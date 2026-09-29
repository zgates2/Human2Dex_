#!/usr/bin/env python3
"""Profile and optionally clean DexUMI episode pickle datasets.

The input dataset is never modified.  In report mode this computes robust,
dataset-level quality statistics and writes a JSON report.  In clean mode it
copies accepted episodes to an output directory and removes only clearly bad
frames from accepted episodes; rejected episodes are omitted and recorded.

ssh -N -f -L 127.0.0.1:8399:127.0.0.1:8317 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
job-70440db2-bacf-41f4-92d2-ff8392cad64f-worker-0.luoshaqi.baai-vision_vision.dx-calc1.job@ssh.platform-multi.baai.ac.cn -p 2222
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


def install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def load(path: Path) -> Any:
    install_numpy_pickle_compat()
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump(obj: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def ep_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.parent.name)
    return (int(match.group(1)) if match else 10**9, str(path))


def pkl_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pkl"), key=ep_sort_key)


def as_float_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    return array if array.size else None


def finite_row(value: Any) -> bool:
    array = as_float_array(value)
    return array is not None and bool(np.all(np.isfinite(array)))


def row_distance(values: list[Any], scale: np.ndarray | None = None) -> np.ndarray:
    arrays = [as_float_array(value).reshape(-1) if as_float_array(value) is not None else None for value in values]
    distances = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(1, len(arrays)):
        left, right = arrays[index - 1], arrays[index]
        if left is None or right is None or left.shape != right.shape:
            continue
        delta = np.abs(right - left)
        if scale is not None and scale.shape == delta.shape:
            delta = delta / np.maximum(scale, 1e-9)
        distances[index] = float(np.max(delta))
    return distances


def rotvec_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    half = angle * 0.5
    return np.concatenate((rotvec / angle * math.sin(half), [math.cos(half)]))


def pose6_distance(values: list[Any]) -> np.ndarray:
    """Return max(position distance in metres, SO(3) angle in radians)."""
    arrays = [as_float_array(value) for value in values]
    distances = np.full(len(values), np.nan, dtype=np.float64)
    for index in range(1, len(arrays)):
        left, right = arrays[index - 1], arrays[index]
        if left is None or right is None or left.shape != (6,) or right.shape != (6,):
            continue
        position_distance = float(np.linalg.norm(right[:3] - left[:3]))
        left_q = rotvec_quaternion(left[3:])
        right_q = rotvec_quaternion(right[3:])
        rotation_distance = 2.0 * math.acos(float(np.clip(abs(np.dot(left_q, right_q)), 0.0, 1.0)))
        distances[index] = max(position_distance, rotation_distance)
    return distances


def jpeg_is_valid(path: Path) -> bool:
    try:
        if path.stat().st_size < 4:
            return False
        with path.open("rb") as handle:
            start = handle.read(3)
            handle.seek(-2, 2)
            end = handle.read(2)
        return start == b"\xff\xd8\xff" and end == b"\xff\xd9"
    except OSError:
        return False


def robust_median(values: list[float], default: float) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(np.median(finite)) if finite.size else default


def contiguous_runs(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in mask:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def episode_metrics(path: Path) -> dict[str, Any]:
    obj = load(path)
    messages = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(messages, list):
        return {"path": str(path), "episode": path.parent.name, "frame_count": 0, "reasons": ["missing_messages"]}

    timestamps = np.asarray(
        [message.get("sampleClockNs", np.nan) / 1e9 if isinstance(message, dict) else np.nan for message in messages],
        dtype=np.float64,
    )
    timestamp_delta = np.diff(timestamps) if len(timestamps) > 1 else np.asarray([], dtype=np.float64)
    rgb_ids = np.asarray([message.get("rgbFrameId", np.nan) if isinstance(message, dict) else np.nan for message in messages], dtype=np.float64)
    rgb_delta = np.diff(rgb_ids) if len(rgb_ids) > 1 else np.asarray([], dtype=np.float64)
    fields = ("trajectoryPose", "o6_command", "wuji_command", "pts21_mano", "fused_pts21_mano")
    field_values = {field: [message.get(field) if isinstance(message, dict) else None for message in messages] for field in fields}
    finite = {field: all(finite_row(value) for value in values) for field, values in field_values.items()}
    distances = {field: row_distance(values) for field, values in field_values.items()}
    distances["trajectoryPose"] = pose6_distance(field_values["trajectoryPose"])
    repeated = np.ones(max(0, len(messages) - 1), dtype=bool)
    for field in ("trajectoryPose", "o6_command", "wuji_command"):
        values = field_values[field]
        field_repeated = np.zeros(max(0, len(messages) - 1), dtype=bool)
        for index in range(1, len(values)):
            left, right = as_float_array(values[index - 1]), as_float_array(values[index])
            if left is not None and right is not None and left.shape == right.shape:
                field_repeated[index - 1] = bool(np.allclose(left, right, rtol=0.0, atol=1e-7))
        repeated &= field_repeated
    quality_flags: list[str] = []
    quality_bad = 0
    missing_images = 0
    unreadable_images = 0
    for message in messages:
        if not isinstance(message, dict):
            quality_bad += 1
            continue
        flags = message.get("qualityFlags") or []
        sample = message.get("sampleQuality") or {}
        if flags or sample.get("ok") is False:
            quality_bad += 1
            quality_flags.extend(str(flag) for flag in flags)
            quality_flags.extend(str(flag) for flag in (sample.get("flags") or []))
        rgb = message.get("rgbImage")
        if isinstance(rgb, str):
            image_path = path.parent / rgb
            if not image_path.exists():
                missing_images += 1
            elif not jpeg_is_valid(image_path):
                unreadable_images += 1
    static_mask = repeated
    static_runs = contiguous_runs(static_mask)
    pose_distances = distances["trajectoryPose"][1:]
    pose_static_mask = np.isfinite(pose_distances) & (pose_distances < 1e-7)
    pose_static_runs = contiguous_runs(pose_static_mask)
    dt_ms = timestamp_delta * 1000.0
    valid_dt = dt_ms[np.isfinite(dt_ms)]
    valid_rgb_delta = rgb_delta[np.isfinite(rgb_delta)]
    return {
        "path": str(path),
        "episode": path.parent.name,
        "frame_count": len(messages),
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 and np.isfinite(timestamps[[0, -1]]).all() else None,
        "dt_median_ms": float(np.median(valid_dt)) if valid_dt.size else None,
        "dt_p01_ms": float(np.percentile(valid_dt, 1)) if valid_dt.size else None,
        "dt_p99_ms": float(np.percentile(valid_dt, 99)) if valid_dt.size else None,
        "dt_bad_count": int(np.sum((valid_dt < 10) | (valid_dt > 100))),
        "rgb_delta_bad_count": int(np.sum(valid_rgb_delta != 1)) if valid_rgb_delta.size else 0,
        "quality_bad_count": quality_bad,
        "quality_flags": sorted(set(quality_flags)),
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "finite_fields": finite,
        "max_step": {field: float(np.nanmax(value)) if np.isfinite(value).any() else None for field, value in distances.items()},
        "p99_step": {field: float(np.nanpercentile(value[np.isfinite(value)], 99)) if np.isfinite(value).any() else None for field, value in distances.items()},
        "static_fraction": float(np.mean(static_mask)) if static_mask.size else 0.0,
        "max_static_run": max(static_runs, default=0),
        "trajectory_static_fraction": float(np.mean(pose_static_mask)) if pose_static_mask.size else 0.0,
        "max_trajectory_static_run": max(pose_static_runs, default=0),
    }


def dataset_thresholds(metrics: list[dict[str, Any]]) -> dict[str, float]:
    counts = np.asarray([item["frame_count"] for item in metrics if item["frame_count"] > 0], dtype=np.float64)
    med_count = float(np.median(counts)) if counts.size else 0.0
    p10_count = float(np.percentile(counts, 10)) if counts.size else 0.0
    p99_steps = {}
    for field in ("trajectoryPose", "o6_command", "wuji_command", "pts21_mano", "fused_pts21_mano"):
        values = [item["p99_step"][field] for item in metrics if item["p99_step"].get(field) is not None]
        p99_steps[field] = robust_median(values, 0.0)
    return {
        "median_frame_count": med_count,
        "min_frame_count": max(30.0, min(p10_count * 0.5, med_count * 0.5)) if med_count else 30.0,
        "max_step_trajectoryPose": max(0.2, p99_steps["trajectoryPose"] * 8.0),
        "max_step_o6_command": max(80.0, p99_steps["o6_command"] * 8.0),
        "max_step_wuji_command": max(0.5, p99_steps["wuji_command"] * 8.0),
        "max_step_pts21_mano": max(0.1, p99_steps["pts21_mano"] * 8.0),
        "max_step_fused_pts21_mano": max(0.1, p99_steps["fused_pts21_mano"] * 8.0),
    }


def classify(item: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if item["frame_count"] < thresholds["min_frame_count"]:
        reasons.append("too_few_frames")
    if item["quality_bad_count"] > max(3, int(item["frame_count"] * 0.02)):
        reasons.append("quality_flags")
    if item["missing_images"] or item["unreadable_images"]:
        reasons.append("missing_or_unreadable_images")
    if not all(item["finite_fields"].values()):
        reasons.append("nonfinite_or_missing_arrays")
    for field in ("trajectoryPose", "o6_command", "wuji_command", "pts21_mano", "fused_pts21_mano"):
        step = item["max_step"].get(field)
        if step is not None and step > thresholds[f"max_step_{field}"]:
            reasons.append(f"large_{field}_jump")
    if item["static_fraction"] > 0.98 and item["frame_count"] > thresholds["median_frame_count"] * 0.5:
        reasons.append("almost_entirely_static")
    if item["max_static_run"] > max(150, int(item["frame_count"] * 0.6)):
        reasons.append("long_static_run")
    if item["trajectory_static_fraction"] > 0.98:
        reasons.append("trajectory_almost_entirely_static")
    elif item["max_trajectory_static_run"] > max(150, int(item["frame_count"] * 0.6)):
        reasons.append("long_static_trajectory_run")
    if item["dt_bad_count"] > max(5, int(item["frame_count"] * 0.03)):
        reasons.append("irregular_timing")
    if item["rgb_delta_bad_count"] > max(5, int(item["frame_count"] * 0.03)):
        reasons.append("rgb_frame_gaps")
    return reasons


def clean_episode(src: Path, dst: Path, bad_frame_indices: set[int]) -> int:
    obj = load(src)
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return 0
    kept = [message for index, message in enumerate(messages) if index not in bad_frame_indices]
    obj["messages"] = kept
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata["cleaning"] = {"removedFrameCount": len(messages) - len(kept), "source": str(src)}
        obj["metadata"] = metadata
    dst.mkdir(parents=True, exist_ok=True)
    dump(obj, dst / src.name)
    image_names = {message.get("rgbImage") for message in kept if isinstance(message, dict) and isinstance(message.get("rgbImage"), str)}
    source_images = src.parent / "images"
    target_images = dst / "images"
    if source_images.exists():
        target_images.mkdir(parents=True, exist_ok=True)
        for name in image_names:
            source_image = src.parent / name
            if source_image.exists():
                target_image = dst / name
                target_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_image, target_image)
    for extra in ("l515_intrinsics.json",):
        source_extra = src.parent / extra
        if source_extra.exists():
            shutil.copy2(source_extra, dst / extra)
    return len(messages) - len(kept)


def bad_frame_indices(path: Path) -> set[int]:
    obj = load(path)
    messages = obj.get("messages") if isinstance(obj, dict) else None
    if not isinstance(messages, list):
        return set()
    bad: set[int] = set()
    required = ("trajectoryPose", "o6_command", "wuji_command", "pts21_mano", "fused_pts21_mano")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            bad.add(index)
            continue
        sample = message.get("sampleQuality") or {}
        flags = message.get("qualityFlags") or []
        image = message.get("rgbImage")
        image_ok = isinstance(image, str) and jpeg_is_valid(path.parent / image)
        if flags or sample.get("ok") is False or not image_ok or not all(finite_row(message.get(field)) for field in required):
            bad.add(index)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path)
    parser.add_argument("--min-frames", type=int, default=None)
    args = parser.parse_args()
    all_reports: list[dict[str, Any]] = []
    for root in args.roots:
        paths = pkl_paths(root)
        episode_dirs = sorted(path for path in root.glob("episode_*") if path.is_dir())
        pkl_dirs = {path.parent.resolve() for path in paths}
        missing_pkl = [path.name for path in episode_dirs if path.resolve() not in pkl_dirs]
        metrics = [episode_metrics(path) for path in paths]
        thresholds = dataset_thresholds(metrics)
        if args.min_frames is not None:
            thresholds["min_frame_count"] = float(args.min_frames)
        for item in metrics:
            item["reasons"] = classify(item, thresholds)
            item["accepted"] = not item["reasons"]
            item["dataset"] = root.name
        all_reports.append({"dataset": root.name, "root": str(root.resolve()), "episode_directory_count": len(episode_dirs), "episode_count": len(metrics), "missing_pkl_episodes": missing_pkl, "thresholds": thresholds, "episodes": metrics})
        if args.clean_root:
            output_root = args.clean_root / root.name
            for path, item in zip(paths, metrics):
                if not item["accepted"]:
                    continue
                target = output_root / path.parent.name
                removed = clean_episode(path, target, bad_frame_indices(path))
                item["removed_frame_count"] = removed
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for report in all_reports:
        accepted = sum(1 for item in report["episodes"] if item["accepted"])
        print(f"{report['dataset']}: {accepted}/{report['episode_count']} accepted")
        print(json.dumps(report["thresholds"], ensure_ascii=False))


if __name__ == "__main__":
    main()
