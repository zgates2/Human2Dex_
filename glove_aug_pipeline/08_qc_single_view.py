#!/usr/bin/env python3
"""08: Validate single-view RGB, object observations and action labels."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline_common import get, load_config, path, require_new_directory, write_manifest


def read_pkl(source: Path) -> dict[str, Any]:
    try:
        importlib.import_module("numpy._core")
    except ImportError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.numeric", core.numeric)
    with source.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"Not a DexUMI PKL: {source}")
    return value


def atomic_pickle(value: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def finite_vector(value: Any, exact_size: int) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return False
    return bool(array.size == exact_size and np.isfinite(array).all())


def resolve_image(pkl_path: Path, message: dict[str, Any], field: str) -> Path | None:
    value = message.get(field)
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else pkl_path.parent / candidate


def valid_frame(pkl_path: Path, message: Any, spec: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(message, dict):
        return False, ["message_not_dict"]
    reasons: list[str] = []
    image = resolve_image(pkl_path, message, spec["source_image"])
    if image is None or not image.is_file():
        reasons.append("missing_rgb")
    if not bool(message.get(spec["object_obs_valid"], False)):
        reasons.append("invalid_object_memory")
    if not finite_vector(message.get(spec["object_obs"]), int(spec["object_obs_dim"])):
        reasons.append("invalid_object_obs")
    if not finite_vector(message.get(spec["trajectory_field"]), int(spec["trajectory_dims"])):
        reasons.append("invalid_trajectory")
    for field, dims in spec["action_fields"]:
        if not finite_vector(message.get(field), int(dims)):
            reasons.append(f"invalid_{field}")
    return not reasons, reasons


def contiguous_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    return runs


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def field_prefix(item: dict[str, Any]) -> str:
    return str(item.get("field_prefix") or f"task_object_{safe_name(str(item['name']))}")


def draw_overlay(
    pkl_path: Path,
    message: dict[str, Any],
    spec: dict[str, Any],
    destination: Path,
) -> None:
    image_path = resolve_image(pkl_path, message, spec["source_image"])
    image = None if image_path is None else cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    pocket = np.asarray(message.get(spec["pocket_uv"]), dtype=np.float64).reshape(-1)
    if pocket.shape == (2,) and np.isfinite(pocket).all():
        cv2.drawMarker(image, tuple(np.rint(pocket).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    colors = [(255, 100, 0), (180, 0, 255), (0, 220, 0), (255, 0, 180)]
    for index, item in enumerate(spec["objects"]):
        prefix = field_prefix(item)
        uv = message.get(f"{prefix}_policy_uv")
        try:
            point = np.asarray(uv, dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if point.shape != (2,) or not np.isfinite(point).all():
            continue
        color = colors[index % len(colors)]
        center = tuple(np.rint(point).astype(int))
        cv2.circle(image, center, 5, color, -1, cv2.LINE_AA)
        cv2.putText(image, str(item["name"]), (center[0] + 7, center[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def process_episode(source_text: str, output_text: str, spec: dict[str, Any]) -> dict[str, Any]:
    source_pkl = Path(source_text)
    output_root = Path(output_text)
    cv2.setNumThreads(1)
    data = read_pkl(source_pkl)
    messages = data["messages"]
    flags: list[bool] = []
    errors: Counter[str] = Counter()
    for message in messages:
        valid, reasons = valid_frame(source_pkl, message, spec)
        flags.append(valid)
        errors.update(reasons)
    runs = [
        (start, end)
        for start, end in contiguous_runs(flags)
        if end - start >= int(spec["min_segment_frames"])
    ]
    segments = []
    for segment_index, (start, end) in enumerate(runs):
        target_dir = output_root / "common" / f"{source_pkl.parent.name}_seg_{segment_index:03d}"
        shutil.copytree(source_pkl.parent, target_dir, copy_function=hardlink_or_copy)
        target_pkl = target_dir / source_pkl.name
        segment_data = read_pkl(target_pkl)
        segment_data["messages"] = segment_data["messages"][start:end]
        segment_data.setdefault("metadata", {})["human2dexObjectSegmentV1"] = {
            "source_pkl": str(source_pkl),
            "source_episode": source_pkl.parent.name,
            "parent_episode": re.sub(str(spec["parent_episode_suffix_regex"]), "", source_pkl.parent.name),
            "start_frame": start,
            "end_frame_exclusive": end,
            "length": end - start,
        }
        atomic_pickle(segment_data, target_pkl)
        segments.append({"path": str(target_dir), "start": start, "end": end, "length": end - start})
        frames = np.linspace(start, end - 1, num=min(int(spec["overlays_per_segment"]), end - start), dtype=int)
        for frame in sorted(set(frames.tolist())):
            draw_overlay(source_pkl, messages[frame], spec, output_root / "overlays" / f"{source_pkl.parent.name}_seg_{segment_index:03d}_frame_{frame:06d}.jpg")
    return {
        "pkl": str(source_pkl),
        "frames": len(messages),
        "valid_frames": int(sum(flags)),
        "discarded_frames": int(len(messages) - sum(flags)),
        "reasons": dict(errors),
        "segments": segments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source = path(config, "paths.appearance_root")
    output = path(config, "paths.qc_segments_root")
    if not source.is_dir():
        raise FileNotFoundError(f"Run stage 07 first: {source}")
    require_new_directory(output)
    pkls = sorted(source.rglob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No PKLs under {source}")
    objects = get(config, "task_objects", required=True)
    action_fields = [(str(get(config, "labels.o6_action_field", "fused_o6_command")), 6)]
    if not bool(get(config, "labels.skip_wuji", True)):
        action_fields.append((str(get(config, "labels.wuji_action_field", "fused_wuji_command")), 20))
    spec = {
        "source_image": str(get(config, "fields.source_image", "rgbImage")),
        "pocket_uv": str(get(config, "fields.pocket_uv", "wrist_grasp_pocket_uv")),
        "object_obs": str(get(config, "fields.object_pocket_obs", "objectPocketObs")),
        "object_obs_valid": str(get(config, "fields.object_pocket_valid", "objectPocketObsValid")),
        "object_obs_dim": 5 * len(objects),
        "trajectory_field": str(get(config, "labels.trajectory_field", "trajectoryPose")),
        "trajectory_dims": int(get(config, "labels.trajectory_min_dims", 6)),
        "action_fields": action_fields,
        "objects": objects,
        "min_segment_frames": int(get(config, "qc.min_segment_frames", 24)),
        "overlays_per_segment": int(get(config, "qc.overlays_per_segment", 3)),
        "parent_episode_suffix_regex": str(get(config, "qc.parent_episode_suffix_regex", r"_app_aug[0-9]+$")),
    }
    workers = max(1, min(args.workers or int(get(config, "parallel.qc_workers", 32)), len(pkls)))
    print(json.dumps({"source": str(source), "output": str(output), "pkls": len(pkls), "workers": workers, "spec": spec}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_episode, str(item), str(output), spec) for item in pkls]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index == 1 or index % 25 == 0 or index == len(futures):
                print(f"[08] {index}/{len(futures)}", flush=True)
    summary = {"source": str(source), "output": str(output), "pkls": len(pkls), "workers": workers, "spec": spec, "results": results}
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(output, "08_qc_single_view", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
