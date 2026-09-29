#!/usr/bin/env python3
"""Read-only quality audit for DexUMI PKL datasets used by DP training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


MP_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

FINGERTIPS = [4, 8, 12, 16, 20]
WUJI_JOINTS = [f"finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]

FIELD_SPECS = {
    "trajectoryPose": (6,),
    "pts21_mano": (21, 3),
    "raw26x7": (26, 7),
    "o6_command": (6,),
    "wuji_command": (5, 4),
    "valid_mask": (21,),
}

TIME_FIELDS_NS = [
    "sampleClockNs",
    "rgbCaptureNs",
    "sourceReceiveNs",
    "mainClockMonotonicNs",
]


class NumpyCompatUnpickler(pickle.Unpickler):
    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(self.MODULE_ALIASES.get(module, module), name)


@dataclass
class Flag:
    episode: str
    frame: int
    prev_frame: int | None
    severity: str
    category: str
    field: str
    metric: str
    value: float | str | None
    threshold: float | str | None
    stage_context: str
    evidence: str
    rgb_delta_ms: float | None = None
    rgb_abs_delta_ms: float | None = None
    rgb_frame_gap: int | None = None
    rgb_repeated: bool | None = None
    image: str | None = None


@dataclass
class EpisodeRecord:
    episode: str
    pkl_path: Path
    n: int
    metadata: dict[str, Any]
    counters: Counter[str] = field(default_factory=Counter)
    flags: list[Flag] = field(default_factory=list)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    step_metrics: dict[str, np.ndarray] = field(default_factory=dict)
    time_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    image_paths: list[str | None] = field(default_factory=list)
    image_exists: np.ndarray | None = None
    quality_flag_counts: Counter[str] = field(default_factory=Counter)
    existing_quality_flags: list[tuple[int, str]] = field(default_factory=list)
    decision: str = "unclassified"
    notes: list[str] = field(default_factory=list)


def natural_key(value: Any) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def read_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        try:
            data = pickle.load(handle)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            handle.seek(0)
            data = NumpyCompatUnpickler(handle).load()
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"unsupported PKL layout: {path}")
    return data


def find_pkl_paths(root: Path, episodes: list[str]) -> list[Path]:
    root = root.expanduser().resolve()
    requested = set(episodes)
    if root.is_file():
        return [root]
    if (root / "images").is_dir():
        paths = sorted(root.glob("*.pkl"), key=natural_key)
    else:
        paths = sorted(root.glob("episode_*/*.pkl"), key=natural_key)
    if requested:
        paths = [path for path in paths if path.parent.name in requested or path.stem in requested]
    return paths


def numeric_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
        out = float(value)
        if np.isfinite(out):
            return out
    return None


def finite_array(value: Any, shape: tuple[int, ...], dtype: Any = np.float64) -> tuple[np.ndarray | None, str | None]:
    if value is None:
        return None, "missing"
    try:
        arr = np.asarray(value, dtype=dtype)
    except Exception as exc:
        return None, f"array_error:{exc}"
    if arr.shape != shape:
        return None, f"shape:{arr.shape}"
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        return None, "nonfinite"
    return arr, None


def add_flag(
    record: EpisodeRecord,
    frame: int,
    prev_frame: int | None,
    severity: str,
    category: str,
    field: str,
    metric: str,
    value: float | str | None,
    threshold: float | str | None,
    stage_context: str,
    evidence: str,
) -> None:
    rgb_delta = value_at(record.time_arrays.get("rgbToPicoReceiveDeltaNs"), frame)
    rgb_abs_delta = value_at(record.time_arrays.get("absRgbToPicoReceiveDeltaNs"), frame)
    rgb_gap = value_at(record.time_arrays.get("rgbFrameGap"), frame)
    rgb_repeated = value_at(record.time_arrays.get("rgbFrameRepeated"), frame)
    record.flags.append(
        Flag(
            episode=record.episode,
            frame=int(frame),
            prev_frame=None if prev_frame is None else int(prev_frame),
            severity=severity,
            category=category,
            field=field,
            metric=metric,
            value=value,
            threshold=threshold,
            stage_context=stage_context,
            evidence=evidence,
            rgb_delta_ms=None if rgb_delta is None else float(rgb_delta) / 1e6,
            rgb_abs_delta_ms=None if rgb_abs_delta is None else float(rgb_abs_delta) / 1e6,
            rgb_frame_gap=None if rgb_gap is None else int(rgb_gap),
            rgb_repeated=None if rgb_repeated is None else bool(rgb_repeated),
            image=record.image_paths[frame] if 0 <= frame < len(record.image_paths) else None,
        )
    )


def value_at(arr: np.ndarray | None, idx: int) -> float | None:
    if arr is None or idx < 0 or idx >= len(arr):
        return None
    value = float(arr[idx])
    if not np.isfinite(value):
        return None
    return value


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def relative_rot_angle(a_rotvec: np.ndarray, b_rotvec: np.ndarray) -> float:
    rel = rotvec_to_rotmat(a_rotvec).T @ rotvec_to_rotmat(b_rotvec)
    cos_theta = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cos_theta)


def parse_wuji_limits(urdf_path: Path | None) -> dict[str, tuple[float, float]]:
    fallback = {
        "finger1_joint1": (-0.0448, 1.6508),
        "finger1_joint2": (-0.1659, 0.9339),
        "finger1_joint3": (-0.4932, 1.6272),
        "finger1_joint4": (-0.4932, 1.6272),
    }
    for finger in range(2, 6):
        fallback[f"finger{finger}_joint1"] = (-0.32695, 1.636)
        fallback[f"finger{finger}_joint2"] = (-0.495, 0.495)
        fallback[f"finger{finger}_joint3"] = (-0.4932, 1.6272)
        fallback[f"finger{finger}_joint4"] = (-0.4932, 1.6272)
    if urdf_path is None or not urdf_path.is_file():
        return fallback
    limits: dict[str, tuple[float, float]] = {}
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if name not in WUJI_JOINTS:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[name] = (
            float(limit.attrib.get("lower", fallback.get(name, (-2.0, 2.0))[0])),
            float(limit.attrib.get("upper", fallback.get(name, (-2.0, 2.0))[1])),
        )
    for name, pair in fallback.items():
        limits.setdefault(name, pair)
    return limits


def q_summary(values: Any) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p999": None,
            "max": None,
            "mean": None,
        }
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p999": float(np.percentile(arr, 99.9)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def percentile(records: list[EpisodeRecord], metric: str, pct: float) -> float:
    arrays = []
    for record in records:
        arr = record.step_metrics.get(metric)
        if arr is not None:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                arrays.append(finite)
    if not arrays:
        return 0.0
    return float(np.percentile(np.concatenate(arrays), pct))


def finite_metric_values(records: list[EpisodeRecord], metric: str) -> np.ndarray:
    arrays = []
    for record in records:
        arr = record.step_metrics.get(metric)
        if arr is None:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            arrays.append(finite)
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float64)


def scan_episode(pkl_path: Path, args: argparse.Namespace) -> EpisodeRecord:
    data = read_pkl(pkl_path)
    messages = data["messages"]
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    episode = pkl_path.parent.name
    n = len(messages)
    record = EpisodeRecord(episode=episode, pkl_path=pkl_path, n=n, metadata=metadata)

    for key, shape in FIELD_SPECS.items():
        dtype = bool if key == "valid_mask" else np.float64
        fill = False if key == "valid_mask" else np.nan
        record.arrays[key] = np.full((n, *shape), fill, dtype=dtype)
        record.masks[f"{key}_valid"] = np.zeros(n, dtype=bool)

    for key in TIME_FIELDS_NS + [
        "timestamp",
        "rgbToPicoReceiveDeltaNs",
        "absRgbToPicoReceiveDeltaNs",
        "rgbAlignResidualNs",
        "rgbSyncDeltaNs",
        "sourceAgeNs",
        "rgbFrameGap",
        "rgbFrameRepeated",
        "rgbFrameId",
        "rgbFramesRead",
    ]:
        record.time_arrays[key] = np.full(n, np.nan, dtype=np.float64)

    image_exists = np.zeros(n, dtype=bool)

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            record.counters["message_not_dict"] += 1
            add_flag(record, idx, None, "hard_error", "structure", "message", "type", type(msg).__name__, "dict", "none", "message is not a dict")
            continue

        rel = msg.get("rgbImage")
        rel_str = str(rel) if rel not in (None, "", False) else None
        record.image_paths.append(rel_str)
        if rel_str is None:
            record.counters["rgbImage_missing"] += 1
            add_flag(record, idx, None, "hard_error", "rgb", "rgbImage", "missing", None, "path", "none", "rgbImage is missing")
        else:
            path = pkl_path.parent / rel_str
            exists = path.is_file()
            image_exists[idx] = exists
            if not exists:
                record.counters["rgbImage_file_missing"] += 1
                add_flag(record, idx, None, "hard_error", "rgb", "rgbImage", "file_missing", rel_str, "existing file", "none", "rgbImage path does not exist")

        for key, shape in FIELD_SPECS.items():
            dtype = bool if key == "valid_mask" else np.float64
            arr, err = finite_array(msg.get(key), shape, dtype=dtype)
            if arr is None:
                record.counters[f"{key}_{err}"] += 1
                add_flag(
                    record,
                    idx,
                    None,
                    "hard_error",
                    "field_integrity",
                    key,
                    "missing_shape_or_nonfinite",
                    err,
                    str(shape),
                    "none",
                    f"{key} failed validation: {err}",
                )
                continue
            record.arrays[key][idx] = arr
            record.masks[f"{key}_valid"][idx] = True

        valid_mask = record.arrays["valid_mask"][idx]
        if record.masks["valid_mask_valid"][idx] and not bool(np.all(valid_mask)):
            record.counters["valid_mask_false_frames"] += 1
            add_flag(
                record,
                idx,
                None,
                "hard_error",
                "field_integrity",
                "valid_mask",
                "not_all_true",
                int(np.count_nonzero(valid_mask)),
                21,
                "none",
                "pts21_mano valid_mask is not all true",
            )

        pico_active = numeric_scalar(msg.get("picoActive"))
        if pico_active is None:
            record.counters["picoActive_missing"] += 1
        elif int(pico_active) != 1:
            record.counters["picoActive_not_1"] += 1
            add_flag(record, idx, None, "hard_error", "field_integrity", "picoActive", "not_active", int(pico_active), 1, "none", "PICO active flag is not 1")

        for key in record.time_arrays:
            value = numeric_scalar(msg.get(key))
            if value is not None:
                record.time_arrays[key][idx] = value

        flags = msg.get("qualityFlags")
        if isinstance(flags, (list, tuple)):
            for flag in flags:
                flag_str = str(flag)
                record.quality_flag_counts[flag_str] += 1
                record.existing_quality_flags.append((idx, flag_str))

    record.image_exists = image_exists
    compute_episode_metrics(record)
    return record


def compute_episode_metrics(record: EpisodeRecord) -> None:
    n = record.n
    pose = record.arrays["trajectoryPose"]
    pose_valid = record.masks["trajectoryPose_valid"]
    trans = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    rot = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    for idx in range(1, n):
        if pose_valid[idx - 1] and pose_valid[idx]:
            trans[idx - 1] = float(np.linalg.norm(pose[idx, :3] - pose[idx - 1, :3]))
            rot[idx - 1] = relative_rot_angle(pose[idx - 1, 3:], pose[idx, 3:])
    record.step_metrics["traj_trans_step_m"] = trans
    record.step_metrics["traj_rot_step_rad"] = rot
    record.step_metrics["traj_rot_step_deg"] = np.rad2deg(rot)

    pts = record.arrays["pts21_mano"]
    pts_valid = record.masks["pts21_mano_valid"]
    pts_joint_max = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    pts_joint_mean = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    pts_tip_max = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    for idx in range(1, n):
        if pts_valid[idx - 1] and pts_valid[idx]:
            dist = np.linalg.norm(pts[idx] - pts[idx - 1], axis=1)
            pts_joint_max[idx - 1] = float(np.max(dist))
            pts_joint_mean[idx - 1] = float(np.mean(dist))
            pts_tip_max[idx - 1] = float(np.max(dist[FINGERTIPS]))
    record.step_metrics["pts_joint_step_max_m"] = pts_joint_max
    record.step_metrics["pts_joint_step_mean_m"] = pts_joint_mean
    record.step_metrics["pts_tip_step_max_m"] = pts_tip_max

    o6 = record.arrays["o6_command"].reshape(n, 6)
    o6_valid = record.masks["o6_command_valid"]
    o6_step_max = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    o6_step_mean = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    o6_mean_delta = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    for idx in range(1, n):
        if o6_valid[idx - 1] and o6_valid[idx]:
            step = o6[idx] - o6[idx - 1]
            o6_step_max[idx - 1] = float(np.max(np.abs(step)))
            o6_step_mean[idx - 1] = float(np.mean(np.abs(step)))
            o6_mean_delta[idx - 1] = float(np.mean(step))
    record.step_metrics["o6_step_max"] = o6_step_max
    record.step_metrics["o6_step_mean"] = o6_step_mean
    record.step_metrics["o6_mean_delta"] = o6_mean_delta

    wuji = record.arrays["wuji_command"].reshape(n, 20)
    wuji_valid = record.masks["wuji_command_valid"]
    wuji_step_max = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    wuji_step_mean = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    wuji_mean_delta = np.full(max(0, n - 1), np.nan, dtype=np.float64)
    for idx in range(1, n):
        if wuji_valid[idx - 1] and wuji_valid[idx]:
            step = wuji[idx] - wuji[idx - 1]
            wuji_step_max[idx - 1] = float(np.max(np.abs(step)))
            wuji_step_mean[idx - 1] = float(np.mean(np.abs(step)))
            wuji_mean_delta[idx - 1] = float(np.mean(step))
    record.step_metrics["wuji_step_max_rad"] = wuji_step_max
    record.step_metrics["wuji_step_mean_rad"] = wuji_step_mean
    record.step_metrics["wuji_mean_delta_rad"] = wuji_mean_delta


def derive_thresholds(records: list[EpisodeRecord], args: argparse.Namespace) -> dict[str, float]:
    return {
        "traj_trans_step_m": max(args.traj_trans_jump_m, percentile(records, "traj_trans_step_m", 99.9) * 1.5),
        "traj_rot_step_deg": max(args.traj_rot_jump_deg, percentile(records, "traj_rot_step_deg", 99.9) * 1.5),
        "pts_joint_step_max_m": max(args.pts_jump_m, percentile(records, "pts_joint_step_max_m", 99.9) * 1.5),
        "pts_joint_step_mean_m": max(args.pts_mean_jump_m, percentile(records, "pts_joint_step_mean_m", 99.9) * 2.0),
        "o6_step_max": max(args.o6_jump, percentile(records, "o6_step_max", 99.9) * 1.2),
        "wuji_step_max_rad": max(args.wuji_jump_rad, percentile(records, "wuji_step_max_rad", 99.9) * 1.2),
        "ctx_traj_trans_step_m": max(args.ctx_traj_trans_m, percentile(records, "traj_trans_step_m", 99.5)),
        "ctx_traj_rot_step_deg": max(args.ctx_traj_rot_deg, percentile(records, "traj_rot_step_deg", 99.5)),
        "ctx_o6_step_max": max(args.ctx_o6_jump, percentile(records, "o6_step_max", 99.5)),
        "ctx_wuji_step_max_rad": max(args.ctx_wuji_jump_rad, percentile(records, "wuji_step_max_rad", 99.5)),
    }


def any_metric_near(record: EpisodeRecord, metric: str, center_step: int, radius: int, threshold: float) -> bool:
    arr = record.step_metrics.get(metric)
    if arr is None:
        return False
    lo = max(0, center_step - radius)
    hi = min(len(arr), center_step + radius + 1)
    if lo >= hi:
        return False
    sub = arr[lo:hi]
    return bool(np.any(np.isfinite(sub) & (sub >= threshold)))


def stage_context(record: EpisodeRecord, step_idx: int, thresholds: dict[str, float], include_self_hand: bool) -> str:
    reasons = []
    o6_fast = any_metric_near(record, "o6_step_max", step_idx, 2, thresholds["ctx_o6_step_max"])
    wuji_fast = any_metric_near(record, "wuji_step_max_rad", step_idx, 2, thresholds["ctx_wuji_step_max_rad"])
    wrist_fast = (
        any_metric_near(record, "traj_trans_step_m", step_idx, 2, thresholds["ctx_traj_trans_step_m"])
        or any_metric_near(record, "traj_rot_step_deg", step_idx, 2, thresholds["ctx_traj_rot_step_deg"])
    )
    if include_self_hand and (o6_fast or wuji_fast):
        o6_delta = value_at(record.step_metrics.get("o6_mean_delta"), step_idx)
        wuji_delta = value_at(record.step_metrics.get("wuji_mean_delta_rad"), step_idx)
        if (o6_delta is not None and o6_delta < -2.0) or (wuji_delta is not None and wuji_delta > 0.015):
            reasons.append("closing_or_grasp_phase")
        elif (o6_delta is not None and o6_delta > 2.0) or (wuji_delta is not None and wuji_delta < -0.015):
            reasons.append("release_or_opening_phase")
        else:
            reasons.append("fast_hand_retargeting_phase")
    elif o6_fast or wuji_fast:
        reasons.append("near_fast_hand_phase")
    if wrist_fast:
        reasons.append("near_fast_wrist_or_contact_candidate")
    frame = min(record.n - 1, step_idx + 1)
    rgb_abs = value_at(record.time_arrays.get("absRgbToPicoReceiveDeltaNs"), frame)
    gap = value_at(record.time_arrays.get("rgbFrameGap"), frame)
    repeated = value_at(record.time_arrays.get("rgbFrameRepeated"), frame)
    if rgb_abs is not None and rgb_abs / 1e6 > 20.0:
        reasons.append("rgb_hand_sync_warning")
    if gap is not None and gap > 0:
        reasons.append("rgb_frame_gap")
    if repeated is not None and repeated > 0:
        reasons.append("rgb_repeated")
    return "+".join(reasons) if reasons else "no_supporting_context"


def single_frame_return(record: EpisodeRecord, field: str, frame: int, return_threshold: float) -> bool:
    if frame <= 0 or frame >= record.n - 1:
        return False
    arr = record.arrays.get(field)
    mask = record.masks.get(f"{field}_valid")
    if arr is None or mask is None or not (mask[frame - 1] and mask[frame] and mask[frame + 1]):
        return False
    prev = arr[frame - 1].reshape(-1).astype(np.float64)
    curr = arr[frame].reshape(-1).astype(np.float64)
    nxt = arr[frame + 1].reshape(-1).astype(np.float64)
    if field == "trajectoryPose":
        into = np.linalg.norm(curr[:3] - prev[:3])
        out = np.linalg.norm(nxt[:3] - curr[:3])
        ret = np.linalg.norm(nxt[:3] - prev[:3])
    elif field == "pts21_mano":
        into = float(np.max(np.linalg.norm((curr - prev).reshape(21, 3), axis=1)))
        out = float(np.max(np.linalg.norm((nxt - curr).reshape(21, 3), axis=1)))
        ret = float(np.max(np.linalg.norm((nxt - prev).reshape(21, 3), axis=1)))
    else:
        into = float(np.max(np.abs(curr - prev)))
        out = float(np.max(np.abs(nxt - curr)))
        ret = float(np.max(np.abs(nxt - prev)))
    return bool(into > return_threshold and out > return_threshold and ret < return_threshold * 0.35)


def classify_jump(
    record: EpisodeRecord,
    step_idx: int,
    field: str,
    metric: str,
    value: float,
    threshold: float,
    category: str,
    thresholds: dict[str, float],
    include_self_hand: bool,
) -> None:
    frame = step_idx + 1
    if has_hard_flag_on_frame(record, frame) or has_hard_flag_on_frame(record, frame - 1):
        add_flag(
            record,
            frame,
            frame - 1,
            "hard_error",
            category,
            field,
            metric,
            float(value),
            float(threshold),
            "adjacent_to_hard_frame",
            "jump includes a frame already hard-flagged; do not interpret as real manipulation",
        )
        return
    context = stage_context(record, step_idx, thresholds, include_self_hand=include_self_hand)
    is_single_return = single_frame_return(record, field, frame, threshold)
    if is_single_return and context == "no_supporting_context":
        severity = "suspect_unexplained"
        evidence = "single-frame return pattern with no nearby gripper/wrist/sync context"
    elif context == "no_supporting_context" and value >= threshold * 1.8:
        severity = "suspect_unexplained"
        evidence = "jump exceeds threshold with no nearby gripper/wrist/sync context"
    elif context == "no_supporting_context":
        severity = "review"
        evidence = "jump exceeds threshold; no automatic deletion because operation phase cannot be inferred"
    else:
        severity = "review_fast_phase"
        evidence = "jump coincides with inferred manipulation/contact/timing context; keep for review, not automatic deletion"
    add_flag(record, frame, frame - 1, severity, category, field, metric, float(value), float(threshold), context, evidence)


def has_hard_flag_on_frame(record: EpisodeRecord, frame: int) -> bool:
    return any(flag.frame == frame and flag.severity == "hard_error" for flag in record.flags)


def classify_records(records: list[EpisodeRecord], thresholds: dict[str, float], limits: dict[str, tuple[float, float]], args: argparse.Namespace) -> None:
    for record in records:
        classify_integrity_and_ranges(record, limits, args)
        classify_step_jumps(record, thresholds)
        classify_time(record, args)
        classify_episode_decision(record, args)


def classify_integrity_and_ranges(record: EpisodeRecord, limits: dict[str, tuple[float, float]], args: argparse.Namespace) -> None:
    pose = record.arrays["trajectoryPose"]
    pose_valid = record.masks["trajectoryPose_valid"]
    for idx in np.where(pose_valid)[0]:
        xyz = pose[idx, :3]
        rot = pose[idx, 3:]
        xyz_abs = float(np.max(np.abs(xyz)))
        rot_norm = float(np.linalg.norm(rot))
        if xyz_abs > args.pose_abs_limit_m:
            record.counters["trajectoryPose_abs_extreme"] += 1
            add_flag(record, int(idx), None, "hard_error", "range", "trajectoryPose", "xyz_abs_max_m", xyz_abs, args.pose_abs_limit_m, "none", "trajectory position is outside conservative room-scale limit")
        if rot_norm > args.rotvec_norm_limit_rad:
            record.counters["trajectoryPose_rotvec_extreme"] += 1
            add_flag(record, int(idx), None, "hard_error", "range", "trajectoryPose", "rotvec_norm_rad", rot_norm, args.rotvec_norm_limit_rad, "none", "rotation vector norm is outside conservative representation limit")

    pts = record.arrays["pts21_mano"]
    pts_valid = record.masks["pts21_mano_valid"]
    for idx in np.where(pts_valid)[0]:
        p = pts[idx]
        max_abs = float(np.max(np.abs(p)))
        span = float(np.max(np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)))
        max_bone = float(max(np.linalg.norm(p[a] - p[b]) for a, b in HAND_BONES))
        wrist_norm = float(np.linalg.norm(p[0]))
        if max_abs > args.pts_abs_limit_m:
            record.counters["pts_abs_extreme"] += 1
            add_flag(record, int(idx), None, "hard_error", "range", "pts21_mano", "coord_abs_max_m", max_abs, args.pts_abs_limit_m, "none", "wrist-centered keypoint coordinate is outside conservative hand-scale limit")
        if span > args.pts_span_limit_m:
            record.counters["pts_span_extreme"] += 1
            add_flag(record, int(idx), None, "hard_error", "range", "pts21_mano", "hand_span_m", span, args.pts_span_limit_m, "none", "21-point hand span is outside conservative hand-scale limit")
        if max_bone > args.pts_bone_limit_m:
            record.counters["pts_bone_extreme"] += 1
            add_flag(record, int(idx), None, "hard_error", "range", "pts21_mano", "max_bone_m", max_bone, args.pts_bone_limit_m, "none", "MediaPipe bone length is outside conservative limit")
        if wrist_norm > args.pts_wrist_origin_limit_m:
            record.counters["pts_wrist_origin_offset"] += 1
            add_flag(record, int(idx), None, "review", "range", "pts21_mano", "wrist_norm_m", wrist_norm, args.pts_wrist_origin_limit_m, "none", "pts21_mano is expected to be wrist-centered")

    o6 = record.arrays["o6_command"].reshape(record.n, 6)
    o6_valid = record.masks["o6_command_valid"]
    for idx in np.where(o6_valid)[0]:
        row = o6[idx]
        if np.any((row < 0) | (row > 255)):
            record.counters["o6_outside_uint8_range"] += 1
            add_flag(record, int(idx), None, "hard_error", "range", "o6_command", "outside_0_255", float(np.max(np.abs(row))), "[0,255]", "none", "o6 command is outside uint8 command range")
        if np.any(row > 250):
            record.counters["o6_over_250"] += int(np.count_nonzero(row > 250))
            add_flag(record, int(idx), None, "review", "range", "o6_command", "over_250_count", int(np.count_nonzero(row > 250)), "<=250 expected by teleop mapping", "none", "o6 command is above teleop 250-open convention but still inside uint8 range")

    wuji = record.arrays["wuji_command"].reshape(record.n, 20)
    wuji_valid = record.masks["wuji_command_valid"]
    lower = np.asarray([limits[name][0] for name in WUJI_JOINTS], dtype=np.float64)
    upper = np.asarray([limits[name][1] for name in WUJI_JOINTS], dtype=np.float64)
    for idx in np.where(wuji_valid)[0]:
        row = wuji[idx]
        below = row < (lower - args.wuji_limit_tolerance_rad)
        above = row > (upper + args.wuji_limit_tolerance_rad)
        if np.any(below | above):
            record.counters["wuji_outside_urdf_limits"] += int(np.count_nonzero(below | above))
            add_flag(record, int(idx), None, "hard_error", "range", "wuji_command", "outside_urdf_limits_count", int(np.count_nonzero(below | above)), f"URDF limits +/- {args.wuji_limit_tolerance_rad}", "none", "Wuji qpos is outside parsed URDF joint limits")


def classify_step_jumps(record: EpisodeRecord, thresholds: dict[str, float]) -> None:
    checks = [
        ("trajectoryPose", "traj_trans_step_m", "trajectory_jump", False),
        ("trajectoryPose", "traj_rot_step_deg", "trajectory_jump", False),
        ("pts21_mano", "pts_joint_step_max_m", "keypoint_jump", False),
        ("pts21_mano", "pts_joint_step_mean_m", "keypoint_jump", False),
        ("o6_command", "o6_step_max", "action_jump", True),
        ("wuji_command", "wuji_step_max_rad", "action_jump", True),
    ]
    for field, metric, category, include_self_hand in checks:
        arr = record.step_metrics.get(metric)
        if arr is None:
            continue
        threshold = thresholds[metric]
        for step_idx in np.where(np.isfinite(arr) & (arr > threshold))[0]:
            classify_jump(record, int(step_idx), field, metric, float(arr[step_idx]), threshold, category, thresholds, include_self_hand=include_self_hand)


def classify_time(record: EpisodeRecord, args: argparse.Namespace) -> None:
    strict_increase_fields = {"sampleClockNs", "rgbCaptureNs"}
    for key in TIME_FIELDS_NS + ["timestamp"]:
        arr = record.time_arrays.get(key)
        if arr is None:
            continue
        finite = np.isfinite(arr)
        if not np.any(finite):
            record.counters[f"{key}_missing_all"] += 1
            continue
        missing = int(np.count_nonzero(~finite))
        if missing:
            record.counters[f"{key}_missing"] += missing
        idxs = np.where(finite)[0]
        vals = arr[idxs]
        if vals.size >= 2:
            diffs = np.diff(vals)
            regressions = np.where(diffs < 0)[0]
            for local in regressions[: args.max_flags_per_episode]:
                frame = int(idxs[local + 1])
                record.counters[f"{key}_non_monotonic"] += 1
                add_flag(record, frame, int(idxs[local]), "hard_error", "time", key, "negative_delta", float(diffs[local]), ">=0", "none", "time field regressed")
            equal_steps = np.where(diffs == 0)[0]
            if key in strict_increase_fields:
                for local in equal_steps[: args.max_flags_per_episode]:
                    frame = int(idxs[local + 1])
                    record.counters[f"{key}_equal_step"] += 1
                    add_flag(record, frame, int(idxs[local]), "hard_error", "time", key, "equal_timestamp", 0.0, ">0", "none", "sample/RGB clock did not advance")
            elif equal_steps.size:
                record.counters[f"{key}_equal_step"] += int(equal_steps.size)
                for local in equal_steps[: min(args.max_flags_per_episode, 10)]:
                    frame = int(idxs[local + 1])
                    add_flag(record, frame, int(idxs[local]), "review", "time", key, "equal_timestamp", 0.0, "non-decreasing", "state_reuse_possible", "auxiliary timestamp repeated; record as reuse/alignment evidence")

    abs_delta = record.time_arrays.get("absRgbToPicoReceiveDeltaNs")
    if abs_delta is not None:
        for frame in np.where(np.isfinite(abs_delta) & (abs_delta / 1e6 > args.sync_warn_ms))[0]:
            record.counters["rgb_hand_sync_over_warn_ms"] += 1
            if record.counters["rgb_hand_sync_over_warn_ms"] <= args.max_flags_per_episode:
                add_flag(record, int(frame), None, "review", "time_alignment", "rgbToPicoReceiveDeltaNs", "abs_ms", float(abs_delta[frame] / 1e6), args.sync_warn_ms, "rgb_hand_sync_warning", "RGB capture and PICO receive timestamps exceed warning threshold")

    for key in ("rgbAlignResidualNs", "rgbSyncDeltaNs"):
        arr = record.time_arrays.get(key)
        if arr is None:
            continue
        over = np.where(np.isfinite(arr) & (np.abs(arr) / 1e6 > args.align_warn_ms))[0]
        for frame in over[: args.max_flags_per_episode]:
            record.counters[f"{key}_over_warn_ms"] += 1
            add_flag(record, int(frame), None, "review", "time_alignment", key, "abs_ms", float(abs(arr[frame]) / 1e6), args.align_warn_ms, "timebase_mismatch_candidate", f"{key} exceeds alignment warning threshold")


def classify_episode_decision(record: EpisodeRecord, args: argparse.Namespace) -> None:
    hard = sum(1 for flag in record.flags if flag.severity == "hard_error")
    suspect = sum(1 for flag in record.flags if flag.severity == "suspect_unexplained")
    timing = sum(1 for flag in record.flags if flag.category == "time_alignment")
    if record.n < args.dp_horizon:
        record.decision = "exclude_candidate"
        record.notes.append(f"episode shorter than dp_horizon={args.dp_horizon}")
    elif hard:
        record.decision = "review_or_filter_frames"
        record.notes.append(f"{hard} hard frame/field flags")
    elif suspect:
        record.decision = "review"
        record.notes.append(f"{suspect} unexplained jump candidates")
    elif timing:
        record.decision = "keep_with_timing_marks"
        record.notes.append(f"{timing} timing alignment warning frames")
    elif record.flags:
        record.decision = "keep_with_review_marks"
    else:
        record.decision = "keep"


def invalid_quality_ranges(record: EpisodeRecord, horizon: int, include_review: bool) -> list[tuple[int, int, str]]:
    bad_frames = set()
    for flag in record.flags:
        if flag.severity in {"hard_error", "suspect_unexplained"} or (include_review and flag.severity.startswith("review")):
            bad_frames.add(flag.frame)
    if record.image_exists is not None:
        for idx in np.where(~record.image_exists)[0]:
            bad_frames.add(int(idx))
    ranges: list[tuple[int, int, str]] = []
    max_valid_start = record.n - horizon
    if max_valid_start < 0:
        return [(0, max(0, record.n - 1), "episode_shorter_than_horizon")]
    frame_ranges = contiguous_ranges(sorted(bad_frames))
    for first_frame, last_frame in frame_ranges:
        start = max(0, first_frame - horizon + 1)
        end = min(max_valid_start, last_frame)
        if start <= end:
            ranges.append((start, end, f"window_contains_flagged_frames_{first_frame}_{last_frame}"))
    return merge_ranges_union(ranges)


def contiguous_ranges(nums: list[int]) -> list[tuple[int, int]]:
    if not nums:
        return []
    out = []
    start = end = nums[0]
    for num in nums[1:]:
        if num == end + 1:
            end = num
        else:
            out.append((start, end))
            start = end = num
    out.append((start, end))
    return out


def merge_ranges_union(ranges: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda item: (item[0], item[1]))
    merged = []
    cur_s, cur_e, reasons = ranges[0][0], ranges[0][1], [ranges[0][2]]
    for start, end, reason in ranges[1:]:
        if start <= cur_e + 1:
            cur_e = max(cur_e, end)
            reasons.append(reason)
        else:
            merged.append((cur_s, cur_e, "+".join(reasons)))
            cur_s, cur_e, reasons = start, end, [reason]
    merged.append((cur_s, cur_e, "+".join(reasons)))
    return merged


def invalid_quality_ranges_unmerged(record: EpisodeRecord, horizon: int, include_review: bool) -> list[tuple[int, int, str]]:
    bad_frames = set()
    for flag in record.flags:
        if flag.severity in {"hard_error", "suspect_unexplained"} or (include_review and flag.severity.startswith("review")):
            bad_frames.add(flag.frame)
    if record.image_exists is not None:
        for idx in np.where(~record.image_exists)[0]:
            bad_frames.add(int(idx))
    ranges: list[tuple[int, int, str]] = []
    max_valid_start = record.n - horizon
    if max_valid_start < 0:
        return [(0, max(0, record.n - 1), "episode_shorter_than_horizon")]
    for frame in sorted(bad_frames):
        start = max(0, frame - horizon + 1)
        end = min(max_valid_start, frame)
        if start <= end:
            ranges.append((start, end, f"window_contains_flagged_frame_{frame}"))
    return merge_ranges(ranges)


def tail_invalid_ranges(record: EpisodeRecord, horizon: int) -> list[tuple[int, int, str]]:
    if record.n <= 0:
        return []
    first_tail = max(0, record.n - horizon + 1)
    if first_tail <= record.n - 1:
        return [(first_tail, record.n - 1, f"tail_start_without_full_horizon_{horizon}")]
    return []


def merge_ranges(ranges: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda item: (item[0], item[1], item[2]))
    merged: list[tuple[int, int, str]] = []
    cur_s, cur_e, cur_r = ranges[0]
    for start, end, reason in ranges[1:]:
        if start <= cur_e + 1 and reason == cur_r:
            cur_e = max(cur_e, end)
        else:
            merged.append((cur_s, cur_e, cur_r))
            cur_s, cur_e, cur_r = start, end, reason
    merged.append((cur_s, cur_e, cur_r))
    return merged


def range_count(ranges: list[tuple[int, int, str]]) -> int:
    return int(sum(max(0, end - start + 1) for start, end, _ in ranges))


def episode_summary_row(record: EpisodeRecord, thresholds: dict[str, float], args: argparse.Namespace) -> dict[str, Any]:
    hard = sum(1 for flag in record.flags if flag.severity == "hard_error")
    suspect = sum(1 for flag in record.flags if flag.severity == "suspect_unexplained")
    review_fast = sum(1 for flag in record.flags if flag.severity == "review_fast_phase")
    review = sum(1 for flag in record.flags if flag.severity == "review")
    tail_ranges = tail_invalid_ranges(record, args.dp_horizon)
    quality_ranges = invalid_quality_ranges(record, args.dp_horizon, include_review=False)
    max_valid_start = max(-1, record.n - args.dp_horizon)
    valid_start_count = max(0, max_valid_start + 1) - range_count(quality_ranges)
    row: dict[str, Any] = {
        "episode": record.episode,
        "pkl_path": str(record.pkl_path),
        "frames": record.n,
        "decision": record.decision,
        "notes": "; ".join(record.notes),
        "hard_flags": hard,
        "suspect_unexplained_flags": suspect,
        "review_fast_phase_flags": review_fast,
        "review_flags": review,
        "existing_quality_flag_count": sum(record.quality_flag_counts.values()),
        "existing_quality_flags": json.dumps(dict(record.quality_flag_counts), sort_keys=True),
        "missing_image_files": int(record.counters.get("rgbImage_file_missing", 0)),
        "missing_rgbImage": int(record.counters.get("rgbImage_missing", 0)),
        "pico_inactive_frames": int(record.counters.get("picoActive_not_1", 0)),
        "dp_horizon": int(args.dp_horizon),
        "dp_valid_start_count": int(max(0, valid_start_count)),
        "dp_tail_invalid_start_count": range_count(tail_ranges),
        "dp_quality_invalid_start_count": range_count(quality_ranges),
        "duration_sec_sample": duration_sec(record.time_arrays.get("sampleClockNs"), 1e-9),
        "fps_median_sample": median_fps(record.time_arrays.get("sampleClockNs"), 1e-9),
        "rgb_to_pico_offset_ms_median": q_summary_ms(record.time_arrays.get("rgbToPicoReceiveDeltaNs"))["p50"],
        "rgb_to_pico_abs_ms_p95": q_summary_ms(record.time_arrays.get("absRgbToPicoReceiveDeltaNs"))["p95"],
        "rgb_to_pico_abs_ms_max": q_summary_ms(record.time_arrays.get("absRgbToPicoReceiveDeltaNs"))["max"],
        "rgb_align_residual_ms_max": q_summary_ms_abs(record.time_arrays.get("rgbAlignResidualNs"))["max"],
        "rgb_sync_delta_ms_max": q_summary_ms_abs(record.time_arrays.get("rgbSyncDeltaNs"))["max"],
        "rgb_frame_gap_count": int(np.count_nonzero(record.time_arrays.get("rgbFrameGap", np.asarray([])) > 0)),
        "rgb_frame_gap_total": finite_sum(record.time_arrays.get("rgbFrameGap")),
        "rgb_repeated_count": int(np.count_nonzero(record.time_arrays.get("rgbFrameRepeated", np.asarray([])) > 0)),
    }
    for metric in [
        "traj_trans_step_m",
        "traj_rot_step_deg",
        "pts_joint_step_max_m",
        "pts_joint_step_mean_m",
        "o6_step_max",
        "wuji_step_max_rad",
    ]:
        arr = record.step_metrics.get(metric)
        finite = arr[np.isfinite(arr)] if arr is not None else np.asarray([])
        row[f"{metric}_max"] = None if finite.size == 0 else float(np.max(finite))
        row[f"{metric}_p99"] = None if finite.size == 0 else float(np.percentile(finite, 99))
        row[f"{metric}_threshold"] = float(thresholds[metric])
        row[f"{metric}_flag_count"] = int(np.count_nonzero(finite > thresholds[metric])) if finite.size else 0
    return row


def finite_sum(arr: np.ndarray | None) -> int:
    if arr is None:
        return 0
    finite = arr[np.isfinite(arr)]
    return int(np.sum(finite)) if finite.size else 0


def duration_sec(arr: np.ndarray | None, scale: float) -> float | None:
    if arr is None:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    return float((finite[-1] - finite[0]) * scale)


def median_fps(arr: np.ndarray | None, scale: float) -> float | None:
    if arr is None:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    dt = np.diff(finite) * scale
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def q_summary_ms(arr: np.ndarray | None) -> dict[str, float | int | None]:
    if arr is None:
        return q_summary([])
    return q_summary(np.asarray(arr, dtype=np.float64) / 1e6)


def q_summary_ms_abs(arr: np.ndarray | None) -> dict[str, float | int | None]:
    if arr is None:
        return q_summary([])
    return q_summary(np.abs(np.asarray(arr, dtype=np.float64)) / 1e6)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flag_to_row(flag: Flag) -> dict[str, Any]:
    return {
        "episode": flag.episode,
        "frame": flag.frame,
        "prev_frame": "" if flag.prev_frame is None else flag.prev_frame,
        "severity": flag.severity,
        "category": flag.category,
        "field": flag.field,
        "metric": flag.metric,
        "value": flag.value,
        "threshold": flag.threshold,
        "stage_context": flag.stage_context,
        "evidence": flag.evidence,
        "rgb_delta_ms": flag.rgb_delta_ms,
        "rgb_abs_delta_ms": flag.rgb_abs_delta_ms,
        "rgb_frame_gap": flag.rgb_frame_gap,
        "rgb_repeated": flag.rgb_repeated,
        "image": flag.image,
    }


def build_start_range_rows(records: list[EpisodeRecord], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for start, end, reason in tail_invalid_ranges(record, args.dp_horizon):
            rows.append(
                {
                    "episode": record.episode,
                    "start_begin": start,
                    "start_end": end,
                    "count": end - start + 1,
                    "reason": reason,
                    "kind": "tail",
                }
            )
        for start, end, reason in invalid_quality_ranges(record, args.dp_horizon, include_review=False):
            rows.append(
                {
                    "episode": record.episode,
                    "start_begin": start,
                    "start_end": end,
                    "count": end - start + 1,
                    "reason": reason,
                    "kind": "quality",
                }
            )
    return rows


def safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def global_time_summary(records: list[EpisodeRecord]) -> dict[str, Any]:
    out = {}
    for key in [
        "rgbToPicoReceiveDeltaNs",
        "absRgbToPicoReceiveDeltaNs",
        "rgbAlignResidualNs",
        "rgbSyncDeltaNs",
        "sourceAgeNs",
    ]:
        values = []
        for record in records:
            arr = record.time_arrays.get(key)
            if arr is not None:
                values.append(arr)
        if values:
            merged = np.concatenate(values)
            if key in {"rgbAlignResidualNs", "rgbSyncDeltaNs", "absRgbToPicoReceiveDeltaNs"}:
                out[f"{key}_ms_abs"] = q_summary_ms_abs(merged)
            out[f"{key}_ms"] = q_summary_ms(merged)
    out["rgbFrameGap_count"] = int(sum(np.count_nonzero(record.time_arrays.get("rgbFrameGap", np.asarray([])) > 0) for record in records))
    out["rgbFrameGap_total"] = int(sum(finite_sum(record.time_arrays.get("rgbFrameGap")) for record in records))
    out["rgbFrameRepeated_count"] = int(sum(np.count_nonzero(record.time_arrays.get("rgbFrameRepeated", np.asarray([])) > 0) for record in records))
    return out


def global_metric_summary(records: list[EpisodeRecord]) -> dict[str, Any]:
    metrics = {}
    for metric in [
        "traj_trans_step_m",
        "traj_rot_step_deg",
        "pts_joint_step_max_m",
        "pts_joint_step_mean_m",
        "o6_step_max",
        "wuji_step_max_rad",
    ]:
        metrics[metric] = q_summary(finite_metric_values(records, metric))
    return metrics


def image_motion_alignment(records: list[EpisodeRecord], args: argparse.Namespace, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    if args.image_motion_sample_episodes <= 0:
        return []
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return [{"episode": "", "status": f"skipped_no_cv2:{exc}"}]

    ranked = sorted(records, key=lambda rec: motion_energy_score(rec, thresholds), reverse=True)
    selected = ranked[: args.image_motion_sample_episodes]
    rows = []
    for record in selected:
        rows.append(compute_image_motion_alignment(record, cv2, args, thresholds))
    return rows


def motion_energy_score(record: EpisodeRecord, thresholds: dict[str, float]) -> float:
    total = np.zeros(max(0, record.n - 1), dtype=np.float64)
    for metric in ["traj_trans_step_m", "traj_rot_step_deg", "pts_joint_step_max_m", "o6_step_max", "wuji_step_max_rad"]:
        arr = record.step_metrics.get(metric)
        if arr is None:
            continue
        denom = max(thresholds.get(metric, 1.0), 1e-9)
        total += np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) / denom
    return float(np.percentile(total, 95)) if total.size else 0.0


def compute_image_motion_alignment(record: EpisodeRecord, cv2: Any, args: argparse.Namespace, thresholds: dict[str, float]) -> dict[str, Any]:
    imgs = []
    missing = 0
    for rel in record.image_paths:
        if rel is None:
            imgs.append(None)
            missing += 1
            continue
        path = record.pkl_path.parent / rel
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            imgs.append(None)
            missing += 1
            continue
        img = cv2.resize(img, (args.image_motion_size, args.image_motion_size), interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32))

    image_motion = np.full(max(0, record.n - 1), np.nan, dtype=np.float64)
    for idx in range(1, record.n):
        if imgs[idx - 1] is not None and imgs[idx] is not None:
            image_motion[idx - 1] = float(np.mean(np.abs(imgs[idx] - imgs[idx - 1])))

    action_motion = np.zeros(max(0, record.n - 1), dtype=np.float64)
    for metric in ["traj_trans_step_m", "traj_rot_step_deg", "pts_joint_step_max_m", "o6_step_max", "wuji_step_max_rad"]:
        arr = record.step_metrics.get(metric)
        if arr is None:
            continue
        denom = max(thresholds.get(metric, 1.0), 1e-9)
        action_motion += np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) / denom

    corrs = {}
    for lag in range(-args.image_motion_max_lag, args.image_motion_max_lag + 1):
        corr = lagged_corr(image_motion, action_motion, lag)
        corrs[str(lag)] = corr
    finite_corrs = {int(k): v for k, v in corrs.items() if v is not None and np.isfinite(v)}
    if finite_corrs:
        best_lag = max(finite_corrs, key=lambda lag: finite_corrs[lag])
        best_corr = finite_corrs[best_lag]
    else:
        best_lag = None
        best_corr = None
    return {
        "episode": record.episode,
        "status": "ok",
        "frames": record.n,
        "missing_or_unreadable_images": missing,
        "best_lag_frames": best_lag,
        "best_lag_seconds_at_30hz": None if best_lag is None else best_lag / 30.0,
        "best_corr": best_corr,
        "corr_lag0": corrs.get("0"),
        "corrs_by_lag": json.dumps(corrs, sort_keys=True),
        "interpretation": "weak visual-action timing evidence; timestamps remain primary",
    }


def lagged_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float | None:
    if lag < 0:
        aa = a[-lag:]
        bb = b[: len(aa)]
    elif lag > 0:
        aa = a[: len(a) - lag]
        bb = b[lag:]
    else:
        aa = a
        bb = b
    mask = np.isfinite(aa) & np.isfinite(bb)
    if np.count_nonzero(mask) < 8:
        return None
    aa = aa[mask]
    bb = bb[mask]
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return None
    return float(np.dot(aa, bb) / denom)


def write_markdown(
    path: Path,
    records: list[EpisodeRecord],
    episode_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    metric_summary: dict[str, Any],
    time_summary: dict[str, Any],
    image_alignment_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decisions = Counter(row["decision"] for row in episode_rows)
    hard_flags = sum(row["hard_flags"] for row in episode_rows)
    suspect_flags = sum(row["suspect_unexplained_flags"] for row in episode_rows)
    review_fast = sum(row["review_fast_phase_flags"] for row in episode_rows)
    total_frames = sum(record.n for record in records)
    tail_invalid = sum(row["dp_tail_invalid_start_count"] for row in episode_rows)
    quality_invalid = sum(row["dp_quality_invalid_start_count"] for row in episode_rows)

    lines = [
        "# DP Dataset Audit Summary",
        "",
        f"- source_root: `{args.input}`",
        f"- generated_at_unix: `{int(time.time())}`",
        f"- read_only: `true`",
        f"- episodes: `{len(records)}`",
        f"- frames: `{total_frames}`",
        f"- dp_horizon_for_start_marks: `{args.dp_horizon}`",
        f"- decisions: `{dict(decisions)}`",
        f"- hard_flags: `{hard_flags}`",
        f"- suspect_unexplained_flags: `{suspect_flags}`",
        f"- review_fast_phase_flags: `{review_fast}`",
        f"- tail_invalid_start_marks: `{tail_invalid}`",
        f"- quality_invalid_start_marks: `{quality_invalid}`",
        "",
        "## Thresholds",
        "",
        "| metric | threshold | global p99 | global p99.9 | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, threshold in thresholds.items():
        if metric.startswith("ctx_"):
            continue
        summary = metric_summary.get(metric, {})
        lines.append(
            f"| `{metric}` | {threshold:.6g} | {fmt(summary.get('p99'))} | {fmt(summary.get('p999'))} | {fmt(summary.get('max'))} |"
        )

    lines.extend(
        [
            "",
            "## Time Alignment",
            "",
            "- Primary evidence is stored timestamps: `rgbCaptureNs`, `sourceReceiveNs`, `sampleClockNs`, `rgbToPicoReceiveDeltaNs`, `rgbAlignResidualNs`, and `rgbSyncDeltaNs`.",
            f"- Estimated RGB-to-hand offset, ms: `{json.dumps(time_summary.get('rgbToPicoReceiveDeltaNs_ms', {}), sort_keys=True)}`",
            f"- Absolute RGB-to-hand offset, ms: `{json.dumps(time_summary.get('absRgbToPicoReceiveDeltaNs_ms_abs', {}), sort_keys=True)}`",
            f"- RGB align residual absolute ms: `{json.dumps(time_summary.get('rgbAlignResidualNs_ms_abs', {}), sort_keys=True)}`",
            f"- RGB sync delta absolute ms: `{json.dumps(time_summary.get('rgbSyncDeltaNs_ms_abs', {}), sort_keys=True)}`",
            f"- RGB frame gaps: count=`{time_summary.get('rgbFrameGap_count')}`, total_gap_frames=`{time_summary.get('rgbFrameGap_total')}`",
            f"- RGB repeated frames: `{time_summary.get('rgbFrameRepeated_count')}`",
            "",
        ]
    )

    if image_alignment_rows:
        ok_rows = [row for row in image_alignment_rows if row.get("status") == "ok"]
        lag_counts = Counter(row.get("best_lag_frames") for row in ok_rows)
        lines.extend(
            [
                "## Image Motion Check",
                "",
                f"- sampled_episodes: `{len(ok_rows)}`",
                f"- best_lag_frame_counts: `{dict(lag_counts)}`",
                "- This is weak evidence only because pixel change is not a hand-state label.",
                "",
            ]
        )

    lines.extend(
        [
            "## Top Episodes To Review",
            "",
            "| episode | decision | hard | suspect | review_fast | max_traj_m | max_rot_deg | max_pts_m | max_o6_step | max_wuji_rad | notes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    top = sorted(
        episode_rows,
        key=lambda row: (
            row["decision"] != "keep",
            row["hard_flags"],
            row["suspect_unexplained_flags"],
            row["traj_trans_step_m_max"] or 0.0,
            row["traj_rot_step_deg_max"] or 0.0,
        ),
        reverse=True,
    )[:30]
    for row in top:
        lines.append(
            "| `{episode}` | `{decision}` | {hard_flags} | {suspect_unexplained_flags} | {review_fast_phase_flags} | {traj:.4g} | {rot:.4g} | {pts:.4g} | {o6:.4g} | {wuji:.4g} | {notes} |".format(
                episode=row["episode"],
                decision=row["decision"],
                hard_flags=row["hard_flags"],
                suspect_unexplained_flags=row["suspect_unexplained_flags"],
                review_fast_phase_flags=row["review_fast_phase_flags"],
                traj=row["traj_trans_step_m_max"] or 0.0,
                rot=row["traj_rot_step_deg_max"] or 0.0,
                pts=row["pts_joint_step_max_m_max"] or 0.0,
                o6=row["o6_step_max_max"] or 0.0,
                wuji=row["wuji_step_max_rad_max"] or 0.0,
                notes=row["notes"],
            )
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `episode_audit.csv`: one row per episode.",
            "- `frame_flags.csv`: frame/pair-level flags with severity and context.",
            "- `dp_start_mark_ranges.csv`: tail and quality invalid DP start ranges; no source data was changed.",
            "- `audit_summary.json`: machine-readable global summary.",
            "- `image_motion_alignment.csv`: optional weak RGB/action correlation evidence when enabled.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def write_filter_json(path: Path, records: list[EpisodeRecord], args: argparse.Namespace) -> None:
    episodes = {}
    for record in records:
        episodes[record.episode] = {
            "pkl_path": str(record.pkl_path),
            "frames": record.n,
            "decision": record.decision,
            "flagged_frames": sorted({int(flag.frame) for flag in record.flags}),
            "hard_or_suspect_frames": sorted({int(flag.frame) for flag in record.flags if flag.severity in {"hard_error", "suspect_unexplained"}}),
            "tail_invalid_start_ranges": [
                {"start": start, "end": end, "reason": reason}
                for start, end, reason in tail_invalid_ranges(record, args.dp_horizon)
            ],
            "quality_invalid_start_ranges": [
                {"start": start, "end": end, "reason": reason}
                for start, end, reason in invalid_quality_ranges(record, args.dp_horizon, include_review=False)
            ],
        }
    payload = {
        "read_only_sidecar": True,
        "source_root": str(args.input),
        "dp_horizon": int(args.dp_horizon),
        "episodes": episodes,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=safe_json_value) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only DP-oriented audit for DexUMI PKL episode bundles.")
    parser.add_argument("--input", type=Path, required=True, help="Dataset root, episode dir, or PKL path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for sidecar reports.")
    parser.add_argument("--episode", action="append", default=[], help="Episode name to include; can be repeated.")
    parser.add_argument("--dp-horizon", type=int, default=32, help="Full future chunk length used to mark invalid tail start frames.")
    parser.add_argument("--wuji-urdf", type=Path, default=Path("wuji_retargeting/wuji_hand_description/urdf/right.urdf"))
    parser.add_argument("--max-flags-per-episode", type=int, default=500)

    parser.add_argument("--pose-abs-limit-m", type=float, default=5.0)
    parser.add_argument("--rotvec-norm-limit-rad", type=float, default=12.566370614359172)
    parser.add_argument("--pts-abs-limit-m", type=float, default=0.30)
    parser.add_argument("--pts-span-limit-m", type=float, default=0.35)
    parser.add_argument("--pts-bone-limit-m", type=float, default=0.12)
    parser.add_argument("--pts-wrist-origin-limit-m", type=float, default=0.01)
    parser.add_argument("--wuji-limit-tolerance-rad", type=float, default=0.003)

    parser.add_argument("--traj-trans-jump-m", type=float, default=0.05)
    parser.add_argument("--traj-rot-jump-deg", type=float, default=10.0)
    parser.add_argument("--pts-jump-m", type=float, default=0.04)
    parser.add_argument("--pts-mean-jump-m", type=float, default=0.012)
    parser.add_argument("--o6-jump", type=float, default=25.0)
    parser.add_argument("--wuji-jump-rad", type=float, default=0.25)

    parser.add_argument("--ctx-traj-trans-m", type=float, default=0.02)
    parser.add_argument("--ctx-traj-rot-deg", type=float, default=5.0)
    parser.add_argument("--ctx-o6-jump", type=float, default=12.0)
    parser.add_argument("--ctx-wuji-jump-rad", type=float, default=0.12)
    parser.add_argument("--sync-warn-ms", type=float, default=20.0)
    parser.add_argument("--align-warn-ms", type=float, default=2.0)

    parser.add_argument("--image-motion-sample-episodes", type=int, default=0)
    parser.add_argument("--image-motion-size", type=int, default=64)
    parser.add_argument("--image-motion-max-lag", type=int, default=3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.input = args.input.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.wuji_urdf is not None and not args.wuji_urdf.is_absolute():
        args.wuji_urdf = (Path.cwd() / args.wuji_urdf).resolve()
    pkl_paths = find_pkl_paths(args.input, args.episode)
    if not pkl_paths:
        raise FileNotFoundError(f"no PKL files found under {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"found episodes: {len(pkl_paths)}", flush=True)
    records: list[EpisodeRecord] = []
    for idx, pkl_path in enumerate(pkl_paths, 1):
        record = scan_episode(pkl_path, args)
        records.append(record)
        print(f"[scan {idx:04d}/{len(pkl_paths):04d}] {record.episode} frames={record.n} flags_so_far={len(record.flags)}", flush=True)

    thresholds = derive_thresholds(records, args)
    limits = parse_wuji_limits(args.wuji_urdf)
    classify_records(records, thresholds, limits, args)

    metric_summary = global_metric_summary(records)
    time_summary = global_time_summary(records)
    episode_rows = [episode_summary_row(record, thresholds, args) for record in records]
    flag_rows = [flag_to_row(flag) for record in records for flag in sorted(record.flags, key=lambda item: (item.frame, item.severity, item.field))]
    start_rows = build_start_range_rows(records, args)
    image_alignment_rows = image_motion_alignment(records, args, thresholds)

    write_csv(args.out_dir / "episode_audit.csv", episode_rows)
    write_csv(args.out_dir / "frame_flags.csv", flag_rows)
    write_csv(args.out_dir / "dp_start_mark_ranges.csv", start_rows)
    if image_alignment_rows:
        write_csv(args.out_dir / "image_motion_alignment.csv", image_alignment_rows)
    write_filter_json(args.out_dir / "recommended_filter.json", records, args)

    summary = {
        "read_only": True,
        "source_root": str(args.input),
        "episodes": len(records),
        "frames": int(sum(record.n for record in records)),
        "dp_horizon": int(args.dp_horizon),
        "thresholds": thresholds,
        "metric_summary": metric_summary,
        "time_summary": time_summary,
        "decision_counts": dict(Counter(row["decision"] for row in episode_rows)),
        "flag_counts_by_severity": dict(Counter(flag["severity"] for flag in flag_rows)),
        "flag_counts_by_category": dict(Counter(flag["category"] for flag in flag_rows)),
        "outputs": {
            "episode_audit_csv": str(args.out_dir / "episode_audit.csv"),
            "frame_flags_csv": str(args.out_dir / "frame_flags.csv"),
            "dp_start_mark_ranges_csv": str(args.out_dir / "dp_start_mark_ranges.csv"),
            "recommended_filter_json": str(args.out_dir / "recommended_filter.json"),
            "audit_summary_md": str(args.out_dir / "audit_summary.md"),
        },
    }
    if image_alignment_rows:
        summary["outputs"]["image_motion_alignment_csv"] = str(args.out_dir / "image_motion_alignment.csv")
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=safe_json_value) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.out_dir / "audit_summary.md",
        records,
        episode_rows,
        thresholds,
        metric_summary,
        time_summary,
        image_alignment_rows,
        args,
    )

    print(f"wrote audit outputs to {args.out_dir}", flush=True)
    print(f"decisions: {summary['decision_counts']}", flush=True)
    print(f"flag severities: {summary['flag_counts_by_severity']}", flush=True)


if __name__ == "__main__":
    main()
