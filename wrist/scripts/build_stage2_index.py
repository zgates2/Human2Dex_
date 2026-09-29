#!/usr/bin/env python3
"""
构建 Stage2 数据索引和质量报告。

用途
----
从采集 PKL 中提取 wrist RGB、pts21_mano、valid_mask 和同步元数据，生成
`stage2_index.jsonl`。该脚本只做数据质量分析，不依赖 MANO / manotorch。

常用命令
--------
  python wrist/scripts/build_stage2_index.py \
      --data-root data/wrist_test_1 \
      --output-dir wrist/outputs/stage2/index

快速检查：
  python wrist/scripts/build_stage2_index.py --limit-episodes 3
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.constants import HAND_BONES  # noqa: E402
from wrist_pose.utils import save_json, write_jsonl  # noqa: E402


NUMPY2_CORE_PREFIX = "numpy._core"
NUMPY1_CORE_PREFIX = "numpy.core"


SYNC_FIELDS = [
    "mainClockMonotonicNs",
    "sampleClockNs",
    "sourceReceiveNs",
    "rgbCaptureNs",
    "rgbAlignResidualNs",
    "rgbToPicoReceiveDeltaNs",
    "absRgbToPicoReceiveDeltaNs",
    "rgbFrameRepeated",
    "rgbFrameGap",
    "rgbMissingReason",
    "picoActive",
]


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == NUMPY2_CORE_PREFIX or module.startswith(f"{NUMPY2_CORE_PREFIX}."):
                compat_module = NUMPY1_CORE_PREFIX + module[len(NUMPY2_CORE_PREFIX) :]
                return super().find_class(compat_module, name)
            raise


def install_numpy_pickle_compat() -> list[str]:
    try:
        core = importlib.import_module("numpy._core")
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            core = importlib.import_module("numpy.core")
    aliases = {
        "numpy._core": core,
        "numpy._core.multiarray": getattr(core, "multiarray", None),
        "numpy._core.numeric": getattr(core, "numeric", None),
    }
    added = []
    for name, module in aliases.items():
        if module is not None and name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    return added


def read_pkl(path: Path) -> dict[str, Any]:
    added = install_numpy_pickle_compat()
    try:
        with path.open("rb") as f:
            data = NumpyCompatUnpickler(f).load()
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict/messages: {path}")
    return data


def iter_pkl_paths(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.glob("**/*.pkl") if path.is_file())


def as_pts21(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (21, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def as_valid_mask(msg: dict[str, Any], pts21: np.ndarray) -> list[bool]:
    value = msg.get("valid_mask")
    if value is not None:
        arr = np.asarray(value).astype(bool)
        if arr.shape == (21,):
            return arr.tolist()
    return np.all(np.isfinite(pts21), axis=1).tolist()


def qpos_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr.reshape(-1)


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def percentile(values: list[float], ps: tuple[int, ...] = (50, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def bone_lengths(pts21: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(pts21[dst] - pts21[src]) for src, dst in HAND_BONES],
        dtype=np.float32,
    )


def should_skip(
    msg: dict[str, Any],
    pts21: np.ndarray | None,
    image_path: Path | None,
    args: argparse.Namespace,
    prev: dict[str, Any] | None,
) -> str | None:
    if pts21 is None:
        return "invalid_pts21_mano"
    if image_path is None or not image_path.is_file():
        return "missing_rgb"
    if args.require_pico_active and msg.get("picoActive") not in (1, True):
        return "pico_inactive"

    abs_delta = numeric(msg.get("absRgbToPicoReceiveDeltaNs"))
    if abs_delta is not None and abs_delta > args.max_abs_rgb_to_pico_ns:
        return "rgb_to_pico_delta_too_large"

    align = numeric(msg.get("rgbAlignResidualNs"))
    if align is not None and abs(align) > args.max_abs_rgb_align_ns:
        return "rgb_align_residual_too_large"

    if bool(msg.get("rgbFrameRepeated")) and args.drop_repeated_rgb:
        return "rgb_frame_repeated"

    gap = numeric(msg.get("rgbFrameGap"))
    if gap is not None and gap > args.max_rgb_frame_gap:
        return "rgb_frame_gap"

    lengths = bone_lengths(pts21)
    if float(lengths.min()) < args.min_bone_m or float(lengths.max()) > args.max_bone_m:
        return "bone_length_outlier"

    if prev is not None:
        dt = numeric(msg.get("timestamp"))
        prev_t = numeric(prev.get("timestamp"))
        prev_pts = prev.get("pts21")
        if dt is not None and prev_t is not None and prev_pts is not None:
            elapsed = max(dt - prev_t, 1e-3)
            max_velocity = float(np.max(np.linalg.norm(pts21 - prev_pts, axis=-1)) / elapsed)
            if max_velocity > args.max_joint_velocity_mps:
                return "joint_velocity_spike"
        qpos = qpos_array(msg.get("wuji_command"))
        prev_qpos = prev.get("qpos")
        if qpos is not None and prev_qpos is not None and qpos.shape == prev_qpos.shape:
            delta = float(np.max(np.abs(qpos - prev_qpos)))
            if delta > args.max_qpos_delta_rad:
                return "qpos_delta_spike"

    return None


def collect_episode(
    pkl_path: Path,
    data_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, Any]]:
    data = read_pkl(pkl_path)
    rows = []
    skips: Counter[str] = Counter()
    prev_valid: dict[str, Any] | None = None
    sync_values: dict[str, list[float]] = defaultdict(list)
    bone_values: list[float] = []
    frame_gaps: list[float] = []
    joint_deltas: list[float] = []
    qpos_deltas: list[float] = []

    sequence_id = str(pkl_path.parent.relative_to(data_root))
    for frame_idx, msg in enumerate(data["messages"]):
        if not isinstance(msg, dict):
            skips["message_not_dict"] += 1
            continue
        pts21 = as_pts21(msg.get("pts21_mano"))
        rel_image = msg.get("rgbImage")
        image_path = (pkl_path.parent / rel_image).resolve() if isinstance(rel_image, str) and rel_image else None
        reason = should_skip(msg, pts21, image_path, args, prev_valid)
        if reason is not None:
            skips[reason] += 1
            continue

        assert pts21 is not None
        assert image_path is not None
        lengths = bone_lengths(pts21)
        bone_values.extend(float(x) for x in lengths)
        for key in ("rgbAlignResidualNs", "absRgbToPicoReceiveDeltaNs", "rgbToPicoReceiveDeltaNs"):
            value = numeric(msg.get(key))
            if value is not None:
                sync_values[key].append(value / 1e6)
        gap = numeric(msg.get("rgbFrameGap"))
        if gap is not None:
            frame_gaps.append(gap)

        qpos = qpos_array(msg.get("wuji_command"))
        if prev_valid is not None:
            joint_deltas.append(float(np.max(np.linalg.norm(pts21 - prev_valid["pts21"], axis=-1))))
            if qpos is not None and prev_valid.get("qpos") is not None and qpos.shape == prev_valid["qpos"].shape:
                qpos_deltas.append(float(np.max(np.abs(qpos - prev_valid["qpos"]))))

        row = {
            "image_path": str(image_path),
            "pts21_mano": pts21.tolist(),
            "valid_mask": as_valid_mask(msg, pts21),
            "raw26x7": np.asarray(msg["raw26x7"], dtype=np.float32).tolist()
            if msg.get("raw26x7") is not None
            else None,
            "wuji_command": qpos.tolist() if qpos is not None else None,
            "sequence_id": sequence_id,
            "frame_idx": int(frame_idx),
            "timestamp": float(msg.get("timestamp", frame_idx)),
            "source_pkl": str(pkl_path.resolve()),
        }
        for key in SYNC_FIELDS:
            row[key] = msg.get(key)
        rows.append(row)
        prev_valid = {"pts21": pts21, "timestamp": row["timestamp"], "qpos": qpos}

    stats = {
        "sequence_id": sequence_id,
        "source_pkl": str(pkl_path.resolve()),
        "valid_frames": len(rows),
        "skips": dict(skips),
        "sync_ms": {key: percentile(vals) for key, vals in sync_values.items()},
        "bone_length_m": percentile(bone_values),
        "frame_gap": percentile(frame_gaps),
        "joint_delta_m": percentile(joint_deltas),
        "qpos_delta_rad": percentile(qpos_deltas),
    }
    return rows, skips, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "wrist_test_1")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "wrist" / "outputs" / "stage2" / "index")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--require-pico-active", action="store_true", default=True)
    parser.add_argument("--allow-inactive-pico", dest="require_pico_active", action="store_false")
    parser.add_argument("--drop-repeated-rgb", action="store_true", default=True)
    parser.add_argument("--keep-repeated-rgb", dest="drop_repeated_rgb", action="store_false")
    parser.add_argument("--max-abs-rgb-to-pico-ns", type=float, default=30_000_000.0)
    parser.add_argument("--max-abs-rgb-align-ns", type=float, default=30_000_000.0)
    parser.add_argument("--max-rgb-frame-gap", type=float, default=5.0)
    parser.add_argument("--min-bone-m", type=float, default=0.005)
    parser.add_argument("--max-bone-m", type=float, default=0.12)
    parser.add_argument("--max-joint-velocity-mps", type=float, default=4.0)
    parser.add_argument("--max-qpos-delta-rad", type=float, default=1.2)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pkl_paths = iter_pkl_paths(data_root)
    if args.limit_episodes is not None:
        pkl_paths = pkl_paths[: int(args.limit_episodes)]
    if not pkl_paths:
        raise SystemExit(f"no PKL files found under {data_root}")

    all_rows = []
    skip_totals: Counter[str] = Counter()
    episodes = []
    for pkl_path in pkl_paths:
        rows, skips, stats = collect_episode(pkl_path, data_root, args)
        all_rows.extend(rows)
        skip_totals.update(skips)
        episodes.append(stats)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "stage2_index.jsonl"
    report_path = output_dir / "quality_report.json"
    write_jsonl(index_path, all_rows)
    save_json(
        report_path,
        {
            "data_root": str(data_root),
            "num_episodes": len(pkl_paths),
            "num_valid_frames": len(all_rows),
            "skip_totals": dict(skip_totals),
            "thresholds": {
                "max_abs_rgb_to_pico_ns": args.max_abs_rgb_to_pico_ns,
                "max_abs_rgb_align_ns": args.max_abs_rgb_align_ns,
                "max_rgb_frame_gap": args.max_rgb_frame_gap,
                "min_bone_m": args.min_bone_m,
                "max_bone_m": args.max_bone_m,
                "max_joint_velocity_mps": args.max_joint_velocity_mps,
                "max_qpos_delta_rad": args.max_qpos_delta_rad,
            },
            "episodes": episodes,
        },
    )
    print(json.dumps({"index": str(index_path), "report": str(report_path), "frames": len(all_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
