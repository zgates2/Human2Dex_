#!/usr/bin/env python3
"""Add Wuji Hand qpos commands to recorded DexUMI pickle episodes."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "pick_sponge"
DEFAULT_YAML = REPO_ROOT / "wuji_retargeting" / "config" / "adaptive_analytical_pico.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wuji_retargeting import Retargeter  # noqa: E402


def _valid_wuji_command(value: object) -> bool:
    if value is None:
        return False
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.shape == (5, 4) and np.all(np.isfinite(arr))


def _retarget_mano_points(retargeter: Retargeter, pts21_mano: object) -> np.ndarray:
    points = np.asarray(pts21_mano, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"pts21_mano shape must be (21, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("pts21_mano contains NaN or Inf")

    keypoints = points.copy()

    rotation_xyz = getattr(retargeter, "rotation_xyz", {}) or {}
    has_rotation = any(float(rotation_xyz.get(axis, 0.0)) != 0.0 for axis in ("x", "y", "z"))
    if has_rotation:
        keypoints = retargeter._apply_rotation(keypoints)

    if getattr(retargeter, "_has_offset", False):
        keypoints = retargeter._apply_offset(keypoints)

    qpos = np.asarray(retargeter.optimizer.solve(keypoints), dtype=np.float32).reshape(-1)
    qpos = np.asarray(retargeter.lp_filter.next(qpos), dtype=np.float32).reshape(-1)
    if qpos.shape != (20,):
        raise RuntimeError(f"Wuji retargeter returned shape {qpos.shape}, expected (20,)")
    if not np.all(np.isfinite(qpos)):
        raise RuntimeError("Wuji retargeter returned NaN or Inf")
    return qpos.reshape(5, 4).astype(np.float32, copy=False)


def _process_pkl(path: Path, retargeter: Retargeter, *, overwrite: bool) -> tuple[int, int, bool]:
    with path.open("rb") as f:
        data = pickle.load(f)

    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        raise ValueError("pickle root must be a dict with list field 'messages'")

    valid_messages = [
        msg for msg in messages
        if isinstance(msg, dict) and msg.get("pts21_mano") is not None
    ]
    if (
        not overwrite
        and valid_messages
        and all(_valid_wuji_command(msg.get("wuji_command")) for msg in valid_messages)
    ):
        return len(valid_messages), 0, False

    retargeter.reset()
    added_or_updated = 0
    valid_count = 0

    for frame_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"message {frame_idx} is not a dict")

        pts21_mano = msg.get("pts21_mano")
        if pts21_mano is None:
            if overwrite or "wuji_command" not in msg:
                msg["wuji_command"] = None
                added_or_updated += 1
            continue

        valid_count += 1
        if overwrite or not _valid_wuji_command(msg.get("wuji_command")):
            msg["wuji_command"] = _retarget_mano_points(retargeter, pts21_mano)
            added_or_updated += 1
        else:
            existing = np.asarray(msg["wuji_command"], dtype=np.float32).reshape(5, 4)
            msg["wuji_command"] = existing
            qpos = existing.reshape(-1)
            retargeter.optimizer.last_qpos = qpos.astype(np.float64)
            retargeter.lp_filter.y = qpos.astype(np.float32)
            retargeter.lp_filter.is_init = True

    if added_or_updated:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp_path.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return valid_count, added_or_updated, bool(added_or_updated)


def _iter_pkl_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*.pkl") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    yaml_path = args.yaml.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Wuji retargeting YAML not found: {yaml_path}")

    paths = _iter_pkl_files(root)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"No pkl files found under {root}")
        return 0

    retargeter = Retargeter.from_yaml(str(yaml_path), hand_side=args.hand)
    start = time.perf_counter()
    total_valid = 0
    total_changed = 0
    changed_files = 0

    for idx, path in enumerate(paths, start=1):
        valid_count, changed_count, changed = _process_pkl(
            path,
            retargeter,
            overwrite=args.overwrite,
        )
        total_valid += valid_count
        total_changed += changed_count
        changed_files += int(changed)
        if idx == 1 or idx % 10 == 0 or idx == len(paths):
            elapsed = time.perf_counter() - start
            print(
                f"[{idx:04d}/{len(paths):04d}] changed_files={changed_files} "
                f"changed_frames={total_changed} valid_frames={total_valid} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    print(
        f"Done: files={len(paths)} changed_files={changed_files} "
        f"valid_frames={total_valid} changed_frames={total_changed} "
        f"elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
