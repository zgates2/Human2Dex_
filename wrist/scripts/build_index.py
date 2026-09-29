#!/usr/bin/env python3
"""
构建 wrist pose baseline 的 train/val/test JSONL 数据索引。

用途
----
从 DexUMI 原始 episode PKL 中提取可训练样本。每个有效样本包含：
  - image_path: wrist RGB 图像绝对路径
  - pts21_mano: MediaPipe 21 点 hand-local / wrist-centered 3D pseudo-GT
  - valid_mask: 21 个关键点是否有效
  - sequence_id: episode 标识
  - frame_idx / timestamp / 同步相关字段

脚本会按 episode 划分 train / val / test，避免同一个 episode 同时出现在不同
split 中造成数据泄漏。

默认输入
--------
  /home/zjc/Desktop/human2dex/data/pick_2/**/*.pkl

默认输出
--------
  /home/zjc/Desktop/human2dex/wrist/outputs/index/train.jsonl
  /home/zjc/Desktop/human2dex/wrist/outputs/index/val.jsonl
  /home/zjc/Desktop/human2dex/wrist/outputs/index/test.jsonl
  /home/zjc/Desktop/human2dex/wrist/outputs/index/summary.json

常用命令
--------
构建完整索引：
  python wrist/scripts/build_index.py \
      --data-root /home/zjc/Desktop/human2dex/data/pick_2 \
      --output-dir /home/zjc/Desktop/human2dex/wrist/outputs/index

只检查前 3 个 episode：
  python wrist/scripts/build_index.py --limit-episodes 3

跳过图片可读性检查，加快索引生成：
  python wrist/scripts/build_index.py --no-check-readable

关键检查
--------
  - pts21_mano 必须是 shape=(21, 3)
  - pts21_mano 必须全 finite
  - rgbImage 必须存在且图片可读
  - pts21_mano[0] 应接近 [0, 0, 0]，异常会写入 summary warning
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "wrist") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "wrist"))

from wrist_pose.utils import write_jsonl  # noqa: E402


def install_numpy_pickle_compat() -> list[str]:
    """Temporarily expose NumPy 2 pickle module names for NumPy 1.x."""
    core = getattr(np, "core", None)
    aliases = {
        "numpy._core": core,
        "numpy._core.multiarray": getattr(core, "multiarray", None) if core is not None else None,
        "numpy._core.numeric": getattr(core, "numeric", None) if core is not None else None,
    }
    added: list[str] = []
    for name, module in aliases.items():
        if module is not None and name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    return added


def restore_numpy_pickle_compat(added_modules: list[str]) -> None:
    for name in reversed(added_modules):
        sys.modules.pop(name, None)


def read_pkl(path: Path) -> dict[str, Any]:
    added_modules = install_numpy_pickle_compat()
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    finally:
        restore_numpy_pickle_compat(added_modules)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict/messages: {path}")
    return data


def episode_sort_key(path: Path) -> tuple[str, int, str]:
    parent = path.parent.name
    prefix = parent
    ep_idx = -1
    if "_ep" in parent:
        prefix, ep_text = parent.rsplit("_ep", 1)
        try:
            ep_idx = int(ep_text)
        except ValueError:
            ep_idx = -1
    elif parent.startswith("episode_"):
        prefix = parent.rsplit("_", 1)[0]
        try:
            ep_idx = int(parent.rsplit("_", 1)[1])
        except ValueError:
            ep_idx = -1
    return prefix, ep_idx, str(path)


def iter_pkl_paths(data_root: Path) -> list[Path]:
    paths = sorted(data_root.glob("**/*.pkl"), key=episode_sort_key)
    return [path for path in paths if path.is_file()]


def valid_pts21(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (21, 3):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def valid_mask_from_message(msg: dict[str, Any], pts21: np.ndarray) -> list[bool]:
    value = msg.get("valid_mask")
    if value is not None:
        arr = np.asarray(value).astype(bool)
        if arr.shape == (21,):
            return arr.tolist()
    return np.all(np.isfinite(pts21), axis=1).tolist()


def image_is_readable(path: Path, check_readable: bool) -> bool:
    if not path.is_file():
        return False
    if not check_readable:
        return True
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required when --check-readable is enabled") from exc
    return cv2.imread(str(path), cv2.IMREAD_COLOR) is not None


def collect_episode_rows(
    pkl_path: Path,
    data_root: Path,
    check_readable: bool,
    wrist_zero_tol: float,
) -> tuple[list[dict[str, Any]], Counter[str], float]:
    data = read_pkl(pkl_path)
    messages = data["messages"]
    rows: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    max_wrist_abs = 0.0

    for frame_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            skips["message_not_dict"] += 1
            continue
        pts21 = valid_pts21(msg.get("pts21_mano"))
        if pts21 is None:
            skips["invalid_pts21_mano"] += 1
            continue
        max_wrist_abs = max(max_wrist_abs, float(np.max(np.abs(pts21[0]))))
        if float(np.max(np.abs(pts21[0]))) > wrist_zero_tol:
            skips["wrist_not_zero_warning"] += 1

        rel_image = msg.get("rgbImage")
        if not isinstance(rel_image, str) or not rel_image:
            skips["missing_rgbImage"] += 1
            continue
        image_path = (pkl_path.parent / rel_image).resolve()
        if not image_is_readable(image_path, check_readable):
            skips["image_not_readable"] += 1
            continue

        rows.append(
            {
                "image_path": str(image_path),
                "pts21_mano": pts21.tolist(),
                "valid_mask": valid_mask_from_message(msg, pts21),
                "sequence_id": str(pkl_path.parent.relative_to(data_root)),
                "frame_idx": int(frame_idx),
                "timestamp": float(msg.get("timestamp", frame_idx)),
                "mainClockMonotonicNs": msg.get("mainClockMonotonicNs"),
                "sampleClockNs": msg.get("sampleClockNs"),
                "rgb_align_residual_ns": msg.get("rgbAlignResidualNs"),
                "source_pkl": str(pkl_path.resolve()),
            }
        )

    return rows, skips, max_wrist_abs


def split_episodes(
    episodes: list[tuple[Path, list[dict[str, Any]]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    shuffled = episodes[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n >= 3:
        n_train = max(1, min(n - 2, n_train))
        n_val = max(1, min(n - n_train - 1, n_val))
    n_test = max(0, n - n_train - n_val)
    if n_test == 0 and n >= 3:
        n_train = max(1, n_train - 1)
        n_test = 1

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    return {
        name: [row for _, rows in split_eps for row in rows]
        for name, split_eps in splits.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "pick_2")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "wrist" / "outputs" / "index")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--wrist-zero-tol", type=float, default=1e-4)
    parser.add_argument("--check-readable", action="store_true", default=True)
    parser.add_argument("--no-check-readable", dest="check_readable", action="store_false")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    pkl_paths = iter_pkl_paths(data_root)
    if args.limit_episodes is not None:
        pkl_paths = pkl_paths[: int(args.limit_episodes)]
    if not pkl_paths:
        raise SystemExit(f"no PKL files found under {data_root}")

    episodes: list[tuple[Path, list[dict[str, Any]]]] = []
    skip_totals: Counter[str] = Counter()
    episode_stats = []
    for pkl_path in pkl_paths:
        rows, skips, max_wrist_abs = collect_episode_rows(
            pkl_path=pkl_path,
            data_root=data_root,
            check_readable=args.check_readable,
            wrist_zero_tol=float(args.wrist_zero_tol),
        )
        skip_totals.update(skips)
        if rows:
            episodes.append((pkl_path, rows))
        episode_stats.append(
            {
                "pkl": str(pkl_path),
                "valid_frames": len(rows),
                "max_abs_wrist_m": max_wrist_abs,
                "skips": dict(skips),
            }
        )

    if not episodes:
        raise SystemExit("no valid frames found")

    splits = split_episodes(
        episodes,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "episodes_found": len(pkl_paths),
        "episodes_with_valid_frames": len(episodes),
        "frames": {name: len(rows) for name, rows in splits.items()},
        "skip_totals": dict(skip_totals),
        "episode_stats": episode_stats,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "episode_stats"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
