#!/usr/bin/env python3
"""功能：复查和导出已经写入 pkl 的 L515 ego 点云。

这个脚本读取转换后 pkl 中的 point_cloud_ego 字段，打印每帧点云 shape、
dtype、xyz/rgb 范围和 x/y/z percentile；也可以把 pkl 内点云重新导出为
PLY 文件。它不负责从 depth 生成点云，主要用于验证转换结果、查看坐标范围，
并生成便于打开检查的可视化文件。
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path(
    "/share/project/liyuanyuan/data/dexglove_data/debug_l515_pointcloud_pkl"
)
DEFAULT_POINT_CLOUD_KEY = "point_cloud_ego"


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to unpickle arrays written by NumPy 2.x."""
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def episode_sort_key(path: Path) -> tuple[str, int, str]:
    parent = path.parent.name
    prefix = parent
    ep_idx = -1
    if "_ep" in parent:
        prefix, ep_text = parent.rsplit("_ep", 1)
        try:
            ep_idx = int(ep_text.split("_", 1)[0])
        except ValueError:
            ep_idx = -1
    return prefix, ep_idx, str(path)


def find_pkl_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("demo_*/*.pkl"), key=episode_sort_key)
    if not paths:
        paths = sorted(input_dir.rglob("*.pkl"), key=episode_sort_key)
    return paths


def load_pkl(path: Path) -> dict[str, Any]:
    install_numpy_pickle_compat()
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must be dict with list messages: {path}")
    return data


def point_cloud_from_message(
    message: dict[str, Any],
    *,
    camera: str,
    point_cloud_key: str,
) -> np.ndarray | None:
    l515 = message.get("l515")
    if not isinstance(l515, dict):
        return None
    fields = l515.get(camera)
    if not isinstance(fields, dict):
        return None
    value = fields.get(point_cloud_key)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(f"invalid point cloud shape: {arr.shape}")
    return arr


def write_ply(path: Path, point_cloud: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = point_cloud.shape[1] >= 6
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {point_cloud.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_color:
            colors = np.clip(point_cloud[:, 3:6], 0, 255).astype(np.uint8)
            for xyz, rgb in zip(point_cloud[:, :3], colors):
                f.write(
                    f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g} "
                    f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
                )
        else:
            for xyz in point_cloud[:, :3]:
                f.write(f"{xyz[0]:.7g} {xyz[1]:.7g} {xyz[2]:.7g}\n")


def fmt_min_max(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "empty"
    return f"[{float(arr.min()):.6g}, {float(arr.max()):.6g}]"


def print_percentiles(label: str, xyz: np.ndarray) -> None:
    if xyz.size == 0:
        print(f"{label} xyz percentiles [1,5,50,95,99]: empty")
        return
    values = np.percentile(xyz[:, :3], [1, 5, 50, 95, 99], axis=0)
    print(f"{label} xyz percentiles [1,5,50,95,99]:")
    for axis, axis_values in zip(("x", "y", "z"), values.T):
        print("  " + axis + ": [" + ", ".join(f"{float(v):.6g}" for v in axis_values) + "]")


def process_pkl(path: Path, args: argparse.Namespace) -> np.ndarray | None:
    data = load_pkl(path)
    messages = data["messages"]
    selected_messages = messages[: args.max_frames] if args.max_frames is not None else messages
    episode_name = path.parent.name
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = path.parent / "visualized_l515_pointcloud"
    else:
        out_dir = out_dir / episode_name

    print("=" * 100)
    print(f"pkl={path}")
    print(f"messages source={len(messages)} selected={len(selected_messages)}")
    all_xyz = []
    for frame_idx, message in enumerate(selected_messages):
        if not isinstance(message, dict):
            continue
        point_cloud = point_cloud_from_message(
            message,
            camera=args.camera,
            point_cloud_key=args.point_cloud_key,
        )
        if point_cloud is None:
            print(f"frame {frame_idx}: missing {args.point_cloud_key}")
            continue
        print(
            f"frame {frame_idx}: shape={point_cloud.shape} dtype={point_cloud.dtype} "
            f"xyz_range={fmt_min_max(point_cloud[:, :3])} "
            f"rgb_range={fmt_min_max(point_cloud[:, 3:6]) if point_cloud.shape[1] >= 6 else None}"
        )
        print_percentiles(f"frame {frame_idx}", point_cloud[:, :3])
        if args.save_ply:
            write_ply(out_dir / f"{args.camera}_frame_{frame_idx:06d}.ply", point_cloud)
        all_xyz.append(point_cloud[:, :3])
    if not all_xyz:
        return None
    xyz = np.concatenate(all_xyz, axis=0)
    print_percentiles(f"episode {episode_name}", xyz)
    if args.save_ply:
        print(f"saved ply dir: {out_dir}")
    return xyz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Point-cloud pkl root or pkl path.")
    parser.add_argument("--output_dir", type=Path, default=None, help="PLY output root.")
    parser.add_argument("--camera", default="ego", help="L515 camera name, default: ego.")
    parser.add_argument("--point_cloud_key", default=DEFAULT_POINT_CLOUD_KEY, help="Point-cloud field name.")
    parser.add_argument("--max_episodes", type=int, default=1, help="Max pkl episodes to process.")
    parser.add_argument("--max_frames", type=int, default=5, help="Max frames per pkl.")
    parser.add_argument("--save_ply", action="store_true", help="Export PLY files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    if input_path.is_file():
        pkl_paths = [input_path]
    else:
        pkl_paths = find_pkl_paths(input_path)
    if args.max_episodes is not None:
        pkl_paths = pkl_paths[: args.max_episodes]
    if not pkl_paths:
        raise SystemExit(f"no pkl files found under {input_path}")

    all_xyz = []
    for path in pkl_paths:
        xyz = process_pkl(path, args)
        if xyz is not None and xyz.size > 0:
            all_xyz.append(xyz)
    if all_xyz:
        print("=" * 100)
        print_percentiles("all selected point clouds", np.concatenate(all_xyz, axis=0))


if __name__ == "__main__":
    main()
