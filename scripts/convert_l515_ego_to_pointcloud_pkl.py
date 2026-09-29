#!/usr/bin/env python3
"""功能：把 DexUMI pkl 中的 L515 ego RGB-D 帧转换为点云并写入新 pkl。

这个脚本读取每帧 message["l515"]["ego"] 引用的 RGB/depth 文件，使用
l515_intrinsics.json 中的 pinhole 相机内参和 depth scale 将 depth 反投影
为 camera frame 下的 xyz 点云；如果 RGB 与 depth 已对齐且 shape 一致，
同时保存 xyzrgb，字段默认写入 point_cloud_ego。脚本只写到 output_dir，
不会修改原始数据，并可输出 PLY 与 RGB/depth/mask debug PNG 供人工检查
workspace、尺度和点云方向。
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path(
    "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_aug/pick_3_crop"
)
DEFAULT_OUTPUT = Path(
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


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


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


def dump_pkl(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_l515_intrinsics(episode_dir: Path, camera: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = episode_dir / "l515_intrinsics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing l515 intrinsics json: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cameras = data.get("cameras")
    if not isinstance(cameras, dict) or not isinstance(cameras.get(camera), dict):
        raise KeyError(f"missing camera '{camera}' in {path}")
    return data, cameras[camera]


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def read_depth(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image)


def copy_or_link(src: Path, dst: Path, *, overwrite: bool, hardlink: bool = True) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def resolve_depth_scale(
    *,
    cli_depth_scale: float | None,
    frame_fields: dict[str, Any],
    camera_intrinsics: dict[str, Any],
) -> float | None:
    if cli_depth_scale is not None:
        return float(cli_depth_scale)
    frame_scale = frame_fields.get("depthScaleMPerUnit")
    if frame_scale is not None:
        return float(frame_scale)
    json_scale = camera_intrinsics.get("depthScaleMPerUnit")
    if json_scale is not None:
        return float(json_scale)
    return None


def depth_to_meters(depth: np.ndarray, scale: float | None) -> np.ndarray:
    if scale is not None:
        return depth.astype(np.float32) * float(scale)
    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32, copy=False)
    raise ValueError("integer depth requires --depth_scale or depthScaleMPerUnit metadata")


def make_point_cloud(
    *,
    depth: np.ndarray,
    rgb: np.ndarray | None,
    intrinsics: dict[str, Any],
    depth_scale: float | None,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_m = depth_to_meters(depth, depth_scale)
    valid = np.isfinite(z_m) & (z_m > float(min_depth)) & (z_m < float(max_depth))
    ys, xs = np.nonzero(valid)
    if ys.size == 0:
        return np.empty((0, 6 if rgb is not None else 3), dtype=np.float32), z_m, valid

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["ppx"])
    cy = float(intrinsics["ppy"])

    zs = z_m[ys, xs].astype(np.float32, copy=False)
    x = (xs.astype(np.float32) - cx) * zs / fx
    y = (ys.astype(np.float32) - cy) * zs / fy
    xyz = np.column_stack((x, y, zs)).astype(np.float32, copy=False)

    if rgb is None:
        return xyz, z_m, valid

    colors = rgb[ys, xs, :3].astype(np.float32, copy=False)
    xyzrgb = np.column_stack((xyz, colors)).astype(np.float32, copy=False)
    return xyzrgb, z_m, valid


def finite_min_max(arr: np.ndarray) -> tuple[float, float] | None:
    if arr.size == 0:
        return None
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min()), float(arr.max())


def fmt_min_max(arr: np.ndarray) -> str:
    bounds = finite_min_max(arr)
    if bounds is None:
        return "empty"
    return f"[{bounds[0]:.6g}, {bounds[1]:.6g}]"


def xyz_percentiles(xyz: np.ndarray) -> dict[str, list[float]]:
    if xyz.size == 0:
        return {"x": [], "y": [], "z": []}
    values = np.percentile(xyz[:, :3], [1, 5, 50, 95, 99], axis=0)
    return {
        "x": [float(v) for v in values[:, 0]],
        "y": [float(v) for v in values[:, 1]],
        "z": [float(v) for v in values[:, 2]],
    }


def print_percentiles(label: str, xyz: np.ndarray) -> None:
    pct = xyz_percentiles(xyz)
    print(f"{label} xyz percentiles [1,5,50,95,99]:")
    for axis in ("x", "y", "z"):
        values = pct[axis]
        if not values:
            print(f"  {axis}: []")
        else:
            print("  " + axis + ": [" + ", ".join(f"{v:.6g}" for v in values) + "]")


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


def save_debug_pngs(
    *,
    out_prefix: Path,
    rgb: np.ndarray | None,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    max_depth: float,
) -> None:
    from PIL import Image

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    height, width = depth_m.shape

    if rgb is None:
        rgb_vis = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        rgb_vis = rgb.astype(np.uint8, copy=False)

    depth_vis = np.zeros((height, width), dtype=np.uint8)
    finite = np.isfinite(depth_m) & (depth_m > 0)
    if np.any(finite):
        scaled = np.clip(depth_m[finite] / float(max_depth), 0.0, 1.0)
        depth_vis[finite] = (scaled * 255.0).astype(np.uint8)
    mask_vis = (valid_mask.astype(np.uint8) * 255)

    Image.fromarray(rgb_vis).save(out_prefix.with_name(out_prefix.name + "_rgb.png"))
    Image.fromarray(depth_vis).save(out_prefix.with_name(out_prefix.name + "_depth.png"))
    Image.fromarray(mask_vis).save(out_prefix.with_name(out_prefix.name + "_mask.png"))

    panel = Image.new("RGB", (width * 3, height))
    panel.paste(Image.fromarray(rgb_vis), (0, 0))
    panel.paste(Image.fromarray(depth_vis).convert("RGB"), (width, 0))
    panel.paste(Image.fromarray(mask_vis).convert("RGB"), (width * 2, 0))
    panel.save(out_prefix.with_name(out_prefix.name + "_panel.png"))


def camera_fields_for_message(message: dict[str, Any], camera: str) -> dict[str, Any] | None:
    l515 = message.get("l515")
    if not isinstance(l515, dict):
        return None
    fields = l515.get(camera)
    return fields if isinstance(fields, dict) else None


def can_attach_rgb(
    *,
    rgb: np.ndarray | None,
    depth: np.ndarray,
    camera_info: dict[str, Any],
) -> tuple[bool, str]:
    if rgb is None:
        return False, "missing rgb"
    if rgb.shape[:2] != depth.shape[:2]:
        return False, f"rgb/depth shape mismatch: {rgb.shape[:2]} vs {depth.shape[:2]}"
    align_to = str(camera_info.get("alignTo", "")).lower()
    if align_to and align_to != "color":
        return False, f"alignTo is {align_to}, not color"
    return True, "rgb aligned to depth/color grid"


def process_frame(
    *,
    src_pkl: Path,
    out_episode: Path,
    frame_idx: int,
    message: dict[str, Any],
    camera: str,
    camera_info: dict[str, Any],
    depth_intrinsics: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray | None]:
    out_message = copy.deepcopy(message)
    fields = camera_fields_for_message(out_message, camera)
    src_fields = camera_fields_for_message(message, camera)
    if fields is None or src_fields is None:
        print(f"frame {frame_idx}: missing l515/{camera}, skipped")
        return out_message, None
    if args.point_cloud_key in fields and not args.overwrite_point_cloud:
        raise KeyError(
            f"frame {frame_idx} already has {args.point_cloud_key}; "
            "use --overwrite_point_cloud to replace it in output"
        )

    rgb_rel = src_fields.get("rgbImage")
    depth_rel = src_fields.get("depthImage")
    if not isinstance(depth_rel, str):
        print(f"frame {frame_idx}: missing depthImage, skipped")
        return out_message, None

    src_depth_path = src_pkl.parent / depth_rel
    out_depth_path = out_episode / depth_rel
    copy_or_link(src_depth_path, out_depth_path, overwrite=args.overwrite, hardlink=True)
    depth = read_depth(src_depth_path)

    rgb = None
    if isinstance(rgb_rel, str):
        src_rgb_path = src_pkl.parent / rgb_rel
        out_rgb_path = out_episode / rgb_rel
        copy_or_link(src_rgb_path, out_rgb_path, overwrite=args.overwrite, hardlink=True)
        rgb = read_rgb(src_rgb_path)

    use_rgb, rgb_reason = can_attach_rgb(rgb=rgb, depth=depth, camera_info=camera_info)
    if not use_rgb:
        rgb_for_cloud = None
    else:
        rgb_for_cloud = rgb

    depth_scale = resolve_depth_scale(
        cli_depth_scale=args.depth_scale,
        frame_fields=src_fields,
        camera_intrinsics=camera_info,
    )
    point_cloud, depth_m, valid_mask = make_point_cloud(
        depth=depth,
        rgb=rgb_for_cloud,
        intrinsics=depth_intrinsics,
        depth_scale=depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    fields[args.point_cloud_key] = point_cloud
    fields[args.point_cloud_key + "_columns"] = (
        ["x", "y", "z", "r", "g", "b"] if point_cloud.shape[1] >= 6 else ["x", "y", "z"]
    )
    fields[args.point_cloud_key + "_frame"] = "l515_ego_camera"

    debug_dir = out_episode / "debug_l515_pointcloud" / f"{camera}_frame_{frame_idx:06d}"
    if args.save_ply:
        write_ply(debug_dir.with_suffix(".ply"), point_cloud)
    if not args.no_debug_png:
        save_debug_pngs(
            out_prefix=debug_dir,
            rgb=rgb,
            depth_m=depth_m,
            valid_mask=valid_mask,
            max_depth=args.max_depth,
        )

    print(f"frame {frame_idx}:")
    print(f"  rgb shape={None if rgb is None else rgb.shape} range={None if rgb is None else fmt_min_max(rgb)}")
    print(
        f"  depth shape={depth.shape} dtype={depth.dtype} raw_range={fmt_min_max(depth)} "
        f"meters_range={fmt_min_max(depth_m)} depth_scale={depth_scale}"
    )
    print(f"  rgb usage={rgb_reason}")
    print(f"  point cloud shape={point_cloud.shape}")
    print(f"  xyz min/max={fmt_min_max(point_cloud[:, :3])}")
    print_percentiles(f"  frame {frame_idx}", point_cloud[:, :3])

    return out_message, point_cloud


def process_episode(src_pkl: Path, output_dir: Path, args: argparse.Namespace) -> np.ndarray | None:
    episode_dir = src_pkl.parent
    out_episode = output_dir / episode_dir.name
    if out_episode.exists():
        if args.overwrite:
            shutil.rmtree(out_episode)
        else:
            print(f"skip existing output episode: {out_episode}")
            return None
    out_episode.mkdir(parents=True, exist_ok=True)

    intr_json, camera_info = load_l515_intrinsics(episode_dir, args.camera)
    depth_intrinsics = camera_info.get("depthIntrinsics")
    if not isinstance(depth_intrinsics, dict):
        raise KeyError(f"missing depthIntrinsics for camera {args.camera}: {episode_dir}")

    intr_src = episode_dir / "l515_intrinsics.json"
    copy_or_link(intr_src, out_episode / intr_src.name, overwrite=args.overwrite, hardlink=True)

    data = load_pkl(src_pkl)
    messages = data["messages"]
    selected_messages = messages[: args.max_frames] if args.max_frames is not None else messages
    out_data = copy.deepcopy(data)
    out_messages = []
    episode_xyz = []
    used_rgb = False

    print("=" * 100)
    print(f"episode={episode_dir.name}")
    print(f"source_pkl={src_pkl}")
    print(f"output_episode={out_episode}")
    print(
        f"camera={args.camera} alignTo={camera_info.get('alignTo')} "
        f"depthScaleMPerUnit={camera_info.get('depthScaleMPerUnit')}"
    )
    print(f"depthIntrinsics={depth_intrinsics}")
    print(f"messages source={len(messages)} selected={len(selected_messages)}")

    for frame_idx, message in enumerate(selected_messages):
        if not isinstance(message, dict):
            out_messages.append(copy.deepcopy(message))
            continue
        out_message, point_cloud = process_frame(
            src_pkl=src_pkl,
            out_episode=out_episode,
            frame_idx=frame_idx,
            message=message,
            camera=args.camera,
            camera_info=camera_info,
            depth_intrinsics=depth_intrinsics,
            args=args,
        )
        out_messages.append(out_message)
        if point_cloud is not None and point_cloud.size > 0:
            used_rgb = used_rgb or point_cloud.shape[1] >= 6
            episode_xyz.append(point_cloud[:, :3])

    out_data["messages"] = out_messages
    metadata = out_data.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["l515PointCloudConversion"] = {
            "camera": args.camera,
            "pointCloudKey": args.point_cloud_key,
            "pointCloudFrame": "l515_ego_camera",
            "usesRgb": bool(used_rgb),
            "usesExtrinsics": False,
            "sourcePkl": str(src_pkl),
            "sourceMessageCount": len(messages),
            "outputMessageCount": len(out_messages),
            "depthScaleOverride": args.depth_scale,
            "minDepth": args.min_depth,
            "maxDepth": args.max_depth,
            "intrinsicsSource": "l515_intrinsics.json depthIntrinsics",
            "intrinsics": intr_json,
        }

    out_pkl = out_episode / src_pkl.name
    dump_pkl(out_data, out_pkl)
    print(f"saved pkl: {out_pkl}")

    if not episode_xyz:
        return None
    xyz_all = np.concatenate(episode_xyz, axis=0)
    print_percentiles(f"episode {episode_dir.name}", xyz_all)
    return xyz_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input dataset root or pkl path.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT, help="Output dataset root.")
    parser.add_argument("--camera", default="ego", help="L515 camera name, default: ego.")
    parser.add_argument("--max_episodes", type=int, default=1, help="Max episodes to process.")
    parser.add_argument("--max_frames", type=int, default=5, help="Max frames per episode.")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=None,
        help="Override depth scale in meters per raw unit. Default reads pkl/json metadata.",
    )
    parser.add_argument("--min_depth", type=float, default=0.05, help="Minimum valid depth in meters.")
    parser.add_argument("--max_depth", type=float, default=2.0, help="Maximum valid depth in meters.")
    parser.add_argument("--save_ply", action="store_true", help="Save per-frame PLY files.")
    parser.add_argument("--no_debug_png", action="store_true", help="Do not save RGB/depth/mask debug PNGs.")
    parser.add_argument("--point_cloud_key", default=DEFAULT_POINT_CLOUD_KEY, help="New field name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output episodes.")
    parser.add_argument(
        "--overwrite_point_cloud",
        action="store_true",
        help="Replace point_cloud_key if it already exists in output frame dict.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_path == output_dir:
        raise SystemExit("output_dir must be different from input")
    if input_path.is_file():
        pkl_paths = [input_path]
    else:
        pkl_paths = find_pkl_paths(input_path)
    if args.max_episodes is not None:
        pkl_paths = pkl_paths[: args.max_episodes]
    if not pkl_paths:
        raise SystemExit(f"no pkl files found under {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_xyz = []
    print(f"input={input_path}")
    print(f"output_dir={output_dir}")
    print(f"pkl_count={len(pkl_paths)}")
    for pkl_path in pkl_paths:
        xyz = process_episode(pkl_path, output_dir, args)
        if xyz is not None and xyz.size > 0:
            all_xyz.append(xyz)
    if all_xyz:
        print("=" * 100)
        print_percentiles("all processed frames", np.concatenate(all_xyz, axis=0))


if __name__ == "__main__":
    main()
