#!/usr/bin/env python3
"""功能：检查 DexUMI pkl 数据集格式，不修改任何原始数据。

这个脚本用于读取 episode pkl、l515_intrinsics.json，以及 pkl 中引用的
RGB/depth 图像，打印每个 episode/frame 的字段、shape、dtype、数值范围、
L515 相机命名方式、depth scale、内参和 alignTo 状态。它的作用是先确认
数据真实结构，避免在转换点云前假设字段名或图像路径。
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path(
    "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_aug/pick_3_crop"
)


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


def load_pkl(path: Path) -> Any:
    install_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def array_range(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    if np.issubdtype(arr.dtype, np.number):
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size == 0:
            return "all_nonfinite"
        return f"min={finite.min()} max={finite.max()}"
    return "non_numeric"


def describe_value(value: Any, *, indent: str = "", depth: int = 0, max_depth: int = 2) -> None:
    prefix = indent
    if isinstance(value, dict):
        keys = list(value.keys())
        print(f"{prefix}dict len={len(value)} keys={keys}")
        if depth >= max_depth:
            return
        for key in keys:
            print(f"{prefix}  {key}: ", end="")
            describe_value(value[key], indent=prefix + "  ", depth=depth + 1, max_depth=max_depth)
        return
    if isinstance(value, (list, tuple)):
        print(f"{type(value).__name__} len={len(value)}")
        if depth >= max_depth:
            return
        for idx, item in enumerate(value[:5]):
            print(f"{prefix}  [{idx}]: ", end="")
            describe_value(item, indent=prefix + "  ", depth=depth + 1, max_depth=max_depth)
        return
    if isinstance(value, np.ndarray):
        print(f"ndarray shape={value.shape} dtype={value.dtype} {array_range(value)}")
        return
    if value is None:
        print("None")
        return
    print(f"{type(value).__name__} value={repr(value)[:160]}")


def read_image(path: Path, *, unchanged: bool = False) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image)
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            return None
        if not unchanged and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def image_summary(label: str, path: Path, *, unchanged: bool = False) -> None:
    exists = path.exists()
    print(f"    {label} path={path} exists={exists}")
    if not exists:
        return
    arr = read_image(path, unchanged=unchanged)
    if arr is None:
        print(f"    {label} read_failed")
        return
    print(
        f"    {label} shape={arr.shape} dtype={arr.dtype} "
        f"{array_range(np.asarray(arr))}"
    )


def inspect_intrinsics(episode_dir: Path, camera: str | None) -> None:
    intr_path = episode_dir / "l515_intrinsics.json"
    if not intr_path.exists():
        print(f"intrinsics: missing {intr_path}")
        return
    with intr_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"intrinsics: {intr_path}")
    print(f"  top keys={list(data.keys())}")
    print(f"  depthMeters={data.get('depthMeters')}")
    cameras = data.get("cameras")
    if not isinstance(cameras, dict):
        print("  cameras: not a dict")
        return
    selected = [camera] if camera else sorted(cameras.keys())
    for cam in selected:
        info = cameras.get(cam)
        if not isinstance(info, dict):
            print(f"  camera {cam}: missing")
            continue
        print(
            f"  camera {cam}: keys={list(info.keys())} "
            f"alignTo={info.get('alignTo')} depthScaleMPerUnit={info.get('depthScaleMPerUnit')}"
        )
        for intr_key in ("colorIntrinsics", "depthIntrinsics"):
            intr = info.get(intr_key)
            if isinstance(intr, dict):
                print(
                    f"    {intr_key}: width={intr.get('width')} height={intr.get('height')} "
                    f"fx={intr.get('fx')} fy={intr.get('fy')} "
                    f"ppx={intr.get('ppx')} ppy={intr.get('ppy')} "
                    f"model={intr.get('model')}"
                )


def inspect_l515_references(pkl_path: Path, message: dict[str, Any], camera: str | None) -> None:
    l515 = message.get("l515")
    if not isinstance(l515, dict):
        print("  l515: missing or not a dict")
        return
    cameras = [camera] if camera else sorted(l515.keys())
    print(f"  l515 cameras={list(l515.keys())}")
    for cam in cameras:
        fields = l515.get(cam)
        if not isinstance(fields, dict):
            print(f"  l515/{cam}: missing or not a dict")
            continue
        print(f"  l515/{cam} keys={list(fields.keys())}")
        rgb_rel = fields.get("rgbImage")
        depth_rel = fields.get("depthImage")
        print(
            f"    colorFrameId={fields.get('colorFrameId')} "
            f"depthFrameId={fields.get('depthFrameId')} "
            f"depthIsNewFrame={fields.get('depthIsNewFrame')} "
            f"depthScaleMPerUnit={fields.get('depthScaleMPerUnit')}"
        )
        if isinstance(rgb_rel, str):
            image_summary("rgb", pkl_path.parent / rgb_rel, unchanged=False)
        else:
            print(f"    rgbImage={rgb_rel!r}")
        if isinstance(depth_rel, str):
            image_summary("depth", pkl_path.parent / depth_rel, unchanged=True)
        else:
            print(f"    depthImage={depth_rel!r}")


def inspect_pkl(path: Path, *, max_frames: int, camera: str | None) -> None:
    print("=" * 100)
    print(f"pkl: {path}")
    obj = load_pkl(path)
    describe_value(obj, max_depth=1)
    if not isinstance(obj, dict):
        return
    inspect_intrinsics(path.parent, camera)
    messages = obj.get("messages")
    if not isinstance(messages, list):
        print("messages: missing or not a list")
        return
    print(f"messages len={len(messages)}")
    for idx, message in enumerate(messages[:max_frames]):
        print("-" * 80)
        print(f"frame[{idx}]")
        describe_value(message, max_depth=2)
        if isinstance(message, dict):
            inspect_l515_references(path, message, camera)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Dataset root or pkl path.")
    parser.add_argument("--max_episodes", type=int, default=2, help="Max pkl episodes to inspect.")
    parser.add_argument("--max_frames", type=int, default=3, help="Max frames per pkl to inspect.")
    parser.add_argument(
        "--camera",
        default="ego",
        help="L515 camera to read for image summaries. Use 'all' to inspect all cameras.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    camera = None if str(args.camera).lower() == "all" else str(args.camera)
    if input_path.is_file():
        pkl_paths = [input_path]
    else:
        pkl_paths = find_pkl_paths(input_path)
    if args.max_episodes is not None:
        pkl_paths = pkl_paths[: args.max_episodes]
    if not pkl_paths:
        raise SystemExit(f"no pkl files found under {input_path}")
    print(f"input={input_path}")
    print(f"found_pkl={len(pkl_paths)} camera={camera or 'all'} max_frames={args.max_frames}")
    for path in pkl_paths:
        inspect_pkl(path, max_frames=args.max_frames, camera=camera)


if __name__ == "__main__":
    main()
