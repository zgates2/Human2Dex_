#!/usr/bin/env python3
"""Render wrist 2D projection skeleton overlays from DexUMI PKLs.

This is a QC utility for outputs written by tools/add_wrist_predictions_to_pkl.py:

  rgbImage + wrist_uv21_rgb + wrist_uv21_valid -> overlay jpg/png

It does not modify PKLs or source images.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HAND_BONES: tuple[tuple[int, int], ...] = (
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
)

JOINT_COLORS_RGB = [
    (245, 245, 245),
    *((255, 180, 80) for _ in range(4)),
    *((80, 255, 120) for _ in range(4)),
    *((80, 210, 255) for _ in range(4)),
    *((255, 120, 180) for _ in range(4)),
    *((180, 120, 255) for _ in range(4)),
]


def natural_key(path: Path | str) -> list[Any]:
    parts = re.split(r"(\d+)", Path(path).name)
    return [int(part) if part.isdigit() else part for part in parts]


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
        raise ValueError(f"PKL must contain dict/messages: {path}")
    return data


def list_episode_dirs(root: Path, episodes: Sequence[str] | None = None) -> list[Path]:
    root = root.expanduser().resolve()
    requested = set(episodes or [])
    if (root / "images").is_dir():
        if requested and root.name not in requested:
            raise FileNotFoundError(f"requested episodes do not include input episode dir: {root.name}")
        return [root]
    dirs = [
        child
        for child in sorted(root.iterdir(), key=natural_key)
        if child.is_dir() and (child / "images").is_dir() and (not requested or child.name in requested)
    ]
    if requested:
        found = {path.name for path in dirs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"episodes not found under {root}: {missing}")
    return dirs


def first_pkl(episode_dir: Path) -> Path | None:
    paths = sorted(episode_dir.glob("*.pkl"), key=natural_key)
    return paths[0] if paths else None


def resolve_image_path(pkl_path: Path, msg: dict[str, Any], image_field: str) -> Path | None:
    value = msg.get(image_field)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    episode_dir = pkl_path.parent
    candidates = [
        episode_dir / path,
        episode_dir / "images" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def valid_uv_mask(uv21: np.ndarray, valid: Any, width: int, height: int) -> np.ndarray:
    arr = np.asarray(uv21, dtype=np.float32)
    mask = np.isfinite(arr).all(axis=1)
    mask &= arr[:, 0] >= 0
    mask &= arr[:, 0] < width
    mask &= arr[:, 1] >= 0
    mask &= arr[:, 1] < height
    if valid is not None:
        valid_arr = np.asarray(valid, dtype=bool).reshape(-1)
        if valid_arr.shape[0] == 21:
            mask &= valid_arr
    return mask


def draw_overlay(
    image_rgb: np.ndarray,
    uv21: np.ndarray,
    valid: Any,
    *,
    line_width: int,
    point_radius: int,
) -> np.ndarray:
    out = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    width, height = out.size
    arr = np.asarray(uv21, dtype=np.float32).reshape(21, 2)
    mask = valid_uv_mask(arr, valid, width, height)

    for a, b in HAND_BONES:
        if not (mask[a] and mask[b]):
            continue
        pa = tuple(float(v) for v in arr[a])
        pb = tuple(float(v) for v in arr[b])
        draw.line([pa, pb], fill=JOINT_COLORS_RGB[b], width=max(1, int(line_width)))

    radius = max(1, int(point_radius))
    for idx, uv in enumerate(arr):
        if not mask[idx]:
            continue
        x, y = float(uv[0]), float(uv[1])
        color = JOINT_COLORS_RGB[idx]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    return np.asarray(out, dtype=np.uint8)


def choose_frame_indices(n: int, max_frames: int | None, stride: int) -> list[int]:
    indices = list(range(0, n, max(1, int(stride))))
    if max_frames is not None and max_frames > 0 and len(indices) > max_frames:
        indices = np.linspace(0, len(indices) - 1, int(max_frames)).round().astype(int).tolist()
        base = list(range(0, n, max(1, int(stride))))
        indices = [base[i] for i in indices]
    return indices


def process_episode(args: argparse.Namespace, episode_dir: Path, output_root: Path) -> dict[str, Any]:
    pkl_path = first_pkl(episode_dir)
    if pkl_path is None:
        return {"episode": episode_dir.name, "error": "missing_pkl", "written": 0}
    data = read_pkl(pkl_path)
    messages = data["messages"]
    frame_indices = choose_frame_indices(len(messages), args.max_frames_per_episode, args.stride)
    out_dir = output_root / episode_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(messages):
            continue
        msg = messages[frame_idx]
        if not isinstance(msg, dict):
            stats["message_not_dict"] += 1
            continue
        uv = msg.get(args.uv_field)
        if uv is None:
            stats["missing_uv"] += 1
            continue
        try:
            uv_arr = np.asarray(uv, dtype=np.float32).reshape(21, 2)
        except Exception:
            stats["bad_uv_shape"] += 1
            continue
        image_path = resolve_image_path(pkl_path, msg, args.image_field)
        if image_path is None:
            stats["missing_image_field"] += 1
            continue
        if not image_path.is_file():
            stats["missing_image_file"] += 1
            if len(records) < 5:
                records.append({"frame_idx": frame_idx, "missing_image": str(image_path)})
            continue
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        valid = msg.get(args.valid_field)
        overlay = draw_overlay(
            rgb,
            uv_arr,
            valid,
            line_width=int(args.line_width),
            point_radius=int(args.point_radius),
        )
        out_name = f"{frame_idx:06d}_{image_path.stem}_wrist_projection.jpg"
        out_path = out_dir / out_name
        if out_path.exists() and not args.overwrite:
            stats["skip_existing"] += 1
            continue
        Image.fromarray(overlay).save(out_path, quality=int(args.jpeg_quality))
        stats["written"] += 1
        if len(records) < 10:
            records.append(
                {
                    "frame_idx": frame_idx,
                    "image": str(image_path),
                    "output": str(out_path),
                }
            )

    return {
        "episode": episode_dir.name,
        "pkl": str(pkl_path),
        "messages": len(messages),
        "selected": len(frame_indices),
        **dict(stats),
        "samples": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--image-field", default="rgbImage")
    parser.add_argument("--uv-field", default="wrist_uv21_rgb")
    parser.add_argument("--valid-field", default="wrist_uv21_valid")
    parser.add_argument("--max-frames-per-episode", type=int, default=80)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    episodes = list_episode_dirs(args.data_root, args.episode)
    if not episodes:
        raise SystemExit(f"no episodes found under {args.data_root}")
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [process_episode(args, episode_dir, output_root) for episode_dir in episodes]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "output_dir": str(output_root),
        "episodes": rows,
        "total_written": int(sum(int(row.get("written", 0)) for row in rows)),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
