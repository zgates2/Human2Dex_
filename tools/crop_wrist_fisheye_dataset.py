#!/usr/bin/env python3
"""Crop black borders from wrist fisheye images in a DexUMI pkl dataset.

The script writes a new dataset root. Episode names, file names, pkl files,
and non-wrist camera data are copied unchanged. Images under each episode's
``wrist/`` directory are cropped to a square fisheye view and resized back to
the original square resolution by default.


python /home/zjc/Desktop/human2dex/tools/crop_wrist_fisheye_dataset.py \
    --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_aug/pick_3_raw_wrist_glove_aug \
    --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_aug/pick_3_crop \
    --workers 16 \
    --hardlink \
    --overwrite

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


DEFAULT_INPUT = Path(
    "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def default_output_root(input_root: Path) -> Path:
    return input_root.parent / f"{input_root.name}_wrist_fisheye_crop"


def iter_episode_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=natural_key)


def list_wrist_images(episode_dir: Path) -> List[Path]:
    wrist_dir = episode_dir / "wrist"
    if not wrist_dir.is_dir():
        return []
    return sorted(
        [p for p in wrist_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def evenly_spaced(items: Sequence[Path], count: int) -> List[Path]:
    if count <= 0 or count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    idxs = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[int(i)] for i in idxs]


def detect_nonblack_bbox(
    image_path: Path,
    black_threshold: int,
    min_axis_occupancy: float,
) -> Tuple[int, int, int, int, int, int]:
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)
    mask = gray > black_threshold
    min_col_pixels = max(2, int(round(h * min_axis_occupancy)))
    min_row_pixels = max(2, int(round(w * min_axis_occupancy)))

    cols = np.flatnonzero(mask.sum(axis=0) >= min_col_pixels)
    rows = np.flatnonzero(mask.sum(axis=1) >= min_row_pixels)
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, w, h, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1), w, h


def square_crop_from_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    padding: int,
) -> Tuple[int, int, int]:
    x1, y1, x2, y2 = bbox
    w, h = image_size
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = min(bw, bh) + 2 * padding
    side = max(8, min(side, w, h))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))
    return left, top, int(side)


def estimate_episode_crop(
    images: Sequence[Path],
    sample_frames: int,
    black_threshold: int,
    min_axis_occupancy: float,
    padding: int,
) -> Dict[str, Any]:
    samples = evenly_spaced(images, sample_frames)
    boxes = []
    image_sizes = []
    for image_path in samples:
        x1, y1, x2, y2, w, h = detect_nonblack_bbox(
            image_path,
            black_threshold=black_threshold,
            min_axis_occupancy=min_axis_occupancy,
        )
        crop = square_crop_from_bbox((x1, y1, x2, y2), (w, h), padding=padding)
        boxes.append(crop)
        image_sizes.append((w, h))

    if not boxes:
        raise ValueError("no images available for crop estimation")

    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    image_w = int(round(float(np.median(widths))))
    image_h = int(round(float(np.median(heights))))
    left = int(round(float(np.median([b[0] for b in boxes]))))
    top = int(round(float(np.median([b[1] for b in boxes]))))
    side = int(round(float(np.median([b[2] for b in boxes]))))
    side = max(8, min(side, image_w, image_h))
    left = max(0, min(left, image_w - side))
    top = max(0, min(top, image_h - side))
    return {
        "left": left,
        "top": top,
        "side": side,
        "input_width": image_w,
        "input_height": image_h,
        "sample_count": len(samples),
    }


def copy_file(src: Path, dst: Path, overwrite: bool, hardlink: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_non_wrist_image_files(
    episode_dir: Path,
    out_episode: Path,
    overwrite: bool,
    hardlink: bool,
) -> int:
    copied = 0
    for src_path in sorted(episode_dir.rglob("*")):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(episode_dir)
        if len(rel.parts) >= 2 and rel.parts[0] == "wrist" and src_path.suffix.lower() in IMAGE_EXTENSIONS:
            continue
        copy_file(src_path, out_episode / rel, overwrite=overwrite, hardlink=hardlink)
        copied += 1
    return copied


def crop_one_image(
    src: Path,
    dst: Path,
    crop: Dict[str, Any],
    output_size: Optional[int],
    quality: int,
    overwrite: bool,
) -> None:
    if dst.exists() and not overwrite:
        return
    with Image.open(src) as img:
        img = img.convert("RGB")
        left = int(crop["left"])
        top = int(crop["top"])
        side = int(crop["side"])
        cropped = img.crop((left, top, left + side, top + side))
        target_size = output_size
        if target_size is None:
            target_size = min(img.size)
        if target_size > 0 and target_size != side:
            cropped = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        suffix = dst.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            cropped.save(dst, quality=quality)
        else:
            cropped.save(dst)


def process_episode(task: Tuple[Path, Path, Dict[str, Any]]) -> Dict[str, Any]:
    episode_dir, output_root, options = task
    overwrite = bool(options["overwrite"])
    hardlink = bool(options["hardlink"])
    dry_run = bool(options["dry_run"])
    sample_frames = int(options["sample_frames"])
    black_threshold = int(options["black_threshold"])
    min_axis_occupancy = float(options["min_axis_occupancy"])
    padding = int(options["padding"])
    output_size = options["output_size"]
    quality = int(options["quality"])

    images = list_wrist_images(episode_dir)
    out_episode = output_root / episode_dir.name
    result: Dict[str, Any] = {
        "episode": episode_dir.name,
        "skipped": False,
        "reason": "",
        "wrist_images": len(images),
        "copied_files": 0,
        "crop": None,
    }

    if not images:
        result.update(skipped=True, reason="missing wrist images")
        return result
    if out_episode.exists() and not overwrite:
        result.update(skipped=True, reason="output episode exists")
        return result

    crop = estimate_episode_crop(
        images,
        sample_frames=sample_frames,
        black_threshold=black_threshold,
        min_axis_occupancy=min_axis_occupancy,
        padding=padding,
    )
    result["crop"] = crop

    if dry_run:
        return result

    if out_episode.exists() and overwrite:
        shutil.rmtree(out_episode)
    out_episode.mkdir(parents=True, exist_ok=True)

    result["copied_files"] = copy_non_wrist_image_files(
        episode_dir,
        out_episode,
        overwrite=overwrite,
        hardlink=hardlink,
    )
    for image_path in images:
        rel = image_path.relative_to(episode_dir)
        crop_one_image(
            image_path,
            out_episode / rel,
            crop=crop,
            output_size=output_size,
            quality=quality,
            overwrite=overwrite,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop wrist fisheye images in a DexUMI-style pkl dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset root. Default: <input>_wrist_fisheye_crop.",
    )
    parser.add_argument("--episode", action="append", default=[], help="Only process this episode. Repeatable.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Process at most N episodes.")
    parser.add_argument("--sample-frames", type=int, default=16, help="Frames sampled per episode to estimate one crop.")
    parser.add_argument("--black-threshold", type=int, default=8, help="Pixels brighter than this are treated as fisheye content.")
    parser.add_argument(
        "--min-axis-occupancy",
        type=float,
        default=0.01,
        help="Minimum non-black occupancy per row/column when finding outer fisheye bounds.",
    )
    parser.add_argument("--padding", type=int, default=0, help="Pixels added around the estimated circle crop.")
    parser.add_argument(
        "--output-size",
        type=int,
        default=-1,
        help="Output square size. -1 keeps original image size; 0 keeps crop size; positive value resizes to that size.",
    )
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality for cropped wrist images.")
    parser.add_argument("--workers", type=int, default=8, help="Episode-level worker processes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output episodes.")
    parser.add_argument("--hardlink", action="store_true", help="Hardlink copied non-wrist files when possible.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate crops without writing files.")
    return parser


def output_size_from_arg(value: int) -> Optional[int]:
    if value < 0:
        return None
    return value


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input.expanduser().resolve()
    output_root = args.output
    if output_root is None:
        output_root = default_output_root(input_root)
    output_root = output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise SystemExit(f"input dataset root not found: {input_root}")
    if input_root == output_root:
        raise SystemExit("output must be different from input")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.output_size < -1:
        raise SystemExit("--output-size must be -1, 0, or a positive integer")

    episodes = iter_episode_dirs(input_root)
    if args.episode:
        wanted = set(args.episode)
        episodes = [p for p in episodes if p.name in wanted]
        missing = sorted(wanted - {p.name for p in episodes})
        if missing:
            raise SystemExit(f"episodes not found: {', '.join(missing)}")
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]

    options = {
        "overwrite": args.overwrite,
        "hardlink": args.hardlink,
        "dry_run": args.dry_run,
        "sample_frames": args.sample_frames,
        "black_threshold": args.black_threshold,
        "min_axis_occupancy": args.min_axis_occupancy,
        "padding": args.padding,
        "output_size": output_size_from_arg(args.output_size),
        "quality": args.quality,
    }

    print(f"input: {input_root}")
    print(f"output: {output_root}")
    print("output episode structure:")
    print("  <output>/<episode>/wrist/<cropped original filenames>")
    print("  <output>/<episode>/<all original non-wrist files/directories copied unchanged>")
    if args.output_size == -1:
        print("output image size: original wrist image size")
    elif args.output_size == 0:
        print("output image size: crop size")
    else:
        print(f"output image size: {args.output_size}x{args.output_size}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    tasks = [(ep, output_root, options) for ep in episodes]
    results: List[Dict[str, Any]] = []
    if args.workers == 1 or len(tasks) <= 1:
        for task in tasks:
            results.append(process_episode(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(process_episode, task): task[0].name for task in tasks}
            for future in as_completed(future_map):
                results.append(future.result())

    results.sort(key=lambda item: item["episode"])
    processed = 0
    skipped = 0
    total_images = 0
    for item in results:
        if item.get("skipped"):
            skipped += 1
            print(f"SKIP {item['episode']}: {item['reason']}")
            continue
        processed += 1
        total_images += int(item["wrist_images"])
        crop = item["crop"]
        print(
            f"OK {item['episode']}: wrist_images={item['wrist_images']} "
            f"crop=({crop['left']},{crop['top']},{crop['side']}) "
            f"input={crop['input_width']}x{crop['input_height']}"
        )

    summary = {
        "input": str(input_root),
        "output": str(output_root),
        "dry_run": args.dry_run,
        "processed": processed,
        "skipped": skipped,
        "wrist_images": total_images,
        "options": {
            "sample_frames": args.sample_frames,
            "black_threshold": args.black_threshold,
            "min_axis_occupancy": args.min_axis_occupancy,
            "padding": args.padding,
            "output_size": args.output_size,
            "quality": args.quality,
        },
        "episodes": results,
    }
    if not args.dry_run:
        with (output_root / "wrist_fisheye_crop_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
    print(
        f"done: processed={processed} skipped={skipped} "
        f"wrist_images={total_images} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
