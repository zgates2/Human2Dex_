#!/usr/bin/env python3
"""Detect white glove regions and draw simple 2D contour keypoints.

This is a lightweight visual sanity-check tool for the pick_sponge data. It
does not run SAM 3D Body. The emitted points are pseudo keypoints estimated
from the white glove mask, useful for checking whether the glove is localized.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from common import (
    bbox_from_mask,
    feather_mask,
    list_episode_dirs,
    list_images,
    load_annotations,
    load_rgb,
    natural_key,
    normalize_bbox,
    segment_white_glove,
)


Point = Tuple[float, float]


def _point_dict(name: str, point: Point, score: float = 1.0) -> Dict:
    return {
        "name": name,
        "x": round(float(point[0]), 3),
        "y": round(float(point[1]), 3),
        "score": round(float(score), 4),
        "keypoint_type": "contour_pseudo_2d",
    }


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, iterations=1)
    boundary = mask & ~eroded
    ys, xs = np.nonzero(boundary)
    if xs.size == 0:
        ys, xs = np.nonzero(mask)
    return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _weighted_mean(points: np.ndarray, weights: Optional[np.ndarray] = None) -> Point:
    if points.size == 0:
        return 0.0, 0.0
    if weights is None or float(np.sum(weights)) <= 1e-6:
        out = np.mean(points, axis=0)
    else:
        out = np.sum(points * weights[:, None], axis=0) / float(np.sum(weights))
    return float(out[0]), float(out[1])


def _max_distance_point(mask: np.ndarray) -> Point:
    dist = ndimage.distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y)


def _extreme_point(points: np.ndarray, direction: np.ndarray) -> Point:
    projection = points @ direction.astype(np.float32)
    thresh = float(np.percentile(projection, 96.0))
    candidates = points[projection >= thresh]
    weights = projection[projection >= thresh] - thresh + 1e-3
    return _weighted_mean(candidates, weights)


def _select_tip_points(
    boundary: np.ndarray,
    palm: Point,
    max_tips: int = 5,
    min_separation: float = 28.0,
) -> List[Point]:
    if boundary.size == 0:
        return []

    palm_arr = np.asarray(palm, dtype=np.float32)
    vec = boundary - palm_arr[None, :]
    distance = np.linalg.norm(vec, axis=1)

    # In this dataset the glove usually enters from the upper image side and
    # fingers point downward. Keep a mild lower-half bias, but still allow other
    # orientations through the distance term.
    y_bias = np.maximum(boundary[:, 1] - palm_arr[1], 0.0) * 0.18
    score = distance + y_bias
    order = np.argsort(score)[::-1]

    selected: List[np.ndarray] = []
    for idx in order:
        pt = boundary[int(idx)]
        if any(float(np.linalg.norm(pt - old)) < min_separation for old in selected):
            continue
        selected.append(pt)
        if len(selected) >= max_tips:
            break

    selected_points = [(float(p[0]), float(p[1])) for p in selected]
    return sorted(selected_points, key=lambda p: p[0])


def estimate_glove_keypoints(mask: np.ndarray) -> List[Dict]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []

    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    boundary = _mask_boundary(mask)
    palm = _max_distance_point(mask)
    palm_arr = np.asarray(palm, dtype=np.float32)

    tips = _select_tip_points(
        boundary,
        palm,
        max_tips=5,
        min_separation=max(18.0, math.sqrt(float(mask.sum())) * 0.10),
    )

    keypoints: List[Dict] = [_point_dict("palm_center", palm)]
    for i, point in enumerate(tips, start=1):
        keypoints.append(_point_dict(f"tip_{i}", point))

    if tips:
        tip_center = np.mean(np.asarray(tips, dtype=np.float32), axis=0)
        wrist_direction = _normalize(palm_arr - tip_center)
        wrist = _extreme_point(boundary, wrist_direction)
    else:
        centered = points - np.mean(points, axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        wrist = _extreme_point(boundary, -_normalize(vh[0]))
    keypoints.append(_point_dict("wrist_center", wrist))

    keypoints.extend(
        [
            _point_dict("mask_left", (float(xs.min()), float(ys[np.argmin(xs)])), 0.75),
            _point_dict("mask_right", (float(xs.max()), float(ys[np.argmax(xs)])), 0.75),
            _point_dict("mask_top", (float(xs[np.argmin(ys)]), float(ys.min())), 0.75),
            _point_dict("mask_bottom", (float(xs[np.argmax(ys)]), float(ys.max())), 0.75),
        ]
    )
    return keypoints


def _draw_circle(draw: ImageDraw.ImageDraw, point: Point, color: Tuple[int, int, int], radius: int = 4):
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=1)


def draw_keypoint_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    keypoints: Sequence[Dict],
    bbox: Optional[Sequence[int]] = None,
    exclude_bboxes: Optional[Sequence[Sequence[int]]] = None,
) -> Image.Image:
    base = Image.fromarray(image.astype(np.uint8), mode="RGB")
    tint = Image.new("RGB", base.size, (30, 190, 255))
    alpha = Image.fromarray((feather_mask(mask, radius=2.0) * 105).astype(np.uint8), mode="L")
    out = Image.composite(tint, base, alpha)
    draw = ImageDraw.Draw(out)

    if bbox is not None:
        draw.rectangle(tuple(normalize_bbox(bbox)), outline=(20, 230, 80), width=2)
    for exclude_bbox in exclude_bboxes or []:
        draw.rectangle(tuple(normalize_bbox(exclude_bbox)), outline=(255, 170, 0), width=2)

    by_name = {kp["name"]: kp for kp in keypoints}
    palm = by_name.get("palm_center")
    if palm is not None:
        p0 = (float(palm["x"]), float(palm["y"]))
        for kp in keypoints:
            if kp["name"].startswith("tip_") or kp["name"] == "wrist_center":
                p1 = (float(kp["x"]), float(kp["y"]))
                draw.line((p0, p1), fill=(255, 255, 255), width=2)

    for kp in keypoints:
        point = (float(kp["x"]), float(kp["y"]))
        name = kp["name"]
        if name == "palm_center":
            color = (255, 60, 80)
            radius = 5
        elif name == "wrist_center":
            color = (255, 210, 40)
            radius = 5
        elif name.startswith("tip_"):
            color = (80, 255, 120)
            radius = 5
        else:
            color = (190, 120, 255)
            radius = 3
        _draw_circle(draw, point, color, radius)
        if name in {"palm_center", "wrist_center"} or name.startswith("tip_"):
            draw.text((point[0] + 6, point[1] - 6), name, fill=(255, 255, 255))

    return out


def _load_episode_roi(roi_data: Dict, episode_name: str, image_shape: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], List[Tuple[int, int, int, int]], bool]:
    item = roi_data.get("episodes", {}).get(episode_name)
    h, w = image_shape
    if not item:
        return (0, 0, w, h), [], False
    bbox = normalize_bbox(item["bbox"])
    exclude = [normalize_bbox(b) for b in item.get("exclude_bboxes", [])]
    return bbox, exclude, True


def _choose_episodes(input_root: Path, roi_data: Dict, requested: Sequence[str], limit: int) -> List[Path]:
    by_name = {p.name: p for p in list_episode_dirs(input_root)}
    if requested:
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise FileNotFoundError(f"episodes not found under {input_root}: {', '.join(missing)}")
        return [by_name[name] for name in requested]

    annotated = [
        by_name[name]
        for name in sorted(roi_data.get("episodes", {}).keys(), key=lambda n: natural_key(Path(n)))
        if name in by_name
    ]
    if annotated:
        return annotated[:limit]
    return list(by_name.values())[:limit]


def _sample_images(images: Sequence[Path], limit: int, stride: int) -> List[Path]:
    if stride <= 1:
        return list(images[:limit])
    return list(images[::stride][:limit])


def process_images(args: argparse.Namespace) -> Dict:
    input_root = Path(args.input)
    output_root = Path(args.output)
    roi_data = load_annotations(Path(args.roi))
    episodes = _choose_episodes(input_root, roi_data, args.episode, args.limit_episodes)

    annotated_dir = output_root / "annotated"
    mask_dir = output_root / "masks"
    records = []

    for episode_dir in episodes:
        images = _sample_images(list_images(episode_dir), args.limit_frames, args.stride)
        prev_mask = None
        current_bbox = None
        for image_path in images:
            rgb = load_rgb(image_path)
            bbox, exclude, has_roi = _load_episode_roi(roi_data, episode_dir.name, rgb.shape[:2])
            if current_bbox is not None and args.track_bbox:
                bbox = current_bbox

            mask, stats = segment_white_glove(
                rgb,
                bbox=bbox,
                exclude_bboxes=exclude,
                prev_mask=prev_mask,
                min_area=args.min_area,
                max_components=args.max_components,
                bbox_padding=args.bbox_padding,
            )
            keypoints = [] if stats.failed else estimate_glove_keypoints(mask)

            tight_bbox = bbox_from_mask(mask, padding=6)
            if tight_bbox is not None and args.track_bbox:
                current_bbox = tight_bbox
            prev_mask = mask

            rel_name = f"{episode_dir.name}/{image_path.stem}_keypoints.jpg"
            out_image_path = annotated_dir / rel_name
            out_image_path.parent.mkdir(parents=True, exist_ok=True)
            overlay = draw_keypoint_overlay(rgb, mask, keypoints, bbox=bbox, exclude_bboxes=exclude)
            overlay.save(out_image_path, quality=95)

            mask_path = None
            if args.save_masks:
                mask_path = mask_dir / episode_dir.name / f"{image_path.stem}_mask.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

            records.append(
                {
                    "episode": episode_dir.name,
                    "image": str(image_path),
                    "annotated_image": str(out_image_path),
                    "mask_image": str(mask_path) if mask_path is not None else None,
                    "has_roi": has_roi,
                    "roi_bbox": [int(v) for v in bbox],
                    "mask_bbox": [int(v) for v in tight_bbox] if tight_bbox is not None else None,
                    "mask_area": stats.area,
                    "mask_area_ratio": round(stats.area_ratio, 6),
                    "failed": stats.failed,
                    "failure_reason": stats.reason,
                    "keypoints": keypoints,
                }
            )

    result = {
        "input": str(input_root),
        "output": str(output_root),
        "roi": str(args.roi),
        "note": "Pseudo 2D contour keypoints from white-glove segmentation; not SAM3D/MANO joints.",
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "keypoints.json"
    tmp_path = json_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    tmp_path.replace(json_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/home/zjc/Desktop/human2dex/data/pick_sponge")
    parser.add_argument("--output", default="/home/zjc/Desktop/human2dex/data/pick_sponge_glove_keypoint_demo")
    parser.add_argument("--roi", default="/home/zjc/Desktop/human2dex/glove_aug_pipeline/roi_annotations.json")
    parser.add_argument("--episode", action="append", default=[], help="Episode name. Repeat to process multiple episodes.")
    parser.add_argument("--limit-episodes", type=int, default=1)
    parser.add_argument("--limit-frames", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=180)
    parser.add_argument("--max-components", type=int, default=1)
    parser.add_argument("--bbox-padding", type=int, default=0)
    parser.add_argument("--track-bbox", action="store_true", help="Update bbox from the previous detected mask.")
    parser.add_argument("--save-masks", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = process_images(args)
    print(f"Wrote {len(result['records'])} records to {Path(args.output) / 'keypoints.json'}")
    for record in result["records"][:8]:
        print(
            f"{record['episode']} {Path(record['image']).name}: "
            f"failed={record['failed']} keypoints={len(record['keypoints'])} "
            f"annotated={record['annotated_image']}"
        )


if __name__ == "__main__":
    main()
