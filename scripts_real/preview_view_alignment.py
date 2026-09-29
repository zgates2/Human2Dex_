#!/usr/bin/env python3
"""Headless preview tool for deployment-side RGB view alignment."""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import cv2
import numpy as np
import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_inference_config import load_yaml_mapping
from real_inference_view_alignment import ViewCanonicalizer


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply view_alignment to saved RGB images without loading a policy "
            "or connecting to robot hardware."
        )
    )
    parser.add_argument("--config", default="eval_franka_o6_config.yaml")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--rotation-deg", type=float)
    parser.add_argument("--tx-ratio", type=float)
    parser.add_argument("--ty-ratio", type=float)
    parser.add_argument("--pivot-x-ratio", type=float)
    parser.add_argument("--pivot-y-ratio", type=float)
    return parser.parse_args()


def _find_images(path: pathlib.Path, limit: int) -> list[pathlib.Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit > 0:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    return images


def _override_config(config: dict, args) -> dict:
    result = copy.deepcopy(config.get("view_alignment", {}) or {})
    result["enabled"] = True
    result.setdefault("transform_type", "similarity")
    if result["transform_type"] != "similarity" and any(
        value is not None
        for value in (
            args.scale,
            args.rotation_deg,
            args.tx_ratio,
            args.ty_ratio,
            args.pivot_x_ratio,
            args.pivot_y_ratio,
        )
    ):
        raise ValueError("CLI parameter overrides require transform_type=similarity")

    if args.scale is not None:
        result["scale"] = args.scale
    if args.rotation_deg is not None:
        result["rotation_deg"] = args.rotation_deg

    translation = list(result.get("translation_ratio", [0.0, 0.0]))
    if args.tx_ratio is not None:
        translation[0] = args.tx_ratio
    if args.ty_ratio is not None:
        translation[1] = args.ty_ratio
    result["translation_ratio"] = translation

    pivot = list(result.get("pivot_ratio", [0.5, 0.5]))
    if args.pivot_x_ratio is not None:
        pivot[0] = args.pivot_x_ratio
    if args.pivot_y_ratio is not None:
        pivot[1] = args.pivot_y_ratio
    result["pivot_ratio"] = pivot
    return result


def _label(image_bgr: np.ndarray, text: str) -> np.ndarray:
    output = image_bgr.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(
        output,
        text,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def main():
    args = _parse_args()
    cfg = load_yaml_mapping(args.config)
    alignment_cfg = _override_config(cfg, args)
    canonicalizer = ViewCanonicalizer(alignment_cfg)

    input_path = pathlib.Path(args.input).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    aligned_dir = output_dir / "aligned"
    compare_dir = output_dir / "compare"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    images = _find_images(input_path, args.max_images)
    first_size = None
    for index, image_path in enumerate(images):
        raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise IOError(f"cv2.imread failed: {image_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        aligned_rgb = canonicalizer.apply_image(raw_rgb)
        aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        if first_size is None:
            first_size = (raw_bgr.shape[1], raw_bgr.shape[0])

        output_name = f"{index:04d}_{image_path.stem}.jpg"
        if not cv2.imwrite(str(aligned_dir / output_name), aligned_bgr):
            raise IOError(f"cv2.imwrite failed: {aligned_dir / output_name}")
        comparison = np.concatenate(
            [_label(raw_bgr, "RAW"), _label(aligned_bgr, "ALIGNED")],
            axis=1,
        )
        if not cv2.imwrite(str(compare_dir / output_name), comparison):
            raise IOError(f"cv2.imwrite failed: {compare_dir / output_name}")

    used = {"view_alignment": alignment_cfg}
    with (output_dir / "alignment_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(used, handle, sort_keys=False, allow_unicode=True)

    metadata = canonicalizer.metadata(first_size)
    print(f"[OK] processed {len(images)} images -> {output_dir}")
    print("[INFO] source-to-canonical matrix:")
    print(np.asarray(metadata["matrix"], dtype=np.float64))
    print(f"[INFO] reusable config: {output_dir / 'alignment_used.yaml'}")


if __name__ == "__main__":
    main()
