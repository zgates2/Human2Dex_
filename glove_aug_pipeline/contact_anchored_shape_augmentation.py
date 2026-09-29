#!/usr/bin/env python3
"""Prototype contact-anchored silhouette contraction for wrist-camera hand images.

The transform deliberately keeps the generated hand silhouette inside the
original SAM mask.  A narrow distal boundary band is copied back unchanged as
an approximate contact anchor.  Pixels removed from the human-hand silhouette
are inpainted, while pixels that were visible outside the original hand mask
are never overwritten.

This is a shape-only prototype.  Appearance randomization should be applied in
a separate stage so the paper can evaluate shape and appearance independently.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ShapeConfig:
    distal_direction_deg: float = 90.0
    proximal_width_scale: float = 0.82
    distal_width_scale: float = 0.82
    length_scale: float = 0.98
    anchor_quantile: float = 0.74
    anchor_radius_px: int = 3
    mask_close_px: int = 2
    support_dilate_px: int = 0
    inpaint_radius_px: float = 5.0
    background_fill: str = "nearest"
    nearest_fill_blur_px: float = 3.0
    min_area_retention: float = 0.72
    max_area_retention: float = 0.90
    max_components: int = 2


def load_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise OSError(f"Failed to write mask: {path}")


def _disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _boundary(mask: np.ndarray, radius: int) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), _disk(radius)) > 0
    return mask & ~eroded


def _smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _principal_coordinates(
    shape: Tuple[int, int],
    distal_direction_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    theta = math.radians(float(distal_direction_deg))
    distal = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
    lateral = np.asarray([-distal[1], distal[0]], dtype=np.float32)
    v = xx * distal[0] + yy * distal[1]
    u = xx * lateral[0] + yy * lateral[1]
    return u, v, lateral, distal


def build_contact_anchored_mask(
    mask: np.ndarray,
    config: ShapeConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Return contracted mask, preserved distal boundary, and diagnostics."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("mask must be a non-empty 2D binary array")

    u, v, lateral, distal = _principal_coordinates(
        mask.shape,
        config.distal_direction_deg,
    )
    mask_v = v[mask]
    v_min = float(mask_v.min())
    v_max = float(mask_v.max())
    span = max(v_max - v_min, 1.0)
    t = np.clip((v - v_min) / span, 0.0, 1.0)

    anchor_cut = float(np.quantile(mask_v, config.anchor_quantile))
    distal_boundary = _boundary(mask, config.anchor_radius_px) & (v >= anchor_cut)

    # Use the proximal half of the hand to estimate a stable lateral center.
    # This is less sensitive to individual finger spread than the full centroid.
    proximal = mask & (t <= 0.55)
    center_source = proximal if proximal.any() else mask
    u_center = float(np.median(u[center_source]))

    # Inverse map from target coordinates to the original mask.  The distal
    # extreme remains fixed, while the proximal hand is pulled toward it.
    source_v = v_max - (v_max - v) / max(config.length_scale, 1e-4)
    source_t = np.clip((source_v - v_min) / span, 0.0, 1.0)
    width_scale = (
        config.proximal_width_scale
        + (config.distal_width_scale - config.proximal_width_scale)
        * _smoothstep(source_t)
    )
    source_u = u_center + (u - u_center) / np.maximum(width_scale, 1e-4)

    source_x = source_u * lateral[0] + source_v * distal[0]
    source_y = source_u * lateral[1] + source_v * distal[1]
    warped = cv2.remap(
        mask.astype(np.uint8),
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    # Safety invariant: never paint a synthetic hand over pixels that were
    # visibly outside the original hand.  A one-pixel support dilation is an
    # optional anti-aliasing allowance and defaults to one pixel.
    support = mask
    if config.support_dilate_px > 0:
        support = cv2.dilate(
            mask.astype(np.uint8),
            _disk(config.support_dilate_px),
        ) > 0
    target = warped & support
    target |= distal_boundary
    if config.mask_close_px > 0:
        target = cv2.morphologyEx(
            target.astype(np.uint8),
            cv2.MORPH_CLOSE,
            _disk(config.mask_close_px),
        ) > 0
        target &= support
        target |= distal_boundary

    label_count, labels = cv2.connectedComponents(target.astype(np.uint8))
    _ = labels
    diagnostics = {
        "original_area": int(mask.sum()),
        "target_area": int(target.sum()),
        "area_ratio_target_over_original": float(target.sum() / mask.sum()),
        "anchor_area": int(distal_boundary.sum()),
        "anchor_retention": float(
            (target & distal_boundary).sum() / max(int(distal_boundary.sum()), 1)
        ),
        "component_count": int(max(label_count - 1, 0)),
        "distal_projection_min": v_min,
        "distal_projection_max": v_max,
        "anchor_projection_cut": anchor_cut,
        "lateral_center": u_center,
    }
    return target, distal_boundary, diagnostics


def apply_shape_contraction(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: ShapeConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    target, anchor, diagnostics = build_contact_anchored_mask(mask, config)
    u, v, lateral, distal = _principal_coordinates(
        mask.shape,
        config.distal_direction_deg,
    )
    mask_v = v[mask]
    v_min = float(mask_v.min())
    v_max = float(mask_v.max())
    span = max(v_max - v_min, 1.0)
    proximal = mask & (((v - v_min) / span) <= 0.55)
    center_source = proximal if proximal.any() else mask
    u_center = float(np.median(u[center_source]))

    source_v = v_max - (v_max - v) / max(config.length_scale, 1e-4)
    source_t = np.clip((source_v - v_min) / span, 0.0, 1.0)
    width_scale = (
        config.proximal_width_scale
        + (config.distal_width_scale - config.proximal_width_scale)
        * _smoothstep(source_t)
    )
    source_u = u_center + (u - u_center) / np.maximum(width_scale, 1e-4)
    source_x = source_u * lateral[0] + source_v * distal[0]
    source_y = source_u * lateral[1] + source_v * distal[1]
    warped_rgb = cv2.remap(
        image_rgb,
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    removed = mask & ~target
    if config.background_fill == "nearest":
        # Propagate only real pixels from outside the original hand mask into
        # the removed contour band.  Unlike Telea on a large mask, this cannot
        # sample the retained inner hand and recreate the old skin silhouette.
        _, nearest_indices = ndimage.distance_transform_edt(
            mask,
            return_indices=True,
        )
        nearest_background = image_rgb[
            nearest_indices[0],
            nearest_indices[1],
        ]
        if config.nearest_fill_blur_px > 0:
            nearest_background = cv2.GaussianBlur(
                nearest_background,
                (0, 0),
                sigmaX=float(config.nearest_fill_blur_px),
                sigmaY=float(config.nearest_fill_blur_px),
            )
        output = image_rgb.copy()
        output[removed] = nearest_background[removed]
    elif config.background_fill == "telea":
        # Kept as an explicit comparison baseline.  Full-mask inpainting is
        # required; inpainting only ``removed`` leaks colors from retained hand.
        inpaint_mask = mask.astype(np.uint8) * 255
        background = cv2.inpaint(
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            inpaint_mask,
            float(config.inpaint_radius_px),
            cv2.INPAINT_TELEA,
        )
        output = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported background_fill: {config.background_fill}")
    output[target] = warped_rgb[target]
    output[anchor] = image_rgb[anchor]

    # Verify the primary label-preservation invariant at runtime.
    visible_outside = ~mask
    changed_outside = np.any(output != image_rgb, axis=-1) & visible_outside
    diagnostics["changed_visible_outside_original_mask"] = int(changed_outside.sum())
    diagnostics["removed_area"] = int(removed.sum())
    qc_reasons = []
    area_retention = diagnostics["area_ratio_target_over_original"]
    if area_retention < config.min_area_retention:
        qc_reasons.append("area_retention_too_low")
    if area_retention > config.max_area_retention:
        qc_reasons.append("area_retention_too_high")
    if diagnostics["anchor_retention"] < 1.0:
        qc_reasons.append("anchor_not_fully_retained")
    if diagnostics["component_count"] > config.max_components:
        qc_reasons.append("too_many_components")
    if diagnostics["changed_visible_outside_original_mask"] > 0:
        qc_reasons.append("visible_pixels_outside_mask_changed")
    diagnostics["qc_pass"] = not qc_reasons
    diagnostics["qc_reasons"] = qc_reasons
    return output, target, anchor, diagnostics


def make_qc(
    image_rgb: np.ndarray,
    original_mask: np.ndarray,
    output_rgb: np.ndarray,
    target_mask: np.ndarray,
    anchor_mask: np.ndarray,
) -> np.ndarray:
    original_overlay = image_rgb.copy()
    original_overlay[original_mask] = (
        0.55 * original_overlay[original_mask] + 0.45 * np.asarray([255, 40, 80])
    ).astype(np.uint8)

    target_overlay = image_rgb.copy()
    removed = original_mask & ~target_mask
    target_overlay[removed] = (
        0.45 * target_overlay[removed] + 0.55 * np.asarray([255, 180, 0])
    ).astype(np.uint8)
    target_overlay[target_mask] = (
        0.55 * target_overlay[target_mask] + 0.45 * np.asarray([40, 220, 100])
    ).astype(np.uint8)
    target_overlay[anchor_mask] = np.asarray([20, 180, 255], dtype=np.uint8)

    gap = np.full((image_rgb.shape[0], 8, 3), 24, dtype=np.uint8)
    return np.concatenate(
        [image_rgb, gap, original_overlay, gap, target_overlay, gap, output_rgb],
        axis=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distal-direction-deg", type=float, default=90.0)
    parser.add_argument("--proximal-width-scale", type=float, default=0.82)
    parser.add_argument("--distal-width-scale", type=float, default=0.82)
    parser.add_argument("--length-scale", type=float, default=0.98)
    parser.add_argument("--anchor-quantile", type=float, default=0.74)
    parser.add_argument("--anchor-radius-px", type=int, default=3)
    parser.add_argument("--mask-close-px", type=int, default=2)
    parser.add_argument("--support-dilate-px", type=int, default=0)
    parser.add_argument("--inpaint-radius-px", type=float, default=5.0)
    parser.add_argument(
        "--background-fill",
        choices=("nearest", "telea"),
        default="nearest",
    )
    parser.add_argument("--nearest-fill-blur-px", type=float, default=3.0)
    parser.add_argument("--min-area-retention", type=float, default=0.72)
    parser.add_argument("--max-area-retention", type=float, default=0.90)
    parser.add_argument("--max-components", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = load_rgb(args.image)
    mask = load_mask(args.mask, image.shape[:2])
    config = ShapeConfig(
        distal_direction_deg=args.distal_direction_deg,
        proximal_width_scale=args.proximal_width_scale,
        distal_width_scale=args.distal_width_scale,
        length_scale=args.length_scale,
        anchor_quantile=args.anchor_quantile,
        anchor_radius_px=args.anchor_radius_px,
        mask_close_px=args.mask_close_px,
        support_dilate_px=args.support_dilate_px,
        inpaint_radius_px=args.inpaint_radius_px,
        background_fill=args.background_fill,
        nearest_fill_blur_px=args.nearest_fill_blur_px,
        min_area_retention=args.min_area_retention,
        max_area_retention=args.max_area_retention,
        max_components=args.max_components,
    )
    output, target, anchor, diagnostics = apply_shape_contraction(image, mask, config)
    qc = make_qc(image, mask, output, target, anchor)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    save_rgb(output_dir / f"{stem}_shape.jpg", output)
    save_rgb(output_dir / f"{stem}_shape_qc.jpg", qc)
    save_mask(output_dir / f"{stem}_shape_mask.png", target)
    save_mask(output_dir / f"{stem}_anchor_mask.png", anchor)
    payload = {
        "image": str(args.image),
        "mask": str(args.mask),
        "config": asdict(config),
        "diagnostics": diagnostics,
    }
    with (output_dir / f"{stem}_shape.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
