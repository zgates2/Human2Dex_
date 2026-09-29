#!/usr/bin/env python3
"""Synthetic invariant checks for contact-anchored shape contraction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_SHAPE_DIR = HERE.parent / "glove_aug_pipeline"
if (REPO_SHAPE_DIR / "contact_anchored_shape_augmentation.py").is_file():
    sys.path.insert(0, str(REPO_SHAPE_DIR))
else:
    sys.path.insert(0, str(HERE))

from contact_anchored_shape_augmentation import (
    ShapeConfig,
    apply_shape_contraction,
    build_contact_anchored_mask,
)


def synthetic_hand() -> np.ndarray:
    mask = np.zeros((160, 160), dtype=bool)
    mask[20:92, 40:120] = True
    for x0, x1, length in (
        (42, 52, 54),
        (57, 68, 62),
        (73, 84, 68),
        (89, 100, 60),
        (105, 116, 50),
    ):
        mask[90 : 90 + length, x0:x1] = True
    return mask


def main() -> None:
    mask = synthetic_hand()
    cfg = ShapeConfig(
        distal_direction_deg=90.0,
        proximal_width_scale=0.82,
        distal_width_scale=0.82,
        length_scale=0.98,
        anchor_quantile=0.75,
        anchor_radius_px=2,
        support_dilate_px=0,
    )
    target, anchor, stats = build_contact_anchored_mask(mask, cfg)
    assert target.shape == mask.shape
    assert np.all(target <= mask), "target mask painted outside original mask"
    assert np.all(target[anchor]), "distal contact anchors were dropped"
    assert target.sum() < mask.sum(), "shape transform did not contract silhouette"
    assert stats["anchor_retention"] == 1.0

    image = np.zeros((160, 160, 3), dtype=np.uint8)
    image[..., 1] = 80
    image[mask] = np.asarray([190, 135, 105], dtype=np.uint8)
    output, output_mask, output_anchor, diagnostics = apply_shape_contraction(
        image,
        mask,
        cfg,
    )
    assert np.array_equal(output_mask, target)
    assert np.array_equal(output_anchor, anchor)
    assert diagnostics["changed_visible_outside_original_mask"] == 0
    assert diagnostics["qc_pass"], diagnostics["qc_reasons"]
    assert np.array_equal(output[~mask], image[~mask])
    print("[OK] contact-anchored shape augmentation invariants passed")


if __name__ == "__main__":
    main()
