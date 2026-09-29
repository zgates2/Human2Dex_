"""Lightweight RGB skeleton overlay for wrist 2D projections.

This file deliberately avoids importing torch.  It is used inside augmentation
CPU workers after mask/color/camera-mount augmentation has already produced the
final image.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw


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


def _valid_uv21(uv21: np.ndarray, valid: Sequence[bool] | None, width: int, height: int) -> np.ndarray:
    arr = np.asarray(uv21, dtype=np.float32)
    mask = np.isfinite(arr).all(axis=1)
    mask &= arr[:, 0] >= 0
    mask &= arr[:, 0] < width
    mask &= arr[:, 1] >= 0
    mask &= arr[:, 1] < height
    if valid is not None:
        src = np.asarray(valid, dtype=bool).reshape(-1)
        if src.shape[0] == 21:
            mask &= src
    return mask


def draw_wrist_skeleton_rgb(
    image_rgb: np.ndarray,
    uv21: np.ndarray | None,
    valid: Sequence[bool] | None = None,
    *,
    line_width: int = 2,
    point_radius: int = 4,
) -> np.ndarray:
    if uv21 is None:
        return image_rgb
    arr = np.asarray(uv21, dtype=np.float32)
    if arr.shape != (21, 2):
        return image_rgb
    out = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    width, height = out.size
    valid21 = _valid_uv21(arr, valid, width=width, height=height)

    for a, b in HAND_BONES:
        if not (valid21[a] and valid21[b]):
            continue
        pa = tuple(float(v) for v in arr[a])
        pb = tuple(float(v) for v in arr[b])
        draw.line([pa, pb], fill=JOINT_COLORS_RGB[b], width=max(1, int(line_width)))

    radius = max(1, int(point_radius))
    for joint_idx, uv in enumerate(arr):
        if not valid21[joint_idx]:
            continue
        x, y = float(uv[0]), float(uv[1])
        color = JOINT_COLORS_RGB[joint_idx]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    return np.asarray(out, dtype=np.uint8)
