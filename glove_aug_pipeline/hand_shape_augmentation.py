"""Conservative 2D hand-silhouette augmentation for wrist-camera data.

The current implementation changes finger-web depth while keeping detected
fingertips almost fixed.  It deliberately skips occluded/contact-heavy frames
when fewer than three clean distal peaks are visible.  This is a 2D silhouette
proxy for finger length variation, not a MANO/3D bone-length edit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


@dataclass
class ShapeAugmentationStats:
    enabled: bool
    applied: bool
    reason: str
    length_scale: float
    detected_tip_count: int = 0
    control_valley_count: int = 0
    max_control_displacement_px: float = 0.0
    max_tip_displacement_px: float = 0.0
    original_mask_area: int = 0
    augmented_mask_area: int = 0
    mask_area_change_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 6)
        return data


def _pixel_scale(shape: Tuple[int, int]) -> float:
    height, width = shape
    return max(float(np.sqrt((height * width) / float(360 * 640))), 0.35)


def _bottom_envelope(mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    envelope = np.full(x1 - x0 + 1, np.nan, dtype=np.float32)
    column_heights = np.zeros_like(envelope)
    for local_x, x in enumerate(range(x0, x1 + 1)):
        column = np.flatnonzero(mask[:, x])
        if column.size:
            envelope[local_x] = float(column.max())
            column_heights[local_x] = float(column.size)
    valid = np.isfinite(envelope)
    if np.any(~valid):
        valid_idx = np.flatnonzero(valid)
        invalid_idx = np.flatnonzero(~valid)
        envelope[~valid] = np.interp(invalid_idx, valid_idx, envelope[valid])
    return x0, envelope, column_heights


def _detect_finger_controls(
    mask: np.ndarray,
    *,
    min_visible_tips: int,
    min_visible_valleys: int,
    peak_prominence_range: Tuple[float, float],
    min_valley_drop_px: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    scale = _pixel_scale(mask.shape)
    distance = ndimage.distance_transform_edt(mask)
    palm_y, palm_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    palm = np.asarray([float(palm_x), float(palm_y)], dtype=np.float32)
    x0, envelope, column_heights = _bottom_envelope(mask)
    smooth = ndimage.gaussian_filter1d(envelope, sigma=max(0.8, 2.0 * scale))

    min_prominence = float(peak_prominence_range[0]) * scale
    max_prominence = float(peak_prominence_range[1]) * scale
    peak_indices, properties = find_peaks(
        smooth,
        prominence=min_prominence,
        distance=max(4, int(round(12 * scale))),
        width=max(1, int(round(3 * scale))),
    )
    tips: List[Dict[str, Any]] = []
    min_distal_offset = max(5.0, 14.0 * scale)
    for i, peak_idx in enumerate(peak_indices):
        x = x0 + int(peak_idx)
        y = float(smooth[peak_idx])
        prominence = float(properties["prominences"][i])
        if (
            y >= float(palm_y) + min_distal_offset
            and column_heights[peak_idx] >= max(3.0, 5.0 * scale)
            and prominence <= max_prominence
        ):
            tips.append(
                {
                    "index": int(peak_idx),
                    "point": np.asarray([float(x), y], dtype=np.float32),
                    "prominence": prominence,
                }
            )

    tips.sort(key=lambda item: float(item["point"][0]))
    max_tip_gap = 48.0 * scale
    best_group: List[Dict[str, Any]] = []
    best_score = (-1, -1.0)
    for start in range(len(tips)):
        group = [tips[start]]
        for item in tips[start + 1 :]:
            if float(item["point"][0] - group[-1]["point"][0]) <= max_tip_gap:
                group.append(item)
            else:
                break
        score = (len(group), sum(float(item["prominence"]) for item in group))
        if score > best_score:
            best_group = group
            best_score = score
    tips = best_group
    if len(tips) > 5:
        tips = sorted(tips, key=lambda item: item["prominence"], reverse=True)[:5]
        tips.sort(key=lambda item: float(item["point"][0]))

    valleys: List[Dict[str, Any]] = []
    min_drop = float(min_valley_drop_px) * scale
    for left_tip, right_tip in zip(tips, tips[1:]):
        lo, hi = int(left_tip["index"]), int(right_tip["index"])
        if hi - lo < 3:
            continue
        local_idx = int(np.argmin(smooth[lo : hi + 1])) + lo
        valley_y = float(smooth[local_idx])
        drop = min(
            float(left_tip["point"][1]),
            float(right_tip["point"][1]),
        ) - valley_y
        if drop >= min_drop:
            valleys.append(
                {
                    "point": np.asarray(
                        [float(x0 + local_idx), valley_y], dtype=np.float32
                    ),
                    "left_tip": left_tip["point"],
                    "right_tip": right_tip["point"],
                    "drop": float(drop),
                }
            )

    if len(tips) < int(min_visible_tips) or len(valleys) < int(min_visible_valleys):
        valleys = []
    return palm, tips, valleys


def _build_displacement_field(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    palm: np.ndarray,
    tips: List[Dict[str, Any]],
    valleys: List[Dict[str, Any]],
    length_scale: float,
    warp_radius_px: float,
    max_control_displacement_px: float,
    max_tip_displacement_px: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    disp_x = np.zeros_like(xx, dtype=np.float32)
    disp_y = np.zeros_like(yy, dtype=np.float32)
    weight_sum = np.full_like(xx, 0.32, dtype=np.float32)
    controls = []
    max_control_displacement = 0.0
    for valley in valleys:
        point = valley["point"]
        target = palm + float(length_scale) * (point - palm)
        delta = target - point
        norm = float(np.linalg.norm(delta))
        if norm > float(max_control_displacement_px):
            delta *= float(max_control_displacement_px) / max(norm, 1e-6)
            norm = float(max_control_displacement_px)
        max_control_displacement = max(max_control_displacement, norm)
        controls.append((point, delta, 1.0))
        controls.append((valley["left_tip"], np.zeros(2, np.float32), 1.8))
        controls.append((valley["right_tip"], np.zeros(2, np.float32), 1.8))
    controls.append((palm, np.zeros(2, np.float32), 2.0))

    sigma = max(float(warp_radius_px), 1.0)
    for point, delta, strength in controls:
        distance_sq = (xx - point[0]) ** 2 + (yy - point[1]) ** 2
        weight = np.exp(-distance_sq / (2.0 * sigma * sigma)).astype(np.float32)
        weight *= float(strength)
        disp_x += weight * float(delta[0])
        disp_y += weight * float(delta[1])
        weight_sum += weight
    disp_x /= weight_sum
    disp_y /= weight_sum

    max_tip_displacement = 0.0
    for tip in tips:
        tx = int(np.clip(round(float(tip["point"][0])), 0, xx.shape[1] - 1))
        ty = int(np.clip(round(float(tip["point"][1])), 0, yy.shape[0] - 1))
        displacement = float(np.hypot(disp_x[ty, tx], disp_y[ty, tx]))
        max_tip_displacement = max(max_tip_displacement, displacement)
    if max_tip_displacement > float(max_tip_displacement_px):
        field_scale = float(max_tip_displacement_px) / max(max_tip_displacement, 1e-6)
        disp_x *= field_scale
        disp_y *= field_scale
        max_tip_displacement = float(max_tip_displacement_px)
        max_control_displacement *= field_scale
    return disp_x, disp_y, max_control_displacement, max_tip_displacement


def apply_hand_shape_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    enabled: bool,
    length_scale: float,
    min_visible_tips: int = 3,
    min_visible_valleys: int = 2,
    peak_prominence_range: Tuple[float, float] = (3.0, 40.0),
    min_valley_drop_px: float = 2.5,
    warp_radius_px: float = 13.0,
    max_control_displacement_px: float = 6.0,
    max_tip_displacement_px: float = 2.0,
    max_mask_area_change_ratio: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray, ShapeAugmentationStats]:
    """Warp finger-web depth while preserving visible fingertips.

    The detector assumes the wrist-camera convention used by the current
    datasets: the hand enters from the upper image side and fingertips point
    toward the lower image side.  Frames with contact/occlusion normally expose
    too few clean peaks and are skipped instead of forcing a risky deformation.
    """

    mask = np.asarray(mask, dtype=bool)
    original_area = int(mask.sum())
    stats = ShapeAugmentationStats(
        enabled=bool(enabled),
        applied=False,
        reason="disabled" if not enabled else "",
        length_scale=float(length_scale),
        original_mask_area=original_area,
        augmented_mask_area=original_area,
    )
    if not enabled:
        return image, mask, stats
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        stats.reason = "shape_mismatch"
        return image, mask, stats
    if original_area == 0:
        stats.reason = "empty_mask"
        return image, mask, stats
    if abs(float(length_scale) - 1.0) < 1e-4:
        stats.reason = "identity_scale"
        return image, mask, stats

    palm, tips, valleys = _detect_finger_controls(
        mask,
        min_visible_tips=min_visible_tips,
        min_visible_valleys=min_visible_valleys,
        peak_prominence_range=peak_prominence_range,
        min_valley_drop_px=min_valley_drop_px,
    )
    stats.detected_tip_count = len(tips)
    stats.control_valley_count = len(valleys)
    if not valleys:
        stats.reason = "insufficient_visible_fingers"
        return image, mask, stats

    height, width = mask.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    scale = _pixel_scale(mask.shape)
    disp_x, disp_y, max_control_disp, max_tip_disp = _build_displacement_field(
        xx,
        yy,
        palm=palm,
        tips=tips,
        valleys=valleys,
        length_scale=length_scale,
        warp_radius_px=float(warp_radius_px) * scale,
        max_control_displacement_px=float(max_control_displacement_px) * scale,
        max_tip_displacement_px=float(max_tip_displacement_px) * scale,
    )
    stats.max_control_displacement_px = max_control_disp
    stats.max_tip_displacement_px = max_tip_disp

    map_x = xx - disp_x
    map_y = yy - disp_y
    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask_u8 = cv2.remap(
        mask.astype(np.uint8) * 255,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = warped_mask_u8 >= 127
    augmented_area = int(warped_mask.sum())
    area_change_ratio = float((augmented_area - original_area) / max(original_area, 1))
    stats.augmented_mask_area = augmented_area
    stats.mask_area_change_ratio = area_change_ratio
    if abs(area_change_ratio) > float(max_mask_area_change_ratio):
        stats.reason = "mask_area_change_exceeded"
        return image, mask, stats

    _, nearest_background = ndimage.distance_transform_edt(
        mask,
        return_indices=True,
    )
    background_fill = image[nearest_background[0], nearest_background[1]]
    base = image.copy()
    removed = mask & ~warped_mask
    base[removed] = background_fill[removed]
    alpha = ndimage.gaussian_filter(warped_mask.astype(np.float32), sigma=1.0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    output = base.astype(np.float32) * (1.0 - alpha)
    output += warped_image.astype(np.float32) * alpha

    stats.applied = True
    stats.reason = ""
    return np.clip(output, 0, 255).astype(np.uint8), warped_mask, stats
