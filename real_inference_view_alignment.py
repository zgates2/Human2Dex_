#!/usr/bin/env python3
"""Deterministic deployment-side wrist-view canonicalization.

The transform is applied only to RGB observations passed to the policy.  It
does not modify robot poses, TCPs, actions, or camera buffers used by the
episode recorder.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import cv2
import numpy as np


_BORDER_MODES = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "reflect101": cv2.BORDER_REFLECT_101,
}


def _pair(value, name: str) -> tuple[float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"view_alignment.{name} must contain two finite values")
    return float(arr[0]), float(arr[1])


def _optional_pair(value, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _pair(value, name)


def _size(value, name: str) -> tuple[int, int]:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.shape != (2,) or np.any(arr <= 1):
        raise ValueError(f"view_alignment.{name} must contain [width, height] > 1")
    return int(arr[0]), int(arr[1])


def _optional_circle(value, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or float(arr[2]) <= 0.0:
        raise ValueError(f"view_alignment.{name} must contain finite [cx, cy, radius]")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _normalize_mode(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _threshold_for_image(image: np.ndarray, threshold_255: float) -> float:
    if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.5:
        return float(threshold_255) / 255.0
    return float(threshold_255)


def _estimate_fisheye_circle(
        image: np.ndarray,
        *,
        threshold_255: float,
        margin_px: float) -> tuple[float, float, float]:
    max_chan = np.max(image[:, :, :3], axis=2)
    threshold = _threshold_for_image(image, threshold_255)
    mask = (max_chan > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    if not contours:
        return (width - 1) / 2.0, (height - 1) / 2.0, max(1.0, min(width, height) / 2.0)
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    return float(cx), float(cy), max(1.0, float(radius) + float(margin_px))


class ViewCanonicalizer:
    """Apply one fixed image-space transform to every frame in an episode."""

    def __init__(self, config: dict):
        self.config = deepcopy(config)
        self.mount_id = str(config.get("mount_id", "unspecified_mount"))
        self.transform_type = str(
            config.get("transform_type", "similarity")
        ).strip().lower()
        self.transform_type = _normalize_mode(self.transform_type)
        self.rgb_keys = config.get("rgb_keys")
        if isinstance(self.rgb_keys, str):
            self.rgb_keys = [self.rgb_keys]
        if self.rgb_keys is not None:
            self.rgb_keys = {str(key) for key in self.rgb_keys}

        self.scale = float(config.get("scale", 1.0))
        self.rotation_deg = float(config.get("rotation_deg", 0.0))
        self.translation_ratio = _pair(
            config.get("translation_ratio", [0.0, 0.0]),
            "translation_ratio",
        )
        self.pivot_ratio = _pair(
            config.get("pivot_ratio", [0.5, 0.5]),
            "pivot_ratio",
        )
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("view_alignment.scale must be finite and > 0")
        if not np.isfinite(self.rotation_deg):
            raise ValueError("view_alignment.rotation_deg must be finite")

        border_name = str(config.get("border_mode", "reflect101")).strip().lower()
        if border_name not in _BORDER_MODES:
            raise ValueError(
                "view_alignment.border_mode must be one of "
                + ", ".join(sorted(_BORDER_MODES))
            )
        self.border_name = border_name
        self.border_mode = _BORDER_MODES[border_name]
        self.interpolation = cv2.INTER_LINEAR

        self.explicit_matrix = None
        self.train_center_px = None
        self.train_image_size = None
        self.deploy_center_px = None
        self.deploy_circle_px = None
        self.reference_circle_px = None
        self.circle_threshold = 2.0
        self.circle_margin_px = 0.0
        self.reflect_inner_margin_px = 0.0
        self.circle_mode = "fixed_source"
        if self.transform_type == "homography":
            matrix = np.asarray(config.get("matrix"), dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    "view_alignment.matrix must be a finite 3x3 matrix when "
                    "transform_type=homography"
                )
            if abs(float(np.linalg.det(matrix))) < 1e-12:
                raise ValueError("view_alignment.matrix must be invertible")
            self.explicit_matrix = matrix
        elif self.transform_type == "cavity_center":
            self.train_center_px = _pair(
                config.get("train_center_px", config.get("train_center")),
                "train_center_px",
            )
            self.train_image_size = _size(
                config.get("train_image_size", config.get("train_size", [224, 224])),
                "train_image_size",
            )
            self.deploy_center_px = _optional_pair(
                config.get("deploy_center_px", config.get("deploy_center")),
                "deploy_center_px",
            )
            if self.deploy_center_px is None:
                raise ValueError(
                    "view_alignment.deploy_center_px is required when "
                    "transform_type=cavity_center. Please re-calibrate the "
                    "deployment O6 cavity center and pass --view-deploy-center x,y."
                )
            self.deploy_circle_px = _optional_circle(
                config.get("deploy_circle_px", config.get("deploy_circle")),
                "deploy_circle_px",
            )
            self.reference_circle_px = _optional_circle(
                config.get("reference_circle_px", config.get("reference_circle")),
                "reference_circle_px",
            )
            self.circle_threshold = float(config.get("circle_threshold", 2.0))
            if not np.isfinite(self.circle_threshold) or self.circle_threshold < 0.0:
                raise ValueError("view_alignment.circle_threshold must be finite and >= 0")
            self.circle_margin_px = float(config.get("circle_margin_px", 0.0))
            if not np.isfinite(self.circle_margin_px):
                raise ValueError("view_alignment.circle_margin_px must be finite")
            self.reflect_inner_margin_px = float(
                config.get("reflect_inner_margin_px", config.get("reflect_margin_px", 0.0))
            )
            if not np.isfinite(self.reflect_inner_margin_px) or self.reflect_inner_margin_px < 0.0:
                raise ValueError(
                    "view_alignment.reflect_inner_margin_px must be finite and >= 0"
                )
            self.circle_mode = _normalize_mode(config.get("circle_mode", "fixed_source"))
            allowed_circle_modes = {
                "fixed_source",
                "transformed_source",
                "fixed_reference",
                "fixed_output",
            }
            if self.circle_mode not in allowed_circle_modes:
                raise ValueError(
                    "view_alignment.circle_mode must be one of "
                    + ", ".join(sorted(allowed_circle_modes))
                )
            if self.circle_mode == "fixed_reference" and self.reference_circle_px is None:
                raise ValueError(
                    "view_alignment.reference_circle_px is required when "
                    "circle_mode=fixed_reference"
                )
        elif self.transform_type != "similarity":
            raise ValueError(
                "view_alignment.transform_type must be similarity, homography, or cavity_center"
            )

        self._matrix_cache: dict[tuple[int, int], np.ndarray] = {}
        self.last_debug: dict | None = None

    def _should_process(self, key: str, value) -> bool:
        if self.rgb_keys is not None and str(key) not in self.rgb_keys:
            return False
        arr = np.asarray(value)
        return str(key).endswith("_rgb") and arr.ndim == 4 and arr.shape[-1] == 3

    def matrix_for_shape(self, height: int, width: int) -> np.ndarray:
        shape_key = (int(height), int(width))
        cached = self._matrix_cache.get(shape_key)
        if cached is not None:
            return cached.copy()

        if self.explicit_matrix is not None:
            matrix = self.explicit_matrix.copy()
        else:
            pivot_x = self.pivot_ratio[0] * float(width)
            pivot_y = self.pivot_ratio[1] * float(height)
            affine = cv2.getRotationMatrix2D(
                (pivot_x, pivot_y),
                self.rotation_deg,
                self.scale,
            )
            affine[0, 2] += self.translation_ratio[0] * float(width)
            affine[1, 2] += self.translation_ratio[1] * float(height)
            matrix = np.eye(3, dtype=np.float64)
            matrix[:2] = affine

        self._matrix_cache[shape_key] = matrix
        return matrix.copy()

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"RGB image must have shape (H,W,3), got {arr.shape}")
        if self.transform_type == "cavity_center":
            aligned, debug = self.apply_image_with_debug(arr)
            self.last_debug = debug
            return aligned

        height, width = arr.shape[:2]
        matrix = self.matrix_for_shape(height, width)

        original_dtype = arr.dtype
        work = np.ascontiguousarray(arr)
        if original_dtype == np.float16:
            work = work.astype(np.float32)
        warped = cv2.warpPerspective(
            work,
            matrix,
            dsize=(width, height),
            flags=self.interpolation,
            borderMode=self.border_mode,
        )
        if warped.dtype != original_dtype:
            warped = warped.astype(original_dtype, copy=False)
        return warped

    def _target_center_for_shape(self, height: int, width: int) -> tuple[float, float]:
        if self.train_center_px is None or self.train_image_size is None:
            raise RuntimeError("cavity-center train center is not configured")
        train_w, train_h = self.train_image_size
        return (
            self.train_center_px[0] * float(width - 1) / float(train_w - 1),
            self.train_center_px[1] * float(height - 1) / float(train_h - 1),
        )

    def _source_circle_for_image(self, arr: np.ndarray) -> tuple[float, float, float]:
        if self.deploy_circle_px is not None:
            return self.deploy_circle_px
        return _estimate_fisheye_circle(
            arr,
            threshold_255=self.circle_threshold,
            margin_px=self.circle_margin_px,
        )

    def _output_circle_for_shape(
            self,
            *,
            height: int,
            width: int,
            source_circle: tuple[float, float, float],
            target_center: tuple[float, float],
            source_center: tuple[float, float]) -> tuple[float, float, float]:
        src_cx, src_cy, src_radius = source_circle
        sx, sy = source_center
        if self.circle_mode == "fixed_source":
            return src_cx, src_cy, src_radius
        if self.circle_mode == "transformed_source":
            return (
                target_center[0] + self.scale * (src_cx - sx),
                target_center[1] + self.scale * (src_cy - sy),
                self.scale * src_radius,
            )
        if self.circle_mode == "fixed_reference":
            assert self.reference_circle_px is not None
            return self.reference_circle_px
        if self.circle_mode == "fixed_output":
            return target_center[0], target_center[1], float(min(width, height)) * 0.5
        raise RuntimeError(f"unknown circle mode: {self.circle_mode}")

    def apply_image_with_debug(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"RGB image must have shape (H,W,3), got {arr.shape}")
        if self.transform_type != "cavity_center":
            aligned = self.apply_image(arr)
            return aligned, {"enabled": False}

        height, width = arr.shape[:2]
        source_center = self.deploy_center_px
        if source_center is None:
            raise RuntimeError("cavity-center deploy center is not configured")
        target_center = self._target_center_for_shape(height, width)
        source_circle = self._source_circle_for_image(arr)
        out_circle = self._output_circle_for_shape(
            height=height,
            width=width,
            source_circle=source_circle,
            target_center=target_center,
            source_center=source_center,
        )

        sx, sy = source_center
        tx = target_center[0] - self.scale * sx
        ty = target_center[1] - self.scale * sy
        yy, xx = np.indices((height, width), dtype=np.float32)
        inv_x = (xx - tx) / self.scale
        inv_y = (yy - ty) / self.scale

        src_cx, src_cy, src_radius = source_circle
        sample_x = inv_x.copy()
        sample_y = inv_y.copy()
        reflect_radius = max(1.0, float(src_radius) - float(self.reflect_inner_margin_px))
        dx = sample_x - float(src_cx)
        dy = sample_y - float(src_cy)
        dist = np.sqrt(dx * dx + dy * dy)
        outside_source_circle = dist > reflect_radius
        if np.any(outside_source_circle):
            period = max(1.0, 2.0 * reflect_radius)
            dist_mod = np.mod(dist, period)
            reflected_dist = np.where(dist_mod > reflect_radius, period - dist_mod, dist_mod)
            safe_dist = np.maximum(dist, 1e-6)
            sample_x = np.where(
                outside_source_circle,
                float(src_cx) + dx / safe_dist * reflected_dist,
                sample_x,
            )
            sample_y = np.where(
                outside_source_circle,
                float(src_cy) + dy / safe_dist * reflected_dist,
                sample_y,
            )

        original_dtype = arr.dtype
        work = np.ascontiguousarray(arr)
        if original_dtype == np.float16:
            work = work.astype(np.float32)
        warped = cv2.remap(
            work,
            sample_x.astype(np.float32),
            sample_y.astype(np.float32),
            interpolation=self.interpolation,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        source_valid = (
            (inv_x >= 0.0)
            & (inv_x <= float(width - 1))
            & (inv_y >= 0.0)
            & (inv_y <= float(height - 1))
            & ((inv_x - src_cx) ** 2 + (inv_y - src_cy) ** 2 <= reflect_radius ** 2)
        )
        out_cx, out_cy, out_radius = out_circle
        output_circle_mask = (xx - out_cx) ** 2 + (yy - out_cy) ** 2 <= out_radius ** 2
        reflected_mask = output_circle_mask & ~source_valid
        warped = warped.copy()
        warped[~output_circle_mask] = 0
        if warped.dtype != original_dtype:
            warped = warped.astype(original_dtype, copy=False)

        circle_area = max(1, int(np.count_nonzero(output_circle_mask)))
        debug = {
            "enabled": True,
            "transformType": self.transform_type,
            "scale": float(self.scale),
            "sourceCenter": [float(source_center[0]), float(source_center[1])],
            "targetCenter": [float(target_center[0]), float(target_center[1])],
            "sourceCircle": [float(src_cx), float(src_cy), float(src_radius)],
            "reflectionCircle": [float(src_cx), float(src_cy), float(reflect_radius)],
            "outputCircle": [float(out_cx), float(out_cy), float(out_radius)],
            "circleMode": self.circle_mode,
            "circleThreshold": float(self.circle_threshold),
            "reflectInnerMarginPx": float(self.reflect_inner_margin_px),
            "reflectedAreaFractionInCircle": float(np.count_nonzero(reflected_mask) / circle_area),
            "blackFraction": float(np.mean(np.all(warped <= _threshold_for_image(warped, 8.0), axis=2))),
            "reflectedMask": reflected_mask,
            "outputCircleMask": output_circle_mask,
        }
        self.last_debug = debug
        return warped, debug

    def apply(self, env_obs: dict) -> dict:
        output = dict(env_obs)
        for key, value in env_obs.items():
            if not self._should_process(key, value):
                continue
            frames = np.asarray(value)
            output[key] = np.stack(
                [self.apply_image(frame) for frame in frames],
                axis=0,
            )
        return output

    def metadata(self, image_size: Iterable[int] | None = None) -> dict:
        result = {
            "enabled": True,
            "mountId": self.mount_id,
            "transformType": self.transform_type,
            "rgbKeys": None if self.rgb_keys is None else sorted(self.rgb_keys),
            "scale": self.scale,
            "rotationDeg": self.rotation_deg,
            "translationRatio": list(self.translation_ratio),
            "pivotRatio": list(self.pivot_ratio),
            "borderMode": self.border_name,
        }
        if self.transform_type == "cavity_center":
            result.update({
                "trainCenterPx": None if self.train_center_px is None else list(self.train_center_px),
                "trainImageSize": None if self.train_image_size is None else list(self.train_image_size),
                "deployCenterPx": None if self.deploy_center_px is None else list(self.deploy_center_px),
                "deployCirclePx": None if self.deploy_circle_px is None else list(self.deploy_circle_px),
                "referenceCirclePx": (
                    None if self.reference_circle_px is None else list(self.reference_circle_px)
                ),
                "circleMode": self.circle_mode,
                "circleThreshold": self.circle_threshold,
                "circleMarginPx": self.circle_margin_px,
                "reflectInnerMarginPx": self.reflect_inner_margin_px,
            })
            if image_size is not None:
                width, height = [int(value) for value in image_size]
                result["imageSize"] = [width, height]
                result["targetCenterPx"] = list(self._target_center_for_shape(height, width))
            return result
        if image_size is not None:
            width, height = [int(value) for value in image_size]
            result["imageSize"] = [width, height]
            result["matrix"] = self.matrix_for_shape(height, width).tolist()
        elif self.explicit_matrix is not None:
            result["matrix"] = self.explicit_matrix.tolist()
        return result


def build_view_canonicalizer(cfg: dict):
    alignment_cfg = cfg.get("view_alignment", {}) or {}
    if not isinstance(alignment_cfg, dict):
        raise ValueError("view_alignment config must be a mapping")
    if not bool(alignment_cfg.get("enabled", False)):
        return None
    canonicalizer = ViewCanonicalizer(alignment_cfg)
    print(
        "[INFO] view_alignment enabled: "
        f"mount_id={canonicalizer.mount_id}, "
        f"type={canonicalizer.transform_type}, "
        f"scale={canonicalizer.scale:.4f}, "
        f"rotation_deg={canonicalizer.rotation_deg:.3f}, "
        f"translation_ratio={canonicalizer.translation_ratio}"
    )
    return canonicalizer
