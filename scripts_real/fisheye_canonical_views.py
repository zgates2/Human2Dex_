#!/usr/bin/env python3
"""Reusable geometric transforms for grasp-anchored fisheye views.

The wide transform is a rotation of calibrated fisheye rays, not a crop or a
2D affine warp.  It preserves the camera field of view while placing a grasp
anchor at a fixed virtual-fisheye pixel.  The local transform is metric and
uses the pocket depth and palm width for near-contact detail.
"""

from __future__ import annotations

import math
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


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if np.isfinite(norm) and norm > 1e-9:
        return value / norm
    if fallback is None:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return normalize(fallback, None)


def border_mode(name: str) -> int:
    key = str(name).strip().lower()
    if key not in _BORDER_MODES:
        raise ValueError(f"Unknown border mode {name!r}; use {sorted(_BORDER_MODES)}")
    return _BORDER_MODES[key]


def fisheye_pixels_to_unit_rays(
        uv: np.ndarray,
        K: np.ndarray,
        D: np.ndarray) -> np.ndarray:
    pixels = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    undistorted = cv2.fisheye.undistortPoints(pixels, K, D).reshape(-1, 2)
    rays = np.concatenate(
        [undistorted, np.ones((len(undistorted), 1), dtype=np.float64)], axis=1
    )
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    if not np.all(np.isfinite(rays)) or np.any(norms <= 1e-9):
        raise ValueError("Fisheye unprojection produced invalid rays")
    return rays / norms


def fisheye_unit_rays_to_pixels(
        rays: np.ndarray,
        K: np.ndarray,
        D: np.ndarray) -> np.ndarray:
    values = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(norms <= 1e-9):
        raise ValueError("Fisheye projection received invalid rays")
    values = values / norms
    zero = np.zeros((3, 1), dtype=np.float64)
    uv, _ = cv2.fisheye.projectPoints(values.reshape(-1, 1, 3), zero, zero, K, D)
    return uv.reshape(-1, 2)


def _basis_from_anchor_and_across(
        anchor_ray: np.ndarray,
        palm_across_ray: np.ndarray,
        across_sign: float) -> np.ndarray:
    z_axis = normalize(anchor_ray)
    across = np.asarray(palm_across_ray, dtype=np.float64).reshape(3) * float(across_sign)
    x_axis = across - float(np.dot(across, z_axis)) * z_axis
    if float(np.linalg.norm(x_axis)) <= 1e-7:
        camera_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = camera_right - float(np.dot(camera_right, z_axis)) * z_axis
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _basis_with_target_ray(target_ray: np.ndarray) -> np.ndarray:
    z_axis = normalize(target_ray)
    image_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = image_right - float(np.dot(image_right, z_axis)) * z_axis
    if float(np.linalg.norm(x_axis)) <= 1e-7:
        image_down = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(image_down, z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _rotation_about_axis(axis: np.ndarray, roll_deg: float) -> np.ndarray:
    """Return a right-handed rotation about a virtual-camera ray."""
    value = float(roll_deg)
    if not math.isfinite(value):
        raise ValueError("virtual_roll_deg must be finite")
    unit_axis = normalize(axis)
    theta = math.radians(value)
    cosine, sine = math.cos(theta), math.sin(theta)
    skew = np.array(
        [[0.0, -unit_axis[2], unit_axis[1]],
         [unit_axis[2], 0.0, -unit_axis[0]],
         [-unit_axis[1], unit_axis[0], 0.0]],
        dtype=np.float64,
    )
    return (
        cosine * np.eye(3, dtype=np.float64)
        + (1.0 - cosine) * np.outer(unit_axis, unit_axis)
        + sine * skew
    )


class FisheyeCanonicalViewRenderer:
    """Render wide and local grasp-anchored views from a calibrated fisheye RGB."""

    def __init__(
            self,
            K: np.ndarray,
            D: np.ndarray,
            *,
            output_size: Iterable[int],
            source_size: Iterable[int] | None = None,
            anchor_uv_ratio: Iterable[float] = (0.5, 0.35),
            palm_across_sign: float = 1.0,
            border: str = "reflect101"):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(4, 1)
        width, height = [int(value) for value in output_size]
        if width < 2 or height < 2:
            raise ValueError("output_size must be [width, height] with values >= 2")
        self.output_size = (width, height)
        ratio = np.asarray(anchor_uv_ratio, dtype=np.float64).reshape(-1)
        if ratio.shape != (2,) or not np.all(np.isfinite(ratio)):
            raise ValueError("anchor_uv_ratio must be two finite values")
        if not (0.0 < ratio[0] < 1.0 and 0.0 < ratio[1] < 1.0):
            raise ValueError("anchor_uv_ratio values must be strictly inside (0,1)")
        self.anchor_uv_ratio = ratio
        self.palm_across_sign = float(palm_across_sign)
        if not math.isfinite(self.palm_across_sign) or abs(self.palm_across_sign) < 1e-8:
            raise ValueError("palm_across_sign must be finite and non-zero")
        self.border_mode = border_mode(border)

        if source_size is None:
            source_width = max(int(round(self.K[0, 2] * 2.0)), width)
            source_height = max(int(round(self.K[1, 2] * 2.0)), height)
        else:
            source_width, source_height = [int(value) for value in source_size]
        if source_width < 2 or source_height < 2:
            raise ValueError("source_size must be [width, height] with values >= 2")
        self.source_size = (source_width, source_height)
        scale_x = float(width) / float(source_width)
        scale_y = float(height) / float(source_height)
        self.output_K = self.K.copy()
        self.output_K[0, 0] *= scale_x
        self.output_K[0, 2] *= scale_x
        self.output_K[1, 1] *= scale_y
        self.output_K[1, 2] *= scale_y

        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float64),
            np.arange(height, dtype=np.float64),
        )
        output_uv = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        self._output_rays = fisheye_pixels_to_unit_rays(output_uv, self.output_K, self.D)
        target_uv = np.asarray(
            [[ratio[0] * (width - 1), ratio[1] * (height - 1)]],
            dtype=np.float64,
        )
        self._target_ray = fisheye_pixels_to_unit_rays(target_uv, self.output_K, self.D)[0]
        self._target_basis = _basis_with_target_ray(self._target_ray)

    def wide_rotation(
            self,
            pocket_camera: np.ndarray,
            palm_across_camera: np.ndarray,
            *,
            virtual_roll_deg: float = 0.0) -> np.ndarray:
        source_basis = _basis_from_anchor_and_across(
            normalize(pocket_camera),
            palm_across_camera,
            self.palm_across_sign,
        )
        # Vector in camera coordinates -> vector in canonical virtual-camera coordinates.
        # The optional left factor rolls around the anchor ray itself.  This
        # preserves the configured anchor pixel even when it is not image-centred.
        return (
            _rotation_about_axis(self._target_ray, virtual_roll_deg)
            @ self._target_basis
            @ source_basis.T
        )

    def render_wide(
            self,
            image: np.ndarray,
            pocket_camera: np.ndarray,
            palm_across_camera: np.ndarray,
            *,
            virtual_roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {source.shape}")
        rotation_virtual_from_camera = self.wide_rotation(
            pocket_camera, palm_across_camera, virtual_roll_deg=virtual_roll_deg
        )
        # Row-vector implementation of d_camera = R^T d_virtual.
        source_rays = self._output_rays @ rotation_virtual_from_camera
        source_uv = fisheye_unit_rays_to_pixels(source_rays, self.K, self.D)
        out_w, out_h = self.output_size
        map_x = source_uv[:, 0].reshape(out_h, out_w).astype(np.float32)
        map_y = source_uv[:, 1].reshape(out_h, out_w).astype(np.float32)
        output = cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
        )
        return output, rotation_virtual_from_camera

    def render_local(
            self,
            image: np.ndarray,
            pocket_camera: np.ndarray,
            palm_width_m: float,
            *,
            width_in_palm: float,
            height_in_palm: float,
            rotation_deg: float = 0.0) -> np.ndarray:
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {source.shape}")
        pocket = np.asarray(pocket_camera, dtype=np.float64).reshape(3)
        palm_width = float(palm_width_m)
        if not np.all(np.isfinite(pocket)) or pocket[2] <= 1e-6:
            raise ValueError("Pocket must be finite and in front of camera")
        if not math.isfinite(palm_width) or palm_width <= 1e-6:
            raise ValueError("palm_width_m must be finite and > 0")
        out_w, out_h = self.output_size
        view_width = palm_width * float(width_in_palm)
        view_height = palm_width * float(height_in_palm)
        x = ((np.arange(out_w, dtype=np.float64) + 0.5) / out_w - 0.5) * view_width
        y = ((np.arange(out_h, dtype=np.float64) + 0.5) / out_h - 0.5) * view_height
        xx, yy = np.meshgrid(x, y)
        theta = math.radians(float(rotation_deg))
        c, s = math.cos(theta), math.sin(theta)
        dx = c * xx - s * yy
        dy = s * xx + c * yy
        camera_points = np.empty((out_h * out_w, 3), dtype=np.float64)
        camera_points[:, 0] = pocket[0] + dx.reshape(-1)
        camera_points[:, 1] = pocket[1] + dy.reshape(-1)
        camera_points[:, 2] = pocket[2]
        source_uv = fisheye_unit_rays_to_pixels(camera_points, self.K, self.D)
        map_x = source_uv[:, 0].reshape(out_h, out_w).astype(np.float32)
        map_y = source_uv[:, 1].reshape(out_h, out_w).astype(np.float32)
        return cv2.remap(
            source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
        )

    def render_local_angular(
            self,
            image: np.ndarray,
            pocket_camera: np.ndarray,
            palm_across_camera: np.ndarray | None = None,
            *,
            horizontal_fov_deg: float = 78.0) -> np.ndarray:
        """Ray-resample a near view with a fixed angular field of view.

        Unlike :meth:`render_local`, this contains no depth or palm-width
        assumption.  It is therefore the human-side L counterpart for a
        directly predicted 2D functional pocket, and can be used identically
        on the robot side after FK projects the pocket ray.  The configured
        anchor_uv_ratio is the output pixel where the pocket ray lands.
        """
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {source.shape}")
        fov = math.radians(float(horizontal_fov_deg))
        if not math.isfinite(fov) or not math.radians(5.0) <= fov <= math.radians(170.0):
            raise ValueError("horizontal_fov_deg must be within [5,170]")
        across = np.array([1.0, 0.0, 0.0], dtype=np.float64) if palm_across_camera is None else palm_across_camera
        source_basis = _basis_from_anchor_and_across(normalize(pocket_camera), across, self.palm_across_sign)
        out_w, out_h = self.output_size
        focal = 0.5 * float(out_w) / math.tan(0.5 * fov)
        xx, yy = np.meshgrid(np.arange(out_w, dtype=np.float64), np.arange(out_h, dtype=np.float64))
        anchor_x = self.anchor_uv_ratio[0] * float(out_w - 1)
        anchor_y = self.anchor_uv_ratio[1] * float(out_h - 1)
        x = (xx.reshape(-1) - anchor_x) / focal
        y = (yy.reshape(-1) - anchor_y) / focal
        virtual_rays = np.stack([x, y, np.ones_like(x)], axis=-1)
        virtual_rays /= np.linalg.norm(virtual_rays, axis=-1, keepdims=True)
        source_rays = virtual_rays @ source_basis.T
        source_uv = fisheye_unit_rays_to_pixels(source_rays, self.K, self.D)
        map_x = source_uv[:, 0].reshape(out_h, out_w).astype(np.float32)
        map_y = source_uv[:, 1].reshape(out_h, out_w).astype(np.float32)
        return cv2.remap(source, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=self.border_mode)
