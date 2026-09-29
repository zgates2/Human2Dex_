"""Camera-mount perturbation for offline wrist-view augmentation.

This module applies an episode-stable full-image affine perturbation to RGB
frames. It is intentionally independent from the DP training transform so the
augmented dataset can be inspected, converted to Zarr, and reproduced.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from PIL import Image


FloatRange = Tuple[float, float]


@dataclass(frozen=True)
class CameraMountAugConfig:
    enabled: bool = False
    p: float = 0.8
    rotation_deg: FloatRange = (-10.0, 10.0)
    translate_ratio_x: FloatRange = (-0.10, 0.10)
    translate_ratio_y: FloatRange = (-0.10, 0.10)
    scale: FloatRange = (0.88, 1.15)
    shear_deg: FloatRange = (-4.0, 4.0)
    frame_jitter_enabled: bool = True
    frame_jitter_rotation_deg: FloatRange = (-1.0, 1.0)
    frame_jitter_translate_px: FloatRange = (-3.0, 3.0)
    frame_jitter_scale: FloatRange = (0.99, 1.01)
    interpolation: str = "bilinear"
    padding_mode: str = "edge"
    fill_value: int = 0


@dataclass(frozen=True)
class CameraMountProfile:
    applied: bool
    seed: int
    rotation_deg: float = 0.0
    translate_ratio_x: float = 0.0
    translate_ratio_y: float = 0.0
    scale: float = 1.0
    shear_x_deg: float = 0.0
    shear_y_deg: float = 0.0


def _stable_seed(*parts) -> int:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:8], "little") % (2**31 - 1)


def _coerce_range(value: Sequence[float], name: str) -> FloatRange:
    if isinstance(value, (int, float)):
        lo = hi = float(value)
    else:
        if len(value) == 1:
            lo = hi = float(value[0])
        elif len(value) == 2:
            lo, hi = float(value[0]), float(value[1])
        else:
            raise ValueError(f"{name} must be VALUE or [MIN, MAX]")
    if hi < lo:
        raise ValueError(f"{name} max must be >= min")
    return lo, hi


def camera_mount_config_from_args(args) -> CameraMountAugConfig:
    return CameraMountAugConfig(
        enabled=bool(getattr(args, "camera_mount_aug_enabled", False)),
        p=float(getattr(args, "camera_mount_aug_p", 0.8)),
        rotation_deg=_coerce_range(getattr(args, "camera_mount_rotation_deg", (-10.0, 10.0)), "camera_mount_rotation_deg"),
        translate_ratio_x=_coerce_range(getattr(args, "camera_mount_translate_ratio_x", (-0.10, 0.10)), "camera_mount_translate_ratio_x"),
        translate_ratio_y=_coerce_range(getattr(args, "camera_mount_translate_ratio_y", (-0.10, 0.10)), "camera_mount_translate_ratio_y"),
        scale=_coerce_range(getattr(args, "camera_mount_scale", (0.88, 1.15)), "camera_mount_scale"),
        shear_deg=_coerce_range(getattr(args, "camera_mount_shear_deg", (-4.0, 4.0)), "camera_mount_shear_deg"),
        frame_jitter_enabled=bool(getattr(args, "camera_mount_frame_jitter_enabled", True)),
        frame_jitter_rotation_deg=_coerce_range(getattr(args, "camera_mount_frame_jitter_rotation_deg", (-1.0, 1.0)), "camera_mount_frame_jitter_rotation_deg"),
        frame_jitter_translate_px=_coerce_range(getattr(args, "camera_mount_frame_jitter_translate_px", (-3.0, 3.0)), "camera_mount_frame_jitter_translate_px"),
        frame_jitter_scale=_coerce_range(getattr(args, "camera_mount_frame_jitter_scale", (0.99, 1.01)), "camera_mount_frame_jitter_scale"),
        interpolation=str(getattr(args, "camera_mount_interpolation", "bilinear")),
        padding_mode=str(getattr(args, "camera_mount_padding_mode", "edge")),
        fill_value=int(getattr(args, "camera_mount_fill_value", 0)),
    )


def validate_camera_mount_config(config: CameraMountAugConfig):
    if not 0.0 <= config.p <= 1.0:
        raise ValueError("camera_mount_aug.p must be in [0, 1]")
    if config.scale[0] <= 0.0:
        raise ValueError("camera_mount_aug.scale must be positive")
    if config.frame_jitter_scale[0] <= 0.0:
        raise ValueError("camera_mount_aug.frame_jitter.scale must be positive")
    if config.interpolation not in {"nearest", "bilinear", "bicubic"}:
        raise ValueError("camera_mount_aug.interpolation must be nearest/bilinear/bicubic")
    if config.padding_mode not in {"edge", "reflect", "constant"}:
        raise ValueError("camera_mount_aug.padding_mode must be edge/reflect/constant")


def _uniform(rng: np.random.Generator, value_range: FloatRange) -> float:
    lo, hi = value_range
    if abs(hi - lo) < 1e-12:
        return float(lo)
    return float(rng.uniform(lo, hi))


def sample_camera_mount_profile(
    config: CameraMountAugConfig,
    base_seed: int,
    episode_name: str,
    variant: int,
    image_subdir: str = "images",
) -> CameraMountProfile:
    """Sample one stable mount profile for an episode/variant/stream."""
    seed = _stable_seed(base_seed, episode_name, variant, image_subdir, "camera_mount")
    if not config.enabled:
        return CameraMountProfile(applied=False, seed=seed)
    rng = np.random.default_rng(seed)
    if rng.random() > config.p:
        return CameraMountProfile(applied=False, seed=seed)
    return CameraMountProfile(
        applied=True,
        seed=seed,
        rotation_deg=_uniform(rng, config.rotation_deg),
        translate_ratio_x=_uniform(rng, config.translate_ratio_x),
        translate_ratio_y=_uniform(rng, config.translate_ratio_y),
        scale=_uniform(rng, config.scale),
        shear_x_deg=_uniform(rng, config.shear_deg),
        shear_y_deg=_uniform(rng, config.shear_deg),
    )


def _frame_jitter(config: CameraMountAugConfig, profile: CameraMountProfile, frame_index: int) -> Dict[str, float]:
    if (not profile.applied) or (not config.frame_jitter_enabled):
        return {"rotation_deg": 0.0, "translate_x_px": 0.0, "translate_y_px": 0.0, "scale": 1.0}
    rng = np.random.default_rng(_stable_seed(profile.seed, frame_index, "frame_jitter"))
    return {
        "rotation_deg": _uniform(rng, config.frame_jitter_rotation_deg),
        "translate_x_px": _uniform(rng, config.frame_jitter_translate_px),
        "translate_y_px": _uniform(rng, config.frame_jitter_translate_px),
        "scale": _uniform(rng, config.frame_jitter_scale),
    }


def frame_camera_mount_params(
    config: CameraMountAugConfig,
    profile: CameraMountProfile,
    frame_index: int,
    image_shape: Tuple[int, int],
) -> Dict[str, float]:
    h, w = int(image_shape[0]), int(image_shape[1])
    jitter = _frame_jitter(config, profile, frame_index)
    return {
        "applied": bool(profile.applied),
        "rotation_deg": float(profile.rotation_deg + jitter["rotation_deg"]),
        "translate_x_px": float(profile.translate_ratio_x * w + jitter["translate_x_px"]),
        "translate_y_px": float(profile.translate_ratio_y * h + jitter["translate_y_px"]),
        "scale": float(profile.scale * jitter["scale"]),
        "shear_x_deg": float(profile.shear_x_deg),
        "shear_y_deg": float(profile.shear_y_deg),
    }


def _forward_affine_matrix(
    width: int,
    height: int,
    rotation_deg: float,
    translate_x_px: float,
    translate_y_px: float,
    scale: float,
    shear_x_deg: float,
    shear_y_deg: float,
) -> np.ndarray:
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    shx = math.tan(math.radians(shear_x_deg))
    shy = math.tan(math.radians(shear_y_deg))

    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    scale_m = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    shear_m = np.array([[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rotate_m = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    back = np.array([[1.0, 0.0, cx + translate_x_px], [0.0, 1.0, cy + translate_y_px], [0.0, 0.0, 1.0]], dtype=np.float64)
    return back @ rotate_m @ shear_m @ scale_m @ to_origin


def camera_mount_affine_matrix(
    config: CameraMountAugConfig,
    profile: CameraMountProfile,
    frame_index: int,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Return the forward source-RGB -> augmented-RGB affine matrix."""
    params = frame_camera_mount_params(config, profile, frame_index, image_shape)
    h, w = int(image_shape[0]), int(image_shape[1])
    return _forward_affine_matrix(
        width=w,
        height=h,
        rotation_deg=params["rotation_deg"],
        translate_x_px=params["translate_x_px"],
        translate_y_px=params["translate_y_px"],
        scale=params["scale"],
        shear_x_deg=params["shear_x_deg"],
        shear_y_deg=params["shear_y_deg"],
    )


def apply_camera_mount_to_points(
    points_uv: np.ndarray,
    config: CameraMountAugConfig,
    profile: CameraMountProfile,
    frame_index: int,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Transform 2D pixel points with the same affine used for RGB warping."""
    points = np.asarray(points_uv, dtype=np.float32)
    if points.shape[-1] != 2:
        raise ValueError(f"points_uv must have last dim 2, got {points.shape}")
    if not profile.applied:
        return points.astype(np.float32, copy=True)
    matrix = camera_mount_affine_matrix(config, profile, frame_index, image_shape)
    flat = points.reshape(-1, 2).astype(np.float64)
    homo = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    warped = homo @ matrix.T
    return warped[:, :2].reshape(points.shape).astype(np.float32)


def _pil_resample(name: str, is_mask: bool):
    if is_mask or name == "nearest":
        return Image.Resampling.NEAREST
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    return Image.Resampling.BILINEAR


def _edge_padding_pixels(width: int, height: int, params: Dict[str, float]) -> int:
    max_dim = max(width, height)
    rotation_pad = abs(math.sin(math.radians(params["rotation_deg"]))) * max_dim
    shear_pad = (
        abs(math.tan(math.radians(params["shear_x_deg"])))
        + abs(math.tan(math.radians(params["shear_y_deg"])))
    ) * max_dim
    scale_pad = abs(1.0 / max(params["scale"], 1e-6) - 1.0) * max_dim * 0.5
    translate_pad = max(abs(params["translate_x_px"]), abs(params["translate_y_px"]))
    return int(max(4, min(max_dim, math.ceil(rotation_pad + shear_pad + scale_pad + translate_pad + 8))))


def _warp_array(
    array: np.ndarray,
    params: Dict[str, float],
    interpolation: str,
    padding_mode: str,
    fill_value: int,
    is_mask: bool = False,
) -> np.ndarray:
    h, w = array.shape[:2]
    forward = _forward_affine_matrix(
        width=w,
        height=h,
        rotation_deg=params["rotation_deg"],
        translate_x_px=params["translate_x_px"],
        translate_y_px=params["translate_y_px"],
        scale=params["scale"],
        shear_x_deg=params["shear_x_deg"],
        shear_y_deg=params["shear_y_deg"],
    )
    inverse = np.linalg.inv(forward)

    if is_mask:
        pil = Image.fromarray((array.astype(np.uint8) * 255), mode="L")
    else:
        pil = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")

    if padding_mode == "constant":
        padded = pil
        offset_x = 0
        offset_y = 0
        fill = int(fill_value)
    else:
        pad = _edge_padding_pixels(w, h, params)
        pad_mode = "edge" if padding_mode == "edge" else "reflect"
        np_array = np.asarray(pil)
        if is_mask:
            padded_np = np.pad(np_array, ((pad, pad), (pad, pad)), mode=pad_mode)
        else:
            padded_np = np.pad(np_array, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)
        padded = Image.fromarray(padded_np, mode=("L" if is_mask else "RGB"))
        offset_x = pad
        offset_y = pad
        fill = None

    data = (
        float(inverse[0, 0]),
        float(inverse[0, 1]),
        float(inverse[0, 2] + offset_x),
        float(inverse[1, 0]),
        float(inverse[1, 1]),
        float(inverse[1, 2] + offset_y),
    )
    kwargs = {}
    if fill is not None:
        kwargs["fillcolor"] = fill
    warped = padded.transform(
        (w, h),
        Image.Transform.AFFINE,
        data,
        resample=_pil_resample(interpolation, is_mask=is_mask),
        **kwargs,
    )
    out = np.asarray(warped)
    if is_mask:
        return out > 127
    return out.astype(np.uint8)


def apply_camera_mount_to_rgb(
    image: np.ndarray,
    config: CameraMountAugConfig,
    profile: CameraMountProfile,
    frame_index: int,
) -> np.ndarray:
    if not profile.applied:
        return image
    params = frame_camera_mount_params(config, profile, frame_index, image.shape[:2])
    return _warp_array(
        image,
        params=params,
        interpolation=config.interpolation,
        padding_mode=config.padding_mode,
        fill_value=config.fill_value,
        is_mask=False,
    )


def apply_camera_mount_to_mask(
    mask: np.ndarray,
    config: CameraMountAugConfig,
    profile: CameraMountProfile,
    frame_index: int,
) -> np.ndarray:
    if not profile.applied:
        return mask
    params = frame_camera_mount_params(config, profile, frame_index, mask.shape[:2])
    return _warp_array(
        mask.astype(bool),
        params=params,
        interpolation="nearest",
        padding_mode=config.padding_mode,
        fill_value=0,
        is_mask=True,
    )


def summarize_camera_mount_config(config: CameraMountAugConfig) -> Dict:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, tuple):
            data[key] = list(value)
    return data


def summarize_camera_mount_profile(profile: CameraMountProfile) -> Dict:
    return asdict(profile)
