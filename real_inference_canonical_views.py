#!/usr/bin/env python3
"""Runtime dual-view observation renderer for Linker O6.

The policy receives only two grasp-anchored views derived from the same source
fisheye frame:
  - a wide canonical fisheye image under the source/global RGB key;
  - a metric near-contact image under a separate local RGB key.

No arm or hand action is modified here.
"""

from __future__ import annotations

import pathlib
import sys
import math

import cv2
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
SCRIPTS_REAL = ROOT / "scripts_real"
if str(SCRIPTS_REAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_REAL))

from fisheye_canonical_views import (  # noqa: E402
    FisheyeCanonicalViewRenderer,
    fisheye_unit_rays_to_pixels,
    normalize,
)
from o6_grasp_view import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_GRASP_POCKET_CONFIG,
    camera_points,
    compute_grasp_pocket,
    load_calibration,
    load_grasp_config,
)
from o6_fk21 import O6Kinematics  # noqa: E402


def _parse_size(value, field_name: str) -> tuple[int, int]:
    if isinstance(value, str):
        pieces = value.lower().replace(" ", "").split("x", 1)
        if len(pieces) != 2:
            raise ValueError(f"{field_name} must be [width,height] or WIDTHxHEIGHT")
        value = [int(piece) for piece in pieces]
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.shape != (2,) or np.any(array < 2):
        raise ValueError(f"{field_name} must contain width,height >= 2")
    return int(array[0]), int(array[1])


def _calibration_source_size(calibration) -> tuple[int, int]:
    raw = getattr(calibration, "raw", {}) or {}
    if "image_width" in raw and "image_height" in raw:
        width, height = int(raw["image_width"]), int(raw["image_height"])
        if width >= 2 and height >= 2:
            return width, height
    if "image_size" in raw:
        return _parse_size(raw["image_size"], "calibration.image_size")
    cx = float(calibration.K[0, 2])
    cy = float(calibration.K[1, 2])
    return max(2, int(round(cx * 2.0))), max(2, int(round(cy * 2.0)))


class O6CanonicalViewsRenderer:
    """Generate wide G and local L policy observations from one fisheye source."""

    def __init__(self, config: dict):
        self.config = dict(config or {})
        self.calibration_path = pathlib.Path(
            self.config.get("calibration", DEFAULT_CALIBRATION)
        ).expanduser().resolve()
        self.grasp_pocket_config_path = pathlib.Path(
            self.config.get("grasp_pocket_config", DEFAULT_GRASP_POCKET_CONFIG)
        ).expanduser().resolve()
        self.calibration = load_calibration(self.calibration_path)
        self.grasp_config = load_grasp_config(self.grasp_pocket_config_path)
        self.kinematics = O6Kinematics(self.calibration.urdf_path)
        self.camera_rotation, _ = cv2.Rodrigues(
            np.asarray(self.calibration.rvec, dtype=np.float64).reshape(3, 1)
        )

        self.source_rgb_key = str(self.config.get("source_rgb_key", "camera0_rgb"))
        self.global_output_key = str(
            self.config.get("global_output_key", self.source_rgb_key)
        )
        self.local_output_key = str(
            self.config.get("local_output_key", "camera0_local_rgb")
        )
        if not self.source_rgb_key or not self.global_output_key or not self.local_output_key:
            raise ValueError("canonical_views RGB keys must be non-empty")
        if self.global_output_key == self.local_output_key:
            raise ValueError("global_output_key and local_output_key must differ")

        global_cfg = dict(self.config.get("global", {}) or {})
        local_cfg = dict(self.config.get("local", {}) or {})
        self.global_size = _parse_size(
            global_cfg.get("output_size", self.config.get("output_size", [224, 224])),
            "canonical_views.global.output_size",
        )
        self.local_size = _parse_size(
            local_cfg.get("output_size", self.global_size),
            "canonical_views.local.output_size",
        )
        self.source_size = _parse_size(
            self.config.get("source_size", _calibration_source_size(self.calibration)),
            "canonical_views.source_size",
        )
        self.global_renderer = FisheyeCanonicalViewRenderer(
            self.calibration.K,
            self.calibration.D,
            output_size=self.global_size,
            source_size=self.source_size,
            anchor_uv_ratio=global_cfg.get("anchor_uv_ratio", [0.5, 0.35]),
            palm_across_sign=float(global_cfg.get("palm_across_sign", 1.0)),
            border=str(global_cfg.get("border_mode", "constant")),
        )
        self.global_roll_mode = str(global_cfg.get("roll_mode", "palm"))
        self.local_renderer = FisheyeCanonicalViewRenderer(
            self.calibration.K,
            self.calibration.D,
            output_size=self.local_size,
            source_size=self.local_size,
            anchor_uv_ratio=local_cfg.get("anchor_uv_ratio", [0.5, 0.5]),
            border=str(local_cfg.get("border_mode", "reflect101")),
        )
        self.local_width_in_palm = float(local_cfg.get("width_in_palm", 2.5))
        self.local_height_in_palm = float(local_cfg.get("height_in_palm", 2.5))
        self.local_rotation_deg = float(local_cfg.get("rotation_deg", 0.0))
        self.fixed_roll_deg = float(self.config.get("fixed_roll_deg", 0.0))
        self.local_mode = str(local_cfg.get("mode", "anchor_wide"))
        self.local_fov_deg = float(local_cfg.get("fov_deg", 78.0))
        self.ema_alpha = float(self.config.get("ema_alpha", 0.3))
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("canonical_views.ema_alpha must be in [0,1]")
        if self.local_mode not in {"metric_palm", "angular", "anchor_wide"}:
            raise ValueError("canonical_views.local.mode must be metric_palm, angular, or anchor_wide")
        if self.global_roll_mode not in {"palm", "camera"}:
            raise ValueError("canonical_views.global.roll_mode must be palm or camera")
        if self.local_mode == "metric_palm" and (self.local_width_in_palm <= 0.0 or self.local_height_in_palm <= 0.0):
            raise ValueError("canonical_views.local width/height_in_palm must be > 0")
        if self.local_mode == "angular" and not 5.0 <= self.local_fov_deg <= 170.0:
            raise ValueError("canonical_views.local.fov_deg must be in [5,170]")
        if not math.isfinite(self.fixed_roll_deg):
            raise ValueError("canonical_views.fixed_roll_deg must be finite")
        self._last_pocket_camera: np.ndarray | None = None
        self._last_palm_width: float | None = None
        self._last_palm_across_camera: np.ndarray | None = None

    def reset(self) -> None:
        self._last_pocket_camera = None
        self._last_palm_width = None
        self._last_palm_across_camera = None

    def validate_shape_meta(self, shape_meta: dict) -> None:
        obs_meta = shape_meta.get("obs", {})
        expected = {
            self.global_output_key: self.global_size,
            self.local_output_key: self.local_size,
        }
        horizons = []
        for key, (width, height) in expected.items():
            if key not in obs_meta:
                raise ValueError(
                    "canonical_views is enabled, but checkpoint shape_meta is missing "
                    f"RGB key {key!r}. Train a dual-view checkpoint first."
                )
            attr = obs_meta[key]
            if str(attr.get("type", "low_dim")) != "rgb":
                raise ValueError(f"shape_meta.obs.{key} must be type=rgb")
            shape = tuple(int(value) for value in attr.get("shape", ()))
            if shape != (3, height, width):
                raise ValueError(
                    f"shape_meta.obs.{key}.shape must be (3,{height},{width}), got {shape}"
                )
            horizons.append(int(attr.get("horizon", 1)))
        if len(set(horizons)) != 1:
            raise ValueError(
                "canonical global/local RGB horizons must match because both views "
                "are derived from the same source frames"
            )

    def metadata(self) -> dict:
        return {
            "enabled": True,
            "hand": "linker_o6",
            "sourceRgbKey": self.source_rgb_key,
            "globalOutputKey": self.global_output_key,
            "localOutputKey": self.local_output_key,
            "sourceSize": list(self.source_size),
            "fixedRollDeg": self.fixed_roll_deg,
            "calibration": str(self.calibration_path),
            "graspPocketConfig": str(self.grasp_pocket_config_path),
            "global": {
                "outputSize": list(self.global_size),
                "anchorUvRatio": self.global_renderer.anchor_uv_ratio.tolist(),
                "palmAcrossSign": self.global_renderer.palm_across_sign,
                "rollMode": self.global_roll_mode,
            },
            "local": {
                "outputSize": list(self.local_size),
                "anchorUvRatio": self.local_renderer.anchor_uv_ratio.tolist(),
                "mode": self.local_mode,
                "fovDeg": self.local_fov_deg,
                "widthInPalm": self.local_width_in_palm,
                "heightInPalm": self.local_height_in_palm,
                "rotationDeg": self.local_rotation_deg,
            },
            "emaAlpha": self.ema_alpha,
            "a": float(self.grasp_config["a"]),
            "b": float(self.grasp_config["b"]),
            "normalSign": float(self.grasp_config["normal_sign"]),
        }

    def apply(self, env_obs: dict, hand_state) -> dict:
        if self.source_rgb_key not in env_obs:
            raise KeyError(
                f"canonical_views source key {self.source_rgb_key!r} is absent from env obs"
            )
        frames = np.asarray(env_obs[self.source_rgb_key])
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"canonical_views source {self.source_rgb_key} must be (T,H,W,3), got {frames.shape}"
            )
        state = np.asarray(hand_state, dtype=np.float64).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("canonical_views requires a finite O6 measured state of shape (6,)")

        pocket_camera, palm_width, palm_across_camera = self._compute_smoothed_geometry(state)
        center_uv = fisheye_unit_rays_to_pixels(
            pocket_camera[None, :], self.calibration.K, self.calibration.D
        )[0]
        height, width = frames.shape[1:3]
        if not (0.0 <= center_uv[0] < width and 0.0 <= center_uv[1] < height):
            raise RuntimeError(
                "canonical_views grasp pocket projects outside the source RGB; "
                "stop instead of feeding an invalid fallback image"
            )

        global_frames = []
        local_frames = []
        for frame in frames:
            image, restore = self._to_uint8(frame)
            global_view = self._render_global_shift(image, center_uv)
            if self.local_mode == "anchor_wide":
                local_view, _ = self.local_renderer.render_wide(
                    image,
                    pocket_camera,
                    palm_across_camera,
                    virtual_roll_deg=self.fixed_roll_deg,
                )
            elif self.local_mode == "angular":
                local_view = self.local_renderer.render_local_angular(
                    image, pocket_camera, palm_across_camera,
                    horizontal_fov_deg=self.local_fov_deg,
                )
            else:
                local_view = self.local_renderer.render_local(
                    image,
                    pocket_camera,
                    palm_width,
                    width_in_palm=self.local_width_in_palm,
                    height_in_palm=self.local_height_in_palm,
                    rotation_deg=self.local_rotation_deg,
                )
            global_frames.append(restore(global_view))
            local_frames.append(restore(local_view))

        output = dict(env_obs)
        output[self.global_output_key] = np.stack(global_frames, axis=0)
        output[self.local_output_key] = np.stack(local_frames, axis=0)
        return output

    def _render_global_shift(self, image: np.ndarray, center_uv: np.ndarray) -> np.ndarray:
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got {source.shape}")
        out_w, out_h = self.global_size
        src_h, src_w = source.shape[:2]
        if (src_w, src_h) != (out_w, out_h):
            source = cv2.resize(source, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            center = np.asarray(center_uv, dtype=np.float64).reshape(2).copy()
            center[0] *= float(out_w) / float(src_w)
            center[1] *= float(out_h) / float(src_h)
        else:
            center = np.asarray(center_uv, dtype=np.float64).reshape(2)
        anchor = np.asarray(
            [
                self.global_renderer.anchor_uv_ratio[0] * float(out_w - 1),
                self.global_renderer.anchor_uv_ratio[1] * float(out_h - 1),
            ],
            dtype=np.float64,
        )
        matrix = np.asarray(
            [[1.0, 0.0, anchor[0] - center[0]],
             [0.0, 1.0, anchor[1] - center[1]]],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            source,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _compute_smoothed_geometry(
            self, state: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        points = self.kinematics.points21(state)
        pocket = compute_grasp_pocket(
            points,
            a=float(self.grasp_config["a"]),
            b=float(self.grasp_config["b"]),
            normal_sign=float(self.grasp_config["normal_sign"]),
        )
        pocket_camera = camera_points(pocket.point3d[None, :], self.calibration)[0]
        palm_across_o6 = np.asarray(points[17] - points[5], dtype=np.float64)
        palm_across_camera = self.camera_rotation @ palm_across_o6
        if not np.all(np.isfinite(pocket_camera)) or pocket_camera[2] <= 1e-6:
            raise RuntimeError("canonical_views grasp pocket is behind camera")
        palm_across_camera = (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
            if self.global_roll_mode == "camera"
            else normalize(palm_across_camera)
        )

        if self._last_pocket_camera is None or self.ema_alpha >= 1.0:
            smooth_pocket = pocket_camera
            smooth_width = float(pocket.palm_width)
            smooth_across = palm_across_camera
        else:
            alpha = self.ema_alpha
            smooth_pocket = (1.0 - alpha) * self._last_pocket_camera + alpha * pocket_camera
            smooth_width = (1.0 - alpha) * float(self._last_palm_width) + alpha * float(pocket.palm_width)
            smooth_across = normalize(
                (1.0 - alpha) * self._last_palm_across_camera + alpha * palm_across_camera,
                fallback=palm_across_camera,
            )
        self._last_pocket_camera = smooth_pocket.copy()
        self._last_palm_width = float(smooth_width)
        self._last_palm_across_camera = smooth_across.copy()
        return smooth_pocket, float(smooth_width), smooth_across

    @staticmethod
    def _to_uint8(image: np.ndarray):
        arr = np.asarray(image)
        if arr.dtype == np.uint8:
            return arr.copy(), lambda value: value.astype(np.uint8, copy=False)
        dtype = arr.dtype
        work = arr.astype(np.float32, copy=False)
        finite = work[np.isfinite(work)]
        max_value = float(np.max(finite)) if finite.size else 1.0
        if max_value <= 1.5:
            converted = np.clip(np.rint(work * 255.0), 0, 255).astype(np.uint8)

            def restore(value: np.ndarray) -> np.ndarray:
                return (value.astype(np.float32) / 255.0).astype(dtype, copy=False)

            return converted, restore
        converted = np.clip(np.rint(work), 0, 255).astype(np.uint8)

        def restore(value: np.ndarray) -> np.ndarray:
            return value.astype(dtype, copy=False)

        return converted, restore


def build_canonical_views_renderer(cfg: dict, hand_backend: str | None):
    view_cfg = cfg.get("canonical_views", {}) or {}
    if not bool(view_cfg.get("enabled", False)):
        return None
    view_hand = str(view_cfg.get("hand", "linker_o6")).strip().lower()
    if view_hand != "linker_o6":
        raise ValueError("canonical_views currently supports hand=linker_o6 only")
    if hand_backend != "linker_o6":
        raise ValueError("canonical_views.hand=linker_o6 requires runtime hand=linker_o6")
    if bool((cfg.get("grasp_view", {}) or {}).get("enabled", False)):
        raise ValueError("Use canonical_views, not the deprecated single grasp_view")
    renderer = O6CanonicalViewsRenderer(view_cfg)
    print(
        "[INFO] canonical_views enabled: "
        f"source={renderer.source_rgb_key}, global={renderer.global_output_key}, "
        f"local={renderer.local_output_key}, global_size={renderer.global_size}, "
        f"local_size={renderer.local_size}, "
        f"anchor_ratio={renderer.global_renderer.anchor_uv_ratio.tolist()}"
    )
    return renderer
