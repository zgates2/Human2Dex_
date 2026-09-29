#!/usr/bin/env python3
"""Runtime O6 grasp-pocket local-view renderer.

This module changes only RGB observations.  It uses the measured O6 state,
the fixed camera<-O6_base calibration, and grasp_pocket_v1 to replace selected
RGB observation keys with fisheye-corrected local views.  It never changes
actions or robot control.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Iterable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
SCRIPTS_REAL = ROOT / "scripts_real"
if str(SCRIPTS_REAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_REAL))

from o6_grasp_view import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_GRASP_POCKET_CONFIG,
    camera_points,
    compute_grasp_pocket,
    load_calibration,
    load_grasp_config,
    render_grasp_view,
)
from o6_fk21 import O6Kinematics  # noqa: E402


class O6GraspViewRenderer:
    """Replace selected RGB sequences with views centered on grasp_pocket_v1."""

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

        rgb_keys = self.config.get("rgb_keys")
        if isinstance(rgb_keys, str):
            rgb_keys = [rgb_keys]
        self.rgb_keys = None if rgb_keys is None else {str(key) for key in rgb_keys}
        output_size = self.config.get("output_size", [224, 224])
        if isinstance(output_size, str):
            text = output_size.lower().replace(" ", "")
            output_size = [int(part) for part in text.split("x", 1)]
        if len(output_size) != 2:
            raise ValueError("grasp_view.output_size must be [width, height]")
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.width_in_palm = float(self.config.get("width_in_palm", 2.5))
        self.height_in_palm = float(self.config.get("height_in_palm", 2.5))
        self.rotation_deg = float(self.config.get("rotation_deg", 0.0))
        self.border_mode = str(self.config.get("border_mode", "reflect101"))
        self.ema_alpha = float(self.config.get("ema_alpha", 0.3))
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("grasp_view.ema_alpha must be in [0,1]")
        self._last_pocket_camera: np.ndarray | None = None
        self._last_palm_width: float | None = None
        self._warned_invalid = False

    def reset(self) -> None:
        self._last_pocket_camera = None
        self._last_palm_width = None
        self._warned_invalid = False

    def metadata(self) -> dict:
        return {
            "enabled": True,
            "hand": "linker_o6",
            "calibration": str(self.calibration_path),
            "graspPocketConfig": str(self.grasp_pocket_config_path),
            "rgbKeys": None if self.rgb_keys is None else sorted(self.rgb_keys),
            "outputSize": list(self.output_size),
            "widthInPalm": self.width_in_palm,
            "heightInPalm": self.height_in_palm,
            "rotationDeg": self.rotation_deg,
            "borderMode": self.border_mode,
            "emaAlpha": self.ema_alpha,
            "a": float(self.grasp_config["a"]),
            "b": float(self.grasp_config["b"]),
            "normalSign": float(self.grasp_config["normal_sign"]),
        }

    def apply(self, env_obs: dict, hand_state) -> dict:
        if hand_state is None:
            return env_obs
        state = np.asarray(hand_state, dtype=np.float64).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            return env_obs
        try:
            points = self.kinematics.points21(state)
            pocket = compute_grasp_pocket(
                points,
                a=float(self.grasp_config["a"]),
                b=float(self.grasp_config["b"]),
                normal_sign=float(self.grasp_config["normal_sign"]),
            )
            pocket_camera = camera_points(
                pocket.point3d[None, :], self.calibration
            )[0]
            if not np.all(np.isfinite(pocket_camera)) or pocket_camera[2] <= 1e-6:
                return self._invalid_result(env_obs, "pocket is behind camera")
            if self._last_pocket_camera is None or self.ema_alpha >= 1.0:
                smooth_pocket = pocket_camera.copy()
                smooth_width = float(pocket.palm_width)
            else:
                alpha = max(0.0, min(1.0, self.ema_alpha))
                smooth_pocket = (
                    (1.0 - alpha) * self._last_pocket_camera
                    + alpha * pocket_camera
                )
                smooth_width = (
                    (1.0 - alpha) * float(self._last_palm_width)
                    + alpha * float(pocket.palm_width)
                )
            self._last_pocket_camera = smooth_pocket
            self._last_palm_width = smooth_width
        except Exception as exc:
            return self._invalid_result(env_obs, f"compute failed: {exc}")

        output = dict(env_obs)
        for key, value in env_obs.items():
            if not self._should_process_key(key, value):
                continue
            frames = np.asarray(value)
            rendered = []
            for frame in frames:
                image, restore = self._to_uint8(frame)
                view = render_grasp_view(
                    image,
                    smooth_pocket,
                    smooth_width,
                    self.calibration,
                    output_size=self.output_size,
                    width_in_palm=self.width_in_palm,
                    height_in_palm=self.height_in_palm,
                    rotation_deg=self.rotation_deg,
                    border_mode=self.border_mode,
                )
                rendered.append(restore(view))
            output[key] = np.stack(rendered, axis=0)
        return output

    def _invalid_result(self, env_obs: dict, reason: str) -> dict:
        if not self._warned_invalid:
            print(f"[WARN] grasp_view skipped: {reason}")
            self._warned_invalid = True
        return env_obs

    def _should_process_key(self, key: str, value) -> bool:
        if self.rgb_keys is not None and str(key) not in self.rgb_keys:
            return False
        array = np.asarray(value)
        return str(key).endswith("_rgb") and array.ndim == 4 and array.shape[-1] == 3

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


def build_grasp_view_renderer(cfg: dict, hand_backend: str | None):
    view_cfg = cfg.get("grasp_view", {}) or {}
    if not bool(view_cfg.get("enabled", False)):
        return None
    view_hand = str(view_cfg.get("hand", "linker_o6")).strip().lower()
    if view_hand != "linker_o6":
        raise ValueError("grasp_view currently supports hand=linker_o6 only")
    if hand_backend != "linker_o6":
        raise ValueError("grasp_view.hand=linker_o6 requires runtime hand=linker_o6")
    renderer = O6GraspViewRenderer(view_cfg)
    print(
        "[INFO] grasp_view enabled: "
        f"calibration={renderer.calibration_path}, "
        f"config={renderer.grasp_pocket_config_path}, "
        f"rgb_keys={renderer.rgb_keys or 'all *_rgb'}, "
        f"size={renderer.output_size}, "
        f"scale={renderer.width_in_palm:.2f}x{renderer.height_in_palm:.2f} palm, "
        f"ema_alpha={renderer.ema_alpha:.2f}"
    )
    return renderer

