#!/usr/bin/env python3
"""Runtime RGB skeleton overlay for deployment-time visual canonicalization."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Iterable

import cv2
import numpy as np


ROOT = pathlib.Path(__file__).parent
SCRIPTS_REAL = ROOT / "scripts_real"
if str(SCRIPTS_REAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_REAL))

from calibrate_o6_camera_extrinsic import project_fisheye
from o6_fk21 import HAND21_BONES, O6Kinematics
from wuji_fk21 import WujiKinematics


DEFAULT_O6_CALIBRATION = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "o6_camera_calibration/mvs_DA9057801_o6_mount_v1/fit_ep0003_v1/"
    "camera_from_o6_base.json"
)
DEFAULT_WUJI_CALIBRATION = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "wuji_camera_calibration/mvs_DA9057802_wuji_mount_v1/fit_20260814_231759/"
    "camera_from_wuji_base.json"
)


# Exact RGB palette used by the training-side
# glove_aug_pipeline/wrist_skeleton_overlay.py.  Keep this explicit instead of
# converting the O6 diagnostic BGR palette, whose index/ring/pinky colors differ.
POINT21_COLORS_RGB = [
    (245, 245, 245),
    *((255, 180, 80) for _ in range(4)),
    *((80, 255, 120) for _ in range(4)),
    *((80, 210, 255) for _ in range(4)),
    *((255, 120, 180) for _ in range(4)),
    *((180, 120, 255) for _ in range(4)),
]


class O6SkeletonOverlayRenderer:
    """Draw O6 FK skeleton onto RGB observation frames before policy inference."""

    def __init__(
            self,
            calibration_path: str | pathlib.Path,
            *,
            rgb_keys: Iterable[str] | None = None,
            line_thickness: int = 1,
            point_radius: int = 2,
            draw_wrist: bool = True,
            wrist_extension_ratio: float = 0.85,
            kinematics_cls=O6Kinematics,
            rvec_key: str = "rvec_camera_from_o6_base",
            tvec_key: str = "tvec_camera_from_o6_base_m",
            expected_state_dim: int = 6,
            hand_name: str = "linker_o6"):
        path = pathlib.Path(calibration_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)
        self.calibration_path = path
        self.K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(calibration["D"], dtype=np.float64).reshape(4, 1)
        rvec_value = calibration.get(rvec_key, calibration.get("rvec_camera_from_o6_base"))
        tvec_value = calibration.get(tvec_key, calibration.get("tvec_camera_from_o6_base_m"))
        if rvec_value is None or tvec_value is None:
            raise KeyError(f"{path} does not contain {rvec_key}/{tvec_key}")
        self.rvec = np.asarray(
            rvec_value, dtype=np.float64
        ).reshape(3)
        self.tvec = np.asarray(
            tvec_value, dtype=np.float64
        ).reshape(3)
        try:
            self.kinematics = kinematics_cls(calibration["urdf_path"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Failed to initialize skeleton_overlay kinematics. "
                f"hand={hand_name!r}, calibration={path}, "
                f"urdf_path={calibration.get('urdf_path')!r}. "
                "Check that skeleton_overlay.hand matches the calibration file."
            ) from exc
        self.rgb_keys = None if rgb_keys is None else {str(key) for key in rgb_keys}
        self.line_thickness = max(1, int(line_thickness))
        self.point_radius = max(1.0, float(point_radius))
        self.draw_wrist = bool(draw_wrist)
        self.wrist_extension_ratio = float(wrist_extension_ratio)
        self.expected_state_dim = int(expected_state_dim)
        self.hand_name = str(hand_name)

    def metadata(self) -> dict:
        return {
            "enabled": True,
            "hand": self.hand_name,
            "calibration": str(self.calibration_path),
            "rgbKeys": None if self.rgb_keys is None else sorted(self.rgb_keys),
            "drawWrist": self.draw_wrist,
            "lineThickness": self.line_thickness,
            "pointRadius": self.point_radius,
            "wristSource": "extrapolated_3d_wrist",
            "wristExtensionRatio": self.wrist_extension_ratio,
        }

    def apply(self, env_obs: dict, hand_state) -> dict:
        if hand_state is None:
            return env_obs
        state = np.asarray(hand_state, dtype=np.float64).reshape(-1)
        if state.shape != (self.expected_state_dim,) or not np.all(np.isfinite(state)):
            return env_obs

        output = dict(env_obs)
        for key, value in env_obs.items():
            if not self._should_process_key(key, value):
                continue
            output[key] = self._overlay_sequence(np.asarray(value), state)
        return output

    def _should_process_key(self, key: str, value) -> bool:
        if self.rgb_keys is not None and str(key) not in self.rgb_keys:
            return False
        array = np.asarray(value)
        return str(key).endswith("_rgb") and array.ndim == 4 and array.shape[-1] == 3

    def _overlay_sequence(self, images: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = np.asarray(images).copy()
        points21 = self.kinematics.points21(state)
        uv21, depth21 = project_fisheye(points21, self.rvec, self.tvec, self.K, self.D)
        for index in range(result.shape[0]):
            result[index] = self._overlay_one(result[index], points21, uv21, depth21)
        return result

    def _overlay_one(
            self,
            image: np.ndarray,
            points21: np.ndarray,
            uv21: np.ndarray,
            depth21: np.ndarray) -> np.ndarray:
        image_uint8, restore = self._to_uint8_rgb(image)
        height, width = image_uint8.shape[:2]
        valid = self._projectable_uv_mask(uv21, depth21)
        uv_draw = self._clip_uv_to_image(uv21, width, height)
        if self.draw_wrist:
            wrist_uv = self._estimate_wrist_uv(
                points21,
                uv21,
                valid,
            )
            if wrist_uv is None:
                valid[0] = False
            else:
                uv_draw[0] = self._clip_uv_to_image(wrist_uv, width, height)
                valid[0] = True
        else:
            valid[0] = False

        for a, b in HAND21_BONES:
            if not (valid[a] and valid[b]):
                continue
            pa = tuple(np.round(uv_draw[a]).astype(int))
            pb = tuple(np.round(uv_draw[b]).astype(int))
            cv2.line(
                image_uint8,
                pa,
                pb,
                POINT21_COLORS_RGB[b],
                self.line_thickness,
                cv2.LINE_AA,
            )

        for point_index, uv in enumerate(uv_draw):
            if not valid[point_index]:
                continue
            self._draw_antialiased_point(
                image_uint8,
                uv,
                self.point_radius,
                POINT21_COLORS_RGB[point_index],
            )
        return restore(image_uint8)

    def _estimate_wrist_uv(
            self,
            points21: np.ndarray,
            uv21: np.ndarray,
            valid: np.ndarray):
        palm_indices = (1, 5, 9, 13, 17)
        tip_indices = (4, 8, 12, 16, 20)
        palm = np.asarray(
            [points21[index] for index in palm_indices if valid[index]],
            dtype=np.float64,
        )
        tips = np.asarray(
            [points21[index] for index in tip_indices if valid[index]],
            dtype=np.float64,
        )
        if len(palm) >= 3 and len(tips) >= 3:
            palm_center = palm.mean(axis=0)
            tip_center = tips.mean(axis=0)
            candidate = (
                palm_center
                + self.wrist_extension_ratio * (palm_center - tip_center)
            )
            candidate_uv, candidate_depth = project_fisheye(
                candidate[None, :],
                self.rvec,
                self.tvec,
                self.K,
                self.D,
            )
            if (
                    float(candidate_depth[0]) > 0.0
                    and np.all(np.isfinite(candidate_uv[0]))):
                return candidate_uv[0]
            candidate_uv = self._estimate_wrist_uv_2d(uv21, valid)
            if candidate_uv is not None:
                return candidate_uv
        if valid[0]:
            return uv21[0]
        return None

    def _estimate_wrist_uv_2d(
            self,
            uv21: np.ndarray,
            valid: np.ndarray) -> np.ndarray | None:
        palm_indices = (1, 5, 9, 13, 17)
        tip_indices = (4, 8, 12, 16, 20)
        palm = np.asarray(
            [uv21[index] for index in palm_indices if valid[index]],
            dtype=np.float64,
        )
        tips = np.asarray(
            [uv21[index] for index in tip_indices if valid[index]],
            dtype=np.float64,
        )
        if len(palm) >= 3 and len(tips) >= 3:
            palm_center = palm.mean(axis=0)
            tip_center = tips.mean(axis=0)
            candidate = (
                palm_center
                + self.wrist_extension_ratio * (palm_center - tip_center)
            )
            if np.all(np.isfinite(candidate)):
                return candidate
        return None

    @staticmethod
    def _projectable_uv_mask(uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        uv_array = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        depth_array = np.asarray(depth, dtype=np.float64).reshape(-1)
        return (
            (depth_array > 0.0)
            & np.isfinite(depth_array)
            & np.all(np.isfinite(uv_array), axis=1)
        )

    @staticmethod
    def _clip_uv_to_image(uv: np.ndarray, width: int, height: int) -> np.ndarray:
        clipped = np.asarray(uv, dtype=np.float64).copy()
        clipped[..., 0] = np.clip(clipped[..., 0], 0.0, max(0.0, float(width - 1)))
        clipped[..., 1] = np.clip(clipped[..., 1], 0.0, max(0.0, float(height - 1)))
        return clipped

    @staticmethod
    def _draw_antialiased_point(
            image_uint8: np.ndarray,
            center: np.ndarray,
            radius: float,
            color: Iterable[int]) -> None:
        height, width = image_uint8.shape[:2]
        x, y = [float(value) for value in np.asarray(center).reshape(2)]
        r = max(1.0, float(radius))
        x0 = max(0, int(np.floor(x - r - 1.0)))
        x1 = min(width - 1, int(np.ceil(x + r + 1.0)))
        y0 = max(0, int(np.floor(y - r - 1.0)))
        y1 = min(height - 1, int(np.ceil(y + r + 1.0)))
        if x1 < x0 or y1 < y0:
            return
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        distance = np.sqrt((xx.astype(np.float64) - x) ** 2 + (yy.astype(np.float64) - y) ** 2)
        alpha = np.clip(r + 0.5 - distance, 0.0, 1.0)[..., None]
        if not np.any(alpha > 0.0):
            return
        patch = image_uint8[y0:y1 + 1, x0:x1 + 1].astype(np.float32, copy=False)
        color_arr = np.asarray([int(value) for value in color], dtype=np.float32).reshape(1, 1, 3)
        blended = patch * (1.0 - alpha) + color_arr * alpha
        image_uint8[y0:y1 + 1, x0:x1 + 1] = np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray):
        arr = np.asarray(image)
        if arr.dtype == np.uint8:
            return arr.copy(), lambda value: value.astype(np.uint8, copy=False)

        dtype = arr.dtype
        arr_float = arr.astype(np.float32, copy=False)
        finite = arr_float[np.isfinite(arr_float)]
        max_value = float(np.max(finite)) if finite.size else 1.0
        if max_value <= 1.5:
            uint8 = np.clip(np.rint(arr_float * 255.0), 0, 255).astype(np.uint8)

            def restore(value: np.ndarray) -> np.ndarray:
                return (value.astype(np.float32) / 255.0).astype(dtype, copy=False)

            return uint8, restore

        uint8 = np.clip(np.rint(arr_float), 0, 255).astype(np.uint8)

        def restore(value: np.ndarray) -> np.ndarray:
            return value.astype(dtype, copy=False)

        return uint8, restore


class WujiSkeletonOverlayRenderer(O6SkeletonOverlayRenderer):
    """Draw Wuji FK skeleton onto RGB observation frames before policy inference."""

    def __init__(self, calibration_path: str | pathlib.Path, **kwargs):
        super().__init__(
            calibration_path,
            kinematics_cls=WujiKinematics,
            rvec_key="rvec_camera_from_wuji_base",
            tvec_key="tvec_camera_from_wuji_base_m",
            expected_state_dim=20,
            hand_name="wuji_hand",
            **kwargs,
        )


def build_skeleton_overlay_renderer(cfg: dict, hand_backend: str | None):
    overlay_cfg = cfg.get("skeleton_overlay", {}) or {}
    if not bool(overlay_cfg.get("enabled", False)):
        return None
    overlay_hand = str(overlay_cfg.get("hand", "linker_o6"))
    if overlay_hand == "wujihand":
        overlay_hand = "wuji_hand"
    if overlay_hand not in {"linker_o6", "wuji_hand"}:
        raise ValueError("skeleton_overlay supports hand=linker_o6 or hand=wuji_hand")
    if hand_backend != overlay_hand:
        raise ValueError(
            f"skeleton_overlay.hand={overlay_hand} requires runtime hand={overlay_hand}"
        )
    rgb_keys = overlay_cfg.get("rgb_keys")
    if rgb_keys is not None and isinstance(rgb_keys, str):
        rgb_keys = [rgb_keys]
    renderer_cls = (
        O6SkeletonOverlayRenderer if overlay_hand == "linker_o6" else WujiSkeletonOverlayRenderer
    )
    default_calibration = (
        DEFAULT_O6_CALIBRATION if overlay_hand == "linker_o6" else DEFAULT_WUJI_CALIBRATION
    )
    renderer = renderer_cls(
        overlay_cfg.get("calibration", default_calibration),
        rgb_keys=rgb_keys,
        line_thickness=int(overlay_cfg.get("line_thickness", 1)),
        point_radius=float(overlay_cfg.get("point_radius", 2)),
        draw_wrist=bool(overlay_cfg.get("draw_wrist", True)),
        wrist_extension_ratio=float(overlay_cfg.get("wrist_extension_ratio", 0.85)),
    )
    print(
        "[INFO] skeleton_overlay enabled: "
        f"calibration={renderer.calibration_path}, rgb_keys={rgb_keys or 'all *_rgb'}, "
        f"draw_wrist={renderer.draw_wrist}, line_thickness={renderer.line_thickness}, "
        f"point_radius={renderer.point_radius}"
    )
    return renderer
