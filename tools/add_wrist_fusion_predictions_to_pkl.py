#!/usr/bin/env python3
"""Run wrist inference, fuse it with stored PICO points, and rewrite PKLs in place.

The GPU stage delegates to ``tools/add_wrist_predictions_to_pkl.py`` so it keeps
the repository's tested multi-GPU batching and wrist retargeting behavior.  The
CPU stage parallelizes across episodes while preserving frame order and state
inside every episode.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Retargeting workers are process-parallel.  Keep native math libraries from
# creating another large thread pool inside every worker.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_ROOT = REPO_ROOT / "tools"
TELEOP_ROOT = REPO_ROOT / "teleop"
for _path in (REPO_ROOT, TOOLS_ROOT, TELEOP_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


WRIST_SCRIPT = TOOLS_ROOT / "add_wrist_predictions_to_pkl.py"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "wrist"
    / "outputs"
    / "test_3"
    / "runs"
    / "stage2_test_3"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_LINKER_YAML = REPO_ROOT / "linker_o6" / "linker_o6" / "linker_o6.yml"
DEFAULT_LINKER_URDF_DIR = REPO_ROOT / "linker_o6"
DEFAULT_WUJI_YAML = (
    REPO_ROOT / "wuji_retargeting" / "config" / "adaptive_analytical_pico.yaml"
)

NUMPY2_CORE_PREFIX = "numpy._core"
NUMPY1_CORE_PREFIX = "numpy.core"

FUSION_FIELDS = (
    "fused_pts21_mano",
    "fusionPicoWeights",
    "fusionVisionWeights",
    "fusionMode",
    "fusionDiagnostics",
    "fused_o6_joint_radians",
    "fused_o6_command",
    "fused_wuji_command",
    "fused_o6_error",
    "fused_wuji_error",
    "fused_error",
)

BONES: tuple[tuple[int, int], ...] = (
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


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy-1/NumPy-2 PKLs in either runtime."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == NUMPY2_CORE_PREFIX or module.startswith(
                f"{NUMPY2_CORE_PREFIX}."
            ):
                compat_module = NUMPY1_CORE_PREFIX + module[len(NUMPY2_CORE_PREFIX) :]
                return super().find_class(compat_module, name)
            raise


def install_numpy_pickle_compat() -> list[str]:
    try:
        core = importlib.import_module("numpy._core")
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            core = importlib.import_module("numpy.core")
    aliases = {
        "numpy._core": core,
        "numpy._core.multiarray": getattr(core, "multiarray", None),
        "numpy._core.numeric": getattr(core, "numeric", None),
    }
    added = []
    for name, module in aliases.items():
        if module is not None and name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    return added


def read_pkl(path: Path) -> dict[str, Any]:
    added = install_numpy_pickle_compat()
    try:
        with path.open("rb") as file_obj:
            data = NumpyCompatUnpickler(file_obj).load()
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict with list field 'messages': {path}")
    return data


def atomic_write_pkl(data: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as file_obj:
            pickle.dump(data, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def iter_pkl_paths(data_root: Path, limit_pkls: int | None) -> list[Path]:
    paths = sorted(path for path in data_root.glob("**/*.pkl") if path.is_file())
    if limit_pkls is not None:
        paths = paths[:limit_pkls]
    return paths


def _as_local_points(points: Any) -> np.ndarray | None:
    if points is None:
        return None
    try:
        arr = np.asarray(points, dtype=np.float32)
    except Exception:
        return None
    if arr.shape != (21, 3) or not np.all(np.isfinite(arr)):
        return None
    arr = arr - arr[0:1]
    span = float(np.max(np.linalg.norm(arr, axis=1)))
    if span < 0.04 or span > 0.30:
        return None
    return arr.astype(np.float32, copy=False)


def _bone_lengths(points: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(points[b] - points[a]) for a, b in BONES],
        dtype=np.float32,
    )


def _joint_bone_scores(
    points: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    if reference is None:
        return np.ones(21, dtype=np.float32)
    lengths = _bone_lengths(points)
    relative_error = np.abs(lengths - reference) / np.maximum(reference, 0.004)
    bone_scores = np.exp(-0.5 * np.square(relative_error / 0.22)).astype(np.float32)
    totals = np.ones(21, dtype=np.float32) * 0.25
    counts = np.ones(21, dtype=np.float32) * 0.25
    for score, (a, b) in zip(bone_scores, BONES):
        totals[a] += score
        totals[b] += score
        counts[a] += 1.0
        counts[b] += 1.0
    return np.clip(totals / counts, 0.02, 1.0)


@dataclass
class FusionOutput:
    points: np.ndarray
    pico_weights: np.ndarray
    vision_weights: np.ndarray
    mode: str
    diagnostics: dict[str, Any]


class AdaptiveKalmanHandFusion:
    """Same adaptive constant-velocity fusion used by collect_data_fused.py."""

    def __init__(
        self,
        *,
        pico_std_mm: float = 5.0,
        vision_std_mm: float = 5.0,
        innovation_gate_mm: float = 22.0,
        process_accel_mps2: float = 2.5,
        min_quality: float = 0.03,
        bone_reference_alpha: float = 0.0,
    ) -> None:
        self.pico_std_m = max(float(pico_std_mm), 0.5) / 1000.0
        self.vision_std_m = max(float(vision_std_mm), 0.5) / 1000.0
        self.innovation_gate_m = max(float(innovation_gate_mm), 2.0) / 1000.0
        self.process_accel_mps2 = max(float(process_accel_mps2), 0.1)
        self.min_quality = float(np.clip(min_quality, 1e-4, 0.25))
        self.bone_reference_alpha = float(
            np.clip(bone_reference_alpha, 0.0, 0.05)
        )
        self.reset()

    def reset(self) -> None:
        self.position: np.ndarray | None = None
        self.velocity = np.zeros((21, 3), dtype=np.float32)
        self.p00 = np.ones(21, dtype=np.float32) * 1e-4
        self.p01 = np.zeros(21, dtype=np.float32)
        self.p11 = np.ones(21, dtype=np.float32) * 0.25
        self.last_timestamp_ns: int | None = None
        self.bone_reference: np.ndarray | None = None

    def _predict(self, dt: float) -> np.ndarray:
        assert self.position is not None
        predicted = self.position + self.velocity * dt
        q = self.process_accel_mps2**2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        p00 = self.p00 + 2.0 * dt * self.p01 + dt2 * self.p11 + q * dt4 * 0.25
        p01 = self.p01 + dt * self.p11 + q * dt3 * 0.5
        p11 = self.p11 + q * dt2
        self.p00, self.p01, self.p11 = p00, p01, p11
        return predicted

    def _quality(
        self,
        points: np.ndarray,
        predicted: np.ndarray,
        dt: float,
        source_quality: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        innovation = np.linalg.norm(points - predicted, axis=1)
        speed_allowance = np.linalg.norm(self.velocity, axis=1) * dt * 1.5
        gate = self.innovation_gate_m + speed_allowance
        innovation_score = np.exp(
            -0.5 * np.square(innovation / np.maximum(gate, 1e-4))
        )
        bone_score = _joint_bone_scores(points, self.bone_reference)
        score = float(np.clip(source_quality, 0.0, 1.0)) * innovation_score * bone_score
        return (
            np.clip(score, self.min_quality, 1.0).astype(np.float32),
            innovation,
        )

    def _project_bones(self, points: np.ndarray) -> np.ndarray:
        if self.bone_reference is None:
            return points
        projected = np.asarray(points, dtype=np.float32).copy()
        projected[0] = 0.0
        for length, (a, b) in zip(self.bone_reference, BONES):
            direction = projected[b] - projected[a]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6 and self.position is not None:
                direction = self.position[b] - self.position[a]
                norm = float(np.linalg.norm(direction))
            if norm >= 1e-6:
                projected[b] = projected[a] + direction * (float(length) / norm)
        return projected

    def update(
        self,
        *,
        pico_points: np.ndarray | None,
        vision_points: np.ndarray | None,
        timestamp_ns: int,
        pico_quality: float = 1.0,
        vision_quality: float = 1.0,
    ) -> FusionOutput:
        pico = _as_local_points(pico_points)
        vision = _as_local_points(vision_points)
        if pico is None and vision is None:
            raise ValueError("fusion requires at least one valid keypoint source")

        if self.last_timestamp_ns is None:
            dt = 1.0 / 30.0
        else:
            dt = float(
                np.clip(
                    (int(timestamp_ns) - self.last_timestamp_ns) / 1e9,
                    1.0 / 120.0,
                    0.10,
                )
            )
        self.last_timestamp_ns = int(timestamp_ns)

        if self.position is None:
            if pico is not None and vision is not None:
                pico_precision = max(float(pico_quality), self.min_quality) / self.pico_std_m**2
                vision_precision = max(float(vision_quality), self.min_quality) / self.vision_std_m**2
                wp = pico_precision / (pico_precision + vision_precision)
                initialized = wp * pico + (1.0 - wp) * vision
                pico_weights = np.full(21, wp, dtype=np.float32)
                vision_weights = 1.0 - pico_weights
                mode = "fused_init"
                self.bone_reference = (
                    wp * _bone_lengths(pico)
                    + (1.0 - wp) * _bone_lengths(vision)
                ).astype(np.float32)
            elif pico is not None:
                initialized = pico
                pico_weights = np.ones(21, dtype=np.float32)
                vision_weights = np.zeros(21, dtype=np.float32)
                mode = "pico_only_init"
                self.bone_reference = _bone_lengths(pico)
            else:
                assert vision is not None
                initialized = vision
                pico_weights = np.zeros(21, dtype=np.float32)
                vision_weights = np.ones(21, dtype=np.float32)
                mode = "vision_only_init"
                self.bone_reference = _bone_lengths(vision)
            self.position = self._project_bones(initialized)
            self.position[0] = 0.0
            return FusionOutput(
                points=self.position.copy(),
                pico_weights=pico_weights,
                vision_weights=vision_weights,
                mode=mode,
                diagnostics={"dtMs": dt * 1000.0, "initialized": True},
            )

        previous = self.position.copy()
        predicted = self._predict(dt)
        pico_score = np.zeros(21, dtype=np.float32)
        vision_score = np.zeros(21, dtype=np.float32)
        pico_innovation = np.full(21, np.nan, dtype=np.float32)
        vision_innovation = np.full(21, np.nan, dtype=np.float32)

        if pico is not None:
            pico_score, pico_innovation = self._quality(
                pico, predicted, dt, pico_quality
            )
        if vision is not None:
            vision_score, vision_innovation = self._quality(
                vision, predicted, dt, vision_quality
            )

        pico_precision = (
            pico_score / self.pico_std_m**2
            if pico is not None
            else np.zeros(21, dtype=np.float32)
        )
        vision_precision = (
            vision_score / self.vision_std_m**2
            if vision is not None
            else np.zeros(21, dtype=np.float32)
        )

        disagreement = None
        if pico is not None and vision is not None:
            disagreement = np.linalg.norm(pico - vision, axis=1)
            median_disagreement = float(np.median(disagreement[1:]))
            if median_disagreement > 0.060:
                pico_bone = float(
                    np.mean(_joint_bone_scores(pico, self.bone_reference))
                )
                vision_bone = float(
                    np.mean(_joint_bone_scores(vision, self.bone_reference))
                )
                if vision_bone + 0.08 < pico_bone:
                    vision_precision *= 0.10
                else:
                    pico_precision *= 0.35

        total_precision = pico_precision + vision_precision
        valid_precision = np.maximum(total_precision, 1e-9)
        pico_weights = pico_precision / valid_precision
        vision_weights = vision_precision / valid_precision

        if pico is not None and vision is not None:
            measurement = (
                pico_weights[:, None] * pico + vision_weights[:, None] * vision
            )
            mode = "fused"
        elif pico is not None:
            measurement = pico
            pico_weights[:] = 1.0
            vision_weights[:] = 0.0
            mode = "pico_only"
        else:
            assert vision is not None
            measurement = vision
            pico_weights[:] = 0.0
            vision_weights[:] = 1.0
            mode = "vision_only"

        measurement_variance = 1.0 / valid_precision
        innovation = measurement - predicted
        denominator = self.p00 + measurement_variance
        k0 = self.p00 / np.maximum(denominator, 1e-12)
        k1 = self.p01 / np.maximum(denominator, 1e-12)
        filtered = predicted + k0[:, None] * innovation
        self.velocity = self.velocity + k1[:, None] * innovation
        old_p00 = self.p00.copy()
        old_p01 = self.p01.copy()
        old_p11 = self.p11.copy()
        self.p00 = (1.0 - k0) * old_p00
        self.p01 = (1.0 - k0) * old_p01
        self.p11 = np.maximum(old_p11 - k1 * old_p01, 1e-8)

        if self.bone_reference is not None:
            candidate_lengths = _bone_lengths(measurement)
            safe = np.logical_and(
                candidate_lengths > 0.004,
                candidate_lengths < 0.12,
            )
            alpha = self.bone_reference_alpha
            self.bone_reference[safe] = (
                (1.0 - alpha) * self.bone_reference[safe]
                + alpha * candidate_lengths[safe]
            )

        projected = self._project_bones(filtered)
        projected[0] = 0.0
        observed_velocity = (projected - previous) / max(dt, 1e-4)
        self.velocity = 0.75 * self.velocity + 0.25 * observed_velocity
        self.position = projected.astype(np.float32)

        diagnostics: dict[str, Any] = {
            "dtMs": dt * 1000.0,
            "initialized": False,
            "picoWeightMean": float(np.mean(pico_weights[1:])),
            "visionWeightMean": float(np.mean(vision_weights[1:])),
            "picoInnovationMmMean": (
                None
                if pico is None
                else float(np.nanmean(pico_innovation[1:]) * 1000.0)
            ),
            "visionInnovationMmMean": (
                None
                if vision is None
                else float(np.nanmean(vision_innovation[1:]) * 1000.0)
            ),
            "crossSensorMmMean": (
                None
                if disagreement is None
                else float(np.mean(disagreement[1:]) * 1000.0)
            ),
        }
        return FusionOutput(
            points=self.position.copy(),
            pico_weights=pico_weights.astype(np.float32),
            vision_weights=vision_weights.astype(np.float32),
            mode=mode,
            diagnostics=diagnostics,
        )


def _timestamp_ns(msg: dict[str, Any], frame_idx: int) -> int:
    for field in ("sampleClockNs", "mainClockMonotonicNs", "rgbCaptureNs"):
        value = msg.get(field)
        if isinstance(value, (int, np.integer)):
            return int(value)
    timestamp = msg.get("timestamp")
    if isinstance(timestamp, (int, float, np.integer, np.floating)):
        return int(float(timestamp) * 1e9)
    return int(round(frame_idx * 1e9 / 30.0))


def _quality_values(msg: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    pico_quality = 1.0
    source_age_ns = msg.get("sourceAgeNs")
    pico_source_age_ms = None
    if isinstance(source_age_ns, (int, float, np.integer, np.floating)):
        pico_source_age_ms = float(source_age_ns) / 1e6
        pico_quality *= float(
            np.exp(-0.5 * (max(pico_source_age_ms, 0.0) / 35.0) ** 2)
        )

    vision_quality = 1.0
    rgb_repeated = bool(msg.get("rgbFrameRepeated", False))
    if rgb_repeated:
        vision_quality *= 0.25
    rgb_pico_delta_ms = None
    delta_ns = msg.get("rgbToPicoReceiveDeltaNs")
    if isinstance(delta_ns, (int, float, np.integer, np.floating)):
        rgb_pico_delta_ms = float(delta_ns) / 1e6
        vision_quality *= float(
            np.exp(-0.5 * (abs(rgb_pico_delta_ms) / 25.0) ** 2)
        )
    return (
        pico_quality,
        vision_quality,
        {
            "picoSourceAgeMs": pico_source_age_ms,
            "rgbPicoDeltaMs": rgb_pico_delta_ms,
            "rgbFrameRepeated": rgb_repeated,
            "picoQuality": pico_quality,
            "visionQuality": vision_quality,
        },
    )


def _clear_fusion_fields(msg: dict[str, Any]) -> None:
    for field in FUSION_FIELDS:
        msg[field] = None


def _error_text(errors: list[str]) -> str | None:
    return None if not errors else "; ".join(errors)


_WORKER_OPTIONS: dict[str, Any] | None = None
_LINKER: Any = None
_WUJI: Any = None
_ANGLES_RAD_TO_CMD: Any = None


def _init_fusion_worker(options: dict[str, Any]) -> None:
    global _WORKER_OPTIONS, _LINKER, _WUJI, _ANGLES_RAD_TO_CMD
    from episode_io import angles_rad_to_cmd
    from retargeter import LinkerO6Retargeter
    from wuji_retargeting import Retargeter as WujiRetargeter

    _WORKER_OPTIONS = options
    _ANGLES_RAD_TO_CMD = angles_rad_to_cmd
    _LINKER = LinkerO6Retargeter(
        yaml_path=str(options["linker_yaml"]),
        urdf_dir=str(options["linker_urdf_dir"]),
        hand=str(options["hand"]),
    )
    _WUJI = WujiRetargeter.from_yaml(
        str(options["wuji_yaml"]),
        hand_side=str(options["hand"]),
    )
    optimizer = getattr(_WUJI, "optimizer", None)
    opt = getattr(optimizer, "opt", None)
    if opt is not None and hasattr(opt, "set_maxeval"):
        opt.set_maxeval(int(options["wuji_maxeval"]))


def _reset_wuji(retargeter: Any) -> None:
    if hasattr(retargeter, "reset"):
        retargeter.reset()
        return
    optimizer = getattr(retargeter, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "last_qpos"):
        optimizer.last_qpos = None
    lp_filter = getattr(retargeter, "lp_filter", None)
    if lp_filter is not None:
        if hasattr(lp_filter, "reset"):
            lp_filter.reset()
        elif hasattr(lp_filter, "is_init"):
            lp_filter.is_init = False


def _retarget_wuji_mano(retargeter: Any, pts21_mano: np.ndarray) -> np.ndarray:
    points = np.asarray(pts21_mano, dtype=np.float64).reshape(21, 3).copy()
    rotation_xyz = getattr(retargeter, "rotation_xyz", {}) or {}
    if any(float(rotation_xyz.get(axis, 0.0)) != 0.0 for axis in ("x", "y", "z")):
        points = retargeter._apply_rotation(points)
    if getattr(retargeter, "_has_offset", False):
        points = retargeter._apply_offset(points)
    qpos = np.asarray(retargeter.optimizer.solve(points), dtype=np.float32).reshape(-1)
    qpos = np.asarray(retargeter.lp_filter.next(qpos), dtype=np.float32).reshape(-1)
    if qpos.shape != (20,) or not np.all(np.isfinite(qpos)):
        raise RuntimeError(f"Wuji retargeter returned invalid data: {qpos.shape}")
    return qpos.reshape(5, 4).astype(np.float32, copy=False)


def _process_fusion_episode(pkl_str: str) -> dict[str, Any]:
    if _WORKER_OPTIONS is None or _LINKER is None or _WUJI is None:
        raise RuntimeError("fusion worker was not initialized")
    pkl_path = Path(pkl_str)
    options = _WORKER_OPTIONS
    stats: Counter[str] = Counter()
    started = time.perf_counter()
    try:
        data = read_pkl(pkl_path)
        messages = data["messages"]
        max_count = len(messages)
        if options["limit_frames"] is not None:
            max_count = min(max_count, int(options["limit_frames"]))

        _LINKER.reset()
        _reset_wuji(_WUJI)
        fusion = AdaptiveKalmanHandFusion(
            pico_std_mm=float(options["pico_std_mm"]),
            vision_std_mm=float(options["vision_std_mm"]),
            innovation_gate_mm=float(options["innovation_gate_mm"]),
            process_accel_mps2=float(options["process_accel_mps2"]),
        )

        for frame_idx in range(max_count):
            msg = messages[frame_idx]
            if not isinstance(msg, dict):
                stats["message_not_dict"] += 1
                continue
            _clear_fusion_fields(msg)
            pico_points = _as_local_points(msg.get("pts21_mano"))
            vision_points = _as_local_points(msg.get("wrist_pts21_mano"))
            if pico_points is None:
                stats["invalid_pico_pts21"] += 1
            if vision_points is None:
                stats["invalid_wrist_pts21"] += 1
            if pico_points is None and vision_points is None:
                wrist_error = msg.get("wrist_infer_error")
                error = "no_valid_pico_or_wrist_points"
                if wrist_error:
                    error += f"; wrist={wrist_error}"
                msg["fusionMode"] = "invalid"
                msg["fusionDiagnostics"] = {
                    "error": error,
                    "picoSourceField": "pts21_mano",
                    "visionSourceField": "wrist_pts21_mano",
                }
                msg["fused_error"] = error
                stats["no_valid_sources"] += 1
                continue

            pico_quality, vision_quality, quality_diagnostics = _quality_values(msg)
            try:
                fused = fusion.update(
                    pico_points=pico_points,
                    vision_points=vision_points,
                    timestamp_ns=_timestamp_ns(msg, frame_idx),
                    pico_quality=pico_quality,
                    vision_quality=vision_quality,
                )
            except Exception as exc:
                error = f"fusion_error: {type(exc).__name__}: {exc}"
                msg["fusionMode"] = "error"
                msg["fusionDiagnostics"] = {"error": error, **quality_diagnostics}
                msg["fused_error"] = error
                stats["fusion_error"] += 1
                continue

            fused_points = np.asarray(fused.points, dtype=np.float32).reshape(21, 3)
            diagnostics = dict(fused.diagnostics)
            diagnostics.update(quality_diagnostics)
            diagnostics.update(
                {
                    "picoSourceField": "pts21_mano",
                    "visionSourceField": "wrist_pts21_mano",
                    "wristInferenceError": msg.get("wrist_infer_error"),
                }
            )
            msg["fused_pts21_mano"] = fused_points
            msg["fusionPicoWeights"] = np.asarray(
                fused.pico_weights, dtype=np.float32
            ).reshape(21)
            msg["fusionVisionWeights"] = np.asarray(
                fused.vision_weights, dtype=np.float32
            ).reshape(21)
            msg["fusionMode"] = fused.mode
            msg["fusionDiagnostics"] = diagnostics

            errors = []
            o6_started = time.perf_counter()
            try:
                o6_radians = np.asarray(
                    _LINKER.retarget(fused_points), dtype=np.float32
                ).reshape(6)
                o6_command = np.asarray(
                    _ANGLES_RAD_TO_CMD(o6_radians), dtype=np.uint8
                ).reshape(6)
                msg["fused_o6_joint_radians"] = o6_radians
                msg["fused_o6_command"] = o6_command
                msg["fused_o6_error"] = None
                stats["fused_o6_success"] += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                msg["fused_o6_error"] = error
                errors.append(f"o6={error}")
                stats["fused_o6_error"] += 1
            diagnostics["linkerRetargetMs"] = (
                time.perf_counter() - o6_started
            ) * 1000.0

            wuji_started = time.perf_counter()
            try:
                msg["fused_wuji_command"] = _retarget_wuji_mano(
                    _WUJI, fused_points
                )
                msg["fused_wuji_error"] = None
                stats["fused_wuji_success"] += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                msg["fused_wuji_error"] = error
                errors.append(f"wuji={error}")
                stats["fused_wuji_error"] += 1
            diagnostics["wujiRetargetMs"] = (
                time.perf_counter() - wuji_started
            ) * 1000.0
            diagnostics["wujiMaxeval"] = int(options["wuji_maxeval"])
            msg["fused_error"] = _error_text(errors)
            stats["fused_points_success"] += 1
            stats[f"fusion_mode_{fused.mode}"] += 1

        stats["frames_seen"] += max_count
        metadata = data.setdefault("metadata", {})
        metadata["wrist_fusion_prediction"] = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "script": "tools/add_wrist_fusion_predictions_to_pkl.py",
            "source_pkl": str(pkl_path),
            "checkpoint": str(options["checkpoint"]),
            "pico_source_field": "pts21_mano",
            "vision_source_field": "wrist_pts21_mano",
            "quality_fields": [
                "sourceAgeNs",
                "rgbFrameRepeated",
                "rgbToPicoReceiveDeltaNs",
            ],
            "parameters": {
                "pico_std_mm": float(options["pico_std_mm"]),
                "vision_std_mm": float(options["vision_std_mm"]),
                "innovation_gate_mm": float(options["innovation_gate_mm"]),
                "process_accel_mps2": float(options["process_accel_mps2"]),
                "wuji_maxeval": int(options["wuji_maxeval"]),
            },
            "fields": list(FUSION_FIELDS),
            "overwrite_generated_fields": True,
            "preserved_source_fields": [
                "pts21_mano",
                "o6_command",
                "wuji_command",
            ],
            "limit_frames": options["limit_frames"],
            "stats": dict(stats),
        }
        stats["files_written"] += 1
        metadata["wrist_fusion_prediction"]["stats"] = dict(stats)
        atomic_write_pkl(data, pkl_path)
        return {
            "ok": True,
            "pkl": str(pkl_path),
            "elapsed_s": time.perf_counter() - started,
            "stats": dict(stats),
        }
    except Exception as exc:
        return {
            "ok": False,
            "pkl": str(pkl_path),
            "elapsed_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "stats": dict(stats),
        }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Infer wrist 21-point poses on every RGB frame, create wrist commands, "
            "fuse wrist points with stored pts21_mano, and atomically rewrite PKLs."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-type", choices=("auto", "stage1", "stage2"), default="stage2")
    parser.add_argument("--projection-head-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--write-wrist-projection",
        action="store_true",
        help="Also write wrist_uv21_rgb / wrist_anchor_uv_rgb during wrist inference.",
    )
    parser.add_argument("--image-field", default="rgbImage")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=8)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--fusion-workers", type=_nonnegative_int, default=0, help="0 chooses min(32, CPU/4, PKL count).")
    parser.add_argument("--log-every", type=_nonnegative_int, default=5)
    parser.add_argument("--limit-pkls", type=_positive_int, default=None)
    parser.add_argument("--limit-frames", type=_positive_int, default=None)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--dino-dir", type=Path, default=None)
    parser.add_argument("--mano-model-dir", type=Path, default=None)
    parser.add_argument("--linker-yaml", type=Path, default=DEFAULT_LINKER_YAML)
    parser.add_argument("--linker-urdf-dir", type=Path, default=DEFAULT_LINKER_URDF_DIR)
    parser.add_argument("--wuji-yaml", type=Path, default=DEFAULT_WUJI_YAML)
    parser.add_argument("--pico-std-mm", type=float, default=5.0)
    parser.add_argument("--vision-std-mm", type=float, default=5.0)
    parser.add_argument("--innovation-gate-mm", type=float, default=22.0)
    parser.add_argument("--process-accel-mps2", type=float, default=2.5)
    parser.add_argument("--wuji-maxeval", type=_positive_int, default=40)
    parser.add_argument("--skip-wrist-inference", action="store_true", help="Fuse already existing wrist_* fields without running the model.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running inference or rewriting PKLs.")
    return parser


def _resolved(path: Path | None) -> str | None:
    return None if path is None else str(path.expanduser().resolve())


def _validate_args(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    required = {
        "wrist inference script": WRIST_SCRIPT,
        "checkpoint": args.checkpoint.expanduser().resolve(),
        "linker yaml": args.linker_yaml.expanduser().resolve(),
        "linker urdf dir": args.linker_urdf_dir.expanduser().resolve(),
        "wuji yaml": args.wuji_yaml.expanduser().resolve(),
    }
    if args.projection_head_checkpoint is not None or bool(args.write_wrist_projection):
        if args.projection_head_checkpoint is None:
            raise ValueError("--write-wrist-projection requires --projection-head-checkpoint")
        required["projection head checkpoint"] = args.projection_head_checkpoint.expanduser().resolve()
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    pkl_paths = iter_pkl_paths(data_root, args.limit_pkls)
    if not pkl_paths:
        raise FileNotFoundError(f"no PKL files found under: {data_root}")
    return data_root, pkl_paths


def _wrist_command(args: argparse.Namespace, data_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(WRIST_SCRIPT),
        "--data-root",
        str(data_root),
        "--checkpoint",
        _resolved(args.checkpoint),
        "--model-type",
        str(args.model_type),
        "--image-field",
        str(args.image_field),
        "--devices",
        str(args.devices),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--precision",
        str(args.precision),
        "--pipeline-mode",
        "chunk",
        "--log-every",
        str(args.log_every),
        "--overwrite-wrist",
        "--hand",
        str(args.hand),
        "--linker-yaml",
        _resolved(args.linker_yaml),
        "--linker-urdf-dir",
        _resolved(args.linker_urdf_dir),
        "--wuji-yaml",
        _resolved(args.wuji_yaml),
    ]
    if args.projection_head_checkpoint is not None or bool(args.write_wrist_projection):
        command.append("--write-wrist-projection")
    if args.projection_head_checkpoint is not None:
        command.extend(("--projection-head-checkpoint", _resolved(args.projection_head_checkpoint)))
    optional_paths = (
        ("--config", args.config),
        ("--dino-dir", args.dino_dir),
        ("--mano-model-dir", args.mano_model_dir),
    )
    for flag, value in optional_paths:
        if value is not None:
            command.extend((flag, _resolved(value)))
    if args.limit_pkls is not None:
        command.extend(("--limit-pkls", str(args.limit_pkls)))
    if args.limit_frames is not None:
        command.extend(("--limit-frames", str(args.limit_frames)))
    return command


def _fusion_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint": _resolved(args.checkpoint),
        "projection_head_checkpoint": _resolved(args.projection_head_checkpoint),
        "write_wrist_projection": bool(args.write_wrist_projection or args.projection_head_checkpoint is not None),
        "linker_yaml": _resolved(args.linker_yaml),
        "linker_urdf_dir": _resolved(args.linker_urdf_dir),
        "wuji_yaml": _resolved(args.wuji_yaml),
        "hand": str(args.hand),
        "limit_frames": args.limit_frames,
        "pico_std_mm": float(args.pico_std_mm),
        "vision_std_mm": float(args.vision_std_mm),
        "innovation_gate_mm": float(args.innovation_gate_mm),
        "process_accel_mps2": float(args.process_accel_mps2),
        "wuji_maxeval": int(args.wuji_maxeval),
    }


def _choose_fusion_workers(requested: int, pkl_count: int) -> int:
    if requested > 0:
        return min(requested, pkl_count)
    cpu_count = os.cpu_count() or 1
    return max(1, min(32, max(1, cpu_count // 4), pkl_count))


def _run_fusion(
    pkl_paths: list[Path],
    options: dict[str, Any],
    workers: int,
    log_every: int,
) -> tuple[Counter[str], list[dict[str, Any]], float]:
    aggregate: Counter[str] = Counter()
    failures = []
    started = time.perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_fusion_worker,
        initargs=(options,),
    ) as executor:
        future_to_path = {
            executor.submit(_process_fusion_episode, str(path)): path
            for path in pkl_paths
        }
        completed = 0
        for future in as_completed(future_to_path):
            completed += 1
            path = future_to_path[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ok": False,
                    "pkl": str(path),
                    "error": f"worker_crash: {type(exc).__name__}: {exc}",
                    "stats": {},
                }
            aggregate.update(result.get("stats", {}))
            if not result.get("ok"):
                failures.append(result)
                print(f"[fusion ERROR] {path}: {result.get('error')}", flush=True)
            if log_every > 0 and (
                completed == 1
                or completed == len(pkl_paths)
                or completed % log_every == 0
            ):
                print(
                    f"[fusion] {completed}/{len(pkl_paths)} "
                    f"{path.name} ok={bool(result.get('ok'))} "
                    f"elapsed={float(result.get('elapsed_s', 0.0)):.2f}s",
                    flush=True,
                )
    return aggregate, failures, time.perf_counter() - started


def main() -> int:
    args = build_parser().parse_args()
    data_root, pkl_paths = _validate_args(args)
    fusion_workers = _choose_fusion_workers(args.fusion_workers, len(pkl_paths))
    wrist_command = _wrist_command(args, data_root)
    plan = {
        "data_root": str(data_root),
        "pkl_files": len(pkl_paths),
        "checkpoint": _resolved(args.checkpoint),
        "projection_head_checkpoint": _resolved(args.projection_head_checkpoint),
        "write_wrist_projection": bool(args.write_wrist_projection or args.projection_head_checkpoint is not None),
        "wrist_inference": not bool(args.skip_wrist_inference),
        "devices": str(args.devices),
        "batch_size_per_gpu": int(args.batch_size),
        "dataloader_workers_per_gpu": int(args.num_workers),
        "fusion_workers": fusion_workers,
        "fusion_parameters": {
            "pico_std_mm": float(args.pico_std_mm),
            "vision_std_mm": float(args.vision_std_mm),
            "innovation_gate_mm": float(args.innovation_gate_mm),
            "process_accel_mps2": float(args.process_accel_mps2),
            "wuji_maxeval": int(args.wuji_maxeval),
        },
        "preserved_fields": ["pts21_mano", "o6_command", "wuji_command"],
        "atomic_in_place_rewrite": True,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("wrist command:", subprocess.list2cmdline(wrist_command))
        return 0

    total_started = time.perf_counter()
    wrist_elapsed = 0.0
    if not args.skip_wrist_inference:
        print("[stage 1/2] multi-GPU wrist inference and wrist retargeting", flush=True)
        wrist_started = time.perf_counter()
        inference_env = os.environ.copy()
        inference_env.setdefault(
            "PYTHONPYCACHEPREFIX",
            f"/tmp/dex_wrist_pycache_{os.getuid()}",
        )
        inference_env.setdefault("TOKENIZERS_PARALLELISM", "false")
        subprocess.run(
            wrist_command,
            cwd=str(REPO_ROOT),
            env=inference_env,
            check=True,
        )
        wrist_elapsed = time.perf_counter() - wrist_started
    else:
        print("[stage 1/2] skipped; reusing existing wrist_* fields", flush=True)

    print("[stage 2/2] ordered per-episode fusion and fused retargeting", flush=True)
    aggregate, failures, fusion_elapsed = _run_fusion(
        pkl_paths=pkl_paths,
        options=_fusion_options(args),
        workers=fusion_workers,
        log_every=int(args.log_every),
    )
    summary = {
        "ok": not failures,
        "pkl_files": len(pkl_paths),
        "fusion_failures": len(failures),
        "wrist_elapsed_s": wrist_elapsed,
        "fusion_elapsed_s": fusion_elapsed,
        "total_elapsed_s": time.perf_counter() - total_started,
        "fusion_stats": dict(aggregate),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
