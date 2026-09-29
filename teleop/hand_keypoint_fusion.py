#!/usr/bin/env python3
"""Adaptive PICO + wrist-RGB hand keypoint fusion and fused retargeting."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from retargeter import (
    LinkerO6Retargeter,
    WujiHandRetargeter,
    pico_raw_to_mano,
    pico_wrist_pose6d,
)
from wrist_mvs_keypoint_viewer import (
    choose_device,
    choose_precision,
    load_wrist_model,
    predict_pts21_mano,
)


BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _as_local_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float32)
    if arr.shape != (21, 3) or not np.all(np.isfinite(arr)):
        return None
    arr = arr - arr[0:1]
    span = float(np.max(np.linalg.norm(arr, axis=1)))
    if span < 0.04 or span > 0.30:
        return None
    return arr


def _bone_lengths(points: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(points[b] - points[a]) for a, b in BONES],
        dtype=np.float32,
    )


def _joint_bone_scores(points: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
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
    """Constant-velocity Kalman-style filter with adaptive per-joint precision."""

    def __init__(
        self,
        *,
        pico_std_mm: float = 5.0,
        vision_std_mm: float = 14.0,
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
        self.bone_reference_alpha = float(np.clip(bone_reference_alpha, 0.0, 0.05))
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
        q = self.process_accel_mps2 ** 2
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
        innovation_score = np.exp(-0.5 * np.square(innovation / np.maximum(gate, 1e-4)))
        bone_score = _joint_bone_scores(points, self.bone_reference)
        score = float(np.clip(source_quality, 0.0, 1.0)) * innovation_score * bone_score
        return np.clip(score, self.min_quality, 1.0).astype(np.float32), innovation

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
            if norm < 1e-6:
                continue
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
            dt = float(np.clip((int(timestamp_ns) - self.last_timestamp_ns) / 1e9, 1.0 / 120.0, 0.10))
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
                    wp * _bone_lengths(pico) + (1.0 - wp) * _bone_lengths(vision)
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
            pico_score, pico_innovation = self._quality(pico, predicted, dt, pico_quality)
        if vision is not None:
            vision_score, vision_innovation = self._quality(vision, predicted, dt, vision_quality)

        pico_precision = pico_score / self.pico_std_m**2 if pico is not None else np.zeros(21, dtype=np.float32)
        vision_precision = vision_score / self.vision_std_m**2 if vision is not None else np.zeros(21, dtype=np.float32)

        disagreement = None
        if pico is not None and vision is not None:
            disagreement = np.linalg.norm(pico - vision, axis=1)
            median_disagreement = float(np.median(disagreement[1:]))
            if median_disagreement > 0.060:
                pico_bone = float(np.mean(_joint_bone_scores(pico, self.bone_reference)))
                vision_bone = float(np.mean(_joint_bone_scores(vision, self.bone_reference)))
                if vision_bone + 0.08 < pico_bone:
                    vision_precision *= 0.10
                else:
                    pico_precision *= 0.35

        total_precision = pico_precision + vision_precision
        valid_precision = np.maximum(total_precision, 1e-9)
        pico_weights = pico_precision / valid_precision
        vision_weights = vision_precision / valid_precision

        if pico is not None and vision is not None:
            measurement = pico_weights[:, None] * pico + vision_weights[:, None] * vision
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
            safe = np.logical_and(candidate_lengths > 0.004, candidate_lengths < 0.12)
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
                None if pico is None else float(np.nanmean(pico_innovation[1:]) * 1000.0)
            ),
            "visionInnovationMmMean": (
                None if vision is None else float(np.nanmean(vision_innovation[1:]) * 1000.0)
            ),
            "crossSensorMmMean": (
                None if disagreement is None else float(np.mean(disagreement[1:]) * 1000.0)
            ),
        }
        return FusionOutput(
            points=self.position.copy(),
            pico_weights=pico_weights.astype(np.float32),
            vision_weights=vision_weights.astype(np.float32),
            mode=mode,
            diagnostics=diagnostics,
        )


@dataclass
class FusedRetargetingResult:
    pts21_mano: np.ndarray
    linker_joint_radians: np.ndarray
    wuji_joint_radians: np.ndarray
    wrist_pose_6d: np.ndarray
    pico_pts21_mano: np.ndarray
    wrist_pts21_mano: np.ndarray | None
    fusion_pico_weights: np.ndarray
    fusion_vision_weights: np.ndarray
    fusion_mode: str
    fusion_diagnostics: dict[str, Any]
    wrist_inference_ms: float | None
    wrist_inference_error: str | None
    wrist_inference_reused: bool = False
    wrist_inference_age_ms: float | None = None

    def message_fields(self) -> dict[str, Any]:
        return {
            "pico_pts21_mano": self.pico_pts21_mano.astype(np.float32),
            "wrist_pts21_mano": (
                None
                if self.wrist_pts21_mano is None
                else self.wrist_pts21_mano.astype(np.float32)
            ),
            "fused_pts21_mano": self.pts21_mano.astype(np.float32),
            "fusionPicoWeights": self.fusion_pico_weights.astype(np.float32),
            "fusionVisionWeights": self.fusion_vision_weights.astype(np.float32),
            "fusionMode": self.fusion_mode,
            "fusionDiagnostics": dict(self.fusion_diagnostics),
            "wristInferenceMs": self.wrist_inference_ms,
            "wristInferenceError": self.wrist_inference_error,
            "wristInferenceReused": bool(self.wrist_inference_reused),
            "wristInferenceAgeMs": self.wrist_inference_age_ms,
        }


@dataclass
class VisionInferenceResult:
    points: np.ndarray | None
    inference_ms: float | None
    error: str | None
    reused: bool = False
    age_ms: float | None = None


@dataclass
class AsyncFusedCompletion:
    sequence_id: int
    context: Any
    result: FusedRetargetingResult | None
    error: str | None


@dataclass
class _AsyncFusedRequest:
    sequence_id: int
    raw26x7: np.ndarray
    rgb: np.ndarray | None
    timestamp_ns: int
    rgb_repeated: bool
    rgb_pico_delta_ms: float | None
    pico_source_age_ms: float | None
    context: Any
    submitted_perf_ns: int


@dataclass
class _AsyncVisionResult:
    request: _AsyncFusedRequest
    vision: VisionInferenceResult
    inference_started_perf_ns: int
    inference_finished_perf_ns: int


class FusedPicoWristRetargeter:
    """PICO global wrist pose + adaptive local-keypoint fusion + fused commands."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        mano_model_dir: Path,
        model_type: str = "auto",
        wrist_config: Path | None = None,
        dino_dir: Path | None = None,
        device: str = "auto",
        precision: str = "auto",
        yaml_path: str | None = None,
        wuji_yaml_path: str | None = None,
        hand: str = "right",
        warmup_iters: int = 20,
        pico_std_mm: float = 5.0,
        vision_std_mm: float = 14.0,
        innovation_gate_mm: float = 22.0,
        process_accel_mps2: float = 2.5,
        wuji_maxeval: int = 40,
        torch_threads: int = 2,
    ) -> None:
        self.torch_threads = max(1, int(torch_threads))
        torch.set_num_threads(self.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        linker_kwargs: dict[str, Any] = {"hand": hand}
        if yaml_path:
            linker_kwargs["yaml_path"] = yaml_path
        self.linker = LinkerO6Retargeter(**linker_kwargs)
        wuji_kwargs: dict[str, Any] = {"hand": hand}
        if wuji_yaml_path:
            wuji_kwargs["yaml_path"] = wuji_yaml_path
        self.wuji = WujiHandRetargeter(**wuji_kwargs)
        self.wuji_maxeval = int(np.clip(int(wuji_maxeval), 5, 100))
        self.wuji.retargeter.optimizer.opt.set_maxeval(self.wuji_maxeval)
        self.hand = hand
        self.device = choose_device(device)
        checkpoint = Path(checkpoint).expanduser().resolve()
        self.model, self.transform, self.model_cfg, self.model_type = load_wrist_model(
            checkpoint_path=checkpoint,
            config_path=None if wrist_config is None else Path(wrist_config),
            dino_dir=None if dino_dir is None else Path(dino_dir),
            mano_model_dir=Path(mano_model_dir),
            device=self.device,
            allow_local_dino_fallback=False,
            model_type=model_type,
        )
        self.precision = choose_precision(precision, self.device)
        self.fusion = AdaptiveKalmanHandFusion(
            pico_std_mm=pico_std_mm,
            vision_std_mm=vision_std_mm,
            innovation_gate_mm=innovation_gate_mm,
            process_accel_mps2=process_accel_mps2,
        )
        image_size = int(self.model_cfg["data"].get("image_size", 448))
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for _ in range(max(0, int(warmup_iters))):
            predict_pts21_mano(
                model=self.model,
                transform=self.transform,
                rgb=dummy,
                device=self.device,
                precision=self.precision,
                model_type=self.model_type,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _retarget_wuji_mano(self, points: np.ndarray) -> np.ndarray:
        backend = self.wuji.retargeter
        keypoints = np.asarray(points, dtype=np.float64).reshape(21, 3).copy()
        rotation_xyz = getattr(backend, "rotation_xyz", {}) or {}
        if any(float(rotation_xyz.get(axis, 0.0)) != 0.0 for axis in ("x", "y", "z")):
            keypoints = backend._apply_rotation(keypoints)
        if getattr(backend, "_has_offset", False):
            keypoints = backend._apply_offset(keypoints)
        qpos = np.asarray(backend.optimizer.solve(keypoints), dtype=np.float32).reshape(-1)
        qpos = np.asarray(backend.lp_filter.next(qpos), dtype=np.float32).reshape(-1)
        if qpos.shape != (20,) or not np.all(np.isfinite(qpos)):
            raise RuntimeError(f"Wuji fused retargeting returned invalid shape/data: {qpos.shape}")
        return qpos

    def infer_vision(self, rgb: np.ndarray | None) -> VisionInferenceResult:
        if rgb is None:
            return VisionInferenceResult(
                points=None,
                inference_ms=None,
                error="rgb_unavailable",
            )
        t0 = time.perf_counter()
        points = None
        error = None
        try:
            points = predict_pts21_mano(
                model=self.model,
                transform=self.transform,
                rgb=np.asarray(rgb),
                device=self.device,
                precision=self.precision,
                model_type=self.model_type,
            )
            points = _as_local_points(points)
            if points is None:
                error = "vision_points_invalid_geometry"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return VisionInferenceResult(
            points=points,
            inference_ms=(time.perf_counter() - t0) * 1000.0,
            error=error,
        )

    def process_with_vision(
        self,
        *,
        raw26x7: np.ndarray,
        vision: VisionInferenceResult,
        timestamp_ns: int,
        rgb_repeated: bool = False,
        rgb_pico_delta_ms: float | None = None,
        pico_source_age_ms: float | None = None,
    ) -> FusedRetargetingResult:
        process_t0 = time.perf_counter()
        pico_points = pico_raw_to_mano(raw26x7, hand=self.hand)
        vision_points = vision.points
        inference_ms = vision.inference_ms
        inference_error = vision.error
        vision_quality = 1.0

        if rgb_repeated:
            vision_quality *= 0.25
        if vision.reused:
            age_ms = max(float(vision.age_ms or 0.0), 0.0)
            vision_quality *= float(np.exp(-0.5 * (age_ms / 80.0) ** 2))
        if rgb_pico_delta_ms is not None:
            vision_quality *= float(np.exp(-0.5 * (abs(rgb_pico_delta_ms) / 25.0) ** 2))
        pico_quality = 1.0
        if pico_source_age_ms is not None:
            pico_quality *= float(np.exp(-0.5 * (max(pico_source_age_ms, 0.0) / 35.0) ** 2))

        fusion_t0 = time.perf_counter()
        fused = self.fusion.update(
            pico_points=pico_points,
            vision_points=vision_points,
            timestamp_ns=timestamp_ns,
            pico_quality=pico_quality,
            vision_quality=vision_quality,
        )
        fusion_ms = (time.perf_counter() - fusion_t0) * 1000.0
        fused_points = fused.points.astype(np.float32)
        linker_t0 = time.perf_counter()
        linker_joint_radians = self.linker.retarget(fused_points)
        linker_ms = (time.perf_counter() - linker_t0) * 1000.0
        wuji_t0 = time.perf_counter()
        wuji_joint_radians = self._retarget_wuji_mano(fused_points)
        wuji_ms = (time.perf_counter() - wuji_t0) * 1000.0
        diagnostics = dict(fused.diagnostics)
        post_inference_ms = (time.perf_counter() - process_t0) * 1000.0
        diagnostics.update(
            {
                "fusionFilterMs": fusion_ms,
                "linkerRetargetMs": linker_ms,
                "wujiRetargetMs": wuji_ms,
                "wujiMaxeval": self.wuji_maxeval,
                "visionInferenceReused": bool(vision.reused),
                "visionInferenceAgeMs": vision.age_ms,
                "postInferenceProcessMs": post_inference_ms,
                "fullProcessMs": post_inference_ms + float(inference_ms or 0.0),
            }
        )
        return FusedRetargetingResult(
            pts21_mano=fused_points,
            linker_joint_radians=linker_joint_radians,
            wuji_joint_radians=wuji_joint_radians,
            wrist_pose_6d=pico_wrist_pose6d(raw26x7),
            pico_pts21_mano=pico_points.astype(np.float32),
            wrist_pts21_mano=(None if vision_points is None else vision_points.astype(np.float32)),
            fusion_pico_weights=fused.pico_weights,
            fusion_vision_weights=fused.vision_weights,
            fusion_mode=fused.mode,
            fusion_diagnostics=diagnostics,
            wrist_inference_ms=inference_ms,
            wrist_inference_error=inference_error,
            wrist_inference_reused=bool(vision.reused),
            wrist_inference_age_ms=vision.age_ms,
        )

    def process(
        self,
        *,
        raw26x7: np.ndarray,
        rgb: np.ndarray | None,
        timestamp_ns: int,
        rgb_repeated: bool = False,
        rgb_pico_delta_ms: float | None = None,
        pico_source_age_ms: float | None = None,
    ) -> FusedRetargetingResult:
        vision = self.infer_vision(rgb)
        return self.process_with_vision(
            raw26x7=raw26x7,
            vision=vision,
            timestamp_ns=timestamp_ns,
            rgb_repeated=rgb_repeated,
            rgb_pico_delta_ms=rgb_pico_delta_ms,
            pico_source_age_ms=pico_source_age_ms,
        )

    def reset(self) -> None:
        self.linker.reset()
        self.wuji.reset()
        self.fusion.reset()

    def close(self) -> None:
        return None


class AsyncFusedPipeline:
    """Ordered GPU-inference/CPU-retarget pipeline for sustained capture FPS."""

    def __init__(
        self,
        retargeter: FusedPicoWristRetargeter,
        *,
        max_queue: int = 32,
        vision_stride: int = 2,
    ) -> None:
        self.retargeter = retargeter
        self.max_queue = max(2, int(max_queue))
        self.vision_stride = max(1, int(vision_stride))
        self._input: queue.Queue[_AsyncFusedRequest | None] = queue.Queue(
            maxsize=self.max_queue
        )
        self._retarget: queue.Queue[_AsyncVisionResult | None] = queue.Queue(
            maxsize=self.max_queue
        )
        self._output: queue.Queue[AsyncFusedCompletion] = queue.Queue()
        self._inference_thread = threading.Thread(
            target=self._run_inference,
            name="FusedVisionInference",
            daemon=True,
        )
        self._retarget_thread = threading.Thread(
            target=self._run_retarget,
            name="FusedHandRetarget",
            daemon=True,
        )
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._last_vision: VisionInferenceResult | None = None
        self._last_vision_timestamp_ns: int | None = None
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.submitted = 0
        self.completed = 0
        self.failed = 0
        self.submit_blocked = 0
        self.submit_blocked_ms = 0.0
        self.input_queue_high_water = 0
        self.retarget_queue_high_water = 0

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("AsyncFusedPipeline 已关闭")
        if not self._started:
            self._started = True
            self._inference_thread.start()
            self._retarget_thread.start()

    def reset(self) -> None:
        self.wait_empty()
        if not self._output.empty():
            raise RuntimeError("reset 前必须先 drain pipeline output")
        self.retargeter.reset()
        self._last_vision = None
        self._last_vision_timestamp_ns = None
        with self._lock:
            self._reset_stats()

    def submit(
        self,
        *,
        sequence_id: int,
        raw26x7: np.ndarray,
        rgb: np.ndarray | None,
        timestamp_ns: int,
        rgb_repeated: bool = False,
        rgb_pico_delta_ms: float | None = None,
        pico_source_age_ms: float | None = None,
        context: Any = None,
    ) -> None:
        if not self._started or self._closed:
            raise RuntimeError("AsyncFusedPipeline 未启动或已关闭")
        request = _AsyncFusedRequest(
            sequence_id=int(sequence_id),
            raw26x7=np.asarray(raw26x7, dtype=np.float32).reshape(26, 7).copy(),
            rgb=None if rgb is None else np.asarray(rgb),
            timestamp_ns=int(timestamp_ns),
            rgb_repeated=bool(rgb_repeated),
            rgb_pico_delta_ms=rgb_pico_delta_ms,
            pico_source_age_ms=pico_source_age_ms,
            context=context,
            submitted_perf_ns=time.perf_counter_ns(),
        )
        blocked_t0 = time.perf_counter_ns()
        try:
            self._input.put_nowait(request)
            blocked_ms = 0.0
        except queue.Full:
            self._input.put(request)
            blocked_ms = (time.perf_counter_ns() - blocked_t0) / 1e6
        with self._lock:
            self.submitted += 1
            if blocked_ms > 0.0:
                self.submit_blocked += 1
                self.submit_blocked_ms += blocked_ms
            self.input_queue_high_water = max(
                self.input_queue_high_water,
                self._input.qsize(),
            )

    def drain(self) -> list[AsyncFusedCompletion]:
        completions: list[AsyncFusedCompletion] = []
        while True:
            try:
                completions.append(self._output.get_nowait())
            except queue.Empty:
                return completions

    def wait_empty(self) -> None:
        if not self._started:
            return
        self._input.join()
        self._retarget.join()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "asyncStages": ["capture", "visionInference", "fusionAndRetarget"],
                "maxQueue": int(self.max_queue),
                "visionStride": int(self.vision_stride),
                "submitted": int(self.submitted),
                "completed": int(self.completed),
                "failed": int(self.failed),
                "submitBlocked": int(self.submit_blocked),
                "submitBlockedMs": float(self.submit_blocked_ms),
                "inputQueueHighWater": int(self.input_queue_high_water),
                "retargetQueueHighWater": int(self.retarget_queue_high_water),
                "inputPending": int(self._input.qsize()),
                "retargetPending": int(self._retarget.qsize()),
                "outputPending": int(self._output.qsize()),
                "inferenceThreadAlive": bool(self._inference_thread.is_alive()),
                "retargetThreadAlive": bool(self._retarget_thread.is_alive()),
                "device": str(self.retargeter.device),
                "precision": str(self.retargeter.precision),
            }

    def close(self, timeout_s: float = 10.0) -> None:
        if not self._started or self._closed:
            return
        self.wait_empty()
        self._input.put(None)
        self._inference_thread.join(timeout=max(float(timeout_s), 0.1))
        self._retarget_thread.join(timeout=max(float(timeout_s), 0.1))
        if self._inference_thread.is_alive() or self._retarget_thread.is_alive():
            raise RuntimeError("AsyncFusedPipeline worker 未能正常退出")
        self._closed = True

    def _run_inference(self) -> None:
        while True:
            request = self._input.get()
            try:
                if request is None:
                    self._retarget.put(None)
                    return
                started_ns = time.perf_counter_ns()
                try:
                    run_inference = (
                        self._last_vision is None
                        or self._last_vision.points is None
                        or request.sequence_id % self.vision_stride == 0
                    )
                    if run_inference:
                        vision = self.retargeter.infer_vision(request.rgb)
                        if vision.points is not None:
                            self._last_vision = vision
                            self._last_vision_timestamp_ns = request.timestamp_ns
                    else:
                        assert self._last_vision is not None
                        age_ms = (
                            None
                            if self._last_vision_timestamp_ns is None
                            else max(
                                0.0,
                                (request.timestamp_ns - self._last_vision_timestamp_ns)
                                / 1e6,
                            )
                        )
                        vision = VisionInferenceResult(
                            points=self._last_vision.points,
                            inference_ms=None,
                            error=self._last_vision.error,
                            reused=True,
                            age_ms=age_ms,
                        )
                except Exception as exc:
                    vision = VisionInferenceResult(
                        points=None,
                        inference_ms=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                finished_ns = time.perf_counter_ns()
                self._retarget.put(
                    _AsyncVisionResult(
                        request=request,
                        vision=vision,
                        inference_started_perf_ns=started_ns,
                        inference_finished_perf_ns=finished_ns,
                    )
                )
                with self._lock:
                    self.retarget_queue_high_water = max(
                        self.retarget_queue_high_water,
                        self._retarget.qsize(),
                    )
            finally:
                self._input.task_done()

    def _run_retarget(self) -> None:
        while True:
            inferred = self._retarget.get()
            try:
                if inferred is None:
                    return
                request = inferred.request
                retarget_started_ns = time.perf_counter_ns()
                result = None
                error = None
                try:
                    result = self.retargeter.process_with_vision(
                        raw26x7=request.raw26x7,
                        vision=inferred.vision,
                        timestamp_ns=request.timestamp_ns,
                        rgb_repeated=request.rgb_repeated,
                        rgb_pico_delta_ms=request.rgb_pico_delta_ms,
                        pico_source_age_ms=request.pico_source_age_ms,
                    )
                    finished_ns = time.perf_counter_ns()
                    result.fusion_diagnostics.update(
                        {
                            "pipelineQueueWaitMs": (
                                inferred.inference_started_perf_ns
                                - request.submitted_perf_ns
                            )
                            / 1e6,
                            "retargetQueueWaitMs": (
                                retarget_started_ns
                                - inferred.inference_finished_perf_ns
                            )
                            / 1e6,
                            "pipelineLatencyMs": (
                                finished_ns - request.submitted_perf_ns
                            )
                            / 1e6,
                        }
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                self._output.put(
                    AsyncFusedCompletion(
                        sequence_id=request.sequence_id,
                        context=request.context,
                        result=result,
                        error=error,
                    )
                )
                with self._lock:
                    self.completed += 1
                    if error is not None:
                        self.failed += 1
            finally:
                self._retarget.task_done()
