#!/usr/bin/env python3
"""Reliability-gated PICO-teacher / wrist-vision-student hand fusion.

This module is intentionally separate from the production collector while the
method is evaluated.  Unlike ordinary precision averaging, each source is
checked against its own temporal track.  Healthy PICO remains the pseudo-label
teacher; wrist vision is selected only for joints with a detected PICO fault.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hand_keypoint_fusion import (
    BONES,
    FusionOutput,
    _as_local_points,
    _bone_lengths,
    _joint_bone_scores,
)


@dataclass
class _SourceTrack:
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    last_timestamp_ns: int | None = None


class TeacherStudentGatedHandFusion:
    """Per-joint fault selector with independent source tracks and hysteresis."""

    def __init__(
        self,
        *,
        innovation_gate_mm: float = 18.0,
        healthy_score: float = 0.55,
        fault_score: float = 0.30,
        source_margin: float = 1.25,
        switch_frames: int = 2,
        recover_frames: int = 3,
        catastrophic_gate_scale: float = 2.5,
        track_update_score: float = 0.20,
        project_mixed_bones: bool = True,
        fallback_blend_initial: float = 0.35,
        fallback_blend_step: float = 0.15,
        fallback_blend_max: float = 0.85,
    ) -> None:
        self.innovation_gate_m = max(float(innovation_gate_mm), 2.0) / 1000.0
        self.healthy_score = float(np.clip(healthy_score, 0.1, 0.95))
        self.fault_score = float(np.clip(fault_score, 0.01, self.healthy_score))
        self.source_margin = max(float(source_margin), 1.0)
        self.switch_frames = max(int(switch_frames), 1)
        self.recover_frames = max(int(recover_frames), 1)
        self.catastrophic_gate_scale = max(float(catastrophic_gate_scale), 1.0)
        self.track_update_score = float(np.clip(track_update_score, 0.01, 0.9))
        self.project_mixed_bones = bool(project_mixed_bones)
        self.fallback_blend_initial = float(np.clip(fallback_blend_initial, 0.0, 1.0))
        self.fallback_blend_step = float(np.clip(fallback_blend_step, 0.0, 1.0))
        self.fallback_blend_max = float(
            np.clip(fallback_blend_max, self.fallback_blend_initial, 1.0)
        )
        self.reset()

    def reset(self) -> None:
        self.pico_track = _SourceTrack()
        self.vision_track = _SourceTrack()
        self.use_vision = np.zeros(21, dtype=bool)
        self.switch_count = np.zeros(21, dtype=np.int32)
        self.recover_count = np.zeros(21, dtype=np.int32)
        self.bone_reference: np.ndarray | None = None

    @staticmethod
    def _dt(track: _SourceTrack, timestamp_ns: int) -> float:
        if track.last_timestamp_ns is None:
            return 1.0 / 30.0
        return float(
            np.clip(
                (int(timestamp_ns) - int(track.last_timestamp_ns)) / 1e9,
                1.0 / 120.0,
                0.10,
            )
        )

    def _score_source(
        self,
        track: _SourceTrack,
        points: np.ndarray,
        timestamp_ns: int,
        source_quality: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        dt = self._dt(track, timestamp_ns)
        if track.position is None:
            predicted = points.copy()
            innovation = np.zeros(21, dtype=np.float32)
            score = np.ones(21, dtype=np.float32) * float(
                np.clip(source_quality, 0.0, 1.0)
            )
            return score, innovation, predicted, dt

        assert track.velocity is not None
        predicted = track.position + track.velocity * dt
        innovation = np.linalg.norm(points - predicted, axis=1)
        speed_allowance = np.linalg.norm(track.velocity, axis=1) * dt * 1.5
        gate = self.innovation_gate_m + speed_allowance
        innovation_score = np.exp(
            -0.5 * np.square(innovation / np.maximum(gate, 1e-5))
        )
        bone_score = _joint_bone_scores(points, self.bone_reference)
        score = (
            float(np.clip(source_quality, 0.0, 1.0))
            * innovation_score
            * bone_score
        )
        return np.clip(score, 0.0, 1.0).astype(np.float32), innovation, predicted, dt

    def _update_track(
        self,
        track: _SourceTrack,
        points: np.ndarray,
        score: np.ndarray,
        predicted: np.ndarray,
        dt: float,
        timestamp_ns: int,
    ) -> None:
        if track.position is None:
            track.position = points.copy()
            track.velocity = np.zeros_like(points)
            track.last_timestamp_ns = int(timestamp_ns)
            return
        assert track.velocity is not None
        previous = track.position.copy()
        accepted = score >= self.track_update_score
        corrected = predicted.copy()
        corrected[accepted] = (
            0.65 * points[accepted] + 0.35 * predicted[accepted]
        )
        observed_velocity = (corrected - previous) / max(dt, 1e-4)
        track.velocity = 0.80 * track.velocity + 0.20 * observed_velocity
        track.velocity[~accepted] *= 0.90
        track.position = corrected.astype(np.float32)
        track.last_timestamp_ns = int(timestamp_ns)

    def _project_mixed_bones(self, points: np.ndarray) -> np.ndarray:
        if self.bone_reference is None:
            return points
        projected = points.copy()
        projected[0] = 0.0
        for length, (parent, child) in zip(self.bone_reference, BONES):
            direction = projected[child] - projected[parent]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            projected[child] = projected[parent] + direction * (float(length) / norm)
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
            raise ValueError("teacher-student fusion requires at least one valid source")

        pico_score = np.zeros(21, dtype=np.float32)
        vision_score = np.zeros(21, dtype=np.float32)
        pico_innovation = np.full(21, np.nan, dtype=np.float32)
        vision_innovation = np.full(21, np.nan, dtype=np.float32)
        pico_predicted = pico
        vision_predicted = vision
        pico_dt = vision_dt = 1.0 / 30.0

        if pico is not None:
            pico_score, pico_innovation, pico_predicted, pico_dt = self._score_source(
                self.pico_track,
                pico,
                timestamp_ns,
                pico_quality,
            )
        if vision is not None:
            vision_score, vision_innovation, vision_predicted, vision_dt = self._score_source(
                self.vision_track,
                vision,
                timestamp_ns,
                vision_quality,
            )

        if self.bone_reference is None:
            reference_points = pico if pico is not None else vision
            assert reference_points is not None
            self.bone_reference = _bone_lengths(reference_points).astype(np.float32)

        if pico is not None:
            self._update_track(
                self.pico_track,
                pico,
                pico_score,
                pico_predicted,
                pico_dt,
                timestamp_ns,
            )
        if vision is not None:
            self._update_track(
                self.vision_track,
                vision,
                vision_score,
                vision_predicted,
                vision_dt,
                timestamp_ns,
            )

        if pico is None:
            self.use_vision[:] = True
        elif vision is None:
            self.use_vision[:] = False
        else:
            catastrophic = pico_innovation >= (
                self.innovation_gate_m * self.catastrophic_gate_scale
            )
            prefer_vision = (
                (pico_score < self.fault_score)
                & (vision_score >= self.fault_score)
                & (vision_score > pico_score * self.source_margin)
            )
            prefer_vision |= catastrophic & (vision_score >= self.fault_score)
            if pico_quality < self.fault_score and vision_quality >= self.fault_score:
                prefer_vision[:] = True

            self.switch_count = np.where(
                prefer_vision,
                self.switch_count + 1,
                0,
            )
            immediate = catastrophic | (pico_quality < self.fault_score)
            self.use_vision |= immediate | (self.switch_count >= self.switch_frames)

            pico_recovered = pico_score >= self.healthy_score
            self.recover_count = np.where(
                self.use_vision & pico_recovered,
                self.recover_count + 1,
                0,
            )
            self.use_vision &= self.recover_count < self.recover_frames

        self.use_vision[0] = bool(pico is None and vision is not None)
        if pico is not None and vision is not None:
            selected = pico.copy()
            fallback_blend = np.clip(
                self.fallback_blend_initial
                + self.fallback_blend_step * np.maximum(self.switch_count - 1, 0),
                self.fallback_blend_initial,
                self.fallback_blend_max,
            ).astype(np.float32)
            fallback = (
                fallback_blend[:, None] * vision
                + (1.0 - fallback_blend[:, None]) * pico_predicted
            )
            selected[self.use_vision] = fallback[self.use_vision]
            if self.project_mixed_bones and np.any(self.use_vision[1:]):
                selected = self._project_mixed_bones(selected)
        elif pico is not None:
            selected = pico.copy()
        else:
            assert vision is not None
            selected = vision.copy()
        selected[0] = 0.0

        if pico is None:
            vision_weights = np.ones(21, dtype=np.float32)
        elif vision is None:
            vision_weights = np.zeros(21, dtype=np.float32)
        else:
            vision_weights = np.zeros(21, dtype=np.float32)
            vision_weights[self.use_vision] = fallback_blend[self.use_vision]
        pico_weights = 1.0 - vision_weights
        if pico is None:
            mode = "vision_only"
        elif vision is None:
            mode = "pico_only"
        elif np.all(~self.use_vision[1:]):
            mode = "pico_teacher"
        elif np.all(self.use_vision[1:]):
            mode = "vision_fallback"
        else:
            mode = "mixed_fallback"

        diagnostics: dict[str, Any] = {
            "picoScoreMean": None if pico is None else float(np.mean(pico_score[1:])),
            "visionScoreMean": None if vision is None else float(np.mean(vision_score[1:])),
            "picoInnovationMmMean": None
            if pico is None
            else float(np.nanmean(pico_innovation[1:]) * 1000.0),
            "visionInnovationMmMean": None
            if vision is None
            else float(np.nanmean(vision_innovation[1:]) * 1000.0),
            "visionSelectedJointCount": int(np.sum(self.use_vision[1:])),
            "visionFallbackBlendMean": None
            if not np.any(self.use_vision[1:]) or pico is None or vision is None
            else float(np.mean(fallback_blend[self.use_vision])),
            "strategy": "teacher_student_gated",
        }
        return FusionOutput(
            points=selected.astype(np.float32),
            pico_weights=pico_weights,
            vision_weights=vision_weights,
            mode=mode,
            diagnostics=diagnostics,
        )
