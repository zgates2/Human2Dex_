#!/usr/bin/env python3
"""Synthetic fault tests for teacher-student hand fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_TELEOP = HERE.parent / "teleop"
for path in (HERE, REPO_TELEOP):
    if path.is_dir():
        sys.path.insert(0, str(path))

from hand_keypoint_fusion import AdaptiveKalmanHandFusion
from teacher_student_hand_fusion import TeacherStudentGatedHandFusion


def base_hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    finger_x = (-0.035, -0.017, 0.0, 0.017, 0.033)
    for finger, x in enumerate(finger_x):
        start = 1 + finger * 4
        for joint in range(4):
            points[start + joint] = [x, 0.025 + joint * 0.022, 0.003 * finger]
    return points


def trajectory(frame: int) -> np.ndarray:
    points = base_hand()
    phase = np.sin(frame * 0.08)
    for tip in (4, 8, 12, 16, 20):
        points[tip, 2] += 0.008 * phase
        points[tip, 1] -= 0.004 * phase
    return points


def mean_error_mm(pred: np.ndarray, target: np.ndarray, joints: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred[joints] - target[joints], axis=1)) * 1000.0)


def main() -> None:
    rng = np.random.default_rng(20260716)
    gated = TeacherStudentGatedHandFusion()
    averaged = AdaptiveKalmanHandFusion()
    tips = np.asarray([4, 8, 12, 16, 20])
    pico_fault_frames = set(range(30, 36))
    vision_fault_frames = set(range(55, 61))

    clean_gated = []
    clean_pico = []
    clean_vision_selected = []
    pico_fault_raw = []
    pico_fault_gated = []
    pico_fault_average = []
    selected_during_pico_fault = []
    selected_during_vision_fault = []

    for frame in range(90):
        truth = trajectory(frame)
        pico = truth + rng.normal(0.0, 0.0015, truth.shape).astype(np.float32)
        vision = truth + rng.normal(0.0, 0.0040, truth.shape).astype(np.float32)
        if frame in pico_fault_frames:
            pico[tips] += rng.normal(0.0, 0.035, (len(tips), 3)).astype(np.float32)
        if frame in vision_fault_frames:
            vision[tips] += rng.normal(0.0, 0.045, (len(tips), 3)).astype(np.float32)

        pico_local = pico - pico[0:1]

        timestamp_ns = frame * 33_333_333
        gated_out = gated.update(
            pico_points=pico,
            vision_points=vision,
            timestamp_ns=timestamp_ns,
        )
        average_out = averaged.update(
            pico_points=pico,
            vision_points=vision,
            timestamp_ns=timestamp_ns,
        )

        if frame not in pico_fault_frames and frame not in vision_fault_frames and frame > 5:
            clean_gated.append(mean_error_mm(gated_out.points, truth, tips))
            clean_pico.append(mean_error_mm(pico_local, truth, tips))
            clean_vision_selected.append(float(np.mean(gated_out.vision_weights[tips])))
        if frame in pico_fault_frames:
            pico_fault_raw.append(mean_error_mm(pico_local, truth, tips))
            pico_fault_gated.append(mean_error_mm(gated_out.points, truth, tips))
            pico_fault_average.append(mean_error_mm(average_out.points, truth, tips))
            selected_during_pico_fault.append(float(np.mean(gated_out.vision_weights[tips])))
        if frame in vision_fault_frames:
            selected_during_vision_fault.append(float(np.mean(gated_out.vision_weights[tips])))

    metrics = {
        "clean_gated_mm": float(np.mean(clean_gated)),
        "clean_pico_mm": float(np.mean(clean_pico)),
        "vision_selected_clean": float(np.mean(clean_vision_selected)),
        "pico_fault_raw_mm": float(np.mean(pico_fault_raw)),
        "pico_fault_gated_mm": float(np.mean(pico_fault_gated)),
        "pico_fault_average_mm": float(np.mean(pico_fault_average)),
        "vision_selected_pico_fault": float(np.mean(selected_during_pico_fault)),
        "vision_selected_vision_fault": float(np.mean(selected_during_vision_fault)),
    }
    print(metrics)
    assert np.mean(clean_gated) <= np.mean(clean_pico) + 0.5
    assert np.mean(pico_fault_gated) < np.mean(pico_fault_raw) * 0.55
    assert np.mean(pico_fault_gated) < np.mean(pico_fault_average)
    assert np.mean(selected_during_pico_fault) > 0.30
    assert np.mean(selected_during_vision_fault) < 0.10

    dropout = gated.update(
        pico_points=None,
        vision_points=trajectory(91),
        timestamp_ns=91 * 33_333_333,
    )
    assert dropout.mode == "vision_only"
    assert np.all(dropout.vision_weights == 1.0)
    print(
        "[OK] teacher-student fusion synthetic faults passed "
        f"clean={np.mean(clean_gated):.2f}mm "
        f"pico_fault={np.mean(pico_fault_gated):.2f}mm "
        f"average={np.mean(pico_fault_average):.2f}mm"
    )


if __name__ == "__main__":
    main()
