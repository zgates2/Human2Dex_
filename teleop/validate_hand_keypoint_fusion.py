#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def bone_lengths(points: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(points[b] - points[a]) for a, b in BONES])


def temporal_second_difference_mm(sequence: list[np.ndarray]) -> float | None:
    if len(sequence) < 3:
        return None
    arr = np.stack(sequence)
    second = arr[2:] - 2.0 * arr[1:-1] + arr[:-2]
    return float(np.mean(np.linalg.norm(second[:, 1:], axis=2)) * 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pkl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mano-model-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--output-pkl", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "teleop"), str(repo / "wrist")]

    from hand_keypoint_fusion import FusedPicoWristRetargeter
    from episode_io import build_message, save_pkl

    with args.pkl.open("rb") as f:
        payload = pickle.load(f)
    messages = payload.get("messages", payload if isinstance(payload, list) else [])
    pipeline = FusedPicoWristRetargeter(
        checkpoint=args.checkpoint,
        mano_model_dir=args.mano_model_dir,
        model_type="stage2",
        device="cuda",
        precision="bf16",
        warmup_iters=20,
    )

    modes = Counter()
    inference_ms = []
    process_ms = []
    pico_weights = []
    vision_weights = []
    cross_mm = []
    pico_seq = []
    vision_seq = []
    fused_seq = []
    fused_bones = []
    pico_bones = []
    vision_bones = []
    command_shapes = set()
    message_field_shapes = set()
    errors = []
    timestamps = []
    last_raw = None
    output_messages = []

    for msg in messages:
        raw = msg.get("raw26x7")
        image_rel = msg.get("rgbImage")
        if raw is None or image_rel is None:
            continue
        image_path = args.pkl.parent / image_rel
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        sample_ns = int(msg.get("sampleClockNs") or 0)
        if sample_ns <= 0:
            sample_ns = len(fused_seq) * 33_333_333
        delta_ns = msg.get("rgbToPicoReceiveDeltaNs")
        age_ns = msg.get("sourceAgeNs")
        try:
            process_t0 = time.perf_counter()
            result = pipeline.process(
                raw26x7=np.asarray(raw),
                rgb=rgb,
                timestamp_ns=sample_ns,
                rgb_repeated=bool(msg.get("rgbFrameRepeated") is True),
                rgb_pico_delta_ms=None if delta_ns is None else float(delta_ns) / 1e6,
                pico_source_age_ms=None if age_ns is None else float(age_ns) / 1e6,
            )
            process_ms.append((time.perf_counter() - process_t0) * 1000.0)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        modes[result.fusion_mode] += 1
        last_raw = np.asarray(raw).copy()
        if result.wrist_inference_ms is not None:
            inference_ms.append(float(result.wrist_inference_ms))
        pico_weights.append(float(np.mean(result.fusion_pico_weights[1:])))
        vision_weights.append(float(np.mean(result.fusion_vision_weights[1:])))
        cross = result.fusion_diagnostics.get("crossSensorMmMean")
        if cross is not None:
            cross_mm.append(float(cross))
        pico_seq.append(result.pico_pts21_mano.copy())
        assert result.wrist_pts21_mano is not None
        vision_seq.append(result.wrist_pts21_mano.copy())
        fused_seq.append(result.pts21_mano.copy())
        timestamps.append(sample_ns)
        pico_bones.append(bone_lengths(result.pico_pts21_mano))
        vision_bones.append(bone_lengths(result.wrist_pts21_mano))
        fused_bones.append(bone_lengths(result.pts21_mano))
        command_shapes.add((result.linker_joint_radians.shape, result.wuji_joint_radians.shape))
        fields = result.message_fields()
        message_field_shapes.add(
            (
                fields["pico_pts21_mano"].shape,
                None if fields["wrist_pts21_mano"] is None else fields["wrist_pts21_mano"].shape,
                fields["fused_pts21_mano"].shape,
                fields["fusionPicoWeights"].shape,
                fields["fusionVisionWeights"].shape,
            )
        )
        if args.output_pkl is not None:
            built = build_message(
                o6_angles=result.linker_joint_radians,
                wuji_qpos=result.wuji_joint_radians,
                pts21_mano=result.pts21_mano,
                wrist_pose_6d=result.wrist_pose_6d,
                sdk_ts_ns=int(msg.get("mainClockMonotonicNs") or sample_ns),
                sample_ns=sample_ns,
                source_receive_ns=int(msg.get("sourceReceiveNs") or sample_ns),
                raw26x7=np.asarray(raw),
                pico_active=1,
                valid_mask=[True] * 21,
                quality_payload={"ok": True, "flags": [], "source": "fusion_validator"},
            )
            built.update(fields)
            output_messages.append(built)
        if len(fused_seq) >= args.max_frames:
            break

    if not fused_seq:
        raise RuntimeError(f"no valid fused frames; errors={errors[:3]}")

    def bone_std_mm(values: list[np.ndarray]) -> float:
        return float(np.mean(np.std(np.stack(values), axis=0)) * 1000.0)

    ordered_inf = sorted(inference_ms)
    ordered_process = sorted(process_ms)
    from hand_keypoint_fusion import AdaptiveKalmanHandFusion
    fault_filter = AdaptiveKalmanHandFusion()
    rng = np.random.default_rng(123)
    requested_injected_frames = set(range(30, 36)) | set(range(75, 81))
    injected_frames = {idx for idx in requested_injected_frames if idx < len(pico_seq)}
    if not injected_frames and len(pico_seq) >= 4:
        start = max(1, len(pico_seq) // 3)
        injected_frames = set(range(start, min(start + 6, len(pico_seq))))
    injected_joints = np.asarray([4, 8, 12, 16, 20], dtype=int)
    corrupted_errors = []
    recovered_errors = []
    injected_vision_weights = []
    normal_vision_weights = []
    for idx, (pico, vision, timestamp_ns) in enumerate(zip(pico_seq, vision_seq, timestamps)):
        corrupted = pico.copy()
        if idx in injected_frames:
            corrupted[injected_joints] += rng.normal(
                loc=0.0,
                scale=0.035,
                size=(len(injected_joints), 3),
            ).astype(np.float32)
        output = fault_filter.update(
            pico_points=corrupted,
            vision_points=vision,
            timestamp_ns=timestamp_ns,
        )
        if idx in injected_frames:
            corrupted_errors.append(
                float(np.mean(np.linalg.norm(corrupted[injected_joints] - pico[injected_joints], axis=1)) * 1000.0)
            )
            recovered_errors.append(
                float(np.mean(np.linalg.norm(output.points[injected_joints] - pico[injected_joints], axis=1)) * 1000.0)
            )
            injected_vision_weights.append(float(np.mean(output.vision_weights[injected_joints])))
        else:
            normal_vision_weights.append(float(np.mean(output.vision_weights[injected_joints])))

    assert last_raw is not None
    fallback = pipeline.process(
        raw26x7=last_raw,
        rgb=None,
        timestamp_ns=timestamps[-1] + 33_333_333,
    )

    output_contract = None
    if args.output_pkl is not None:
        save_pkl(
            output_messages,
            args.output_pkl,
            metadata={"source": "validate_hand_keypoint_fusion", "frames": len(output_messages)},
        )
        with args.output_pkl.open("rb") as f:
            reloaded = pickle.load(f)
        first = reloaded["messages"][0]
        output_contract = {
            "path": str(args.output_pkl),
            "formatVersion": int(reloaded["formatVersion"]),
            "frames": len(reloaded["messages"]),
            "pts21Shape": list(np.asarray(first["pts21_mano"]).shape),
            "o6Shape": list(np.asarray(first["o6_command"]).shape),
            "wujiShape": list(np.asarray(first["wuji_command"]).shape),
            "fusionWeightsShape": list(np.asarray(first["fusionPicoWeights"]).shape),
            "allRequiredFields": all(
                key in first
                for key in (
                    "pico_pts21_mano",
                    "wrist_pts21_mano",
                    "fused_pts21_mano",
                    "fusionPicoWeights",
                    "fusionVisionWeights",
                    "fusionMode",
                    "fusionDiagnostics",
                    "wristInferenceMs",
                    "wristInferenceError",
                )
            ),
        }

    report = {
        "frames": len(fused_seq),
        "modes": dict(modes),
        "inferenceMeanMs": statistics.fmean(inference_ms),
        "inferenceP95Ms": ordered_inf[min(len(ordered_inf) - 1, int(len(ordered_inf) * 0.95))],
        "fullProcessMeanMs": statistics.fmean(process_ms),
        "fullProcessP95Ms": ordered_process[min(len(ordered_process) - 1, int(len(ordered_process) * 0.95))],
        "picoWeightMean": statistics.fmean(pico_weights),
        "visionWeightMean": statistics.fmean(vision_weights),
        "crossSensorMmMean": statistics.fmean(cross_mm),
        "boneLengthStdMm": {
            "pico": bone_std_mm(pico_bones),
            "vision": bone_std_mm(vision_bones),
            "fused": bone_std_mm(fused_bones),
        },
        "temporalSecondDifferenceMm": {
            "pico": temporal_second_difference_mm(pico_seq),
            "vision": temporal_second_difference_mm(vision_seq),
            "fused": temporal_second_difference_mm(fused_seq),
        },
        "commandShapes": [str(item) for item in command_shapes],
        "messageFieldShapes": [str(item) for item in message_field_shapes],
        "rgbFailureFallback": {
            "mode": fallback.fusion_mode,
            "error": fallback.wrist_inference_error,
            "picoWeightMean": float(np.mean(fallback.fusion_pico_weights[1:])),
            "commandsFinite": bool(
                np.all(np.isfinite(fallback.linker_joint_radians))
                and np.all(np.isfinite(fallback.wuji_joint_radians))
            ),
        },
        "allFusedFinite": bool(np.all(np.isfinite(np.stack(fused_seq)))),
        "outputPklContract": output_contract,
        "injectedPicoFaultTest": {
            "frames": sorted(injected_frames),
            "joints": injected_joints.tolist(),
            "rawCorruptedErrorMm": statistics.fmean(corrupted_errors),
            "fusedRecoveredErrorMm": statistics.fmean(recovered_errors),
            "errorReductionPercent": 100.0 * (1.0 - statistics.fmean(recovered_errors) / statistics.fmean(corrupted_errors)),
            "visionWeightNormal": statistics.fmean(normal_vision_weights),
            "visionWeightDuringFault": statistics.fmean(injected_vision_weights),
        },
        "errors": errors[:5],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
