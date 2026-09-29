#!/usr/bin/env python3
"""Click Wuji hand landmarks, fit camera<-Wuji palm extrinsic, and render skeletons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import calibrate_o6_camera_extrinsic as base
from wuji_fk21 import (
    ANCHOR_SETS,
    ANCHOR_TO_POINT21_INDEX,
    CALIBRATION_ANCHOR_TO_LINK,
    HAND21_BONES,
    HAND21_NAMES,
    POINT21_COLORS_BGR,
    WUJI_ACTIVE_JOINTS,
    DEFAULT_WUJI_URDF,
    WujiKinematics,
)


DEFAULT_URDF = DEFAULT_WUJI_URDF
DEFAULT_CAMERA_CONTRACT = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "wuji_camera_calibration/mvs_DA9057802_wuji_mount_v1/"
    "camera_contract_mvs_DA9057802_wuji_mount_v1.yaml"
)
DEFAULT_EPISODES_ROOT = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "wuji_camera_calibration/mvs_DA9057802_wuji_mount_v1/calibration_episodes"
)
STATE_KEY = "wuji_qpos"


def _patch_base() -> None:
    base.ANCHOR_SETS = ANCHOR_SETS
    base.ANCHOR_TO_POINT21_INDEX = ANCHOR_TO_POINT21_INDEX
    base.CALIBRATION_ANCHOR_TO_LINK = CALIBRATION_ANCHOR_TO_LINK
    base.HAND21_BONES = HAND21_BONES
    base.HAND21_NAMES = HAND21_NAMES
    base.POINT21_COLORS_BGR = POINT21_COLORS_BGR
    base.O6Kinematics = WujiKinematics
    base.DEFAULT_URDF = DEFAULT_URDF
    base.DEFAULT_CAMERA_CONTRACT = DEFAULT_CAMERA_CONTRACT
    base.DEFAULT_EPISODES_ROOT = DEFAULT_EPISODES_ROOT
    base.make_annotation_payload = make_annotation_payload
    base.collect_correspondences = collect_correspondences
    base.correspondence_diagnostics = correspondence_diagnostics
    base.render_frame = render_frame


def _as_qpos20(value: Any, name: str = "wuji_qpos") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (20,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite shape (20,), got {arr}")
    return arr


def collect_candidates(
    episodes_root: Path,
    episode_numbers: list[int],
    state_source: str,
    state_time_alignment: str,
    max_align_ms: float,
    include_repeated: bool,
) -> list[dict[str, Any]]:
    if state_time_alignment not in {"rgb", "message"}:
        raise ValueError(
            "state_time_alignment must be 'rgb' or 'message', got "
            f"{state_time_alignment!r}"
        )
    candidates: list[dict[str, Any]] = []
    for episode_number in episode_numbers:
        pkl_path = base.episode_pkl_path(episodes_root, episode_number)
        if not pkl_path.is_file():
            raise FileNotFoundError(pkl_path)
        payload = base.load_pickle(pkl_path)
        metadata = payload.get("metadata", {})
        if metadata.get("handBackend") not in {"wuji_hand", "wujihand"}:
            raise ValueError(f"{pkl_path} is not a wuji_hand episode")
        messages = payload.get("messages", [])
        state_times = []
        state_values = []
        for message in messages:
            timestamp = message.get("timestamp")
            state = message.get(state_source)
            if timestamp is None or state is None:
                continue
            state_array = np.asarray(state, dtype=np.float64).reshape(-1)
            if (
                state_array.shape == (20,)
                and np.all(np.isfinite(state_array))
                and np.isfinite(float(timestamp))
            ):
                state_times.append(float(timestamp))
                state_values.append(state_array)
        if not state_times:
            continue
        state_times_array = np.asarray(state_times, dtype=np.float64)
        state_values_array = np.stack(state_values, axis=0)
        order = np.argsort(state_times_array)
        state_times_array = state_times_array[order]
        state_values_array = state_values_array[order]
        state_times_array, unique_indices = np.unique(
            state_times_array, return_index=True
        )
        state_values_array = state_values_array[unique_indices]

        for message_index, message in enumerate(messages):
            state = message.get(state_source)
            if state is None:
                continue
            state_array = np.asarray(state, dtype=np.float64).reshape(-1)
            if state_array.shape != (20,) or not np.all(np.isfinite(state_array)):
                continue
            if not include_repeated and bool(message.get("rgbFrameRepeated", False)):
                continue
            residual_ns = message.get("rgbAlignResidualNs")
            align_ms = None if residual_ns is None else float(residual_ns) * 1e-6
            if align_ms is not None and abs(align_ms) > max_align_ms:
                continue
            message_timestamp = message.get("timestamp")
            if message_timestamp is None or not np.isfinite(float(message_timestamp)):
                continue
            state_query_timestamp = float(message_timestamp)
            aligned_state = state_array.copy()
            if state_time_alignment == "rgb" and residual_ns is not None:
                state_query_timestamp -= float(residual_ns) * 1e-9
                if (
                    len(state_times_array) < 2
                    or state_query_timestamp < state_times_array[0]
                    or state_query_timestamp > state_times_array[-1]
                ):
                    continue
                aligned_state = np.asarray(
                    [
                        np.interp(
                            state_query_timestamp,
                            state_times_array,
                            state_values_array[:, joint_index],
                        )
                        for joint_index in range(20)
                    ],
                    dtype=np.float64,
                )
            image_value = message.get("rgbImage")
            if not image_value:
                continue
            image_path = base.resolve_image_path(pkl_path.parent, str(image_value))
            if not image_path.is_file():
                continue
            candidate = {
                "episode": f"episode_{episode_number:04d}",
                "episode_number": int(episode_number),
                "pkl_path": str(pkl_path),
                "message_index": int(message_index),
                "image_path": str(image_path),
                "rgb_frame_id": message.get("rgbFrameId"),
                "timestamp": message.get("timestamp"),
                "rgb_capture_timestamp": message.get("rgbCaptureTimestamp"),
                "rgb_align_ms": align_ms,
                "state_source": state_source,
                "state_time_alignment": state_time_alignment,
                "state_query_timestamp": state_query_timestamp,
                "state_interpolation_shift_ms": (
                    state_query_timestamp - float(message_timestamp)
                ) * 1000.0,
                "state_interpolation_max_delta": float(
                    np.max(np.abs(aligned_state - state_array))
                ),
                STATE_KEY: aligned_state.astype(float).tolist(),
            }
            # Compatibility with the shared O6-derived annotation UI and fit path.
            candidate["o6_state"] = candidate[STATE_KEY]
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No eligible Wuji frames matched the filters")
    return candidates


def state_feature(frame: dict[str, Any]) -> np.ndarray:
    qpos = np.asarray(frame[STATE_KEY], dtype=np.float64).reshape(20)
    return np.clip(qpos / np.pi, -1.0, 1.0)


def select_diverse_frames(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    count = min(int(count), len(candidates))
    features = np.stack([state_feature(frame) for frame in candidates], axis=0)
    mean = np.mean(features, axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(features - mean, axis=1)))
    selected = [first]
    min_distance = np.linalg.norm(features - features[first], axis=1)
    min_distance[first] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(min_distance))
        if index in selected:
            break
        selected.append(index)
        distance = np.linalg.norm(features - features[index], axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    frames = [dict(candidates[index]) for index in selected]
    frames.sort(key=lambda frame: (frame["episode_number"], frame["message_index"]))
    return frames


def make_annotation_payload(args: argparse.Namespace) -> dict[str, Any]:
    contract = None
    if getattr(args, "camera_contract", None):
        contract = base.load_camera_contract(Path(args.camera_contract))
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    episode_numbers = base.parse_episode_spec(args.episodes)
    candidates = collect_candidates(
        episodes_root=episodes_root,
        episode_numbers=episode_numbers,
        state_source=args.state_source,
        state_time_alignment=getattr(args, "state_time_alignment", "rgb"),
        max_align_ms=float(args.max_align_ms),
        include_repeated=bool(args.include_repeated),
    )
    frames = select_diverse_frames(candidates, int(args.num_frames))
    for index, frame in enumerate(frames):
        frame["split"] = "holdout" if (index + 1) % int(args.holdout_every) == 0 else "fit"
        frame["annotations"] = {}
        frame["skipped_anchors"] = []
        frame["done"] = False
    first_image = cv2.imread(frames[0]["image_path"], cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(frames[0]["image_path"])
    selected_image_size = (int(first_image.shape[1]), int(first_image.shape[0]))
    if contract is not None and selected_image_size != contract["image_size"]:
        raise ValueError(
            f"Selected RGB size {selected_image_size} does not match contract policy "
            f"image size {contract['image_size']}"
        )
    urdf_path = Path(
        args.urdf or (contract["urdf_path"] if contract is not None else DEFAULT_URDF)
    ).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if contract is not None and base.file_sha256(urdf_path) != contract["record"]["urdf_sha256"]:
        raise ValueError("Selected URDF does not match the camera contract")
    anchor_names = list(ANCHOR_SETS[args.anchor_set])
    result = {
        "formatVersion": 2,
        "tool": "calibrate_wuji_camera_extrinsic.py",
        "episodes_root": str(episodes_root),
        "episodes": episode_numbers,
        "urdf_path": str(urdf_path),
        "urdf_sha256": base.file_sha256(urdf_path),
        "hand": "wuji_hand",
        "wuji_base_link": "palm_link",
        "state_source": args.state_source,
        "selection": {
            "method": "greedy_farthest_wuji_qpos",
            "eligible_frames": len(candidates),
            "selected_frames": len(frames),
            "max_align_ms": float(args.max_align_ms),
            "include_repeated": bool(args.include_repeated),
            "state_time_alignment": getattr(args, "state_time_alignment", "rgb"),
            "holdout_every": int(args.holdout_every),
        },
        "anchor_set": args.anchor_set,
        "anchor_names": anchor_names,
        "frames": frames,
    }
    if contract is not None:
        result["camera_mount_id"] = contract["record"]["camera_mount_id"]
        result["camera_contract"] = contract["record"]
    return result


class WujiAnnotationApp(base.AnnotationApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = "Wuji camera extrinsic annotation"


def run_annotate(args: argparse.Namespace) -> None:
    _patch_base()
    base.AnnotationApp = WujiAnnotationApp
    base.run_annotate(args)


def collect_correspondences(
    payload: dict[str, Any],
    split: str,
    kinematics: WujiKinematics,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(payload["frames"]):
        if frame.get("split") != split:
            continue
        qpos = _as_qpos20(frame.get(STATE_KEY, frame.get("o6_state")))
        anchors = kinematics.calibration_anchor_points(qpos)
        for anchor_name, uv in frame.get("annotations", {}).items():
            if anchor_name not in anchors:
                continue
            object_points.append(anchors[anchor_name])
            image_points.append(np.asarray(uv, dtype=np.float64).reshape(2))
            metadata.append(
                {
                    "frame_index": frame_index,
                    "episode": frame["episode"],
                    "message_index": frame["message_index"],
                    "anchor": anchor_name,
                    STATE_KEY: qpos.astype(float).tolist(),
                    "o6_state": qpos.astype(float).tolist(),
                }
            )
    if object_points:
        return np.stack(object_points), np.stack(image_points), metadata
    return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), metadata


def correspondence_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        residuals = np.asarray(
            [row["residual_uv"] for row in group_rows], dtype=np.float64
        ).reshape(-1, 2)
        errors = np.asarray(
            [row["error_px"] for row in group_rows], dtype=np.float64
        )
        bias = np.mean(residuals, axis=0)
        residual_norm_sum = float(np.sum(np.linalg.norm(residuals, axis=1)))
        direction_consistency = (
            0.0
            if residual_norm_sum <= 1e-12
            else float(np.linalg.norm(np.sum(residuals, axis=0)) / residual_norm_sum)
        )
        states = np.asarray(
            [row.get(STATE_KEY, row["o6_state"]) for row in group_rows], dtype=np.float64
        ).reshape(-1, 20)
        q_correlations = {}
        for joint_index in range(20):
            corr_u = base._correlation(states[:, joint_index], residuals[:, 0])
            corr_v = base._correlation(states[:, joint_index], residuals[:, 1])
            corr_error = base._correlation(states[:, joint_index], errors)
            if corr_u is not None or corr_v is not None or corr_error is not None:
                q_correlations[str(joint_index)] = {
                    "joint": WUJI_ACTIVE_JOINTS[joint_index],
                    "residual_u": corr_u,
                    "residual_v": corr_v,
                    "error": corr_error,
                }
        return {
            "error": base.error_statistics(errors),
            "bias_uv_px": bias.astype(float).tolist(),
            "bias_norm_px": float(np.linalg.norm(bias)),
            "direction_consistency": direction_consistency,
            "q_correlation": q_correlations,
        }

    by_anchor: dict[str, list[dict[str, Any]]] = {}
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_anchor.setdefault(str(row["anchor"]), []).append(row)
        frame_key = f"{row['episode']}:message_{int(row['message_index']):06d}"
        by_frame.setdefault(frame_key, []).append(row)
    return {
        "by_anchor": {
            key: summarize(value) for key, value in sorted(by_anchor.items())
        },
        "by_frame": {
            key: summarize(value) for key, value in sorted(by_frame.items())
        },
    }


def render_frame(
    frame: dict[str, Any],
    kinematics: WujiKinematics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(frame["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame["image_path"])
    qpos = _as_qpos20(frame.get(STATE_KEY, frame.get("o6_state")))
    points21 = kinematics.points21(qpos)
    uv21, depth21 = base.project_fisheye(points21, rvec, tvec, K, D)
    overlay = image.copy()
    for a, b in HAND21_BONES:
        if depth21[a] <= 0.0 or depth21[b] <= 0.0:
            continue
        pa = tuple(np.round(uv21[a]).astype(int))
        pb = tuple(np.round(uv21[b]).astype(int))
        cv2.line(overlay, pa, pb, POINT21_COLORS_BGR[b], 2, cv2.LINE_AA)
    for index, uv in enumerate(uv21):
        if depth21[index] <= 0.0:
            continue
        point = tuple(np.round(uv).astype(int))
        cv2.circle(overlay, point, 3 if index else 4, POINT21_COLORS_BGR[index], -1, cv2.LINE_AA)

    anchors = kinematics.calibration_anchor_points(qpos)
    clicked_errors = []
    for name, gt_uv in frame.get("annotations", {}).items():
        pred_uv, pred_depth = base.project_fisheye(
            np.asarray([anchors[name]]), rvec, tvec, K, D
        )
        gt = np.asarray(gt_uv, dtype=np.float64)
        if pred_depth[0] > 0.0:
            index = ANCHOR_TO_POINT21_INDEX[name]
            cv2.circle(overlay, tuple(np.round(pred_uv[0]).astype(int)), 6, POINT21_COLORS_BGR[index], 1, cv2.LINE_AA)
            base.draw_cross(overlay, gt, (255, 255, 255))
            clicked_errors.append(float(np.linalg.norm(pred_uv[0] - gt)))

    rendered = cv2.addWeighted(image, 0.55, overlay, 0.45, 0.0)
    median_error = None if not clicked_errors else float(np.median(clicked_errors))
    text = f"{frame['split'].upper()} {frame['episode']} m={frame['message_index']}"
    if median_error is not None:
        text += f" click-med={median_error:.2f}px"
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(rendered, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return rendered, {
        "episode": frame["episode"],
        "message_index": frame["message_index"],
        "split": frame["split"],
        "clicked_median_error_px": median_error,
        "projected_points_in_front": int(np.sum(depth21 > 0.0)),
    }


def _write_wuji_alias(output_dir: Path) -> Path:
    o6_path = output_dir / "camera_from_o6_base.json"
    wuji_path = output_dir / "camera_from_wuji_base.json"
    with o6_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    calibration["hand"] = "wuji_hand"
    calibration["base_link"] = calibration.get("o6_base_link", "palm_link")
    calibration["wuji_base_link"] = calibration.get("o6_base_link", "palm_link")
    calibration["transform_convention"] = (
        "X_camera = R_camera_from_wuji_base @ X_wuji_base + "
        "t_camera_from_wuji_base"
    )
    calibration["T_camera_from_wuji_base"] = calibration["T_camera_from_o6_base"]
    calibration["rvec_camera_from_wuji_base"] = calibration["rvec_camera_from_o6_base"]
    calibration["tvec_camera_from_wuji_base_m"] = calibration["tvec_camera_from_o6_base_m"]
    diagnostics = dict(calibration.get("diagnostics", {}) or {})
    diagnostics["q_state_order"] = list(WUJI_ACTIVE_JOINTS)
    calibration["diagnostics"] = diagnostics
    base.atomic_write_json(o6_path, calibration)
    base.atomic_write_json(wuji_path, calibration)
    return wuji_path


def run_fit(args: argparse.Namespace) -> None:
    _patch_base()
    base.run_fit(args)
    wuji_path = _write_wuji_alias(Path(args.output_dir).expanduser().resolve())
    print(f"[saved] {wuji_path}")


def run_self_test(args: argparse.Namespace) -> None:
    kinematics = WujiKinematics(args.urdf)
    rng = np.random.default_rng(42)
    K = np.array([[76.0, 0.0, 111.0], [0.0, 75.0, 112.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.015, 0.004, -0.001, 0.0002], dtype=np.float64).reshape(4, 1)
    true_rvec = np.array([0.12, -0.18, 0.05], dtype=np.float64)
    true_tvec = np.array([0.02, 0.09, 0.26], dtype=np.float64)
    object_points = []
    image_points = []
    for _ in range(10):
        qpos = np.zeros(20, dtype=np.float64)
        qpos += rng.normal(0.35, 0.25, size=20)
        anchors = kinematics.calibration_anchor_points(qpos)
        for name in ANCHOR_SETS["tips_dips"]:
            object_points.append(anchors[name])
    object_array = np.stack(object_points)
    image_array, depth = base.project_fisheye(object_array, true_rvec, true_tvec, K, D)
    if np.any(depth <= 0.0):
        raise RuntimeError("Synthetic Wuji points are behind the camera")
    image_array = image_array + rng.normal(0.0, 0.05, image_array.shape)
    rvec, tvec, optimizer = base.fit_extrinsic(object_array, image_array, K, D)
    predicted, _ = base.project_fisheye(object_array, rvec, tvec, K, D)
    errors = np.linalg.norm(predicted - image_array, axis=1)
    stats = base.error_statistics(errors)
    if stats["median_px"] is None or stats["median_px"] > 0.2:
        raise RuntimeError(f"Wuji fit self-test residual too high: {stats}")
    print(json.dumps({
        "urdf": str(Path(args.urdf).expanduser().resolve()),
        "points21_shape": list(kinematics.points21(np.zeros(20)).shape),
        "fit": stats,
        "optimizer": optimizer,
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="select frames and open the click UI")
    annotate.add_argument("--episodes-root", default=DEFAULT_EPISODES_ROOT)
    annotate.add_argument("--episodes", required=True, help="e.g. 1 or 1-3")
    annotate.add_argument("--urdf", default=None)
    annotate.add_argument("--camera-contract", default=DEFAULT_CAMERA_CONTRACT)
    annotate.add_argument("--output", required=True)
    annotate.add_argument("--state-source", default=STATE_KEY, choices=[STATE_KEY, "wuji_command"])
    annotate.add_argument(
        "--state-time-alignment",
        choices=["rgb", "message"],
        default="rgb",
        help="interpolate q samples to each RGB timestamp (recommended) or use message q directly",
    )
    annotate.add_argument("--num-frames", type=int, default=32)
    annotate.add_argument("--holdout-every", type=int, default=4)
    annotate.add_argument("--anchor-set", choices=sorted(ANCHOR_SETS), default="tips")
    annotate.add_argument("--max-align-ms", type=float, default=25.0)
    annotate.add_argument("--include-repeated", action="store_true")
    annotate.add_argument("--display-scale", type=int, default=3)
    annotate.add_argument("--min-points", type=int, default=4)
    annotate.add_argument("--overwrite", action="store_true")
    annotate.set_defaults(func=run_annotate)

    fit = subparsers.add_parser("fit", help="fit camera<-Wuji_base and render held-out frames")
    fit.add_argument("--annotations", required=True)
    fit.add_argument("--camera-contract", default=None)
    fit.add_argument("--intrinsics", default=None)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--urdf", default=None)
    fit.add_argument("--contact-sheet-frames", type=int, default=24)
    fit.set_defaults(func=run_fit)

    self_test = subparsers.add_parser("self-test", help="synthetic FK/projection/fitting test")
    self_test.add_argument("--urdf", default=DEFAULT_URDF)
    self_test.set_defaults(func=run_self_test)

    contract_check = subparsers.add_parser(
        "contract-check", help="validate and print the active camera contract"
    )
    contract_check.add_argument("--camera-contract", default=DEFAULT_CAMERA_CONTRACT)
    contract_check.set_defaults(func=base.run_contract_check)
    return parser


def main() -> None:
    _patch_base()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
