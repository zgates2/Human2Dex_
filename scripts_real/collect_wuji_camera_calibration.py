#!/usr/bin/env python3
"""Collect policy RGB plus measured Wuji Hand qpos for camera extrinsic fitting."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collect_o6_camera_calibration import (  # noqa: E402
    atomic_write_pickle,
    build_camera_open_config,
    file_sha256,
    interpolate_state,
    load_camera_runtime,
    load_contract,
    load_yaml,
    next_episode_dir,
)


DEFAULT_CONTRACT = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "wuji_camera_calibration/mvs_DA9057802_wuji_mount_v1/"
    "camera_contract_mvs_DA9057802_wuji_mount_v1.yaml"
)
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "example/eval_robots_config.yaml"
DEFAULT_GUIDED_POSES = REPO_ROOT / "scripts_real/wuji_camera_calibration_poses.yaml"
EPISODE_RE = re.compile(r"^episode_(\d+)$")


def normalize_wuji_qpos(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float64).reshape(-1)
    if state.shape != (20,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Wuji qpos must be finite shape (20,), got {state}")
    return state


def qpos_distance(state: np.ndarray, messages: list[dict[str, Any]]) -> float | None:
    if not messages:
        return None
    current = np.asarray(state, dtype=np.float64).reshape(20) / np.pi
    previous = np.stack([
        np.asarray(message["wuji_qpos"], dtype=np.float64).reshape(20) / np.pi
        for message in messages
    ])
    return float(np.min(np.linalg.norm(previous - current[None], axis=1)))


def load_guided_poses(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = load_yaml(path)
    raw_poses = raw.get("poses")
    if not isinstance(raw_poses, list) or not raw_poses:
        raise ValueError(f"guided pose file has no poses: {path}")
    poses = []
    names = set()
    for index, item in enumerate(raw_poses):
        if not isinstance(item, dict):
            raise ValueError(f"pose {index} must be a mapping")
        name = str(item.get("name", f"pose_{index + 1:02d}"))
        if name in names:
            raise ValueError(f"duplicate guided pose name {name!r}")
        names.add(name)
        qpos = normalize_wuji_qpos(item.get("q"))
        poses.append({"index": index, "name": name, "q": qpos.astype(float).tolist()})
    return {"path": path, "sha256": file_sha256(path), "poses": poses, "raw": raw}


def load_wuji_runtime_config(path: Path, override_root: str | None) -> tuple[Path, dict[str, Any]]:
    path = path.expanduser().resolve()
    raw = load_yaml(path)
    cfg = dict(raw.get("wuji_hand", {}) or {})
    if override_root:
        cfg["wuji_demo_root"] = override_root
    cfg.setdefault("wuji_demo_root", "/home/zjc/wuji_demo")
    return path, cfg


def import_wuji_driver(wuji_cfg: dict[str, Any]):
    wuji_root = Path(
        wuji_cfg.get("wuji_demo_root", wuji_cfg.get("wuji_root", "/home/zjc/wuji_demo"))
    ).expanduser().resolve()
    if str(wuji_root) not in sys.path:
        sys.path.insert(0, str(wuji_root))
    from wuji_pico_hand.hand_driver import WujiHandDriver

    return WujiHandDriver


def save_snapshot(
    episode_dir: Path,
    payload: dict[str, Any],
    image_rgb: np.ndarray,
    frame_info: dict[str, Any],
    qpos: np.ndarray,
    qpos_before: np.ndarray,
    before_ns: int,
    qpos_after: np.ndarray,
    after_ns: int,
    interpolation_alpha: float,
    guided_pose: dict[str, Any] | None,
) -> None:
    index = len(payload["messages"])
    episode_name = episode_dir.name
    image_relative = Path("images") / f"{episode_name}_rgb_{index:06d}.jpg"
    image_path = episode_dir / image_relative
    bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 97]):
        raise IOError(f"cv2.imwrite failed: {image_path}")
    capture_ns = int(frame_info["timestamp_ns"])
    message = {
        "timestamp": capture_ns * 1e-9,
        "sampleClockNs": capture_ns,
        "rgbImage": str(image_relative),
        "rgbFrameId": int(frame_info["frame_id"]),
        "rgbCaptureTimestamp": capture_ns * 1e-9,
        "rgbCaptureNs": capture_ns,
        "rgbDeviceTimestampTicks": int(frame_info["device_timestamp_ticks"]),
        "rgbHostTimestampMs": int(frame_info["host_timestamp_ms"]),
        "rgbFrameRepeated": False,
        "rgbAlignResidualNs": 0,
        "wuji_qpos": qpos.astype(float).tolist(),
        "wujiQposBefore": qpos_before.astype(float).tolist(),
        "wujiQposBeforeTimestampNs": int(before_ns),
        "wujiQposAfter": qpos_after.astype(float).tolist(),
        "wujiQposAfterTimestampNs": int(after_ns),
        "wujiQposInterpolationAlpha": float(interpolation_alpha),
        "wuji_command": None,
    }
    if guided_pose is not None:
        message["guidedPose"] = {
            "index": int(guided_pose["index"]),
            "name": str(guided_pose["name"]),
            "command": list(guided_pose["q"]),
        }
    payload["messages"].append(message)
    payload["metadata"]["savedSnapshots"] = len(payload["messages"])
    atomic_write_pickle(episode_dir / f"{episode_name}.pkl", payload)


def preview_image(
    image_rgb: np.ndarray,
    mount_id: str,
    qpos: np.ndarray,
    saved: int,
    target: int,
    nearest_distance: float | None,
    notice: str,
    guided_pose: dict[str, Any] | None,
    guided_enabled: bool,
) -> np.ndarray:
    view = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    view = cv2.resize(view, (672, 672), interpolation=cv2.INTER_NEAREST)
    panel_height = 154
    canvas = np.zeros((view.shape[0] + panel_height, view.shape[1], 3), dtype=np.uint8)
    canvas[: view.shape[0]] = view
    pose_text = "read-only"
    if guided_enabled:
        pose_text = "N: next guided pose"
        if guided_pose is not None:
            pose_text += f" | active={guided_pose['index'] + 1}:{guided_pose['name']}"
    q = np.round(np.asarray(qpos, dtype=np.float64).reshape(5, 4), 2)
    lines = [
        f"mount={mount_id}",
        f"saved={saved}/{target}  q mean={float(np.mean(q)):+.2f} range=[{float(np.min(q)):+.2f},{float(np.max(q)):+.2f}]",
        f"thumb={q[0].tolist()} index={q[1].tolist()}",
        f"S/Space save | F force duplicate | Q quit | {pose_text}",
        f"nearest_pose_distance={nearest_distance}  {notice}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (10, view.shape[0] + 25 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230) if index != 4 else (70, 220, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def run_capture(args: argparse.Namespace) -> None:
    contract = load_contract(Path(args.camera_contract))
    if args.confirm_mount_id != contract["mount_id"]:
        raise ValueError(
            f"--confirm-mount-id must exactly match {contract['mount_id']!r}"
        )
    runtime = load_camera_runtime(Path(args.robot_config), contract)
    camera_config = build_camera_open_config(contract, runtime)
    _, wuji_cfg = load_wuji_runtime_config(Path(args.robot_config), args.wuji_demo_root)
    guided = None
    if args.guided_poses:
        if args.confirm_guided_motion != "operator_controls_each_pose":
            raise ValueError(
                "Guided Wuji motion requires --confirm-guided-motion "
                "operator_controls_each_pose"
            )
        guided = load_guided_poses(Path(args.guided_poses))

    from umi.real_world.mvs_camera import open_kwargs_from_collection_config
    from umi.real_world._mvs_cpp import open_mvs_cpp_device

    WujiHandDriver = import_wuji_driver(wuji_cfg)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else Path(contract["raw"]["artifacts"]["annotation_json"]).parent
        / "calibration_episodes"
    )
    episode_dir = next_episode_dir(output_root)
    episode_name = episode_dir.name
    pkl_path = episode_dir / f"{episode_name}.pkl"
    payload = {
        "metadata": {
            "formatVersion": 1,
            "taskName": "wuji_camera_extrinsic_calibration",
            "handBackend": "wuji_hand",
            "cameraMountId": contract["mount_id"],
            "cameraContract": {
                "path": str(contract["path"]),
                "sha256": contract["sha256"],
            },
            "cameraSerial": contract["serial"],
            "policyImage": contract["raw"]["policy_image"],
            "robotConfig": {
                "path": str(Path(args.robot_config).expanduser().resolve()),
                "sha256": file_sha256(Path(args.robot_config).expanduser().resolve()),
            },
            "urdf": {
                "path": str(contract["urdf_path"]),
                "sha256": contract["urdf_sha256"],
            },
            "collector": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
                "hostname": socket.gethostname(),
                "python": sys.executable,
            },
            "captureMode": "manual_stationary_snapshot",
            "stateAlignment": "bracketed_Wuji_qpos_interpolated_to_MVS_host_timestamp",
            "frankaConnected": False,
            "policyLoaded": False,
            "wujiMotionCommandsIssued": False,
            "guidedPoses": (
                None
                if guided is None
                else {
                    "path": str(guided["path"]),
                    "sha256": guided["sha256"],
                    "count": len(guided["poses"]),
                    "operatorStepRequired": True,
                }
            ),
            "targetSnapshots": int(args.target_snapshots),
            "savedSnapshots": 0,
            "createdWallTime": time.time(),
        },
        "messages": [],
    }
    atomic_write_pickle(pkl_path, payload)

    device = None
    driver = None
    notice = "wait until hand is stationary before saving"
    last_frame_id = None
    guided_pose_index = -1
    active_guided_pose = None
    print(f"[Output] {pkl_path}")
    if guided is None:
        print("[Safety] read-only Wuji mode; no Franka connection and no Wuji motion commands")
    else:
        print(
            "[Safety] no Franka connection; guided Wuji mode moves only after each "
            "operator N keypress"
        )
    try:
        driver = WujiHandDriver(
            serial_number=wuji_cfg.get("serial_number"),
            dry_run=False,
            timeout=wuji_cfg.get("timeout"),
        )
        if bool(wuji_cfg.get("refresh_limits", True)):
            limits = driver.refresh_limits()
            if limits is not None:
                print("[Wuji] lower:", np.round(limits.lower, 4).tolist())
                print("[Wuji] upper:", np.round(limits.upper, 4).tolist())
        driver.enable()
        warmup_deadline = time.monotonic() + float(args.state_warmup_sec)
        qpos = normalize_wuji_qpos(driver.read_positions())
        while time.monotonic() < warmup_deadline:
            qpos = normalize_wuji_qpos(driver.read_positions())
            time.sleep(0.05)

        open_kwargs = open_kwargs_from_collection_config(camera_config)
        device = open_mvs_cpp_device(**open_kwargs)
        print(f"[Camera] exact policy path: {open_kwargs}")
        cv2.namedWindow("Wuji camera calibration snapshots", cv2.WINDOW_NORMAL)
        while True:
            qpos_before = normalize_wuji_qpos(driver.read_positions())
            before_ns = time.monotonic_ns()
            frame = device.read_frame(timeout_ms=int(args.camera_timeout_ms))
            try:
                image_rgb = frame.copy_array()
                frame_info = {
                    "frame_id": int(frame.frame_id),
                    "timestamp_ns": int(frame.timestamp_ns),
                    "device_timestamp_ticks": int(frame.device_timestamp_ticks),
                    "host_timestamp_ms": int(frame.host_timestamp_ms),
                }
            finally:
                frame.release()
            after_ns = time.monotonic_ns()
            qpos_after = normalize_wuji_qpos(driver.read_positions())
            qpos, alpha = interpolate_state(
                qpos_before,
                before_ns,
                qpos_after,
                after_ns,
                frame_info["timestamp_ns"],
            )
            if image_rgb.shape[:2][::-1] != contract["output_size"]:
                raise RuntimeError(
                    f"MVS returned {image_rgb.shape[:2][::-1]}, expected {contract['output_size']}"
                )
            nearest = qpos_distance(qpos, payload["messages"])
            cv2.imshow(
                "Wuji camera calibration snapshots",
                preview_image(
                    image_rgb,
                    contract["mount_id"],
                    qpos,
                    len(payload["messages"]),
                    int(args.target_snapshots),
                    nearest,
                    notice,
                    active_guided_pose,
                    guided is not None,
                ),
            )
            key = cv2.waitKey(1) & 0xFF
            should_save = key in (ord("s"), ord(" "))
            force_save = key == ord("f")
            if key in (27, ord("q")):
                break
            if key == ord("n"):
                if guided is None:
                    notice = "read-only mode: restart with --guided-poses to enable N"
                    continue
                next_index = guided_pose_index + 1
                if next_index >= len(guided["poses"]):
                    notice = "all guided poses have been issued"
                    continue
                active_guided_pose = guided["poses"][next_index]
                sent = driver.send_positions(np.asarray(active_guided_pose["q"], dtype=np.float64))
                guided_pose_index = next_index
                payload["metadata"]["wujiMotionCommandsIssued"] = True
                payload["metadata"]["lastGuidedPoseIssued"] = {
                    "index": int(active_guided_pose["index"]),
                    "name": str(active_guided_pose["name"]),
                    "command": np.asarray(sent, dtype=np.float64).astype(float).tolist(),
                    "wallTime": time.time(),
                }
                atomic_write_pickle(pkl_path, payload)
                notice = (
                    f"issued {next_index + 1}/{len(guided['poses'])} "
                    f"{active_guided_pose['name']}; wait until stationary, then S"
                )
                print(
                    f"[Guided pose] {next_index + 1:02d}/{len(guided['poses'])} "
                    f"{active_guided_pose['name']} q={np.round(sent, 3).reshape(5, 4).tolist()}"
                )
                continue
            if should_save or force_save:
                if guided is not None and active_guided_pose is None:
                    notice = "press N to issue the first guided pose before saving"
                    continue
                if (
                    not force_save
                    and nearest is not None
                    and nearest < float(args.min_pose_distance)
                ):
                    notice = (
                        f"pose too similar ({nearest:.3f} < {args.min_pose_distance:.3f}); "
                        "change q or press F"
                    )
                    continue
                if last_frame_id == frame_info["frame_id"]:
                    notice = "same camera frame; wait for a new frame"
                    continue
                save_snapshot(
                    episode_dir,
                    payload,
                    image_rgb,
                    frame_info,
                    qpos,
                    qpos_before,
                    before_ns,
                    qpos_after,
                    after_ns,
                    alpha,
                    active_guided_pose,
                )
                last_frame_id = frame_info["frame_id"]
                notice = f"saved snapshot {len(payload['messages'])}"
                print(
                    f"[Saved] {len(payload['messages']):02d} frame={frame_info['frame_id']} "
                    f"q_mean={float(np.mean(qpos)):+.3f}"
                )
                if 0 < int(args.target_snapshots) <= len(payload["messages"]):
                    print("[Done] target snapshot count reached")
                    break
    finally:
        payload["metadata"]["completedWallTime"] = time.time()
        payload["metadata"]["savedSnapshots"] = len(payload["messages"])
        atomic_write_pickle(pkl_path, payload)
        cv2.destroyAllWindows()
        if device is not None:
            device.close()
        if driver is not None:
            driver.close()
    print(f"[Saved episode] {pkl_path} snapshots={len(payload['messages'])}")


def run_list_cameras(_args: argparse.Namespace) -> None:
    from umi.real_world._mvs_cpp import list_mvs_device_serials

    print(json.dumps({"serials": list(list_mvs_device_serials())}, indent=2))


def run_self_test(args: argparse.Namespace) -> None:
    contract = load_contract(Path(args.camera_contract))
    runtime = load_camera_runtime(Path(args.robot_config), contract)
    config = build_camera_open_config(contract, runtime)
    guided = load_guided_poses(Path(args.guided_poses))
    before = np.linspace(0.0, 0.5, 20, dtype=np.float64)
    after = before + 0.1
    interpolated, alpha = interpolate_state(before, 100, after, 200, 150)
    expected = (before + after) / 2.0
    if not np.allclose(interpolated, expected) or abs(alpha - 0.5) > 1e-12:
        raise RuntimeError("state interpolation self-test failed")
    from calibrate_wuji_camera_extrinsic import collect_candidates

    with tempfile.TemporaryDirectory(prefix="wuji-camera-calibration-") as temp_dir:
        root = Path(temp_dir)
        episode_dir = root / "episode_0001"
        (episode_dir / "images").mkdir(parents=True)
        payload = {
            "metadata": {"handBackend": "wuji_hand", "savedSnapshots": 0},
            "messages": [],
        }
        for index in range(3):
            state = np.linspace(0.0, 0.6 + index * 0.1, 20, dtype=np.float64)
            frame_ns = 1_000_000_000 + index * 40_000_000
            save_snapshot(
                episode_dir,
                payload,
                np.zeros((224, 224, 3), dtype=np.uint8),
                {
                    "frame_id": index + 1,
                    "timestamp_ns": frame_ns,
                    "device_timestamp_ticks": index + 10,
                    "host_timestamp_ms": frame_ns // 1_000_000,
                },
                state,
                state,
                frame_ns - 1_000_000,
                state,
                frame_ns + 1_000_000,
                0.5,
                None,
            )
        candidates = collect_candidates(
            episodes_root=root,
            episode_numbers=[1],
            state_source="wuji_qpos",
            state_time_alignment="rgb",
            max_align_ms=25.0,
            include_repeated=False,
        )
        if len(candidates) != 3:
            raise RuntimeError(
                f"calibrator compatibility self-test expected 3 candidates, got {len(candidates)}"
            )
    print(json.dumps({
        "camera_mount_id": contract["mount_id"],
        "contract_sha256": contract["sha256"],
        "camera_serial": contract["serial"],
        "input_size": list(contract["input_size"]),
        "crop_rect": list(contract["crop_rect"]),
        "output_size": list(contract["output_size"]),
        "rotate_180": contract["rotate_180"],
        "generated_open_config": {
            "crop": [config.crop_x, config.crop_y, config.crop_width, config.crop_height],
            "output": [config.width, config.height],
            "fps": config.fps,
        },
        "guided_pose_file": {
            "path": str(guided["path"]),
            "sha256": guided["sha256"],
            "count": len(guided["poses"]),
            "first": guided["poses"][0],
            "last": guided["poses"][-1],
        },
        "franka_imported": False,
        "policy_imported": False,
        "guided_motion_requires_explicit_file_confirmation_and_each_N_keypress": True,
        "calibrator_compatible_candidates": len(candidates),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="operator-triggered RGB + Wuji qpos snapshots")
    capture.add_argument("--camera-contract", default=DEFAULT_CONTRACT)
    capture.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    capture.add_argument("--confirm-mount-id", required=True)
    capture.add_argument("--output-root", default=None)
    capture.add_argument("--wuji-demo-root", default=None)
    capture.add_argument("--target-snapshots", type=int, default=20)
    capture.add_argument("--min-pose-distance", type=float, default=0.05)
    capture.add_argument("--state-warmup-sec", type=float, default=1.0)
    capture.add_argument("--camera-timeout-ms", type=int, default=1000)
    capture.add_argument(
        "--guided-poses",
        default=None,
        help="optional YAML; each command is sent only after an operator N keypress",
    )
    capture.add_argument("--confirm-guided-motion", default=None)
    capture.set_defaults(func=run_capture)

    list_cameras = subparsers.add_parser("list-cameras", help="list MVS serials and exit")
    list_cameras.set_defaults(func=run_list_cameras)

    self_test = subparsers.add_parser("self-test", help="contract/config/state-alignment test without hardware")
    self_test.add_argument("--camera-contract", default=DEFAULT_CONTRACT)
    self_test.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    self_test.add_argument("--guided-poses", default=str(DEFAULT_GUIDED_POSES))
    self_test.set_defaults(func=run_self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
