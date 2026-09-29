#!/usr/bin/env python3
"""Collect policy RGB plus measured Linker O6 state for camera extrinsic fitting.

This program deliberately does not import a policy or connect to Franka.  Its
default mode is O6-read-only.  An optional guided-pose file can be enabled only
with an explicit confirmation token; even then the operator must press N for
every individual hand command.  Snapshots are saved only on S/Space.
"""

from __future__ import annotations

import argparse
import hashlib
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
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONTRACT = (
    "/home/zjc/Desktop/human2dex/converted_data/params/"
    "camera_contract_mvs_DA9057801_o6_mount_v1.yaml"
)
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "example/eval_robots_config.yaml"
DEFAULT_GUIDED_POSES = REPO_ROOT / "scripts_real/o6_camera_calibration_poses.yaml"
EPISODE_RE = re.compile(r"^episode_(\d+)$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_contract(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = load_yaml(path)
    mount_id = str(raw.get("mount", {}).get("camera_mount_id", ""))
    if not mount_id or raw.get("contract_id") != mount_id:
        raise ValueError("contract_id must equal mount.camera_mount_id")
    policy = raw.get("policy_image", {})
    input_size = tuple(int(value) for value in policy.get("input_size", []))
    output_size = tuple(int(value) for value in policy.get("resize", []))
    crop_rect = tuple(int(value) for value in policy.get("crop_rect_xywh", []))
    if len(input_size) != 2 or len(output_size) != 2 or len(crop_rect) != 4:
        raise ValueError("camera contract has an incomplete policy image chain")
    camera = raw.get("camera", {})
    serial = str(camera.get("serial", ""))
    if not serial:
        raise ValueError("camera contract is missing camera.serial")
    urdf_path = Path(raw.get("kinematics", {}).get("urdf_path", "")).expanduser().resolve()
    urdf_sha = str(raw.get("kinematics", {}).get("urdf_sha256", ""))
    if not urdf_path.is_file() or (urdf_sha and file_sha256(urdf_path) != urdf_sha):
        raise ValueError("camera contract URDF is missing or changed")
    return {
        "path": path,
        "sha256": file_sha256(path),
        "raw": raw,
        "mount_id": mount_id,
        "serial": serial,
        "input_size": input_size,
        "output_size": output_size,
        "crop_rect": crop_rect,
        "rotate_180": bool(policy.get("rotate_180", False)),
        "urdf_path": urdf_path,
        "urdf_sha256": file_sha256(urdf_path),
    }


def load_camera_runtime(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = load_yaml(path)
    cameras = raw.get("cameras", {})
    if str(cameras.get("backend", "")).lower() != "mvs":
        raise ValueError("runtime camera backend must be mvs")
    serials = [str(value) for value in cameras.get("serials", [])]
    if contract["serial"] not in serials:
        raise ValueError(
            f"contract camera {contract['serial']} is absent from runtime serials {serials}"
        )
    resolution = tuple(int(value) for value in cameras.get("resolution", []))
    input_size = tuple(int(value) for value in cameras.get("input_res", []))
    if resolution != contract["output_size"] or input_size != contract["input_size"]:
        raise ValueError(
            f"runtime image chain {input_size}->{resolution} differs from contract "
            f"{contract['input_size']}->{contract['output_size']}"
        )
    if bool(cameras.get("rotate_180", True)) != contract["rotate_180"]:
        raise ValueError("runtime rotate_180 differs from contract")
    return {"path": path, "sha256": file_sha256(path), "raw": raw, "cameras": cameras}


def build_camera_open_config(contract: dict[str, Any], runtime: dict[str, Any]):
    from umi.real_world.mvs_camera import (
        compute_center_crop_rect,
        create_camera_configs_from_serials,
    )

    expected_crop = compute_center_crop_rect(
        contract["input_size"], contract["output_size"]
    )
    if tuple(expected_crop) != contract["crop_rect"]:
        raise ValueError(
            f"contract crop {contract['crop_rect']} differs from runtime center crop {expected_crop}"
        )
    cameras = runtime["cameras"]
    configs = list(create_camera_configs_from_serials(
        serials=[contract["serial"]],
        fps=int(cameras.get("capture_fps", 30)),
        output_res=contract["output_size"],
        input_res=contract["input_size"],
        exposure_time_us=float(cameras.get("exposure_time_us", 15000.0)),
        gain_auto=str(cameras.get("gain_auto", "continuous")),
        gain_db=cameras.get("gain_db"),
        prefer_sensor_roi=bool(cameras.get("prefer_sensor_roi", False)),
    ))
    config = configs[0]
    config.rotate_180 = contract["rotate_180"]
    config.balance_white_auto = str(cameras.get("balance_white_auto", "continuous"))
    config.validate()
    actual_crop = (config.crop_x, config.crop_y, config.crop_width, config.crop_height)
    if actual_crop != contract["crop_rect"]:
        raise ValueError(f"generated MVS crop {actual_crop} differs from contract")
    return config


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
        q = normalize_o6_state(item.get("q"))
        poses.append({"index": index, "name": name, "q": np.rint(q).astype(int).tolist()})
    return {"path": path, "sha256": file_sha256(path), "poses": poses, "raw": raw}


def next_episode_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    used = set()
    for path in root.iterdir():
        match = EPISODE_RE.fullmatch(path.name)
        if match:
            used.add(int(match.group(1)))
    index = 1
    while index in used:
        index += 1
    result = root / f"episode_{index:04d}"
    result.mkdir(parents=False, exist_ok=False)
    (result / "images").mkdir()
    return result


def atomic_write_pickle(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def interpolate_state(
    before: np.ndarray,
    before_ns: int,
    after: np.ndarray,
    after_ns: int,
    target_ns: int,
) -> tuple[np.ndarray, float]:
    if after_ns <= before_ns:
        return after.astype(np.float64), 1.0
    alpha = float(np.clip((target_ns - before_ns) / (after_ns - before_ns), 0.0, 1.0))
    return (1.0 - alpha) * before + alpha * after, alpha


def normalize_o6_state(value: Any) -> np.ndarray:
    state = np.asarray(value, dtype=np.float64).reshape(-1)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError(f"O6 state must be finite shape (6,), got {state}")
    if np.any(state < 0) or np.any(state > 255):
        raise ValueError(f"O6 state outside [0,255]: {state.tolist()}")
    return state


def pose_distance(state: np.ndarray, messages: list[dict[str, Any]]) -> float | None:
    if not messages:
        return None
    current = np.asarray(state, dtype=np.float64) / 250.0
    previous = np.stack([
        np.asarray(message["o6_measured_state"], dtype=np.float64) / 250.0
        for message in messages
    ])
    return float(np.min(np.linalg.norm(previous - current[None], axis=1)))


def save_snapshot(
    episode_dir: Path,
    payload: dict[str, Any],
    image_rgb: np.ndarray,
    frame_info: dict[str, Any],
    state: np.ndarray,
    state_before: np.ndarray,
    before_ns: int,
    state_after: np.ndarray,
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
        "o6_measured_state": state.astype(float).tolist(),
        "o6StateBefore": state_before.astype(float).tolist(),
        "o6StateBeforeTimestampNs": int(before_ns),
        "o6StateAfter": state_after.astype(float).tolist(),
        "o6StateAfterTimestampNs": int(after_ns),
        "o6StateInterpolationAlpha": float(interpolation_alpha),
        "o6Command": None,
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
    state: np.ndarray,
    saved: int,
    target: int,
    nearest_distance: float | None,
    notice: str,
    guided_pose: dict[str, Any] | None,
    guided_enabled: bool,
) -> np.ndarray:
    view = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    view = cv2.resize(view, (672, 672), interpolation=cv2.INTER_NEAREST)
    panel_height = 128
    canvas = np.zeros((view.shape[0] + panel_height, view.shape[1], 3), dtype=np.uint8)
    canvas[: view.shape[0]] = view
    pose_text = "read-only"
    if guided_enabled:
        pose_text = "N: next guided pose"
        if guided_pose is not None:
            pose_text += f" | active={guided_pose['index'] + 1}:{guided_pose['name']}"
    lines = [
        f"mount={mount_id}",
        f"saved={saved}/{target}  q={np.rint(state).astype(int).tolist()}",
        f"S/Space save | F force duplicate | Q quit | {pose_text}",
        f"nearest_pose_distance={nearest_distance}  {notice}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (10, view.shape[0] + 25 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230) if index != 3 else (70, 220, 255),
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
    guided = None
    if args.guided_poses:
        if args.confirm_guided_motion != "operator_controls_each_pose":
            raise ValueError(
                "Guided O6 motion requires --confirm-guided-motion "
                "operator_controls_each_pose"
            )
        guided = load_guided_poses(Path(args.guided_poses))
        if not 0 <= int(args.guided_speed) <= 255:
            raise ValueError("--guided-speed must be in [0,255]")
    from umi.real_world.mvs_camera import open_kwargs_from_collection_config
    from umi.real_world._mvs_cpp import open_mvs_cpp_device

    dexumi_root = Path(args.dexumi_root).expanduser().resolve()
    o6_dir = dexumi_root / "o6_right_hand"
    if str(o6_dir) not in sys.path:
        sys.path.insert(0, str(o6_dir))
    from controller import O6RightHand

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
            "taskName": "o6_camera_extrinsic_calibration",
            "handBackend": "linker_o6",
            "cameraMountId": contract["mount_id"],
            "cameraContract": {
                "path": str(contract["path"]),
                "sha256": contract["sha256"],
            },
            "cameraSerial": contract["serial"],
            "policyImage": contract["raw"]["policy_image"],
            "robotConfig": {
                "path": str(runtime["path"]),
                "sha256": runtime["sha256"],
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
            "stateAlignment": "bracketed_O6_read_interpolated_to_MVS_host_timestamp",
            "frankaConnected": False,
            "policyLoaded": False,
            "o6MotionCommandsIssued": False,
            "guidedPoses": (
                None
                if guided is None
                else {
                    "path": str(guided["path"]),
                    "sha256": guided["sha256"],
                    "count": len(guided["poses"]),
                    "operatorStepRequired": True,
                    "speed": int(args.guided_speed),
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
    hand = None
    notice = "wait until hand is stationary before saving"
    last_frame_id = None
    guided_pose_index = -1
    active_guided_pose = None
    guided_speed_set = False
    print(f"[Output] {pkl_path}")
    if guided is None:
        print("[Safety] read-only O6 mode; no Franka connection and no O6 motion commands")
    else:
        print(
            "[Safety] no Franka connection; guided O6 mode moves only after each "
            "operator N keypress"
        )
    try:
        hand = O6RightHand(can_channel=args.can_channel, bitrate=args.bitrate)
        warmup_deadline = time.monotonic() + float(args.state_warmup_sec)
        while time.monotonic() < warmup_deadline:
            state = normalize_o6_state(hand.get_state())
            if not np.allclose(state, 0.0):
                break
            time.sleep(0.05)
        if np.allclose(state, 0.0):
            raise RuntimeError(
                "O6 readback stayed all-zero during warmup; start from a non-fully-closed pose "
                "so placeholder cache data cannot be mistaken for q_meas"
            )

        open_kwargs = open_kwargs_from_collection_config(camera_config)
        device = open_mvs_cpp_device(**open_kwargs)
        print(f"[Camera] exact policy path: {open_kwargs}")
        cv2.namedWindow("O6 camera calibration snapshots", cv2.WINDOW_NORMAL)
        while True:
            state_before = normalize_o6_state(hand.get_state())
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
            state_after = normalize_o6_state(hand.get_state())
            state, alpha = interpolate_state(
                state_before,
                before_ns,
                state_after,
                after_ns,
                frame_info["timestamp_ns"],
            )
            if image_rgb.shape[:2][::-1] != contract["output_size"]:
                raise RuntimeError(
                    f"MVS returned {image_rgb.shape[:2][::-1]}, expected {contract['output_size']}"
                )
            nearest = pose_distance(state, payload["messages"])
            cv2.imshow(
                "O6 camera calibration snapshots",
                preview_image(
                    image_rgb,
                    contract["mount_id"],
                    state,
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
                if not guided_speed_set:
                    hand.set_speed(int(args.guided_speed))
                    guided_speed_set = True
                active_guided_pose = guided["poses"][next_index]
                hand.move(active_guided_pose["q"])
                guided_pose_index = next_index
                payload["metadata"]["o6MotionCommandsIssued"] = True
                payload["metadata"]["lastGuidedPoseIssued"] = {
                    "index": int(active_guided_pose["index"]),
                    "name": str(active_guided_pose["name"]),
                    "command": list(active_guided_pose["q"]),
                    "wallTime": time.time(),
                }
                atomic_write_pickle(pkl_path, payload)
                notice = (
                    f"issued {next_index + 1}/{len(guided['poses'])} "
                    f"{active_guided_pose['name']}; wait until stationary, then S"
                )
                print(
                    f"[Guided pose] {next_index + 1:02d}/{len(guided['poses'])} "
                    f"{active_guided_pose['name']} q={active_guided_pose['q']}"
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
                    state,
                    state_before,
                    before_ns,
                    state_after,
                    after_ns,
                    alpha,
                    active_guided_pose,
                )
                last_frame_id = frame_info["frame_id"]
                notice = f"saved snapshot {len(payload['messages'])}"
                print(
                    f"[Saved] {len(payload['messages']):02d} frame={frame_info['frame_id']} "
                    f"q={np.round(state, 2).tolist()}"
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
        if hand is not None:
            hand.close()
    print(f"[Saved episode] {pkl_path} snapshots={len(payload['messages'])}")


def run_list_cameras(_args: argparse.Namespace) -> None:
    from umi.real_world._mvs_cpp import list_mvs_device_serials

    print(json.dumps({"serials": list(list_mvs_device_serials())}, indent=2))


def run_self_test(args: argparse.Namespace) -> None:
    contract = load_contract(Path(args.camera_contract))
    runtime = load_camera_runtime(Path(args.robot_config), contract)
    config = build_camera_open_config(contract, runtime)
    guided = load_guided_poses(Path(args.guided_poses))
    before = np.array([250, 240, 200, 150, 100, 50], dtype=np.float64)
    after = before - 10.0
    interpolated, alpha = interpolate_state(before, 100, after, 200, 150)
    expected = (before + after) / 2.0
    if not np.allclose(interpolated, expected) or abs(alpha - 0.5) > 1e-12:
        raise RuntimeError("state interpolation self-test failed")
    from calibrate_o6_camera_extrinsic import collect_candidates

    with tempfile.TemporaryDirectory(prefix="o6-camera-calibration-") as temp_dir:
        root = Path(temp_dir)
        episode_dir = root / "episode_0001"
        (episode_dir / "images").mkdir(parents=True)
        payload = {
            "metadata": {"handBackend": "linker_o6", "savedSnapshots": 0},
            "messages": [],
        }
        for index in range(3):
            state = np.asarray([240, 220, 200 - 20 * index, 180, 160, 140], dtype=np.float64)
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
            state_source="o6_measured_state",
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

    capture = subparsers.add_parser("capture", help="operator-triggered RGB + q_meas snapshots")
    capture.add_argument("--camera-contract", default=DEFAULT_CONTRACT)
    capture.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG))
    capture.add_argument("--confirm-mount-id", required=True)
    capture.add_argument("--output-root", default=None)
    capture.add_argument("--dexumi-root", default="/home/zjc/Desktop/human2dex")
    capture.add_argument("--can-channel", default="can0")
    capture.add_argument("--bitrate", type=int, default=1_000_000)
    capture.add_argument("--target-snapshots", type=int, default=20)
    capture.add_argument("--min-pose-distance", type=float, default=0.08)
    capture.add_argument("--state-warmup-sec", type=float, default=3.0)
    capture.add_argument("--camera-timeout-ms", type=int, default=1000)
    capture.add_argument(
        "--guided-poses",
        default=None,
        help="optional YAML; each command is sent only after an operator N keypress",
    )
    capture.add_argument("--confirm-guided-motion", default=None)
    capture.add_argument("--guided-speed", type=int, default=80)
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
