#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DexUMI inference entrypoint for Data-Scaling-Laws-Infer checkpoints.

This script keeps the original eval_real.py checkpoint/policy loading path, but
replaces the UMI robot environment with DexUMI observations:
  - camera_0: RGB image sequence
  - robot_eef_pose / trajectoryPose: wrist pose sequence, if present in cfg
  - hand_command: Linker O6 0..255 command sequence, if present in cfg
  - pts21_mano / pts21_mano_flat and timing fields, if present in cfg

For validation, run it offline on a converted replay_buffer.zarr or directly on
a DexUMI PKL bundle before enabling real CAN execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import dill
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


FIELD_ALIASES = {
    "camera_0": ["camera_0", "camera0_rgb", "rgb", "rgbImage"],
    "camera0_rgb": ["camera0_rgb", "camera_0", "rgb", "rgbImage"],
    "robot_eef_pose": ["robot_eef_pose", "trajectoryPose", "trajectory_pose"],
    "trajectoryPose": ["trajectoryPose", "robot_eef_pose", "trajectory_pose"],
    "trajectory_pose": ["trajectory_pose", "trajectoryPose", "robot_eef_pose"],
    "hand_command": ["hand_command", "handCommand"],
    "handCommand": ["handCommand", "hand_command"],
    "pts21_mano": ["pts21_mano"],
    "pts21_mano_flat": ["pts21_mano_flat", "pts21_mano"],
    "timestamp": ["timestamp"],
    "sampleClockNs": ["sampleClockNs"],
    "rgbFrameId": ["rgbFrameId"],
    "rgbCaptureNs": ["rgbCaptureNs"],
    "rgbAlignResidualNs": ["rgbAlignResidualNs"],
}

URDF_UPPER = {
    "thumb_cmc_pitch": 0.58,
    "thumb_cmc_yaw": 1.36,
    "index_mcp_pitch": 1.60,
    "middle_mcp_pitch": 1.60,
    "ring_mcp_pitch": 1.60,
    "pinky_mcp_pitch": 1.60,
}

LINKER_JOINTS = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


@dataclass
class LoadedPolicy:
    cfg: OmegaConf
    policy: Any
    device: torch.device
    action_shape: tuple[int, ...]
    obs_horizon: int


def _install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def _as_list_config(value: Any) -> list[Any]:
    return list(OmegaConf.to_container(value, resolve=True))


def _shape_tuple(attr: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in _as_list_config(attr["shape"]))


def _obs_horizon_from_cfg(cfg: OmegaConf) -> int:
    if "n_obs_steps" in cfg:
        return int(cfg.n_obs_steps)
    horizons = []
    for attr in cfg.task.shape_meta.obs.values():
        if "horizon" in attr:
            horizons.append(int(attr.horizon))
    return max(horizons) if horizons else 1


def _load_policy(ckpt_path: str, device_name: str, num_inference_steps: int | None) -> LoadedPolicy:
    ckpt = Path(ckpt_path).expanduser()
    if not ckpt.suffix == ".ckpt":
        ckpt = ckpt / "checkpoints" / "latest.ckpt"
    payload = torch.load(open(ckpt, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(num_inference_steps)

    device = torch.device(device_name)
    policy.eval().to(device)
    action_shape = tuple(int(x) for x in _as_list_config(cfg.task.shape_meta.action.shape))
    if len(action_shape) != 1:
        raise ValueError(f"DexUMI expects a flat action shape, got {action_shape}")

    print("checkpoint:", ckpt)
    print("workspace:", cfg._target_)
    print("policy:", cfg.policy._target_)
    print("dataset_path:", cfg.task.dataset.dataset_path)
    print("obs_keys:", list(cfg.task.shape_meta.obs.keys()))
    print("action_shape:", action_shape)
    print("obs_horizon:", _obs_horizon_from_cfg(cfg))
    if hasattr(policy, "num_inference_steps"):
        print("num_inference_steps:", policy.num_inference_steps)

    return LoadedPolicy(
        cfg=cfg,
        policy=policy,
        device=device,
        action_shape=action_shape,
        obs_horizon=_obs_horizon_from_cfg(cfg),
    )


def _find_alias(mapping: dict[str, Any], key: str) -> str | None:
    for candidate in FIELD_ALIASES.get(key, [key]):
        if candidate in mapping:
            return candidate
    return None


def _read_rgb(path: Path, resize_hw: tuple[int, int] | None = None) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    if resize_hw is not None:
        height, width = resize_hw
        image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _pose7_xyzw_to_pose6(pose7: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    arr = np.asarray(pose7, dtype=np.float32).reshape(7)
    rotvec = R.from_quat(arr[3:7]).as_rotvec().astype(np.float32)
    return np.concatenate([arr[:3], rotvec], axis=0).astype(np.float32)


def _angles_rad_to_cmd(angles_rad: np.ndarray) -> np.ndarray:
    cmds = np.zeros(len(LINKER_JOINTS), dtype=np.float32)
    for i, (name, q) in enumerate(zip(LINKER_JOINTS, np.asarray(angles_rad).reshape(-1))):
        upper = URDF_UPPER[name]
        q_c = float(np.clip(q, 0.0, upper))
        cmds[i] = float(np.clip(round(250.0 * (1.0 - q_c / upper)), 0, 255))
    return cmds


def _valid_array(value: Any, expected_shape: tuple[int, ...] | None, dtype: np.dtype) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if expected_shape is not None and arr.shape != expected_shape:
        return None
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        return None
    return arr.astype(dtype, copy=False)


def _load_pkl_messages(pkl_path: Path) -> list[dict[str, Any]]:
    _install_numpy_pickle_compat()
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict/messages: {pkl_path}")
    return [m for m in data["messages"] if isinstance(m, dict)]


def _messages_to_episode(
        pkl_path: Path,
        shape_meta: Any,
        action_source: str = "auto") -> dict[str, np.ndarray]:
    obs_meta = shape_meta.obs
    required_rgb_shapes: dict[str, tuple[int, int]] = {}
    for key, attr in obs_meta.items():
        if attr.get("type", "low_dim") == "rgb":
            c, h, w = _shape_tuple(attr)
            if c != 3:
                raise ValueError(f"{key} expects {c} channels; only RGB is supported")
            required_rgb_shapes[key] = (h, w)

    messages = _load_pkl_messages(pkl_path)
    rows: dict[str, list[np.ndarray | float | int]] = {str(k): [] for k in obs_meta.keys()}
    rows["action"] = []
    rows["timestamp"] = []

    action_dim = int(shape_meta.action.shape[0])
    for msg in messages:
        obs_values: dict[str, np.ndarray] = {}
        ok = True
        for key, attr in obs_meta.items():
            key = str(key)
            obs_type = attr.get("type", "low_dim")
            shape = _shape_tuple(attr)
            if obs_type == "rgb":
                rgb_rel = msg.get("rgbImage")
                if not rgb_rel:
                    ok = False
                    break
                rgb = _read_rgb(pkl_path.parent / str(rgb_rel), resize_hw=required_rgb_shapes[key])
                obs_values[key] = rgb
            else:
                src_key = _find_alias(msg, key)
                if src_key is None:
                    ok = False
                    break
                value = msg[src_key]
                if np.asarray(value).shape == (7,) and shape == (6,):
                    value = _pose7_xyzw_to_pose6(value)
                if key == "pts21_mano_flat" and np.asarray(value).shape == (21, 3):
                    value = np.asarray(value).reshape(-1)
                arr = _valid_array(value, shape, np.float32)
                if arr is None:
                    ok = False
                    break
                obs_values[key] = arr

        if not ok:
            continue

        action = None
        if action_dim == 6:
            selected = action_source
            if selected == "auto":
                selected = "hand_command" if any(k in ("hand_command", "handCommand") for k in obs_meta.keys()) else "trajectory_pose"
            if selected == "hand_command":
                src = _find_alias(msg, "hand_command")
                if src is not None:
                    action = _valid_array(msg[src], (6,), np.float32)
            elif selected == "trajectory_pose":
                src = _find_alias(msg, "trajectoryPose")
                if src is not None:
                    value = msg[src]
                    if np.asarray(value).shape == (7,):
                        value = _pose7_xyzw_to_pose6(value)
                    action = _valid_array(value, (6,), np.float32)
            else:
                raise ValueError(f"unsupported pkl action source: {action_source}")
        elif action_dim == 7:
            pose = _valid_array(msg.get("trajectoryPose"), (6,), np.float32)
            hand = _valid_array(msg.get("handCommand"), (6,), np.float32)
            if pose is not None and hand is not None:
                action = np.concatenate([pose[:6], hand[:1]], axis=0)
        if action is None:
            continue

        for key, value in obs_values.items():
            rows[key].append(value)
        rows["action"].append(action.astype(np.float32))
        rows["timestamp"].append(float(msg.get("timestamp", 0.0)))

    if not rows["action"]:
        raise ValueError(f"no valid frames matched checkpoint shape_meta in {pkl_path}")

    episode: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        if not values:
            continue
        if key == "timestamp":
            episode[key] = np.asarray(values, dtype=np.float64)
        else:
            episode[key] = np.stack(values, axis=0)
    return episode


def _load_zarr_episode(zarr_path: Path, episode_index: int | None = None) -> dict[str, np.ndarray]:
    import zarr

    root_path = zarr_path.expanduser()
    if root_path.name != "replay_buffer.zarr":
        root_path = root_path / "replay_buffer.zarr"
    root = zarr.open(str(root_path), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0:
        raise ValueError(f"no episodes in {root_path}")

    if episode_index is None:
        start, end = 0, int(episode_ends[-1])
    else:
        ep = int(episode_index)
        if ep < 0 or ep >= len(episode_ends):
            raise IndexError(f"episode_index={ep} out of range [0, {len(episode_ends)})")
        start = 0 if ep == 0 else int(episode_ends[ep - 1])
        end = int(episode_ends[ep])

    return {
        str(key): np.asarray(arr[start:end])
        for key, arr in root["data"].items()
    }


def _slice_obs_sequence(
        episode: dict[str, np.ndarray],
        shape_meta: Any,
        end_index: int,
        obs_horizon: int) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {}
    end = int(end_index) + 1
    start = max(0, end - int(obs_horizon))
    for key, attr in shape_meta.obs.items():
        key = str(key)
        src_key = _find_alias(episode, key)
        if src_key is None:
            raise KeyError(f"episode does not contain obs key {key}; available={sorted(episode.keys())}")
        arr = np.asarray(episode[src_key])
        seq = arr[start:end]
        if len(seq) == 0:
            raise ValueError(f"empty observation sequence for key {key}")
        if len(seq) < obs_horizon:
            pad = np.repeat(seq[[0]], obs_horizon - len(seq), axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        obs[key] = seq
    return obs


def _format_obs_for_policy(env_obs: dict[str, np.ndarray], shape_meta: Any) -> dict[str, np.ndarray]:
    obs_dict: dict[str, np.ndarray] = {}
    for key, attr in shape_meta.obs.items():
        key = str(key)
        obs_type = attr.get("type", "low_dim")
        shape = _shape_tuple(attr)
        data = np.asarray(env_obs[key])
        if obs_type == "rgb":
            if data.ndim != 4:
                raise ValueError(f"{key} expected THWC image sequence, got {data.shape}")
            t, hi, wi, ci = data.shape
            co, ho, wo = shape
            if ci != co:
                raise ValueError(f"{key} expected {co} channels, got {ci}")
            out = data
            if (hi != ho) or (wi != wo) or out.dtype == np.uint8:
                tf = get_image_transform(
                    input_res=(wi, hi),
                    output_res=(wo, ho),
                    bgr_to_rgb=False,
                )
                out = np.stack([tf(x) for x in out], axis=0)
                if data.dtype == np.uint8:
                    out = out.astype(np.float32) / 255.0
            obs_dict[key] = np.moveaxis(out.astype(np.float32), -1, 1)
        else:
            out = data.astype(np.float32)
            if out.shape[1:] != shape:
                raise ValueError(f"{key} expected T{shape}, got {out.shape}")
            obs_dict[key] = out
    return obs_dict


def _predict_action(
        loaded: LoadedPolicy,
        obs_seq: dict[str, np.ndarray],
        fixed_action_prefix: torch.Tensor | None = None) -> np.ndarray:
    obs_np = _format_obs_for_policy(obs_seq, loaded.cfg.task.shape_meta)
    obs_torch = dict_apply(
        obs_np,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(loaded.device),
    )
    with torch.no_grad():
        try:
            if fixed_action_prefix is None:
                result = loaded.policy.predict_action(obs_torch)
            else:
                result = loaded.policy.predict_action(obs_torch, fixed_action_prefix=fixed_action_prefix)
        except TypeError:
            result = loaded.policy.predict_action(obs_torch)
    key = "action" if "action" in result else "action_pred"
    return result[key][0].detach().to("cpu").numpy()


def _coerce_hand_command(action: np.ndarray, mode: str, last_obs: dict[str, np.ndarray] | None = None) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] < 6:
        raise ValueError(f"cannot convert action with shape {arr.shape} to 6D hand command")

    if mode == "auto":
        mode = "hand_command"

    if mode == "hand_command":
        cmd = arr[:6]
    elif mode == "trajectory_pose":
        if last_obs is None:
            raise ValueError("--action-mode trajectory_pose needs last_obs fallback for hand command")
        src_key = _find_alias(last_obs, "hand_command")
        if src_key is None:
            raise ValueError("trajectory_pose action has no hand_command observation fallback")
        cmd = np.asarray(last_obs[src_key])[-1].astype(np.float32)
    else:
        raise ValueError(f"unsupported action mode: {mode}")
    return np.clip(np.rint(cmd), 0, 255).astype(np.uint8)


def _open_o6_hand(can_channel: str, bitrate: int, speed: int, torque: int | None, dexumi_root: Path | None):
    if dexumi_root is not None:
        root = dexumi_root.expanduser().resolve()
        sys.path.insert(0, str(root / "o6_right_hand"))
        sys.path.insert(0, str(root))
    from controller import O6RightHand

    hand = O6RightHand(can_channel=can_channel, bitrate=bitrate)
    hand.set_speed(speed)
    if torque is not None:
        hand.set_torque(torque)
    return hand


def _dump_actions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_offline(args: argparse.Namespace, loaded: LoadedPolicy) -> None:
    if args.zarr:
        episode = _load_zarr_episode(Path(args.zarr), episode_index=args.episode_index)
        source = args.zarr
    elif args.pkl:
        episode = _messages_to_episode(
            Path(args.pkl).expanduser(),
            loaded.cfg.task.shape_meta,
            action_source=args.pkl_action_source,
        )
        source = args.pkl
    else:
        raise SystemExit("offline mode requires --zarr or --pkl")

    total = len(next(iter(episode.values())))
    start = max(0, int(args.start_index))
    stop = total if args.max_steps is None else min(total, start + int(args.max_steps))
    rows: list[dict[str, Any]] = []
    mse_values = []
    print(f"offline source: {source}")
    print(f"frames: {total}, running indices [{start}, {stop})")

    for idx in range(start, stop):
        obs_seq = _slice_obs_sequence(
            episode,
            loaded.cfg.task.shape_meta,
            end_index=idx,
            obs_horizon=loaded.obs_horizon,
        )
        t0 = time.time()
        pred = _predict_action(loaded, obs_seq)
        latency = time.time() - t0
        first = pred[0]
        row = {
            "index": idx,
            "latency_s": latency,
            "pred_shape": list(pred.shape),
        }
        for j, value in enumerate(first):
            row[f"pred_{j}"] = float(value)

        if "action" in episode and idx < len(episode["action"]):
            gt = np.asarray(episode["action"][idx], dtype=np.float32)
            n = min(gt.shape[-1], first.shape[-1])
            mse = float(np.mean((first[:n] - gt[:n]) ** 2))
            mse_values.append(mse)
            row["gt_mse_first_step"] = mse
            for j, value in enumerate(gt[: first.shape[-1]]):
                row[f"gt_{j}"] = float(value)
        rows.append(row)
        print(f"{idx}: pred_shape={pred.shape} first={np.round(first, 4).tolist()} latency={latency:.4f}s")

    if args.output:
        _dump_actions_csv(Path(args.output).expanduser(), rows)
        print("wrote:", args.output)
    if mse_values:
        print("mean first-step MSE:", float(np.mean(mse_values)))


def _build_mvs_source(args: argparse.Namespace):
    dexumi_root = Path(args.dexumi_root).expanduser().resolve()
    sys.path.insert(0, str(dexumi_root / "teleop"))
    sys.path.insert(0, str(dexumi_root / "teleop" / "mvs_cpp"))
    from mvs_cpp import LatestMVSCache, MVSConfig

    config = MVSConfig(
        serial=args.mvs_serial,
        width=args.mvs_width,
        height=args.mvs_height,
        fps=args.frequency,
        exposure_time_us=args.mvs_exposure_us,
        gain_auto=args.mvs_gain_auto,
        balance_white_auto=args.mvs_white_balance_auto,
        auto_crop_black_border=args.mvs_auto_crop_black_border,
        center_crop=args.mvs_center_crop,
        rotate_180=args.mvs_rotate_180,
    )
    cache = LatestMVSCache(config)
    cache.start()
    return cache


def _open_pico_source(dexumi_root: Path, hand: str, yaml_path: str | None):
    root = dexumi_root.expanduser().resolve()
    sys.path.insert(0, str(root / "teleop"))
    sys.path.insert(0, str(root))
    from pico_hand import PicoHandReader
    from retargeter import PicoToLinkerO6Retargeter

    reader = PicoHandReader(hand=hand)
    retargeter = PicoToLinkerO6Retargeter(
        **({"yaml_path": yaml_path} if yaml_path else {}),
        hand=hand,
    )
    return reader, retargeter


def _read_pico_obs(reader: Any, retargeter: Any) -> dict[str, np.ndarray] | None:
    raw26x7, active = reader.read_raw()
    if int(active) != 1 or raw26x7 is None or np.asarray(raw26x7).shape[:2] != (26, 7):
        return None
    retargeted = retargeter.retarget(raw26x7)
    hand_command = _angles_rad_to_cmd(retargeted.linker_joint_radians)
    wrist_pose = np.asarray(retargeted.wrist_pose_6d, dtype=np.float32).reshape(6)
    pts21 = np.asarray(retargeted.pts21_mano, dtype=np.float32).reshape(21, 3)
    return {
        "hand_command": hand_command.astype(np.float32),
        "handCommand": hand_command.astype(np.float32),
        "robot_eef_pose": wrist_pose,
        "trajectoryPose": wrist_pose,
        "trajectory_pose": wrist_pose,
        "pts21_mano": pts21,
        "pts21_mano_flat": pts21.reshape(-1),
    }


def run_real(args: argparse.Namespace, loaded: LoadedPolicy) -> None:
    if args.action_mode == "trajectory_pose":
        raise SystemExit(
            "This script can only execute Linker O6 hand commands. "
            "If your checkpoint predicts trajectoryPose, run offline validation or retrain action=hand_command."
        )

    hand = None
    camera = None
    pico_reader = None
    pico_retargeter = None
    last_pico_obs: dict[str, np.ndarray] | None = None
    try:
        if args.real_camera == "mvs":
            camera = _build_mvs_source(args)
            print("MVS camera started")
        else:
            camera = cv2.VideoCapture(int(args.camera_index))
            if not camera.isOpened():
                raise RuntimeError(f"failed to open cv2 camera index {args.camera_index}")

        if args.dry_run:
            print("dry run: CAN hand is not opened")
        else:
            hand = _open_o6_hand(
                can_channel=args.can_channel,
                bitrate=args.bitrate,
                speed=args.speed,
                torque=args.torque,
                dexumi_root=Path(args.dexumi_root),
            )
            if args.home_on_start:
                hand.home()
                time.sleep(1.0)

        if not args.no_pico:
            pico_reader, pico_retargeter = _open_pico_source(
                dexumi_root=Path(args.dexumi_root),
                hand=args.hand,
                yaml_path=args.retarget_yaml,
            )
            print("PICO source started")
        else:
            print("PICO disabled: non-image low-dim obs use hand state or zeros")

        obs_meta = loaded.cfg.task.shape_meta.obs
        obs_buffers: dict[str, deque[np.ndarray]] = {
            str(key): deque(maxlen=loaded.obs_horizon)
            for key in obs_meta.keys()
        }
        period = 1.0 / max(float(args.frequency), 1e-3)
        iter_idx = 0
        rows: list[dict[str, Any]] = []
        print("Press Ctrl+C to stop.")

        while args.max_steps is None or iter_idx < int(args.max_steps):
            loop_t0 = time.time()
            frame = None
            if args.real_camera == "mvs":
                cached = camera.latest()
                frame = None if cached is None else cached.image
            else:
                ok, bgr = camera.read()
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
            if frame is None:
                print("camera frame unavailable; skipping")
                time.sleep(period)
                continue

            if pico_reader is not None and pico_retargeter is not None:
                pico_obs = _read_pico_obs(pico_reader, pico_retargeter)
                if pico_obs is not None:
                    last_pico_obs = pico_obs

            for key, attr in obs_meta.items():
                key = str(key)
                obs_type = attr.get("type", "low_dim")
                shape = _shape_tuple(attr)
                if obs_type == "rgb":
                    obs_buffers[key].append(frame)
                else:
                    src_key = None
                    if last_pico_obs is not None:
                        src_key = _find_alias(last_pico_obs, key)
                    if src_key is not None:
                        obs_buffers[key].append(np.asarray(last_pico_obs[src_key], dtype=np.float32))
                    elif key in ("hand_command", "handCommand") and hand is not None:
                        obs_buffers[key].append(np.asarray(hand.get_state(), dtype=np.float32))
                    elif key in ("hand_command", "handCommand"):
                        obs_buffers[key].append(np.full(shape, args.default_hand_command, dtype=np.float32))
                    else:
                        obs_buffers[key].append(np.zeros(shape, dtype=np.float32))

            if any(len(buf) < loaded.obs_horizon for buf in obs_buffers.values()):
                time.sleep(max(0.0, period - (time.time() - loop_t0)))
                continue

            obs_seq = {key: np.stack(list(buf), axis=0) for key, buf in obs_buffers.items()}
            pred = _predict_action(loaded, obs_seq)
            cmd = _coerce_hand_command(pred[0], mode=args.action_mode, last_obs=obs_seq)
            if hand is None:
                print(iter_idx, "dry-run cmd", cmd.tolist(), "pred_first", np.round(pred[0], 4).tolist())
            else:
                hand.move(cmd.tolist())
                print(iter_idx, "sent", cmd.tolist())

            rows.append({
                "iter": iter_idx,
                "time": time.time(),
                "cmd": cmd.tolist(),
                "pred_first": np.asarray(pred[0], dtype=float).tolist(),
            })
            iter_idx += 1
            time.sleep(max(0.0, period - (time.time() - loop_t0)))
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if args.output:
            out = Path(args.output).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print("wrote:", out)
        if hand is not None:
            if args.home_on_exit:
                hand.home()
                time.sleep(0.5)
            hand.close()
        if camera is not None and hasattr(camera, "stop"):
            camera.stop()
        elif camera is not None and hasattr(camera, "release"):
            camera.release()
        if pico_reader is not None and hasattr(pico_reader, "close"):
            pico_reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="checkpoint .ckpt or output dir containing checkpoints/latest.ckpt")
    parser.add_argument("--mode", choices=["offline", "real"], default="offline")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--zarr", default=None, help="converted dataset dir or replay_buffer.zarr for offline validation")
    parser.add_argument("--pkl", default=None, help="DexUMI PKL bundle for offline validation")
    parser.add_argument("--pkl-action-source", choices=["auto", "hand_command", "trajectory_pose"], default="auto")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("-o", "--output", default=None, help="CSV for offline mode, JSON for real mode")
    parser.add_argument("--action-mode", choices=["auto", "hand_command", "trajectory_pose"], default="auto")

    parser.add_argument("--dexumi-root", default="/home/zjc/Desktop/human2dex", help="DexUMI repo root on this machine")
    parser.add_argument("--dry-run", action="store_true", help="real mode: do not open CAN, only print commands")
    parser.add_argument("--real-camera", choices=["mvs", "cv2"], default="mvs")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frequency", "-f", type=float, default=10.0)
    parser.add_argument("--can-channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--torque", type=int, default=None)
    parser.add_argument("--home-on-start", action="store_true")
    parser.add_argument("--home-on-exit", action="store_true")
    parser.add_argument("--default-hand-command", type=float, default=250.0)
    parser.add_argument("--no-pico", action="store_true", help="real mode: do not read PICO retargeted obs")
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--retarget-yaml", default=None, help="optional DexUMI retargeting yaml")

    parser.add_argument("--mvs-serial", default=None)
    parser.add_argument("--mvs-width", type=int, default=640)
    parser.add_argument("--mvs-height", type=int, default=480)
    parser.add_argument("--mvs-exposure-us", type=float, default=6000.0)
    parser.add_argument("--mvs-gain-auto", default="Off")
    parser.add_argument("--mvs-white-balance-auto", default="Continuous")
    parser.add_argument("--mvs-auto-crop-black-border", action="store_true")
    parser.add_argument("--mvs-center-crop", action="store_true")
    parser.add_argument("--mvs-rotate-180", action="store_true")
    args = parser.parse_args()

    loaded = _load_policy(
        ckpt_path=args.input,
        device_name=args.device,
        num_inference_steps=args.num_inference_steps,
    )
    if args.mode == "offline":
        run_offline(args, loaded)
    else:
        run_real(args, loaded)


if __name__ == "__main__":
    main()
