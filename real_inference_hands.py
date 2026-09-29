#!/usr/bin/env python3
"""Runtime hand-state reads and timestamped command execution."""

from __future__ import annotations

from typing import Optional

import numpy as np

from real_inference_actions import execute_timestamped_hand_commands
from real_inference_config import _as_qpos20


_WUJI_FINGER_ROWS = {
    "thumb": 0,
    "index": 1,
    "index_finger": 1,
    "middle": 2,
    "middle_finger": 2,
    "ring": 3,
    "ring_finger": 3,
    "pinky": 4,
    "little": 4,
    "little_finger": 4,
}


_O6_JOINT_COLUMNS = {
    "thumb_cmc_pitch": 0,
    "thumb_pitch": 0,
    "thumb_cmc_yaw": 1,
    "thumb_yaw": 1,
    "index_mcp_pitch": 2,
    "index": 2,
    "middle_mcp_pitch": 3,
    "middle": 3,
    "ring_mcp_pitch": 4,
    "ring": 4,
    "pinky_mcp_pitch": 5,
    "pinky": 5,
    "little": 5,
}


def _fixed_bias_vector(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _o6_command_bias(linker_o6_cfg: dict) -> Optional[np.ndarray]:
    spec = linker_o6_cfg.get("command_bias")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        return _fixed_bias_vector(spec, 6, "O6 command_bias")
    if not bool(spec.get("enabled", True)):
        return None

    bias = np.zeros(6, dtype=np.float64)
    for key in ("values", "command", "bias"):
        if key in spec:
            bias += _fixed_bias_vector(spec[key], 6, f"O6 command_bias.{key}")

    joints = spec.get("joints", {})
    if joints is None:
        joints = {}
    if not isinstance(joints, dict):
        raise ValueError("O6 command_bias.joints must be a mapping")
    for joint_name, raw_value in joints.items():
        key = str(joint_name).strip().lower()
        if key not in _O6_JOINT_COLUMNS:
            raise ValueError(f"unknown O6 joint in command_bias: {joint_name!r}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"O6 command_bias.joints.{joint_name} is NaN or Inf")
        bias[_O6_JOINT_COLUMNS[key]] += value

    indices = spec.get("indices", {})
    if indices is None:
        indices = {}
    if not isinstance(indices, dict):
        raise ValueError("O6 command_bias.indices must be a mapping")
    for raw_index, raw_value in indices.items():
        index = int(raw_index)
        if index < 0 or index >= 6:
            raise ValueError(f"O6 command_bias index out of range: {raw_index!r}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"O6 command_bias.indices.{raw_index} is NaN or Inf")
        bias[index] += value

    if not np.any(bias):
        return None
    return bias


def _wuji_bias_vector(value, name: str = "wuji command_bias") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (5, 4):
        arr = arr.reshape(20)
    else:
        arr = arr.reshape(-1)
    if arr.shape != (20,):
        raise ValueError(f"{name} must contain 20 values or be shaped [5,4], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _wuji_command_bias(wuji_cfg: dict) -> Optional[np.ndarray]:
    spec = wuji_cfg.get("command_bias")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        return _wuji_bias_vector(spec)
    if not bool(spec.get("enabled", True)):
        return None

    bias = np.zeros(20, dtype=np.float64)
    for key in ("values", "qpos", "bias"):
        if key in spec:
            bias += _wuji_bias_vector(spec[key], f"wuji command_bias.{key}")

    fingers = spec.get("fingers", {})
    if fingers is None:
        fingers = {}
    if not isinstance(fingers, dict):
        raise ValueError("wuji command_bias.fingers must be a mapping")
    for finger_name, values in fingers.items():
        key = str(finger_name).strip().lower()
        if key not in _WUJI_FINGER_ROWS:
            raise ValueError(f"unknown Wuji finger in command_bias: {finger_name!r}")
        finger_bias = np.asarray(values, dtype=np.float64).reshape(-1)
        if finger_bias.shape != (4,):
            raise ValueError(
                f"wuji command_bias.fingers.{finger_name} must contain 4 values"
            )
        if not np.all(np.isfinite(finger_bias)):
            raise ValueError(f"wuji command_bias.fingers.{finger_name} contains NaN or Inf")
        start = _WUJI_FINGER_ROWS[key] * 4
        bias[start:start + 4] += finger_bias

    indices = spec.get("indices", {})
    if indices is None:
        indices = {}
    if not isinstance(indices, dict):
        raise ValueError("wuji command_bias.indices must be a mapping")
    for raw_index, raw_value in indices.items():
        index = int(raw_index)
        if index < 0 or index >= 20:
            raise ValueError(f"wuji command_bias index out of range: {raw_index!r}")
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"wuji command_bias.indices.{raw_index} is NaN or Inf")
        bias[index] += value

    if not np.any(bias):
        return None
    return bias


def read_hand_state(
        hand_state: Optional[np.ndarray],
        o6_hand,
        wuji_driver,
        dry_run_hand: bool,
        o6_warning: str,
        wuji_warning: str) -> Optional[np.ndarray]:
    """Read the active hand when available, otherwise retain the last state."""
    latest_state = hand_state
    if o6_hand is not None and not dry_run_hand:
        try:
            latest_state = np.asarray(o6_hand.get_state(), dtype=np.float32).reshape(6)
        except Exception as exc:
            print(f"[WARN] {o6_warning}: {exc}")
    if wuji_driver is not None and not dry_run_hand:
        try:
            latest_state = wuji_driver.read_positions().astype(np.float32)
        except Exception as exc:
            print(f"[WARN] {wuji_warning}: {exc}")
    return latest_state


def execute_hand_actions(
        *,
        execute_enabled: bool,
        scheduled_hand_actions: Optional[np.ndarray],
        hand_backend: Optional[str],
        hand_timestamps: np.ndarray,
        o6_hand,
        wuji_driver,
        wuji_filter,
        linker_o6_cfg: dict,
        wuji_cfg: dict,
        synchronize_hand_actions: bool,
        should_continue_actions=None):
    """Execute one aligned hand chunk and return sent commands plus feedback."""
    executed_hand_actions = None
    measured_hand_states = None

    if (
            execute_enabled
            and scheduled_hand_actions is not None
            and hand_backend == "linker_o6"):
        command_bias = _o6_command_bias(linker_o6_cfg)

        def send_o6(command):
            command = np.asarray(command, dtype=np.float64).reshape(-1)
            if command.shape != (6,):
                raise ValueError(f"O6 command must contain 6 values, got {command.shape}")
            if command_bias is not None:
                command = command + command_bias
            normalized = np.clip(np.rint(command), 0, 255).astype(np.uint8)
            if o6_hand is not None:
                o6_hand.move(normalized.tolist())
            return normalized

        read_o6_state = None
        if o6_hand is not None:
            def read_o6_state():
                state = np.asarray(o6_hand.get_state(), dtype=np.float32).reshape(6)
                if not np.all(np.isfinite(state)) or np.any(state < 0):
                    return None
                return state

        executed_hand_actions, measured_hand_states = execute_timestamped_hand_commands(
            hand_timestamps,
            scheduled_hand_actions,
            send_command=send_o6,
            read_state=read_o6_state,
            synchronize=synchronize_hand_actions,
            should_continue=should_continue_actions,
        )

    elif (
            execute_enabled
            and scheduled_hand_actions is not None
            and hand_backend == "wuji_hand"):
        if wuji_driver is None or wuji_filter is None:
            raise RuntimeError("Wuji backend selected but runtime is not initialized")

        command_bias = _wuji_command_bias(wuji_cfg)

        def send_wuji(qpos):
            qpos = _as_qpos20(qpos)
            if command_bias is not None:
                qpos = qpos + command_bias
            qpos = wuji_driver.clamp(qpos)
            if bool(wuji_cfg.get("filter_enabled", True)):
                qpos = wuji_filter.apply(qpos)
            sent_qpos = wuji_driver.send_positions(qpos)
            return qpos if sent_qpos is None else sent_qpos

        def read_wuji_state():
            state = np.asarray(
                wuji_driver.read_positions(), dtype=np.float32
            ).reshape(20)
            if not np.all(np.isfinite(state)):
                return None
            return state

        executed_hand_actions, measured_hand_states = execute_timestamped_hand_commands(
            hand_timestamps,
            scheduled_hand_actions,
            send_command=send_wuji,
            read_state=read_wuji_state,
            synchronize=synchronize_hand_actions,
            should_continue=should_continue_actions,
        )

    return executed_hand_actions, measured_hand_states
