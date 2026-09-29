#!/usr/bin/env python3
"""Pure observation, action, timestamp, and diagnostics helpers."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from umi.common.precise_sleep import precise_wait

def disable_pretrained_download(cfg) -> None:
    if (
        "policy" in cfg
        and "obs_encoder" in cfg.policy
        and "pretrained" in cfg.policy.obs_encoder
    ):
        cfg.policy.obs_encoder.pretrained = False

def obs_horizon(shape_meta, key: str, default: int = 2) -> int:
    obs_meta = shape_meta["obs"]
    if key not in obs_meta:
        return default
    return int(obs_meta[key].get("horizon", default))

def down_sample_steps(task_cfg):
    obs_steps = task_cfg.get("obs_down_sample_steps", 1)
    action_steps = int(task_cfg.get("action_down_sample_steps", 1))
    if isinstance(obs_steps, int):
        return max(1, int(obs_steps) // max(1, action_steps))
    return ([0] + [int(x) // max(1, action_steps) for x in obs_steps])[::-1]

def filter_obs_for_shape_meta(obs_dict: dict, shape_meta) -> dict:
    allowed = {str(k) for k in shape_meta["obs"].keys()}
    return {str(k): v for k, v in obs_dict.items() if str(k) in allowed}

def patch_object_pocket_obs(env_obs: dict, shape_meta) -> dict:
    """Fill missing objectPocketObs with an explicit invalid observation.

    Real checkpoints trained with objectPocketObs require the key to exist at
    inference time. Until the online object tracker is enabled, zeros represent:
    no relative offset, no log-size cue, confidence=0, valid=0.
    """
    key = "objectPocketObs"
    if key not in shape_meta["obs"] or key in env_obs:
        return env_obs
    shape = tuple(int(x) for x in shape_meta["obs"][key].get("shape", ()))
    if len(shape) != 1:
        return env_obs
    horizon = obs_horizon(
        shape_meta,
        key,
        len(np.asarray(env_obs.get("timestamp", [0]))),
    )
    out = dict(env_obs)
    out[key] = np.zeros((horizon, shape[0]), dtype=np.float32)
    return out

def patch_hand_obs(env_obs: dict, hand_state: Optional[np.ndarray], shape_meta) -> dict:
    key = "robot0_gripper_width"
    if hand_state is None or key not in shape_meta["obs"]:
        return patch_object_pocket_obs(env_obs, shape_meta)
    shape = tuple(int(x) for x in shape_meta["obs"][key].get("shape", ()))
    state = np.asarray(hand_state, dtype=np.float32).reshape(-1)
    if shape != (state.shape[0],):
        return patch_object_pocket_obs(env_obs, shape_meta)
    out = dict(env_obs)
    horizon = obs_horizon(shape_meta, key, len(np.asarray(env_obs.get("timestamp", [0]))))
    out[key] = np.repeat(state[None, :], horizon, axis=0)
    return patch_object_pocket_obs(out, shape_meta)

def obs_feature_shape(shape_meta, key: str) -> Optional[tuple[int, ...]]:
    obs_meta = shape_meta["obs"]
    if key not in obs_meta:
        return None
    shape = obs_meta[key].get("shape", ())
    return tuple(int(x) for x in shape)

def _format_feature_shape(shape: Optional[tuple[int, ...]]) -> str:
    return "missing" if shape is None else str(shape)

def validate_hand_checkpoint_compatibility(
        *,
        hand_backend: Optional[str],
        action_dim: int,
        shape_meta,
        pts21_mode: bool,
        task_name: str = "<unknown>") -> list[str]:
    """Validate hand/backend dimensions before hardware is opened.

    Returns warning messages that should be printed by the caller.
    """
    if pts21_mode:
        return []

    action_dim = int(action_dim)
    gripper_shape = obs_feature_shape(shape_meta, "robot0_gripper_width")
    gripper_shape_text = _format_feature_shape(gripper_shape)
    warnings = []

    if hand_backend == "linker_o6":
        if action_dim != 15:
            raise ValueError(
                "hand=linker_o6 is incompatible with checkpoint "
                f"task.name={task_name}, action_dim={action_dim}, "
                f"robot0_gripper_width shape={gripper_shape_text}. "
                "Use a Linker O6 checkpoint with action_dim 15 and 6D hand obs."
            )
        if gripper_shape is not None and gripper_shape != (6,):
            raise ValueError(
                "hand=linker_o6 is incompatible with checkpoint "
                f"task.name={task_name}, action_dim={action_dim}, "
                f"robot0_gripper_width shape={gripper_shape_text}. "
                "Use a Linker O6 checkpoint with 6D hand obs."
            )
    elif hand_backend == "wuji_hand":
        if action_dim in (20, 29):
            if gripper_shape is not None and gripper_shape != (20,):
                raise ValueError(
                    "hand=wuji_hand is incompatible with checkpoint "
                    f"task.name={task_name}, action_dim={action_dim}, "
                    f"robot0_gripper_width shape={gripper_shape_text}. "
                    "Use a Wuji checkpoint with 20D hand obs."
                )
        elif action_dim == 15 or gripper_shape == (6,):
            raise ValueError(
                "hand=wuji_hand is incompatible with checkpoint "
                f"task.name={task_name}, action_dim={action_dim}, "
                f"robot0_gripper_width shape={gripper_shape_text}. "
                "This looks like a Linker O6 checkpoint. Use hand=linker_o6 "
                "or a Wuji checkpoint with action_dim 20/29 and 20D hand obs."
            )
        else:
            warnings.append(
                f"hand=wuji_hand with action_dim={action_dim}; "
                "the policy must return result['wuji_command']."
            )
    elif hand_backend is None:
        if gripper_shape is not None and gripper_shape not in ((1,),):
            raise ValueError(
                "hand disabled but checkpoint "
                f"task.name={task_name}, action_dim={action_dim}, "
                f"robot0_gripper_width shape={gripper_shape_text}. "
                "Enable the matching hand backend or use an arm-only checkpoint."
            )
        if action_dim not in (10,):
            warnings.append(
                f"hand disabled but checkpoint action_dim={action_dim}; "
                "hand tail will be ignored."
            )

    return warnings

def latest_env_gripper_width(env_obs: dict) -> float:
    arr = np.asarray(env_obs.get("robot0_gripper_width", [[0.0]]))
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[-1])

def split_policy_action(
        pred: np.ndarray,
        action_dim: int,
        gripper_width: float,
        hand_backend: Optional[str],
        hand_state: Optional[np.ndarray] = None,
        hand_command_representation: str = "absolute"):
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 1:
        pred = pred[None, :]
    n = pred.shape[0]
    arm_action = None
    hand_action = None
    if action_dim == 10:
        arm_action = pred[:, :10]
    elif action_dim == 15:
        dummy_gripper = np.full((n, 1), gripper_width, dtype=np.float32)
        arm_action = np.concatenate([pred[:, :9], dummy_gripper], axis=-1)
        if hand_backend == "linker_o6":
            hand_pred = pred[:, 9:15]
            representation = str(hand_command_representation).strip().lower()
            if representation == "delta_from_current":
                if hand_state is None:
                    raise ValueError(
                        "delta_from_current O6 action requires the current 6D hand_state"
                    )
                base = np.asarray(hand_state, dtype=np.float32).reshape(-1)
                if base.shape != (6,) or not np.all(np.isfinite(base)):
                    raise ValueError(
                        f"invalid O6 hand_state for delta action: shape={base.shape}"
                    )
                # One fixed base for the complete predicted horizon.
                hand_pred = hand_pred + base[None, :]
            elif representation != "absolute":
                raise ValueError(
                    "hand_command_representation must be absolute or "
                    f"delta_from_current, got {hand_command_representation!r}"
                )
            hand_action = np.clip(np.rint(hand_pred), 0, 255).astype(np.uint8)
    elif action_dim == 20:
        if hand_backend == "wuji_hand":
            hand_action = pred[:, :20].astype(np.float64)
    elif action_dim == 29:
        dummy_gripper = np.full((n, 1), gripper_width, dtype=np.float32)
        arm_action = np.concatenate([pred[:, :9], dummy_gripper], axis=-1)
        if hand_backend == "wuji_hand":
            hand_action = pred[:, 9:29].astype(np.float64)
    else:
        raise ValueError(f"Unsupported action_dim={action_dim}")
    return arm_action, hand_action

def scale_franka_z_delta(
        actions: np.ndarray,
        current_z: float,
        gain: float,
        robot_idx: int = 0) -> np.ndarray:
    """Scale target z displacement around the current Franka base-frame z."""
    gain = float(gain)
    current_z = float(current_z)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"franka_z_delta_gain must be finite and > 0, got {gain!r}")
    if not np.isfinite(current_z):
        raise ValueError(f"current_z must be finite, got {current_z!r}")

    arr = np.asarray(actions)
    if arr.ndim != 2:
        raise ValueError(f"actions must be a 2D array, got {arr.shape}")
    z_col = 7 * int(robot_idx) + 2
    if z_col >= arr.shape[1]:
        raise ValueError(
            f"actions with shape {arr.shape} do not contain robot{robot_idx} z"
        )
    if gain == 1.0:
        return arr

    out = np.array(arr, copy=True)
    out[:, z_col] = current_z + gain * (out[:, z_col] - current_z)
    return out

def scale_franka_negative_x_delta(
        actions: np.ndarray,
        current_x: float,
        gain: float,
        robot_idx: int = 0) -> np.ndarray:
    """Scale only negative base-frame x displacement around current Franka x."""
    gain = float(gain)
    current_x = float(current_x)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"franka_neg_x_delta_gain must be finite and > 0, got {gain!r}")
    if not np.isfinite(current_x):
        raise ValueError(f"current_x must be finite, got {current_x!r}")

    arr = np.asarray(actions)
    if arr.ndim != 2:
        raise ValueError(f"actions must be a 2D array, got {arr.shape}")
    x_col = 7 * int(robot_idx)
    if x_col >= arr.shape[1]:
        raise ValueError(
            f"actions with shape {arr.shape} do not contain robot{robot_idx} x"
        )
    if gain == 1.0:
        return arr

    out = np.array(arr, copy=True)
    dx = out[:, x_col] - current_x
    negative = dx < 0.0
    out[negative, x_col] = current_x + gain * dx[negative]
    return out

def scale_franka_negative_y_delta(
        actions: np.ndarray,
        current_y: float,
        gain: float,
        robot_idx: int = 0) -> np.ndarray:
    """Scale only negative base-frame y displacement around current Franka y."""
    gain = float(gain)
    current_y = float(current_y)
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"franka_neg_y_delta_gain must be finite and > 0, got {gain!r}")
    if not np.isfinite(current_y):
        raise ValueError(f"current_y must be finite, got {current_y!r}")

    arr = np.asarray(actions)
    if arr.ndim != 2:
        raise ValueError(f"actions must be a 2D array, got {arr.shape}")
    y_col = 7 * int(robot_idx) + 1
    if y_col >= arr.shape[1]:
        raise ValueError(
            f"actions with shape {arr.shape} do not contain robot{robot_idx} y"
        )
    if gain == 1.0:
        return arr

    out = np.array(arr, copy=True)
    dy = out[:, y_col] - current_y
    negative = dy < 0.0
    out[negative, y_col] = current_y + gain * dy[negative]
    return out

def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").numpy()
    return np.asarray(value)

def _coerce_wuji_action_sequence(value, max_steps: Optional[int] = None) -> np.ndarray:
    arr = _to_numpy(value).astype(np.float64, copy=False)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[-2:] == (5, 4):
        arr = arr.reshape(arr.shape[:-2] + (20,))
    if arr.ndim == 1:
        arr = arr.reshape(1, 20)
    if arr.ndim != 2 or arr.shape[-1] != 20:
        raise ValueError(f"Wuji command must be (H,20), (H,5,4), (20,), or (5,4), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Wuji command contains NaN or Inf")
    if max_steps is not None:
        arr = arr[:max_steps]
    return arr

def future_timestamps(
        obs: dict,
        steps: int,
        frequency: float,
        start_delay: float = 0.25) -> np.ndarray:
    now = time.time()
    start = now + float(start_delay)
    if "timestamp" in obs:
        obs_time = float(np.asarray(obs["timestamp"]).reshape(-1)[-1])
        if np.isfinite(obs_time):
            start = obs_time + float(start_delay)
    return start + np.arange(steps, dtype=np.float64) / float(frequency)

def keep_future_actions(
        actions: np.ndarray,
        timestamps: np.ndarray,
        min_lead_time: float,
        fallback_delay: float):
    now = time.time()
    is_future = timestamps > (now + float(min_lead_time))
    if np.any(is_future):
        return actions[is_future], timestamps[is_future], is_future

    if len(actions) == 0:
        return actions, timestamps, is_future

    fallback_timestamps = np.array([now + float(fallback_delay)], dtype=np.float64)
    return actions[[-1]], fallback_timestamps, is_future

def align_recording_hand_commands(
        hand_commands: Optional[np.ndarray],
        kept_mask: np.ndarray,
        submitted_timestamps: np.ndarray):
    """Match post-processed hand commands to the arm waypoints actually submitted."""
    timestamps = np.asarray(submitted_timestamps, dtype=np.float64).reshape(-1)
    if hand_commands is None or len(timestamps) == 0:
        return timestamps[:0], None

    commands = np.asarray(hand_commands)
    if commands.ndim == 1:
        commands = commands[None, :]
    mask = np.asarray(kept_mask, dtype=bool).reshape(-1)
    if np.any(mask):
        kept_indices = np.flatnonzero(mask)
        pairs = [
            (timestamp, commands[action_idx])
            for timestamp, action_idx in zip(timestamps, kept_indices)
            if action_idx < len(commands)
        ]
        if not pairs:
            return timestamps[:0], None
        return (
            np.asarray([item[0] for item in pairs], dtype=np.float64),
            np.stack([item[1] for item in pairs], axis=0),
        )

    # keep_future_actions() reschedules the final waypoint when the whole
    # predicted horizon is already stale.
    return timestamps[-1:], commands[-1:]

def execute_timestamped_hand_commands(
        timestamps: np.ndarray,
        hand_commands: Optional[np.ndarray],
        send_command,
        read_state=None,
        synchronize: bool = True,
        wait_fn=precise_wait,
        time_func=time.time,
        should_continue=None):
    """Send hand commands at the same timestamps used by the arm waypoints."""
    command_timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if hand_commands is None or len(command_timestamps) == 0:
        return None, None
    commands = np.asarray(hand_commands)
    if commands.ndim == 1:
        commands = commands[None, :]
    if len(commands) != len(command_timestamps):
        raise ValueError(
            "hand command/timestamp length mismatch: "
            f"{len(commands)} vs {len(command_timestamps)}"
        )

    sent_commands = []
    measured_states = []
    measurement_complete = read_state is not None
    for timestamp, command in zip(command_timestamps, commands):
        if should_continue is not None and not should_continue():
            break
        if synchronize:
            if should_continue is None:
                wait_fn(float(timestamp), time_func=time_func)
            elif not interruptible_wait_until(
                    float(timestamp), should_continue, time_func=time_func):
                break
        if should_continue is not None and not should_continue():
            break
        if read_state is not None:
            try:
                state = read_state()
            except Exception as exc:
                print(f"[WARN] hand state read failed before command: {exc}")
                state = None
            if state is None:
                measurement_complete = False
            elif measurement_complete:
                measured_states.append(np.asarray(state).copy())
        sent = send_command(np.asarray(command).copy())
        sent_commands.append(
            np.asarray(command if sent is None else sent).copy()
        )

    if not sent_commands:
        return None, None
    measured = None
    if measurement_complete and len(measured_states) == len(sent_commands):
        measured = np.stack(measured_states, axis=0)
    return np.stack(sent_commands, axis=0), measured

def interruptible_wait_until(
        deadline: float,
        should_continue,
        time_func=time.time,
        poll_interval: float = 0.01) -> bool:
    """Wait until a wall-clock deadline while polling a safety predicate."""
    while True:
        if not should_continue():
            return False
        remaining = float(deadline) - float(time_func())
        if remaining <= 0:
            return should_continue()
        time.sleep(min(float(poll_interval), remaining))

def latest_obs_time(obs: dict) -> Optional[float]:
    if "timestamp" not in obs:
        return None
    timestamps = np.asarray(obs["timestamp"], dtype=np.float64).reshape(-1)
    if timestamps.size == 0:
        return None
    return float(timestamps[-1])

def latest_robot_pose(obs: dict, robot_idx: int = 0) -> Optional[np.ndarray]:
    pos_key = f"robot{robot_idx}_eef_pos"
    rot_key = f"robot{robot_idx}_eef_rot_axis_angle"
    if pos_key not in obs or rot_key not in obs:
        return None
    pos = np.asarray(obs[pos_key], dtype=np.float64)
    rot = np.asarray(obs[rot_key], dtype=np.float64)
    if pos.size == 0 or rot.size == 0:
        return None
    return np.concatenate([pos.reshape(-1, 3)[-1], rot.reshape(-1, 3)[-1]])

def fmt_vec(values, precision: int = 4) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return np.array2string(
        arr,
        precision=precision,
        suppress_small=True,
        separator=",",
    )

def action_state_gap_summary(
        actions: Optional[np.ndarray],
        current_pose: Optional[np.ndarray],
        next_pose: Optional[np.ndarray],
        robot_idx: int = 0) -> str:
    if actions is None or len(actions) == 0 or current_pose is None:
        return "state_action_gap=unavailable"

    start = 7 * robot_idx
    cmd_pose = np.asarray(actions[:, start:start + 6], dtype=np.float64)
    if cmd_pose.shape[1] != 6:
        return "state_action_gap=unavailable"

    first_delta = cmd_pose[0] - current_pose
    last_delta = cmd_pose[-1] - current_pose
    if len(cmd_pose) > 1:
        step_delta = np.diff(cmd_pose, axis=0)
        cmd_step_pos_max = float(np.max(np.linalg.norm(step_delta[:, :3], axis=-1)))
        cmd_step_rot_max = float(np.max(np.linalg.norm(step_delta[:, 3:6], axis=-1)))
    else:
        cmd_step_pos_max = 0.0
        cmd_step_rot_max = 0.0

    actual_msg = ""
    if next_pose is not None:
        actual_delta = next_pose - current_pose
        actual_msg = (
            f" actual_dpos={fmt_vec(actual_delta[:3])}"
            f" actual_drot={fmt_vec(actual_delta[3:6])}"
        )

    return (
        f"cmd0_dpos={fmt_vec(first_delta[:3])} "
        f"cmd0_drot={fmt_vec(first_delta[3:6])} "
        f"cmdN_dpos={fmt_vec(last_delta[:3])} "
        f"cmdN_drot={fmt_vec(last_delta[3:6])} "
        f"cmd_step_max(pos={cmd_step_pos_max:.4f},rot={cmd_step_rot_max:.4f})"
        f"{actual_msg}"
    )

def run_timing_self_test() -> None:
    actions = np.zeros((4, 7), dtype=np.float64)

    future = time.time() + 0.1 + np.arange(4, dtype=np.float64) * 0.05
    kept_actions, kept_ts, kept_mask = keep_future_actions(
        actions,
        future,
        min_lead_time=0.02,
        fallback_delay=0.25,
    )
    assert kept_actions.shape == actions.shape
    assert np.all(kept_mask)
    assert np.allclose(kept_ts, future)

    past = time.time() - 1.0 + np.arange(4, dtype=np.float64) * 0.05
    rescheduled_actions, rescheduled_ts, rescheduled_mask = keep_future_actions(
        actions,
        past,
        min_lead_time=0.02,
        fallback_delay=0.25,
    )
    assert rescheduled_actions.shape == (1, actions.shape[1])
    assert not np.any(rescheduled_mask)
    lead = rescheduled_ts - time.time()
    assert lead[0] > 0.20

    hand_commands = np.arange(18, dtype=np.float64).reshape(3, 6)
    hand_timestamps = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    waited = []
    sent = []
    state_counter = []

    def fake_wait(timestamp, time_func):
        waited.append(float(timestamp))

    def fake_send(command):
        sent.append(command.copy())
        return command

    def fake_state():
        state = np.full(6, len(state_counter), dtype=np.float64)
        state_counter.append(state)
        return state

    executed, measured = execute_timestamped_hand_commands(
        hand_timestamps,
        hand_commands,
        send_command=fake_send,
        read_state=fake_state,
        synchronize=True,
        wait_fn=fake_wait,
        time_func=lambda: 0.0,
    )
    assert waited == hand_timestamps.tolist()
    assert np.array_equal(executed, hand_commands)
    assert np.array_equal(np.stack(sent), hand_commands)
    assert np.array_equal(measured[:, 0], np.arange(3))

    print("[OK] Franka + hand timing self-test passed.")
