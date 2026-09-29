#!/usr/bin/env python3
"""Dashboard-facing preview, safety hold, reset, and publishing helpers."""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from real_inference_actions import latest_robot_pose
from real_inference_config import _as_qpos20

def dashboard_frames(obs_dict_np: dict) -> dict:
    frames = {}
    for key, value in obs_dict_np.items():
        array = np.asarray(value)
        if not key.endswith("_rgb") or array.ndim != 4:
            continue
        frames[key] = np.moveaxis(array[-1], 0, -1)
    return frames

def dashboard_rgb_inputs_from_env_obs(env_obs: dict, shape_meta) -> dict:
    """Build only the RGB tensors used by the policy, without pose interpolation."""
    rgb_inputs = {}
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") != "rgb" or key not in env_obs:
            continue
        images = np.asarray(env_obs[key])
        if images.ndim != 4:
            continue
        channels, out_h, out_w = [int(x) for x in attr["shape"]]
        processed = []
        for image in images:
            if image.shape[:2] != (out_h, out_w):
                interpolation = (
                    cv2.INTER_AREA
                    if image.shape[0] >= out_h and image.shape[1] >= out_w
                    else cv2.INTER_LINEAR
                )
                image = cv2.resize(image, (out_w, out_h), interpolation=interpolation)
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32, copy=False)
            if image.shape[-1] != channels:
                raise ValueError(
                    f"{key} expected {channels} channels, got {image.shape}"
                )
            processed.append(image)
        if processed:
            rgb_inputs[key] = np.moveaxis(np.stack(processed), -1, 1)
    return rgb_inputs

def hold_robot_motion(env) -> None:
    """Cancel queued trajectories and hold each robot at its measured pose."""
    for robot in env.robots:
        hold = getattr(robot, "hold_position", None)
        if callable(hold):
            hold()
            continue
        clear_queue = getattr(robot, "clear_queue", None)
        if callable(clear_queue):
            clear_queue()
        state = robot.get_state()
        pose = np.asarray(state["ActualTCPPose"], dtype=np.float64).reshape(6)
        robot.schedule_waypoint(pose=pose, target_time=time.time() + 0.03)

def reset_robot_and_hand(
        env,
        robots_config: list,
        o6_hand,
        linker_o6_cfg: dict,
        wuji_driver,
        wuji_filter,
        wuji_cfg: dict,
        hand_backend: Optional[str],
        dry_run_franka: bool,
        dry_run_hand: bool) -> Optional[np.ndarray]:
    """Reuse configured startup poses, then leave execution paused."""
    hold_robot_motion(env)
    if not dry_run_franka:
        for robot, robot_cfg in zip(env.robots, robots_config):
            reset = getattr(robot, "reset_joints", None)
            if not callable(reset):
                raise RuntimeError(
                    f"{type(robot).__name__} does not support dashboard joint reset"
                )
            if "joints_init" not in robot_cfg:
                raise RuntimeError("robot config has no joints_init for dashboard reset")
            duration = float(robot_cfg.get("joints_init_duration", 4.0))
            reset(
                np.asarray(robot_cfg["joints_init"], dtype=np.float64),
                time_to_go=duration,
                timeout=max(30.0, duration + 15.0),
            )

    hand_state = None
    if hand_backend == "linker_o6":
        init_pose = np.asarray(
            linker_o6_cfg.get("init_pose", [250] * 6), dtype=np.float32
        ).reshape(6)
        if o6_hand is not None and not dry_run_hand:
            o6_hand.move(init_pose.astype(np.uint8).tolist())
        hand_state = init_pose
    elif hand_backend == "wuji_hand":
        init_qpos = _as_qpos20(
            wuji_cfg.get("init_qpos", wuji_cfg.get("init_pose", np.zeros(20))),
            "wuji init_qpos",
        )
        if wuji_driver is not None and not dry_run_hand:
            sent = wuji_driver.send_positions(init_qpos)
            hand_state = np.asarray(
                init_qpos if sent is None else sent, dtype=np.float32
            ).reshape(20)
        else:
            hand_state = init_qpos.astype(np.float32)
        if wuji_filter is not None:
            wuji_filter.reset()
            wuji_filter.apply(hand_state)
        settle_s = float(wuji_cfg.get("init_settle_s", 0.5))
        if settle_s > 0:
            time.sleep(settle_s)
    return hand_state

def publish_dashboard(
        dashboard,
        obs_dict_np: dict,
        obs: dict,
        hand_state,
        raw_model_chunk=None,
        processed_arm_chunk=None,
        submitted_arm_chunk=None,
        processed_hand_chunk=None,
        submitted_hand_chunk=None,
        metadata: Optional[dict] = None) -> None:
    if dashboard is None:
        return
    dashboard.update(
        frames=dashboard_frames(obs_dict_np),
        raw_model_chunk=raw_model_chunk,
        processed_arm_chunk=processed_arm_chunk,
        submitted_arm_chunk=submitted_arm_chunk,
        processed_hand_chunk=processed_hand_chunk,
        submitted_hand_chunk=submitted_hand_chunk,
        observed_arm=latest_robot_pose(obs, robot_idx=0),
        observed_hand=hand_state,
        metadata=metadata,
    )
