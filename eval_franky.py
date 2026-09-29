#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Franky + 可选灵巧手真实部署推理脚本

用法：
    python eval_franky.py -c eval_franka_o6_config.yaml

也可通过命令行覆盖 yaml 中的任意参数：
    python eval_franky.py --save-episode
"""

import sys
import os
import time
import argparse
import copy
import numpy as np
import torch
import click
import dill
import hydra
import pathlib
from multiprocessing.managers import SharedMemoryManager

# ── 项目内部 ─────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.camera_factory import build_camera_from_yaml
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_obs_dict,
    get_real_umi_action,
)
from diffusion_policy.common.pytorch_util import dict_apply
from umi.common.precise_sleep import precise_wait
from inference_episode_recorder import InferenceEpisodeRecorder
from real_inference_actions import (
    _coerce_wuji_action_sequence,
    _to_numpy,
    action_state_gap_summary,
    align_recording_hand_commands,
    disable_pretrained_download,
    down_sample_steps,
    execute_timestamped_hand_commands,
    filter_obs_for_shape_meta,
    future_timestamps,
    keep_future_actions,
    latest_env_gripper_width,
    latest_obs_time,
    latest_robot_pose,
    obs_horizon,
    patch_hand_obs,
    run_timing_self_test,
    split_policy_action,
)
from real_inference_config import (
    HAND_BACKENDS,
    NoOpScalarGripper,
    _as_qpos20,
    load_config,
    load_yaml_mapping,
    merged_hand_config,
    open_o6_hand,
    open_wuji_hand,
    resolve_hand_backend,
    resolve_episode_record_fps,
)


# ── 配置加载 ──────────────────────────────────────────────────────────────────


def configure_franky_ports(robots_config: list[dict], franky_port: int) -> list[dict]:
    """Route Franka robots to the dedicated Franky chunk controller.

    The input is the already-copied runtime robot configuration, so the shared
    YAML and the original Polymetis path remain untouched.
    """
    port = int(franky_port)
    if not 1 <= port <= 65535:
        raise ValueError(f"franky_port must be in [1, 65535], got {port}")

    configured = 0
    for robot_config in robots_config:
        if str(robot_config.get("robot_type", "")).startswith("franka"):
            robot_config["robot_port"] = port
            robot_config["control_backend"] = "franky_chunk"
            robot_config["rpc_timeout"] = float(
                robot_config.get("franky_rpc_timeout", 60.0)
            )
            robot_config["launch_timeout"] = float(
                robot_config.get("franky_launch_timeout", 60.0)
            )
            robot_config["joints_init_duration"] = float(
                robot_config.get("franky_joints_init_duration", 4.0)
            )
            robot_config["franky_state_frequency"] = float(
                robot_config.get("franky_state_frequency", 100.0)
            )
            robot_config["franky_max_chunk_size"] = int(
                robot_config.get("franky_max_chunk_size", 64)
            )
            configured += 1
    if configured == 0:
        raise ValueError("No Franka robot entry found in robot configuration")
    return robots_config


# ── 模型工具 ──────────────────────────────────────────────────────────────────
def load_policy(ckpt_path: str, device: torch.device):
    """从 checkpoint 加载策略模型。"""
    payload = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    disable_pretrained_download(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(ROOT / "tmp_ws"))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model
    policy.eval().to(device)
    return policy, cfg


# ── 动作解析 ──────────────────────────────────────────────────────────────────


# ── 主推理循环 ────────────────────────────────────────────────────────────────
def run_inference(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 加载模型 ──
    print(f"[INFO] Loading checkpoint: {cfg['checkpoint']}")
    policy, model_cfg = load_policy(cfg["checkpoint"], device)

    if hasattr(policy, "action_dim"):
        action_dim = int(policy.action_dim)
    else:
        action_dim = int(model_cfg.task.shape_meta.action.shape[0])
    print(f"[INFO] action_dim = {action_dim}")

    robot_cfg = load_yaml_mapping(cfg["robot_config"])
    robots_config = copy.deepcopy(robot_cfg["robots"])
    franky_port = int(cfg.get("franky_port", 4243))
    configure_franky_ports(robots_config, franky_port)
    print(
        f"[INFO] robot backend = Franky Cartesian impedance endpoint mode; "
        f"ZeroRPC port = {franky_port} (no fallback to Polymetis 4242)"
    )
    grippers_config = copy.deepcopy(robot_cfg["grippers"])
    grippers = [NoOpScalarGripper(init_pos=0.0) for _ in robots_config]

    hand_backend = resolve_hand_backend(cfg, robot_cfg)
    linker_o6_cfg = merged_hand_config(cfg, robot_cfg, "linker_o6")
    wuji_cfg = merged_hand_config(cfg, robot_cfg, "wuji_hand")
    active_hand_cfg = {}
    if hand_backend == "linker_o6":
        active_hand_cfg = linker_o6_cfg
    elif hand_backend == "wuji_hand":
        active_hand_cfg = wuji_cfg
    dry_run_hand = bool(
        cfg.get("dry_run_hand", cfg.get("dry_run_o6", False))
        or active_hand_cfg.get("dry_run", False)
    )
    print(f"[INFO] hand = {hand_backend or 'none'}")

    save_episode = bool(cfg.get("save_episode", False))
    if save_episode and hand_backend not in HAND_BACKENDS:
        raise ValueError("--save-episode requires --hand linker_o6 or wuji_hand")
    episode_recorder = None
    if save_episode:
        episode_record_fps = resolve_episode_record_fps(cfg, robot_cfg)
        episode_recorder = InferenceEpisodeRecorder(
            output_dir=cfg["output"],
            frequency=float(cfg["frequency"]),
            hand_backend=hand_backend,
            record_fps=episode_record_fps,
        )
        print(
            "[INFO] inference episode recording enabled: "
            f"{pathlib.Path(cfg['output']) / 'inference_episodes'} "
            f"@ {episode_record_fps:.3f} Hz"
        )

    if hand_backend == "linker_o6" and action_dim != 15:
        raise ValueError(f"hand=linker_o6 requires action_dim=15, got {action_dim}")
    if hand_backend == "wuji_hand" and action_dim not in (20, 29):
        print(
            f"[WARN] hand=wuji_hand with action_dim={action_dim}; "
            "the policy must return result['wuji_command']."
        )
    if hand_backend is None and action_dim not in (10,):
        print(f"[WARN] hand disabled but checkpoint action_dim={action_dim}; hand tail will be ignored.")

    shape_meta = model_cfg.task.shape_meta
    obs_res = get_real_obs_resolution(shape_meta)
    obs_pose_repr = model_cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = model_cfg.task.pose_repr.action_pose_repr
    tx_robot1_robot0 = np.asarray(robot_cfg.get("tx_left_right", np.eye(4)), dtype=np.float64)
    ds_steps = down_sample_steps(model_cfg.task)
    action_start_delay = float(cfg.get("action_start_delay", 0.05))
    action_min_lead_time = float(cfg.get("action_min_lead_time", 0.02))
    wait_for_horizon = bool(cfg.get("wait_for_horizon", True))
    synchronize_hand_actions = bool(cfg.get("synchronize_hand_actions", True))
    profile_action_timing = bool(cfg.get("profile_action_timing", False))
    profile_action_every = max(1, int(cfg.get("profile_action_every", 1)))
    print(
        f"[INFO] action_start_delay={action_start_delay:.3f}s, "
        f"action_min_lead_time={action_min_lead_time:.3f}s, "
        f"wait_for_horizon={wait_for_horizon}, "
        f"synchronize_hand_actions={synchronize_hand_actions}"
    )
    if synchronize_hand_actions and not wait_for_horizon:
        print(
            "[WARN] synchronize_hand_actions=true waits through the hand horizon; "
            "--no-wait-for-horizon will not make the loop streaming."
        )
    if profile_action_timing:
        print(f"[INFO] profile_action_timing enabled, every {profile_action_every} control loops.")

    # ── 可选：初始化 Wuji 手 ──
    wuji_driver = None
    wuji_filter = None
    if hand_backend == "wuji_hand":
        wuji_driver, wuji_filter = open_wuji_hand(wuji_cfg, dry_run=dry_run_hand)

    # ── 可选：初始化 O6 手 ──
    o6_hand = None
    if hand_backend == "linker_o6" and not dry_run_hand:
        o6_hand = open_o6_hand(linker_o6_cfg)
        print("[INFO] O6 hand connected.")

    init_joints = bool(cfg.get("init_joints", False))
    if cfg.get("dry_run_franka", False):
        init_joints = False
        for rc in robots_config:
            if str(rc.get("robot_type", "")).startswith("franka"):
                rc["read_only"] = True
        print("[INFO] Franka dry-run: robot read_only=True, skipping exec_actions().")

    hand_state = None
    if hand_backend == "linker_o6":
        hand_state = np.asarray(linker_o6_cfg.get("init_pose", [250] * 6), dtype=np.float32).reshape(6)
    elif hand_backend == "wuji_hand":
        init_qpos = wuji_cfg.get("init_qpos", wuji_cfg.get("init_pose", np.zeros(20)))
        hand_state = _as_qpos20(init_qpos, "wuji init_qpos").astype(np.float32)

    enable_multi_cam_vis = bool(
        cfg.get("enable_multi_cam_vis", bool(os.environ.get("DISPLAY")))
    )
    if not enable_multi_cam_vis:
        print(
            "[INFO] Multi-camera Qt visualization disabled "
            "(enable_multi_cam_vis=false or DISPLAY is unavailable)."
        )

    try:
        with SharedMemoryManager() as shm_manager:
            cameras_cfg = robot_cfg.get("cameras") or {}
            camera, camera_capture_fps, camera_obs_latency = build_camera_from_yaml(
                shm_manager=shm_manager,
                cameras_cfg=cameras_cfg,
                obs_image_resolution=obs_res,
                camera_obs_latency=0.17,
            )

            env = BimanualUmiEnv(
                output_dir=cfg["output"],
                robots_config=robots_config,
                grippers_config=grippers_config,
                grippers=grippers,
                force_sensors_config=robot_cfg.get("force_sensors"),
                frequency=int(cfg["frequency"]),
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_obs_latency=camera_obs_latency,
                camera_capture_fps=camera_capture_fps,
                camera=camera,
                camera_down_sample_steps=ds_steps,
                robot_down_sample_steps=ds_steps,
                gripper_down_sample_steps=ds_steps,
                force_down_sample_steps=ds_steps,
                camera_obs_horizon=obs_horizon(shape_meta, "camera0_rgb", 2),
                robot_obs_horizon=obs_horizon(shape_meta, "robot0_eef_pos", 2),
                gripper_obs_horizon=obs_horizon(shape_meta, "robot0_gripper_width", 2),
                force_obs_horizon=obs_horizon(
                    shape_meta,
                    "robot0_wrench",
                    obs_horizon(shape_meta, "robot0_eef_pos", 2),
                ),
                max_pos_speed=float(cfg["max_pos_speed"]),
                max_rot_speed=float(cfg["max_rot_speed"]),
                init_joints=init_joints,
                enable_multi_cam_vis=enable_multi_cam_vis,
                shm_manager=shm_manager,
            )

            with env:
                if o6_hand is not None and cfg.get("init_joints", False):
                    o6_init = linker_o6_cfg.get("init_pose", [250] * 6)
                    print(f"[INFO] Initializing O6 hand to {o6_init} ...")
                    o6_hand.move(o6_init)
                    hand_state = np.asarray(o6_init, dtype=np.float32).reshape(6)
                if wuji_driver is not None and cfg.get("init_joints", False):
                    wuji_init = _as_qpos20(
                        wuji_cfg.get("init_qpos", wuji_cfg.get("init_pose", np.zeros(20))),
                        "wuji init_qpos",
                    )
                    print(f"[INFO] Initializing Wuji hand to {np.round(wuji_init, 4).tolist()} ...")
                    sent = wuji_driver.send_positions(wuji_init)
                    hand_state = np.asarray(sent, dtype=np.float32).reshape(20)
                    if wuji_filter is not None:
                        wuji_filter.reset()
                        wuji_filter.apply(hand_state)
                    settle_s = float(wuji_cfg.get("init_settle_s", 0.5))
                    if settle_s > 0:
                        time.sleep(settle_s)

                click.echo("Ready. Press SPACE to start episode, 'q' to quit.")

                episode = 0
                while True:
                    key = click.getchar()
                    if key == "q":
                        print("[INFO] Quit.")
                        break
                    if key != " ":
                        continue

                    episode += 1
                    print(f"\n{'='*50}\nEpisode {episode}\n{'='*50}")

                    if episode_recorder is not None:
                        episode_recorder.start_episode({
                            "checkpoint": str(cfg["checkpoint"]),
                            "robotConfig": str(cfg["robot_config"]),
                            "policyActionDim": int(action_dim),
                            "stepsPerInference": int(cfg["steps_per_inference"]),
                        })

                    raw_obs = env.get_obs()
                    env_gripper_width = latest_env_gripper_width(raw_obs)
                    if o6_hand is not None:
                        try:
                            hand_state = np.asarray(o6_hand.get_state(), dtype=np.float32).reshape(6)
                        except Exception as exc:
                            print(f"[WARN] O6 get_state failed, using last command: {exc}")
                    if wuji_driver is not None and not dry_run_hand:
                        try:
                            hand_state = wuji_driver.read_positions().astype(np.float32)
                        except Exception as exc:
                            print(f"[WARN] Wuji read_positions failed, using last command: {exc}")
                    obs = patch_hand_obs(raw_obs, hand_state, shape_meta)
                    episode_start_pose = []
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f"robot{robot_id}_eef_pos"],
                            obs[f"robot{robot_id}_eef_rot_axis_angle"],
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    policy.reset()
                    t_start = time.monotonic()
                    step = 0
                    max_steps = int(cfg.get("max_timesteps", 3000))
                    max_duration = cfg.get("max_duration")
                    steps_per_infer = int(cfg["steps_per_inference"])
                    dt = 1.0 / float(cfg["frequency"])
                    profile_prev_last_submitted_ts = None
                    episode_finish_reason = "max_timesteps"

                    while step < max_steps:
                        should_profile = profile_action_timing and (step % profile_action_every == 0)
                        profile_t0 = time.time()
                        if max_duration and (time.monotonic() - t_start) > float(max_duration):
                            print("[INFO] Max duration reached.")
                            episode_finish_reason = "max_duration"
                            break
                        cycle_start_wall = time.time()
                        profile_obs_time = latest_obs_time(obs)
                        profile_curr_pose = latest_robot_pose(obs, robot_idx=0)

                        profile_obs_start = time.time()
                        obs_dict_np = get_real_umi_obs_dict(
                            env_obs=obs,
                            shape_meta=shape_meta,
                            obs_pose_repr=obs_pose_repr,
                            tx_robot1_robot0=tx_robot1_robot0,
                            episode_start_pose=episode_start_pose,
                        )
                        obs_dict_np = filter_obs_for_shape_meta(obs_dict_np, shape_meta)
                        profile_obs_dict_done = time.time()
                        obs_dict = dict_apply(
                            obs_dict_np,
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(device),
                        )
                        profile_tensor_done = time.time()

                        with torch.no_grad():
                            profile_policy_start = time.time()
                            result = policy.predict_action(obs_dict)
                            profile_policy_done = time.time()

                        exec_steps = min(steps_per_infer, max_steps - step)
                        has_wuji_command = hand_backend == "wuji_hand" and "wuji_command" in result
                        if "action" in result:
                            action_tensor = result["action"]
                        elif "action_pred" in result:
                            action_tensor = result["action_pred"]
                        elif has_wuji_command:
                            action_tensor = None
                        else:
                            raise RuntimeError("Unknown model output keys: " + str(result.keys()))

                        if action_tensor is None:
                            pred = None
                        else:
                            pred = _to_numpy(action_tensor)
                            if pred.ndim == 3:
                                pred = pred[0]
                        arm_pred = None if pred is None else pred

                        if has_wuji_command and (pred is None or action_dim not in (10, 15, 20, 29)):
                            arm_action = None
                            hand_actions = _coerce_wuji_action_sequence(
                                result["wuji_command"], exec_steps
                            )
                        else:
                            arm_action, hand_actions = split_policy_action(
                                pred[:exec_steps],
                                action_dim,
                                env_gripper_width,
                                hand_backend,
                            )
                            if has_wuji_command:
                                hand_actions = _coerce_wuji_action_sequence(
                                    result["wuji_command"], exec_steps
                                )
                        if hand_backend == "wuji_hand" and hand_actions is None:
                            raise RuntimeError(
                                "hand=wuji_hand requires action_dim 20/29 or policy result['wuji_command']"
                            )

                        if arm_action is not None:
                            profile_action_start = time.time()
                            if arm_pred is not None:
                                arm_action_for_franka, _ = split_policy_action(
                                    arm_pred,
                                    action_dim,
                                    env_gripper_width,
                                    hand_backend,
                                )
                            else:
                                arm_action_for_franka = arm_action
                            franka_actions = get_real_umi_action(
                                arm_action_for_franka,
                                obs,
                                action_pose_repr=action_pose_repr,
                            )
                            profile_action_done = time.time()
                            timestamps = future_timestamps(
                                obs,
                                len(franka_actions),
                                cfg["frequency"],
                                start_delay=action_start_delay,
                            )
                            submitted_actions, submitted_timestamps, kept_mask = keep_future_actions(
                                franka_actions,
                                timestamps,
                                min_lead_time=action_min_lead_time,
                                fallback_delay=action_start_delay,
                            )
                            profile_schedule_done = time.time()
                            profile_submit_time = time.time()
                            profile_submit_lead = submitted_timestamps - profile_submit_time
                            profile_segment_gap = None
                            if profile_prev_last_submitted_ts is not None and len(submitted_timestamps) > 0:
                                profile_segment_gap = float(submitted_timestamps[0] - profile_prev_last_submitted_ts)
                            if not cfg.get("dry_run_franka", False):
                                env.exec_actions(actions=submitted_actions, timestamps=submitted_timestamps)
                            profile_exec_done = time.time()
                        else:
                            profile_action_start = profile_action_done = time.time()
                            timestamps = future_timestamps(
                                obs,
                                exec_steps,
                                cfg["frequency"],
                                start_delay=action_start_delay,
                            )
                            submitted_actions = None
                            submitted_timestamps = timestamps
                            kept_mask = np.ones(len(timestamps), dtype=bool)
                            profile_schedule_done = profile_exec_done = time.time()
                            profile_submit_time = profile_schedule_done
                            profile_submit_lead = submitted_timestamps - profile_submit_time
                            profile_segment_gap = None
                            if profile_prev_last_submitted_ts is not None and len(submitted_timestamps) > 0:
                                profile_segment_gap = float(submitted_timestamps[0] - profile_prev_last_submitted_ts)

                        hand_timestamps, scheduled_hand_actions = align_recording_hand_commands(
                            hand_actions,
                            kept_mask,
                            submitted_timestamps,
                        )
                        profile_hand_start = time.time()
                        executed_hand_actions = None
                        measured_hand_states = None
                        if scheduled_hand_actions is not None and hand_backend == "linker_o6":
                            def send_o6(command):
                                normalized = np.clip(np.rint(command), 0, 255).astype(np.uint8)
                                if o6_hand is not None:
                                    o6_hand.move(normalized.tolist())
                                return normalized

                            read_o6_state = None
                            if o6_hand is not None:
                                def read_o6_state():
                                    state = np.asarray(
                                        o6_hand.get_state(), dtype=np.float32
                                    ).reshape(6)
                                    if not np.all(np.isfinite(state)) or np.any(state < 0):
                                        return None
                                    return state
                            executed_hand_actions, measured_hand_states = (
                                execute_timestamped_hand_commands(
                                    hand_timestamps,
                                    scheduled_hand_actions,
                                    send_command=send_o6,
                                    read_state=read_o6_state,
                                    synchronize=synchronize_hand_actions,
                                )
                            )
                            if measured_hand_states is not None:
                                hand_state = measured_hand_states[-1].astype(np.float32)
                            elif executed_hand_actions is not None:
                                hand_state = executed_hand_actions[-1].astype(np.float32)
                        elif scheduled_hand_actions is not None and hand_backend == "wuji_hand":
                            if wuji_driver is None or wuji_filter is None:
                                raise RuntimeError("Wuji backend selected but runtime is not initialized")

                            def send_wuji(qpos):
                                qpos = wuji_driver.clamp(_as_qpos20(qpos))
                                if bool(wuji_cfg.get("filter_enabled", True)):
                                    qpos = wuji_filter.apply(qpos)
                                sent_qpos = wuji_driver.send_positions(qpos)
                                return qpos if sent_qpos is None else sent_qpos

                            executed_hand_actions, measured_hand_states = (
                                execute_timestamped_hand_commands(
                                    hand_timestamps,
                                    scheduled_hand_actions,
                                    send_command=send_wuji,
                                    synchronize=synchronize_hand_actions,
                                )
                            )
                            if executed_hand_actions is not None:
                                hand_state = executed_hand_actions[-1].astype(np.float32)
                        profile_hand_done = time.time()

                        step += exec_steps
                        if wait_for_horizon:
                            wait_until = cycle_start_wall + exec_steps * dt
                            profile_wait_start = time.time()
                            precise_wait(wait_until, time_func=time.time)
                            profile_wait_done = time.time()
                        else:
                            profile_wait_start = profile_wait_done = time.time()
                        profile_get_obs_start = time.time()
                        raw_obs = env.get_obs()
                        profile_get_obs_done = time.time()
                        env_gripper_width = latest_env_gripper_width(raw_obs)
                        if o6_hand is not None and not dry_run_hand:
                            try:
                                hand_state = np.asarray(
                                    o6_hand.get_state(), dtype=np.float32
                                ).reshape(6)
                            except Exception as exc:
                                print(f"[WARN] O6 get_state failed, using last state: {exc}")
                        if wuji_driver is not None and not dry_run_hand:
                            try:
                                hand_state = wuji_driver.read_positions().astype(np.float32)
                            except Exception as exc:
                                print(f"[WARN] Wuji read_positions failed, using last command: {exc}")
                        obs = patch_hand_obs(raw_obs, hand_state, shape_meta)
                        if episode_recorder is not None:
                            if executed_hand_actions is not None:
                                try:
                                    episode_recorder.record_segment(
                                        env=env,
                                        timestamps=hand_timestamps,
                                        hand_commands=executed_hand_actions,
                                        measured_hand_states=measured_hand_states,
                                    )
                                except Exception as exc:
                                    print(f"[WARN] inference episode recorder delayed: {exc}")
                        if should_profile:
                            now = time.time()
                            lead = submitted_timestamps - now if submitted_timestamps is not None else np.array([])
                            lead_msg = "[]"
                            if len(lead) > 0:
                                lead_msg = f"[{lead[0]:.3f}, {lead[-1]:.3f}]"
                            submit_lead_msg = "[]"
                            if len(profile_submit_lead) > 0:
                                submit_lead_msg = f"[{profile_submit_lead[0]:.3f}, {profile_submit_lead[-1]:.3f}]"
                            submitted_count = 0 if submitted_actions is None else len(submitted_actions)
                            dropped_count = int(len(kept_mask) - np.count_nonzero(kept_mask))
                            obs_age_msg = "n/a"
                            if profile_obs_time is not None:
                                obs_age_msg = f"{profile_t0 - profile_obs_time:.3f}s"
                            segment_gap_msg = "n/a"
                            if profile_segment_gap is not None:
                                segment_gap_msg = f"{profile_segment_gap:.3f}s"
                            next_pose = latest_robot_pose(obs, robot_idx=0)
                            gap_msg = action_state_gap_summary(
                                submitted_actions,
                                profile_curr_pose,
                                next_pose,
                                robot_idx=0,
                            )
                            print(
                                "[PROFILE Franka] "
                                f"step={step} exec_steps={exec_steps} "
                                f"obs_age={obs_age_msg} "
                                f"obs_dict={profile_obs_dict_done - profile_obs_start:.3f}s "
                                f"tensor={profile_tensor_done - profile_obs_dict_done:.3f}s "
                                f"policy={profile_policy_done - profile_policy_start:.3f}s "
                                f"action={profile_action_done - profile_action_start:.3f}s "
                                f"schedule={profile_schedule_done - profile_action_done:.3f}s "
                                f"exec={profile_exec_done - profile_schedule_done:.3f}s "
                                f"hand={profile_hand_done - profile_hand_start:.3f}s "
                                f"wait={profile_wait_done - profile_wait_start:.3f}s "
                                f"get_obs={profile_get_obs_done - profile_get_obs_start:.3f}s "
                                f"loop={time.time() - profile_t0:.3f}s "
                                f"submitted={submitted_count}/{len(timestamps)} "
                                f"dropped={dropped_count} "
                                f"segment_gap_s={segment_gap_msg} "
                                f"submit_lead_s={submit_lead_msg} end_lead_s={lead_msg} "
                                f"{gap_msg}"
                            )
                        if submitted_timestamps is not None and len(submitted_timestamps) > 0:
                            profile_prev_last_submitted_ts = float(submitted_timestamps[-1])

                    if episode_recorder is not None:
                        episode_recorder.finish_episode(
                            env=env,
                            reason=episode_finish_reason,
                        )
                    print(f"[INFO] Episode {episode} finished. Steps: {step}")
    finally:
        if episode_recorder is not None and episode_recorder.active:
            try:
                episode_recorder.finish_episode(reason="exception_or_exit")
            except Exception as exc:
                print(f"[WARN] failed to finalize inference episode: {exc}")
        if o6_hand is not None:
            o6_hand.close()
        if wuji_driver is not None:
            wuji_driver.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Franky + 可选灵巧手推理脚本")
    parser.add_argument("-c", "--config", default="eval_franka_o6_config.yaml",
                        help="YAML 配置文件路径")
    # 允许命令行覆盖常用参数
    parser.add_argument("-i", "--checkpoint",   default=None, help="checkpoint 路径")
    parser.add_argument("-o", "--output",        default=None, help="输出目录")
    parser.add_argument("-rc","--robot-config",  default=None, dest="robot_config")
    parser.add_argument("--franky-port", default=4243, type=int, dest="franky_port",
                        help="Franky ZeroRPC port; defaults to 4243 and never falls back to 4242")
    parser.add_argument("--timing-self-test", action="store_true",
                        help="Run local Franka timestamp scheduling checks and exit")
    parser.add_argument("--hand", choices=["linker_o6", "wuji_hand", "wujihand", "none"],
                        default=None, help="选择灵巧手: linker_o6 / wuji_hand / none")
    parser.add_argument("-f", "--frequency",     default=None, type=int)
    parser.add_argument("-si","--steps-per-inference", default=None, type=int,
                        dest="steps_per_inference")
    parser.add_argument("--max-timesteps",  default=None, type=int, dest="max_timesteps")
    parser.add_argument("--max-duration",   default=None, type=float, dest="max_duration")
    parser.add_argument("--max-pos-speed",  default=None, type=float, dest="max_pos_speed")
    parser.add_argument("--max-rot-speed",  default=None, type=float, dest="max_rot_speed")
    parser.add_argument("--action-start-delay", default=None, type=float,
                        dest="action_start_delay",
                        help="Franka waypoint timestamps start this many seconds in the future")
    parser.add_argument("--action-min-lead-time", default=None, type=float,
                        dest="action_min_lead_time",
                        help="Discard/reschedule waypoints closer than this many seconds")
    parser.add_argument("--no-wait-for-horizon", action="store_false",
                        dest="wait_for_horizon", default=None,
                        help="Do not wait for the submitted action horizon before the next inference")
    parser.add_argument("--no-synchronize-hand-actions", action="store_false",
                        dest="synchronize_hand_actions", default=None,
                        help="Use legacy immediate hand command submission")
    parser.add_argument("--profile-action-timing", action="store_true",
                        dest="profile_action_timing", default=None,
                        help="Print timing diagnostics for inference/action scheduling")
    parser.add_argument("--profile-action-every", type=int, default=None,
                        dest="profile_action_every",
                        help="Print timing diagnostics every N executed steps")
    parser.add_argument("--init",           action="store_true", dest="init_joints",
                        default=None)
    parser.add_argument("--dry-run-franka", action="store_true", dest="dry_run_franka",
                        default=None)
    parser.add_argument("--dry-run-o6",     action="store_true", dest="dry_run_o6",
                        default=None)
    parser.add_argument("--dry-run-hand",   action="store_true", dest="dry_run_hand",
                        default=None)
    parser.add_argument("--save-episode", action="store_true", dest="save_episode",
                        default=None,
                        help="Save each inference rollout as PKL + JPG episode data")
    parser.add_argument("--episode-record-fps", "--save-episode-fps",
                        dest="episode_record_fps", default=None, type=float,
                        help="Fixed FPS for --save-episode JPG/PKL samples; defaults to camera capture_fps or 30")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    if args.timing_self_test:
        run_timing_self_test()
        return
    cfg = load_config(args.config, overrides)

    # 创建输出目录
    os.makedirs(cfg["output"], exist_ok=True)

    run_inference(cfg)


if __name__ == "__main__":
    main()
