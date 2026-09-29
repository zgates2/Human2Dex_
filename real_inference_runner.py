#!/usr/bin/env python3
"""Franka plus dexterous-hand inference lifecycle."""

from __future__ import annotations

import copy
import pathlib
import select
import sys
import termios
import time
import tty
from contextlib import nullcontext
from multiprocessing.managers import SharedMemoryManager

import click
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from inference_episode_recorder import InferenceEpisodeRecorder
from real_inference_actions import (
    _coerce_wuji_action_sequence,
    _to_numpy,
    action_state_gap_summary,
    align_recording_hand_commands,
    down_sample_steps,
    filter_obs_for_shape_meta,
    future_timestamps,
    interruptible_wait_until,
    keep_future_actions,
    latest_env_gripper_width,
    latest_obs_time,
    latest_robot_pose,
    obs_horizon,
    patch_hand_obs,
    scale_franka_negative_x_delta,
    scale_franka_negative_y_delta,
    scale_franka_z_delta,
    split_policy_action,
    validate_hand_checkpoint_compatibility,
)
from real_inference_config import (
    HAND_BACKENDS,
    NoOpScalarGripper,
    _as_qpos20,
    load_yaml_mapping,
    merged_hand_config,
    open_o6_hand,
    open_wuji_hand,
    resolve_hand_backend,
    resolve_episode_record_fps,
)
from real_inference_dashboard import (
    dashboard_rgb_inputs_from_env_obs,
    hold_robot_motion,
    publish_dashboard,
    reset_robot_and_hand,
)
from real_inference_hands import execute_hand_actions, read_hand_state
from real_inference_policy import load_policy
from real_inference_pts21 import build_pts21_action_adapter
from real_inference_skeleton_overlay import build_skeleton_overlay_renderer
from real_inference_canonical_views import build_canonical_views_renderer
from real_inference_object_pocket import build_object_pocket_observer
from real_inference_view_alignment import build_view_canonicalizer
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.camera_factory import build_camera_from_yaml
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_action,
    get_real_umi_obs_dict,
)


class _OperatorKeyPoller:
    """Non-blocking terminal controls for the local CLI inference loop."""

    def __init__(self):
        self._fd = None
        self._old_termios = None
        self.enabled = False
        self._episode_command = None

    def __enter__(self):
        stream = sys.stdin
        if stream is None or not hasattr(stream, "fileno") or not stream.isatty():
            return self
        try:
            self._fd = stream.fileno()
            self._old_termios = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        except termios.error as exc:
            self._fd = None
            self._old_termios = None
            self.enabled = False
            print(f"[WARN] Non-blocking keyboard controls disabled: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None and self._old_termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        self.enabled = False

    def _read_chars(self) -> list[str]:
        if not self.enabled or self._fd is None:
            return []
        chars = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                break
            char = sys.stdin.read(1)
            if char == "":
                break
            chars.append(char)
        return chars

    def clear(self) -> None:
        self._episode_command = None
        self._read_chars()

    def wait_for_start_or_quit(self) -> bool:
        if not self.enabled:
            while True:
                key = click.getchar()
                if key.lower() == "q":
                    print("[INFO] Quit.")
                    return False
                if key == " ":
                    return True

        self.clear()
        while True:
            for char in self._read_chars():
                if char == "\x03":
                    raise KeyboardInterrupt
                if char.lower() == "q":
                    print("[INFO] Quit.")
                    return False
                if char == " ":
                    return True
            time.sleep(0.02)

    def poll_episode_command(self):
        for char in self._read_chars():
            if char == "\x03":
                raise KeyboardInterrupt
            key = char.lower()
            if key == "q":
                self._episode_command = "quit"
            elif key == "r" and self._episode_command != "quit":
                self._episode_command = "reset"
        return self._episode_command

    def should_continue_episode(self) -> bool:
        return self.poll_episode_command() is None

    def consume_episode_command(self):
        command = self.poll_episode_command()
        self._episode_command = None
        return command


def _reset_dashboard_runtime(
        *,
        dashboard,
        env,
        policy,
        hand_state,
        robots_config,
        o6_hand,
        linker_o6_cfg,
        wuji_driver,
        wuji_filter,
        wuji_cfg,
        hand_backend,
        dry_run_franka,
        dry_run_hand):
    reset_state = reset_robot_and_hand(
        env=env,
        robots_config=robots_config,
        o6_hand=o6_hand,
        linker_o6_cfg=linker_o6_cfg,
        wuji_driver=wuji_driver,
        wuji_filter=wuji_filter,
        wuji_cfg=wuji_cfg,
        hand_backend=hand_backend,
        dry_run_franka=dry_run_franka,
        dry_run_hand=dry_run_hand,
    )
    if reset_state is not None:
        hand_state = reset_state
    policy.reset()
    if dashboard is not None:
        dashboard.mark_ready("复位完成，等待开始推理")
    else:
        print("[INFO] Reset complete. Press SPACE to start inference again, 'q' to quit.")
    return hand_state


def _wait_for_episode_start(
        *,
        dashboard,
        env,
        policy,
        hand_state,
        robots_config,
        o6_hand,
        linker_o6_cfg,
        wuji_driver,
        wuji_filter,
        wuji_cfg,
        hand_backend,
        dry_run_franka,
        dry_run_hand,
        pts21_action_adapter,
        skeleton_overlay_renderer,
        canonical_views_renderer,
        object_pocket_observer,
        view_canonicalizer,
        shape_meta,
        action_dim,
        frequency,
        keyboard_control=None):
    if dashboard is None:
        if keyboard_control is None:
            keyboard_control = _OperatorKeyPoller()
        return keyboard_control.wait_for_start_or_quit(), hand_state

    while not dashboard.can_execute():
        if dashboard.stop_pending():
            print("[INFO] Dashboard requested stop.")
            break
        if dashboard.consume_reset_request():
            dashboard.mark_resetting()
            try:
                hand_state = _reset_dashboard_runtime(
                    dashboard=dashboard,
                    env=env,
                    policy=policy,
                    hand_state=hand_state,
                    robots_config=robots_config,
                    o6_hand=o6_hand,
                    linker_o6_cfg=linker_o6_cfg,
                    wuji_driver=wuji_driver,
                    wuji_filter=wuji_filter,
                    wuji_cfg=wuji_cfg,
                    hand_backend=hand_backend,
                    dry_run_franka=dry_run_franka,
                    dry_run_hand=dry_run_hand,
                )
                if pts21_action_adapter is not None:
                    pts21_action_adapter.reset()
                if canonical_views_renderer is not None:
                    canonical_views_renderer.reset()
                if object_pocket_observer is not None:
                    object_pocket_observer.reset()
            except Exception as exc:
                dashboard.mark_error(f"复位失败：{exc}")
                print(f"[ERROR] Dashboard reset failed: {exc}")

        try:
            preview_raw_obs = env.get_obs()
            hand_state = read_hand_state(
                hand_state,
                o6_hand,
                wuji_driver,
                dry_run_hand,
                o6_warning="O6 preview get_state failed",
                wuji_warning="Wuji preview read_positions failed",
            )
            if pts21_action_adapter is None:
                preview_obs = patch_hand_obs(preview_raw_obs, hand_state, shape_meta)
            else:
                preview_obs = pts21_action_adapter.patch_observation(
                    preview_raw_obs, shape_meta
                )
            if object_pocket_observer is not None:
                preview_obs = object_pocket_observer.apply(preview_obs, hand_state)
            if skeleton_overlay_renderer is not None:
                preview_obs = skeleton_overlay_renderer.apply(preview_obs, hand_state)
            if canonical_views_renderer is not None:
                preview_obs = canonical_views_renderer.apply(preview_obs, hand_state)
            if view_canonicalizer is not None:
                preview_obs = view_canonicalizer.apply(preview_obs)
            preview_obs_dict = dashboard_rgb_inputs_from_env_obs(
                preview_obs, shape_meta
            )
            publish_dashboard(
                dashboard,
                preview_obs_dict,
                preview_obs,
                hand_state,
                metadata={
                    "mode": "paused_preview",
                    "action_dim": int(action_dim),
                    "hand_backend": hand_backend or "none",
                    "frequency_hz": float(frequency),
                },
            )
        except Exception as exc:
            dashboard.mark_error(f"暂停预览失败：{exc}")
            print(f"[ERROR] Dashboard preview failed: {exc}")
        time.sleep(0.1)

    if dashboard.stop_pending():
        return False, hand_state
    dashboard.mark_running()
    return True, hand_state


def _complete_dashboard_episode(
        *,
        dashboard,
        episode,
        reset_after_episode,
        env,
        policy,
        hand_state,
        robots_config,
        o6_hand,
        linker_o6_cfg,
        wuji_driver,
        wuji_filter,
        wuji_cfg,
        hand_backend,
        dry_run_franka,
        dry_run_hand):
    if dashboard is not None and dashboard.stop_pending():
        return True, hand_state
    if reset_after_episode:
        try:
            hand_state = _reset_dashboard_runtime(
                dashboard=dashboard,
                env=env,
                policy=policy,
                hand_state=hand_state,
                robots_config=robots_config,
                o6_hand=o6_hand,
                linker_o6_cfg=linker_o6_cfg,
                wuji_driver=wuji_driver,
                wuji_filter=wuji_filter,
                wuji_cfg=wuji_cfg,
                hand_backend=hand_backend,
                dry_run_franka=dry_run_franka,
                dry_run_hand=dry_run_hand,
            )
        except Exception as exc:
            if dashboard is not None:
                dashboard.mark_error(f"复位失败：{exc}")
            print(f"[ERROR] Runtime reset failed: {exc}")
            if dashboard is None:
                raise
    else:
        if dashboard is not None:
            dashboard.mark_ready(f"Episode {episode} 已结束，等待再次开始")
    return False, hand_state


def _consume_operator_episode_command(keyboard_control, env, hold_sent: bool):
    if keyboard_control is None:
        return None, hold_sent
    command = keyboard_control.consume_episode_command()
    if command is None:
        return None, hold_sent
    if not hold_sent:
        hold_robot_motion(env)
        hold_sent = True
    if command == "reset":
        print("[INFO] Operator requested reset with 'r'.")
    elif command == "quit":
        print("[INFO] Operator requested quit with 'q'.")
    return command, hold_sent


def run_inference(cfg: dict, dashboard=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 加载模型 ──
    print(f"[INFO] Loading checkpoint: {cfg['checkpoint']}")
    policy, model_cfg = load_policy(cfg["checkpoint"], device)

    if hasattr(policy, "action_dim"):
        action_dim = int(policy.action_dim)
    else:
        action_dim = int(model_cfg.task.shape_meta.action.shape[0])
    print(f"[INFO] action_dim = {action_dim}")
    shape_meta = model_cfg.task.shape_meta

    robot_cfg = load_yaml_mapping(cfg["robot_config"])
    robots_config = copy.deepcopy(robot_cfg["robots"])
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
    hand_action_mode = str(cfg.get("hand_action_mode", "direct_command")).strip().lower()
    pts21_mode = hand_action_mode in ("pts21", "pts21_mano", "mono_keypoints")
    pts21_action_adapter = None
    hand_command_representation = str(
        cfg.get("hand_command_representation", "auto")
    ).strip().lower()
    if hand_command_representation == "auto":
        hand_command_representation = str(
            model_cfg.task.get("hand_action_representation", "absolute")
        ).strip().lower()
    if hand_command_representation not in ("absolute", "delta_from_current"):
        raise ValueError(
            "hand_command_representation must be auto, absolute, or "
            f"delta_from_current, got {hand_command_representation!r}"
        )
    if hand_command_representation == "delta_from_current" and hand_backend != "linker_o6":
        raise ValueError("delta_from_current hand commands currently support linker_o6 only")
    print(f"[INFO] hand_command_representation = {hand_command_representation}")

    task_name = str(model_cfg.task.get("name", "<unknown>"))
    for warning in validate_hand_checkpoint_compatibility(
            hand_backend=hand_backend,
            action_dim=action_dim,
            shape_meta=shape_meta,
            pts21_mode=pts21_mode,
            task_name=task_name):
        print(f"[WARN] {warning}")

    skeleton_overlay_renderer = build_skeleton_overlay_renderer(cfg, hand_backend)
    canonical_views_renderer = build_canonical_views_renderer(cfg, hand_backend)
    view_canonicalizer = build_view_canonicalizer(cfg)
    if canonical_views_renderer is not None and view_canonicalizer is not None:
        raise ValueError(
            "canonical_views and legacy view_alignment cannot both be enabled; "
            "the former already defines the policy camera geometry"
        )

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

    object_pocket_observer = build_object_pocket_observer(cfg, hand_backend, shape_meta)
    if canonical_views_renderer is not None:
        canonical_views_renderer.validate_shape_meta(shape_meta)
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
    franka_neg_x_delta_gain = float(cfg.get("franka_neg_x_delta_gain", 1.0))
    if not np.isfinite(franka_neg_x_delta_gain) or franka_neg_x_delta_gain <= 0.0:
        raise ValueError(
            "franka_neg_x_delta_gain must be finite and > 0, "
            f"got {franka_neg_x_delta_gain!r}"
        )
    franka_neg_y_delta_gain = float(cfg.get("franka_neg_y_delta_gain", 1.0))
    if not np.isfinite(franka_neg_y_delta_gain) or franka_neg_y_delta_gain <= 0.0:
        raise ValueError(
            "franka_neg_y_delta_gain must be finite and > 0, "
            f"got {franka_neg_y_delta_gain!r}"
        )
    franka_z_delta_gain = float(cfg.get("franka_z_delta_gain", 1.0))
    if not np.isfinite(franka_z_delta_gain) or franka_z_delta_gain <= 0.0:
        raise ValueError(
            "franka_z_delta_gain must be finite and > 0, "
            f"got {franka_z_delta_gain!r}"
        )
    print(
        f"[INFO] action_start_delay={action_start_delay:.3f}s, "
        f"action_min_lead_time={action_min_lead_time:.3f}s, "
        f"wait_for_horizon={wait_for_horizon}, "
        f"synchronize_hand_actions={synchronize_hand_actions}"
    )
    if franka_neg_x_delta_gain != 1.0:
        print(f"[INFO] franka_neg_x_delta_gain={franka_neg_x_delta_gain:.3f}")
    if franka_neg_y_delta_gain != 1.0:
        print(f"[INFO] franka_neg_y_delta_gain={franka_neg_y_delta_gain:.3f}")
    if franka_z_delta_gain != 1.0:
        print(f"[INFO] franka_z_delta_gain={franka_z_delta_gain:.3f}")
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
    try:
        pts21_action_adapter = build_pts21_action_adapter(
            cfg,
            policy=policy,
            action_dim=action_dim,
            hand_backend=hand_backend,
        )
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
                shm_manager=shm_manager,
            )

            keyboard_control = None if dashboard is not None else _OperatorKeyPoller()
            keyboard_context = nullcontext() if keyboard_control is None else keyboard_control

            with env, keyboard_context:
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

                if dashboard is None:
                    click.echo(
                        "Ready. Press SPACE to start episode, "
                        "'r' during inference to reset, 'q' to quit."
                    )
                else:
                    dashboard.mark_ready("硬件就绪，等待网页点击开始推理")
                    print("[INFO] Dashboard control ready; inference starts paused.")

                episode = 0
                while True:
                    should_start, hand_state = _wait_for_episode_start(
                        dashboard=dashboard,
                        env=env,
                        policy=policy,
                        hand_state=hand_state,
                        robots_config=robots_config,
                        o6_hand=o6_hand,
                        linker_o6_cfg=linker_o6_cfg,
                        wuji_driver=wuji_driver,
                        wuji_filter=wuji_filter,
                        wuji_cfg=wuji_cfg,
                        hand_backend=hand_backend,
                        dry_run_franka=bool(cfg.get("dry_run_franka", False)),
                        dry_run_hand=dry_run_hand,
                        pts21_action_adapter=pts21_action_adapter,
                        skeleton_overlay_renderer=skeleton_overlay_renderer,
                        canonical_views_renderer=canonical_views_renderer,
                        object_pocket_observer=object_pocket_observer,
                        view_canonicalizer=view_canonicalizer,
                        shape_meta=shape_meta,
                        action_dim=action_dim,
                        frequency=cfg["frequency"],
                        keyboard_control=keyboard_control,
                    )
                    if not should_start:
                        break

                    episode += 1
                    print(f"\n{'='*50}\nEpisode {episode}\n{'='*50}")

                    if pts21_action_adapter is not None:
                        pts21_action_adapter.reset()
                    if canonical_views_renderer is not None:
                        canonical_views_renderer.reset()
                    if object_pocket_observer is not None:
                        object_pocket_observer.reset()

                    if episode_recorder is not None:
                        episode_recorder.start_episode({
                            "checkpoint": str(cfg["checkpoint"]),
                            "robotConfig": str(cfg["robot_config"]),
                            "policyActionDim": int(action_dim),
                            "stepsPerInference": int(cfg["steps_per_inference"]),
                            "viewAlignment": (
                                {"enabled": False}
                                if view_canonicalizer is None
                                else view_canonicalizer.metadata(obs_res)
                            ),
                            "skeletonOverlay": (
                                {"enabled": False}
                                if skeleton_overlay_renderer is None
                                else skeleton_overlay_renderer.metadata()
                            ),
                            "canonicalViews": (
                                {"enabled": False}
                                if canonical_views_renderer is None
                                else canonical_views_renderer.metadata()
                            ),
                            "objectPocketObs": (
                                {"enabled": False}
                                if object_pocket_observer is None
                                else object_pocket_observer.metadata()
                            ),
                        })

                    raw_obs = env.get_obs()
                    env_gripper_width = latest_env_gripper_width(raw_obs)
                    hand_state = read_hand_state(
                        hand_state,
                        o6_hand,
                        wuji_driver,
                        dry_run_hand,
                        o6_warning="O6 get_state failed, using last command",
                        wuji_warning="Wuji read_positions failed, using last command",
                    )
                    if pts21_action_adapter is None:
                        obs = patch_hand_obs(raw_obs, hand_state, shape_meta)
                    else:
                        obs = pts21_action_adapter.patch_observation(raw_obs, shape_meta)
                    if object_pocket_observer is not None:
                        obs = object_pocket_observer.apply(obs, hand_state)
                    if skeleton_overlay_renderer is not None:
                        obs = skeleton_overlay_renderer.apply(obs, hand_state)
                    if canonical_views_renderer is not None:
                        obs = canonical_views_renderer.apply(obs, hand_state)
                    if view_canonicalizer is not None:
                        obs = view_canonicalizer.apply(obs)
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
                    reset_after_episode = False
                    stop_after_episode = False
                    dashboard_hold_sent = False

                    while step < max_steps:
                        operator_command, dashboard_hold_sent = (
                            _consume_operator_episode_command(
                                keyboard_control, env, dashboard_hold_sent
                            )
                        )
                        if operator_command == "reset":
                            episode_finish_reason = "operator_reset"
                            reset_after_episode = True
                            break
                        if operator_command == "quit":
                            episode_finish_reason = "operator_quit"
                            stop_after_episode = True
                            break
                        if dashboard is not None and dashboard.stop_pending():
                            episode_finish_reason = "dashboard_stop"
                            if not dashboard_hold_sent:
                                hold_robot_motion(env)
                                dashboard_hold_sent = True
                            break
                        if dashboard is not None and dashboard.reset_pending():
                            episode_finish_reason = "dashboard_reset"
                            dashboard.consume_reset_request()
                            dashboard.mark_resetting()
                            if not dashboard_hold_sent:
                                hold_robot_motion(env)
                                dashboard_hold_sent = True
                            reset_after_episode = True
                            break
                        should_profile = profile_action_timing and (step % profile_action_every == 0)
                        profile_t0 = time.time()
                        elapsed_for_limit = time.monotonic() - t_start
                        if dashboard is not None:
                            elapsed_for_limit = step / float(cfg["frequency"])
                        if max_duration and elapsed_for_limit > float(max_duration):
                            print("[INFO] Max duration reached.")
                            episode_finish_reason = "max_duration"
                            break
                        cycle_start_wall = time.time()
                        profile_obs_time = latest_obs_time(obs)
                        profile_curr_pose = latest_robot_pose(obs, robot_idx=0)
                        policy_obs = obs

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

                        operator_command, dashboard_hold_sent = (
                            _consume_operator_episode_command(
                                keyboard_control, env, dashboard_hold_sent
                            )
                        )
                        if operator_command == "reset":
                            episode_finish_reason = "operator_reset"
                            reset_after_episode = True
                            break
                        if operator_command == "quit":
                            episode_finish_reason = "operator_quit"
                            stop_after_episode = True
                            break

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
                        policy_hand_points = None

                        if pts21_action_adapter is not None:
                            if pred is None:
                                raise RuntimeError(
                                    "pts21 checkpoint must return result['action'] or "
                                    "result['action_pred']"
                                )
                            arm_action, hand_actions, policy_hand_points = (
                                pts21_action_adapter.split(
                                    pred[:exec_steps], env_gripper_width
                                )
                            )
                        elif has_wuji_command and (
                                pred is None or action_dim not in (10, 15, 20, 29)):
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
                                hand_state=hand_state,
                                hand_command_representation=hand_command_representation,
                            )
                            if has_wuji_command:
                                hand_actions = _coerce_wuji_action_sequence(
                                    result["wuji_command"], exec_steps
                                )
                        if hand_backend == "wuji_hand" and hand_actions is None:
                            raise RuntimeError(
                                "hand=wuji_hand requires action_dim 20/29 or policy result['wuji_command']"
                            )

                        execute_enabled = dashboard is None or dashboard.can_execute()
                        if dashboard is not None and not execute_enabled and not dashboard_hold_sent:
                            hold_robot_motion(env)
                            dashboard_hold_sent = True
                        elif execute_enabled:
                            dashboard_hold_sent = False
                        cycle_executed = bool(execute_enabled)
                        franka_actions = None
                        actual_submitted_actions = None

                        if arm_action is not None:
                            profile_action_start = time.time()
                            if pts21_action_adapter is not None:
                                arm_action_for_franka = pts21_action_adapter.split_arm(
                                    arm_pred, env_gripper_width
                                )
                            elif arm_pred is not None:
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
                            if (
                                    franka_neg_x_delta_gain != 1.0
                                    or franka_neg_y_delta_gain != 1.0
                                    or franka_z_delta_gain != 1.0):
                                current_pose_for_gain = latest_robot_pose(obs, robot_idx=0)
                                if current_pose_for_gain is None:
                                    raise RuntimeError(
                                        "Franka delta gains require robot0 eef pose observation"
                                    )
                                franka_actions = scale_franka_negative_x_delta(
                                    franka_actions,
                                    current_x=current_pose_for_gain[0],
                                    gain=franka_neg_x_delta_gain,
                                    robot_idx=0,
                                )
                                franka_actions = scale_franka_negative_y_delta(
                                    franka_actions,
                                    current_y=current_pose_for_gain[1],
                                    gain=franka_neg_y_delta_gain,
                                    robot_idx=0,
                                )
                                franka_actions = scale_franka_z_delta(
                                    franka_actions,
                                    current_z=current_pose_for_gain[2],
                                    gain=franka_z_delta_gain,
                                    robot_idx=0,
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
                            if dashboard is not None and not dashboard.can_execute():
                                execute_enabled = False
                                cycle_executed = False
                            if execute_enabled and not cfg.get("dry_run_franka", False):
                                env.exec_actions(actions=submitted_actions, timestamps=submitted_timestamps)
                                actual_submitted_actions = submitted_actions
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
                        scheduled_policy_hand_points = None
                        if pts21_action_adapter is not None:
                            scheduled_policy_hand_points = pts21_action_adapter.align_points(
                                policy_hand_points,
                                kept_mask,
                                submitted_timestamps,
                            )
                        profile_hand_start = time.time()
                        if dashboard is not None:
                            should_continue_actions = dashboard.can_execute
                        elif keyboard_control is not None and keyboard_control.enabled:
                            should_continue_actions = keyboard_control.should_continue_episode
                        else:
                            should_continue_actions = None
                        executed_hand_actions, measured_hand_states = execute_hand_actions(
                            execute_enabled=execute_enabled,
                            scheduled_hand_actions=scheduled_hand_actions,
                            hand_backend=hand_backend,
                            hand_timestamps=hand_timestamps,
                            o6_hand=o6_hand,
                            wuji_driver=wuji_driver,
                            wuji_filter=wuji_filter,
                            linker_o6_cfg=linker_o6_cfg,
                            wuji_cfg=wuji_cfg,
                            synchronize_hand_actions=synchronize_hand_actions,
                            should_continue_actions=should_continue_actions,
                        )
                        if measured_hand_states is not None:
                            hand_state = measured_hand_states[-1].astype(np.float32)
                        elif executed_hand_actions is not None:
                            hand_state = executed_hand_actions[-1].astype(np.float32)
                        if pts21_action_adapter is not None:
                            executed_count = (
                                0 if executed_hand_actions is None
                                else len(executed_hand_actions)
                            )
                            pts21_action_adapter.commit_executed(
                                scheduled_policy_hand_points,
                                executed_count,
                            )
                        if (
                                dashboard is not None
                                and not dashboard.can_execute()
                                and not dashboard_hold_sent):
                            hold_robot_motion(env)
                            dashboard_hold_sent = True
                        profile_hand_done = time.time()

                        operator_command, dashboard_hold_sent = (
                            _consume_operator_episode_command(
                                keyboard_control, env, dashboard_hold_sent
                            )
                        )
                        if operator_command == "reset":
                            episode_finish_reason = "operator_reset"
                            reset_after_episode = True
                            break
                        if operator_command == "quit":
                            episode_finish_reason = "operator_quit"
                            stop_after_episode = True
                            break

                        if cycle_executed:
                            step += exec_steps
                        if wait_for_horizon and cycle_executed:
                            wait_until = cycle_start_wall + exec_steps * dt
                            profile_wait_start = time.time()
                            if (
                                    keyboard_control is not None
                                    and keyboard_control.enabled):
                                completed = interruptible_wait_until(
                                    wait_until,
                                    keyboard_control.should_continue_episode,
                                    time_func=time.time,
                                )
                                if not completed:
                                    operator_command, dashboard_hold_sent = (
                                        _consume_operator_episode_command(
                                            keyboard_control, env, dashboard_hold_sent
                                        )
                                    )
                                    if operator_command == "reset":
                                        episode_finish_reason = "operator_reset"
                                        reset_after_episode = True
                                        profile_wait_done = time.time()
                                        break
                                    if operator_command == "quit":
                                        episode_finish_reason = "operator_quit"
                                        stop_after_episode = True
                                        profile_wait_done = time.time()
                                        break
                            elif dashboard is None:
                                precise_wait(wait_until, time_func=time.time)
                            else:
                                completed = interruptible_wait_until(
                                    wait_until,
                                    dashboard.can_execute,
                                    time_func=time.time,
                                )
                                if not completed and not dashboard_hold_sent:
                                    hold_robot_motion(env)
                                    dashboard_hold_sent = True
                            profile_wait_done = time.time()
                        else:
                            profile_wait_start = time.time()
                            if dashboard is not None and not cycle_executed:
                                time.sleep(0.1)
                            profile_wait_done = time.time()
                        operator_command, dashboard_hold_sent = (
                            _consume_operator_episode_command(
                                keyboard_control, env, dashboard_hold_sent
                            )
                        )
                        if operator_command == "reset":
                            episode_finish_reason = "operator_reset"
                            reset_after_episode = True
                            break
                        if operator_command == "quit":
                            episode_finish_reason = "operator_quit"
                            stop_after_episode = True
                            break
                        profile_get_obs_start = time.time()
                        raw_obs = env.get_obs()
                        profile_get_obs_done = time.time()
                        env_gripper_width = latest_env_gripper_width(raw_obs)
                        hand_state = read_hand_state(
                            hand_state,
                            o6_hand,
                            wuji_driver,
                            dry_run_hand,
                            o6_warning="O6 get_state failed, using last state",
                            wuji_warning="Wuji read_positions failed, using last command",
                        )
                        if pts21_action_adapter is None:
                            obs = patch_hand_obs(raw_obs, hand_state, shape_meta)
                        else:
                            obs = pts21_action_adapter.patch_observation(
                                raw_obs, shape_meta
                            )
                        if object_pocket_observer is not None:
                            obs = object_pocket_observer.apply(obs, hand_state)
                        if skeleton_overlay_renderer is not None:
                            obs = skeleton_overlay_renderer.apply(obs, hand_state)
                        if canonical_views_renderer is not None:
                            obs = canonical_views_renderer.apply(obs, hand_state)
                        if view_canonicalizer is not None:
                            obs = view_canonicalizer.apply(obs)
                        publish_dashboard(
                            dashboard,
                            obs_dict_np,
                            obs,
                            hand_state,
                            raw_model_chunk=(
                                pred if pred is not None else np.empty((0,), dtype=np.float32)
                            ),
                            processed_arm_chunk=(
                                franka_actions
                                if franka_actions is not None
                                else np.empty((0,), dtype=np.float32)
                            ),
                            submitted_arm_chunk=(
                                actual_submitted_actions
                                if actual_submitted_actions is not None
                                else np.empty((0,), dtype=np.float32)
                            ),
                            processed_hand_chunk=(
                                scheduled_hand_actions
                                if scheduled_hand_actions is not None
                                else np.empty((0,), dtype=np.float32)
                            ),
                            submitted_hand_chunk=(
                                executed_hand_actions
                                if executed_hand_actions is not None
                                else np.empty((0,), dtype=np.float32)
                            ),
                            metadata={
                                "mode": (
                                    "executing" if cycle_executed else "paused_inference"
                                ),
                                "episode": int(episode),
                                "step": int(step),
                                "action_dim": int(action_dim),
                                "hand_backend": hand_backend or "none",
                                "frequency_hz": float(cfg["frequency"]),
                                "steps_per_inference": int(steps_per_infer),
                            },
                        )
                        if episode_recorder is not None:
                            if executed_hand_actions is not None:
                                try:
                                    recorded_hand_timestamps = hand_timestamps[
                                        :len(executed_hand_actions)
                                    ]
                                    episode_recorder.record_segment(
                                        env=env,
                                        timestamps=recorded_hand_timestamps,
                                        hand_commands=executed_hand_actions,
                                        measured_hand_states=measured_hand_states,
                                        policy_obs=policy_obs,
                                        shape_meta=shape_meta,
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
                    should_stop, hand_state = _complete_dashboard_episode(
                        dashboard=dashboard,
                        episode=episode,
                        reset_after_episode=reset_after_episode,
                        env=env,
                        policy=policy,
                        hand_state=hand_state,
                        robots_config=robots_config,
                        o6_hand=o6_hand,
                        linker_o6_cfg=linker_o6_cfg,
                        wuji_driver=wuji_driver,
                        wuji_filter=wuji_filter,
                        wuji_cfg=wuji_cfg,
                        hand_backend=hand_backend,
                        dry_run_franka=bool(cfg.get("dry_run_franka", False)),
                        dry_run_hand=dry_run_hand,
                    )
                    if reset_after_episode and pts21_action_adapter is not None:
                        pts21_action_adapter.reset()
                    if should_stop or stop_after_episode:
                        break
    finally:
        if episode_recorder is not None and episode_recorder.active:
            try:
                episode_recorder.finish_episode(reason="exception_or_exit")
            except Exception as exc:
                print(f"[WARN] failed to finalize inference episode: {exc}")
        if pts21_action_adapter is not None:
            pts21_action_adapter.close()
        if object_pocket_observer is not None:
            object_pocket_observer.close()
        if o6_hand is not None:
            o6_hand.close()
        if wuji_driver is not None:
            wuji_driver.close()
