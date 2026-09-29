#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Franka + optional dexterous-hand inference CLI facade."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Re-export the established helper surface for existing launchers and ad-hoc tools.
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
    fmt_vec,
    future_timestamps,
    interruptible_wait_until,
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
    DISABLED_HAND_VALUES,
    HAND_BACKENDS,
    NoOpScalarGripper,
    _as_qpos20,
    _import_wuji_runtime,
    legacy_enabled_hand,
    load_config,
    load_yaml_mapping,
    merged_hand_config,
    normalize_hand_name,
    open_o6_hand,
    open_wuji_hand,
    resolve_hand_backend,
)
from real_inference_dashboard import (
    dashboard_frames,
    dashboard_rgb_inputs_from_env_obs,
    hold_robot_motion,
    publish_dashboard,
    reset_robot_and_hand,
)
from real_inference_policy import load_policy, validate_checkpoint_metadata
from real_inference_runner import run_inference


def _parse_pair_arg(value: str) -> list[float]:
    normalized = str(value).lower().replace("x", ",")
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("value must look like x,y")
    return [float(parts[0]), float(parts[1])]


def _parse_size_arg(value: str) -> list[int]:
    normalized = str(value).lower().replace(",", "x")
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("value must look like width,height or widthxheight")
    width, height = int(parts[0]), int(parts[1])
    if width <= 1 or height <= 1:
        raise argparse.ArgumentTypeError("width and height must be > 1")
    return [width, height]


def _parse_circle_arg(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("value must look like cx,cy,radius")
    circle = [float(part) for part in parts]
    if circle[2] <= 0.0:
        raise argparse.ArgumentTypeError("circle radius must be > 0")
    return circle


def build_arg_parser(description: str = "Franka + O6 推理脚本"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--config", default="eval_franka_o6_config.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("-i", "--checkpoint", default=None, help="checkpoint 路径")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("-rc", "--robot-config", default=None, dest="robot_config")
    parser.add_argument("--timing-self-test", action="store_true",
                        help="Run local Franka timestamp scheduling checks and exit")
    parser.add_argument("--hand", choices=["linker_o6", "wuji_hand", "wujihand", "none"],
                        default=None, help="选择灵巧手: linker_o6 / wuji_hand / none")
    parser.add_argument("-f", "--frequency", default=None, type=int)
    parser.add_argument("-si", "--steps-per-inference", default=None, type=int,
                        dest="steps_per_inference")
    parser.add_argument("--max-timesteps", default=None, type=int, dest="max_timesteps")
    parser.add_argument("--max-duration", default=None, type=float, dest="max_duration")
    parser.add_argument("--max-pos-speed", default=None, type=float, dest="max_pos_speed")
    parser.add_argument("--max-rot-speed", default=None, type=float, dest="max_rot_speed")
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
    parser.add_argument("--init", action="store_true", dest="init_joints", default=None)
    parser.add_argument("--dry-run-franka", action="store_true",
                        dest="dry_run_franka", default=None)
    parser.add_argument("--dry-run-o6", action="store_true",
                        dest="dry_run_o6", default=None)
    parser.add_argument("--dry-run-hand", action="store_true",
                        dest="dry_run_hand", default=None)
    parser.add_argument("--save-episode", action="store_true",
                        dest="save_episode", default=None,
                        help="Save each inference rollout as PKL + JPG episode data")
    parser.add_argument("--episode-record-fps", "--save-episode-fps",
                        dest="episode_record_fps", default=None, type=float,
                        help="Fixed FPS for --save-episode JPG/PKL samples; defaults to camera capture_fps or 30")
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument(
        "--view-align",
        action="store_true",
        dest="view_alignment_enabled",
        default=None,
        help="Enable deployment-side RGB view alignment for this run",
    )
    view_group.add_argument(
        "--no-view-align",
        action="store_false",
        dest="view_alignment_enabled",
        default=None,
        help="Disable deployment-side RGB view alignment for this run",
    )
    parser.add_argument("--view-scale", type=float, default=None)
    parser.add_argument("--view-rotation-deg", type=float, default=None)
    parser.add_argument("--view-tx-ratio", type=float, default=None)
    parser.add_argument("--view-ty-ratio", type=float, default=None)
    parser.add_argument(
        "--view-transform-type",
        choices=["similarity", "homography", "cavity_center", "cavity-center"],
        default=None,
        help="view_alignment transform type; cavity_center enables cavity-center scale/translate",
    )
    parser.add_argument("--view-train-center", type=_parse_pair_arg, default=None,
                        help="Training/reference cavity center in pixels, x,y")
    parser.add_argument("--view-train-image-size", type=_parse_size_arg, default=None,
                        help="Training/reference image size, width,height")
    parser.add_argument("--view-deploy-center", type=_parse_pair_arg, default=None,
                        help="Deployment cavity center in policy-image pixels, x,y")
    parser.add_argument("--view-circle", type=_parse_circle_arg, default=None,
                        help="Optional fixed deployment fisheye circle in pixels, cx,cy,radius")
    parser.add_argument(
        "--view-circle-mode",
        choices=["fixed-source", "fixed_source", "transformed-source", "transformed_source",
                 "fixed-reference", "fixed_reference", "fixed-output", "fixed_output"],
        default=None,
    )
    parser.add_argument("--view-circle-threshold", type=float, default=None,
                        help="Auto fisheye-circle threshold in 0-255 intensity units")
    parser.add_argument("--view-circle-margin-px", type=float, default=None)
    parser.add_argument(
        "--view-reflect-inner-margin-px",
        "--view-reflect-margin-px",
        type=float,
        default=None,
        dest="view_reflect_inner_margin_px",
        help="Shrink the reflection sampling circle inward by N pixels to avoid fisheye dark rims",
    )
    return parser


def config_from_args(args) -> dict:
    view_override_names = {
        "view_alignment_enabled",
        "view_scale",
        "view_rotation_deg",
        "view_tx_ratio",
        "view_ty_ratio",
        "view_transform_type",
        "view_train_center",
        "view_train_image_size",
        "view_deploy_center",
        "view_circle",
        "view_circle_mode",
        "view_circle_threshold",
        "view_circle_margin_px",
        "view_reflect_inner_margin_px",
    }
    excluded = {"config", "timing_self_test", *view_override_names}
    overrides = {key: value for key, value in vars(args).items() if key not in excluded}
    cfg = load_config(args.config, overrides)

    view_cfg = dict(cfg.get("view_alignment", {}) or {})
    if args.view_alignment_enabled is not None:
        view_cfg["enabled"] = bool(args.view_alignment_enabled)
    if args.view_scale is not None:
        view_cfg["scale"] = float(args.view_scale)
    if args.view_rotation_deg is not None:
        view_cfg["rotation_deg"] = float(args.view_rotation_deg)
    if args.view_transform_type is not None:
        view_cfg["transform_type"] = args.view_transform_type.replace("-", "_")
    if args.view_train_center is not None:
        view_cfg["train_center_px"] = list(args.view_train_center)
    if args.view_train_image_size is not None:
        view_cfg["train_image_size"] = list(args.view_train_image_size)
    if args.view_deploy_center is not None:
        view_cfg["deploy_center_px"] = list(args.view_deploy_center)
    if args.view_circle is not None:
        view_cfg["deploy_circle_px"] = list(args.view_circle)
    if args.view_circle_mode is not None:
        view_cfg["circle_mode"] = args.view_circle_mode.replace("-", "_")
    if args.view_circle_threshold is not None:
        view_cfg["circle_threshold"] = float(args.view_circle_threshold)
    if args.view_circle_margin_px is not None:
        view_cfg["circle_margin_px"] = float(args.view_circle_margin_px)
    if args.view_reflect_inner_margin_px is not None:
        view_cfg["reflect_inner_margin_px"] = float(args.view_reflect_inner_margin_px)
    translation = list(view_cfg.get("translation_ratio", [0.0, 0.0]))
    if args.view_tx_ratio is not None:
        translation[0] = float(args.view_tx_ratio)
    if args.view_ty_ratio is not None:
        translation[1] = float(args.view_ty_ratio)
    view_cfg["translation_ratio"] = translation
    if view_cfg:
        cfg["view_alignment"] = view_cfg

    os.makedirs(cfg["output"], exist_ok=True)
    return cfg


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.timing_self_test:
        run_timing_self_test()
        return
    cfg = config_from_args(args)
    run_inference(cfg)


if __name__ == "__main__":
    main()
