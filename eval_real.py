"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import copy
import pathlib
import time
from contextlib import ExitStack
from multiprocessing.managers import SharedMemoryManager
import omegaconf

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
import csv
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.camera_factory import build_camera_from_yaml
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.common.pose_util import pose_to_mat, mat_to_pose
from scipy.spatial.transform import Rotation as R, Slerp

OmegaConf.register_new_resolver("eval", eval, replace=True)


class _NullKeyCounter:
    def get_press_events(self):
        return []

    def clear(self):
        pass


class _NullSpacemouse:
    def get_motion_state_transformed(self):
        return np.zeros(6, dtype=np.float32)

    def is_button_pressed(self, _idx):
        return False


def _load_keyboard_controls():
    from umi.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode
    return KeystrokeCounter, Key, KeyCode


def _load_spacemouse():
    from umi.real_world.spacemouse_shared_memory import Spacemouse
    return Spacemouse


def _debug_image_to_uint8_hwc(img):
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size > 0 and np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def _debug_write_rgb(path, img):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _debug_image_to_uint8_hwc(img)
    cv2.imwrite(str(path), rgb[..., ::-1])


def _apply_gripper_action_lead(target_poses, n_robots, lead_steps):
    lead_steps = int(lead_steps)
    if lead_steps <= 0:
        return target_poses
    poses = np.asarray(target_poses)
    if poses.shape[0] == 0:
        return poses
    shifted = poses.copy()
    src_idx = np.minimum(
        np.arange(poses.shape[0], dtype=np.int64) + lead_steps,
        poses.shape[0] - 1,
    )
    for robot_idx in range(n_robots):
        gripper_col = robot_idx * 7 + 6
        shifted[:, gripper_col] = poses[src_idx, gripper_col]
    return shifted


def _interp_rotvec_at_time(rotvecs, timestamps, query_time):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    rotvecs = np.asarray(rotvecs, dtype=np.float64)
    if len(timestamps) == 0:
        return None
    if len(timestamps) == 1:
        return rotvecs[0].copy()

    idx = int(np.searchsorted(timestamps, float(query_time), side='left'))
    if idx <= 0:
        return rotvecs[0].copy()
    if idx >= len(timestamps):
        return rotvecs[-1].copy()

    t0 = float(timestamps[idx - 1])
    t1 = float(timestamps[idx])
    if t1 <= t0:
        return rotvecs[idx].copy()

    alpha = (float(query_time) - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    key_rots = R.from_rotvec(np.stack([rotvecs[idx - 1], rotvecs[idx]], axis=0))
    return Slerp([0.0, 1.0], key_rots)([alpha]).as_rotvec()[0]


def _interp_robot_pose_plan(target_poses, timestamps, query_time, robot_idx):
    target_poses = np.asarray(target_poses, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if target_poses.size == 0 or timestamps.size == 0:
        return None
    if query_time < timestamps[0] or query_time > timestamps[-1]:
        return None

    base = robot_idx * 7
    pose = np.zeros(6, dtype=np.float64)
    for axis_idx in range(3):
        pose[axis_idx] = np.interp(query_time, timestamps, target_poses[:, base + axis_idx])
    pose[3:6] = _interp_rotvec_at_time(target_poses[:, base + 3:base + 6], timestamps, query_time)
    return pose


def _blend_rotvec(prev_rotvec, curr_rotvec, alpha):
    key_rots = R.from_rotvec(np.stack([prev_rotvec, curr_rotvec], axis=0))
    return Slerp([0.0, 1.0], key_rots)([float(alpha)]).as_rotvec()[0]


def _stitch_robot_action_chunk(
        target_poses,
        action_timestamps,
        prev_target_poses,
        prev_action_timestamps,
        n_robots,
        stitch_steps):
    stitch_steps = int(stitch_steps)
    stats = {
        'enabled_steps': stitch_steps,
        'applied_robot_waypoints': 0,
        'max_xyz_delta_before_m': 0.0,
        'max_rot_delta_before_rad': 0.0,
    }
    target_poses = np.asarray(target_poses, dtype=np.float64)
    action_timestamps = np.asarray(action_timestamps, dtype=np.float64)
    if (
        stitch_steps <= 0
        or target_poses.size == 0
        or prev_target_poses is None
        or prev_action_timestamps is None
    ):
        return target_poses, stats

    prev_target_poses = np.asarray(prev_target_poses, dtype=np.float64)
    prev_action_timestamps = np.asarray(prev_action_timestamps, dtype=np.float64)
    if prev_target_poses.size == 0 or prev_action_timestamps.size == 0:
        return target_poses, stats

    stitched = target_poses.copy()
    n_steps = min(stitch_steps, len(stitched))
    for step_idx in range(n_steps):
        alpha = float(step_idx + 1) / float(n_steps + 1)
        query_time = float(action_timestamps[step_idx])
        for robot_idx in range(n_robots):
            prev_pose = _interp_robot_pose_plan(
                prev_target_poses,
                prev_action_timestamps,
                query_time,
                robot_idx,
            )
            if prev_pose is None:
                continue
            base = robot_idx * 7
            curr_pose = stitched[step_idx, base:base + 6].copy()
            stats['max_xyz_delta_before_m'] = max(
                stats['max_xyz_delta_before_m'],
                float(np.linalg.norm(curr_pose[:3] - prev_pose[:3])),
            )
            stats['max_rot_delta_before_rad'] = max(
                stats['max_rot_delta_before_rad'],
                float(np.linalg.norm(curr_pose[3:6] - prev_pose[3:6])),
            )
            stitched[step_idx, base:base + 3] = (
                (1.0 - alpha) * prev_pose[:3] + alpha * curr_pose[:3]
            )
            stitched[step_idx, base + 3:base + 6] = _blend_rotvec(
                prev_pose[3:6],
                curr_pose[3:6],
                alpha,
            )
            stats['applied_robot_waypoints'] += 1

    return stitched, stats


def _debug_pose_rows(poses, n_robots, timestamps=None, obs_time=None, curr_time=None):
    poses = np.asarray(poses)
    timestamps = None if timestamps is None else np.asarray(timestamps)
    rows = []
    for step_idx, pose in enumerate(poses):
        row = {'step': int(step_idx)}
        if timestamps is not None and step_idx < len(timestamps):
            timestamp = float(timestamps[step_idx])
            row['target_time_unix'] = timestamp
            row['dt_from_obs_s'] = '' if obs_time is None else timestamp - float(obs_time)
            row['dt_from_now_s'] = '' if curr_time is None else timestamp - float(curr_time)
        else:
            row['target_time_unix'] = ''
            row['dt_from_obs_s'] = ''
            row['dt_from_now_s'] = ''
        for robot_idx in range(n_robots):
            base = robot_idx * 7
            prefix = f'robot{robot_idx}'
            row[f'{prefix}_x'] = float(pose[base + 0])
            row[f'{prefix}_y'] = float(pose[base + 1])
            row[f'{prefix}_z'] = float(pose[base + 2])
            row[f'{prefix}_rx'] = float(pose[base + 3])
            row[f'{prefix}_ry'] = float(pose[base + 4])
            row[f'{prefix}_rz'] = float(pose[base + 5])
            row[f'{prefix}_gripper_rad'] = float(pose[base + 6])
        rows.append(row)
    return rows


def _debug_write_csv(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _debug_write_gripper_plot(path, full_action, submitted_action, action_timestamps,
                              obs_time, curr_time, latest_gripper, n_robots):
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except Exception:
        return False

    full_action = np.asarray(full_action)
    submitted_action = np.asarray(submitted_action)
    action_timestamps = np.asarray(action_timestamps)
    fig, axes = plt.subplots(n_robots, 1, figsize=(8, max(3, 2.5 * n_robots)), squeeze=False)
    axes = axes[:, 0]
    for robot_idx, ax in enumerate(axes):
        gripper_col = robot_idx * 7 + 6
        full_t = np.arange(full_action.shape[0], dtype=np.float64)
        if len(action_timestamps) > 1:
            dt_est = float(np.median(np.diff(action_timestamps)))
            full_t = full_t * dt_est
        submitted_t = action_timestamps - float(obs_time)

        ax.plot(full_t, full_action[:, gripper_col], marker='o', label='policy full horizon')
        if submitted_action.size > 0:
            ax.plot(
                submitted_t,
                submitted_action[:, gripper_col],
                marker='x',
                label='submitted after filter/lead',
            )
        if latest_gripper is not None and len(latest_gripper) > robot_idx:
            ax.axhline(
                float(latest_gripper[robot_idx]),
                color='tab:red',
                linestyle='--',
                label='latest observed gripper',
            )
        ax.axvline(float(curr_time) - float(obs_time), color='0.4', linestyle=':', label='dump time')
        ax.set_title(f'robot{robot_idx} gripper target (smaller rad = more closed)')
        ax.set_xlabel('seconds from observation timestamp')
        ax.set_ylabel('gripper rad')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _debug_write_robot_plot(path, full_action, submitted_action, action_timestamps,
                            obs_time, curr_time, dt, latest_poses, n_robots):
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except Exception:
        return False

    full_action = np.asarray(full_action)
    submitted_action = np.asarray(submitted_action)
    action_timestamps = np.asarray(action_timestamps)
    latest_poses = [] if latest_poses is None else latest_poses
    pose_names = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    fig, axes = plt.subplots(
        6,
        n_robots,
        figsize=(4 * n_robots, 10),
        squeeze=False,
        sharex=True,
    )
    full_t = np.arange(full_action.shape[0], dtype=np.float64) * float(dt)
    submitted_t = action_timestamps - float(obs_time)
    now_t = float(curr_time) - float(obs_time)

    for robot_idx in range(n_robots):
        base = robot_idx * 7
        latest_pose = latest_poses[robot_idx] if robot_idx < len(latest_poses) else None
        for pose_idx, name in enumerate(pose_names):
            ax = axes[pose_idx, robot_idx]
            col = base + pose_idx
            ax.plot(full_t, full_action[:, col], marker='o', label='policy full horizon')
            if submitted_action.size > 0:
                ax.plot(submitted_t, submitted_action[:, col], marker='x', label='submitted')
            if latest_pose is not None and len(latest_pose) > pose_idx:
                ax.axhline(float(latest_pose[pose_idx]), color='tab:red', linestyle='--', label='latest observed')
            ax.axvline(now_t, color='0.4', linestyle=':', label='dump time')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            if pose_idx == 0:
                ax.set_title(f'robot{robot_idx} pose target')
            if pose_idx == len(pose_names) - 1:
                ax.set_xlabel('seconds from observation timestamp')
            if robot_idx == 0 and pose_idx == 0:
                ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _debug_dump_inference(
        debug_dir,
        iter_idx,
        obs,
        obs_dict_np,
        raw_action,
        action,
        target_poses,
        action_timestamps,
        obs_timestamps,
        curr_time,
        dt,
        n_robots,
        gripper_action_lead_steps,
        robot_action_stitch_stats=None,
        save_plots=False):
    debug_dir = pathlib.Path(debug_dir)
    step_dir = debug_dir.joinpath(f"iter_{iter_idx:06d}")
    step_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for key, value in obs.items():
        if key.endswith('_rgb'):
            image_path = step_dir.joinpath(f"env_{key}_last.png")
            _debug_write_rgb(image_path, value[-1])
            image_files.append(str(image_path.name))

    for key, value in obs_dict_np.items():
        if key.endswith('_rgb'):
            # obs_dict_np image layout is TCHW float in [0, 1].
            for t_idx, img in enumerate(value):
                image_path = step_dir.joinpath(f"model_{key}_t{t_idx:02d}.png")
                _debug_write_rgb(image_path, img)
                image_files.append(str(image_path.name))

    np.savez_compressed(
        step_dir.joinpath("inference_arrays.npz"),
        raw_action=np.asarray(raw_action),
        converted_action=np.asarray(action),
        submitted_target_poses=np.asarray(target_poses),
        action_timestamps=np.asarray(action_timestamps),
        obs_timestamps=np.asarray(obs_timestamps),
        latest_robot0_eef_pos=np.asarray(obs.get('robot0_eef_pos', []))[-1:],
        latest_robot0_eef_rot_axis_angle=np.asarray(obs.get('robot0_eef_rot_axis_angle', []))[-1:],
        latest_robot0_gripper_width=np.asarray(obs.get('robot0_gripper_width', []))[-1:],
    )

    obs_time = float(np.asarray(obs_timestamps)[-1])
    latest_grippers = []
    latest_poses = []
    for robot_idx in range(n_robots):
        grip_arr = np.asarray(obs.get(f'robot{robot_idx}_gripper_width', []))
        if grip_arr.size > 0:
            latest_grippers.append(float(grip_arr.reshape(-1)[-1]))
        eef_pos = np.asarray(obs.get(f'robot{robot_idx}_eef_pos', []))
        eef_rot = np.asarray(obs.get(f'robot{robot_idx}_eef_rot_axis_angle', []))
        if eef_pos.size > 0 and eef_rot.size > 0:
            latest_poses.append(np.concatenate([eef_pos[-1], eef_rot[-1]], axis=-1))

    full_action = np.asarray(action)
    submitted_action = np.asarray(target_poses)
    full_rows = _debug_pose_rows(
        full_action,
        n_robots=n_robots,
        timestamps=obs_time + np.arange(len(full_action), dtype=np.float64) * float(dt),
        obs_time=obs_time,
        curr_time=curr_time,
    )
    submitted_rows = _debug_pose_rows(
        submitted_action,
        n_robots=n_robots,
        timestamps=action_timestamps,
        obs_time=obs_time,
        curr_time=curr_time,
    )
    _debug_write_csv(step_dir.joinpath("policy_action_full.csv"), full_rows)
    _debug_write_csv(step_dir.joinpath("submitted_actions.csv"), submitted_rows)

    gripper_rows = []
    for robot_idx in range(n_robots):
        if len(latest_grippers) > robot_idx:
            gripper_rows.append({
                'source': 'observed',
                'robot': robot_idx,
                'step': -1,
                'dt_from_obs_s': 0.0,
                'dt_from_now_s': obs_time - float(curr_time),
                'gripper_rad': latest_grippers[robot_idx],
            })
        gripper_col = robot_idx * 7 + 6
        for step_idx, pose in enumerate(full_action):
            gripper_rows.append({
                'source': 'policy_full',
                'robot': robot_idx,
                'step': int(step_idx),
                'dt_from_obs_s': float(step_idx) * float(dt),
                'dt_from_now_s': obs_time + float(step_idx) * float(dt) - float(curr_time),
                'gripper_rad': float(pose[gripper_col]),
            })
        for step_idx, pose in enumerate(submitted_action):
            target_time = float(action_timestamps[step_idx])
            gripper_rows.append({
                'source': 'submitted',
                'robot': robot_idx,
                'step': int(step_idx),
                'dt_from_obs_s': target_time - obs_time,
                'dt_from_now_s': target_time - float(curr_time),
                'gripper_rad': float(pose[gripper_col]),
            })
    _debug_write_csv(step_dir.joinpath("gripper_timeline.csv"), gripper_rows)

    plot_written = False
    robot_plot_written = False
    if save_plots:
        plot_written = _debug_write_gripper_plot(
            step_dir.joinpath("gripper_timeline.png"),
            full_action=full_action,
            submitted_action=submitted_action,
            action_timestamps=action_timestamps,
            obs_time=obs_time,
            curr_time=curr_time,
            latest_gripper=latest_grippers,
            n_robots=n_robots,
        )
        robot_plot_written = _debug_write_robot_plot(
            step_dir.joinpath("robot_timeline.png"),
            full_action=full_action,
            submitted_action=submitted_action,
            action_timestamps=action_timestamps,
            obs_time=obs_time,
            curr_time=curr_time,
            dt=dt,
            latest_poses=latest_poses,
            n_robots=n_robots,
        )

    def _grip_stats(poses):
        arr = np.asarray(poses)
        stats = {}
        for robot_idx in range(n_robots):
            col = robot_idx * 7 + 6
            if arr.size == 0:
                stats[f'robot{robot_idx}'] = {}
            else:
                g = arr[:, col]
                stats[f'robot{robot_idx}'] = {
                    'first': float(g[0]),
                    'last': float(g[-1]),
                    'min': float(np.min(g)),
                    'max': float(np.max(g)),
                }
        return stats

    meta = {
        'iter_idx': int(iter_idx),
        'time': float(curr_time),
        'dt': float(dt),
        'gripper_action_lead_steps': int(gripper_action_lead_steps),
        'robot_action_stitch_stats': robot_action_stitch_stats or {},
        'latest_gripper_rad': latest_grippers,
        'latest_robot_pose': [pose.tolist() for pose in latest_poses],
        'obs_keys': sorted([str(k) for k in obs.keys()]),
        'model_obs_shapes': {
            str(k): list(np.asarray(v).shape)
            for k, v in obs_dict_np.items()
        },
        'raw_action_shape': list(np.asarray(raw_action).shape),
        'converted_action_shape': list(np.asarray(action).shape),
        'submitted_target_poses_shape': list(np.asarray(target_poses).shape),
        'first_submitted_target_pose': np.asarray(target_poses)[0].tolist(),
        'action_timestamps': np.asarray(action_timestamps).tolist(),
        'obs_timestamps': np.asarray(obs_timestamps).tolist(),
        'submitted_time_from_obs_s': (np.asarray(action_timestamps) - obs_time).tolist(),
        'submitted_time_from_now_s': (np.asarray(action_timestamps) - float(curr_time)).tolist(),
        'policy_full_gripper_stats': _grip_stats(full_action),
        'submitted_gripper_stats': _grip_stats(submitted_action),
        'image_files': image_files,
        'readable_files': [
            'policy_action_full.csv',
            'submitted_actions.csv',
            'gripper_timeline.csv',
            'gripper_timeline.png' if plot_written else None,
            'robot_timeline.png' if robot_plot_written else None,
            'debug_summary.txt',
        ],
    }
    meta['readable_files'] = [x for x in meta['readable_files'] if x is not None]
    with step_dir.joinpath("meta.json").open('w') as f:
        json.dump(meta, f, indent=2)

    summary_lines = [
        f"iter_idx: {iter_idx}",
        f"obs_time_unix: {obs_time:.6f}",
        f"dump_time_unix: {float(curr_time):.6f}",
        f"dump_delay_from_obs_s: {float(curr_time) - obs_time:.4f}",
        f"dt_s: {float(dt):.4f}",
        f"gripper_action_lead_steps: {int(gripper_action_lead_steps)}",
        f"robot_action_stitch_stats: {meta['robot_action_stitch_stats']}",
        f"latest_gripper_rad: {latest_grippers}",
    ]
    for robot_idx in range(n_robots):
        key = f'robot{robot_idx}'
        summary_lines.append(
            f"{key} policy_full_gripper(first/min/last): "
            f"{meta['policy_full_gripper_stats'][key].get('first', '')} / "
            f"{meta['policy_full_gripper_stats'][key].get('min', '')} / "
            f"{meta['policy_full_gripper_stats'][key].get('last', '')}"
        )
        summary_lines.append(
            f"{key} submitted_gripper(first/min/last): "
            f"{meta['submitted_gripper_stats'][key].get('first', '')} / "
            f"{meta['submitted_gripper_stats'][key].get('min', '')} / "
            f"{meta['submitted_gripper_stats'][key].get('last', '')}"
        )
    if len(action_timestamps) > 0:
        summary_lines.append(
            "submitted_time_from_now_s(first/last): "
            f"{float(action_timestamps[0]) - float(curr_time):.4f} / "
            f"{float(action_timestamps[-1]) - float(curr_time):.4f}"
        )
    summary_lines.append("open policy_action_full.csv / submitted_actions.csv for numeric targets.")
    if save_plots:
        summary_lines.append("open gripper_timeline.png for a quick visual check.")
        summary_lines.append("open robot_timeline.png for robot target curves.")
    else:
        summary_lines.append("plots disabled; set --debug-save-plots to write timeline PNGs.")
    step_dir.joinpath("debug_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding='utf-8',
    )

    summary_path = debug_dir.joinpath("summary.csv")
    summary_exists = summary_path.exists()
    summary_row = {
        'iter_idx': int(iter_idx),
        'dump_delay_from_obs_s': float(curr_time) - obs_time,
        'n_submitted': int(len(submitted_action)),
        'gripper_action_lead_steps': int(gripper_action_lead_steps),
        'robot_stitch_applied_waypoints': meta['robot_action_stitch_stats'].get('applied_robot_waypoints', ''),
        'robot_stitch_max_xyz_delta_before_m': meta['robot_action_stitch_stats'].get('max_xyz_delta_before_m', ''),
        'robot_stitch_max_rot_delta_before_rad': meta['robot_action_stitch_stats'].get('max_rot_delta_before_rad', ''),
        'first_submitted_dt_from_now_s': '' if len(action_timestamps) == 0 else float(action_timestamps[0]) - float(curr_time),
        'last_submitted_dt_from_now_s': '' if len(action_timestamps) == 0 else float(action_timestamps[-1]) - float(curr_time),
    }
    for robot_idx in range(n_robots):
        key = f'robot{robot_idx}'
        summary_row[f'{key}_observed_gripper_rad'] = '' if len(latest_grippers) <= robot_idx else latest_grippers[robot_idx]
        for prefix, stats_key in [('policy', 'policy_full_gripper_stats'), ('submitted', 'submitted_gripper_stats')]:
            stats = meta[stats_key][key]
            summary_row[f'{key}_{prefix}_gripper_first'] = stats.get('first', '')
            summary_row[f'{key}_{prefix}_gripper_min'] = stats.get('min', '')
            summary_row[f'{key}_{prefix}_gripper_last'] = stats.get('last', '')
    with summary_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if not summary_exists:
            writer.writeheader()
        writer.writerow(summary_row)


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config, tx_robot1_robot0=None):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    if tx_robot1_robot0 is not None:
        this_that_mat = np.asarray(tx_robot1_robot0, dtype=np.float64).copy()
    else:
        this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # legacy fallback

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))


def enforce_action_collisions(target_poses, robots_config, tx_robot1_robot0=None):
    for target_pose in target_poses:
        for robot_idx in range(len(robots_config)):
            solve_table_collision(
                ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                gripper_width=target_pose[robot_idx * 7 + 6],
                height_threshold=robots_config[robot_idx]['height_threshold']
            )

        solve_sphere_collision(
            ee_poses=target_pose.reshape([len(robots_config), -1]),
            robots_config=robots_config,
            tx_robot1_robot0=tx_robot1_robot0,
        )


def get_policy_start_delay(grippers_config, default_delay=1.0):
    delay = float(default_delay)
    for gc in grippers_config:
        open_pos = gc.get('start_episode_open_rad', gc.get('open_on_start_rad', None))
        if open_pos is None:
            continue
        open_duration = float(gc.get(
            'start_episode_open_duration_s',
            gc.get('open_on_start_duration_s', 1.0),
        ))
        delay = max(delay, open_duration + 0.1)
    return delay


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--max_timesteps', '-mt', default=500, help='Max steps for each epoch.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--temporal_agg', is_flag=True, default=False)
@click.option('--ensemble_steps', type=int, default=8)
@click.option('--headless', is_flag=True, default=False,
              help='Run without pynput/OpenCV windows/SpaceMouse; useful over SSH.')
@click.option('--dry-run-actions', is_flag=True, default=False,
              help='Run predict_action and action conversion, but do not call env.exec_actions().')
@click.option('--pdb-before-joint-init', is_flag=True, default=False,
              help='Enter pdb before starting env when -j/--init_joints is active.')
@click.option('--pdb-before-policy-action', is_flag=True, default=False,
              help='Enter pdb before each real policy env.exec_actions() call.')
@click.option('--pdb-before-predict-action', is_flag=True, default=False,
              help='Enter pdb before policy.predict_action() calls.')
@click.option('--debug-action-start-delay', type=float, default=0.3,
              help='When resuming from --pdb-before-policy-action, reschedule the first action this many seconds in the future.')
@click.option('--debug-save-inference', is_flag=True, default=False,
              help='Save model input images and submitted action chunks for debugging.')
@click.option('--debug-save-dir', type=str, default=None,
              help='Directory for --debug-save-inference. Defaults to <output>/debug_inference.')
@click.option('--debug-save-every', type=int, default=1,
              help='Save every N submitted action chunks when --debug-save-inference is enabled.')
@click.option('--debug-save-plots', is_flag=True, default=False,
              help='Also write matplotlib timeline PNGs. This is slow; avoid during real robot smoothness tests.')
@click.option('--gripper-action-lead-steps', type=int, default=0,
              help='Use gripper targets this many policy steps ahead while keeping robot poses unchanged.')
@click.option('--robot-action-stitch-steps', type=int, default=0,
              help='Blend the first N submitted robot pose waypoints toward the previous plan; gripper targets are unchanged.')
@click.option('--predict-every-step', is_flag=True, default=False,
              help='Run policy.predict_action() on every control tick, while still submitting every --steps_per_inference ticks.')
@click.option('--keep-robot-queue', is_flag=True, default=False,
              help='Do not clear pending robot waypoints before submitting a new chunk. This is closer to the original UMI scheduling.')
@click.option('--frame-latency-s', type=float, default=None,
              help='Override frame latency wait before policy start. Use 0.0166667 to match the original fixed 1/60s wait.')
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration, max_timesteps,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap, temporal_agg, ensemble_steps,
    headless, dry_run_actions, pdb_before_joint_init, pdb_before_policy_action,
    pdb_before_predict_action, debug_action_start_delay,
    debug_save_inference, debug_save_dir, debug_save_every, debug_save_plots,
    gripper_action_lead_steps, robot_action_stitch_steps,
    predict_every_step, keep_robot_queue, frame_latency_s):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    debug_save_every = max(1, int(debug_save_every))
    debug_save_path = pathlib.Path(debug_save_dir) if debug_save_dir else pathlib.Path(output).joinpath('debug_inference')
    if debug_save_inference:
        print(
            f"Debug inference dump: {debug_save_path} "
            f"(every {debug_save_every} submitted chunk, plots={debug_save_plots})"
        )
    gripper_action_lead_steps = max(0, int(gripper_action_lead_steps))
    if gripper_action_lead_steps > 0:
        print(f"Gripper action lead: using targets {gripper_action_lead_steps} policy steps ahead.")
    robot_action_stitch_steps = max(0, int(robot_action_stitch_steps))
    if robot_action_stitch_steps > 0:
        print(f"Robot action stitching: blending first {robot_action_stitch_steps} submitted robot waypoints.")
    if predict_every_step:
        print("Predict scheduling: running policy.predict_action() on every control tick.")
    if keep_robot_queue:
        print("Robot queue scheduling: keeping pending robot waypoints between submitted chunks.")
    if frame_latency_s is not None:
        frame_latency_s = float(frame_latency_s)
        print(f"Frame latency override: using {frame_latency_s:.6f}s before policy start.")
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    robots_config_for_env = copy.deepcopy(robots_config)
    init_joints_for_env = init_joints
    if dry_run_actions:
        init_joints_for_env = False
        for rc in robots_config_for_env:
            if rc['robot_type'].startswith('franka'):
                rc['read_only'] = True
        print("Dry-run actions: Franka read-only mode enabled; no impedance, joint init, or action updates.")

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    if type(cfg.task.obs_down_sample_steps) == int:
        down_sample_steps = cfg.task.obs_down_sample_steps // cfg.task.action_down_sample_steps
    elif type(cfg.task.obs_down_sample_steps) == omegaconf.listconfig.ListConfig:
        down_sample_steps = [0] + [x // cfg.task.action_down_sample_steps for x in cfg.task.obs_down_sample_steps]
        down_sample_steps = down_sample_steps[::-1]
    
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    force_obs_horizon = (
        cfg.task.shape_meta.obs.robot0_wrench.horizon
        if 'robot0_wrench' in cfg.task.shape_meta.obs
        else cfg.task.shape_meta.obs.robot0_eef_pos.horizon
    )
    with SharedMemoryManager() as shm_manager:
        # 从 yaml 决定相机后端;UVC 时返回 None,env 内部走默认 v4l 自动检测路径
        camera_inst, camera_capture_fps, camera_obs_latency_from_yaml = build_camera_from_yaml(
            shm_manager=shm_manager,
            cameras_cfg=robot_config_data.get('cameras'),
            obs_image_resolution=obs_res,
            camera_obs_latency=0.17,
        )
        with ExitStack() as stack:
            Key = None
            KeyCode = None
            if headless:
                print("Headless mode: no pynput, no OpenCV windows, no SpaceMouse.")
                sm = _NullSpacemouse()
                key_counter = _NullKeyCounter()
            else:
                KeystrokeCounter, Key, KeyCode = _load_keyboard_controls()
                Spacemouse = _load_spacemouse()
                sm = stack.enter_context(Spacemouse(shm_manager=shm_manager))
                key_counter = stack.enter_context(KeystrokeCounter())

            env_kwargs = dict(
                output_dir=output,
                robots_config=robots_config_for_env,
                grippers_config=grippers_config,
                force_sensors_config=robot_config_data.get('force_sensors'),
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints_for_env,
                enable_multi_cam_vis=not headless,
                # latency
                camera_obs_latency=camera_obs_latency_from_yaml,
                camera_capture_fps=camera_capture_fps,
                camera=camera_inst,

                # downsample
                camera_down_sample_steps=down_sample_steps,
                robot_down_sample_steps=down_sample_steps,
                gripper_down_sample_steps=down_sample_steps,
                force_down_sample_steps=down_sample_steps,

                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                force_obs_horizon=force_obs_horizon,
                clear_robot_queue_on_exec=not keep_robot_queue,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager)

            if init_joints_for_env and pdb_before_joint_init:
                joints_init_preview = [
                    rc.get('joints_init')
                    for rc in robots_config_for_env
                ]
                print("[PDB] before env start / joint init.")
                print("      joints_init_preview =", joints_init_preview)
                print("      Continue with `c` to start env and send move_to_joint_positions.")
                breakpoint()

            env = stack.enter_context(BimanualUmiEnv(**env_kwargs))
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                if pdb_before_predict_action:
                    debug_predict_stage = 'warmup'
                    debug_obs_dict_np = obs_dict_np
                    debug_obs_dict = obs_dict
                    debug_obs_shapes = {
                        key: tuple(value.shape)
                        for key, value in obs_dict.items()
                    }
                    debug_env_obs_keys = list(obs.keys())
                    print("[PDB] before warmup policy.predict_action().")
                    print("      Inspect: debug_predict_stage, debug_obs_shapes, debug_obs_dict_np, debug_env_obs_keys")
                    print("      Continue with `c` to run warmup predict_action().")
                    breakpoint()
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)
                action_horizon = action.shape[0]
                action_dim = action.shape[-1]
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            if_first_time = True
            while True:
                # ========= human control loop ==========
                if headless:
                    print("Headless mode: skipping human control and starting policy.")
                else:
                    print("Human in control!")
                    robot_states = env.get_robot_state()
                    target_pose = np.stack([rs['ActualTCPPose'] for rs in robot_states])

                    gripper_states = env.get_gripper_state()
                    gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])

                    control_robot_idx_list = [0]

                    t_start = time.monotonic()
                    iter_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_sample = t_cycle_end - command_latency
                        t_command_target = t_cycle_end + dt

                        # pump obs
                        obs = env.get_obs()

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera{match_camera}_rgb'][-1]
                        match_episode_id = episode_id
                        if match_episode is not None:
                            match_episode_id = match_episode
                        if match_episode_id in episode_first_frame_map:
                            match_img = episode_first_frame_map[match_episode_id]
                            ih, iw, _ = match_img.shape
                            oh, ow, _ = vis_img.shape
                            tf = get_image_transform(
                                input_res=(iw, ih),
                                output_res=(ow, oh),
                                bgr_to_rgb=False)
                            match_img = tf(match_img).astype(np.float32) / 255
                            vis_img = (vis_img + match_img) / 2
                        obs_left_img = obs['camera0_rgb'][-1]
                        obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)

                        text = f'Episode: {episode_id}'
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            lineType=cv2.LINE_AA,
                            thickness=3,
                            color=(0,0,0)
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])
                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        start_policy = False
                        if init_joints and if_first_time:
                            if_first_time = False
                            break
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='q'):
                                # Exit program
                                env.end_episode()
                                exit(0)
                            elif key_stroke == KeyCode(char='c'):
                                # Exit human control loop
                                # hand control over to the policy
                                start_policy = True
                            elif key_stroke == KeyCode(char='e'):
                                # Next episode
                                if match_episode is not None:
                                    match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                            elif key_stroke == KeyCode(char='w'):
                                # Prev episode
                                if match_episode is not None:
                                    match_episode = max(match_episode - 1, 0)
                            elif key_stroke == KeyCode(char='m'):
                                # move the robot
                                duration = 3.0
                                ep = match_replay_buffer.get_episode(match_episode_id)

                                for robot_idx in range(1):
                                    pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                    rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                    grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                    pose = np.concatenate([pos, rot])
                                    env.robots[robot_idx].servoL(pose, duration=duration)
                                    env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                    target_pose[robot_idx] = pose
                                    gripper_target_pos[robot_idx] = grip
                                time.sleep(duration)

                            elif key_stroke == Key.backspace:
                                if click.confirm('Are you sure to drop an episode?'):
                                    env.drop_episode()
                                    key_counter.clear()
                            elif key_stroke == KeyCode(char='a'):
                                control_robot_idx_list = list(range(target_pose.shape[0]))
                            elif key_stroke == KeyCode(char='1'):
                                control_robot_idx_list = [0]
                            elif key_stroke == KeyCode(char='2'):
                                control_robot_idx_list = [1]

                        if start_policy:
                            break

                        precise_wait(t_sample)
                        # get teleop command
                        sm_state = sm.get_motion_state_transformed()
                        # print(sm_state)
                        dpos = sm_state[:3] * (0.5 / frequency)
                        drot_xyz = sm_state[3:] * (1.5 / frequency)

                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        for robot_idx in control_robot_idx_list:
                            target_pose[robot_idx, :3] += dpos
                            target_pose[robot_idx, 3:] = (drot * st.Rotation.from_rotvec(
                                target_pose[robot_idx, 3:])).as_rotvec()

                        dpos = 0
                        if sm.is_button_pressed(0):
                            # close gripper
                            dpos = -gripper_speed / frequency
                        if sm.is_button_pressed(1):
                            dpos = gripper_speed / frequency
                        for robot_idx in control_robot_idx_list:
                            gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)

                        # solve collision with table
                        for robot_idx in control_robot_idx_list:
                            solve_table_collision(
                                ee_pose=target_pose[robot_idx],
                                gripper_width=gripper_target_pos[robot_idx],
                                height_threshold=robots_config[robot_idx]['height_threshold'])

                        # solve collison between two robots
                        solve_sphere_collision(
                            ee_poses=target_pose,
                            robots_config=robots_config,
                            tx_robot1_robot0=tx_robot1_robot0,
                        )

                        action = np.zeros((7 * target_pose.shape[0],))

                        for robot_idx in range(target_pose.shape[0]):
                            action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                            action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]

                        # execute teleop command
                        env.exec_actions(
                            actions=[action],
                            timestamps=[t_command_target-time.monotonic()+time.time()],
                            compensate_latency=False)
                        precise_wait(t_cycle_end)
                        iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = get_policy_start_delay(grippers_config)
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    if dry_run_actions:
                        env.prepare_episode_start(eval_t_start)
                        print("Dry-run actions: recording disabled.")
                    else:
                        env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/capture_fps sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = frame_latency_s
                    if frame_latency is None:
                        frame_latency = 1 / env.camera_capture_fps
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    inference_idx = steps_per_inference
                    all_time_actions = np.zeros((max_timesteps, max_timesteps + action_horizon, action_dim))
                    prev_submitted_target_poses = None
                    prev_submitted_action_timestamps = None

                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        should_submit_actions = inference_idx == steps_per_inference
                        should_run_inference = predict_every_step or temporal_agg or should_submit_actions
                        needs_obs = should_run_inference or (not headless)

                        if needs_obs:
                            # get obs
                            obs = env.get_obs()
                            obs_timestamps = obs['timestamp']
                            print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        else:
                            obs = None
                            obs_timestamps = None

                        if should_run_inference:
                            # run inference
                            with torch.no_grad():
                                s = time.time()
                                obs_dict_np = get_real_umi_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta,
                                    obs_pose_repr=obs_pose_rep,
                                    tx_robot1_robot0=tx_robot1_robot0,
                                    episode_start_pose=episode_start_pose)
                                obs_dict = dict_apply(obs_dict_np,
                                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                if pdb_before_predict_action:
                                    debug_predict_stage = 'policy_loop'
                                    debug_obs_dict_np = obs_dict_np
                                    debug_obs_dict = obs_dict
                                    debug_obs_shapes = {
                                        key: tuple(value.shape)
                                        for key, value in obs_dict.items()
                                    }
                                    debug_env_obs_keys = list(obs.keys())
                                    debug_obs_timestamp = obs_timestamps[-1]
                                    print("[PDB] before policy.predict_action().")
                                    print("      Inspect: debug_predict_stage, debug_obs_shapes, debug_obs_dict_np, debug_env_obs_keys")
                                    print("      Continue with `c` to run predict_action().")
                                    breakpoint()
                                    s = time.time()
                                result = policy.predict_action(obs_dict)
                                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                                action = get_real_umi_action(raw_action, obs, action_pose_repr)  # (16, 7)
                                print('Inference latency:', time.time() - s)
                                all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = action

                        if should_submit_actions:
                            inference_idx = 0

                            if temporal_agg:
                                # temporal ensemble
                                action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                                target_pose_list = []
                                for i in range(action_horizon):
                                    actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_steps + 1):iter_idx + 1, i]
                                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                                    k = -0.01
                                    exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                    exp_weights = exp_weights / exp_weights.sum()
                                    weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(weights=exp_weights).as_rotvec()
                                    weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
                                    weighted_action[0][3:6] = weighted_rotvec
                                    target_pose_list.append(weighted_action)
                                this_target_poses = np.concatenate(target_pose_list, axis=0)
                            else:
                                this_target_poses = action

                            this_target_poses = _apply_gripper_action_lead(
                                this_target_poses,
                                n_robots=len(robots_config),
                                lead_steps=gripper_action_lead_steps,
                            )

                            assert this_target_poses.shape[1] == len(robots_config) * 7
                            enforce_action_collisions(
                                this_target_poses,
                                robots_config=robots_config,
                                tx_robot1_robot0=tx_robot1_robot0,
                            )

                            # deal with timing
                            # the same step actions are always the target for
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                this_target_poses = this_target_poses[[-1]]  # (1, 7)
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            this_target_poses, robot_stitch_stats = _stitch_robot_action_chunk(
                                target_poses=this_target_poses,
                                action_timestamps=action_timestamps,
                                prev_target_poses=prev_submitted_target_poses,
                                prev_action_timestamps=prev_submitted_action_timestamps,
                                n_robots=len(robots_config),
                                stitch_steps=robot_action_stitch_steps,
                            )
                            if robot_stitch_stats['applied_robot_waypoints'] > 0:
                                print(
                                    iter_idx,
                                    "Robot stitch: blended "
                                    f"{robot_stitch_stats['applied_robot_waypoints']} robot waypoints, "
                                    f"max_delta_before_xyz={robot_stitch_stats['max_xyz_delta_before_m']:.4f}m, "
                                    f"max_delta_before_rot={robot_stitch_stats['max_rot_delta_before_rad']:.4f}rad"
                                )
                                enforce_action_collisions(
                                    this_target_poses,
                                    robots_config=robots_config,
                                    tx_robot1_robot0=tx_robot1_robot0,
                                )

                            if (
                                debug_save_inference
                                and (iter_idx % debug_save_every == 0)
                            ):
                                _debug_dump_inference(
                                    debug_save_path,
                                    iter_idx,
                                    obs,
                                    obs_dict_np,
                                    raw_action,
                                    action,
                                    this_target_poses,
                                    action_timestamps,
                                    obs_timestamps,
                                    curr_time,
                                    dt,
                                    len(robots_config),
                                    gripper_action_lead_steps,
                                    robot_stitch_stats,
                                    debug_save_plots,
                                )

                            # execute actions
                            if dry_run_actions:
                                first_action = np.array2string(
                                    this_target_poses[0],
                                    precision=4,
                                    suppress_small=True,
                                )
                                print(iter_idx, f"Dry-run: would submit {len(this_target_poses)} steps. first={first_action}")
                            else:
                                if pdb_before_policy_action:
                                    debug_target_poses = this_target_poses.copy()
                                    debug_action_timestamps = action_timestamps.copy()
                                    debug_raw_action = raw_action.copy()
                                    debug_obs_timestamp = obs_timestamps[-1]
                                    print("[PDB] before policy env.exec_actions().")
                                    print("      Inspect: debug_target_poses, debug_action_timestamps, debug_raw_action")
                                    print("      Continue with `c` to reschedule this same action chunk and send it.")
                                    breakpoint()
                                    action_timestamps = (
                                        time.time()
                                        + float(debug_action_start_delay)
                                        + np.arange(len(this_target_poses), dtype=np.float64) * dt
                                    )
                                env.exec_actions(
                                    actions=this_target_poses,
                                    timestamps=action_timestamps,
                                    compensate_latency=True
                                )
                                print(iter_idx, f"Submitted {len(this_target_poses)} steps of actions.")

                            prev_submitted_target_poses = this_target_poses.copy()
                            prev_submitted_action_timestamps = action_timestamps.copy()

                        stop_episode = False
                        if not headless:
                            # visualize
                            episode_id = env.replay_buffer.n_episodes
                            obs_left_img = obs['camera0_rgb'][-1]
                            obs_right_img = obs['camera0_rgb'][-1]
                            vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255,255,255)
                            )
                            cv2.imshow('default', vis_img[...,::-1])

                            _ = cv2.pollKey()
                            press_events = key_counter.get_press_events()
                            for key_stroke in press_events:
                                if key_stroke == KeyCode(char='s'):
                                    # Stop episode
                                    # Hand control back to human
                                    print('Stopped.')
                                    stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if iter_idx + 1 >= max_timesteps:
                            print("Max Timesteps reached.")
                            stop_episode = True
                        if stop_episode:
                            if not dry_run_actions:
                                env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += 1
                        inference_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    if not dry_run_actions:
                        env.end_episode()
                
                print("Stopped.")
                if headless:
                    break



# %%
if __name__ == '__main__':
    main()
