# 加入了检测图片全黑、重复和delta xyz为0的功能

import sys
from turtle import pos
sys.path.append("/home/ps/omniUMI")

import gc
import numpy as np
import pickle
from typing import Tuple, Dict
from pathlib import Path
import glob
import copy
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import transformations
from scipy.signal import savgol_filter
from reactive_diffusion_policy.filter import OfflineRotationFilter
from reactive_diffusion_policy.common.space_utils import delta_pose_6d


if __name__ == "__main__":
    SAVE_FILTERED_PKL = False
    SKIP_EXISTING_IMAGES = True
    PLOT_FILTERED_DATA = False

    # INPUT_DATA_DIR = "/mnt/wd/wipe/data/dataset_full/wipe1015"
    # OUTPUT_FIGURE_DIR = Path("/mnt/wd/wipe/data/dataset_full/wipe1015_filtered_full_figure_combined/")
    # OUTPUT_PKL_DIR = Path("/mnt/wd/wipe/data/dataset_full/wipe1015_filter")
    
    # INPUT_DATA_DIR = "/mnt/wd/ominiUMI_data_demo"
    # OUTPUT_FIGURE_DIR = Path("/mnt/wd/ominiUMI_data_demo/vio_test_figure")
    # OUTPUT_PKL_DIR = Path("/mnt/wd/ominiUMI_data_demo/output")
    
    # INPUT_DATA_DIR = "/mnt/wd/ominiUMI_data_forward"
    # OUTPUT_FIGURE_DIR = Path("/mnt/wd/ominiUMI_data_forward/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/wd/ominiUMI_data_forwardoutput")
    
    # INPUT_DATA_DIR = "/mnt/wd/ominiUMI_data_forward_roll"
    # OUTPUT_FIGURE_DIR = Path("/mnt/wd/ominiUMI_data_forward_roll/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/wd/ominiUMI_data_forward_roll/output")

    # INPUT_DATA_DIR = "/mnt/1127pickplace"
    # OUTPUT_FIGURE_DIR = Path("/mnt/1127pickplace/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/1127pickplace/output")

    # INPUT_DATA_DIR = "/mnt/1201pickplace"
    # OUTPUT_FIGURE_DIR = Path("/mnt/1201pickplace/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/1201pickplace/output")

    # INPUT_DATA_DIR = "/mnt/storage/1201pickplace_mvs"
    # OUTPUT_FIGURE_DIR = Path("/mnt/storage/1201pickplace_mvs/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/storage/1201pickplace_mvs/output")

    # INPUT_DATA_DIR = "/mnt/storage/1222wipeblackboard"
    # OUTPUT_FIGURE_DIR = Path("/mnt/storage/1222wipeblackboard/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/storage/1222wipeblackboard/output")

    # INPUT_DATA_DIR = "/mnt/storage/20260107wipeblackboard"
    # OUTPUT_FIGURE_DIR = Path("/mnt/storage/20260107wipeblackboard/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/storage/20260107wipeblackboard/output")

    # INPUT_DATA_DIR = "/mnt/storage/20260119wipeblackboard"
    # OUTPUT_FIGURE_DIR = Path("/mnt/storage/20260119wipeblackboard/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/storage/20260119wipeblackboard/output")
    
    # INPUT_DATA_DIR = "/mnt/storage/20260302_wipeblackboard_30fps"
    # OUTPUT_FIGURE_DIR = Path("/mnt/storage/20260302_wipeblackboard_30fps/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/storage/20260302_wipeblackboard_30fps/output")
    
    # INPUT_DATA_DIR = "/mnt/storage/20260311_wipeblackboard_30fps"
    # OUTPUT_FIGURE_DIR = Path("/mnt/storage/20260311_wipeblackboard_30fps/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/storage/20260311_wipeblackboard_30fps/output")

    # INPUT_DATA_DIR = "/mnt/wd/20260323_pickplace_sponge_30fps"
    # OUTPUT_FIGURE_DIR = Path("/mnt/wd/20260323_pickplace_sponge_30fps/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/wd/20260323_pickplace_sponge_30fps/output")

    # INPUT_DATA_DIR = "/mnt/wd/20260325_pickplace_photochip_30fps"
    # OUTPUT_FIGURE_DIR = Path("/mnt/wd/20260325_pickplace_photochip_30fps/figure")
    # OUTPUT_PKL_DIR = Path("/mnt/wd/20260325_pickplace_photochip_30fps/output")

    
    INPUT_DATA_DIR = "/mnt/wd/20260327_pickplace_watercup_30fps"
    OUTPUT_FIGURE_DIR = Path("/mnt/wd/20260327_pickplace_watercup_30fps/figure")
    OUTPUT_PKL_DIR = Path("/mnt/wd/20260327_pickplace_watercup_30fps/output")
    
    TRANS_DELTA_THRESHOLD = 0.05
    ROT_DELTA_THRESHOLD = 5.0

    FILTER_CONFIG = {
        'filter_type': 'savgol',
        'window_size': 7,
        'polyorder': 2
    }
    
    input_data_dir = Path(INPUT_DATA_DIR)

    output_pkl_dir = None
    if SAVE_FILTERED_PKL:
        if OUTPUT_PKL_DIR is None:
            output_pkl_dir = input_data_dir.parent / f"{input_data_dir.name}_filtered"
        else:
            output_pkl_dir = Path(OUTPUT_PKL_DIR)
        
        output_pkl_dir.mkdir(parents=True, exist_ok=True)
        print(f"滤波后的 .pkl 文件将被保存至: '{output_pkl_dir}'")

    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    all_pkl_files = glob.glob(f"{input_data_dir}/**/*.pkl", recursive=True)
    if not all_pkl_files:
        print(f"错误: 在 '{input_data_dir}' 中未找到任何 .pkl 文件。")
        exit()
        
    print(f"找到 {len(all_pkl_files)} 个 .pkl 文件。开始生成分析图...")
    
    filter_obj = OfflineRotationFilter(**FILTER_CONFIG)

    for pkl_file in all_pkl_files:
        pkl_path = Path(pkl_file)
        episode_name = pkl_path.stem

        if SKIP_EXISTING_IMAGES:
            search_pattern = str(OUTPUT_FIGURE_DIR / f"{episode_name}_dashboard*.png")
            existing_images = glob.glob(search_pattern)
            if existing_images:
                # 兼容旧命名：旧文件只有 fisheyedup，没有 fps/ratio 信息，允许自动重绘。
                has_legacy_fisheye_dup_name = any(img.endswith("_fisheyedup.png") for img in existing_images)
                if not has_legacy_fisheye_dup_name:
                    print(f"  - 跳过 (图像已存在): {episode_name}")
                    continue
                print(f"  - 检测到旧fisheyedup命名，重新生成: {episode_name}")

        print(f"正在处理和绘图: {episode_name}")
        
        try:
            with open(pkl_file, 'rb') as f:
                all_data = pickle.load(f)

            sensor_messages = all_data.sensorMessages
            poses = np.array([frame.leftRobotTCP for frame in sensor_messages])
            timestamps = np.array([frame.timestamp for frame in sensor_messages])
            wrench = np.array([frame.leftRobotTCPWrench for frame in sensor_messages])

            if len(poses) < FILTER_CONFIG['window_size']:
                print(f"  - 跳过 (数据点不足): {episode_name}")
                continue

            # Fisheye 异常检测（字段缺失时自动跳过）
            fisheye_all_black = False
            fisheye_has_duplicate = False
            fisheye_frame_count = 0
            fisheye_hashes = set()
            fisheye_duplicate_consecutive_count = 0
            fisheye_duplicate_duration = 0.0
            fisheye_first_timestamp = None
            fisheye_last_timestamp = None
            prev_fisheye_hash = None
            prev_fisheye_timestamp = None
            for frame in sensor_messages:
                fisheye_img = getattr(frame, "fisheyeCameraRGB", None)
                if fisheye_img is None:
                    continue
                frame_timestamp = float(getattr(frame, "timestamp", 0.0))
                fisheye_frame = np.asarray(fisheye_img)
                fisheye_frame_count += 1
                if fisheye_first_timestamp is None:
                    fisheye_first_timestamp = frame_timestamp
                fisheye_last_timestamp = frame_timestamp
                if fisheye_frame_count == 1:
                    fisheye_all_black = True
                if fisheye_all_black and not np.all(fisheye_frame == 0):
                    fisheye_all_black = False
                frame_hash = hashlib.sha1(np.ascontiguousarray(fisheye_frame).tobytes()).digest()
                if frame_hash in fisheye_hashes:
                    fisheye_has_duplicate = True
                else:
                    fisheye_hashes.add(frame_hash)
                if prev_fisheye_hash is not None and frame_hash == prev_fisheye_hash:
                    fisheye_duplicate_consecutive_count += 1
                    dt = frame_timestamp - prev_fisheye_timestamp
                    if dt > 0:
                        fisheye_duplicate_duration += dt
                prev_fisheye_hash = frame_hash
                prev_fisheye_timestamp = frame_timestamp
            if fisheye_frame_count == 0:
                fisheye_all_black = False

            if fisheye_first_timestamp is None or fisheye_last_timestamp is None:
                fisheye_total_duration = 0.0
            else:
                fisheye_total_duration = max(0.0, fisheye_last_timestamp - fisheye_first_timestamp)
            if fisheye_total_duration > 0:
                fisheye_duplicate_fps = fisheye_duplicate_consecutive_count / fisheye_total_duration
                fisheye_duplicate_ratio = min(1.0, fisheye_duplicate_duration / fisheye_total_duration)
            else:
                fisheye_duplicate_fps = 0.0
                fisheye_duplicate_ratio = 0.0

            print(
                "  - Fisheye连续重复帧统计: "
                f"重复帧 {fisheye_duplicate_consecutive_count} 帧, "
                f"重复帧率 {fisheye_duplicate_fps:.3f} fps, "
                f"重复时长占比 {fisheye_duplicate_ratio * 100:.2f}%"
            )

            filtered_poses = filter_obj.filter_trajectory(poses)

            if SAVE_FILTERED_PKL:
                print(f"  - 正在保存滤波后的 .pkl 文件...")
                filtered_data = copy.deepcopy(all_data)
                
                for i in range(len(filtered_poses)):
                    filtered_data.sensorMessages[i].leftRobotTCP = filtered_poses[i]
                
                output_pkl_path = output_pkl_dir / pkl_path.name
                with open(output_pkl_path, 'wb') as f_out:
                    pickle.dump(filtered_data, f_out)
                print(f"  - 已成功保存到: {output_pkl_path}")
                del filtered_data

            # 释放大量内存: sensor_messages 持有对 all_data 中所有帧(含图像)的引用
            del sensor_messages
            del all_data

            relative_time = timestamps - timestamps[0]
            force, torque = wrench[:, :3], wrench[:, 3:]
            force_magnitude = np.linalg.norm(force, axis=1)
            torque_magnitude = np.linalg.norm(torque, axis=1)
            
            # print("  - 正在使用 delta_pose_6d 计算帧间差异...")
            delta_poses_list = [delta_pose_6d(poses[i-1], poses[i]) for i in range(1, len(poses))]
            delta_poses = np.array([[0.0]*6] + delta_poses_list)
            
            delta_translations = delta_poses[:, :3]
            delta_rotations_rad = delta_poses[:, 3:]
            delta_xyz_all_zero = len(delta_translations) > 1 and np.all(delta_translations[1:] == 0)
            
            translation_distances = np.linalg.norm(delta_translations, axis=1)
            # print("  - 差异计算完成。")
            
            # pose_diff = np.diff(poses[:, :3], axis=0)
            # translation_distances = np.insert(np.linalg.norm(pose_diff, axis=1), 0, 0)
            
            rpy_array = poses[:, 3:]
            quaternions = filter_obj._rpy_to_quaternion(rpy_array)
            axis_angles = filter_obj._quaternion_to_rotvec(quaternions)
            
            # dot_products = (quaternions[:-1, :] * quaternions[1:, :]).sum(axis=1)
            # angle_rad = 2 * np.arccos(np.clip(np.abs(dot_products), -1.0, 1.0))
            # rotation_degrees = np.insert(np.rad2deg(angle_rad), 0, 0)
            
            fig = plt.figure(figsize=(20, 35))
            fig.suptitle(
                f"Episode Analysis: {episode_name}\n"
                f"Fisheye duplicate: {fisheye_duplicate_fps:.3f} fps, {fisheye_duplicate_ratio * 100:.2f}% duration",
                fontsize=16
            )
            gs = gridspec.GridSpec(7, 4, figure=fig)
            
            ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
            ax_3d.plot(poses[:, 0], poses[:, 1], poses[:, 2], color='green', label='Trajectory')
            ax_3d.scatter(poses[0, 0], poses[0, 1], poses[0, 2], color='blue', s=100, label='Start Point', depthshade=True)
            ax_3d.scatter(poses[-1, 0], poses[-1, 1], poses[-1, 2], color='red', s=100, label='End Point', depthshade=True)
            ax_3d.set_title('3D Trajectory'); ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
            ax_3d.legend()
        
            plot_labels_pos = ['X', 'Y', 'Z']
            for i, label in enumerate(plot_labels_pos):
                ax = fig.add_subplot(gs[0, i + 1])
                ax.plot(relative_time, poses[:, i], label='Original', color='dimgray', alpha=0.9, linewidth=1.5)
                if PLOT_FILTERED_DATA:
                    ax.plot(relative_time, filtered_poses[:, i], label='Filtered', color='crimson', linewidth=1.2)
                ax.set_title(f'{label} Position (m)')
                ax.grid(True)
                ax.legend()

            fig.add_subplot(gs[1, 0]).axis('off')
            plot_labels_rot = ['Roll', 'Pitch', 'Yaw']
            for i, label in enumerate(plot_labels_rot):
                ax = fig.add_subplot(gs[1, i + 1])
                ax.plot(relative_time, np.rad2deg(poses[:, i+3]), label='Original', color='dimgray', alpha=0.9, linewidth=1.5)
                if PLOT_FILTERED_DATA:
                    ax.plot(relative_time, np.rad2deg(filtered_poses[:, i+3]), label='Filtered', color='crimson', linewidth=1.2)
                ax.set_title(f'{label} (degrees)')
                ax.grid(True)
                ax.legend()

            max_trans_dist = np.max(translation_distances)
            ax_dt = fig.add_subplot(gs[2, 0]); ax_dt.plot(relative_time, translation_distances, color='royalblue'); ax_dt.set_title(f'Delta Translation\nMax: {max_trans_dist:.4f} m/frame'); ax_dt.grid(True)
            if max_trans_dist < TRANS_DELTA_THRESHOLD: ax_dt.set_ylim(0, TRANS_DELTA_THRESHOLD)
            # max_rot_deg = np.max(rotation_degrees)
            delta_rotations_deg = np.rad2deg(delta_rotations_rad)
            max_rot_deg = np.max(np.abs(delta_rotations_deg))
            # ax_dr = fig.add_subplot(gs[2, 1]); ax_dr.plot(relative_time, rotation_degrees, color='crimson'); ax_dr.set_title(f'Delta Rotation\nMax: {max_rot_deg:.2f} deg/frame'); ax_dr.grid(True)
            ax_dr = fig.add_subplot(gs[2, 1]); ax_dr.plot(relative_time, delta_rotations_deg, color='crimson'); ax_dr.set_title(f'Delta Rotation\nMax: {max_rot_deg:.2f} deg/frame'); ax_dr.grid(True)
            if max_rot_deg < ROT_DELTA_THRESHOLD: ax_dr.set_ylim(0, ROT_DELTA_THRESHOLD)
            fig.add_subplot(gs[2, 2]).axis('off'); fig.add_subplot(gs[2, 3]).axis('off')

            ax_fm = fig.add_subplot(gs[3, 0]); ax_fm.plot(relative_time, force_magnitude, color='purple'); ax_fm.set_title('Force Norm (N)'); ax_fm.grid(True)
            for i, label in enumerate(['Fx', 'Fy', 'Fz']): ax = fig.add_subplot(gs[3, i+1]); ax.plot(relative_time, force[:, i]); ax.set_title(f'{label} (N)'); ax.grid(True)
            
            ax_tqm = fig.add_subplot(gs[4, 0]); ax_tqm.plot(relative_time, torque_magnitude, color='purple'); ax_tqm.set_title('Torque Norm (Nm)'); ax_tqm.grid(True)
            for i, label in enumerate(['Tx', 'Ty', 'Tz']): ax = fig.add_subplot(gs[4, i+1]); ax.plot(relative_time, torque[:, i]); ax.set_title(f'{label} (Nm)'); ax.grid(True)
            
            quat_labels = ['Qx', 'Qy', 'Qz', 'Qw']
            for i, label in enumerate(quat_labels):
                ax = fig.add_subplot(gs[5, i])
                ax.plot(relative_time, quaternions[:, i])
                ax.set_title(f'Quaternion {label}')
                ax.grid(True)
            
            aa_labels = ['Rx', 'Ry', 'Rz']
            for i, label in enumerate(aa_labels):
                ax = fig.add_subplot(gs[6, i])
                ax.plot(relative_time, axis_angles[:, i])
                ax.set_title(f'Axis-Angle {label}')
                ax.grid(True)
            fig.add_subplot(gs[6, 3]).axis('off')

            fig.tight_layout(rect=[0, 0, 1, 0.98])
            
            filename_suffix = ""
            if max_trans_dist > TRANS_DELTA_THRESHOLD or max_rot_deg > ROT_DELTA_THRESHOLD:
                trans_m_str = f"{max_trans_dist:.4f}"; rot_deg_int = int(max_rot_deg)
                filename_suffix = f"_maxdelta_{trans_m_str}m_{rot_deg_int}d"
            anomaly_tags = []
            if fisheye_all_black:
                anomaly_tags.append("fisheyeblack")
            if fisheye_has_duplicate:
                dup_fps_tag = f"{fisheye_duplicate_fps:.3f}".replace(".", "p")
                dup_ratio_tag = f"{fisheye_duplicate_ratio * 100:.2f}".replace(".", "p")
                anomaly_tags.append(f"fisheyedup_{dup_fps_tag}fps_{dup_ratio_tag}pct")
            if delta_xyz_all_zero:
                anomaly_tags.append("deltaxyzzero")
            if anomaly_tags:
                filename_suffix += "_" + "_".join(anomaly_tags)
            output_filename = OUTPUT_FIGURE_DIR / f"{episode_name}_dashboard{filename_suffix}.png"
            fig.savefig(output_filename, dpi=150)
            plt.close(fig)
            plt.close('all')
            gc.collect()

        except Exception as e:
            print(f"  - 跳过文件 {episode_name} 因为出现错误: {e}")
            plt.close('all')
            gc.collect()

    print(f"\n--- 所有分析图已成功保存至 '{OUTPUT_FIGURE_DIR}'. ---")
