#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
离线旋转滤波器
使用旋转向量的对数映射进行离线旋转滤波

输入格式: (n, 6) numpy array，其中 n 是采样点个数
格式: [x, y, z, roll, pitch, yaw]

流程：
RPY -> 四元数 -> 旋转向量(log map, 3D) -> 滤波 -> exp map -> 四元数 -> RPY

作者: Reactive Diffusion Policy Team
版本: 1.0.0
"""

import sys
sys.path.append("/home/ps/reactive_diffusion_policy")

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal
from typing import Union, Optional, Tuple
import matplotlib.pyplot as plt
import pickle

class OfflineRotationFilter:
    """
    离线旋转滤波器
    
    使用旋转向量的对数映射进行离线旋转滤波，这是数学上更严格的方法。
    支持批量处理 (n, 6) 格式的轨迹数据。
    
    流程：
    1. RPY -> 四元数
    2. 四元数 -> 旋转向量 (log map)
    3. 对旋转向量的每个分量进行滤波
    4. 旋转向量 -> 四元数 (exp map)
    5. 四元数 -> RPY
    """
    
    def __init__(self, 
                 filter_type: str = 'savgol',
                 window_size: int = 5,
                 polyorder: int = 2,
                 cutoff_freq: float = 1.0,
                 sampling_rate: float = 50.0,
                 order: int = 4):
        """
        初始化离线旋转滤波器
        
        Args:
            filter_type: 滤波类型 ('savgol', 'butterworth', 'moving_average')
            window_size: 滤波窗口大小
            polyorder: Savitzky-Golay 多项式阶数
            cutoff_freq: 截止频率 (Hz)
            sampling_rate: 采样率 (Hz)
            order: Butterworth 滤波器阶数
        """
        self.filter_type = filter_type
        self.window_size = window_size
        self.polyorder = polyorder
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        
        self._init_filter()
        
    def _init_filter(self):
        """初始化滤波器"""
        if self.filter_type == 'butterworth':
            # Butterworth 滤波器
            nyquist = self.sampling_rate / 2
            normal_cutoff = self.cutoff_freq / nyquist
            self.b, self.a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
            
        elif self.filter_type == 'savgol':
            # Savitzky-Golay 滤波器参数
            if self.window_size % 2 == 0:
                self.window_size += 1  # 确保窗口大小为奇数
            if self.polyorder >= self.window_size:
                self.polyorder = self.window_size - 1
                
    def _rpy_to_quaternion(self, rpy: np.ndarray) -> np.ndarray:
        """
        RPY 转四元数
        
        Args:
            rpy: [roll, pitch, yaw] 角度 (弧度)，形状 (n, 3)
            
        Returns:
            四元数 [x, y, z, w]，形状 (n, 4)
        """
        r = R.from_euler('xyz', rpy, degrees=False)
        return r.as_quat()  # [x, y, z, w]
    
    def _quaternion_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转 RPY
        
        Args:
            quat: 四元数 [x, y, z, w]，形状 (n, 4)
            
        Returns:
            [roll, pitch, yaw] 角度 (弧度)，形状 (n, 3)
        """
        r = R.from_quat(quat)
        return r.as_euler('xyz', degrees=False)
    
    def _quaternion_to_rotvec(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转旋转向量 (log map)
        
        Args:
            quat: 四元数 [x, y, z, w]，形状 (n, 4)
            
        Returns:
            旋转向量 [rx, ry, rz] (3D)，形状 (n, 3)
        """
        r = R.from_quat(quat)
        return r.as_rotvec()
    
    def _rotvec_to_quaternion(self, rotvec: np.ndarray) -> np.ndarray:
        """
        旋转向量转四元数 (exp map)
        
        Args:
            rotvec: 旋转向量 [rx, ry, rz] (3D)，形状 (n, 3)
            
        Returns:
            四元数 [x, y, z, w]，形状 (n, 4)
        """
        r = R.from_rotvec(rotvec)
        return r.as_quat()
    
    def _filter_rotation_vector(self, rotvec_array: np.ndarray) -> np.ndarray:
        """
        对旋转向量进行离线滤波
        
        Args:
            rotvec_array: 旋转向量数组，形状 (n, 3)
            
        Returns:
            滤波后的旋转向量，形状 (n, 3)
        """
        if len(rotvec_array) == 1:
            return rotvec_array
        
        if self.filter_type == 'moving_average':
            # 移动平均滤波
            filtered_rotvec = np.zeros_like(rotvec_array)
            for i in range(len(rotvec_array)):
                start_idx = max(0, i - self.window_size + 1)
                end_idx = i + 1
                window_data = rotvec_array[start_idx:end_idx]
                filtered_rotvec[i] = np.mean(window_data, axis=0)
            
        elif self.filter_type == 'butterworth':
            # Butterworth 滤波
            min_length = max(3 * self.order, 15)  # 至少需要 3*order 或 15 个点
            if len(rotvec_array) >= min_length:
                filtered_rotvec = signal.filtfilt(self.b, self.a, rotvec_array, axis=0)
            else:
                # 数据点不够，使用移动平均作为替代
                filtered_rotvec = np.zeros_like(rotvec_array)
                for i in range(len(rotvec_array)):
                    start_idx = max(0, i - self.window_size + 1)
                    end_idx = i + 1
                    window_data = rotvec_array[start_idx:end_idx]
                    filtered_rotvec[i] = np.mean(window_data, axis=0)
                
        elif self.filter_type == 'savgol':
            # Savitzky-Golay 滤波
            if len(rotvec_array) >= self.window_size:
                # 确保 polyorder < window_size
                polyorder = min(self.polyorder, self.window_size - 1)
                if polyorder < 1:
                    polyorder = 1
                    
                filtered_rotvec = signal.savgol_filter(
                    rotvec_array, 
                    self.window_size, 
                    polyorder, 
                    axis=0
                )
            else:
                # 数据点不够，使用移动平均
                filtered_rotvec = np.zeros_like(rotvec_array)
                for i in range(len(rotvec_array)):
                    start_idx = max(0, i - self.window_size + 1)
                    end_idx = i + 1
                    window_data = rotvec_array[start_idx:end_idx]
                    filtered_rotvec[i] = np.mean(window_data, axis=0)
        else:
            raise ValueError(f"不支持的滤波类型: {self.filter_type}")
        
        return filtered_rotvec
    
    def _filter_position(self, pos_array: np.ndarray) -> np.ndarray:
        """
        对位置进行离线滤波
        
        Args:
            pos_array: 位置数组，形状 (n, 3)
            
        Returns:
            滤波后的位置，形状 (n, 3)
        """
        if len(pos_array) == 1:
            return pos_array
        
        if self.filter_type == 'moving_average':
            # 移动平均滤波
            filtered_pos = np.zeros_like(pos_array)
            for i in range(len(pos_array)):
                start_idx = max(0, i - self.window_size + 1)
                end_idx = i + 1
                window_data = pos_array[start_idx:end_idx]
                filtered_pos[i] = np.mean(window_data, axis=0)
            
        elif self.filter_type == 'butterworth':
            # Butterworth 滤波
            min_length = max(3 * self.order, 15)  # 至少需要 3*order 或 15 个点
            if len(pos_array) >= min_length:
                filtered_pos = signal.filtfilt(self.b, self.a, pos_array, axis=0)
            else:
                # 数据点不够，使用移动平均作为替代
                filtered_pos = np.zeros_like(pos_array)
                for i in range(len(pos_array)):
                    start_idx = max(0, i - self.window_size + 1)
                    end_idx = i + 1
                    window_data = pos_array[start_idx:end_idx]
                    filtered_pos[i] = np.mean(window_data, axis=0)
                    
        elif self.filter_type == 'savgol':
            # Savitzky-Golay 滤波
            if len(pos_array) >= self.window_size:
                # 确保 polyorder < window_size
                polyorder = min(self.polyorder, self.window_size - 1)
                if polyorder < 1:
                    polyorder = 1
                    
                filtered_pos = signal.savgol_filter(
                    pos_array, 
                    self.window_size, 
                    polyorder, 
                    axis=0
                )
            else:
                # 数据点不够，使用移动平均
                filtered_pos = np.zeros_like(pos_array)
                for i in range(len(pos_array)):
                    start_idx = max(0, i - self.window_size + 1)
                    end_idx = i + 1
                    window_data = pos_array[start_idx:end_idx]
                    filtered_pos[i] = np.mean(window_data, axis=0)
        else:
            raise ValueError(f"不支持的滤波类型: {self.filter_type}")
        
        return filtered_pos
    
    def filter_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        对轨迹进行离线滤波
        
        Args:
            trajectory: 输入轨迹，形状 (n, 6)，格式 [x, y, z, roll, pitch, yaw]
            
        Returns:
            滤波后的轨迹，形状 (n, 6)
        """
        # 验证输入格式
        if trajectory.ndim != 2 or trajectory.shape[1] != 6:
            raise ValueError("轨迹必须是形状为 (n, 6) 的 numpy 数组")
        
        n_points = trajectory.shape[0]
        if n_points == 0:
            return trajectory.copy()
        
        # 分离位置和旋转
        pos_array = trajectory[:, :3]  # (n, 3)
        rpy_array = trajectory[:, 3:]  # (n, 3)
        
        # 转换为四元数
        quat_array = self._rpy_to_quaternion(rpy_array)  # (n, 4)
        
        # 转换为旋转向量
        rotvec_array = self._quaternion_to_rotvec(quat_array)  # (n, 3)
        
        # 滤波位置
        filtered_pos = self._filter_position(pos_array)
        
        # 滤波旋转向量
        filtered_rotvec = self._filter_rotation_vector(rotvec_array)
        
        # 转换回四元数
        filtered_quat = self._rotvec_to_quaternion(filtered_rotvec)  # (n, 4)
        
        # 转换回 RPY
        filtered_rpy = self._quaternion_to_rpy(filtered_quat)  # (n, 3)
        
        # 组合最终轨迹
        filtered_trajectory = np.concatenate([filtered_pos, filtered_rpy], axis=1)  # (n, 6)
        
        return filtered_trajectory
    
    def get_filter_info(self) -> dict:
        """
        获取滤波器信息
        
        Returns:
            滤波器参数字典
        """
        return {
            'filter_type': self.filter_type,
            'window_size': self.window_size,
            'polyorder': self.polyorder if self.filter_type == 'savgol' else None,
            'cutoff_freq': self.cutoff_freq if self.filter_type == 'butterworth' else None,
            'sampling_rate': self.sampling_rate if self.filter_type == 'butterworth' else None,
            'order': self.order if self.filter_type == 'butterworth' else None,
        }


def plot_filter_comparison(original_traj: np.ndarray, 
                           filtered_traj: np.ndarray,
                           filter_info: dict, 
                           save_path: Optional[str] = None):
    """
    绘制滤波前后轨迹的对比图。
    此版本只对比“原始轨迹”和“滤波后轨迹”。

    Args:
        original_traj: 从PKL文件加载的原始轨迹，形状 (n, 6)
        filtered_traj: 滤波后的轨迹，形状 (n, 6)
        filter_info: 包含滤波器信息的字典
        save_path: 图片保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    
    # 使用数据点索引作为横坐标（时间轴）
    time_steps = np.arange(original_traj.shape[0])
    
    for i in range(6):
        ax = axes[i]

        ax.plot(time_steps, original_traj[:, i], color='gray', linestyle=':', 
                label='Original', alpha=0.8)
        
        # 绘制滤波后轨迹
        ax.plot(time_steps, filtered_traj[:, i], color='red', 
                label='Filtered', linewidth=1.8)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel(labels[i])
        ax.set_title(f'Comparison for {labels[i]}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    filter_type_str = filter_info.get('filter_type', 'Unknown').upper()
    plt.suptitle(f'Filter Performance: {filter_type_str}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  -> 对比图已保存至: {save_path}")
    
    # 如果你不想在运行时看到图片弹出，可以取消下面这行的注释
    # plt.close(fig) 
    # plt.show() # 在脚本运行时显示图片

# ==============================================================================
# 重构后的主函数
# ==============================================================================

def main():
    """
    主函数 - 演示离线旋转滤波器的使用 (从 PKL 文件加载数据)
    """
    print("=" * 60)
    print("离线旋转滤波器演示 (从 PKL 文件加载)")
    print("=" * 60)
    
    pkl_file_path = 'episode0006.pkl' 
    print(f"准备从文件加载轨迹: {pkl_file_path}")

    try:
        with open(pkl_file_path, 'rb') as f:
            all_data = pickle.load(f)

            original_trajectory = np.array([frame.leftRobotTCP for frame in all_data.sensorMessages])

            if len(original_trajectory) < 20:
                print(f"错误: 数据点不足 ({len(original_trajectory)}个)，无法进行有效滤波。")
                return

            print("轨迹加载成功!")
            print(f"轨迹形状: {original_trajectory.shape}")

    except FileNotFoundError:
        print(f"错误: 文件未找到 '{pkl_file_path}'。")
        return
    except Exception as e:
        print(f"加载或处理文件时发生错误: {e}")
        return

    filter_configs = [
        {
            'name': 'Savitzky-Golay',
            'config': {'filter_type': 'savgol', 'window_size': 7, 'polyorder': 2}
        },
        {
            'name': 'Butterworth',
            'config': {'filter_type': 'butterworth', 'window_size': 7, 'cutoff_freq': 2.0, 'sampling_rate': 50.0}
        },
        {
            'name': 'Moving Average',
            'config': {'filter_type': 'moving_average', 'window_size': 7}
        }
    ]
    
    results = {}
    
    for config in filter_configs:
        print(f"\n测试 {config['name']} 滤波器...")
        
        filter_obj = OfflineRotationFilter(**config['config'])
        filtered_trajectory = filter_obj.filter_trajectory(original_trajectory)
        
        results[config['name']] = {
            'filtered_trajectory': filtered_trajectory,
            'filter_info': filter_obj.get_filter_info()
        }
        
    print("\n正在生成并保存滤波对比图...")
    for name, result in results.items():
        plot_filter_comparison(
            original_traj=original_trajectory,
            filtered_traj=result['filtered_trajectory'],
            filter_info=result['filter_info'],
            save_path=f'filter_comparison_{name.lower().replace(" ", "_")}.png'
        )
    
    demo_filter_name = 'Savitzky-Golay'
    print(f"\n以 {demo_filter_name} 为例，显示前10个数据点的变化:")
    print("-" * 80)
    
    demo_result = results[demo_filter_name]
    print("索引 | 原始轨迹 (X, Y, Z, R, P, Y)        | 滤波后轨迹 (X, Y, Z, R, P, Y)")
    print("-" * 80)
    
    for i in range(min(10, original_trajectory.shape[0])):
        original_str = np.array2string(original_trajectory[i], formatter={'float_kind': lambda x: f"{x:6.3f}"})
        filtered_str = np.array2string(demo_result['filtered_trajectory'][i], formatter={'float_kind': lambda x: f"{x:6.3f}"})
        print(f"{i:3d}  | {original_str} | {filtered_str}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()