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

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal
from typing import Union, Optional, Tuple
import matplotlib.pyplot as plt

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
        
        # 初始化滤波器
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


def generate_test_trajectory(num_points: int = 100, 
                           noise_level: float = 0.1,
                           rotation_amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成测试轨迹
    
    Args:
        num_points: 轨迹点数
        noise_level: 噪声水平
        rotation_amplitude: 旋转幅度
        
    Returns:
        (真实轨迹, 噪声轨迹)，形状均为 (n, 6)
    """
    t = np.linspace(0, 4*np.pi, num_points)
    
    true_trajectory = np.zeros((num_points, 6))
    noisy_trajectory = np.zeros((num_points, 6))
    
    for i, time in enumerate(t):
        # 真实轨迹
        x = 0.1 * np.sin(0.5 * time)
        y = 0.1 * np.cos(0.5 * time)
        z = 0.05 * time
        
        roll = rotation_amplitude * 0.1 * np.sin(0.3 * time)
        pitch = rotation_amplitude * 0.1 * np.cos(0.3 * time)
        yaw = rotation_amplitude * 0.2 * time
        
        true_trajectory[i] = [x, y, z, roll, pitch, yaw]
        
        # 添加噪声
        noisy_trajectory[i] = true_trajectory[i] + noise_level * np.random.randn(6)
    
    return true_trajectory, noisy_trajectory


def plot_filter_comparison(true_traj: np.ndarray, 
                          noisy_traj: np.ndarray, 
                          filtered_traj: np.ndarray,
                          filter_info: dict, 
                          save_path: Optional[str] = None):
    """
    绘制滤波对比图
    
    Args:
        true_traj: 真实轨迹，形状 (n, 6)
        noisy_traj: 噪声轨迹，形状 (n, 6)
        filtered_traj: 滤波轨迹，形状 (n, 6)
        filter_info: 滤波器信息
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    colors = ['red', 'green', 'blue']
    
    n_points = true_traj.shape[0]
    t = np.linspace(0, 4*np.pi, n_points)
    
    for i in range(6):
        ax = axes[i]
        
        # 真实轨迹
        true_data = true_traj[:, i]
        ax.plot(t, true_data, 'k-', label='True', linewidth=2)
        
        # 噪声轨迹
        noisy_data = noisy_traj[:, i]
        ax.plot(t, noisy_data, 'gray', label='Noisy', alpha=0.7)
        
        # 滤波轨迹
        filtered_data = filtered_traj[:, i]
        ax.plot(t, filtered_data, colors[i % len(colors)], 
                label=f'Filtered ({filter_info["filter_type"]})', linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{labels[i]}')
        ax.set_title(f'{labels[i]} Position')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle(f'Offline Rotation Filter - {filter_info["filter_type"].upper()}', 
                 fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_filter_error(true_traj: np.ndarray, filtered_traj: np.ndarray) -> dict:
    """
    计算滤波误差
    
    Args:
        true_traj: 真实轨迹，形状 (n, 6)
        filtered_traj: 滤波轨迹，形状 (n, 6)
        
    Returns:
        误差统计字典
    """
    # 位置误差
    pos_errors = np.linalg.norm(true_traj[:, :3] - filtered_traj[:, :3], axis=1)
    
    # 旋转误差（使用旋转矩阵）
    rot_errors = np.zeros(true_traj.shape[0])
    for i in range(true_traj.shape[0]):
        r_true = R.from_euler('xyz', true_traj[i, 3:], degrees=False)
        r_filtered = R.from_euler('xyz', filtered_traj[i, 3:], degrees=False)
        rot_errors[i] = np.linalg.norm((r_true.inv() * r_filtered).as_rotvec())
    
    total_errors = pos_errors + rot_errors
    
    return {
        'position_error_mean': np.mean(pos_errors),
        'position_error_std': np.std(pos_errors),
        'rotation_error_mean': np.mean(rot_errors),
        'rotation_error_std': np.std(rot_errors),
        'total_error_mean': np.mean(total_errors),
        'total_error_std': np.std(total_errors)
    }


def main():
    """
    主函数 - 演示离线旋转滤波器的使用
    """
    print("=" * 60)
    print("离线旋转滤波器演示")
    print("=" * 60)
    
    # 生成测试数据
    print("生成测试轨迹...")
    true_trajectory, noisy_trajectory = generate_test_trajectory(
        num_points=200, 
        noise_level=0.15,
        rotation_amplitude=1.0
    )
    
    print(f"轨迹形状: {true_trajectory.shape}")
    print(f"数据格式: (n_points, 6) = (x, y, z, roll, pitch, yaw)")
    
    # 测试不同的滤波器
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
        
        # 创建滤波器
        filter_obj = OfflineRotationFilter(**config['config'])
        
        # 滤波轨迹
        filtered_trajectory = filter_obj.filter_trajectory(noisy_trajectory)
        
        # 验证输出形状
        print(f"  输入形状: {noisy_trajectory.shape}")
        print(f"  输出形状: {filtered_trajectory.shape}")
        
        # 计算误差
        error_stats = calculate_filter_error(true_trajectory, filtered_trajectory)
        
        # 存储结果
        results[config['name']] = {
            'filter_obj': filter_obj,
            'filtered_trajectory': filtered_trajectory,
            'error_stats': error_stats,
            'filter_info': filter_obj.get_filter_info()
        }
        
        # 打印误差统计
        print(f"  位置误差: {error_stats['position_error_mean']:.4f} ± {error_stats['position_error_std']:.4f}")
        print(f"  旋转误差: {error_stats['rotation_error_mean']:.4f} ± {error_stats['rotation_error_std']:.4f}")
        print(f"  总误差: {error_stats['total_error_mean']:.4f} ± {error_stats['total_error_std']:.4f}")
    
    # 绘制对比图
    print("\n绘制滤波对比图...")
    for name, result in results.items():
        plot_filter_comparison(
            true_trajectory, 
            noisy_trajectory, 
            result['filtered_trajectory'],
            result['filter_info'],
            save_path=f'offline_filter_comparison_{name.lower().replace(" ", "_")}.png'
        )
    
    # 打印最佳滤波器
    best_filter = min(results.keys(), 
                     key=lambda x: results[x]['error_stats']['total_error_mean'])
    print(f"\n最佳滤波器: {best_filter}")
    print(f"总误差: {results[best_filter]['error_stats']['total_error_mean']:.4f}")
    
    # 显示前几个数据点的对比
    print("\n" + "=" * 60)
    print("数据点对比 (前10个点)")
    print("=" * 60)
    
    best_result = results[best_filter]
    print("索引 | 原始轨迹                    | 滤波轨迹")
    print("-" * 80)
    
    for i in range(min(10, true_trajectory.shape[0])):
        original = noisy_trajectory[i]
        filtered = best_result['filtered_trajectory'][i]
        
        print(f"{i:3d}  | "
              f"[{original[0]:6.3f}, {original[1]:6.3f}, {original[2]:6.3f}, "
              f"{original[3]:6.3f}, {original[4]:6.3f}, {original[5]:6.3f}] | "
              f"[{filtered[0]:6.3f}, {filtered[1]:6.3f}, {filtered[2]:6.3f}, "
              f"{filtered[3]:6.3f}, {filtered[4]:6.3f}, {filtered[5]:6.3f}]")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    # 打印使用说明
    print("\n使用说明:")
    print("1. 创建滤波器: filter_obj = OfflineRotationFilter(filter_type='savgol')")
    print("2. 滤波轨迹: filtered_traj = filter_obj.filter_trajectory(trajectory)")
    print("3. 输入格式: (n, 6) numpy array，[x, y, z, roll, pitch, yaw]")
    print("4. 输出格式: (n, 6) numpy array，[x, y, z, roll, pitch, yaw]")
    print("\n支持的滤波类型:")
    print("- 'savgol': Savitzky-Golay 滤波 (推荐)")
    print("- 'butterworth': Butterworth 低通滤波")
    print("- 'moving_average': 移动平均滤波")


if __name__ == "__main__":
    main()
