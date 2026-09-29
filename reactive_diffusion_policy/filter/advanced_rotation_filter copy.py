#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高级旋转滤波器
使用旋转向量的对数映射进行旋转滤波，避免四元数平均的问题

流程：
RPY -> 四元数 -> 旋转向量(log map, 3D) -> 滤波 -> exp map -> 四元数 -> RPY

作者: Reactive Diffusion Policy Team
版本: 1.0.0
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal
from collections import deque
from typing import List, Union, Optional, Tuple
import matplotlib.pyplot as plt

class AdvancedRotationFilter:
    """
    高级旋转滤波器
    
    使用旋转向量的对数映射进行旋转滤波，这是数学上更严格的方法。
    避免了四元数平均可能遇到的问题，如四元数双覆盖问题。
    
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
        初始化高级旋转滤波器
        
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
        
        # 状态变量
        self.is_initialized = False
        self.last_pose = None
        self.rotation_buffer = deque(maxlen=window_size)
        self.position_buffer = deque(maxlen=window_size)
        
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
            rpy: [roll, pitch, yaw] 角度 (弧度)
            
        Returns:
            四元数 [x, y, z, w]
        """
        r = R.from_euler('xyz', rpy, degrees=False)
        return r.as_quat()  # [x, y, z, w]
    
    def _quaternion_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转 RPY
        
        Args:
            quat: 四元数 [x, y, z, w]
            
        Returns:
            [roll, pitch, yaw] 角度 (弧度)
        """
        r = R.from_quat(quat)
        return r.as_euler('xyz', degrees=False)
    
    def _quaternion_to_rotvec(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转旋转向量 (log map)
        
        Args:
            quat: 四元数 [x, y, z, w]
            
        Returns:
            旋转向量 [rx, ry, rz] (3D)
        """
        r = R.from_quat(quat)
        return r.as_rotvec()
    
    def _rotvec_to_quaternion(self, rotvec: np.ndarray) -> np.ndarray:
        """
        旋转向量转四元数 (exp map)
        
        Args:
            rotvec: 旋转向量 [rx, ry, rz] (3D)
            
        Returns:
            四元数 [x, y, z, w]
        """
        r = R.from_rotvec(rotvec)
        return r.as_quat()
    
    def _filter_rotation_vector(self, rotvec_history: List[np.ndarray]) -> np.ndarray:
        """
        对旋转向量历史进行滤波
        
        Args:
            rotvec_history: 旋转向量历史列表
            
        Returns:
            滤波后的旋转向量
        """
        if len(rotvec_history) == 1:
            return rotvec_history[0]
        
        # 转换为 numpy 数组
        rotvec_array = np.array(rotvec_history)
        
        if self.filter_type == 'moving_average':
            # 移动平均
            filtered_rotvec = np.mean(rotvec_array, axis=0)
            
        elif self.filter_type == 'butterworth':
            # Butterworth 滤波器需要足够的数据点
            min_length = max(3 * self.order, 15)  # 至少需要 3*order 或 15 个点
            if len(rotvec_array) >= min_length:
                filtered_rotvec = signal.filtfilt(self.b, self.a, rotvec_array, axis=0)[-1]
            else:
                # 数据点不够，使用移动平均作为替代
                filtered_rotvec = np.mean(rotvec_array, axis=0)
                
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
                )[-1]
            else:
                # 使用较小的窗口
                window = min(len(rotvec_array), self.window_size)
                if window % 2 == 0:
                    window -= 1
                if window < 3:  # 至少需要3个点
                    filtered_rotvec = rotvec_array[-1]
                else:
                    polyorder = min(self.polyorder, window - 1)
                    if polyorder < 1:
                        polyorder = 1
                        
                    filtered_rotvec = signal.savgol_filter(
                        rotvec_array, 
                        window, 
                        polyorder, 
                        axis=0
                    )[-1]
        else:
            raise ValueError(f"不支持的滤波类型: {self.filter_type}")
        
        return filtered_rotvec
    
    def _filter_position(self, pos_history: List[np.ndarray]) -> np.ndarray:
        """
        对位置历史进行滤波
        
        Args:
            pos_history: 位置历史列表
            
        Returns:
            滤波后的位置
        """
        if len(pos_history) == 1:
            return pos_history[0]
        
        pos_array = np.array(pos_history)
        
        if self.filter_type == 'moving_average':
            return np.mean(pos_array, axis=0)
        elif self.filter_type == 'butterworth':
            # Butterworth 滤波器需要足够的数据点
            min_length = max(3 * self.order, 15)  # 至少需要 3*order 或 15 个点
            if len(pos_array) >= min_length:
                return signal.filtfilt(self.b, self.a, pos_array, axis=0)[-1]
            else:
                # 数据点不够，使用移动平均作为替代
                return np.mean(pos_array, axis=0)
        elif self.filter_type == 'savgol':
            if len(pos_array) >= self.window_size:
                # 确保 polyorder < window_size
                polyorder = min(self.polyorder, self.window_size - 1)
                if polyorder < 1:
                    polyorder = 1
                    
                return signal.savgol_filter(
                    pos_array, 
                    self.window_size, 
                    polyorder, 
                    axis=0
                )[-1]
            else:
                window = min(len(pos_array), self.window_size)
                if window % 2 == 0:
                    window -= 1
                if window < 3:  # 至少需要3个点
                    return pos_array[-1]
                else:
                    polyorder = min(self.polyorder, window - 1)
                    if polyorder < 1:
                        polyorder = 1
                        
                    return signal.savgol_filter(
                        pos_array, 
                        window, 
                        polyorder, 
                        axis=0
                    )[-1]
        else:
            return pos_array[-1]
    
    def filter_pose(self, pose_xyzrpy: Union[List, np.ndarray]) -> np.ndarray:
        """
        对单个位姿进行滤波
        
        Args:
            pose_xyzrpy: 输入位姿 [x, y, z, roll, pitch, yaw]
            
        Returns:
            滤波后的位姿 [x, y, z, roll, pitch, yaw]
        """
        pose = np.array(pose_xyzrpy, dtype=np.float64)
        if len(pose) != 6:
            raise ValueError("位姿必须是6维向量 [x, y, z, roll, pitch, yaw]")
        
        # 分离位置和旋转
        pos = pose[:3]
        rpy = pose[3:]
        
        # 转换为四元数
        quat = self._rpy_to_quaternion(rpy)
        
        # 转换为旋转向量
        rotvec = self._quaternion_to_rotvec(quat)
        
        if not self.is_initialized:
            self.last_pose = pose.copy()
            self.position_buffer.append(pos.copy())
            self.rotation_buffer.append(rotvec.copy())
            self.is_initialized = True
            return pose
        
        # 添加到缓冲区
        self.position_buffer.append(pos.copy())
        self.rotation_buffer.append(rotvec.copy())
        
        # 滤波位置
        filtered_pos = self._filter_position(list(self.position_buffer))
        
        # 滤波旋转向量
        filtered_rotvec = self._filter_rotation_vector(list(self.rotation_buffer))
        
        # 转换回四元数
        filtered_quat = self._rotvec_to_quaternion(filtered_rotvec)
        
        # 转换回 RPY
        filtered_rpy = self._quaternion_to_rpy(filtered_quat)
        
        # 组合最终位姿
        filtered_pose = np.concatenate([filtered_pos, filtered_rpy])
        
        self.last_pose = filtered_pose.copy()
        return filtered_pose
    
    def filter_trajectory(self, trajectory: List[List[float]]) -> List[List[float]]:
        """
        对整个轨迹进行滤波
        
        Args:
            trajectory: 轨迹列表，每个元素是 [x, y, z, roll, pitch, yaw]
            
        Returns:
            滤波后的轨迹
        """
        filtered_trajectory = []
        for pose in trajectory:
            filtered_pose = self.filter_pose(pose)
            filtered_trajectory.append(filtered_pose.tolist())
        return filtered_trajectory
    
    def reset(self):
        """重置滤波器状态"""
        self.is_initialized = False
        self.last_pose = None
        self.position_buffer.clear()
        self.rotation_buffer.clear()
    
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
            'is_initialized': self.is_initialized,
            'buffer_size': len(self.position_buffer)
        }


def generate_test_trajectory(num_points: int = 100, 
                           noise_level: float = 0.1,
                           rotation_amplitude: float = 1.0) -> Tuple[List, List]:
    """
    生成测试轨迹
    
    Args:
        num_points: 轨迹点数
        noise_level: 噪声水平
        rotation_amplitude: 旋转幅度
        
    Returns:
        (真实轨迹, 噪声轨迹)
    """
    t = np.linspace(0, 4*np.pi, num_points)
    
    true_trajectory = []
    noisy_trajectory = []
    
    for i, time in enumerate(t):
        # 真实轨迹
        x = 0.1 * np.sin(0.5 * time)
        y = 0.1 * np.cos(0.5 * time)
        z = 0.05 * time
        
        roll = rotation_amplitude * 0.1 * np.sin(0.3 * time)
        pitch = rotation_amplitude * 0.1 * np.cos(0.3 * time)
        yaw = rotation_amplitude * 0.2 * time
        
        true_pose = [x, y, z, roll, pitch, yaw]
        true_trajectory.append(true_pose)
        
        # 添加噪声
        noisy_pose = [p + noise_level * np.random.randn() for p in true_pose]
        noisy_trajectory.append(noisy_pose)
    
    return true_trajectory, noisy_trajectory


def plot_filter_comparison(true_traj, noisy_traj, filtered_traj, 
                          filter_info: dict, save_path: Optional[str] = None):
    """
    绘制滤波对比图
    
    Args:
        true_traj: 真实轨迹
        noisy_traj: 噪声轨迹
        filtered_traj: 滤波轨迹
        filter_info: 滤波器信息
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    colors = ['red', 'green', 'blue']
    
    for i in range(6):
        ax = axes[i]
        
        t = np.linspace(0, 4*np.pi, len(true_traj))
        
        # 真实轨迹
        true_data = [pose[i] for pose in true_traj]
        ax.plot(t, true_data, 'k-', label='True', linewidth=2)
        
        # 噪声轨迹
        noisy_data = [pose[i] for pose in noisy_traj]
        ax.plot(t, noisy_data, 'gray', label='Noisy', alpha=0.7)
        
        # 滤波轨迹
        filtered_data = [pose[i] for pose in filtered_traj]
        ax.plot(t, filtered_data, colors[i % len(colors)], 
                label=f'Filtered ({filter_info["filter_type"]})', linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{labels[i]}')
        ax.set_title(f'{labels[i]} Position')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle(f'Advanced Rotation Filter - {filter_info["filter_type"].upper()}', 
                 fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_filter_error(true_traj, filtered_traj) -> dict:
    """
    计算滤波误差
    
    Args:
        true_traj: 真实轨迹
        filtered_traj: 滤波轨迹
        
    Returns:
        误差统计字典
    """
    errors = []
    
    for true, filtered in zip(true_traj, filtered_traj):
        # 位置误差
        pos_error = np.linalg.norm(np.array(true[:3]) - np.array(filtered[:3]))
        
        # 旋转误差（使用旋转矩阵）
        r_true = R.from_euler('xyz', true[3:], degrees=False)
        r_filtered = R.from_euler('xyz', filtered[3:], degrees=False)
        rot_error = np.linalg.norm((r_true.inv() * r_filtered).as_rotvec())
        
        errors.append([pos_error, rot_error])
    
    errors = np.array(errors)
    
    return {
        'position_error_mean': np.mean(errors[:, 0]),
        'position_error_std': np.std(errors[:, 0]),
        'rotation_error_mean': np.mean(errors[:, 1]),
        'rotation_error_std': np.std(errors[:, 1]),
        'total_error_mean': np.mean(np.sum(errors, axis=1)),
        'total_error_std': np.std(np.sum(errors, axis=1))
    }


def main():
    """
    主函数 - 演示高级旋转滤波器的使用
    """
    print("=" * 60)
    print("高级旋转滤波器演示")
    print("=" * 60)
    
    # 生成测试数据
    print("生成测试轨迹...")
    true_trajectory, noisy_trajectory = generate_test_trajectory(
        num_points=200, 
        noise_level=0.15,
        rotation_amplitude=1.0
    )
    
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
        filter_obj = AdvancedRotationFilter(**config['config'])
        
        # 滤波轨迹
        filtered_trajectory = filter_obj.filter_trajectory(noisy_trajectory)
        
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
            save_path=f'filter_comparison_{name.lower().replace(" ", "_")}.png'
        )
    
    # 打印最佳滤波器
    best_filter = min(results.keys(), 
                     key=lambda x: results[x]['error_stats']['total_error_mean'])
    print(f"\n最佳滤波器: {best_filter}")
    print(f"总误差: {results[best_filter]['error_stats']['total_error_mean']:.4f}")
    
    # 实时滤波演示
    print("\n" + "=" * 60)
    print("实时滤波演示")
    print("=" * 60)
    
    # 使用最佳滤波器进行实时演示
    best_filter_obj = results[best_filter]['filter_obj']
    best_filter_obj.reset()  # 重置滤波器
    
    print("实时滤波示例 (按 Ctrl+C 退出):")
    print("格式: [x, y, z, roll, pitch, yaw]")
    print("-" * 60)
    
    try:
        import time
        for i in range(20):  # 演示20个样本
            # 使用噪声轨迹中的样本
            sample_pose = noisy_trajectory[i % len(noisy_trajectory)]
            
            # 滤波
            filtered_pose = best_filter_obj.filter_pose(sample_pose)
            
            print(f"样本 {i+1:2d}:")
            print(f"  输入: {[f'{x:.3f}' for x in sample_pose]}")
            print(f"  输出: {[f'{x:.3f}' for x in filtered_pose]}")
            print()
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n实时演示结束")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    # 打印使用说明
    print("\n使用说明:")
    print("1. 创建滤波器: filter_obj = AdvancedRotationFilter(filter_type='savgol')")
    print("2. 滤波单个位姿: filtered_pose = filter_obj.filter_pose(pose_xyzrpy)")
    print("3. 滤波整个轨迹: filtered_traj = filter_obj.filter_trajectory(trajectory)")
    print("4. 重置滤波器: filter_obj.reset()")
    print("\n支持的滤波类型:")
    print("- 'savgol': Savitzky-Golay 滤波 (推荐)")
    print("- 'butterworth': Butterworth 低通滤波")
    print("- 'moving_average': 移动平均滤波")


if __name__ == "__main__":
    main()
