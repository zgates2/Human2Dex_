#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
旋转滤波器
使用旋转向量的对数映射进行旋转滤波

流程：
RPY -> 四元数 -> 旋转向量(log map) -> 滤波 -> exp map -> 四元数 -> RPY

作者: Reactive Diffusion Policy Team
版本: 1.0.0
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal
from collections import deque
from typing import List, Union, Optional

class RotationFilter:
    """
    旋转滤波器类
    
    使用旋转向量的对数映射进行旋转滤波，这是数学上更严格的方法。
    避免了四元数平均可能遇到的问题。
    
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
        初始化旋转滤波器
        
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
        """RPY 转四元数"""
        r = R.from_euler('xyz', rpy, degrees=False)
        return r.as_quat()  # [x, y, z, w]
    
    def _quaternion_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """四元数转 RPY"""
        r = R.from_quat(quat)
        return r.as_euler('xyz', degrees=False)
    
    def _quaternion_to_rotvec(self, quat: np.ndarray) -> np.ndarray:
        """四元数转旋转向量 (log map)"""
        r = R.from_quat(quat)
        return r.as_rotvec()
    
    def _rotvec_to_quaternion(self, rotvec: np.ndarray) -> np.ndarray:
        """旋转向量转四元数 (exp map)"""
        r = R.from_rotvec(rotvec)
        return r.as_quat()
    
    def _filter_rotation_vector(self, rotvec_history: List[np.ndarray]) -> np.ndarray:
        """对旋转向量历史进行滤波"""
        if len(rotvec_history) == 1:
            return rotvec_history[0]
        
        rotvec_array = np.array(rotvec_history)
        
        if self.filter_type == 'moving_average':
            return np.mean(rotvec_array, axis=0)
        elif self.filter_type == 'butterworth':
            # Butterworth 滤波器需要足够的数据点
            min_length = max(3 * self.order, 15)  # 至少需要 3*order 或 15 个点
            if len(rotvec_array) >= min_length:
                return signal.filtfilt(self.b, self.a, rotvec_array, axis=0)[-1]
            else:
                # 数据点不够，使用移动平均作为替代
                return np.mean(rotvec_array, axis=0)
        elif self.filter_type == 'savgol':
            if len(rotvec_array) >= self.window_size:
                # 确保 polyorder < window_size
                polyorder = min(self.polyorder, self.window_size - 1)
                if polyorder < 1:
                    polyorder = 1
                    
                return signal.savgol_filter(
                    rotvec_array, 
                    self.window_size, 
                    polyorder, 
                    axis=0
                )[-1]
            else:
                window = min(len(rotvec_array), self.window_size)
                if window % 2 == 0:
                    window -= 1
                if window < 3:  # 至少需要3个点
                    return rotvec_array[-1]
                    
                polyorder = min(self.polyorder, window - 1)
                if polyorder < 1:
                    polyorder = 1
                    
                return signal.savgol_filter(
                    rotvec_array, 
                    window, 
                    polyorder, 
                    axis=0
                )[-1]
        else:
            return rotvec_array[-1]
    
    def _filter_position(self, pos_history: List[np.ndarray]) -> np.ndarray:
        """对位置历史进行滤波"""
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
        """获取滤波器信息"""
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


def main():
    """
    主函数 - 演示旋转滤波器的使用
    """
    print("=" * 60)
    print("旋转滤波器演示")
    print("=" * 60)
    
    # 创建滤波器实例
    print("创建 Savitzky-Golay 滤波器...")
    filter_obj = RotationFilter(
        filter_type='savgol',
        window_size=7,
        polyorder=2
    )
    
    # 生成测试数据
    print("生成测试轨迹...")
    t = np.linspace(0, 4*np.pi, 100)
    test_trajectory = []
    
    for i, time in enumerate(t):
        # 生成带噪声的轨迹
        x = 0.1 * np.sin(0.5 * time) + 0.02 * np.random.randn()
        y = 0.1 * np.cos(0.5 * time) + 0.02 * np.random.randn()
        z = 0.05 * time + 0.01 * np.random.randn()
        
        roll = 0.1 * np.sin(0.3 * time) + 0.05 * np.random.randn()
        pitch = 0.1 * np.cos(0.3 * time) + 0.05 * np.random.randn()
        yaw = 0.2 * time + 0.1 * np.random.randn()
        
        pose = [x, y, z, roll, pitch, yaw]
        test_trajectory.append(pose)
    
    print(f"生成了 {len(test_trajectory)} 个位姿点")
    
    # 滤波轨迹
    print("开始滤波...")
    filtered_trajectory = filter_obj.filter_trajectory(test_trajectory)
    
    # 显示结果
    print("\n滤波结果对比 (前10个点):")
    print("索引 | 原始位姿                    | 滤波位姿")
    print("-" * 70)
    
    for i in range(min(10, len(test_trajectory))):
        original = test_trajectory[i]
        filtered = filtered_trajectory[i]
        
        print(f"{i:3d}  | "
              f"[{original[0]:6.3f}, {original[1]:6.3f}, {original[2]:6.3f}, "
              f"{original[3]:6.3f}, {original[4]:6.3f}, {original[5]:6.3f}] | "
              f"[{filtered[0]:6.3f}, {filtered[1]:6.3f}, {filtered[2]:6.3f}, "
              f"{filtered[3]:6.3f}, {filtered[4]:6.3f}, {filtered[5]:6.3f}]")
    
    # 实时滤波演示
    print("\n" + "=" * 60)
    print("实时滤波演示")
    print("=" * 60)
    
    # 重置滤波器
    filter_obj.reset()
    
    print("实时滤波示例 (按 Ctrl+C 退出):")
    print("格式: [x, y, z, roll, pitch, yaw]")
    print("-" * 60)
    
    try:
        import time
        for i in range(20):  # 演示20个样本
            # 使用测试轨迹中的样本
            sample_pose = test_trajectory[i % len(test_trajectory)]
            
            # 滤波
            filtered_pose = filter_obj.filter_pose(sample_pose)
            
            print(f"样本 {i+1:2d}:")
            print(f"  输入: {[f'{x:.3f}' for x in sample_pose]}")
            print(f"  输出: {[f'{x:.3f}' for x in filtered_pose]}")
            print()
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n实时演示结束")
    
    # 显示滤波器信息
    print("\n" + "=" * 60)
    print("滤波器信息")
    print("=" * 60)
    
    filter_info = filter_obj.get_filter_info()
    for key, value in filter_info.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    
    # 打印使用说明
    print("\n使用说明:")
    print("1. 创建滤波器: filter_obj = RotationFilter(filter_type='savgol')")
    print("2. 滤波单个位姿: filtered_pose = filter_obj.filter_pose(pose_xyzrpy)")
    print("3. 滤波整个轨迹: filtered_traj = filter_obj.filter_trajectory(trajectory)")
    print("4. 重置滤波器: filter_obj.reset()")
    print("\n支持的滤波类型:")
    print("- 'savgol': Savitzky-Golay 滤波 (推荐)")
    print("- 'butterworth': Butterworth 低通滤波")
    print("- 'moving_average': 移动平均滤波")


if __name__ == "__main__":
    main()
