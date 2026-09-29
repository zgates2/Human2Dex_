#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from typing import List, Union
from scipy.spatial.transform import Rotation as R

class SimpleTrajectoryFilter:
    """
    简化的轨迹滤波器，专门用于实时 xyzrpy 轨迹滤波
    支持移动平均、指数平滑、低通滤波等方法
    """
    
    def __init__(self, 
                 filter_type: str = 'moving_average',
                 window_size: int = 5,
                 alpha: float = 0.3,
                 cutoff_freq: float = 1.0,
                 sampling_rate: float = 50.0):
        """
        初始化简化轨迹滤波器
        
        Args:
            filter_type: 滤波类型 ('moving_average', 'exponential', 'lowpass')
            window_size: 移动平均窗口大小
            alpha: 指数平滑系数 (0-1)
            cutoff_freq: 截止频率 (Hz)
            sampling_rate: 采样率 (Hz)
        """
        self.filter_type = filter_type
        self.window_size = window_size
        self.alpha = alpha
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        
        # 初始化滤波器状态
        self.is_initialized = False
        self.last_pose = None
        self.pose_buffer = deque(maxlen=window_size)
        
        # 低通滤波器参数
        if filter_type == 'lowpass':
            self._init_lowpass_filter()
    
    def _init_lowpass_filter(self):
        """初始化低通滤波器"""
        from scipy import signal
        nyquist = self.sampling_rate / 2
        normal_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = signal.butter(2, normal_cutoff, btype='low', analog=False)
        self.filter_state = None
    
    def _normalize_angle(self, angle):
        """将角度归一化到 [-π, π]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def _rpy_to_quaternion(self, rpy):
        """RPY转四元数"""
        r = R.from_euler('xyz', rpy, degrees=False)
        return r.as_quat()  # 返回 [x, y, z, w]
    
    def _quaternion_to_rpy(self, quat_xyzw):
        """四元数转RPY"""
        r = R.from_quat(quat_xyzw)  # 输入 [x, y, z, w]
        return r.as_euler('xyz', degrees=False)
    
    def _quaternion_slerp(self, q1, q2, t):
        """四元数球面线性插值"""
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        r_slerp = R.from_quat(r1.as_quat()).inv() * R.from_quat(r2.as_quat())
        r_result = R.from_quat(r1.as_quat()) * R.from_quat(r_slerp.as_quat() * t)
        return r_result.as_quat()
    
    def _quaternion_average(self, quaternions):
        """计算四元数的平均值（使用球面平均）"""
        if len(quaternions) == 1:
            return quaternions[0]
        
        # 使用迭代方法计算球面平均
        q_avg = quaternions[0].copy()
        
        for _ in range(10):  # 迭代10次
            q_sum = np.zeros(4)
            for q in quaternions:
                # 确保四元数在同一半球
                if np.dot(q_avg, q) < 0:
                    q = -q
                q_sum += q
            
            q_avg = q_sum / np.linalg.norm(q_sum)
        
        return q_avg
    
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
        quat_xyzw = self._rpy_to_quaternion(rpy)
        
        if not self.is_initialized:
            self.last_pose = pose.copy()
            self.pose_buffer.append({
                'pos': pos.copy(),
                'quat': quat_xyzw.copy()
            })
            self.is_initialized = True
            return pose
        
        if self.filter_type == 'moving_average':
            # 移动平均滤波
            self.pose_buffer.append({
                'pos': pos.copy(),
                'quat': quat_xyzw.copy()
            })
            
            # 对位置进行线性平均
            pos_list = [p['pos'] for p in self.pose_buffer]
            filtered_pos = np.mean(pos_list, axis=0)
            
            # 对四元数进行球面平均
            quat_list = [p['quat'] for p in self.pose_buffer]
            filtered_quat = self._quaternion_average(quat_list)
            
        elif self.filter_type == 'exponential':
            # 指数平滑滤波
            last_pos = self.last_pose[:3]
            last_rpy = self.last_pose[3:]
            last_quat = self._rpy_to_quaternion(last_rpy)
            
            # 位置指数平滑
            filtered_pos = self.alpha * pos + (1 - self.alpha) * last_pos
            
            # 四元数指数平滑（使用球面插值）
            filtered_quat = self._quaternion_slerp(last_quat, quat_xyzw, self.alpha)
            
        elif self.filter_type == 'lowpass':
            # 低通滤波
            self.pose_buffer.append({
                'pos': pos.copy(),
                'quat': quat_xyzw.copy()
            })
            
            if len(self.pose_buffer) >= 3:  # 需要足够的样本
                from scipy import signal
                
                # 对位置进行低通滤波
                pos_array = np.array([p['pos'] for p in self.pose_buffer])
                filtered_pos = signal.filtfilt(self.b, self.a, pos_array, axis=0)[-1]
                
                # 对四元数进行低通滤波（在四元数空间）
                quat_array = np.array([p['quat'] for p in self.pose_buffer])
                filtered_quat = signal.filtfilt(self.b, self.a, quat_array, axis=0)[-1]
                # 归一化四元数
                filtered_quat = filtered_quat / np.linalg.norm(filtered_quat)
            else:
                filtered_pos = pos
                filtered_quat = quat_xyzw
        else:
            raise ValueError(f"不支持的滤波类型: {self.filter_type}")
        
        # 将四元数转换回RPY
        filtered_rpy = self._quaternion_to_rpy(filtered_quat)
        
        # 组合最终位姿
        filtered_pose = np.concatenate([filtered_pos, filtered_rpy])
        
        self.last_pose = filtered_pose.copy()
        return filtered_pose
    
    def reset(self):
        """重置滤波器状态"""
        self.is_initialized = False
        self.last_pose = None
        self.pose_buffer.clear()
        if self.filter_type == 'lowpass':
            self.filter_state = None


# 使用示例
if __name__ == "__main__":
    # 创建滤波器
    filter_obj = SimpleTrajectoryFilter(
        filter_type='moving_average',
        window_size=5
    )
    
    # 模拟轨迹数据
    import time
    import random
    
    print("轨迹滤波演示:")
    print("输入格式: [x, y, z, roll, pitch, yaw]")
    print("按 Ctrl+C 退出\n")
    
    try:
        while True:
            # 模拟带噪声的位姿数据
            true_pose = [
                0.1 * np.sin(time.time()),
                0.1 * np.cos(time.time()),
                0.5 + 0.05 * np.sin(0.5 * time.time()),
                0.1 * np.sin(0.3 * time.time()),
                0.1 * np.cos(0.3 * time.time()),
                0.2 * time.time()
            ]
            
            # 添加噪声
            noisy_pose = [p + 0.02 * random.gauss(0, 1) for p in true_pose]
            
            # 滤波
            filtered_pose = filter_obj.filter_pose(noisy_pose)
            
            print(f"原始: {[f'{x:.3f}' for x in noisy_pose]}")
            print(f"滤波: {[f'{x:.3f}' for x in filtered_pose]}")
            print("-" * 50)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n程序结束")
