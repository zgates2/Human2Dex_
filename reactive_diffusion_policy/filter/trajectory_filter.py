#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union

class TrajectoryFilter:
    """
    轨迹滤波器，用于对 xyzrpy 格式的轨迹进行滤波处理
    支持多种滤波方法：低通滤波、卡尔曼滤波、移动平均等
    """
    
    def __init__(self, 
                 filter_type: str = 'lowpass',
                 cutoff_freq: float = 1.0,
                 sampling_rate: float = 50.0,
                 order: int = 4,
                 window_size: int = 5):
        """
        初始化轨迹滤波器
        
        Args:
            filter_type: 滤波类型 ('lowpass', 'kalman', 'moving_average', 'butterworth')
            cutoff_freq: 截止频率 (Hz)
            sampling_rate: 采样率 (Hz)
            order: 滤波器阶数
            window_size: 移动平均窗口大小
        """
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        self.window_size = window_size
        
        # 初始化滤波器
        self._init_filter()
        
        # 状态变量
        self.is_initialized = False
        self.last_pose = None
        self.pose_history = []
        
    def _init_filter(self):
        """初始化滤波器"""
        if self.filter_type == 'lowpass':
            # 低通滤波器
            nyquist = self.sampling_rate / 2
            normal_cutoff = self.cutoff_freq / nyquist
            self.b, self.a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
            
        elif self.filter_type == 'butterworth':
            # 巴特沃斯滤波器
            nyquist = self.sampling_rate / 2
            normal_cutoff = self.cutoff_freq / nyquist
            self.b, self.a = signal.butter(self.order, normal_cutoff, btype='low', analog=False)
            
        elif self.filter_type == 'kalman':
            # 卡尔曼滤波器参数
            self.dt = 1.0 / self.sampling_rate
            self.kf_pos = self._init_kalman_filter()
            self.kf_rot = self._init_kalman_filter()
            
        elif self.filter_type == 'moving_average':
            # 移动平均滤波器
            self.filter_buffer = []
            
    def _init_kalman_filter(self):
        """初始化卡尔曼滤波器"""
        # 状态向量: [x, vx, y, vy, z, vz] 或 [rx, vrx, ry, vry, rz, vrz]
        # 观测向量: [x, y, z] 或 [rx, ry, rz]
        
        # 状态转移矩阵
        F = np.array([
            [1, self.dt, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, self.dt, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, self.dt],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        
        # 过程噪声协方差
        Q = np.eye(6) * 0.01
        
        # 观测噪声协方差
        R = np.eye(3) * 0.1
        
        # 初始状态协方差
        P = np.eye(6) * 1.0
        
        return {
            'F': F, 'H': H, 'Q': Q, 'R': R, 'P': P,
            'x': np.zeros(6),  # 初始状态
            'initialized': False
        }
    
    def _kalman_predict(self, kf):
        """卡尔曼滤波预测步骤"""
        # 状态预测
        kf['x'] = kf['F'] @ kf['x']
        # 协方差预测
        kf['P'] = kf['F'] @ kf['P'] @ kf['F'].T + kf['Q']
        return kf
    
    def _kalman_update(self, kf, measurement):
        """卡尔曼滤波更新步骤"""
        # 卡尔曼增益
        S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']
        K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)
        
        # 状态更新
        y = measurement - kf['H'] @ kf['x']
        kf['x'] = kf['x'] + K @ y
        
        # 协方差更新
        kf['P'] = (np.eye(6) - K @ kf['H']) @ kf['P']
        
        return kf
    
    def _quaternion_to_rpy(self, quat_wxyz):
        """四元数转RPY"""
        r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # w,x,y,z -> x,y,z,w
        return r.as_euler('xyz', degrees=False)
    
    def _rpy_to_quaternion(self, rpy):
        """RPY转四元数"""
        r = R.from_euler('xyz', rpy, degrees=False)
        quat_xyzw = r.as_quat()  # x,y,z,w
        return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # w,x,y,z
    
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
    
    def _normalize_angle(self, angle):
        """将角度归一化到 [-π, π]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def filter_pose(self, pose_xyzrpy: Union[List, np.ndarray]) -> np.ndarray:
        """
        对单个位姿进行滤波
        
        Args:
            pose_xyzrpy: 输入位姿 [x, y, z, roll, pitch, yaw]
            
        Returns:
            滤波后的位姿 [x, y, z, roll, pitch, yaw]
        """
        pose = np.array(pose_xyzrpy)
        if len(pose) != 6:
            raise ValueError("位姿必须是6维向量 [x, y, z, roll, pitch, yaw]")
        
        if not self.is_initialized:
            self.last_pose = pose.copy()
            self.pose_history = [pose.copy()]
            self.is_initialized = True
            return pose
        
        # 分离位置和姿态
        pos = pose[:3]
        rpy = pose[3:]
        
        if self.filter_type == 'lowpass' or self.filter_type == 'butterworth':
            # 低通滤波
            self.pose_history.append(pose)
            if len(self.pose_history) > self.window_size:
                self.pose_history.pop(0)
            
            # 对位置和姿态分别滤波
            pos_history = np.array([p[:3] for p in self.pose_history])
            rpy_history = np.array([p[3:] for p in self.pose_history])
            
            # 应用滤波器
            if len(self.pose_history) >= self.order + 1:
                filtered_pos = signal.filtfilt(self.b, self.a, pos_history, axis=0)
                filtered_rpy = signal.filtfilt(self.b, self.a, rpy_history, axis=0)
                
                # 归一化角度
                for i in range(3):
                    filtered_rpy[:, i] = self._normalize_angle(filtered_rpy[:, i])
                
                filtered_pose = np.concatenate([filtered_pos[-1], filtered_rpy[-1]])
            else:
                filtered_pose = pose
                
        elif self.filter_type == 'kalman':
            # 卡尔曼滤波
            # 位置滤波
            if not self.kf_pos['initialized']:
                self.kf_pos['x'][::2] = pos
                self.kf_pos['initialized'] = True
            else:
                self.kf_pos = self._kalman_predict(self.kf_pos)
                self.kf_pos = self._kalman_update(self.kf_pos, pos)
            
            # 姿态滤波
            if not self.kf_rot['initialized']:
                self.kf_rot['x'][::2] = rpy
                self.kf_rot['initialized'] = True
            else:
                self.kf_rot = self._kalman_predict(self.kf_rot)
                self.kf_rot = self._kalman_update(self.kf_rot, rpy)
            
            # 归一化角度
            filtered_rpy = self.kf_rot['x'][::2].copy()
            for i in range(3):
                filtered_rpy[i] = self._normalize_angle(filtered_rpy[i])
            
            filtered_pose = np.concatenate([self.kf_pos['x'][::2], filtered_rpy])
            
        elif self.filter_type == 'moving_average':
            # 移动平均滤波
            self.pose_history.append(pose)
            if len(self.pose_history) > self.window_size:
                self.pose_history.pop(0)
            
            # 计算移动平均
            filtered_pose = np.mean(self.pose_history, axis=0)
            
            # 归一化角度
            for i in range(3, 6):
                filtered_pose[i] = self._normalize_angle(filtered_pose[i])
        
        else:
            raise ValueError(f"不支持的滤波类型: {self.filter_type}")
        
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
        self.pose_history = []
        if self.filter_type == 'kalman':
            self.kf_pos = self._init_kalman_filter()
            self.kf_rot = self._init_kalman_filter()
        elif self.filter_type == 'moving_average':
            self.filter_buffer = []


def demo():
    """演示轨迹滤波器的使用"""
    # 生成测试轨迹（带噪声的正弦波）
    t = np.linspace(0, 10, 500)
    dt = t[1] - t[0]
    sampling_rate = 1.0 / dt
    
    # 生成带噪声的轨迹
    noise_level = 0.1
    true_trajectory = []
    noisy_trajectory = []
    
    for i, time in enumerate(t):
        # 真实轨迹：正弦波运动
        x = 0.1 * np.sin(0.5 * time)
        y = 0.1 * np.cos(0.5 * time)
        z = 0.05 * time
        roll = 0.1 * np.sin(0.3 * time)
        pitch = 0.1 * np.cos(0.3 * time)
        yaw = 0.2 * time
        
        true_pose = [x, y, z, roll, pitch, yaw]
        noisy_pose = [x + noise_level * np.random.randn(),
                     y + noise_level * np.random.randn(),
                     z + noise_level * np.random.randn(),
                     roll + noise_level * np.random.randn(),
                     pitch + noise_level * np.random.randn(),
                     yaw + noise_level * np.random.randn()]
        
        true_trajectory.append(true_pose)
        noisy_trajectory.append(noisy_pose)
    
    # 测试不同的滤波器
    filters = {
        'lowpass': TrajectoryFilter('lowpass', cutoff_freq=2.0, sampling_rate=sampling_rate),
        'kalman': TrajectoryFilter('kalman', sampling_rate=sampling_rate),
        'moving_average': TrajectoryFilter('moving_average', window_size=10)
    }
    
    filtered_trajectories = {}
    for name, filter_obj in filters.items():
        filtered_trajectories[name] = filter_obj.filter_trajectory(noisy_trajectory)
    
    # 绘制结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    
    for i in range(6):
        ax = axes[i]
        
        # 绘制真实轨迹
        true_data = [pose[i] for pose in true_trajectory]
        ax.plot(t, true_data, 'g-', label='True', linewidth=2)
        
        # 绘制噪声轨迹
        noisy_data = [pose[i] for pose in noisy_trajectory]
        ax.plot(t, noisy_data, 'r--', label='Noisy', alpha=0.7)
        
        # 绘制滤波后的轨迹
        colors = ['b-', 'm-', 'c-']
        for j, (name, filtered_data) in enumerate(filtered_trajectories.items()):
            filtered_values = [pose[i] for pose in filtered_data]
            ax.plot(t, filtered_values, colors[j], label=f'Filtered ({name})', linewidth=1.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(labels[i])
        ax.set_title(f'{labels[i]} Position')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算滤波效果
    print("滤波效果评估:")
    for name, filtered_trajectory in filtered_trajectories.items():
        mse = np.mean([np.mean((np.array(true) - np.array(filtered))**2) 
                      for true, filtered in zip(true_trajectory, filtered_trajectory)])
        print(f"{name}: MSE = {mse:.6f}")


if __name__ == "__main__":
    demo()
