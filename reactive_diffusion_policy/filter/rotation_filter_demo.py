#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
旋转滤波演示
展示为什么不能直接对 RPY 进行滤波，以及如何正确地对旋转进行滤波
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from simple_trajectory_filter import SimpleTrajectoryFilter

def generate_rotation_trajectory(num_points=100, noise_level=0.1):
    """生成带噪声的旋转轨迹"""
    t = np.linspace(0, 4*np.pi, num_points)
    
    # 生成真实旋转轨迹（绕Z轴连续旋转）
    true_rotations = []
    true_rpy = []
    
    for i, time in enumerate(t):
        # 真实旋转：绕Z轴连续旋转
        r = R.from_euler('xyz', [0, 0, time], degrees=False)
        true_rotations.append(r)
        
        # 转换为RPY
        rpy = r.as_euler('xyz', degrees=False)
        true_rpy.append(rpy)
    
    # 添加噪声
    noisy_rpy = []
    for rpy in true_rpy:
        noisy_rpy.append(rpy + noise_level * np.random.randn(3))
    
    return true_rpy, noisy_rpy, true_rotations

def wrong_rpy_filtering(noisy_rpy, window_size=5):
    """错误的RPY滤波方法（直接对RPY进行移动平均）"""
    filtered_rpy = []
    
    for i in range(len(noisy_rpy)):
        start_idx = max(0, i - window_size + 1)
        window_rpy = noisy_rpy[start_idx:i+1]
        
        # 直接对RPY进行平均（错误方法）
        avg_rpy = np.mean(window_rpy, axis=0)
        filtered_rpy.append(avg_rpy)
    
    return filtered_rpy

def correct_quaternion_filtering(noisy_rpy, window_size=5):
    """正确的旋转滤波方法（在四元数空间进行滤波）"""
    from simple_trajectory_filter import SimpleTrajectoryFilter
    
    # 创建滤波器
    filter_obj = SimpleTrajectoryFilter(
        filter_type='moving_average',
        window_size=window_size
    )
    
    # 滤波
    filtered_poses = []
    for rpy in noisy_rpy:
        # 构造完整的位姿 [x, y, z, roll, pitch, yaw]
        pose = [0, 0, 0] + list(rpy)
        filtered_pose = filter_obj.filter_pose(pose)
        filtered_poses.append(filtered_pose[3:])  # 只取RPY部分
    
    return filtered_poses

def quaternion_average(quaternions):
    """计算四元数的球面平均"""
    if len(quaternions) == 1:
        return quaternions[0]
    
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

def manual_quaternion_filtering(noisy_rpy, window_size=5):
    """手动实现四元数滤波"""
    filtered_rpy = []
    
    for i in range(len(noisy_rpy)):
        start_idx = max(0, i - window_size + 1)
        window_rpy = noisy_rpy[start_idx:i+1]
        
        # 转换为四元数
        quaternions = []
        for rpy in window_rpy:
            r = R.from_euler('xyz', rpy, degrees=False)
            quaternions.append(r.as_quat())
        
        # 计算四元数平均
        avg_quat = quaternion_average(quaternions)
        
        # 转换回RPY
        avg_r = R.from_quat(avg_quat)
        avg_rpy = avg_r.as_euler('xyz', degrees=False)
        filtered_rpy.append(avg_rpy)
    
    return filtered_rpy

def plot_rotation_comparison(true_rpy, noisy_rpy, wrong_filtered, correct_filtered):
    """绘制旋转滤波对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['red', 'blue', 'green']
    
    for i in range(3):
        ax = axes[i]
        
        t = np.linspace(0, 4*np.pi, len(true_rpy))
        
        # 真实轨迹
        true_data = [rpy[i] for rpy in true_rpy]
        ax.plot(t, true_data, 'k-', label='True', linewidth=2)
        
        # 噪声轨迹
        noisy_data = [rpy[i] for rpy in noisy_rpy]
        ax.plot(t, noisy_data, 'gray', label='Noisy', alpha=0.7)
        
        # 错误滤波
        wrong_data = [rpy[i] for rpy in wrong_filtered]
        ax.plot(t, wrong_data, 'r--', label='Wrong RPY Filter', linewidth=1.5)
        
        # 正确滤波
        correct_data = [rpy[i] for rpy in correct_filtered]
        ax.plot(t, correct_data, 'g-', label='Correct Quat Filter', linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{labels[i]} (rad)')
        ax.set_title(f'{labels[i]} Angle Comparison')
        ax.legend()
        ax.grid(True)
    
    # 绘制角度差异
    for i in range(3):
        ax = axes[i + 3]
        
        t = np.linspace(0, 4*np.pi, len(true_rpy))
        
        # 计算角度差异
        true_data = np.array([rpy[i] for rpy in true_rpy])
        wrong_data = np.array([rpy[i] for rpy in wrong_filtered])
        correct_data = np.array([rpy[i] for rpy in correct_filtered])
        
        # 处理角度跳跃
        wrong_diff = true_data - wrong_data
        correct_diff = true_data - correct_data
        
        # 归一化角度差异到 [-π, π]
        wrong_diff = np.arctan2(np.sin(wrong_diff), np.cos(wrong_diff))
        correct_diff = np.arctan2(np.sin(correct_diff), np.cos(correct_diff))
        
        ax.plot(t, wrong_diff, 'r--', label='Wrong RPY Filter Error', linewidth=1.5)
        ax.plot(t, correct_diff, 'g-', label='Correct Quat Filter Error', linewidth=1.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{labels[i]} Error (rad)')
        ax.set_title(f'{labels[i]} Filtering Error')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def calculate_rotation_error(true_rpy, filtered_rpy):
    """计算旋转滤波误差"""
    errors = []
    
    for true, filtered in zip(true_rpy, filtered_rpy):
        # 转换为旋转矩阵
        r_true = R.from_euler('xyz', true, degrees=False)
        r_filtered = R.from_euler('xyz', filtered, degrees=False)
        
        # 计算旋转差异
        r_diff = r_true.inv() * r_filtered
        
        # 计算角度误差
        angle_error = np.linalg.norm(r_diff.as_rotvec())
        errors.append(angle_error)
    
    return np.mean(errors), np.std(errors)

def main():
    """主函数"""
    print("旋转滤波演示")
    print("=" * 50)
    
    # 生成测试数据
    print("生成测试数据...")
    true_rpy, noisy_rpy, true_rotations = generate_rotation_trajectory(200, 0.2)
    
    # 应用不同的滤波方法
    print("应用滤波方法...")
    wrong_filtered = wrong_rpy_filtering(noisy_rpy, window_size=5)
    correct_filtered = correct_quaternion_filtering(noisy_rpy, window_size=5)
    manual_filtered = manual_quaternion_filtering(noisy_rpy, window_size=5)
    
    # 计算误差
    print("\n滤波误差分析:")
    wrong_mean, wrong_std = calculate_rotation_error(true_rpy, wrong_filtered)
    correct_mean, correct_std = calculate_rotation_error(true_rpy, correct_filtered)
    manual_mean, manual_std = calculate_rotation_error(true_rpy, manual_filtered)
    
    print(f"错误RPY滤波:    平均误差 = {wrong_mean:.4f} ± {wrong_std:.4f} rad")
    print(f"正确四元数滤波:  平均误差 = {correct_mean:.4f} ± {correct_std:.4f} rad")
    print(f"手动四元数滤波:  平均误差 = {manual_mean:.4f} ± {manual_std:.4f} rad")
    
    # 绘制对比图
    print("\n绘制对比图...")
    plot_rotation_comparison(true_rpy, noisy_rpy, wrong_filtered, correct_filtered)
    
    print("\n演示完成！")
    print("关键点:")
    print("1. 直接对RPY进行滤波会导致角度跳跃问题")
    print("2. 在四元数空间进行滤波可以避免角度不连续问题")
    print("3. 四元数滤波使用球面平均，保持旋转的几何意义")

if __name__ == "__main__":
    main()
