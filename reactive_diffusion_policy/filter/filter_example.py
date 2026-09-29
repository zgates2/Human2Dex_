#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轨迹滤波使用示例
演示如何使用不同的滤波器对 xyzrpy 轨迹进行滤波
"""

import numpy as np
import matplotlib.pyplot as plt
from simple_trajectory_filter import SimpleTrajectoryFilter

def generate_test_trajectory(num_points=100, noise_level=0.1):
    """生成测试轨迹"""
    t = np.linspace(0, 10, num_points)
    
    # 生成真实轨迹
    true_trajectory = []
    for i, time in enumerate(t):
        x = 0.1 * np.sin(0.5 * time)
        y = 0.1 * np.cos(0.5 * time)
        z = 0.05 * time
        roll = 0.1 * np.sin(0.3 * time)
        pitch = 0.1 * np.cos(0.3 * time)
        yaw = 0.2 * time
        
        true_trajectory.append([x, y, z, roll, pitch, yaw])
    
    # 添加噪声
    noisy_trajectory = []
    for pose in true_trajectory:
        noisy_pose = [p + noise_level * np.random.randn() for p in pose]
        noisy_trajectory.append(noisy_pose)
    
    return true_trajectory, noisy_trajectory

def test_filters():
    """测试不同滤波器的效果"""
    # 生成测试数据
    true_traj, noisy_traj = generate_test_trajectory(200, 0.05)
    
    # 创建不同的滤波器
    filters = {
        'moving_average_3': SimpleTrajectoryFilter('moving_average', window_size=3),
        'moving_average_7': SimpleTrajectoryFilter('moving_average', window_size=7),
        'exponential_0.3': SimpleTrajectoryFilter('exponential', alpha=0.3),
        'exponential_0.7': SimpleTrajectoryFilter('exponential', alpha=0.7),
    }
    
    # 应用滤波器
    filtered_trajectories = {}
    for name, filter_obj in filters.items():
        filtered_traj = []
        for pose in noisy_traj:
            filtered_pose = filter_obj.filter_pose(pose)
            filtered_traj.append(filtered_pose)
        filtered_trajectories[name] = filtered_traj
    
    # 绘制结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    colors = ['b-', 'r-', 'g-', 'm-', 'c-', 'y-']
    
    for i in range(6):
        ax = axes[i]
        
        # 真实轨迹
        true_data = [pose[i] for pose in true_traj]
        ax.plot(true_data, 'k-', label='True', linewidth=2)
        
        # 噪声轨迹
        noisy_data = [pose[i] for pose in noisy_traj]
        ax.plot(noisy_data, 'gray', label='Noisy', alpha=0.5)
        
        # 滤波轨迹
        for j, (name, filtered_traj) in enumerate(filtered_trajectories.items()):
            filtered_data = [pose[i] for pose in filtered_traj]
            ax.plot(filtered_data, colors[j % len(colors)], label=name, linewidth=1.5)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(labels[i])
        ax.set_title(f'{labels[i]} Position')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算滤波效果
    print("滤波效果评估 (MSE):")
    for name, filtered_traj in filtered_trajectories.items():
        mse = np.mean([np.mean((np.array(true) - np.array(filtered))**2) 
                      for true, filtered in zip(true_traj, filtered_traj)])
        print(f"{name}: {mse:.6f}")

def real_time_example():
    """实时滤波示例"""
    print("实时轨迹滤波示例")
    print("=" * 50)
    
    # 创建滤波器
    filter_obj = SimpleTrajectoryFilter(
        filter_type='moving_average',
        window_size=5
    )
    
    # 模拟实时数据
    import time
    import random
    
    print("开始实时滤波... 按 Ctrl+C 退出")
    print("格式: [x, y, z, roll, pitch, yaw]")
    print("-" * 50)
    
    try:
        step = 0
        while True:
            # 生成模拟位姿数据
            t = step * 0.1
            true_pose = [
                0.1 * np.sin(0.5 * t),
                0.1 * np.cos(0.5 * t),
                0.5 + 0.05 * np.sin(0.3 * t),
                0.1 * np.sin(0.2 * t),
                0.1 * np.cos(0.2 * t),
                0.1 * t
            ]
            
            # 添加噪声
            noisy_pose = [p + 0.02 * random.gauss(0, 1) for p in true_pose]
            
            # 滤波
            filtered_pose = filter_obj.filter_pose(noisy_pose)
            
            if step % 10 == 0:  # 每10步打印一次
                print(f"Step {step:3d}:")
                print(f"  真实: {[f'{x:.3f}' for x in true_pose]}")
                print(f"  噪声: {[f'{x:.3f}' for x in noisy_pose]}")
                print(f"  滤波: {[f'{x:.3f}' for x in filtered_pose]}")
                print()
            
            step += 1
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n程序结束")

if __name__ == "__main__":
    print("轨迹滤波器演示")
    print("1. 测试不同滤波器效果")
    print("2. 实时滤波示例")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == '1':
        test_filters()
    elif choice == '2':
        real_time_example()
    else:
        print("无效选择，运行测试示例...")
        test_filters()
