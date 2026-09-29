#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试滤波模块是否正常工作
"""

import sys
import os
import numpy as np

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simple_filter():
    """测试简化滤波器"""
    print("测试 SimpleTrajectoryFilter...")
    
    try:
        from filter import SimpleTrajectoryFilter
        
        # 创建滤波器
        filter_obj = SimpleTrajectoryFilter(
            filter_type='moving_average',
            window_size=3
        )
        
        # 测试数据（包含角度跳跃）
        test_poses = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            # 测试角度跳跃
            [0.4, 0.4, 0.4, 0.4, 0.4, 3.0],  # yaw 从 0.3 跳到 3.0
        ]
        
        # 滤波测试
        for i, pose in enumerate(test_poses):
            filtered_pose = filter_obj.filter_pose(pose)
            print(f"  输入 {i+1}: {[f'{x:.3f}' for x in pose]}")
            print(f"  输出 {i+1}: {[f'{x:.3f}' for x in filtered_pose]}")
        
        print("✓ SimpleTrajectoryFilter 测试通过")
        return True
        
    except Exception as e:
        print(f"✗ SimpleTrajectoryFilter 测试失败: {e}")
        return False

def test_trajectory_filter():
    """测试完整滤波器"""
    print("\n测试 TrajectoryFilter...")
    
    try:
        from filter import TrajectoryFilter
        
        # 创建滤波器
        filter_obj = TrajectoryFilter(
            filter_type='moving_average',
            window_size=3
        )
        
        # 测试数据
        test_trajectory = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        ]
        
        # 滤波测试
        filtered_trajectory = filter_obj.filter_trajectory(test_trajectory)
        print(f"  输入轨迹长度: {len(test_trajectory)}")
        print(f"  输出轨迹长度: {len(filtered_trajectory)}")
        
        print("✓ TrajectoryFilter 测试通过")
        return True
        
    except Exception as e:
        print(f"✗ TrajectoryFilter 测试失败: {e}")
        return False

def test_imports():
    """测试导入"""
    print("测试模块导入...")
    
    try:
        # 测试从 filter 包导入
        from filter import SimpleTrajectoryFilter, TrajectoryFilter
        print("✓ 包导入成功")
        
        # 测试直接导入
        from filter.simple_trajectory_filter import SimpleTrajectoryFilter as SimpleFilter
        from filter.trajectory_filter import TrajectoryFilter as TrajectoryFilter
        print("✓ 直接导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("轨迹滤波模块测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_simple_filter,
        test_trajectory_filter,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！模块工作正常。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
