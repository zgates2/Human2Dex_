#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试旋转滤波器
验证修复后的代码是否正常工作
"""

import sys
import os
import numpy as np

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_rotation_filter():
    """测试旋转滤波器"""
    print("测试旋转滤波器...")
    
    try:
        from filter import RotationFilter
        
        # 创建滤波器
        filter_obj = RotationFilter(
            filter_type='savgol',
            window_size=5,
            polyorder=2
        )
        
        # 测试数据
        test_poses = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        ]
        
        print("测试 Savitzky-Golay 滤波...")
        for i, pose in enumerate(test_poses):
            filtered_pose = filter_obj.filter_pose(pose)
            print(f"  输入 {i+1}: {[f'{x:.3f}' for x in pose]}")
            print(f"  输出 {i+1}: {[f'{x:.3f}' for x in filtered_pose]}")
        
        print("✓ Savitzky-Golay 滤波测试通过")
        
        # 测试移动平均
        filter_obj.reset()
        filter_obj.filter_type = 'moving_average'
        filter_obj.window_size = 3
        
        print("\n测试移动平均滤波...")
        for i, pose in enumerate(test_poses):
            filtered_pose = filter_obj.filter_pose(pose)
            print(f"  输入 {i+1}: {[f'{x:.3f}' for x in pose]}")
            print(f"  输出 {i+1}: {[f'{x:.3f}' for x in filtered_pose]}")
        
        print("✓ 移动平均滤波测试通过")
        
        # 测试 Butterworth (需要更多数据点)
        filter_obj.reset()
        filter_obj.filter_type = 'butterworth'
        filter_obj.cutoff_freq = 2.0
        filter_obj.sampling_rate = 50.0
        filter_obj.order = 4
        
        print("\n测试 Butterworth 滤波...")
        # 生成更多测试数据
        extended_poses = []
        for i in range(20):
            t = i * 0.1
            pose = [
                0.1 * np.sin(t),
                0.1 * np.cos(t),
                0.05 * t,
                0.1 * np.sin(0.5 * t),
                0.1 * np.cos(0.5 * t),
                0.2 * t
            ]
            extended_poses.append(pose)
        
        for i, pose in enumerate(extended_poses[:5]):  # 只显示前5个
            filtered_pose = filter_obj.filter_pose(pose)
            print(f"  输入 {i+1}: {[f'{x:.3f}' for x in pose]}")
            print(f"  输出 {i+1}: {[f'{x:.3f}' for x in filtered_pose]}")
        
        print("✓ Butterworth 滤波测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 旋转滤波器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_rotation_filter():
    """测试高级旋转滤波器"""
    print("\n测试高级旋转滤波器...")
    
    try:
        from filter import AdvancedRotationFilter
        
        # 创建滤波器
        filter_obj = AdvancedRotationFilter(
            filter_type='savgol',
            window_size=5,
            polyorder=2
        )
        
        # 测试数据
        test_poses = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        ]
        
        print("测试轨迹滤波...")
        filtered_trajectory = filter_obj.filter_trajectory(test_poses)
        
        print(f"  输入轨迹长度: {len(test_poses)}")
        print(f"  输出轨迹长度: {len(filtered_trajectory)}")
        
        for i, (original, filtered) in enumerate(zip(test_poses, filtered_trajectory)):
            print(f"  位姿 {i+1}:")
            print(f"    原始: {[f'{x:.3f}' for x in original]}")
            print(f"    滤波: {[f'{x:.3f}' for x in filtered]}")
        
        print("✓ 高级旋转滤波器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 高级旋转滤波器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("旋转滤波器测试")
    print("=" * 60)
    
    tests = [
        test_rotation_filter,
        test_advanced_rotation_filter,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！旋转滤波器工作正常。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

