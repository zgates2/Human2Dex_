#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试离线旋转滤波器
验证 (n, 6) 格式的 numpy array 输入和离线滤波功能
"""

import sys
import os
import numpy as np

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_offline_filter():
    """测试离线旋转滤波器"""
    print("测试离线旋转滤波器...")
    
    try:
        from filter import OfflineRotationFilter
        
        # 创建滤波器
        filter_obj = OfflineRotationFilter(
            filter_type='savgol',
            window_size=5,
            polyorder=2
        )
        
        # 生成测试数据 (n, 6) 格式
        n_points = 50
        t = np.linspace(0, 2*np.pi, n_points)
        
        # 生成带噪声的轨迹
        trajectory = np.zeros((n_points, 6))
        for i, time in enumerate(t):
            x = 0.1 * np.sin(0.5 * time) + 0.02 * np.random.randn()
            y = 0.1 * np.cos(0.5 * time) + 0.02 * np.random.randn()
            z = 0.05 * time + 0.01 * np.random.randn()
            roll = 0.1 * np.sin(0.3 * time) + 0.05 * np.random.randn()
            pitch = 0.1 * np.cos(0.3 * time) + 0.05 * np.random.randn()
            yaw = 0.2 * time + 0.1 * np.random.randn()
            
            trajectory[i] = [x, y, z, roll, pitch, yaw]
        
        print(f"输入轨迹形状: {trajectory.shape}")
        print(f"数据格式: (n_points, 6) = (x, y, z, roll, pitch, yaw)")
        
        # 滤波轨迹
        filtered_trajectory = filter_obj.filter_trajectory(trajectory)
        
        print(f"输出轨迹形状: {filtered_trajectory.shape}")
        
        # 验证输出形状
        assert filtered_trajectory.shape == trajectory.shape, f"输出形状不匹配: {filtered_trajectory.shape} != {trajectory.shape}"
        
        # 显示前几个数据点
        print("\n前5个数据点对比:")
        print("索引 | 原始轨迹                    | 滤波轨迹")
        print("-" * 80)
        
        for i in range(min(5, trajectory.shape[0])):
            original = trajectory[i]
            filtered = filtered_trajectory[i]
            
            print(f"{i:3d}  | "
                  f"[{original[0]:6.3f}, {original[1]:6.3f}, {original[2]:6.3f}, "
                  f"{original[3]:6.3f}, {original[4]:6.3f}, {original[5]:6.3f}] | "
                  f"[{filtered[0]:6.3f}, {filtered[1]:6.3f}, {filtered[2]:6.3f}, "
                  f"{filtered[3]:6.3f}, {filtered[4]:6.3f}, {filtered[5]:6.3f}]")
        
        print("✓ 离线旋转滤波器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 离线旋转滤波器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_filters():
    """测试不同的滤波方法"""
    print("\n测试不同滤波方法...")
    
    try:
        from filter import OfflineRotationFilter
        
        # 生成测试数据
        n_points = 30
        trajectory = np.random.randn(n_points, 6) * 0.1
        
        filter_types = ['savgol', 'moving_average', 'butterworth']
        
        for filter_type in filter_types:
            print(f"\n测试 {filter_type} 滤波...")
            
            # 创建滤波器
            if filter_type == 'butterworth':
                filter_obj = OfflineRotationFilter(
                    filter_type=filter_type,
                    window_size=5,
                    cutoff_freq=2.0,
                    sampling_rate=50.0,
                    order=4
                )
            else:
                filter_obj = OfflineRotationFilter(
                    filter_type=filter_type,
                    window_size=5,
                    polyorder=2
                )
            
            # 滤波轨迹
            filtered_trajectory = filter_obj.filter_trajectory(trajectory)
            
            # 验证输出
            assert filtered_trajectory.shape == trajectory.shape, f"输出形状不匹配"
            
            # 计算滤波效果
            mse = np.mean((trajectory - filtered_trajectory) ** 2)
            print(f"  均方误差: {mse:.6f}")
            
            print(f"✓ {filter_type} 滤波测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 不同滤波方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    
    try:
        from filter import OfflineRotationFilter
        
        filter_obj = OfflineRotationFilter(filter_type='savgol', window_size=5, polyorder=2)
        
        # 测试空轨迹
        empty_trajectory = np.zeros((0, 6))
        filtered_empty = filter_obj.filter_trajectory(empty_trajectory)
        assert filtered_empty.shape == empty_trajectory.shape, "空轨迹处理失败"
        print("✓ 空轨迹测试通过")
        
        # 测试单点轨迹
        single_point = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])
        filtered_single = filter_obj.filter_trajectory(single_point)
        assert filtered_single.shape == single_point.shape, "单点轨迹处理失败"
        print("✓ 单点轨迹测试通过")
        
        # 测试两点轨迹
        two_points = np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
                              [1.1, 2.1, 3.1, 0.11, 0.21, 0.31]])
        filtered_two = filter_obj.filter_trajectory(two_points)
        assert filtered_two.shape == two_points.shape, "两点轨迹处理失败"
        print("✓ 两点轨迹测试通过")
        
        return True
        
    except Exception as e:
        print(f"✗ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("离线旋转滤波器测试")
    print("=" * 60)
    
    tests = [
        test_offline_filter,
        test_different_filters,
        test_edge_cases,
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
        print("🎉 所有测试通过！离线旋转滤波器工作正常。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    print("=" * 60)
    
    # 打印使用示例
    print("\n使用示例:")
    print("```python")
    print("from filter import OfflineRotationFilter")
    print("import numpy as np")
    print("")
    print("# 创建滤波器")
    print("filter_obj = OfflineRotationFilter(")
    print("    filter_type='savgol',")
    print("    window_size=7,")
    print("    polyorder=2")
    print(")")
    print("")
    print("# 准备轨迹数据 (n, 6) 格式")
    print("trajectory = np.random.randn(100, 6)  # 100个点，6维位姿")
    print("")
    print("# 滤波轨迹")
    print("filtered_trajectory = filter_obj.filter_trajectory(trajectory)")
    print("print(f'输入形状: {trajectory.shape}')")
    print("print(f'输出形状: {filtered_trajectory.shape}')")
    print("```")

if __name__ == "__main__":
    main()

