#!/usr/bin/env python3
"""
性能监控测试脚本
"""

import time
import threading
from single_flexiv_controller import FlexivController

def test_basic_performance():
    """测试基本性能"""
    
    print("=== 基本性能测试 ===")
    
    controller = FlexivController()
    
    try:
        # 启动控制线程
        controller._start_control_thread()
        time.sleep(2.0)
        
        print("\n1. 测试连续设置目标位姿的性能:")
        num_poses = 100
        
        start_time = time.time()
        for i in range(num_poses):
            # 生成测试位姿
            test_pose = [0.5 + i*0.001, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]
            controller.set_target_pose(test_pose)
            
            # 每20次检查一次性能
            if i % 20 == 0:
                stats = controller.get_performance_stats()
                print(f"   设置 {i} 个位姿: 频率={stats['current_frequency']:.2f}Hz, "
                      f"队列大小={stats['queue_size']}")
        
        total_time = time.time() - start_time
        avg_time = total_time / num_poses
        print(f"   总耗时: {total_time:.4f}秒")
        print(f"   平均每次设置时间: {avg_time*1000:.2f}ms")
        print(f"   理论频率: {1/avg_time:.1f}Hz")
        
        # 性能监控
        print("\n2. 性能监控:")
        controller.monitor_performance(duration=5.0)
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        controller.close()

def test_high_frequency_control():
    """测试高频率控制"""
    
    print("=== 高频率控制测试 ===")
    
    controller = FlexivController()
    
    try:
        controller._start_control_thread()
        time.sleep(2.0)
        
        print("\n1. 测试1000Hz控制频率:")
        target_freq = 1000  # 1000Hz
        interval = 1.0 / target_freq  # 1ms间隔
        
        start_time = time.time()
        for i in range(1000):
            target_pose = [0.5 + i*0.0001, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]
            controller.set_target_pose(target_pose)
            
            # 精确控制时间间隔
            elapsed = time.time() - start_time
            expected_time = i * interval
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)
            
            # 每100次检查频率
            if i % 100 == 0:
                stats = controller.get_performance_stats()
                print(f"   第 {i} 次: 实际频率={stats['current_frequency']:.1f}Hz, "
                      f"目标频率={target_freq}Hz")
        
        actual_time = time.time() - start_time
        actual_freq = 1000 / actual_time
        print(f"   实际平均频率: {actual_freq:.1f}Hz")
        print(f"   目标频率: {target_freq}Hz")
        print(f"   频率误差: {abs(actual_freq - target_freq):.1f}Hz")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        controller.close()

def test_none_pose_performance():
    """测试None位姿处理的性能"""
    
    print("=== None位姿处理性能测试 ===")
    
    controller = FlexivController()
    
    try:
        controller._start_control_thread()
        time.sleep(2.0)
        
        print("\n1. 设置初始位姿:")
        initial_pose = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]
        controller.set_target_pose(initial_pose)
        time.sleep(2.0)
        
        print("\n2. 测试None位姿设置性能:")
        start_time = time.time()
        for i in range(100):
            controller.set_target_pose(None)
            if i % 20 == 0:
                stats = controller.get_performance_stats()
                print(f"   第 {i} 次设置None: 频率={stats['current_frequency']:.2f}Hz")
        
        none_time = time.time() - start_time
        print(f"   设置None总耗时: {none_time:.4f}秒")
        print(f"   平均每次None设置时间: {none_time*1000/100:.2f}ms")
        
        print("\n3. 测试恢复控制性能:")
        start_time = time.time()
        for i in range(100):
            test_pose = [0.5 + i*0.001, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]
            controller.set_target_pose(test_pose)
            if i % 20 == 0:
                stats = controller.get_performance_stats()
                print(f"   第 {i} 次恢复控制: 频率={stats['current_frequency']:.2f}Hz")
        
        resume_time = time.time() - start_time
        print(f"   恢复控制总耗时: {resume_time:.4f}秒")
        print(f"   平均每次恢复时间: {resume_time*1000/100:.2f}ms")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        controller.close()

def test_mixed_operations():
    """测试混合操作性能"""
    
    print("=== 混合操作性能测试 ===")
    
    controller = FlexivController()
    
    try:
        controller._start_control_thread()
        time.sleep(2.0)
        
        print("\n1. 混合操作测试（设置位姿 + None + 恢复）:")
        operations = []
        
        # 生成混合操作序列
        for i in range(300):
            if i % 3 == 0:
                operations.append(("pose", [0.5 + i*0.001, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]))
            elif i % 3 == 1:
                operations.append(("none", None))
            else:
                operations.append(("pose", [0.6 + i*0.001, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]))
        
        start_time = time.time()
        for i, (op_type, value) in enumerate(operations):
            if op_type == "pose":
                controller.set_target_pose(value)
            else:
                controller.set_target_pose(None)
            
            if i % 50 == 0:
                stats = controller.get_performance_stats()
                print(f"   第 {i} 次操作 ({op_type}): 频率={stats['current_frequency']:.2f}Hz")
        
        total_time = time.time() - start_time
        print(f"   混合操作总耗时: {total_time:.4f}秒")
        print(f"   平均每次操作时间: {total_time*1000/len(operations):.2f}ms")
        print(f"   总操作数: {len(operations)}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        controller.close()

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 基本性能测试")
    print("2. 高频率控制测试")
    print("3. None位姿处理性能测试")
    print("4. 混合操作性能测试")
    
    choice = input("请输入选择 (1, 2, 3 或 4): ").strip()
    
    if choice == "1":
        test_basic_performance()
    elif choice == "2":
        test_high_frequency_control()
    elif choice == "3":
        test_none_pose_performance()
    elif choice == "4":
        test_mixed_operations()
    else:
        print("无效选择，运行基本性能测试")
        test_basic_performance()

