#!/usr/bin/env python3
"""
测试多线程控制功能的脚本
"""

import time
import threading
from single_flexiv_controller import FlexivController

def test_multithread_control():
    """测试多线程控制功能"""
    
    # 创建控制器实例
    controller = FlexivController()
    
    try:
        print("=== 多线程控制功能测试 ===")
        
        # 1. 检查初始状态
        print("\n1. 检查初始控制状态:")
        status = controller.get_control_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # 2. 测试tcp_move_v3实时控制
        print("\n2. 测试tcp_move_v3实时控制:")
        test_poses = [
            [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],  # 位置1
            [0.6, 0.1, 0.4, 1.0, 0.0, 0.0, 0.0],  # 位置2
            [0.5, -0.1, 0.4, 1.0, 0.0, 0.0, 0.0], # 位置3
        ]
        
        for i, pose in enumerate(test_poses):
            print(f"   设置目标位姿 {i+1}: {pose}")
            controller.tcp_move_v3(pose)
            time.sleep(3.0)  # 等待3秒观察效果
            
            # 检查当前状态
            current_status = controller.get_control_status()
            print(f"   当前状态: 有目标={current_status['has_target']}, 目标位姿={current_status['target_pose']}")
        
        # 3. 测试暂停和恢复控制
        print("\n3. 测试暂停和恢复控制:")
        print("   暂停控制...")
        controller.pause_control()
        time.sleep(2.0)
        
        print("   恢复控制...")
        controller.resume_control([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
        time.sleep(2.0)
        
        # 4. 测试reset_to_home（会停止并重启控制线程）
        print("\n4. 测试reset_to_home:")
        print("   执行重置操作...")
        controller.reset_to_home()
        time.sleep(2.0)
        
        # 检查重置后的状态
        status_after_reset = controller.get_control_status()
        print(f"   重置后状态: {status_after_reset}")
        
        # 5. 测试连续实时控制
        print("\n5. 测试连续实时控制:")
        for i in range(5):
            # 生成一个圆形轨迹
            angle = i * 0.5  # 弧度
            x = 0.5 + 0.05 * np.cos(angle)
            y = 0.0 + 0.05 * np.sin(angle)
            z = 0.4
            pose = [x, y, z, 1.0, 0.0, 0.0, 0.0]
            
            print(f"   轨迹点 {i+1}: [{x:.3f}, {y:.3f}, {z:.3f}]")
            controller.tcp_move_v3(pose)
            time.sleep(1.0)
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭控制器
        print("\n关闭控制器...")
        controller.close()

def test_thread_safety():
    """测试线程安全性"""
    
    controller = FlexivController()
    
    try:
        print("=== 线程安全性测试 ===")
        
        # 创建多个线程同时调用tcp_move_v3
        def worker_thread(thread_id, poses):
            for i, pose in enumerate(poses):
                try:
                    controller.tcp_move_v3(pose)
                    print(f"线程 {thread_id} 设置位姿 {i+1}: {pose}")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"线程 {thread_id} 错误: {e}")
        
        # 创建两个工作线程
        poses1 = [[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] for _ in range(3)]
        poses2 = [[0.6, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] for _ in range(3)]
        
        thread1 = threading.Thread(target=worker_thread, args=(1, poses1))
        thread2 = threading.Thread(target=worker_thread, args=(2, poses2))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        print("线程安全性测试完成")
        
    except Exception as e:
        print(f"线程安全性测试错误: {e}")
    
    finally:
        controller.close()

if __name__ == "__main__":
    import numpy as np
    
    print("选择测试模式:")
    print("1. 基本功能测试")
    print("2. 线程安全性测试")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_multithread_control()
    elif choice == "2":
        test_thread_safety()
    else:
        print("无效选择，运行基本功能测试")
        test_multithread_control()

