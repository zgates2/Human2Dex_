#!/usr/bin/env python3
"""
测试当target_pose为None时使用上一个有效位姿的功能
"""

import time
import threading
from single_flexiv_controller import FlexivController

def test_none_pose_behavior():
    """测试设置None位姿时的行为"""
    
    print("=== 测试None位姿行为 ===")
    
    # 创建控制器实例
    controller = FlexivController()
    
    try:
        # 启动控制线程
        controller._start_control_thread()
        time.sleep(2.0)  # 等待控制线程启动
        
        print("\n1. 设置初始目标位姿:")
        initial_pose = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]
        controller.set_target_pose(initial_pose)
        print(f"   设置初始位姿: {initial_pose}")
        
        # 等待一段时间让控制线程执行
        time.sleep(3.0)
        
        print("\n2. 检查控制状态:")
        status = controller.get_control_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n3. 设置None位姿（暂停控制但保持上一个位姿）:")
        controller.set_target_pose(None)
        print("   目标位姿设置为None")
        
        # 等待一段时间观察行为
        print("   等待5秒观察控制行为...")
        time.sleep(5.0)
        
        print("\n4. 再次检查控制状态:")
        status_after_none = controller.get_control_status()
        for key, value in status_after_none.items():
            print(f"   {key}: {value}")
        
        print("\n5. 设置新的目标位姿:")
        new_pose = [0.6, 0.1, 0.4, 1.0, 0.0, 0.0, -1.0]
        controller.set_target_pose(new_pose)
        print(f"   设置新位姿: {new_pose}")
        
        # 等待执行
        time.sleep(3.0)
        
        print("\n6. 再次设置None位姿:")
        controller.set_target_pose(None)
        print("   目标位姿再次设置为None")
        
        # 等待观察
        time.sleep(5.0)
        
        print("\n7. 最终状态检查:")
        final_status = controller.get_control_status()
        for key, value in final_status.items():
            print(f"   {key}: {value}")
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭控制器
        print("\n关闭控制器...")
        controller.close()

def test_pause_resume_behavior():
    """测试暂停和恢复控制的行为"""
    
    print("=== 测试暂停和恢复控制行为 ===")
    
    controller = FlexivController()
    
    try:
        controller._start_control_thread()
        time.sleep(2.0)
        
        print("\n1. 设置目标位姿:")
        target_pose = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]
        controller.set_target_pose(target_pose)
        print(f"   目标位姿: {target_pose}")
        
        time.sleep(3.0)
        
        print("\n2. 暂停控制:")
        controller.pause_control()
        
        print("   等待5秒观察暂停行为...")
        time.sleep(5.0)
        
        print("\n3. 恢复控制:")
        controller.resume_control([0.6, 0.1, 0.4, 1.0, 0.0, 0.0, -1.0])
        
        time.sleep(3.0)
        
        print("\n4. 最终状态:")
        status = controller.get_control_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n=== 暂停恢复测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        controller.close()

def test_continuous_none_behavior():
    """测试连续设置None的行为"""
    
    print("=== 测试连续设置None的行为 ===")
    
    controller = FlexivController()
    
    try:
        controller._start_control_thread()
        time.sleep(2.0)
        
        print("\n1. 设置初始位姿:")
        initial_pose = [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0]
        controller.set_target_pose(initial_pose)
        print(f"   初始位姿: {initial_pose}")
        
        time.sleep(3.0)
        
        print("\n2. 连续设置None位姿:")
        for i in range(10):
            controller.set_target_pose(None)
            print(f"   第{i+1}次设置None")
            time.sleep(0.5)
        
        print("\n3. 检查最终状态:")
        status = controller.get_control_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        print("\n=== 连续None测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        controller.close()

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 测试None位姿行为")
    print("2. 测试暂停和恢复控制")
    print("3. 测试连续设置None")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == "1":
        test_none_pose_behavior()
    elif choice == "2":
        test_pause_resume_behavior()
    elif choice == "3":
        test_continuous_none_behavior()
    else:
        print("无效选择，运行None位姿行为测试")
        test_none_pose_behavior()

