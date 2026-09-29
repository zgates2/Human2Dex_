#!/usr/bin/env python3
"""
测试队列控制功能的脚本
"""

import time
import threading
import queue
from single_flexiv_controller import FlexivController

def test_queue_performance():
    """测试队列控制的性能"""
    
    print("=== 队列控制性能测试 ===")
    
    # 创建控制器实例
    controller = FlexivController()
    
    try:
        # 启动控制线程
        controller._start_control_thread()
        time.sleep(1.0)
        
        # 测试连续设置目标位姿的性能
        print("\n1. 测试连续设置目标位姿的性能:")
        num_poses = 100
        
        start_time = time.time()
        for i in range(num_poses):
            # 生成测试位姿
            test_pose = [0.5 + i*0.001, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
            controller.set_target_pose(test_pose)
            
            # 测量每次设置的时间
            if i % 20 == 0:
                elapsed = time.time() - start_time
                print(f"   设置 {i} 个位姿，耗时: {elapsed:.4f}秒")
        
        total_time = time.time() - start_time
        avg_time = total_time / num_poses
        print(f"   总耗时: {total_time:.4f}秒")
        print(f"   平均每次设置时间: {avg_time*1000:.2f}ms")
        print(f"   理论频率: {1/avg_time:.1f}Hz")
        
        # 测试控制状态
        print("\n2. 测试控制状态:")
        status = controller.get_control_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # 测试暂停和恢复
        print("\n3. 测试暂停和恢复控制:")
        print("   暂停控制...")
        controller.pause_control()
        time.sleep(2.0)
        
        print("   恢复控制...")
        controller.resume_control([0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
        time.sleep(2.0)
        
        # 测试重置功能
        print("\n4. 测试reset_to_home:")
        controller.reset_to_home()
        time.sleep(2.0)
        
        # 检查重置后的状态
        status_after_reset = controller.get_control_status()
        print(f"   重置后状态: {status_after_reset}")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭控制器
        print("\n关闭控制器...")
        controller.close()

def test_queue_thread_safety():
    """测试队列的线程安全性"""
    
    print("=== 队列线程安全性测试 ===")
    
    controller = FlexivController()
    
    try:
        controller._start_control_thread()
        time.sleep(1.0)
        
        # 创建多个线程同时调用set_target_pose
        def worker_thread(thread_id, poses):
            for i, pose in enumerate(poses):
                try:
                    start_time = time.time()
                    controller.set_target_pose(pose)
                    elapsed = time.time() - start_time
                    print(f"线程 {thread_id} 设置位姿 {i+1}: 耗时 {elapsed*1000:.2f}ms")
                    time.sleep(0.01)  # 10ms间隔
                except Exception as e:
                    print(f"线程 {thread_id} 错误: {e}")
        
        # 创建两个工作线程
        poses1 = [[0.5 + i*0.01, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0] for i in range(10)]
        poses2 = [[0.6 + i*0.01, 0.0, 0.4, 1.0, 0.0, 0.0, -1.0] for i in range(10)]
        
        thread1 = threading.Thread(target=worker_thread, args=(1, poses1))
        thread2 = threading.Thread(target=worker_thread, args=(2, poses2))
        
        print("启动多线程测试...")
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        print("多线程测试完成")
        
        # 检查最终状态
        final_status = controller.get_control_status()
        print(f"最终控制状态: {final_status}")
        
    except Exception as e:
        print(f"线程安全性测试错误: {e}")
    
    finally:
        controller.close()

def test_queue_vs_lock_performance():
    """对比队列和锁的性能"""
    
    print("=== 队列 vs 锁性能对比 ===")
    
    # 测试队列性能
    print("\n1. 测试队列性能:")
    q = queue.Queue(maxsize=1)
    
    def queue_writer():
        for i in range(1000):
            try:
                while not q.empty():
                    q.get_nowait()
                q.put_nowait([i, i, i])
            except:
                pass
    
    def queue_reader():
        for i in range(1000):
            try:
                q.get_nowait()
            except queue.Empty:
                pass
    
    # 测试队列写入性能
    start_time = time.time()
    queue_writer()
    queue_time = time.time() - start_time
    print(f"   队列操作耗时: {queue_time*1000:.2f}ms")
    
    # 测试锁性能
    print("\n2. 测试锁性能:")
    import threading
    
    lock = threading.Lock()
    data = [0, 0, 0]
    
    def lock_writer():
        for i in range(1000):
            with lock:
                data[0] = i
                data[1] = i
                data[2] = i
    
    def lock_reader():
        for i in range(1000):
            with lock:
                _ = data[0]
                _ = data[1]
                _ = data[2]
    
    # 测试锁写入性能
    start_time = time.time()
    lock_writer()
    lock_time = time.time() - start_time
    print(f"   锁操作耗时: {lock_time*1000:.2f}ms")
    
    # 性能对比
    if queue_time < lock_time:
        print(f"   队列比锁快: {lock_time/queue_time:.1f}倍")
    else:
        print(f"   锁比队列快: {queue_time/lock_time:.1f}倍")

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 队列控制性能测试")
    print("2. 队列线程安全性测试")
    print("3. 队列 vs 锁性能对比")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == "1":
        test_queue_performance()
    elif choice == "2":
        test_queue_thread_safety()
    elif choice == "3":
        test_queue_vs_lock_performance()
    else:
        print("无效选择，运行队列控制性能测试")
        test_queue_performance()

