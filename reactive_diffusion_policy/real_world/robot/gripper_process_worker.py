import sys
sys.path.append("/home/legion/collect/reactive_diffusion_policy")

import time
from multiprocessing import Process, Queue
import queue
# from reactive_diffusion_policy.real_world.robot.gripper_controller_map import GripperController
# from reactive_diffusion_policy.real_world.robot.gripper_controller import GripperController
from reactive_diffusion_policy.real_world.robot.gripper_controller_1011 import GripperController
# from reactive_diffusion_policy.real_world.robot.gripper_controller_set_speed import GripperController


from loguru import logger
from typing import List, Optional
import traceback
import psutil

class GripperProcessProxy:
    """
    GripperController 进程的代理。
    负责启动、停止进程，并通过队列发送命令和接收状态。
    """
    def __init__(self, config: dict):
        self._command_queue = Queue()
        self._state_queue = Queue(maxsize=1)
        self._process = Process(
            target=gripper_process_worker,
            args=(self._command_queue, self._state_queue, config),
            daemon=True
        )
        # 单电机模式：返回 [position, velocity, torque, 0, 0, 0]
        self._last_known_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def start(self):
        """启动夹爪子进程。"""
        self._process.start()
        logger.info("Gripper process has been started.")

    def stop(self):
        """停止夹爪子进程。"""
        logger.info("Stopping gripper process...")
        if self._process.is_alive():
            self._command_queue.put(('shutdown', None))
            self._process.join(timeout=5)
            if self._process.is_alive():
                logger.warning("Gripper process did not terminate gracefully. Forcing termination.")
                self._process.terminate()
        logger.info("Gripper process has been stopped.")

    def get_current_gripper_states(self) -> List[float]:
        """获取最新的夹爪状态（非阻塞）。"""
        try:
            self._last_known_state = self._state_queue.get_nowait()
        except queue.Empty:
            pass
        return self._last_known_state

    def __getattr__(self, name):
        """动态生成方法，将方法调用转换为队列中的命令。"""
        def method(*args):
            if name == 'stop_gripper':
                 self._command_queue.put(('stop', args))
            else:
                 self._command_queue.put((name, args))

        return method


def gripper_process_worker(command_queue: Queue, state_queue: Queue, config: dict):
    """
    在独立进程中运行的夹爪控制器工作函数。
    """
    # p = psutil.Process()
    # p.cpu_affinity([1])
    # logger.info("夹爪进程已绑定到CPU核心1。")
    try:
        logger.info(f"Gripper process started. Config: {config}")
        gripper = GripperController(**config)
        gripper.start()
    except Exception as e:
        logger.error(f"Failed to initialize GripperController in a separate process: {e}")
        return

    running = True
    
    try:
        while running:
            try:
                # 1. 检查来自主进程的新命令
                if not command_queue.empty():
                    command, args = command_queue.get()

                    # 'shutdown' 是我们用来终止这个 worker 循环的特殊命令
                    # 它会调用 gripper 自己的 stop 方法
                    if command == 'shutdown': 
                        logger.info("Shutdown command received. Stopping gripper and worker process.")
                        running = False
                        continue

                    # 查找 gripper 对象上是否存在该方法
                    if hasattr(gripper, command):
                        method = getattr(gripper, command)
                        # 调用该方法，将命令转发给 GripperController 的内部队列
                        if args:
                            method(*args)
                        else:
                            method()
                    else:
                        logger.warning(f"GripperController has no method named '{command}'")

                # 2. 从 gripper 获取当前状态
                current_state = gripper.get_current_gripper_states()

                try:
                    state_queue.put_nowait(current_state)
                except queue.Full:
                    try:
                        state_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        state_queue.put_nowait(current_state)
                    except queue.Full:
                        pass

                time.sleep(0.02)

            except Exception as e:
                logger.error("An unexpected error occurred in the gripper process loop. Full traceback below:")
                logger.error(traceback.format_exc()) 
                time.sleep(1)
    except KeyboardInterrupt:
        pass
        
    except Exception as e:
        logger.error("An unexpected error occurred in the gripper process loop. Full traceback below:")
        logger.error(traceback.format_exc()) 
        time.sleep(1)

    gripper.stop()
    logger.info("Gripper process finished.")


def test_dual_mode():
    """测试双电机互控模式"""
    logger.info("\n--- Main Process: Testing DUAL motor mode ---")
    
    gripper_config = {
        'mode': 'dual', 
        'port1': "/dev/ttyACM3", 'motor1_id': 1, 
        'port2': "/dev/ttyACM2", 'motor2_id': 2
    }

    proxy = GripperProcessProxy(config=gripper_config)
    proxy.start()
    time.sleep(5)

    try:
        logger.info("双电机同步已启动。可以手动移动一个电机，另一个会跟随。")
        logger.info("按 Ctrl+C 停止。")
        
        loop_count = 0
        last_freq_time = time.perf_counter()
        
        while True:
            current_state = proxy.get_current_gripper_states()
            
            loop_count += 1

            current_time = time.perf_counter()
            if current_time - last_freq_time >= 1.0:
                freq = loop_count / (current_time - last_freq_time)
                logger.info(f"Main Loop Info Acquisition Freq: {freq:.2f} Hz")
                loop_count = 0
                last_freq_time = current_time
                
            pos1, torque1, pos2, torque2 = current_state
            print(f"电机1: Pos={pos1:.3f} rad, Torque={torque1:.3f} Nm | 电机2: Pos={pos2:.3f} rad, Torque={torque2:.3f} Nm", end='\r')
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\n检测到手动中断...")
    finally:
        logger.info("\n--- Main Process: Test finished. Stopping proxy. ---")
        proxy.stop()

def test_single_mode():
    """测试单电机柔顺控制模式"""
    logger.info("\n--- Main Process: Testing SINGLE motor mode ---")

    # 1. 配置真实的端口和ID（新格式：移除mode参数）
    gripper_config = {
        'port': "/dev/ttyUSB2", 
        'motor_id': 2
    }

    proxy = GripperProcessProxy(config=gripper_config)
    proxy.start()
    time.sleep(3) # 等待电机初始化、校准零点

    try:
        # 2. 调用您控制器中定义好的方法
        logger.info("\n[任务] 移动到 50mm，力限制 1.2 Nm...")
        proxy.move_gripper(width_m=0.05, force_limit_nm=1.2)
        
        # 监控移动过程
        for _ in range(50): # 持续5秒
            states = proxy.get_current_gripper_states()
            width_m, force_nm = states
            print(f"状态: 宽度={width_m*1000:.2f} mm, 力={force_nm:.3f} Nm", end='\r')
            time.sleep(0.1)
        print("\n移动任务应已完成或被力打断。")

        time.sleep(2)
        logger.info("\n[任务] 以 1.0 Nm 的力夹持...")
        proxy.move_gripper_force(force_limit_nm=1.0)
        time.sleep(5) # 等待夹持完成

        time.sleep(2)
        logger.info("\n[任务] 完全打开 (移动到 80mm)...")
        proxy.move_gripper(width_m=0.08, force_limit_nm=1.2)
        time.sleep(3) # 等待打开

    except KeyboardInterrupt:
        print("\n检测到手动中断...")
    finally:
        logger.info("\n--- Main Process: Test finished. Stopping proxy. ---")
        proxy.stop()

if __name__ == "__main__":
    # 配置日志记录器
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # --- 选择要运行的测试 ---
    # 运行双电机测试
    test_dual_mode()
    
    # 或者，注释掉上面的，取消注释下面的来运行单电机测试
    # test_single_mode()

    logger.info("--- Main Process: Done. ---")