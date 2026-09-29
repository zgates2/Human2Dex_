import math
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import time
import queue
from motor.motor import LkMotor
from motor.protocol import radian_to_degree
from loguru import logger

class GripperController:
    MM_PER_SINGLE_MOTOR_RADIAN = 45.0000
    # MOTOR_PHYSICAL_LIMIT_RAD = math.pi / 2
    MOTOR_PHYSICAL_LIMIT_M = 0.16
    POSITION_REACHED_THRESHOLD_RAD = 0.02
    
    SOFT_TORQUE_THRESHOLD_RATIO = 0.8
    BACK_OFF_STEP_RAD = 0.01
    
    VELOCITY_P_GAIN = 200.0
    MAX_SPEED_DPS = 60.0

    def __init__(self, mode: str, port1: str, motor1_id: int, port2: str = None, motor2_id: int = None):
        
        self.mode = mode
        
        if self.mode not in ['dual', 'single']:
            raise ValueError("Mode must be 'dual' or 'single'")

        print(f"控制器以 '{self.mode}' 模式初始化...")
        self.motor1 = LkMotor(port1, motor_id=motor1_id)
        self.motor2 = None
        if self.mode == 'dual':
            if not port2 or not motor2_id:
                raise ValueError("Port2 and motor2_id are required for 'dual' mode")
            self.motor2 = LkMotor(port2, motor_id=motor2_id)

        self.ratio = 1.1 # 1.2
        self.KP = 2.0 / 2 / 6 * 1.5 * 1.3 * self.ratio
        self.KD = 0.01 / 1.5 * 1.5 * self.ratio * 1.0
        self.TORQUE_LIMIT_DUAL = 2.0 # 2.5
        self.TORQUE_LIMIT_SINGLE = 2.0
        self.DT = 0.025
        
        # 位置和力的放大参数
        self.a = 1.0  # 力的放大倍数（主端感受到的力的倒数）
        self.b = 1.0  # 位置的放大倍数（从端跟随主端位置的倍数）
        # self.a = 1.0
        # self.b = 2.0
        # self.DT = 0.01
        
        # single motor control
        self.MIN_SPEED_DPS = 8.0            # 起步最小速度（°/s）
        self.MAX_ACCEL_DPS2 = 800.0         # 最大角加速度（°/s^2）
        self.VELOCITY_P_GAIN = 120.0        # 位置误差 -> 速度（°/s / rad）
        self.SOFT_TORQUE_THRESHOLD_RATIO = 0.75
        self.SOFT_HOLD_CYCLES = 3
        self.SOFT_RELEASE_RATIO = 0.6
        self.BACKOFF_SPEED_RATIO = 0.35
        self.BACKOFF_TIME_S = 0.15
        self.TORQUE_DEADBAND = 0.08
        self._speed_dir_sign = 1.0

        # 扭矩低通滤波状态
        self._torque_filt = 0.0
        self._torque_lpf_alpha = 0.2

        self.thread = None
        self.command_queue = queue.Queue()

        self.current_position_rad = 0.0
        self.current_torque_nm = 0.0
        self.current_position_rad_2 = 0.0
        self.current_torque_nm_2 = 0.0
        
        self.busy_wait_threshold = 0.0015
        
    def _angle_to_width(self, angle_rad: float) -> float:
        width_mm = angle_rad * self.MM_PER_SINGLE_MOTOR_RADIAN
        return width_mm / 1000.0 * 2.0

    def _width_to_angle(self, width_m: float) -> float:
        limited_width_m = min(width_m, self.MOTOR_PHYSICAL_LIMIT_M)
        width_mm = limited_width_m * 1000.0 / 2.0
        angle_rad = width_mm / self.MM_PER_SINGLE_MOTOR_RADIAN
        
        return angle_rad
        # return min(angle_rad, self.MOTOR_PHYSICAL_LIMIT_RAD)

    def get_current_gripper_width(self) -> float:
        """获取当前夹爪的开合宽度 (m)"""
        return self._angle_to_width(self.current_position_rad)

    def get_current_gripper_force(self) -> float:
        """获取当前夹爪的夹持力 (Nm)"""
        return self.current_torque_nm

    def get_current_gripper_states(self):
        """获取当前夹爪的状态"""
        if self.mode == 'dual':
            return [
                self.current_position_rad,
                self.current_torque_nm,
                self.current_position_rad_2,
                self.current_torque_nm_2
            ]
        else:
            width = self._angle_to_width(self.current_position_rad)
            force = self.current_torque_nm
            return [width, force]
    
    def move_gripper(self, width_m: float, force_limit_nm: float):
        if self.mode != 'single':
            print("[错误] move_gripper 只能在 'single' 模式下使用。")
            return
        target_rad = self._width_to_angle(width_m)
        self.command_queue.put(('move', target_rad, abs(force_limit_nm)))
        print(f"指令已发送: 移动到宽度 {width_m*1000:.2f} mm (角度: {target_rad:.3f} rad) 力限制: {abs(force_limit_nm):.2f} Nm")

    def move_gripper_force(self, force_limit_nm: float, close_speed_dps: float = 30.0):
        if self.mode != 'single':
            print("[错误] move_gripper_force 只能在 'single' 模式下使用。")
            return
        self.command_queue.put(('grasp', abs(force_limit_nm), abs(close_speed_dps)))
        print(f"指令已发送: 夹持至力矩达到 {abs(force_limit_nm):.2f} Nm")
    
    def stop_gripper(self):
        if self.mode != 'single':
            print("[警告] stop_gripper 只在 'single' 模式下有效。")
            return
        self.command_queue.put(('stop',))
        print("[事件] 停止指令已发送。")
            
    def _run(self):
        if self.mode == 'dual':
            self._dual_control_loop()
        else:
            self._single_control_loop()
        print("控制循环已结束。")
        
    def precise_sleep(self, delay):
        target_time = time.perf_counter() + delay
        
        sleep_duration = delay - self.busy_wait_threshold
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        
        while time.perf_counter() < target_time:
            pass

    def _single_control_loop(self):
        """
        单电机控制循环，实现“位置保持”功能。(修正版)
        """
        print("开始单电机控制循环 (位置保持模式)...")
        running = True

        control_state = 'IDLE' 
        
        target_position_rad = 0.0

        while running:
            loop_start = time.perf_counter()

            try:
                command = self.command_queue.get_nowait()
                cmd_type = command[0]

                if cmd_type == 'shutdown':
                    running = False
                    continue
                
                elif cmd_type == 'move':
                    _, target_position_rad, force_limit_nm = command
                    control_state = 'POSITION_MODE' 

            except queue.Empty:
                pass
            
            if control_state == 'POSITION_MODE':
                self.motor1.refresh()

                if getattr(self.motor1, 'refresh_ok', True) is False:
                    print("[警告] 电机数据无效，跳过此次循环。")
                    self.precise_sleep(self.DT)
                    continue

                pos_rad, vel_rad_s, torque_nm = self.motor1.getPosition(), self.motor1.getVelocity(), self.motor1.getTorque()
                
                self.current_position_rad = pos_rad
                self.current_torque_nm = torque_nm
                
                pos_error = target_position_rad - pos_rad
                vel_error = 0.0 - vel_rad_s
                
                torque_cmd = self.KP * pos_error + self.KD * vel_error
                
                torque_cmd = max(-self.TORQUE_LIMIT_SINGLE, min(self.TORQUE_LIMIT_SINGLE, torque_cmd))

                self.motor1.set_torque_nm(torque_cmd)

            elapsed = time.perf_counter() - loop_start
            remaining_time = self.DT - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time) 
        
        self.motor1.stop()
        print("单电机控制循环已停止。")
                
    def _dual_control_loop(self):
        print("开始双电机互控...")
        running = True
        
        # loop_count = 0
        # last_freq_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            while running:
                loop_start = time.perf_counter()
                
                try:
                    command = self.command_queue.get_nowait()
                    if command[0] == 'shutdown': running = False; continue
                except queue.Empty:
                    pass

                # flag_motor1 = self.motor1.refresh()
                # flag_motor2 = self.motor2.refresh()

                # refresh_thread_1 = threading.Thread(target=self.motor1.refresh)
                # refresh_thread_2 = threading.Thread(target=self.motor2.refresh)
                refresh_futures = [
                    executor.submit(self.motor1.refresh),
                    executor.submit(self.motor2.refresh)
                ]
                
                wait(refresh_futures)

                # refresh_thread_1.start()
                # refresh_thread_2.start()

                # refresh_thread_1.join()
                # refresh_thread_2.join()

                # if not (self.motor1.is_valid() and self.motor2.is_valid()):
                # if not (flag_motor1 and flag_motor2):
                if (self.motor1.refresh_ok is False) or (self.motor2.refresh_ok is False):
                    time.sleep(self.DT); continue

                pos1, vel1, torque1_val = self.motor1.getPosition(), self.motor1.getVelocity(), self.motor1.getTorque() 
                pos2, vel2, torque2_val = self.motor2.getPosition(), self.motor2.getVelocity(), self.motor2.getTorque()

                self.current_position_rad = pos1
                self.current_torque_nm = torque1_val
                self.current_position_rad_2 = pos2
                self.current_torque_nm_2 = torque2_val
                    
                # pos_error_1, vel_error_1 = pos2 - pos1, vel2 - vel1
                # pos_error_2, vel_error_2 = pos1 - pos2, vel1 - vel2

                # torque1 = self.KP * pos_error_1 + self.KD * vel_error_1
                # torque2 = self.KP * pos_error_2 + self.KD * vel_error_2
                
                # 使用放大公式计算力矩
                # τ_s = K_p(b*θ_m - θ_s) + K_d(b*θ_m_dot - θ_s_dot)
                # τ_m = -(1/a) * [K_p(b*θ_m - θ_s) + K_d(b*θ_m_dot - θ_s_dot)]
                pos_error = self.b * pos1 - pos2
                vel_error = self.b * vel1 - vel2
                
                torque_s = self.KP * pos_error + self.KD * vel_error
                torque1 = -(1.0 / self.a) * torque_s  # 主端 motor1
                torque2 = torque_s  # 从端 motor2
                
                
                # pos_e rror_1 = 0.5 * pos2 - pos1
                # pos_error_2 = 2.0 * pos1 - pos2

                # # 速度误差必须跟随位置误差
                # vel_error_1 = 0.5 * vel2 - vel1
                # vel_error_2 = 2.0 * vel1 - vel2

                # # --- 2. 带有正确物理反馈的力矩计算 ---
                # # 定义行程比例
                # scaling_ratio = 2.0

                # # Motor1 (短行程, 更"硬") 使用原始增益
                # torque1 = self.KP * pos_error_1 + self.KD * vel_error_1

                # # Motor2 (长行程, 更"软") 使用按比例缩放后的增益
                # # 这是实现正确物理手感的关键
                # kp2_scaled = self.KP / scaling_ratio
                # kd2_scaled = self.KD / scaling_ratio
                # torque2 = kp2_scaled * pos_error_2 + kd2_scaled * vel_error_2

                # --- 3. 力矩限幅和发送 ---
                torque1 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque1))
                torque2 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque2))

                # torque1 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque1))
                # torque2 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque2))

                # self.motor1.set_torque_nm(torque1)
                # self.motor2.set_torque_nm(torque2)

                # set_torque_thread1 = threading.Thread(target=self.motor1.set_torque_nm, args=(torque1,))
                # set_torque_thread2 = threading.Thread(target=self.motor2.set_torque_nm, args=(torque2,))
                # set_torque_thread1.start()
                # set_torque_thread2.start()
                # set_torque_thread1.join()
                # set_torque_thread2.join()
                
                set_torque_futures = [
                    executor.submit(self.motor1.set_torque_nm, torque1),
                    executor.submit(self.motor2.set_torque_nm, torque2)
                ]
                wait(set_torque_futures)

                elapsed = time.perf_counter() - loop_start
                if elapsed < self.DT:
                    time.sleep(self.DT - elapsed)
                else:
                    logger.info(f"[警告] 循环执行超时! 耗时: {elapsed*1000:.2f}ms, 目标 DT: {self.DT*1000:.0f}ms")
                
                # loop_count += 1
                # current_time = time.perf_counter()
                # if current_time - last_freq_time >= 1.0:
                #     freq = loop_count / (current_time - last_freq_time)
                #     logger.debug(f"Gripper Control Freq: {freq:.2f} Hz")
                #     loop_count = 0
                #     last_freq_time = current_time

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            print("控制器已经启动。"); return
            
        print("启动电机..."); self.motor1.enable(); time.sleep(1)
        if self.mode == 'dual': self.motor2.enable(); time.sleep(1)

        print("设置当前位置为零点..."); self.motor1.set_zero_ram(); time.sleep(2)
        if self.mode == 'dual': self.motor2.set_zero_ram(); time.sleep(2)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("控制器已启动。")

    def stop(self):
        print("正在停止控制循环...")
        if self.thread and self.thread.is_alive():
            self.command_queue.put(('shutdown',))
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                print("[警告] 控制线程未能正常结束。")
                
        if self.mode == 'dual' and self.motor1 and self.motor2:
            motor1_fail_rate = self.motor1.get_failure_percentage()
            motor2_fail_rate = self.motor2.get_failure_percentage()
            logger.info(f"Motor 1 (ID: {self.motor1.motor_id}) refresh failure rate: {motor1_fail_rate:.2f}%")
            logger.info(f"Motor 2 (ID: {self.motor2.motor_id}) refresh failure rate: {motor2_fail_rate:.2f}%")
        
        print("正在关闭电机...")
        try:
            self.motor1.disable()
            if self.mode == 'dual': self.motor2.disable()
        except Exception as e:
            print(f"关闭电机时发生错误: {e}")
        print("控制器已安全停止。")


if __name__ == "__main__":
    
    # --- 示例 1: 双电机模式 ---
    print("\n--- 测试双电机互控模式 ---")
    gripper = GripperController(mode='dual', port1="/dev/ttyUSB0", motor1_id=1, port2="/dev/ttyACM2", motor2_id=2)
    try:
        gripper.start()
        print("双电机同步已启动。可以手动移动一个电机，另一个会跟随。")
        print("按 Ctrl+C 停止。")
        while True:
            pos1, torque1, pos2, torque2 = gripper.get_current_gripper_states()
            print(f"电机1: Pos={pos1:.3f} rad, Torque={torque1:.3f} Nm | 电机2: Pos={pos2:.3f} rad, Torque={torque2:.3f} Nm", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n检测到手动中断...")
    finally:
        gripper.stop()

    # --- 示例 2: 单电机模式 (取消下面的注释来测试) ---
    # print("\n--- 测试单电机控制模式 ---")
    # gripper = GripperController(mode='single', port1="/dev/ttyUSB0", motor1_id=1)
    # try:
    #     gripper.start()
    #     time.sleep(1)

    #     print("\n[任务] 移动到 50mm，力限制 1.2 Nm...")
    #     gripper.move_gripper(width_m=0.06, force_limit_nm=1.2)
        
    #     for _ in range(50):
    #         time.sleep(0.1)
    #         width, force = gripper.get_current_gripper_states()
    #         print(f"状态: 宽度={width*1000:.2f} mm, 力={force:.3f} Nm", end='\r')
    #     print("\n移动任务可能已完成。")

    #     time.sleep(1)
    #     print("\n[任务] 以 1.0 Nm 的力夹持...")
    #     gripper.move_gripper(width_m=0.00, force_limit_nm=1.2)
    #     # gripper.move_gripper_force(force_limit_nm=1.0, close_speed_dps=30)
    #     time.sleep(5)

        # time.sleep(2)
        # print("\n[任务] 完全打开...")
        # gripper.move_gripper(width_m=0.08, force_limit_nm=1.2)
        # time.sleep(3)

    # except KeyboardInterrupt:
    #     print("\n检测到手动中断...")
    # finally:
    #     gripper.stop()

