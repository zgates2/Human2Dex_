import math
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import time
import queue
from motor.motor import LkMotor
from motor.protocol import radian_to_degree
from loguru import logger

class GripperController:
    # MM_PER_SINGLE_MOTOR_RADIAN = 45.0000
    # MM_PER_SINGLE_MOTOR_RADIAN = 30.0000
    # MOTOR_PHYSICAL_LIMIT_RAD = math.pi / 2
    MOTOR_PHYSICAL_LIMIT_M = 0.16
    MOTOR_PHYSICAL_LIMIT_M = 0.20
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

        self.MM_PER_SINGLE_MOTOR_RADIAN = 30.0000
        # self.ratio = 1.1 # 1.2
        # self.ratio = 0.4
        # self.ratio = 0.4
        # self.KP = 2.0 / 2 / 6 * 1.5 * 1.3 * self.ratio
        # # self.KD = 0.01 / 1.5 * 1.5 * self.ratio * 1.0
        # self.KD = 0.01 / 1.5 * 1.5 * self.ratio * 1.0 * 0.5
        self.ratio = 1.2
        # self.ratio = 0.1
        self.KP = 2.0 / 2 / 6 * 1.5 * 1.3 * self.ratio
        # self.KD = 0.01 / 1.5 * 1.5 * self.ratio * 1.0
        self.KD = 0.01 / 1.5 * 1.5 * self.ratio * 1.0
        
        self.TORQUE_LIMIT_DUAL = 2.0 # 2.5
        self.TORQUE_LIMIT_SINGLE = 2.0
        # self.DT = 0.025
        # self.DT = 1 / 21.0
        
        # self.DT = 1 / 20.0
        self.DT = 1 / 120.0
        # self.DT = 1 / 140.0
        # self.DT = 0.01
        
        # 目标位置变化率限制，防止突变导致不稳定
        # 最大变化速度: 3.14 rad/s
        self.target_position_rad_max_delta = 3.14 * self.DT  # ≈ 0.0785 rad per cycle
        
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
        self.current_vel = 0.0
        self.current_torque_nm = 0.0
        self.current_position_rad_2 = 0.0
        self.current_vel_2 = 0.0
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
                self.current_vel,
                self.current_torque_nm,
                self.current_position_rad_2,
                self.current_vel_2,
                self.current_torque_nm_2
            ]
        else:
            # width = self._angle_to_width(self.current_position_rad)
            width = self.current_position_rad
            vel = self.current_vel
            force = self.current_torque_nm
            return [width, vel, force, 0, 0, 0]
    
    def move_gripper(self, width_m: float, force_limit_nm: float):
        if self.mode != 'single':
            print("[错误] move_gripper 只能在 'single' 模式下使用。")
            return
        # target_rad = self._width_to_angle(width_m)
        target_rad = width_m
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
        单电机控制循环，实现"位置保持"功能。
        """
        print("开始单电机控制循环 (位置保持模式)...")
        running = True

        control_state = 'IDLE' 
        
        # 期望的目标位置（可能突变）
        desired_position_rad = 0.0
        # 实际使用的目标位置（经过变化率限制平滑处理）
        actual_target_position_rad = 0.0

        while running:
            loop_start = time.perf_counter()

            try:
                command = self.command_queue.get_nowait()
                cmd_type = command[0]

                if cmd_type == 'shutdown':
                    running = False
                    continue
                
                elif cmd_type == 'move':
                    _, desired_position_rad, force_limit_nm = command
                    control_state = 'POSITION_MODE' 

            except queue.Empty:
                pass
            
            if control_state == 'POSITION_MODE':
                # 限制目标位置变化率，防止突变导致不稳定
                position_diff = desired_position_rad - actual_target_position_rad
                if abs(position_diff) > self.target_position_rad_max_delta:
                    actual_target_position_rad += math.copysign(self.target_position_rad_max_delta, position_diff)
                else:
                    actual_target_position_rad = desired_position_rad
                
                self.motor1.refresh()

                if getattr(self.motor1, 'refresh_ok', True) is False:
                    print("[警告] 电机数据无效，跳过此次循环。")
                    self.precise_sleep(self.DT)
                    continue

                pos_rad, vel_rad_s, torque_nm = self.motor1.getPosition(), self.motor1.getVelocity(), self.motor1.getTorque()
                
                self.current_position_rad = pos_rad
                self.vel_rad_s = vel_rad_s
                self.current_torque_nm = torque_nm
                
                # 使用经过平滑处理的目标位置
                pos_error = actual_target_position_rad - pos_rad
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
                self.current_vel = vel1
                self.current_torque_nm = torque1_val
                self.current_position_rad_2 = pos2
                self.current_vel_2 = vel2
                self.current_torque_nm_2 = torque2_val
                    
                pos_error_1, vel_error_1 = pos2 - pos1, vel2 - vel1
                pos_error_2, vel_error_2 = pos1 - pos2, vel1 - vel2

                torque1 = self.KP * pos_error_1 + self.KD * vel_error_1
                torque2 = self.KP * pos_error_2 + self.KD * vel_error_2

                torque1 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque1))
                torque2 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque2))

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

    def start(self, auto_home=True, home_torque_nm=-0.05, home_timeout_s=2.0):
        if self.thread is not None and self.thread.is_alive():
            print("控制器已经启动。"); return
            
        print("启动电机..."); self.motor1.enable(); time.sleep(1)
        if self.mode == 'dual': self.motor2.enable(); time.sleep(1)

        if auto_home:
            print(f"MIT力控回零: 施加 {home_torque_nm:.2f} Nm 力矩...")
            start_time = time.time()
            stall_counter = 0
            stall_threshold = 10  # 连续10次检测到低速才认为堵转
            
            while (time.time() - start_time) < home_timeout_s:
                # 串行刷新电机状态，避免通信冲突
                self.motor1.refresh()
                motor1_ok = getattr(self.motor1, 'refresh_ok', False)
                
                if self.mode == 'dual':
                    time.sleep(0.005)  # 电机间延时5ms，避免串口冲突
                    self.motor2.refresh()
                    motor2_ok = getattr(self.motor2, 'refresh_ok', False)
                    
                    if not motor1_ok or not motor2_ok:
                        # print(f"[警告] 电机数据刷新失败 (motor1: {motor1_ok}, motor2: {motor2_ok})")
                        time.sleep(0.01)
                        continue
                else:
                    if not motor1_ok:
                        # print("[警告] 电机1数据刷新失败")
                        time.sleep(0.01)
                        continue
                
                vel1 = self.motor1.getVelocity()
                
                # 检查速度是否有效
                if vel1 is None:
                    time.sleep(0.01)
                    continue
                
                # 施加恒定力矩
                self.motor1.set_torque_nm(home_torque_nm)
                if self.mode == 'dual':
                    vel2 = self.motor2.getVelocity()
                    if vel2 is None:
                        time.sleep(0.01)
                        continue
                    time.sleep(0.005)  # 发送力矩命令间延时
                    self.motor2.set_torque_nm(home_torque_nm)
                    # 双电机都需要检测堵转
                    if abs(vel1) < 0.05 and abs(vel2) < 0.05:
                        stall_counter += 1
                    else:
                        stall_counter = 0
                else:
                    # 单电机检测堵转
                    if abs(vel1) < 0.05:
                        stall_counter += 1
                    else:
                        stall_counter = 0
                
                # 连续检测到堵转，认为到达机械限位
                if stall_counter >= stall_threshold:
                    print("检测到机械限位，回零完成")
                    break
                
                time.sleep(0.01)  # 100Hz刷新率
            
            # 停止施加力矩
            self.motor1.set_torque_nm(0)
            if self.mode == 'dual':
                self.motor2.set_torque_nm(0)
            time.sleep(0.5)
            
            if (time.time() - start_time) >= home_timeout_s:
                print("[警告] 回零超时，可能未到达机械限位")

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
    # print("\n--- 测试双电机互控模式 ---")
    # # gripper = GripperController(mode='dual', port1="/dev/ttyUSB0", motor1_id=1, port2="/dev/ttyACM2", motor2_id=2)
    # gripper = GripperController(mode='dual', port1="/dev/ttyUSB1", motor1_id=2, port2="/dev/ttyUSB2", motor2_id=1)
    # try:
    #     gripper.start()
    #     print("双电机同步已启动。可以手动移动一个电机，另一个会跟随。")
    #     print("按 Ctrl+C 停止。")
    #     while True:
    #         pos1, torque1, pos2, torque2 = gripper.get_current_gripper_states()
    #         print(f"电机1: Pos={pos1:.3f} rad, Torque={torque1:.3f} Nm| 电机2: Pos={pos2:.3f} rad, Torque={torque2:.3f} Nm", end='\r')
    #         time.sleep(0.1)
    # except KeyboardInterrupt:
    #     print("\n检测到手动中断...")
    # finally:
    #     gripper.stop()

    # --- 示例 2: 单电机模式 (取消下面的注释来测试) ---
    print("\n--- 测试单电机控制模式 ---")
    # gripper = GripperController(mode='single', port1="/dev/ttyUSB2", motor1_id=2)
    # gripper = GripperController(mode='single', port1="/dev/ttyACM2", motor1_id=2)
    gripper = GripperController(mode='single', port1="/dev/ttyUSB2", motor1_id=1)
    
    try:
        gripper.start()
        time.sleep(1)

        print("\n[任务] 移动到 50mm，力限制 1.2 Nm...")
        gripper.move_gripper(width_m=3.0, force_limit_nm=1.5)
        
        for _ in range(10000):
            time.sleep( 1 / 10.0)
            width, _, force, _, _, _ = gripper.get_current_gripper_states()
            print(f"状态: 弧度={width:.4f} rad, 力={force:.3f} Nm", end='\r')
        print("\n移动任务可能已完成。")

        time.sleep(1)
        print("\n[任务] 以 1.0 Nm 的力夹持...")
        # gripper.move_gripper(width_m=0.00, force_limit_nm=1.2)
        # gripper.move_gripper_force(force_limit_nm=1.0, close_speed_dps=30)
        time.sleep(5)

        # time.sleep(2)
        # print("\n[任务] 完全打开...")
        # gripper.move_gripper(width_m=0.08, force_limit_nm=1.2)
        # time.sleep(3)

    except KeyboardInterrupt:
        print("\n检测到手动中断...")
    finally:
        gripper.stop()

