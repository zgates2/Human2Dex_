import math
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import time
import queue
from motor.motor import LkMotor
from motor.protocol import radian_to_degree
from loguru import logger


class GripperControllerNoFeedback:
    """
    不带力反馈的双边遥操作夹爪控制器

    motor1: 主端(Master) - 操作员控制端，只提供轻微阻尼，不施加跟随力
    motor2: 从端(Slave) - 机器人端，跟随主端位置
    """
    MM_PER_SINGLE_MOTOR_RADIAN = 45.0000
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

        print(f"控制器以 '{self.mode}' 模式初始化（无力反馈）...")
        self.motor1 = LkMotor(port1, motor_id=motor1_id)
        self.motor2 = None
        if self.mode == 'dual':
            if not port2 or not motor2_id:
                raise ValueError("Port2 and motor2_id are required for 'dual' mode")
            self.motor2 = LkMotor(port2, motor_id=motor2_id)
            print("双电机模式: 单向遥操作（motor1=主端，motor2=从端）")

        self.ratio = 1.1 # 1.2
        self.KP = 2.0 / 2 / 6 * 1.5 * 1.3 * self.ratio
        self.KD = 0.01 / 1.5 * 1.5 * self.ratio * 1.0
        self.TORQUE_LIMIT_DUAL = 2.0 # 2.5
        self.TORQUE_LIMIT_SINGLE = 2.0
        self.DT = 0.025

        # 主端（无力反馈模式）的参数
        self.KD_MASTER = 0.005  # 主端阻尼系数，提供轻微的阻尼感
        self.TORQUE_LIMIT_MASTER = 0.3  # 主端最大力矩限制，保持手感轻便

        # single motor control
        self.MIN_SPEED_DPS = 8.0
        self.MAX_ACCEL_DPS2 = 800.0
        self.VELOCITY_P_GAIN = 120.0
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
            print("[错误] move_gripper_force 只能��� 'single' 模式下使用。")
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
        """
        双电机单向遥操作（无力反馈）
        - motor1 (主端): 只施加轻微阻尼，操作员可以自由移动
        - motor2 (从端): 跟随motor1的位置
        """
        print("开始双电机单向遥操作（无力反馈）...")
        running = True

        with ThreadPoolExecutor(max_workers=4) as executor:
            while running:
                loop_start = time.perf_counter()

                try:
                    command = self.command_queue.get_nowait()
                    if command[0] == 'shutdown': running = False; continue
                except queue.Empty:
                    pass

                refresh_futures = [
                    executor.submit(self.motor1.refresh),
                    executor.submit(self.motor2.refresh)
                ]

                wait(refresh_futures)

                if (self.motor1.refresh_ok is False) or (self.motor2.refresh_ok is False):
                    time.sleep(self.DT); continue

                pos1, vel1, torque1_val = self.motor1.getPosition(), self.motor1.getVelocity(), self.motor1.getTorque()
                pos2, vel2, torque2_val = self.motor2.getPosition(), self.motor2.getVelocity(), self.motor2.getTorque()

                self.current_position_rad = pos1
                self.current_torque_nm = torque1_val
                self.current_position_rad_2 = pos2
                self.current_torque_nm_2 = torque2_val

                # 主端(motor1): 只施加轻微阻尼力矩，让操作更顺滑
                # 不施加位置跟随力，所以操作员不会感受到从端的阻力
                torque1 = -self.KD_MASTER * vel1
                torque1 = max(-self.TORQUE_LIMIT_MASTER, min(self.TORQUE_LIMIT_MASTER, torque1))

                # 从端(motor2): 跟随主端位置
                pos_error_2 = pos1 - pos2
                vel_error_2 = vel1 - vel2
                torque2 = self.KP * pos_error_2 + self.KD * vel_error_2
                torque2 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque2))

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

    # --- 双电机模式（无力反馈） ---
    print("\n--- 测试双电机单向遥操作模式（无力反馈） ---")
    gripper = GripperControllerNoFeedback(mode='dual', port1="/dev/ttyUSB1", motor1_id=1, port2="/dev/ttyUSB2", motor2_id=2)
    try:
        gripper.start()
        print("单向遥操作已启动。移动主端(motor1)，从端(motor2)会跟随，但主端不会感受到从端的力。")
        print("按 Ctrl+C 停止。")
        while True:
            pos1, torque1, pos2, torque2 = gripper.get_current_gripper_states()
            print(f"主端: Pos={pos1:.3f} rad, Torque={torque1:.3f} Nm | 从端: Pos={pos2:.3f} rad, Torque={torque2:.3f} Nm", end='\r')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n检测到手动中断...")
    finally:
        gripper.stop()
