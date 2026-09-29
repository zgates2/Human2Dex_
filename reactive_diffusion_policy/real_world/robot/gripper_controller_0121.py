import sys
sys.path.insert(0, "/home/zjc/work/LK_MOTOR")

import math
import threading
from concurrent.futures import ThreadPoolExecutor, wait
import time
import queue
from motor.motor import LkMotor
from motor.protocol import radian_to_degree
from loguru import logger

class GripperController:
    MOTOR_PHYSICAL_LIMIT_M = 0.20
    POSITION_REACHED_THRESHOLD_RAD = 0.02

    SOFT_TORQUE_THRESHOLD_RATIO = 0.8
    BACK_OFF_STEP_RAD = 0.01

    VELOCITY_P_GAIN = 200.0
    MAX_SPEED_DPS = 60.0

    def __init__(self, mode: str = 'dual',
                 port: str = None, motor_id: int = None,
                 port1: str = None, motor1_id: int = None,
                 port2: str = None, motor2_id: int = None,
                 **kwargs):
        """
        初始化夹爪控制器

        Args:
            mode: 控制模式，'single' 或 'dual'
            port/motor_id: 单电机模式的端口和ID
            port1/motor1_id: 电机1的端口和ID
            port2/motor2_id: 电机2的端口和ID（双电机模式必需）
        """
        self.mode = mode

        if self.mode not in ['dual', 'single']:
            raise ValueError("Mode must be 'dual' or 'single'")

        # 兼容新旧参数格式
        if port1 is None and port is not None:
            port1 = port
        if motor1_id is None and motor_id is not None:
            motor1_id = motor_id

        if port1 is None or motor1_id is None:
            raise ValueError("必须提供 port1/motor1_id 或 port/motor_id 参数")

        print(f"控制器以 '{self.mode}' 模式初始化...")
        self.motor1 = LkMotor(port1, motor_id=motor1_id)
        self.motor2 = None
        if self.mode == 'dual':
            if not port2 or not motor2_id:
                raise ValueError("双电机模式必须提供 port2 和 motor2_id")
            self.motor2 = LkMotor(port2, motor_id=motor2_id)

        self.MM_PER_RAD = 30.59  # 校准值: 3.4rad=104mm → 104/3.4≈30.59
        self.ratio = 1.2

        # MIT阻抗控制参数
        self.KP = 0.7    # 位置刚度 [Nm/rad]
        self.KD = 0.1   # 速度阻尼 [Nm·s/rad]
        self.K_TRANS  = 0.1  # 力透明度系数（0=无透传,)

        self.TORQUE_LIMIT = 2.0       # 单电机最大力矩限制 [Nm]
        self.TORQUE_LIMIT_DUAL = 2.0  # 双电机最大力矩限制 [Nm]

        # 主端自动打开引导
        self.VIRTUAL_FIXTURE_ENABLED = True
        self.VIRTUAL_FIXTURE_TARGET_RAD = 2.5       # 目标位置 [rad]
        self.VIRTUAL_FIXTURE_KP = 0.15              # 刚度
        self.VIRTUAL_FIXTURE_MAX_TORQUE = 0.0      # 最大引导力矩 [Nm]

        # 电机1相对电机2的角度偏置（正值表示电机1比电机2多转该角度）
        # 例如：设为 0.1 时，电机2在 0~1 rad，电机1在 0.1~1.1 rad
        self.MOTOR1_BIAS_RAD = 0

        # 夹持力增强：目标位置偏移量
        self.GRIPPER_CLOSE_OFFSET_RAD = 0.0
        self.DT = 1 / 120.0

        # 目标位置变化率限制，防止突变导致不稳定
        self.target_position_rad_max_delta = 3.14 * self.DT

        # 扭矩低通滤波状态（单电机）
        self._torque_filt = 0.0
        self._torque_lpf_alpha = 0.3

        # 双电机力透传低通滤波状态
        self._torque1_filt = 0.0  # 电机1测量扭矩滤波值
        self._torque2_filt = 0.0  # 电机2测量扭矩滤波值
        self._trans_lpf_alpha = 0.3  # 透传滤波系数

        # 四通道控制：干扰观测器（DOB）参数
        self.J1 = 0              # 电机1等效转动惯量 [kg·m²]
        self.J2 = 0              # 电机2等效转动惯量 [kg·m²]
        self.torque_deadzone = 0.04   # 环境力矩死区 [Nm] (原来0.04，增大到0.08抑制低频正反馈和底噪)
        self.prev_vel1 = 0.0       # 上一周期电机1速度，用于差分计算加速度
        self.prev_vel2 = 0.0       # 上一周期电机2速度，用于差分计算加速度

        self._acc1_filt = 0.0
        self._acc2_filt = 0.0
        self._acc_lpf_alpha = 0  # 加速度滤波系数，加速度噪声最大，需较强滤波

        self.thread = None
        self.command_queue = queue.Queue()

        # 电机1状态
        self.current_position_rad = 0.0
        self.current_vel = 0.0
        self.current_torque_nm = 0.0
        # 电机2状态（双电机模式）
        self.current_position_rad_2 = 0.0
        self.current_vel_2 = 0.0
        self.current_torque_nm_2 = 0.0

        self.busy_wait_threshold = 0.0015

    def _angle_to_width(self, angle_rad: float) -> float:
        """弧度 -> 开合宽度(m)，校准: 3.4rad=104mm"""
        width_mm = angle_rad * self.MM_PER_RAD
        return width_mm / 1000.0

    def _width_to_angle(self, width_m: float) -> float:
        """开合宽度(m) -> 弧度，校准: 104mm=3.4rad"""
        limited_width_m = min(width_m, self.MOTOR_PHYSICAL_LIMIT_M)
        width_mm = limited_width_m * 1000.0
        angle_rad = width_mm / self.MM_PER_RAD
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
                self.current_vel,
                self.current_torque_nm,
                self.current_position_rad_2,
                self.current_vel_2,
                self.current_torque_nm_2
            ]
        else:
            return [
                self.current_position_rad,
                self.current_vel,
                self.current_torque_nm,
                0, 0, 0
            ]

    def move_gripper(self, width_m: float, force_limit_nm: float, feedforward_torque_nm: float = 0.0):
        """
        移动夹爪到目标位置（单电机模式）

        Args:
            width_m: 目标位置 (rad)
            force_limit_nm: 最大允许力矩 (Nm)
            feedforward_torque_nm: 前馈力矩 (Nm)，默认为0
        """
        if self.mode != 'single':
            print("[警告] move_gripper 主要用于单电机模式，双电机模式下无效")
            return
        target_rad = width_m
        self.command_queue.put(('move', target_rad, abs(force_limit_nm), feedforward_torque_nm))
        if feedforward_torque_nm != 0.0:
            logger.info(f"指令已发送: 移动到 {target_rad:.3f} rad, 力限制: {abs(force_limit_nm):.2f} Nm, 前馈力矩: {feedforward_torque_nm:.2f} Nm")
        else:
            logger.info(f"指令已发送: 移动到 {target_rad:.3f} rad, 力限制: {abs(force_limit_nm):.2f} Nm")

    def move_gripper_force(self, force_limit_nm: float, close_speed_dps: float = 30.0):
        if self.mode != 'single':
            print("[错误] move_gripper_force 只能在 'single' 模式下使用。")
            return
        self.command_queue.put(('grasp', abs(force_limit_nm), abs(close_speed_dps)))
        print(f"指令已发送: 夹持至力矩达到 {abs(force_limit_nm):.2f} Nm")

    def stop_gripper(self):
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
        单电机控制循环，MIT阻抗控制 + 前馈力矩
        """
        print("开始单电机控制循环 (MIT阻抗控制)...")
        running = True

        control_state = 'IDLE'

        desired_position_rad = 0.0
        actual_target_position_rad = 0.0
        feedforward_torque_nm = 0.0
        current_force_limit_nm = self.TORQUE_LIMIT

        while running:
            loop_start = time.perf_counter()

            try:
                command = self.command_queue.get_nowait()
                cmd_type = command[0]

                if cmd_type == 'shutdown':
                    running = False
                    continue

                elif cmd_type == 'move':
                    _, desired_position_rad, force_limit_nm, feedforward_torque = command
                    current_force_limit_nm = force_limit_nm
                    feedforward_torque_nm = feedforward_torque
                    control_state = 'POSITION_MODE'

            except queue.Empty:
                pass

            if control_state == 'POSITION_MODE':
                # 应用夹持偏移
                adjusted_desired_position = desired_position_rad - self.GRIPPER_CLOSE_OFFSET_RAD

                # 限制目标位置变化率
                position_diff = adjusted_desired_position - actual_target_position_rad
                if abs(position_diff) > self.target_position_rad_max_delta:
                    actual_target_position_rad += math.copysign(self.target_position_rad_max_delta, position_diff)
                else:
                    actual_target_position_rad = adjusted_desired_position

                self.motor1.refresh()

                if getattr(self.motor1, 'refresh_ok', True) is False:
                    print("[警告] 电机数据无效，跳过此次循环。")
                    self.precise_sleep(self.DT)
                    continue

                pos_rad = self.motor1.getPosition()
                vel_rad_s = self.motor1.getVelocity()
                torque_nm = self.motor1.getTorque()

                self.current_position_rad = pos_rad
                self.current_vel = vel_rad_s
                self.current_torque_nm = torque_nm

                # MIT阻抗控制: τ = τff + Kp*(θd-θ) + Kd*(θ̇d-θ̇)
                pos_error = actual_target_position_rad - pos_rad
                vel_error = 0.0 - vel_rad_s

                torque_cmd = feedforward_torque_nm + self.KP * pos_error + self.KD * vel_error
                torque_cmd = max(-current_force_limit_nm, min(current_force_limit_nm, torque_cmd))

                self.motor1.set_torque_nm(torque_cmd)

            elapsed = time.perf_counter() - loop_start
            remaining_time = self.DT - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time)

        self.motor1.stop()
        print("单电机控制循环已停止。")

    def _dual_control_loop(self):
        """
        双电机互控循环，MIT阻抗控制
        电机1的目标是电机2的位置，反之亦然
        """
        print("开始双电机互控循环 (MIT阻抗控制)...")
        running = True

        with ThreadPoolExecutor(max_workers=4) as executor:
            while running:
                loop_start = time.perf_counter()

                try:
                    command = self.command_queue.get_nowait()
                    if command[0] == 'shutdown':
                        running = False
                        continue
                except queue.Empty:
                    pass

                # 并行刷新电机状态
                refresh_futures = [
                    executor.submit(self.motor1.refresh),
                    executor.submit(self.motor2.refresh)
                ]
                wait(refresh_futures)

                if (self.motor1.refresh_ok is False) or (self.motor2.refresh_ok is False):
                    time.sleep(self.DT)
                    continue

                pos1 = self.motor1.getPosition()
                vel1 = self.motor1.getVelocity()
                torque1_val = self.motor1.getTorque()
                # status1  = self.motor1.read_status_2()
                # iq_raw   = status2.get("iq_or_power", 0)
                # iq_read  = iq_raw / 2048.0 * 33.0          # 单位 A，直接硬件值
                # actual_torque_nm_1 = iq_to_torque_nm(iq_read)

                pos2 = self.motor2.getPosition()
                vel2 = self.motor2.getVelocity()
                torque2_val = self.motor2.getTorque()
                # status2  = self.motor2.read_status_2()
                # iq_raw   = status2.get("iq_or_power", 0)
                # iq_read  = iq_raw / 2048.0 * 33.0          # 单位 A，直接硬件值
                # actual_torque_nm_2 = iq_to_torque_nm(iq_read)

                # 更新状态
                self.current_position_rad = pos1
                self.current_vel = vel1
                self.current_torque_nm = torque1_val
                self.current_position_rad_2 = pos2
                self.current_vel_2 = vel2
                self.current_torque_nm_2 = torque2_val

                # -------------------------
                # 扭矩低通滤波，抑制电流传感器噪声
                # -------------------------
                self._torque1_filt = (self._trans_lpf_alpha * torque1_val
                                      + (1 - self._trans_lpf_alpha) * self._torque1_filt)
                self._torque2_filt = (self._trans_lpf_alpha * torque2_val
                                      + (1 - self._trans_lpf_alpha) * self._torque2_filt)

                # -------------------------
                # 加速度估计与滤波（用于惯性解耦）
                # -------------------------
                acc1 = (vel1 - self.prev_vel1) / self.DT
                acc2 = (vel2 - self.prev_vel2) / self.DT
                self.prev_vel1 = vel1
                self.prev_vel2 = vel2

                self._acc1_filt = (self._acc_lpf_alpha * acc1 + (1 - self._acc_lpf_alpha) * self._acc1_filt)
                self._acc2_filt = (self._acc_lpf_alpha * acc2 + (1 - self._acc_lpf_alpha) * self._acc2_filt)

                # -------------------------
                # 环境力矩观测：放弃加速度补偿，直接使用低通滤波后的力矩，避免速度差分带来严重噪声
                # -------------------------
                tau_env1 = self._torque1_filt
                tau_env2 = self._torque2_filt

                # 死区过滤：消除静摩擦和传感器底噪引起的微震荡
                if abs(tau_env1) < self.torque_deadzone:
                    tau_env1 = 0.0
                if abs(tau_env2) < self.torque_deadzone:
                    tau_env2 = 0.0

                # -------------------------
                # 位置耦合误差与速度误差计算及滤波
                # -------------------------
                bias = self.MOTOR1_BIAS_RAD
                pos_error_1 = (pos2 + bias) - pos1
                pos_error_2 = (pos1 - bias) - pos2

                vel_error_1 = vel2 - vel1
                vel_error_2 = vel1 - vel2

                # -------------------------
                # 位置-力 (P-F) 控制架构：位置跟踪 + 纯环境力反向透传
                # -------------------------
                # 1. 从端保持阻抗/位置追踪，紧跟主端位置（不再受主端力的干扰）
                torque2 = (self.KP * pos_error_2 + self.KD * vel_error_2)

                # 2. 主端取消位置追踪（去掉了虚拟弹簧），只进行环境力透传
                # 加入微小的本地阻尼 (LOCAL_DAMPING) 以压制失去位置约束后的高频震荡
                LOCAL_DAMPING = 0.003
                # torque1 =  -self.K_TRANS * tau_env2   + LOCAL_DAMPING * vel1
                torque1 = LOCAL_DAMPING * vel1

                # 在主端(motor1)叠加一个向打开方向的引导力
                if self.VIRTUAL_FIXTURE_ENABLED:
                    fixture_error = self.VIRTUAL_FIXTURE_TARGET_RAD - pos1
                    fixture_torque = self.VIRTUAL_FIXTURE_KP * fixture_error
                    # clip到最大引导力矩范围
                    fixture_torque = max(-self.VIRTUAL_FIXTURE_MAX_TORQUE,
                                        min(self.VIRTUAL_FIXTURE_MAX_TORQUE, fixture_torque))
                    torque1 += fixture_torque

                torque1 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque1))
                torque2 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque2))

                # 并行发送力矩命令
                set_torque_futures = [
                    executor.submit(self.motor1.set_torque_nm, torque1),
                    executor.submit(self.motor2.set_torque_nm, torque2)
                ]
                wait(set_torque_futures)

                elapsed = time.perf_counter() - loop_start
                if elapsed < self.DT:
                    time.sleep(self.DT - elapsed)
                else:
                    logger.debug(f"[警告] 循环执行超时! 耗时: {elapsed*1000:.2f}ms, 目标: {self.DT*1000:.0f}ms")

        self.motor1.stop()
        self.motor2.stop()
        print("双电机控制循环已停止。")

    def start(self, auto_home=True, home_torque_nm=-0.05, home_timeout_s=2.0):
        """
        启动夹爪控制器

        Args:
            auto_home: 是否自动回零（默认True）
            home_torque_nm: 回零力矩，负值表示关闭方向（默认-0.05 Nm）
            home_timeout_s: 回零超时时间（默认2秒）
        """
        if self.thread is not None and self.thread.is_alive():
            print("控制器已经启动。")
            return

        print("启动电机...")
        self.motor1.enable()
        time.sleep(1)
        if self.mode == 'dual':
            self.motor2.enable()
            time.sleep(1)

        if auto_home:
            print(f"MIT力控回零: 施加 {home_torque_nm:.2f} Nm 力矩...")
            start_time = time.time()
            stall_counter = 0
            stall_threshold = 10

            while (time.time() - start_time) < home_timeout_s:
                self.motor1.refresh()
                motor1_ok = getattr(self.motor1, 'refresh_ok', False)

                if self.mode == 'dual':
                    time.sleep(0.005)
                    self.motor2.refresh()
                    motor2_ok = getattr(self.motor2, 'refresh_ok', False)

                    if not motor1_ok or not motor2_ok:
                        time.sleep(0.01)
                        continue
                else:
                    if not motor1_ok:
                        time.sleep(0.01)
                        continue

                vel1 = self.motor1.getVelocity()
                if vel1 is None:
                    time.sleep(0.01)
                    continue

                self.motor1.set_torque_nm(home_torque_nm)
                if self.mode == 'dual':
                    vel2 = self.motor2.getVelocity()
                    if vel2 is None:
                        time.sleep(0.01)
                        continue
                    time.sleep(0.005)
                    self.motor2.set_torque_nm(home_torque_nm)
                    if abs(vel1) < 0.05 and abs(vel2) < 0.05:
                        stall_counter += 1
                    else:
                        stall_counter = 0
                else:
                    if abs(vel1) < 0.05:
                        stall_counter += 1
                    else:
                        stall_counter = 0

                if stall_counter >= stall_threshold:
                    print("检测到机械限位，回零完成")
                    break

                time.sleep(0.01)

            self.motor1.set_torque_nm(0)
            if self.mode == 'dual':
                self.motor2.set_torque_nm(0)
            time.sleep(0.5)

            if (time.time() - start_time) >= home_timeout_s:
                print("[警告] 回零超时，可能未到达机械限位")

        print("设置当前位置为零点...")
        self.motor1.set_zero_ram()
        time.sleep(2)
        if self.mode == 'dual':
            self.motor2.set_zero_ram()
            time.sleep(2)

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

        print("正在关闭电机...")
        try:
            self.motor1.disable()
            if self.mode == 'dual' and self.motor2:
                self.motor2.disable()
        except Exception as e:
            print(f"关闭电机时发生错误: {e}")
        print("控制器已安全停止。")


if __name__ == "__main__":
    import csv
    from datetime import datetime

    # ========== 选择测试模式 ==========
    TEST_MODE = 'dual'  # 'single' 或 'dual'

    if TEST_MODE == 'dual':
        # ==================== 双电机互控测试====================
        print("\n===（双电机互控）===")

        PORT1 = "/dev/ttyUSB1"
        MOTOR1_ID = 1
        PORT2 = "/dev/ttyUSB2"
        MOTOR2_ID = 1
        LOG_FILE = f"mit_dual_fixture_{datetime.now().strftime('%H%M%S')}.csv"

        gripper = GripperController(
            mode='dual',
            port1=PORT1, motor1_id=MOTOR1_ID,
            port2=PORT2, motor2_id=MOTOR2_ID
        )
        data_log = []

        # 限制目标弧度
        fixture_target_rad = gripper.VIRTUAL_FIXTURE_TARGET_RAD
        fixture_target_width_cm = gripper._angle_to_width(fixture_target_rad) * 100

        try:
            gripper.start()
            print(f"\n双电机同步已启动")
            print(f"  MIT参数: Kp={gripper.KP}, Kd={gripper.KD}")
            print(f"  采集位置限制: {'开启' if gripper.VIRTUAL_FIXTURE_ENABLED else '关闭'}")
            print(f"    目标位置: {fixture_target_rad:.2f} rad (约 {fixture_target_width_cm:.1f} cm)")
            print(f"    引导刚度: {gripper.VIRTUAL_FIXTURE_KP}")
            print(f"    最大力矩: {gripper.VIRTUAL_FIXTURE_MAX_TORQUE} Nm")
            print("按 Ctrl+C 停止。\n")

            t_start = time.time()
            while True:
                t = time.time() - t_start
                pos1, vel1, torque1, pos2, vel2, torque2 = gripper.get_current_gripper_states()

                # 计算当前宽度
                width1_cm = gripper._angle_to_width(pos1) * 100
                width2_cm = gripper._angle_to_width(pos2) * 100

                data_log.append({
                    't': t,
                    'pos1': pos1, 'width1_cm': width1_cm, 'torque1': torque1,
                    'pos2': pos2, 'width2_cm': width2_cm, 'torque2': torque2,
                    'fixture_target_rad': fixture_target_rad
                })

                print(f"主端: {width1_cm:5.1f}cm ({pos1:+.2f}rad) τ={torque1:+.3f}Nm | "
                      f"从端: {width2_cm:5.1f}cm ({pos2:+.2f}rad) τ={torque2:+.3f}Nm | "
                      f"目标: {fixture_target_width_cm:.0f}cm", end='\r')
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n检测到手动中断...")

            # 保存数据
            if data_log:
                with open(LOG_FILE, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['t','pos1','width1_cm','torque1','pos2','width2_cm','torque2','fixture_target_rad'])
                    writer.writeheader()
                    writer.writerows(data_log)
                print(f"数据已保存至: {LOG_FILE}")
        finally:
            gripper.stop()

    else:
        # ==================== 单电机MIT控制测试 ====================
        print("\n=== MIT阻抗控制验证测试（单电机）===")

        PORT = "/dev/ttyUSB2"
        MOTOR_ID = 1
        LOG_FILE = f"mit_single_test_{datetime.now().strftime('%H%M%S')}.csv"

        gripper = GripperController(mode='single', port=PORT, motor_id=MOTOR_ID)
        data_log = []

        try:
            gripper.start()
            time.sleep(1)

            # ========== 测试1: 纯位置跟踪（验证Kp） ==========
            print("\n[测试1] 位置阶跃响应 (0 -> 2.0 rad)")
            print(f"  Kp={gripper.KP}, Kd={gripper.KD}")

            target_rad = 2.0
            gripper.move_gripper(width_m=target_rad, force_limit_nm=1.5, feedforward_torque_nm=0.0)

            t_start = time.time()
            for i in range(100):
                time.sleep(0.1)
                t = time.time() - t_start
                pos, vel, torque, _, _, _ = gripper.get_current_gripper_states()
                error = target_rad - pos

                data_log.append({
                    'test': 1, 't': t, 'target': target_rad,
                    'pos': pos, 'vel': vel, 'torque': torque, 'error': error
                })
                print(f"  t={t:.1f}s | pos={pos:.3f} | err={error:.3f} | τ={torque:.3f} Nm")

            # ========== 测试2: 位置保持刚度（验证阻抗行为）==========
            print("\n[测试2] 位置保持 - 请手动施加外力干扰...")
            print("  观察：电机应产生抵抗力矩")

            for i in range(50):
                time.sleep(0.1)
                t = time.time() - t_start
                pos, vel, torque, _, _, _ = gripper.get_current_gripper_states()
                error = target_rad - pos

                data_log.append({
                    'test': 2, 't': t, 'target': target_rad,
                    'pos': pos, 'vel': vel, 'torque': torque, 'error': error
                })
                print(f"  pos={pos:.3f} | err={error:.3f} | τ={torque:.3f} Nm", end='\r')
            print()

            # ========== 测试3: 回零 ==========
            print("\n[测试3] 回零 (-> 0 rad)")
            target_rad = 0.0
            gripper.move_gripper(width_m=target_rad, force_limit_nm=1.5, feedforward_torque_nm=0.0)

            for i in range(50):
                time.sleep(0.1)
                t = time.time() - t_start
                pos, vel, torque, _, _, _ = gripper.get_current_gripper_states()
                error = target_rad - pos

                data_log.append({
                    'test': 3, 't': t, 'target': target_rad,
                    'pos': pos, 'vel': vel, 'torque': torque, 'error': error
                })
                print(f"  t={t:.1f}s | pos={pos:.3f} | err={error:.3f} | τ={torque:.3f} Nm")

            # 保存数据
            with open(LOG_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['test','t','target','pos','vel','torque','error'])
                writer.writeheader()
                writer.writerows(data_log)
            print(f"\n数据已保存至: {LOG_FILE}")

        except KeyboardInterrupt:
            print("\n手动中断...")
        finally:
            gripper.stop()
