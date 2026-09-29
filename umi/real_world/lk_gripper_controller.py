"""LkMotor 6-axis impedance gripper controller, ported from omniumi.drivers.gripper.controller.

upstream 行为说明：
- ``move_gripper(width_m, force_limit_nm)`` 的 ``width_m`` 参数名是 “米” 但实现里
  直接当 rad 用 ( ``target_rad = width_m`` )。本文件保留 upstream 行为，调用方
  应当显式做 米/rad 转换（用 ``_width_to_angle``）再传入；
  ``LkGripperProxy`` 的 ``schedule_waypoint`` 会在边界做这个转换。
- ``motor.motor.LkMotor`` 是供应商 SDK，pip 单独装。本文件在 ``__init__`` 里
  lazy import，让没装 SDK 的环境（如 mac 开发机）仍能 import 这个模块本身。
"""
import logging
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

logger = logging.getLogger(__name__)


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
        try:
            from motor.motor import LkMotor
        except ImportError as e:
            raise RuntimeError(
                "motor.motor.LkMotor SDK not installed; required for GripperController. "
                "Install vendor SDK on the deployment host."
            ) from e

        self.mode = mode

        if self.mode not in ['dual', 'single']:
            raise ValueError("Mode must be 'dual' or 'single'")

        if port1 is None and port is not None:
            port1 = port
        if motor1_id is None and motor_id is not None:
            motor1_id = motor_id

        if port1 is None or motor1_id is None:
            raise ValueError("必须提供 port1/motor1_id 或 port/motor_id 参数")

        logger.info("控制器以 '%s' 模式初始化...", self.mode)
        self.motor1 = LkMotor(port1, motor_id=motor1_id)
        self.motor2 = None
        if self.mode == 'dual':
            if not port2 or not motor2_id:
                raise ValueError("双电机模式必须提供 port2 和 motor2_id")
            self.motor2 = LkMotor(port2, motor_id=motor2_id)

        # 校准值: 3.4rad=104mm → 104/3.4≈30.59
        self.MM_PER_RAD = 30.59
        self.ratio = 1.2

        # MIT 阻抗控制参数
        self.KP = 0.7
        self.KD = 0.1
        self.K_TRANS = 0.1

        self.TORQUE_LIMIT = 2.0
        self.TORQUE_LIMIT_DUAL = 2.0

        self.VIRTUAL_FIXTURE_ENABLED = True
        self.VIRTUAL_FIXTURE_TARGET_RAD = 2.5
        self.VIRTUAL_FIXTURE_KP = 0.15
        self.VIRTUAL_FIXTURE_MAX_TORQUE = 0.0

        self.MOTOR1_BIAS_RAD = 0
        self.GRIPPER_CLOSE_OFFSET_RAD = float(kwargs.get('close_offset_rad', 0.0))
        self.GRIPPER_CLOSE_OFFSET_ACTIVE_BELOW_RAD = float(
            kwargs.get('close_offset_active_below_rad', 0.05)
        )
        if self.GRIPPER_CLOSE_OFFSET_RAD > 0:
            logger.info(
                "close offset enabled: %.3f rad for targets <= %.3f rad",
                self.GRIPPER_CLOSE_OFFSET_RAD,
                self.GRIPPER_CLOSE_OFFSET_ACTIVE_BELOW_RAD,
            )
        self.DT = 1 / 120.0
        self.target_position_rad_max_delta = 3.14 * self.DT

        # 扭矩低通滤波状态
        self._torque_filt = 0.0
        self._torque_lpf_alpha = 0.3
        self._torque1_filt = 0.0
        self._torque2_filt = 0.0
        self._trans_lpf_alpha = 0.3

        # 干扰观测器参数
        self.J1 = 0
        self.J2 = 0
        self.torque_deadzone = 0.04
        self.prev_vel1 = 0.0
        self.prev_vel2 = 0.0
        self._acc1_filt = 0.0
        self._acc2_filt = 0.0
        self._acc_lpf_alpha = 0

        self.thread = None
        self.command_queue: queue.Queue = queue.Queue()

        self.current_position_rad = 0.0
        self.current_vel = 0.0
        self.current_torque_nm = 0.0
        self.current_position_rad_2 = 0.0
        self.current_vel_2 = 0.0
        self.current_torque_nm_2 = 0.0

        self.busy_wait_threshold = 0.0015

    def _angle_to_width(self, angle_rad: float) -> float:
        width_mm = angle_rad * self.MM_PER_RAD
        return width_mm / 1000.0

    def _width_to_angle(self, width_m: float) -> float:
        limited_width_m = min(width_m, self.MOTOR_PHYSICAL_LIMIT_M)
        width_mm = limited_width_m * 1000.0
        angle_rad = width_mm / self.MM_PER_RAD
        return angle_rad

    def get_current_gripper_width(self) -> float:
        return self._angle_to_width(self.current_position_rad)

    def get_current_gripper_force(self) -> float:
        return self.current_torque_nm

    def get_current_gripper_states(self):
        if self.mode == 'dual':
            return [
                self.current_position_rad,
                self.current_vel,
                self.current_torque_nm,
                self.current_position_rad_2,
                self.current_vel_2,
                self.current_torque_nm_2,
            ]
        return [
            self.current_position_rad,
            self.current_vel,
            self.current_torque_nm,
            0, 0, 0,
        ]

    def move_gripper(self, width_m: float, force_limit_nm: float, feedforward_torque_nm: float = 0.0):
        if self.mode != 'single':
            logger.warning("move_gripper 主要用于单电机模式，双电机模式下无效")
            return
        # 注意 upstream 行为：width_m 实际当 rad 用。调用方负责单位转换。
        target_rad = width_m
        self.command_queue.put(('move', target_rad, abs(force_limit_nm), feedforward_torque_nm))

    def move_gripper_force(self, force_limit_nm: float, close_speed_dps: float = 30.0):
        if self.mode != 'single':
            logger.warning("move_gripper_force 只能在 'single' 模式下使用")
            return
        self.command_queue.put(('grasp', abs(force_limit_nm), abs(close_speed_dps)))

    def stop_gripper(self):
        self.command_queue.put(('stop',))

    def _run(self):
        if self.mode == 'dual':
            self._dual_control_loop()
        else:
            self._single_control_loop()
        logger.info("控制循环已结束")

    def precise_sleep(self, delay: float) -> None:
        target_time = time.perf_counter() + delay
        sleep_duration = delay - self.busy_wait_threshold
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        while time.perf_counter() < target_time:
            pass

    def _single_control_loop(self):
        logger.info("开始单电机控制循环 (MIT阻抗控制)")
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
                close_offset = (
                    self.GRIPPER_CLOSE_OFFSET_RAD
                    if desired_position_rad <= self.GRIPPER_CLOSE_OFFSET_ACTIVE_BELOW_RAD
                    else 0.0
                )
                adjusted_desired_position = desired_position_rad - close_offset
                position_diff = adjusted_desired_position - actual_target_position_rad
                if abs(position_diff) > self.target_position_rad_max_delta:
                    actual_target_position_rad += math.copysign(self.target_position_rad_max_delta, position_diff)
                else:
                    actual_target_position_rad = adjusted_desired_position

                self.motor1.refresh()
                if getattr(self.motor1, 'refresh_ok', True) is False:
                    self.precise_sleep(self.DT)
                    continue

                pos_rad = self.motor1.getPosition()
                vel_rad_s = self.motor1.getVelocity()
                torque_nm = self.motor1.getTorque()

                self.current_position_rad = pos_rad
                self.current_vel = vel_rad_s
                self.current_torque_nm = torque_nm

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
        logger.info("单电机控制循环已停止")

    def _dual_control_loop(self):
        logger.info("开始双电机互控循环 (MIT阻抗控制)")
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

                refresh_futures = [
                    executor.submit(self.motor1.refresh),
                    executor.submit(self.motor2.refresh),
                ]
                wait(refresh_futures)

                if (self.motor1.refresh_ok is False) or (self.motor2.refresh_ok is False):
                    time.sleep(self.DT)
                    continue

                pos1 = self.motor1.getPosition()
                vel1 = self.motor1.getVelocity()
                torque1_val = self.motor1.getTorque()
                pos2 = self.motor2.getPosition()
                vel2 = self.motor2.getVelocity()
                torque2_val = self.motor2.getTorque()

                self.current_position_rad = pos1
                self.current_vel = vel1
                self.current_torque_nm = torque1_val
                self.current_position_rad_2 = pos2
                self.current_vel_2 = vel2
                self.current_torque_nm_2 = torque2_val

                self._torque1_filt = (self._trans_lpf_alpha * torque1_val
                                      + (1 - self._trans_lpf_alpha) * self._torque1_filt)
                self._torque2_filt = (self._trans_lpf_alpha * torque2_val
                                      + (1 - self._trans_lpf_alpha) * self._torque2_filt)

                acc1 = (vel1 - self.prev_vel1) / self.DT
                acc2 = (vel2 - self.prev_vel2) / self.DT
                self.prev_vel1 = vel1
                self.prev_vel2 = vel2

                self._acc1_filt = (self._acc_lpf_alpha * acc1 + (1 - self._acc_lpf_alpha) * self._acc1_filt)
                self._acc2_filt = (self._acc_lpf_alpha * acc2 + (1 - self._acc_lpf_alpha) * self._acc2_filt)

                tau_env1 = self._torque1_filt
                tau_env2 = self._torque2_filt
                if abs(tau_env1) < self.torque_deadzone:
                    tau_env1 = 0.0
                if abs(tau_env2) < self.torque_deadzone:
                    tau_env2 = 0.0

                bias = self.MOTOR1_BIAS_RAD
                pos_error_2 = (pos1 - bias) - pos2
                vel_error_2 = vel1 - vel2

                torque2 = (self.KP * pos_error_2 + self.KD * vel_error_2)
                # P-F 主从:主端只透传环境力 + 微阻尼,不做位置追踪
                LOCAL_DAMPING = 0.003
                torque1 = -self.K_TRANS * tau_env2 + LOCAL_DAMPING * vel1

                if self.VIRTUAL_FIXTURE_ENABLED:
                    fixture_error = self.VIRTUAL_FIXTURE_TARGET_RAD - pos1
                    fixture_torque = self.VIRTUAL_FIXTURE_KP * fixture_error
                    fixture_torque = max(-self.VIRTUAL_FIXTURE_MAX_TORQUE,
                                         min(self.VIRTUAL_FIXTURE_MAX_TORQUE, fixture_torque))
                    torque1 += fixture_torque

                torque1 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque1))
                torque2 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque2))

                set_torque_futures = [
                    executor.submit(self.motor1.set_torque_nm, torque1),
                    executor.submit(self.motor2.set_torque_nm, torque2),
                ]
                wait(set_torque_futures)

                elapsed = time.perf_counter() - loop_start
                if elapsed < self.DT:
                    time.sleep(self.DT - elapsed)
                else:
                    logger.debug("循环执行超时: %.2fms, target %.0fms", elapsed * 1000, self.DT * 1000)

        self.motor1.stop()
        self.motor2.stop()
        logger.info("双电机控制循环已停止")

    def start(self, auto_home: bool = True, home_torque_nm: float = -0.05, home_timeout_s: float = 2.0):
        if self.thread is not None and self.thread.is_alive():
            logger.info("控制器已经启动")
            return

        logger.info("启动电机...")
        self.motor1.enable()
        time.sleep(1)
        if self.mode == 'dual':
            self.motor2.enable()
            time.sleep(1)

        if auto_home:
            logger.info("MIT力控回零: 施加 %.2f Nm 力矩", home_torque_nm)
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
                    logger.info("检测到机械限位，回零完成")
                    break

                time.sleep(0.01)

            self.motor1.set_torque_nm(0)
            if self.mode == 'dual':
                self.motor2.set_torque_nm(0)
            time.sleep(0.5)

            if (time.time() - start_time) >= home_timeout_s:
                logger.warning("回零超时，可能未到达机械限位")

        logger.info("设置当前位置为零点")
        self.motor1.set_zero_ram()
        time.sleep(2)
        if self.mode == 'dual':
            self.motor2.set_zero_ram()
            time.sleep(2)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info("控制器已启动")

    def stop(self):
        logger.info("正在停止控制循环")
        if self.thread and self.thread.is_alive():
            self.command_queue.put(('shutdown',))
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("控制线程未能正常结束")

        logger.info("正在关闭电机")
        try:
            self.motor1.disable()
            if self.mode == 'dual' and self.motor2:
                self.motor2.disable()
        except Exception as e:
            logger.warning("关闭电机时发生错误: %s", e)
        logger.info("控制器已安全停止")
