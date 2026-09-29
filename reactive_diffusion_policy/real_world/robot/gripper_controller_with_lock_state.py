import math
import threading
import time
from motor.motor import LkMotor
from motor.protocol import radian_to_degree


class GripperController:
    # MM_PER_SINGLE_MOTOR_RADIAN = 46.5641
    MM_PER_SINGLE_MOTOR_RADIAN = 45.0000
    MOTOR_PHYSICAL_LIMIT_RAD = math.pi / 2
    POSITION_REACHED_THRESHOLD_RAD = 0.02
    
    SOFT_TORQUE_THRESHOLD_RATIO = 0.8
    BACK_OFF_STEP_RAD = 0.01
    
    VELOCITY_P_GAIN = 200.0
    MAX_SPEED_DPS = 60.0  # deg/s

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

        self.ratio = 1.2
        self.KP = 2.0 / 2 / 6 * 1.5 * 1.3 * self.ratio
        self.KD = 0.01 / 1.5 * 1.5 * self.ratio * 1.0
        self.TORQUE_LIMIT_DUAL = 2.5
        self.DT = 0.02

        self.running = False
        self.thread = None
        self.state_lock = threading.Lock()

        self.current_position_rad = 0.0
        self.current_torque_nm = 0.0
        
        self.current_position_rad_2 = 0.0
        self.current_torque_nm_2 = 0.0
        
        self.control_state = 'IDLE'  # 'IDLE', 'POSITION_MODE', 'FORCE_MODE'
        self.target_position_rad = 0.0
        self.force_limit_nm = 0.0
        self.close_speed_dps = 15.0

    def _angle_to_width(self, angle_rad: float) -> float:
        """角度(rad) -> 夹爪开合宽度 (m)"""
        width_mm = angle_rad * self.MM_PER_SINGLE_MOTOR_RADIAN
        return width_mm / 1000.0 * 2.0    # 两个夹爪间的距离

    def _width_to_angle(self, width_m: float) -> float:
        """夹爪开合宽度 (m) -> 角度(rad)"""
        width_mm = width_m * 1000.0 / 2.0
        angle_rad = width_mm / self.MM_PER_SINGLE_MOTOR_RADIAN
        return min(angle_rad, self.MOTOR_PHYSICAL_LIMIT_RAD)

    def get_current_gripper_width(self) -> float:
        """获取当前夹爪的开合宽度 (m)"""
        with self.state_lock:
            return self._angle_to_width(self.current_position_rad)

    def get_current_gripper_force(self) -> float:
        """获取当前夹爪的夹持力 (Nm)"""
        with self.state_lock:
            return self.current_torque_nm

    # def get_current_gripper_states(self):
    #     """获取当前夹爪的宽度(m)和力(Nm)"""
    #     width = self.get_current_gripper_width()
    #     force = self.get_current_gripper_force()
    #     return [width, force]
    def get_current_gripper_states(self):
        with self.state_lock:
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
        with self.state_lock:
            self.target_position_rad = self._width_to_angle(width_m)
            self.force_limit_nm = abs(force_limit_nm)
            self.control_state = 'POSITION_MODE'
            print(f"指令: 移动到宽度 {width_m*1000:.2f} mm (角度: {self.target_position_rad:.3f} rad) "
                  f"力限制: {self.force_limit_nm:.2f} Nm")

    def move_gripper_force(self, force_limit_nm: float, close_speed_dps: float = 15.0):
        if self.mode != 'single':
            print("[错误] move_gripper_force 只能在 'single' 模式下使用。")
            return
        with self.state_lock:
            self.force_limit_nm = abs(force_limit_nm)
            self.close_speed_dps = abs(close_speed_dps)
            self.control_state = 'FORCE_MODE'
            print(f"指令: 夹持至力矩达到 {self.force_limit_nm:.2f} Nm")
    
    def stop_gripper(self):
        if self.mode != 'single':
            print("[警告] stop_gripper 只在 'single' 模式下有效。")
            return
        with self.state_lock:
            if self.control_state != 'IDLE':
                self.control_state = 'IDLE'
                self.motor1.stop()
                print("[事件] 夹爪运动被手动停止。")
            
    def _run(self):
        if self.mode == 'dual': self._dual_control_loop()
        else: self._single_control_loop()
        
    def _single_control_loop(self):
        print("开始单电机柔顺控制循环...")
        while self.running:
            loop_start = time.perf_counter()
            self.motor1.refresh()
            if not self.motor1.is_valid():
                time.sleep(self.DT); continue

            pos_rad, torque_nm = self.motor1.getPosition(), self.motor1.getTorque()

            with self.state_lock:
                self.current_position_rad = pos_rad
                self.current_torque_nm = torque_nm
                
                if self.control_state == 'POSITION_MODE':
                    if abs(self.target_position_rad - pos_rad) < self.POSITION_REACHED_THRESHOLD_RAD:
                        self.motor1.stop()
                        self.control_state = 'IDLE'
                        print(f"[事件] 已到达目标位置。")
                        continue

                    soft_limit = self.force_limit_nm * self.SOFT_TORQUE_THRESHOLD_RATIO
                    
                    if abs(torque_nm) >= self.force_limit_nm:
                        self.motor1.stop()
                        self.control_state = 'IDLE'
                        print(f"[事件] 位置移动中断：达到硬力矩限制 {abs(torque_nm):.2f} Nm")
                    elif abs(torque_nm) >= soft_limit:
                        back_off_pos_rad = pos_rad + math.copysign(self.BACK_OFF_STEP_RAD, -torque_nm)
                        self.motor1.move_to_position(radian_to_degree(back_off_pos_rad))
                    else:
                        pos_error_rad = self.target_position_rad - pos_rad
                        desired_speed_dps = pos_error_rad * (180 / math.pi) * self.VELOCITY_P_GAIN
                        desired_speed_dps = max(-self.MAX_SPEED_DPS, min(self.MAX_SPEED_DPS, desired_speed_dps))
                        self.motor1.set_speed(desired_speed_dps)

                elif self.control_state == 'FORCE_MODE':
                    if abs(torque_nm) >= self.force_limit_nm:
                        self.motor1.stop()
                        self.control_state = 'IDLE'
                        print(f"[事件] 已达到目标力矩 {abs(torque_nm):.2f} Nm")
                    else:
                        # FORCE_MODE 本身就是速度控制，所以是平滑的
                        self.motor1.set_speed(-self.close_speed_dps)
            
            elapsed = time.perf_counter() - loop_start
            if elapsed < self.DT: time.sleep(self.DT - elapsed)

    def _dual_control_loop(self):
        print("开始双电机互控...")
        while self.running:
            loop_start = time.perf_counter()

            self.motor1.refresh()
            self.motor2.refresh()

            if not (self.motor1.is_valid() and self.motor2.is_valid()):
                time.sleep(self.DT)
                continue

            pos1, vel1, torque1_val = self.motor1.getPosition(), self.motor1.getVelocity(), self.motor1.getTorque() 
            pos2, vel2, torque2_val = self.motor2.getPosition(), self.motor2.getVelocity(), self.motor2.getTorque()

            with self.state_lock:
                self.current_position_rad = pos1
                self.current_torque_nm = torque1_val
                self.current_position_rad_2 = pos2
                self.current_torque_nm_2 = torque2_val
                

            pos_error_1, vel_error_1 = pos2 - pos1, vel2 - vel1
            pos_error_2, vel_error_2 = pos1 - pos2, vel1 - vel2

            torque1 = self.KP * pos_error_1 + self.KD * vel_error_1
            torque2 = self.KP * pos_error_2 + self.KD * vel_error_2

            torque1 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque1))
            torque2 = max(-self.TORQUE_LIMIT_DUAL, min(self.TORQUE_LIMIT_DUAL, torque2))

            # self.motor1.set_torque_nm(torque1)
            # self.motor2.set_torque_nm(torque2)

            loop_end = time.perf_counter()
            elapsed = loop_end - loop_start
            if elapsed < self.DT:
                time.sleep(self.DT - elapsed)
            else:
                print(f"[警告] 循环执行超时! 耗时: {elapsed*1000:.2f}ms, 目标 DT: {self.DT*1000:.0f}ms")

    def start(self):
        print("启动电机...")
        self.motor1.enable()
        time.sleep(2)
        if self.mode == 'dual': self.motor2.enable()
        time.sleep(2)

        print("设置当前位置为零点...")
        self.motor1.set_zero_ram()
        time.sleep(4)
        if self.mode == 'dual': self.motor2.set_zero_ram()
        time.sleep(4)

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("控制器已启动。")

    def stop(self):
        print("正在停止控制循环...")
        self.running = False
        if self.thread: self.thread.join()
        print("正在关闭电机...")
        try:
            self.motor1.disable()
            if self.mode == 'dual': self.motor2.disable()
        except Exception as e:
            print(f"关闭电机时发生错误: {e}")
        print("控制器已安全停止。")


if __name__ == "__main__":
    print("\n--- 测试单电机柔顺控制模式 ---")
    # single_gripper = GripperController(mode='single', port1="/dev/ttyUSB0", motor1_id=1)
    single_gripper = GripperController(mode='dual', port1="/dev/ttyACM3", motor1_id=1, port2="/dev/ttyACM2", motor2_id=2)
    try:
        single_gripper.start()
        
        print("双电机同步已启动。可以手动移动一个电机，另一个会跟随。")
        print("按 Ctrl+C 停止。")
        
        while True:
            width1, force1, witdh2, force2 = single_gripper.get_current_gripper_states()
            print(f"状态: 宽度={width1*1000:.2f} mm, 力={force1:.3f} Nm", end='\r')
            time.sleep(0.1) # 每0.1秒打印一次状态

        
        # print("\n[任务] 平滑移动到 50mm，力限制 1.2 Nm...")
        # single_gripper.move_gripper(width_m=0.05, force_limit_nm=1.2)
        
        # for i in range(50):
        #     time.sleep(0.1)
        #     width, force = single_gripper.get_current_gripper_states()
        #     print(f"状态: 宽度={width*1000:.2f} mm, 力={force:.3f} Nm", end='\r')
        # print("\n移动任务完成。")

        # time.sleep(2)
        # print("\n[任务] 平滑地完全打开...")
        # single_gripper.move_gripper(width_m=0.08, force_limit_nm=1.2)
        # time.sleep(3)

    except KeyboardInterrupt:
        print("\n检测到手动中断...")
    finally:
        single_gripper.stop()