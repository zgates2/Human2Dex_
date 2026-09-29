import threading
import time
import math
from typing import Dict

from motor.motor import LkMotor

KP = 0.80
KD = 0.05

MAX_VELOCITY_RAD_S = 1.0
DEFAULT_TORQUE_LIMIT_NM = 1.0

MM_PER_SINGLE_MOTOR_RADIAN = 46.5641 
MOTOR_PHYSICAL_LIMIT_RAD = math.pi / 2
CONTROL_LOOP_DT = 0.02


class GripperController:
    def __init__(self, motor_port: str, motor_id: int = 1):
        """
        初始化夹爪控制器，所有配置均从文件顶部的全局常量读取。

        Args:
            motor_port (str): 电机连接的串口。
            motor_id (int): 电机的ID。
        """
        self.KP = KP
        self.KD = KD
        self.max_velocity_rad_s = MAX_VELOCITY_RAD_S
        self.DT = CONTROL_LOOP_DT
        self.MM_PER_SINGLE_MOTOR_RADIAN = MM_PER_SINGLE_MOTOR_RADIAN
        self.MOTOR_PHYSICAL_LIMIT_RAD = MOTOR_PHYSICAL_LIMIT_RAD
        
        print(f"正在连接电机 (ID: {motor_id}) on {motor_port}...")
        self.motor = LkMotor(motor_port, motor_id=motor_id)
        self.motor.enable()
        time.sleep(0.5)
        self.motor.set_zero_ram()
        time.sleep(0.2)
        
        self.state_lock = threading.Lock()
        self.current_position_rad: float = 0.0
        self.current_torque_nm: float = 0.0
        self._immediate_target_rad: float = 0.0
        self._final_target_rad: float = 0.0
        self._current_torque_limit: float = DEFAULT_TORQUE_LIMIT_NM

        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print("最终版夹爪控制器初始化完成，已可接收指令。")
        print(f"  - 当前配置: KP={self.KP}, KD={self.KD}, MaxVel={self.max_velocity_rad_s} rad/s")

    def _control_loop(self):
        """后台高频控制循环"""
        # while self.is_running:
        #     loop_start = time.perf_counter()
        #     with self.state_lock:
        #         diff = self._final_target_rad - self._immediate_target_rad
        #         step = self.max_velocity_rad_s * self.DT
        #         if abs(diff) <= step:
        #             self._immediate_target_rad = self._final_target_rad
        #         else:
        #             self._immediate_target_rad += math.copysign(step, diff)

        #         self.motor.refresh()
        #         if self.motor.is_valid():
        #             self.current_position_rad = self.motor.getPosition()
        #             self.current_torque_nm = self.motor.getTorque()
                
        #         pos_error = self._immediate_target_rad - self.current_position_rad
        #         vel_error = 0 - self.motor.getVelocity()
                
        #         torque_cmd = self.KP * pos_error + self.KD * vel_error
        #         torque_cmd = max(-self._current_torque_limit, min(self._current_torque_limit, torque_cmd))
            
        #     self.motor.set_torque_nm(torque_cmd)
        #     time.sleep(max(0, self.DT - (time.perf_counter() - loop_start)))
        pass

    def shutdown(self):
        """安全地关闭控制器和电机。"""
        print("正在关闭控制器...")
        self.is_running = False
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=1)
        self.motor.disable()
        print("电机已安全关闭。")

    def move_gripper(self, width: float, force_limit: float):
        """以安全、恒定的速度移动夹爪到指定的宽度。"""
        print(f"指令: 移动夹爪至宽度 {width:.2f}mm, 力限制 {force_limit:.2f}Nm")
        target_angle = self._width_to_angle(width)
        with self.state_lock:
            self._final_target_rad = target_angle
            self._current_torque_limit = abs(force_limit)
            self.current_position_rad = target_angle

    def move_gripper_force(self, force_limit: float):
        """以力控模式抓取物体 (以恒定速度闭合直到力达标)。"""
        print(f"指令: 以力控模式抓取, 力限制 {force_limit:.2f}Nm")
        with self.state_lock:
            self._final_target_rad = 0.0
            self._current_torque_limit = abs(force_limit)

    def stop_gripper(self):
        """立即平滑地停止夹爪运动并保持当前位置。"""
        print("指令: 停止夹爪运动")
        with self.state_lock:
            self._final_target_rad = self._immediate_target_rad

    def get_current_gripper_width(self) -> float:
        """获取当前夹爪的开合宽度(mm)"""
        with self.state_lock:
            return self._angle_to_width(self.current_position_rad)

    def get_current_gripper_force(self) -> float:
        """获取当前夹爪的夹持力(等同于电机力矩Nm)"""
        with self.state_lock:
            return self.current_torque_nm

    def get_current_gripper_states(self):
        """获取当前夹爪的宽度和力状态"""
        width = self.get_current_gripper_width()
        force = self.get_current_gripper_force()
        return [width, force]
        # return {"width": width, "force": force}
    
    def is_busy(self, tolerance_mm: float = 0.5) -> bool:
        """
        检查夹爪是否仍在运动中。
        """
        # tolerance_rad = tolerance_mm / self.MM_PER_SINGLE_MOTOR_RADIAN
        # with self.state_lock:
        #     return abs(self._final_target_rad - self.current_position_rad) > tolerance_rad
        return False
            
    def _angle_to_width(self, angle_rad: float) -> float:
        return angle_rad * self.MM_PER_SINGLE_MOTOR_RADIAN

    def _width_to_angle(self, width_mm: float) -> float:
        angle_rad = width_mm / self.MM_PER_SINGLE_MOTOR_RADIAN
        return min(angle_rad, self.MOTOR_PHYSICAL_LIMIT_RAD)

if __name__ == "__main__":
    MOTOR_SERIAL_PORT = "/dev/ttyACM1"
    
    gripper = GripperController(motor_port=MOTOR_SERIAL_PORT)
    
    try:
        print("\n--- 最终版控制器API测试 (全局配置版) ---")
        time.sleep(1)

        print("\n[测试1] 指令: 移动到 50mm")
        gripper.move_gripper(width=50.0, force_limit=1.0)
        
        while gripper.is_busy():
            states = gripper.get_current_gripper_states()
            print(f"\r移动中... 当前状态: 宽度={states['width']:.2f}mm, 力={states['force']:.3f}Nm", end="")
            time.sleep(0.1)
        print("\n移动完成!")
        time.sleep(1)

        print("\n[测试2] 指令: 以 0.8Nm 的力抓取")
        gripper.move_gripper_force(force_limit=0.8)
        print("抓取中 (您现在应该能看到夹爪正在闭合)...")
        time.sleep(3) 
        
        print("\n[测试3] 指令: 停止抓取")
        gripper.stop_gripper()
        print("运动已停止。")
        time.sleep(1)

        stopped_states = gripper.get_current_gripper_states()
        print(f"停止后的最终状态: 宽度={stopped_states['width']:.2f}mm, 力={stopped_states['force']:.3f}Nm")
        time.sleep(1)

    except Exception as e:
        print(f"\n程序发生严重错误: {e}")
    finally:
        if gripper:
            print("\n--- 测试结束，关闭控制器 ---")
            gripper.shutdown()