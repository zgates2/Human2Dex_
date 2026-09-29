"""Example script of moving robot joint positions."""
import argparse
import pickle
import threading
import time
import queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable, Any
import collections

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

import deoxys.utils.transform_utils as T

import sys
sys.path.append('/home/ps/work/deoxys_control/deoxys/beta_scripts')
sys.path.append('/home/ps/reactive_diffusion_policy')

from reactive_diffusion_policy.real_world.robot.gripper_process_worker import GripperProcessProxy 
from reactive_diffusion_policy.real_world.robot.force_publisher import ForceSensorReader
from reactive_diffusion_policy.common.data_models import RobotStates

from spatialmath import SE3
from spatialmath.base import *
import transformations as tft
from loguru import logger

from spatialmath import *

from reactive_diffusion_policy.real_world.robot.teleop_helper import HTCTeleopHelper
from reactive_diffusion_policy.real_world.robot.vive_tracker import ViveForceSensorTracker

import threading

import os
import time
from reactive_diffusion_policy.common.space_utils import pose_6d_to_pose_7d, matrix4x4_to_pose_6d

class ThreadedReader:
    """
    一个通用的线程读取器，用于将任何阻塞的I/O函数异步化。
    """
    def __init__(self, read_function: Callable[[], Any], poll_interval: float = 0.005):
        """
        :param read_function: 一个无参数的、会阻塞并返回数据的函数
        :param poll_interval: 两次读取之间的休眠时间，防止CPU占用过高
        """
        self.read_function = read_function
        self.poll_interval = poll_interval
        
        self._latest_data = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = threading.Thread(target=self._update_loop, daemon=True)

    def _update_loop(self):
        """线程执行的内部循环"""
        while self._running:
            try:
                data = self.read_function()
                with self._lock:
                    self._latest_data = data
            except Exception as e:
                logger.error(f"Error in ThreadedReader for {self.read_function.__qualname__}: {e}")
            time.sleep(self.poll_interval)

    def start(self):
        """启动后台读取线程"""
        if not self._running:
            self._running = True
            try:
                self._latest_data = self.read_function()
            except Exception as e:
                logger.warning(f"Initial read failed for {self.read_function.__qualname__}: {e}. Will proceed without initial data.")
            self._thread.start()
            logger.info(f"Started threaded reader for: {self.read_function.__qualname__}")

    def stop(self):
        """停止后台线程"""
        if self._running:
            self._running = False
            self._thread.join()
            logger.info(f"Stopped threaded reader for: {self.read_function.__qualname__}")

    def get_data(self) -> Any:
        """
        非阻塞地获取最新读取到的数据。
        :return: 最新的数据
        """
        with self._lock:
            return self._latest_data


class FlexivController:
    """Franka机器人控制器类，提供笛卡尔空间控制功能"""
    
    def __init__(self, 
                 interface_cfg: str = "charmander.yml",
                 gripper_config: Dict = {
                    "mode": "single",
                    "port1": "/dev/ttyACM2",
                    "motor1_id": 2,
                 },
                 force_sensor_port: str = '/dev/ttyACM0'
                 ) -> None:
        self.RobotStates = RobotStates()
        self.cali_info = None
        self.cali_bias = None
        self.load_params()
        
        self.gripper_config = gripper_config
        
        self.gripper_controller = GripperProcessProxy(config=self.gripper_config)
        self.vive_tracker = ViveForceSensorTracker()
        
        logger.info("正在等待 Vive Tracker 稳定，这可能需要几秒钟...")
        start_time = time.time()
        timeout_seconds = 10
        while time.time() - start_time < timeout_seconds:
            if self.vive_tracker.get_force_sensor_pose() is not None:
                logger.info("Vive Tracker 已成功稳定并获取到初始位姿。")
                break
            time.sleep(0.1)
        else:
             raise RuntimeError(f"在 {timeout_seconds} 秒内未能从 Vive Tracker 获取到稳定位姿，程序终止。")
        
        self.force_sensor = ForceSensorReader(port=force_sensor_port)

        self.pose_reader = ThreadedReader(read_function=self.vive_tracker.get_force_sensor_pose)
        self.wrench_reader = ThreadedReader(read_function=self.force_sensor.get_wrench)
        self.gripper_reader = ThreadedReader(read_function=self.gripper_controller.get_current_gripper_states)

        self.pose_reader.start()
        self.wrench_reader.start()
        self.gripper_reader.start()

        self.call_index = 0
        self.last_time = time.time()
        
        starter_thread = threading.Thread(target=self._start_gripper_process_in_thread, daemon=True)
        starter_thread.start()
        
    def _start_gripper_process_in_thread(self):
        time.sleep(0.2)
        self.gripper_controller.start()
    
    def get_current_robot_states(self) -> Optional[dict]:
        pose_matrix_from_vive = self.pose_reader.get_data()
        raw_wrench = self.wrench_reader.get_data()
        gripper_states_raw = self.gripper_reader.get_data()

        if pose_matrix_from_vive is None or raw_wrench is None or gripper_states_raw is None:
            logger.warning("Sensor data not ready yet, skipping this cycle.")
            return None

        position = pose_matrix_from_vive[:3]
        quaternion_wxyz = pose_matrix_from_vive[3:]
        pose_matrix_from_vive = tft.quaternion_matrix(quaternion_wxyz)
        pose_matrix_from_vive[:3, 3] = position
        pose_se3 = SE3(pose_matrix_from_vive, check=False)
        position = pose_se3.t
        quaternion = np.array(pose_se3.UnitQuaternion())
        tcp_pose_7d = np.concatenate([position, quaternion]).tolist()

        rotation_matrix_vive = pose_matrix_from_vive[:3, :3]
        tool_offset = np.array([[0], [0], [0.156]])

        f_corrected, t_ee = self.gravity_compensation(
            np.array(raw_wrench), rotation_matrix_vive, tool_offset
        )
        
        if f_corrected is None:
             logger.warning("Gravity compensation failed. Skipping cycle.")
             return None
            
        compensated_wrench = np.concatenate([f_corrected.flatten(), t_ee.flatten()]).tolist()

        if gripper_states_raw is None:
            logger.warning("获取夹爪状态失败，将使用默认零值。")
            gripper_states_raw = [0.0, 0.0]

        if self.gripper_controller.mode == 'dual':
            gripper_state = gripper_states_raw
        else:
            gripper_state = gripper_states_raw + [0.0, 0.0]
            
        tcp_pose_target_7d = tcp_pose_7d
        
        return {
            "leftRobotTCP": tcp_pose_7d,
            "leftRobotTCPVel": [0.0] * 6,
            "leftRobotTCPWrench": compensated_wrench,
            "leftGripperState": gripper_state,
            "leftRobotTCPTarget": tcp_pose_target_7d,
            "leftJoinStates": [0.0] * 7,
        }

    def get_current_gripper_states(self):
        return self.gripper_reader.get_data()

    def get_current_gripper_force(self):
        return self.gripper_controller.get_current_gripper_force()

    def get_current_gripper_width(self):
        return self.gripper_controller.get_current_gripper_width()

    def get_current_pose(self):
        return self.pose_reader.get_data()

    def get_current_orientation(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        current_pose = self.get_current_pose()
        if current_pose is None: return None
        rot_matrix = tft.quaternion_matrix(current_pose[3:])[:3,:3]
        current_quat = T.mat2quat(rot_matrix)
        current_axis_angle = T.quat2axisangle(current_quat)
        return current_quat, current_axis_angle

    def get_current_tcp(self) -> Optional[List[float]]:
        current_pose = self.get_current_pose()
        if current_pose is None: return None
        return current_pose[:3].flatten().tolist()

    def gravity_compensation(self, raw_wrench_array, T_map_force, T_force_ee):
        """重力补偿计算函数"""
        # --- 该函数保持您的原始逻辑不变 ---
        if self.cali_info is None or self.cali_bias is None:
            logger.info("标定参数未加载，无法进行重力补偿")
            return None, None
        f_raw = raw_wrench_array[:3].reshape(3, 1)
        t_raw = raw_wrench_array[3:].reshape(3, 1)
        f_bias = self.cali_bias[3:6].reshape(3,1)
        t_bias = self.cali_info[3:6].reshape(3,1)
        r_cg = self.cali_info[0:3].reshape(3,1)
        g_vec = np.array([[0],[0],[-9.81]])
        m = np.linalg.norm(self.cali_bias[0:3]) / 9.81
        g_sensor = T_map_force[:3, :3].T @ g_vec
        f_gravity = m * g_sensor
        t_gravity = m * np.cross(r_cg.flatten(), g_sensor.flatten()).reshape(3,1)
        f_corrected = f_raw - f_bias - f_gravity
        t_corrected = t_raw - t_bias - t_gravity
        t_ee = t_corrected + np.cross(T_force_ee.flatten(), f_corrected.flatten()).reshape(3,1)
        return f_corrected, t_ee
    
    def load_params(self, filename="cali_params.pkl"):
        """自动加载标定参数"""
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [ os.path.join(p, filename) for p in ['.', base_path, os.path.join(base_path, '..')] ]
            param_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if param_path is None:
                logger.info(f"标定文件 '{filename}' 未在任何检查路径中找到。")
                return

            with open(param_path, 'rb') as f:
                params = pickle.load(f)
            self.cali_info = params['cali_info']
            self.cali_bias = params['cali_bias']
            logger.info(f"已成功从 '{param_path}' 加载标定参数。")
            self.T_force_ee = np.array([[0], [0], [0.156]])
        except Exception as e:
            logger.info(f"加载标定参数失败: {e}")
    
    def close(self):
        """关闭机器人接口"""
        # --- 新增：优雅地停止所有后台读取线程 ---
        logger.info("正在停止传感器读取线程...")
        self.pose_reader.stop()
        self.wrench_reader.stop()
        self.gripper_reader.stop()
        
        if self.gripper_controller:
            logger.info("正在停止夹爪控制器...")
            self.gripper_controller.stop()