"""Example script of moving robot joint positions."""
import argparse
import pickle
import threading
import time
import queue
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config,
                                       verify_controller_config)
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger
import deoxys.utils.transform_utils as T
import deoxys.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
import transformations as tft

import sys
sys.path.append('/home/ps/work/deoxys_control/deoxys/beta_scripts')
sys.path.append('/home/ps/reactive_diffusion_policy')

# from reactive_diffusion_policy.real_world.robot.gripper_controller import GripperController
from reactive_diffusion_policy.real_world.robot.gripper_controller_1011 import GripperController
from reactive_diffusion_policy.real_world.robot.gripper_process_worker import GripperProcessProxy 
from reactive_diffusion_policy.real_world.robot.force_publisher import ForceSensorReader
from reactive_diffusion_policy.real_world.robot.later_force_sensor_ros_pub import WrenchSensor
from reactive_diffusion_policy.common.data_models import RobotStates


# from gripper_control import GripperController
# from force_publisher import ForceSensorReader

from spatialmath import SE3
from spatialmath.base import *
from loguru import logger

# from benchmark_osc import move_to_target_pose
# from trajectory_interpolator import TrajectoryInterpolator
from spatialmath import *

# from reactive_diffusion_policy.real_world.robot.teleop_helper import HTCTeleopHelper

# from pynput import keyboard
import threading

import os
import time
from reactive_diffusion_policy.common.space_utils import pose_6d_to_pose_7d, matrix4x4_to_pose_6d, pose_7d_to_4x4matrix, matrix4x4_to_pose_7d

# from geometry_msgs.msg import TransformStamped
# from tf2_ros import TransformBroadcaster

# def _publish_tf(self, parent_frame, child_frame, pos, quat_wxyz):
#     """Publish a TF transform from parent_frame to child_frame"""
#     if not self.pub:
#         return
        
#     t = TransformStamped()
#     t.header.stamp = self.node.get_clock().now().to_msg()
#     t.header.frame_id = parent_frame
#     t.child_frame_id = child_frame
    
#     # Position
#     t.transform.translation.x = float(pos[0])
#     t.transform.translation.y = float(pos[1])
#     t.transform.translation.z = float(pos[2])
    
#     # Quaternion (w, x, y, z)
#     t.transform.rotation.w = float(quat_wxyz[0])
#     t.transform.rotation.x = float(quat_wxyz[1])
#     t.transform.rotation.y = float(quat_wxyz[2])
#     t.transform.rotation.z = float(quat_wxyz[3])
    
#     self.tf_broadcaster.sendTransform(t)
        
class FlexivController:
    """Franka机器人控制器类，提供笛卡尔空间控制功能"""
    
    def __init__(self, 
                 interface_cfg: str = "charmander.yml",
                 controller_type: str = "OSC_POSE",
                 control_frequency: int = 120,
                 use_visualizer: bool = False,
                 gripper_config: Dict = {
                    "mode": "single",
                    "port1": "/dev/ttyACM2",
                    "motor1_id": 2,
                 },
                #  left_gripper_port: Optional[str] = None,
                #  left_gripper_port: str = '/dev/ttyACM2',
                #  left_gripper_id: int = 1,
                 force_sensor_port: str = '/dev/ttyACM0'
                 ) -> None:
        """
        初始化Franka控制器
        
        Args:
            interface_cfg: 接口配置文件
            controller_type: 控制器类型
            control_frequency: 控制频率
            use_visualizer: 是否使用可视化
        """
        self.interface_cfg = interface_cfg
        self.controller_type = controller_type
        self.control_frequency = control_frequency
        self.use_visualizer = use_visualizer
        
        # 默认关节位置
        # self.reset_joint_positions = [0.09162008114028396, -0.19826458111314524, -0.01990020486871322, 
                                    #  -2.4732269941140346, -0.01307073642274261, 2.30396583422025, 0.8480939705504309]
        # self.reset_joint_positions = [0.01120143266724818, -0.1332651821553325, -0.4430850127232918, -2.686229667345102, -0.5912380438248317, 3.221344792381947, 0.26939041427278326] #  for lager wipe blackboard

        self.reset_joint_positions = [0.24886237143896478, -0.10823570083692469, -0.2110291881155267, -2.6155513998496334, -0.17867161943735246, 3.024342456234826, 0.3109925351142883] # for pick place egg

        # self.reset_joint_positions = [-0.07642651704829324, -0.1808715642308099, -0.4330556412920376, -2.8049991027729444, -0.3826065352881766, 3.115773088725849, -0.0039805526130518645]  # for smalll wipe blackboard
        
        # self.reset_joint_positions = [0.4169293399756414, -0.43932892444321603, -1.1142299904906958, -2.8021394537875524, -0.7932445713990193, 2.9412088879437213, 0.21236763791077667] # for wipe blackboard 2025.12.25

        # self.reset_joint_positions = [0.1931246121829016, -0.7368797204536303, -0.6086643421665371, -2.991080145667289, -0.41208872289790044, 2.6703454704814487, 0.015567715635730159] # for wipe blackboard 2025.12.25

        # self.reset_joint_positions = [0.3930702242621204, 0.10499223843587838, -0.5578696471931037, -2.397923722016184, 0.18889465236131772, 2.7496264518974196, -0.3547688272015888]
        
        # self.reset_joint_positions = [0.23450814764750627, -0.8940697880962419, -0.2984101930208373, -2.9593612909103766, -0.3203218846652242, 2.4125077548615725, 0.3435769063409206]

        # self.reset_joint_positions = [0.027478965141619675, -0.7353882745023358, -0.3617626738222395, -2.7666900015713876, -0.28187733776851465, 2.371029925717026, -0.13891724530999086]# for wipe blackboard 2026.01.05

        # 初始化机器人接口
        self.robot_interface = None
        self.controller_cfg = None
        self._initialize_robot()
        self.state = {}
        self.RobotStates = RobotStates()
        # self.load_params()

        # self.left_gripper = GripperController(motor_port=left_gripper_port, motor_id=left_gripper_id)
        # self.gripper_controller = GripperController(
        #     mode='single',
        #     port1='/dev/ttyACM3', motor1_id=1,
        # )
        
        # double motor for data collect
        # gripper_config = {
        #     'mode': 'dual',
        #     'port1': '/dev/ttyACM2',
        #     'motor1_id': 2,
        #     # 'port2': '/dev/ttyACM3',
        #     'port2': '/dev/ttyUSB0',
        #     'motor2_id': 1,
        # }
        
        # gripper_config = {
        #     'mode': 'single',
        #     'port1': '/dev/ttyACM2',
        #     'motor1_id': 2,
        #     # 'port1': '/dev/ttyUSB0',
        #     # 'motor1_id': 1,
        # }
        
        self.gripper_config = gripper_config
        
        self.gripper_controller = GripperProcessProxy(config=self.gripper_config)
        
        self.gripper_controller.start()
        # self.gripper_controller = GripperController(
        #     mode='dual',
        #     port1="/dev/ttyACM2", motor1_id=2,
        #     # port1="/dev/ttyUSB0", motor1_id=1,
        #     port2="/dev/ttyACM3", motor2_id=1,
        # )
        
        # self.force_sensor = ForceSensorReader(port=force_sensor_port)
        # self.force_sensor = WrenchSensor(port="/dev/ttyUSB0", baudrate=1000000)

        self.call_index = 0
        self.last_time = time.time()
        
        # 多线程控制相关属性
        self.stop_event = threading.Event()
        self.control_thread = None
        # 使用队列替代锁和直接变量，提高性能
        self.target_pose_queue = queue.Queue(maxsize=1)  # 只保留最新的目标位姿
        self.is_control_thread_running = False
        
        while len(self.robot_interface._state_buffer) == 0:
            time.sleep(0.1)
        current_ee_pose_se3 = SE3(np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
        theta, v = tr2angvec(current_ee_pose_se3.R)
        axisangle = v * theta
        self.last_valid_target_pose = current_ee_pose_se3.t.tolist() + axisangle.tolist() + [-1.0]
        
        # self.gripper_state_call_count = 0
        # self.last_gripper_freq_time = time.perf_counter()
        
                    
                    
        # 启动控制线程
        # self._start_control_thread()
        
        # time.sleep(1.0)
        # current_pose = np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
        # print(f"current_pose: {current_pose}")
        # pass
        
    def _initialize_robot(self):
        """初始化机器人接口和控制器配置"""
        try:
            # 创建机器人接口
            self.robot_interface = FrankaInterface(
                config_root + f"/{self.interface_cfg}", 
                control_freq=self.control_frequency, 
                use_visualizer=self.use_visualizer
            )
            
            # 获取控制器配置
            self.controller_cfg = get_default_controller_config(self.controller_type)
            self.controller_cfg.is_delta = False
            self.controller_cfg.action_scale.translation = 1.0 
            self.controller_cfg.action_scale.rotation = 1.0
            # self.controller_cfg.Kp.translation = 300.0 # 500.0
            # self.controller_cfg.Kp.translation = 500.0
            # self.controller_cfg.Kp.rotation = 80.0 # 50.0
            # self.controller_cfg.Kp.rotation = 150.0
            self.controller_cfg.Kp.translation = 300.0
            self.controller_cfg.Kp.rotation = 80.0 # 50.0
            # self.controller_cfg.Kp.translation = 10.0
            # self.controller_cfg.Kp.rotation = 1.0 # 50.0
            self.controller_cfg.control_frequency = self.control_frequency
            
            logger.info("Franka控制器初始化成功")
            
        except Exception as e:
            logger.info(f"初始化机器人接口失败: {str(e)}")
            raise
    
    def _start_control_thread(self):
        """启动控制线程"""
        if self.control_thread is None or not self.is_control_thread_running:
            # if self.gripper_controller:
            #     logger.info("正在启动夹爪控制器...")
            #     self.gripper_controller.start()
                
            self.stop_event.clear()
            self.control_thread = threading.Thread(target=self._control_thread_worker, daemon=True)
            self.control_thread.start()
            self.is_control_thread_running = True
            logger.info("控制线程已启动,需等1s")
            # 将当前 pose 作为 target pose
            current_ee_pose_se3 = SE3(np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
            theta, v = tr2angvec(current_ee_pose_se3.R)
            axisangle = v * theta
            target_pose_7d = current_ee_pose_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist() + [-1.0]
            self.set_target_pose(target_pose_7d)
            time.sleep(1.1)
    
    def _stop_control_thread(self):
        """停止控制线程"""
        if self.is_control_thread_running:
            self.stop_event.set()
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=2.0)  # 等待最多2秒
            self.is_control_thread_running = False
            logger.info("控制线程已停止")
    
    def _control_thread_worker(self):
        """控制线程工作函数，持续执行absolute_control"""
        logger.info("控制线程开始运行")
        
        # last_valid_target_pose = None  # 记录上一个有效的目标位姿
        # i = 0
        while not self.stop_event.is_set():
            # print(f"i: {i}")
            # i += 1
            try:
                # 从队列中获取目标位姿，非阻塞方式
                try:
                    target_pose = self.target_pose_queue.get_nowait()
                    # print(f"target_pose: {target_pose}")
                    if target_pose is not None:
                        # 更新上一个有效的目标位姿
                        self.last_valid_target_pose = target_pose
                        # 执行绝对位置控制
                        self.absolute_control(target_pose)
                        # print(f"self.last_valid_target_pose: {self.last_valid_target_pose}")
                    elif self.last_valid_target_pose is not None:
                        # 如果当前target_pose为None，使用上一个有效的目标位姿
                        # logger.debug(f"使用上一个有效目标位姿: {self.last_valid_target_pose}")
                        self.absolute_control(self.last_valid_target_pose)
                except queue.Empty:
                    # 队列为空时，如果有上一个有效的目标位姿，继续使用
                    if self.last_valid_target_pose is not None:
                        # logger.debug(f"队列为空，使用上一个有效目标位姿: {self.last_valid_target_pose}")
                        self.absolute_control(self.last_valid_target_pose)
                    # print('queue.Empty')
                
                # 控制频率
                # time.sleep(1.0 / self.control_frequency)
                
            except Exception as e:
                logger.error(f"控制线程执行错误: {str(e)}")
                time.sleep(0.1)  # 出错时短暂等待
        
        logger.info("控制线程已退出")
    
    def set_target_pose(self, target_pose: List[float]):
        # laststart_time = time.time()
        """
        设置目标位姿（线程安全，使用队列）
        
        Args:
            target_pose: 目标位姿 [x, y, z, rx, ry, rz, gripper] 或 None
                        None表示暂停控制，但保持上一个有效位姿
        """
        try:
            # 清空队列中的旧值，放入新值（包括None）
            while not self.target_pose_queue.empty():
                self.target_pose_queue.get_nowait()
            self.target_pose_queue.put_nowait(target_pose)
            
            # if target_pose is None:
            #     logger.debug("目标位姿设置为None，暂停控制但保持上一个有效位姿")
            # else:
            #     logger.debug(f"目标位姿已更新: {target_pose}")
                
        except queue.Full:
            logger.warning("目标位姿队列已满，跳过更新")
        except Exception as e:
            logger.error(f"设置目标位姿失败: {str(e)}")
        # logger.info('freq: ', 1 / (time.time()-self.last_time))
        self.last_time = time.time()
    
    def get_target_pose(self) -> Optional[List[float]]:
        """获取当前目标位姿（线程安全，使用队列）"""
        try:
            # 非阻塞方式获取目标位姿
            target_pose = self.target_pose_queue.get_nowait()
            # 重新放回队列，这样不会丢失数据
            self.target_pose_queue.put_nowait(target_pose)
            return target_pose
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"获取目标位姿失败: {str(e)}")
            return None
    
    def is_control_active(self) -> bool:
        """检查控制线程是否正在运行"""
        return self.is_control_thread_running and not self.stop_event.is_set()
    
    def pause_control(self):
        """暂停控制（设置目标位姿为None，但保持上一个有效位姿）"""
        # self.set_target_pose(None)
        logger.info("控制已暂停，但保持上一个有效位姿")
    
    def resume_control(self, target_pose: List[float]):
        """恢复控制（设置新的目标位姿）"""
        self.set_target_pose(target_pose)
        logger.info("控制已恢复")
    
    def get_control_status(self) -> dict:
        """获取控制状态信息"""
        return {
            "is_running": self.is_control_thread_running,
            "is_active": self.is_control_active(),
            "has_target": not self.target_pose_queue.empty(),
            "queue_size": self.target_pose_queue.qsize(),
            "current_target_pose": self.get_target_pose(),
            "control_frequency": self.control_frequency,
            "note": "当target_pose为None时，会使用上一个有效位姿继续控制"
        }
    
    def get_last_valid_target_pose(self) -> Optional[List[float]]:
        """获取上一个有效的目标位姿（用于调试和监控）"""
        try:
            # 尝试从队列中获取当前值
            current_pose = self.get_target_pose()
            if current_pose is not None:
                return current_pose
            
            # 如果队列为空，返回None（表示还没有设置过有效位姿）
            return None
        except Exception as e:
            logger.error(f"获取上一个有效目标位姿失败: {str(e)}")
            return None
    
    def reset_to_home(self):
        """重置机器人到初始位置"""
        if self.robot_interface is not None:
            # 停止控制线程
            logger.info("重置前停止控制线程...")
            self._stop_control_thread()
            
            # 执行重置操作
            reset_joints_to(self.robot_interface, self.reset_joint_positions)
            time.sleep(1.0)  # 等待机器人到达初始位置
            # move_to_position(self.robot_interface, self.reset_joint_positions)
            # target_7dpose = [0.5196294293292169, -0.2156082942890752, 0.28196828856550654, 0.18133782671797274, -0.817560749762115, 0.3117777430203325, -0.44889380927649053]
            # target_7dpose = [0.5507221748557546, -0.095460914511574, 0.3087936619958489, 0.06936711141534938, -0.8790220996431051, 0.3574700422381382, -0.30777186534051676]
            # use to collect data
            target_7dpose = [0.3507221748557546, -0.095460914511574, 0.3087936619958489, 0.06936711141534938, -0.8790220996431051, 0.3574700422381382, -0.30777186534051676]
            
            # use to replay traj
            target_7dpose = [0.3507221748557546, -0.095460914511574, 0.4087936619958489, 0.06936711141534938, -0.8790220996431051, 0.3574700422381382, -0.30777186534051676]
            
            # target_7dpose = [0.5393377382628499, -0.07896847017187847, 0.3110309643745146, -0.6312972256719985, 0.3165361797693695, -0.015941616976786522, 0.7078237948840211]
            
            # use to inference
            target_7dpose = [0.4007221748557546, -0.095460914511574, 0.3087936619958489, 0.06936711141534938, -0.8790220996431051, 0.3574700422381382, -0.30777186534051676]

            # use to test generalization model
            # target_7dpose = [0.4007221748557546, 0.095460914511574, 0.3087936619958489, 0.06936711141534938, -0.8790220996431051, 0.3574700422381382, -0.30777186534051676]
            
            # Rz -45deg
            target_7dpose = matrix4x4_to_pose_7d(pose_7d_to_4x4matrix(target_7dpose) @ tft.rotation_matrix(-np.pi/4, (0, 0, 1)))

            # target_7dpose = [0.40509667182418657, -0.09404870995994667, 0.310499728662094, 0.003678251485221603, 0.9935792247339545, -0.06586662739840773, 0.09191508058117949]

            # target_7dpose = [0.4162551106637166, -0.1275177298768997, 0.32455379896288183, 0.09266498922978518, -0.07638207557914826, 0.18357529602592285, 0.9756429105929382]

            target_7dpose = [0.42643067112976923, -0.1347191513655038, 0.34296087717027834, 0.0704599030929515, -0.051515171260330714, 0.3119421432227898, 0.9460833411849736]

            # self.tcp_move(target_7dpose, duration=3.0)
            
            # 重新启动控制线程
            logger.info("重置完成后重新启动控制线程...")
            self._start_control_thread()
            
            logger.info("机器人已重置到初始位置")
    
    def get_current_robot_states(self):
        if self.robot_interface is None:
            raise RuntimeError("机器人接口未初始化")
        
        # state = {}
        self.state['O_F_ext_hat_K'] = self.robot_interface._state_buffer[-1].O_F_ext_hat_K
        self.state['O_T_EE'] = np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
        
        # import pdb;pdb.set_trace()
        self.state['q'] = self.robot_interface._state_buffer[-1].q
        self.RobotStates.joinStates = np.array(self.state['q'], dtype=float).flatten().tolist()
        self.state['q_d'] = self.robot_interface._state_buffer[-1].q_d
        # self.state['6dpose'] = self.state['O_T_EE'].A[:3, 3:].tolist()+self.state['O_T_EE'].rpy().tolist # xyz,rpy
        pose_se3 = SE3(self.state['O_T_EE'], check=False)
        pose_6d_xyzrpy = np.concatenate([pose_se3.t, pose_se3.rpy()]).tolist()
        
        # tcpPose = self.get_current_pose()
        current_tmat = np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
        current_tmat_se3 = SE3(current_tmat)
        current_pos = current_tmat_se3.A[:3, 3:]
        current_quat = current_tmat_se3.UnitQuaternion()
        # np.roll(np.array(current_tmat_se3.UnitQuaternion()),-1)
        current_quat = np.array(current_tmat_se3.UnitQuaternion())
        
        self.RobotStates.tcpPose = current_pos.flatten().tolist() + current_quat.flatten().tolist()
        self.RobotStates.tcpVel = [0]*6
        # self.RobotStates.extWrenchInTcp = [0]*6
        # original_wrench = self.force_sensor.get_wrench()
        original_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # logger.info(f"original_wrench: {original_wrench}")
        # logger.info(f"Failure rate is {self.force_sensor.failure_count / self.force_sensor.success_count  * 100 }%")
        # self.load_params()
        
        # f_corrected, t_ee = self.gravity_compensation(np.array(original_wrench), current_tmat[:3, :3], np.array([[0], [0], [0.156]]))
        # self.RobotStates.extWrenchInTcp = f_corrected.flatten().tolist() + t_ee.flatten().tolist()
        self.RobotStates.extWrenchInTcp = original_wrench
        
        # for gravity compensation
        # self.RobotStates.extWrenchInTcp = original_wrench
        # self.RobotStates.width = [0] # width
        # self.RobotStates.force = [0] # torque
        # self.RobotStates.gripperState = self.get_current_gripper_states()
        gripper_states_raw = self.get_current_gripper_states()
        
        # self.gripper_state_call_count += 1
        # current_time = time.perf_counter()
        # if current_time - self.last_gripper_freq_time >= 1.0:
        #     freq = self.gripper_state_call_count / (current_time - self.last_gripper_freq_time)
        #     logger.info(f"get_current_gripper_states() Call Freq: {freq:.2f} Hz")
        #     self.gripper_state_call_count = 0
        #     self.last_gripper_freq_time = current_time
        
        self.RobotStates.tcpPoseTarget = pose_6d_to_pose_7d(self.last_valid_target_pose)
        
        # self.last_valid_target_pose: xyz,axayaz,gripper 
        # 需要将 xyz,axayaz,gripper --> xyz,qwxyz 方便后续控制
        axisangle = self.last_valid_target_pose[3:-1]
        assert len(axisangle) == 3
        so3 = SO3.AngleAxis(np.linalg.norm(axisangle), axisangle / np.linalg.norm(axisangle))
        quant = np.array(so3.UnitQuaternion())
        xyz = self.last_valid_target_pose[:3]
        # wxyz = np.concatenate([xyz, quant])
        # self.RobotStates.tcpPose = np.concatenate([xyz, quant])
        
        # self.RobotStates.tcpPose = pose_6d_to_pose_7d(self.last_valid_target_pose)
        
        # if self.gripper_controller.mode == 'dual':
        #     self.RobotStates.gripperState = gripper_states_raw
        # else:
        #     # for single motor interence
        #     self.RobotStates.gripperState = gripper_states_raw + [0.0, 0.0]
        self.RobotStates.gripperState = gripper_states_raw
        # self.RobotStates.width = [gripper_state_list[0]] # width
        # self.RobotStates.force = [gripper_state_list[1]] # torque
        return {
            "leftRobotTCP": self.RobotStates.tcpPose,
            # "leftRobotTCP": self.RobotStates.tcpPoseTarget,
            # "rightRobotTCP": [0.0] * 7,
            "leftRobotTCPVel": self.RobotStates.tcpVel,
            # "rightRobotTCPVel": [0.0] * 6,
            "leftRobotTCPWrench": self.RobotStates.extWrenchInTcp,
            # "rightRobotTCPWrench": [0.0] * 6,
            "leftGripperState": self.RobotStates.gripperState,
            # "rightGripperState": [0.0] * 2
            "leftRobotTCPTarget": self.RobotStates.tcpPoseTarget, # [0.0] * 7
            "leftJoinStates": self.RobotStates.joinStates
        }  
                    # return BimanualRobotStates(leftRobotTCP=left_robot_state.tcpPose,
                    #                    rightRobotTCP=right_robot_state.tcpPose,
                    #                    leftRobotTCPVel=left_robot_state.tcpVel,
                    #                    rightRobotTCPVel=right_robot_state.tcpVel,
                    #                    leftRobotTCPWrench=left_robot_state.extWrenchInTcp,
                    #                    rightRobotTCPWrench=right_robot_state.extWrenchInTcp,
                    #                    leftGripperState=[left_robot_gripper_state.width,
                    #                                         left_robot_gripper_state.force],
                    #                    rightGripperState=[right_robot_gripper_state.width,
                    #                                              right_robot_gripper_state.force])
        
        # return pose_6d_xyzrpy

    def get_current_gripper_states(self):
        # return self.left_gripper.get_current_gripper_states()
        return self.gripper_controller.get_current_gripper_states()
        # pass

    def get_current_gripper_force(self):
        # return self.left_gripper.get_current_gripper_force()
        return self.gripper_controller.get_current_gripper_force()
        # pass

    def get_current_gripper_width(self):
        # return self.left_gripper.get_current_gripper_width()
        return self.gripper_controller.get_current_gripper_width()
        # pass

    def get_current_q(self) -> List[float]:
        # 返回flexivAPI下机械臂当前joints值
        return self.robot_interface._state_buffer[-1].q
    
    def get_current_pose(self) -> np.ndarray:
        """
        获取当前末端执行器位姿
        
        Returns:
            np.ndarray: 4x4变换矩阵
        """
        # return np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
        #  = np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
        current_tmat = np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
        current_tmat_se3 = SE3(current_tmat)
        current_pos = current_tmat_se3.A[:3, 3:]
        current_quat = current_tmat_se3.UnitQuaternion()
        # np.roll(np.array(current_tmat_se3.UnitQuaternion()),-1)
        current_quat = np.array(current_tmat_se3.UnitQuaternion())
        # current_quat = np.array(current_tmat_se3.UnitQuaternion()).tolist()
        # current_quat_tg = T.mat2quat(current_tmat_se3.A[:3, :3]) # (x,y,z,w)
        # print(f"current_quat_tg--------------------{current_quat_tg}")
        # print(f"current_quat--------------------{np.roll(np.array(current_tmat_se3.UnitQuaternion()),-1)}")
        # print(f"current_quat--------------------{current_quat}")
        return current_pos.flatten().tolist() + current_quat.flatten().tolist()
        # return np.concatenate([current_pos, current_quat])

    def get_current_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前姿态
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (四元数, 轴角)
        """
        current_pose = self.get_current_pose()
        current_rot = current_pose[:3, :3]
        current_quat = T.mat2quat(current_rot)
        current_axis_angle = T.quat2axisangle(current_quat)
        return current_quat, current_axis_angle

    def get_current_tcp(self) -> List[float]:
        # 返回flexivAPI下机械臂当前tcp值
        current_pose = self.get_current_pose()
        return current_pose[:3, 3:].flatten()
    
    def absolute_control(self, target_pose: List[float]):
        """
        基于绝对笛卡尔空间位置的控制
        
        Args:
            target_pose: 目标位姿 [x, y, z, ax, ay, az, gripper]
        """
        # print(f"absolute_control--------------------{target_pose}")
        if self.robot_interface is None:
            raise RuntimeError("机器人接口未初始化")
        
        self.robot_interface.control(
            controller_type=self.controller_type,
            action=target_pose,
            controller_cfg=self.controller_cfg,
        )
    
    def move(self, goal_pose, duration: float):
        """
        从当前位置移动到目标位置，使用插值控制
        
        Args:
            goal_pose: 目标位姿 (SE3对象或4x4矩阵)
            duration: 运动持续时间(秒)
        """
        if self.robot_interface is None:
            raise RuntimeError("机器人接口未初始化")
        
        # print(goal_pose)
        # print(goal_pose.A)
        # current_ee_pose_se3 = SE3(self.get_current_pose(),check=False)
        current_ee_pose_se3 = SE3(np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
        goal_pose_se3 = SE3(goal_pose, check=False)
        # print(goal_pose_se3)
        num_steps = int(self.control_frequency * duration)
        
        print(f"开始运动到目标位置，持续时间: {duration}秒，步数: {num_steps}")
        
        for i in range(num_steps+int(num_steps*0.2)):
            # 计算插值位姿
            # target_pose = current_ee_pose_se3.interp(goal_pose_se3, i/num_steps).A
            s = (i+1) / num_steps
            interpolated_pose_se3 = current_ee_pose_se3.interp(goal_pose_se3, s)
            # print(f"-------------------{i}---------------------")
            # print(f"interplated--------------------{interpolated_pose_se3}")
            # target_pose = interpolated_pose_se3.A
            # print(f"target_pose--------------------{target_pose}")

            # target_pos = target_pose[:3, 3:]
            # target_quat = T.mat2quat(target_pose[:3, :3])
            # target_axis_angle = T.quat2axisangle(target_quat)
            # print(target_axis_angle)
            theta, v = tr2angvec(interpolated_pose_se3.R)
            axisangle = v * theta
            # print(f"axisangle--------------------{axisangle}")
            
            # print(interpolated_pose_se3.AngleAxis())
            # target_pose_7d = target_pos.flatten().tolist() + target_axis_angle.flatten().tolist() + [-1.0]
            target_pose_7d = interpolated_pose_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist() + [-1.0]

            # print(target_pose_7d)
            # print(target_pos.flatten().tolist() + target_axis_angle.flatten().tolist() + [-1.0])
            
            # 发送控制命令
            self.absolute_control(target_pose_7d)

            # 控制频率
            # time.sleep(1.0 / self.control_frequency)
        
        print("运动完成")

    
    def tcp_move_delta(self, delta_6dpose: List[float], duration: float):
        """
        基于增量的目标控制
        
        Args:
            delta_6dpose: 增量位姿 [dx, dy, dz, drx, dry, drz]
            duration: 运动持续时间(秒)
        """
        print(f"------Delta_6dPose: {delta_6dpose}")
        # current_ee_pose_se3 = SE3(self.get_current_pose())
        current_ee_pose_se3 = SE3(np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
        goal_pose_se3 = current_ee_pose_se3 * SE3(delta_6dpose[:3]) * SE3.RPY(delta_6dpose[3:])
        # self.move(goal_pose_se3, duration)

    def tcp_move(self, goal_pose: List[float], duration: float = 1.0/2.0): # xyz,wxyz
        
        goal_se3 = SE3(goal_pose[:3])*UnitQuaternion(goal_pose[3:]).SE3()


        ############################## debug ##############################
        
        # goal_se3 = SE3(goal_pose[:3])*Quaternion(goal_pose[3:])
        
        current_time = time.time()
        elapsed_time = current_time - self.last_time

        

        # goal_se3 = SE3(goal_pose[:3])*SE3.UnitQuaternion(UnitQuaternion(goal_pose[3:]))
        # goal_se3 = SE3(goal_pose[:3])*SE3.UnitQuaternion(UnitQuaternion(goal_pose[3:]))

        # current_ee_pose_se3 = SE3(self.get_current_pose())
        current_ee_pose_se3 = SE3(np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
        delta_se3 = current_ee_pose_se3.inv() * goal_se3
        theta, v = tr2angvec(delta_se3.R)
        axisangle = v * theta
        delta_6dpose = delta_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist()
        delta_6dpose = np.array(delta_6dpose)

        self.call_index += 1
        logger.info(f"goal_pose frequency { 1 / elapsed_time} hz )")
        logger.info(f"************{goal_pose}*****{self.call_index}****")
        print(f"delta_6dpose: {delta_6dpose[:3]*1000, delta_6dpose[3:]*180/np.pi}")

        self.last_time = time.time()
        # goal_se3 = SE3(goal_se3)
        
        
        self.move(goal_se3.A, duration)
    
    def tcp_move_v2(self, goal_pose: List[float]): # xyz,wxyz
        # goal_se3 = SE3(goal_pose[:3])*Quaternion(goal_pose[3:])
        logger.info(f"************{goal_pose}*********")
        # goal_se3 = SE3(goal_pose[:3])*SE3.UnitQuaternion(UnitQuaternion(goal_pose[3:]))
        # goal_se3 = SE3(goal_pose[:3])*SE3.UnitQuaternion(UnitQuaternion(goal_pose[3:]))
        goal_se3 = SE3(goal_pose[:3])*UnitQuaternion(goal_pose[3:]).SE3()
        # goal_se3 = SE3(goal_se3)
        theta, v = tr2angvec(goal_se3.R)
        axisangle = v * theta
        target_pose_7d = goal_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist() + [-1.0]

        # print(target_pose_7d)
        # print(target_pos.flatten().tolist() + target_axis_angle.flatten().tolist() + [-1.0])
        
        # 发送控制命令
        self.absolute_control(target_pose_7d)
        
        # self.move(goal_se3.A, duration)
    
    def tcp_move_v3(self, goal_pose: List[float]): # 多线程控制
        """
        实时修改目标位姿，通过控制线程持续执行
        
        Args:
            goal_pose: 目标位姿 [x, y, z, w, x, y, z] (位置 + 四元数)
        """
        try:
            # goal_se3 = SE3(position) * UnitQuaternion(quaternion).SE3()
            
            current_ee_pose_se3 = SE3(np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
            goal_se3 = SE3(goal_pose[:3])*UnitQuaternion(goal_pose[3:]).SE3()
            delta_se3 = current_ee_pose_se3.inv() * goal_se3
            theta, v = tr2angvec(delta_se3.R)
            axisangle = v * theta
            delta_6dpose = delta_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist()
            delta_6dpose = np.array(delta_6dpose)

            # self.call_index += 1
            # logger.info(f"goal_pose frequency { 1 / elapsed_time} hz )")
            # logger.info(f"************{goal_pose}*****{self.call_index}****")
            print(f"delta_6dpose: {delta_6dpose[:3]*1000, delta_6dpose[3:]*180/np.pi}")
            
            # 将 xyzwxyz 格式转换为 absolute_control 所需的格式
            # goal_pose: [x, y, z, w, x, y, z]
            position = goal_pose[:3]  # [x, y, z]
            quaternion = goal_pose[3:]  # [w, x, y, z]
            
            # 创建SE3对象
            goal_se3 = SE3(position) * UnitQuaternion(quaternion).SE3()
            
            # 转换为轴角表示
            theta, v = tr2angvec(goal_se3.R)
            axisangle = v * theta
            
            # 构建7D目标位姿 [x, y, z, rx, ry, rz, gripper]
            target_pose_7d = goal_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist() + [-1.0]
            
            # 设置目标位姿，控制线程会自动执行
            self.set_target_pose(target_pose_7d)
            
            # logger.info(f"目标位姿已更新: 位置={position}, 轴角={[f'{x:.4f}' for x in axisangle]}")
            
        except Exception as e:
            logger.error(f"tcp_move_v3 执行错误: {str(e)}")
            raise
        
    def tcp_move_v3_relative(self, relative_pose_7d: List[float]): # 多线程控制
        """
        实时修改目标位姿，通过控制线程持续执行
        
        Args:
            goal_pose: 目标位姿 [x, y, z, w, x, y, z] (位置 + 四元数)
        """
        try:
            # goal_se3 = SE3(position) * UnitQuaternion(quaternion).SE3()
            
            current_ee_pose_se3 = SE3(np.array(self.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
            delta_se3 = SE3(relative_pose_7d[:3])*UnitQuaternion(relative_pose_7d[3:]).SE3()
            goal_se3 = current_ee_pose_se3 * delta_se3
            # goal_se3 = SE3(goal_pose[:3])*UnitQuaternion(goal_pose[3:]).SE3()
            # delta_se3 = current_ee_pose_se3.inv() * goal_se3
            theta, v = tr2angvec(delta_se3.R)
            axisangle = v * theta
            delta_6dpose = delta_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist()
            delta_6dpose = np.array(delta_6dpose)

            # self.call_index += 1
            # logger.info(f"goal_pose frequency { 1 / elapsed_time} hz )")
            # logger.info(f"************{goal_pose}*****{self.call_index}****")
            print(f"delta_6dpose: {delta_6dpose[:3]*1000, delta_6dpose[3:]*180/np.pi}")
            
            # 将 xyzwxyz 格式转换为 absolute_control 所需的格式
            # goal_pose: [x, y, z, w, x, y, z]
            # position = goal_pose[:3]  # [x, y, z]
            # quaternion = goal_pose[3:]  # [w, x, y, z]
            
            # 创建SE3对象
            # goal_se3 = SE3(position) * UnitQuaternion(quaternion).SE3()
            
            # 转换为轴角表示
            theta, v = tr2angvec(goal_se3.R)
            axisangle = v * theta
            
            # 构建7D目标位姿 [x, y, z, rx, ry, rz, gripper]
            target_pose_7d = goal_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist() + [-1.0]
            
            # 设置目标位姿，控制线程会自动执行
            self.set_target_pose(target_pose_7d)
            
            # logger.info(f"目标位姿已更新: 位置={position}, 轴角={[f'{x:.4f}' for x in axisangle]}")
            
        except Exception as e:
            logger.error(f"tcp_move_v3 执行错误: {str(e)}")
            raise
        
    def move_to_position(self, target_pos: np.ndarray, target_quat: np.ndarray, duration: float = 3.0):
        """
        移动到指定位置和姿态
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标四元数 [w, x, y, z]
            duration: 运动持续时间(秒)
        """
        # 构建目标位姿矩阵
        target_rot = T.quat2mat(target_quat)
        target_pose = np.eye(4)
        target_pose[:3, :3] = target_rot
        target_pose[:3, 3] = target_pos
        
        self.move(target_pose, duration)
    
    def move_delta(self, delta_pos: np.ndarray, delta_rpy: np.ndarray, duration: float = 3.0):
        """
        相对当前位置移动
        
        Args:
            delta_pos: 位置增量 [dx, dy, dz]
            delta_rpy: 姿态增量 [drx, dry, drz] (弧度)
            duration: 运动持续时间(秒)
        """
        delta_6dpose = np.concatenate([delta_pos, delta_rpy]).tolist()
        self.tcp_move(delta_6dpose, duration)
    
    ############ gravity compensation ############   
    
    # def process_and_compensate(self, wrench_msg, vive_odom_msg):
    #     """处理力传感器数据：坐标系转换 + 重力补偿"""
    #     # 1. 坐标系转换：从Vive坐标系转换到力传感器坐标系
    #     # f_transformed, t_transformed, vive_rotation = self.transform_wrench_to_force_sensor_frame(
    #     #     wrench_msg, vive_odom_msg
    #     # )
    #     pos = [vive_odom_msg.pose.pose.position.x, vive_odom_msg.pose.pose.position.y, vive_odom_msg.pose.pose.position.z]
    #     quat = [vive_odom_msg.pose.pose.orientation.w, vive_odom_msg.pose.pose.orientation.x, vive_odom_msg.pose.pose.orientation.y, vive_odom_msg.pose.pose.orientation.z]
    #     T_map_vive = pose_to_matrix(pos, quat)
    #     self.T_map_force = np.dot(T_map_vive, self.T_vive_to_force)
    #     # 2. 重力补偿
    #     f_corrected, t_ee = self.gravity_compensation(wrench_msg, self.T_map_force, self.T_force_ee)
        
    #     return f_corrected, t_ee

    # def gravity_compensation(self, wrench_msg, T_map_force, T_force_ee):
    def gravity_compensation(self, raw_wrench_array, T_map_force, T_force_ee):
        """重力补偿计算函数"""
        # 检查标定参数是否已加载
        if self.cali_info is None or self.cali_bias is None:
            logger.info("标定参数未加载，无法进行重力补偿")
            return None, None
        # 原始力/力矩
        # f_raw = np.array([[wrench_msg.wrench.force.x], [wrench_msg.wrench.force.y], [wrench_msg.wrench.force.z]])
        # t_raw = np.array([[wrench_msg.wrench.torque.x], [wrench_msg.wrench.torque.y], [wrench_msg.wrench.torque.z]])
        f_raw = raw_wrench_array[:3].reshape(3, 1)
        t_raw = raw_wrench_array[3:].reshape(3, 1)
        # 零偏
        f_bias = self.cali_bias[3:6].reshape(3,1)
        t_bias = self.cali_info[3:6].reshape(3,1)
        # 重心
        r_cg = self.cali_info[0:3].reshape(3,1)
        # 估算质量
        g_vec = np.array([[0],[0],[-9.81]])
        m = np.linalg.norm(self.cali_bias[0:3]) / 9.81
        # 当前重力在传感器坐标系下
        g_sensor = T_map_force[:3, :3].T @ g_vec
        # 重力补偿项
        f_gravity = m * g_sensor
        t_gravity = m * np.cross(r_cg.flatten(), g_sensor.flatten()).reshape(3,1)
        # 补偿
        f_corrected = f_raw - f_bias - f_gravity
        t_corrected = t_raw - t_bias - t_gravity
        # ee_link在力传感器坐标系下的位置
        # r_ee = np.array([[0], [0], [0.156]])
        # ee_link处的力和力矩
        t_ee = t_corrected + np.cross(T_force_ee.flatten(), f_corrected.flatten()).reshape(3,1)
        
        return f_corrected, t_ee
    
    def load_params(self, filename="cali_params.pkl"):
        """自动加载标定参数"""
        try:
            # 在常见路径中搜索参数文件
            base_path = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join('.', filename),
                os.path.join(base_path, filename),
                os.path.join(base_path, '..', filename)
            ]
            
            param_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    param_path = p
                    break

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
        # 停止控制线程
        logger.info("关闭前停止控制线程...")
        self._stop_control_thread()
        
        self.force_sensor.close()
        
        if self.gripper_controller:
            logger.info("正在停止夹爪控制器...")
            # self.force_sensor.close()
            self.gripper_controller.stop()
        
        if self.robot_interface is not None:
            self.robot_interface.close()
            # self.force_sensor.close()
            logger.info("机器人接口已关闭")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="charmander.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    args = parser.parse_args()
    return args


def compute_errors(pose_1, pose_2):
    pose_a = (
        pose_1[:3]
        + transform_utils.quat2axisangle(np.array(pose_1[3:]).flatten()).tolist()
    )
    pose_b = (
        pose_2[:3]
        + transform_utils.quat2axisangle(np.array(pose_2[3:]).flatten()).tolist()
    )
    return np.abs(np.array(pose_a) - np.array(pose_b))


def demo_usage_thread():
    """演示如何使用FrankaController类"""
    # args = parse_args()
    
    GRIPPER_PORT = "/dev/ttyACM2"
    FORCE_SENSOR_PORT = '/dev/ttyACM0'

    # 创建控制器实例
    controller = FlexivController(
        # left_gripper_port=GRIPPER_PORT,
        force_sensor_port=FORCE_SENSOR_PORT
    )

    controller.reset_to_home()
    time.sleep(100.0)

    controller._start_control_thread()
    # controller.tcp_move_delta([0.1, -0.2, 0.1, 0.0, 0.0, -np.pi/4], duration=3.0)
    # controller.control_thread.start()
    # goal_pose = 
    time.sleep(1.0)
    current_ee_pose_se3 = SE3(np.array(controller.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
    # import pdb;pdb.set_trace()
    robot_state = controller.get_current_robot_states()
    delta_6dpose = [0.1, -0.2, 0.1, 0.0, 0.0, -np.pi/4]
    current_pose = controller.state['O_T_EE']
    delta_pose = SE3.Trans(delta_6dpose[0], delta_6dpose[1], delta_6dpose[2]) * SE3.RPY(delta_6dpose[3], delta_6dpose[4], delta_6dpose[5])
    # goal_pose_se3 = SE3(controller.state['O_T_EE']) * delta_pose
    goal_pose_se3 = SE3(current_ee_pose_se3) * delta_pose
    
    # goal_pose_se3 = SE3(goal_pose, check=False)
    # print(goal_pose_se3)
    duration = 5.0
    num_steps = int(controller.control_frequency * duration)
    
    print(f"开始运动到目标位置，持续时间: {duration}秒，步数: {num_steps}")
    last_time = time.time()
    
    for i in range(num_steps):
        s = (i+1) / num_steps
        interpolated_pose_se3 = current_ee_pose_se3.interp(goal_pose_se3, s)
        print(f"-------------------{i} / {num_steps}---------------------")
        theta, v = tr2angvec(interpolated_pose_se3.R)
        axisangle = v * theta
        target_pose_7d = interpolated_pose_se3.A[:3, 3:].flatten().tolist() + axisangle.flatten().tolist() + [-1.0]
        controller.set_target_pose(target_pose_7d)
        time.sleep(1.0 / controller.control_frequency)
        # 发送控制命令
        # controller.absolute_control(target_pose_7d)
        print('send target freq:', 1/(time.time()-last_time))
        last_time = time.time()
    print("-------------------------------- 定点阻抗模式 ---------------")
    time.sleep(10.0)
    controller.reset_to_home()
    controller._stop_control_thread()

def demo_usage():
    """演示如何使用FrankaController类"""
    # args = parse_args()
    
    GRIPPER_PORT = "/dev/ttyACM2"
    FORCE_SENSOR_PORT = '/dev/ttyACM0'

    # 创建控制器实例
    controller = FlexivController(
        # left_gripper_port=GRIPPER_PORT,
        force_sensor_port=FORCE_SENSOR_PORT
    )
    
    try:
        # try:
        #     gripper_state = controller.get_current_gripper_states()
        #     gripper_width = gripper_state[0]
        #     gripper_force = gripper_state[1]

        #     wrench = controller.force_sensor.get_wrench()
        #     if wrench:
        #         formatted_wrench = [f"{x:8.3f}" for x in wrench]
        #     else:
        #         formatted_wrench = ["N/A"] * 6
            
        #     print(
        #         f"\rGripper: Width={gripper_width:6.2f}, Force={gripper_force:6.2f} | "
        #         f"Wrench: [Fx, Fy, Fz, Mx, My, Mz] = {formatted_wrench}", end=""
        #     )
        #     time.sleep(0.1)
        # except KeyboardInterrupt:
        #     logger.info("--- Peripheral status check finished. Proceeding to movement demo. ---")

        # 重置到初始位置
        controller.reset_to_home()
        time.sleep(1.0)
        
        # 获取当前位置和姿态
        # current_pos = controller.get_current_tcp()
        # current_quat, current_axis_angle = controller.get_current_orientation()
        
        # print(f"当前位置: {current_pos}")
        # print(f"当前四元数: {current_quat}")
        # print(f"当前轴角: {np.rad2deg(current_axis_angle)}")

        # current_q = controller.get_current_q()
        # print(f"当前关节角度 (q): {np.round(np.rad2deg(current_q), 2)}")
        
        robot_state = controller.get_current_robot_states()
        print(f"------------------------------------------------------------")
        for key, value in robot_state.items():
            print(f"{key}: {value}")

        delta_6dpose = [0.1, -0.2, 0.1, 0.0, 0.0, -np.pi/4]
        current_pose = controller.state['O_T_EE']
        delta_pose = SE3.Trans(delta_6dpose[0], delta_6dpose[1], delta_6dpose[2]) * SE3.RPY(delta_6dpose[3], delta_6dpose[4], delta_6dpose[5])
        target_tmat = SE3(controller.state['O_T_EE']) * delta_pose
        # print(target_tmat)
        # print(target_tmat.A)
        # target_pos = target_tmat.A[:3, 3:]
        # target_quat = T.mat2quat(target_tmat.A[:3, :3])
        # target_axis_angle = T.quat2axisangle(target_quat)
        # target_pose_7d = target_pos.flatten().tolist() + target_axis_angle.flatten().tolist() + [-1.0]
        
        # 示例1: 绝对位置控制
        print("\n=== 示例1: 绝对位置控制 ===")
        # target_pos = current_pos + np.array([0.1, 0.0, 0.0])  # 在x方向移动10cm
        # target_quat = T.euler2quat([0, 0, np.pi/4])  # 绕z轴旋转45度
        controller.move(target_tmat, duration=3.0)
        # controller.move_to_position(target_pos, target_quat, duration=3.0)
        
        # 示例2: 增量控制
        print("\n=== 示例2: 增量控制 ===")
        controller.reset_to_home()
        time.sleep(1.0)
        controller.move_delta(
            delta_pos=np.array([0.1, -0.2, 0.1]),
            delta_rpy=np.array([0.0, 0.0, -np.pi/4]),
            duration=3.0
        )
        # goto_delta_goal
        
        # 示例3: 使用SE3对象
        print("\n=== 示例3: 使用SE3对象 ===")
        controller.reset_to_home()
        time.sleep(1.0)
        current_pose = controller.get_current_pose()
        delta_pose = SE3.Trans(-0.1, -0.3, 0.1) * SE3.RPY(0.0, -np.pi/4, np.pi/4)
        target_pose = SE3(current_pose) * delta_pose
        controller.tcp_move(target_pose, duration=3.0)
        
    except Exception as e:
        print(f"演示过程中出现错误: {str(e)}")
    
    finally:
        # 关闭控制器
        controller.close()



# def demo_usage():
#     """演示如何使用FrankaController类"""
#     # args = parse_args()
    
#     # 创建控制器实例
#     controller = FlexivController()
    
#     try:
#         # 重置到初始位置
#         controller.reset_to_home()
#         time.sleep(1.0)
        
#         # 获取当前位置和姿态
#         current_pos = controller.get_current_tcp()
#         current_quat, current_axis_angle = controller.get_current_orientation()
        
#         print(f"当前位置: {current_pos}")
#         print(f"当前四元数: {current_quat}")
#         print(f"当前轴角: {np.rad2deg(current_axis_angle)}")

#         current_q = controller.get_current_q()
#         print(f"当前关节角度 (q): {np.round(np.rad2deg(current_q), 2)}")
        
#         controller.get_current_robot_states()
#         delta_6dpose = [0.1, -0.2, 0.1, 0.0, 0.0, -np.pi/4]
#         current_pose = controller.state['O_T_EE']
#         delta_pose = SE3.Trans(delta_6dpose[0], delta_6dpose[1], delta_6dpose[2]) * SE3.RPY(delta_6dpose[3], delta_6dpose[4], delta_6dpose[5])
#         target_tmat = SE3(controller.state['O_T_EE']) * delta_pose
#         # print(target_tmat)
#         # print(target_tmat.A)
#         # target_pos = target_tmat.A[:3, 3:]
#         # target_quat = T.mat2quat(target_tmat.A[:3, :3])
#         # target_axis_angle = T.quat2axisangle(target_quat)
#         # target_pose_7d = target_pos.flatten().tolist() + target_axis_angle.flatten().tolist() + [-1.0]
        
#         # 示例1: 绝对位置控制
#         print("\n=== 示例1: 绝对位置控制 ===")
#         # target_pos = current_pos + np.array([0.1, 0.0, 0.0])  # 在x方向移动10cm
#         # target_quat = T.euler2quat([0, 0, np.pi/4])  # 绕z轴旋转45度
#         controller.move(target_tmat, duration=3.0)
#         # controller.move_to_position(target_pos, target_quat, duration=3.0)
        
#         # 示例2: 增量控制
#         print("\n=== 示例2: 增量控制 ===")
#         controller.reset_to_home()
#         time.sleep(1.0)
#         controller.move_delta(
#             delta_pos=np.array([0.1, -0.2, 0.1]),
#             delta_rpy=np.array([0.0, 0.0, -np.pi/4]),
#             duration=3.0
#         )
#         # goto_delta_goal
        
#         # 示例3: 使用SE3对象
#         print("\n=== 示例3: 使用SE3对象 ===")
#         controller.reset_to_home()
#         time.sleep(1.0)
#         current_pose = controller.get_current_pose()
#         delta_pose = SE3.Trans(-0.1, -0.3, 0.1) * SE3.RPY(0.0, -np.pi/4, np.pi/4)
#         target_pose = SE3(current_pose) * delta_pose
#         controller.tcp_move(target_pose, duration=3.0)
        
#     except Exception as e:
#         print(f"演示过程中出现错误: {str(e)}")
    
#     finally:
#         # 关闭控制器
#         controller.close()
def debug():
    current_pose=[[ 0.71919628, -0.69431253 , 0.02620715 , 0.45430258],
                    [-0.69446869, -0.71951144, -0.0040643  , 0.03234122],
                    [ 0.02167824 ,-0.01527702 ,-0.99964827 , 0.36789881],
                    [ 0.     ,     0.  ,        0.     ,     1.        ]]
    current_pose_se3 = SE3(current_pose)
    current_pose_se3.rpy()
    pose_9d = np.array([ 0.45430296,  0.03234144,  0.36789929,  0.71920178, -0.69446295,
        0.02167984, -0.69430669, -0.719517  , -0.01528114])
    '''
    (Pdb) obs['left_robot_tcp_pose'][-1]
    array([ 0.45430296,  0.03234144,  0.36789929,  0.71920178, -0.69446295,
            0.02167984, -0.69430669, -0.719517  , -0.01528114])
    (Pdb) base_absolute_action
    array([ 0.45430296,  0.03234144,  0.36789929,  0.71920178, -0.69446295,
            0.02167984, -0.69430669, -0.719517  , -0.01528114])
    '''
    from reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
    )
    import requests
    from reactive_diffusion_policy.common.space_utils import (
        ortho6d_to_rotation_matrix,
        pose_3d_9d_to_homo_matrix_batch,
        homo_matrix_to_pose_9d_batch,
        matrix4x4_to_pose_6d,
        pose_6d_to_4x4matrix
    )
    obs_pose_mat = pose_3d_9d_to_homo_matrix_batch(pose_9d[np.newaxis,:])
    
    obs_pose_se3 = SE3(obs_pose_mat[0])
    delta_pose = current_pose_se3.inv() @ obs_pose_se3
    theta, v = tr2angvec(delta_pose.R)
    
    print(f"theta: {theta}, v: {v}")
    # print(f"delta_pose: {delta_pose}")
    # print(f"delta_pose.A: {delta_pose.A}")
    # print(f"delta_pose.A[:3,3:]")
    # print(f"delta_pose.A[:3,:3]")
    # print(f"delta_pose.A[:3,3:]")
    # print(f"delta_pose.A[:3,3:]")
    '''
        reactive_diffusion_policy.env_runner.real_runner:action_command_thread:420 - execute_action: 
    [ 5.70955753e-01 -3.97547819e-02  3.46713781e-01 -3.14159000e+00
    0.00000000e+00  3.14159000e+00 -1.37814321e-03  0.00000000e+00
    5.70955753e-01 -3.97547819e-02  3.46713781e-01 -3.14159000e+00
    0.00000000e+00  3.14159000e+00 -1.37814321e-03  0.00000000e+00]
    '''
    # 这是绝对action的6d pose,来源于上面的前 6 维
    pose_6d = [5.70955753e-01, -3.97547819e-02,  3.46713781e-01, -3.14159000e+00, 0.00000000e+00 , 3.14159000e+00]
    import transforms3d as t3d
    def pose_6d_to_pose_7d(pose: np.ndarray) -> np.ndarray:
        # convert 6D pose (x, y, z, r, p, y) to 7D pose (x, y, z, qw, qx, qy, qz)
        quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
        return np.concatenate([pose[:3], quat])
    import numpy as np
    pose_7d = pose_6d_to_pose_7d(pose_6d)
    goal_se3 = SE3(pose_7d[:3])*UnitQuaternion(pose_7d[3:]).SE3()

    # 绝对 action，3 + 6d rot + gripper_width 共 10 维
    tcp_step_action = np.array([[ 4.7675380e-01,  1.0069788e-02,  3.7382472e-01,  6.9767851e-01,
        -7.1600801e-01,  2.4029391e-02, -7.1554691e-01, -6.9809175e-01,
        -2.5699863e-02, -1.7638505e-04],
       [ 4.8484194e-01,  1.7393850e-03,  3.7530276e-01,  6.8980342e-01,
        -7.2363079e-01,  2.3013348e-02, -7.2308886e-01, -6.9018167e-01,
        -2.8136604e-02, -3.2162294e-05],
       [ 4.9311304e-01, -6.9806492e-03,  3.7661573e-01,  6.8174922e-01,
        -7.3126322e-01,  2.1724707e-02, -7.3065650e-01, -6.8207926e-01,
        -3.0146932e-02,  9.2804432e-05],
       [ 5.0123441e-01, -1.5689343e-02,  3.7775943e-01,  6.7362237e-01,
        -7.3878890e-01,  2.0588426e-02, -7.3813301e-01, -6.7390859e-01,
        -3.1731065e-02,  1.7773546e-04],
       [ 5.0890851e-01, -2.3935761e-02,  3.7871405e-01,  6.6571021e-01,
        -7.4594641e-01,  1.9844398e-02, -7.4525338e-01, -6.6596776e-01,
        -3.2927420e-02,  2.0235404e-04]], dtype=np.float32)
    tcp_step_action_9d = tcp_step_action[-1,:9]
    tcp_step_action_mat = pose_3d_9d_to_homo_matrix_batch(tcp_step_action_9d[np.newaxis,:])
    tcp_step_action_se3 = SE3(tcp_step_action_mat[0].tolist())
    delta_pose = current_pose_se3.inv() @ tcp_step_action_se3
    theta, v = tr2angvec(delta_pose.R)
    print(f"theta: {theta*57}, v: {v}")
    
    
    # 绝对值，9dpose 没问题
    # combined_action = np.array([ 5.0890851e-01, -2.3935761e-02,  3.7871405e-01,
    #     -3.10865278e+00, -1.98457015e-02, -8.42175394e-01,
    #      1.67096034e-04,  0.00000000e+00,  5.08908510e-01,
    #     -2.39357613e-02,  3.78714055e-01, -3.10865278e+00,
    #     -1.98457015e-02, -8.42175394e-01,  1.67096034e-04,
    #      0.00000000e+00]])
    # action_all_6d = action_all[:, :6]
    # action_all_6d_mat = pose_6d_to_4x4matrix(action_all_6d[0])
    # action_all_6d_se3 = SE3(action_all_6d_mat.tolist())
    # delta_pose_to_axisangle(current_pose_se3, action_all_6d_se3)
    
    '''
    theta (deg): 177.3042807154596, v: [ 0.50320059  0.49894076 -0.70558294]
    delta_pose (deg): [ 0.  0.  0.]
    '''
    
    rot_6d = combined_action[3:9]
    rot_mat = ortho6d_to_rotation_matrix(rot_6d[np.newaxis,:])
    print(f"rot_mat: {rot_mat}")
    
    rot_mat_so3 = SO3(rot_mat[0].tolist())
    rot_mat_se3 = SE3(rot_mat_so3)
    rot_mat_so3.rpy()
    delta_pose_to_axisangle(current_pose_se3, rot_mat_se3)
    
def delta_pose_to_axisangle(se1: SE3, se2: SE3):
    delta_pose = se1.inv() @ se2
    theta, v = tr2angvec(delta_pose.R)
    print(f"theta (deg): {np.degrees(theta)}, v: {v}")
    print(f"delta_pose (deg): {np.degrees(delta_pose.rpy())}")
    # return np.degrees(theta), v
    
# def delta_pose_to_axisangle(se1, se2):
#     delta_pose = se1.inv() @ se2
#     theta, v = tr2angvec(delta_pose.R)
#     return theta, v
# def teleop_control(): 


def teleop_demo_usage():
    robot = None
    teleop_helper = None
    
    s_key_pressed = threading.Event()
    q_key_pressed = threading.Event()

    def on_press(key):
        try:
            if key.char == 's':
                s_key_pressed.set()
            elif key.char == 'q':
                q_key_pressed.set()
                return False
        except AttributeError:
            pass
        
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    try:
        robot = FlexivController()
        time.sleep(1)
        teleop_helper = HTCTeleopHelper()
        robot.reset_to_home()
        time.sleep(1)
        is_teleop_active = False

        logger.info("\n=======================================================")
        logger.info("按 's' 键开始/停止 HTC 远程控制。")
        logger.info("按 'q' 键退出程序。")
        logger.info("=======================================================")
        
        while not q_key_pressed.is_set():
            robot_state = robot.get_current_robot_states()
            if s_key_pressed.is_set():
                s_key_pressed.clear()
                if not is_teleop_active:
                    try:
                        current_pose = robot.get_current_pose()
                        if teleop_helper.calculate_offset(current_pose):
                            is_teleop_active = True
                            logger.info("远程控制已激活。")
                    except RuntimeError as e:
                        logger.error(f"无法启动远程控制: {e}")
                else:
                    is_teleop_active = False
                    teleop_helper.reset_offset()
                    # robot.pause_control()
                    logger.info("远程控制已停用。")

            if is_teleop_active:
                target_pose_xyz_quat, current_htc_pose_7d = teleop_helper.get_target_pose()
                
                if target_pose_xyz_quat is not None:
                    position = target_pose_xyz_quat[:3]
                    quaternion = target_pose_xyz_quat[3:]
                    # quaternion = current_pose[3:]
                    goal_se3 = SE3(position) * UnitQuaternion(quaternion).SE3()
                    
                    theta, v = tr2angvec(goal_se3.R)
                    axisangle = v * theta
                    current_pose_se3 = SE3(np.array(robot.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
                    delta_pose_to_axisangle(current_pose_se3, goal_se3)
                    print('current_htc_pose_7d', [float(item) for item in current_htc_pose_7d])
                    print('delta_pos',current_pose_se3.t-goal_se3.t, 'current_pose_se3.t', current_pose_se3.t, 'goal_se3.t', goal_se3.t)
                    # print('position_offset', teleop_helper.position_offset)
                    # print('delta_rot',current_pose_se3.R-goal_se3.R)
                    
                    target_pose_for_queue = goal_se3.t.tolist() + axisangle.tolist() + [-1.0]

                    robot.set_target_pose(target_pose_for_queue)
                    
            time.sleep(1.0 / robot.control_frequency)

    except KeyboardInterrupt:
        logger.info("\n检测到 Ctrl+C，正在退出...")
    except Exception as e:
        logger.error(f"程序遇到严重错误，即将退出: {e}")
    finally:
        logger.info("正在关闭所有设备...")
        listener.stop()
        if robot:
            robot.close()
        if teleop_helper:
            teleop_helper.shutdown()
        logger.info("所有设备已安全关闭。")


def get_current_pose():
    """演示如何使用FrankaController类"""
    # args = parse_args()
    
    GRIPPER_PORT = "/dev/ttyACM2"
    FORCE_SENSOR_PORT = '/dev/ttyACM0'

    # 创建控制器实例
    controller = FlexivController(
        left_gripper_port=GRIPPER_PORT,
        force_sensor_port=FORCE_SENSOR_PORT
    )
    
    time.sleep(1.0)
    current_pose = np.array(controller.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
    print(f"current_pose: {current_pose}")
    
    
def demo_usage_thread_reset_to_home():
    """演示如何使用FrankaController类"""
    # args = parse_args()
    
    GRIPPER_PORT = "/dev/ttyACM2"
    FORCE_SENSOR_PORT = '/dev/ttyACM0'

    # 创建控制器实例
    controller = FlexivController(
        left_gripper_port=GRIPPER_PORT,
        force_sensor_port=FORCE_SENSOR_PORT
    )

    controller.reset_to_home()
    time.sleep(1.0)

    controller._start_control_thread()
    # controller.tcp_move_delta([0.1, -0.2, 0.1, 0.0, 0.0, -np.pi/4], duration=3.0)
    # controller.control_thread.start()
    # goal_pose = 
    time.sleep(1.0)
    current_ee_pose_se3 = SE3(np.array(controller.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose())
    robot_state = controller.get_current_robot_states()
    delta_6dpose = [0.1, -0.2, 0.1, 0.0, 0.0, -np.pi/4]
    current_pose = controller.state['O_T_EE']
    delta_pose = SE3.Trans(delta_6dpose[0], delta_6dpose[1], delta_6dpose[2]) * SE3.RPY(delta_6dpose[3], delta_6dpose[4], delta_6dpose[5])
    # goal_pose_se3 = SE3(controller.state['O_T_EE']) * delta_pose
    goal_pose_se3 = SE3(current_ee_pose_se3) * delta_pose
    
    # goal_pose_se3 = SE3(goal_pose, check=False)
    # print(goal_pose_se3)
    duration = 5.0
    num_steps = int(controller.control_frequency * duration)
    
    # print(f"开始运动到目标位置，持续时间: {duration}秒，步数: {num_steps}")
    # last_time = time.time()

if __name__ == "__main__":
    demo_usage()
    # get_current_pose()
    # demo_usage_thread() 
    # teleop_demo_usage()
    # demo_usage_thread_reset_to_home()
    # demo_usage_thread()
    