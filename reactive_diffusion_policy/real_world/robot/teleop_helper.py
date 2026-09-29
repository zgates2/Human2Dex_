import numpy as np
import time
from reactive_diffusion_policy.real_world.robot.vive_tracker import ViveForceSensorTracker
from loguru import logger
from transformations import quaternion_matrix, quaternion_from_matrix

from spatialmath import SE3, UnitQuaternion

    # def _pose_to_matrix(pos, quat_wxyz):
    #     T = tft.quaternion_matrix(quat_wxyz)
    #     T[:3, 3] = pos
    #     return T
    
from reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
# import requests
from reactive_diffusion_policy.common.space_utils import (
    ortho6d_to_rotation_matrix,
    pose_3d_9d_to_homo_matrix_batch,
    homo_matrix_to_pose_9d_batch,
    matrix4x4_to_pose_6d,
    pose_7d_to_4x4matrix
)

class HTCTeleopHelper:
    def __init__(self):
        try:
            self.vive_tracker = ViveForceSensorTracker()
            logger.info("HTC Teleop Helper 初始化成功。")
        except Exception as e:
            logger.info(f"初始化 Vive Tracker 失败: {e}")
            self.vive_tracker = None
        
        self.T_offset = None  # 直接存储变换矩阵

    def calculate_offset(self, robot_current_pose_7d: list):
        if self.vive_tracker is None:
            raise RuntimeError("Vive Tracker 未初始化，无法计算偏移量。请检查设备连接。")

        logger.info("--- 正在计算远程操作偏移量 ---")
        
        robot_pos_initial = np.array(robot_current_pose_7d[:3])
        robot_quat_initial = np.array(robot_current_pose_7d[3:])  # [w, x, y, z]
        # logger.info(f"已获取机器人初始位置: {robot_pos_initial}")
        # logger.info(f"已获取机器人初始旋转: {robot_quat_initial}")

        logger.info("正在获取HTC Vive数据以进行对齐...")
        for i in range(50):
            htc_pose_7d = self.vive_tracker.get_force_sensor_pose()
            htc_pos_initial = np.array(htc_pose_7d[:3])
            htc_quat_initial = np.array(htc_pose_7d[3:])  # [w, x, y, z]
            # logger.info(f"已获取HTC初始位置: {htc_pos_initial}")
            # logger.info(f"已获取HTC初始旋转: {htc_quat_initial}")
            
            # 直接计算变换矩阵offset：T_robot * T_htc^(-1)
            # 1. 将四元数转换为旋转矩阵
            # 机器人初始位姿的变换矩阵
            T_robot = quaternion_matrix(robot_quat_initial)
            T_robot[:3, 3] = robot_pos_initial
            
            # HTC初始位姿的变换矩阵
            T_htc = quaternion_matrix(htc_quat_initial)
            T_htc[:3, 3] = htc_pos_initial
            
            # 2. 计算相对变换：T_robot * T_htc^(-1)
            # 这表示从HTC坐标系到机器人坐标系的变换
            T_htc_inv = np.linalg.inv(T_htc)
            self.T_offset = np.dot(T_robot, T_htc_inv)
            
            # logger.info(f"变换矩阵offset计算完成:")
            # logger.info(f"位置部分: {self.T_offset[:3, 3]}")
            # logger.info(f"旋转部分:\n{self.T_offset[:3, :3]}")
            # logger.info(f"完整变换矩阵:\n{self.T_offset}")

        return True

    def get_target_pose(self) -> list | None:
        if self.T_offset is None or self.vive_tracker is None:
            logger.info("\r[警告] 远程操作未激活或未初始化，无法计算目标位姿。", end="")
            return None
        
        current_htc_pose_7d = self.vive_tracker.get_force_sensor_pose()
        htc_pos_current = np.array(current_htc_pose_7d[:3])
        htc_quat_current = np.array(current_htc_pose_7d[3:])  # [w, x, y, z]
        
        # 使用变换矩阵方法计算目标位姿
        # 1. 构建HTC当前位姿的变换矩阵
        T_htc_current = quaternion_matrix(htc_quat_current)
        T_htc_current[:3, 3] = htc_pos_current
        
        # 2. 计算目标机器人位姿：T_offset * T_htc_current
        T_target = np.dot(self.T_offset, T_htc_current)
        
        # 3. 提取目标位置和旋转
        target_robot_pos = T_target[:3, 3]
        target_robot_quat = quaternion_from_matrix(T_target)
        
        # 归一化目标旋转
        # target_robot_quat = target_robot_quat / np.linalg.norm(target_robot_quat)
        
        final_target_pose_7d = np.concatenate([target_robot_pos, target_robot_quat]).tolist()
        
        return final_target_pose_7d, current_htc_pose_7d

    def reset_offset(self):
        """重置偏移量状态。"""
        self.T_offset = None  # 重置变换矩阵
        logger.info("远程操作偏移量已重置。")

    def shutdown(self):
        """安全关闭Vive Tracker。"""
        if self.vive_tracker:
            self.vive_tracker.shutdown()