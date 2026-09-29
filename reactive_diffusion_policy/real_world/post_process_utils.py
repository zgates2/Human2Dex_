import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from typing import Dict
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.data_models import SensorMessage, SensorMode
from reactive_diffusion_policy.common.visualization_utils import visualize_pcd_from_numpy, visualize_rgb_image
from reactive_diffusion_policy.common.space_utils import pose_6d_to_4x4matrix, pose_6d_to_pose_9d
from reactive_diffusion_policy.model.common.pca_embedding import PCAEmbedding
from omegaconf import DictConfig

MM_PER_SINGLE_MOTOR_RADIAN = 30.0
def rad_to_width_m(angle_rad: float) -> float:
    return angle_rad * MM_PER_SINGLE_MOTOR_RADIAN / 1000.0 * 2.0

class DataPostProcessingManager:
    def __init__(self,
                 transforms: RealWorldTransforms,
                #  mode: str = 'single_arm_one_realsense_no_tactile',
                 mode: str = 'single_arm_one_realsense_two_tactile',
                 image_resize_shape: tuple = (320, 240),
                #  image_resize_shape: tuple = (640, 480),
                 use_6d_rotation: bool = True,
                 marker_dimension: int = 2,
                 pca_param_dict: DictConfig = None,
                 debug: bool = False):
        self.transforms = transforms
        self.mode = SensorMode[mode]
        self.use_6d_rotation = use_6d_rotation
        self.marker_dimension = marker_dimension
        self.resize_shape = image_resize_shape
        if pca_param_dict is not None:
            self.pca_embedding_dict = dict()
            for tactile_sensor_name in pca_param_dict.keys():
                self.pca_embedding_dict[tactile_sensor_name] = PCAEmbedding(**pca_param_dict[tactile_sensor_name])
        else:
            self.pca_embedding_dict = None
        self.debug = debug
        
        
    @staticmethod
    def process_fisheye_image(image: np.ndarray) -> np.ndarray:
        x, y, w, h = 10, 0, 460, 460
        cropped = image[y:y+h, x:x+w]
        return cv2.resize(cropped, (240, 240), interpolation=cv2.INTER_AREA)

    def convert_sensor_msg_to_obs_dict(self, sensor_msg: SensorMessage) -> Dict[str, np.ndarray]:
        # import pdb;pdb.set_trace()
        obs_dict = dict()
        obs_dict['timestamp'] = np.array([sensor_msg.timestamp])

        # Add independent key-value pairs for left robot
        obs_dict['left_robot_tcp_pose'] = sensor_msg.leftRobotTCP
        obs_dict['left_robot_tcp_vel'] = sensor_msg.leftRobotTCPVel
        obs_dict['left_robot_tcp_wrench'] = sensor_msg.leftRobotTCPWrench
        obs_dict['left_robot_tcp_xyzrpy'] = sensor_msg.leftRobotTCPTarget
        
        arr = np.asarray(sensor_msg.leftRobotGripperState, dtype=np.float32)

        # array[0,2] is motor with force sensor data, array[2:] is motor with real control motor
        # width_m = rad_to_width_m(float(arr[0]))
        # force_nm = float(arr[1])
        # logger.error(f"width1 : {float(arr[0])}, width2: {float(arr[2])}")
        # width_m = rad_to_width_m(float(arr[2]))
        # force_nm = float(arr[3])

        # for inference (single motor mode)
        # width_m = rad_to_width_m(float(arr[0]))
        # force_nm = float(arr[1])
        # obs_dict['left_robot_gripper_width'] = np.array([width_m], dtype=np.float32)
        # obs_dict['left_robot_gripper_force'] = np.array([force_nm], dtype=np.float32)
        # width_m_for_action = rad_to_width_m(float(arr[0]))
        width_m_for_action = float(arr[0])
        vel_for_action =  float(arr[1])
        force_nm_for_action = float(arr[2])
        
        
        # width_m_for_state = rad_to_width_m(float(arr[2]))
        width_m_for_state = float(arr[3])
        vel_for_state =  float(arr[4])
        force_nm_for_state = float(arr[5])
        obs_dict['left_robot_gripper_width'] = np.array([width_m_for_state], dtype=np.float32)
        # obs_dict['left_robot_gripper_vel_state'] = np.array([vel_for_state], dtype=np.float32)
        obs_dict['left_robot_gripper_force'] = np.array([force_nm_for_state], dtype=np.float32)
        
        
        obs_dict['left_robot_gripper_width_for_action'] = np.array([width_m_for_action], dtype=np.float32)
        obs_dict['left_robot_gripper_torque_for_action'] = np.array([force_nm_for_action], dtype=np.float32)
        obs_dict['left_robot_gripper_vel_for_action'] = np.array([force_nm_for_action], dtype=np.float32)
        
        # import pdb;pdb.set_trace()
        obs_dict['left_robot_tcp_vel'] = np.array([vel_for_action, vel_for_action], dtype=np.float32)
        
        # obs_dict['left_robot_gripper_width'] = sensor_msg.leftRobotGripperState[0][np.newaxis]
        # obs_dict['left_robot_gripper_force'] = sensor_msg.leftRobotGripperState[1][np.newaxis]

        # Add independent key-value pairs for right robot
        # obs_dict['right_robot_tcp_pose'] = sensor_msg.rightRobotTCP
        # obs_dict['right_robot_tcp_vel'] = sensor_msg.rightRobotTCPVel
        # obs_dict['right_robot_tcp_wrench'] = sensor_msg.rightRobotTCPWrench
        # obs_dict['right_robot_gripper_width'] = sensor_msg.rightRobotGripperState[0][np.newaxis]
        # obs_dict['right_robot_gripper_force'] = sensor_msg.rightRobotGripperState[1][np.newaxis]

        if self.use_6d_rotation:
            obs_dict['left_robot_tcp_pose'] = pose_6d_to_pose_9d(sensor_msg.leftRobotTCP)
            # obs_dict['right_robot_tcp_pose'] = pose_6d_to_pose_9d(sensor_msg.rightRobotTCP)

        if self.debug:
            logger.debug(f'left_robot_tcp_pose: {obs_dict["left_robot_tcp_pose"]}, ')
                        #  f'right_robot_tcp_pose: {obs_dict["right_robot_tcp_pose"]}')
            logger.debug(f'left_robot_tcp_vel: {obs_dict["left_robot_tcp_vel"]}, ')
                            # f'right_robot_tcp_vel: {obs_dict["right_robot_tcp_vel"]}')
            logger.debug(f'left_robot_tcp_wrench: {obs_dict["left_robot_tcp_wrench"]}, ')
                            # f'right_robot_tcp_wrench: {obs_dict["right_robot_tcp_wrench"]}')
            logger.debug(f'left_robot_gripper_width: {obs_dict["left_robot_gripper_width"]}, ')
                            # f'right_robot_gripper_width: {obs_dict["right_robot_gripper_width"]}')
            logger.debug(f'left_robot_gripper_force: {obs_dict["left_robot_gripper_force"]}, ')
                            # f'right_robot_gripper_force: {obs_dict["right_robot_gripper_force"]}')

        # TODO: make all sensor post-processing in parallel
        obs_dict['external_img'] = self.resize_image_by_size(sensor_msg.externalCameraRGB, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['external_img'])
        if self.mode == SensorMode.single_arm_one_realsense_no_tactile:
            return obs_dict

        obs_dict['left_wrist_img'] = self.resize_image_by_size(sensor_msg.leftWristCameraRGB, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['left_wrist_img'])
        if self.mode == SensorMode.single_arm_two_realsense_no_tactile:
            return obs_dict
        
        # obs_dict['fisheye_img'] = self.resize_image_by_size(sensor_msg.fisheyeCameraRGB, size=self.resize_shape)
        obs_dict['fisheye_img'] = self.process_fisheye_image(sensor_msg.fisheyeCameraRGB)
        if self.debug:
            visualize_rgb_image(obs_dict['fisheye_img'])
        # visualize_rgb_image(obs_dict['fisheye_img'])
        # cv2.imshow('fisheye_img', obs_dict['fisheye_img'])
        # cv2.waitKey(1)
        if self.mode == SensorMode.single_arm_two_realsense_no_tactile:
            return obs_dict

        obs_dict['left_gripper1_img'] = self.resize_image_by_size(sensor_msg.leftGripperCameraRGB1, size=self.resize_shape)
        obs_dict['left_gripper2_img'] = self.resize_image_by_size(sensor_msg.leftGripperCameraRGB2, size=self.resize_shape)
        if self.debug:
            visualize_rgb_image(obs_dict['left_gripper1_img'])
            visualize_rgb_image(obs_dict['left_gripper2_img'])
        if self.mode == SensorMode.single_arm_two_realsense_two_tactile or self.mode == SensorMode.dual_arm_two_realsense_four_tactile or self.mode == SensorMode.single_arm_one_realsense_two_tactile:
            obs_dict['left_gripper1_initial_marker'] = sensor_msg.leftGripperCameraMarker1
            obs_dict['left_gripper1_marker_offset'] = sensor_msg.leftGripperCameraMarkerOffset1
            obs_dict['left_gripper2_initial_marker'] = sensor_msg.leftGripperCameraMarker2
            obs_dict['left_gripper2_marker_offset'] = sensor_msg.leftGripperCameraMarkerOffset2
            # TODO: more flexible way to choose which tactile sensor to use
            if self.pca_embedding_dict is not None:
                try:
                    obs_dict['left_gripper1_marker_offset_emb'] = self.pca_embedding_dict['GelSight'].pca_reduction(
                        sensor_msg.leftGripperCameraMarkerOffset1.reshape(-1)[np.newaxis, :])[0]
                except ValueError as e:
                    obs_dict['left_gripper1_marker_offset_emb'] = sensor_msg.leftGripperCameraMarkerOffset1.reshape(-1)
                try:
                    # obs_dict['left_gripper2_marker_offset_emb'] = self.pca_embedding_dict['McTac'].pca_reduction(
                    #     sensor_msg.leftGripperCameraMarkerOffset2.reshape(-1)[np.newaxis, :])[0]
                    obs_dict['left_gripper2_marker_offset_emb'] = self.pca_embedding_dict['GelSight'].pca_reduction(
                        sensor_msg.leftGripperCameraMarkerOffset2.reshape(-1)[np.newaxis, :])[0]
                except ValueError as e:
                    obs_dict['left_gripper2_marker_offset_emb'] = sensor_msg.leftGripperCameraMarkerOffset2.reshape(-1)
        else:
            raise NotImplementedError

        # Add synexens depth processing here
        if hasattr(sensor_msg, 'synexensDepth') and sensor_msg.synexensDepth is not None and sensor_msg.synexensDepth.any():
            depth_m = sensor_msg.synexensDepth.astype(np.float32) * 0.001
            depth_m = cv2.resize(depth_m, (240, 240), interpolation=cv2.INTER_LINEAR)
            depth_m = depth_m.astype(np.float16)
            obs_dict['depth_0'] = np.repeat(depth_m[..., np.newaxis], 3, axis=-1)
        else:
            # Fallback if no valid synexensDepth
            obs_dict['depth_0'] = np.zeros((240, 240, 3), dtype=np.float16)

        # obs_dict['right_wrist_img'] = self.resize_image_by_size(sensor_msg.rightWristCameraRGB, size=self.resize_shape)
        # obs_dict['right_gripper1_img'] = self.resize_image_by_size(sensor_msg.rightGripperCameraRGB1, size=self.resize_shape)
        # obs_dict['right_gripper2_img'] = self.resize_image_by_size(sensor_msg.rightGripperCameraRGB2, size=self.resize_shape)
        # if self.debug:
        #     visualize_rgb_image(obs_dict['right_wrist_img'])
        #     visualize_rgb_image(obs_dict['right_gripper1_img'])
        #     visualize_rgb_image(obs_dict['right_gripper2_img'])
        # if self.mode == SensorMode.dual_arm_two_realsense_four_tactile:
        #     obs_dict['right_gripper1_initial_marker'] = sensor_msg.rightGripperCameraMarker1
        #     obs_dict['right_gripper1_marker_offset'] = sensor_msg.rightGripperCameraMarkerOffset1
        #     obs_dict['right_gripper2_initial_marker'] = sensor_msg.rightGripperCameraMarker2
        #     obs_dict['right_gripper2_marker_offset'] = sensor_msg.rightGripperCameraMarkerOffset2
        #     # TODO: more flexible way to choose which tactile sensor to use
        #     if self.pca_embedding_dict is not None:
        #         try:
        #             obs_dict['right_gripper1_marker_offset_emb'] = self.pca_embedding_dict['GelSight'].pca_reduction(
        #                 sensor_msg.rightGripperCameraMarkerOffset1.reshape(-1)[np.newaxis, :])[0]
        #         except ValueError as e:
        #             obs_dict['right_gripper1_marker_offset_emb'] = sensor_msg.rightGripperCameraMarkerOffset1.reshape(-1)
        #         try:
        #             obs_dict['right_gripper2_marker_offset_emb'] = self.pca_embedding_dict['McTac'].pca_reduction(
        #                 sensor_msg.rightGripperCameraMarkerOffset2.reshape(-1)[np.newaxis, :])[0]
        #         except ValueError as e:
        #             obs_dict['right_gripper2_marker_offset_emb'] = sensor_msg.rightGripperCameraMarkerOffset2.reshape(-1)

        #     return obs_dict
        # else:
        #     raise NotImplementedError

        return obs_dict

    @staticmethod
    def resize_image_by_size(image: np.ndarray, size: tuple) -> np.ndarray:
        return cv2.resize(image, size)
