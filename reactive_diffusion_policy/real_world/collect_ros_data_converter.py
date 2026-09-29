# Filename: data_converter.py
# -- coding: UTF-8
# --- FINAL VERSION with Point Cloud generation logic ---

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import transformations as tf
import copy
from typing import Dict, Tuple, List, Optional

from sensor_msgs.msg import Image, CompressedImage, PointCloud2, JointState, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped, Pose, Twist

from sensor_msgs_py import point_cloud2
from numpy.lib.recfunctions import structured_to_unstructured

from reactive_diffusion_policy.common.space_utils import ros_pose_to_6d_pose
from reactive_diffusion_policy.common.data_models import SensorMessage
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms

MM_PER_SINGLE_MOTOR_RADIAN = 46.5641

class ROS2DataConverter:
    def __init__(self, camera_intrinsics: dict, depth_scale: float = 0.001):
        """
        Initializes the DataConverter.
        :param camera_intrinsics: A dictionary containing camera intrinsic parameters (fx, fy, cx, cy).
        :param depth_scale: The scale factor to convert depth values to meters (e.g., 0.001 for RealSense).
        """
        self.bridge = CvBridge()
        self.marker_fields = ['marker_location_x', 'marker_location_y', 'marker_offset_x', 'marker_offset_y']
        self.GS_MARKER_EXPECTED_WIDTH = 63
        
        self.cam_intrinsics = camera_intrinsics
        self.cam_fx = self.cam_intrinsics['fx']
        self.cam_fy = self.cam_intrinsics['fy']
        self.cam_cx = self.cam_intrinsics['cx']
        self.cam_cy = self.cam_intrinsics['cy']
        self.depth_scale = depth_scale

    def _get_left_gripper_state(self, joint_states_msg: JointState):
        position = 0.0
        effort = 0.0

        if len(joint_states_msg.position) >= 2:
            position = joint_states_msg.position[1]
        
        if len(joint_states_msg.effort) >= 2:
            effort = joint_states_msg.effort[1]
            
        return np.array([position, effort], dtype=np.float16)

    def _create_colored_point_cloud_from_depth(self, depth_image, color_image):
        """
        根据对齐的深度图像和彩色图像，以及相机内参生成带颜色的点云 (XYZRGB)。
        """
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        u, v = u.flatten(), v.flatten()
        d = depth_image.flatten()
        
        valid_indices = d > 0
        u, v, d = u[valid_indices], v[valid_indices], d[valid_indices]
        
        z_cam = d.astype(np.float32) * self.depth_scale
        x_cam = (u - self.cam_cx) * z_cam / self.cam_fx
        y_cam = (v - self.cam_cy) * z_cam / self.cam_fy
        
        points_3d = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        colors = color_image[v, u]
        
        colors_float = colors.astype(np.float32) / 255.0
        
        points_6d = np.hstack([points_3d, colors_float])

        return points_6d.astype(np.float32)

    def _quat_to_euler(self, quat_xyzw):
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]] # wxyz
        return tf.euler_from_quaternion(quat_wxyz, 'sxyz')

    def convert_pose_to_array(self, pose: Pose):
        pos = pose.position; ori = pose.orientation
        return np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w], dtype=np.float32)

    def convert_wrench_to_array(self, wrench_stamped: WrenchStamped):
        force = wrench_stamped.wrench.force; torque = wrench_stamped.wrench.torque
        return np.array([force.x, force.y, force.z, torque.x, torque.y, torque.z], dtype=np.float32)

    def _convert_ros_image_to_rgb(self, image_msg, sensor_name="Image"):
        """
        [新增/恢复的函数] 使用CvBridge转换标准的、未压缩的ROS Image消息。
        """
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except CvBridgeError as e:
            print(f"ERROR: CvBridge failed to convert {sensor_name}: {e}"); raise
    
    def _convert_compressed_image_to_rgb(self, compressed_image_msg: CompressedImage, sensor_name="CompressedImage"):
        """
        [关键修正] 解码标准的 CompressedImage 消息。
        """
        try:
            np_arr = np.frombuffer(compressed_image_msg.data, np.uint8)
            bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr_image is None:
                raise CvBridgeError(f"cv2.imdecode failed for {sensor_name}")
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"ERROR: Failed to convert {sensor_name}: {e}"); raise
    
    def _convert_compressed_depth_to_cv2(self, compressed_depth_msg: CompressedImage):
        """
        解码包含在CompressedImage消息中的PNG深度数据。
        """
        try:
            np_arr = np.frombuffer(compressed_depth_msg.data, np.uint8)
            depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise CvBridgeError("cv2.imdecode failed for depth image")
            return depth_image
        except Exception as e:
            print(f"ERROR: Failed to convert compressed depth image: {e}"); raise

    def _process_marker_pointcloud(self, pc_msg: PointCloud2, pc_name: str):
        if pc_msg.width == 0:
            nan_array = np.full((self.GS_MARKER_EXPECTED_WIDTH, 3), np.nan, dtype=np.float32)
            return nan_array, nan_array

        data = np.frombuffer(pc_msg.data, dtype=np.float32).reshape(-1, 4)
        
        marker_locations_2d = copy.deepcopy(data[:, :2])
        marker_offsets_2d = copy.deepcopy(data[:, 2:4])

        num_points = data.shape[0]
        locations_3d = np.zeros((num_points, 3), dtype=np.float32)
        locations_3d[:, :2] = marker_locations_2d
        
        offsets_3d = np.zeros((num_points, 3), dtype=np.float32)
        offsets_3d[:, :2] = marker_offsets_2d

        def align_array(arr):
            n_pts = arr.shape[0]
            if n_pts == self.GS_MARKER_EXPECTED_WIDTH:
                return arr
            if n_pts > self.GS_MARKER_EXPECTED_WIDTH:
                return arr[:self.GS_MARKER_EXPECTED_WIDTH, :]
            else:
                padding = np.full((self.GS_MARKER_EXPECTED_WIDTH - n_pts, 3), 0.0, dtype=np.float32)
                return np.vstack((arr, padding))

        return align_array(locations_3d), align_array(offsets_3d)


    def convert_robot_states(self, topic_dict: Dict) -> (
            Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_tcp_pose: PoseStamped = topic_dict['left_tcp_pose']
        # right_tcp_pose: PoseStamped = topic_dict['/right_tcp_pose']

        left_gripper_state: JointState = topic_dict['left_gripper_state']
        # right_gripper_state: JointState = topic_dict['/right_gripper_state']

        left_tcp_vel: TwistStamped = topic_dict['left_tcp_vel']
        # right_tcp_vel: TwistStamped = topic_dict['/right_tcp_vel']

        left_tcp_wrench: WrenchStamped = topic_dict['left_tcp_wrench']
        # right_tcp_wrench: WrenchStamped = topic_dict['/right_tcp_wrench']

        left_tcp_pose_array = ros_pose_to_6d_pose(left_tcp_pose.pose)
        # right_tcp_pose_array = ros_pose_to_6d_pose(right_tcp_pose.pose)

        left_tcp_vel_array = np.array([left_tcp_vel.twist.linear.x, left_tcp_vel.twist.linear.y, left_tcp_vel.twist.linear.z,
                                 left_tcp_vel.twist.angular.x, left_tcp_vel.twist.angular.y,
                                 left_tcp_vel.twist.angular.z])
        # right_tcp_vel_array = np.array(
        #     [right_tcp_vel.twist.linear.x, right_tcp_vel.twist.linear.y, right_tcp_vel.twist.linear.z,
        #      right_tcp_vel.twist.angular.x, right_tcp_vel.twist.angular.y, right_tcp_vel.twist.angular.z])

        left_tcp_wrench_array = np.array(
            [left_tcp_wrench.wrench.force.x, left_tcp_wrench.wrench.force.y, left_tcp_wrench.wrench.force.z,
             left_tcp_wrench.wrench.torque.x, left_tcp_wrench.wrench.torque.y, left_tcp_wrench.wrench.torque.z])
        # right_tcp_wrench_array = np.array(
        #     [right_tcp_wrench.wrench.force.x, right_tcp_wrench.wrench.force.y, right_tcp_wrench.wrench.force.z,
        #      right_tcp_wrench.wrench.torque.x, right_tcp_wrench.wrench.torque.y, right_tcp_wrench.wrench.torque.z])

        left_gripper_state_array = np.array([left_gripper_state.position[0], left_gripper_state.effort[0]])
        # right_gripper_state_array = np.array([right_gripper_state.position[0], right_gripper_state.effort[0]])

        # return (left_tcp_pose_array, right_tcp_pose_array, left_tcp_vel_array, right_tcp_vel_array,
        #         left_tcp_wrench_array, right_tcp_wrench_array, left_gripper_state_array, right_gripper_state_array)
        return (left_tcp_pose_array, left_tcp_vel_array, left_tcp_wrench_array, left_gripper_state_array)
    
    def convert_synced_messages_to_data(self, synced_messages: dict, odometry_offset: np.ndarray) -> SensorMessage:
    # def convert_all_data(self, synced_messages: dict, odometry_offset: np.ndarray) -> SensorMessage:
        try:
            latest_timestamp = max([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                                for msg in synced_messages.values()])

            # odom_msg = synced_messages['baton_odometry']
            # pose_arr = self.convert_pose_to_array(odom_msg.pose.pose) # x, y, z, x, y, z, w
            # pos = pose_arr[0:3] - odometry_offset
            # quat_xyzw = pose_arr[3:7]
            # euler_rpy = self._quat_to_euler(quat_xyzw)
            
            # left_tcp_pose = np.concatenate([pos, euler_rpy]).astype(np.float32)
            # compensated_wrench = self.convert_wrench_to_array(synced_messages['force_compensated_wrench'])
            
            (left_tcp_pose_array, 
             left_tcp_vel_array, 
             left_tcp_wrench_array, 
             left_gripper_state_array) = self.convert_robot_states(synced_messages)
            
            external_rgb = self._convert_compressed_image_to_rgb(synced_messages['fisheye_rgb'], "External Camera")
            # wrist_depth_image = self._convert_compressed_depth_to_cv2(synced_messages['wrist_camera_depth'])
            wrist_color_image = self._convert_compressed_image_to_rgb(synced_messages['wrist_camera_color'], "Wrist Camera Color")

            # left_wrist_point_cloud = self._create_colored_point_cloud_from_depth(wrist_depth_image, wrist_color_image)

            left_gripper_locations, left_gripper_offsets = self._process_marker_pointcloud(synced_messages['gsmini_0_points'], "Left Gripper PC")
            right_gripper_locations, right_gripper_offsets = self._process_marker_pointcloud(synced_messages['gsmini_1_points'], "Right Gripper PC")
        
            # left_gripper_state = self._get_left_gripper_state(synced_messages['motor_states'])

            # timestamp_sec = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec / 1e9
            
            return SensorMessage(
                timestamp=latest_timestamp,
                externalCameraRGB=external_rgb,
                # leftWristCameraPointCloud=left_wrist_point_cloud,
                leftWristCameraRGB=wrist_color_image,
                leftGripperCameraMarker1=left_gripper_locations,
                leftGripperCameraMarkerOffset1=left_gripper_offsets,
                rightGripperCameraMarker1=right_gripper_locations,
                rightGripperCameraMarkerOffset1=right_gripper_offsets,
                leftRobotTCP=left_tcp_pose_array,
                leftRobotTCPVel=left_tcp_vel_array,
                leftRobotTCPWrench=left_tcp_wrench_array,
                leftGripperState=left_gripper_state_array,
                # leftRobotTCP=left_tcp_pose,
                # leftRobotTCPWrench=compensated_wrench,
                # leftGripperState=left_gripper_state,
            )
        except Exception as e:
            print(f"ERROR in DataConverter during processing: {e}")
            import traceback
            traceback.print_exc()
            return None
