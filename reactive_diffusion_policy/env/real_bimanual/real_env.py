import pickle
import os
import os.path as osp
import datetime
import threading

import cv2
import time
import torch
import numpy as np
import requests
from omegaconf import DictConfig
from copy import deepcopy
from typing import Union, List, Dict, Optional
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from collections import deque
from termcolor import colored

from loguru import logger
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.real_world.device_mapping.device_mapping_utils import get_topic_and_type
from reactive_diffusion_policy.real_world.device_mapping.device_mapping_server import DeviceToTopic
from reactive_diffusion_policy.real_world.ros_data_converter import ROS2DataConverter
from reactive_diffusion_policy.common.data_models import SensorMessage, SensorMessageList, BimanualRobotStates
from reactive_diffusion_policy.common.time_utils import convert_ros_time_to_float
from reactive_diffusion_policy.common.ring_buffer import RingBuffer
from reactive_diffusion_policy.real_world.post_process_utils import DataPostProcessingManager
from reactive_diffusion_policy.common.space_utils import (pose_6d_to_pose_7d, pose_6d_to_4x4matrix, matrix4x4_to_pose_6d)

from reactive_diffusion_policy.real_world.ros_data_converter import ( CompressedImage, WrenchStamped, PointCloud2, JointState,
    PoseStamped, TwistStamped
)

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import pyinstrument

def stack_last_n_obs(all_obs, n_steps: int) -> Union[np.ndarray, torch.Tensor]:
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    if isinstance(all_obs[0], np.ndarray):
        result = np.zeros((n_steps,) + all_obs[-1].shape,
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    elif isinstance(all_obs[0], torch.Tensor):
        result = torch.zeros((n_steps,) + all_obs[-1].shape,
            dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = torch.stack(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
    else:
        raise RuntimeError(f'Unsupported obs type {type(all_obs[0])}')
    return result


class RealRobotEnvironment(Node):
    start_gripper_interval_control: bool = False
    gripper_interval_count: int = 0
    last_gripper_width_target: List[float] = [0.1, 0.1]
    def __init__(self,
                 robot_server_ip: str,
                 robot_server_port: int,
                 transforms: RealWorldTransforms,
                 device_mapping_server_ip: str,
                 device_mapping_server_port: int,
                 data_processing_params: DictConfig,
                 max_fps: int = 30,
                 # gripper control parameters
                 use_force_control_for_gripper: bool = True,
                #  max_gripper_width: float = 0.05,
                 max_gripper_width: float = 0.06,
                 min_gripper_width: float = 0.,
                 grasp_force: float = 5.0,
                 enable_gripper_interval_control: bool = False,
                 gripper_control_time_interval: float = 60,
                 gripper_control_width_precision: float = 0.02,
                 gripper_width_threshold: float = 0.04,
                 enable_gripper_width_clipping: bool = True,
                 enable_exp_recording: bool = False,
                 output_dir: Optional[str] = None,
                 vcamera_server_ip: Optional[str] = None,
                 vcamera_server_port: Optional[int] = None,
                 time_check: bool = False,
                 debug: bool = False,
                 **kwargs):
        # import pdb;pdb.set_trace()
        super().__init__('real_env')
        self.robot_server_ip = robot_server_ip
        self.robot_server_port = robot_server_port
        self.transforms = transforms
        self.max_fps = max_fps

        # gripper control parameters
        self.use_force_control_for_gripper = use_force_control_for_gripper
        self.max_gripper_width = max_gripper_width
        self.min_gripper_width = min_gripper_width
        self.grasp_force = grasp_force
        self.enable_gripper_interval_control = enable_gripper_interval_control
        self.gripper_control_time_interval = gripper_control_time_interval
        self.gripper_control_width_precision = gripper_control_width_precision
        self.gripper_width_threshold = gripper_width_threshold
        self.enable_gripper_width_clipping = enable_gripper_width_clipping

        self.data_processing_manager = DataPostProcessingManager(transforms,
                                                                 **data_processing_params)
        self.debug = debug
        self.subscribers = []
        self.obs_buffer = RingBuffer(size=1024, fps=max_fps)

        self.mutex = threading.Lock()

        self.odometry_offset = np.zeros(3, dtype=np.float32)

        self.enable_exp_recording = enable_exp_recording
        if self.enable_exp_recording:
            assert output_dir is not None, "output_dir must be provided for experiment recording"
            assert vcamera_server_ip is not None and vcamera_server_port is not None, "vcamera_server_ip and vcamera_server_port must be provided for experiment recording"
        self.exp_dir = osp.join(output_dir, 'exp_data') if output_dir is not None else None
        self.vcamera_server_ip = vcamera_server_ip
        self.vcamera_server_port = vcamera_server_port
        self.predicted_full_tcp_action_buffer = RingBuffer(size=1024, fps=max_fps)
        self.predicted_full_gripper_action_buffer = RingBuffer(size=1024, fps=max_fps)
        self.predicted_partial_tcp_action_buffer = RingBuffer(size=1024, fps=max_fps)
        self.predicted_partial_gripper_action_buffer = RingBuffer(size=1024, fps=max_fps)
        self.sensor_msg_list: SensorMessageList = SensorMessageList(sensorMessages=[])

        # Video recording for fisheye camera
        self.video_writer = None
        self.video_recording = False
        self.video_recording_lock = threading.Lock()

        logger.debug("Initializing RealEnv node...")
        # Get device to topic mapping
        response = requests.get(
            f"http://{device_mapping_server_ip}:{device_mapping_server_port}/get_mapping")
        self.device_to_topic_mapping = DeviceToTopic.model_validate(response.json())

        subs_name_type = get_topic_and_type(self.device_to_topic_mapping)
        # import pdb;pdb.set_trace()
        depth_camera_point_cloud_topic_names: List[Optional[str]] = [None, None, None]  # external, left wrist, right wrist
        depth_camera_rgb_topic_names: List[Optional[str]] = [None, None, None]  # external, left wrist, right wrist
        depth_camera_depth_topic_names: List[Optional[str]] = [None, None, None] # external, left wrist, right wrist
        tactile_camera_rgb_topic_names: List[Optional[str]] = [None, None, None, None]  # left gripper1, left gripper2, right gripper1, right gripper2
        tactile_camera_marker_topic_names: List[Optional[str]] = [None, None, None, None]  # left gripper1, left gripper2, right gripper1, right gripper2
        fisheye_camera_topic_names: List[Optional[str]] = [None] # fisheye
        synexens_camera_depth_topic_names: List[Optional[str]] = [None]
        synexens_camera_ir_topic_names: List[Optional[str]] = [None]
        
        for topic, msg_type in subs_name_type:
            if "depth/points" in topic:
                if "external_camera" in topic:
                    depth_camera_point_cloud_topic_names[0] = topic
                elif "left_wrist_camera" in topic:
                    depth_camera_point_cloud_topic_names[1] = topic
                elif "right_wrist_camera" in topic:
                    depth_camera_point_cloud_topic_names[2] = topic
            elif "color/image_raw" in topic:
                if "gripper_camera" in topic:
                    if "left_gripper_camera_1" in topic:
                        tactile_camera_rgb_topic_names[0] = topic
                    elif "left_gripper_camera_2" in topic:
                        tactile_camera_rgb_topic_names[1] = topic
                    elif "right_gripper_camera_1" in topic:
                        tactile_camera_rgb_topic_names[2] = topic
                    elif "right_gripper_camera_2" in topic:
                        tactile_camera_rgb_topic_names[3] = topic
                else:
                    if "external_camera" in topic:
                        depth_camera_rgb_topic_names[0] = topic
                    elif "left_wrist_camera" in topic:
                        depth_camera_rgb_topic_names[1] = topic
                    elif "right_wrist_camera" in topic:
                        depth_camera_rgb_topic_names[2] = topic
            elif "depth/image_raw" in topic:
                if "external_camera" in topic:
                    depth_camera_depth_topic_names[0] = topic
                elif "left_wrist_camera" in topic:
                    depth_camera_depth_topic_names[1] = topic
                elif "right_wrist_camera" in topic:
                    depth_camera_depth_topic_names[2] = topic
                elif "synexens" in topic:
                    synexens_camera_depth_topic_names[0] = topic
            elif "ir/image_raw" in topic:
                if "synexens" in topic:
                    synexens_camera_ir_topic_names[0] = topic
            elif "marker_offset/information" in topic:
                if "left_gripper_camera_1" in topic:
                    tactile_camera_marker_topic_names[0] = topic
                elif "left_gripper_camera_2" in topic:
                    tactile_camera_marker_topic_names[1] = topic
                elif "right_gripper_camera_1" in topic:
                    tactile_camera_marker_topic_names[2] = topic
                elif "right_gripper_camera_2" in topic:
                    tactile_camera_marker_topic_names[3] = topic
            elif "image_raw/compressed" in topic:
                fisheye_camera_topic_names[0] = topic

        self.time_check = time_check
        self.timestamps = {name: [] for name, _ in get_topic_and_type(self.device_to_topic_mapping)}

        

        # self.topic_info = {
        #     # 'baton_odometry': ('/vive/stereo3/odometry', Odometry),
        #     # 'wrist_camera_depth': ('/external_camera/depth/image_raw/compressed', CompressedImage),
        #     'wrist_camera_color': ('/external_camera/color/image_raw/compressed', CompressedImage),
        #     'fisheye_rgb': ('/fisheye/rgb/image_raw', CompressedImage),
        #     # 'compensated_wrench': ('/force_sensor/compensated_wrench', WrenchStamped),
        #     # 'raw_wrench': ('/force_sensor/wrench', WrenchStamped),
        #     'gsmini_0_points': ('/left_gripper_camera_1/marker_offset/information', PointCloud2),
        #     'gsmini_1_points': ('/right_gripper_camera_1/marker_offset/information', PointCloud2),
        #     # 'motor_joint_states': ('/motor_joint_states', JointState),
        #     'camera_info': ('/external_camera/depth/camera_info', CameraInfo),

        #     'left_gripper_state': ('/left_gripper_state', JointState),
        #     'left_tcp_pose': ('/left_tcp_pose', PoseStamped),
        #     'left_tcp_vel': ('/left_tcp_vel', TwistStamped),
        #     'left_tcp_wrench': ('/left_tcp_wrench', WrenchStamped),
        # }

        # try:
        #     cam_info_topic = self.topic_info['camera_info'][0]
        #     logger.info(f"Waiting for camera intrinsics on topic: {cam_info_topic}")
        #     cam_info_msg = self.wait_for_first_message(self, cam_info_topic, CameraInfo, timeout_sec=10.0)
        #     intrinsics = { 'fx': cam_info_msg.k[0], 'fy': cam_info_msg.k[4], 'cx': cam_info_msg.k[2], 'cy': cam_info_msg.k[5] }
        #     logger.info(f"Successfully received camera intrinsics: {intrinsics}")
        # except Exception as e:
        #     logger.error(f"Failed to get camera intrinsics. Aborting. Error: {e}")
        #     raise e

        # self.data_converter = ROS2DataConverter(camera_intrinsics=intrinsics, depth_scale=0.001)

        # self.subscribers = []

        # self.sync_topic_keys = [
        #     'baton_odometry', 'wrist_camera_depth', 'wrist_camera_color', 'fisheye_rgb', 
        #     'compensated_wrench', 'raw_wrench', 'gsmini_0_points', 'gsmini_1_points', 'motor_joint_states'
        # ]
        # self.sync_topic_keys = [
        #     'wrist_camera_color', 
        #     'fisheye_rgb', 
        #     'gsmini_0_points', 
        #     'gsmini_1_points',
        #     'left_gripper_state',
        #     'left_tcp_pose',
        #     'left_tcp_vel',
        #     'left_tcp_wrench'
        # ]

        # subs = [Subscriber(self, self.topic_info[key][1], self.topic_info[key][0]) for key in self.sync_topic_keys]
        # for sub in subs:
        #     logger.debug(f"Subscribing to: {sub.topic}")
        #     self.subscribers.append(sub)

        # for calculating FPS
        self.prev_time = time.time()
        self.frame_count = 0

        if self.debug:
            logger.debug(f"Depth camera point cloud topic names: {depth_camera_point_cloud_topic_names}")
            logger.debug(f"Depth camera rgb topic names: {depth_camera_rgb_topic_names}")
            logger.debug(f"Tactile camera rgb topic names: {tactile_camera_rgb_topic_names}")
            logger.debug(f"Tactile camera marker topic names: {tactile_camera_marker_topic_names}")

        self.data_converter = ROS2DataConverter(depth_camera_point_cloud_topic_names,
                                                depth_camera_rgb_topic_names,
                                                depth_camera_depth_topic_names,
                                                tactile_camera_rgb_topic_names,
                                                tactile_camera_marker_topic_names,
                                                fisheye_camera_topic_names,
                                                synexens_camera_depth_topic_names,
                                                synexens_camera_ir_topic_names,
                                                debug=self.debug)
        
        # logger.info("data convert is success")
 
        for name, msg_type in subs_name_type:
            self.subscribers.append(Subscriber(self, msg_type, name))
            logger.debug(f"Subscribed to topic: {name} with type: {msg_type}")

        # ApproximateTimeSynchronizer is used to synchronize multiple topics
        self.ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=40, slop=0.4,
                                              allow_headerless=False)
        # self.ts = ApproximateTimeSynchronizer(self.subscribers, queue_size=40, slop=0.1, allow_headerless=False)

        self.ts.registerCallback(self.callback)

        # Create a session with robot server
        self.session = requests.session()

    def send_command(self, endpoint: str, data: dict = None):
        url = f"http://{self.robot_server_ip}:{self.robot_server_port}{endpoint}"
        # print("********************************")
        # print(url)
        # print("********************************")
        if 'get' in endpoint:
            response = self.session.get(url)
        else:
            if 'move' in endpoint:
                # low-level control commands
                try:
                    response = self.session.post(url, json=data, timeout=0.001)
                except requests.exceptions.ReadTimeout:
                    # Ignore the timeout error for low-level control commands to reduce latency
                    # TODO: use a more robust way to handle the timeout error
                    response = None
            else:
                response = self.session.post(url, json=data)
        if response is not None:
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        else:
            return dict()

    # @pyinstrument.profile()
    def callback(self, *msgs):
        # logger.info("the callback is success trigger")
        topic_dict = dict()
        for i, msg in enumerate(msgs):
            topic_name = self.subscribers[i].topic
            topic_dict[topic_name] = msg

        if self.time_check:
            # check the time differences across topics and interval between time stamps
            for i, msg in enumerate(msgs):
                topic_name = self.subscribers[i].topic
                self.timestamps[topic_name].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        if self.debug:
        # if True:
            # calculate the lastest timestamp in the topic_dict
            latest_timestamp = max([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 for msg in msgs])
            # convert current time (ROS time) to python time
            current_timestamp = convert_ros_time_to_float(self.get_clock().now())
            # find out the latency compared to current time
            latency = current_timestamp - latest_timestamp
            # the latency is approximately 5ms - 10ms (~5000 points per pcd)
            # the latency is approximately 10ms - 26ms (without point cloud)
            logger.debug(f"Latency for time synchronizer: {latency:.4f} seconds")

        # this part takes about 10ms - 15ms for now (~5000 points per pcd)
        # this part takes about 3ms (with RGB only) for now
        # import pdb;pdb.set_trace()
        sensor_msg: SensorMessage = self.data_converter.convert_all_data(topic_dict)
        # logger.info(f"--- DataConverter Output ---\n{sensor_msg}\n--------------------------")

        # convert sensor msg to obs dict
        # this part takes about 2ms (without point cloud)
        raw_obs_dict = self.data_processing_manager.convert_sensor_msg_to_obs_dict(sensor_msg)

        self.obs_buffer.push(raw_obs_dict)

        # Write frames to video if recording is active
        with self.video_recording_lock:
            if self.video_recording:
                if sensor_msg is not None:
                    # Fisheye recording
                    if getattr(sensor_msg, 'fisheyeCameraRGB', None) is not None:
                        frame = sensor_msg.fisheyeCameraRGB
                        # Initialize writer if not done yet
                        if self.video_writer is None and hasattr(self, 'pending_video_path'):
                            if np.any(frame):
                                h, w = frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                self.video_writer = cv2.VideoWriter(self.pending_video_path, fourcc, self.pending_video_fps, (w, h))
                                logger.info(f"Initialized fisheye video writer with size {(w, h)} at {self.pending_video_fps} FPS")
                        
                        if self.video_writer is not None:
                            # fisheyeCameraRGB is RGB, VideoWriter expects BGR
                            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            self.video_writer.write(bgr_frame)

                    # Tactile 2 recording
                    if getattr(sensor_msg, 'leftGripperCameraRGB2', None) is not None:
                        frame2 = sensor_msg.leftGripperCameraRGB2
                        if np.any(frame2):
                            if getattr(self, 'video_writer_tactile_2', None) is None and hasattr(self, 'pending_video_path_tactile_2'):
                                h, w = frame2.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                self.video_writer_tactile_2 = cv2.VideoWriter(self.pending_video_path_tactile_2, fourcc, self.pending_video_fps, (w, h))
                                logger.info(f"Initialized tactile_2 video writer with size {(w, h)} at {self.pending_video_fps} FPS")
                            if getattr(self, 'video_writer_tactile_2', None) is not None:
                                bgr_frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                                self.video_writer_tactile_2.write(bgr_frame2)

        # record experiment data
        if self.enable_exp_recording:  # is False
            recorded_sensor_msg = deepcopy(sensor_msg)

            vcamera_image = self.get_vcamera_image()
            recorded_sensor_msg.vCameraImage = vcamera_image

            predicted_full_tcp_action, exception = self.predicted_full_tcp_action_buffer.peek_last_n(1)
            predicted_full_gripper_action, exception = self.predicted_full_gripper_action_buffer.peek_last_n(1)
            predicted_partial_tcp_action, exception = self.predicted_partial_tcp_action_buffer.peek_last_n(1)
            predicted_partial_gripper_action, exception = self.predicted_partial_gripper_action_buffer.peek_last_n(1)

            recorded_sensor_msg.predictedFullTCPAction = predicted_full_tcp_action[0] if len(predicted_full_tcp_action) == 1 else []
            recorded_sensor_msg.predictedFullGipperAction = predicted_full_gripper_action[0] if len(predicted_full_gripper_action) == 1 else []
            recorded_sensor_msg.predictedPartialTCPAction = predicted_partial_tcp_action[0] if len(predicted_partial_tcp_action) == 1 else []
            recorded_sensor_msg.predictedPartialGipperAction = predicted_partial_gripper_action[0] if len(predicted_partial_gripper_action) == 1 else []

            with self.mutex:
                self.sensor_msg_list.sensorMessages.append(recorded_sensor_msg)

        # calculate fps
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time >= 1.0:
            frame_rate = self.frame_count / elapsed_time
            self.prev_time = current_time
            self.frame_count = 0
            if self.time_check:
                logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
                self.check_sync()
                self.check_timestamp()

    # def callback(self, odom_msg, wrist_depth_msg, wrist_color_msg, fisheye_msg, comp_wrench_msg, raw_wrench_msg, 
    #              gs0_pc_msg, gs1_pc_msg):
    # def callback(self, wrist_color_msg, fisheye_msg, gs0_pc_msg, gs1_pc_msg, left_gripper_state, left_tcp_pose, left_tcp_vel, left_tcp_wrench):
    #     # if self.time_check:
    #     #     # check the time differences across topics and interval between time stamps
    #     #     for i, msg in enumerate(msgs):
    #     #         topic_name = self.subscribers[i].topic
        
    #     synced_messages_dict = {
    #         # 'baton_odometry': odom_msg,
    #         # 'wrist_camera_depth': wrist_depth_msg,
    #         'wrist_camera_color': wrist_color_msg,
    #         'fisheye_rgb': fisheye_msg, 
    #         # 'force_compensated_wrench': comp_wrench_msg,
    #         # 'force_raw_wrench': raw_wrench_msg,
    #         'gsmini_0_points': gs0_pc_msg, 
    #         'gsmini_1_points': gs1_pc_msg,

    #         'left_gripper_state': left_gripper_state,
    #         'left_tcp_pose': left_tcp_pose,
    #         'left_tcp_vel': left_tcp_vel,
    #         'left_tcp_wrench': left_tcp_wrench
    #         # 'motor_states': motor_states_msg,
    #     }
        
    #     sensor_msg: SensorMessage = self.data_converter.convert_synced_messages_to_data(
    #         synced_messages_dict,
    #         self.odometry_offset 
    #     )

    #     # if sensor_msg is not None:
    #     #     logger.info(f"--- DataConverter Output ---\n{sensor_msg}\n--------------------------")
        
    #     if sensor_msg is None:
    #         logger.warning("Your custom DataConverter returned None. Skipping this frame.")
    #         return

    #     raw_obs_dict = self.data_processing_manager.convert_sensor_msg_to_obs_dict(sensor_msg)

    #     self.obs_buffer.push(raw_obs_dict)

    #     if self.enable_exp_recording:
    #         recorded_sensor_msg = deepcopy(sensor_msg)

    #         vcamera_image = self.get_vcamera_image()
    #         recorded_sensor_msg.vCameraImage = vcamera_image

    #         predicted_full_tcp_action, _ = self.predicted_full_tcp_action_buffer.peek_last_n(1)
    #         predicted_full_gripper_action, _ = self.predicted_full_gripper_action_buffer.peek_last_n(1)
    #         predicted_partial_tcp_action, _ = self.predicted_partial_tcp_action_buffer.peek_last_n(1)
    #         predicted_partial_gripper_action, _ = self.predicted_partial_gripper_action_buffer.peek_last_n(1)

    #         recorded_sensor_msg.predictedFullTCPAction = predicted_full_tcp_action[0] if len(predicted_full_tcp_action) == 1 else []
    #         recorded_sensor_msg.predictedFullGipperAction = predicted_full_gripper_action[0] if len(predicted_full_gripper_action) == 1 else []
    #         recorded_sensor_msg.predictedPartialTCPAction = predicted_partial_tcp_action[0] if len(predicted_partial_tcp_action) == 1 else []
    #         recorded_sensor_msg.predictedPartialGipperAction = predicted_partial_gripper_action[0] if len(predicted_partial_gripper_action) == 1 else []

    #         with self.mutex:
    #             self.sensor_msg_list.sensorMessages.append(recorded_sensor_msg)
                
    #     if self.debug:
    #         latest_timestamp = sensor_msg.timestamp
    #         current_timestamp = convert_ros_time_to_float(self.get_clock().now())
    #         latency = current_timestamp - latest_timestamp
    #         logger.debug(f"Data conversion to obs_dict latency: {latency:.4f} seconds")

    #     self.frame_count += 1
    #     current_time = time.time()
    #     elapsed_time = current_time - self.prev_time
    #     if elapsed_time >= 1.0:
    #         frame_rate = self.frame_count / elapsed_time
    #         # if self.time_check:
    #         #     logger.debug(f"Frame processing rate: {frame_rate:.2f} FPS")
    #         self.prev_time = current_time
    #         self.frame_count = 0

    def check_sync(self):
        # Check and log timestamp differences across topics
        all_times = list(self.timestamps.values())
        if not all(all_times):
            return

        # Calculate time differences for each frame across topics
        for i in range(len(all_times[0])):
            max_diff = 0
            for j in range(len(all_times)):
                for k in range(j + 1, len(all_times)):
                    if i < len(all_times[j]) and i < len(all_times[k]):
                        time_diff = abs(all_times[j][i] - all_times[k][i])
                        max_diff = max(max_diff, time_diff)
            logger.info(f"Frame {i}: Maximum time difference across topics: {max_diff:.6f} seconds")

    def check_timestamp(self):
        # check the interval between different time stampss
        all_times = list(self.timestamps.values())
        if not all(all_times):
            return

        time_stamps = []
        for i in range(len(all_times[0])):
            timestamps_for_frame = []
            for j in range(len(all_times)):
                if i < len(all_times[j]):
                    timestamp = all_times[j][i]
                    timestamps_for_frame.append(timestamp)

            if timestamps_for_frame:
                mean_time_stamp = sum(timestamps_for_frame) / len(timestamps_for_frame)
                time_stamps.append(mean_time_stamp)
                logger.info(f"Frame {i}: Mean timestamp: {mean_time_stamp:.6f} seconds")


    def start_video_recording(self, video_dir: str, fps: int = 30, prefix: str = 'fisheye'):
        """
        Start recording fisheye camera video.
        
        Args:
            video_dir: directory to save the video file
            fps: frames per second for the video
            prefix: filename prefix
        """
        if not osp.exists(video_dir):
            os.makedirs(video_dir)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = osp.join(video_dir, f'{prefix}_{timestamp}.mp4')
        
        video_path_tactile_2 = osp.join(video_dir, f'tactile_2_{timestamp}.mp4')

        # Store parameters to initialize writer in callback using actual frame size
        with self.video_recording_lock:
            self.video_recording = True
            self.pending_video_path = video_path
            self.pending_video_path_tactile_2 = video_path_tactile_2
            self.pending_video_fps = fps
            self.video_writer = None
            self.video_writer_tactile_2 = None

        logger.info(f'Started video recording: {video_path} (and tactiles if active)')
        return video_path

    def stop_video_recording(self):
        """Stop recording video and release the writers."""
        with self.video_recording_lock:
            self.video_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            if getattr(self, 'video_writer_tactile_2', None) is not None:
                self.video_writer_tactile_2.release()
                self.video_writer_tactile_2 = None
        logger.info('Stopped video recording')

    def reset(self) -> None:
        self.start_gripper_interval_control = False
        self.obs_buffer.reset()
        if self.enable_exp_recording:
            self.sensor_msg_list.sensorMessages = []
            self.predicted_full_tcp_action_buffer.reset()
            self.predicted_full_gripper_action_buffer.reset()
            self.predicted_partial_tcp_action_buffer.reset()
            self.predicted_partial_gripper_action_buffer.reset()
            self.odometry_offset = np.zeros(3, dtype=np.float32)

    # @pyinstrument.profile()
    def get_obs(self,obs_steps: int = 2, temporal_downsample_ratio: int = 2, ) -> Dict[str, np.ndarray]:
        """
        Get observations with temporal downsampling support.

        Args:
            obs_steps: The number of observations to stack.
            temporal_downsample_ratio: The ratio for temporal downsampling.
                For example, if ratio=2, it will sample every other observation.
        Returns:
            A dictionary containing stacked observations
        """
        # Get last n*ratio observations to ensure we have enough samples after downsampling
        last_n_obs_list, _ = self.obs_buffer.peek_last_n(
            obs_steps * temporal_downsample_ratio)  # newest to oldest

        result = dict()
        # Filter out None observations
        last_n_obs_list = [obs for obs in last_n_obs_list if obs is not None]
        if len(last_n_obs_list) == 0:
            return result

        # Apply temporal downsampling
        # If ratio=2, it will take every other observation: [0, 2, 4, ...]
        # If ratio=3, it will take every third observation: [0, 3, 6, ...]
        downsampled_obs_list = last_n_obs_list[::temporal_downsample_ratio]
        # Take only the last n_obs_steps observations after downsampling
        downsampled_obs_list = downsampled_obs_list[:obs_steps]

        # reverse the order to oldest to newest
        downsampled_obs_list = downsampled_obs_list[::-1]

        # Stack observations for each key
        for key in downsampled_obs_list[0].keys():
            result[key] = stack_last_n_obs(
                [obs[key] for obs in downsampled_obs_list], obs_steps)

        # convert current time (ROS time) to python time
        current_timestamp = convert_ros_time_to_float(self.get_clock().now())
        # find out the latency compared to current time
        latency = current_timestamp - downsampled_obs_list[-1]['timestamp'][0]
        # the overall latency is approximately 20ms - 70ms (max 110ms) (~5000 points per pcd)
        logger.debug(f"Overall latency for get_obs() : {latency:.4f} seconds")

        return result

    def send_gripper_command_direct(self, left_gripper_width_target: float, right_gripper_width_target: float):
        """
        Send gripper command (width) directly to robot
        """
        self.send_command('/move_gripper/left', {
            'width': left_gripper_width_target,
            # 'velocity': 10.0,
            'velocity': 2.0,
            'force_limit': self.grasp_force
        })
        self.last_gripper_width_target[0] = left_gripper_width_target
        self.send_command('/move_gripper/right', {
            'width': right_gripper_width_target,
            # 'velocity': 10.0,
            'velocity': 2.0,
            'force_limit': self.grasp_force
        })
        self.last_gripper_width_target[1] = right_gripper_width_target

    def send_gripper_command(self, left_gripper_width_target: float, right_gripper_width_target: float, is_bimanual: bool = False):
        if self.enable_gripper_interval_control and self.start_gripper_interval_control:
            self.gripper_interval_count += 1
            if self.gripper_interval_count % self.gripper_control_time_interval == 0:
                self.gripper_interval_count = 0

            if self.gripper_interval_count != 0:
                return

        if self.enable_gripper_width_clipping: # is True
            if left_gripper_width_target < self.gripper_width_threshold:
                left_gripper_width_target = self.min_gripper_width
                self.start_gripper_interval_control = True
            if is_bimanual:
                if right_gripper_width_target < self.gripper_width_threshold:
                    right_gripper_width_target = self.min_gripper_width
                    self.start_gripper_interval_control = True
        else:
            self.start_gripper_interval_control = True

        robot_states = BimanualRobotStates.model_validate(self.send_command('/get_current_robot_states'))

        grasp_force = self.grasp_force
        gripper_control_width_precision = self.gripper_control_width_precision
        left_current_width = robot_states.leftGripperState[0]
        if abs(self.last_gripper_width_target[0] - left_gripper_width_target) >= gripper_control_width_precision:
            if self.use_force_control_for_gripper and self.last_gripper_width_target[0] > left_gripper_width_target:
                # try to close gripper with pure force control
                logger.debug(f"left gripper moving from {left_current_width} to target: {left_gripper_width_target} "
                             f"with force {grasp_force}")
                self.send_command('/move_gripper_force/left', {
                    'force_limit': grasp_force,
                    'width': left_gripper_width_target,
                    'velocity': 2.0
                })
            else:
                # open gripper with position control
                logger.debug(f"left gripper moving from {left_current_width} to target: {left_gripper_width_target}")
                self.send_command('/move_gripper/left', {
                    'width': left_gripper_width_target,
                    # 'velocity': 10.0,
                    'velocity': 2.0,
                    'force_limit': grasp_force
                })
            self.last_gripper_width_target[0] = left_gripper_width_target

        if is_bimanual: # this variable is from real_runner post_process_action function
            right_current_width = robot_states.rightGripperState[0]
            if abs(self.last_gripper_width_target[1] - right_gripper_width_target) >= gripper_control_width_precision:
                if self.use_force_control_for_gripper and self.last_gripper_width_target[1] > right_gripper_width_target:
                    # try to close gripper with pure force control
                    logger.debug(f"right gripper moving from {right_current_width} to target: {right_gripper_width_target} "
                                 f"with force {grasp_force}")
                    self.send_command('/move_gripper_force/right', {
                        'force_limit': grasp_force
                    })
                else:
                    # open gripper with position control
                    logger.debug(f"right gripper moving from {right_current_width} to target: {right_gripper_width_target}")
                    self.send_command('/move_gripper/right', {
                        'width': right_gripper_width_target,
                        # 'velocity': 10.0,
                        'velocity': 2.0,
                        'force_limit': grasp_force
                    })
                self.last_gripper_width_target[1] = right_gripper_width_target


    def execute_action(self, action: np.ndarray, use_relative_action: bool = False, is_bimanual: bool = False) -> None:
        """
        Send action (in robot coordinate system) to robot
        :param action: np.ndarray, shape (16,) (left+right) (x, y, z, r, p, y, gripper_width, gripper_force)
        """
        left_action = action[:8]
        right_action = action[8:]

        # calculate target gripper width
        if use_relative_action:
            raise NotImplementedError
        else:
            left_gripper_width_target = float(left_action[-2])
            right_gripper_width_target = float(right_action[-2])
        self.send_gripper_command(left_gripper_width_target, right_gripper_width_target, is_bimanual=is_bimanual)

        if use_relative_action:
            raise NotImplementedError
        else:
            left_tcp_target_6d_in_robot = left_action[:6]
            right_tcp_target_6d_in_robot = right_action[:6]
        left_tcp_target_7d_in_robot = pose_6d_to_pose_7d(left_tcp_target_6d_in_robot)
        right_tcp_target_7d_in_robot = pose_6d_to_pose_7d(right_tcp_target_6d_in_robot)

        # print("----------------")
        # print(left_tcp_target_7d_in_robot)
        # print("----------------")
        self.send_command('/move_tcp/left', {'target_tcp': left_tcp_target_7d_in_robot.tolist()})
        if is_bimanual:
            self.send_command('/move_tcp/right', {'target_tcp': right_tcp_target_7d_in_robot.tolist()})
    
    def execute_relative_action(self, action: np.ndarray, use_relative_action: bool = False, is_bimanual: bool = False) -> None:
        """
        Send action (in robot coordinate system) to robot
        :param action: np.ndarray, shape (16,) (left+right) (x, y, z, r, p, y, gripper_width, gripper_force)
        """
        left_action = action[:8]
        right_action = action[8:]

        # calculate target gripper width
        if use_relative_action:
            raise NotImplementedError
        else:
            left_gripper_width_target = float(left_action[-2])
            right_gripper_width_target = float(right_action[-2])
        self.send_gripper_command(left_gripper_width_target, right_gripper_width_target, is_bimanual=is_bimanual)

        if use_relative_action:
            raise NotImplementedError
        else:
            left_tcp_target_6d_in_robot = left_action[:6]
            right_tcp_target_6d_in_robot = right_action[:6]
        left_tcp_target_7d_in_robot = pose_6d_to_pose_7d(left_tcp_target_6d_in_robot)
        right_tcp_target_7d_in_robot = pose_6d_to_pose_7d(right_tcp_target_6d_in_robot)

        # print("----------------")
        # print(left_tcp_target_7d_in_robot)
        # print("----------------")
        self.send_command('/move_tcp/left', {'target_tcp': left_tcp_target_7d_in_robot.tolist()})
        if is_bimanual:
            self.send_command('/move_tcp/right', {'target_tcp': right_tcp_target_7d_in_robot.tolist()})

    def get_vcamera_image(self):
        response = self.session.get(f'http://{self.vcamera_server_ip}:{self.vcamera_server_port}/peek_latest_capture')
        if response.status_code == 200 and len(response.content) != 0:
            img = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            return img
        else:
            logger.warning(f"Failed to get vcamera image, status code: {response.status_code}")
            return []

    def get_predicted_action(self, action: np.ndarray, type):
        if self.enable_exp_recording:
            if type == 'full_tcp':
                self.predicted_full_tcp_action_buffer.push(action)
            elif type == "full_gripper":
                self.predicted_full_gripper_action_buffer.push(action)
            elif type == "partial_tcp":
                self.predicted_partial_tcp_action_buffer.push(action)
            elif type == "partial_gripper":
                self.predicted_partial_gripper_action_buffer.push(action)
            else:
                raise ValueError(f"Unknown action type: {type}")

    def save_exp(self, episode_idx):
        if self.enable_exp_recording:
            logger.debug('Trying to save sensor messages...')
            if not osp.exists(self.exp_dir):
                os.makedirs(self.exp_dir)
            record_path = osp.join(self.exp_dir, f'episode_{episode_idx}.pkl')
            if osp.exists(record_path):
                record_path = ".".join(record_path.split('.')[:-1]) + f'{time.strftime("_%Y%m%d_%H%M%S")}.pkl'
                logger.warning(f'Experiment path already exists, save to {record_path}')
            with open(record_path, 'wb') as f:
                with self.mutex:
                    pickle.dump(self.sensor_msg_list, f)
            logger.debug(f'Saved experiment record to {record_path}')
    
    # def wait_for_first_message(self, node: Node, topic: str, msg_type, timeout_sec=5.0):
    #     node.get_logger().info(f"Waiting for first message on topic '{topic}'...")
    #     msg_received = None
    #     event = threading.Event()

    #     def callback(msg):
    #         nonlocal msg_received
    #         msg_received = msg
    #         event.set()

    #     qos_profile = QoSProfile(
    #         reliability=ReliabilityPolicy.RELIABLE,
    #         history=HistoryPolicy.KEEP_LAST,
    #         durability=DurabilityPolicy.VOLATILE,
    #         depth=100
    #     )
    #     sub = node.create_subscription(msg_type, topic, callback, qos_profile)

    #     start_time = time.time()
    #     while not event.is_set():
    #         rclpy.spin_once(node, timeout_sec=0.1)
    #         if time.time() - start_time > timeout_sec:
    #             break

    #     node.destroy_subscription(sub)

    #     if not event.is_set() or msg_received is None:
    #         raise TimeoutError(f"Timed out waiting for message on topic {topic}")
        
    #     print(colored(" OK", "green"), end='\n', flush=True)
    #     return msg_received