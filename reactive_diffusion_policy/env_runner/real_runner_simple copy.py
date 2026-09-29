import threading
import time
import os
import sys
import os.path as osp
import pickle
import numpy as np
import torch
import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, Optional, List
import transforms3d as t3d
import py_cli_interaction
from omegaconf import DictConfig, ListConfig, OmegaConf
from copy import deepcopy
import requests

from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.common.precise_sleep import precise_sleep
from reactive_diffusion_policy.env.real_bimanual.real_env import OmniUMIRealEnv
from reactive_diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix
from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
from reactive_diffusion_policy.common.space_utils import (
    pose_3d_9d_to_homo_matrix_batch,
    homo_matrix_to_pose_9d_batch,
    matrix4x4_to_pose_6d
)

from reactive_diffusion_policy.real_world.robot.force_publisher import ForceSensorAsync
from reactive_diffusion_policy.real_world.robot.flexiv_controller_async import FlexivControllerAsync
from reactive_diffusion_policy.real_world.publisher.gelsight_camera_publisher import GelsightImageCamera
from reactive_diffusion_policy.real_world.publisher.realsense_camera_publisher import RealsenseImageCamera
from reactive_diffusion_policy.real_world.publisher.mvs_camera_publisher import MVSImageCamera, create_camera_configs_from_serials
from reactive_diffusion_policy.real_world.publisher.mvs_utils.mvs_cam import get_all_mvs_dev_serial
from reactive_diffusion_policy.real_world.robot.inference_observation_manager import InferenceObservationManager

import cv2

from reactive_diffusion_policy.common.space_utils import (
    pose_6d_to_pose_7d,
    pose_6d_to_pose_9d,
    pose_6d_to_4x4matrix,
    matrix4x4_to_pose_6d
)

os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
cv2.setNumThreads(12)


class RealRunner:
    """
    OmniUMI Real Runner (Synchronous/Sequential Mode)
    """

    def __init__(
        self,
        output_dir: str,
        shape_meta: DictConfig,
        sensor_config: DictConfig,
        robot_interface_cfg: str = "charmander.yml",
        robot_controller_type: str = "OSC_POSE",
        robot_control_frequency: int = 120,
        gripper_port: str = "/dev/ttyUSB1",
        gripper_motor_id: int = 1,
        enable_gripper: bool = True,
        use_force_control_for_gripper: bool = False,
        max_gripper_width: float = 0.12,
        min_gripper_width: float = 0.0,
        grasp_force: float = 5.0,
        enable_gripper_interval_control: bool = False,
        gripper_control_time_interval: float = 30,
        gripper_control_width_precision: float = 0.01,
        gripper_width_threshold: float = 0.046,
        enable_gripper_width_clipping: bool = True,
        image_resize_shape: tuple = (320, 240),
        use_6d_rotation: bool = True,
        pca_param_dict: DictConfig = None,
        gripper_index: int = 0,
        tcp_ensemble_buffer_params: DictConfig = None,
        gripper_ensemble_buffer_params: DictConfig = None,
        latent_tcp_ensemble_buffer_params: DictConfig = None,
        latent_gripper_ensemble_buffer_params: DictConfig = None,
        use_latent_action_with_rnn_decoder: bool = False,
        use_relative_action: bool = False,
        use_relative_tcp_obs_for_relative_action: bool = True,
        action_interpolation_ratio: int = 1,
        eval_episodes: int = 10,
        max_duration_time: float = 30,
        tcp_action_update_interval: int = 6,
        gripper_action_update_interval: int = 10,
        tcp_pos_clip_range: ListConfig = None,
        tcp_rot_clip_range: ListConfig = None,
        tqdm_interval_sec: float = 5.0,
        control_fps: float = 12,
        inference_fps: float = 6,
        latency_step: int = 0,
        gripper_latency_step: Optional[int] = None,
        n_obs_steps: int = 2,
        obs_temporal_downsample_ratio: int = 2,
        dataset_obs_temporal_downsample_ratio: int = 1,
        downsample_extended_obs: bool = True,
        max_fps: int = 30,
        enable_video_recording: bool = False,
        vcamera_server_ip: Optional[Union[str, ListConfig]] = None,
        vcamera_server_port: Optional[Union[int, ListConfig]] = None,
        task_name: str = None,
        debug: bool = False,
        **kwargs
    ):
        self.task_name = task_name
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes
        self.debug = debug
        self.output_dir = output_dir

        # Parse observation keys
        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                extended_rgb_keys.append(key)
            elif obs_type == 'low_dim':
                extended_lowdim_keys.append(key)
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys

        logger.info("=" * 80)
        logger.info("Initializing sensors...")
        logger.info("=" * 80)

        self.obs_manager = self._init_sensors(
            sensor_config,
            robot_interface_cfg,
            robot_controller_type,
            robot_control_frequency,
            gripper_port,
            gripper_motor_id,
            enable_gripper
        )

        logger.info("Initializing OmniUMI Real Environment...")
        self.env = OmniUMIRealEnv(
            shape_meta=shape_meta,
            robot_controller=self.robot_controller,
            obs_manager=self.obs_manager,
            image_resize_shape=image_resize_shape,
            use_6d_rotation=use_6d_rotation,
            pca_param_dict=pca_param_dict,
            gripper_index=gripper_index,
            use_force_control_for_gripper=use_force_control_for_gripper,
            max_gripper_width=max_gripper_width,
            min_gripper_width=min_gripper_width,
            grasp_force=grasp_force,
            enable_gripper_interval_control=enable_gripper_interval_control,
            gripper_control_time_interval=gripper_control_time_interval,
            gripper_control_width_precision=gripper_control_width_precision,
            gripper_width_threshold=gripper_width_threshold,
            enable_gripper_width_clipping=enable_gripper_width_clipping,
            max_fps=max_fps,
            debug=debug
        )

        # self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
        self.env.send_gripper_command_direct(0.15, self.env.max_gripper_width)
        # self.env.send_gripper_command_direct(2.7, self.env.max_gripper_width)
        time.sleep(2)

        self.max_duration_time = max_duration_time
        self.tcp_action_update_interval = tcp_action_update_interval
        self.gripper_action_update_interval = gripper_action_update_interval
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.tcp_rot_clip_range = tcp_rot_clip_range
        self.tqdm_interval_sec = tqdm_interval_sec
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0
        self.latency_step = latency_step
        self.gripper_latency_step = gripper_latency_step if gripper_latency_step is not None else latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder

        if self.use_latent_action_with_rnn_decoder:
            assert latent_tcp_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            assert latent_gripper_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            self.tcp_ensemble_buffer = EnsembleBuffer(**latent_tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**latent_gripper_ensemble_buffer_params)
        else:
            self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)

        self.use_relative_action = use_relative_action
        self.use_relative_tcp_obs_for_relative_action = use_relative_tcp_obs_for_relative_action
        self.action_interpolation_ratio = action_interpolation_ratio

        self.enable_video_recording = enable_video_recording
        if enable_video_recording:
            assert isinstance(vcamera_server_ip, str) and isinstance(vcamera_server_port, int) or \
                   isinstance(vcamera_server_ip, ListConfig) and isinstance(vcamera_server_port, ListConfig), \
                "vcamera_server_ip and vcamera_server_port should be a string or ListConfig."
        if isinstance(vcamera_server_ip, str):
            vcamera_server_ip_list = [vcamera_server_ip]
            vcamera_server_port_list = [vcamera_server_port]
        elif isinstance(vcamera_server_ip, ListConfig):
            vcamera_server_ip_list = list(vcamera_server_ip)
            vcamera_server_port_list = list(vcamera_server_port)
        else:
            vcamera_server_ip_list = []
            vcamera_server_port_list = []
        self.vcamera_server_ip_list = vcamera_server_ip_list
        self.vcamera_server_port_list = vcamera_server_port_list
        self.video_dir = osp.join(output_dir, 'videos')

        self.stop_event = threading.Event()
        self.session = requests.Session()
        
        self.action_step_count = 0

        logger.info("RealRunner initialized successfully (Synchronous Mode)")

    def _init_sensors(self, sensor_config, robot_interface_cfg, robot_controller_type, robot_control_frequency, gripper_port, gripper_motor_id, enable_gripper):
        cali_info = None
        cali_bias = None
        force_tool_offset = np.array([[0], [0], [0.1864]])

        cali_params_file = sensor_config.get('cali_params_file', None)
        if cali_params_file and os.path.exists(cali_params_file):
            try:
                with open(cali_params_file, 'rb') as f:
                    params = pickle.load(f)
                cali_info = params['cali_info']
                cali_bias = params['cali_bias']
                logger.info(f"Loaded calibration from {cali_params_file}")
            except Exception as e:
                logger.warning(f"Failed to load calibration: {e}")

        logger.info("Initializing FlexivControllerAsync (direct robot control)...")
        self.robot_controller = FlexivControllerAsync(
            interface_cfg=robot_interface_cfg,
            controller_type=robot_controller_type,
            control_frequency=robot_control_frequency,
            gripper_port=gripper_port,
            gripper_motor_id=gripper_motor_id,
            enable_gripper=enable_gripper,
            allow_repeat_frames=True,
            debug=self.debug,
        )
        self.robot_controller.connect()

        logger.info("Initializing Force Sensor...")
        self.force_sensor = ForceSensorAsync(
            port=sensor_config.get('force_port', "/dev/ttyUSB3"),
            baud_rate=sensor_config.get('force_baud', 1000000),
            sensor_name="force_sensor",
            allow_repeat_frames=True,
            debug=self.debug,
        )
        self.force_sensor.connect()

        logger.info("Gripper control integrated into FlexivControllerAsync")

        gelsight_left_device = sensor_config.get('gelsight_left_device', None)
        if gelsight_left_device is not None:
            self.gelsight_left = GelsightImageCamera(
                device_path=gelsight_left_device, camera_name="gelsight_left",
                target_width=640, target_height=480, fps=30, color_mode='RGB', allow_repeat_frames=True, debug=self.debug
            )
            self.gelsight_left.connect(warmup=True)
        else:
            self.gelsight_left = None

        gelsight_right_device = sensor_config.get('gelsight_right_device', None)
        if gelsight_right_device is not None:
            self.gelsight_right = GelsightImageCamera(
                device_path=gelsight_right_device, camera_name="gelsight_right",
                target_width=640, target_height=480, fps=30, color_mode='RGB', allow_repeat_frames=True, debug=self.debug
            )
            self.gelsight_right.connect(warmup=True)
        else:
            self.gelsight_right = None

        realsense_external_serial = sensor_config.get('realsense_external_serial', None)
        if realsense_external_serial is not None:
            self.realsense_external = RealsenseImageCamera(
                camera_serial_number=realsense_external_serial, camera_name="realsense_external", camera_type='D400',
                rgb_resolution=(640, 480), depth_resolution=(640, 480), fps=30, color_mode='RGB',
                auto_exposure=True, auto_white_balance=True, allow_repeat_frames=True, debug=self.debug
            )
            self.realsense_external.connect(warmup=True)
        else:
            self.realsense_external = None

        realsense_wrist_serial = sensor_config.get('realsense_wrist_serial', None)
        if realsense_wrist_serial is not None:
            self.realsense_wrist = RealsenseImageCamera(
                camera_serial_number=realsense_wrist_serial, camera_name="realsense_wrist", camera_type='D400',
                rgb_resolution=(640, 480), depth_resolution=(640, 480), fps=30, color_mode='RGB',
                auto_exposure=True, auto_white_balance=True, allow_repeat_frames=True, debug=self.debug
            )
            self.realsense_wrist.connect(warmup=True)
        else:
            self.realsense_wrist = None
        
        time.sleep(5.0)
            
        mvs_camera_serial = sensor_config.get('mvs_camera_serial', None)
        if mvs_camera_serial is None:
            try:
                available_serials = get_all_mvs_dev_serial()
                if available_serials:
                    mvs_camera_serial = available_serials[0]
                    logger.info(f"Auto-detected MVS camera: {mvs_camera_serial}")
            except Exception as e:
                logger.warning(f"Failed to detect MVS camera: {e}")

        if mvs_camera_serial is not None:
            cam_configs = create_camera_configs_from_serials(
                serials=[mvs_camera_serial], fps=sensor_config.get('mvs_camera_fps', 30),
                output_res=(sensor_config.get('mvs_camera_width', 480), sensor_config.get('mvs_camera_height', 480)),
                crop_func=lambda img: img
            )
            self.mvs_camera = MVSImageCamera(config=cam_configs[0], allow_repeat_frames=True, debug=self.debug)
            self.mvs_camera.connect(warmup=True)
            time.sleep(3.0)
        else:
            self.mvs_camera = None

        logger.info("Creating InferenceObservationManager...")
        obs_manager = InferenceObservationManager(
            robot_server=self.robot_controller,
            force_sensor=self.force_sensor,
            gelsight_left=self.gelsight_left,
            gelsight_right=self.gelsight_right,
            fisheye=self.mvs_camera,
            realsense_external=self.realsense_external,
            realsense_wrist=self.realsense_wrist,
            force_tool_offset=force_tool_offset,
            cali_info=cali_info,
            cali_bias=cali_bias,
            num_workers=sensor_config.get('num_workers', 6),
            target_fps=sensor_config.get('target_fps', 30),
            debug=self.debug,
        )
        return obs_manager

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        obs_dict = deepcopy(obs_dict)
        for key in self.lowdim_keys:
            if "wrt" not in key:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, None))
        for key in self.lowdim_keys:
            obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = obs_dict[key][-1].copy()
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)
        return obs_dict, absolute_obs_dict

    def pre_process_extended_obs(self, extended_obs_dict: Dict) -> Tuple[Dict, Dict]:
        extended_obs_dict = deepcopy(extended_obs_dict)
        absolute_extended_obs_dict = dict()
        for key in self.extended_lowdim_keys:
            extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()

        # convert absolute action to relative action
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = extended_obs_dict[key][-1].copy()
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key], base_absolute_action=base_absolute_action)
        
        return extended_obs_dict, absolute_extended_obs_dict

    def post_process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Post-process the action before sending to the robot"""
        assert len(action.shape) == 2  # (action_steps, d_a)

        if self.env.data_processing_manager.use_6d_rotation: # lsq: true in yaml file
            if action.shape[-1] == 4 or action.shape[-1] == 8:
                # convert to 6D pose
                left_trans_batch = action[:, :3] # (action_steps, 3)
                # we use default euler angles as 0
                left_euler_batch = np.zeros_like(left_trans_batch)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 8:
                    right_trans_batch = action[:, 3:6] # (action_steps, 3)
                    right_euler_batch = np.zeros_like(right_trans_batch)
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            elif action.shape[-1] == 10 or action.shape[-1] == 20:
                # convert to 6D pose
                left_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 3:9])  # (action_steps, 3, 3)
                # left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                left_euler_batch = np.asarray([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                # left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch], dtype=np.float32)
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 20:
                    right_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 12:18])
                    right_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in right_rot_mat_batch])
                    right_trans_batch = action[:, 9:12]
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # clip action (x, y, z)
        left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[0]), np.array(self.tcp_pos_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, :3] = np.clip(right_action_6d[:, :3], np.array(self.tcp_pos_clip_range[2]), np.array(self.tcp_pos_clip_range[3]))
        # clip action (r, p, y)
        left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[0]), np.array(self.tcp_rot_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, 3:] = np.clip(right_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[2]), np.array(self.tcp_rot_clip_range[3]))
        # add gripper action
        if action.shape[-1] == 4:
            left_action = np.concatenate([left_action_6d, action[:, 3][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 8:
            left_action = np.concatenate([left_action_6d, action[:, 6][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 7][:, np.newaxis],
                                           np.zeros((action.shape[0], 1))], axis=1)
        elif action.shape[-1] == 10:
            left_action = np.concatenate([left_action_6d, action[:, 9][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 20:
            left_action = np.concatenate([left_action_6d, action[:, 18][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 19][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
        else:
            raise NotImplementedError

        if right_action is None:
            right_action = left_action.copy()
            is_bimanual = False
        else:
            is_bimanual = True
        action_all = np.concatenate([left_action, right_action], axis=-1)
        return (action_all, is_bimanual)

    def action_command_thread(self, policy: Union[DiffusionUnetImagePolicy], stop_event):
        # while not stop_event.is_set():
        start_time_loop = time.time()
        lenght_actions = len(self.tcp_ensemble_buffer.actions)
        while not stop_event.is_set() and (len(self.tcp_ensemble_buffer.actions) > 0 or len(self.gripper_ensemble_buffer.actions) > 0):
            cur_time_loop = time.time()
             # if cur_time_loop - start_time_loop > 0.5:
            if len(self.tcp_ensemble_buffer.actions) / lenght_actions < 0.2:
                break
            start_time = time.time()
            print(f"thread tcp_ensemble_buffer length: {len(self.tcp_ensemble_buffer.actions)} , gripper_ensemble_buffer length: {len(self.gripper_ensemble_buffer.actions)}")
            print(f"tcp_ensemble_buffer status: {self.tcp_ensemble_buffer.get_buffer_status()}")
            print(f"self.tcp_ensemble_buffer.actions: \n{self.tcp_ensemble_buffer.actions}") # len(self.tcp_ensemble_buffer.actions) = 24 for the first loop

            # import pdb; pdb.set_trace()
            # get step action from ensemble buffer
            tcp_step_action = self.tcp_ensemble_buffer.get_action()
            gripper_step_action = self.gripper_ensemble_buffer.get_action()

            print(f"tcp_step_action | get_action | shape: {tcp_step_action.shape}  \n{tcp_step_action}") if tcp_step_action is not None else print(f"tcp_step_action is None, tcp_step_action: {tcp_step_action}")
            print(f"gripper_step_action | get_action | shape: {gripper_step_action.shape}  \n{gripper_step_action}") if gripper_step_action is not None else print(f"gripper_step_action is None, gripper_step_action: {gripper_step_action}")
            # 目前测的 tcp_step_action == gripper_step_action；，shape 都为(42,)；
            # e.g. 
            '''
            tcp_step_action = 
            array([ 3.16610813e-01, -1.03490400e+00,  9.51986074e-01, -7.84015596e-01,
                -3.88795510e-03, -1.52957439e+00,  1.42907286e+00, -9.50710356e-01,
                -2.75679648e-01, -1.69892776e+00,  1.54201174e+00, -1.02298236e+00,
                -4.23640996e-01, -1.78198719e+00,  1.59088206e+00, -1.10941136e+00,
                -4.48576272e-01, -1.83473134e+00,  1.65653586e+00, -1.14325953e+00,
                -4.35820282e-01, -1.85737598e+00,  1.68194425e+00, -1.19034147e+00,
                -4.32880551e-01, -1.85945642e+00,  1.66750312e+00, -1.24657023e+00,
                3.15546572e-01, -1.49330556e+00,  1.47601748e+00, -1.37307203e+00,
                4.53025689e-01,  3.24377458e-02,  3.65129963e-01,  7.19680836e-01,
                -6.94221204e-01,  1.07895342e-02, -6.94223839e-01, -7.19747437e-01,
                -4.10947806e-03,  6.00000000e+00])
            '''
            if tcp_step_action is None or gripper_step_action is None:  # no action in the buffer => no movement.
                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                logger.debug(f"Step: {self.action_step_count}, control_interval_time: {self.control_interval_time}, "
                             f"cur_time-start_time: {cur_time - start_time}")
                self.action_step_count += 1
                continue

            if self.use_latent_action_with_rnn_decoder:
                tcp_extended_obs_step = int(tcp_step_action[-1])
                gripper_extended_obs_step = int(gripper_step_action[-1])
                tcp_step_action = tcp_step_action[:-1]
                gripper_step_action = gripper_step_action[:-1]

                longer_extended_obs_step = max(tcp_extended_obs_step, gripper_extended_obs_step)
                obs_temporal_downsample_ratio = self.obs_temporal_downsample_ratio if self.downsample_extended_obs else 1
                extended_obs = self.env.get_obs(longer_extended_obs_step, temporal_downsample_ratio=obs_temporal_downsample_ratio)

                if self.use_relative_action:
                    logger.info("use_relative_action")
                    # action_dim = self.shape_meta['obs']['left_robot_tcp_pose']['shape'][0]
                    # logger.info(f"left robot tcp pose action dim {action_dim}")
                    action_dim = 9

                    if 'right_robot_tcp_pose' in self.shape_meta['obs']:
                        action_dim += self.shape_meta['obs']['right_robot_tcp_pose']['shape'][0]
                        logger.info(f"the action have right_robot_tcp_pose,the action dim is {action_dim} ")
                    tcp_base_absolute_action = tcp_step_action[-action_dim:]
                    gripper_base_absolute_action = gripper_step_action[-action_dim:]
                    tcp_step_action = tcp_step_action[:-action_dim] # lsq : shape = (32,) , latent action dim = 32, absolute dim = 9 step count = 1, thus (42,) = (32+9+1,)
                    gripper_step_action = gripper_step_action[:-action_dim] # lsq : shape = (32,) 

                np_extended_obs_dict = dict(extended_obs) # extended_obs['left_robot_tcp_wrench'].shape = (9, 6) 有时候为 （6,6），是不是会根据 step 来决定第一个维度是多少？
                np_extended_obs_dict = get_real_obs_dict(env_obs=np_extended_obs_dict, shape_meta=self.shape_meta, is_extended_obs=True)
                np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)
                extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0)) # extended_obs_dict['left_robot_tcp_wrench'].shape = (1, 9, 6)

                tcp_step_latent_action = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0) # lsq: shape = (1, 32)
                gripper_step_latent_action = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0) # lsq: shape = (1, 32)

                dataset_obs_temporal_downsample_ratio = self.dataset_obs_temporal_downsample_ratio
                tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy() # lsq: shape = (5, 10) 
                #  input: tcp_step_latent_action.shape = torch.Size([1, 32]);  extended_obs_dict['left_robot_tcp_wrench'].shape = torch.Size([1, 6, 6]);  tcp_extended_obs_step = 6 , dataset_obs_temporal_downsample_ratio = 2
                # output: tcp_step_action.shape = (3, 10)
                '''
                tcp_step_action = 
                array([[-1.3159011e-03, -1.1339219e-03, -6.2940246e-03,  1.0078665e+00,
                    -4.2338315e-03,  1.7646770e-03,  5.8489908e-03,  9.9942338e-01,
                    -1.1007907e-02,  9.2301086e-02],
                    [-1.7117941e-03,  2.6780218e-04, -1.1092811e-02,  1.0083706e+00,
                    -3.3790562e-03,  2.7948762e-03,  6.1783586e-03,  1.0002395e+00,
                    -1.0377841e-02,  9.1626011e-02],
                    [-2.0495886e-03,  1.7487550e-03, -1.5864503e-02,  1.0085688e+00,
                    -2.6818532e-03,  3.5665776e-03,  6.9377143e-03,  1.0011752e+00,
                    -1.0692695e-02,  9.0959571e-02]], dtype=float32)
                '''
                gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()  # lsq: shape = (6, 10)
                # gripper_step_action.shape = (3, 10)
                if tcp_step_action is None or gripper_step_action is None:
                    logger.warning("Policy returned a None action, skipping this command step.")
                    continue
                
                # import pdb; pdb.set_trace()
                # logger.info("---------------------------------")
                # logger.info(f"tcp_step_action: {tcp_step_action}")
                # logger.info(f"tcp_base_absolute_action: {tcp_base_absolute_action}")
                # logger.info("---------------------------------")
                
                # tcp_step_action_pos = tcp_step_action[:,:3]
                # import pdb; pdb.set_trace()
                # print(f"tcp_step_action: {tcp_step_action.shape}")
                tcp_step_action_mat = pose_3d_9d_to_homo_matrix_batch(tcp_step_action[:,:9]) # lsq (5,4,4) 
                tcp_step_action_6d = np.zeros((tcp_step_action.shape[0], 6))
                for i in range(tcp_step_action_mat.shape[0]):
                    tcp_step_action_6d[i] = matrix4x4_to_pose_6d(tcp_step_action_mat[i])

                if self.use_relative_action:
                    # logger.info(f"longer_extended_obs_step: {longer_extended_obs_step}, relative_action (mm): \n {tcp_step_action_6d[:,:3]*1000} \n relative_rotation (deg): \n {tcp_step_action_6d[:,3:]*180/np.pi}") # tcp_step_action.shape = (6,10),其中 6 不是固定的
                    # logger.info(f"tcp_base_absolute_action.shape: {tcp_base_absolute_action.shape} \n tcp_base_absolute_action: \n {tcp_base_absolute_action}")
                    print(f"longer_extended_obs_step: {longer_extended_obs_step}, relative_action (mm): \n {tcp_step_action_6d[:,:3]*1000} \n relative_rotation (deg): \n {tcp_step_action_6d[:,3:]*180/np.pi}") # tcp_step_action.shape = (6,10),其中 6 不是固定的
                    print(f"tcp_base_absolute_action.shape: {tcp_base_absolute_action.shape} \n tcp_base_absolute_action: \n {tcp_base_absolute_action}")
                    '''
                    longer_extended_obs_step: 6, relative_action (mm): 
                    [[ -1.31590106  -1.13392191  -6.29402464]
                    [ -1.71179406   0.26780218 -11.09281089]
                    [ -2.0495886    1.74875499 -15.86450264]] 
                    relative_rotation (deg): 
                    [[-0.63119976 -0.1003184  -0.24068593]
                    [-0.59488026 -0.15880402 -0.1919978 ]
                    [-0.61275617 -0.20261214 -0.15235303]]
                    '''
                    # print(f"tcp_step_action: {tcp_step_action}")
                    tcp_step_action = relative_actions_to_absolute_actions(tcp_step_action, tcp_base_absolute_action) # lsq: shape = (5, 10)
                    gripper_step_action = relative_actions_to_absolute_actions(gripper_step_action, gripper_base_absolute_action) # lsq: shape = (6, 10)
                    # logger.info(f"After use_relative_action tcp_step_action: {tcp_step_action}")
                    # logger.info(f"After use_relative_action tcp_base_absolute_action: {tcp_base_absolute_action}")

                if tcp_step_action.shape[-1] == 4: # (x, y, z, gripper_width)
                    tcp_len = 3
                elif tcp_step_action.shape[-1] == 8: # (x_l, y_l, z_l, x_r, y_r, z_r, gripper_width_l, gripper_width_r)
                    tcp_len = 6
                elif tcp_step_action.shape[-1] == 10: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
                    tcp_len = 9
                elif tcp_step_action.shape[-1] == 20: # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r, gripper_width_l, gripper_width_r)
                    tcp_len = 18
                else:
                    raise NotImplementedError
                
                # if self.env.enable_exp_recording: # lsq: 不执行这行代码
                #     self.env.get_predicted_action(tcp_step_action[:, :tcp_len], type='partial_tcp')
                #     self.env.get_predicted_action(gripper_step_action[:, tcp_len:], type='partial_gripper')

                #     full_tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                #     full_gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                #     if self.use_relative_action:
                #         full_tcp_step_action = relative_actions_to_absolute_actions(full_tcp_step_action, tcp_base_absolute_action)
                #         full_gripper_step_action = relative_actions_to_absolute_actions(full_gripper_step_action, gripper_base_absolute_action)
                #     self.env.get_predicted_action(full_tcp_step_action[:, :tcp_len], type='full_tcp')
                #     self.env.get_predicted_action(full_gripper_step_action[:, tcp_len:], type='full_gripper')
                print(f"tcp_step_action | predict_from_latent_action | shape: {tcp_step_action.shape} \n{tcp_step_action}")
                # import pdb; pdb.set_trace()
                tcp_step_action = tcp_step_action[-1]  # 只取最后一个动作, shape = (10,)
                print(f"tcp_step_action | [-1] | shape: {tcp_step_action.shape} \n{tcp_step_action}")
                gripper_step_action = gripper_step_action[-1] # shape = (10,)

                tcp_step_action = tcp_step_action[:tcp_len] # (9,)
                gripper_step_action = gripper_step_action[tcp_len:] # (1,)

            combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1) # shape = (10,), 和机械臂初始位置计算axisangle theta: 4.332082530773278, v: [ 2.11951944e-01 -5.63084703e-04  9.77279927e-01]
            # convert to 16-D robot action (TCP + gripper of both arms)
            # TODO: handle rotation in temporal ensemble buffer!
            step_action, is_bimanual = self.post_process_action(combined_action[np.newaxis, :]) # shape = (16,), left arm dim=8: xyz rpy gripper_width 0 + right arm 和机械臂初始位置计算axisangle theta: 177.3042807154596, v: [ 0.50320059  0.49894076 -0.70558294]
            # logger.info(f"-----------{step_action}")
            step_action = step_action.squeeze(0)

            logger.info(f"execute_action shape: {step_action.shape} \n {step_action}") # shape = (16,) xyz + 6d rot + gripper_width + 0, 0 是 step_count; 再加右臂的 xyz + 6d rot + gripper_width + 0 共记 16 维

            # send action to the robot
            # input("continue")
            self.env.execute_action(step_action, use_relative_action=False, is_bimanual=is_bimanual)

            cur_time = time.time()
            precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
             # precise_sleep(max(0., 1/5 - (cur_time - start_time)))
            self.action_step_count += 1

    def start_record_video(self, video_path):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/start_recording/{video_path}')
            if response.status_code == 200:
                logger.info(f"Start recording video to {video_path}")
            else:
                logger.error(f"Failed to start recording video to {video_path}")

    def stop_record_video(self):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/stop_recording')
            if response.status_code == 200:
                logger.info(f"Stop recording video")
            else:
                logger.error(f"Failed to stop recording video")

    def run(self, policy: Union[DiffusionUnetImagePolicy]):
        """Run inference loop (Synchronous Mode)"""
        if self.use_latent_action_with_rnn_decoder:
            assert policy.at.use_rnn_decoder, "Policy should use rnn decoder for latent action."
        else:
            assert not hasattr(policy, 'at') or not policy.at.use_rnn_decoder, "Policy should not use rnn decoder for action."

        device = policy.device

        # Start observation collection
        self.env.start_obs_collection()

        try:
            time.sleep(2)
            for episode_idx in tqdm.tqdm(range(0, self.eval_episodes),
                                         desc=f"Eval for {self.task_name}",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                logger.info(f"Start evaluation episode {episode_idx}")

                # ask user whether the environment resetting is done
                reset_flag = py_cli_interaction.parse_cli_bool('Has the environment reset finished?', default_value=True)
                if not reset_flag:
                    logger.warning("Skip this episode.")
                    continue

                logger.info("Start episode rollout.")
                # Start rollout
                self.env.reset()
                self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
                time.sleep(1)

                policy.reset()
                self.tcp_ensemble_buffer.clear()
                self.gripper_ensemble_buffer.clear()
                logger.debug("Reset environment and policy.")

                self.stop_event.clear()
                time.sleep(0.5)

                self.action_step_count = 0
                step_count = 0
                steps_per_inference = int(self.control_fps / self.inference_fps) # 24 / 6 = 4
                start_timestamp = time.time()

                try:
                    while True:
                        # profiler = Profiler()
                        # profiler.start()
                        start_time = time.time()

                        # get obs
                        # import pdb; pdb.set_trace()
                        # input("continue")
                        obs = self.env.get_obs(obs_steps=self.n_obs_steps, temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
                        # obs = dict()
                        # import pdb; pdb.set_trace()
                        # dict_keys(['timestamp', 'left_robot_tcp_pose', 'left_robot_tcp_wrench', 'left_robot_gripper_width', 'left_robot_gripper_force', 'external_img', 'left_wrist_img', 'left_gripper1_initial_marker', 'left_gripper1_marker_offset', 'left_gripper2_initial_marker', 'left_gripper2_marker_offset', 'left_gripper1_marker_offset_emb', 'left_gripper2_marker_offset_emb'])
                        # 检查 ensemble buffer 状态
                        # 如果 buffer 不为空，说明还有动作等待执行，跳过推理，继续循环
                        # 如果 buffer 为空，说明需要新的推理来生成动作

                        
                        # buffer 为空，执行后续推理和动作添加逻辑
                        # logger.info(f"Buffer empty, starting inference for step {step_count}")
                        if len(obs) == 0:
                            logger.warning("No observation received! Skip this step.")
                            cur_time = time.time()
                            precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                            step_count += steps_per_inference
                            continue

                        # create obs dict
                        np_obs_dict = dict(obs)
                        np_obs_dict = get_real_obs_dict(env_obs=np_obs_dict, shape_meta=self.shape_meta)
                        np_obs_dict, np_absolute_obs_dict = self.pre_process_obs(np_obs_dict)

                        # device transfer
                        obs_dict = dict_apply(np_obs_dict,
                                              lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device))
                        
                        # for key, tensor in obs_dict.items():
                        #     print(f"|-- 输入观察 (Input Obs) - '{key}': shape={tensor.shape}")

                        policy_time = time.time()
                        # run policy
                        with torch.no_grad():
                            if self.use_latent_action_with_rnn_decoder:
                                action_dict = policy.predict_action(obs_dict,
                                                                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                                    return_latent_action=True)
                            else:
                                action_dict = policy.predict_action(obs_dict)
                        logger.info(f"Policy inference time: {time.time() - policy_time:.3f}s")
                        # if 'action' in action_dict:
                        #     action_tensor = action_dict['action']
                        #     logger.info(f"|-- 输出动作 (Output Action): shape={action_tensor.shape}")
                        # print("="*24 + " 模型推理结束 " + "="*24 + "\n")

                        # device_transfer
                        np_action_dict = dict_apply(action_dict,
                                                    lambda x: x.detach().to('cpu').numpy())
                        action_all = np_action_dict['action'].squeeze(0) # np_action_dict['action']: np.ndarray, shape = (B, T, Da), squeeze后shape = (T, Da)
                        
                        # 截取action_all的start,end区间的动作，并发布出去
                        start_index = 1
                        end_index = action_all.shape[0]
                        action_all_truncated = action_all[start_index:end_index]
                        left_rot_mat_batch = ortho6d_to_rotation_matrix(action_all_truncated[:, 3:9])  # (action_steps, 3, 3)
                        # left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                        left_euler_batch = np.asarray([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                        # left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch], dtype=np.float32)
                        left_trans_batch = action_all_truncated[:, :3]  # (action_steps, 3)
                        left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                        gripper_all_truncated = action_all_truncated[:, 9:]
                        
                        # 直接写一个循环，把action_all按照指定频率发布出去
                        for i in range(left_action_6d.shape[0]):
                            # if i % self.control_fps == 0:
                            step_action = left_action_6d[i] # relative action
                            step_gripper = gripper_all_truncated[i]
                            # self.action_command_thread(policy, self.stop_event)
                            # self.env.execute_relative_action(np.concatenate([step_action, step_gripper], axis=-1))
                            self.robot_controller.move_tcp(pose_6d_to_pose_7d(step_action))
                            self.env.send_gripper_command(step_gripper, step_gripper, is_bimanual=False)
                            time.sleep(0.04)

                        # if self.use_latent_action_with_rnn_decoder:
                        #     # add first absolute action to get absolute action
                        #     if self.use_relative_action:
                        #         # import pdb; pdb.set_trace()
                        #         # base_absolute_action = np.concatenate([
                        #         #     np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                        #         #     np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                        #         # ], axis=-1) # lsq: 目前为空，需要处理
                        #         base_absolute_action = np.concatenate([
                        #             obs['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs else np.array([]),
                        #             # obs['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs else np.array([])
                        #         ], axis=-1) # lsq: 把原始 obs 作为base_absolute_action # work 当yaml的 obs有 tcp pose 时
                        #         # base_absolute_action = np.concatenate([obs['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs else np.array([])], axis=-1) # lsq: 把原始 obs 作为base_absolute_action
                        #         # base_absolute_action = obs['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs else np.array([]) # lsq: 把原始 obs 作为base_absolute_action

                        #         action_all = np.concatenate([
                        #             action_all,
                        #             base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0)
                        #         ], axis=-1)
                        #     # add action step to get corresponding observation
                        #     action_all = np.concatenate([
                        #         action_all,
                        #         np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]
                        #     ], axis=-1)
                        # else: # 不执行这行代码
                        #     if self.use_relative_action:
                        #         # base_absolute_action = np.concatenate([
                        #         #     np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                        #         #     np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                        #         # ], axis=-1)
                        #         base_absolute_action = np.concatenate([
                        #             obs['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs else np.array([]),
                        #             # obs['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs else np.array([])
                        #         ], axis=-1) # lsq: 把原始 obs 作为base_absolute_action # work 当yaml的 obs有 tcp pose 时
                        #         action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)

                        # if self.action_interpolation_ratio > 1: # 当前设置 = 1 不执行这行代码
                        #     if self.use_latent_action_with_rnn_decoder:
                        #         action_all = action_all.repeat(self.action_interpolation_ratio, axis=0)
                        #     else:
                        #         action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)

                        # # TODO: only takes the first n_action_steps and add to the ensemble buffer
                        # # if step_count % self.tcp_action_update_interval == 0:

                        # if True:
                        #     if self.use_latent_action_with_rnn_decoder:
                        #         tcp_action = action_all[self.latency_step:, ...]
                        #     else:
                        #         if action_all.shape[-1] == 4:
                        #             tcp_action = action_all[self.latency_step:, :3]
                        #         elif action_all.shape[-1] == 8:
                        #             tcp_action = action_all[self.latency_step:, :6]
                        #         elif action_all.shape[-1] == 10:
                        #             tcp_action = action_all[self.latency_step:, :9]
                        #         elif action_all.shape[-1] == 20:
                        #             tcp_action = action_all[self.latency_step:, :18]
                        #         else:
                        #             raise NotImplementedError
                        #     # add to ensemble buffer
                        #     # logger.debug(f"Step: {step_count}, TCP_Shape {tcp_action.shape}.Add TCP action to ensemble buffer: {tcp_action}")
                        #     logger.debug(f"Step: {step_count}, TCP_Shape {tcp_action.shape}.") # TCP_Shape (25, 42)
                        #     tcp_action_len = tcp_action.shape[0]
                        #     # tcp_action = tcp_action[:tcp_action_len//2]
                        #     # nn = -4
                        #     # import pdb; pdb.set_trace()
                        #     # tcp_action = tcp_action[:nn]
                        #     print(f"tcp_ensemble_buffer length: {len(self.tcp_ensemble_buffer.actions)}")
                        #     print(f"gripper_ensemble_buffer length: {len(self.gripper_ensemble_buffer.actions)}")
                        #     step_count = 0
                        #     self.tcp_ensemble_buffer.clear()
                        #     # tcp_action = tcp_action[:-3]
                        #     self.tcp_ensemble_buffer.add_action(tcp_action, step_count)
                            
                        #     # step_count = 0
                            
                        #     # if len(self.tcp_ensemble_buffer.actions) > 0 or len(self.gripper_ensemble_buffer.actions) > 0:
                        #     # if len(self.tcp_ensemble_buffer.actions) > 0:
                        #     #     # buffer 不为空，跳过推理，等待动作被消费
                        #     #     cur_time = time.time()
                        #     #     precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                        #     #     continue
                        #     # else:
                        #     #     self.tcp_ensemble_buffer.add_action(tcp_action, step_count)

                        #     # if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder: # lsq:false
                        #     #     self.env.get_predicted_action(tcp_action, type='full_tcp')

                        # # if step_count % self.gripper_action_update_interval == 0:
                        # if True:
                        #     if self.use_latent_action_with_rnn_decoder:
                        #         gripper_action = action_all[self.gripper_latency_step:, ...]
                        #     else:
                        #         if action_all.shape[-1] == 4:
                        #             gripper_action = action_all[self.gripper_latency_step:, 3:]
                        #         elif action_all.shape[-1] == 8:
                        #             gripper_action = action_all[self.gripper_latency_step:, 6:]
                        #         elif action_all.shape[-1] == 10:
                        #             gripper_action = action_all[self.gripper_latency_step:, 9:]
                        #         elif action_all.shape[-1] == 20:
                        #             gripper_action = action_all[self.gripper_latency_step:, 18:]
                        #         else:
                        #             raise NotImplementedError
                        #     # add to ensemble buffer
                        #     # logger.debug(f"Step: {step_count}, Add gripper action to ensemble buffer: {gripper_action}")
                        #     logger.debug(f"Step: {step_count}, Add gripper action to ensemble buffer")
                        #     print(f"tcp_ensemble_buffer length: {len(self.tcp_ensemble_buffer.actions)}")
                        #     print(f"gripper_ensemble_buffer length: {len(self.gripper_ensemble_buffer.actions)}")
                        #     # if len(self.tcp_ensemble_buffer.actions) > 0 or len(self.gripper_ensemble_buffer.actions) > 0:
                        #     #     # buffer 不为空，跳过推理，等待动作被消费
                        #     #     cur_time = time.time()
                        #     #     precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                        #     #     continue
                        #     # else:
                        #     # step_count = 0
                        #     # gripper_action_len = gripper_action.shape[0]
                        #     # gripper_action = gripper_action[:nn]
                        #     self.gripper_ensemble_buffer.clear()
                        #     # gripper_action = gripper_action[:-8]
                        #     self.gripper_ensemble_buffer.add_action(gripper_action, step_count)
                            
                        #     # if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder: # lsq: false
                        #     #     self.env.get_predicted_action(gripper_action, type='full_gripper')

                        #     if len(self.tcp_ensemble_buffer.actions) > 0 or len(self.gripper_ensemble_buffer.actions) > 0:
                        #         # import pdb; pdb.set_trace()
                        #         self.action_command_thread(policy, self.stop_event)
                        #         # self.action_command(policy, tcp_action, gripper_action)
                        #         time.sleep(0.5)

                        # cur_time = time.time()
                        # precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                        # if cur_time - start_timestamp >= self.max_duration_time:
                        #     logger.info(f"Episode {episode_idx} reaches max duration time {self.max_duration_time} seconds.")
                        #     break
                        # step_count += steps_per_inference
                        # # profiler.stop()
                        # # profiler.print()

                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt! Terminate the episode now!")
                finally:
                    self.stop_event.set()
                    # action_thread.join()
                    if self.enable_video_recording:
                        self.stop_record_video()
                    # self.env.save_exp(episode_idx)

        finally:
            self.env.stop_obs_collection()
            self.shutdown()

    def shutdown(self):
        """Shutdown all sensors and environment"""
        logger.info("Shutting down...")

        self.env.shutdown()

        if hasattr(self, 'robot_controller'):
            self.robot_controller.disconnect()
        if hasattr(self, 'force_sensor'):
            self.force_sensor.disconnect()
        if self.gelsight_left is not None:
            self.gelsight_left.disconnect()
        if self.gelsight_right is not None:
            self.gelsight_right.disconnect()
        if self.mvs_camera is not None:
            self.mvs_camera.disconnect()
        if self.realsense_external is not None:
            self.realsense_external.disconnect()
        if self.realsense_wrist is not None:
            self.realsense_wrist.disconnect()

        logger.info("Shutdown complete")