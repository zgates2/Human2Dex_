from typing import Optional, List
import pathlib
import pickle
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.franky_chunk_controller import FrankyChunkController
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import draw_predefined_mask
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator


def _force_vector(value, name: str, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != size:
        raise ValueError(f"{name} must contain {size} values, got shape {np.asarray(value).shape}")
    return arr


def _force_tool_offset(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] == 6:
        arr = arr[:3]
    if arr.shape[0] != 3:
        raise ValueError(f"force_tool_offset must contain 3 values, got shape {np.asarray(value).shape}")
    return arr


def _load_force_calibration(path):
    cali_path = pathlib.Path(path).expanduser()
    with cali_path.open('rb') as f:
        cali_params = pickle.load(f)
    return cali_params['cali_info'], cali_params['cali_bias']


def _parse_force_compensation_config(config):
    if config is None:
        return None
    if not config.get('gravity_compensation', False):
        return None

    if 'cali_params_path' in config:
        cali_info, cali_bias = _load_force_calibration(config['cali_params_path'])
    else:
        if 'cali_info' not in config or 'cali_bias' not in config:
            raise ValueError(
                "force gravity_compensation requires cali_info/cali_bias "
                "or cali_params_path"
            )
        cali_info = config['cali_info']
        cali_bias = config['cali_bias']

    tool_offset = config.get('force_tool_offset', config.get('tool_offset', [0.0, 0.0, 0.2165]))
    cali_bias = _force_vector(cali_bias, 'cali_bias', 6)
    mass = np.linalg.norm(cali_bias[:3]) / 9.81
    if mass <= 1e-9:
        raise ValueError("force cali_bias[:3] gravity vector is near zero")

    return {
        'cali_info': _force_vector(cali_info, 'cali_info', 6),
        'cali_bias': cali_bias,
        'force_tool_offset': _force_tool_offset(tool_offset),
        'mass': mass,
    }


def _parse_device_id(value):
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def _create_force_sensor_from_config(
        shm_manager,
        config,
        default_receive_latency: float = 0.0):
    if config is None:
        return None
    if 'port' not in config:
        raise ValueError("force sensor config requires port")

    from umi.real_world.force_sensor_controller import ForceSensorController

    return ForceSensorController(
        shm_manager=shm_manager,
        port=config['port'],
        baudrate=int(config.get('baudrate', config.get('baud', 1000000))),
        device_id=_parse_device_id(config.get('device_id', 0x01)),
        frequency=float(config.get('frequency', 500.0)),
        get_max_k=config.get('get_max_k', None),
        command_queue_size=int(config.get('command_queue_size', 64)),
        launch_timeout=float(config.get('launch_timeout', 5.0)),
        receive_latency=float(config.get('obs_latency', default_receive_latency)),
        zero_on_start=False,
        verbose=bool(config.get('verbose', False)),
    )


def _gravity_compensate_wrench(raw_wrench, rotation_map_force, comp_config):
    raw = np.asarray(raw_wrench, dtype=np.float64)
    single = raw.ndim == 1
    raw = raw.reshape(-1, 6)

    rot = np.asarray(rotation_map_force, dtype=np.float64)
    if rot.ndim == 2:
        rot = rot[None, ...]
    if rot.shape[0] != raw.shape[0]:
        raise ValueError(
            f"rotation_map_force has {rot.shape[0]} samples, raw_wrench has {raw.shape[0]}"
        )

    cali_info = comp_config['cali_info']
    cali_bias = comp_config['cali_bias']
    force_tool_offset = comp_config['force_tool_offset']
    mass = comp_config['mass']

    f_raw = raw[:, :3]
    t_raw = raw[:, 3:]

    f_bias = cali_bias[3:6]
    t_bias = cali_info[3:6]
    r_cg_param = cali_info[:3]
    mg_base = cali_bias[:3]

    f_gravity = np.einsum('nji,j->ni', rot, mg_base)
    g_sensor = f_gravity / mass
    t_gravity = np.cross(np.broadcast_to(r_cg_param, f_gravity.shape), g_sensor)

    f_corrected = f_raw - f_bias - f_gravity
    t_corrected = t_raw - t_bias - t_gravity
    t_ee = t_corrected + np.cross(
        np.broadcast_to(force_tool_offset, f_corrected.shape),
        f_corrected,
    )

    compensated = np.concatenate([f_corrected, t_ee], axis=-1).astype(np.float32)
    if single:
        return compensated[0]
    return compensated


class BimanualUmiEnv:
    def __init__(self,
            # required params
            output_dir,
            robots_config, # list of dict[{robot_type: 'ur5', robot_ip: XXX, obs_latency: 0.0001, action_latency: 0.1, tcp_offset: 0.21}]
            grippers_config, # list of dict[{gripper_ip: XXX, gripper_port: 1000, obs_latency: 0.01, , action_latency: 0.1}]
            # env params
            frequency=20,
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=False,
            fisheye_converter=None,
            mirror_swap=False,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            init_joints=False,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None,
            # ---- new: 注入式硬件 ----
            # 相机的物理捕获 fps,影响 get_obs 里向后取多少帧;原先硬编码 60
            camera_capture_fps: float = 60.0,
            # 预构建的相机实例,提供则跳过默认 MultiUvcCamera 构造(用于 MultiMvsCamera 等)
            camera=None,
            # 预构建的夹爪实例列表,提供则跳过默认 WSGController 构造
            grippers: Optional[List] = None,
            # 力传感器:每机器人一个;None 表示该机器人无力传感器
            force_sensors: Optional[List] = None,
            force_sensors_config: Optional[List] = None,
            force_obs_horizon: int = 2,
            force_down_sample_steps: int = 1,
            force_obs_latency: float = 0.0,
            clear_robot_queue_on_exec: bool = True,
            vis_rgb_to_bgr: Optional[bool] = None,
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        injected_camera = camera is not None
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        if camera is None:
            # ====== 默认路径:UVC + Elgato 采集卡 ======
            # Find and reset all Elgato capture cards.
            # Required to workaround a firmware bug.
            reset_all_elgato_devices()

            # Wait for all v4l cameras to be back online
            time.sleep(0.1)
            v4l_paths = get_sorted_v4l_paths()
            if camera_reorder is not None:
                paths = [v4l_paths[i] for i in camera_reorder]
                v4l_paths = paths

            # compute resolution for vis
            rw, rh, col, row = optimal_row_cols(
                n_cameras=len(v4l_paths),
                in_wh_ratio=4/3,
                max_resolution=multi_cam_vis_resolution
            )

            # HACK: Separate video setting for each camera
            # Elagto Cam Link 4k records at 4k 30fps
            # Other capture card records at 720p 60fps
            resolution = list()
            capture_fps = list()
            cap_buffer_size = list()
            video_recorder = list()
            transform = list()
            vis_transform = list()
            for path in v4l_paths:
                if 'Cam_Link_4K' in path:
                    res = (3840, 2160)
                    fps = 30
                    buf = 3
                    bit_rate = 6000*1000
                    def tf4k(data, input_res=res):
                        img = data['color']
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution,
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = f(img)
                        if obs_float32:
                            img = img.astype(np.float32) / 255
                        data['color'] = img
                        return data
                    transform.append(tf4k)
                else:
                    res = (1920, 1080)
                    fps = 60
                    buf = 1
                    bit_rate = 3000*1000

                    is_mirror = None
                    if mirror_swap:
                        mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                        mirror_mask = draw_predefined_mask(
                            mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
                        is_mirror = (mirror_mask[...,0] == 0)

                    def tf(data, input_res=res):
                        img = data['color']
                        if fisheye_converter is None:
                            f = get_image_transform(
                                input_res=input_res,
                                output_res=obs_image_resolution,
                                # obs output rgb
                                bgr_to_rgb=True)
                            img = np.ascontiguousarray(f(img))
                            if is_mirror is not None:
                                img[is_mirror] = img[:,::-1,:][is_mirror]
                            img = draw_predefined_mask(img, color=(0,0,0),
                                mirror=no_mirror, gripper=True, finger=False, use_aa=True)
                        else:
                            img = fisheye_converter.forward(img)
                            img = img[...,::-1]
                        if obs_float32:
                            img = img.astype(np.float32) / 255
                        data['color'] = img
                        return data
                    transform.append(tf)

                resolution.append(res)
                capture_fps.append(fps)
                cap_buffer_size.append(buf)
                video_recorder.append(VideoRecorder.create_hevc_nvenc(
                    fps=fps,
                    input_pix_fmt='bgr24',
                    bit_rate=bit_rate
                ))

                def vis_tf(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=(rw,rh),
                        bgr_to_rgb=False
                    )
                    img = f(img)
                    data['color'] = img
                    return data
                vis_transform.append(vis_tf)

            camera = MultiUvcCamera(
                dev_video_paths=v4l_paths,
                shm_manager=shm_manager,
                resolution=resolution,
                capture_fps=capture_fps,
                # send every frame immediately after arrival
                # ignores put_fps
                put_downsample=False,
                get_max_k=max_obs_buffer_size,
                receive_latency=camera_obs_latency,
                cap_buffer_size=cap_buffer_size,
                transform=transform,
                vis_transform=vis_transform,
                video_recorder=video_recorder,
                verbose=False
            )
        else:
            # ====== 注入路径:caller 已经构造好 camera(可以是 MultiMvsCamera 等)======
            rw, rh, col, row = optimal_row_cols(
                n_cameras=camera.n_cameras,
                in_wh_ratio=4/3,
                max_resolution=multi_cam_vis_resolution,
            )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            if vis_rgb_to_bgr is None:
                # Default UVC visualization data is BGR already. Injected cameras
                # from camera_factory, especially MVS, provide RGB frames.
                vis_rgb_to_bgr = injected_camera
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=vis_rgb_to_bgr
            )

        cube_diag = np.linalg.norm([1,1,1])

        assert len(robots_config) == len(grippers_config)
        robots: List = list()
        for rc in robots_config:
            if rc['robot_type'].startswith('ur5'):
                from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController

                assert rc['robot_type'] in ['ur5', 'ur5e']
                this_robot = RTDEInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    frequency=500 if rc['robot_type'] == 'ur5e' else 125,
                    lookahead_time=0.1,
                    gain=300,
                    max_pos_speed=max_pos_speed*cube_diag,
                    max_rot_speed=max_rot_speed*cube_diag,
                    launch_timeout=3,
                    tcp_offset_pose=[0, 0, rc['tcp_offset'], 0, 0, 0],
                    payload_mass=None,
                    payload_cog=None,
                    joints_init=rc['joints_init'] if init_joints else None,
                    joints_init_speed=1.05,
                    soft_real_time=False,
                    verbose=False,
                    receive_keys=None,
                    receive_latency=rc['robot_obs_latency']
                )
            elif rc.get('control_backend') == 'franky_chunk':
                this_robot = FrankyChunkController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    robot_port=rc.get('robot_port', 4243),
                    frequency=rc.get('franky_state_frequency', 100),
                    launch_timeout=rc.get('launch_timeout', 60.0),
                    rpc_timeout=rc.get('rpc_timeout', 60.0),
                    read_only=rc.get('read_only', False),
                    joints_init=rc['joints_init'] if init_joints else None,
                    joints_init_duration=rc.get('joints_init_duration', 4.0),
                    verbose=rc.get('verbose', False),
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    max_chunk_size=rc.get('franky_max_chunk_size', 64),
                    min_segment_duration=rc.get('franky_min_segment_duration', 0.01),
                    tx_flange_tip=rc.get('tx_flange_tip', None),
                    receive_latency=rc['robot_obs_latency'],
                )
            elif rc['robot_type'].startswith('franka'):
                this_robot = FrankaInterpolationController(
                    shm_manager=shm_manager,
                    robot_ip=rc['robot_ip'],
                    robot_port=rc.get('robot_port', 4242),
                    frequency=rc.get('frequency', 200),
                    Kx_scale=rc.get('Kx_scale', 1.0),
                    Kxd_scale=rc.get('Kxd_scale', np.array([2.0,1.5,2.0,1.0,1.0,1.0])),
                    launch_timeout=rc.get('launch_timeout', 20),
                    pose_api=rc.get('pose_api', 'rpy'),
                    rpc_timeout=rc.get('rpc_timeout', 5.0),
                    read_only=rc.get('read_only', False),
                    start_impedance_on_start=rc.get('start_impedance_on_start', False),
                    impedance_start_delay=rc.get('impedance_start_delay', 0.5),
                    joints_init=rc['joints_init'] if init_joints else None,
                    joints_init_duration=rc.get('joints_init_duration', 4),
                    verbose=rc.get('verbose', False),
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    max_commands_per_cycle=rc.get('max_commands_per_cycle', 1),
                    tx_flange_tip=rc.get('tx_flange_tip', None),
                    receive_latency=rc['robot_obs_latency']
                )
            else:
                raise NotImplementedError()
            robots.append(this_robot)

        if grippers is None:
            # 默认路径:按 grippers_config[i].gripper_type 分派
            #   "wsg"      -> WSGController (旧默认)
            #   "lkmotor"  -> LkGripperProxy (omniumi-style RS485 + MIT impedance)
            assert len(robots_config) == len(grippers_config)
            grippers = []
            for gc in grippers_config:
                gtype = gc.get('gripper_type', 'wsg')
                if gtype == 'wsg':
                    from umi.real_world.wsg_controller import WSGController

                    this_gripper = WSGController(
                        shm_manager=shm_manager,
                        hostname=gc['gripper_ip'],
                        port=gc['gripper_port'],
                        receive_latency=gc['gripper_obs_latency'],
                        use_meters=True,
                    )
                elif gtype == 'lkmotor':
                    from umi.real_world.lk_gripper_proxy import LkGripperProxy
                    proxy_cfg = {
                        'mode': gc.get('mode', 'single'),
                        'port1': gc['port1'],
                        'motor1_id': gc['motor1_id'],
                    }
                    for key in ['close_offset_rad', 'close_offset_active_below_rad']:
                        if key in gc:
                            proxy_cfg[key] = gc[key]
                    if proxy_cfg['mode'] == 'dual':
                        proxy_cfg['port2'] = gc['port2']
                        proxy_cfg['motor2_id'] = gc['motor2_id']
                    this_gripper = LkGripperProxy(
                        config=proxy_cfg,
                        receive_latency=gc.get('gripper_obs_latency', 0.0),
                        launch_timeout=gc.get('launch_timeout', 30.0),
                    )
                else:
                    raise NotImplementedError(f"unknown gripper_type: {gtype!r}")
                grippers.append(this_gripper)
        else:
            # 注入路径:caller 提供 (LkGripperProxy / WSGController / 混合);
            # grippers_config 的 gripper_action_latency / gripper_obs_latency 仍由 env 用于 exec_actions 时序补偿
            assert len(grippers) == len(robots_config), \
                f"len(grippers)={len(grippers)} must match len(robots_config)={len(robots_config)}"

        if force_sensors_config is None:
            force_sensors_config = [None] * len(robots_config)
        else:
            assert len(force_sensors_config) == len(robots_config), \
                f"len(force_sensors_config)={len(force_sensors_config)} must match len(robots_config)={len(robots_config)}"
        # force sensors: 长度对齐 robots,None 表示该机器人无力传感器
        if force_sensors is None:
            force_sensors = [
                _create_force_sensor_from_config(
                    shm_manager=shm_manager,
                    config=config,
                    default_receive_latency=force_obs_latency,
                )
                for config in force_sensors_config
            ]
        else:
            assert len(force_sensors) == len(robots_config), \
                f"len(force_sensors)={len(force_sensors)} must match len(robots_config)={len(robots_config)}"

        self.camera = camera

        self.robots = robots
        self.robots_config = robots_config
        self.grippers = grippers
        self.grippers_config = grippers_config
        self.force_sensors = force_sensors
        self.force_compensation_configs = [
            _parse_force_compensation_config(config)
            for config in force_sensors_config
        ]
        if any(config is not None for config in self.force_compensation_configs[1:]):
            raise NotImplementedError(
                "force gravity compensation is currently supported only for robot0/single-arm setup"
            )

        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.camera_capture_fps = camera_capture_fps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        # timing
        self.camera_obs_latency = camera_obs_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.force_down_sample_steps = force_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.force_obs_horizon = force_obs_horizon
        self.force_obs_latency = force_obs_latency
        self.clear_robot_queue_on_exec = bool(clear_robot_queue_on_exec)
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self.last_time_step = 0
    
    # ======== start-stop API =============
    def get_ready_status(self):
        def component_status(component):
            status = {}
            try:
                status['ready'] = bool(component.is_ready)
            except Exception as e:
                status['ready'] = False
                status['ready_error'] = repr(e)

            if hasattr(component, 'is_alive'):
                try:
                    status['alive'] = bool(component.is_alive())
                except Exception as e:
                    status['alive_error'] = repr(e)
            if hasattr(component, 'exitcode'):
                try:
                    status['exitcode'] = component.exitcode
                except Exception:
                    pass
            proc = getattr(component, '_process', None)
            if proc is not None:
                try:
                    status['process_alive'] = bool(proc.is_alive())
                    status['process_exitcode'] = proc.exitcode
                except Exception as e:
                    status['process_error'] = repr(e)
            thread = getattr(component, '_state_thread', None)
            if thread is not None:
                try:
                    status['state_thread_alive'] = bool(thread.is_alive())
                except Exception as e:
                    status['state_thread_error'] = repr(e)
            children = getattr(component, 'cameras', None)
            if children is not None:
                status['children'] = {
                    str(name): component_status(child)
                    for name, child in children.items()
                }
            return status

        return {
            'camera': component_status(self.camera),
            'robots': [
                component_status(robot)
                for robot in self.robots
            ],
            'grippers': [
                component_status(gripper)
                for gripper in self.grippers
            ],
            'force_sensors': [
                None if fs is None else component_status(fs)
                for fs in self.force_sensors
            ],
        }

    @property
    def is_ready(self):
        ready_flag = self.camera.is_ready
        for robot in self.robots:
            ready_flag = ready_flag and robot.is_ready
        for gripper in self.grippers:
            ready_flag = ready_flag and gripper.is_ready
        for fs in self.force_sensors:
            if fs is not None:
                ready_flag = ready_flag and fs.is_ready
        return ready_flag

    def start(self, wait=True):
        self.camera.start(wait=False)
        for robot in self.robots:
            robot.start(wait=False)
        for gripper in self.grippers:
            gripper.start(wait=False)
        for fs in self.force_sensors:
            if fs is not None:
                fs.start(wait=False)

        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        if self.obs_accumulator is not None or self.action_accumulator is not None:
            try:
                self.end_episode()
            except Exception as e:
                print(f"[WARN] end_episode during stop failed: {e}")
                print(f"[WARN] ready status: {self.get_ready_status()}")
                self.obs_accumulator = None
                self.action_accumulator = None
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        for fs in self.force_sensors:
            if fs is not None:
                fs.stop(wait=False)
        for robot in self.robots:
            robot.stop(wait=False)
        for gripper in self.grippers:
            gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        for robot in self.robots:
            robot.start_wait()
        for gripper in self.grippers:
            gripper.start_wait()
        for fs in self.force_sensors:
            if fs is not None:
                fs.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        for fs in self.force_sensors:
            if fs is not None:
                fs.stop_wait()
        for robot in self.robots:
            robot.stop_wait()
        for gripper in self.grippers:
            gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Timestamp alignment policy
        We assume the cameras used for obs are always [0, k - 1], where k is the number of robots
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        """

        "observation dict"
        if not self.is_ready:
            raise RuntimeError(
                f"BimanualUmiEnv is not ready: {self.get_ready_status()}"
            )

        # get data
        # camera_capture_fps Hz, camera_calibrated_timestamp
        if type(self.camera_down_sample_steps) == int:
            k = math.ceil(
                self.camera_obs_horizon * self.camera_down_sample_steps \
                * (self.camera_capture_fps / self.frequency)) + 2 # here 2 is adjustable, typically 1 should be enough
        elif type(self.camera_down_sample_steps) == list:
            k = math.ceil(max(self.camera_down_sample_steps) * (self.camera_capture_fps / self.frequency)) + 2
        # print('==>k  ', k, self.camera_obs_horizon, self.camera_down_sample_steps, self.frequency)
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # both have more than n_obs_steps data
        last_robots_data = list()
        last_grippers_data = list()
        last_force_data = list()  # 长度对齐 robots,None 表示该机器人无力传感器
        # 125/500 hz, robot_receive_timestamp
        for robot in self.robots:
            last_robots_data.append(robot.get_all_state())
        # 30 hz, gripper_receive_timestamp
        for gripper in self.grippers:
            last_grippers_data.append(gripper.get_all_state())
        # ~500 hz, force_receive_timestamp
        for fs in self.force_sensors:
            last_force_data.append(fs.get_all_state() if fs is not None else None)

        # select align_camera_idx
        num_obs_cameras = len(self.robots)
        align_camera_idx = None
        running_best_error = np.inf
   
        for camera_idx in range(num_obs_cameras):
            this_error = 0
            this_timestamp = self.last_camera_data[camera_idx]['timestamp'][-1]
            for other_camera_idx in range(num_obs_cameras):
                if other_camera_idx == camera_idx:
                    continue
                other_timestep_idx = -1
                while True:
                    if self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx] < this_timestamp:
                        this_error += this_timestamp - self.last_camera_data[other_camera_idx]['timestamp'][other_timestep_idx]
                        break
                    other_timestep_idx -= 1
            if align_camera_idx is None or this_error < running_best_error:
                running_best_error = this_error
                align_camera_idx = camera_idx

        last_timestamp = self.last_camera_data[align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # align camera obs timestamps
        if type(self.camera_down_sample_steps) == int:
            camera_obs_timestamps = last_timestamp - (
                np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        elif type(self.camera_down_sample_steps) == list:
            camera_obs_timestamps = last_timestamp - np.array(self.camera_down_sample_steps) * dt
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                # if np.abs(this_timestamps - t)[nn_idx] > 1.0 / 120 and camera_idx != 3:
                #     print('ERROR!!!  ', camera_idx, len(this_timestamps), nn_idx, (this_timestamps - t)[nn_idx-1: nn_idx+2])
                this_idxs.append(nn_idx)
            # remap key
            camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        # obs_data to return (it only includes camera data at this stage)
        obs_data = dict(camera_obs)

        # include camera timesteps
        obs_data['timestamp'] = camera_obs_timestamps

        # align robot obs
        if type(self.robot_down_sample_steps) == int:
            robot_obs_timestamps = last_timestamp - (
                np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        elif type(self.robot_down_sample_steps) == list:
            robot_obs_timestamps = last_timestamp - np.array(self.robot_down_sample_steps) * dt
        robot_pose_interpolators = list()
        for robot_idx, last_robot_data in enumerate(last_robots_data):
            robot_pose_interpolator = PoseInterpolator(
                t=last_robot_data['robot_timestamp'], 
                x=last_robot_data['ActualTCPPose'])
            robot_pose_interpolators.append(robot_pose_interpolator)
            robot_pose = robot_pose_interpolator(robot_obs_timestamps)
            robot_obs = {
                f'robot{robot_idx}_eef_pos': robot_pose[...,:3],
                f'robot{robot_idx}_eef_rot_axis_angle': robot_pose[...,3:]
            }
            # update obs_data
            obs_data.update(robot_obs)

        # align gripper obs
        if type(self.gripper_down_sample_steps) == int:
            gripper_obs_timestamps = last_timestamp - (
                np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        elif type(self.gripper_down_sample_steps) == list:
            gripper_obs_timestamps = last_timestamp - np.array(self.gripper_down_sample_steps) * dt
        for robot_idx, last_gripper_data in enumerate(last_grippers_data):
            # align gripper obs
            gripper_interpolator = get_interp1d(
                t=last_gripper_data['gripper_timestamp'],
                x=last_gripper_data['gripper_position'][...,None]
            )
            gripper_obs = {
                f'robot{robot_idx}_gripper_width': gripper_interpolator(gripper_obs_timestamps)
            }

            # update obs_data
            obs_data.update(gripper_obs)

        # align force obs (optional, per-robot)
        if any(fs is not None for fs in self.force_sensors):
            if isinstance(self.force_down_sample_steps, int):
                force_obs_timestamps = last_timestamp - (
                    np.arange(self.force_obs_horizon)[::-1]
                    * self.force_down_sample_steps * dt)
            else:
                force_obs_timestamps = last_timestamp - np.array(self.force_down_sample_steps) * dt
            for robot_idx, fs_data in enumerate(last_force_data):
                if fs_data is None:
                    continue
                if fs_data['force_timestamp'].shape[0] < 2:
                    # buffer 还没攒够;先填 0,policy 跑到这里通常已经攒够了。
                    zeros = np.zeros((force_obs_timestamps.shape[0], 6), dtype=np.float32)
                    obs_data[f'robot{robot_idx}_wrench_raw'] = zeros
                    obs_data[f'robot{robot_idx}_wrench'] = zeros
                    continue
                force_interpolator = get_interp1d(
                    t=fs_data['force_timestamp'],
                    x=fs_data['wrench']
                )
                raw_wrench = force_interpolator(force_obs_timestamps).astype(np.float32)
                obs_data[f'robot{robot_idx}_wrench_raw'] = raw_wrench

                comp_config = self.force_compensation_configs[robot_idx]
                if comp_config is not None:
                    force_robot_pose = robot_pose_interpolators[robot_idx](force_obs_timestamps)
                    _, force_robot_rot = pose_to_pos_rot(force_robot_pose)
                    rotation_map_force = force_robot_rot.as_matrix()
                    obs_data[f'robot{robot_idx}_wrench'] = _gravity_compensate_wrench(
                        raw_wrench,
                        rotation_map_force,
                        comp_config,
                    )
                else:
                    obs_data[f'robot{robot_idx}_wrench'] = raw_wrench

        # accumulate obs
        if self.obs_accumulator is not None:
            for robot_idx, last_robot_data in enumerate(last_robots_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_eef_pose': last_robot_data['ActualTCPPose'],
                        f'robot{robot_idx}_joint_pos': last_robot_data['ActualQ'],
                        f'robot{robot_idx}_joint_vel': last_robot_data['ActualQd'],
                    },
                    timestamps=last_robot_data['robot_timestamp']
                )

            for robot_idx, last_gripper_data in enumerate(last_grippers_data):
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_gripper_width': last_gripper_data['gripper_position'][...,None]
                    },
                    timestamps=last_gripper_data['gripper_timestamp']
                )

            # 力数据原始流。end_episode() 会按 action timestamps 插值写入
            # replay buffer;不要在这里让在线补偿流参与录制时间轴。
            for robot_idx, fs_data in enumerate(last_force_data):
                if fs_data is None or fs_data['force_timestamp'].shape[0] == 0:
                    continue
                self.obs_accumulator.put(
                    data={
                        f'robot{robot_idx}_wrench_raw': fs_data['wrench'],
                    },
                    timestamps=fs_data['force_timestamp'],
                )

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        assert new_actions.shape[1] // len(self.robots) == 7
        assert new_actions.shape[1] % len(self.robots) == 0

        if len(new_actions) > 0 and self.clear_robot_queue_on_exec:
            for robot in self.robots:
                robot.clear_queue()

        chunk_capable = [
            callable(getattr(robot, 'schedule_waypoints', None))
            for robot in self.robots
        ]

        # Franky consumes one complete action chunk instead of a high-rate
        # stream of newly-created Cartesian motions.
        for robot_idx, (robot, rc) in enumerate(zip(self.robots, self.robots_config)):
            if not chunk_capable[robot_idx] or len(new_actions) == 0:
                continue
            r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
            robot.schedule_waypoints(
                poses=new_actions[:, 7 * robot_idx: 7 * robot_idx + 6],
                target_times=new_timestamps - r_latency,
            )

        # Preserve the original per-waypoint path for Polymetis/UR robots and
        # keep gripper scheduling identical for every backend.
        for i in range(len(new_actions)):
            for robot_idx, (robot, gripper, rc, gc) in enumerate(zip(self.robots, self.grippers, self.robots_config, self.grippers_config)):
                r_latency = rc['robot_action_latency'] if compensate_latency else 0.0
                g_latency = gc['gripper_action_latency'] if compensate_latency else 0.0
                r_actions = new_actions[i, 7 * robot_idx + 0: 7 * robot_idx + 6]
                g_actions = new_actions[i, 7 * robot_idx + 6]
                if not chunk_capable[robot_idx]:
                    robot.schedule_waypoint(
                        pose=r_actions,
                        target_time=new_timestamps[i] - r_latency
                    )
                gripper.schedule_waypoint(
                    pos=g_actions,
                    target_time=new_timestamps[i] - g_latency
                )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
    
    def get_robot_state(self):
        return [robot.get_state() for robot in self.robots]
    
    def get_gripper_state(self):
        return [gripper.get_state() for gripper in self.grippers]

    def prepare_episode_start(self, start_time=None):
        """Run configured pre-episode hardware preparation."""
        if start_time is None:
            start_time = time.time()
        now = time.time()

        for robot_idx, (gripper, gc) in enumerate(zip(self.grippers, self.grippers_config)):
            open_pos = gc.get('start_episode_open_rad', gc.get('open_on_start_rad', None))
            if open_pos is None:
                continue
            duration = float(gc.get(
                'start_episode_open_duration_s',
                gc.get('open_on_start_duration_s', 1.0),
            ))
            force_limit = float(gc.get('start_episode_open_force_limit_nm', 1.5))
            target_time = min(float(start_time), now + max(0.0, duration))
            target_time = max(now, target_time)
            try:
                gripper.schedule_waypoint(
                    pos=float(open_pos),
                    target_time=target_time,
                    force_limit_nm=force_limit,
                )
            except TypeError:
                gripper.schedule_waypoint(float(open_pos), target_time)
            print(
                f"Pre-start gripper{robot_idx}: opening to {float(open_pos):.3f} rad "
                f"by {target_time - now:.2f}s"
            )
        return start_time

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready
        self.prepare_episode_start(start_time)

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
    
    def end_episode(self):
        "Stop recording"
        ready_status = None
        if not self.is_ready:
            ready_status = self.get_ready_status()
            print(f"[WARN] end_episode while env not ready: {ready_status}")

        # stop video recorder
        try:
            self.camera.stop_recording()
        except Exception as e:
            print(f"[WARN] camera.stop_recording failed: {e}")

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                if '_wrench' in key:
                    continue
                end_time = min(end_time, value[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            if action_timestamps.shape[0] == 0:
                self.obs_accumulator = None
                self.action_accumulator = None
                return
            end_time = min(end_time, action_timestamps[-1])
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }

                def interp_accumulator_key(key):
                    if key not in self.obs_accumulator.timestamps:
                        return None
                    key_timestamps = np.array(self.obs_accumulator.timestamps[key])
                    key_data = np.array(self.obs_accumulator.data[key])
                    if key_timestamps.shape[0] == 0:
                        return None
                    if key_timestamps.shape[0] == 1:
                        return np.repeat(key_data[:1], len(timestamps), axis=0)
                    return get_interp1d(
                        t=key_timestamps,
                        x=key_data,
                    )(timestamps)

                for robot_idx in range(len(self.robots)):
                    robot_pose_interpolator = PoseInterpolator(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_eef_pose']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_eef_pose'])
                    )
                    robot_pose = robot_pose_interpolator(timestamps)
                    episode[f'robot{robot_idx}_eef_pos'] = robot_pose[:,:3]
                    episode[f'robot{robot_idx}_eef_rot_axis_angle'] = robot_pose[:,3:]
                    joint_pos_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_pos']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_pos'])
                    )
                    joint_vel_interpolator = get_interp1d(
                        np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_joint_vel']),
                        np.array(self.obs_accumulator.data[f'robot{robot_idx}_joint_vel'])
                    )
                    episode[f'robot{robot_idx}_joint_pos'] = joint_pos_interpolator(timestamps)
                    episode[f'robot{robot_idx}_joint_vel'] = joint_vel_interpolator(timestamps)

                    gripper_interpolator = get_interp1d(
                        t=np.array(self.obs_accumulator.timestamps[f'robot{robot_idx}_gripper_width']),
                        x=np.array(self.obs_accumulator.data[f'robot{robot_idx}_gripper_width'])
                    )
                    episode[f'robot{robot_idx}_gripper_width'] = gripper_interpolator(timestamps)

                    force_raw = interp_accumulator_key(f'robot{robot_idx}_wrench_raw')
                    if force_raw is not None:
                        force_raw = force_raw.astype(np.float32)
                        episode[f'robot{robot_idx}_wrench_raw'] = force_raw

                        comp_config = self.force_compensation_configs[robot_idx]
                        if comp_config is not None:
                            _, force_robot_rot = pose_to_pos_rot(robot_pose)
                            episode[f'robot{robot_idx}_wrench'] = _gravity_compensate_wrench(
                                force_raw,
                                force_robot_rot.as_matrix(),
                                comp_config,
                            )
                        else:
                            episode[f'robot{robot_idx}_wrench'] = force_raw

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
