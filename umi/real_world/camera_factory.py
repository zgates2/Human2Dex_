"""Helpers for building cameras from the robot-config YAML.

Both eval_real.py and umi_policy_client.py call build_camera_from_yaml() so
the YAML `cameras` section is honored in one place.
"""
from typing import Optional, Tuple
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from umi.real_world.video_recorder import VideoRecorder


def _tuple_or_none(value):
    if value is None:
        return None
    if len(value) > 0 and isinstance(value[0], (list, tuple)):
        return [tuple(v) for v in value]
    return tuple(value)


def build_camera_from_yaml(
        shm_manager: SharedMemoryManager,
        cameras_cfg: Optional[dict],
        obs_image_resolution: Optional[Tuple[int, int]] = None,
        camera_obs_latency: Optional[float] = None,
        ):
    """Return (camera_or_None, capture_fps, obs_latency).

    - camera_or_None is None only for the legacy "uvc" backend; the caller
      falls back to the UVC auto-detect path inside BimanualUmiEnv.
    - "mvs" keeps the existing wrist-camera path.
    - "l515_subprocess" uses a dedicated RealSense worker environment for
      the L515 ego view; "realsense"/"l515" is also available when the main
      environment itself has pyrealsense2. Both expose camera0_rgb unchanged.
    - capture_fps / obs_latency are extracted so the caller can pass them
      through to env even on the UVC path (env needs camera_capture_fps for
      get_obs window sizing).
    """
    if cameras_cfg is None:
        return None, 60.0, camera_obs_latency

    view = str(cameras_cfg.get('view', 'wrist')).strip().lower()
    backend = str(cameras_cfg.get('backend', 'uvc')).lower()
    # Allow a config to switch only the physical view while leaving the
    # remainder of the inference pipeline untouched.  If view=ego is set but
    # an old backend value is still present, use the explicitly configured
    # ego backend (default: the cross-environment L515 worker).
    if view in {'ego', 'l515'} and backend in {'uvc', 'mvs'}:
        backend = str(cameras_cfg.get('ego_backend', 'l515_subprocess')).lower()
    capture_fps = float(cameras_cfg.get('capture_fps', 60.0))
    obs_latency = float(cameras_cfg.get('obs_latency', camera_obs_latency or 0.0))

    if backend == 'uvc':
        return None, capture_fps, obs_latency

    if backend == 'mvs':
        from umi.real_world.multi_mvs_camera import MultiMvsCamera
        from umi.real_world.mvs_camera import create_camera_configs_from_serials
        serials = list(cameras_cfg.get('serials') or [])
        if not serials:
            raise ValueError("cameras.backend='mvs' but cameras.serials is empty")
        cfg_res = cameras_cfg.get('resolution')
        if obs_image_resolution is not None:
            resolution = tuple(obs_image_resolution)
        elif cfg_res is not None:
            resolution = tuple(cfg_res)
        else:
            resolution = (640, 480)
        sensor_resolution = _tuple_or_none(
            cameras_cfg.get('sensor_resolution', cameras_cfg.get('input_resolution'))
        )
        crop_rect = _tuple_or_none(cameras_cfg.get('crop_rect'))
        auto_center_crop = bool(cameras_cfg.get('auto_center_crop', False))

        capture_fps_int = int(capture_fps)
        use_collection_config = bool(cameras_cfg.get('use_collection_camera_config', True))
        camera_configs = None
        gain_db = cameras_cfg.get('gain_db', None)
        if gain_db is not None and not isinstance(gain_db, (list, tuple)):
            gain_db = float(gain_db)
        if use_collection_config:
            input_res = tuple(cameras_cfg.get(
                'input_res',
                cameras_cfg.get('input_resolution', (1440, 1080)),
            ))
            camera_configs = list(create_camera_configs_from_serials(
                serials=serials,
                fps=capture_fps_int,
                output_res=resolution,
                input_res=input_res,
                exposure_time_us=float(cameras_cfg.get('exposure_time_us', 15000.0)),
                gain_auto=cameras_cfg.get('gain_auto', 'continuous'),
                gain_db=gain_db,
                prefer_sensor_roi=bool(cameras_cfg.get('prefer_sensor_roi', False)),
            ))
            if 'rotate_180' in cameras_cfg:
                for cfg in camera_configs:
                    cfg.rotate_180 = bool(cameras_cfg['rotate_180'])
            if 'balance_white_auto' in cameras_cfg:
                for cfg in camera_configs:
                    cfg.balance_white_auto = cameras_cfg['balance_white_auto']
            if 'black_level_enable' in cameras_cfg:
                for cfg in camera_configs:
                    cfg.black_level_enable = bool(cameras_cfg['black_level_enable'])
            if 'black_level' in cameras_cfg:
                for cfg in camera_configs:
                    cfg.black_level = int(cameras_cfg['black_level'])
            if 'brightness' in cameras_cfg:
                for cfg in camera_configs:
                    cfg.brightness = int(cameras_cfg['brightness'])

        video_bit_rate = int(cameras_cfg.get('video_bit_rate', 3000 * 1000))
        video_recorder = [
            VideoRecorder.create_hevc_nvenc(
                fps=capture_fps_int,
                input_pix_fmt='rgb24',
                bit_rate=video_bit_rate,
            )
            for _ in serials
        ]

        camera = MultiMvsCamera(
            mvs_serials=serials,
            shm_manager=shm_manager,
            resolution=resolution,
            camera_configs=camera_configs,
            sensor_resolution=sensor_resolution,
            auto_center_crop=auto_center_crop,
            crop_rect=crop_rect,
            capture_fps=capture_fps_int,
            put_downsample=False,
            receive_latency=obs_latency,
            exposure_time_us=float(cameras_cfg.get('exposure_time_us', 20000.0)),
            gain_auto=cameras_cfg.get('gain_auto', 'continuous'),
            gain_db=gain_db,
            balance_white_auto=cameras_cfg.get('balance_white_auto', 'continuous'),
            rotate_180=bool(cameras_cfg.get('rotate_180', True)),
            auto_crop_black_border=bool(cameras_cfg.get('auto_crop_black_border', True)),
            auto_crop_probe_frames=int(cameras_cfg.get('auto_crop_probe_frames', 4)),
            auto_crop_min_probe_frames=int(cameras_cfg.get('auto_crop_min_probe_frames', 2)),
            auto_crop_warmup_frames=int(cameras_cfg.get('auto_crop_warmup_frames', 2)),
            auto_crop_black_threshold=int(cameras_cfg.get('auto_crop_black_threshold', 12)),
            auto_crop_margin_px=int(cameras_cfg.get('auto_crop_margin_px', 0)),
            auto_crop_min_size_ratio=float(cameras_cfg.get('auto_crop_min_size_ratio', 0.80)),
            auto_crop_reopen_delay_ms=int(cameras_cfg.get('auto_crop_reopen_delay_ms', 200)),
            auto_crop_detect_max_dim=int(cameras_cfg.get('auto_crop_detect_max_dim', 720)),
            launch_timeout=float(cameras_cfg.get('launch_timeout', 10.0)),
            video_recorder=video_recorder,
            verbose=False,
        )
        return camera, capture_fps, obs_latency

    if backend in {'l515_subprocess', 'realsense_subprocess', 'ego_l515'}:
        from umi.real_world.l515_ego_camera import L515EgoCamera

        serials = list(cameras_cfg.get('serials') or [])
        if not serials:
            serial = cameras_cfg.get(
                'serial',
                cameras_cfg.get('ego_serial', cameras_cfg.get('l515_serial')),
            )
            if serial not in (None, ''):
                serials = [str(serial)]
        if len(serials) != 1:
            raise ValueError(
                "cameras.backend='l515_subprocess' requires exactly one "
                "L515 serial in cameras.serials"
            )

        input_res = cameras_cfg.get(
            'input_res',
            cameras_cfg.get('input_resolution', (960, 540)),
        )
        input_res = tuple(int(v) for v in input_res)
        if len(input_res) != 2 or min(input_res) <= 0:
            raise ValueError(f'invalid L515 input_res: {input_res!r}')

        cfg_res = cameras_cfg.get('resolution')
        if obs_image_resolution is not None:
            output_res = tuple(int(v) for v in obs_image_resolution)
        elif cfg_res is not None:
            output_res = tuple(int(v) for v in cfg_res)
        else:
            output_res = (224, 224)
        worker_python = cameras_cfg.get(
            'worker_python',
            '/home/zjc/miniconda3/envs/l515_mvs310/bin/python',
        )
        worker_script = cameras_cfg.get(
            'worker_script',
            '/home/zjc/Desktop/human2dex/'
            'scripts_real/l515_ego_worker.py',
        )
        camera = L515EgoCamera(
            serial=str(serials[0]),
            worker_python=str(worker_python),
            worker_script=str(worker_script),
            input_resolution=input_res,
            output_resolution=output_res,
            capture_fps=int(capture_fps),
            warmup_frames=int(cameras_cfg.get('warmup_frames', 30)),
            timeout_ms=int(cameras_cfg.get('timeout_ms', 5000)),
            shm_slots=int(cameras_cfg.get('shm_slots', 4)),
            max_history=int(cameras_cfg.get(
                'max_history',
                max(64, round(float(capture_fps) * 5.0)),
            )),
            rotate_180=bool(cameras_cfg.get('rotate_180', False)),
        )
        return camera, capture_fps, obs_latency

    if backend in {'realsense', 'l515', 'realsense_l515'}:
        # The training-side ego stream is the L515 color stream
        # (960x540 RGB).  The existing MultiRealsense implementation already
        # provides the lifecycle/get()/recording interface expected by
        # BimanualUmiEnv, so no policy, action, or observation code needs to
        # know which physical camera is selected.
        from diffusion_policy.common.cv2_util import get_image_transform
        from diffusion_policy.real_world.multi_realsense import MultiRealsense

        serials = list(cameras_cfg.get('serials') or [])
        if not serials:
            serial = cameras_cfg.get(
                'serial',
                cameras_cfg.get('ego_serial', cameras_cfg.get('l515_serial')),
            )
            if serial not in (None, ''):
                serials = [str(serial)]
        if not serials:
            raise ValueError(
                "cameras.backend='realsense' requires cameras.serials (or "
                "serial/ego_serial/l515_serial)"
            )

        input_res = cameras_cfg.get(
            'input_res',
            cameras_cfg.get('input_resolution', (960, 540)),
        )
        input_res = tuple(int(v) for v in input_res)
        if len(input_res) != 2 or min(input_res) <= 0:
            raise ValueError(f'invalid RealSense input_res: {input_res!r}')

        cfg_res = cameras_cfg.get('resolution')
        if obs_image_resolution is not None:
            output_res = tuple(int(v) for v in obs_image_resolution)
        elif cfg_res is not None:
            output_res = tuple(int(v) for v in cfg_res)
        else:
            output_res = (224, 224)
        if len(output_res) != 2 or min(output_res) <= 0:
            raise ValueError(f'invalid RealSense resolution: {output_res!r}')

        capture_fps_int = int(capture_fps)
        if capture_fps_int <= 0:
            raise ValueError(f'capture_fps must be positive, got {capture_fps!r}')
        rotate_180 = bool(cameras_cfg.get('rotate_180', False))
        image_tf = get_image_transform(
            input_res=input_res,
            output_res=output_res,
            # pyrealsense color stream is BGR8 in SingleRealsense; policy
            # observations everywhere else are RGB.
            bgr_to_rgb=True,
        )

        def transform(data):
            image = data['color']
            if rotate_180:
                image = image[::-1, ::-1]
            data['color'] = np.ascontiguousarray(image_tf(image))
            return data

        camera = MultiRealsense(
            serial_numbers=[str(s) for s in serials],
            shm_manager=shm_manager,
            resolution=input_res,
            capture_fps=capture_fps_int,
            put_fps=capture_fps_int,
            put_downsample=False,
            record_fps=capture_fps_int,
            enable_color=True,
            enable_depth=False,
            transform=transform,
            # Visualizer expects injected cameras to provide RGB frames.
            vis_transform=transform,
            recording_transform=None,
            verbose=False,
        )
        return camera, capture_fps, obs_latency

    raise NotImplementedError(f"unknown cameras.backend: {backend!r}")
