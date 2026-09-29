from typing import Callable, Dict, List, Optional, Union
import copy
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from umi.real_world.mvs_camera import MvsCamera, MvsCameraOpenConfig
from umi.real_world.video_recorder import VideoRecorder


class MultiMvsCamera:
    """Drop-in for MultiUvcCamera but backed by Hikrobot MVS USB3 cameras.

    The order of `mvs_serials` is the camera index used by BimanualUmiEnv:
    serials[0] -> camera0_rgb, serials[1] -> camera1_rgb, etc. No reorder
    parameter -- caller is expected to put serials in the desired order.
    """

    def __init__(
        self,
        mvs_serials: List[str],
        shm_manager: Optional[SharedMemoryManager] = None,
        resolution=(640, 480),
        camera_configs: Optional[List[MvsCameraOpenConfig]] = None,
        sensor_resolution=None,
        auto_center_crop: bool = False,
        crop_rect=None,
        capture_fps: int = 60,
        put_fps: Optional[int] = None,
        put_downsample: bool = True,
        get_max_k: int = 30,
        receive_latency: float = 0.0,
        exposure_time_us: float = 20000.0,
        gain_auto: str = 'continuous',
        gain_db: Optional[Union[float, List[float]]] = None,
        balance_white_auto: str = 'continuous',
        rotate_180: bool = True,
        auto_crop_black_border: bool = True,
        auto_crop_probe_frames: int = 4,
        auto_crop_min_probe_frames: int = 2,
        auto_crop_warmup_frames: int = 2,
        auto_crop_black_threshold: int = 12,
        auto_crop_margin_px: int = 0,
        auto_crop_min_size_ratio: float = 0.80,
        auto_crop_reopen_delay_ms: int = 200,
        auto_crop_detect_max_dim: int = 720,
        backend_read_timeout_ms: int = 500,
        launch_timeout: float = 10.0,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]] = None,
        verbose: bool = False,
    ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        n = len(mvs_serials)
        if camera_configs is not None:
            assert len(camera_configs) == n
            for serial, cfg in zip(mvs_serials, camera_configs):
                cfg.validate()
                if cfg.serial != serial:
                    raise ValueError(f"MVS config serial {cfg.serial!r} != requested serial {serial!r}")
            resolution = [(cfg.width, cfg.height) for cfg in camera_configs]
            capture_fps = [cfg.fps for cfg in camera_configs]
            exposure_time_us = [cfg.exposure_time_us for cfg in camera_configs]
            gain_db = [cfg.gain_db for cfg in camera_configs]

        resolution = repeat_to_list(resolution, n, tuple)
        sensor_resolution = repeat_to_list(sensor_resolution, n, tuple)
        crop_rect = repeat_to_list(crop_rect, n, tuple)
        capture_fps = repeat_to_list(capture_fps, n, (int, float))
        exposure_time_us = repeat_to_list(exposure_time_us, n, (int, float))
        gain_db = repeat_to_list(gain_db, n, (int, float))
        transform = repeat_to_list(transform, n, Callable)
        vis_transform = repeat_to_list(vis_transform, n, Callable)
        recording_transform = repeat_to_list(recording_transform, n, Callable)
        video_recorder = repeat_to_list(video_recorder, n, VideoRecorder)

        cameras: Dict[str, MvsCamera] = {}
        for i, serial in enumerate(mvs_serials):
            cameras[serial] = MvsCamera(
                shm_manager=shm_manager,
                mvs_serial=serial,
                resolution=resolution[i],
                open_config=None if camera_configs is None else camera_configs[i],
                sensor_resolution=sensor_resolution[i],
                auto_center_crop=auto_center_crop,
                crop_rect=crop_rect[i],
                capture_fps=capture_fps[i],
                put_fps=put_fps,
                put_downsample=put_downsample,
                get_max_k=get_max_k,
                receive_latency=receive_latency,
                exposure_time_us=exposure_time_us[i],
                gain_auto=gain_auto,
                gain_db=gain_db[i],
                balance_white_auto=balance_white_auto,
                rotate_180=rotate_180,
                auto_crop_black_border=auto_crop_black_border,
                auto_crop_probe_frames=auto_crop_probe_frames,
                auto_crop_min_probe_frames=auto_crop_min_probe_frames,
                auto_crop_warmup_frames=auto_crop_warmup_frames,
                auto_crop_black_threshold=auto_crop_black_threshold,
                auto_crop_margin_px=auto_crop_margin_px,
                auto_crop_min_size_ratio=auto_crop_min_size_ratio,
                auto_crop_reopen_delay_ms=auto_crop_reopen_delay_ms,
                auto_crop_detect_max_dim=auto_crop_detect_max_dim,
                backend_read_timeout_ms=backend_read_timeout_ms,
                launch_timeout=launch_timeout,
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                verbose=verbose,
            )

        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self) -> int:
        return len(self.cameras)

    @property
    def is_ready(self) -> bool:
        for camera in self.cameras.values():
            if not camera.is_ready:
                return False
        return True

    def start(self, wait: bool = True, put_start_time: Optional[float] = None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        if wait:
            self.start_wait()

    def stop(self, wait: bool = True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.end_wait()

    def get(self, k: Optional[int] = None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        if out is None:
            out = {}
        for i, camera in enumerate(self.cameras.values()):
            this_out = out.get(i, None)
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = []
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = {}
                for key, v in out.items():
                    this_out[key] = v[i:i + 1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = {}
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = [
                str(video_dir.joinpath(f'{i}.mp4').absolute())
                for i in range(self.n_cameras)
            ]
        assert len(video_path) == self.n_cameras
        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)

    def stop_recording(self):
        for camera in self.cameras.values():
            camera.stop_recording()

    def restart_put(self, start_time: float):
        for camera in self.cameras.values():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        return [None] * n
    if isinstance(x, cls):
        return [copy.deepcopy(x) for _ in range(n)]
    assert len(x) == n
    return x
