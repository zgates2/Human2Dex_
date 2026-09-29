import enum
import multiprocessing as mp
import time
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from umi.common.timesync import DeviceHostClockCalibrator
from umi.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from umi.real_world.video_recorder import VideoRecorder
from umi.shared_memory.shared_memory_queue import Empty, SharedMemoryQueue
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

# 延迟到子进程的 run() 里再 import，避免在 mac 上未编译 nanobind 扩展时
# 主进程 import umi.real_world.mvs_camera 直接炸掉。这样 BimanualUmiEnv 在
# 不使用 MVS 后端的情况下仍能正常 import。
_MVS_CPP_AVAILABLE: Optional[bool] = None
_MVS_CPP_IMPORT_ERROR: Optional[Exception] = None


def compute_center_crop_rect(
    input_res: Sequence[int],
    output_res: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Return x/y/w/h crop preserving output aspect before resize."""
    input_width, input_height = map(int, input_res)
    output_width, output_height = map(int, output_res)
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"invalid input_res: {input_res!r}")
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"invalid output_res: {output_res!r}")

    input_aspect = input_width / input_height
    output_aspect = output_width / output_height
    if abs(input_aspect - output_aspect) < 1e-6:
        return 0, 0, input_width, input_height
    if input_aspect > output_aspect:
        crop_height = input_height
        crop_width = int(round(crop_height * output_aspect))
        crop_x = max(0, (input_width - crop_width) // 2)
        crop_y = 0
    else:
        crop_width = input_width
        crop_height = int(round(crop_width / output_aspect))
        crop_x = 0
        crop_y = max(0, (input_height - crop_height) // 2)
    return crop_x, crop_y, min(crop_width, input_width), min(crop_height, input_height)


def _identity_transform(x):
    return x


def _is_identity_transform(fn) -> bool:
    return fn is _identity_transform or getattr(fn, "__name__", "") == "_identity_transform"


@dataclass
class MvsCameraOpenConfig:
    """Collection-side MVS open config, reduced to fields used by this runtime."""

    name: str
    serial: str
    fps: int = 30
    put_desired_frequency: Optional[int] = None
    receive_latency: float = 0.0
    width: int = 480
    height: int = 480
    transformed_width: int = 480
    transformed_height: int = 480
    sensor_width: Optional[int] = None
    sensor_height: Optional[int] = None
    offset_x: int = 0
    offset_y: int = 0
    crop_x: int = 0
    crop_y: int = 0
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None
    image_node_num: int = 1
    frame_pool_size: int = 4
    rotate_180: bool = True
    acquisition_frame_rate_enable: bool = True
    exposure_auto: str = "off"
    exposure_time_us: float = 20000.0
    gain_auto: str = "continuous"
    gain_db: Optional[float] = None
    balance_white_auto: str = "continuous"
    black_level_enable: bool = True
    black_level: int = 240
    brightness: Optional[int] = None

    def validate(self) -> None:
        assert isinstance(self.name, str) and self.name, f"invalid camera name: {self.name!r}"
        assert isinstance(self.serial, str) and self.serial, f"invalid serial: {self.serial!r}"
        assert self.width > 0 and self.height > 0
        assert self.transformed_width > 0 and self.transformed_height > 0
        assert self.crop_x >= 0 and self.crop_y >= 0
        if self.crop_width is not None:
            assert self.crop_width > 0
        if self.crop_height is not None:
            assert self.crop_height > 0
        assert self.image_node_num > 0
        assert self.frame_pool_size > 0
        if self.put_desired_frequency is None:
            self.put_desired_frequency = self.fps


def create_camera_configs_from_serials(
    serials: Sequence[str],
    fps: int = 30,
    output_res: Tuple[int, int] = (320, 240),
    input_res: Tuple[int, int] = (1440, 1080),
    crop_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    exposure_time_us: float = 15000.0,
    gain_auto: str = "continuous",
    gain_db: Optional[Union[float, Sequence[float]]] = None,
    prefer_sensor_roi: bool = False,
) -> Sequence[MvsCameraOpenConfig]:
    """Match the collection-side MVS config generator.

    The C++ backend returns already-processed RGB frames. For the default
    identity crop_func this means: request full sensor readout, center crop to
    match output aspect ratio, then resize in C++ before Python sees the image.
    """
    configs = []
    crop = crop_func or _identity_transform
    use_sensor_roi = (
        prefer_sensor_roi
        and _is_identity_transform(crop)
        and tuple(output_res) == tuple(input_res)
    )
    sensor_width = output_res[0] if use_sensor_roi else input_res[0]
    sensor_height = output_res[1] if use_sensor_roi else input_res[1]
    crop_x = 0
    crop_y = 0
    crop_width = None
    crop_height = None
    if not use_sensor_roi and _is_identity_transform(crop):
        crop_x, crop_y, crop_width, crop_height = compute_center_crop_rect(
            input_res=input_res,
            output_res=output_res,
        )

    for idx, serial in enumerate(serials):
        gain_db_i = gain_db[idx] if isinstance(gain_db, (list, tuple)) else gain_db
        config = MvsCameraOpenConfig(
            receive_latency=0.125,
            serial=serial,
            name=f"camera{idx}_rgb",
            fps=fps,
            put_desired_frequency=fps,
            width=output_res[0],
            height=output_res[1],
            transformed_width=output_res[0],
            transformed_height=output_res[1],
            exposure_time_us=exposure_time_us,
            gain_auto=gain_auto,
            gain_db=None if gain_db_i is None else float(gain_db_i),
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            offset_x=0,
            offset_y=0,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
        )
        config.validate()
        configs.append(config)
    return configs


def open_kwargs_from_collection_config(config: MvsCameraOpenConfig) -> Dict:
    return {
        'serial': config.serial,
        'sensor_width': config.sensor_width,
        'sensor_height': config.sensor_height,
        'offset_x': config.offset_x,
        'offset_y': config.offset_y,
        'crop_x': config.crop_x,
        'crop_y': config.crop_y,
        'crop_width': config.crop_width,
        'crop_height': config.crop_height,
        'output_width': config.width,
        'output_height': config.height,
        'image_node_num': config.image_node_num,
        'frame_pool_size': config.frame_pool_size,
        'rotate_180': config.rotate_180,
        'acquisition_frame_rate_enable': config.acquisition_frame_rate_enable,
        'acquisition_frame_rate_fps': float(config.fps),
        'exposure_auto': config.exposure_auto,
        'exposure_time_us': config.exposure_time_us,
        'gain_auto': config.gain_auto,
        'gain_db': config.gain_db,
        'balance_white_auto': config.balance_white_auto,
        'black_level_enable': config.black_level_enable,
        'black_level': config.black_level,
        'brightness': config.brightness,
    }


def _read_mvs_frame_array(device, timeout_ms: int) -> np.ndarray:
    frame = device.read_frame(timeout_ms=timeout_ms)
    try:
        return frame.copy_array()
    finally:
        frame.release()


def _capture_auto_crop_probe_frames(
        device,
        *,
        wanted: int,
        warmup: int,
        timeout_ms: int,
        no_data_error,
        deadline_s: float) -> list:
    frames = []
    discarded = 0
    while discarded < warmup and time.monotonic() < deadline_s:
        try:
            _ = _read_mvs_frame_array(device, timeout_ms=timeout_ms)
        except no_data_error:
            continue
        discarded += 1

    while len(frames) < wanted and time.monotonic() < deadline_s:
        try:
            frames.append(_read_mvs_frame_array(device, timeout_ms=timeout_ms))
        except no_data_error:
            continue
    return frames


def _probe_gray(frames: list, max_dim: int):
    if not frames:
        return None
    src_h, src_w = frames[0].shape[:2]
    target = max(32, int(max_dim))
    longest = max(src_h, src_w)
    if longest > target:
        scale = longest / float(target)
        new_w = max(8, int(round(src_w / scale)))
        new_h = max(8, int(round(src_h / scale)))
    else:
        scale = 1.0
        new_w, new_h = src_w, src_h

    try:
        import cv2
    except Exception:
        return None

    grays = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] < 3:
            continue
        chan_max = np.max(frame[:, :, :3], axis=2).astype(np.uint8)
        if scale != 1.0:
            chan_max = cv2.resize(chan_max, (new_w, new_h), interpolation=cv2.INTER_AREA)
        grays.append(chan_max)
    if not grays:
        return None
    if len(grays) == 1:
        return grays[0], scale
    return np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8), scale


def _fit_crop_rect_to_bounds(
        rect: Tuple[int, int, int, int],
        *,
        bounds_width: int,
        bounds_height: int) -> Tuple[int, int, int, int]:
    x, y, crop_w, crop_h = rect
    center_x = float(x) + (float(crop_w) / 2.0)
    center_y = float(y) + (float(crop_h) / 2.0)
    scale = min(
        1.0,
        float(bounds_width) / max(1.0, float(crop_w)),
        float(bounds_height) / max(1.0, float(crop_h)),
    )
    crop_w = max(1, int(round(float(crop_w) * scale)))
    crop_h = max(1, int(round(float(crop_h) * scale)))
    x = int(round(center_x - (float(crop_w) / 2.0)))
    y = int(round(center_y - (float(crop_h) / 2.0)))
    x = max(0, min(x, int(bounds_width) - crop_w))
    y = max(0, min(y, int(bounds_height) - crop_h))
    return x, y, crop_w, crop_h


def _detect_auto_crop_rect(
        frames: list,
        *,
        sensor_width: int,
        sensor_height: int,
        output_width: int,
        output_height: int,
        min_probe_frames: int,
        black_threshold: int,
        margin_px: int,
        min_size_ratio: float,
        detect_max_dim: int):
    if len(frames) < max(1, int(min_probe_frames)):
        return None
    probe = _probe_gray(frames, max_dim=int(detect_max_dim))
    if probe is None:
        return None
    gray_small, scale = probe

    try:
        import cv2
    except Exception:
        return None

    valid = (gray_small > int(black_threshold)).astype(np.uint8)
    valid_ratio = float(np.mean(valid))
    if valid_ratio < 0.05:
        return None

    kernel_size = max(3, min(valid.shape[:2]) // 80)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel, iterations=1)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(largest)) <= 0:
        return None

    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(largest)
    margin = int(margin_px)
    bbox_x = int(round((float(bbox_x) * scale) - float(margin)))
    bbox_y = int(round((float(bbox_y) * scale) - float(margin)))
    bbox_w = int(round(float(bbox_w) * scale + (2.0 * float(margin))))
    bbox_h = int(round(float(bbox_h) * scale + (2.0 * float(margin))))
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    output_aspect = float(output_width) / float(output_height)
    bbox_cx = float(bbox_x) + (float(bbox_w) / 2.0)
    bbox_cy = float(bbox_y) + (float(bbox_h) / 2.0)
    if output_aspect >= 1.0:
        crop_w = max(float(bbox_w), float(bbox_h) * output_aspect)
        crop_h = crop_w / output_aspect
    else:
        crop_h = max(float(bbox_h), float(bbox_w) / output_aspect)
        crop_w = crop_h * output_aspect

    rect = (
        int(round(bbox_cx - (crop_w / 2.0))),
        int(round(bbox_cy - (crop_h / 2.0))),
        max(1, int(round(crop_w))),
        max(1, int(round(crop_h))),
    )
    sensor_rect = _fit_crop_rect_to_bounds(
        rect,
        bounds_width=int(sensor_width),
        bounds_height=int(sensor_height),
    )

    _, _, crop_w, crop_h = sensor_rect
    min_ratio = max(0.05, min(1.0, float(min_size_ratio)))
    min_sensor_side = min(int(sensor_width), int(sensor_height))
    if crop_w < min_sensor_side * min_ratio or crop_h < min_sensor_side * min_ratio:
        return None
    return sensor_rect


def _auto_crop_open_kwargs(
        *,
        serial: str,
        open_kwargs: Dict,
        open_mvs_cpp_device,
        no_data_error,
        backend_read_timeout_ms: int,
        probe_frames: int,
        min_probe_frames: int,
        warmup_frames: int,
        black_threshold: int,
        margin_px: int,
        min_size_ratio: float,
        reopen_delay_ms: int,
        detect_max_dim: int):
    if 'output_width' not in open_kwargs or 'output_height' not in open_kwargs:
        return open_kwargs

    sensor_width = int(
        open_kwargs.get('sensor_width')
        or open_kwargs.get('crop_width')
        or open_kwargs['output_width']
    )
    sensor_height = int(
        open_kwargs.get('sensor_height')
        or open_kwargs.get('crop_height')
        or open_kwargs['output_height']
    )
    probe_kwargs = dict(open_kwargs)
    probe_kwargs.update({
        'crop_x': 0,
        'crop_y': 0,
        'crop_width': sensor_width,
        'crop_height': sensor_height,
        'output_width': sensor_width,
        'output_height': sensor_height,
        'rotate_180': False,
    })

    probe_device = None
    try:
        print(f"[MvsCamera {serial}] auto-crop probing fisheye black border", flush=True)
        probe_device = open_mvs_cpp_device(**probe_kwargs)
        timeout_ms = max(50, int(backend_read_timeout_ms))
        wanted = max(1, int(probe_frames))
        warmup = max(0, int(warmup_frames))
        deadline = time.monotonic() + max(
            2.0, (wanted + warmup) * timeout_ms / 1000.0 + 1.0
        )
        frames = _capture_auto_crop_probe_frames(
            probe_device,
            wanted=wanted,
            warmup=warmup,
            timeout_ms=timeout_ms,
            no_data_error=no_data_error,
            deadline_s=deadline,
        )
        sensor_rect = _detect_auto_crop_rect(
            frames,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            output_width=int(open_kwargs['output_width']),
            output_height=int(open_kwargs['output_height']),
            min_probe_frames=int(min_probe_frames),
            black_threshold=int(black_threshold),
            margin_px=int(margin_px),
            min_size_ratio=float(min_size_ratio),
            detect_max_dim=int(detect_max_dim),
        )
        if sensor_rect is None:
            print(
                f"[MvsCamera {serial}] auto-crop could not locate fisheye circle; "
                "using configured crop",
                flush=True,
            )
            return open_kwargs
        crop_x, crop_y, crop_width, crop_height = sensor_rect
        new_kwargs = dict(open_kwargs)
        new_kwargs.update({
            'crop_x': crop_x,
            'crop_y': crop_y,
            'crop_width': crop_width,
            'crop_height': crop_height,
        })
        print(
            f"[MvsCamera {serial}] auto-crop sensor_crop="
            f"({crop_x},{crop_y},{crop_width},{crop_height}) -> "
            f"{open_kwargs['output_width']}x{open_kwargs['output_height']}",
            flush=True,
        )
        return new_kwargs
    except Exception as exc:
        print(
            f"[MvsCamera {serial}] auto-crop probe failed "
            f"({type(exc).__name__}: {exc}); using configured crop",
            flush=True,
        )
        return open_kwargs
    finally:
        if probe_device is not None:
            try:
                probe_device.close()
            except Exception:
                pass
        delay_s = max(0, int(reopen_delay_ms)) / 1000.0
        if delay_s > 0:
            time.sleep(delay_s)


def is_mvs_cpp_available() -> bool:
    global _MVS_CPP_AVAILABLE, _MVS_CPP_IMPORT_ERROR
    if _MVS_CPP_AVAILABLE is not None:
        return _MVS_CPP_AVAILABLE
    try:
        from umi.real_world._mvs_cpp import (  # noqa: F401
            MVSCPPDevice, open_mvs_cpp_device, list_mvs_device_serials,
        )
        _MVS_CPP_AVAILABLE = True
    except Exception as e:
        _MVS_CPP_IMPORT_ERROR = e
        _MVS_CPP_AVAILABLE = False
    return _MVS_CPP_AVAILABLE


def list_mvs_serials():
    if not is_mvs_cpp_available():
        raise RuntimeError(
            "MVS C++ backend not built. See umi/real_world/_mvs_cpp/README.md."
        ) from _MVS_CPP_IMPORT_ERROR
    from umi.real_world._mvs_cpp import list_mvs_device_serials
    return list(list_mvs_device_serials())


class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2


class MvsCamera(mp.Process):
    """Hikrobot MVS USB3 camera, SHM-backed, drop-in for UvcCamera.

    Ring buffer schema matches UvcCamera so MultiUvcCamera/MultiCameraVisualizer/
    BimanualUmiEnv.get_obs alignment logic works unchanged:
        color                      : (H, W, 3) uint8
        camera_capture_timestamp   : wall-clock seconds at sensor exposure
                                     (calibrated via DeviceHostClockCalibrator
                                     once enough samples land)
        camera_receive_timestamp   : wall-clock seconds at driver receive
        timestamp                  : capture - receive_latency
        step_idx                   : int, monotonically increasing
    """

    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        mvs_serial: str,
        resolution=(640, 480),
        open_config: Optional[MvsCameraOpenConfig] = None,
        sensor_resolution: Optional[Sequence[int]] = None,
        auto_center_crop: bool = False,
        crop_rect: Optional[Sequence[int]] = None,
        capture_fps: int = 60,
        put_fps: Optional[int] = None,
        put_downsample: bool = True,
        get_max_k: int = 30,
        receive_latency: float = 0.0,
        exposure_time_us: float = 20000.0,
        gain_auto: str = 'continuous',
        gain_db: Optional[float] = None,
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
        transform: Optional[Callable[[Dict], Dict]] = None,
        vis_transform: Optional[Callable[[Dict], Dict]] = None,
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        verbose: bool = False,
    ):
        super().__init__(name=f"MvsCamera({mvs_serial})")

        if put_fps is None:
            put_fps = capture_fps

        if open_config is not None:
            open_config.validate()
            if open_config.serial != mvs_serial:
                raise ValueError(
                    f"open_config serial {open_config.serial!r} != camera serial {mvs_serial!r}"
                )
            resolution = (int(open_config.width), int(open_config.height))

        resolution = tuple(map(int, resolution))
        if sensor_resolution is not None:
            sensor_resolution = tuple(map(int, sensor_resolution))
            if len(sensor_resolution) != 2:
                raise ValueError(f"sensor_resolution must be [width, height], got {sensor_resolution!r}")
        if crop_rect is not None:
            crop_rect = tuple(map(int, crop_rect))
            if len(crop_rect) != 4:
                raise ValueError(f"crop_rect must be [x, y, width, height], got {crop_rect!r}")
        if auto_center_crop and crop_rect is None:
            if sensor_resolution is None:
                raise ValueError("auto_center_crop requires sensor_resolution")
            crop_rect = compute_center_crop_rect(sensor_resolution, resolution)
        shape = resolution[::-1]
        examples = {
            'color': np.empty(shape=shape + (3,), dtype=np.uint8),
            'camera_capture_timestamp': 0.0,
            'camera_receive_timestamp': 0.0,
            'camera_frame_id': 0,
            'camera_device_timestamp_ticks': 0,
            'camera_host_timestamp_ms': 0,
            'camera_backend_total_ns': 0,
            'timestamp': 0.0,
            'step_idx': 0,
        }

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps,
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None else transform(dict(examples)),
            get_max_k=get_max_k * 10,
            get_time_budget=0.2,
            put_desired_frequency=put_fps,
        )

        cmd_example = {
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': 0.0,
            'video_path': np.array('a' * self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
        }
        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=cmd_example,
            buffer_size=128,
        )

        if video_recorder is None:
            video_recorder = VideoRecorder.create_hevc_nvenc(
                fps=capture_fps,
                input_pix_fmt='rgb24',
                bit_rate=3000 * 1000,
            )
        assert video_recorder.fps == capture_fps

        self.shm_manager = shm_manager
        self.mvs_serial = mvs_serial
        self.resolution = resolution
        self.open_config = open_config
        self.sensor_resolution = sensor_resolution
        self.auto_center_crop = auto_center_crop
        self.crop_rect = crop_rect
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.exposure_time_us = exposure_time_us
        self.gain_auto = gain_auto
        self.gain_db = gain_db
        self.balance_white_auto = balance_white_auto
        self.rotate_180 = rotate_180
        self.auto_crop_black_border = auto_crop_black_border
        self.auto_crop_probe_frames = auto_crop_probe_frames
        self.auto_crop_min_probe_frames = auto_crop_min_probe_frames
        self.auto_crop_warmup_frames = auto_crop_warmup_frames
        self.auto_crop_black_threshold = auto_crop_black_threshold
        self.auto_crop_margin_px = auto_crop_margin_px
        self.auto_crop_min_size_ratio = auto_crop_min_size_ratio
        self.auto_crop_reopen_delay_ms = auto_crop_reopen_delay_ms
        self.auto_crop_detect_max_dim = auto_crop_detect_max_dim
        self.backend_read_timeout_ms = backend_read_timeout_ms
        self.launch_timeout = launch_timeout
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose

        self.put_start_time: Optional[float] = None
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue

    # ========= context manager ==========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= lifecycle ==========
    def start(self, wait: bool = True, put_start_time: Optional[float] = None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape + (3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=self.shm_manager,
            data_example=data_example,
        )
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait: bool = True):
        self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self, timeout: Optional[float] = None):
        if timeout is None:
            timeout = self.launch_timeout
        deadline = time.monotonic() + timeout

        while not self.ready_event.wait(timeout=0.05):
            if not self.is_alive():
                self.video_recorder.stop()
                raise RuntimeError(
                    f"MvsCamera({self.mvs_serial}) exited before ready "
                    f"(exitcode={self.exitcode}). Check MVS backend, serial, "
                    "camera connection, and SDK installation."
                )
            if time.monotonic() >= deadline:
                self.stop(wait=True)
                raise TimeoutError(
                    f"MvsCamera({self.mvs_serial}) did not become ready "
                    f"within {timeout:.1f}s"
                )

        if not self.is_alive():
            self.video_recorder.stop()
            raise RuntimeError(
                f"MvsCamera({self.mvs_serial}) exited after setting ready "
                f"(exitcode={self.exitcode})"
            )

        remaining = max(0.0, deadline - time.monotonic())
        if not self.video_recorder.ready_event.wait(timeout=remaining):
            self.stop(wait=True)
            raise TimeoutError(
                f"MvsCamera({self.mvs_serial}) video recorder did not become "
                f"ready within {timeout:.1f}s"
            )

    def end_wait(self, timeout: Optional[float] = None):
        if timeout is None:
            timeout = self.launch_timeout
        self.join(timeout=timeout)
        if self.is_alive():
            self.terminate()
            self.join(timeout=2.0)
        self.video_recorder.end_wait()

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set() and self.is_alive()

    def get(self, k: Optional[int] = None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)

    # ========= recording commands ==========
    def start_recording(self, video_path: str, start_time: float = -1):
        path_len = len(video_path.encode('utf-8'))
        assert path_len < self.MAX_PATH_LENGTH, "video_path too long"
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time,
        })

    def stop_recording(self):
        self.command_queue.put({'cmd': Command.STOP_RECORDING.value})

    def restart_put(self, start_time: float):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time,
        })

    # ========= subprocess main ==========
    def run(self):
        try:
            from umi.real_world._mvs_cpp import MVSCPPNoDataError, open_mvs_cpp_device
        except Exception as e:
            raise RuntimeError(
                f"MvsCamera({self.mvs_serial}): MVS C++ backend not available. "
                "Build it on the deployment host: see umi/real_world/_mvs_cpp/README.md"
            ) from e

        if self.open_config is not None:
            open_kwargs = open_kwargs_from_collection_config(self.open_config)
        else:
            open_kwargs = {
                'serial': self.mvs_serial,
                'output_width': self.resolution[0],
                'output_height': self.resolution[1],
                'acquisition_frame_rate_enable': True,
                'acquisition_frame_rate_fps': float(self.capture_fps),
                'exposure_auto': 'off',
                'exposure_time_us': self.exposure_time_us,
                'gain_auto': self.gain_auto,
                'gain_db': self.gain_db,
                'balance_white_auto': self.balance_white_auto,
                'rotate_180': self.rotate_180,
            }
            if self.sensor_resolution is not None:
                open_kwargs['sensor_width'] = self.sensor_resolution[0]
                open_kwargs['sensor_height'] = self.sensor_resolution[1]
            if self.crop_rect is not None:
                crop_x, crop_y, crop_width, crop_height = self.crop_rect
                open_kwargs.update({
                    'crop_x': crop_x,
                    'crop_y': crop_y,
                    'crop_width': crop_width,
                    'crop_height': crop_height,
                })

        if self.auto_crop_black_border:
            open_kwargs = _auto_crop_open_kwargs(
                serial=self.mvs_serial,
                open_kwargs=open_kwargs,
                open_mvs_cpp_device=open_mvs_cpp_device,
                no_data_error=MVSCPPNoDataError,
                backend_read_timeout_ms=self.backend_read_timeout_ms,
                probe_frames=self.auto_crop_probe_frames,
                min_probe_frames=self.auto_crop_min_probe_frames,
                warmup_frames=self.auto_crop_warmup_frames,
                black_threshold=self.auto_crop_black_threshold,
                margin_px=self.auto_crop_margin_px,
                min_size_ratio=self.auto_crop_min_size_ratio,
                reopen_delay_ms=self.auto_crop_reopen_delay_ms,
                detect_max_dim=self.auto_crop_detect_max_dim,
            )

        device = open_mvs_cpp_device(
            **open_kwargs,
        )

        clock_calibrator = DeviceHostClockCalibrator(
            window_size=256, min_samples=30,
        )
        wall_minus_monotonic_s = time.time() - time.monotonic()

        try:
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            while not self.stop_event.is_set():
                try:
                    frame = device.read_frame(timeout_ms=self.backend_read_timeout_ms)
                except Exception as e:
                    if self.verbose:
                        print(f"[MvsCamera {self.mvs_serial}] read_frame error: {e}")
                    time.sleep(0.005)
                    continue

                try:
                    image = frame.copy_array()
                    frame_id = int(frame.frame_id)
                    receive_host_ns = int(frame.timestamp_ns)
                    device_ts = int(frame.device_timestamp_ticks)
                    host_timestamp_ms = int(frame.host_timestamp_ms)
                    backend_total_ns = int(frame.timing_ns.get('total_ns', 0))
                finally:
                    frame.release()

                if device_ts > 0:
                    clock_calibrator.record(device_ts, receive_host_ns)
                if clock_calibrator.is_ready and device_ts > 0:
                    capture_monotonic_ns = clock_calibrator.device_to_host_ns(device_ts)
                else:
                    capture_monotonic_ns = receive_host_ns

                t_recv = time.time()
                t_cap = capture_monotonic_ns * 1e-9 + wall_minus_monotonic_s
                t_cal = t_cap - self.receive_latency

                if (
                    self.video_recorder is not None
                    and self.video_recorder.is_ready()
                ):
                    try:
                        self.video_recorder.write_frame(image, frame_time=t_cal)
                    except Exception as e:
                        if self.verbose:
                            print(f"[MvsCamera {self.mvs_serial}] video write failed: {e}")

                data = {
                    'color': image,
                    'camera_receive_timestamp': t_recv,
                    'camera_capture_timestamp': t_cap,
                    'camera_frame_id': frame_id,
                    'camera_device_timestamp_ticks': device_ts,
                    'camera_host_timestamp_ms': host_timestamp_ms,
                    'camera_backend_total_ns': backend_total_ns,
                }

                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[t_cal],
                        start_time=put_start_time,
                        dt=1 / self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = t_cal
                    self.ring_buffer.put(put_data, wait=False)

                if iter_idx == 0:
                    self.ready_event.set()

                vis_data = data
                if self.vis_transform is self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)

                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                for i in range(n_cmd):
                    cmd = int(commands['cmd'][i])
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = float(commands['put_start_time'][i])
                    elif cmd == Command.START_RECORDING.value:
                        if self.video_recorder is None:
                            continue
                        video_path = str(commands['video_path'][i])
                        start_time = float(commands['recording_start_time'][i])
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        if self.video_recorder is not None:
                            self.video_recorder.stop_recording()

                iter_idx += 1
        finally:
            try:
                device.close()
            except Exception:
                pass
            self.ready_event.set()
            if self.video_recorder is not None and self.video_recorder.is_ready():
                self.video_recorder.stop_recording()
            if self.verbose:
                print(f"[MvsCamera {self.mvs_serial}] subprocess exited")
