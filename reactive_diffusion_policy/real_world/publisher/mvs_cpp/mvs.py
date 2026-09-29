#!/usr/bin/env python3
"""
MVS camera async reader with a C++ backend aligned to the official Hikrobot MVS
SDK C/C++ sample flow.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger


if __package__ in (None, ""):
    _CURRENT_DIR = Path(__file__).resolve().parent
    if str(_CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(_CURRENT_DIR))
    from mvs_cpp.backend import (  # type: ignore
        DEFAULT_MVS_CPP_BACKEND,
        MVSCPPBackendError,
        MVSCPPBackendUnavailableError,
        MVSCameraRuntimeConfig,
        MVSUSBLinkInfo,
        MVSCPPDevice,
        MVSCPPNoDataError,
        get_mvs_usb_link_info,
        list_mvs_device_serials,
        open_mvs_cpp_device,
    )
else:
    from .mvs_cpp.backend import (
        DEFAULT_MVS_CPP_BACKEND,
        MVSCPPBackendError,
        MVSCPPBackendUnavailableError,
        MVSCameraRuntimeConfig,
        MVSUSBLinkInfo,
        MVSCPPDevice,
        MVSCPPNoDataError,
        get_mvs_usb_link_info,
        list_mvs_device_serials,
        open_mvs_cpp_device,
    )


def _identity_transform(img: np.ndarray) -> np.ndarray:
    return img


def _is_identity_transform(func: Optional[Callable[[np.ndarray], np.ndarray]]) -> bool:
    return func is None or func is _identity_transform


def _can_use_cpp_postprocess(config: "MVSCamControllerConfig") -> bool:
    return _is_identity_transform(config.crop_func) and _is_identity_transform(
        config.img_transform_func
    )


def _normalize_backend_name(name: Optional[str]) -> str:
    value = (name or DEFAULT_MVS_CPP_BACKEND or "cpp").strip().lower()
    alias_map = {
        "zero_copy": "cpp",
        "cpp_zero_copy": "cpp",
        "legacy": "cpp",
        "auto": "cpp",
    }
    value = alias_map.get(value, value)
    if value != "cpp":
        raise ValueError(f"Unsupported MVS backend: {name!r}")
    return value


def _compute_center_crop_rect(
    input_res: Tuple[int, int],
    output_res: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    input_width, input_height = map(int, input_res)
    output_width, output_height = map(int, output_res)
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"Invalid input resolution: {input_res!r}")
    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"Invalid output resolution: {output_res!r}")

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

    crop_width = min(crop_width, input_width)
    crop_height = min(crop_height, input_height)
    return crop_x, crop_y, crop_width, crop_height


@dataclass
class MVSCamControllerConfig:
    name: str
    serial: str

    fps: int = 30
    put_desired_frequency: Optional[int] = None
    receive_latency: float = 0.0

    width: int = 480
    height: int = 480
    transformed_width: int = 480
    transformed_height: int = 480

    crop_func: Callable[[np.ndarray], np.ndarray] = field(default=_identity_transform)
    img_transform_func: Optional[Callable[[np.ndarray], np.ndarray]] = field(
        default=_identity_transform
    )

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
    balance_white_auto: str = "continuous"
    black_level_enable: bool = True
    black_level: int = 240
    brightness: int = 40

    def validate(self) -> None:
        assert isinstance(self.name, str) and self.name, f"Invalid camera name: {self.name!r}"
        assert isinstance(self.serial, str) and self.serial, f"Invalid serial: {self.serial!r}"
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


class _BaseCameraDriver:
    performs_rotation = False
    delivers_processed_frames = True

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def read_frame(self, timeout_ms: int = 1000) -> "_DriverFrame":
        raise NotImplementedError


@dataclass
class _DriverFrame:
    image: np.ndarray
    frame_id: int
    timestamp_ns: int
    owner: Optional[object] = None


class _CppCameraDriver(_BaseCameraDriver):
    performs_rotation = True
    _CAPTURE_PATH_REASON = (
        "MVSImageCamera returns owned arrays so frame slots are released immediately"
    )

    def __init__(self, config: MVSCamControllerConfig):
        self.config = config
        self.device: Optional[MVSCPPDevice] = None
        self.delivers_processed_frames = _can_use_cpp_postprocess(config)
        self.capture_path = "copy"
        self.capture_path_reason: Optional[str] = self._CAPTURE_PATH_REASON
        self.runtime_config: Optional[MVSCameraRuntimeConfig] = None

    def start(self) -> None:
        try:
            self.device = open_mvs_cpp_device(
                self.config.serial,
                sensor_width=self.config.sensor_width,
                sensor_height=self.config.sensor_height,
                offset_x=self.config.offset_x,
                offset_y=self.config.offset_y,
                crop_x=self.config.crop_x if self.delivers_processed_frames else 0,
                crop_y=self.config.crop_y if self.delivers_processed_frames else 0,
                crop_width=self.config.crop_width if self.delivers_processed_frames else None,
                crop_height=self.config.crop_height if self.delivers_processed_frames else None,
                output_width=self.config.width if self.delivers_processed_frames else None,
                output_height=self.config.height if self.delivers_processed_frames else None,
                image_node_num=self.config.image_node_num,
                frame_pool_size=self.config.frame_pool_size,
                rotate_180=self.config.rotate_180,
                acquisition_frame_rate_enable=self.config.acquisition_frame_rate_enable,
                acquisition_frame_rate_fps=float(self.config.fps),
                exposure_auto=self.config.exposure_auto,
                exposure_time_us=self.config.exposure_time_us,
                gain_auto=self.config.gain_auto,
                balance_white_auto=self.config.balance_white_auto,
                black_level_enable=self.config.black_level_enable,
                black_level=self.config.black_level,
                brightness=self.config.brightness,
            )
            self.runtime_config = self.device.get_runtime_config()
            self.capture_path = "cpp_postprocess" if self.delivers_processed_frames else "copy"
            self.capture_path_reason = (
                "crop/resize handled inside the C++ backend"
                if self.delivers_processed_frames
                else self._CAPTURE_PATH_REASON
            )
        except Exception:
            if self.device is not None:
                try:
                    self.device.close()
                except Exception:
                    pass
            self.device = None
            self.runtime_config = None
            raise

    def stop(self) -> None:
        if self.device is not None:
            self.device.close()
            self.device = None
        self.runtime_config = None

    def read_frame(self, timeout_ms: int = 1000) -> _DriverFrame:
        if self.device is None:
            raise RuntimeError("C++ nanobind MVS device is not connected")

        frame = self.device.read_frame(timeout_ms=timeout_ms)
        try:
            image = np.array(frame.as_array(), dtype=np.uint8, copy=True)
        finally:
            frame.release()
        return _DriverFrame(image=image, frame_id=frame.frame_id, timestamp_ns=frame.timestamp_ns)


class _ProxyCameraDriver(_BaseCameraDriver):
    # 跟 _CppCameraDriver 相同的接口,但 mvs_cpp.Device 跑在独立子进程里。
    # 这是 P1 方案的核心:把 MVS SDK 与 Synexens SDK 隔离到不同 Python 进程,
    # 避免共进程 GIL/全局状态/kernel USB 调度互相踩踏。
    performs_rotation = True
    _CAPTURE_PATH_REASON = (
        "MVS runs in a dedicated subprocess to avoid sharing SDK state with Synexens"
    )

    def __init__(self, config: MVSCamControllerConfig):
        from .mvs_proxy import (
            MVSAsyncProxy,
            build_worker_config_from_camcontroller,
        )
        self._MVSAsyncProxy = MVSAsyncProxy
        self._build_worker_config = build_worker_config_from_camcontroller

        self.config = config
        self._proxy: Optional[object] = None
        # 第一版要求 cpp 内部完成 crop+resize(对应 _identity_transform 配置)。
        # 这与现有 collector 默认配置一致。
        self.delivers_processed_frames = _can_use_cpp_postprocess(config)
        self.capture_path = "subprocess"
        self.capture_path_reason: Optional[str] = self._CAPTURE_PATH_REASON
        self.runtime_config: Optional[MVSCameraRuntimeConfig] = None

    def start(self) -> None:
        if not self.delivers_processed_frames:
            raise NotImplementedError(
                "MVSAsyncProxy first version requires cpp post-processing; "
                "set crop_func/img_transform_func to identity (the default in collector.py)."
            )
        worker_config = self._build_worker_config(
            self.config, enable_cpp_postprocess=True
        )
        proxy = self._MVSAsyncProxy(
            worker_config, camera_name=self.config.name
        )
        try:
            proxy.start()
        except Exception:
            try:
                proxy.stop()
            except Exception:
                pass
            raise
        self._proxy = proxy
        # runtime_config 真值在子进程,主进程拿不到;用 config 推一份用于日志展示。
        # 实际帧 shape 与配置匹配性已经在子进程 worker 启动时校验。
        from .mvs_cpp.backend import _enum_mode  # noqa: WPS433
        self.runtime_config = MVSCameraRuntimeConfig(
            width=int(self.config.width),
            height=int(self.config.height),
            offset_x=int(self.config.offset_x),
            offset_y=int(self.config.offset_y),
            acquisition_frame_rate_enable=bool(self.config.acquisition_frame_rate_enable),
            acquisition_frame_rate_fps=float(self.config.fps),
            exposure_auto_mode=_enum_mode(self.config.exposure_auto),
            exposure_time_us=float(self.config.exposure_time_us),
            gain_auto_mode=_enum_mode(self.config.gain_auto),
            balance_white_auto_mode=_enum_mode(self.config.balance_white_auto),
            black_level_enable=bool(self.config.black_level_enable),
            black_level=int(self.config.black_level),
            brightness=int(self.config.brightness),
        )

    def stop(self) -> None:
        if self._proxy is not None:
            try:
                self._proxy.stop()
            except Exception as exc:
                logger.warning(f"MVSAsyncProxy stop failed: {exc}")
            self._proxy = None
        self.runtime_config = None

    def read_frame(self, timeout_ms: int = 1000) -> _DriverFrame:
        if self._proxy is None:
            raise RuntimeError("MVS subprocess proxy is not connected")
        try:
            image, frame_id, timestamp_ns = self._proxy.wait_next_capture(
                timeout_ms=timeout_ms
            )
        except TimeoutError as exc:
            # 兼容 _processing_loop 里的异常分发:转成 NoData
            from .mvs_cpp.backend import MVSCPPNoDataError
            raise MVSCPPNoDataError(str(exc)) from exc
        return _DriverFrame(
            image=image,
            frame_id=int(frame_id),
            timestamp_ns=int(timestamp_ns),
        )


def _postprocess_frame(config: MVSCamControllerConfig, image: np.ndarray) -> np.ndarray:
    frame = config.crop_func(image)
    if frame.shape[1] != config.width or frame.shape[0] != config.height:
        frame = cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_LINEAR)
    if config.img_transform_func is not None:
        frame = config.img_transform_func(frame)
    return np.asarray(frame, dtype=np.uint8)


def get_all_mvs_dev_serial(backend: Optional[str] = None) -> List[str]:
    _normalize_backend_name(backend)
    return list(list_mvs_device_serials())


class MVSImageCamera:
    def __init__(
        self,
        config: MVSCamControllerConfig,
        warmup_s: float = 1.0,
        allow_repeat_frames: bool = True,
        debug: bool = False,
        backend: Optional[str] = None,
    ):
        self.config = config
        self.config.validate()

        self.camera_name = config.name
        self.warmup_s = warmup_s
        self.allow_repeat_frames = allow_repeat_frames
        self.enable_frame_skip = True
        self.debug = debug
        self.backend = _normalize_backend_name(backend)
        self.backend_in_use: Optional[str] = None
        self.capture_path_in_use: Optional[str] = None
        self.runtime_config_in_use: Optional[MVSCameraRuntimeConfig] = None
        self.usb_link_in_use: Optional[MVSUSBLinkInfo] = None

        self.camera: Optional[_BaseCameraDriver] = None

        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None

        self.frame_lock = Lock()
        self.latest_image: Optional[np.ndarray] = None
        self.latest_image_processed = True
        self.latest_output_image: Optional[np.ndarray] = None
        self.latest_output_frame_id = 0
        self.latest_frame_id = 0
        self.latest_timestamp_ns = 0
        self.new_frame_event = Event()

        self.target_width = config.width
        self.target_height = config.height
        self.color_mode = "RGB"
        self.backend_read_timeout_ms = max(50, min(200, int(1000 / max(self.config.fps, 1)) * 2))

    @property
    def is_connected(self) -> bool:
        return self.camera is not None and self.thread is not None and self.thread.is_alive()

    def _create_driver(self) -> _BaseCameraDriver:
        # 默认走 proxy:MVS SDK 在独立子进程里跑,避免跟 Synexens SDK 共进程冲突。
        # 老的 _CppCameraDriver 路径保留,需要走老路径时设环境变量
        # OMNIUMI_MVS_INPROCESS=1。
        import os
        use_inprocess = os.getenv("OMNIUMI_MVS_INPROCESS", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

        try:
            if use_inprocess:
                driver = _CppCameraDriver(self.config)
                driver.start()
                self.backend_in_use = "cpp_inprocess"
                self.capture_path_in_use = driver.capture_path
                self.runtime_config_in_use = driver.runtime_config
            else:
                driver = _ProxyCameraDriver(self.config)
                driver.start()
                self.backend_in_use = "cpp_subprocess"
                self.capture_path_in_use = driver.capture_path
                self.runtime_config_in_use = driver.runtime_config
            if self.debug and driver.capture_path_reason is not None:
                logger.debug(
                    f"{self.camera_name}: backend path: {driver.capture_path_reason}"
                )
            return driver
        except Exception as exc:
            raise ConnectionError(f"Failed to create MVS backend for {self.camera_name}: {exc}") from exc

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self.camera_name} is already connected.")

        logger.info(f"Connecting {self.camera_name} (S/N: {self.config.serial})...")
        self.latest_image = None
        self.latest_image_processed = True
        self.latest_output_image = None
        self.latest_output_frame_id = 0
        self.latest_frame_id = 0
        self.latest_timestamp_ns = 0
        self.new_frame_event.clear()

        try:
            self.camera = self._create_driver()
            logger.info(
                f"{self.camera_name} backend ready: requested={self.backend}, "
                f"using={self.backend_in_use}, path={self.capture_path_in_use}"
            )
            if self.runtime_config_in_use is not None:
                logger.info(
                    f"{self.camera_name} runtime config: "
                    f"{self.runtime_config_in_use}"
                )
            self.usb_link_in_use = get_mvs_usb_link_info(self.config.serial)
            if self.usb_link_in_use is not None:
                if self.usb_link_in_use.is_usb2_fallback:
                    logger.warning(
                        f"{self.camera_name} usb link degraded: {self.usb_link_in_use.summary()} "
                        "This camera is not on a SuperSpeed link and may not sustain 30fps full-frame capture."
                    )
                else:
                    logger.info(
                        f"{self.camera_name} usb link: {self.usb_link_in_use.summary()}"
                    )
            else:
                logger.warning(
                    f"{self.camera_name}: unable to determine USB link speed from sysfs."
                )

            if warmup:
                logger.info(f"Warming up {self.camera_name} for {self.warmup_s:.1f}s...")
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    try:
                        self._read_frame(timeout_ms=1000)
                    except Exception as exc:
                        if self.debug:
                            logger.debug(f"{self.camera_name} warmup retry: {exc}")
                    time.sleep(0.1)

            self._start_thread()
            logger.info(f"{self.camera_name} connected and ready.")
        except Exception as exc:
            logger.error(f"Failed to connect {self.camera_name}: {exc}", exc_info=True)
            if self.camera is not None:
                try:
                    self.camera.stop()
                except Exception:
                    pass
            self.camera = None
            raise ConnectionError(f"Failed to open {self.camera_name}.") from exc

    def disconnect(self) -> None:
        if self.camera is None and self.thread is None:
            return

        logger.info(f"Disconnecting {self.camera_name}...")
        self._stop_thread()

        with self.frame_lock:
            self.latest_image = None
            self.latest_image_processed = True
            self.latest_output_image = None
            self.latest_output_frame_id = 0
            self.latest_frame_id = 0
            self.latest_timestamp_ns = 0

        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception as exc:
                logger.warning(f"{self.camera_name}: Error during backend stop: {exc}")
            self.camera = None

        self.backend_in_use = None
        self.capture_path_in_use = None
        self.runtime_config_in_use = None
        self.usb_link_in_use = None
        logger.info(f"{self.camera_name} disconnected.")

    def _read_frame(self, timeout_ms: int = 1000) -> _DriverFrame:
        if self.camera is None:
            raise RuntimeError(f"{self.camera_name} is not connected.")

        frame = self.camera.read_frame(timeout_ms=timeout_ms)
        if self.config.rotate_180 and not getattr(self.camera, "performs_rotation", False):
            frame.image = cv2.rotate(frame.image, cv2.ROTATE_180)
        return frame

    def _processing_loop(self) -> None:
        logger.info(f"{self.camera_name}: Background thread started.")

        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                loop_start = time.perf_counter()
                frame = self._read_frame(timeout_ms=self.backend_read_timeout_ms)
                image = frame.image
                frame_id = int(frame.frame_id)
                image_is_processed = bool(getattr(self.camera, "delivers_processed_frames", True))

                if self.debug and frame_id % 30 == 0:
                    loop_duration = (time.perf_counter() - loop_start) * 1000
                    logger.debug(
                        f"{self.camera_name} [Frame {frame_id}] backend read: {loop_duration:.1f}ms"
                    )

                if (
                    not self.allow_repeat_frames
                    and self.enable_frame_skip
                    and self.new_frame_event.is_set()
                ):
                    if self.debug and frame_id % 30 == 1:
                        logger.debug(f"{self.camera_name} [Frame {frame_id}] skipped")
                    continue

                with self.frame_lock:
                    self.latest_image = image
                    self.latest_image_processed = image_is_processed
                    self.latest_output_image = image if image_is_processed else None
                    self.latest_output_frame_id = frame_id if image_is_processed else 0
                    self.latest_frame_id = frame_id
                    self.latest_timestamp_ns = int(frame.timestamp_ns)
                self.new_frame_event.set()

            except MVSCPPNoDataError as exc:
                if self.debug:
                    logger.debug(f"{self.camera_name}: No frame yet ({exc})")
                time.sleep(0.01)
            except RuntimeError as exc:
                logger.warning(f"{self.camera_name}: Device error in background thread: {exc}")
                time.sleep(0.1)
                if self.camera is None:
                    break
            except Exception as exc:
                logger.error(f"{self.camera_name}: Error in background thread: {exc}", exc_info=True)
                time.sleep(0.1)

        logger.info(f"{self.camera_name}: Background thread stopped.")

    def _start_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self._stop_thread()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._processing_loop,
            name=f"{self.camera_name}_processing",
            daemon=True,
        )
        self.thread.start()

    def _stop_thread(self) -> None:
        if self.stop_event is None:
            return

        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
            if self.thread.is_alive():
                logger.error(f"{self.camera_name}: Thread did not stop gracefully!")
        self.thread = None
        self.stop_event = None
        self.new_frame_event.clear()

    def async_read(self, timeout_ms: float = 500) -> Tuple[np.ndarray, int]:
        if not self.is_connected:
            raise RuntimeError(f"{self.camera_name} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            logger.warning(f"{self.camera_name}: Thread not running, restarting...")
            self._start_thread()
            if not self.new_frame_event.wait(timeout=max(timeout_ms, 2000) / 1000.0):
                raise RuntimeError(f"{self.camera_name}: Thread restarted but no data.")

        wait_success = self.new_frame_event.wait(timeout=timeout_ms / 1000.0)
        if not wait_success:
            if not self.allow_repeat_frames:
                raise TimeoutError(
                    f"{self.camera_name}: Timeout waiting for new frame after {timeout_ms}ms."
                )

        with self.frame_lock:
            image = self.latest_image
            image_is_processed = self.latest_image_processed
            cached_output = self.latest_output_image
            cached_output_frame_id = self.latest_output_frame_id
            frame_id = self.latest_frame_id
            if wait_success:
                self.new_frame_event.clear()

        if image is None:
            raise RuntimeError(f"{self.camera_name}: No data available (timeout={timeout_ms}ms).")

        if image_is_processed:
            return image, frame_id

        if cached_output is not None and cached_output_frame_id == frame_id:
            return cached_output, frame_id

        output_image = _postprocess_frame(self.config, image)
        with self.frame_lock:
            if self.latest_frame_id == frame_id and not self.latest_image_processed:
                self.latest_output_image = output_image
                self.latest_output_frame_id = frame_id
        return output_image, frame_id

    def read(self) -> np.ndarray:
        image, _ = self.async_read(timeout_ms=1000)
        return image

    def __enter__(self) -> "MVSImageCamera":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"MVSImageCamera(name={self.camera_name!r}, serial={self.config.serial!r}, "
            f"output={self.target_width}x{self.target_height}, "
            f"backend={self.backend_in_use or self.backend}, "
            f"path={self.capture_path_in_use}, usb={self.usb_link_in_use.summary() if self.usb_link_in_use else 'unknown'}, "
            f"connected={self.is_connected})"
        )


def create_camera_configs_from_serials(
    serials: Sequence[str],
    fps: int = 30,
    output_res: Tuple[int, int] = (320, 240),
    input_res: Tuple[int, int] = (1440, 1080),
    crop_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    exposure_time_us: float = 15000.0,
    gain_auto: str = "continuous",
    prefer_sensor_roi: bool = False,
) -> List[MVSCamControllerConfig]:
    configs: List[MVSCamControllerConfig] = []
    crop = crop_func or _identity_transform
    use_sensor_roi = (
        prefer_sensor_roi
        and _is_identity_transform(crop)
        and tuple(output_res) == tuple(input_res)
    )
    sensor_width: Optional[int] = output_res[0] if use_sensor_roi else None
    sensor_height: Optional[int] = output_res[1] if use_sensor_roi else None
    offset_x = 0
    offset_y = 0
    crop_x = 0
    crop_y = 0
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None

    if not use_sensor_roi and _is_identity_transform(crop):
        crop_x, crop_y, crop_width, crop_height = _compute_center_crop_rect(
            input_res=input_res,
            output_res=output_res,
        )

    for idx, serial in enumerate(serials):
        config = MVSCamControllerConfig(
            receive_latency=0.125,
            serial=serial,
            name=f"camera{idx}_rgb",
            fps=fps,
            put_desired_frequency=fps,
            img_transform_func=_identity_transform,
            width=output_res[0],
            height=output_res[1],
            transformed_width=output_res[0],
            transformed_height=output_res[1],
            crop_func=crop,
            exposure_time_us=exposure_time_us,
            gain_auto=gain_auto,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            offset_x=offset_x,
            offset_y=offset_y,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
        )
        config.validate()
        configs.append(config)

    return configs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVS async reader test")
    parser.add_argument("--serial", type=str, default=None, help="Camera serial")
    parser.add_argument("--list-cameras", action="store_true", help="List MVS cameras and exit")
    parser.add_argument("--fps", type=int, default=60, help="Target fps")
    parser.add_argument("--frames", type=int, default=1000, help="Frames to sample")
    parser.add_argument("--width", type=int, default=320, help="Output width")
    parser.add_argument("--height", type=int, default=240, help="Output height")
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_MVS_CPP_BACKEND,
        choices=("cpp",),
        help="MVS backend to use",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("--show", action="store_true", help="Preview frames with OpenCV")
    return parser.parse_args()


def _resolve_target_serial(serial: Optional[str], backend: str) -> str:
    if serial is not None:
        return serial

    serials = get_all_mvs_dev_serial(backend=backend)
    if not serials:
        raise RuntimeError("No MVS cameras found")
    return serials[0]


def main() -> int:
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    if args.list_cameras:
        serials = get_all_mvs_dev_serial(backend=args.backend)
        if serials:
            logger.info(f"Found {len(serials)} MVS camera(s)")
            for idx, serial in enumerate(serials):
                usb_link = get_mvs_usb_link_info(serial)
                usb_suffix = f" usb={usb_link.summary()}" if usb_link is not None else ""
                logger.info(f"  {idx}: {serial}{usb_suffix}")
            return 0
        logger.warning("No MVS cameras found")
        return 1

    try:
        target_serial = _resolve_target_serial(args.serial, args.backend)
    except Exception as exc:
        logger.error(f"Failed to resolve camera serial: {exc}")
        return 1

    config = create_camera_configs_from_serials(
        serials=[target_serial],
        fps=args.fps,
        output_res=(args.width, args.height),
        prefer_sensor_roi=True,
    )[0]
    camera = MVSImageCamera(
        config=config,
        allow_repeat_frames=True,
        debug=args.debug,
        backend=args.backend,
    )

    try:
        camera.connect(warmup=True)
        logger.info(f"Connected: {camera}")

        frame_ids = []
        latencies = []
        start_time = time.time()

        for idx in range(args.frames):
            read_start = time.perf_counter()
            image, frame_id = camera.async_read(timeout_ms=1000)
            latencies.append((time.perf_counter() - read_start) * 1000)
            frame_ids.append(frame_id)

            if args.show:
                cv2.imshow("MVS Camera", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif (idx + 1) % 30 == 0:
                logger.info(f"Progress: {idx + 1}/{args.frames}")

        elapsed = time.time() - start_time
        logger.info(
            "Capture summary: "
            f"elapsed={elapsed:.2f}s, fps={len(frame_ids) / max(elapsed, 1e-6):.1f}, "
            f"avg_latency={np.mean(latencies):.2f}ms, max_latency={np.max(latencies):.2f}ms, "
            f"unique_frames={len(set(frame_ids))}/{len(frame_ids)}"
        )
        return 0
    except Exception as exc:
        logger.error(f"MVS test failed: {exc}", exc_info=True)
        return 1
    finally:
        if args.show:
            cv2.destroyAllWindows()
        camera.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
