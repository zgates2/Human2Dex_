#!/usr/bin/env python3

import argparse
import os
import sys
import time
from ctypes import POINTER, byref, cast, c_int, c_int32, c_ubyte, c_uint, c_ushort, c_void_p, memmove, sizeof
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from loguru import logger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SYNEXENS_SDK_DIR = os.path.join(CURRENT_DIR, "synexens_utils")

if SYNEXENS_SDK_DIR not in sys.path:
    sys.path.insert(0, SYNEXENS_SDK_DIR)

try:
    import SynexensPythonSDK as synexens_sdk
except ModuleNotFoundError as exc:
    synexens_sdk = None
    _SYNEXENS_IMPORT_ERROR = exc
else:
    _SYNEXENS_IMPORT_ERROR = None


DEFAULT_CAMERA_NAME = "synexens_depth_camera"
DEFAULT_RESOLUTION = (320, 240)
DEFAULT_WARMUP_S = 1.0
DEFAULT_INTEGRAL_TIME_US = 80
DEFAULT_DISTANCE_RANGE_MM = (100, 400)
SDK_POLL_INTERVAL_S = 0.005
SUPPORTED_RESOLUTION_ENUMS = {
    (320, 240): "SYRESOLUTION_320_240",
}


def _parse_resolution(resolution_text: str) -> Tuple[int, int]:
    try:
        width_text, height_text = resolution_text.lower().split("x", 1)
        return int(width_text), int(height_text)
    except Exception as exc:
        raise ValueError(
            f"Invalid resolution '{resolution_text}', expected format like 320x240"
        ) from exc


class SynexensDepthCamera:
    _sdk_lock = Lock()
    _sdk_ref_count = 0

    def __init__(
        self,
        camera_name: str = DEFAULT_CAMERA_NAME,
        device_id: Optional[int] = None,
        device_index: int = 0,
        depth_resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
        warmup_s: float = DEFAULT_WARMUP_S,
        integral_time_us: Optional[int] = DEFAULT_INTEGRAL_TIME_US,
        distance_range_mm: Optional[Tuple[int, int]] = DEFAULT_DISTANCE_RANGE_MM,
        enable_depth_color: bool = True,
        enable_frame_skip: bool = True,
        allow_repeat_frames: bool = True,
        show: bool = False,
        debug: bool = False,
    ):
        self.camera_name = camera_name
        self.requested_device_id = device_id
        self.device_index = device_index
        self.depth_resolution = depth_resolution
        self.warmup_s = warmup_s
        self.integral_time_us = integral_time_us
        self.distance_range_mm = distance_range_mm
        self.enable_depth_color = enable_depth_color
        self.enable_frame_skip = enable_frame_skip
        self.allow_repeat_frames = allow_repeat_frames
        self.show = show
        self.debug = debug

        self.device_id: Optional[int] = None
        self.device_info = None
        self.device_type = None
        self.serial_number: Optional[str] = None
        self.hardware_version: Optional[str] = None

        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None
        self.frame_lock = Lock()
        self.new_frame_event = Event()

        self.latest_data: Optional[Dict[str, Any]] = None
        self.latest_frame_id = 0

        self._sdk_ready = False
        self._streaming = False
        self._depth_window_name = f"{self.camera_name}_depth"
        self._ir_window_name = f"{self.camera_name}_ir"

    @property
    def is_connected(self) -> bool:
        return self._streaming and self.thread is not None and self.thread.is_alive()

    @classmethod
    def _require_sdk(cls):
        if synexens_sdk is None:
            raise ImportError(
                "SynexensPythonSDK is not installed or not on PYTHONPATH. "
                "Please source the Synexens SDK environment first."
            ) from _SYNEXENS_IMPORT_ERROR
        return synexens_sdk

    @classmethod
    def _sdk_init_ref(cls):
        sdk = cls._require_sdk()
        with cls._sdk_lock:
            if cls._sdk_ref_count == 0:
                ec = sdk.InitSDK()
                if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    raise RuntimeError(f"InitSDK failed with error code: {ec}")
            cls._sdk_ref_count += 1
        return sdk

    @classmethod
    def _sdk_release_ref(cls):
        sdk = synexens_sdk
        if sdk is None:
            return

        with cls._sdk_lock:
            if cls._sdk_ref_count <= 0:
                return
            cls._sdk_ref_count -= 1
            if cls._sdk_ref_count == 0:
                try:
                    sdk.UnInitSDK()
                except Exception as exc:
                    logger.warning(f"UnInitSDK failed: {exc}")

    @staticmethod
    def _enum_value(enum_like: Any) -> int:
        if hasattr(enum_like, "value"):
            return int(enum_like.value)
        return int(enum_like)

    @staticmethod
    def _pointer_address(pointer_like: Any) -> int:
        if isinstance(pointer_like, int):
            return pointer_like
        if hasattr(pointer_like, "value") and pointer_like.value is not None:
            return int(pointer_like.value)
        return int(cast(pointer_like, c_void_p).value)

    @staticmethod
    def _clone_data(data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                result[key] = value.copy()
            else:
                result[key] = value
        return result

    @classmethod
    def list_devices(cls) -> List[Dict[str, Any]]:
        sdk = cls._sdk_init_ref()
        try:
            device_count = c_int32()
            ec = sdk.FindDevice(byref(device_count), None)
            if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                raise RuntimeError(f"FindDevice failed with error code: {ec}")

            if device_count.value <= 0:
                return []

            device_infos = (sdk.SYDeviceInfo * device_count.value)()
            ec = sdk.FindDevice(device_count, device_infos)
            if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                raise RuntimeError(f"FindDevice(2nd call) failed with error code: {ec}")

            devices: List[Dict[str, Any]] = []
            for index in range(device_count.value):
                dev = device_infos[index]
                devices.append({
                    "index": index,
                    "device_id": int(dev.m_nDeviceID),
                    "device_type": getattr(dev.m_deviceType, "value", dev.m_deviceType),
                })
            return devices
        finally:
            cls._sdk_release_ref()

    def _get_resolution_enum(self):
        sdk = self._require_sdk()

        if self.depth_resolution not in SUPPORTED_RESOLUTION_ENUMS:
            raise ValueError(
                f"Unsupported Synexens resolution {self.depth_resolution}. "
                f"Supported resolutions: {sorted(SUPPORTED_RESOLUTION_ENUMS.keys())}"
            )

        enum_name = SUPPORTED_RESOLUTION_ENUMS[self.depth_resolution]
        if not hasattr(sdk.SYResolutionEnum, enum_name):
            raise AttributeError(f"Synexens SDK missing resolution enum {enum_name}")

        return getattr(sdk.SYResolutionEnum, enum_name)

    def _enumerate_device_infos(self):
        sdk = self._require_sdk()
        device_count = c_int32()
        ec = sdk.FindDevice(byref(device_count), None)
        if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            raise RuntimeError(f"FindDevice failed with error code: {ec}")
        if device_count.value <= 0:
            raise RuntimeError("No Synexens device found. Please check USB connection.")

        device_infos = (sdk.SYDeviceInfo * device_count.value)()
        ec = sdk.FindDevice(device_count, device_infos)
        if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            raise RuntimeError(f"FindDevice(2nd call) failed with error code: {ec}")

        return device_infos, device_count.value

    def _select_device(self):
        device_infos, device_count = self._enumerate_device_infos()

        if self.requested_device_id is not None:
            for index in range(device_count):
                if int(device_infos[index].m_nDeviceID) == int(self.requested_device_id):
                    return device_infos[index]
            raise RuntimeError(f"Requested device_id={self.requested_device_id} not found.")

        if self.device_index < 0 or self.device_index >= device_count:
            raise RuntimeError(
                f"device_index={self.device_index} out of range, found {device_count} device(s)."
            )

        return device_infos[self.device_index]

    def _configure_integral_time(self, resolution_enum) -> Optional[int]:
        sdk = self._require_sdk()
        if self.integral_time_us is None or self.device_id is None:
            return self.integral_time_us

        get_integral_range = getattr(sdk, "GetIntegralTimeRange", None)
        if not callable(get_integral_range):
            ec = sdk.SetIntegralTime(self.device_id, int(self.integral_time_us))
            if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                logger.warning(f"SetIntegralTime({self.integral_time_us}) failed with error code: {ec}")
                return None
            logger.info(f"{self.camera_name}: integral_time_us set to {self.integral_time_us}")
            return int(self.integral_time_us)

        min_value = c_int(0)
        max_value = c_int(0)
        ec = get_integral_range(self.device_id, resolution_enum, min_value, max_value)
        if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            logger.warning(f"GetIntegralTimeRange failed with error code: {ec}")
            return None

        requested = int(self.integral_time_us)
        clamped = min(max(requested, min_value.value), max_value.value)
        if clamped != requested:
            logger.warning(
                f"{self.camera_name}: requested integral_time_us={requested} is outside "
                f"[{min_value.value}, {max_value.value}], clamp to {clamped}"
            )

        ec = sdk.SetIntegralTime(self.device_id, clamped)
        if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
            logger.warning(
                f"SetIntegralTime({clamped}) failed with error code: {ec} "
                f"(range=[{min_value.value}, {max_value.value}])"
            )
            return None

        logger.info(
            f"{self.camera_name}: integral_time_us={clamped} "
            f"(supported range [{min_value.value}, {max_value.value}])"
        )
        return clamped

    def connect(self, warmup: bool = True):
        if self._streaming:
            raise RuntimeError(f"{self.camera_name} is already connected.")

        sdk = self._sdk_init_ref()
        self._sdk_ready = True

        try:
            self.device_info = self._select_device()
            self.device_id = int(self.device_info.m_nDeviceID)
            self.device_type = getattr(self.device_info.m_deviceType, "value", self.device_info.m_deviceType)

            logger.info(
                f"Connecting {self.camera_name} (device_id={self.device_id}, type={self.device_type})..."
            )

            ec = sdk.OpenDevice(self.device_info)
            if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                raise RuntimeError(f"OpenDevice failed with error code: {ec}")

            get_device_sn = getattr(sdk, "GetDeviceSN", None)
            if callable(get_device_sn):
                try:
                    self.serial_number = str(get_device_sn(self.device_id))
                except Exception:
                    self.serial_number = None

            get_hw_version = getattr(sdk, "GetDeviceHWVersion", None)
            if callable(get_hw_version):
                try:
                    self.hardware_version = str(get_hw_version(self.device_id))
                except Exception:
                    self.hardware_version = None

            resolution_enum = self._get_resolution_enum()
            depth_type = sdk.SYFrameTypeEnum.SYFRAMETYPE_DEPTH
            ec = sdk.SetFrameResolution(self.device_id, depth_type, resolution_enum)
            if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                raise RuntimeError(f"SetFrameResolution(DEPTH) failed with error code: {ec}")

            ir_type = getattr(sdk.SYFrameTypeEnum, "SYFRAMETYPE_IR", None)
            if ir_type is not None:
                ec = sdk.SetFrameResolution(self.device_id, ir_type, resolution_enum)
                if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    logger.warning(f"SetFrameResolution(IR) failed with error code: {ec}")

            if self.distance_range_mm is not None:
                min_mm, max_mm = self.distance_range_mm
                ec = sdk.SetDistanceUserRange(self.device_id, int(min_mm), int(max_mm))
                if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                    logger.warning(
                        f"SetDistanceUserRange({min_mm}, {max_mm}) failed with error code: {ec}"
                    )

            ec = sdk.StartStreaming(self.device_id, sdk.SYStreamTypeEnum.SYSTREAMTYPE_DEPTHIR)
            if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                raise RuntimeError(f"StartStreaming failed with error code: {ec}")

            actual_integral_time_us = self._configure_integral_time(resolution_enum)
            self._streaming = True
            self._start_thread()

            if warmup:
                logger.info(f"Warming up {self.camera_name} for {self.warmup_s:.1f}s...")
                self.new_frame_event.wait(timeout=max(2.0, self.warmup_s + 1.0))
                time.sleep(max(0.0, self.warmup_s))

            logger.info(
                f"{self.camera_name} connected. resolution={self.depth_resolution[0]}x{self.depth_resolution[1]}, "
                f"integral_time_us={actual_integral_time_us}, distance_range_mm={self.distance_range_mm}, "
                f"sn={self.serial_number}, hw={self.hardware_version}"
            )

        except Exception:
            self.disconnect()
            raise

    def disconnect(self):
        sdk = synexens_sdk

        self._stop_thread()

        if sdk is not None and self.device_id is not None:
            if self._streaming:
                try:
                    sdk.StopStreaming(self.device_id)
                except Exception as exc:
                    logger.warning(f"StopStreaming failed for {self.camera_name}: {exc}")

            close_device = getattr(sdk, "CloseDevice", None)
            if callable(close_device):
                try:
                    close_device(self.device_id)
                except Exception as exc:
                    logger.warning(f"CloseDevice failed for {self.camera_name}: {exc}")

        self._streaming = False
        self.device_id = None
        self.device_info = None
        self.device_type = None

        with self.frame_lock:
            self.latest_data = None
            self.latest_frame_id = 0

        self.new_frame_event.clear()
        self._close_preview_windows()

        if self._sdk_ready:
            self._sdk_ready = False
            self._sdk_release_ref()

    def _copy_uint16_frame(self, base_address: int, offset_bytes: int, count: int, shape: Tuple[int, int]) -> np.ndarray:
        frame_buffer = (c_ushort * count)()
        source_ptr = cast(c_void_p(base_address + offset_bytes), POINTER(c_ushort))
        memmove(frame_buffer, source_ptr, sizeof(c_ushort) * count)
        return np.frombuffer(frame_buffer, dtype=np.uint16).reshape(shape).copy()

    def _normalize_depth_to_mono8(self, depth_image: np.ndarray) -> np.ndarray:
        if depth_image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.convertScaleAbs(normalized)

    def _build_depth_visual(self, depth_image: np.ndarray) -> np.ndarray:
        sdk = self._require_sdk()
        height, width = depth_image.shape
        count = int(depth_image.size)

        depth_buffer = (c_ushort * count).from_buffer_copy(depth_image)
        color_buffer = (c_ubyte * (count * 3))()
        disp = np.zeros((height, width, 3), dtype=np.uint8)

        if self.enable_depth_color and self.device_id is not None:
            ec = sdk.GetDepthColor(c_uint(self.device_id), count, depth_buffer, color_buffer)
            if ec == sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                memmove(disp.ctypes.data, color_buffer, height * width * 3)
                return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        gray8 = self._normalize_depth_to_mono8(depth_image)
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2RGB)

    def _build_depth_visual_color(self, depth_image: np.ndarray) -> np.ndarray:
        sdk = self._require_sdk()
        height, width = depth_image.shape
        count = int(depth_image.size)

        if self.enable_depth_color and self.device_id is not None:
            depth_buffer = (c_ushort * count).from_buffer_copy(depth_image)
            color_buffer = (c_ubyte * (count * 3))()
            ec = sdk.GetDepthColor(c_uint(self.device_id), count, depth_buffer, color_buffer)
            if ec == sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS:
                color_np = np.ctypeslib.as_array(color_buffer).reshape(height, width, 3).copy()
                return cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

        normalized = self._normalize_depth_to_mono8(depth_image)
        color_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def normalize_ir_to_mono8(ir_image: np.ndarray) -> np.ndarray:
        if ir_image.dtype != np.uint16:
            ir_image = ir_image.astype(np.uint16)

        if ir_image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        gray8 = np.zeros(ir_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(ir_image, gray8, 0.5, 0)
        return gray8

    def _parse_frame_packet(self, frame_pointer) -> Dict[str, Optional[np.ndarray]]:
        sdk = self._require_sdk()
        packet = frame_pointer.contents

        base_address = self._pointer_address(packet.m_pData)
        depth_type = self._enum_value(sdk.SYFrameTypeEnum.SYFRAMETYPE_DEPTH)
        ir_type = self._enum_value(sdk.SYFrameTypeEnum.SYFRAMETYPE_IR)

        offset_bytes = 0
        depth_image = None
        ir_image = None

        for frame_index in range(packet.m_nFrameCount):
            frame_info = packet.m_pFrameInfo[frame_index]
            height = int(frame_info.m_nFrameHeight)
            width = int(frame_info.m_nFrameWidth)
            count = height * width
            frame_bytes = count * sizeof(c_ushort)
            frame_type = self._enum_value(frame_info.m_frameType)

            if frame_type == depth_type:
                depth_image = self._copy_uint16_frame(base_address, offset_bytes, count, (height, width))
            elif frame_type == ir_type:
                ir_image = self._copy_uint16_frame(base_address, offset_bytes, count, (height, width))

            offset_bytes += frame_bytes

        depth_visual = None if depth_image is None else self._build_depth_visual(depth_image)
        depth_visual_color = None if depth_image is None else self._build_depth_visual_color(depth_image)
        ir_visual = None if ir_image is None else self.normalize_ir_to_mono8(ir_image)

        return {
            "depth_mm": depth_image,
            "ir": ir_image,
            "depth_visual": depth_visual,
            "depth_visual_color": depth_visual_color,
            "ir_visual": ir_visual,
        }

    def _display_preview(self, depth_visual: Optional[np.ndarray], ir_visual: Optional[np.ndarray]) -> None:
        if depth_visual is not None:
            cv2.imshow(self._depth_window_name, depth_visual)
        if ir_visual is not None:
            cv2.imshow(self._ir_window_name, ir_visual)
        cv2.waitKey(1)

    def _close_preview_windows(self) -> None:
        if not self.show:
            return
        for window_name in [self._depth_window_name, self._ir_window_name]:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    def _processing_loop(self):
        sdk = self._require_sdk()
        frame_id = 0
        processing_times: List[float] = []
        logger.info(f"{self.camera_name}: Background thread started.")

        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                loop_start = time.perf_counter()
                frame_pointer = POINTER(sdk.SYFrameData)()
                ec = sdk.GetLastFrameData(self.device_id, byref(frame_pointer))
                if ec != sdk.SYErrorCodeEnum.SYERRORCODE_SUCCESS or not bool(frame_pointer):
                    time.sleep(SDK_POLL_INTERVAL_S)
                    continue

                parsed = self._parse_frame_packet(frame_pointer)
                if parsed["depth_mm"] is None and parsed["ir"] is None:
                    time.sleep(SDK_POLL_INTERVAL_S)
                    continue

                if not self.allow_repeat_frames and self.enable_frame_skip and self.new_frame_event.is_set():
                    continue

                frame_id += 1
                parsed["timestamp_ns"] = time.time_ns()

                with self.frame_lock:
                    self.latest_data = parsed
                    self.latest_frame_id = frame_id

                self.new_frame_event.set()

                if self.show:
                    self._display_preview(parsed["depth_visual"], parsed["ir_visual"])

                loop_duration_ms = (time.perf_counter() - loop_start) * 1000.0
                processing_times.append(loop_duration_ms)
                if len(processing_times) > 100:
                    processing_times.pop(0)

                if self.debug and frame_id % 30 == 0:
                    logger.debug(
                        f"{self.camera_name} [Frame {frame_id}]: avg={np.mean(processing_times):.1f}ms, "
                        f"max={np.max(processing_times):.1f}ms"
                    )

            except Exception as exc:
                if self.stop_event is not None and self.stop_event.is_set():
                    break
                logger.error(f"{self.camera_name}: Error in background thread: {exc}", exc_info=True)
                time.sleep(0.05)

        logger.info(f"{self.camera_name}: Background thread stopped.")

    def _start_thread(self):
        if self.thread is not None and self.thread.is_alive():
            self._stop_thread()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._processing_loop,
            name=f"{self.camera_name}_processing",
            daemon=True,
        )
        self.thread.start()

    def _stop_thread(self):
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
            if self.thread.is_alive():
                logger.error(f"{self.camera_name}: Thread did not stop gracefully.")

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 500) -> Tuple[Dict[str, Any], int]:
        if not self._streaming:
            raise RuntimeError(f"{self.camera_name} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            logger.warning(f"{self.camera_name}: Thread not running, restarting...")
            self._start_thread()
            if not self.new_frame_event.wait(timeout=max(timeout_ms, 2000) / 1000.0):
                raise RuntimeError(f"{self.camera_name}: Thread restarted but no data.")

        wait_success = self.new_frame_event.wait(timeout=timeout_ms / 1000.0)
        if not wait_success and not self.allow_repeat_frames:
            raise TimeoutError(
                f"{self.camera_name}: Timeout waiting for NEW frame after {timeout_ms}ms."
            )

        with self.frame_lock:
            data = None if self.latest_data is None else self._clone_data(self.latest_data)
            frame_id = self.latest_frame_id
            if not self.allow_repeat_frames:
                self.new_frame_event.clear()

        if data is None:
            raise RuntimeError(f"{self.camera_name}: No data available (timeout={timeout_ms}ms).")

        return data, frame_id

    def read(self) -> Dict[str, Any]:
        data, _ = self.async_read(timeout_ms=1000)
        return data

    @staticmethod
    def get_depth_in_meters(depth_image_mm: np.ndarray) -> np.ndarray:
        return depth_image_mm.astype(np.float32) * 0.001

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"SynexensDepthCamera(name='{self.camera_name}', "
            f"device_id={self.requested_device_id}, "
            f"resolution={self.depth_resolution[0]}x{self.depth_resolution[1]}, "
            f"connected={self.is_connected})"
        )


class SynexensDepthCameraPublisher(Node):
    def __init__(self,
                 camera_name: str,
                 device_id: Optional[int] = None,
                 depth_resolution: Tuple[int, int] = DEFAULT_RESOLUTION,
                 fps: int = 30,
                 integral_time_us: Optional[int] = DEFAULT_INTEGRAL_TIME_US,
                 distance_range_mm: Optional[Tuple[int, int]] = DEFAULT_DISTANCE_RANGE_MM,
                 debug: bool = False,
                 **kwargs
                 ):
        super().__init__(f'{camera_name}_publisher')
        self.camera_name = camera_name
        self.fps = fps
        self.debug = debug
        self.bridge = CvBridge()
        
        self.camera = SynexensDepthCamera(
            camera_name=camera_name,
            device_id=device_id,
            depth_resolution=depth_resolution,
            integral_time_us=integral_time_us,
            distance_range_mm=distance_range_mm,
            allow_repeat_frames=True,
            show=False,
            debug=debug
        )
        
        self.camera.connect(warmup=True)
        
        self.depth_pub = self.create_publisher(Image, f'/{camera_name}/depth/image_raw', 10)
        self.ir_pub = self.create_publisher(Image, f'/{camera_name}/ir/image_raw', 10)
        
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Initialized {camera_name} publisher at {fps} FPS")

    def timer_callback(self):
        try:
            # allow_repeat_frames=True means this won't block if there's no new hardware frame ready, achieving 30Hz natively
            data, frame_id = self.camera.async_read(timeout_ms=100)
            target_time = self.get_clock().now()
            
            depth_img = data.get("depth_mm")
            ir_img = data.get("ir")
            
            if depth_img is not None:
                msg = Image()
                msg.header.stamp = target_time.to_msg()
                msg.header.frame_id = f"{self.camera_name}_depth_optical_frame"
                msg.height = depth_img.shape[0]
                msg.width = depth_img.shape[1]
                msg.encoding = "mono16"
                msg.is_bigendian = False
                msg.step = msg.width * depth_img.itemsize
                msg.data = depth_img.tobytes()
                self.depth_pub.publish(msg)

            if ir_img is not None:
                msg = Image()
                msg.header.stamp = target_time.to_msg()
                msg.header.frame_id = f"{self.camera_name}_ir_optical_frame"
                msg.height = ir_img.shape[0]
                msg.width = ir_img.shape[1]
                msg.encoding = "mono16"
                msg.is_bigendian = False
                msg.step = msg.width * ir_img.itemsize
                msg.data = ir_img.tobytes()
                self.ir_pub.publish(msg)

        except Exception as e:
            if self.debug:
                self.get_logger().error(f"Error reading from {self.camera_name}: {e}")

    def __del__(self):
        if hasattr(self, 'camera'):
            self.camera.disconnect()

def main(args=None):
    rclpy.init(args=args)
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="Synexens Depth+IR async wrapper test")
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_CAMERA_NAME,
        help=f"camera name (default: {DEFAULT_CAMERA_NAME})",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="SDK device ID to open. If omitted, use --device-index.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Enumerated device index to open when --device-id is omitted.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]}",
        help="depth/IR resolution, currently only 320x240 is guaranteed.",
    )
    parser.add_argument(
        "--integral-time-us",
        type=int,
        default=DEFAULT_INTEGRAL_TIME_US,
        help=f"sensor integral time in microseconds (default: {DEFAULT_INTEGRAL_TIME_US})",
    )
    parser.add_argument(
        "--distance-min-mm",
        type=int,
        default=DEFAULT_DISTANCE_RANGE_MM[0],
        help="minimum user depth range in mm",
    )
    parser.add_argument(
        "--distance-max-mm",
        type=int,
        default=DEFAULT_DISTANCE_RANGE_MM[1],
        help="maximum user depth range in mm",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list all discoverable Synexens devices and exit",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show depth and IR preview windows",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="number of frames to print before exit, 0 means run until Ctrl+C",
    )
    parser.add_argument(
        "--no-repeat-frames",
        action="store_true",
        help="wait for a NEW frame on every async_read, useful for measuring actual hardware fps",
    )
    parser.add_argument(
        "--measure-fps",
        action="store_true",
        help="report measured fps from received frames before exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable extra logging",
    )
    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    try:
        depth_resolution = _parse_resolution(args.resolution)
    except ValueError as exc:
        print(exc)
        return

    if args.list_devices:
        try:
            devices = SynexensDepthCamera.list_devices()
        except Exception as exc:
            print(f"Failed to list Synexens devices: {exc}")
            return

        if not devices:
            print("No Synexens device found.")
            return

        print(f"Found {len(devices)} Synexens device(s):")
        for item in devices:
            print(
                f"  [index={item['index']}] device_id={item['device_id']} "
                f"type={item['device_type']}"
            )
        return

    camera = SynexensDepthCamera(
        camera_name=args.name,
        device_id=args.device_id,
        device_index=args.device_index,
        depth_resolution=depth_resolution,
        integral_time_us=args.integral_time_us,
        distance_range_mm=(args.distance_min_mm, args.distance_max_mm),
        allow_repeat_frames=not args.no_repeat_frames,
        show=args.show,
        debug=args.debug,
    )

    frame_count = 0
    frame_ids: List[int] = []
    timestamps_ns: List[int] = []
    try:
        camera.connect(warmup=True)
        while True:
            data, frame_id = camera.async_read(timeout_ms=1000)
            depth_image = data.get("depth_mm")
            ir_image = data.get("ir")
            timestamp_ns = data.get("timestamp_ns")
            frame_ids.append(frame_id)
            if isinstance(timestamp_ns, int):
                timestamps_ns.append(timestamp_ns)

            if depth_image is not None:
                depth_min = int(depth_image[depth_image > 0].min()) if np.any(depth_image > 0) else 0
                depth_max = int(depth_image.max())
                depth_shape = depth_image.shape
            else:
                depth_min = 0
                depth_max = 0
                depth_shape = None

            ir_shape = None if ir_image is None else ir_image.shape
            print(
                f"frame_id={frame_id} timestamp_ns={timestamp_ns} "
                f"depth_shape={depth_shape} depth_range_mm=({depth_min}, {depth_max}) ir_shape={ir_shape}"
            )

            frame_count += 1
            if args.frames > 0 and frame_count >= args.frames:
                break

    except KeyboardInterrupt:
        logger.info("User interrupted.")
    finally:
        camera.disconnect()

        if args.measure_fps:
            unique_frame_count = len(set(frame_ids))
            if len(timestamps_ns) >= 2 and unique_frame_count >= 2:
                elapsed_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
                measured_fps = (len(timestamps_ns) - 1) / elapsed_s if elapsed_s > 0 else 0.0
                unique_elapsed_fps = (
                    (unique_frame_count - 1) / elapsed_s if elapsed_s > 0 else 0.0
                )
                print()
                print("FPS Summary")
                print(f"  frames_read={len(frame_ids)}")
                print(f"  unique_frames={unique_frame_count}")
                print(f"  frame_id_range={frame_ids[0]} -> {frame_ids[-1]}")
                print(f"  elapsed_s={elapsed_s:.3f}")
                print(f"  measured_fps={measured_fps:.2f}")
                if unique_frame_count != len(frame_ids):
                    print(f"  unique_frame_fps={unique_elapsed_fps:.2f}")
                    print("  note=repeat frames detected; use --no-repeat-frames for actual hardware fps")
            else:
                print()
                print("FPS Summary")
                print("  insufficient frame timestamps to measure fps")


if __name__ == "__main__":
    main()
