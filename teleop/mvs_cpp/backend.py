from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MVS_CPP_BACKEND = "cpp"
_MODULE_NAME = "_mvs_camera"
_module_lock = Lock()
_cached_module = None


class MVSCPPBackendError(RuntimeError):
    pass


class MVSCPPBackendUnavailableError(MVSCPPBackendError):
    pass


class MVSCPPNoDataError(MVSCPPBackendError):
    pass


@dataclass(frozen=True)
class MVSCameraRuntimeConfig:
    width: int
    height: int
    offset_x: int
    offset_y: int
    acquisition_frame_rate_enable: bool
    acquisition_frame_rate_fps: float
    exposure_auto_mode: int
    exposure_time_us: float
    gain_auto_mode: int
    gain_db: float
    balance_white_auto_mode: int
    black_level_enable: bool
    black_level: int
    brightness: int


@dataclass(frozen=True)
class MVSUSBLinkInfo:
    serial: str
    sysfs_name: Optional[str]
    busnum: Optional[int]
    devnum: Optional[int]
    speed_mbps: Optional[float]
    usb_version: Optional[str]
    vendor_id: Optional[str]
    product_id: Optional[str]
    manufacturer: Optional[str]
    product: Optional[str]

    @property
    def is_superspeed(self) -> bool:
        return self.speed_mbps is not None and self.speed_mbps >= 5000.0

    @property
    def is_usb2_fallback(self) -> bool:
        return self.speed_mbps is not None and self.speed_mbps < 5000.0

    def summary(self) -> str:
        details: List[str] = []
        if self.usb_version:
            details.append(f"USB {self.usb_version}")
        if self.speed_mbps is not None:
            details.append(f"{self.speed_mbps:.0f}M")
        if self.busnum is not None and self.devnum is not None:
            details.append(f"bus={self.busnum} dev={self.devnum}")
        return " ".join(details) if details else "unavailable"


@dataclass(frozen=True)
class _CameraOpenConfig:
    sensor_width: Optional[int]
    sensor_height: Optional[int]
    offset_x: int
    offset_y: int
    crop_x: int
    crop_y: int
    crop_width: Optional[int]
    crop_height: Optional[int]
    output_width: Optional[int]
    output_height: Optional[int]
    image_node_num: int
    frame_pool_size: int
    rotate_180: bool
    acquisition_frame_rate_enable: bool
    acquisition_frame_rate_fps: float
    exposure_auto: str
    exposure_time_us: float
    gain_auto: str
    gain_db: Optional[float]
    balance_white_auto: str
    black_level_enable: bool
    black_level: int
    brightness: Optional[int]


def _read_text_file(path: Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None
    return value or None


def _read_int_file(path: Path) -> Optional[int]:
    value = _read_text_file(path)
    if value is None:
        return None
    try:
        return int(value, 10)
    except ValueError:
        return None


def _read_float_file(path: Path) -> Optional[float]:
    value = _read_text_file(path)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _iter_usb_sysfs_devices(root: Path = Path("/sys/bus/usb/devices")) -> Iterable[Path]:
    if not root.exists():
        return ()
    return sorted(path for path in root.iterdir() if path.is_dir())


def _build_usb_link_info(device_path: Path, serial: str) -> MVSUSBLinkInfo:
    return MVSUSBLinkInfo(
        serial=serial,
        sysfs_name=device_path.name,
        busnum=_read_int_file(device_path / "busnum"),
        devnum=_read_int_file(device_path / "devnum"),
        speed_mbps=_read_float_file(device_path / "speed"),
        usb_version=_read_text_file(device_path / "version"),
        vendor_id=_read_text_file(device_path / "idVendor"),
        product_id=_read_text_file(device_path / "idProduct"),
        manufacturer=_read_text_file(device_path / "manufacturer"),
        product=_read_text_file(device_path / "product"),
    )


def _float_close(lhs: float, rhs: float, tolerance: float) -> bool:
    return abs(float(lhs) - float(rhs)) <= float(tolerance)


def _verify_runtime_config_matches_request(
    runtime_config: MVSCameraRuntimeConfig,
    requested_config: _CameraOpenConfig,
) -> None:
    mismatches = []
    if requested_config.sensor_width is not None and runtime_config.width != int(requested_config.sensor_width):
        mismatches.append(
            f"width requested={int(requested_config.sensor_width)} actual={runtime_config.width}"
        )
    if requested_config.sensor_height is not None and runtime_config.height != int(requested_config.sensor_height):
        mismatches.append(
            f"height requested={int(requested_config.sensor_height)} actual={runtime_config.height}"
        )
    if runtime_config.offset_x != int(requested_config.offset_x):
        mismatches.append(f"offset_x requested={int(requested_config.offset_x)} actual={runtime_config.offset_x}")
    if runtime_config.offset_y != int(requested_config.offset_y):
        mismatches.append(f"offset_y requested={int(requested_config.offset_y)} actual={runtime_config.offset_y}")

    if runtime_config.acquisition_frame_rate_enable != bool(requested_config.acquisition_frame_rate_enable):
        mismatches.append(
            "acquisition_frame_rate_enable "
            f"requested={bool(requested_config.acquisition_frame_rate_enable)} "
            f"actual={runtime_config.acquisition_frame_rate_enable}"
        )
    if bool(requested_config.acquisition_frame_rate_enable) and not _float_close(
        runtime_config.acquisition_frame_rate_fps,
        requested_config.acquisition_frame_rate_fps,
        tolerance=max(0.1, abs(float(requested_config.acquisition_frame_rate_fps)) * 0.01),
    ):
        mismatches.append(
            "acquisition_frame_rate_fps "
            f"requested={float(requested_config.acquisition_frame_rate_fps):.3f} "
            f"actual={runtime_config.acquisition_frame_rate_fps:.3f}"
        )

    exposure_auto_mode = _enum_mode(requested_config.exposure_auto)
    if runtime_config.exposure_auto_mode != exposure_auto_mode:
        mismatches.append(
            f"exposure_auto requested={exposure_auto_mode} actual={runtime_config.exposure_auto_mode}"
        )
    if exposure_auto_mode == 0 and not _float_close(
        runtime_config.exposure_time_us,
        requested_config.exposure_time_us,
        tolerance=max(1.0, abs(float(requested_config.exposure_time_us)) * 0.01),
    ):
        mismatches.append(
            "exposure_time_us "
            f"requested={float(requested_config.exposure_time_us):.3f} "
            f"actual={runtime_config.exposure_time_us:.3f}"
        )

    gain_auto_mode = _enum_mode(requested_config.gain_auto)
    if runtime_config.gain_auto_mode != gain_auto_mode:
        mismatches.append(f"gain_auto requested={gain_auto_mode} actual={runtime_config.gain_auto_mode}")
    if (
        gain_auto_mode == 0
        and requested_config.gain_db is not None
        and not _float_close(
            runtime_config.gain_db,
            requested_config.gain_db,
            tolerance=max(0.05, abs(float(requested_config.gain_db)) * 0.01),
        )
    ):
        mismatches.append(
            "gain_db "
            f"requested={float(requested_config.gain_db):.3f} "
            f"actual={runtime_config.gain_db:.3f}"
        )

    balance_white_auto_mode = _enum_mode(requested_config.balance_white_auto)
    if runtime_config.balance_white_auto_mode != balance_white_auto_mode:
        mismatches.append(
            "balance_white_auto "
            f"requested={balance_white_auto_mode} actual={runtime_config.balance_white_auto_mode}"
        )

    if runtime_config.black_level_enable != bool(requested_config.black_level_enable):
        mismatches.append(
            "black_level_enable "
            f"requested={bool(requested_config.black_level_enable)} actual={runtime_config.black_level_enable}"
        )
    if bool(requested_config.black_level_enable) and runtime_config.black_level != int(requested_config.black_level):
        mismatches.append(
            f"black_level requested={int(requested_config.black_level)} actual={runtime_config.black_level}"
        )
    if requested_config.brightness is not None and runtime_config.brightness != int(requested_config.brightness):
        mismatches.append(f"brightness requested={int(requested_config.brightness)} actual={runtime_config.brightness}")

    if mismatches:
        raise MVSCPPBackendError(
            "MVS camera runtime config does not match the requested configuration: "
            + "; ".join(mismatches)
        )


def _verify_frame_shape_matches_request(
    frame_shape: Tuple[int, int],
    requested_config: _CameraOpenConfig,
) -> None:
    expected_width = int(requested_config.output_width or requested_config.crop_width or requested_config.sensor_width or 0)
    expected_height = int(
        requested_config.output_height or requested_config.crop_height or requested_config.sensor_height or 0
    )
    if expected_width <= 0 or expected_height <= 0:
        return
    actual_width = int(frame_shape[0])
    actual_height = int(frame_shape[1])
    if actual_width != expected_width or actual_height != expected_height:
        raise MVSCPPBackendError(
            "MVS processed frame shape does not match the requested output shape: "
            f"requested={expected_width}x{expected_height} actual={actual_width}x{actual_height}"
        )


def _enum_mode(mode: str) -> int:
    normalized = mode.strip().lower()
    mapping = {
        "off": 0,
        "once": 1,
        "continuous": 2,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported MVS auto mode: {mode!r}")
    return mapping[normalized]


def _candidate_module_paths(base_dir: Path) -> Iterable[Path]:
    env_path = (
        os.getenv("OMNIUMI_MVS_CPP_MODULE")
        or os.getenv("OMNIUMI_MVS_CPP_LIB")
        or os.getenv("ACTIVEUMI_MVS_CPP_MODULE")
        or os.getenv("ACTIVEUMI_MVS_CPP_LIB")
    )
    if env_path:
        yield Path(env_path).expanduser()

    roots = (
        base_dir,
        base_dir / "build",
        base_dir / "build" / "Release",
        base_dir / "build" / "Debug",
    )
    for root in roots:
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            yield root / f"{_MODULE_NAME}{suffix}"


def _reraise_module_error(exc: Exception) -> None:
    module = get_mvs_cpp_module()
    backend_error = getattr(module, "BackendError", RuntimeError)
    no_data_error = getattr(module, "NoDataError", RuntimeError)

    if isinstance(exc, no_data_error):
        raise MVSCPPNoDataError(str(exc)) from exc
    if isinstance(exc, backend_error):
        raise MVSCPPBackendError(str(exc)) from exc
    raise exc


def get_mvs_cpp_module():
    global _cached_module
    if _cached_module is not None:
        return _cached_module

    with _module_lock:
        if _cached_module is not None:
            return _cached_module

        package_name = __package__ or "mvs_cpp"
        try:
            _cached_module = importlib.import_module(f"{package_name}.{_MODULE_NAME}")
            return _cached_module
        except ImportError:
            pass

        base_dir = Path(__file__).resolve().parent
        tried = []
        for candidate in _candidate_module_paths(base_dir):
            tried.append(str(candidate))
            if not candidate.exists():
                continue

            spec = importlib.util.spec_from_file_location(f"{package_name}.{_MODULE_NAME}", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _cached_module = module
            return _cached_module

    tried_paths = "\n".join(f"  - {path}" for path in tried) or "  - <none>"
    raise MVSCPPBackendUnavailableError(
        "MVS nanobind extension module not found. Build it first or set "
        "OMNIUMI_MVS_CPP_MODULE.\n"
        f"Searched paths:\n{tried_paths}"
    )


def get_mvs_cpp_api():
    return get_mvs_cpp_module()


class MVSCPPFrame:
    def __init__(self, native_frame):
        self._native_frame = native_frame

    @property
    def frame_id(self) -> int:
        return int(self._native_frame.frame_id)

    @property
    def timestamp_ns(self) -> int:
        return int(self._native_frame.timestamp_ns)

    @property
    def device_timestamp_ticks(self) -> int:
        try:
            return int(self._native_frame.device_timestamp_ticks)
        except AttributeError:
            return 0

    @property
    def host_timestamp_ms(self) -> int:
        try:
            return int(self._native_frame.host_timestamp_ms)
        except AttributeError:
            return 0

    @property
    def timing_ns(self) -> dict:
        keys = (
            "get_buffer_ns",
            "convert_ns",
            "rotate_ns",
            "transform_ns",
            "free_buffer_ns",
            "total_ns",
        )
        values = {}
        for key in keys:
            try:
                values[key] = int(getattr(self._native_frame, key))
            except AttributeError:
                values[key] = 0
        return values

    @property
    def released(self) -> bool:
        return bool(self._native_frame.released)

    def as_array(self) -> np.ndarray:
        return np.asarray(self._native_frame.as_array(), dtype=np.uint8)

    def copy_array(self) -> np.ndarray:
        return np.array(self._native_frame.as_array(), dtype=np.uint8, copy=True)

    def release(self) -> None:
        self._native_frame.release()

    def __enter__(self) -> "MVSCPPFrame":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class MVSCPPDevice:
    def __init__(self, native_device, serial: str):
        self._device = native_device
        self.serial = serial
        self._frame_shape: Optional[Tuple[int, int]] = None

    @property
    def is_open(self) -> bool:
        return bool(self._device is not None and self._device.is_open)

    @property
    def frame_shape(self) -> Tuple[int, int]:
        if self._frame_shape is None:
            self._frame_shape = self.refresh_frame_shape()
        return self._frame_shape

    def refresh_frame_shape(self) -> Tuple[int, int]:
        if self._device is None:
            raise RuntimeError("MVS nanobind device is not connected")
        try:
            width, height = self._device.get_frame_shape()
        except Exception as exc:
            _reraise_module_error(exc)
        self._frame_shape = (int(width), int(height))
        return self._frame_shape

    def get_runtime_config(self) -> MVSCameraRuntimeConfig:
        if self._device is None:
            raise RuntimeError("MVS nanobind device is not connected")
        try:
            native = self._device.get_runtime_config()
        except Exception as exc:
            _reraise_module_error(exc)
        return MVSCameraRuntimeConfig(
            width=int(native.width),
            height=int(native.height),
            offset_x=int(native.offset_x),
            offset_y=int(native.offset_y),
            acquisition_frame_rate_enable=bool(native.acquisition_frame_rate_enable),
            acquisition_frame_rate_fps=float(native.acquisition_frame_rate_fps),
            exposure_auto_mode=int(native.exposure_auto_mode),
            exposure_time_us=float(native.exposure_time_us),
            gain_auto_mode=int(native.gain_auto_mode),
            gain_db=float(native.gain_db),
            balance_white_auto_mode=int(native.balance_white_auto_mode),
            black_level_enable=bool(native.black_level_enable),
            black_level=int(native.black_level),
            brightness=int(native.brightness),
        )

    def read_frame(self, timeout_ms: int = 1000) -> MVSCPPFrame:
        if self._device is None:
            raise RuntimeError("MVS nanobind device is not connected")
        try:
            native_frame = self._device.read_frame(timeout_ms=int(timeout_ms))
        except Exception as exc:
            _reraise_module_error(exc)
        return MVSCPPFrame(native_frame)

    def grab_rgb(self, timeout_ms: int = 1000) -> Tuple[np.ndarray, int, int]:
        frame = self.read_frame(timeout_ms=timeout_ms)
        try:
            return frame.copy_array(), frame.frame_id, frame.timestamp_ns
        finally:
            frame.release()

    def close(self) -> None:
        if self._device is None:
            return
        try:
            self._device.close()
        except Exception as exc:
            if not self.is_open:
                self._device = None
                self._frame_shape = None
            _reraise_module_error(exc)
        self._device = None
        self._frame_shape = None

    def __enter__(self) -> "MVSCPPDevice":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def list_mvs_device_serials() -> Sequence[str]:
    module = get_mvs_cpp_module()
    try:
        return list(module.enumerate_serials())
    except Exception as exc:
        _reraise_module_error(exc)


def get_mvs_usb_link_info(serial: str) -> Optional[MVSUSBLinkInfo]:
    if not serial:
        return None
    for device_path in _iter_usb_sysfs_devices():
        device_serial = _read_text_file(device_path / "serial")
        if device_serial != serial:
            continue
        return _build_usb_link_info(device_path, serial=device_serial)
    return None


def list_mvs_usb_link_info(serials: Optional[Sequence[str]] = None) -> Sequence[MVSUSBLinkInfo]:
    serial_filter = None if serials is None else {serial for serial in serials if serial}
    infos: List[MVSUSBLinkInfo] = []
    seen = set()
    for device_path in _iter_usb_sysfs_devices():
        serial = _read_text_file(device_path / "serial")
        if serial is None or serial in seen:
            continue
        if serial_filter is not None and serial not in serial_filter:
            continue
        infos.append(_build_usb_link_info(device_path, serial=serial))
        seen.add(serial)
    return infos


def open_mvs_cpp_device(
    serial: str,
    *,
    sensor_width: Optional[int] = None,
    sensor_height: Optional[int] = None,
    offset_x: int = 0,
    offset_y: int = 0,
    crop_x: int = 0,
    crop_y: int = 0,
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
    output_width: Optional[int] = None,
    output_height: Optional[int] = None,
    image_node_num: int = 1,
    frame_pool_size: int = 4,
    rotate_180: bool = True,
    acquisition_frame_rate_enable: bool = True,
    acquisition_frame_rate_fps: Optional[float] = None,
    exposure_auto: str = "off",
    exposure_time_us: float = 20000.0,
    gain_auto: str = "continuous",
    gain_db: Optional[float] = None,
    balance_white_auto: str = "continuous",
    black_level_enable: bool = True,
    black_level: int = 240,
    brightness: Optional[int] = None,
) -> MVSCPPDevice:
    config = _CameraOpenConfig(
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        offset_x=offset_x,
        offset_y=offset_y,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_width,
        crop_height=crop_height,
        output_width=output_width,
        output_height=output_height,
        image_node_num=image_node_num,
        frame_pool_size=frame_pool_size,
        rotate_180=rotate_180,
        acquisition_frame_rate_enable=acquisition_frame_rate_enable,
        acquisition_frame_rate_fps=float(acquisition_frame_rate_fps if acquisition_frame_rate_fps is not None else 30.0),
        exposure_auto=exposure_auto,
        exposure_time_us=exposure_time_us,
        gain_auto=gain_auto,
        gain_db=gain_db,
        balance_white_auto=balance_white_auto,
        black_level_enable=black_level_enable,
        black_level=black_level,
        brightness=brightness,
    )
    module = get_mvs_cpp_module()
    try:
        native_device = module.Device(
            serial=serial,
            roi_width=int(config.sensor_width or 0),
            roi_height=int(config.sensor_height or 0),
            offset_x=int(config.offset_x),
            offset_y=int(config.offset_y),
            crop_x=int(config.crop_x),
            crop_y=int(config.crop_y),
            crop_width=int(config.crop_width or 0),
            crop_height=int(config.crop_height or 0),
            output_width=int(config.output_width or 0),
            output_height=int(config.output_height or 0),
            image_node_num=max(1, int(config.image_node_num)),
            frame_pool_size=max(1, int(config.frame_pool_size)),
            rotate_180=bool(config.rotate_180),
            acquisition_frame_rate_enable=bool(config.acquisition_frame_rate_enable),
            acquisition_frame_rate_fps=float(config.acquisition_frame_rate_fps),
            exposure_auto_mode=_enum_mode(config.exposure_auto),
            exposure_time_us=float(config.exposure_time_us),
            gain_auto_mode=_enum_mode(config.gain_auto),
            gain_db=float(config.gain_db) if config.gain_db is not None else -1.0,
            balance_white_auto_mode=_enum_mode(config.balance_white_auto),
            black_level_enable=bool(config.black_level_enable),
            black_level=int(config.black_level),
            # Brightness is an optional GenICam node.  Some MVS firmware does
            # not expose it and returns 0x80000106 if we try to write it; use
            # the -1 sentinel so the C++ backend skips this optional setting
            # unless the caller explicitly requests a value.
            brightness=int(config.brightness) if config.brightness is not None else -1,
        )
    except Exception as exc:
        _reraise_module_error(exc)
    device = MVSCPPDevice(native_device=native_device, serial=serial)
    try:
        runtime_config = device.get_runtime_config()
        _verify_runtime_config_matches_request(runtime_config, config)
        _verify_frame_shape_matches_request(device.frame_shape, config)
    except Exception:
        device.close()
        raise
    return device
