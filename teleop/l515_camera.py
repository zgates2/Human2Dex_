from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pyrealsense2 as rs


@dataclass(frozen=True)
class L515Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: list[float]


@dataclass(frozen=True)
class L515Config:
    name: str
    serial: str | None = None
    rgb_width: int = 640
    rgb_height: int = 480
    depth_width: int = 640
    depth_height: int = 480
    rgb_fps: int = 60
    depth_fps: int = 30
    align_to: str = "color"
    timeout_ms: int = 5000
    warmup_frames: int = 90
    strict_global_time: bool = True
    auto_exposure: bool = True
    auto_exposure_priority: float = 0.0
    auto_white_balance: bool = True


@dataclass
class L515CachedFrame:
    color_rgb: np.ndarray | None
    depth_z16: np.ndarray | None
    color_frame_number: int | None
    depth_frame_number: int | None
    color_timestamp_ms: float | None
    depth_timestamp_ms: float | None
    color_timestamp_domain: str | None
    depth_timestamp_domain: str | None
    color_capture_ns: int | None
    depth_capture_ns: int | None
    color_intrinsics: L515Intrinsics | None
    depth_intrinsics: L515Intrinsics | None
    color_metadata: dict[str, float]
    depth_metadata: dict[str, float]
    depth_scale_m_per_unit: float

    @property
    def color_timestamp_ns(self) -> int | None:
        if self.color_timestamp_ms is None:
            return None
        return int(round(self.color_timestamp_ms * 1_000_000.0))

    @property
    def depth_timestamp_ns(self) -> int | None:
        if self.depth_timestamp_ms is None:
            return None
        return int(round(self.depth_timestamp_ms * 1_000_000.0))

    @property
    def is_global_time(self) -> bool:
        return (
            self.color_timestamp_domain == "global_time"
            and self.depth_timestamp_domain == "global_time"
        )


def timestamp_domain_name(domain) -> str:
    for name in ("hardware_clock", "system_time", "global_time"):
        if hasattr(rs.timestamp_domain, name) and domain == getattr(rs.timestamp_domain, name):
            return name
    return str(domain)


def distortion_model_name(model) -> str:
    for name in (
        "none",
        "modified_brown_conrady",
        "inverse_brown_conrady",
        "brown_conrady",
        "kannala_brandt4",
    ):
        if hasattr(rs.distortion, name) and model == getattr(rs.distortion, name):
            return name
    return str(model)


def intrinsics_from_frame(frame: rs.frame) -> L515Intrinsics:
    intr = frame.profile.as_video_stream_profile().get_intrinsics()
    return L515Intrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        ppx=float(intr.ppx),
        ppy=float(intr.ppy),
        model=distortion_model_name(intr.model),
        coeffs=[float(v) for v in intr.coeffs],
    )


def frame_metadata(frame: rs.frame) -> dict[str, float]:
    result: dict[str, float] = {}
    names = (
        "frame_counter",
        "frame_timestamp",
        "sensor_timestamp",
        "backend_timestamp",
        "time_of_arrival",
        "actual_exposure",
        "gain_level",
    )
    for name in names:
        if not hasattr(rs.frame_metadata_value, name):
            continue
        key = getattr(rs.frame_metadata_value, name)
        try:
            if frame.supports_frame_metadata(key):
                result[name] = float(frame.get_frame_metadata(key))
        except Exception:
            continue
    return result


def set_option_if_supported(sensor: rs.sensor, option, value: float, label: str) -> bool:
    if not sensor.supports(option):
        return False
    sensor.set_option(option, float(value))
    try:
        sensor_name = sensor.get_info(rs.camera_info.name)
    except Exception:
        sensor_name = "sensor"
    print(f"[Init] {sensor_name}: set {label}={value}")
    return True


def configure_l515_sensor_options(device: rs.device, config: L515Config) -> None:
    for sensor in device.query_sensors():
        set_option_if_supported(
            sensor,
            rs.option.global_time_enabled,
            1.0,
            "global_time_enabled",
        )
        if config.auto_exposure:
            set_option_if_supported(
                sensor,
                rs.option.enable_auto_exposure,
                1.0,
                "enable_auto_exposure",
            )
            if hasattr(rs.option, "auto_exposure_priority"):
                set_option_if_supported(
                    sensor,
                    rs.option.auto_exposure_priority,
                    float(config.auto_exposure_priority),
                    "auto_exposure_priority",
                )
        if config.auto_white_balance:
            set_option_if_supported(
                sensor,
                rs.option.enable_auto_white_balance,
                1.0,
                "enable_auto_white_balance",
            )


def list_l515_devices() -> list[dict[str, str]]:
    devices = []
    for dev in rs.context().query_devices():
        devices.append(
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "product_line": dev.get_info(rs.camera_info.product_line),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            }
        )
    return devices


def fallback_rs_enumerate_summary() -> str:
    exe = shutil.which("rs-enumerate-devices")
    if exe is None:
        return ""
    try:
        proc = subprocess.run(
            [exe, "-s"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5.0,
        )
    except Exception as exc:
        return f"rs-enumerate-devices failed: {exc}"
    return proc.stdout.strip()


def find_l515_serial(requested_serial: Optional[str]) -> str:
    if requested_serial:
        # Do not create a pyrealsense2 context just to validate an explicit
        # serial. In spawn/fork multi-camera setups, that extra enumeration can
        # leave librealsense/libusb state in the wrong process. pipeline.start()
        # will report a clear error if the serial is wrong or unavailable.
        return requested_serial
    devices = list_l515_devices()

    l500 = [dev for dev in devices if dev["product_line"] == "L500"]
    if len(l500) == 1:
        return l500[0]["serial"]
    if len(l500) > 1:
        serials = ", ".join(dev["serial"] for dev in l500)
        raise RuntimeError(f"Multiple L515/L500 cameras found, pass serial. Found: {serials}")

    summary = fallback_rs_enumerate_summary()
    raise RuntimeError(
        "pyrealsense2 did not enumerate any L515/L500 camera. "
        "Use a Python binding built against the same librealsense stack that "
        "rs-enumerate-devices uses.\n"
        f"rs-enumerate-devices output:\n{summary}"
    )


class LatestL515Cache:
    """
    Background RealSense L515 reader that keeps the newest RGB and depth frames.

    L515 color can run at 60 Hz while depth runs at 30 Hz. The RealSense pipeline
    may repeat the latest depth frame on every color frameset, so this cache
    tracks color and depth independently by frame number and preserves separate
    host-cache timestamps for each stream.
    """

    def __init__(self, config: L515Config):
        self.config = config
        self.serial = find_l515_serial(config.serial)
        self.pipeline = rs.pipeline()
        self.align = self._make_align(config.align_to)
        self.profile = None
        self.depth_scale_m_per_unit = math.nan
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._latest = L515CachedFrame(
            color_rgb=None,
            depth_z16=None,
            color_frame_number=None,
            depth_frame_number=None,
            color_timestamp_ms=None,
            depth_timestamp_ms=None,
            color_timestamp_domain=None,
            depth_timestamp_domain=None,
            color_capture_ns=None,
            depth_capture_ns=None,
            color_intrinsics=None,
            depth_intrinsics=None,
            color_metadata={},
            depth_metadata={},
            depth_scale_m_per_unit=math.nan,
        )
        self._frames_read = 0

    def _make_align(self, align_to: str):
        if align_to == "color":
            return rs.align(rs.stream.color)
        if align_to == "depth":
            return rs.align(rs.stream.depth)
        if align_to == "none":
            return None
        raise ValueError(f"Unsupported L515 align_to: {align_to}")

    def start(self) -> None:
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(
            rs.stream.color,
            int(self.config.rgb_width),
            int(self.config.rgb_height),
            rs.format.rgb8,
            int(self.config.rgb_fps),
        )
        cfg.enable_stream(
            rs.stream.depth,
            int(self.config.depth_width),
            int(self.config.depth_height),
            rs.format.z16,
            int(self.config.depth_fps),
        )

        resolved = cfg.resolve(rs.pipeline_wrapper(self.pipeline))
        configure_l515_sensor_options(resolved.get_device(), self.config)
        self.profile = self.pipeline.start(cfg)
        device = self.profile.get_device()
        configure_l515_sensor_options(device, self.config)
        self.depth_scale_m_per_unit = float(device.first_depth_sensor().get_depth_scale())
        print(
            f"[Init] L515 name={self.config.name} serial={self.serial} "
            f"rgb={self.config.rgb_width}x{self.config.rgb_height}@{self.config.rgb_fps} "
            f"depth={self.config.depth_width}x{self.config.depth_height}@{self.config.depth_fps} "
            f"align_to={self.config.align_to} depth_scale={self.depth_scale_m_per_unit}"
        )
        self._warmup()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, float(self.config.timeout_ms) / 1000.0 + 0.5))
        try:
            self.pipeline.stop()
        except Exception as exc:
            print(f"[Warn] L515 {self.config.name} stop failed: {exc}")

    def latest(self, copy_images: bool = True) -> L515CachedFrame:
        with self._lock:
            color = self._latest.color_rgb
            depth = self._latest.depth_z16
            if color is not None and copy_images:
                color = color.copy()
            if depth is not None and copy_images:
                depth = depth.copy()
            return L515CachedFrame(
                color_rgb=color,
                depth_z16=depth,
                color_frame_number=self._latest.color_frame_number,
                depth_frame_number=self._latest.depth_frame_number,
                color_timestamp_ms=self._latest.color_timestamp_ms,
                depth_timestamp_ms=self._latest.depth_timestamp_ms,
                color_timestamp_domain=self._latest.color_timestamp_domain,
                depth_timestamp_domain=self._latest.depth_timestamp_domain,
                color_capture_ns=self._latest.color_capture_ns,
                depth_capture_ns=self._latest.depth_capture_ns,
                color_intrinsics=self._latest.color_intrinsics,
                depth_intrinsics=self._latest.depth_intrinsics,
                color_metadata=dict(self._latest.color_metadata),
                depth_metadata=dict(self._latest.depth_metadata),
                depth_scale_m_per_unit=self._latest.depth_scale_m_per_unit,
            )

    def metadata(self) -> dict:
        latest = self.latest(copy_images=False)
        return {
            "name": self.config.name,
            "serial": self.serial,
            "rgbWidth": int(self.config.rgb_width),
            "rgbHeight": int(self.config.rgb_height),
            "depthWidth": int(self.config.depth_width),
            "depthHeight": int(self.config.depth_height),
            "rgbFps": int(self.config.rgb_fps),
            "depthFps": int(self.config.depth_fps),
            "alignTo": self.config.align_to,
            "strictGlobalTime": bool(self.config.strict_global_time),
            "autoExposure": bool(self.config.auto_exposure),
            "autoExposurePriority": float(self.config.auto_exposure_priority),
            "autoWhiteBalance": bool(self.config.auto_white_balance),
            "depthScaleMPerUnit": float(self.depth_scale_m_per_unit),
            "colorIntrinsics": (
                None if latest.color_intrinsics is None else asdict(latest.color_intrinsics)
            ),
            "depthIntrinsics": (
                None if latest.depth_intrinsics is None else asdict(latest.depth_intrinsics)
            ),
            "lastColorTimestampDomain": latest.color_timestamp_domain,
            "lastDepthTimestampDomain": latest.depth_timestamp_domain,
            "framesRead": int(self._frames_read),
        }

    def _warmup(self) -> None:
        ready_at = None
        last_domains = None
        for idx in range(max(0, int(self.config.warmup_frames))):
            frame = self._read_once(require_global_time=False)
            last_domains = (frame.color_timestamp_domain, frame.depth_timestamp_domain)
            with self._lock:
                self._latest = frame
            if frame.is_global_time and ready_at is None:
                ready_at = idx + 1
        if ready_at is not None:
            print(
                f"[Init] L515 {self.config.name} global_time ready after "
                f"{ready_at} frames; dropped {self.config.warmup_frames} warmup frames"
            )
            return
        if self.config.strict_global_time:
            raise RuntimeError(
                f"L515 {self.config.name} global_time not ready. "
                f"Last domains={last_domains}. Raw hardware_clock timestamps are "
                "not safe for multi-camera synchronization."
            )
        print(f"[Warn] L515 {self.config.name} global_time not ready; last domains={last_domains}")

    def _read_once(self, require_global_time: bool | None = None) -> L515CachedFrame:
        frames = self.pipeline.wait_for_frames(int(self.config.timeout_ms))
        host_ns = time.monotonic_ns()
        if self.align is not None:
            frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame and not depth_frame:
            raise RuntimeError(f"L515 {self.config.name} missing both color and depth frames")

        with self._lock:
            latest = self._latest
            color_rgb = latest.color_rgb
            depth_z16 = latest.depth_z16
            color_frame_number = latest.color_frame_number
            depth_frame_number = latest.depth_frame_number
            color_timestamp_ms = latest.color_timestamp_ms
            depth_timestamp_ms = latest.depth_timestamp_ms
            color_timestamp_domain = latest.color_timestamp_domain
            depth_timestamp_domain = latest.depth_timestamp_domain
            color_capture_ns = latest.color_capture_ns
            depth_capture_ns = latest.depth_capture_ns
            color_intrinsics = latest.color_intrinsics
            depth_intrinsics = latest.depth_intrinsics
            color_meta = dict(latest.color_metadata)
            depth_meta = dict(latest.depth_metadata)

        if color_frame:
            frame_number = int(color_frame.get_frame_number())
            if frame_number != color_frame_number:
                color_rgb = np.asanyarray(color_frame.get_data()).copy()
                color_frame_number = frame_number
                color_timestamp_ms = float(color_frame.get_timestamp())
                color_timestamp_domain = timestamp_domain_name(color_frame.get_frame_timestamp_domain())
                color_capture_ns = host_ns
                color_intrinsics = intrinsics_from_frame(color_frame)
                color_meta = frame_metadata(color_frame)

        if depth_frame:
            frame_number = int(depth_frame.get_frame_number())
            if frame_number != depth_frame_number:
                depth_z16 = np.asanyarray(depth_frame.get_data()).copy()
                depth_frame_number = frame_number
                depth_timestamp_ms = float(depth_frame.get_timestamp())
                depth_timestamp_domain = timestamp_domain_name(depth_frame.get_frame_timestamp_domain())
                depth_capture_ns = host_ns
                depth_intrinsics = intrinsics_from_frame(depth_frame)
                depth_meta = frame_metadata(depth_frame)

        frame = L515CachedFrame(
            color_rgb=color_rgb,
            depth_z16=depth_z16,
            color_frame_number=color_frame_number,
            depth_frame_number=depth_frame_number,
            color_timestamp_ms=color_timestamp_ms,
            depth_timestamp_ms=depth_timestamp_ms,
            color_timestamp_domain=color_timestamp_domain,
            depth_timestamp_domain=depth_timestamp_domain,
            color_capture_ns=color_capture_ns,
            depth_capture_ns=depth_capture_ns,
            color_intrinsics=color_intrinsics,
            depth_intrinsics=depth_intrinsics,
            color_metadata=color_meta,
            depth_metadata=depth_meta,
            depth_scale_m_per_unit=float(self.depth_scale_m_per_unit),
        )
        strict = self.config.strict_global_time if require_global_time is None else require_global_time
        if strict and color_rgb is not None and depth_z16 is not None and not frame.is_global_time:
            raise RuntimeError(
                f"L515 {self.config.name} timestamp domain is not global_time: "
                f"color={frame.color_timestamp_domain}, depth={frame.depth_timestamp_domain}"
            )
        return frame

    def _run(self) -> None:
        miss_count = 0
        warning_interval = max(int(round(float(self.config.rgb_fps))), 1)
        while not self._stop.is_set():
            try:
                frame = self._read_once()
                with self._lock:
                    self._latest = frame
                self._frames_read += 1
                miss_count = 0
            except Exception as exc:
                miss_count += 1
                if miss_count % warning_interval == 0:
                    print(f"[Warn] L515 {self.config.name} cache read failed: {exc}")
