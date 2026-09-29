#!/usr/bin/env python3
"""
Non-ROS Intel RealSense L515 reader with timestamp diagnostics.

The usable timestamp for synchronization is frame.get_timestamp() only after
global_time_enabled has taken effect and the frame timestamp domain reports
global_time. Raw hardware_clock timestamps are kept for diagnosis, but they are
not comparable across two cameras.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_RGB_RESOLUTION = (640, 480)
DEFAULT_DEPTH_RESOLUTION = (640, 480)
DEFAULT_RGB_FPS = 60
DEFAULT_DEPTH_FPS = 30


@dataclass
class L515Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: list[float]


@dataclass
class L515Frame:
    color_bgr: np.ndarray
    depth_z16: np.ndarray
    color_timestamp_ms: float
    depth_timestamp_ms: float
    color_timestamp_domain: str
    depth_timestamp_domain: str
    color_frame_number: int
    depth_frame_number: int
    host_arrival_monotonic_ns: int
    depth_scale_m_per_unit: float
    color_intrinsics: L515Intrinsics
    depth_intrinsics: L515Intrinsics
    color_metadata: dict[str, float]
    depth_metadata: dict[str, float]

    @property
    def color_timestamp_ns(self) -> int:
        return int(round(self.color_timestamp_ms * 1_000_000.0))

    @property
    def depth_timestamp_ns(self) -> int:
        return int(round(self.depth_timestamp_ms * 1_000_000.0))

    @property
    def is_global_time(self) -> bool:
        return (
            self.color_timestamp_domain == "global_time"
            and self.depth_timestamp_domain == "global_time"
        )


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be WIDTHxHEIGHT")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


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


def set_l515_timestamp_options(device: rs.device) -> None:
    for sensor in device.query_sensors():
        set_option_if_supported(
            sensor,
            rs.option.global_time_enabled,
            1.0,
            "global_time_enabled",
        )
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
                0.0,
                "auto_exposure_priority",
            )
        set_option_if_supported(
            sensor,
            rs.option.enable_auto_white_balance,
            1.0,
            "enable_auto_white_balance",
        )


def list_devices() -> list[dict[str, str]]:
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
    devices = list_devices()
    if requested_serial:
        for dev in devices:
            if dev["serial"] == requested_serial:
                return requested_serial
        summary = fallback_rs_enumerate_summary()
        raise RuntimeError(
            f"pyrealsense2 cannot see serial {requested_serial}. "
            f"rs-enumerate-devices output:\n{summary}"
        )
    l500 = [dev for dev in devices if dev["product_line"] == "L500"]
    if len(l500) == 1:
        return l500[0]["serial"]
    if len(l500) > 1:
        serials = ", ".join(dev["serial"] for dev in l500)
        raise RuntimeError(f"Multiple L515/L500 cameras found, pass --serial. Found: {serials}")
    summary = fallback_rs_enumerate_summary()
    raise RuntimeError(
        "pyrealsense2 did not enumerate any L515/L500 camera. "
        "Use a Python binding built against the same librealsense stack that "
        "rs-enumerate-devices uses.\n"
        f"rs-enumerate-devices output:\n{summary}"
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


class L515TimestampCamera:
    def __init__(
        self,
        serial: Optional[str] = None,
        rgb_resolution: tuple[int, int] = DEFAULT_RGB_RESOLUTION,
        depth_resolution: tuple[int, int] = DEFAULT_DEPTH_RESOLUTION,
        rgb_fps: int = DEFAULT_RGB_FPS,
        depth_fps: int = DEFAULT_DEPTH_FPS,
        align_to: str = "color",
        timeout_ms: int = 5000,
        strict_global_time: bool = True,
    ) -> None:
        self.serial = find_l515_serial(serial)
        self.rgb_resolution = rgb_resolution
        self.depth_resolution = depth_resolution
        self.rgb_fps = int(rgb_fps)
        self.depth_fps = int(depth_fps)
        self.align_to = align_to
        self.timeout_ms = int(timeout_ms)
        self.strict_global_time = bool(strict_global_time)
        self.pipeline = rs.pipeline()
        self.align = self._make_align(align_to)
        self.profile = None
        self.depth_scale_m_per_unit = math.nan

    def _make_align(self, align_to: str):
        if align_to == "color":
            return rs.align(rs.stream.color)
        if align_to == "depth":
            return rs.align(rs.stream.depth)
        if align_to == "none":
            return None
        raise ValueError(f"Unsupported align_to: {align_to}")

    def start(self, warmup_frames: int = 90) -> None:
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color,
            self.rgb_resolution[0],
            self.rgb_resolution[1],
            rs.format.bgr8,
            self.rgb_fps,
        )
        config.enable_stream(
            rs.stream.depth,
            self.depth_resolution[0],
            self.depth_resolution[1],
            rs.format.z16,
            self.depth_fps,
        )

        wrapper = rs.pipeline_wrapper(self.pipeline)
        resolved = config.resolve(wrapper)
        set_l515_timestamp_options(resolved.get_device())
        self.profile = self.pipeline.start(config)
        device = self.profile.get_device()
        set_l515_timestamp_options(device)
        self.depth_scale_m_per_unit = float(device.first_depth_sensor().get_depth_scale())
        print(
            f"[Init] L515 serial={self.serial} depth_scale={self.depth_scale_m_per_unit} "
            f"rgb={self.rgb_resolution}@{self.rgb_fps} "
            f"depth={self.depth_resolution}@{self.depth_fps}"
        )
        self.wait_for_global_time_ready(warmup_frames)

    def stop(self) -> None:
        self.pipeline.stop()

    def wait_for_global_time_ready(self, max_frames: int) -> None:
        last_domains = None
        ready_at = None
        for idx in range(max(0, int(max_frames))):
            frame = self.read(require_global_time=False)
            last_domains = (frame.color_timestamp_domain, frame.depth_timestamp_domain)
            if frame.is_global_time and ready_at is None:
                ready_at = idx + 1
        if ready_at is not None:
            print(
                f"[Init] global_time ready after {ready_at} frames; "
                f"dropped {max_frames} warmup frames"
            )
            return
        if self.strict_global_time:
            raise RuntimeError(
                "global_time_enabled did not produce global_time frame timestamps. "
                f"Last domains={last_domains}. Raw hardware_clock timestamps are not "
                "safe for multi-camera synchronization."
            )
        print(f"[Warn] global_time not ready after warmup; last domains={last_domains}")

    def read(self, require_global_time: Optional[bool] = None) -> L515Frame:
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        host_arrival_ns = time.monotonic_ns()
        if self.align is not None:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("missing color or depth frame")

        frame = L515Frame(
            color_bgr=np.asanyarray(color_frame.get_data()).copy(),
            depth_z16=np.asanyarray(depth_frame.get_data()).copy(),
            color_timestamp_ms=float(color_frame.get_timestamp()),
            depth_timestamp_ms=float(depth_frame.get_timestamp()),
            color_timestamp_domain=timestamp_domain_name(color_frame.get_frame_timestamp_domain()),
            depth_timestamp_domain=timestamp_domain_name(depth_frame.get_frame_timestamp_domain()),
            color_frame_number=int(color_frame.get_frame_number()),
            depth_frame_number=int(depth_frame.get_frame_number()),
            host_arrival_monotonic_ns=host_arrival_ns,
            depth_scale_m_per_unit=float(self.depth_scale_m_per_unit),
            color_intrinsics=intrinsics_from_frame(color_frame),
            depth_intrinsics=intrinsics_from_frame(depth_frame),
            color_metadata=frame_metadata(color_frame),
            depth_metadata=frame_metadata(depth_frame),
        )
        strict = self.strict_global_time if require_global_time is None else require_global_time
        if strict and not frame.is_global_time:
            raise RuntimeError(
                "Frame timestamp domain is not global_time: "
                f"color={frame.color_timestamp_domain}, depth={frame.depth_timestamp_domain}"
            )
        return frame


def save_frame(frame: L515Frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "color_timestamp_ms": frame.color_timestamp_ms,
        "depth_timestamp_ms": frame.depth_timestamp_ms,
        "color_timestamp_ns": frame.color_timestamp_ns,
        "depth_timestamp_ns": frame.depth_timestamp_ns,
        "color_timestamp_domain": frame.color_timestamp_domain,
        "depth_timestamp_domain": frame.depth_timestamp_domain,
        "color_frame_number": frame.color_frame_number,
        "depth_frame_number": frame.depth_frame_number,
        "host_arrival_monotonic_ns": frame.host_arrival_monotonic_ns,
        "depth_scale_m_per_unit": frame.depth_scale_m_per_unit,
        "color_intrinsics": asdict(frame.color_intrinsics),
        "depth_intrinsics": asdict(frame.depth_intrinsics),
        "color_metadata": frame.color_metadata,
        "depth_metadata": frame.depth_metadata,
    }
    np.savez_compressed(
        path,
        color_bgr=frame.color_bgr,
        depth_z16=frame.depth_z16,
        metadata=json.dumps(metadata, sort_keys=True),
    )


def print_frame_summary(frame: L515Frame, idx: int) -> None:
    print(
        f"[{idx:06d}] color_ts_ms={frame.color_timestamp_ms:.3f} "
        f"depth_ts_ms={frame.depth_timestamp_ms:.3f} "
        f"delta_ms={frame.color_timestamp_ms - frame.depth_timestamp_ms:.3f} "
        f"domains=({frame.color_timestamp_domain},{frame.depth_timestamp_domain}) "
        f"frames=({frame.color_frame_number},{frame.depth_frame_number}) "
        f"host_ns={frame.host_arrival_monotonic_ns}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe L515 global/hardware timestamps without ROS.")
    parser.add_argument("--serial", default=None, help="L515 serial number. Required when more than one L515 is present.")
    parser.add_argument("--rgb-resolution", type=parse_resolution, default=DEFAULT_RGB_RESOLUTION)
    parser.add_argument("--depth-resolution", type=parse_resolution, default=DEFAULT_DEPTH_RESOLUTION)
    parser.add_argument("--rgb-fps", type=int, default=DEFAULT_RGB_FPS)
    parser.add_argument("--depth-fps", type=int, default=DEFAULT_DEPTH_FPS)
    parser.add_argument("--fps", type=int, default=None, help="Legacy shortcut: set both RGB and depth FPS.")
    parser.add_argument("--align-to", choices=("color", "depth", "none"), default="color")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--allow-non-global-time",
        action="store_true",
        help="Do not fail when timestamps remain in hardware_clock/system_time domain.",
    )
    parser.add_argument("--list-devices", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.list_devices:
        devices = list_devices()
        if devices:
            print(json.dumps(devices, indent=2, sort_keys=True))
        else:
            print("pyrealsense2 devices: []")
            summary = fallback_rs_enumerate_summary()
            if summary:
                print("rs-enumerate-devices:")
                print(summary)
        return

    camera = L515TimestampCamera(
        serial=args.serial,
        rgb_resolution=args.rgb_resolution,
        depth_resolution=args.depth_resolution,
        rgb_fps=args.fps if args.fps is not None else args.rgb_fps,
        depth_fps=args.fps if args.fps is not None else args.depth_fps,
        align_to=args.align_to,
        timeout_ms=args.timeout_ms,
        strict_global_time=not args.allow_non_global_time,
    )
    camera.start(warmup_frames=args.warmup_frames)
    try:
        for idx in range(args.frames):
            frame = camera.read()
            if args.print_every > 0 and idx % args.print_every == 0:
                print_frame_summary(frame, idx)
            if args.save_dir is not None and args.save_every > 0 and idx % args.save_every == 0:
                save_frame(frame, args.save_dir / f"l515_frame_{idx:06d}.npz")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
