#!/usr/bin/env python3
"""
MVS Camera ROS2 Publisher (nanobind C++ backend).

Default mode: publishes /{camera_name}/rgb/image_raw/compressed (CompressedImage).
Debug mode (--show): publishes /{camera_name}/rgb/image_raw (Image, encoding rgb8) for rviz2.

This file is a drop-in replacement for mvs_ros2_publisher.py that uses the new
nanobind-based MVS C++ backend. The SDK lives next to this file under
publisher/mvs_cpp/. The compiled extension is cp310 + x86_64-linux-gnu and was
built against /opt/MVS. If the host or Python version changes, rebuild via:
    cd publisher/mvs_cpp/mvs_cpp && ./build_backend.sh
(set MVS_SDK_ROOT first if /opt/MVS is not the SDK root).
"""

import os

os.environ.setdefault("OMNIUMI_MVS_INPROCESS", "1")

import sys
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from loguru import logger

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mvs_cpp.mvs import (
    MVSImageCamera,
    MVSCamControllerConfig,
    get_all_mvs_dev_serial,
)

CAMERA_SERIAL = "DA6242414"
CAMERA_NAME = "fisheye_front"
CAMERA_FPS = 30.0
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480
JPEG_QUALITY = 95


class MVSCameraPublisher(Node):
    """ROS2 Node that publishes MVS camera frames using the new C++ SDK."""

    def __init__(self, camera_name: str, camera_type: str, device_path: str, show: bool = False):
        super().__init__("mvs_camera_publisher")

        self.camera_name = camera_name
        self.camera_type = camera_type
        self.device_path = device_path
        self.show = show

        self.frame_count = 0
        self.start_time: Optional[float] = None

        config = MVSCamControllerConfig(
            name=camera_name,
            serial=device_path,
            fps=int(CAMERA_FPS),
            put_desired_frequency=int(CAMERA_FPS),
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            transformed_width=CAMERA_WIDTH,
            transformed_height=CAMERA_HEIGHT,
            
            crop_x=180,
            crop_y=0,
            crop_width=1080,
            crop_height=1080,


            rotate_180=True,
            acquisition_frame_rate_enable=True,
            exposure_auto="off",
            # exposure_time_us=15000.0,
            exposure_time_us=10000.0,
            gain_auto="continuous",
            balance_white_auto="continuous",
            black_level_enable=True,
            black_level=240,
            brightness=40,
        )
        self.camera = MVSImageCamera(
            config,
            warmup_s=1.0,
            allow_repeat_frames=True,
            debug=False,
            backend="cpp",
        )

        frame_interval_ms = int(1000.0 / max(CAMERA_FPS, 1.0))
        self._read_timeout_ms = max(50, min(500, frame_interval_ms * 2))

        self.start()

        if self.show:
            self.publisher = self.create_publisher(
                Image,
                f"/{self.camera_name}/rgb/image_raw",
                10,
            )
            self.topic_name = f"/{self.camera_name}/rgb/image_raw"
        else:
            self.publisher = self.create_publisher(
                CompressedImage,
                f"/{self.camera_name}/rgb/image_raw/compressed",
                10,
            )
            self.topic_name = f"/{self.camera_name}/rgb/image_raw/compressed"

        self.timer = self.create_timer(1.0 / CAMERA_FPS, self._timer_callback)

        logger.info("MVS Camera Publisher (C++ backend) initialized")
        logger.info(f"  Mode: {'Debug (Image)' if self.show else 'Normal (CompressedImage)'}")
        logger.info(f"  Topic: {self.topic_name}")
        logger.info(f"  Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        logger.info(f"  FPS: {CAMERA_FPS}")
        if not self.show:
            logger.info(f"  JPEG Quality: {JPEG_QUALITY}")

    def start(self):
        logger.info("Starting MVS camera (C++ backend)...")

        try:
            os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
        except Exception:
            pass
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            SCHED_OTHER = 0

            class SchedParam(ctypes.Structure):
                _fields_ = [("sched_priority", ctypes.c_int)]

            param = SchedParam()
            param.sched_priority = 0
            libc.sched_setscheduler(0, SCHED_OTHER, ctypes.byref(param))
        except Exception:
            pass

        self.camera.connect(warmup=True)
        self.start_time = time.time()
        logger.info("Camera started, publishing images...")

    def stop(self):
        logger.info("Stopping camera...")
        try:
            self.camera.disconnect()
        except Exception as exc:
            logger.warning(f"Error disconnecting camera: {exc}")

        if self.start_time is not None and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            actual_fps = self.frame_count / max(elapsed, 1e-6)
            logger.info(f"Published {self.frame_count} frames in {elapsed:.1f}s ({actual_fps:.1f} fps)")

    def _timer_callback(self):
        try:
            image, _frame_id = self.camera.async_read(timeout_ms=self._read_timeout_ms)

            if self.show:
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"{self.camera_name}_color_frame"
                msg.height, msg.width, _ = image.shape
                msg.encoding = "rgb8"
                msg.step = msg.width * 3
                # rclpy in humble validates uint8[] data via two all() iterations,
                # which segfaults / mistypes on bytes / array.array at production
                # size (480x480x3 = 691200). list of int passes cleanly at the
                # cost of ~30 ms / frame; OK for debug mode.
                msg.data = image.flatten().tolist()
            else:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                success, encoded = cv2.imencode(".jpg", image_bgr, encode_param)
                if not success:
                    logger.warning("Failed to encode image")
                    return
                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"{self.camera_name}_color_frame"
                msg.format = "jpeg"
                msg.data = encoded.tobytes()

            self.publisher.publish(msg)
        except Exception as exc:
            logger.warning(f"Error in timer callback: {exc}")


def main():
    global CAMERA_SERIAL, CAMERA_NAME

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="MVS Camera ROS2 Publisher (C++ backend)")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Debug mode: publish Image (for rviz2). Default publishes CompressedImage.",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=CAMERA_SERIAL,
        help=f"Camera serial number (default: {CAMERA_SERIAL})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=CAMERA_NAME,
        help=f"Camera name / topic prefix (default: {CAMERA_NAME})",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List all available MVS cameras and exit",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Startup delay in seconds (avoids USB bandwidth contention with depth cameras)",
    )
    args = parser.parse_args()

    if args.list_cameras:
        try:
            available_serials = get_all_mvs_dev_serial()
            print(f"Found {len(available_serials)} MVS camera(s):")
            for i, serial in enumerate(available_serials):
                print(f"  [{i}] {serial}")
        except Exception as exc:
            print(f"Failed to enumerate cameras: {exc}")
        return

    CAMERA_SERIAL = args.serial
    CAMERA_NAME = args.name

    print("=" * 80)
    print(" " * 18 + "MVS Camera ROS2 Publisher (nanobind C++ backend)")
    print("=" * 80)
    print(f"Camera serial: {CAMERA_SERIAL}")
    print(f"Camera name:   {CAMERA_NAME}")
    print(f"Resolution:    {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"FPS:           {CAMERA_FPS}")
    print(f"Mode:          {'Debug (Image)' if args.show else 'Normal (CompressedImage)'}")
    if args.show:
        print(f"Topic:         /{CAMERA_NAME}/rgb/image_raw")
    else:
        print(f"Topic:         /{CAMERA_NAME}/rgb/image_raw/compressed")
        print(f"JPEG quality:  {JPEG_QUALITY}")
    print("=" * 80)

    try:
        available_serials = get_all_mvs_dev_serial()
        print(f"Available cameras: {available_serials}")
        if CAMERA_SERIAL not in available_serials:
            print(f"WARNING: target camera {CAMERA_SERIAL} not found!")
    except Exception as exc:
        print(f"Failed to enumerate cameras: {exc}")

    if args.delay > 0:
        logger.info(f"Sleeping {args.delay}s to stagger USB bandwidth contention...")
        time.sleep(args.delay)

    rclpy.init()

    publisher_node = MVSCameraPublisher(
        camera_name=CAMERA_NAME,
        camera_type="MVS",
        device_path=CAMERA_SERIAL,
        show=args.show,
    )

    try:
        rclpy.spin(publisher_node)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        publisher_node.stop()
        publisher_node.destroy_node()
        rclpy.shutdown()

    print("\n" + "=" * 80)
    print(" " * 32 + "done")
    print("=" * 80)


if __name__ == "__main__":
    main()
