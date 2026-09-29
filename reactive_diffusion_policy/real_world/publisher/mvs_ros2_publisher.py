#!/usr/bin/env python3
"""
MVS Camera ROS2 Publisher - Using mvs_utils

默认发布: /{camera_name}/rgb/image_raw/compressed (CompressedImage)
调试模式: /{camera_name}/rgb/image_raw (Image) - 用于 rviz2 显示
"""

import sys
sys.path.append("/home/legion/reactive_diffusion_policy/reactive_diffusion_policy/real_world/publisher")
import time
import argparse
from typing import Optional

import numpy as np
import cv2
from loguru import logger

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage

MVS_SDK_PATH = "/opt/MVS/Samples/64/Python/MvImport"
sys.path.append(MVS_SDK_PATH)

# Import from mvs_utils
from mvs_utils.mvs_cam import MVSCamControllerConfig, get_all_mvs_dev_serial
from mvs_utils.multi_mvs_cam import MultiMVSCamera

# Camera configuration
CAMERA_SERIAL = "DA6242414"
# CAMERA_SERIAL = "DA9057802"
CAMERA_NAME = "fisheye_front"
CAMERA_FPS = 30.0
# CAMERA_FPS = 60.0
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 480
# CAMERA_WIDTH = 320
# CAMERA_HEIGHT = 240
JPEG_QUALITY = 95


def _identity_transform(img):
    return img


class MVSImageCameraROS:
    """
    MVS camera wrapper that uses mvs_utils.MultiMVSCamera
    """
    def __init__(
        self,
        serial: str,
        name: str = "mvs_camera",
        fps: int = 30,
        width: int = 480,
        height: int = 480,
        warmup_s: float = 1.0,
        rotate_180: bool = True,
    ):
        self.serial = serial
        self.camera_name = name
        self.fps = fps
        self.width = width
        self.height = height
        self.warmup_s = warmup_s
        self.rotate_180 = rotate_180

        self.camera: Optional[MultiMVSCamera] = None
        self.cam_idx = 0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.camera is not None

    def _create_config(self) -> MVSCamControllerConfig:
        """Create camera configuration"""
        return MVSCamControllerConfig(
            receive_latency=0.125,
            serial=self.serial,
            name=self.camera_name,
            fps=self.fps,
            put_desired_frequency=self.fps,
            # img_transform_func=lambda img: img,
            img_transform_func=_identity_transform,
            width=self.width,
            height=self.height,
            transformed_width=self.width,
            transformed_height=self.height,
            crop_func=_identity_transform
            # crop_func=lambda img: img
        )

    def connect(self, warmup: bool = True):
        """Connect to camera"""
        if self.is_connected:
            raise RuntimeError(f"{self.camera_name} is already connected")

        logger.info(f"Connecting {self.camera_name} (S/N: {self.serial})...")

        try:
            config = self._create_config()
            self.camera = MultiMVSCamera([config])

            logger.info(f"Starting {self.camera_name}...")
            self.camera.start(wait=True)

            if warmup:
                logger.info(f"Warming up {self.camera_name} for {self.warmup_s}s...")
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    self._read_frame_internal()
                    time.sleep(0.1)

            self._connected = True
            logger.info(f"{self.camera_name} connected and ready")

        except Exception as e:
            logger.error(f"Failed to connect {self.camera_name}: {e}")
            self.camera = None
            raise

    def disconnect(self):
        """Disconnect camera"""
        if not self._connected and self.camera is None:
            return

        logger.info(f"Disconnecting {self.camera_name}...")

        if self.camera is not None:
            try:
                self.camera.stop(wait=True)
            except Exception as e:
                logger.warning(f"{self.camera_name}: Error during stop: {e}")
            self.camera = None

        self._connected = False
        logger.info(f"{self.camera_name} disconnected")

    def _read_frame_internal(self) -> Optional[np.ndarray]:
        """Internal frame read from MultiMVSCamera"""
        if self.camera is None:
            return None

        camera_data = self.camera.get(k=None)

        if not isinstance(camera_data, dict) or self.cam_idx not in camera_data:
            return None

        img_data = camera_data[self.cam_idx]['color']

        # Handle dimension variations
        if len(img_data.shape) == 4:
            image = img_data[-1].copy()
        elif len(img_data.shape) == 3:
            image = img_data.copy()
        else:
            return None

        if self.rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)

        return image

    def read(self, timeout_ms: float = 500) -> np.ndarray:
        """Read a frame from the camera"""
        if not self.is_connected:
            raise RuntimeError(f"{self.camera_name} is not connected")

        image = self._read_frame_internal()
        if image is None:
            raise RuntimeError(f"{self.camera_name}: Failed to read frame")

        return image

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def __del__(self):
        try:
            self.disconnect()
        except:
            pass


class MVSCameraPublisher(Node):
    """ROS2 Node that publishes MVS camera images"""

    def __init__(self, camera_name: str, camera_type: str, device_path: str, show: bool = False):
        super().__init__('mvs_camera_publisher')

        self.camera_name = camera_name
        self.camera_type = camera_type
        self.device_path = device_path
        self.show = show  # True: 发布 Image (用于 rviz2 调试), False: 发布 CompressedImage

        self.frame_count = 0
        self.start_time = None

        # Create camera using mvs_utils
        self.camera = MVSImageCameraROS(
            serial=self.device_path,
            name=camera_name,
            fps=int(CAMERA_FPS),
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            rotate_180=True,
        )

        self.start()

        # 根据 show 参数选择发布类型
        if self.show:
            # 调试模式: 发布 Image (rviz2 可以直接显示)
            self.publisher = self.create_publisher(
                Image,
                f'/{self.camera_name}/rgb/image_raw',
                10
            )
            self.topic_name = f'/{self.camera_name}/rgb/image_raw'
        else:
            # 默认模式: 发布 CompressedImage
            self.publisher = self.create_publisher(
                CompressedImage,
                f'/{self.camera_name}/rgb/image_raw/compressed',
                10
            )
            self.topic_name = f'/{self.camera_name}/rgb/image_raw/compressed'

        # Create timer for publishing
        self.timer = self.create_timer(
            1.0 / CAMERA_FPS,
            self._timer_callback
        )

        logger.info(f"MVS Camera Publisher initialized")
        logger.info(f"  Mode: {'Debug (Image)' if self.show else 'Normal (CompressedImage)'}")
        logger.info(f"  Topic: {self.topic_name}")
        logger.info(f"  Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        logger.info(f"  FPS: {CAMERA_FPS}")
        if not self.show:
            logger.info(f"  JPEG Quality: {JPEG_QUALITY}")

    def start(self):
        logger.info("Starting camera...")
        
        # Reset CPU affinity so MVS SDK background threads are not starved on a single core
        try:
            import os
            os.sched_setaffinity(0, set(range(os.cpu_count())))
        except Exception:
            pass

        # Reset real-time priority so SDK threads are not preempted completely
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
        self.camera.disconnect()

        if self.start_time and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            actual_fps = self.frame_count / elapsed
            logger.info(f"Published {self.frame_count} frames in {elapsed:.1f}s ({actual_fps:.1f} fps)")

    def _timer_callback(self):
        try:
            image = self.camera.read(timeout_ms=100)

            if self.show:
                # 调试模式: 发布 Image
                msg = Image()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"{self.camera_name}_color_frame"
                msg.height, msg.width, _ = image.shape
                msg.encoding = "rgb8"
                msg.step = msg.width * 3
                msg.data = image.tobytes()
            else:
                # 默认模式: 发布 CompressedImage
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                success, encoded = cv2.imencode('.jpg', image_bgr, encode_param)

                if not success:
                    logger.warning("Failed to encode image")
                    return

                msg = CompressedImage()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = f"{self.camera_name}_color_frame"
                msg.format = 'jpeg'
                msg.data = encoded.flatten().tolist()

            self.publisher.publish(msg)
            # self.frame_count += 1

            # if self.frame_count % 100 == 0:
            #     elapsed = time.time() - self.start_time
            #     actual_fps = self.frame_count / elapsed
            #     logger.info(f"Published {self.frame_count} frames ({actual_fps:.1f} fps)")

        except Exception as e:
            logger.warning(f"Error in timer callback: {e}")


def main():
    global CAMERA_SERIAL, CAMERA_NAME

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(description="MVS Camera ROS2 Publisher")
    parser.add_argument(
        "--show",
        action="store_true",
        help="调试模式: 发布 Image 消息 (用于 rviz2 显示), 默认发布 CompressedImage"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=CAMERA_SERIAL,
        help=f"相机序列号 (默认: {CAMERA_SERIAL})"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=CAMERA_NAME,
        help=f"相机名称/话题前缀 (默认: {CAMERA_NAME})"
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="列出所有可用的 MVS 相机并退出"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="延迟启动时间(秒)，用于避免多相机同时启动抢占 USB 带宽"
    )
    args = parser.parse_args()

    # 列出相机
    if args.list_cameras:
        try:
            available_serials = get_all_mvs_dev_serial()
            print(f"找到 {len(available_serials)} 个 MVS 相机:")
            for i, serial in enumerate(available_serials):
                print(f"  [{i}] {serial}")
        except Exception as e:
            print(f"枚举相机失败: {e}")
        return

    # 更新配置
    CAMERA_SERIAL = args.serial
    CAMERA_NAME = args.name

    print("=" * 80)
    print(" " * 20 + "MVS Camera ROS2 Publisher (mvs_utils)")
    print("=" * 80)
    print(f"相机序列号: {CAMERA_SERIAL}")
    print(f"相机名称: {CAMERA_NAME}")
    print(f"分辨率: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"帧率: {CAMERA_FPS}")
    print(f"模式: {'调试 (Image)' if args.show else '正常 (CompressedImage)'}")
    if args.show:
        print(f"话题: /{CAMERA_NAME}/rgb/image_raw")
    else:
        print(f"话题: /{CAMERA_NAME}/rgb/image_raw/compressed")
        print(f"JPEG 质量: {JPEG_QUALITY}")
    print("=" * 80)

    # 检查相机是否存在
    try:
        available_serials = get_all_mvs_dev_serial()
        print(f"可用相机: {available_serials}")
        if CAMERA_SERIAL not in available_serials:
            print(f"警告: 目标相机 {CAMERA_SERIAL} 未找到!")
    except Exception as e:
        print(f"枚举相机失败: {e}")

    if args.delay > 0:
        logger.info(f"等待 {args.delay} 秒以错开深度相机 USB 带宽抢占阶段...")
        time.sleep(args.delay)

    rclpy.init()

    publisher_node = MVSCameraPublisher(
        camera_name=CAMERA_NAME,
        camera_type="MVS",
        device_path=CAMERA_SERIAL,
        show=args.show
    )

    try:
        rclpy.spin(publisher_node)

    except KeyboardInterrupt:
        logger.info("用户中断")

    finally:
        publisher_node.stop()
        publisher_node.destroy_node()
        rclpy.shutdown()

    print("\n" + "=" * 80)
    print(" " * 30 + "完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
