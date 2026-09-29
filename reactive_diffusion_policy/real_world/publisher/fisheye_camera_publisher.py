#!/usr/bin/env python3
import rclpy
import time
import subprocess
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
from loguru import logger
import copy
import numpy as np

class FisheyeCameraPublisher(Node):
    def __init__(self,
                 camera_name: str,
                 camera_type: str,
                 device_path: str,
                 resolution: tuple = (640, 480),
                 fps: float = 30.0,
                 jpeg_quality: int = 95,
                 debug: bool = False,
                 auto_exposure: bool = True,
                 exposure_time_absolute: int = None,
                 brightness: int = None,
                 contrast: int = None,
                 saturation: int = None,
                 hue: int = None,
                 gain: int = None,
                 gamma: int = None,
                 sharpness: int = None,
                 white_balance_automatic: bool = True,
                 white_balance_temperature: int = None,
                 backlight_compensation: int = None):

        node_name = f'{camera_name}_publisher'
        super().__init__(node_name)

        self.camera_name = camera_name
        self.device_path = device_path
        self.resolution = resolution
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.debug = debug

        self.controls = {
            'auto_exposure': auto_exposure,
            'exposure_time_absolute': exposure_time_absolute,
            'brightness': brightness,
            'contrast': contrast,
            'saturation': saturation,
            'hue': hue,
            'gain': gain,
            'gamma': gamma,
            'sharpness': sharpness,
            'white_balance_automatic': white_balance_automatic,
            'white_balance_temperature': white_balance_temperature,
            'backlight_compensation': backlight_compensation
        }

        self.cap = None
        self.start()

        self.publisher_ = self.create_publisher(
            CompressedImage,
            f'/{self.camera_name}/rgb/image_raw/compressed',
            10
        )
        # self.publisher_ = self.create_publisher(
        #     Image,
        #     f'/{self.camera_name}/rgb/image_raw/compressed',
        #     10
        # )
        self.timer = self.create_timer(1.0 / self.fps, self.capture_and_publish_once)

        self.get_logger().info(f"摄像头 '{self.camera_name}' 在 '{self.device_path}' 上已打开，"
                               f"将以 {self.fps} Hz 的频率发布图像。")

    def start(self):
        """打开摄像头并应用设置。"""
        self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开摄像头: {self.device_path}")
            raise RuntimeError(f"无法打开摄像头: {self.device_path}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.apply_cv2_settings()

    def apply_cv2_settings(self):
        """使用 OpenCV 的 cap.set() 方法来设置相机参数。"""
        param_map = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'hue': cv2.CAP_PROP_HUE,
            'gain': cv2.CAP_PROP_GAIN,
            'gamma': cv2.CAP_PROP_GAMMA,
            'sharpness': cv2.CAP_PROP_SHARPNESS,
            'exposure_time_absolute': cv2.CAP_PROP_EXPOSURE,
            'white_balance_temperature': cv2.CAP_PROP_WB_TEMPERATURE,
            'backlight_compensation': cv2.CAP_PROP_BACKLIGHT
        }

        if self.controls['auto_exposure']:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.get_logger().info(f"成功设置 '{self.camera_name}': 自动曝光模式")
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.get_logger().info(f"成功设置 '{self.camera_name}': 手动曝光模式")
            if self.controls['exposure_time_absolute'] is not None:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.controls['exposure_time_absolute'])
                self.get_logger().info(f"成功设置 '{self.camera_name}': 曝光时间={self.controls['exposure_time_absolute']}")

        if self.controls['white_balance_automatic']:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.get_logger().info(f"成功设置 '{self.camera_name}': 自动白平衡")
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.get_logger().info(f"成功设置 '{self.camera_name}': 手动白平衡")
            if self.controls['white_balance_temperature'] is not None:
                self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, self.controls['white_balance_temperature'])
                self.get_logger().info(f"成功设置 '{self.camera_name}': 白平衡色温={self.controls['white_balance_temperature']}")

        handled_params = {'exposure_time_absolute', 'white_balance_temperature'}
        for param_name, value in self.controls.items():
            if value is not None and param_name in param_map and param_name not in handled_params:
                prop_id = param_map[param_name]
                if self.cap.set(prop_id, value):
                    self.get_logger().info(f"成功设置 '{self.camera_name}': {param_name}={value}")
                elif self.debug:
                    self.get_logger().warn(f"无法设置 '{self.camera_name}': {param_name}={value}")

    def capture_and_publish_once(self):
        """捕获一帧图像，压缩并发布。"""
        if not self.cap or not self.cap.isOpened():
            self.get_logger().warn("摄像头未准备好或已关闭。")
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.debug:
                self.get_logger().warn("未读取到图像帧")
            return

        # color_image = copy.deepcopy(np.asanyarray(frame))
        # success, encoded_image = cv2.imencode('.jpg', color_image)
        # success = False

        # msg = Image()
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = "fisheye_camera_color_frame"
        # msg.height, msg.width, _ = color_image.shape
        # msg.encoding = "bgr8"
        # msg.step = msg.width * 3
        # if success:
        #     image_bytes = encoded_image.tobytes()
        #     msg.data = image_bytes
        # else:
        #     logger.debug('fail to image encoding!')
        #     msg.data = color_image.tobytes()
        
        # # msg = numpy_to_image(color_image, "bgr8")
        # self.publisher_.publish(msg)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        result, compressed_frame = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            self.get_logger().warn("JPEG 编码失败。")
            return

        # self.get_logger().info(f"Type of compressed_frame: {type(compressed_frame)}")

        # if isinstance(compressed_frame, np.ndarray):
        #     self.get_logger().info(f"NumPy array dtype: {compressed_frame.dtype}, shape: {compressed_frame.shape}")
        
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_name
        msg.format = "jpeg"
        # msg.data = compressed_frame.tobytes()
        msg.data = compressed_frame.flatten().tolist()
        # msg.data = bytes(compressed_frame)
        self.publisher_.publish(msg)

    def stop(self):
        """释放摄像头资源。"""
        self.get_logger().info(f"正在关闭摄像头 '{self.camera_name}'...")
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info(f"摄像头 '{self.camera_name}' 已成功关闭。")

    def destroy_node(self):
        """在销毁节点前调用 stop 方法。"""
        self.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        # node = FisheyeCameraPublisher(
        #     camera_name='fisheye_test_manual',
        #     device_path='/dev/v4l/by-id/usb-icSpring_icspring_camera_20240307110607-video-index0',
        #     camera_type='Fisheye',
        #     fps=14.0,
        #     debug=True,
        #     auto_exposure=False,
        #     exposure_time_absolute=500,
        #     white_balance_automatic=False,
        #     white_balance_temperature=4600,
        #     gain=32,
        #     brightness=10
        # ) # 手动设置曝光
        
        node = FisheyeCameraPublisher(
            camera_name='fisheye_auto_exposure_test',
            device_path='/dev/v4l/by-id/usb-icSpring_icspring_camera_20240307110607-video-index0',
            camera_type='Fisheye',
            fps=14.0,
            debug=True
        ) # 自动曝光

        # node = FisheyeCameraPublisher(
        #     camera_name='fisheye_auto_exposure_test',
        #     device_path='/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0',
        #     camera_type='Fisheye',
        #     fps=60.0,
        #     debug=True
        # ) # 自动曝光
        
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError) as e:
        logger.error(f"节点运行出错: {e}")
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
