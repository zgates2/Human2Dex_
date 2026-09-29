# 文件名: marker_processor_node.py

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge # 重要：用于ROS图像和OpenCV图像的转换
import cv2
import numpy as np
import copy
import struct
from loguru import logger

# ================= 将所有计算相关的依赖和函数都搬到这里 =================
import sys
# 注意：这里的路径需要根据你实际的运行环境来确定
sys.path.append("/home/legion/collect/reactive_diffusion_policy")

from reactive_diffusion_policy.common.data_models import TactileSensorMessage, Arrow
from reactive_diffusion_policy.common.tactile_marker_utils import marker_normalization
from reactive_diffusion_policy.real_world.publisher.lib import find_marker
from reactive_diffusion_policy.real_world.publisher.gelsight_utility import GelsightUtility
# =====================================================================


class MarkerProcessorNode(Node):
    def __init__(self):
        super().__init__('marker_processor_node')
        self.get_logger().info('Marker Processor Node has been started.')

        # 与原发布器相同的参数，用于计算
        self.dimension = 2
        self.width = 640
        self.height = 480
        self.camera_name = 'left_gripper_camera_1'

        # 初始化图像转换工具
        self.bridge = CvBridge()

        # 初始化所有计算需要用到的对象
        self.GelsightHandler = GelsightUtility(RESCALE=1)
        self.m = find_marker.Matching(
            N_=self.GelsightHandler.N,
            M_=self.GelsightHandler.M,
            fps_=self.GelsightHandler.fps,
            x0_=self.GelsightHandler.x0,
            y0_=self.GelsightHandler.y0,
            dx_=self.GelsightHandler.dx,
            dy_=self.GelsightHandler.dy)
        self.initial_markers_3d = None
        self.vertical_scale = 0.05

        # 1. 创建订阅者，接收原始图像
        self.image_subscription = self.create_subscription(
            Image,
            f'/{self.camera_name}/color/image_raw',
            self.image_callback,
            10) # QoS profile can be adjusted for reliability

        # 2. 创建发布者，发布计算结果
        self.marker_publisher = self.create_publisher(
            PointCloud2,
            f'/{self.camera_name}/marker_offset/information',
            10)

    def image_callback(self, msg: Image):
        """当接收到图像消息时，此回调函数被触发"""
        try:
            # a. 将ROS Image消息转为OpenCV图像格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # b. 执行标记点检测和追踪 (这部分逻辑和原来完全一样)
            initial_markers, marker_motion = self.get_marker_image(cv_image)
            
            if initial_markers is None or marker_motion is None:
                self.get_logger().warn("Marker detection failed.")
                return

            # c. 标准化
            initial_markers_norm, marker_motion_norm = marker_normalization(
                copy.deepcopy(initial_markers), 
                copy.deepcopy(marker_motion),
                self.dimension,
                width=self.width, 
                height=self.height)

            # d. 获取时间戳，保持与输入图像一致
            camera_timestamp = Time.from_msg(msg.header.stamp)

            # e. 发布PointCloud2消息
            self.publish_marker_offset(initial_markers_norm, marker_motion_norm, camera_timestamp)
            self.get_logger().info(f"Processed frame and published markers at timestamp: {camera_timestamp.nanoseconds / 1e9}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}", exc_info=True)
    
    def get_marker_image(self, img):
        mask = self.GelsightHandler.find_marker(img)
        markers_detected = self.GelsightHandler.marker_center(mask)
        initial_markers, marker_motion = self.track_marker(markers_detected, self.dimension)
        return initial_markers, marker_motion

    def track_marker(self, marker_center, dimension):
        self.m.init(marker_center)
        self.m.run()
        flow = self.m.get_flow()
        Ox, Oy, Cx, Cy, _ = flow
        M, N = len(Ox), len(Ox[0])

        if self.initial_markers_3d is None and dimension == 3:
            self.initial_markers_3d = self.GelsightHandler.ComputesurroundingArea(Ox, Oy)
                    
        initial_marker = np.zeros((M * N, 3))
        marker_motion = np.zeros((M * N, 2))
        if dimension == 3:
            current_marker_3d = self.GelsightHandler.ComputesurroundingArea(Cx, Cy)

        k = 0
        for i in range(M):
            for j in range(N):
                if dimension == 2:
                    initial_marker[k] = [Ox[i][j], Oy[i][j], 0]
                elif dimension == 3:
                    initial_marker[k] = [Ox[i][j], Oy[i][j], max((current_marker_3d[i][j] - self.initial_markers_3d[i][j]) * self.vertical_scale, 0)]
                marker_motion[k] = [Cx[i][j] - Ox[i][j], Cy[i][j] - Oy[i][j]]
                k += 1
        return initial_marker, marker_motion

    def publish_marker_offset(self, marker_loc, marker_offset, camera_timestamp: Time):
        cur_marker = marker_loc[:, :2]
        marker_information = np.hstack((cur_marker, marker_offset)).astype(np.float32)
        msg = PointCloud2()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f'camera_marker_offset_{self.camera_name}'
        msg.is_bigendian = False
        msg.point_step = 16
        msg.is_dense = True
        msg.fields = [
            PointField(name='marker_location_x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_location_y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_x', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_y', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        pointcloud_data = b''.join(
            map(lambda row: struct.pack('ffff', row[0], row[1], row[2], row[3]), marker_information)
        )
        msg.data = pointcloud_data
        self.marker_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()