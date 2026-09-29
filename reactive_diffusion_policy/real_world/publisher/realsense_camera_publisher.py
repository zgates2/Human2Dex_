import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import pyrealsense2 as rs
import numpy as np
import copy
import cv2
from loguru import logger
import sys
sys.path.append("/home/ps/reactive_diffusion_policy")
from reactive_diffusion_policy.common.time_utils import convert_float_to_ros_time
import uuid
import time
import socket
import math
import bson
import zstandard as zstd
from pyinstrument import Profiler
from loguru import logger


class RealsenseCameraPublisher(Node):
    """
    Realsense Camera publisher class
    """
    def __init__(self,
                #  camera_serial_number: str = '036422060422', # rdp realsense 435 wrist camera
                 camera_serial_number: str = '218622273046', #  realsense 405
                #  camera_serial_number: str = 'f1422067', # realsense 515
                #  camera_type: str = 'L500',  # L500
                 camera_type: str = 'D400', 
                #  camera_type: str = 'D455',
                 camera_name: str = 'camera_base',
                 rgb_resolution: tuple = (640, 480),
                 exposure: int = 120,
                 white_balance: int = 5900,  # 2800-6500
                 depth_resolution: tuple = (640, 480),
                 fps: int = 30,
                 decimate: int = 2,  # (0-4) decimation_filter magnitude for point cloud
                 random_sample_point_num: int = 10000,
                 enable_streaming: bool = False,
                 streaming_server_ip: str = '127.0.0.1',
                 streaming_server_port: int = 10004,
                 streaming_quality: int = 10,
                 streaming_chunk_size: int = 1024,
                 streaming_display_params_list: list = None,
                 debug: bool = False
                 ):
        node_name = f'{camera_name}_publisher'
        super().__init__(node_name)
        self.camera_serial_number = camera_serial_number
        self.camera_type = camera_type
        self.camera_name = camera_name
        self.fps = fps
        self.rgb_resolution = rgb_resolution
        self.exposure = exposure
        self.white_balance = white_balance
        self.depth_resolution = depth_resolution
        self.random_sample_point_num = random_sample_point_num

        # streaming configuration
        self.enable_streaming = enable_streaming
        if self.enable_streaming:
            self.id = uuid.uuid4()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.streaming_server_ip = streaming_server_ip
            self.streaming_server_port = streaming_server_port
            self.streaming_quality = streaming_quality
            self.streaming_chunk_size = streaming_chunk_size
            streaming_display_params_list = [{k:list(v) for k, v in d.items()} for d in streaming_display_params_list]
            self.streaming_display_params_list = streaming_display_params_list

        self.debug = debug

        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        
        # self.depth_publisher_ = self.create_publisher(Image, f'/{camera_name}/depth/image_raw', 10)
        self.depth_publisher_ = self.create_publisher(CompressedImage, f'/{camera_name}/depth/image_raw', 10)
        # self.pointcloud_publisher_ = self.create_publisher(PointCloud2, f'/{camera_name}/depth/points', 10)

        # self.color_publisher_ = self.create_publisher(CompressedImage, f'/{camera_name}/color/image_raw', 10)
        self.timer = self.create_timer(1 / fps, self.timer_callback)
        self.pipeline = None
        self.timestamp_offset = None
        self.depth_scale = None
        
        self.color_sensor = None
        
        self.pc = rs.pointcloud()
        # self.zstd_compressor = zstd.ZstdCompressor(level=3)

        self.prev_time = time.time()
        self.last_print_time = time.time()  # Add a variable to keep track of the last print time
        self.frame_count = 0
        self.fps_list = []
        self.frame_intervals = []
        self.last_frame_time = None

        # Create a decimation filter
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2 ** decimate)

        # Start the camera
        self.start()
        
        # self.profiler = Profiler()

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        
        if not self.color_sensor:
            logger.warning("无法设置曝光：未找到独立的颜色传感器")
            return
            
        if exposure is None and gain is None:
            # auto exposure
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.color_sensor.set_option(rs.option.exposure, exposure)
            if gain is not None:
                self.color_sensor.set_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if not self.color_sensor:
            logger.warning("无法设置白平衡：未找到独立的颜色传感器")
            return
        
        if white_balance is None:
            self.color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
            self.color_sensor.set_option(rs.option.white_balance, white_balance)

    def start(self):
        # get the context of the connected devices
        context = rs.context()
        devices = context.query_devices()

        # check if there are connected devices
        if len(devices) == 0:
            logger.error("No connected devices found")
            raise Exception("No connected devices found")

        config = rs.config()
        is_camera_valid = False
        for device in devices:
            # check if the device serial number matches the provided serial number
            serial_number = device.get_info(rs.camera_info.serial_number)
            if serial_number == self.camera_serial_number:
                is_camera_valid = True
                break

        # if the provided camera is not found, raise an exception
        if not is_camera_valid:
            logger.error("Camera with serial number {} not found".format(self.camera_serial_number))
            raise Exception("Camera with serial number {} not found".format(self.camera_serial_number))

        # Start the camera
        config.enable_device(self.camera_serial_number)
        self.pipeline = rs.pipeline()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        assert device_product_line == self.camera_type, f'With {self.camera_name}, Camera type does not match the camera product line.'
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        logger.info(f"Depth Scale is: {self.depth_scale}")

        self.depth_sensor.set_option(rs.option.global_time_enabled, 1.0)
        # report global time
        # https://github.com/IntelRealSense/librealsense/pull/3909
        # self.color_sensor = device.first_color_sensor()
        # self.color_sensor.set_option(rs.option.global_time_enabled, 1)
        # # realsense exposure
        # self.set_exposure(exposure=self.exposure, gain=0)
        # # realsense white balance
        # self.set_white_balance(white_balance=self.white_balance)
        
        try:
            self.color_sensor = device.first_color_sensor()
            if self.color_sensor:
                self.color_sensor.set_option(rs.option.global_time_enabled, 1.0)
                # self.set_exposure(exposure=self.exposure, gain=0)
                # self.set_white_balance(white_balance=self.white_balance)
                logger.info("已成功配置独立的颜色传感器。")
            else:
                logger.warning("未找到独立的颜色传感器，将使用默认的相机设置。")
        except Exception as e:
            logger.warning(f"无法获取或配置独立的颜色传感器（这在D405上是正常现象）。错误: {e}")
            self.color_sensor = None

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # set the resolution and format of the camera
        config.enable_stream(rs.stream.color, self.rgb_resolution[0], self.rgb_resolution[1], rs.format.bgr8, self.fps)
        # config.enable_stream(rs.stream.depth, self.depth_resolution[0], self.depth_resolution[1], rs.format.z16, self.fps)  # 启动depth数据流
        config.enable_stream(rs.stream.depth, self.depth_resolution[0], self.depth_resolution[1], rs.format.z16, 30)
        # self.pipeline.start(config)
        profile = self.pipeline.start(config)
        logger.debug("Camera started!")
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1.0)

        # capture some frames for the camera to stabilize
        logger.debug("Capturing some frames for the camera to stabilize...")
        for _ in range(self.fps):
            self.pipeline.wait_for_frames()

        # Capture initial frames to get initial timestamps
        frames = self.pipeline.wait_for_frames()

        initial_frame = frames.get_color_frame()
        if not initial_frame:
            logger.error("Failed to get initial frame")
            raise ValueError("Failed to get initial frame")

        # convert the camera timestamp to system timestamp
        initial_camera_timestamp = convert_float_to_ros_time(initial_frame.get_timestamp() / 1000)  # convert to time class in ROS
        # we assume that the internal clock of realsense is synchronized with the system clock
        initial_system_timestamp = self.get_clock().now()

        # Calculate timestamp offset
        # TODO: measure accurate latency with QR code
        self.timestamp_offset = initial_system_timestamp - initial_system_timestamp
        logger.debug(f"Timestamp offset: {self.timestamp_offset.nanoseconds / 1e6} ms")
        logger.debug("Camera is ready! Start publishing images...")

    def stop(self):
        # Stop the camera
        self.pipeline.stop()
        logger.info("Camera stopped!")
        
        # print(self.profiler.output_text(unicode=True, color=True))

    def convert_to_system_timestamp(self, camera_timestamp: Time) -> Time:
        """
        Convert camera timestamp to system timestamp
        """
        return camera_timestamp + self.timestamp_offset

    def publish_color_image(self, color_frame: rs.composite_frame, camera_timestamp: Time):
        """
        Publish color image
        """
        color_image = copy.deepcopy(np.asanyarray(color_frame.get_data()))
        success, encoded_image = cv2.imencode('.jpg', color_image)
        # success = False
        # stream_profile = color_frame.get_profile().as_video_stream_profile()
        # fmt = stream_profile.format()
        # if fmt == rs.format.rgb8:
        #     color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        # elif fmt == rs.format.yuyv:
        #     color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2)
        # elif fmt == rs.format.bgr8:
        #     pass
        # else:
        #     logger.warn(f"Unhandled format {fmt}, try to encode raw data directly.")
            
        # success, encoded_image = cv2.imencode('.jpg', color_image)
        # if not success:
        #     logger.error("JPEG encode failed")
        #     return

        # Fill the message
        msg = Image()
        # msg = CompressedImage()
        # msg.header.stamp = self.convert_to_system_timestamp(camera_timestamp).to_msg()
        # msg.header.frame_id = "camera_color_frame"
        # msg.format = "jpeg"  # 这里必须是 jpeg 或 png
        # if success:
        #     msg.data = encoded_image.tobytes()
        msg.header.stamp = self.convert_to_system_timestamp(camera_timestamp).to_msg()
        msg.header.frame_id = "camera_color_frame"
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        if success:
            image_bytes = encoded_image.tobytes()
            msg.data = image_bytes
        else:
            logger.debug('fail to image encoding!')
            msg.data = color_image.tobytes()
        
        # msg = numpy_to_image(color_image, "bgr8")
        self.color_publisher_.publish(msg)
        
    def publish_depth_image(self, depth_frame: rs.frame, camera_timestamp: Time):
        depth_image = np.asanyarray(depth_frame.get_data())
        if depth_image.dtype != np.uint16:
            depth_image = depth_image.astype(np.uint16, copy=False)

        success, encoded = cv2.imencode('.png', depth_image)
        if not success:
            self.get_logger().warning('PNG encode failed; skip this depth frame')
            return

        msg = CompressedImage()
        msg.header.stamp = self.convert_to_system_timestamp(camera_timestamp).to_msg()
        msg.header.frame_id = "camera_depth_frame"
        msg.format = "png"
        msg.data = encoded.tobytes()

        self.depth_publisher_.publish(msg)
        # depth_image = np.asanyarray(depth_frame.get_data())
        # success, compressed_data = cv2.imencode('.png', depth_image)
        # # success = False
    
        # msg = Image()
        # msg.header.stamp = self.convert_to_system_timestamp(camera_timestamp).to_msg()
        # msg.header.frame_id = "camera_depth_frame"
        # msg.height = depth_image.shape[0]
        # msg.width = depth_image.shape[1]
        # # msg.encoding = "16UC1" 
        # # msg.encoding = "16UC1; png compressed"
        # msg.encoding = 'png'
        # msg.is_bigendian = 0
        # # msg.step = msg.width * 2
        # msg.step = 0
        
        # if success:
        #     image_bytes = compressed_data.tobytes()
        #     msg.data = image_bytes
        #     # msg.data = list(image_bytes)
        # else:
        #     logger.debug('fail to depth image encoding!')
        #     msg.data = depth_frame.tobytes()
        

        # self.depth_publisher_.publish(msg)

    def publish_point_cloud(self, depth_frame: rs.frame, color_frame: rs.frame, camera_timestamp: Time):
        
        compression_success = False
        
        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        valid_indices = vertices[:, 2] > 0
        vertices = vertices[valid_indices]
        color_image = np.asanyarray(color_frame.get_data())
        colors = color_image.reshape(-1, 3)[valid_indices]
        colors_rgb = colors[:, ::-1]

        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
        structured_points = np.empty(len(vertices), dtype=dtype)
        structured_points['x'], structured_points['y'], structured_points['z'] = vertices.T
        structured_points['r'], structured_points['g'], structured_points['b'] = colors_rgb.T
        
        header = Header()
        header.stamp = camera_timestamp.to_msg()
        header.frame_id = f"{self.camera_name}_color_optical_frame"
        
        fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
                  PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
                  PointField(name='b', offset=14, datatype=PointField.UINT8, count=1)]
        
        
        # start_time = time.time()
        uncompressed_msg = point_cloud2.create_cloud(header, fields, structured_points)
        
        # freq = 1 / (time.time() - start_time)
        
        # logger.info(f"freq: {freq}")

        if compression_success:
            if not hasattr(self, 'zstd_compressor'):
                self.zstd_compressor = zstd.ZstdCompressor(level=3)

            compressed_data = self.zstd_compressor.compress(uncompressed_msg.data)
            
            msg_to_publish = PointCloud2()
            msg_to_publish.header = uncompressed_msg.header
            msg_to_publish.height = uncompressed_msg.height
            msg_to_publish.width = uncompressed_msg.width
            msg_to_publish.fields = uncompressed_msg.fields
            msg_to_publish.is_bigendian = uncompressed_msg.is_bigendian
            msg_to_publish.point_step = uncompressed_msg.point_step
            msg_to_publish.row_step = uncompressed_msg.row_step
            msg_to_publish.is_dense = uncompressed_msg.is_dense
            
            msg_to_publish.data = compressed_data
            
            self.pointcloud_publisher_.publish(msg_to_publish)
        else:
            # logger.debug('fail to compress pointcloud! publishing raw data.')
            self.pointcloud_publisher_.publish(uncompressed_msg)

    def send_streaming_msg(self, color_image):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.streaming_quality]
        ret, color_image_encoded = cv2.imencode('.jpg', color_image, encode_param)
        color_image_bytes = color_image_encoded.tobytes()
        packed_data_dict = {"images": [{"id": self.id,
                                        "inHeadSpace": False,
                                        **display_params,
                                        **{"image": color_image_bytes}}
                                        for display_params in self.streaming_display_params_list]}
        packed_data = bson.dumps(packed_data_dict)

        arrow_address = (self.streaming_server_ip, self.streaming_server_port)
        chunk_size = self.streaming_chunk_size

        self.socket.sendto(len(packed_data).to_bytes(length=4, byteorder='little', signed=False), arrow_address)
        if self.debug:
            logger.debug(f"Sending streaming image to VR server with size {len(packed_data)}")

        self.socket.sendto(chunk_size.to_bytes(length=4, byteorder='little', signed=False), arrow_address)
        count = math.ceil(len(packed_data) / chunk_size)
        if self.debug:
            logger.debug(f"Sending streaming image to VR server with {count} chunks of size {chunk_size}")

        for i in range(count):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            if end > len(packed_data):
                end = len(packed_data)
            self.socket.sendto(packed_data[start:end], arrow_address)
        if self.debug:
            logger.debug(f"Sent streaming image to VR server")

    def timer_callback(self):
        """
        Publish the color and depth frames
        """

        while True:
            # capture frames
            frames = self.pipeline.wait_for_frames()

            camera_timestamp = self.get_clock().now()
            
            aligned_frames = self.align.process(frames)

            # we only record the raw color frame
            # raw_color_frame = frames.get_color_frame()
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            # depth_frame = self.decimate_filter.process(depth_frame) # for depth fps

            # camera_timestamp_ms = color_frame.get_timestamp()
            # camera_timestamp_ms = depth_frame.get_timestamp()
            # camera_timestamp = convert_float_to_ros_time(camera_timestamp_ms / 1000.0)
            # color_frame = raw_color_frame
            # depth_frame = None
            if not color_frame:
                continue

            # publish the color image
            # self.publish_color_image(raw_color_frame, camera_timestamp)
            self.publish_color_image(color_frame, camera_timestamp)
            # start_time = time.time()
            # self.publish_point_cloud(depth_frame, color_frame, camera_timestamp)
            # end_time = time.time() - start_time
            # freq = 1/end_time
            # logger.info(f"freq : {freq}")
            self.publish_depth_image(depth_frame, camera_timestamp)

            # send streaming image
            if self.enable_streaming:
                color_image = np.asanyarray(color_frame.get_data())
                self.send_streaming_msg(color_image)

            # calculate fps
            self.frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time >= 1.0:
                frame_rate = self.frame_count / elapsed_time
                self.fps_list.append(frame_rate)
                logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
                self.prev_time = current_time
                self.frame_count = 0

            # calculate interval between frames
            if self.last_frame_time is not None:
                frame_interval = (current_time - self.last_frame_time) * 1000
                self.frame_intervals.append(frame_interval)
            self.last_frame_time = current_time

            # Print info and make plots every 5 seconds
            if current_time - self.last_print_time >= 5:
                logger.info(f"Publishing image from {self.camera_name} at timestamp (s): {camera_timestamp.nanoseconds / 1e9}")
                self.last_print_time = current_time
            break


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseCameraPublisher(
            # camera_serial_number='218622273046',
            camera_serial_number='218722270752',
            camera_type='D400',
            camera_name='external_camera_d405',
            fps=30
        ) # D405
    # node = RealsenseCameraPublisher(
    #         # camera_serial_number= 'f1380685',
    #         camera_serial_number = 'f1422067',
    #         camera_type='L500',
    #         camera_name='external_camera_l515',
    #         # rgb_resolution=(640, 480),
    #         rgb_resolution=(960, 540),
    #         depth_resolution=(640, 480)
    #         # exposure=300
    #     )  # L515
    
    # profiler = Profiler()
    try:
        # profiler.start()
        rclpy.spin(node)
    except IndentationError as e:
    # except (KeyboardInterrupt, SystemExit):
        # logger.info("Shutdown requested by user.")
        logger.exception(e)
    # finally:
        # profiler.stop()
        node.stop()
        node.destroy_node()
        # if rclpy.ok():
            # rclpy.shutdown()
        # rclpy.shutdown()
        
        # print("\n--- pyinstrument performance report ---")
        # print(profiler.output_text(unicode=True, color=True))
        # print("-------------------------------------\n")




if __name__ == '__main__':
    main()
