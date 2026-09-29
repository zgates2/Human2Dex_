import os.path
import numpy as np
import rclpy
import bson
import socket
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2, PointField
import json
from loguru import logger
import math
import uuid
import time
import cv2
import time as tm
import copy
import struct
import requests
import open3d as o3d
import threading
from threading import Lock

import sys
sys.path.append("/home/ps/reactive_diffusion_policy")

from reactive_diffusion_policy.common.data_models import TactileSensorMessage, Arrow
from reactive_diffusion_policy.common.tactile_marker_utils import marker_normalization
from reactive_diffusion_policy.real_world.publisher.lib import find_marker
from reactive_diffusion_policy.real_world.publisher.gelsight_utility import GelsightUtility
import pyinstrument


class GelsightCameraPublisher(Node):
    '''
    GelSight Camera publisher Class (Thread-safe version)
    '''

    def __init__(self,
                 device_path: str,
                 camera_type: str = 'gelsight',
                 fps: int = 30, 
                 exposure: int = -6,
                 contrast: int = 100,
                 camera_name: str = 'left_gripper_camera_1',
                 vr_server_ip: str = '127.0.0.1',
                 vr_server_port: int = 10002,
                 teleop_server_ip: str = '192.168.2.187',
                 teleop_server_port: int = 8082,
                 dimension=3,
                 marker_vis_rotation_angle: float = 0.,  # in degrees
                 debug=False,
                 video_path="../../../data/tactile_video/video_001.mp4",
                 recorded=False,
                 enable_streaming: bool = False,
                 streaming_server_ip: str = '127.0.0.1',
                 streaming_server_port: int = 10004,
                 streaming_quality: int = 10,
                 streaming_chunk_size: int = 1024,
                 streaming_display_params_list: list = None,
                 vis_latency_steps: int = 5,
                 ):
        node_name = f'{camera_name}_publisher'
        super().__init__(node_name)

        # --- Parameter Initialization ---
        self.device_path = device_path
        self.camera_name = camera_name
        self.fps = fps
        self.contrast = contrast
        self.exposure = exposure
        self.width = 640
        self.height = 480
        self.debug = debug
        self.dimension = dimension
        self.marker_vis_rotation_angle = np.deg2rad(marker_vis_rotation_angle)
        self.recorded = recorded
        # ... (other parameters)

        # --- ROS Publishers and Timer ---
        self.color_publisher_ = self.create_publisher(Image, f'/{camera_name}/color/image_raw', 10)
        self.marker_publisher = self.create_publisher(PointCloud2, f'/{camera_name}/marker_offset/information', 10)
        # Timer can now truly trigger at the specified fps without being blocked
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

        # --- Marker Tracking State ---
        self.initial_markers = None
        self.prev_markers = None
        self.initial_markers_3d = None
        self.vertical_scale = 0.05
        
        # --- MODIFICATION START: Thread-safe camera reading and alternating logic state ---
        self.cap = None
        self.latest_frame = None
        self.frame_lock = Lock()
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._camera_read_loop)

        # State variables for the alternating "double publish" logic
        self.frame_toggle_counter = 0
        self.last_processed_color_frame = None
        self.last_processed_initial_markers = None
        self.last_processed_marker_motion = None
        # --- MODIFICATION END ---
        
        # --- Performance Stats ---
        self.last_print_time = tm.time()

        # GelSight specific utilities
        self.GelsightHandler = GelsightUtility(RESCALE=1)
        self.m = find_marker.Matching(
            N_=self.GelsightHandler.N, M_=self.GelsightHandler.M,
            fps_=self.GelsightHandler.fps, x0_=self.GelsightHandler.x0,
            y0_=self.GelsightHandler.y0, dx_=self.GelsightHandler.dx,
            dy_=self.GelsightHandler.dy
        )
        
        # ... (Other initializations like teleop, streaming etc.)
        self.video_path = video_path
        if recorded:
            assert os.path.exists(self.video_path), f"Video path {self.video_path} does not exist!"
        self.enable_streaming = enable_streaming
        if self.enable_streaming:
            # ... (streaming setup)
            pass

        # Start the camera and the reading thread
        self.start()
        self.camera_thread.start()

    # --- MODIFICATION START: New methods for threaded camera handling ---
    def _camera_read_loop(self):
        """
        This function runs in a separate background thread, dedicated to reading from the camera.
        This prevents the blocking `cap.read()` from stalling the main ROS event loop.
        """
        logger.info("Camera reading thread started...")
        while self.is_running and rclpy.ok():
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Pre-process the frame here to offload work from the main thread
                    resized_frame = cv2.resize(frame, (self.width, self.height))
                    processed_frame = self.GelsightHandler.img_initiation(resized_frame)
                    with self.frame_lock:
                        self.latest_frame = processed_frame
                else:
                    if self.recorded: # If it's a recorded video and it ends
                        logger.info("End of video file. Stopping camera thread.")
                        break
                    tm.sleep(0.01) # Small sleep if read fails on a live camera
            else:
                tm.sleep(0.1) # Wait if camera is not opened
        logger.info("Camera reading thread stopped.")

    def get_latest_frame(self):
        """Thread-safely gets the latest frame."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def start(self):
        """Initializes the camera capture object."""
        if self.recorded:
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(self.device_path)

        if not self.cap.isOpened():
            logger.error(f"Could not open video source: {self.device_path or self.video_path}")
            raise Exception("Could not open video source")
        
        if not self.recorded:
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        
        logger.info(f"{self.camera_name} started.")

    def stop(self):
        """Stops the camera thread and releases resources."""
        logger.info("Stopping camera node...")
        self.is_running = False
        if self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2) # Wait for the thread to finish
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Camera released.")
    # --- MODIFICATION END ---
    
    def timer_callback(self):
        """
        This callback now runs at the desired `fps` and is not blocked by I/O.
        It alternates between processing new frames and re-publishing old data.
        """
        self.frame_toggle_counter += 1

        # Case 1: Real processing (on odd-numbered frames: 1, 3, 5...)
        if self.frame_toggle_counter % 2 == 1:
            color_frame = self.get_latest_frame()
            if color_frame is None:
                logger.warning("No frame available from camera thread, skipping cycle.")
                return
            
            # This is a new frame, so we get a new timestamp
            camera_timestamp = self.get_clock().now()
            
            # Perform the expensive image processing
            initial_markers, marker_motion = self.get_marker_image(color_frame)

            # Normalize the results
            initial_markers_norm, marker_motion_norm = marker_normalization(
                copy.deepcopy(initial_markers), copy.deepcopy(marker_motion),
                self.dimension, width=self.width, height=self.height
            )
            
            # Store the processed data for the next "fake" frame
            self.last_processed_color_frame = color_frame
            self.last_processed_initial_markers = initial_markers_norm
            self.last_processed_marker_motion = marker_motion_norm
        
        # Case 2: Fake publish (on even-numbered frames: 2, 4, 6...)
        else:
            # If we haven't processed a real frame yet, just skip
            if self.last_processed_color_frame is None:
                return
            
            # Reuse the data from the last real processing step
            color_frame = self.last_processed_color_frame
            initial_markers_norm = self.last_processed_initial_markers
            marker_motion_norm = self.last_processed_marker_motion
            
            # CRITICAL: Get a fresh, new timestamp for this repeated data
            camera_timestamp = self.get_clock().now()
        
        # --- Common Publishing Logic (for both real and fake frames) ---
        if (color_frame is None) or (initial_markers_norm is None) or (marker_motion_norm is None):
            return

        # Publish the marker offset
        self.publish_marker_offset(initial_markers_norm, marker_motion_norm, camera_timestamp)

        # Publish the color image
        self.publish_color_image(color_frame, camera_timestamp)

        # (Optional) Send streaming image
        if self.enable_streaming:
            self.send_streaming_msg(color_frame.copy())

        # Print info every 5 seconds
        current_time = tm.time()
        if current_time - self.last_print_time >= 5:
            # logger.info(f"Publishing from {self.camera_name} at timestamp (s): {camera_timestamp.nanoseconds / 1e9}")
            self.last_print_time = current_time

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

        if self.initial_markers_3d is None:
            self.initial_markers_3d = self.GelsightHandler.ComputesurroundingArea(Ox, Oy)
        
        initial_marker = np.zeros((M * N, 3))
        marker_motion = np.zeros((M * N, 2))
        if dimension == 3:
            current_marker_3d = self.GelsightHandler.ComputesurroundingArea(Cx, Cy)

        k = 0
        for i in range(M):
            for j in range(N):
                if self.dimension == 2:
                    initial_marker[k] = [Ox[i][j], Oy[i][j], 0]
                elif self.dimension == 3:
                    initial_marker[k] = [Ox[i][j], Oy[i][j], max((current_marker_3d[i][j] - self.initial_markers_3d[i][j]) * self.vertical_scale, 0)]
                marker_motion[k] = [Cx[i][j] - Ox[i][j], Cy[i][j] - Oy[i][j]]
                k += 1
        return initial_marker, marker_motion

    def publish_marker_offset(self, marker_loc, marker_offset, camera_timestamp: Time):
        cur_marker = copy.deepcopy(marker_loc)[:, :2]
        marker_information = np.hstack((cur_marker, marker_offset)).astype(np.float32)

        msg = PointCloud2()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f'camera_marker_offset_{self.camera_name}'
        msg.is_bigendian = False
        msg.point_step = 16  # 4 fields * 4 bytes/field
        msg.is_dense = True
        msg.fields = [
            PointField(name='marker_location_x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_location_y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_x', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='marker_offset_y', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = b''.join(map(lambda row: struct.pack('ffff', *row), marker_information))
        self.marker_publisher.publish(msg)

    def publish_color_image(self, color_image, camera_timestamp: Time):
        msg = Image()
        msg.header.stamp = camera_timestamp.to_msg()
        msg.header.frame_id = f"camera_color_frame_{self.camera_name}"
        msg.height, msg.width, _ = color_image.shape
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        
        success, encoded_image = cv2.imencode('.jpg', color_image)
        if success:
            msg.data = encoded_image.tobytes()
        else:
            logger.warning('Failed to encode image!')
            msg.data = color_image.tobytes()
        self.color_publisher_.publish(msg)

def main(args=None):
    import psutil
    cpu_core_id = set([11, 12, 13])
    total_cores = psutil.cpu_count()
    for id in cpu_core_id:
        if id >= total_cores:
            raise ValueError(f"Invalid cpu_id: {id}, total cores: {total_cores}")
    os.sched_setaffinity(0, cpu_core_id)
    
    rclpy.init(args=args)
    device_path = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._GelSight_Mini_R0B_2BVR-3BUH_2BVR3BUH-video-index0"
    
    node = GelsightCameraPublisher(
        device_path=device_path,
        camera_name='left_gripper_camera_1',
        fps=30,
        debug=False,
        recorded=False,
        dimension=2,
        camera_type="gelsight"
    )
    
    if node.debug:
        # node.marker_track_visualization() # This might need adjustments for the new threaded model
        logger.warning("Debug visualization may not work as expected with the new threaded model.")
    else:
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected, shutting down.")
        finally:
            # MODIFICATION: Ensure the stop method is called to cleanly exit the thread
            node.stop()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    os.environ["OPENBLAS_NUM_THREADS"] = "12"
    cv2.setNumThreads(12)

    main()
