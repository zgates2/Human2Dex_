import multiprocessing
import rclpy
from realsense_camera_publisher import RealsenseCameraPublisher
import time

def run_camera(serial_number, camera_type, camera_name, rgb_resolution=None):
    rclpy.init()  # 每个子进程都要单独初始化 rclpy

    if rgb_resolution is not None:
        node = RealsenseCameraPublisher(
            camera_serial_number=serial_number,
            camera_type=camera_type,
            camera_name=camera_name,
            rgb_resolution=rgb_resolution
        )
    else:
        node = RealsenseCameraPublisher(
            camera_serial_number=serial_number,
            camera_type=camera_type,
            camera_name=camera_name
        )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # 推荐在 ROS2 里用 spawn 而不是 fork
   
    # L515
    p1 = multiprocessing.Process(
        target=run_camera,
        args=("f1422067", "L500", "external_camera"),
        kwargs={"rgb_resolution": (960, 540)}
    )
    
    
    
    # D405
    p2 = multiprocessing.Process(
        target=run_camera,
        args=("218622273046", "D400", "external_camera_d405")
    )


    

    p1.start()
    time.sleep(10)
    p2.start()

    p1.join()
    p2.join()