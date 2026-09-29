#!/usr/bin/env python3
"""
Pico VR Controller Reader with TCP Transformation & ROS2 TF.
从 Redis 异步读取 VR 手柄数据，应用 TCP 偏移，并可选择发布 ROS2 TF。

四元数格式约定:
- 内部处理/输出: [x, y, z, qw, qx, qy, qz] (Position + Quaternion w,x,y,z - Scalar First)
"""

import sys
# 确保包含项目路径以便导入 common.space_utils
sys.path.append("/home/ps/omniUMI")

import time
import json
import redis
import numpy as np
from threading import Event, Lock, Thread
from typing import Optional, Tuple, Dict
from loguru import logger

# === 移植的数学库引用 ===
from spatialmath import SE3
# 假设您环境中有此库，若无请确保路径正确
try:
    from reactive_diffusion_policy.common.space_utils import (
        pose_7d_to_4x4matrix, 
        matrix4x4_to_pose_7d, 
        pose_7d_to_pose_6d, 
        pose_6d_to_pose_7d
    )
except ImportError:
    logger.error("无法导入 reactive_diffusion_policy.common.space_utils，请检查路径。")
    sys.exit(1)

# === ROS2 引用 (用于 TF 发布) ===
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster
except ImportError:
    rclpy = None
    Node = object
    TransformStamped = object
    TransformBroadcaster = object
    logger.warning("ROS2 (rclpy/tf2) 模块未找到，TF 发布功能将禁用。")


T_htc_ultimate = SE3()

# HTC 坐标系对齐变换：将 HTC 坐标系对齐到期望的坐标系

# T_offset = SE3.Rx(90, unit='deg') * SE3.Ry(-90, unit='deg')
# T_offset.A[:3, 3] = np.array([0.2, 0.2, 0.2])
# T_htc_to_flange = (SE3.Rz(45, unit='deg') * 
#                    SE3.Rx(-90, unit='deg') * 
#                    SE3.Tx(-0.0532) * 
#                    SE3.Ty(0.05324) * 
#                    SE3.Tz(-0.04147))



pico2flange_matrix_data = np.array([
    [ 0.707106781186545,    0.636264628923053,   0.308492012832445,   56.6581914048022],
    [ 5.44789105054735E-15,-0.436273588431422,   0.89981406748126,   -42.7682717438898],
    [ 0.70710678118655,    -0.636264628923045,  -0.30849201283245,   -64.9824918626055],
    [ 0.0,                  0.0,                  0.0,                  1.0]
])

# pico2flange_matrix_data = np.array([
#     [ 0.707106781186545,    5.44789105054735E-15,   0.70710678118655	,   5.89201000436006],
#     [ 0.636264628923053 ,  -0.436273588431422,   -0.636264628923045,    -96.0595871316353],
#     [ 0.308492012832445,    0.89981406748126,  -0.30849201283245,       0.955765189510426],
#     [ 0.0,                  0.0,                  0.0,                  1.0]
# ])

pico2flange_matrix_data[:3, 3] = pico2flange_matrix_data[:3, 3] / 1000.0

T_htc_to_flange = SE3(pico2flange_matrix_data)
T_flange = T_htc_ultimate * T_htc_to_flange

# 3. 法兰坐标系 -> 力传感器坐标系
# 沿 z 轴移动 22.5mm 
T_flange_to_sensor = SE3.Tz((15.5+7.0)/1000)
T_htc_to_sensor = T_flange * T_flange_to_sensor
 
# 4. 法兰坐标系 -> TCP 坐标系
# 沿 z 轴移动 216.5mm
T_flange_to_tcp = SE3.Tz(0.2165)
T_htc_to_tcp = T_flange * T_flange_to_tcp
print(T_htc_to_tcp)


# === 辅助函数 ===
def quat_mul_np(x, y, scalar_first=True):
    """四元数乘法"""
    x = np.array(x)
    y = np.array(y)

    if not scalar_first:
        x = x[..., [3, 0, 1, 2]]
        y = y[..., [3, 0, 1, 2]]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
        x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
        x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
        x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0
    ], axis=-1)

    if not scalar_first:
        res = res[..., [1, 2, 3, 0]]

    return res


def coordinate_transform_unity_to_right_hand(position, rotation_quat):
    """
    从 Unity (左手坐标系) 转换到右手坐标系。
    
    Args:
        position: Unity 左手系位置 [x, y, z]
        rotation_quat: 四元数 [w, x, y, z] (scalar first)
    
    Returns:
        position_transformed: 右手系位置 [x, y, z]
        orientation: 右手系四元数 [w, x, y, z] (scalar first)
    """
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    m = rotation_matrix
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25 * s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25 * s

    rotation_quat_transform = np.array([w, x, y, z])  # scalar-first
    # 这里传入的 rotation_quat 应该是 [w, x, y, z] 格式
    orientation = quat_mul_np(rotation_quat_transform, np.array(rotation_quat), scalar_first=True)
    position_transformed = np.array(position) @ rotation_matrix.T

    return position_transformed, orientation


class PicoControllerAsync:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        tracker_name: str = "pico",
        allow_repeat_frames: bool = True,
        debug: bool = False,
        pub: bool = False,  # 新增: 是否发布 TF
    ):
        """
        Args:
            host: Redis 主机地址
            port: Redis 端口
            db: Redis 数据库
            tracker_name: 传感器名称
            allow_repeat_frames: 是否允许重复帧
            debug: 调试模式
            pub: 是否通过 ROS2 发布 TF
        """
        self.host = host
        self.port = port
        self.db = db
        self.tracker_name = tracker_name
        self.allow_repeat_frames = allow_repeat_frames
        self.debug = debug
        self.pub = pub

        self.redis_client: Optional[redis.Redis] = None
        self.success_count = 0
        self.failure_count = 0

        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None

        self.frame_lock = Lock()
        # 最新位姿 (TCP): [x, y, z, qw, qx, qy, qz]
        self.latest_pose_left: Optional[np.ndarray] = None
        self.latest_pose_right: Optional[np.ndarray] = None
        self.latest_frame_id = 0
        self.new_frame_event = Event()

        # ROS2 TF 初始化
        self.node: Optional[Node] = None
        self.tf_broadcaster: Optional[TransformBroadcaster] = None

        if self.pub and rclpy is not None:
            try:
                if not rclpy.ok():
                    rclpy.init(args=sys.argv)
                self.node = Node(f'pico_{self.tracker_name}_tf_publisher')
                self.tf_broadcaster = TransformBroadcaster(self.node)
                logger.info("ROS2 TF publishing enabled.")
            except Exception as e:
                logger.error(f"Failed to initialize ROS2 for TF publishing: {e}")
                self.pub = False

    @property
    def is_connected(self) -> bool:
        return (
            self.redis_client is not None
            and self.thread is not None
            and self.thread.is_alive()
        )

    def _publish_tf(self, parent_frame, child_frame, pos, quat_wxyz):
        """发布 TF 变换"""
        if not self.pub or self.tf_broadcaster is None:
            return

        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = parent_frame 
        t.child_frame_id = child_frame

        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        # ROS 使用 [x, y, z, w] 顺序，这里输入是 [w, x, y, z]
        t.transform.rotation.w = float(quat_wxyz[0])
        t.transform.rotation.x = float(quat_wxyz[1])
        t.transform.rotation.y = float(quat_wxyz[2])
        t.transform.rotation.z = float(quat_wxyz[3])

        self.tf_broadcaster.sendTransform(t)

    def connect(self):
        if self.is_connected:
            raise RuntimeError(f"{self.tracker_name} 已经连接。")

        logger.info(f"正在连接 {self.tracker_name} 到 redis://{self.host}:{self.port}/{self.db}...")

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                socket_timeout=0.1,
                socket_connect_timeout=0.5,
            )
            self.redis_client.ping()
            logger.info(f"{self.tracker_name} 已连接到 Redis。")
            self._start_thread()

        except redis.RedisError as e:
            logger.error(f"连接 {self.tracker_name} 失败: {e}")
            raise

    def disconnect(self):
        if not self.is_connected and self.thread is None:
            return

        logger.info(f"正在断开 {self.tracker_name}...")
        self._stop_thread()

        if self.redis_client is not None:
            self.redis_client.close()
            self.redis_client = None

        logger.info(f"{self.tracker_name} 已断开。")

    def _receive_and_parse(self) -> Optional[Dict]:
        """从 Redis 获取原始数据并转换为右手坐标系"""
        if self.redis_client is None:
            return None

        try:
            controller_data_raw = self.redis_client.get("controller_data")
            if controller_data_raw is None:
                return None

            controller_data = json.loads(controller_data_raw)
            result = {}

            # 处理左手柄
            left_ctrl = controller_data.get('LeftController', {})
            left_pos = left_ctrl.get('position', [0, 0, 0])
            left_rot = left_ctrl.get('rotation', [1, 0, 0, 0]) # Redis 中已经是 [w, x, y, z] 格式
            # 直接使用，不需要转换
            left_rot_sf = np.array(left_rot)
            left_pos_rh, left_rot_rh = coordinate_transform_unity_to_right_hand(
                np.array(left_pos), left_rot_sf
            )
            result['left'] = np.concatenate([left_pos_rh, left_rot_rh])

            # 处理右手柄
            right_ctrl = controller_data.get('RightController', {})
            right_pos = right_ctrl.get('position', [0, 0, 0])
            right_rot = right_ctrl.get('rotation', [1, 0, 0, 0]) # Redis 中已经是 [w, x, y, z] 格式
            # 直接使用，不需要转换
            right_rot_sf = np.array(right_rot)
            right_pos_rh, right_rot_rh = coordinate_transform_unity_to_right_hand(
                np.array(right_pos), right_rot_sf
            )
            result['right'] = np.concatenate([right_pos_rh, right_rot_rh])

            return result

        except json.JSONDecodeError:
            return None
        except redis.RedisError:
            return None

    def _processing_loop(self):
        """后台线程：接收 -> TCP变换 -> 发布TF"""
        frame_id = 0
        logger.info(f"{self.tracker_name}: 后台线程已启动。")

        while not self.stop_event.is_set():
            try:
                # 1. 获取基础右手系位姿 (Raw RH)
                # 格式: [x, y, z, qw, qx, qy, qz]
                data = self._receive_and_parse()

                if data is not None:
                    pose_left_raw = data['left']
                    pose_right_raw = data['right']
                    
                    # Left
                    mat_left_raw = pose_7d_to_4x4matrix(pose_left_raw)
                    mat_left_tcp = mat_left_raw @ T_htc_to_flange.A @ T_htc_to_tcp.A
                    pose_left_tcp = matrix4x4_to_pose_7d(mat_left_tcp)

                    # Right
                    mat_right_raw = pose_7d_to_4x4matrix(pose_right_raw)
                    mat_right_tcp = mat_right_raw @ T_htc_to_flange.A @ T_htc_to_tcp.A
                    pose_right_tcp = matrix4x4_to_pose_7d(mat_right_tcp)

                    # 3. 发布 TF (可选)
                    if self.pub:
                        # 发布 Offset (模拟 map -> offset) - 仅平移，无旋转
                        offset_matrix = np.eye(4)
                        offset_matrix[:3, 3] = np.array([0.2, 0.2, 0.2])
                        offset_7d = matrix4x4_to_pose_7d(offset_matrix)
                        self._publish_tf("map", "pico_offset", offset_7d[:3], offset_7d[3:])
                        
                        # 发布 Right Controller TCP
                        # 为了演示清晰，这里主要发布右手柄作为 tcp_end_effector
                        self._publish_tf("pico_offset", "pico_right_raw", pose_right_raw[:3], pose_right_raw[3:])
                        # self._publish_tf("offset", "tcp_end_effector_right", pose_right_tcp[:3], pose_right_tcp[3:])
                        
                        # 也发布左手柄
                        self._publish_tf("pico_offset", "pico_tcp_end_effector_right", pose_right_tcp[:3], pose_right_tcp[3:])

                    frame_id += 1

                    with self.frame_lock:
                        self.latest_pose_left = pose_left_tcp
                        self.latest_pose_right = pose_right_tcp
                        self.latest_frame_id = frame_id

                    self.success_count += 1
                    self.new_frame_event.set()

                    if self.debug and frame_id % 100 == 0:
                        logger.debug(f"{self.tracker_name} [Frame {frame_id}]")
                else:
                    self.failure_count += 1

                time.sleep(0.001)

            except Exception as e:
                logger.error(f"{self.tracker_name}: 线程错误: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info(f"{self.tracker_name}: 线程已停止。")

    def _start_thread(self):
        if self.thread is not None and self.thread.is_alive():
            self._stop_thread()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._processing_loop,
            name=f"{self.tracker_name}_processing",
            daemon=True
        )
        self.thread.start()

    def _stop_thread(self):
        if self.stop_event is None:
            return
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 500) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        异步读取最新的 TCP 位姿。
        Returns:
            (pose_left_tcp, pose_right_tcp, frame_id)
        """
        if not self.is_connected:
            raise RuntimeError(f"{self.tracker_name} 未连接。")

        if self.thread is None or not self.thread.is_alive():
            self._start_thread()
            time.sleep(0.5)

        if self.allow_repeat_frames:
            self.new_frame_event.wait(timeout=timeout_ms / 1000.0)
            with self.frame_lock:
                return self.latest_pose_left, self.latest_pose_right, self.latest_frame_id
        else:
            if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
                raise TimeoutError(f"超时 {timeout_ms}ms")
            with self.frame_lock:
                self.new_frame_event.clear()
                return self.latest_pose_left, self.latest_pose_right, self.latest_frame_id


# === 测试代码 ===
REDIS_HOST = "localhost"
# REDIS_HOST = "172.16.13.91"
REDIS_PORT = 6379

def main():
    print("\n" + "=" * 80)
    print("Pico TCP 变换 & ROS2 TF 发布测试")
    print("=" * 80)

    # 启用 pub=True 以发布 TF (需要安装 ROS2)
    pico = PicoControllerAsync(host=REDIS_HOST, port=REDIS_PORT, pub=True)

    try:
        pico.connect()
        print("✓ 连接成功，正在后台计算 TCP 位姿并发布 TF...")
        print("按 Ctrl+C 停止。")

        while True:
            try:
                # 获取经过变换后的 TCP 位姿
                left_tcp, right_tcp, fid = pico.async_read(timeout_ms=1000)
                
                # 打印右手柄 TCP 数据
                if right_tcp is not None:
                    x, y, z, qw, qx, qy, qz = right_tcp
                    print(f"\r[Frame {fid}] Right TCP: pos=[{x:.3f}, {y:.3f}, {z:.3f}] quat=[{qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}]", end="")
                
                time.sleep(0.01)

            except TimeoutError:
                print("\r等待数据中...", end="")

    except KeyboardInterrupt:
        print("\n\n✓ 用户中断。")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        pico.disconnect()

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    main()