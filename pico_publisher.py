#!/usr/bin/env python3
"""
Pico VR Controller Reader with TCP Transformation.
从 Redis 异步读取 VR 手柄数据，应用 TCP 变换。

四元数格式约定:
- Redis 中存储: [w, x, y, z] (scalar-first)
- 内部/输出: [w, x, y, z] (scalar-first)
"""

import sys
sys.path.append("/home/ps/omniUMI")
import time
import json
import redis
import numpy as np
from threading import Event, Lock, Thread
from typing import Optional, Tuple, Dict
from loguru import logger

# === 数学库引用 ===
from spatialmath import SE3
try:
    from reactive_diffusion_policy.common.space_utils import (
        pose_7d_to_4x4matrix, 
        matrix4x4_to_pose_7d
    )
except ImportError:
    logger.error("无法导入 reactive_diffusion_policy.common.space_utils，请检查路径。")
    sys.exit(1)


# === TCP 变换矩阵定义 ===
T_htc_ultimate = SE3()

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

# === 数学库引用 ===
from spatialmath import SE3
try:
    from reactive_diffusion_policy.common.space_utils import (
        pose_7d_to_4x4matrix, 
        matrix4x4_to_pose_7d
    )
except ImportError:
    logger.error("无法导入 reactive_diffusion_policy.common.space_utils，请检查路径。")
    sys.exit(1)


def quat_mul_np(x, y, scalar_first=True):
    """
    四元数乘法。
    Args:
        x, y: 四元数，格式为 [w, x, y, z] (scalar_first=True) 或 [x, y, z, w]
    Returns:
        四元数乘积，格式与输入相同
    """
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
        position: [x, y, z] Unity 坐标系下的位置
        rotation_quat: [w, x, y, z] 四元数 (scalar-first)，Unity 坐标系
    Returns:
        position: [x, y, z] 右手坐标系下的位置
        rotation: [w, x, y, z] 四元数 (scalar-first)，右手坐标系
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
    ):
        """
        Args:
            host: Redis 主机地址 (默认: localhost)
            port: Redis 端口 (默认: 6379)
            db: Redis 数据库编号 (默认: 0)
            tracker_name: 传感器名称 (用于日志)
            allow_repeat_frames: 是否允许返回重复帧
            debug: 是否开启调试日志
        """
        self.host = host
        self.port = port
        self.db = db
        self.tracker_name = tracker_name
        self.allow_repeat_frames = allow_repeat_frames
        self.debug = debug

        self.redis_client: Optional[redis.Redis] = None

        self.success_count = 0
        self.failure_count = 0

        self.thread: Optional[Thread] = None
        self.stop_event: Optional[Event] = None

        self.frame_lock = Lock()
        # 最新位姿 (TCP): [x, y, z, qw, qx, qy, qz] 右手坐标系
        self.latest_pose_left: Optional[np.ndarray] = None
        self.latest_pose_right: Optional[np.ndarray] = None
        self.latest_frame_id = 0
        self.new_frame_event = Event()

    @property
    def is_connected(self) -> bool:
        return (
            self.redis_client is not None
            and self.thread is not None
            and self.thread.is_alive()
        )

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
            # 测试连接
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

        total_reads = self.success_count + self.failure_count
        if total_reads > 0:
            failure_rate = self.failure_count / total_reads * 100
            logger.info(
                f"{self.tracker_name} 统计: "
                f"成功={self.success_count}, 失败={self.failure_count}, "
                f"失败率={failure_rate:.2f}%"
            )

        logger.info(f"{self.tracker_name} 已断开。")

    def _receive_and_parse(self) -> Optional[Dict]:
        """
        从 Redis 接收并解析手柄数据。
        Returns:
            Dict，包含 'left' 和 'right' 键:
            - 7D 位姿: [x, y, z, qw, qx, qy, qz] 右手坐标系
        """
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

        except json.JSONDecodeError as e:
            logger.warning(f"{self.tracker_name}: JSON 解析失败: {e}")
            return None
        except redis.RedisError as e:
            logger.error(f"{self.tracker_name}: Redis 错误: {e}")
            return None

    def _processing_loop(self):
        """后台线程：持续从 Redis 接收消息，应用 TCP 变换。"""
        frame_id = 0

        logger.info(f"{self.tracker_name}: 后台线程已启动。")

        while not self.stop_event.is_set():
            try:
                data = self._receive_and_parse()

                if data is not None:
                    pose_left_raw = data['left']
                    pose_right_raw = data['right']
                    
                    # 应用 TCP 变换 - Left
                    mat_left_raw = pose_7d_to_4x4matrix(pose_left_raw)
                    mat_left_tcp = mat_left_raw @ T_htc_to_flange.A @ T_htc_to_tcp.A
                    pose_left_tcp = matrix4x4_to_pose_7d(mat_left_tcp)

                    # 应用 TCP 变换 - Right
                    mat_right_raw = pose_7d_to_4x4matrix(pose_right_raw)
                    mat_right_tcp = mat_right_raw @ T_htc_to_flange.A @ T_htc_to_tcp.A
                    pose_right_tcp = matrix4x4_to_pose_7d(mat_right_tcp)

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
                logger.error(
                    f"{self.tracker_name}: 后台线程错误: {e}",
                    exc_info=True
                )
                time.sleep(0.1)

        logger.info(f"{self.tracker_name}: 后台线程已停止。")

    def _start_thread(self):
        if self.thread is not None and self.thread.is_alive():
            logger.warning(f"{self.tracker_name}: 线程已在运行，先停止...")
            self._stop_thread()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._processing_loop,
            name=f"{self.tracker_name}_processing",
            daemon=True
        )
        self.thread.start()

        logger.info(f"{self.tracker_name}: 后台线程已启动。")

    def _stop_thread(self):
        """停止后台线程。"""
        if self.stop_event is None:
            return

        self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)

            if self.thread.is_alive():
                logger.error(f"{self.tracker_name}: 线程未能正常停止!")
            else:
                logger.info(f"{self.tracker_name}: 线程已停止。")

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 500) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        异步读取最新手柄 TCP 位姿。
        Returns:
            (pose_left, pose_right, frame_id)
            - pose_left: np.ndarray, shape (7,), [x, y, z, qw, qx, qy, qz] TCP 位姿
            - pose_right: np.ndarray, shape (7,), [x, y, z, qw, qx, qy, qz] TCP 位姿
            - frame_id: int, 帧编号
        """
        if not self.is_connected:
            raise RuntimeError(f"{self.tracker_name} 未连接。")

        if self.thread is None or not self.thread.is_alive():
            logger.warning(f"{self.tracker_name}: 线程未运行，正在重启...")
            self._start_thread()
            time.sleep(0.5)

        if self.allow_repeat_frames:
            self.new_frame_event.wait(timeout=timeout_ms / 1000.0)

            with self.frame_lock:
                pose_left = self.latest_pose_left
                pose_right = self.latest_pose_right
                frame_id = self.latest_frame_id

            if pose_left is None or pose_right is None:
                raise RuntimeError(f"{self.tracker_name}: 无可用数据。")

            return pose_left, pose_right, frame_id

        else:
            if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
                raise TimeoutError(f"{self.tracker_name}: 超时 {timeout_ms}ms。")

            with self.frame_lock:
                pose_left = self.latest_pose_left
                pose_right = self.latest_pose_right
                frame_id = self.latest_frame_id
                self.new_frame_event.clear()

            if pose_left is None or pose_right is None:
                raise RuntimeError(f"{self.tracker_name}: 无可用数据。")

            return pose_left, pose_right, frame_id

    def get_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取最新手柄 TCP 位姿。
        Returns:
            (pose_left, pose_right): 元组，每个为 np.ndarray, shape (7,), [x, y, z, qw, qx, qy, qz] 或 None
        """
        try:
            pose_left, pose_right, _ = self.async_read(timeout_ms=1000)
            return pose_left.copy(), pose_right.copy()
        except (RuntimeError, TimeoutError):
            left = self.latest_pose_left.copy() if self.latest_pose_left is not None else None
            right = self.latest_pose_right.copy() if self.latest_pose_right is not None else None
            return left, right

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"PicoControllerAsync(name='{self.tracker_name}', "
            f"host='{self.host}', port={self.port}, "
            f"connected={self.is_connected})"
        )


# 默认配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
TEST_FRAMES = 100


def main():
    """测试：连续读取手柄 TCP 数据"""

    print("\n" + "=" * 80)
    print(" " * 20 + "Pico 手柄 TCP 异步读取测试")
    print("=" * 80)
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"测试帧数: {TEST_FRAMES}")
    print("=" * 80)

    pico = PicoControllerAsync(host=REDIS_HOST, port=REDIS_PORT)

    try:
        pico.connect()

        print("\n[1/3] 连接中...")
        print("✓ 连接成功")

        print("\n[2/3] 等待首帧数据 (1秒)...")
        time.sleep(1.0)

        pose_left, pose_right, frame_id = pico.async_read(timeout_ms=5000)

        print(f"✓ 首帧读取成功")
        print(f"  - 帧编号: {frame_id}")

        print(f"\n[3/3] 采集 {TEST_FRAMES} 帧...")
        poses_left = []
        poses_right = []
        frame_ids = []
        latencies = []
        start_time = time.time()

        for i in range(TEST_FRAMES):
            read_start = time.perf_counter()
            pose_left, pose_right, frame_id = pico.async_read(timeout_ms=1000)
            latency = (time.perf_counter() - read_start) * 1000

            poses_left.append(pose_left)
            poses_right.append(pose_right)
            frame_ids.append(frame_id)
            latencies.append(latency)

            time.sleep(1.0 / 30)

        elapsed = time.time() - start_time
        unique_frames = len(set(frame_ids))

        print(f"\n✓ 采集完成")
        print(f"  - 总耗时: {elapsed:.2f}s")
        print(f"  - 读取速率: {TEST_FRAMES/elapsed:.1f} fps (目标 30 fps)")
        print(f"  - 读取延迟: 平均={np.mean(latencies):.2f}ms, 最大={np.max(latencies):.1f}ms")
        print(f"  - 唯一帧数: {unique_frames} / {TEST_FRAMES}")

        x, y, z, qw, qx, qy, qz = poses_right[-1]
        print(f"\n最后一帧 TCP 位姿 (右手柄):")
        print(f"  位置: [x={x:.4f}, y={y:.4f}, z={z:.4f}]")
        print(f"  四元数 [w,x,y,z]: [w={qw:.4f}, x={qx:.4f}, y={qy:.4f}, z={qz:.4f}]")

        x, y, z, qw, qx, qy, qz = poses_left[-1]
        print(f"\n最后一帧 TCP 位姿 (左手柄):")
        print(f"  位置: [x={x:.4f}, y={y:.4f}, z={z:.4f}]")
        print(f"  四元数 [w,x,y,z]: [w={qw:.4f}, x={qx:.4f}, y={qy:.4f}, z={qz:.4f}]")

    except KeyboardInterrupt:
        print("\n\n✓ 用户中断。")
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
    finally:
        pico.disconnect()

    print("\n" + "=" * 80)
    print(" " * 30 + "✓ 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    main()
