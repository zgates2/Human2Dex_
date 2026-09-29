#!/usr/bin/env python3
"""
Publish PICO hand tracking wrist TCP pose to ROS2 for RViz visualization.

Data source:
    xrobotoolkit_sdk via human2dex/teleop/pico_hand.py

Input pose convention from PICO hand tracking:
    [x, y, z, qx, qy, qz, qw] in PICO SDK world tracking frame.

Published ROS frames/topics:
    TF offset:   map -> pico_offset
    TF raw:      pico_offset -> pico_<hand>_wrist_raw
    TF aligned:  pico_offset -> pico_<hand>_wrist_tcp
    PoseStamped: /pico/<hand>/wrist_tcp_pose
    Path:        /pico/<hand>/wrist_tcp_path
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    pass

DEX_TELEOP_DIR = Path(__file__).resolve().parent / "teleop"
if not DEX_TELEOP_DIR.exists():
    DEX_TELEOP_DIR = Path("/home/zjc/Desktop/human2dex/teleop")
if str(DEX_TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(DEX_TELEOP_DIR))

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from nav_msgs.msg import Path as RosPath
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster
except ImportError as exc:
    raise ImportError(
        "ROS2 Python packages are not available. Source ROS2 setup.bash before "
        "running this script."
    ) from exc


LOGGER = logging.getLogger("wrist_tcp_ros2_publisher")


def load_pico_hand_reader():
    try:
        from pico_hand import PicoHandReader
    except ImportError as exc:
        raise ImportError(
            "Cannot import PicoHandReader. Check that /home/zjc/Desktop/human2dex/teleop "
            "exists and that xrobotoolkit_sdk is available in this Python env."
        ) from exc
    return PicoHandReader


# PICO raw wrist local axes:
#   x = right, y = up, z = backward
#
# The target wrist/TCP frame used by DexUMI/DP is:
#   x = up, y = right, z = forward
#
# R_world_tcp = R_world_pico_wrist @ PICO_WRIST_TO_TARGET_TCP_ROT.
PICO_WRIST_TO_TARGET_TCP_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
PICO_OFFSET_POSE_XYZW = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def normalize_quat_xyzw(quat: np.ndarray) -> Optional[np.ndarray]:
    """Return normalized [qx, qy, qz, qw], or None if invalid."""
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-9:
        return None
    return quat / norm


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert [qx, qy, qz, qw] to a 3x3 rotation matrix."""
    quat = normalize_quat_xyzw(quat_xyzw)
    if quat is None:
        raise ValueError("invalid wrist quaternion")
    qx, qy, qz, qw = quat
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to normalized [qx, qy, qz, qw]."""
    rot = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = normalize_quat_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))
    if quat is None:
        raise ValueError("invalid rotation matrix")
    return quat


def _wrist_pose_to_matrix(raw26x7: np.ndarray) -> np.ndarray:
    """
    构造 wrist 基准 4x4 位姿矩阵。

    PICO[1] 是 wrist；输入四元数为 [qx,qy,qz,qw]。

    输出矩阵语义：
        T_picoWorld_wristTcp

    其中平移部分直接使用 wrist 的世界坐标，旋转部分先由 PICO 原始 wrist
    quaternion 得到，再右乘 PICO_WRIST_TO_TARGET_TCP_ROT，将局部轴从
    “x右/y上/z后”重定义为“x上/y右/z前”。

    这里没有做 PICO world 到机器人 base 的外参变换。原因是当前数据主要用于
    DexUMI/DP 的相对位姿训练，训练时通常会再计算 inv(T0) @ Tt；固定 world
    外参会在相对变换中抵消。但局部 TCP frame 的轴定义会影响相对平移和旋转轴
    的含义，所以必须在这里统一。
    """
    wrist_pose = np.asarray(raw26x7[1, :7], dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_xyzw_to_rotmat(wrist_pose[3:7]) @ PICO_WRIST_TO_TARGET_TCP_ROT
    mat[:3, 3] = wrist_pose[:3]
    return mat.astype(np.float32)


def matrix_to_pose_xyzw(mat: np.ndarray) -> np.ndarray:
    """Convert a homogeneous matrix to [x, y, z, qx, qy, qz, qw]."""
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = mat[:3, 3]
    pose[3:] = _rotmat_to_quat_xyzw(mat[:3, :3])
    return pose


class WristTcpRos2Publisher(Node):
    """ROS2 node that publishes one PICO hand wrist TCP pose."""

    def __init__(
        self,
        hand: str,
        hz: float,
        root_frame: str,
        parent_frame: str,
        child_frame: str,
        raw_child_frame: str,
        publish_path: bool,
        path_length: int,
        publish_invalid: bool,
        debug: bool,
    ):
        super().__init__(f"pico_{hand}_wrist_tcp_publisher")

        self.hand = hand
        self.root_frame = root_frame
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.raw_child_frame = raw_child_frame
        self.publish_path = publish_path
        self.path_length = max(1, int(path_length))
        self.publish_invalid = publish_invalid
        self.debug = debug

        pico_hand_reader_cls = load_pico_hand_reader()
        self.reader = pico_hand_reader_cls(hand=hand)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(
            PoseStamped,
            f"/pico/{hand}/wrist_tcp_pose",
            10,
        )
        self.path_pub = self.create_publisher(
            RosPath,
            f"/pico/{hand}/wrist_tcp_path",
            10,
        )
        self.path_msg = RosPath()
        self.path_msg.header.frame_id = parent_frame

        self.frame_count = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.last_sdk_timestamp_ns: Optional[int] = None

        period = 1.0 / max(float(hz), 1.0)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Publishing PICO {hand} wrist TCP pose: offset TF {root_frame} -> "
            f"{parent_frame}, raw TF {parent_frame} -> {raw_child_frame}, "
            f"aligned TF {parent_frame} -> {child_frame}, "
            f"pose topic /pico/{hand}/wrist_tcp_pose, "
            f"path topic /pico/{hand}/wrist_tcp_path"
        )

    def close(self) -> None:
        self.reader.close()

    def _on_timer(self) -> None:
        self.frame_count += 1

        try:
            raw26x7, active = self.reader.read_raw()
            sdk_ts_ns = self.reader.get_timestamp_ns()
        except Exception as exc:
            self.invalid_count += 1
            self.get_logger().warning(f"Failed to read PICO hand tracking: {exc}")
            return

        if (
            raw26x7 is None
            or raw26x7.ndim != 2
            or raw26x7.shape[0] != 26
            or raw26x7.shape[1] < 7
        ):
            self.invalid_count += 1
            self._maybe_log_invalid(active, "bad raw shape")
            return

        if active != 1 and not self.publish_invalid:
            self.invalid_count += 1
            self._maybe_log_invalid(active, "inactive hand tracking")
            return

        wrist = raw26x7[1, :7].astype(np.float64)
        if not np.all(np.isfinite(wrist)):
            self.invalid_count += 1
            self._maybe_log_invalid(active, "non-finite wrist pose")
            return

        if normalize_quat_xyzw(wrist[3:7]) is None:
            self.invalid_count += 1
            self._maybe_log_invalid(active, "invalid wrist quaternion")
            return

        wrist_tcp_mat = _wrist_pose_to_matrix(raw26x7)
        wrist_tcp_pose = matrix_to_pose_xyzw(wrist_tcp_mat)

        stamp = self.get_clock().now().to_msg()
        offset_tf_msg = self._make_tf_msg(
            stamp,
            self.root_frame,
            self.parent_frame,
            PICO_OFFSET_POSE_XYZW,
        )
        raw_tf_msg = self._make_tf_msg(
            stamp,
            self.parent_frame,
            self.raw_child_frame,
            wrist,
        )
        aligned_tf_msg = self._make_tf_msg(
            stamp,
            self.parent_frame,
            self.child_frame,
            wrist_tcp_pose,
        )
        pose_msg = self._make_pose_msg(stamp, wrist_tcp_pose)

        self.tf_broadcaster.sendTransform(offset_tf_msg)
        self.tf_broadcaster.sendTransform(raw_tf_msg)
        self.tf_broadcaster.sendTransform(aligned_tf_msg)
        self.pose_pub.publish(pose_msg)

        if self.publish_path:
            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose_msg)
            if len(self.path_msg.poses) > self.path_length:
                self.path_msg.poses = self.path_msg.poses[-self.path_length :]
            self.path_pub.publish(self.path_msg)

        self.valid_count += 1
        self.last_sdk_timestamp_ns = sdk_ts_ns

        if self.debug and self.valid_count % 100 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count} valid={self.valid_count} "
                f"invalid={self.invalid_count} "
                f"raw_wrist=[{wrist[0]:.3f}, {wrist[1]:.3f}, {wrist[2]:.3f}] "
                f"wrist_tcp=[{wrist_tcp_pose[0]:.3f}, {wrist_tcp_pose[1]:.3f}, {wrist_tcp_pose[2]:.3f}]"
            )

    def _maybe_log_invalid(self, active: int, reason: str) -> None:
        if self.debug and self.invalid_count % 100 == 1:
            self.get_logger().warning(
                f"Skipping frame {self.frame_count}: {reason}, active={active}"
            )

    def _make_pose_msg(
        self,
        stamp,
        pose_xyzw: np.ndarray,
    ) -> PoseStamped:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.parent_frame
        pose_msg.pose.position.x = float(pose_xyzw[0])
        pose_msg.pose.position.y = float(pose_xyzw[1])
        pose_msg.pose.position.z = float(pose_xyzw[2])
        pose_msg.pose.orientation.x = float(pose_xyzw[3])
        pose_msg.pose.orientation.y = float(pose_xyzw[4])
        pose_msg.pose.orientation.z = float(pose_xyzw[5])
        pose_msg.pose.orientation.w = float(pose_xyzw[6])
        return pose_msg

    def _make_tf_msg(
        self,
        stamp,
        parent_frame: str,
        child_frame: str,
        pose_xyzw: np.ndarray,
    ) -> TransformStamped:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = float(pose_xyzw[0])
        tf_msg.transform.translation.y = float(pose_xyzw[1])
        tf_msg.transform.translation.z = float(pose_xyzw[2])
        tf_msg.transform.rotation.x = float(pose_xyzw[3])
        tf_msg.transform.rotation.y = float(pose_xyzw[4])
        tf_msg.transform.rotation.z = float(pose_xyzw[5])
        tf_msg.transform.rotation.w = float(pose_xyzw[6])
        return tf_msg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish PICO hand tracking wrist TCP pose to ROS2 TF/Pose/Path."
    )
    parser.add_argument(
        "--hand",
        choices=("right", "left"),
        default="right",
        help="PICO hand tracking side to publish.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=60.0,
        help="Read/publish frequency.",
    )
    parser.add_argument(
        "--root-frame",
        default="map",
        help="Root frame for the RViz TF tree. Defaults to map.",
    )
    parser.add_argument(
        "--parent-frame",
        default="pico_offset",
        help="Parent frame for wrist TFs. Defaults to pico_offset.",
    )
    parser.add_argument(
        "--child-frame",
        default=None,
        help="Aligned child frame name. Defaults to pico_<hand>_wrist_tcp.",
    )
    parser.add_argument(
        "--raw-child-frame",
        default=None,
        help="Raw wrist child frame name. Defaults to pico_<hand>_wrist_raw.",
    )
    parser.add_argument(
        "--path-length",
        type=int,
        default=1000,
        help="Maximum number of poses kept in the published Path.",
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="Disable nav_msgs/Path publishing.",
    )
    parser.add_argument(
        "--publish-invalid",
        action="store_true",
        help="Publish even when PICO reports active != 1.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print periodic read statistics.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = build_arg_parser()
    args, ros_args = parser.parse_known_args()

    child_frame = args.child_frame or f"pico_{args.hand}_wrist_tcp"
    raw_child_frame = args.raw_child_frame or f"pico_{args.hand}_wrist_raw"

    rclpy.init(args=[sys.argv[0], *ros_args])
    node = WristTcpRos2Publisher(
        hand=args.hand,
        hz=args.hz,
        root_frame=args.root_frame,
        parent_frame=args.parent_frame,
        child_frame=child_frame,
        raw_child_frame=raw_child_frame,
        publish_path=not args.no_path,
        path_length=args.path_length,
        publish_invalid=args.publish_invalid,
        debug=args.debug,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if rclpy.ok():
            raise
        LOGGER.info("ROS context already shut down: %s", exc)
    finally:
        LOGGER.info(
            "Stopping. frames=%d valid=%d invalid=%d last_sdk_timestamp_ns=%s",
            node.frame_count,
            node.valid_count,
            node.invalid_count,
            node.last_sdk_timestamp_ns,
        )
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
