#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pysurvive
import sys
import numpy as np
import transformations as tft
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

Rz = tft.rotation_matrix(np.pi/4, (0, 0, 1))
Rz_180 = tft.rotation_matrix(np.pi, (0, 0, 1))
# tool_offset = tft.translation_matrix(np.array([[0], [0], [0.156]]))
# tool_offset = np.array([[0], [0], [0.156]])
tool_offset = np.array([[0], [0], [0.140]])
tool_offset_mat = np.eye(4)
tool_offset_mat[:3, 3] = tool_offset.flatten()

class ViveForceSensorTracker:
    def __init__(self, args=None, stabilization_frame_count: int = 30, pub=False):
        if args is None:
            args = sys.argv
        
        try:
            self.actx = pysurvive.SimpleContext(args)
            print("pysurvive 上下文初始化成功。")
            for obj in self.actx.Objects():
                print(f"  - 检测到追踪器: {str(obj.Name(), 'utf-8')}")
        except Exception as e:
            print(f"pysurvive 初始化失败: {e}")
            sys.exit(1)

        Ry = tft.rotation_matrix(np.pi/4, (0, 1, 0))

        Rz = tft.rotation_matrix(np.pi/2, (0, 0, 1))

        Rx = tft.rotation_matrix(-np.pi/2, (1, 0, 0))

        R = np.dot(np.dot(Ry, Rz), Rx)
        
        T = np.eye(4)
        T[:3, :3] = R[:3, :3]
        T[:3, 3] = [0.128673204636989, 0, -0.000668236302979977]
        
        self.T_vive_to_force = np.linalg.inv(T)
        
        self.last_pose = None
        self.last_mat = None
        self.inital_pos = None
        
        self.stabilization_threshold = stabilization_frame_count
        self.frame_count = 0
        self.is_stabilized = False
        print(f"Vive Tracker 将在捕获 {self.stabilization_threshold} 帧有效数据后进入稳定状态。")
        
        # Initialize ROS2 TF publishing if enabled
        self.pub = pub
        if self.pub:
            rclpy.init()
            self.node = Node('vive_tracker')
            self.tf_broadcaster = TransformBroadcaster(self.node)
            print("ROS2 TF publishing enabled.")
        
        # print("坐标变换矩阵设置完成。")

    @staticmethod
    def _pose_to_matrix(pos, quat_wxyz):
        T = tft.quaternion_matrix(quat_wxyz)
        T[:3, 3] = pos
        return T

    def _publish_tf(self, parent_frame, child_frame, pos, quat_wxyz):
        """Publish a TF transform from parent_frame to child_frame"""
        if not self.pub:
            return
            
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        
        # Position
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        
        # Quaternion (w, x, y, z)
        t.transform.rotation.w = float(quat_wxyz[0])
        t.transform.rotation.x = float(quat_wxyz[1])
        t.transform.rotation.y = float(quat_wxyz[2])
        t.transform.rotation.z = float(quat_wxyz[3])
        
        self.tf_broadcaster.sendTransform(t)

    def get_force_sensor_pose(self):
        if not self.actx.Running():
            return self.last_pose if self.is_stabilized else None
        
        updated = self.actx.NextUpdated()
        if not updated:
            return self.last_pose if self.is_stabilized else None
            
        try:
            pose_obj = updated.Pose()
            pose_data = pose_obj[0]
            if self.is_stabilized:
                vive_pos = np.array(pose_data.Pos) - self.inital_pos # np.array([0, 0, 18.00])
            else:
                vive_pos = pose_data.Pos
            # vive_pos = pose_data.Pos - self.inital_pos # np.array([0, 0, 18.00])
            vive_quat_wxyz = pose_data.Rot
            if not self.is_stabilized:
                self.frame_count += 1
                print(f"\r正在等待 Vive Tracker 稳定: {self.frame_count}/{self.stabilization_threshold}...", end="")
                self.inital_pos = np.array(vive_pos)
                if self.frame_count >= self.stabilization_threshold:
                    self.is_stabilized = True
                    print("\nVive Tracker 已稳定, 开始输出位姿数据。")
                else:
                    return None
            T_map_vive = self._pose_to_matrix(vive_pos, vive_quat_wxyz)

            T_map_force = np.dot(T_map_vive, self.T_vive_to_force) # 之前手持采集
            # T_map_force = np.dot(np.dot(T_map_vive, self.T_vive_to_force), Rz) # 与 franka 同向
            # T_map_force = np.dot(np.dot(np.dot(T_map_vive, self.T_vive_to_force), Rz), Rz_180) # 面对 franka 操作
            T_map_tcp = np.dot(T_map_force, tool_offset_mat) # 添加 tool_offset
            tcp_pos = T_map_tcp[:3, 3]
            tcp_quat_wxyz = tft.quaternion_from_matrix(T_map_tcp)
            
            # Publish TF transforms if enabled
            if self.pub:
                # Publish vive frame (world -> vive)
                self._publish_tf("world", "vive", vive_pos, vive_quat_wxyz)

                # Publish force frame (world -> force)
                force_pos = T_map_force[:3, 3]
                force_quat_wxyz = tft.quaternion_from_matrix(T_map_force)
                self._publish_tf("world", "force", force_pos, force_quat_wxyz)
                
                # Publish tcp frame (world -> tcp)
                self._publish_tf("world", "tcp", tcp_pos, tcp_quat_wxyz)
            
            self.last_pose = list(tcp_pos) + list(tcp_quat_wxyz)
            self.last_mat = T_map_tcp
            # return self.last_pose, self.last_mat
            return self.last_pose

        except Exception as e:
            print(f"处理Vive数据时出错: {e}, 返回上一次的Vive数据")
            return self.last_pose

    def shutdown(self):
        print("\n正在关闭 pysurvive 上下文...")
        self.actx.close()
        if self.pub:
            print("正在关闭 ROS2 节点...")
            self.node.destroy_node()
            rclpy.shutdown()
        print("已关闭。")

def main():
    # Example usage with TF publishing enabled
    # tracker = ViveForceSensorTracker(pub=True)
    tracker = ViveForceSensorTracker(pub=True)
    
    time.sleep(2)
    
    print_interval = 1
    last_print_time = time.time()
    
    try:
        print("\n开始追踪... 按 Ctrl+C 退出。")
        while True:
            pose_data = tracker.get_force_sensor_pose()
            
            current_time = time.time()
            
            if (current_time - last_print_time) >= print_interval:
                if pose_data:
                    x, y, z, w, qx, qy, qz = pose_data
                    print(f"力传感器位姿 | "
                        f"Pos: [x={x:.4f}, y={y:.4f}, z={z:.4f}] | "
                        f"Quat: [w={w:.4f}, x={qx:.4f}, y={qy:.4f}, z={qz:.4f}]",
                        end='\n')
                
                last_print_time = current_time
            time.sleep(0.001)
        
    except KeyboardInterrupt:
        pass
    finally:
        tracker.shutdown()

if __name__ == "__main__":
    main()