#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactive_diffusion_policy.real_world.robot.vive_tracker import ViveForceSensorTracker
from filter.simple_trajectory_filter import SimpleTrajectoryFilter
import time
import numpy as np

class ViveTrackerWithFilter:
    """
    带滤波功能的 Vive Tracker
    """
    
    def __init__(self, 
                 args=None, 
                 stabilization_frame_count: int = 50, 
                 pub=False,
                 filter_type: str = 'moving_average',
                 filter_window_size: int = 5):
        """
        初始化带滤波的 Vive Tracker
        
        Args:
            args: pysurvive 参数
            stabilization_frame_count: 稳定化帧数
            pub: 是否发布 ROS2 TF
            filter_type: 滤波类型
            filter_window_size: 滤波窗口大小
        """
        # 初始化 Vive Tracker
        self.vive_tracker = ViveForceSensorTracker(
            args=args, 
            stabilization_frame_count=stabilization_frame_count, 
            pub=pub
        )
        
        # 初始化滤波器
        self.pose_filter = SimpleTrajectoryFilter(
            filter_type=filter_type,
            window_size=filter_window_size
        )
        
        # 用于存储原始和滤波后的位姿
        self.raw_pose = None
        self.filtered_pose = None
        
    def get_filtered_pose(self):
        """
        获取滤波后的位姿
        
        Returns:
            滤波后的位姿 [x, y, z, w, qx, qy, qz] 或 None
        """
        # 获取原始位姿
        raw_pose = self.vive_tracker.get_force_sensor_pose()
        if raw_pose is None:
            return None
        
        self.raw_pose = raw_pose
        
        # 将四元数位姿转换为 xyzrpy 格式进行滤波
        x, y, z, w, qx, qy, qz = raw_pose
        
        # 四元数转 RPY
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([qx, qy, qz, w])  # x,y,z,w
        rpy = r.as_euler('xyz', degrees=False)
        
        # 组合 xyzrpy
        xyzrpy = [x, y, z, rpy[0], rpy[1], rpy[2]]
        
        # 滤波
        filtered_xyzrpy = self.pose_filter.filter_pose(xyzrpy)
        
        # 转换回四元数格式
        filtered_r = R.from_euler('xyz', filtered_xyzrpy[3:], degrees=False)
        filtered_quat = filtered_r.as_quat()  # x,y,z,w
        
        # 组合最终位姿 [x, y, z, w, qx, qy, qz]
        self.filtered_pose = [
            filtered_xyzrpy[0],  # x
            filtered_xyzrpy[1],  # y
            filtered_xyzrpy[2],  # z
            filtered_quat[3],    # w
            filtered_quat[0],    # qx
            filtered_quat[1],    # qy
            filtered_quat[2]     # qz
        ]
        
        return self.filtered_pose
    
    def get_raw_pose(self):
        """获取原始位姿"""
        return self.raw_pose
    
    def reset_filter(self):
        """重置滤波器"""
        self.pose_filter.reset()
    
    def shutdown(self):
        """关闭 tracker"""
        self.vive_tracker.shutdown()


def main():
    """主函数演示"""
    print("Vive Tracker 带滤波功能演示")
    print("=" * 50)
    
    # 创建带滤波的 tracker
    tracker = ViveTrackerWithFilter(
        stabilization_frame_count=30,
        pub=True,  # 启用 ROS2 TF 发布
        filter_type='moving_average',
        filter_window_size=5
    )
    
    time.sleep(2)
    
    print_interval = 1
    last_print_time = time.time()
    
    try:
        print("\n开始追踪... 按 Ctrl+C 退出。")
        print("显示格式: [x, y, z, w, qx, qy, qz]")
        print("-" * 80)
        
        while True:
            # 获取滤波后的位姿
            filtered_pose = tracker.get_filtered_pose()
            raw_pose = tracker.get_raw_pose()
            
            current_time = time.time()
            
            if (current_time - last_print_time) >= print_interval:
                if filtered_pose and raw_pose:
                    print(f"原始位姿: [{', '.join([f'{x:.4f}' for x in raw_pose])}]")
                    print(f"滤波位姿: [{', '.join([f'{x:.4f}' for x in filtered_pose])}]")
                    print("-" * 80)
                
                last_print_time = current_time
            time.sleep(0.001)
        
    except KeyboardInterrupt:
        pass
    finally:
        tracker.shutdown()


if __name__ == "__main__":
    main()
