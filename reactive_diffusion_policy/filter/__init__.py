#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轨迹滤波模块

包含多种轨迹滤波器，用于对 xyzrpy 格式的轨迹进行滤波处理。

主要模块：
- simple_trajectory_filter: 简化版轨迹滤波器，适合实时应用
- trajectory_filter: 完整版轨迹滤波器，支持多种滤波方法
- vive_tracker_with_filter: 集成到 Vive Tracker 的滤波功能
- filter_example: 使用示例和演示
"""

from .simple_trajectory_filter import SimpleTrajectoryFilter
from .trajectory_filter import TrajectoryFilter
from .rotation_filter import RotationFilter
from .advanced_rotation_filter import AdvancedRotationFilter
from .offline_rotation_filter import OfflineRotationFilter

__version__ = "1.0.0"
__author__ = "Reactive Diffusion Policy Team"

__all__ = [
    'SimpleTrajectoryFilter',
    'TrajectoryFilter',
    'RotationFilter',
    'AdvancedRotationFilter',
    'OfflineRotationFilter'
]
