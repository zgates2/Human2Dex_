"""
Franka 控制接口提取（基于 reactive_diffusion_policy）

底层实现: deoxys.FrankaInterface + FlexivController (OSC_POSE 笛卡尔控制)
主实现文件: reactive_diffusion_policy/real_world/robot/single_flexiv_controller.py

位姿约定:
  - control 7D: [x, y, z, ax, ay, az, gripper]  (轴角, gripper=-1 表示不控夹爪)
  - tcp 7D (状态/数据): [x, y, z, qw, qx, qy, qz]
  - 6D (pkl/轨迹): [x, y, z, roll, pitch, yaw] (xyz 欧拉角, rad)
  - O_T_EE: Franka 状态 4x4 需 reshape(4,4).transpose() 后使用
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np

# 优先使用本仓库内的 reactive_diffusion_policy
_REPO_ROOT = Path(__file__).resolve().parent
_RDP = _REPO_ROOT / "reactive_diffusion_policy"
if _RDP.is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reactive_diffusion_policy.real_world.robot.single_flexiv_controller import (  # noqa: E402
    FlexivController,
)
from reactive_diffusion_policy.common.space_utils import (  # noqa: E402
    matrix4x4_to_pose_6d,
    matrix4x4_to_pose_7d,
    pose_6d_to_4x4matrix,
    pose_6d_to_pose_7d,
    pose_7d_to_4x4matrix,
)

__all__ = [
    "FlexivController",
    "FrankaControlAPI",
    "get_ee_matrix",
    "se3_to_control_7d",
    "control_7d_from_matrix",
    "pose_utils",
]


@runtime_checkable
class FrankaControlAPI(Protocol):
    """Franka 控制对外 API（FlexivController 已实现）"""

    robot_interface: object
    control_frequency: int
    last_valid_target_pose: Optional[List[float]]

    def reset_to_home(self) -> None: ...
    def absolute_control(self, target_pose: List[float]) -> None: ...
    def set_target_pose(self, target_pose: List[float]) -> None: ...
    def get_current_pose(self) -> np.ndarray: ...
    def get_current_q(self) -> List[float]: ...
    def get_current_robot_states(self) -> dict: ...
    def close(self) -> None: ...


def get_ee_matrix(controller: FlexivController) -> np.ndarray:
    """读取当前末端 4x4 齐次变换矩阵 (base frame)。"""
    return np.array(controller.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()


def se3_to_control_7d(se3, gripper: float = -1.0) -> List[float]:
    """SE3 位姿 -> deoxys OSC 7D: [xyz, axisangle, gripper]。"""
    from spatialmath.base import tr2angvec

    theta, v = tr2angvec(se3.R)
    axisangle = (v * theta).flatten()
    return se3.t.tolist() + axisangle.tolist() + [gripper]


def control_7d_from_matrix(T: np.ndarray, gripper: float = -1.0) -> List[float]:
    """4x4 矩阵 -> control 7D。"""
    from spatialmath import SE3

    return se3_to_control_7d(SE3(T, check=False), gripper=gripper)


class _PoseUtils:
    pose_6d_to_4x4matrix = staticmethod(pose_6d_to_4x4matrix)
    matrix4x4_to_pose_6d = staticmethod(matrix4x4_to_pose_6d)
    pose_6d_to_pose_7d = staticmethod(pose_6d_to_pose_7d)
    pose_7d_to_4x4matrix = staticmethod(pose_7d_to_4x4matrix)
    matrix4x4_to_pose_7d = staticmethod(matrix4x4_to_pose_7d)


pose_utils = _PoseUtils()
