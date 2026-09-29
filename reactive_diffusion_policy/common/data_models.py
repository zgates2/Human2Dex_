from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from enum import Enum, auto
from typing import List, final, Optional

# class HandMes(BaseModel):
#     wristPos: List[float]  # (x, y, z)
#     wristQuat: List[float]  # (w, qx, qy, qz)
#     triggerState:float
#     # # for left controller (B, A, joystick, trigger, side_trigger)
#     # # for right controller (Y, X, joystick, trigger, side_trigger)
#     buttonState: List[bool]

# class UnityMes(BaseModel):
#     timestamp: float
#     leftHand: HandMes
#     rightHand: HandMes

class Arrow(BaseModel):
    start: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    end: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

class TactileSensorMessage(BaseModel):
    device_id: str
    arrows: List[Arrow]
    scale: List[float] = Field(default_factory=lambda: [0.01, 0.005, 0.005])  # meters (sphere radius, arrow x scale, arrow y scale)

# class ForceSensorMessage(BaseModel):
#     device_id: str
#     arrow: Arrow
#     scale: List[float] = Field(default_factory=lambda: [0.01, 0.005, 0.005])  # meters (sphere radius, arrow x scale, arrow y scale)

# class BimanualRobotStates(BaseModel):
#     leftRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
#     rightRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
#     leftRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
#     rightRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
#     leftRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
#     rightRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
#     leftGripperState: List[float] = [0.0] * 2  # (2) (width, force)
#     rightGripperState: List[float] = [0.0] * 2  # (2) (width, force)

class BimanualRobotStates(BaseModel):
    # leftRobotTCP: List[List[float]]
    leftRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
    # rightRobotTCP: List[float] = [0.0] * 7  # (7) (x, y, z, qw, qx, qy, qz)
    leftRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
    # rightRobotTCPVel: List[float] = [0.0] * 6  # (6) (vx, vy, vz, wx, wy, wz)
    leftRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
    # rightRobotTCPWrench: List[float] = [0.0] * 6  # (6) (fx, fy, fz, mx, my, mz)
    # leftGripperState: List[float] = [0.0] * 2  # (2) (width, force)
    # leftGripperState: List[float] = [0.0] * 4  # (2) (width, force, width, force)
    leftGripperState: List[float] = [0.0] * 6  # (2) (width, vel, force, width, vel, force)
    leftRobotTCPTarget: List[float] = [0.0] * 7
    leftJoinStates: List[float] = [0.0] * 7
    # rightGripperState: List[float] = [0.0] * 2  # (2) (width, force)
    # rightRobotTCP: List[List[float]]
    # leftRobotQ: List[float]
    # rightRobotQ: List[float]

class RobotStates(BaseModel):
    tcpPose: List[float] = Field(default_factory=lambda: [0.0] * 7)
    tcpVel: List[float] = Field(default_factory=lambda: [0.0] * 6)
    extWrenchInTcp: List[float] = Field(default_factory=lambda: [0.0] * 6)
    # gripperState: List[float] = Field(default_factory=lambda: [0.0] * 2)
    # gripperState: List[float] = Field(default_factory=lambda: [0.0] * 4)   # 0-1 index is motor with force, ID is 2; 2-3 index is motor ID 1.
    gripperState: List[float] = Field(default_factory=lambda: [0.0] * 6)
    tcpPoseTarget: List[float] = Field(default_factory=lambda: [0.0] * 7)
    joinStates: List[float] = Field(default_factory=lambda: [0.0] * 7)

# class TargetTCPMatrixRequest(BaseModel):
#     target_tcp_matrix: List[List[float]]
#     # duration: float = 3.0
class TargetTCPRequest(BaseModel):
    target_tcp: List[float]  # (7) (x, y, z, qw, qx, qy, qz)

class MoveGripperRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')

    width: float
    force_limit: float
    feedforward_torque: Optional[float] = 0.0  # 前馈力矩，用于MIT模式阻抗控制
    
    # velocity: Optional[float] = None

class GraspGripperRequest(BaseModel):
    force_limit: float

# class MoveGripperRequest(BaseModel):
#     width: float = 0.05
#     velocity: float = 10.0
#     force_limit: float = 5.0

# class TargetTCPRequest(BaseModel):
#     target_tcp: List[float]  # (7) (x, y, z, qw, qx, qy, qz)

# class ActionPrimitiveRequest(BaseModel):
#     primitive_cmd: str

class SensorMessage(BaseModel):
    # TODO: adaptable for different dimensions, considering abolishing the 2-D version
    timestamp: float
    leftRobotTCP: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (x, y, z, r, p, y)
    leftRobotTCPTarget: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32)) # (6) (x, y, z, r, p, y)
    # rightRobotTCP: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (x, y, z, r, p, y)
    leftRobotTCPVel: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (vx, vy, vz, wx, wy, wz)
    # rightRobotTCPVel: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (vx, vy, vz, wx, wy, wz)
    leftRobotTCPWrench: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (fx, fy, fz, mx, my, mz)
    # rightRobotTCPWrench: np.ndarray = Field(default_factory=lambda: np.zeros((6, ), dtype=np.float32))  # (6) (fx, fy, fz, mx, my, mz)
    leftRobotGripperState: np.ndarray = Field(default_factory=lambda: np.zeros((2, ), dtype=np.float32))  # (2) gripper (width, force)
    # rightRobotGripperState: np.ndarray = Field(default_factory=lambda: np.zeros((2, ), dtype=np.float32))  # (2) gripper (width, force)
    # externalCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16)) # (N, 6) (x, y, z, r, g, b)
    externalCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    externalCameraDepth: np.ndarray = Field(default_factory=lambda: np.zeros((480, 640), dtype=np.uint16))
    # externalCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16)) # (N, 6) (x, y, z, r, g, b)
    externalCameraPointCloud: Optional[np.ndarray] = Field(default=None)
    
    leftWristCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftWristCameraDepth: np.ndarray = Field(default_factory=lambda: np.zeros((480, 640), dtype=np.uint16))
    # leftWristCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16))  # (N, 6) (x, y, z, r, g, b)
    leftWristCameraPointCloud: Optional[np.ndarray] = Field(default=None)
    
    # rightWristCameraPointCloud: np.ndarray = Field(default_factory=lambda: np.zeros((10, 6), dtype=np.float16))  # (N, 6) (x, y, z, r, g, b)
    # rightWristCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftGripperCameraRGB1: np.ndarray = Field(
        default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftGripperCameraMarker1: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    leftGripperCameraMarkerOffset1: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    leftGripperCameraRGB2: np.ndarray = Field(
        default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    leftGripperCameraMarker2: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    leftGripperCameraMarkerOffset2: np.ndarray = Field(
        default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 2)(x, y,z)
    # rightGripperCameraRGB1: np.ndarray = Field(default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    # rightGripperCameraMarker1: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    # rightGripperCameraMarkerOffset1: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    # rightGripperCameraRGB2: np.ndarray = Field(default_factory=lambda: np.zeros((24, 32, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    # rightGripperCameraMarker2: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    # rightGripperCameraMarkerOffset2: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((63, 3), dtype=np.float32))  # (num_markers, 3)(x, y, z)
    # vCameraImage: np.ndarray = Field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8))  # (H, W, 3) (r, g, b)
    # predictedFullTCPAction: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 3), (N, 6), (N, 9) or (N, 18)
    # predictedFullGipperAction: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 1), (N, 2)
    # predictedPartialTCPAction: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 3), (N, 6), (N, 9) or (N, 18)
    # predictedPartialGipperAction: np.ndarray = Field(
    #     default_factory=lambda: np.zeros((0, 0), dtype=np.float32))  # (N, 1), (N, 2)
    fisheyeCameraRGB: np.ndarray = Field(default_factory=lambda: np.zeros((48, 64, 3), dtype=np.uint8))
    synexensDepth: np.ndarray = Field(default_factory=lambda: np.zeros((240, 320), dtype=np.uint16))
    synexensIR: np.ndarray = Field(default_factory=lambda: np.zeros((240, 320), dtype=np.uint16))
    synexensFrameId: int = 0
    synexensTimestampNs: int = 0

    class Config:
        arbitrary_types_allowed = True

class SensorMessageList(BaseModel):
    sensorMessages: List[SensorMessage]

class SensorMode(Enum):
    single_arm_one_realsense_no_tactile = auto()
    single_arm_one_realsense_two_tactile = auto() # 新增
    single_arm_two_realsense_no_tactile = auto()
    single_arm_two_realsense_two_tactile = auto()
    dual_arm_two_realsense_four_tactile = auto()

# class TeleopMode(Enum):
#     left_arm_6DOF = auto()
#     left_arm_3D_translation = auto()
#     left_arm_3D_translation_Y_rotation = auto()
#     dual_arm_3D_translation = auto()
