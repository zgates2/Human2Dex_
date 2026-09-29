#!/usr/bin/env python3
"""Wuji Hand qpos FK and MediaPipe-style 21-point skeleton helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_WUJI_URDF = (
    "/home/zjc/wuji_demo/wuji-retargeting/wuji_retargeting/"
    "wuji_hand_description/urdf/right.urdf"
)

WUJI_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
WUJI_ACTIVE_JOINTS = [
    f"finger{finger_index}_joint{joint_index}"
    for finger_index in range(1, 6)
    for joint_index in range(1, 5)
]

HAND21_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

HAND21_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS_BGR = {
    "thumb": (80, 180, 255),
    "index": (80, 255, 120),
    "middle": (255, 210, 80),
    "ring": (255, 120, 180),
    "pinky": (180, 120, 255),
}

POINT21_COLORS_BGR = [
    (245, 245, 245),
    *([FINGER_COLORS_BGR["thumb"]] * 4),
    *([FINGER_COLORS_BGR["index"]] * 4),
    *([FINGER_COLORS_BGR["middle"]] * 4),
    *([FINGER_COLORS_BGR["ring"]] * 4),
    *([FINGER_COLORS_BGR["pinky"]] * 4),
]

# finger1 is thumb, finger2 is index, then middle/ring/pinky.  The mapping
# follows wuji-retargeting/example/config/vector/*.yaml: link3/link4/tip are
# the PIP/DIP/TIP targets for four fingers, and MCP is the first finger link.
CALIBRATION_ANCHOR_TO_LINK = {
    "thumb_cmc": "finger1_link1",
    "thumb_mcp": "finger1_link3",
    "thumb_ip": "finger1_link4",
    "thumb_tip": "finger1_tip_link",
    "index_mcp": "finger2_link1",
    "index_pip": "finger2_link3",
    "index_dip": "finger2_link4",
    "index_tip": "finger2_tip_link",
    "middle_mcp": "finger3_link1",
    "middle_pip": "finger3_link3",
    "middle_dip": "finger3_link4",
    "middle_tip": "finger3_tip_link",
    "ring_mcp": "finger4_link1",
    "ring_pip": "finger4_link3",
    "ring_dip": "finger4_link4",
    "ring_tip": "finger4_tip_link",
    "pinky_mcp": "finger5_link1",
    "pinky_pip": "finger5_link3",
    "pinky_dip": "finger5_link4",
    "pinky_tip": "finger5_tip_link",
}

ANCHOR_TO_POINT21_INDEX = {
    name: HAND21_NAMES.index(name)
    for name in CALIBRATION_ANCHOR_TO_LINK
}

ANCHOR_SETS = {
    "tips": [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ],
    "tips_dips": [
        "thumb_ip", "thumb_tip",
        "index_dip", "index_tip",
        "middle_dip", "middle_tip",
        "ring_dip", "ring_tip",
        "pinky_dip", "pinky_tip",
    ],
    "all_physical": list(CALIBRATION_ANCHOR_TO_LINK),
}


def _parse_vector(text: str | None, length: int, default: Iterable[float]) -> np.ndarray:
    if text is None or not text.strip():
        return np.asarray(list(default), dtype=np.float64).reshape(length)
    values = [float(value) for value in text.split()]
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12 or abs(float(angle)) <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return result


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    mimic_joint: str | None
    mimic_multiplier: float
    mimic_offset: float


class WujiKinematics:
    """Small URDF FK implementation for the Wuji right hand."""

    def __init__(self, urdf_path: str | Path = DEFAULT_WUJI_URDF, hand_base_link: str = "palm_link"):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        self.hand_base_link = str(hand_base_link)
        root = ET.parse(self.urdf_path).getroot()
        self.links = {element.attrib["name"] for element in root.findall("link")}
        self.joints = [self._parse_joint(element) for element in root.findall("joint")]
        child_links = {joint.child for joint in self.joints}
        root_links = sorted(self.links - child_links)
        if len(root_links) != 1:
            raise ValueError(f"Expected one URDF root link, got {root_links}")
        self.root_link = root_links[0]
        self.children: dict[str, list[JointSpec]] = {}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)
        if self.hand_base_link not in self.links:
            raise KeyError(f"Missing hand base link {self.hand_base_link!r}")
        missing = [link for link in CALIBRATION_ANCHOR_TO_LINK.values() if link not in self.links]
        if missing:
            raise KeyError(f"Wuji URDF is missing required calibration links: {missing}")
        self.active_joint_limits = {
            joint.name: (joint.lower, joint.upper)
            for joint in self.joints
            if joint.name in WUJI_ACTIVE_JOINTS
        }

    @staticmethod
    def _parse_joint(element: ET.Element) -> JointSpec:
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        mimic = element.find("mimic")
        return JointSpec(
            name=element.attrib["name"],
            joint_type=element.attrib.get("type", "fixed"),
            parent=element.find("parent").attrib["link"],
            child=element.find("child").attrib["link"],
            origin_xyz=_parse_vector(None if origin is None else origin.attrib.get("xyz"), 3, [0, 0, 0]),
            origin_rpy=_parse_vector(None if origin is None else origin.attrib.get("rpy"), 3, [0, 0, 0]),
            axis=_parse_vector(None if axis is None else axis.attrib.get("xyz"), 3, [1, 0, 0]),
            lower=None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"]),
            upper=None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"]),
            mimic_joint=None if mimic is None else mimic.attrib.get("joint"),
            mimic_multiplier=1.0 if mimic is None else float(mimic.attrib.get("multiplier", 1.0)),
            mimic_offset=0.0 if mimic is None else float(mimic.attrib.get("offset", 0.0)),
        )

    def normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        state = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if state.shape != (20,):
            raise ValueError(f"Wuji qpos must have shape (20,), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("Wuji qpos contains NaN or Inf")
        clipped = state.copy()
        for index, name in enumerate(WUJI_ACTIVE_JOINTS):
            lower, upper = self.active_joint_limits.get(name, (None, None))
            if lower is not None:
                clipped[index] = max(float(lower), clipped[index])
            if upper is not None:
                clipped[index] = min(float(upper), clipped[index])
        return clipped

    def joint_positions(self, qpos: np.ndarray) -> dict[str, float]:
        state = self.normalize_qpos(qpos)
        values = {
            joint_name: float(value)
            for joint_name, value in zip(WUJI_ACTIVE_JOINTS, state)
        }
        unresolved = [joint for joint in self.joints if joint.mimic_joint is not None]
        for _ in range(len(unresolved) + 1):
            next_unresolved = []
            for joint in unresolved:
                if joint.mimic_joint not in values:
                    next_unresolved.append(joint)
                    continue
                value = joint.mimic_multiplier * values[joint.mimic_joint] + joint.mimic_offset
                if joint.lower is not None:
                    value = max(value, joint.lower)
                if joint.upper is not None:
                    value = min(value, joint.upper)
                values[joint.name] = float(value)
            if not next_unresolved:
                break
            if len(next_unresolved) == len(unresolved):
                names = [joint.name for joint in next_unresolved]
                raise RuntimeError(f"Could not resolve mimic joints: {names}")
            unresolved = next_unresolved
        return values

    def link_transforms_in_hand_base(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        joint_values = self.joint_positions(qpos)
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_transform = transforms[parent]
            for joint in self.children.get(parent, []):
                origin_transform = _transform(
                    _rpy_matrix(joint.origin_rpy), joint.origin_xyz
                )
                motion = np.eye(4, dtype=np.float64)
                if joint.joint_type in ("revolute", "continuous"):
                    motion[:3, :3] = _axis_angle_matrix(
                        joint.axis, joint_values.get(joint.name, 0.0)
                    )
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis * joint_values.get(joint.name, 0.0)
                elif joint.joint_type != "fixed":
                    raise NotImplementedError(
                        f"Unsupported Wuji URDF joint type {joint.joint_type!r}"
                    )
                transforms[joint.child] = parent_transform @ origin_transform @ motion
                stack.append(joint.child)
        if self.hand_base_link not in transforms:
            raise RuntimeError(f"FK did not produce {self.hand_base_link}")
        hand_from_root = np.linalg.inv(transforms[self.hand_base_link])
        return {name: hand_from_root @ pose for name, pose in transforms.items()}

    def calibration_anchor_points(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        transforms = self.link_transforms_in_hand_base(qpos)
        return {
            name: transforms[link][:3, 3].copy()
            for name, link in CALIBRATION_ANCHOR_TO_LINK.items()
        }

    def points21(self, qpos: np.ndarray) -> np.ndarray:
        transforms = self.link_transforms_in_hand_base(qpos)

        def point(link: str) -> np.ndarray:
            return transforms[link][:3, 3].copy()

        points = [point(self.hand_base_link)]
        points.extend(
            [
                point("finger1_link1"),
                point("finger1_link3"),
                point("finger1_link4"),
                point("finger1_tip_link"),
            ]
        )
        for finger_index in range(2, 6):
            points.extend(
                [
                    point(f"finger{finger_index}_link1"),
                    point(f"finger{finger_index}_link3"),
                    point(f"finger{finger_index}_link4"),
                    point(f"finger{finger_index}_tip_link"),
                ]
            )
        result = np.stack(points, axis=0)
        if result.shape != (21, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"Invalid Wuji 21-point FK output: {result.shape}")
        return result


def point21_finger(index: int) -> str:
    if index == 0:
        return "wrist"
    if 1 <= index <= 4:
        return "thumb"
    if 5 <= index <= 8:
        return "index"
    if 9 <= index <= 12:
        return "middle"
    if 13 <= index <= 16:
        return "ring"
    if 17 <= index <= 20:
        return "pinky"
    raise IndexError(index)
