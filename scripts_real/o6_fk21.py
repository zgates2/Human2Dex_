#!/usr/bin/env python3
"""Linker O6 measured-state FK and MediaPipe-style 21-point skeleton helpers.

This module intentionally uses only NumPy and the Python standard library so
it can run inside the deployment ``umi204`` environment without Pinocchio.
The six O6 values are interpreted with the same command/radian convention as
DexUMI: 250 is open and 0 is fully flexed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np


O6_ACTIVE_JOINTS = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

O6_ACTIVE_UPPER_RAD = {
    "thumb_cmc_pitch": 0.58,
    "thumb_cmc_yaw": 1.36,
    "index_mcp_pitch": 1.60,
    "middle_mcp_pitch": 1.60,
    "ring_mcp_pitch": 1.60,
    "pinky_mcp_pitch": 1.60,
}

HAND21_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip_virtual",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip_virtual",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip_virtual",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip_virtual",
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

# These are physical URDF link-frame origins that a human can identify in RGB.
# The full 21-point skeleton additionally contains virtual PIP samples for the
# four coupled two-segment fingers, but those virtual points are never requested
# as click annotations.
CALIBRATION_ANCHOR_TO_LINK = {
    "thumb_cmc": "thumb_metacarpals_base2",
    "thumb_mcp": "thumb_metacarpals",
    "thumb_ip": "thumb_distal",
    "thumb_tip": "thumb_tip",
    "index_mcp": "index_proximal",
    "index_dip": "index_distal",
    "index_tip": "index_tip",
    "middle_mcp": "middle_proximal",
    "middle_dip": "middle_distal",
    "middle_tip": "middle_tip",
    "ring_mcp": "ring_proximal",
    "ring_dip": "ring_distal",
    "ring_tip": "ring_tip",
    "pinky_mcp": "pinky_proximal",
    "pinky_dip": "pinky_distal",
    "pinky_tip": "pinky_tip",
}

ANCHOR_TO_POINT21_INDEX = {
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_dip": 19,
    "pinky_tip": 20,
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


class O6Kinematics:
    """Small URDF FK implementation for the Linker O6 right hand."""

    def __init__(self, urdf_path: str | Path, hand_base_link: str = "hand_base_link"):
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
            raise KeyError(f"O6 URDF is missing required calibration links: {missing}")

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

    @staticmethod
    def measured_state_to_active_radians(measured_state: np.ndarray) -> dict[str, float]:
        state = np.asarray(measured_state, dtype=np.float64).reshape(-1)
        if state.shape != (6,):
            raise ValueError(f"O6 measured state must have shape (6,), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("O6 measured state contains NaN or Inf")
        state = np.clip(state, 0.0, 250.0)
        result = {}
        for name, raw in zip(O6_ACTIVE_JOINTS, state):
            upper = O6_ACTIVE_UPPER_RAD[name]
            result[name] = float(upper * (1.0 - raw / 250.0))
        return result

    def joint_positions(self, measured_state: np.ndarray) -> dict[str, float]:
        values = self.measured_state_to_active_radians(measured_state)
        unresolved = [joint for joint in self.joints if joint.mimic_joint is not None]
        for _ in range(len(unresolved) + 1):
            next_unresolved = []
            for joint in unresolved:
                if joint.mimic_joint not in values:
                    next_unresolved.append(joint)
                    continue
                value = (
                    joint.mimic_multiplier * values[joint.mimic_joint]
                    + joint.mimic_offset
                )
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

    def link_transforms_in_hand_base(self, measured_state: np.ndarray) -> dict[str, np.ndarray]:
        joint_values = self.joint_positions(measured_state)
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
                        f"Unsupported O6 URDF joint type {joint.joint_type!r}"
                    )
                transforms[joint.child] = parent_transform @ origin_transform @ motion
                stack.append(joint.child)
        if self.hand_base_link not in transforms:
            raise RuntimeError(f"FK did not produce {self.hand_base_link}")
        hand_from_root = np.linalg.inv(transforms[self.hand_base_link])
        return {name: hand_from_root @ pose for name, pose in transforms.items()}

    def calibration_anchor_points(self, measured_state: np.ndarray) -> dict[str, np.ndarray]:
        transforms = self.link_transforms_in_hand_base(measured_state)
        return {
            name: transforms[link][:3, 3].copy()
            for name, link in CALIBRATION_ANCHOR_TO_LINK.items()
        }

    def points21(self, measured_state: np.ndarray) -> np.ndarray:
        transforms = self.link_transforms_in_hand_base(measured_state)

        def point(link: str) -> np.ndarray:
            return transforms[link][:3, 3].copy()

        points = [point(self.hand_base_link)]
        points.extend(
            [
                point("thumb_metacarpals_base2"),
                point("thumb_metacarpals"),
                point("thumb_distal"),
                point("thumb_tip"),
            ]
        )
        for finger in ("index", "middle", "ring", "pinky"):
            mcp = point(f"{finger}_proximal")
            dip = point(f"{finger}_distal")
            tip = point(f"{finger}_tip")
            pip_virtual = 0.5 * (mcp + dip)
            points.extend([mcp, pip_virtual, dip, tip])
        result = np.stack(points, axis=0)
        if result.shape != (21, 3) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"Invalid O6 21-point FK output: {result.shape}")
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
