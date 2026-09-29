#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dex-retargeting 调用封装。

- LinkerO6Retargeter: MANO (21, 3) 关键点 → Linker O6 六个主动关节弧度
- PicoToLinkerO6Retargeter: PICO raw26x7 → MANO 21 点 + O6/Wuji 弧度 + wrist 6D pose

YAML 注意事项
-------------
当前 linker_o6/linker_o6/linker_o6.yml 顶层是 left/right 两个段，并不是
dex-retargeting `load_from_file` 期望的 `retargeting:` 顶层结构；并且 right 段里写了
非标字段 `target_link_human_indices_dexpilot`（dex-retargeting dataclass 不识别）。
本文件采取：
  1. 用 yaml 自行读取 left/right 段
  2. 剥离非标字段 → DexPilot 在 target_link_human_indices=None 时会基于 MediaPipe 21
     关键点自动生成正确的 origin/task 索引
  3. 调用 `RetargetingConfig.from_dict()` 构建 `SeqRetargeting`
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import yaml
try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:  # pragma: no cover - scipy is expected in normal envs
    _Rotation = None

# 把本仓库和 dex-retargeting 源码加入 sys.path，避免要求用户先 pip install
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEX_SRC = _REPO_ROOT / "dex-retargeting" / "src"
if _DEX_SRC.is_dir() and str(_DEX_SRC) not in sys.path:
    sys.path.insert(0, str(_DEX_SRC))

from dex_retargeting.retargeting_config import RetargetingConfig  # noqa: E402
from dex_retargeting.seq_retarget import SeqRetargeting  # noqa: E402


# 与 dex-retargeting 一致的 operator→MANO 矩阵（右手）
_OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float32,
)
OPERATOR2MANO_RIGHT = _OPERATOR2MANO_RIGHT


# PICO 26 → MediaPipe/MANO 21 索引映射。丢弃 Palm 和四指 metacarpal 点。
PICO2MEDIAPIPE = np.array(
    [
        1,                  # 0  wrist        ← PICO Wrist
        2, 3, 4, 5,         # 1-4  thumb cmc/mcp/ip/tip
        7, 8, 9, 10,        # 5-8  index mcp/pip/dip/tip
        12, 13, 14, 15,     # 9-12 middle
        17, 18, 19, 20,     # 13-16 ring
        22, 23, 24, 25,     # 17-20 pinky
    ],
    dtype=int,
)


# PICO 原始 wrist 局部轴：x=右, y=上, z=后。
# 目标 wrist/TCP 局部轴：x=上, y=右, z=前。
# 右乘该矩阵只改变 wrist 局部坐标轴定义，不改变世界系位置。
PICO_WRIST_TO_TARGET_TCP_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PicoRetargetingResult:
    """单帧 PICO raw26x7 重定向结果。"""

    pts21_mano: np.ndarray
    linker_joint_radians: np.ndarray
    wuji_joint_radians: np.ndarray | None
    wrist_pose_6d: np.ndarray


def _validate_pico_raw(raw26x7: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw26x7, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 26 or raw.shape[1] < 7:
        raise ValueError(
            f"期望 PICO raw shape 为 (26, 7) 或 (26, >=7)，收到 {raw.shape}"
        )
    if not np.all(np.isfinite(raw[:, :7])):
        raise ValueError("PICO raw26x7 前 7 列包含 NaN/Inf")
    return raw


def _estimate_wrist_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    用 wrist/index_mcp/middle_mcp 三点估计 operator 手部局部坐标系。

    输入应为已减 wrist 的 MediaPipe 21 点，返回列向量形式的 operator frame。
    """
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"期望 (21, 3) 输入，收到 {keypoint_3d_array.shape}")

    points = keypoint_3d_array[[0, 5, 9], :]
    x_vector = points[0] - points[2]

    points = points - np.mean(points, axis=0, keepdims=True)
    _, _, v = np.linalg.svd(points)
    normal = v[2, :]

    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / (np.linalg.norm(x) + 1e-12)
    z = np.cross(x, normal)

    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1

    return np.stack([x, normal, z], axis=1)


def pico_raw_to_mano(raw26x7: np.ndarray, hand: str = "right") -> np.ndarray:
    """
    PICO raw26x7 → wrist-centered MANO 21 keypoints.

    返回 shape=(21, 3), dtype=float32，单位米。左手目前沿用右手 MANO 对齐流程，
    与现有 `pico_hand.py` 行为保持一致。
    """
    if hand not in ("right", "left"):
        raise ValueError(f"hand 必须为 'right' 或 'left'，收到 {hand!r}")

    raw = _validate_pico_raw(raw26x7)
    pts26 = raw[:, :3]
    pts21_world = pts26[PICO2MEDIAPIPE]
    centered = pts21_world - pts21_world[0:1, :]
    rot = _estimate_wrist_frame(centered)
    return (centered @ rot @ _OPERATOR2MANO_RIGHT).astype(np.float32)


def pico_raw_to_mediapipe(raw26x7: np.ndarray) -> np.ndarray:
    """PICO raw26x7 → MediaPipe 21 keypoints in PICO world frame, meters."""
    raw = _validate_pico_raw(raw26x7)
    return raw[:, :3][PICO2MEDIAPIPE].astype(np.float32)


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """PICO quaternion [qx, qy, qz, qw] → 3x3 rotation matrix."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12 or not np.all(np.isfinite(q)):
        return np.eye(3, dtype=np.float64)

    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pico_wrist_pose_matrix(raw26x7: np.ndarray) -> np.ndarray:
    """
    PICO raw26x7 → wrist/TCP 4x4 pose matrix.

    平移使用 PICO wrist 点 raw[1, :3]；旋转先从 raw[1, 3:7] 的 xyzw
    四元数得到原始 wrist 姿态，再把局部轴从 x右/y上/z后重定义为
    x上/y右/z前。
    """
    raw = _validate_pico_raw(raw26x7)
    wrist_pose = raw[1, :7]
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_xyzw_to_rotmat(wrist_pose[3:7]) @ PICO_WRIST_TO_TARGET_TCP_ROT
    mat[:3, 3] = wrist_pose[:3]
    return mat.astype(np.float32)


def _rotmat_to_rotvec(rotmat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → rotation vector fallback when scipy is unavailable."""
    r = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
    cos_angle = (float(np.trace(r)) - 1.0) * 0.5
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    axis = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=np.float64,
    )
    sin_angle = float(np.sin(angle))
    if abs(sin_angle) > 1e-6:
        axis = axis / (2.0 * sin_angle)
    else:
        idx = int(np.argmax(np.diag(r)))
        axis = np.zeros(3, dtype=np.float64)
        axis[idx] = np.sqrt(max(r[idx, idx] + 1.0, 0.0) * 0.5)
        denom = 4.0 * axis[idx] + 1e-12
        if idx == 0:
            axis[1] = (r[0, 1] + r[1, 0]) / denom
            axis[2] = (r[0, 2] + r[2, 0]) / denom
        elif idx == 1:
            axis[0] = (r[0, 1] + r[1, 0]) / denom
            axis[2] = (r[1, 2] + r[2, 1]) / denom
        else:
            axis[0] = (r[0, 2] + r[2, 0]) / denom
            axis[1] = (r[1, 2] + r[2, 1]) / denom
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    return axis * angle


def matrix_to_pose6d(matrix4x4: np.ndarray) -> np.ndarray:
    """4x4 pose matrix → [x, y, z, rx, ry, rz] rotvec pose."""
    mat = np.asarray(matrix4x4, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"期望 (4, 4) 位姿矩阵，收到 {mat.shape}")
    xyz = mat[:3, 3]
    if _Rotation is None:
        rotvec = _rotmat_to_rotvec(mat[:3, :3])
    else:
        rotvec = _Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([xyz, rotvec]).astype(np.float32)


def pico_wrist_pose6d(raw26x7: np.ndarray) -> np.ndarray:
    """PICO raw26x7 → wrist/TCP [x, y, z, rx, ry, rz] pose."""
    return matrix_to_pose6d(pico_wrist_pose_matrix(raw26x7))


def _estimate_mano_axes_from_three_points(
    wrist: np.ndarray, index_mcp: np.ndarray, middle_mcp: np.ndarray
) -> np.ndarray:
    """
    复刻 dex-retargeting 的 estimate_frame_from_hand_points，但只用 3 个关键点。

    输入是 base_link 系下的 (wrist, index_mcp, middle_mcp) 三个坐标。
    输出: (3, 3) 矩阵，列向量 = MANO 系基向量在 base_link 系下的表示。

    即满足 `vec_base = vec_mano @ R.T`（按 dex-retargeting 的约定）。
    """
    pts = np.stack([wrist, index_mcp, middle_mcp]).astype(np.float64)

    # plane fit
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]

    x_vec = pts[0] - pts[2]  # middle_mcp → wrist 方向
    x = x_vec - np.dot(x_vec, normal) * normal
    x = x / (np.linalg.norm(x) + 1e-9)
    z = np.cross(x, normal)

    # 让 z 大致与 middle → index 同向
    if np.dot(z, pts[1] - pts[2]) < 0:
        normal *= -1
        z *= -1

    op_frame = np.stack([x, normal, z], axis=1)  # 列=operator 系基
    mano_frame = op_frame @ _OPERATOR2MANO_RIGHT.astype(np.float64)
    return mano_frame.astype(np.float32)


# Linker O6 真机/MuJoCo 都按这个顺序使用 6 个主动关节角度
LINKER_JOINTS: List[str] = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


DEFAULT_YAML = "/home/zjc/Desktop/human2dex/linker_o6/linker_o6/linker_o6.yml"
DEFAULT_URDF_DIR = "/home/zjc/Desktop/human2dex/linker_o6"
DEFAULT_WUJI_YAML = str(
    _REPO_ROOT / "wuji_retargeting" / "config" / "adaptive_analytical_pico.yaml"
)

WUJI_JOINTS: List[str] = [
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]


# 在 dataclass 里 RetargetingConfig 的允许字段集合（白名单过滤用）
_ALLOWED_CFG_KEYS = {
    "type",
    "urdf_path",
    "add_dummy_free_joint",
    "target_link_human_indices",
    "wrist_link_name",
    "target_link_names",
    "target_joint_names",
    "target_origin_link_names",
    "target_task_link_names",
    "finger_tip_link_names",
    "scaling_factor",
    "normal_delta",
    "huber_delta",
    "project_dist",
    "escape_dist",
    "has_joint_limits",
    "ignore_mimic_joint",
    "low_pass_alpha",
}


def _resolve_urdf_path(rel_path: str, yaml_path: str) -> str:
    """
    将 yml 中可能写错的相对 URDF 路径解析成绝对路径。

    依次尝试以下候选根目录，命中第一个存在的：
      1. yaml 所在目录（如 yaml 在 linker_o6/linker_o6/ ）
      2. yaml 父目录       （如 /home/zjc//DexUMI/linker_o6）
      3. yaml 父父目录     （如 /home/zjc/Desktop/human2dex）
      4. 仓库根目录         （硬编码兜底）
    """
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return str(p)

    yaml_dir = Path(yaml_path).resolve().parent
    candidates = [
        yaml_dir,
        yaml_dir.parent,
        yaml_dir.parent.parent,
        Path("/home/zjc/Desktop/human2dex"),
    ]
    for root in candidates:
        cand = (root / rel_path).resolve()
        if cand.is_file():
            return str(cand)

    tried = "\n  ".join(str((r / rel_path).resolve()) for r in candidates)
    raise FileNotFoundError(
        f"无法在以下候选位置找到 URDF '{rel_path}':\n  {tried}"
    )


def _load_section(yaml_path: str, hand: str) -> dict:
    """
    从 yml 取出 retargeting 段并清洗未知字段，兼容三种顶层结构：
      - 顶层 `retargeting:` （dex-retargeting 标准）
      - 顶层 `left:` + `right:` （旧版 linker_o6.yml）

    同时把 yml 中可能写错的相对 urdf_path 自动转成绝对路径。
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)

    if not isinstance(all_cfg, dict) or not all_cfg:
        raise ValueError(f"YAML {yaml_path} 内容为空或格式异常")

    if "retargeting" in all_cfg:
        section_key = "retargeting"
    elif hand in all_cfg:
        section_key = hand
    else:
        raise KeyError(
            f"YAML {yaml_path} 中既没有 'retargeting' 也没有 '{hand}' 段，"
            f"可用段: {list(all_cfg.keys())}"
        )

    print(f"[Retargeter] 使用 YAML 段 '{section_key}'")

    raw = dict(all_cfg[section_key])

    # 把相对 urdf_path 解析成绝对路径，避免 dex-retargeting 的 set_default_urdf_dir 拼接歧义
    if "urdf_path" in raw:
        raw["urdf_path"] = _resolve_urdf_path(str(raw["urdf_path"]), yaml_path)
        print(f"[Retargeter] URDF -> {raw['urdf_path']}")

    cleaned = {k: v for k, v in raw.items() if k in _ALLOWED_CFG_KEYS}

    dropped = set(raw.keys()) - set(cleaned.keys())
    if dropped:
        # 已知会被丢弃的有 target_link_human_indices_dexpilot（DexPilot 默认自动生成索引）
        print(f"[Retargeter] 忽略 YAML 中不支持的字段: {sorted(dropped)}")

    return cleaned


class LinkerO6Retargeter:
    """把 (21, 3) MANO 关键点重定向为 Linker O6 六个主动关节弧度。"""

    def __init__(
        self,
        yaml_path: str = DEFAULT_YAML,
        urdf_dir: str = DEFAULT_URDF_DIR,
        hand: str = "right",
    ):
        if hand not in ("right", "left"):
            raise ValueError(f"hand 必须为 'right' 或 'left'，收到 {hand!r}")
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"找不到 YAML: {yaml_path}")
        self.hand = hand

        # _resolve_urdf_path 已把 urdf_path 改为绝对路径，set_default_urdf_dir 仅作兜底
        if os.path.isdir(urdf_dir):
            RetargetingConfig.set_default_urdf_dir(urdf_dir)

        cfg = _load_section(yaml_path, hand)
        config = RetargetingConfig.from_dict(cfg)
        self.seq: SeqRetargeting = config.build()

        # DexPilot / Vector 类型：取出 (2, K) 的人手关键点索引
        indices = self.seq.optimizer.target_link_human_indices
        if indices is None or indices.ndim != 2 or indices.shape[0] != 2:
            raise RuntimeError(
                f"重定向器返回的 target_link_human_indices 形状异常: "
                f"{None if indices is None else indices.shape}"
            )
        self.origin_indices = np.asarray(indices[0, :], dtype=int)
        self.task_indices = np.asarray(indices[1, :], dtype=int)

        # 从 retargeting 输出的全 qpos 中取出 Linker 6 个主动关节的下标
        retargeting_joints = self.seq.joint_names
        missing = [n for n in LINKER_JOINTS if n not in retargeting_joints]
        if missing:
            raise RuntimeError(
                f"以下 Linker 关节在 URDF 中找不到: {missing}\n"
                f"URDF 实际关节: {retargeting_joints}"
            )
        self.ret2linker = np.array(
            [retargeting_joints.index(n) for n in LINKER_JOINTS], dtype=int
        )

        # 校验 MediaPipe 索引范围
        max_idx = int(max(self.origin_indices.max(), self.task_indices.max()))
        if max_idx > 20:
            raise RuntimeError(
                f"target_link_human_indices 中出现 {max_idx} > 20，"
                f"不兼容 MediaPipe 21 关键点输入"
            )

        # ── 计算 R_mano_to_base：把 MANO 系的人手 ref 旋转到 robot base_link 系
        self.R_mano_to_base = self._auto_calibrate_axes()

    def _auto_calibrate_axes(self) -> np.ndarray:
        """
        让 robot 处于 qpos=0（手张开），forward kinematics 取出 robot 在 base_link
        系下的 wrist / index_mcp / middle_mcp 三个 link 位置，然后按 dex-retargeting
        的同套 SVD 流程估出 robot 自身的"虚拟 MANO frame"。

        返回 (3, 3) 矩阵 R 使得 vec_base ≈ vec_mano @ R，从而让 ref 与 robot 端
        forward kinematics 输出的 vector 朝向一致.
        """
        try:
            optimizer = self.seq.optimizer
            robot = optimizer.robot

            # 候选 link：先尝试 index/middle 的 proximal/mcp link，再退化
            candidate_index = [
                "index_proximal", "index_mcp", "index_mcp_pitch_link",
            ]
            candidate_middle = [
                "middle_proximal", "middle_mcp", "middle_mcp_pitch_link",
            ]
            wrist_link = "hand_base_link"

            def _try_pose(name):
                try:
                    idx = robot.get_link_index(name)
                    pose = robot.get_link_pose(idx)
                    return np.asarray(pose[:3, 3], dtype=np.float64)
                except Exception:
                    return None

            n_joints = robot.dof
            robot.compute_forward_kinematics(np.zeros(n_joints))

            wrist_pos = _try_pose(wrist_link)
            if wrist_pos is None:
                wrist_pos = _try_pose("base_link")
            index_pos = next(
                (p for p in (_try_pose(n) for n in candidate_index) if p is not None),
                None,
            )
            middle_pos = next(
                (p for p in (_try_pose(n) for n in candidate_middle) if p is not None),
                None,
            )

            if wrist_pos is None or index_pos is None or middle_pos is None:
                print(
                    "[Retargeter] WARN: 找不到 robot 端的 wrist / index_mcp / middle_mcp "
                    "link，自动校准失败，跳过 (R = I)"
                )
                return np.eye(3, dtype=np.float32)

            mano_frame_in_base = _estimate_mano_axes_from_three_points(
                wrist_pos, index_pos, middle_pos
            )
            # mano_frame_in_base 的列向量 = MANO 系基向量在 base_link 系下的坐标
            # 我们希望：vec_base = vec_mano @ R   (按行向量约定)
            # 即 R 的第 i 行 = mano 第 i 个基向量在 base_link 系下的坐标
            R = mano_frame_in_base.T.astype(np.float32)
            print("[Retargeter] R_mano_to_base 自动校准:")
            for row in R:
                print(f"    [{row[0]:+.3f}  {row[1]:+.3f}  {row[2]:+.3f}]")
            print(
                f"  det={float(np.linalg.det(R)):+.4f}  "
                f"(应为 +1)"
            )
            print(
                "  说明：mano +z (手指方向) -> base_link 系下 "
                f"({R[2,0]:+.2f}, {R[2,1]:+.2f}, {R[2,2]:+.2f})"
            )
            return R
        except Exception as exc:
            print(f"[Retargeter] 校准异常: {exc}, 退化为 R = I")
            return np.eye(3, dtype=np.float32)

    # ── 主接口 ──────────────────────────────────────────────────────────

    def retarget(self, joint_pos_21x3: np.ndarray) -> np.ndarray:
        """
        把 (21, 3) 关键点重定向为 Linker O6 主动关节弧度。

        Parameters
        ----------
        joint_pos_21x3 : np.ndarray, shape=(21, 3), 已转到 MANO 局部坐标系

        Returns
        -------
        np.ndarray, shape=(6,), dtype=float32，顺序与 LINKER_JOINTS 一致
        """
        if joint_pos_21x3.shape != (21, 3):
            raise ValueError(
                f"期望 (21, 3) 输入，收到 {joint_pos_21x3.shape}"
            )
        pts = np.asarray(joint_pos_21x3, dtype=np.float32)
        # 把 MANO 系的关键点旋转到 robot base_link 系
        pts_base = pts @ self.R_mano_to_base
        ref = pts_base[self.task_indices, :] - pts_base[self.origin_indices, :]
        qpos = self.seq.retarget(ref)
        return np.asarray(qpos[self.ret2linker], dtype=np.float32)

    def retarget_pico(self, raw26x7: np.ndarray) -> PicoRetargetingResult:
        """
        从一帧 PICO raw26x7 同步得到 MANO 点、O6 六主动关节弧度和 wrist 6D pose。
        """
        pts21_mano = pico_raw_to_mano(raw26x7, hand=self.hand)
        linker_joint_radians = self.retarget(pts21_mano)
        wrist_pose_6d = pico_wrist_pose6d(raw26x7)
        return PicoRetargetingResult(
            pts21_mano=pts21_mano,
            linker_joint_radians=linker_joint_radians,
            wuji_joint_radians=None,
            wrist_pose_6d=wrist_pose_6d,
        )

    # ── 辅助方法 ────────────────────────────────────────────────────────

    def reset(self):
        """episode 切换时调用，重置上一帧 qpos 缓存。"""
        self.seq.reset()

    @property
    def joint_names(self) -> List[str]:
        return list(LINKER_JOINTS)


def _resolve_wuji_yaml_path(yaml_path: str | None) -> str:
    raw = Path(yaml_path or DEFAULT_WUJI_YAML).expanduser()
    candidates = [
        raw,
        Path.cwd() / raw,
        _REPO_ROOT / raw,
        Path(__file__).resolve().parent / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    tried = "\n  ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(f"无法找到 Wuji retargeting YAML:\n  {tried}")


class WujiHandRetargeter:
    """把 MediaPipe 21 关键点重定向为 Wuji Hand 20 维关节弧度。"""

    def __init__(
        self,
        yaml_path: str | None = DEFAULT_WUJI_YAML,
        hand: str = "right",
    ):
        if hand not in ("right", "left"):
            raise ValueError(f"hand 必须为 'right' 或 'left'，收到 {hand!r}")
        from wuji_retargeting import Retargeter  # noqa: WPS433

        self.hand = hand
        self.yaml_path = _resolve_wuji_yaml_path(yaml_path)
        self.retargeter = Retargeter.from_yaml(self.yaml_path, hand_side=hand)

    def retarget(self, points21_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points21_m, dtype=np.float32)
        if points.shape != (21, 3):
            raise ValueError(f"期望 (21, 3) 输入，收到 {points.shape}")
        qpos = np.asarray(self.retargeter.retarget(points), dtype=np.float32).reshape(-1)
        if qpos.shape != (20,):
            raise RuntimeError(f"Wuji retargeter 返回 {qpos.shape}，期望 (20,)")
        return qpos

    def reset(self) -> None:
        lp_filter = getattr(self.retargeter, "lp_filter", None)
        if lp_filter is not None and hasattr(lp_filter, "reset"):
            lp_filter.reset()
        optimizer = getattr(self.retargeter, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "last_qpos"):
            optimizer.last_qpos = None

    @property
    def joint_names(self) -> List[str]:
        return list(WUJI_JOINTS)


class PicoToLinkerO6Retargeter:
    """
    PICO raw26x7 → MANO 21 keypoints + O6/Wuji 关节弧度 + wrist 6D pose。

    该类把原来分散在采集脚本里的 PICO 拓扑映射、MANO 对齐、wrist 局部轴重定义
    以及 O6/Wuji retargeting 收口到一个入口。
    """

    def __init__(
        self,
        yaml_path: str = DEFAULT_YAML,
        urdf_dir: str = DEFAULT_URDF_DIR,
        wuji_yaml_path: str | None = DEFAULT_WUJI_YAML,
        hand: str = "right",
    ):
        self.linker = LinkerO6Retargeter(
            yaml_path=yaml_path,
            urdf_dir=urdf_dir,
            hand=hand,
        )
        self.wuji = WujiHandRetargeter(
            yaml_path=wuji_yaml_path,
            hand=hand,
        )

    def retarget(self, raw26x7: np.ndarray) -> PicoRetargetingResult:
        pts21_mano = pico_raw_to_mano(raw26x7, hand=self.linker.hand)
        pts21_mediapipe = pico_raw_to_mediapipe(raw26x7)
        return PicoRetargetingResult(
            pts21_mano=pts21_mano,
            linker_joint_radians=self.linker.retarget(pts21_mano),
            wuji_joint_radians=self.wuji.retarget(pts21_mediapipe),
            wrist_pose_6d=pico_wrist_pose6d(raw26x7),
        )

    def process(self, raw26x7: np.ndarray) -> PicoRetargetingResult:
        return self.retarget(raw26x7)

    def reset(self):
        self.linker.reset()
        self.wuji.reset()

    @property
    def joint_names(self) -> List[str]:
        return self.linker.joint_names


# ── CLI 自测 ────────────────────────────────────────────────────────────────

def _selftest():
    """构造一次 retargeting 并喂一个伪手姿态。"""
    rt = LinkerO6Retargeter(hand="right")
    print("Linker 关节顺序:", rt.joint_names)
    print("origin_indices :", rt.origin_indices)
    print("task_indices   :", rt.task_indices)

    # 构造一个"基本伸开"的伪 MANO 手：手腕 0,0,0；中指尖 +Z 方向；其他指张开
    pts = np.zeros((21, 3), dtype=np.float32)
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [1, 5, 9, 13, 17]
    for i, (mcp, tip) in enumerate(zip(finger_mcps, finger_tips)):
        x_offset = (i - 2) * 0.02
        pts[mcp] = [x_offset, 0.0, 0.05]
        pts[tip] = [x_offset, 0.0, 0.13]

    angles = rt.retarget(pts)
    print("retarget 输出 (rad):", angles)


if __name__ == "__main__":
    _selftest()
