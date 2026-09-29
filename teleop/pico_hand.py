#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PICO 头显手部数据采集与格式转换。中间文件

负责：
  1. 调用 xrobotoolkit_sdk 取 26 关节 × 7 维 (x,y,z,qx,qy,qz,qw) 原始数据
  2. PICO 26 关键点 → MediaPipe 21 关键点拓扑
  3. 平移到手腕原点 + 对齐到 MANO 局部坐标系（X 掌心向内, Y 食指→小指, Z 手腕→中指）

最终输出供 dex-retargeting 使用的 (21, 3) float32 数组（米制）。

使用示例：
    with PicoHandReader(hand="right") as reader:
        joint_pos, active = reader.read_mano()
        if active and joint_pos is not None:
            # joint_pos.shape == (21, 3)
            ...
"""

from __future__ import annotations

import numpy as np

try:
    import xrobotoolkit_sdk as xrt
except ImportError as e:
    raise ImportError(
        "未找到 xrobotoolkit_sdk，请确认已安装 PICO XRoboToolkit-PC-Service SDK，"
        "并 `pip install xrobotoolkit_sdk`"
    ) from e


# ── PICO 26 → MediaPipe 21 索引映射 ─────────────────────────────────────────
# 丢弃 PICO[0] Palm，以及 PICO[6/11/16/21] 四指 Metacarpal（MediaPipe 不区分）。
# 拇指无 intermediate，PICO 拇指本就是 4 段（metacarpal/proximal/distal/tip）。
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


# 来自 dex_retargeting/constants.py，右手 operator → MANO 旋转矩阵。
# X 掌心向内 / Y 食指→小指 / Z 手腕→中指。
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)


def _estimate_wrist_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    通过手腕(0)、食指 MCP(5)、中指 MCP(9) 三个点用 SVD 拟合一个手部局部坐标系。

    实现复制自 dex-retargeting/example/vector_retargeting/single_hand_detector.py 的
    SingleHandDetector.estimate_frame_from_hand_points，保持口径完全一致，避免
    引入 mediapipe 依赖。
    """
    assert keypoint_3d_array.shape == (21, 3), keypoint_3d_array.shape
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


class PicoHandReader:
    """读取并转换 PICO 单手数据。"""

    def __init__(self, hand: str = "right"):
        if hand not in ("right", "left"):
            raise ValueError(f"hand 必须为 'right' 或 'left'，收到 {hand!r}")
        self.hand = hand
        xrt.init()
        self._closed = False

    # ── 基础读取 ────────────────────────────────────────────────────────

    def read_raw(self) -> tuple[np.ndarray, int]:
        """
        返回原始 PICO 数据。

        Returns
        -------
        arr    : np.ndarray, shape=(26, 7), [x, y, z, qx, qy, qz, qw]
        active : 0=低质量, 1=高质量
        """
        if self.hand == "right":
            raw = xrt.get_right_hand_tracking_state()
            active = xrt.get_right_hand_is_active()
        else:
            raw = xrt.get_left_hand_tracking_state()
            active = xrt.get_left_hand_is_active()
        arr = np.asarray(raw, dtype=np.float64)
        return arr, int(active)

    def get_timestamp_ns(self) -> int:
        """当前样本对应的 SDK 时间戳（纳秒）。"""
        return int(xrt.get_time_stamp_ns())

    # ── 转换为 dex-retargeting 所需格式 ────────────────────────────────────

    def read_mano(
        self, return_debug: bool = False
    ):
        """
        读取并转换为 dex-retargeting 期望的 (21, 3) 关键点。

        - 单位: 米
        - 编号: MediaPipe Hands 21 关键点
        - 原点: 手腕 (索引 0) = [0,0,0]
        - 朝向: MANO 局部坐标系（消除全局旋转）

        Parameters
        ----------
        return_debug : bool
            True 时额外返回一个 debug dict，包含原始 PICO 世界坐标、SVD 估出的 rot
            矩阵等信息，用于排查重定向问题。

        Returns
        -------
        joint_pos : np.ndarray, shape=(21, 3), dtype=float32； 如果原始数据缺失返回 None
        active    : 0=低质量, 1=高质量
        debug     : dict（仅 return_debug=True 时存在）
        """
        raw26x7, active = self.read_raw()

        if raw26x7.ndim != 2 or raw26x7.shape[0] != 26 or raw26x7.shape[1] < 3:
            if return_debug:
                return None, active, {}
            return None, active

        pts26 = raw26x7[:, :3]
        pts21 = pts26[PICO2MEDIAPIPE]

        centered = pts21 - pts21[0:1, :]
        rot = _estimate_wrist_frame(centered)

        # 注意：左手目前仅做相同流程；OPERATOR2MANO_LEFT 已在 dex-retargeting 里定义，
        # 此处右手优先，扩展左手时再加分支即可。
        joint_pos = (centered @ rot @ OPERATOR2MANO_RIGHT).astype(np.float32)

        if not return_debug:
            return joint_pos, active

        debug = {
            "pts21_world": pts21.astype(np.float32),     # 变换前，PICO 世界系
            "centered": centered.astype(np.float32),     # wrist 平移到原点后
            "rot": rot.astype(np.float32),               # SVD 估出的手部局部系
            "rot_det": float(np.linalg.det(rot)),        # 应当 +1（右手系）
        }
        return joint_pos, active, debug

    # ── 资源管理 ────────────────────────────────────────────────────────

    def close(self):
        if not self._closed:
            try:
                xrt.close()
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── 调试格式化（两个 teleop 入口共用） ──────────────────────────────────────

# MediaPipe 21 关键点编号
_MP_NAMES = [
    "wrist",
    "thumb_cmc",  "thumb_mcp",  "thumb_ip",   "thumb_tip",
    "index_mcp",  "index_pip",  "index_dip",  "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp",   "ring_pip",   "ring_dip",   "ring_tip",
    "pinky_mcp",  "pinky_pip",  "pinky_dip",  "pinky_tip",
]


def format_keypoints_debug(
    pts21: np.ndarray | None,
    active: int,
    ts_ns: int | None = None,
    frame_idx: int | None = None,
    debug: dict | None = None,
) -> str:
    """
    把一帧 (21, 3) MANO 关键点格式化成可打印的调试字符串。

    当传入 ``debug``（来自 PicoHandReader.read_mano(return_debug=True)）时，会
    额外打印 sanity check：原始 PICO 世界坐标、手部尺度、SVD 矩阵行列式。
    """
    head_parts = []
    if frame_idx is not None:
        head_parts.append(f"frame={frame_idx}")
    if ts_ns is not None:
        head_parts.append(f"ts_ns={ts_ns}")
    head_parts.append(f"active={active}")
    head = "  ".join(head_parts)

    if pts21 is None:
        return f"[PICO {head}] NO DATA"

    p = np.asarray(pts21) * 1000.0  # m -> mm

    def fmt(idx: int) -> str:
        x, y, z = p[idx]
        return f"{_MP_NAMES[idx]:>10s}=[{x:+6.1f},{y:+6.1f},{z:+6.1f}]"

    lines = [
        f"[PICO {head}]  wrist={p[0].round(1).tolist()}  (mm, MANO frame)",
        "  TIPS: " + "  ".join(fmt(i) for i in (4, 8, 12, 16, 20)),
        "  MCPs: " + "  ".join(fmt(i) for i in (1, 5, 9, 13, 17)),
    ]

    # ── sanity check ────────────────────────────────────────────────────
    def _dist(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) * 1000.0)

    d_w2mid_mcp = _dist(pts21[0], pts21[9])
    d_w2mid_tip = _dist(pts21[0], pts21[12])
    d_w2thumb_tip = _dist(pts21[0], pts21[4])
    d_thumb2index = _dist(pts21[4], pts21[8])

    lines.append("  尺度判定 (期望值供参考):")
    lines.append(
        f"    wrist→middle_mcp = {d_w2mid_mcp:6.1f} mm  (~90 mm)"
        f"    wrist→middle_tip = {d_w2mid_tip:6.1f} mm  (张开 ~180 mm)"
    )
    lines.append(
        f"    wrist→thumb_tip  = {d_w2thumb_tip:6.1f} mm  (~120 mm)"
        f"    thumb_tip→index_tip = {d_thumb2index:6.1f} mm  (张开 ~80, 捏合 ~10)"
    )

    # 变换后 middle_tip 在 MANO 系下：dex-retargeting 默认手势下应位于 +Z（手指方向）
    mid_tip_mano = pts21[12] * 1000.0
    lines.append(
        f"  MANO 系 middle_tip = [{mid_tip_mano[0]:+6.1f},"
        f" {mid_tip_mano[1]:+6.1f}, {mid_tip_mano[2]:+6.1f}] mm "
        f"(张开时 z 应明显为正/负，xy 接近 0)"
    )

    if debug:
        pts_world = debug.get("pts21_world")
        if pts_world is not None:
            pw = np.asarray(pts_world) * 1000.0
            lines.append("  变换前 PICO 世界系 (mm):")
            lines.append(
                f"    wrist     =[{pw[0,0]:+8.1f}, {pw[0,1]:+8.1f}, {pw[0,2]:+8.1f}]"
            )
            lines.append(
                f"    index_mcp =[{pw[5,0]:+8.1f}, {pw[5,1]:+8.1f}, {pw[5,2]:+8.1f}]"
                f"    middle_mcp=[{pw[9,0]:+8.1f}, {pw[9,1]:+8.1f}, {pw[9,2]:+8.1f}]"
            )

        rot = debug.get("rot")
        rot_det = debug.get("rot_det")
        if rot is not None:
            r = np.asarray(rot)
            lines.append("  SVD 估出的旋转矩阵 rot (列=[x_axis, normal, z_axis]):")
            for row in r:
                lines.append(
                    f"    [{row[0]:+.3f}  {row[1]:+.3f}  {row[2]:+.3f}]"
                )
            if rot_det is not None:
                ok = "OK ✓" if 0.95 < rot_det < 1.05 else "异常 ✗ 可能存在镜像"
                lines.append(f"  rot 行列式 = {rot_det:+.4f}  ({ok})")

    return "\n".join(lines)


def format_angles_debug(
    angles_rad: np.ndarray,
    joint_names: list[str] | None = None,
    cmd_0_255: list[int] | None = None,
) -> str:
    """
    把重定向输出 (6 个弧度) 与可选的 0~255 命令打印成易读字符串。
    """
    if joint_names is None:
        joint_names = [
            "thumb_pitch", "thumb_yaw",
            "index", "middle", "ring", "pinky",
        ]
    arr = np.asarray(angles_rad)
    rad_str = "  ".join(
        f"{n}={float(v):+.3f}" for n, v in zip(joint_names, arr)
    )
    if cmd_0_255 is None:
        return f"[Retarget rad]  {rad_str}"

    cmd_str = " ".join(f"{int(c):3d}" for c in cmd_0_255)
    return f"[Retarget rad]  {rad_str}\n[Cmd  0~255 ]  {cmd_str}"


# ── CLI 自测 ────────────────────────────────────────────────────────────────

def _selftest():
    """简单自检：连接 PICO 并打印一帧数据。"""
    import time

    with PicoHandReader(hand="right") as reader:
        for i in range(10):
            pts21, active = reader.read_mano()
            ts = reader.get_timestamp_ns()
            if pts21 is None:
                print(f"[{i}] active={active} ts={ts} (no data)")
            else:
                print(
                    f"[{i}] active={active} ts={ts} "
                    f"wrist={pts21[0]} thumb_tip={pts21[4]} "
                    f"index_tip={pts21[8]}"
                )
            time.sleep(0.1)


if __name__ == "__main__":
    _selftest()
