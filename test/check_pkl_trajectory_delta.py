#!/usr/bin/env python3
"""
Offline PKL trajectory and hand-keypoint delta checker.

Examples:
    python3 test/check_pkl_trajectory_delta.py data/demo/demo.pkl
    python3 test/check_pkl_trajectory_delta.py data --recursive
    python3 test/check_pkl_trajectory_delta.py demo.pkl --frame-a 10 --frame-b 11
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

try:
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401

    HAS_MPL_3D = True
except Exception:
    HAS_MPL_3D = False


MP_NAMES = [
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

FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=int)
FINGERTIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class SequenceData:
    frame_indices: np.ndarray
    times_sec: np.ndarray
    values: np.ndarray


@dataclass
class DeltaData:
    prev_frame_indices: np.ndarray
    curr_frame_indices: np.ndarray
    times_sec: np.ndarray
    values: np.ndarray


@dataclass
class EpisodeData:
    pkl_path: Path
    messages: list[Any]
    pose_mats: SequenceData | None
    pts21: SequenceData | None
    hand_cmds: SequenceData | None
    format_version: Any


@dataclass
class CheckResult:
    pkl_path: Path
    figure_path: Path | None
    report_path: Path | None
    num_messages: int
    num_pose_frames: int
    num_pts21_frames: int
    max_trans_delta: float
    max_trans_pair: tuple[int, int] | None
    max_rot_delta_deg: float
    max_rot_pair: tuple[int, int] | None
    max_keypoint_delta: float
    max_keypoint_pair: tuple[int, int] | None
    max_keypoint_name: str | None
    max_mean_keypoint_delta: float
    pair_report: str | None


def _get_field(msg: Any, name: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(name, default)
    return getattr(msg, name, default)


def _as_finite_array(value: Any, expected_shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if expected_shape is not None and arr.shape != expected_shape:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _load_messages(pkl_path: Path) -> tuple[list[Any], Any]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "messages" in data:
        return list(data["messages"]), data.get("formatVersion")
    if hasattr(data, "sensorMessages"):
        return list(data.sensorMessages), getattr(data, "formatVersion", "sensorMessages")
    if isinstance(data, list):
        return data, "list"
    raise ValueError("Unsupported PKL format: expected dict['messages'], list, or .sensorMessages")


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid quaternion")
    qx, qy, qz, qw = quat / norm
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


def _rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def _rotmat_to_rotvec(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    cos_theta = float((np.trace(rot) - 1.0) * 0.5)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)

    if abs(math.pi - theta) < 1e-5:
        axis = np.sqrt(np.maximum(np.diag(rot) + 1.0, 0.0) * 0.5)
        if rot[2, 1] - rot[1, 2] < 0:
            axis[0] = -axis[0]
        if rot[0, 2] - rot[2, 0] < 0:
            axis[1] = -axis[1]
        if rot[1, 0] - rot[0, 1] < 0:
            axis[2] = -axis[2]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return axis / norm * theta

    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(theta)
    return axis * theta


def _trajectory_pose_to_matrix(value: Any) -> np.ndarray | None:
    arr = _as_finite_array(value)
    if arr is None:
        return None

    if arr.shape == (4, 4):
        mat = arr.astype(np.float64, copy=True)
        mat[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
        return mat

    flat = arr.reshape(-1)
    mat = np.eye(4, dtype=np.float64)
    if flat.shape[0] == 6:
        mat[:3, 3] = flat[:3]
        mat[:3, :3] = _rotvec_to_rotmat(flat[3:6])
        return mat
    if flat.shape[0] == 7:
        mat[:3, 3] = flat[:3]
        mat[:3, :3] = _quat_xyzw_to_rotmat(flat[3:7])
        return mat
    return None


def _relative_time_from_messages(messages: list[Any]) -> np.ndarray:
    candidates = [
        ("mainClockMonotonicNs", 1e-9),
        ("sampleClockNs", 1e-9),
        ("timestamp", 1.0),
    ]
    for name, scale in candidates:
        raw = []
        ok = True
        for msg in messages:
            value = _get_field(msg, name)
            if value is None:
                ok = False
                break
            try:
                raw.append(float(value))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and raw and np.all(np.isfinite(raw)):
            arr = np.asarray(raw, dtype=np.float64)
            return (arr - arr[0]) * scale
    return np.arange(len(messages), dtype=np.float64)


def load_episode(pkl_path: Path) -> EpisodeData:
    messages, format_version = _load_messages(pkl_path)
    base_times = _relative_time_from_messages(messages)

    pose_indices, pose_times, pose_mats = [], [], []
    pts_indices, pts_times, pts_values = [], [], []
    cmd_indices, cmd_times, cmd_values = [], [], []

    for idx, msg in enumerate(messages):
        mat = _trajectory_pose_to_matrix(_get_field(msg, "trajectoryPose"))
        if mat is not None:
            pose_indices.append(idx)
            pose_times.append(base_times[idx])
            pose_mats.append(mat)

        pts = _as_finite_array(_get_field(msg, "pts21_mano"), expected_shape=(21, 3))
        if pts is not None:
            pts_indices.append(idx)
            pts_times.append(base_times[idx])
            pts_values.append(pts)

        cmd_value = _get_field(msg, "o6_command")
        if cmd_value is None:
            cmd_value = _get_field(msg, "handCommand")
        cmd = _as_finite_array(cmd_value, expected_shape=(6,))
        if cmd is not None:
            cmd_indices.append(idx)
            cmd_times.append(base_times[idx])
            cmd_values.append(cmd)

    pose_seq = None
    if pose_mats:
        pose_seq = SequenceData(
            frame_indices=np.asarray(pose_indices, dtype=int),
            times_sec=np.asarray(pose_times, dtype=np.float64),
            values=np.stack(pose_mats, axis=0),
        )

    pts_seq = None
    if pts_values:
        pts_seq = SequenceData(
            frame_indices=np.asarray(pts_indices, dtype=int),
            times_sec=np.asarray(pts_times, dtype=np.float64),
            values=np.stack(pts_values, axis=0),
        )

    cmd_seq = None
    if cmd_values:
        cmd_seq = SequenceData(
            frame_indices=np.asarray(cmd_indices, dtype=int),
            times_sec=np.asarray(cmd_times, dtype=np.float64),
            values=np.stack(cmd_values, axis=0),
        )

    return EpisodeData(
        pkl_path=pkl_path,
        messages=messages,
        pose_mats=pose_seq,
        pts21=pts_seq,
        hand_cmds=cmd_seq,
        format_version=format_version,
    )


def _pose_deltas(pose_seq: SequenceData | None) -> tuple[DeltaData | None, DeltaData | None, np.ndarray | None]:
    if pose_seq is None or len(pose_seq.values) < 2:
        return None, None, None

    mats = pose_seq.values
    trans_delta = np.linalg.norm(np.diff(mats[:, :3, 3], axis=0), axis=1)
    rot_delta = []
    rel_rotvec = []
    for prev, curr in zip(mats[:-1], mats[1:]):
        rel_rot = prev[:3, :3].T @ curr[:3, :3]
        rv = _rotmat_to_rotvec(rel_rot)
        rel_rotvec.append(rv)
        rot_delta.append(np.linalg.norm(rv))

    common = {
        "prev_frame_indices": pose_seq.frame_indices[:-1],
        "curr_frame_indices": pose_seq.frame_indices[1:],
        "times_sec": pose_seq.times_sec[1:],
    }
    return (
        DeltaData(values=trans_delta, **common),
        DeltaData(values=np.asarray(rot_delta, dtype=np.float64), **common),
        np.asarray(rel_rotvec, dtype=np.float64),
    )


def _keypoint_deltas(pts_seq: SequenceData | None) -> dict[str, Any]:
    if pts_seq is None or len(pts_seq.values) < 2:
        return {}

    diff = pts_seq.values[1:] - pts_seq.values[:-1]
    per_joint = np.linalg.norm(diff, axis=2)
    common = {
        "prev_frame_indices": pts_seq.frame_indices[:-1],
        "curr_frame_indices": pts_seq.frame_indices[1:],
        "times_sec": pts_seq.times_sec[1:],
    }
    return {
        "per_joint": DeltaData(values=per_joint, **common),
        "mean": DeltaData(values=np.mean(per_joint, axis=1), **common),
        "max": DeltaData(values=np.max(per_joint, axis=1), **common),
        "max_joint": np.argmax(per_joint, axis=1),
        "fingertips": DeltaData(values=per_joint[:, FINGERTIP_INDICES], **common),
    }


def _max_pair(delta: DeltaData | None) -> tuple[float, tuple[int, int] | None, int | None]:
    if delta is None or delta.values.size == 0:
        return 0.0, None, None
    flat = np.asarray(delta.values)
    if flat.ndim > 1:
        reduced = np.max(flat, axis=tuple(range(1, flat.ndim)))
    else:
        reduced = flat
    idx = int(np.argmax(reduced))
    return (
        float(reduced[idx]),
        (int(delta.prev_frame_indices[idx]), int(delta.curr_frame_indices[idx])),
        idx,
    )


def _frame_lookup(seq: SequenceData | None) -> dict[int, int]:
    if seq is None:
        return {}
    return {int(frame_idx): i for i, frame_idx in enumerate(seq.frame_indices)}


def _format_pair_report(episode: EpisodeData, frame_a: int | None, frame_b: int | None) -> str | None:
    if frame_a is None and frame_b is None:
        return None
    if frame_a is None or frame_b is None:
        return "需要同时提供 --frame-a 和 --frame-b 才能计算指定两帧差异。"

    lines = [f"Specified pair: frame {frame_a} -> {frame_b}"]

    pose_map = _frame_lookup(episode.pose_mats)
    if episode.pose_mats is not None and frame_a in pose_map and frame_b in pose_map:
        a = episode.pose_mats.values[pose_map[frame_a]]
        b = episode.pose_mats.values[pose_map[frame_b]]
        trans = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
        rot_deg = float(np.rad2deg(np.linalg.norm(_rotmat_to_rotvec(a[:3, :3].T @ b[:3, :3]))))
        lines.append(f"  pose translation delta: {trans:.6f} m")
        lines.append(f"  pose rotation delta:    {rot_deg:.3f} deg")
    else:
        lines.append("  pose delta: unavailable for this pair")

    pts_map = _frame_lookup(episode.pts21)
    if episode.pts21 is not None and frame_a in pts_map and frame_b in pts_map:
        a = episode.pts21.values[pts_map[frame_a]]
        b = episode.pts21.values[pts_map[frame_b]]
        dist = np.linalg.norm(b - a, axis=1)
        max_joint = int(np.argmax(dist))
        lines.append(f"  keypoint mean delta:    {float(np.mean(dist)):.6f} m")
        lines.append(
            f"  keypoint max delta:     {float(dist[max_joint]):.6f} m "
            f"({max_joint}:{MP_NAMES[max_joint]})"
        )
        fingertip_desc = ", ".join(
            f"{name}={dist[idx]:.5f}m" for name, idx in zip(FINGERTIP_NAMES, FINGERTIP_INDICES)
        )
        lines.append(f"  fingertip deltas:       {fingertip_desc}")
    else:
        lines.append("  keypoint delta: unavailable for this pair")

    return "\n".join(lines)


def _set_3d_equal(ax, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    if radius < 1e-6:
        radius = 0.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_hand(ax, pts: np.ndarray, title: str, color: str) -> None:
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=14, color=color)
    for a, b in HAND_BONES:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            [pts[a, 2], pts[b, 2]],
            color=color,
            linewidth=1.3,
            alpha=0.85,
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_3d_equal(ax, pts)


def _plot_hand_2d(ax, pts: np.ndarray, title: str, color: str) -> None:
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=color)
    for a, b in HAND_BONES:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            color=color,
            linewidth=1.3,
            alpha=0.85,
        )
    ax.set_title(f"{title} (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True)


def _write_report(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_dashboard(
    episode: EpisodeData,
    out_path: Path,
    trans_delta: DeltaData | None,
    rot_delta: DeltaData | None,
    key_deltas: dict[str, Any],
    pair_report: str | None,
    trans_threshold: float,
    rot_threshold_deg: float,
    keypoint_threshold: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(22, 26))
    fig.suptitle(
        f"PKL Delta Check: {episode.pkl_path.name}\n"
        f"formatVersion={episode.format_version}, messages={len(episode.messages)}, "
        f"pose_frames={0 if episode.pose_mats is None else len(episode.pose_mats.values)}, "
        f"pts21_frames={0 if episode.pts21 is None else len(episode.pts21.values)}",
        fontsize=16,
    )
    gs = gridspec.GridSpec(5, 4, figure=fig)

    if episode.pose_mats is not None:
        poses = episode.pose_mats.values
        pos = poses[:, :3, 3]
        time = episode.pose_mats.times_sec
        rotvec = np.stack([_rotmat_to_rotvec(m[:3, :3]) for m in poses], axis=0)

        if HAS_MPL_3D:
            ax = fig.add_subplot(gs[0, 0], projection="3d")
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="seagreen", linewidth=1.5)
            ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="blue", s=45, label="start")
            ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color="red", s=45, label="end")
            ax.set_title("3D trajectoryPose")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()
            _set_3d_equal(ax, pos)
        else:
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(pos[:, 0], pos[:, 1], color="seagreen", linewidth=1.5)
            ax.scatter(pos[0, 0], pos[0, 1], color="blue", s=45, label="start")
            ax.scatter(pos[-1, 0], pos[-1, 1], color="red", s=45, label="end")
            ax.set_title("trajectoryPose XY")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis("equal")
            ax.grid(True)
            ax.legend()

        ax = fig.add_subplot(gs[0, 1])
        for i, label in enumerate(["x", "y", "z"]):
            ax.plot(time, pos[:, i], label=label)
        ax.set_title("Trajectory position")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m")
        ax.grid(True)
        ax.legend()

        ax = fig.add_subplot(gs[0, 2])
        for i, label in enumerate(["rx", "ry", "rz"]):
            ax.plot(time, rotvec[:, i], label=label)
        ax.set_title("Trajectory rotation vector")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("rad")
        ax.grid(True)
        ax.legend()
    else:
        fig.add_subplot(gs[0, 0]).text(0.1, 0.5, "No valid trajectoryPose", fontsize=13)
        fig.add_subplot(gs[0, 1]).axis("off")
        fig.add_subplot(gs[0, 2]).axis("off")

    ax = fig.add_subplot(gs[0, 3])
    if trans_delta is not None:
        ax.plot(trans_delta.times_sec, trans_delta.values, color="royalblue")
        ax.axhline(trans_threshold, color="crimson", linestyle="--", linewidth=1.0)
        max_v, max_pair, _ = _max_pair(trans_delta)
        ax.set_title(f"Frame delta translation\nmax={max_v:.5f} m pair={max_pair}")
    else:
        ax.text(0.1, 0.5, "No pose delta")
        ax.set_title("Frame delta translation")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/frame")
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    if rot_delta is not None:
        rot_deg = np.rad2deg(rot_delta.values)
        ax.plot(rot_delta.times_sec, rot_deg, color="darkorange")
        ax.axhline(rot_threshold_deg, color="crimson", linestyle="--", linewidth=1.0)
        max_v, max_pair, _ = _max_pair(DeltaData(rot_delta.prev_frame_indices, rot_delta.curr_frame_indices, rot_delta.times_sec, rot_deg))
        ax.set_title(f"Frame delta rotation\nmax={max_v:.3f} deg pair={max_pair}")
    else:
        ax.text(0.1, 0.5, "No rotation delta")
        ax.set_title("Frame delta rotation")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deg/frame")
    ax.grid(True)

    if episode.pts21 is not None:
        if HAS_MPL_3D:
            ax = fig.add_subplot(gs[1, 1], projection="3d")
            _plot_hand(ax, episode.pts21.values[0], "First valid pts21_mano", "tab:blue")
        else:
            ax = fig.add_subplot(gs[1, 1])
            _plot_hand_2d(ax, episode.pts21.values[0], "First valid pts21_mano", "tab:blue")

        if HAS_MPL_3D:
            ax = fig.add_subplot(gs[1, 2], projection="3d")
            _plot_hand(ax, episode.pts21.values[-1], "Last valid pts21_mano", "tab:red")
        else:
            ax = fig.add_subplot(gs[1, 2])
            _plot_hand_2d(ax, episode.pts21.values[-1], "Last valid pts21_mano", "tab:red")
    else:
        fig.add_subplot(gs[1, 1]).text(0.1, 0.5, "No valid pts21_mano", fontsize=13)
        fig.add_subplot(gs[1, 2]).axis("off")

    ax = fig.add_subplot(gs[1, 3])
    key_mean = key_deltas.get("mean")
    key_max = key_deltas.get("max")
    if key_mean is not None and key_max is not None:
        ax.plot(key_mean.times_sec, key_mean.values, label="mean", color="slateblue")
        ax.plot(key_max.times_sec, key_max.values, label="max", color="crimson", alpha=0.85)
        ax.axhline(keypoint_threshold, color="black", linestyle="--", linewidth=1.0)
        max_v, max_pair, max_i = _max_pair(key_max)
        joint_name = "?"
        if max_i is not None:
            joint_name = MP_NAMES[int(key_deltas["max_joint"][max_i])]
        ax.set_title(f"pts21 frame delta\nmax={max_v:.5f} m pair={max_pair} joint={joint_name}")
        ax.legend()
    else:
        ax.text(0.1, 0.5, "No keypoint delta")
        ax.set_title("pts21 frame delta")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/frame")
    ax.grid(True)

    ax = fig.add_subplot(gs[2, 0])
    fingertips = key_deltas.get("fingertips")
    if fingertips is not None:
        for i, name in enumerate(FINGERTIP_NAMES):
            ax.plot(fingertips.times_sec, fingertips.values[:, i], label=name)
        ax.set_title("Fingertip deltas")
        ax.legend(ncol=2)
    else:
        ax.text(0.1, 0.5, "No fingertip delta")
        ax.set_title("Fingertip deltas")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/frame")
    ax.grid(True)

    ax = fig.add_subplot(gs[2, 1:3])
    per_joint = key_deltas.get("per_joint")
    if per_joint is not None:
        im = ax.imshow(
            per_joint.values.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[
                float(per_joint.times_sec[0]),
                float(per_joint.times_sec[-1]),
                -0.5,
                20.5,
            ],
        )
        ax.set_yticks(np.arange(21))
        ax.set_yticklabels(MP_NAMES, fontsize=8)
        ax.set_title("Per-joint delta heatmap")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("joint")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="m/frame")
    else:
        ax.text(0.1, 0.5, "No per-joint delta")
        ax.set_title("Per-joint delta heatmap")

    ax = fig.add_subplot(gs[2, 3])
    if episode.hand_cmds is not None:
        for i in range(episode.hand_cmds.values.shape[1]):
            ax.plot(episode.hand_cmds.times_sec, episode.hand_cmds.values[:, i], label=f"cmd{i}")
        ax.set_title("o6_command")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("0-255")
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
    else:
        ax.text(0.1, 0.5, "No o6_command")
        ax.set_title("o6_command")

    for row, label, seq in [
        (3, "position", episode.pose_mats),
    ]:
        if seq is None:
            for col in range(4):
                fig.add_subplot(gs[row, col]).axis("off")
            continue
        pos = seq.values[:, :3, 3]
        time = seq.times_sec
        labels = ["X", "Y", "Z"]
        for i in range(3):
            ax = fig.add_subplot(gs[row, i])
            ax.plot(time, pos[:, i], color="dimgray")
            ax.set_title(f"{label} {labels[i]}")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("m")
            ax.grid(True)
        fig.add_subplot(gs[row, 3]).axis("off")

    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    summary_lines = build_report_lines(
        episode=episode,
        trans_delta=trans_delta,
        rot_delta=rot_delta,
        key_deltas=key_deltas,
        pair_report=pair_report,
        trans_threshold=trans_threshold,
        rot_threshold_deg=rot_threshold_deg,
        keypoint_threshold=keypoint_threshold,
    )
    ax.text(
        0.0,
        1.0,
        "\n".join(summary_lines[:22]),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def build_report_lines(
    episode: EpisodeData,
    trans_delta: DeltaData | None,
    rot_delta: DeltaData | None,
    key_deltas: dict[str, Any],
    pair_report: str | None,
    trans_threshold: float,
    rot_threshold_deg: float,
    keypoint_threshold: float,
) -> list[str]:
    lines = [
        f"PKL: {episode.pkl_path}",
        f"formatVersion: {episode.format_version}",
        f"messages: {len(episode.messages)}",
        f"valid trajectoryPose frames: {0 if episode.pose_mats is None else len(episode.pose_mats.values)}",
        f"valid pts21_mano frames: {0 if episode.pts21 is None else len(episode.pts21.values)}",
    ]

    max_trans, trans_pair, _ = _max_pair(trans_delta)
    max_rot_rad, rot_pair, _ = _max_pair(rot_delta)
    max_rot_deg = float(np.rad2deg(max_rot_rad))
    lines.extend(
        [
            "",
            f"Max trajectory translation delta: {max_trans:.6f} m/frame pair={trans_pair}",
            f"Max trajectory rotation delta:    {max_rot_deg:.3f} deg/frame pair={rot_pair}",
        ]
    )

    if trans_delta is not None:
        count = int(np.count_nonzero(trans_delta.values > trans_threshold))
        lines.append(f"Translation spikes > {trans_threshold:.4f} m: {count}")
        if len(trans_delta.values) > 0 and np.allclose(trans_delta.values, 0.0):
            lines.append("WARNING: all trajectory translation deltas are zero")
    else:
        lines.append("No valid trajectory delta.")

    if rot_delta is not None:
        rot_deg = np.rad2deg(rot_delta.values)
        count = int(np.count_nonzero(rot_deg > rot_threshold_deg))
        lines.append(f"Rotation spikes > {rot_threshold_deg:.2f} deg: {count}")
        if len(rot_deg) > 0 and np.allclose(rot_deg, 0.0):
            lines.append("WARNING: all trajectory rotation deltas are zero")

    key_max = key_deltas.get("max")
    key_mean = key_deltas.get("mean")
    if key_max is not None and key_mean is not None:
        max_key, key_pair, key_i = _max_pair(key_max)
        max_mean, mean_pair, _ = _max_pair(key_mean)
        joint_name = None
        if key_i is not None:
            joint_idx = int(key_deltas["max_joint"][key_i])
            joint_name = f"{joint_idx}:{MP_NAMES[joint_idx]}"
        lines.extend(
            [
                "",
                f"Max pts21 joint delta: {max_key:.6f} m/frame pair={key_pair} joint={joint_name}",
                f"Max pts21 mean delta:  {max_mean:.6f} m/frame pair={mean_pair}",
                f"Keypoint spikes > {keypoint_threshold:.4f} m: {int(np.count_nonzero(key_max.values > keypoint_threshold))}",
            ]
        )
        if len(key_max.values) > 0 and np.allclose(key_max.values, 0.0):
            lines.append("WARNING: all pts21_mano keypoint deltas are zero")
    else:
        lines.append("No valid pts21_mano delta.")

    if pair_report:
        lines.extend(["", pair_report])
    return lines


def check_one_pkl(
    pkl_path: Path,
    out_dir: Path,
    trans_threshold: float,
    rot_threshold_deg: float,
    keypoint_threshold: float,
    frame_a: int | None,
    frame_b: int | None,
    dpi: int,
) -> CheckResult:
    episode = load_episode(pkl_path)
    trans_delta, rot_delta, _rel_rotvec = _pose_deltas(episode.pose_mats)
    key_deltas = _keypoint_deltas(episode.pts21)
    pair_report = _format_pair_report(episode, frame_a, frame_b)

    out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = out_dir / f"{pkl_path.stem}_trajectory_delta.png"
    report_path = out_dir / f"{pkl_path.stem}_trajectory_delta.txt"

    _make_dashboard(
        episode=episode,
        out_path=figure_path,
        trans_delta=trans_delta,
        rot_delta=rot_delta,
        key_deltas=key_deltas,
        pair_report=pair_report,
        trans_threshold=trans_threshold,
        rot_threshold_deg=rot_threshold_deg,
        keypoint_threshold=keypoint_threshold,
        dpi=dpi,
    )

    report_lines = build_report_lines(
        episode=episode,
        trans_delta=trans_delta,
        rot_delta=rot_delta,
        key_deltas=key_deltas,
        pair_report=pair_report,
        trans_threshold=trans_threshold,
        rot_threshold_deg=rot_threshold_deg,
        keypoint_threshold=keypoint_threshold,
    )
    _write_report(report_path, report_lines)

    max_trans, trans_pair, _ = _max_pair(trans_delta)
    max_rot_rad, rot_pair, _ = _max_pair(rot_delta)
    key_max = key_deltas.get("max")
    key_mean = key_deltas.get("mean")
    max_key, key_pair, key_i = _max_pair(key_max)
    max_mean, _, _ = _max_pair(key_mean)
    key_name = None
    if key_i is not None:
        key_name = MP_NAMES[int(key_deltas["max_joint"][key_i])]

    return CheckResult(
        pkl_path=pkl_path,
        figure_path=figure_path,
        report_path=report_path,
        num_messages=len(episode.messages),
        num_pose_frames=0 if episode.pose_mats is None else len(episode.pose_mats.values),
        num_pts21_frames=0 if episode.pts21 is None else len(episode.pts21.values),
        max_trans_delta=max_trans,
        max_trans_pair=trans_pair,
        max_rot_delta_deg=float(np.rad2deg(max_rot_rad)),
        max_rot_pair=rot_pair,
        max_keypoint_delta=max_key,
        max_keypoint_pair=key_pair,
        max_keypoint_name=key_name,
        max_mean_keypoint_delta=max_mean,
        pair_report=pair_report,
    )


def _iter_pkl_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.pkl" if recursive else "*.pkl"
    return sorted(input_path.glob(pattern))


def _write_summary_csv(results: list[CheckResult], out_dir: Path) -> Path:
    csv_path = out_dir / "trajectory_delta_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pkl",
                "messages",
                "pose_frames",
                "pts21_frames",
                "max_trans_delta_m",
                "max_trans_pair",
                "max_rot_delta_deg",
                "max_rot_pair",
                "max_keypoint_delta_m",
                "max_keypoint_pair",
                "max_keypoint_name",
                "max_mean_keypoint_delta_m",
                "figure",
                "report",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    str(r.pkl_path),
                    r.num_messages,
                    r.num_pose_frames,
                    r.num_pts21_frames,
                    f"{r.max_trans_delta:.8f}",
                    r.max_trans_pair,
                    f"{r.max_rot_delta_deg:.6f}",
                    r.max_rot_pair,
                    f"{r.max_keypoint_delta:.8f}",
                    r.max_keypoint_pair,
                    r.max_keypoint_name,
                    f"{r.max_mean_keypoint_delta:.8f}",
                    str(r.figure_path) if r.figure_path else "",
                    str(r.report_path) if r.report_path else "",
                ]
            )
    return csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check adjacent-frame trajectoryPose and pts21_mano deltas in DexUMI PKL files."
    )
    parser.add_argument("input", type=Path, help="PKL file or directory containing PKL files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for figures/reports. Default: <input_dir>/trajectory_delta_checks.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, search PKL files recursively.",
    )
    parser.add_argument(
        "--trans-threshold",
        type=float,
        default=0.05,
        help="Translation spike threshold in meters per valid frame.",
    )
    parser.add_argument(
        "--rot-threshold-deg",
        type=float,
        default=5.0,
        help="Rotation spike threshold in degrees per valid frame.",
    )
    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=0.03,
        help="Per-joint keypoint spike threshold in meters per valid frame.",
    )
    parser.add_argument("--frame-a", type=int, default=None, help="Optional original frame index A.")
    parser.add_argument("--frame-b", type=int, default=None, help="Optional original frame index B.")
    parser.add_argument("--dpi", type=int, default=150, help="Saved figure DPI.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.out_dir is None:
        base_dir = input_path.parent if input_path.is_file() else input_path
        out_dir = base_dir / "trajectory_delta_checks"
    else:
        out_dir = args.out_dir.expanduser().resolve()

    pkl_files = _iter_pkl_files(input_path, recursive=args.recursive)
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found under: {input_path}")

    print(f"[Info] Found {len(pkl_files)} PKL file(s). Output: {out_dir}")
    results: list[CheckResult] = []
    for pkl_path in pkl_files:
        try:
            result = check_one_pkl(
                pkl_path=pkl_path,
                out_dir=out_dir,
                trans_threshold=args.trans_threshold,
                rot_threshold_deg=args.rot_threshold_deg,
                keypoint_threshold=args.keypoint_threshold,
                frame_a=args.frame_a,
                frame_b=args.frame_b,
                dpi=args.dpi,
            )
            results.append(result)
            print(
                f"[OK] {pkl_path.name}: "
                f"max_trans={result.max_trans_delta:.5f}m pair={result.max_trans_pair}, "
                f"max_rot={result.max_rot_delta_deg:.2f}deg pair={result.max_rot_pair}, "
                f"max_kp={result.max_keypoint_delta:.5f}m pair={result.max_keypoint_pair} "
                f"joint={result.max_keypoint_name}"
            )
            print(f"     figure: {result.figure_path}")
            print(f"     report: {result.report_path}")
            if result.pair_report:
                print(result.pair_report)
        except Exception as exc:
            print(f"[Skip] {pkl_path}: {exc}")

    if results:
        csv_path = _write_summary_csv(results, out_dir)
        print(f"[Done] Summary CSV: {csv_path}")
    else:
        raise RuntimeError("No PKL file was processed successfully.")


if __name__ == "__main__":
    main()
