#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKL → Franka 机械臂轨迹回放（PICO 采集格式）

参考:
  - replay_traj.py: 相对位姿回放思路（inv(T0) @ T_i → 叠加到机器人当前位姿）
  - teleop/PKL_REPLAY_SPEC.md: PKL 数据格式与坐标系约定

PKL 数据格式（teleop/collect_data.py 产出）:
    {
      "messages": [
        {
          "timestamp":            float,         # time.time() 墙钟秒
          "mainClockMonotonicNs": int,           # PICO 单调时钟，纳秒（推荐用于节拍）
          "o6_command":           np.uint8 (6,) or None,
          "trajectoryPose":       np.float32 (7,) = [x,y,z,qx,qy,qz,qw] or None,
          ...
        }, ...
      ]
    }

坐标系（重要！见 PKL_REPLAY_SPEC.md §2.1）:
    trajectoryPose 是 PICO SDK 世界追踪系下 Palhum 的绝对位姿，
    既不是机器人 base 系，也不是 MANO 系。
    本脚本采取「相对回放」策略：
        T_rel(i) = inv(T_pico_palm[0]) @ T_pico_palm[i]
        T_robot_target(i) = T_robot_start @ T_rel(i)
    这样无需 PICO→Robot 外参标定，只回放手掌相对录制起点的运动。

    注意：PICO World 与 Robot base 通常旋转轴不对齐，
    "相对" 位姿的姿态分量仍是在 PICO World 表达，
    直接乘到 robot_start 上会得到与录制时手掌运动 *形状一致* 但
    姿态轴方向不一定符合直觉的轨迹。若需精确对齐，请改用
    外参标定后的绝对回放（在 PKL_REPLAY_SPEC.md §4.2 中描述）。

用法:
+
    python3 replay_traj_pico.py /path/to/demo_xxx.pkl

    # 调整回放速率（默认 1.0 = 原速）
    python3 replay_traj_pico.py demo.pkl --rate 0.5

    # 跳过前若干帧（与 replay_traj.py 中的 i=3 起步类似）
    python3 replay_traj_pico.py demo.pkl --skip 3

    # 不连接真机，只打印目标位姿（调试用）
    python3 replay_traj_pico.py demo.pkl --dry-run
"""
from __future__ import annotations

import argparse
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


sys.path.append("/home/ps/reactive_diffusion_policy")
sys.path.append(str(Path(__file__).resolve().parent / "reactive_diffusion_policy"))

import transforms3d as t3d
from spatialmath import SE3
from spatialmath.base import tr2angvec

from reactive_diffusion_policy.real_world.robot.single_flexiv_controller import (
    FlexivController,
)
from reactive_diffusion_policy.common.space_utils import matrix4x4_to_pose_6d


# ── PKL 加载 ──────────────────────────────────────────────────────────────────

def _trajectory_pose_to_matrix(pose_xyzqxqyqzqw: np.ndarray) -> np.ndarray:
    """trajectoryPose [x,y,z,qx,qy,qz,qw] → 4x4 齐次变换矩阵。

    注意四元数顺序：PICO 给出的是 (qx, qy, qz, qw)，
    而 transforms3d 的 quat2mat 需要 (w, x, y, z)，需要重排。
    """
    p = np.asarray(pose_xyzqxqyqzqw, dtype=np.float64).reshape(-1)
    if p.shape[0] != 7:
        raise ValueError(f"trajectoryPose 必须为长度 7 的向量，实际 shape={p.shape}")

    xyz = p[:3]
    qx, qy, qz, qw = p[3], p[4], p[5], p[6]
    quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat_wxyz /= (np.linalg.norm(quat_wxyz) + 1e-12)

    mat = np.eye(4)
    mat[:3, :3] = t3d.quaternions.quat2mat(quat_wxyz)
    mat[:3, 3] = xyz
    return mat


def load_pkl_trajectory(
    pkl_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 PKL 并提取轨迹相关数据。

    Returns
    -------
    poses_mat   : (N, 4, 4) PICO World 下 Palm 位姿矩阵序列
    times_sec   : (N,) 相对录制起点的时间序列，秒（来自 mainClockMonotonicNs；
                  若 PKL 中缺失，则用 timestamp 退回）
    hand_cmds   : (N, 6) uint8 同步的手部命令（无效填零）
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL 文件不存在: {pkl_path}")

    print(f"[Load] 读取 {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError("PKL 格式不符合预期：缺少 'messages' 键")

    messages: list = data["messages"]
    if len(messages) == 0:
        raise ValueError("PKL 中 messages 为空")

    poses, times_ns, times_sec_wall, hand_cmds = [], [], [], []
    skipped = 0
    for m in messages:
        traj = m.get("trajectoryPose")
        if traj is None:
            skipped += 1
            continue
        poses.append(_trajectory_pose_to_matrix(np.asarray(traj)))

        ns = m.get("mainClockMonotonicNs")
        times_ns.append(int(ns) if ns is not None else -1)
        ts = m.get("timestamp")
        times_sec_wall.append(float(ts) if ts is not None else float("nan"))

        cmd = m.get("o6_command", m.get("handCommand"))
        if cmd is not None and np.asarray(cmd).shape == (6,):
            hand_cmds.append(np.asarray(cmd, dtype=np.uint8))
        else:
            hand_cmds.append(np.zeros(6, dtype=np.uint8))

    if len(poses) == 0:
        raise ValueError("PKL 中没有任何带 trajectoryPose 的有效帧")

    poses_mat = np.stack(poses, axis=0)
    times_ns_arr = np.asarray(times_ns, dtype=np.int64)
    times_wall_arr = np.asarray(times_sec_wall, dtype=np.float64)
    hand_cmds_arr = np.stack(hand_cmds, axis=0)

    if np.all(times_ns_arr > 0):
        times_sec = (times_ns_arr - times_ns_arr[0]) * 1e-9
    elif not np.all(np.isnan(times_wall_arr)):
        times_sec = times_wall_arr - times_wall_arr[0]
    else:
        times_sec = np.arange(len(poses_mat)) / 30.0
        print("[Warn] PKL 中无可用时间戳，回退为固定 30 Hz")

    for i in range(1, len(times_sec)):
        if times_sec[i] < times_sec[i - 1]:
            times_sec[i] = times_sec[i - 1]

    print(
        f"[Load] 总 {len(messages)} 条 messages，"
        f"有效轨迹帧 {len(poses_mat)}，跳过 {skipped}，"
        f"原时长 {times_sec[-1]:.2f}s"
    )
    return poses_mat, times_sec, hand_cmds_arr


# ── 回放 ──────────────────────────────────────────────────────────────────────

def _matrix_to_control_7d(target_mat: np.ndarray, gripper: float = -1.0) -> List[float]:
    """4x4 → deoxys OSC 7D [x,y,z,ax,ay,az,gripper]（轴角）。"""
    se3 = SE3(target_mat, check=False)
    theta, v = tr2angvec(se3.R)
    axisangle = (v * theta).flatten().tolist()
    return se3.t.tolist() + axisangle + [gripper]


def replay_relative_from_pkl(
    controller: Optional[FlexivController],
    pkl_path: str,
    rate: float = 1.0,
    skip: int = 0,
    log_every: int = 30,
    dry_run: bool = False,
    interrupted: Optional[dict] = None,
) -> None:
    """以 PKL 第 skip 帧为参考，将后续相对运动叠加到机器人当前位姿。"""
    if interrupted is None:
        interrupted = {"flag": False}

    poses_mat, times_sec, _hand_cmds = load_pkl_trajectory(Path(pkl_path))
    n = len(poses_mat)
    if n < 2:
        print("[Replay] 轨迹点不足 2 个，无法回放")
        return

    skip = max(0, min(skip, n - 2))
    ref_mat = poses_mat[skip]
    ref_mat_inv = np.linalg.inv(ref_mat)

    if dry_run:
        robot_start_mat = np.eye(4)
        print("[DryRun] 假设机器人起点为单位矩阵")
    else:
        assert controller is not None
        print("[Replay] 等待机器人状态稳定...")
        time.sleep(2.0)
        robot_start_mat = (
            np.array(controller.robot_interface._state_buffer[-1].O_T_EE)
            .reshape(4, 4)
            .transpose()
        )
        print(f"[Replay] 机器人起点位置 (xyz): {robot_start_mat[:3, 3]}")

    print(
        f"[Replay] 共 {n} 帧，参考帧索引={skip}，将回放 {n - skip - 1} 个相对运动，"
        f"速率 {rate:.2f}x"
    )

    rate = max(float(rate), 1e-3)
    t0_data = float(times_sec[skip])
    t0_wall = time.perf_counter()
    last_log_time = t0_wall

    for i in range(skip + 1, n):
        if interrupted["flag"]:
            print("\n[Replay] 用户中断")
            break

        target_wall = t0_wall + (float(times_sec[i]) - t0_data) / rate
        while True:
            wait = target_wall - time.perf_counter()
            if wait <= 0 or interrupted["flag"]:
                break
            time.sleep(min(wait, 0.05))

        rel_mat = ref_mat_inv @ poses_mat[i]
        target_mat = robot_start_mat @ rel_mat
        target_7d = _matrix_to_control_7d(target_mat, gripper=-1.0)

        if dry_run:
            if log_every > 0 and (i % log_every == 0 or i == n - 1):
                rel_6d = matrix4x4_to_pose_6d(rel_mat)
                print(
                    f"[DryRun] {i}/{n - 1} t={times_sec[i] - t0_data:6.2f}s  "
                    f"rel_6d={np.round(rel_6d, 4).tolist()}  "
                    f"target_xyz={np.round(target_7d[:3], 4).tolist()}"
                )
        else:
            controller.set_target_pose(target_7d)

            if log_every > 0 and (i % log_every == 0 or i == n - 1):
                now = time.perf_counter()
                hz = log_every / max(now - last_log_time, 1e-6)
                last_log_time = now
                print(
                    f"[Replay] {i}/{n - 1}  t={times_sec[i] - t0_data:6.2f}s  "
                    f"实际频率≈{hz:.1f} Hz  target_xyz={np.round(target_7d[:3], 4).tolist()}"
                )

    print("-------------------------------- 相对轨迹回放完成 --------------------------------")
    if not dry_run:
        time.sleep(2.0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于 PICO 采集 PKL 的 Franka 机械臂相对轨迹回放"
    )
    parser.add_argument("pkl", type=Path, help="PICO 采集的 PKL 文件路径")
    parser.add_argument("--rate", type=float, default=1.0, help="回放速率倍率，默认 1.0")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="参考帧索引；从该帧开始计算相对位移（默认 0，即第一帧）",
    )
    parser.add_argument(
        "--force-sensor-port",
        type=str,
        default="/dev/ttyUSB0",
        help="FlexivController 的力传感器端口（默认 /dev/ttyUSB0）",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="每 N 帧打印一次进度（设为 0 关闭）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不连接机器人，仅打印目标位姿（调试 PKL/坐标）",
    )
    args = parser.parse_args()

    interrupted = {"flag": False}

    def _sigint(_sig, _frame):
        print("\n[Ctrl+C] 准备退出...")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _sigint)

    if args.dry_run:
        replay_relative_from_pkl(
            controller=None,
            pkl_path=str(args.pkl),
            rate=args.rate,
            skip=args.skip,
            log_every=args.log_every,
            dry_run=True,
            interrupted=interrupted,
        )
        return

    controller = FlexivController(force_sensor_port=args.force_sensor_port)
    try:
        print("[Init] 重置到初始位置...")
        controller.reset_to_home()
        print("[Init] 已到达初始位置，准备回放。")

        replay_relative_from_pkl(
            controller=controller,
            pkl_path=str(args.pkl),
            rate=args.rate,
            skip=args.skip,
            log_every=args.log_every,
            dry_run=False,
            interrupted=interrupted,
        )

    except Exception as e:
        print(f"[Error] 程序执行出错: {e}")
        raise
    finally:
        print("[Exit] 回放结束或出现错误，正在停止机器人...")
        if (
            hasattr(controller, "control_thread")
            and controller.control_thread is not None
            and controller.control_thread.is_alive()
        ):
            print("[Exit] 正在停止控制线程...")
            controller._stop_control_thread()
        print("[Exit] 程序已安全退出。")


if __name__ == "__main__":
    main()
