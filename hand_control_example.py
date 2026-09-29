#!/usr/bin/env python3
"""
灵巧手关节控制示例。

两种控制方式：
  方式 A  运动学控制（直接写 data.qpos）
          不使用执行器，适合回放轨迹 / 可视化
  方式 B  动力学控制（写 data.ctrl，位置执行器）
          在物理仿真中闭环跟踪目标角度

运行示例：
  python3 hand_control_example.py                   # 方式 A，打开 / 合拢循环
  python3 hand_control_example.py --mode dynamic    # 方式 B，打开 / 合拢循环
  python3 hand_control_example.py --mode dynamic --urdf /path/to/xxx.urdf
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from load_in_mujoco import load_hand, get_joint_index, get_actuator_index, print_model_summary


# ── 手势定义（主动关节目标角度，单位 rad） ─────────────────────────────────────
# 说明：slave (mimic) 关节无需设置，会由 equality 约束自动跟随。
#
# 对于 linkerhand_o6_right 系列，主动关节为：
#   thumb_cmc_yaw, thumb_cmc_pitch,
#   index_mcp_pitch, middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch
#
# 若你的 URDF 关节名带 rh_ 前缀，改成 rh_thumb_cmc_yaw 等即可。

POSES: dict[str, dict[str, float]] = {
    "open": {
        "thumb_cmc_yaw":   0.0,
        "thumb_cmc_pitch": 0.0,
        "index_mcp_pitch": 0.0,
        "middle_mcp_pitch": 0.0,
        "ring_mcp_pitch":  0.0,
        "pinky_mcp_pitch": 0.0,
    },
    "close": {
        "thumb_cmc_yaw":   0.8,
        "thumb_cmc_pitch": 0.5,
        "index_mcp_pitch": 1.4,
        "middle_mcp_pitch": 1.4,
        "ring_mcp_pitch":  1.4,
        "pinky_mcp_pitch": 1.4,
    },
    "pinch": {
        "thumb_cmc_yaw":   1.0,
        "thumb_cmc_pitch": 0.4,
        "index_mcp_pitch": 1.2,
        "middle_mcp_pitch": 0.0,
        "ring_mcp_pitch":  0.0,
        "pinky_mcp_pitch": 0.0,
    },
    "victory": {
        "thumb_cmc_yaw":   0.5,
        "thumb_cmc_pitch": 0.3,
        "index_mcp_pitch": 0.0,
        "middle_mcp_pitch": 0.0,
        "ring_mcp_pitch":  1.4,
        "pinky_mcp_pitch": 1.4,
    },
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── 方式 A：运动学控制（直接设 qpos） ─────────────────────────────────────────

def _apply_mimic_equalities(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """
    在纯运动学（kinematic）模式下手动求解所有 mjEQ_JOINT 约束。

    MuJoCo 的 mj_forward 不会强制 equality 约束（只在 mj_step 物理求解中处理），
    所以 mimic 关节（IP/DIP 等从动指节）的 qpos 不会自动跟随主控关节。这里我们
    按 URDF 注入的 polycoef = [offset, multiplier, 0, 0, 0] 直接计算：

        q_slave = c0 + c1 * q_master + c2 * q_master^2 + c3 * q_master^3 + c4 * q_master^4

    并写回 data.qpos[slave]。调用方应在主控关节赋值完毕后再调用本函数。
    """
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        slave_jid = model.eq_obj1id[i]
        master_jid = model.eq_obj2id[i]
        if slave_jid < 0 or master_jid < 0:
            continue
        slave_qi = model.jnt_qposadr[slave_jid]
        master_qi = model.jnt_qposadr[master_jid]
        c = model.eq_data[i]
        qm = data.qpos[master_qi]
        q_slave = c[0] + c[1] * qm + c[2] * qm**2 + c[3] * qm**3 + c[4] * qm**4
        lo, hi = model.jnt_range[slave_jid]
        if lo < hi:
            q_slave = max(lo, min(hi, q_slave))
        data.qpos[slave_qi] = q_slave


def kinematic_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: dict[str, float],
) -> None:
    """
    直接将目标角度写入 data.qpos，然后调用 mj_forward 更新位置。
    不涉及物理仿真，适合可视化和轨迹回放。

    在写完主控关节后会**手动求解 mimic equality** 以驱动 IP/DIP 等从动指节，
    因为 mj_forward 不会自动处理 equality 约束。
    """
    for joint_name, angle in target.items():
        try:
            qi = get_joint_index(model, joint_name)
            lo, hi = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
            data.qpos[qi] = _clamp(angle, lo, hi)
        except KeyError:
            pass  # 当前 URDF 不含该关节，跳过
    _apply_mimic_equalities(model, data)
    mujoco.mj_forward(model, data)


# ── 方式 B：动力学控制（位置执行器 data.ctrl） ────────────────────────────────

def dynamic_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: dict[str, float],
) -> None:
    """
    通过位置执行器发送目标角度（写入 data.ctrl）。
    执行器命名约定：act_<joint_name>（由 load_hand(add_actuators=True) 自动生成）。
    需配合 mj_step 使用（物理仿真闭环）。
    """
    for joint_name, angle in target.items():
        act_name = f"act_{joint_name}"
        try:
            ai = get_actuator_index(model, act_name)
            lo, hi = model.actuator_ctrlrange[ai]
            data.ctrl[ai] = _clamp(angle, lo, hi)
        except KeyError:
            pass


# ── 插值工具 ─────────────────────────────────────────────────────────────────

def interpolate(
    start: dict[str, float],
    end: dict[str, float],
    t: float,
) -> dict[str, float]:
    """在两个姿态之间按 t∈[0,1] 线性插值。"""
    keys = set(start) | set(end)
    return {k: start.get(k, 0.0) * (1 - t) + end.get(k, 0.0) * t for k in keys}


# ── 主逻辑 ──────────────────────────────────────────────────────────────────

def run_kinematic(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """运动学模式：在 open/close/pinch 之间循环，查看器静态展示。"""
    sequence = ["open", "close", "open", "pinch", "victory", "open"]
    steps = 60          # 每段插值帧数
    hold = 30           # 到达姿态后停留帧数

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.4
        idx = 0
        while viewer.is_running():
            a = POSES[sequence[idx % len(sequence)]]
            b = POSES[sequence[(idx + 1) % len(sequence)]]

            for step in range(steps):
                if not viewer.is_running():
                    return
                t = step / steps
                t_smooth = t * t * (3 - 2 * t)   # smoothstep
                kinematic_control(model, data, interpolate(a, b, t_smooth))
                viewer.sync()
                time.sleep(1 / 60)

            for _ in range(hold):
                if not viewer.is_running():
                    return
                kinematic_control(model, data, b)
                viewer.sync()
                time.sleep(1 / 60)

            idx += 1


def run_dynamic(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """动力学模式：物理仿真 + 位置执行器，open/close 循环。"""
    model.opt.gravity[:] = 0       # 无重力，避免手指因自重下垂
    sequence = ["open", "close", "open", "pinch", "victory", "open"]
    phase_time = 2.0               # 每段持续秒数
    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.4
        t_wall = 0.0
        idx = 0
        while viewer.is_running():
            a = POSES[sequence[idx % len(sequence)]]
            b = POSES[sequence[(idx + 1) % len(sequence)]]

            phase_steps = int(phase_time / dt)
            for step in range(phase_steps):
                if not viewer.is_running():
                    return
                t = step / phase_steps
                t_smooth = t * t * (3 - 2 * t)
                dynamic_control(model, data, interpolate(a, b, t_smooth))
                mujoco.mj_step(model, data)
                viewer.sync()

            idx += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="灵巧手关节控制示例")
    parser.add_argument(
        "--urdf", type=Path, default=None,
        help="URDF 路径（默认：脚本同目录 linkerhand_o6_right.urdf）",
    )
    parser.add_argument(
        "--mode", choices=["kinematic", "dynamic"], default="kinematic",
        help="kinematic=直接写 qpos（默认），dynamic=位置执行器仿真",
    )
    args = parser.parse_args()

    urdf = args.urdf or (Path(__file__).resolve().parent / "linkerhand_o6_right.urdf")
    use_actuators = (args.mode == "dynamic")

    try:
        model, data = load_hand(
            urdf,
            fix_mimic=True,
            add_actuators=use_actuators,
            actuator_kp=5.0,
        )
    except Exception as e:
        print(f"加载失败: {e}")
        return 1

    print_model_summary(model)
    print(f"控制模式: {args.mode}")

    if args.mode == "kinematic":
        run_kinematic(model, data)
    else:
        run_dynamic(model, data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
