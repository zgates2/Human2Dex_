#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PICO -> dex-retargeting -> Linker O6 真机 CAN 控制。

调用链：
    PICO 26x7 → PicoHandReader.read_mano()    -> (21, 3) MANO 关键点
                LinkerO6Retargeter.retarget() -> (6,) 弧度
                _angles_rad_to_cmd()           -> (6,) 0~255 整数
                O6RightHand.move()             -> CAN 帧

弧度 → 0~255 线性映射
    cmd = 250 * (1 - qpos / upper),  clip 到 [0, 255]
    qpos=0 (URDF 张开) → cmd=250 (INIT_POS)
    qpos=upper(URDF 最大屈曲) → cmd=0 (完全闭合)

URDF 上限值从 /home/zjc/桌面/DexUMI/linker_o6/linker_o6/right/linkerhand_o6_right.urdf 摘出，
若 URDF 修改请同步本文件中的 URDF_UPPER。

运行：
    conda activate pico

    sudo ip link set can0 up type can bitrate 1000000

    python3 teleop_real.py
    python3 teleop_real.py --hz 60 --can-channel can0
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_O6_DIR = _REPO_ROOT / "o6_right_hand"
if str(_O6_DIR) not in sys.path:
    sys.path.insert(0, str(_O6_DIR))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from controller import O6RightHand  # noqa: E402

from pico_hand import (  # noqa: E402
    PicoHandReader,
    format_keypoints_debug,
    format_angles_debug,
)
from retargeter import LinkerO6Retargeter, LINKER_JOINTS  # noqa: E402


# URDF 中各主动关节的弧度上限 (lower 均为 0)
URDF_UPPER: dict[str, float] = {
    "thumb_cmc_pitch": 0.58,
    "thumb_cmc_yaw":   1.36,
    "index_mcp_pitch": 1.60,
    "middle_mcp_pitch": 1.60,
    "ring_mcp_pitch":  1.60,
    "pinky_mcp_pitch": 1.60,
}


def _angles_rad_to_cmd(angles_rad: np.ndarray) -> list[int]:
    """把 (6,) 弧度向量按 LINKER_JOINTS 顺序映射为 (6,) 0~255 整数。"""
    if len(angles_rad) != len(LINKER_JOINTS):
        raise ValueError(
            f"angles_rad 长度 {len(angles_rad)} 不等于 {len(LINKER_JOINTS)}"
        )
    cmds: list[int] = []
    for name, q in zip(LINKER_JOINTS, angles_rad):
        upper = URDF_UPPER[name]
        q_clipped = float(np.clip(q, 0.0, upper))
        v = 250.0 * (1.0 - q_clipped / upper)
        cmds.append(int(np.clip(round(v), 0, 255)))
    return cmds


def main():
    parser = argparse.ArgumentParser(
        description="PICO -> Linker O6 真机 CAN 控制"
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="覆盖 retargeting 用的 linker_o6.yml 路径",
    )
    parser.add_argument(
        "--hand",
        choices=["right"],
        default="right",
        help="目前只支持右手",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="主循环目标频率，默认 30 Hz",
    )
    parser.add_argument(
        "--can-channel",
        type=str,
        default="can0",
        help="SocketCAN 接口名，默认 can0",
    )
    parser.add_argument(
        "--bitrate",
        type=int,
        default=1_000_000,
        help="CAN 波特率，默认 1 Mbps",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=200,
        help="O6 全关节运行速度 0~255，默认 200",
    )
    parser.add_argument(
        "--torque",
        type=int,
        default=None,
        help="可选：统一设置力矩上限 0~255",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印映射后的指令，不连接 CAN（用于离线调试映射公式）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印每帧的 PICO 关键点与重定向角度调试信息",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=None,
        help="调试节流：每 N 帧打印一次（默认按 --hz 推算 ≈ 1 秒一次）",
    )
    args = parser.parse_args()

    # ── 初始化 PICO + retargeter ──────────────────────────────────────
    print("[Init] 初始化 PICO 数据采集...")
    reader = PicoHandReader(hand=args.hand)

    print("[Init] 加载 dex-retargeting 配置...")
    if args.yaml is None:
        retargeter = LinkerO6Retargeter(hand=args.hand)
    else:
        retargeter = LinkerO6Retargeter(yaml_path=args.yaml, hand=args.hand)

    # 主循环节拍
    period = 1.0 / max(args.hz, 1e-3)
    interrupted = {"flag": False}

    def _sigint(_sig, _frame):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    debug_every = args.debug_every if args.debug_every is not None else max(
        int(args.hz), 1
    )

    if args.dry_run:
        print("[DryRun] 不连接 CAN，仅打印指令")
        if args.debug:
            print(f"[Debug] 已开启，每 {debug_every} 帧打印关键点详情")
        frame_idx = 0
        try:
            while not interrupted["flag"]:
                t0 = time.perf_counter()

                if args.debug:
                    pts21, active, dbg = reader.read_mano(return_debug=True)
                else:
                    pts21, active = reader.read_mano()
                    dbg = None

                if pts21 is not None and active == 1:
                    angles = retargeter.retarget(pts21)
                    cmd = _angles_rad_to_cmd(angles)
                else:
                    angles = None
                    cmd = None

                if args.debug and frame_idx % debug_every == 0:
                    try:()
                    except Exception:
                        ts = None
                    print(
                        format_keypoints_debug(
                            pts21, active, ts, frame_idx, debug=dbg
                        )
                    )
                    if angles is not None:
                        print(
                            format_angles_debug(
                                angles, joint_names=LINKER_JOINTS, cmd_0_255=cmd
                            )
                        )

                if not args.debug:
                    if cmd is not None:
                        print(
                            f"active={active} "
                            f"rad={np.round(angles, 3).tolist()} cmd={cmd}"
                        )
                    else:
                        print(f"active={active} (skip)")

                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
                frame_idx += 1
        finally:
            reader.close()
        return

    print(
        f"[Init] 连接 Linker O6 (channel={args.can_channel}, bitrate={args.bitrate})..."
    )
    try:
        with O6RightHand(can_channel=args.can_channel, bitrate=args.bitrate) as hand:
            hand.set_speed(args.speed)
            if args.torque is not None:
                hand.set_torque(args.torque)

            print("[Init] 回到张开位 ...")
            hand.home()
            time.sleep(1.0)

            print("[Run] 开始遥操作，Ctrl+C 退出")
            if args.debug:
                print(f"[Debug] 已开启，每 {debug_every} 帧打印一次")

            miss_count = 0
            frame_idx = 0
            last_angles = np.zeros(len(LINKER_JOINTS), dtype=np.float32)
            last_cmd: list[int] = [250] * len(LINKER_JOINTS)
            while not interrupted["flag"]:
                t0 = time.perf_counter()

                if args.debug:
                    pts21, active, dbg = reader.read_mano(return_debug=True)
                else:
                    pts21, active = reader.read_mano()
                    dbg = None

                if pts21 is not None and active == 1:
                    angles = retargeter.retarget(pts21)
                    cmd = _angles_rad_to_cmd(angles)
                    hand.move(cmd)
                    last_angles = angles
                    last_cmd = cmd
                    miss_count = 0
                else:
                    miss_count += 1
                    if miss_count == int(args.hz):
                        print("[Warn] 已连续 1s 未收到高质量手部数据")

                if args.debug and frame_idx % debug_every == 0:
                    try:
                        ts = reader.get_timestamp_ns()
                    except Exception:
                        ts = None
                    print(
                        format_keypoints_debug(
                            pts21, active, ts, frame_idx, debug=dbg
                        )
                    )
                    print(
                        format_angles_debug(
                            last_angles,
                            joint_names=LINKER_JOINTS,
                            cmd_0_255=last_cmd,
                        )
                    )

                dt = time.perf_counter() - t0
                sleep_for = period - dt
                if sleep_for > 0:
                    time.sleep(sleep_for)
                frame_idx += 1

            print("\n[Exit] 用户中断，回到张开位 ...")
            try:
                hand.home()
                time.sleep(0.5)
            except Exception as e:
                print(f"[Warn] home() 失败: {e}")
    finally:
        reader.close()
        print("[Exit] PICO 已释放")


if __name__ == "__main__":
    main()
