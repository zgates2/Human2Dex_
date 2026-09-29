#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PICO -> dex-retargeting -> MuJoCo 实时遥操作（仅可视化）。

复用：
  - /home/zjc/桌面/DexUMI/load_in_mujoco.py      (load_hand)
  - /home/zjc/桌面/DexUMI/hand_control_example.py (kinematic_control)
  - teleop/pico_hand.py                          (PicoHandReader)
  - teleop/retargeter.py                         (LinkerO6Retargeter)

运行：
    python3 teleop_mujoco.py
    python3 teleop_mujoco.py --hz 60
    python3 teleop_mujoco.py --urdf /custom/path.urdf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

# 把仓库根目录加入 sys.path 以便导入 load_in_mujoco / hand_control_example
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from load_in_mujoco import load_hand, print_model_summary  # noqa: E402
from hand_control_example import kinematic_control  # noqa: E402

from pico_hand import (  # noqa: E402
    PicoHandReader,
    format_keypoints_debug,
    format_angles_debug,
)
from retargeter import LinkerO6Retargeter, LINKER_JOINTS  # noqa: E402


DEFAULT_URDF = (
    "/home/zjc/桌面/DexUMI/linker_o6/linker_o6/right/linkerhand_o6_right.urdf"
)


def main():
    parser = argparse.ArgumentParser(
        description="PICO -> Linker O6 实时遥操作（MuJoCo 可视化）"
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(DEFAULT_URDF),
        help=f"URDF 路径，默认 {DEFAULT_URDF}",
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
        "--no-gravity",
        action="store_true",
        default=True,
        help="禁用重力，避免手指自重下垂（默认开启）",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="启动时打印 MuJoCo 模型摘要",
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

    if not args.urdf.is_file():
        sys.exit(f"找不到 URDF: {args.urdf}")

    # ── 初始化 PICO + dex-retargeting ─────────────────────────────────
    print("[Init] 初始化 PICO 数据采集...")
    reader = PicoHandReader(hand=args.hand)

    print("[Init] 加载 dex-retargeting 配置...")
    if args.yaml is None:
        retargeter = LinkerO6Retargeter(hand=args.hand)
    else:
        retargeter = LinkerO6Retargeter(yaml_path=args.yaml, hand=args.hand)

    # ── 初始化 MuJoCo ─────────────────────────────────────────────────
    print(f"[Init] 加载 MuJoCo 模型: {args.urdf}")
    model, data = load_hand(
        args.urdf, fix_mimic=True, add_actuators=False, actuator_kp=5.0
    )
    if args.no_gravity:
        model.opt.gravity[:] = 0

    if args.summary:
        print_model_summary(model)

    period = 1.0 / max(args.hz, 1e-3)
    last_target: dict[str, float] = {n: 0.0 for n in LINKER_JOINTS}
    last_angles = np.zeros(len(LINKER_JOINTS), dtype=np.float32)
    miss_count = 0
    frame_idx = 0

    debug_every = args.debug_every if args.debug_every is not None else max(
        int(args.hz), 1
    )

    print("[Run] 启动 viewer，按窗口 ESC 或 Ctrl+C 退出")
    if args.debug:
        print(f"[Debug] 已开启，每 {debug_every} 帧打印一次")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 0.4

            while viewer.is_running():
                t0 = time.perf_counter()

                if args.debug:
                    pts21, active, dbg = reader.read_mano(return_debug=True)
                else:
                    pts21, active = reader.read_mano()
                    dbg = None

                if pts21 is not None and active == 1:
                    angles = retargeter.retarget(pts21)
                    last_angles = angles
                    last_target = {
                        n: float(a) for n, a in zip(LINKER_JOINTS, angles)
                    }
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
                        format_angles_debug(last_angles, joint_names=LINKER_JOINTS)
                    )

                kinematic_control(model, data, last_target)
                viewer.sync()

                dt = time.perf_counter() - t0
                sleep_for = period - dt
                if sleep_for > 0:
                    time.sleep(sleep_for)
                frame_idx += 1
    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C 已捕获")
    finally:
        reader.close()
        print("[Exit] PICO 已释放")


if __name__ == "__main__":
    main()
