#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linker Hand O6 右手 CAN 控制 —— 交互式命令行入口

运行：
    python3 main.py [--channel can0] [--speed 150] [--torque 200]

内置命令：
    <预设名>   执行预设动作（见下方列表）
    home       回到完全张开位置
    state      打印当前关节位置
    temp       打印电机温度
    fault      打印故障码
    touch      读取压感摘要（若硬件支持）
    speed <n>  运行时修改速度（0~255）
    torque <n> 运行时修改力矩（0~255）
    help       显示帮助
    q / quit   退出

之行前先进行  sudo chmod a+x main.py

"""

import argparse
import sys

from controller import O6RightHand
from presets import JOINT_NAMES, PRESET_ACTIONS


def parse_args():
    p = argparse.ArgumentParser(description="Linker Hand O6 右手 CAN 控制台")
    p.add_argument("--channel", default="can0", help="SocketCAN 接口名（默认 can0）")
    p.add_argument("--speed",   type=int, default=255, help="初始速度 0~255（默认 255）")
    p.add_argument("--torque",  type=int, default=255, help="初始力矩 0~255（默认 255）")
    return p.parse_args()


def print_help():
    print(
        "\n可用命令：\n"
        f"  预设动作 : {list(PRESET_ACTIONS)}\n"
        "  home     : 回到初始位置\n"
        "  state    : 查看关节位置\n"
        "  temp     : 查看电机温度\n"
        "  fault    : 查看故障码\n"
        "  touch    : 读取压感摘要\n"
        "  speed <n>  : 修改速度（0~255）\n"
        "  torque <n> : 修改力矩（0~255）\n"
        "  help     : 显示本帮助\n"
        "  q/quit   : 退出\n"
    )


def touch_summary(data: dict) -> dict:
    """把压感原始数据压缩成每维度最大值，方便快速查看。"""
    import numpy as np
    result = {}
    for key, val in data.items():
        if val is None:
            result[key] = 0
        elif isinstance(val, np.ndarray):
            result[key] = int(val.max())
        elif isinstance(val, (list, tuple)):
            flat = []
            for item in val:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                else:
                    flat.append(item)
            result[key] = max(flat) if flat else 0
        else:
            result[key] = 0
    return result


def main():
    args = parse_args()

    print("═" * 46)
    print("  Linker Hand O6 右手 CAN 控制台")
    print("═" * 46)
    print(f"  接口: {args.channel}  速度: {args.speed}  力矩: {args.torque}")
    print(f"  关节: {JOINT_NAMES}")
    print("  输入 help 查看命令列表")
    print("═" * 46 + "\n")

    try:
        hand = O6RightHand(can_channel=args.channel)
    except Exception as e:
        print(f"[错误] 无法打开 CAN 接口：{e}")
        sys.exit(1)

    with hand:
        hand.set_speed(args.speed)
        hand.set_torque(args.torque)

        while True:
            try:
                raw = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出")
                break

            if not raw:
                continue

            cmd, *rest = raw.split()
            cmd = cmd.lower()

            if cmd in ("q", "quit"):
                break

            elif cmd == "help":
                print_help()

            elif cmd == "home":
                hand.home()
                print("  -> 已回到初始位置")

            elif cmd == "state":
                print(f"  关节位置: {hand.get_state()}")

            elif cmd == "temp":
                print(f"  温度: {hand.get_temperature()} ℃")

            elif cmd == "fault":
                codes = hand.get_fault()
                ok = all(c == 0 for c in codes)
                print(f"  故障码: {codes}  {'✓ 正常' if ok else '⚠ 有故障'}")

            elif cmd == "touch":
                data = hand.get_touch()
                if data:
                    print(f"  压感摘要: {touch_summary(data)}")
                else:
                    print("  当前硬件无压感支持")

            elif cmd == "speed" and rest:
                try:
                    hand.set_speed(int(rest[0]))
                    print(f"  速度已设为 {rest[0]}")
                except ValueError as e:
                    print(f"  [错误] {e}")

            elif cmd == "torque" and rest:
                try:
                    hand.set_torque(int(rest[0]))
                    print(f"  力矩已设为 {rest[0]}")
                except ValueError as e:
                    print(f"  [错误] {e}")

            elif raw in PRESET_ACTIONS:
                hand.preset(raw)
                print(f"  -> {raw}: {PRESET_ACTIONS[raw]}")

            else:
                print(f"  未知命令 '{raw}'，输入 help 查看帮助")

    print("CAN 连接已关闭，再见。")


if __name__ == "__main__":
    main()
