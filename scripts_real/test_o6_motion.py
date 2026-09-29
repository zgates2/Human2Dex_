#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linker O6 右手运动测试。

本脚本复用 ``eval_real_franka_o6.py`` 使用的 O6 接口：

* ``open_o6_hand(config)`` 建立 CAN 连接；
* ``set_speed`` / ``set_torque`` 设置运动参数；
* ``move([p0, ..., p5])`` 发送 6 轴位置（每轴 0~255）；
* ``get_state`` 读取当前位置；
* ``close`` 释放 CAN 资源。

默认动作比较保守：先张开，再逐个关节从 ``open_position`` 移到
``close_position``，然后返回张开位。O6 命令空间约定为：250 附近更张开，
0 附近更弯曲。正式连接硬件前可以先用 ``--dry-run`` 检查动作序列。

示例：

    # 只生成动作，不连接 CAN
    python scripts_real/test_o6_motion.py --dry-run

    # 使用 eval 配置中的 robot_config，连接 can0；运行前要求输入 YES
    python scripts_real/test_o6_motion.py

    # 已确认安全环境时跳过交互确认，并只测试食指和中指
    python scripts_real/test_o6_motion.py --yes --joints index,middle

    # 实时键盘控制（需要在真实终端中运行，按键无需回车）
    python scripts_real/test_o6_motion.py --keyboard
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from typing import Iterable, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOINT_NAMES = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
JOINT_ALIASES = {
    "thumb": 0,
    "thumb_pitch": 0,
    "thumb_cmc_pitch": 0,
    "thumb_yaw": 1,
    "thumb_cmc_yaw": 1,
    "index": 2,
    "index_mcp_pitch": 2,
    "middle": 3,
    "middle_mcp_pitch": 3,
    "ring": 4,
    "ring_mcp_pitch": 4,
    "pinky": 5,
    "little": 5,
    "pinky_mcp_pitch": 5,
}


def _position(value: object, name: str) -> int:
    """Validate one O6 position and return it as an integer."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer in [0, 255]") from exc
    if not number.is_integer() or not 0 <= number <= 255:
        raise argparse.ArgumentTypeError(f"{name} must be an integer in [0, 255]")
    return int(number)


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be a finite number >= 0")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return number


def _parse_joints(value: str) -> list[int]:
    """Parse comma-separated joint names or indices."""
    text = str(value).strip().lower()
    if text in ("", "all"):
        return list(range(6))

    result: list[int] = []
    for token in text.replace("，", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            index = int(token)
            if not 0 <= index < 6:
                raise argparse.ArgumentTypeError(f"joint index out of range: {token}")
        else:
            try:
                index = JOINT_ALIASES[token]
            except KeyError as exc:
                choices = ", ".join(JOINT_NAMES)
                raise argparse.ArgumentTypeError(
                    f"unknown joint {token!r}; use 0~5 or one of: {choices}"
                ) from exc
        if index not in result:
            result.append(index)
    if not result:
        raise argparse.ArgumentTypeError("at least one joint is required")
    return result


def _validate_pose(value: Sequence[object], name: str = "pose") -> list[int]:
    pose = list(value)
    if len(pose) != 6:
        raise ValueError(f"{name} must contain 6 values, got {len(pose)}")
    try:
        numbers = [float(item) for item in pose]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric value: {pose!r}") from exc
    if any(not math.isfinite(item) or not item.is_integer() for item in numbers):
        raise ValueError(f"{name} must contain integer-valued positions: {pose!r}")
    result = [int(item) for item in numbers]
    if any(item < 0 or item > 255 for item in result):
        raise ValueError(f"{name} values must be in [0, 255], got {result!r}")
    return result


def _pose_with_joint(base: Sequence[int], joint: int, value: int) -> list[int]:
    pose = list(base)
    pose[joint] = value
    return pose


def _read_state(hand) -> list[float]:
    state = list(hand.get_state())
    if len(state) != 6:
        raise RuntimeError(f"O6 get_state() returned {len(state)} values, expected 6: {state!r}")
    return state


class _DryRunO6:
    """Small local substitute that implements the O6 methods used by this script."""

    def __init__(self, initial_pose: Sequence[int]):
        self.state = list(initial_pose)

    def set_speed(self, speed: int) -> None:
        print(f"[DRY-RUN] set_speed({speed})")

    def set_torque(self, torque: int) -> None:
        print(f"[DRY-RUN] set_torque({torque})")

    def move(self, positions: Sequence[int]) -> None:
        self.state = _validate_pose(positions)
        print(f"[DRY-RUN] move({self.state})")

    def get_state(self) -> list[int]:
        return list(self.state)

    def close(self) -> None:
        print("[DRY-RUN] close()")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Linker O6 右手逐关节运动测试（直接发送 6D 位置命令）"
    )
    parser.add_argument(
        "-c",
        "--config",
        default="eval_franka_pts21_config.yaml",
        help="评估 YAML；用于读取 robot_config 和 linker_o6 配置",
    )
    parser.add_argument(
        "--robot-config",
        default=None,
        help="覆盖 YAML 中的 robot_config 路径",
    )
    parser.add_argument("--channel", default=None, help="SocketCAN 通道，默认使用配置值")
    parser.add_argument("--bitrate", type=int, default=None, help="CAN bitrate，默认使用配置值")
    parser.add_argument(
        "--speed",
        type=lambda value: _position(value, "speed"),
        default=None,
        help="统一速度 0~255，默认使用配置值或 200",
    )
    parser.add_argument(
        "--torque",
        type=lambda value: _position(value, "torque"),
        default=None,
        help="统一力矩上限 0~255；默认不修改配置",
    )
    parser.add_argument(
        "--open-position",
        type=lambda value: _position(value, "open-position"),
        default=None,
        help="张开位置，默认取 linker_o6.init_pose（通常为 250）",
    )
    parser.add_argument(
        "--close-position",
        type=lambda value: _position(value, "close-position"),
        default=180,
        help="单关节测试位置，默认 180；数值越小越弯曲",
    )
    parser.add_argument(
        "--joints",
        type=_parse_joints,
        default=list(range(6)),
        metavar="LIST",
        help="要测试的关节：all、0~5 或名称逗号列表（默认 all）",
    )
    parser.add_argument("--cycles", type=_nonnegative_int, default=1, help="完整测试循环次数")
    parser.add_argument(
        "--pause",
        type=_positive_float,
        default=1.0,
        help="每个动作后的等待时间（秒）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不连接 CAN，只打印将要执行的命令",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过硬件运动前的 YES 确认（仅在确认周围安全时使用）",
    )
    parser.add_argument(
        "--skip-open",
        action="store_true",
        help="跳过开始时的张开动作；仍会在结束时返回张开位",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="进入实时键盘控制模式（需要交互式终端；与自动测试序列互斥）",
    )
    parser.add_argument(
        "--step",
        type=lambda value: _position(value, "step"),
        default=5,
        help="键盘模式每次调节的步长，默认 5",
    )
    parser.add_argument(
        "--keyboard-delay",
        type=_positive_float,
        default=0.05,
        help="键盘模式每次发送后的等待时间（秒），默认 0.05",
    )
    return parser


def _load_o6_config(args: argparse.Namespace) -> dict:
    # Keep --dry-run usable in a lightweight environment where the deployment
    # stack (PyYAML/numpy) is not installed. Real hardware still uses the exact
    # same helpers as eval_real_franka_o6.py.
    try:
        from real_inference_config import load_yaml_mapping, merged_hand_config
    except ModuleNotFoundError as exc:
        if args.dry_run and exc.name in {"yaml", "numpy"}:
            print(f"[WARN] dry-run 无法加载配置依赖 {exc.name!r}，使用内置默认值。")
            return {}
        raise

    eval_cfg = load_yaml_mapping(args.config)
    robot_config_path = args.robot_config or eval_cfg.get(
        "robot_config", "example/eval_robots_config.yaml"
    )
    robot_cfg = load_yaml_mapping(robot_config_path)
    o6_cfg = merged_hand_config(eval_cfg, robot_cfg, "linker_o6")
    if args.channel is not None:
        o6_cfg["can_channel"] = args.channel
    if args.bitrate is not None:
        if args.bitrate <= 0:
            raise ValueError("bitrate must be > 0")
        o6_cfg["bitrate"] = args.bitrate
    return o6_cfg


def _confirm_motion() -> None:
    print("\n[安全提示] 即将驱动 O6 右手运动。请确认手指周围没有障碍物或人员。")
    if not sys.stdin.isatty():
        raise RuntimeError("非交互终端不能确认硬件运动，请使用 --yes（确认安全后再用）")
    answer = input("请输入大写 YES 开始运动，其他输入取消：").strip()
    if answer != "YES":
        raise RuntimeError("用户取消运动测试")


def _print_state(label: str, state: Iterable[object]) -> None:
    values = list(state)
    print(f"{label}: {[round(float(value), 2) for value in values]}")


def _keyboard_control(hand, current_pose: Sequence[int], open_pose: Sequence[int], args) -> None:
    """Run a single-key-at-a-time O6 position controller in a POSIX terminal."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError(
            "键盘模式需要交互式终端；请先 conda activate l515_mvs310，再直接运行 python，"
            "不要使用 conda run"
        )

    # Import terminal helpers lazily so --dry-run/自动测试仍可在非 POSIX 环境查看帮助。
    import select
    import termios
    import tty

    selected = 0
    pose = list(current_pose)
    step = int(args.step)
    if step <= 0:
        raise ValueError("keyboard step must be > 0")

    print(
        "\n键盘控制已启动（无需回车）：\n"
        "  0~5  选择关节       w/+  当前关节张开（+step）\n"
        "  s/-  当前关节弯曲（-step）   o/h  全部张开\n"
        "  p    读取反馈状态   r  用反馈状态重新作为控制起点\n"
        "  q/ESC 退出（退出前自动回到张开位）\n"
        f"当前步长: {step}；发送间隔: {args.keyboard_delay:.3f}s；"
        f"当前关节: 0:{JOINT_NAMES[0]}；当前位置: {pose}"
    )

    def send_pose(reason: str) -> None:
        hand.move(pose)
        print(f"\n[{reason}] command={pose}")
        if args.keyboard_delay > 0:
            time.sleep(args.keyboard_delay)
        _print_state("feedback", _read_state(hand))

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        try:
            while True:
                key = sys.stdin.read(1)
                if key in ("q", "Q", "\x1b"):
                    break
                if key in "012345":
                    selected = int(key)
                    print(f"\n[selected] {selected}:{JOINT_NAMES[selected]} position={pose[selected]}")
                    continue
                if key in ("w", "W", "+", "="):
                    pose[selected] = min(255, pose[selected] + step)
                    send_pose(f"joint {selected}:{JOINT_NAMES[selected]} open")
                    continue
                if key in ("s", "S", "-", "_"):
                    pose[selected] = max(0, pose[selected] - step)
                    send_pose(f"joint {selected}:{JOINT_NAMES[selected]} close")
                    continue
                if key in ("o", "O", "h", "H"):
                    pose = list(open_pose)
                    send_pose("open all")
                    continue
                if key in ("p", "P"):
                    _print_state("feedback", _read_state(hand))
                    continue
                if key in ("r", "R"):
                    pose = _validate_pose(_read_state(hand), "feedback state")
                    print(f"\n[reset] control pose={pose}")
                    continue
                print(f"\n[提示] 未知按键 {key!r}，按 o/h 张开全部关节")
        except KeyboardInterrupt:
            pose = list(open_pose)
            try:
                send_pose("interrupt: open all")
            except Exception as exc:
                print(f"\n[WARN] 中断时返回张开位失败: {exc}")
            raise
        # Keyboard mode returns before the common automatic-test epilogue, so
        # explicitly open the hand before leaving this loop.
        pose = list(open_pose)
        send_pose("exit: open all")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()


def run(args: argparse.Namespace) -> None:
    o6_cfg = _load_o6_config(args)
    configured_open = o6_cfg.get("init_pose", [250] * 6)
    open_pose = _validate_pose(
        configured_open if args.open_position is None else [args.open_position] * 6,
        "open_pose",
    )
    close_position = int(args.close_position)
    joints = list(args.joints)
    speed = int(args.speed if args.speed is not None else o6_cfg.get("speed", 200))
    if not 0 <= speed <= 255:
        raise ValueError(f"speed must be in [0, 255], got {speed}")
    torque = args.torque
    if torque is None and o6_cfg.get("torque") is not None:
        torque = int(o6_cfg["torque"])
    if torque is not None and not 0 <= torque <= 255:
        raise ValueError(f"torque must be in [0, 255], got {torque}")

    print("=" * 64)
    print("Linker O6 右手运动测试")
    print(f"关节顺序: {list(enumerate(JOINT_NAMES))}")
    print(f"测试关节: {[f'{i}:{JOINT_NAMES[i]}' for i in joints]}")
    print(f"张开位: {open_pose}; 单关节测试位: {close_position}")
    print(f"速度: {speed}; 力矩: {torque if torque is not None else '保持配置'}")
    print(f"循环: {args.cycles}; 动作等待: {args.pause:.2f}s; dry_run: {args.dry_run}")
    if args.keyboard:
        print(f"模式: 键盘控制; 步长: {args.step}; 发送间隔: {args.keyboard_delay:.3f}s")
    print("注意：本脚本直接发送原始位置值，不应用 eval 配置中的 command_bias。")
    print("=" * 64)

    if not args.dry_run and not args.yes:
        _confirm_motion()

    hand = _DryRunO6(open_pose) if args.dry_run else None
    try:
        if hand is None:
            from real_inference_config import open_o6_hand

            hand = open_o6_hand(o6_cfg)
        hand.set_speed(speed)
        if torque is not None:
            hand.set_torque(torque)

        initial_state = _read_state(hand)
        _print_state("初始反馈", initial_state)
        base_pose = _validate_pose(initial_state, "initial_state")

        if not args.skip_open:
            print(f"\n[动作] 张开: {open_pose}")
            hand.move(open_pose)
            time.sleep(args.pause)
            _print_state("张开后反馈", _read_state(hand))
            base_pose = open_pose
        else:
            print(f"[信息] 跳过开始张开动作，测试时保持其余关节在当前位: {base_pose}")

        if args.keyboard:
            _keyboard_control(hand, base_pose, open_pose, args)
            return

        for cycle in range(args.cycles):
            print(f"\n===== cycle {cycle + 1}/{args.cycles} =====")
            for joint in joints:
                test_pose = _pose_with_joint(base_pose, joint, close_position)
                print(f"\n[动作] {joint}:{JOINT_NAMES[joint]} -> {test_pose}")
                hand.move(test_pose)
                time.sleep(args.pause)
                _print_state("到位反馈", _read_state(hand))

                print(f"[动作] {joint}:{JOINT_NAMES[joint]} <- {base_pose}")
                hand.move(base_pose)
                time.sleep(args.pause)
                _print_state("回弹反馈", _read_state(hand))

        print(f"\n[动作] 测试结束，返回张开位: {open_pose}")
        hand.move(open_pose)
        time.sleep(args.pause)
        _print_state("最终反馈", _read_state(hand))
    finally:
        if hand is not None:
            hand.close()
            print("[INFO] O6 连接已关闭。")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断，正在退出。")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
