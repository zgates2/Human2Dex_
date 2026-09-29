#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PKL 回放：从 collect_data.py 采集的 PKL 中读取 timestamp + o6_command，
驱动 Linker O6 真机灵巧手按原速度复现动作。

PKL 结构 (collect_data.py 写入):
    {
      "messages": [
        {
          "timestamp":   float,                       # time.time(), 秒
          "o6_command": np.ndarray(6,) uint8 / None, # 0~255
          ...
        },
        ...
      ]
    }

调用链：
    pkl → load → 按 (timestamp - timestamp[0]) 节拍 → O6RightHand.move(cmd)

运行：
    conda activate pico

    sudo ip link set can0 up type can bitrate 1000000

    # 基础回放
    python3 replay_pkl.py data/demo_20250513_170000.pkl

    # 调整速度上限、播放倍率
    python3 replay_pkl.py demo.pkl --speed 200 --rate 1.0

    # 离线 dry-run（不连接 CAN，仅打印）
    python3 replay_pkl.py demo.pkl --dry-run
"""

from __future__ import annotations

import argparse
import pickle
import signal
import sys
import time
from pathlib import Path

import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_O6_DIR = _REPO_ROOT / "o6_right_hand"
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _O6_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from controller import O6RightHand  # noqa: E402


# ── 工具：解析 pkl ────────────────────────────────────────────────────────────

def _load_messages(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"PKL 不存在: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "messages" not in data:
        raise ValueError(f"PKL 格式不符合预期，缺少 'messages' 键: {path}")
    msgs = data["messages"]
    if not isinstance(msgs, list) or len(msgs) == 0:
        raise ValueError(f"PKL 中 messages 为空: {path}")
    return msgs


def _extract_frames(messages: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    从 messages 中抽取有效帧 (o6_command 非空；兼容旧 handCommand)。

    Returns
    -------
    timestamps : (N,) float64，单位秒
    commands   : (N, 6) uint8
    """
    ts_list: list[float] = []
    cmd_list: list[np.ndarray] = []
    for m in messages:
        cmd = m.get("o6_command", m.get("handCommand"))
        ts = m.get("timestamp")
        if cmd is None or ts is None:
            continue
        cmd_arr = np.asarray(cmd, dtype=np.uint8).reshape(-1)
        if cmd_arr.shape != (6,):
            continue
        ts_list.append(float(ts))
        cmd_list.append(cmd_arr)

    if len(ts_list) == 0:
        raise ValueError("PKL 中无有效 o6_command 帧")

    timestamps = np.asarray(ts_list, dtype=np.float64)
    commands = np.stack(cmd_list, axis=0)  # (N, 6) uint8

    # 时间戳单调化（防止 PICO/系统时钟轻微回跳）
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            timestamps[i] = timestamps[i - 1]

    return timestamps, commands


# ── 主回放循环 ────────────────────────────────────────────────────────────────

def _replay_loop(
    timestamps: np.ndarray,
    commands: np.ndarray,
    send_fn,
    rate: float = 1.0,
    interrupted: dict | None = None,
    log_every: int = 30,
) -> None:
    """
    根据原录制时间戳节拍调用 send_fn(cmd_list)。

    Parameters
    ----------
    timestamps : (N,) 录制时刻 (秒)
    commands   : (N, 6) uint8 指令
    send_fn    : callable(list[int])，发送单帧指令
    rate       : 回放倍率，>1 加速，<1 慢放
    interrupted: 可选 {"flag": bool}，用于 Ctrl+C 中断
    log_every  : 每隔多少帧打印一次进度
    """
    rate = max(float(rate), 1e-3)
    n = len(timestamps)
    if interrupted is None:
        interrupted = {"flag": False}

    t0_data = float(timestamps[0])
    t0_wall = time.perf_counter()

    print(f"[Replay] 共 {n} 帧，原时长 {(timestamps[-1] - t0_data):.2f}s"
          f"，回放倍率 {rate:.2f}x")

    last_cmd_logged: list[int] | None = None

    for i in range(n):
        if interrupted["flag"]:
            print("\n[Replay] 用户中断")
            break

        # 目标墙钟时刻：按 rate 缩放
        target_wall = t0_wall + (float(timestamps[i]) - t0_data) / rate
        now = time.perf_counter()
        wait = target_wall - now
        if wait > 0:
            # 长等待分段休眠，让 Ctrl+C 能及时响应
            while wait > 0 and not interrupted["flag"]:
                step = min(wait, 0.05)
                time.sleep(step)
                wait = target_wall - time.perf_counter()

        cmd = [int(v) for v in commands[i].tolist()]
        send_fn(cmd)

        if log_every > 0 and (i % log_every == 0 or i == n - 1):
            elapsed_data = float(timestamps[i]) - t0_data
            print(f"[Replay] frame {i+1}/{n}  t={elapsed_data:6.2f}s  cmd={cmd}")
            last_cmd_logged = cmd

    if last_cmd_logged is None:
        print("[Replay] 未发送任何帧")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PKL → Linker O6 真机回放（仅使用 timestamp + o6_command）"
    )
    parser.add_argument(
        "pkl",
        type=Path,
        help="collect_data.py 产出的 PKL 文件路径",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="回放速率倍率，1.0=原速，2.0=两倍速，0.5=半速",
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
        "--no-home-before",
        action="store_true",
        help="回放前不先回到 home 位（默认会回 home 后等待 1s）",
    )
    parser.add_argument(
        "--no-home-after",
        action="store_true",
        help="回放结束后不回到 home 位",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="日志节流：每 N 帧打印一次（默认 30，设 0 关闭）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不连接 CAN，仅按节拍打印指令（验证 PKL 与节拍）",
    )
    args = parser.parse_args()

    # ── 加载 PKL ──────────────────────────────────────────────────────
    print(f"[Load] 读取 {args.pkl}")
    messages = _load_messages(args.pkl)
    timestamps, commands = _extract_frames(messages)
    print(
        f"[Load] 总 {len(messages)} 条 messages，"
        f"有效 o6_command 帧 {len(timestamps)} 条"
    )

    # ── 中断处理 ──────────────────────────────────────────────────────
    interrupted = {"flag": False}

    def _sigint(_sig, _frame):
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    # ── DRY RUN：不连 CAN，仅打印 ─────────────────────────────────────
    if args.dry_run:
        print("[DryRun] 不连接 CAN，仅按节拍打印指令")

        def _dry_send(cmd: list[int]):
            # 节拍由 _replay_loop 控制，此处不打印每帧避免刷屏
            return

        _replay_loop(
            timestamps, commands,
            send_fn=_dry_send,
            rate=args.rate,
            interrupted=interrupted,
            log_every=max(args.log_every, 1),
        )
        return

    # ── 连接灵巧手 ────────────────────────────────────────────────────
    print(
        f"[Init] 连接 Linker O6 (channel={args.can_channel},"
        f" bitrate={args.bitrate})..."
    )
    with O6RightHand(can_channel=args.can_channel, bitrate=args.bitrate) as hand:
        hand.set_speed(args.speed)
        if args.torque is not None:
            hand.set_torque(args.torque)

        if not args.no_home_before:
            print("[Init] 回到张开位 ...")
            hand.home()
            time.sleep(1.0)

        print("[Run] 开始回放，Ctrl+C 中断")
        try:
            _replay_loop(
                timestamps, commands,
                send_fn=hand.move,
                rate=args.rate,
                interrupted=interrupted,
                log_every=args.log_every,
            )
        finally:
            if not args.no_home_after:
                print("\n[Exit] 回到张开位 ...")
                try:
                    hand.home()
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[Warn] home() 失败: {e}")

    print("[Done] 回放结束")


if __name__ == "__main__":
    main()
