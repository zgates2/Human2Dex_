#!/usr/bin/env python3
"""Unified launcher for Franka inference and the SSH-forwarded web dashboard."""

from __future__ import annotations

import sys
import time

import numpy as np

import eval_real_franka_o6 as inference
from inference_web_dashboard import InferenceWebDashboard


def run_dashboard_self_test(dashboard: InferenceWebDashboard) -> None:
    dashboard.mark_ready("网页自检模式，等待开始")
    started = time.time()
    step = 0
    while not dashboard.stop_pending():
        if dashboard.consume_reset_request():
            dashboard.mark_resetting("网页自检：模拟复位")
            time.sleep(0.25)
            dashboard.mark_ready("网页自检：复位完成")
        phase = time.time() - started
        x = np.linspace(0, 1, 224, dtype=np.float32)
        y = np.linspace(0, 1, 224, dtype=np.float32)[:, None]
        frame = np.stack([
            np.broadcast_to(x, (224, 224)),
            np.broadcast_to(y, (224, 224)),
            np.full((224, 224), (np.sin(phase) + 1.0) / 2.0, dtype=np.float32),
        ], axis=-1)
        raw = np.stack([
            np.sin(phase + np.arange(15) * 0.15 + horizon * 0.08)
            for horizon in range(8)
        ])
        arm = np.stack([
            np.array([0.45, -0.1, 0.35, 0.0, 1.57, 0.0, 0.0])
            + horizon * 0.001
            for horizon in range(8)
        ])
        hand = np.stack([
            np.clip(128 + 80 * np.sin(phase + np.arange(6) * 0.2), 0, 255)
            for _ in range(8)
        ])
        running = dashboard.can_execute()
        dashboard.update(
            frames={"camera0_rgb": frame},
            raw_model_chunk=raw,
            processed_arm_chunk=arm,
            submitted_arm_chunk=arm if running else np.empty((0, 7)),
            processed_hand_chunk=hand,
            submitted_hand_chunk=hand if running else np.empty((0, 6)),
            observed_arm=arm[-1, :6] + 0.002 * np.sin(phase),
            observed_hand=hand[-1] + 1.5 * np.sin(phase),
            metadata={"mode": "web_self_test", "step": step},
        )
        if running:
            step += 1
        time.sleep(0.1)


def main() -> int:
    parser = inference.build_arg_parser(
        description="Franka + O6 推理与本地 SSH 转发网页面板"
    )
    parser.add_argument("--web-host", default="127.0.0.1",
                        choices=["127.0.0.1", "localhost", "::1"],
                        help="Dashboard bind address; keep loopback-only for SSH forwarding")
    parser.add_argument("--web-port", type=int, default=8765,
                        help="Remote dashboard port (default: 8765)")
    parser.add_argument("--web-heartbeat-timeout", type=float, default=5.0,
                        help="Auto-pause after this many seconds without browser heartbeat")
    parser.add_argument("--web-history-seconds", type=float, default=10.0,
                        help="Rolling action/state chart duration")
    parser.add_argument("--web-self-test", action="store_true",
                        help="Run the dashboard with synthetic data and no hardware/model")
    args = parser.parse_args()

    dashboard = InferenceWebDashboard(
        host=args.web_host,
        port=args.web_port,
        heartbeat_timeout=args.web_heartbeat_timeout,
        history_seconds=args.web_history_seconds,
    )
    dashboard.start()
    print("\n[WEB] 在本地电脑另开一个终端执行：", flush=True)
    print(
        f"      ssh -N -L {args.web_port}:127.0.0.1:{args.web_port} zjc",
        flush=True,
    )
    print("[WEB] 然后在本地浏览器打开：", flush=True)
    print(f"      {dashboard.url}\n", flush=True)

    try:
        if args.web_self_test:
            run_dashboard_self_test(dashboard)
        elif args.timing_self_test:
            inference.run_timing_self_test()
        else:
            cfg = inference.config_from_args(args)
            inference.run_inference(cfg, dashboard=dashboard)
        dashboard.mark_stopped("推理已安全结束")
        return 0
    except KeyboardInterrupt:
        dashboard.request_stop()
        dashboard.mark_stopped("用户通过 Ctrl+C 结束")
        return 130
    except Exception as exc:
        dashboard.mark_error(f"推理异常：{type(exc).__name__}: {exc}")
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("[WEB] 页面保留错误状态；点击结束或按 Ctrl+C 关闭。", flush=True)
        try:
            while not dashboard.stop_pending():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        dashboard.mark_stopped("异常后已关闭")
        return 1
    finally:
        dashboard.close()


if __name__ == "__main__":
    raise SystemExit(main())
