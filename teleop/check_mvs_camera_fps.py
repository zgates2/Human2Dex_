#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVS 相机链路帧率/抖动诊断。

用途
----
只启动 MVS 相机和 LatestMVSCache，不加载 wrist 模型、不打开 OpenCV 窗口，
统计 wait_next/latest 拿到的帧间隔、等待耗时、frame_id 跳变和缓存状态。

示例
----
  conda run -n pico python teleop/check_mvs_camera_fps.py \
      --camera-hz 30 \
      --seconds 10

如果要测试相机后台最大吐帧，不等待新帧：

  conda run -n pico python teleop/check_mvs_camera_fps.py \
      --camera-hz 30 \
      --latest-only \
      --no-copy-image
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = REPO_ROOT / "teleop"
for path in (REPO_ROOT, TELEOP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collect_config import build_mvs_config, load_pico_mvs_config  # noqa: E402
from sensor_runtime import MVSSensor  # noqa: E402


DEFAULT_COLLECT_CONFIG = REPO_ROOT / "teleop" / "collect_data.yaml"


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def integer_delta_counts(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def make_sensor(args: argparse.Namespace) -> MVSSensor:
    cfg = load_pico_mvs_config(args.collect_config.expanduser(), task_name=args.task_name)
    if args.mvs_serial:
        cfg.mvs_serial = args.mvs_serial
    if args.camera_hz is not None:
        cfg.hz = float(args.camera_hz)
    if args.mvs_resolution is not None:
        cfg.mvs_resolution = tuple(args.mvs_resolution)
    if args.mvs_input_resolution is not None:
        cfg.mvs_input_resolution = tuple(args.mvs_input_resolution)
    if args.exposure_us is not None:
        cfg.mvs_exposure_time_us = float(args.exposure_us)
    if args.sensor_crop_roi is not None:
        cfg.mvs_use_sensor_crop_roi = bool(args.sensor_crop_roi)

    print(
        "[Init] MVS "
        f"serial={cfg.mvs_serial or 'auto'} fps={cfg.hz} "
        f"output={cfg.mvs_resolution} input={cfg.mvs_input_resolution} "
        f"sensor_crop_roi={cfg.mvs_use_sensor_crop_roi} exposure_us={cfg.mvs_exposure_time_us}"
    )
    return MVSSensor(build_mvs_config(cfg))


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 MVS 相机帧率和帧间隔抖动")
    parser.add_argument("--collect-config", type=Path, default=DEFAULT_COLLECT_CONFIG)
    parser.add_argument("--task-name", type=str, default="mvs_fps_check")
    parser.add_argument("--mvs-serial", type=str, default=None)
    parser.add_argument("--camera-hz", type=float, default=None)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--latest-only", action="store_true", help="不等待新帧，直接轮询 latest")
    parser.add_argument("--no-copy-image", action="store_true", help="取帧时不复制 image buffer")
    parser.add_argument("--mvs-resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--mvs-input-resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument(
        "--sensor-crop-roi",
        type=int,
        choices=(0, 1),
        default=None,
        help="覆盖 mvs_use_sensor_crop_roi，1 开启，0 关闭",
    )
    args = parser.parse_args()

    sensor = make_sensor(args)
    copy_image = not args.no_copy_image
    last_frame_id: int | None = None
    last_capture_ns: int | None = None
    last_device_ticks: int | None = None
    last_sdk_host_ms: int | None = None
    last_host_t: float | None = None
    wait_ms: list[float] = []
    host_dt_ms: list[float] = []
    capture_dt_ms: list[float] = []
    device_tick_delta: list[int] = []
    sdk_host_dt_ms: list[int] = []
    frame_deltas: list[int] = []
    repeated = 0
    skipped = 0
    frames = 0
    start_t = time.perf_counter()

    try:
        sensor.start()
        print("[Init] metadata:")
        print(json.dumps(sensor.metadata(), indent=2, ensure_ascii=False))
        if args.warmup_seconds > 0:
            time.sleep(float(args.warmup_seconds))
        start_t = time.perf_counter()
        deadline = start_t + max(0.1, float(args.seconds))
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            if args.latest_only:
                frame = sensor.latest(copy_image=copy_image)
            else:
                try:
                    frame = sensor.wait_next(
                        last_frame_id=last_frame_id,
                        timeout_s=max(0.2, 2.0 / max(float(args.camera_hz or sensor.config.fps), 1.0)),
                        copy_image=copy_image,
                    )
                except TimeoutError:
                    frame = sensor.latest(copy_image=copy_image)
            t1 = time.perf_counter()
            wait_ms.append((t1 - t0) * 1000.0)

            if last_host_t is not None:
                host_dt_ms.append((t1 - last_host_t) * 1000.0)
            last_host_t = t1

            fid = None if frame.frame_id is None else int(frame.frame_id)
            if fid is not None:
                if last_frame_id is not None:
                    delta = fid - int(last_frame_id)
                    frame_deltas.append(delta)
                    if delta == 0:
                        repeated += 1
                    elif delta > 1:
                        skipped += delta - 1
                last_frame_id = fid

            if frame.capture_ns is not None:
                capture_ns = int(frame.capture_ns)
                if last_capture_ns is not None:
                    capture_dt_ms.append((capture_ns - last_capture_ns) / 1_000_000.0)
                last_capture_ns = capture_ns
            if frame.device_timestamp_ticks is not None:
                device_ticks = int(frame.device_timestamp_ticks)
                if device_ticks > 0 and last_device_ticks is not None:
                    device_tick_delta.append(device_ticks - last_device_ticks)
                if device_ticks > 0:
                    last_device_ticks = device_ticks
            if frame.host_timestamp_ms is not None:
                sdk_host_ms = int(frame.host_timestamp_ms)
                if sdk_host_ms > 0 and last_sdk_host_ms is not None:
                    sdk_host_dt_ms.append(sdk_host_ms - last_sdk_host_ms)
                if sdk_host_ms > 0:
                    last_sdk_host_ms = sdk_host_ms

            frames += 1
            if args.print_every > 0 and frames % int(args.print_every) == 0:
                elapsed = max(time.perf_counter() - start_t, 1e-6)
                print(
                    f"[Frame] n={frames} fps={frames / elapsed:.2f} "
                    f"fid={fid} wait_ms={wait_ms[-1]:.2f} "
                    f"host_dt_ms={(host_dt_ms[-1] if host_dt_ms else -1.0):.2f} "
                    f"capture_dt_ms={(capture_dt_ms[-1] if capture_dt_ms else -1.0):.2f} "
                    f"sdk_host_dt_ms={(sdk_host_dt_ms[-1] if sdk_host_dt_ms else None)} "
                    f"dev_tick_delta={(device_tick_delta[-1] if device_tick_delta else None)} "
                    f"frame_delta={(frame_deltas[-1] if frame_deltas else None)} "
                    f"gap={frame.frame_gap}"
                )
    finally:
        sensor.stop()

    elapsed = max(time.perf_counter() - start_t, 1e-6)
    summary = {
        "frames_returned": frames,
        "elapsed_s": elapsed,
        "return_fps": frames / elapsed,
        "repeated_returns": repeated,
        "skipped_frames_from_frame_id": skipped,
        "frame_delta_counts": integer_delta_counts(frame_deltas),
        "wait_ms": percentiles(wait_ms),
        "host_dt_ms": percentiles(host_dt_ms),
        "capture_dt_ms": percentiles(capture_dt_ms),
        "sdk_host_dt_ms": percentiles([float(v) for v in sdk_host_dt_ms]),
        "sdk_host_dt_counts": integer_delta_counts(sdk_host_dt_ms),
        "device_tick_delta": percentiles([float(v) for v in device_tick_delta]),
        "device_tick_delta_counts_top20": dict(
            sorted(
                integer_delta_counts(device_tick_delta).items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ),
        "sensor_stats": sensor.stats(),
    }
    print("[Summary]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
