#!/usr/bin/env python3
"""Minimal L515 RGB worker for deployment ego-view inference.

This script intentionally has only numpy + pyrealsense2 runtime dependencies.
It writes RGB8 frames into parent-owned shared memory and emits JSON metadata
on stdout.  The parent process can therefore keep using the existing inference
environment even when pyrealsense2 is installed in a separate environment.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker

import numpy as np


def _set_global_time_if_supported(device, rs) -> None:
    for sensor in device.query_sensors():
        option = getattr(rs.option, "global_time_enabled", None)
        if option is None:
            continue
        try:
            if sensor.supports(option):
                sensor.set_option(option, 1.0)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--shm-name", required=True)
    parser.add_argument("--shm-slots", type=int, default=4)
    args = parser.parse_args()

    shm = None
    pipeline = None
    try:
        import pyrealsense2 as rs

        width = int(args.width)
        height = int(args.height)
        slots = max(2, int(args.shm_slots))
        frame_bytes = width * height * 3
        shm = SharedMemory(name=args.shm_name)
        # The parent owns the shared-memory lifetime.  Do not let this worker
        # attempt a second unlink at interpreter shutdown.
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(args.serial))
        config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.rgb8,
            int(args.fps),
        )
        resolved = config.resolve(rs.pipeline_wrapper(pipeline))
        _set_global_time_if_supported(resolved.get_device(), rs)
        profile = pipeline.start(config)
        _set_global_time_if_supported(profile.get_device(), rs)

        for _ in range(max(0, int(args.warmup_frames))):
            pipeline.wait_for_frames(int(args.timeout_ms))

        print(json.dumps({
            "type": "ready",
            "width": width,
            "height": height,
            "fps": int(args.fps),
        }), flush=True)

        slot = -1
        last_frame_id = None
        while True:
            frames = pipeline.wait_for_frames(int(args.timeout_ms))
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame_id = int(color_frame.get_frame_number())
            if frame_id == last_frame_id:
                continue
            slot = (slot + 1) % slots
            image = np.asanyarray(color_frame.get_data())
            if image.shape != (height, width, 3):
                raise RuntimeError(
                    f"unexpected L515 RGB shape {image.shape}, "
                    f"expected {(height, width, 3)}"
                )
            view = np.ndarray(
                (height, width, 3),
                dtype=np.uint8,
                buffer=shm.buf,
                offset=slot * frame_bytes,
            )
            view[...] = image
            last_frame_id = frame_id
            print(json.dumps({
                "type": "frame",
                "slot": int(slot),
                "frame_id": frame_id,
                # One camera only: host wall-clock time is sufficient for
                # timestamp alignment with the Franka/episode recorder.
                "timestamp": float(time.time()),
            }), flush=True)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(json.dumps({
            "type": "error",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }), flush=True)
        return 1
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
