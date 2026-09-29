"""Smoke-test MVS camera timestamps on the deployment host.

What it measures (no external time reference required):
- Device-to-host clock calibration residuals (how stable the SDK timestamp is).
- Frame interval consistency (1 / capture_fps should match observed inter-frame dt).

The runtime uses the MVS hardware/device timestamp path for frame timing, so
`cameras.obs_latency` should stay `0.0` for the MVS backend. This script is a
smoke/statistics check, not a latency calibration step.

Run:
    python scripts_real/calibrate_mvs_latency.py

Edit the constants at the top of `main()` if your config differs.
"""
import os
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.real_world.mvs_camera import MvsCamera, is_mvs_cpp_available, list_mvs_serials


def _sec_per_frame_stats(ts_arr: np.ndarray, expected_dt: float) -> dict:
    diffs = np.diff(ts_arr)
    return {
        'n_samples': int(diffs.size),
        'mean_ms': float(diffs.mean() * 1000),
        'std_ms': float(diffs.std() * 1000),
        'min_ms': float(diffs.min() * 1000),
        'max_ms': float(diffs.max() * 1000),
        'expected_ms': float(expected_dt * 1000),
    }


def _print_interval_stats(label: str, ts_arr: np.ndarray, expected_dt: float):
    stats = _sec_per_frame_stats(ts_arr, expected_dt)
    diffs = np.diff(ts_arr)
    late = diffs > expected_dt * 1.5
    print(f"{label:<31}: mean={stats['mean_ms']:.3f}ms "
          f"std={stats['std_ms']:.3f}ms "
          f"min={stats['min_ms']:.3f}ms max={stats['max_ms']:.3f}ms "
          f"(expected {stats['expected_ms']:.3f}ms)")
    print(f"{'  intervals >1.5x expected':<31}: "
          f"{int(late.sum())}/{diffs.size}")


def _print_optional_gap_stats(data: dict, keep: np.ndarray):
    if 'camera_frame_id' in data:
        frame_id = np.asarray(data['camera_frame_id'])[keep].astype(np.int64)
        if frame_id.size >= 2:
            diffs = np.diff(frame_id)
            dropped = diffs > 1
            backward = diffs < 0
            print(f"{'frame id range':<31}: first={int(frame_id[0])} "
                  f"last={int(frame_id[-1])}")
            print(f"{'frame id delta':<31}: min={int(diffs.min())} "
                  f"max={int(diffs.max())} dropped_gaps={int(dropped.sum())} "
                  f"backward_jumps={int(backward.sum())}")
            if backward.any():
                jump_idx = int(np.flatnonzero(backward)[0])
                lo = max(0, jump_idx - 3)
                hi = min(frame_id.size, jump_idx + 5)
                print(f"{'  first backward window':<31}: "
                      f"{frame_id[lo:hi].tolist()}")

    if 'camera_backend_total_ns' in data:
        total_ms = np.asarray(data['camera_backend_total_ns'])[keep].astype(np.float64) / 1e6
        if total_ms.size:
            print(f"{'backend total time':<31}: mean={total_ms.mean():.3f}ms "
                  f"p95={np.percentile(total_ms, 95):.3f}ms "
                  f"max={total_ms.max():.3f}ms")


def main():
    # -------- 配置(按需修改) --------
    cameras_cfg = _load_mvs_camera_config()
    CAPTURE_FPS = int(cameras_cfg.get('capture_fps', 60))
    RESOLUTION = tuple(cameras_cfg.get('resolution', [480, 480]))
    EXPOSURE_US = float(cameras_cfg.get('exposure_time_us', 15000.0))
    GAIN_AUTO = cameras_cfg.get('gain_auto', 'continuous')
    GAIN_DB = cameras_cfg.get('gain_db', None)
    if GAIN_DB is not None and not isinstance(GAIN_DB, (list, tuple)):
        GAIN_DB = float(GAIN_DB)
    DURATION_S = 5.0
    WARMUP_S = 1.0
    serials_from_cfg = list(cameras_cfg.get('serials') or [])
    MVS_SERIAL = serials_from_cfg[0] if serials_from_cfg else ""
    if isinstance(GAIN_DB, (list, tuple)):
        GAIN_DB = None if GAIN_DB[0] is None else float(GAIN_DB[0])

    if not is_mvs_cpp_available():
        print("[FAIL] MVS C++ backend not built; build first via umi/real_world/_mvs_cpp/build_backend.sh")
        sys.exit(1)

    serial = MVS_SERIAL.strip()
    if not serial:
        serials = list_mvs_serials()
        if not serials:
            print("[FAIL] no MVS device detected")
            sys.exit(1)
        serial = serials[0]
        print(f"[info] auto-picked serial={serial} from {serials}")

    print(f"[info] opening serial={serial} {RESOLUTION} @ {CAPTURE_FPS}Hz, "
          f"exposure {EXPOSURE_US:.0f}us")

    with SharedMemoryManager() as shm:
        cam = MvsCamera(
            shm_manager=shm,
            mvs_serial=serial,
            resolution=RESOLUTION,
            capture_fps=CAPTURE_FPS,
            put_downsample=False,
            get_max_k=int(CAPTURE_FPS * (DURATION_S + WARMUP_S + 2)),
            receive_latency=0.0,
            exposure_time_us=EXPOSURE_US,
            gain_auto=GAIN_AUTO,
            gain_db=GAIN_DB,
            verbose=False,
        )
        cam.start(wait=True)
        try:
            print(f"[info] warming up {WARMUP_S:.1f}s...")
            time.sleep(WARMUP_S)
            t_start = time.time()
            time.sleep(DURATION_S)
            ring_count = int(cam.ring_buffer.count)
            n_frames = min(
                ring_count,
                cam.ring_buffer.get_max_k,
                int(CAPTURE_FPS * (DURATION_S + WARMUP_S + 2)),
            )
            data = cam.get(k=n_frames)
        finally:
            cam.stop(wait=True)

    cap_ts_all = np.asarray(data['camera_capture_timestamp'])
    recv_ts_all = np.asarray(data['camera_receive_timestamp'])
    cap_ts = cap_ts_all
    recv_ts = recv_ts_all
    keep = cap_ts >= t_start
    cap_ts = cap_ts[keep]
    recv_ts = recv_ts[keep]

    if cap_ts.size < 10:
        print(f"[FAIL] only {cap_ts.size} frames captured in {DURATION_S}s")
        sys.exit(1)

    expected_dt = 1.0 / CAPTURE_FPS
    drv_lag = (recv_ts - cap_ts) * 1000.0

    print()
    print(f"ring buffer frames read      : {cap_ts_all.size}")
    print(f"frames analyzed              : {cap_ts.size}")
    _print_interval_stats("capture timestamp interval", cap_ts, expected_dt)
    _print_interval_stats("receive timestamp interval", recv_ts, expected_dt)
    _print_optional_gap_stats(data, keep)
    print(f"{'receive-capture delta':<31}: mean={drv_lag.mean():.3f}ms "
          f"std={drv_lag.std():.3f}ms "
          f"max={drv_lag.max():.3f}ms")
    print()
    print("suggested cameras.obs_latency: 0.000000 s")
    print("  MVS path uses hardware/device timestamps; do not subtract receive latency.")
    print(f"  fill this into example/eval_robots_config.yaml under cameras.obs_latency")


def _load_mvs_camera_config() -> dict:
    cfg_path = os.environ.get(
        "UMI_ROBOT_CONFIG", str(ROOT_DIR / "example/eval_robots_config.yaml"),
    )
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cameras_cfg = cfg.get('cameras') or {}
    if cameras_cfg.get('backend') != 'mvs':
        print(f"[warn] cameras.backend is {cameras_cfg.get('backend')!r}; "
              "using MVS smoke-test defaults")
    return cameras_cfg


if __name__ == "__main__":
    main()
