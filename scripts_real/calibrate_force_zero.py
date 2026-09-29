"""Smoke test + statistics check for the 6D force/torque sensor.

Reports:
- mean raw wrench (6,) over a quiet period
- noise std (6,) = sensor noise floor for sanity

It does NOT produce runtime calibration parameters. Online inference uses
gravity compensation from `cali_info`/`cali_bias` or `cali_params_path`, so do
not paste the mean wrench into eval_robots_config.yaml. Call this with the
end-effector hanging free, no contact, no gripper motion.

Run:
    python scripts_real/calibrate_force_zero.py
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

from umi.real_world.force_sensor_controller import ForceSensorController


DEFAULT_CONFIG_REL = "example/eval_robots_config.yaml"


def _load_first_force_sensor() -> dict:
    cfg_path = os.environ.get(
        "UMI_ROBOT_CONFIG", str(ROOT_DIR / DEFAULT_CONFIG_REL),
    )
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    for fs in cfg.get('force_sensors', []):
        if fs is not None and 'port' in fs:
            return fs
    raise RuntimeError(f"no usable force sensor entry in {cfg_path}")


def main():
    # -------- 标定参数(按需修改) --------
    WARMUP_S = 3.0     # 让 RS485 流量稳定 + 让用户把工具挂稳
    SAMPLE_S = 5.0     # 取样窗口
    EXPECTED_HZ = 500.0  # 期望帧率,用来判断是否取够

    fs_cfg = _load_first_force_sensor()
    print(f"[info] force sensor: port={fs_cfg['port']} baud={fs_cfg.get('baud', 1000000)}")

    with SharedMemoryManager() as shm:
        sensor = ForceSensorController(
            shm_manager=shm,
            port=fs_cfg['port'],
            baudrate=fs_cfg.get('baud', 1000000),
            device_id=fs_cfg.get('device_id', 1),
            frequency=fs_cfg.get('frequency', EXPECTED_HZ),
            receive_latency=0.0,
            zero_on_start=False,
            verbose=False,
        )
        sensor.start(wait=True)
        try:
            print(f"[info] warming up {WARMUP_S}s; keep tool still and free of contact")
            time.sleep(WARMUP_S)

            print(f"[info] sampling {SAMPLE_S}s")
            t_start = time.time()
            time.sleep(SAMPLE_S)
            t_end = time.time()
            data = sensor.get_all_state()
        finally:
            sensor.stop(wait=True)

    if 'wrench' not in data or data['wrench'].shape[0] == 0:
        print("[FAIL] no samples captured")
        sys.exit(1)

    cap_ts = np.asarray(data['force_capture_timestamp'])
    keep = (cap_ts >= t_start) & (cap_ts <= t_end)
    wrench = np.asarray(data['wrench'])[keep]
    cap_ts = cap_ts[keep]
    if wrench.shape[0] < 50:
        print(f"[FAIL] only {wrench.shape[0]} samples in window, need at least 50")
        sys.exit(1)

    actual_hz = wrench.shape[0] / SAMPLE_S
    mean = wrench.mean(axis=0)
    std = wrench.std(axis=0)

    print()
    print(f"samples in window            : {wrench.shape[0]}")
    print(f"effective sample rate        : {actual_hz:.1f} Hz (expected ~{EXPECTED_HZ:.0f})")
    print(f"mean  Fx Fy Fz Mx My Mz      : {np.array2string(mean, precision=4, suppress_small=True)}")
    print(f"std   Fx Fy Fz Mx My Mz      : {np.array2string(std,  precision=4, suppress_small=True)}")
    print()
    print("NOTE: this is a connectivity/noise check only. Runtime gravity")
    print("compensation must use cali_info/cali_bias or cali_params_path.")


if __name__ == "__main__":
    main()
