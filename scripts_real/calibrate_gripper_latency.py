"""Step-response calibration for LkMotor gripper action latency.

Method:
    Repeatedly command an immediate position step (close <-> open) and
    measure the delay between the schedule_waypoint() call and the moment a
    fresh gripper state reports motion onset.

That delay is `gripper_action_latency` -- the value you put into
example/eval_robots_config.yaml under each gripper. The script filters stale
state frames and waits for the gripper to settle before each step, so negative
latencies should not appear in a valid run.

This script reads gripper port/motor_id from eval_robots_config.yaml so it
runs with zero arguments after the yaml is populated.

Run:
    python scripts_real/calibrate_gripper_latency.py
"""
import os
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import List

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.real_world.lk_gripper_proxy import LkGripperProxy


DEFAULT_CONFIG_REL = "example/eval_robots_config.yaml"


def _load_first_lkmotor_gripper() -> dict:
    cfg_path = os.environ.get(
        "UMI_ROBOT_CONFIG", str(ROOT_DIR / DEFAULT_CONFIG_REL),
    )
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    for g in cfg.get('grippers', []):
        if g.get('gripper_type') == 'lkmotor':
            return g
    raise RuntimeError(f"no lkmotor gripper found in {cfg_path}")


def main():
    # -------- 标定参数(按需修改) --------
    POS_LOW_RAD = 0.0       # 关闭位置(rad)
    POS_HIGH_RAD = 2.0      # 张开位置(rad)
    SAMPLE_DT = 0.005       # 5ms 采样间隔
    N_TRIALS = 30           # 总试验次数(half closing + half opening)
    VEL_ONSET_RAD_S = 0.05  # 视为开始运动的速度阈值
    VEL_SETTLE_RAD_S = 0.03 # 视为静止的速度阈值
    SETTLE_STABLE_S = 0.20  # 连续静止这么久后才进入下一次试验
    SETTLE_TIMEOUT_S = 4.0
    ONSET_TIMEOUT_S = 1.0

    g_cfg = _load_first_lkmotor_gripper()
    proxy_cfg = {
        'mode': g_cfg.get('mode', 'single'),
        'port1': g_cfg['port1'],
        'motor1_id': g_cfg['motor1_id'],
    }
    for key in ['close_offset_rad', 'close_offset_active_below_rad']:
        if key in g_cfg:
            proxy_cfg[key] = g_cfg[key]
    if proxy_cfg['mode'] == 'dual':
        proxy_cfg['port2'] = g_cfg['port2']
        proxy_cfg['motor2_id'] = g_cfg['motor2_id']

    print(f"[info] config: {proxy_cfg}")
    print(f"[info] step {POS_LOW_RAD}rad <-> {POS_HIGH_RAD}rad, {N_TRIALS} trials")

    latencies_s: List[float] = []
    with SharedMemoryManager() as shm:
        gripper = LkGripperProxy(
            config=proxy_cfg,
            receive_latency=0.0,
            worker_loop_hz=120.0,
        )
        gripper.start(wait=True)
        try:
            time.sleep(1.0)
            cur_target = POS_LOW_RAD
            gripper.move_gripper(pos=cur_target)
            _wait_until_still(
                gripper=gripper,
                vel_threshold=VEL_SETTLE_RAD_S,
                stable_s=SETTLE_STABLE_S,
                timeout_s=SETTLE_TIMEOUT_S,
                sample_dt=SAMPLE_DT,
            )

            for trial in range(N_TRIALS):
                cur_target = POS_HIGH_RAD if cur_target == POS_LOW_RAD else POS_LOW_RAD
                settled = _wait_until_still(
                    gripper=gripper,
                    vel_threshold=VEL_SETTLE_RAD_S,
                    stable_s=SETTLE_STABLE_S,
                    timeout_s=SETTLE_TIMEOUT_S,
                    sample_dt=SAMPLE_DT,
                )
                start_pos = settled['pos']
                direction = np.sign(cur_target - start_pos)
                if direction == 0:
                    direction = 1.0 if cur_target == POS_HIGH_RAD else -1.0

                schedule_call_t = time.time()
                # Use an immediate waypoint. A future target_time creates an
                # interpolation ramp and measures planned motion, not command latency.
                gripper.schedule_waypoint(pos=cur_target, target_time=schedule_call_t)

                onset = _wait_for_motion_onset(
                    gripper=gripper,
                    min_timestamp=schedule_call_t,
                    direction=direction,
                    vel_threshold=VEL_ONSET_RAD_S,
                    timeout_s=ONSET_TIMEOUT_S,
                    sample_dt=SAMPLE_DT,
                )
                if onset is None:
                    print(f"[trial {trial}] no motion onset detected within window")
                    continue

                latency = onset['t'] - schedule_call_t
                if latency < -1e-4:
                    print(f"[trial {trial}] stale timestamp detected, ignoring "
                          f"latency={latency*1000:.1f}ms")
                    continue
                latencies_s.append(latency)
                print(f"[trial {trial}] target={cur_target:.2f}rad "
                      f"start={start_pos:.3f}rad "
                      f"onset_vel={onset['vel']:.3f}rad/s "
                      f"action_latency={latency*1000:.1f}ms")
        finally:
            gripper.stop(wait=True)

    if not latencies_s:
        print("[FAIL] no valid trials, can't compute statistics")
        sys.exit(1)

    arr = np.asarray(latencies_s)
    median = float(np.median(arr))
    print()
    print(f"trials succeeded             : {arr.size}/{N_TRIALS}")
    print(f"action_latency mean/median   : {arr.mean()*1000:.1f}ms / {median*1000:.1f}ms")
    print(f"action_latency p95           : {np.quantile(arr, 0.95)*1000:.1f}ms")
    print(f"action_latency std           : {arr.std()*1000:.1f}ms")
    print()
    print(f"suggested gripper_action_latency : {max(0.0, median):.4f} s")
    print(f"  fill into example/eval_robots_config.yaml under grippers[*].gripper_action_latency")


def _latest_sample(gripper):
    sample = gripper.get_state(k=1)
    ts = sample.get('gripper_measure_timestamp')
    if ts is None or ts.size == 0:
        return None
    return {
        't': float(ts[-1]),
        'pos': float(sample['gripper_position'][-1]),
        'vel': float(sample['gripper_velocity'][-1]),
        'force': float(sample['gripper_force'][-1]),
    }


def _wait_until_still(gripper, vel_threshold: float, stable_s: float,
                      timeout_s: float, sample_dt: float):
    deadline = time.time() + timeout_s
    stable_since = None
    last_sample = None

    while time.time() < deadline:
        sample = _latest_sample(gripper)
        if sample is not None:
            last_sample = sample
            if abs(sample['vel']) <= vel_threshold:
                if stable_since is None:
                    stable_since = sample['t']
                if sample['t'] - stable_since >= stable_s:
                    return sample
            else:
                stable_since = None
        time.sleep(sample_dt)

    if last_sample is None:
        raise RuntimeError("no gripper state while waiting for settle")
    raise RuntimeError(
        "gripper did not settle: "
        f"last_pos={last_sample['pos']:.3f}rad "
        f"last_vel={last_sample['vel']:.3f}rad/s"
    )


def _wait_for_motion_onset(gripper, min_timestamp: float, direction: float,
                           vel_threshold: float, timeout_s: float,
                           sample_dt: float):
    deadline = time.time() + timeout_s
    last_seen_t = -np.inf

    while time.time() < deadline:
        sample = _latest_sample(gripper)
        if sample is None:
            time.sleep(sample_dt)
            continue
        if sample['t'] < min_timestamp or sample['t'] <= last_seen_t:
            time.sleep(sample_dt)
            continue

        last_seen_t = sample['t']
        if direction * sample['vel'] > vel_threshold:
            return sample
        time.sleep(sample_dt)

    return None


if __name__ == "__main__":
    main()
