"""Step-response calibration for Franka robot action latency.

This script measures the delay between a small immediate Cartesian waypoint
command and the first observed TCP motion onset. It is intended for the
Franka/ZeroRPC path used by eval_real.py, and reads robot connection settings
from example/eval_robots_config.yaml by default.

Run on the robot machine:
    python scripts_real/calibrate_robot_latency.py

The result is a candidate for robots[*].robot_action_latency. Treat it as a
starting point; too large a value can make the first scheduled waypoint stale
in eval_real.py, so compare it against real eval timing logs before committing
the config.
"""
import os
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Optional

import click
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.real_world.franka_interpolation_controller import FrankaInterpolationController


DEFAULT_CONFIG_REL = "example/eval_robots_config.yaml"
AXIS_TO_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _load_franka_config(config_path: Path, robot_idx: int) -> dict:
    with config_path.open('r') as f:
        cfg = yaml.safe_load(f)
    robots = cfg.get('robots', [])
    if robot_idx >= len(robots):
        raise RuntimeError(f"robot_idx={robot_idx} out of range for {config_path}")
    rc = robots[robot_idx]
    if not str(rc.get('robot_type', '')).startswith('franka'):
        raise RuntimeError(
            f"robots[{robot_idx}] is not franka: {rc.get('robot_type')!r}"
        )
    return rc


def _make_controller(shm_manager, rc: dict, init_joints: bool):
    joints_init = rc.get('joints_init') if init_joints else None
    return FrankaInterpolationController(
        shm_manager=shm_manager,
        robot_ip=rc['robot_ip'],
        robot_port=rc.get('robot_port', 4242),
        frequency=rc.get('frequency', 200),
        Kx_scale=rc.get('Kx_scale', 1.0),
        Kxd_scale=rc.get('Kxd_scale', np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0])),
        launch_timeout=rc.get('launch_timeout', 20),
        pose_api=rc.get('pose_api', 'rpy'),
        rpc_timeout=rc.get('rpc_timeout', 5.0),
        read_only=False,
        start_impedance_on_start=rc.get('start_impedance_on_start', False),
        impedance_start_delay=rc.get('impedance_start_delay', 0.5),
        joints_init=joints_init,
        joints_init_duration=4,
        verbose=rc.get('verbose', False),
        max_commands_per_cycle=rc.get('max_commands_per_cycle', 1),
        receive_latency=rc.get('robot_obs_latency', 0.0),
    )


def _latest_sample(controller, axis_idx: int) -> Optional[dict]:
    state = controller.get_state(k=1)
    ts = state.get('robot_timestamp')
    pose = state.get('ActualTCPPose')
    if ts is None or pose is None or len(ts) == 0:
        return None
    pose = np.asarray(pose[-1], dtype=np.float64)
    return {
        't': float(ts[-1]),
        'pose': pose,
        'axis_pos': float(pose[axis_idx]),
    }


def _axis_velocity(controller, axis_idx: int, k: int = 8) -> float:
    state = controller.get_state(k=k)
    ts = np.asarray(state.get('robot_timestamp', []), dtype=np.float64)
    poses = np.asarray(state.get('ActualTCPPose', []), dtype=np.float64)
    if ts.size < 3 or poses.shape[0] < 3:
        return 0.0
    x = poses[:, axis_idx]
    valid = np.isfinite(ts) & np.isfinite(x)
    ts = ts[valid]
    x = x[valid]
    if len(ts) < 3 or ts[-1] <= ts[0]:
        return 0.0
    # Linear fit over the latest samples is less noisy than pairwise diff.
    return float(np.polyfit(ts - ts[-1], x, deg=1)[0])


def _wait_until_still(
        controller,
        axis_idx: int,
        target_pose: np.ndarray,
        pos_tolerance: float,
        vel_threshold: float,
        stable_s: float,
        timeout_s: float,
        sample_dt: float) -> dict:
    deadline = time.time() + timeout_s
    stable_since = None
    last_sample = None
    target_axis = float(target_pose[axis_idx])

    while time.time() < deadline:
        sample = _latest_sample(controller, axis_idx)
        if sample is not None:
            vel = _axis_velocity(controller, axis_idx)
            sample['vel'] = vel
            last_sample = sample
            pos_err = abs(sample['axis_pos'] - target_axis)
            pos_ok = True if pos_tolerance <= 0 else pos_err <= pos_tolerance
            if pos_ok and abs(vel) <= vel_threshold:
                if stable_since is None:
                    stable_since = sample['t']
                if sample['t'] - stable_since >= stable_s:
                    sample['pos_err'] = pos_err
                    return sample
            else:
                stable_since = None
        time.sleep(sample_dt)

    if last_sample is None:
        raise RuntimeError("no robot state while waiting for settle")
    raise RuntimeError(
        "robot did not settle: "
        f"target_axis={target_axis:.5f} "
        f"last_axis={last_sample['axis_pos']:.5f} "
        f"last_vel={last_sample.get('vel', 0.0):.5f}"
    )


def _wait_for_motion_onset(
        controller,
        axis_idx: int,
        min_timestamp: float,
        start_axis_pos: float,
        direction: float,
        vel_threshold: float,
        displacement_threshold: float,
        timeout_s: float,
        sample_dt: float) -> Optional[dict]:
    deadline = time.time() + timeout_s
    last_seen_t = -np.inf

    while time.time() < deadline:
        sample = _latest_sample(controller, axis_idx)
        if sample is None:
            time.sleep(sample_dt)
            continue
        if sample['t'] < min_timestamp or sample['t'] <= last_seen_t:
            time.sleep(sample_dt)
            continue

        last_seen_t = sample['t']
        vel = _axis_velocity(controller, axis_idx)
        displacement = direction * (sample['axis_pos'] - start_axis_pos)
        sample['vel'] = vel
        sample['displacement'] = displacement
        if (
            direction * vel >= vel_threshold
            or displacement >= displacement_threshold
        ):
            return sample
        time.sleep(sample_dt)

    return None


@click.command()
@click.option(
    '-c', '--config',
    default=None,
    show_default=DEFAULT_CONFIG_REL,
    help='Path to eval_robots_config.yaml.',
)
@click.option('--robot-idx', default=0, type=int, show_default=True)
@click.option('--axis', type=click.Choice(['x', 'y', 'z']), default='z', show_default=True)
@click.option('--amplitude-m', default=0.005, type=float, show_default=True,
              help='Step amplitude in meters. Positive direction only, then return to start.')
@click.option('--step-duration-s', default=0.15, type=float, show_default=True,
              help='Waypoint arrival time after each command call; onset is still measured from the call.')
@click.option('--trials', default=20, type=int, show_default=True)
@click.option('--sample-dt', default=0.005, type=float, show_default=True)
@click.option('--vel-onset-m-s', default=0.003, type=float, show_default=True)
@click.option('--disp-onset-m', default=0.00025, type=float, show_default=True)
@click.option('--vel-settle-m-s', default=0.002, type=float, show_default=True)
@click.option('--pos-settle-m', default=0.0, type=float, show_default=True,
              help='Settle position tolerance in meters; 0 disables position check.')
@click.option('--settle-stable-s', default=0.20, type=float, show_default=True)
@click.option('--settle-timeout-s', default=4.0, type=float, show_default=True)
@click.option('--onset-timeout-s', default=1.0, type=float, show_default=True)
@click.option('--init-joints', is_flag=True, default=False,
              help='Move to robots[robot_idx].joints_init before calibration.')
def main(
        config,
        robot_idx,
        axis,
        amplitude_m,
        step_duration_s,
        trials,
        sample_dt,
        vel_onset_m_s,
        disp_onset_m,
        vel_settle_m_s,
        pos_settle_m,
        settle_stable_s,
        settle_timeout_s,
        onset_timeout_s,
        init_joints):
    if config is None:
        config = os.environ.get('UMI_ROBOT_CONFIG', str(ROOT_DIR / DEFAULT_CONFIG_REL))
    config_path = Path(config).expanduser()
    rc = _load_franka_config(config_path, robot_idx)
    axis_idx = AXIS_TO_INDEX[axis]
    amplitude_m = abs(float(amplitude_m))
    if amplitude_m <= 0:
        raise ValueError('--amplitude-m must be positive')
    step_duration_s = float(step_duration_s)
    if step_duration_s <= 0:
        raise ValueError('--step-duration-s must be positive')

    print(f"[info] config: {config_path}")
    print(
        "[info] robot: "
        f"type={rc.get('robot_type')} ip={rc['robot_ip']} port={rc.get('robot_port', 4242)} "
        f"axis={axis} amplitude={amplitude_m:.4f}m "
        f"step_duration={step_duration_s:.3f}s trials={trials}"
    )
    print("[info] warmup starts impedance with a hold-current-pose waypoint; trials exclude startup delay.")
    print("[info] each trial commands a relative step from the current settled TCP pose.")

    latencies_s = []
    with SharedMemoryManager() as shm_manager:
        with _make_controller(shm_manager, rc, init_joints=init_joints) as controller:
            time.sleep(0.5)
            center_pose = controller.get_state()['ActualTCPPose'].copy()

            # Warm up lazy impedance without moving.
            warmup_t = time.time() + 0.2
            controller.schedule_waypoint(center_pose, warmup_t)
            time.sleep(1.0)
            center_pose = controller.get_state()['ActualTCPPose'].copy()

            hold_pose = center_pose.copy()
            controller.schedule_waypoint(hold_pose, time.time() + 0.2)

            _wait_until_still(
                controller=controller,
                axis_idx=axis_idx,
                target_pose=hold_pose,
                pos_tolerance=pos_settle_m,
                vel_threshold=vel_settle_m_s,
                stable_s=settle_stable_s,
                timeout_s=settle_timeout_s,
                sample_dt=sample_dt,
            )

            for trial in range(int(trials)):
                settled = _wait_until_still(
                    controller=controller,
                    axis_idx=axis_idx,
                    target_pose=hold_pose,
                    pos_tolerance=pos_settle_m,
                    vel_threshold=vel_settle_m_s,
                    stable_s=settle_stable_s,
                    timeout_s=settle_timeout_s,
                    sample_dt=sample_dt,
                )
                start_pose = settled['pose'].copy()
                start_axis = settled['axis_pos']
                settle_pos_err = settled.get('pos_err', np.nan)
                direction = 1.0 if trial % 2 == 0 else -1.0
                target_pose = start_pose.copy()
                target_pose[axis_idx] += direction * amplitude_m
                commanded_delta = direction * (float(target_pose[axis_idx]) - start_axis)

                schedule_call_t = time.time()
                controller.schedule_waypoint(target_pose, schedule_call_t + step_duration_s)
                onset = _wait_for_motion_onset(
                    controller=controller,
                    axis_idx=axis_idx,
                    min_timestamp=schedule_call_t,
                    start_axis_pos=start_axis,
                    direction=direction,
                    vel_threshold=vel_onset_m_s,
                    displacement_threshold=disp_onset_m,
                    timeout_s=onset_timeout_s,
                    sample_dt=sample_dt,
                )
                if onset is None:
                    print(f"[trial {trial}] no motion onset detected within window")
                    hold_pose = target_pose
                    continue

                latency = onset['t'] - schedule_call_t
                if latency < -1e-4:
                    print(
                        f"[trial {trial}] stale timestamp detected, ignoring "
                        f"latency={latency * 1000:.1f}ms"
                    )
                    hold_pose = target_pose
                    continue

                latencies_s.append(latency)
                print(
                    f"[trial {trial}] target_{axis}={target_pose[axis_idx]:.5f}m "
                    f"start={start_axis:.5f}m "
                    f"cmd_delta={commanded_delta * 1000:.2f}mm "
                    f"settle_err={settle_pos_err * 1000:.2f}mm "
                    f"onset_vel={onset['vel']:.4f}m/s "
                    f"onset_disp={onset['displacement'] * 1000:.2f}mm "
                    f"action_latency={latency * 1000:.1f}ms"
                )
                hold_pose = target_pose

            # Return to the original center pose.
            controller.schedule_waypoint(center_pose, time.time() + 0.5)
            time.sleep(0.8)

    if not latencies_s:
        print("[FAIL] no valid trials, can't compute statistics")
        sys.exit(1)

    arr = np.asarray(latencies_s, dtype=np.float64)
    median = float(np.median(arr))
    print()
    print(f"trials succeeded              : {arr.size}/{trials}")
    print(f"action_latency mean/median    : {arr.mean() * 1000:.1f}ms / {median * 1000:.1f}ms")
    print(f"action_latency p95            : {np.quantile(arr, 0.95) * 1000:.1f}ms")
    print(f"action_latency std            : {arr.std() * 1000:.1f}ms")
    print()
    print(f"measured closed_loop_onset_latency : {max(0.0, median):.4f} s")
    print("  Do not fill this directly into robot_action_latency without checking eval timing.")


if __name__ == "__main__":
    main()
