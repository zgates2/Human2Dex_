"""UMI Policy Client - decoupled inference client for BimanualUmiEnv.

Fixes applied vs the oper-source/zuraron variant:

1. Inference is run on a background thread. The main control loop only
   submits the latest observation and consumes whatever action chunk has
   arrived. Network latency no longer pins the control frequency.

2. action_timestamps are anchored to "obs_timestamp + (i+1)*dt" of the
   observation that *generated* the chunk, not the most recent observation.
   This stops action[0] from being scheduled into the past every iteration.

3. The over-budget fallback uses the same wall clock as the normal path
   (obs-timestamp baseline), so the two branches don't fight.

4. The dead `exec_start=0, exec_end=16` slice was removed.

5. The redundant gripper command issued by the client right before
   env.exec_actions was removed. env.exec_actions is the single source of
   truth for both arm + gripper waypoints, all going through the same
   timing path. Action index for env.exec_actions also fixed (was [3,6]).

6. Observation horizons are now CLI parameters (--camera-obs-horizon etc.)
   so they can be aligned to the policy server's expected shape_meta. The
   prior `gripper_obs_horizon=2` hardcode is gone.

Unit convention:
    Server output gripper value is in **radians**, matching the policy
    training. The client passes it through unchanged to env.exec_actions,
    which routes to LkGripperProxy (rad in, rad out) or WSGController
    (meters in -- in which case use a different policy server or model).
"""
import json
import os
import queue
import sys
import threading
import time
from typing import Optional

import click
import cv2
import numpy as np
import requests
import scipy.spatial.transform as st
import yaml
from multiprocessing.managers import SharedMemoryManager

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.camera_factory import build_camera_from_yaml
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode
from umi.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
from scipy.spatial.transform import Rotation as R

try:
    import json_numpy as _json_numpy
    _json_numpy.patch()
    json_numpy = _json_numpy
    USE_JSON_NUMPY = True
except ImportError:
    json_numpy = None
    USE_JSON_NUMPY = False
    print("[client] json_numpy not found, falling back to plain json (slower).")


# ============ pose conversions ============

def axis_angle_to_rpy(axis_angle: np.ndarray) -> np.ndarray:
    shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)
    rpy = st.Rotation.from_rotvec(aa).as_euler('xyz')
    return rpy.reshape(*shape, 3)


def rpy_to_axis_angle(rpy: np.ndarray) -> np.ndarray:
    shape = rpy.shape[:-1]
    rpy_flat = rpy.reshape(-1, 3)
    aa = st.Rotation.from_euler('xyz', rpy_flat).as_rotvec()
    return aa.reshape(*shape, 3)


# ============ collisions ============

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed[:, 2]), 0)
    ee_pose[2] += delta


def solve_sphere_collision(ee_poses, robots_config, tx_robot1_robot0=None):
    n = len(robots_config)
    tx_that_this = np.identity(4)
    if tx_robot1_robot0 is not None:
        tx_that_this = np.asarray(tx_robot1_robot0, dtype=np.float64).copy()
    else:
        tx_that_this[:3, 3] = np.array([0, 0.89, 0])
    for i in range(n):
        for j in range(i + 1, n):
            ee_i = pose_to_mat(ee_poses[i][:6])
            li = np.identity(4); li[:3, 3] = np.asarray(robots_config[i]['sphere_center'])
            gi = ee_i @ li
            ci = gi[:3, 3]
            ee_j = pose_to_mat(ee_poses[j][:6])
            lj = np.identity(4); lj[:3, 3] = np.asarray(robots_config[j]['sphere_center'])
            gj = tx_that_this @ ee_j @ lj
            cj = gj[:3, 3]
            d = np.linalg.norm(cj - ci)
            thr = robots_config[i]['sphere_radius'] + robots_config[j]['sphere_radius']
            if d < thr:
                print('avoid collision between two arms')
                half = (thr - d) / 2
                normal = (cj - ci) / d
                gi[:3, 3] -= half * normal
                gj[:3, 3] += half * normal
                ee_poses[i][:6] = mat_to_pose(gi @ np.linalg.inv(li))
                ee_poses[j][:6] = mat_to_pose(np.linalg.inv(tx_that_this) @ gj @ np.linalg.inv(lj))


# ============ policy client ============

class PolicyClient:
    def __init__(self, server_url: str, timeout: float = 30.0):
        if not server_url.startswith('http'):
            server_url = f'http://{server_url}'
        self.server_url = server_url
        self.session = requests.Session()
        self.timeout = timeout

    def call_server(self, obs_dict: dict) -> Optional[np.ndarray]:
        if USE_JSON_NUMPY:
            body = json_numpy.dumps(obs_dict)
        else:
            def to_serializable(o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if isinstance(o, dict):
                    return {k: to_serializable(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [to_serializable(v) for v in o]
                return o
            body = json.dumps(to_serializable(obs_dict))

        try:
            resp = self.session.post(
                f"{self.server_url}/dp",
                data=body,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            if USE_JSON_NUMPY:
                result = json_numpy.loads(resp.text)
            else:
                result = json.loads(resp.text)
            action = result["action"]
            if isinstance(action, dict) and "__numpy__" in action:
                action = json_numpy.decode(action)
            return np.asarray(action)
        except Exception as e:
            print(f"[PolicyClient] server error: {e}")
            return None


# ============ async inference worker ============

class _InferenceWorker:
    """Run policy_client.call_server in a background thread.

    submit(server_obs, obs_timestamp) — non-blocking, replaces any
        pending input so only the freshest obs is served.
    latest() — returns the most recent finished result, or None.
    """

    def __init__(self, policy_client: PolicyClient):
        self.policy_client = policy_client
        self._in: queue.Queue = queue.Queue(maxsize=1)
        self._out: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()

    def submit(self, server_obs: dict, obs_timestamp: float) -> None:
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        try:
            self._in.put_nowait((server_obs, obs_timestamp))
        except queue.Full:
            pass

    def latest(self) -> Optional[dict]:
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                server_obs, obs_ts = self._in.get(timeout=0.05)
            except queue.Empty:
                continue
            t0 = time.time()
            raw_action = self.policy_client.call_server(server_obs)
            t1 = time.time()
            try:
                self._out.get_nowait()
            except queue.Empty:
                pass
            try:
                self._out.put_nowait({
                    'raw_action': raw_action,
                    'obs_timestamp': obs_ts,
                    'inference_started_at': t0,
                    'inference_finished_at': t1,
                })
            except queue.Full:
                pass


# ============ obs/action helpers ============

def build_server_obs(obs: dict, num_robots: int, episode_start_pose, tx_robot1_robot0) -> dict:
    eef_pos = obs['robot0_eef_pos']
    eef_rot_rpy = axis_angle_to_rpy(obs['robot0_eef_rot_axis_angle'])
    server_obs = {
        'fisheye_img': obs['camera0_rgb'],
        'left_robot_tcp_xyzrpy': np.concatenate([eef_pos, eef_rot_rpy], axis=-1),
        'left_robot_gripper_width_for_action': obs['robot0_gripper_width'],
        'episode_start_pose': episode_start_pose,
        'tx_robot1_robot0': (
            tx_robot1_robot0.tolist()
            if isinstance(tx_robot1_robot0, np.ndarray)
            else tx_robot1_robot0
        ),
    }
    if 'robot0_wrench' in obs:
        server_obs['left_robot_tcp_wrench'] = obs['robot0_wrench']
    if 'robot0_wrench_raw' in obs:
        server_obs['left_robot_tcp_wrench_raw'] = obs['robot0_wrench_raw']
    if num_robots > 1:
        eef_pos1 = obs['robot1_eef_pos']
        eef_rot_rpy1 = axis_angle_to_rpy(obs['robot1_eef_rot_axis_angle'])
        server_obs['right_robot_tcp_xyzrpy'] = np.concatenate([eef_pos1, eef_rot_rpy1], axis=-1)
        server_obs['right_robot_gripper_width_for_action'] = obs['robot1_gripper_width']
    return server_obs


def convert_server_action(raw_action: np.ndarray, num_robots: int) -> np.ndarray:
    """Server returns (horizon, 7 * n_robots) with each robot's 7 dims =
    xyz(3) + rpy(3) + gripper(1). Convert rpy → axis_angle in place and
    return the same shape (horizon, 7 * n_robots) ready for env.exec_actions."""
    dim_per = 7
    out = []
    for i in range(num_robots):
        s = i * dim_per
        xyz = raw_action[..., s:s + 3]
        rpy = raw_action[..., s + 3:s + 6]
        grip = raw_action[..., s + 6:s + 7]
        aa = rpy_to_axis_angle(rpy)
        out.append(np.concatenate([xyz, aa, grip], axis=-1))
    return np.concatenate(out, axis=-1)


# ============ main ============

@click.command()
@click.option('--server-url', '-s', required=True, help='Policy server URL (e.g. http://192.168.1.100:8099)')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot-config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--camera-reorder', '-cr', default='0', help='Camera reorder indices')
@click.option('--vis-camera-idx', default=0, type=int)
@click.option('--init-joints', '-j', is_flag=True, default=False)
@click.option('--max-duration', '-md', default=2_000_000.0, type=float, help='Max episode duration (s)')
@click.option('--max-timesteps', '-mt', default=500, type=int)
@click.option('--frequency', '-f', default=5.0, type=float, help='Control frequency (Hz)')
@click.option('--obs-image-resolution', default=224, type=int)
@click.option('--camera-capture-fps', default=60.0, type=float,
              help='Physical camera capture fps; pass through to env (default 60 for UVC).')
@click.option('--camera-obs-latency', default=0.17, type=float)
@click.option('--camera-obs-horizon', default=2, type=int)
@click.option('--robot-obs-horizon', default=2, type=int)
@click.option('--gripper-obs-horizon', default=2, type=int)
@click.option('--force-obs-horizon', default=2, type=int)
@click.option('--no-mirror', '-nm', is_flag=True, default=False)
@click.option('--mirror-swap', is_flag=True, default=False)
@click.option('--temporal-agg', is_flag=True, default=False)
@click.option('--ensemble-steps', type=int, default=8)
@click.option('--action-horizon', default=16, type=int,
              help='Expected horizon length of server-returned action chunks. '
                   'Used to size the temporal-aggregation buffer. Set to the '
                   'training-time horizon (typically 16).')
@click.option('--init-pose', default=None,
              help='Optional 6-float comma-separated home pose (x,y,z,rx,ry,rz) used with --init-joints')
@click.option('--init-gripper-rad', default=None, type=float,
              help='Optional initial gripper angle in rad; only valid with --init-joints')
def main(server_url, output, robot_config,
         camera_reorder, vis_camera_idx, init_joints,
         max_duration, max_timesteps,
         frequency, obs_image_resolution, camera_capture_fps, camera_obs_latency,
         camera_obs_horizon, robot_obs_horizon, gripper_obs_horizon, force_obs_horizon,
         no_mirror, mirror_swap, temporal_agg, ensemble_steps, action_horizon,
         init_pose, init_gripper_rad):

    dt = 1.0 / frequency
    obs_res = (obs_image_resolution, obs_image_resolution)

    cfg = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    tx_robot1_robot0 = np.array(cfg.get('tx_left_right', np.eye(4)))
    robots_config = cfg['robots']
    grippers_config = cfg['grippers']
    num_robots = len(robots_config)

    print("=" * 60)
    print("UMI Policy Client")
    print(f"  server: {server_url}")
    print(f"  control freq: {frequency} Hz   (dt={dt*1000:.0f} ms)")
    print(f"  robots: {num_robots}")
    print(f"  obs horizons: cam={camera_obs_horizon} robot={robot_obs_horizon} grip={gripper_obs_horizon} force={force_obs_horizon}")
    print("=" * 60)

    policy_client = PolicyClient(server_url)
    inference = _InferenceWorker(policy_client)

    try:
        with SharedMemoryManager() as shm_manager:
            camera_inst, capture_fps_yaml, obs_latency_yaml = build_camera_from_yaml(
                shm_manager=shm_manager,
                cameras_cfg=cfg.get('cameras'),
                obs_image_resolution=obs_res,
                camera_obs_latency=camera_obs_latency,
            )
            with KeystrokeCounter() as key_counter, \
                BimanualUmiEnv(
                    output_dir=output,
                    robots_config=robots_config,
                    grippers_config=grippers_config,
                    force_sensors_config=cfg.get('force_sensors'),
                    frequency=frequency,
                    obs_image_resolution=obs_res,
                    obs_float32=False,
                    camera_reorder=[int(x) for x in camera_reorder],
                    init_joints=init_joints,
                    enable_multi_cam_vis=True,
                    camera_obs_latency=obs_latency_yaml,
                    camera_capture_fps=capture_fps_yaml,
                    camera=camera_inst,
                    camera_down_sample_steps=1,
                    robot_down_sample_steps=1,
                    gripper_down_sample_steps=1,
                    force_down_sample_steps=1,
                    camera_obs_horizon=camera_obs_horizon,
                    robot_obs_horizon=robot_obs_horizon,
                    gripper_obs_horizon=gripper_obs_horizon,
                    force_obs_horizon=force_obs_horizon,
                    no_mirror=no_mirror,
                    mirror_swap=mirror_swap,
                    max_pos_speed=2.0,
                    max_rot_speed=6.0,
                    shm_manager=shm_manager,
                ) as env:
                cv2.setNumThreads(2)
                print("Waiting for camera...")
                time.sleep(1.0)

                _maybe_init_robot(env, init_joints, init_pose, init_gripper_rad)

                print("\n" + "=" * 60)
                print("Ready. Press 'C' to start policy, 'Q' to quit.")
                print("=" * 60)

                action_dim = 7 * num_robots
                all_time_actions = np.zeros(
                    (max_timesteps, max_timesteps + action_horizon, action_dim),
                    dtype=np.float64,
                )

                while True:
                    if not _wait_for_start(env, vis_camera_idx, key_counter):
                        return
                    _run_policy_episode(
                        env=env,
                        inference=inference,
                        num_robots=num_robots,
                        robots_config=robots_config,
                        tx_robot1_robot0=tx_robot1_robot0,
                        frequency=frequency,
                        dt=dt,
                        max_duration=max_duration,
                        max_timesteps=max_timesteps,
                        temporal_agg=temporal_agg,
                        ensemble_steps=ensemble_steps,
                        vis_camera_idx=vis_camera_idx,
                        key_counter=key_counter,
                        all_time_actions=all_time_actions,
                    )
    finally:
        inference.stop()


def _maybe_init_robot(env, init_joints: bool,
                      init_pose: Optional[str], init_gripper_rad: Optional[float]):
    if not init_joints:
        return
    print("Initializing robot...")
    try:
        time.sleep(0.5)
        obs = env.get_obs()
        current_pos = obs['robot0_eef_pos'][-1]
        current_rot = obs['robot0_eef_rot_axis_angle'][-1]
        print(f"current EEF pose: {np.concatenate([current_pos, current_rot])}")

        if init_pose is not None:
            target = np.array([float(x) for x in init_pose.split(',')], dtype=np.float64)
            assert target.shape == (6,), f"--init-pose needs 6 values, got {target.shape}"
            print(f"target home pose: {target}")
            init_duration = 3.0
            env.robots[0].servoL(target, duration=init_duration)
            time.sleep(init_duration + 0.5)

        if init_gripper_rad is not None:
            print(f"setting initial gripper to {init_gripper_rad} rad...")
            env.grippers[0].schedule_waypoint(
                pos=float(init_gripper_rad),
                target_time=time.time() + 1.0,
            )
            time.sleep(1.5)
        print("init done.")
    except Exception as e:
        import traceback
        print(f"[init warning] {e}")
        traceback.print_exc()


def _wait_for_start(env, vis_camera_idx, key_counter) -> bool:
    """Block until user presses C (start) or Q (quit). Returns True on start, False on quit."""
    print("\nWaiting for start command (C=start, Q=quit)...")
    while True:
        obs = env.get_obs()
        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
        cv2.putText(vis_img, 'Press C to start, Q to quit', (10, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, thickness=1, color=(255, 255, 255))
        cv2.imshow('default', vis_img[..., ::-1])
        cv2.pollKey()
        for keystroke in key_counter.get_press_events():
            if keystroke == KeyCode(char='q'):
                env.end_episode()
                return False
            if keystroke == KeyCode(char='c'):
                return True
        time.sleep(0.05)


def _run_policy_episode(env, inference, num_robots, robots_config,
                        tx_robot1_robot0, frequency, dt,
                        max_duration, max_timesteps,
                        temporal_agg, ensemble_steps,
                        vis_camera_idx, key_counter, all_time_actions):
    print("\nPolicy control started (S=stop).")
    start_delay = 1.0
    eval_t_start = time.time() + start_delay
    t_start = time.monotonic() + start_delay
    env.start_episode(eval_t_start)

    obs = env.get_obs()
    episode_start_pose = [
        np.concatenate([obs[f'robot{r}_eef_pos'], obs[f'robot{r}_eef_rot_axis_angle']], axis=-1)[-1]
        for r in range(num_robots)
    ]

    frame_latency = 1.0 / env.camera_capture_fps
    precise_wait(eval_t_start - frame_latency, time_func=time.time)
    print("Started!")

    iter_idx = 0
    last_logged_inference_ms: Optional[float] = None
    last_consumed_inference_finished_at: Optional[float] = None

    try:
        while True:
            t_cycle_end = t_start + (iter_idx + 1) * dt

            obs = env.get_obs()
            obs_timestamps = obs['timestamp']
            obs_latest_ts = float(obs_timestamps[-1])

            # 1) submit latest obs for inference (non-blocking; replaces pending)
            server_obs = build_server_obs(obs, num_robots, episode_start_pose, tx_robot1_robot0)
            inference.submit(server_obs, obs_latest_ts)

            # 2) consume newest finished result if any
            result = inference.latest()
            if result is not None:
                if result.get('raw_action') is None:
                    print("[warn] empty action from server, skipping")
                else:
                    inference_ms = (result['inference_finished_at'] - result['inference_started_at']) * 1000.0
                    last_logged_inference_ms = inference_ms
                    last_consumed_inference_finished_at = result['inference_finished_at']
                    action_chunk_xyzrpy = result['raw_action']
                    chunk_obs_ts = result['obs_timestamp']
                    action_chunk = convert_server_action(action_chunk_xyzrpy, num_robots)
                    action_horizon = action_chunk.shape[0]

                    if iter_idx < all_time_actions.shape[0]:
                        # 容量保护:若 server 返回的 horizon 超过 buffer 第二维剩余空间(很罕见),
                        # 截断写入并打印一次提醒,而不是抛 IndexError 直接挂掉。
                        slot_end = min(iter_idx + action_horizon, all_time_actions.shape[1])
                        slot_len = slot_end - iter_idx
                        if slot_len < action_horizon:
                            print(f"[warn] server returned horizon={action_horizon}, "
                                  f"only {slot_len} fits in buffer; truncating writes")
                        all_time_actions[
                            [iter_idx],
                            iter_idx:slot_end,
                        ] = action_chunk[:slot_len]

                    this_target_poses = (
                        _temporal_aggregate(
                            all_time_actions, iter_idx, action_horizon,
                            action_chunk, ensemble_steps,
                        )
                        if temporal_agg else action_chunk
                    )

                    # collisions
                    for tp in this_target_poses:
                        for r in range(num_robots):
                            solve_table_collision(
                                ee_pose=tp[r * 7: r * 7 + 6],
                                gripper_width=tp[r * 7 + 6],
                                height_threshold=robots_config[r].get('height_threshold', 0.01),
                            )
                        if num_robots > 1:
                            solve_sphere_collision(
                                ee_poses=tp.reshape([num_robots, -1]),
                                robots_config=robots_config,
                                tx_robot1_robot0=tx_robot1_robot0,
                            )

                    # action_timestamps anchor: action[i] should execute at
                    # chunk_obs_ts + (i+1)*dt -- i.e. the dt grid that starts
                    # one step after the observation that produced this chunk.
                    action_timestamps = chunk_obs_ts + (np.arange(this_target_poses.shape[0]) + 1) * dt
                    action_exec_latency = 0.01
                    now = time.time()
                    is_new = action_timestamps > (now + action_exec_latency)
                    if np.any(is_new):
                        this_target_poses = this_target_poses[is_new]
                        action_timestamps = action_timestamps[is_new]
                    else:
                        # all in the past: schedule the tail at the next dt grid,
                        # using the same wall-clock baseline as action_timestamps.
                        next_t = chunk_obs_ts + (
                            np.ceil((now + action_exec_latency - chunk_obs_ts) / dt)
                        ) * dt
                        print(f"[over budget] all chunk actions past, fall back to {next_t - now:.3f}s ahead")
                        this_target_poses = this_target_poses[[-1]]
                        action_timestamps = np.array([next_t])

                    env.exec_actions(
                        actions=this_target_poses,
                        timestamps=action_timestamps,
                        compensate_latency=True,
                    )
                    print(f"step={iter_idx} submitted={len(this_target_poses)} "
                          f"inference={inference_ms:.0f}ms")

            # 3) visualize + handle stop
            vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
            text = f'step={iter_idx} t={time.monotonic() - t_start:.1f}s'
            if last_logged_inference_ms is not None:
                text += f' inf={last_logged_inference_ms:.0f}ms'
            cv2.putText(vis_img, text, (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, color=(255, 255, 255))
            cv2.imshow('default', vis_img[..., ::-1])
            cv2.pollKey()

            stop_episode = False
            for keystroke in key_counter.get_press_events():
                if keystroke == KeyCode(char='s'):
                    stop_episode = True
                    print("stop requested.")
            if time.time() - eval_t_start > max_duration:
                stop_episode = True
                print("max duration reached.")
            if iter_idx + 1 >= max_timesteps:
                stop_episode = True
                print("max timesteps reached.")
            if stop_episode:
                env.end_episode()
                break

            precise_wait(t_cycle_end - frame_latency)
            iter_idx += 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        env.end_episode()

    print("Episode ended.")


def _temporal_aggregate(all_time_actions: np.ndarray, iter_idx: int,
                        action_horizon: int, action_chunk: np.ndarray,
                        ensemble_steps: int) -> np.ndarray:
    """ACT-style exponentially-weighted temporal ensemble across past chunks."""
    seq = all_time_actions[:, iter_idx:iter_idx + action_horizon]
    out = []
    for i in range(action_horizon):
        cands = seq[max(0, iter_idx - ensemble_steps + 1):iter_idx + 1, i]
        populated = np.all(cands != 0, axis=1)
        cands = cands[populated]
        if len(cands) == 0:
            out.append(action_chunk[[i]])
            continue
        k_ = -0.01
        w = np.exp(k_ * np.arange(len(cands)))
        w = w / w.sum()
        rotvec = R.from_rotvec(np.array(cands)[:, 3:6]).mean(weights=w).as_rotvec()
        weighted = (cands * w[:, np.newaxis]).sum(axis=0, keepdims=True)
        weighted[0][3:6] = rotvec
        out.append(weighted)
    return np.concatenate(out, axis=0)


if __name__ == '__main__':
    main()
