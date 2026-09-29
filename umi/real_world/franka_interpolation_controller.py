import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
from umi.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    HOLD_POSITION = 3
    RESET_JOINTS = 4

tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([0, 0.0336, 0.274])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

DEFAULT_TX_FLANGE_TIP = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip


def _parse_tx_flange_tip(value):
    '''
        把传进来的 tx_flange_tip 参数统一解析成一个 4x4 位姿变换矩阵
    '''
    if value is None:
        return DEFAULT_TX_FLANGE_TIP.copy()

    tx = np.asarray(value, dtype=np.float64)
    if tx.shape == (4, 4):
        if not np.allclose(tx[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError(f"tx_flange_tip last row must be [0, 0, 0, 1], got {tx[3]}")
        return tx.copy()

    if tx.shape == (6,):
        return pose_to_mat(tx)

    raise ValueError(
        "tx_flange_tip must be either a 4x4 homogeneous matrix or "
        "a 6D pose [x, y, z, rx, ry, rz]."
    )

class FrankaInterface:
    def __init__(self, ip='172.16.0.1', port=4242, pose_api='rpy', rpc_timeout=5.0, tx_flange_tip=None):
        self.pose_api = pose_api
        self.tx_flange_tip = _parse_tx_flange_tip(tx_flange_tip)
        try:
            self.server = zerorpc.Client(heartbeat=20, timeout=rpc_timeout)
        except TypeError:
            self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        if self.pose_api == 'rpy':
            pose_rpy = np.array(self.server.get_ee_pose_rpy(), dtype=np.float64)
            rotvec = st.Rotation.from_euler('xyz', pose_rpy[3:]).as_rotvec()
            return np.concatenate([pose_rpy[:3], rotvec])
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ self.tx_flange_tip)
        return tip_pose
    
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        if self.pose_api == 'rpy':
            return np.zeros(7, dtype=np.float64)
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )

    def move_to_ee_pose(self, pose: np.ndarray, time_to_go: float):
        if self.pose_api == 'rpy':
            self.update_desired_ee_pose(pose)
            time.sleep(float(time_to_go))
            return
        self.server.move_to_ee_pose(pose.tolist(), time_to_go)
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        if self.pose_api == 'rpy':
            pose = np.asarray(pose, dtype=np.float64)
            rpy = st.Rotation.from_rotvec(pose[3:]).as_euler('xyz')
            pose_rpy = np.concatenate([pose[:3], rpy])
            self.server.update_desired_ee_pose_rpy(pose_rpy.tolist())
            return
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()


class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        pose_api='rpy',
        rpc_timeout=5.0,
        read_only=False,
        start_impedance_on_start=True,
        impedance_start_delay=0.5,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        max_commands_per_cycle=1,
        tx_flange_tip=None
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.pose_api = pose_api
        self.rpc_timeout = rpc_timeout
        self.read_only = read_only
        self.start_impedance_on_start = start_impedance_on_start
        self.impedance_start_delay = impedance_start_delay
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.max_pos_speed = float(max_pos_speed)
        self.max_rot_speed = float(max_rot_speed)
        self.max_commands_per_cycle = max(1, int(max_commands_per_cycle))
        self.tx_flange_tip = _parse_tx_flange_tip(tx_flange_tip)
        self.tx_tip_flange = np.linalg.inv(self.tx_flange_tip)

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'joint_positions': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.reset_done_event = mp.Event()
        self.reset_failed_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        if not self.ready_event.wait(self.launch_timeout):
            alive = self.is_alive()
            exitcode = self.exitcode
            if alive:
                self.stop(wait=False)
            raise TimeoutError(
                "FrankaInterpolationController did not become ready within "
                f"{self.launch_timeout:.1f}s "
                f"(alive={alive}, exitcode={exitcode}). Check Franka ZeroRPC "
                "server, robot state, and cartesian impedance startup."
            )
        if not self.is_alive():
            raise RuntimeError(
                "FrankaInterpolationController exited before ready "
                f"(exitcode={self.exitcode})"
            )
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    def clear_queue(self):
        self.input_queue.clear()

    def hold_position(self):
        """Discard queued waypoints and hold the latest measured TCP pose."""
        self.input_queue.clear()
        self.input_queue.put({'cmd': Command.HOLD_POSITION.value})

    def reset_joints(
            self,
            positions: np.ndarray,
            time_to_go: float = 4.0,
            timeout: float = 30.0):
        """Move to the configured joint reset pose inside the controller process."""
        positions = np.asarray(positions, dtype=np.float64).reshape(7)
        if self.read_only:
            return
        self.reset_done_event.clear()
        self.reset_failed_event.clear()
        self.input_queue.clear()
        self.input_queue.put({
            'cmd': Command.RESET_JOINTS.value,
            'joint_positions': positions,
            'duration': float(time_to_go),
        })
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            if self.reset_done_event.wait(timeout=0.05):
                return
            if self.reset_failed_event.is_set():
                raise RuntimeError("Franka joint reset failed in controller process")
            if not self.is_alive():
                raise RuntimeError(
                    f"Franka controller exited during reset (exitcode={self.exitcode})"
                )
        raise TimeoutError(f"Franka joint reset timed out after {timeout:.1f}s")

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        robot = None
        impedance_started = False

        try:
            if self.verbose:
                print(
                    f"[FrankaPositionalController] Connecting to robot: "
                    f"{self.robot_ip}:{self.robot_port}",
                    flush=True,
                )
            # start polymetis interface
            robot = FrankaInterface(
                self.robot_ip,
                self.robot_port,
                pose_api=self.pose_api,
                rpc_timeout=self.rpc_timeout,
                tx_flange_tip=self.tx_flange_tip,
            )
            if self.verbose:
                print(f"[FrankaPositionalController] Connected to robot: {self.robot_ip}", flush=True)
            
            # init pose
            if self.joints_init is not None and not self.read_only:
                if self.verbose:
                    print("[FrankaPositionalController] Moving to joints_init", flush=True)
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )
                if self.verbose:
                    print("[FrankaPositionalController] joints_init done", flush=True)

            # main loop
            dt = 1. / self.frequency
            if self.verbose:
                print("[FrankaPositionalController] Reading initial EE pose", flush=True)
            curr_pose = robot.get_ee_pose()
            if self.verbose:
                print(f"[FrankaPositionalController] Initial EE pose: {curr_pose}", flush=True)

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            def is_no_controller_running_error(exc):
                return 'no controller running' in repr(exc).lower()

            def start_impedance():
                nonlocal impedance_started
                if self.read_only or impedance_started:
                    return
                if self.verbose:
                    print("[FrankaPositionalController] Starting cartesian impedance", flush=True)
                robot.start_cartesian_impedance(
                    Kx=self.Kx,
                    Kxd=self.Kxd
                )
                # Polymetis needs a short registration window before accepting updates.
                time.sleep(self.impedance_start_delay)
                impedance_started = True
                if self.verbose:
                    print("[FrankaPositionalController] Cartesian impedance started", flush=True)

            if not self.read_only and self.start_impedance_on_start:
                # start franka cartesian impedance policy
                start_impedance()
            elif self.verbose:
                if self.read_only:
                    print("[FrankaPositionalController] Read-only mode: impedance disabled", flush=True)
                else:
                    print("[FrankaPositionalController] Lazy impedance mode: wait for first action", flush=True)

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                if not self.read_only and impedance_started:
                    tip_pose = pose_interp(t_now)
                    if self.pose_api == 'rpy':
                        command_pose = tip_pose
                    else:
                        command_pose = mat_to_pose(pose_to_mat(tip_pose) @ self.tx_tip_flange)

                    # send command to robot
                    try:
                        robot.update_desired_ee_pose(command_pose)
                    except Exception as e:
                        if is_no_controller_running_error(e):
                            if self.verbose:
                                print(
                                    "[FrankaPositionalController] Impedance not running; restarting",
                                    flush=True,
                                )
                            impedance_started = False
                            start_impedance()
                            robot.update_desired_ee_pose(command_pose)
                        else:
                            raise

                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                    
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    n_pending = self.input_queue.qsize()
                    if n_pending <= 0:
                        raise Empty()
                    commands = self.input_queue.get_k(
                        min(self.max_commands_per_cycle, n_pending)
                    )
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        if self.read_only:
                            continue
                        if not impedance_started:
                            curr_pose = state['ActualTCPPose']
                            curr_time = t_now + dt
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[curr_time],
                                poses=[curr_pose],
                            )
                            last_waypoint_time = curr_time
                            start_impedance()
                            t_now = time.monotonic()
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = float(pose_interp.times[-1])
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        if self.read_only:
                            continue
                        if not impedance_started:
                            curr_pose = state['ActualTCPPose']
                            curr_time = t_now + dt
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[curr_time],
                                poses=[curr_pose],
                            )
                            last_waypoint_time = curr_time
                            start_impedance()
                            t_now = time.monotonic()
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        if target_time <= curr_time:
                            target_time = curr_time + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = float(pose_interp.times[-1])
                    elif cmd == Command.HOLD_POSITION.value:
                        if self.read_only:
                            continue
                        curr_time = t_now + dt
                        curr_pose = np.asarray(state['ActualTCPPose'], dtype=np.float64)
                        pose_interp = PoseTrajectoryInterpolator(
                            times=[curr_time],
                            poses=[curr_pose],
                        )
                        last_waypoint_time = curr_time
                    elif cmd == Command.RESET_JOINTS.value:
                        if self.read_only:
                            self.reset_done_event.set()
                            continue
                        try:
                            if impedance_started:
                                robot.terminate_current_policy()
                                impedance_started = False
                            robot.move_to_joint_positions(
                                positions=np.asarray(command['joint_positions'], dtype=np.float64),
                                time_to_go=float(command['duration']),
                            )
                            curr_pose = np.asarray(robot.get_ee_pose(), dtype=np.float64)
                            curr_time = time.monotonic() + dt
                            pose_interp = PoseTrajectoryInterpolator(
                                times=[curr_time],
                                poses=[curr_pose],
                            )
                            last_waypoint_time = curr_time
                            start_impedance()
                            # Reset the loop schedule after the blocking joint move;
                            # otherwise the controller would spin to "catch up" several
                            # seconds of missed iterations.
                            t_start = time.monotonic()
                            iter_idx = 0
                            self.reset_done_event.set()
                        except Exception:
                            self.reset_failed_event.set()
                            raise
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            if robot is not None:
                if not self.read_only and impedance_started:
                    if self.verbose:
                        print("[FrankaPositionalController] terminate_current_policy", flush=True)
                    try:
                        robot.terminate_current_policy()
                    except Exception as e:
                        if self.verbose:
                            print(f"[FrankaPositionalController] terminate_current_policy failed: {e}", flush=True)
                try:
                    robot.close()
                except Exception:
                    pass
                del robot

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
