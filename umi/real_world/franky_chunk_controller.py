import enum
import multiprocessing as mp
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import zerorpc

from umi.common.pose_util import mat_to_pose, pose_to_mat
from umi.common.precise_sleep import precise_wait
from umi.real_world.franka_interpolation_controller import _parse_tx_flange_tip
from umi.shared_memory.shared_memory_queue import Empty, SharedMemoryQueue
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    STOP = 0
    EXECUTE_CHUNK = 1


class FrankyRpcClient:
    """Small RPC client for the Franky action-chunk server."""

    def __init__(self, ip: str, port: int, timeout: float):
        try:
            self.server = zerorpc.Client(heartbeat=20, timeout=float(timeout))
        except TypeError:
            self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{int(port)}")

    def get_robot_state(self):
        return self.server.get_robot_state()

    def recover_from_errors(self):
        return self.server.recover_from_errors()

    def move_to_joint_positions(self, positions, time_to_go):
        return self.server.move_to_joint_positions(
            np.asarray(positions, dtype=np.float64).reshape(7).tolist(),
            float(time_to_go),
        )

    def execute_cartesian_waypoints(
        self,
        poses,
        durations,
        max_linear_velocity,
        max_angular_velocity,
    ):
        return self.server.execute_cartesian_waypoints(
            np.asarray(poses, dtype=np.float64).tolist(),
            np.asarray(durations, dtype=np.float64).tolist(),
            max_linear_velocity,
            max_angular_velocity,
        )

    def terminate_current_policy(self):
        return self.server.terminate_current_policy()

    def close(self):
        self.server.close()


class FrankyChunkController(mp.Process):
    """Execute each Diffusion Policy action chunk as one Franky motion.

    Unlike ``FrankaInterpolationController``, this process never turns a pose
    trajectory into a 200 Hz stream of ``Robot.move`` calls.  It submits one
    timestamped Cartesian waypoint batch per policy inference and uses a
    separate low-rate state polling loop for observations.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        robot_port=4243,
        frequency=100,
        launch_timeout=60.0,
        rpc_timeout=60.0,
        read_only=False,
        joints_init=None,
        joints_init_duration=4.0,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0,
        max_pos_speed=np.inf,
        max_rot_speed=np.inf,
        max_chunk_size=64,
        min_segment_duration=0.01,
        tx_flange_tip=None,
    ):
        if joints_init is not None:
            joints_init = np.asarray(joints_init, dtype=np.float64)
            if joints_init.shape != (7,):
                raise ValueError(f"joints_init must have shape (7,), got {joints_init.shape}")

        super().__init__(name="FrankyChunkController")
        self.robot_ip = str(robot_ip)
        self.robot_port = int(robot_port)
        self.frequency = float(frequency)
        self.launch_timeout = float(launch_timeout)
        self.rpc_timeout = float(rpc_timeout)
        self.read_only = bool(read_only)
        self.joints_init = joints_init
        self.joints_init_duration = float(joints_init_duration)
        self.verbose = bool(verbose)
        self.receive_latency = float(receive_latency)
        self.max_pos_speed = float(max_pos_speed)
        self.max_rot_speed = float(max_rot_speed)
        self.max_chunk_size = int(max_chunk_size)
        self.min_segment_duration = float(min_segment_duration)
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if self.min_segment_duration <= 0:
            raise ValueError("min_segment_duration must be positive")

        self.tx_flange_tip = _parse_tx_flange_tip(tx_flange_tip)
        self.tx_tip_flange = np.linalg.inv(self.tx_flange_tip)

        if get_max_k is None:
            get_max_k = max(32, int(self.frequency * 5))

        command_example = {
            "cmd": Command.EXECUTE_CHUNK.value,
            "count": 0,
            "target_poses": np.zeros((self.max_chunk_size, 6), dtype=np.float64),
            "target_times": np.zeros((self.max_chunk_size,), dtype=np.float64),
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=command_example,
            buffer_size=32,
        )

        state_example = {
            "ActualTCPPose": np.zeros(6, dtype=np.float64),
            "ActualQ": np.zeros(7, dtype=np.float64),
            "ActualQd": np.zeros(7, dtype=np.float64),
            "robot_receive_timestamp": time.time(),
            "robot_timestamp": time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=self.frequency,
        )
        self.ready_event = mp.Event()

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def _best_effort_remote_stop(self):
        client = None
        try:
            client = FrankyRpcClient(
                self.robot_ip,
                self.robot_port,
                timeout=min(3.0, self.rpc_timeout),
            )
            client.terminate_current_policy()
        except Exception as exc:
            if self.verbose:
                print(f"[FrankyChunkController] remote stop warning: {exc}", flush=True)
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def stop(self, wait=True):
        self._best_effort_remote_stop()
        try:
            self.input_queue.put({"cmd": Command.STOP.value})
        except Exception:
            pass
        if wait:
            self.stop_wait()

    def start_wait(self):
        if not self.ready_event.wait(self.launch_timeout):
            alive = self.is_alive()
            exitcode = self.exitcode
            if alive:
                self.stop(wait=False)
            raise TimeoutError(
                "FrankyChunkController did not become ready within "
                f"{self.launch_timeout:.1f}s (alive={alive}, exitcode={exitcode}). "
                "Check the Franky ZeroRPC server and joint initialization."
            )
        if not self.is_alive():
            raise RuntimeError(
                f"FrankyChunkController exited before ready (exitcode={self.exitcode})"
            )

    def stop_wait(self):
        self.join(timeout=max(5.0, min(self.rpc_timeout + 1.0, 15.0)))
        if self.is_alive():
            self.terminate()
            self.join(timeout=2.0)

    @property
    def is_ready(self):
        return self.ready_event.is_set() and self.is_alive()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def clear_queue(self):
        self.input_queue.clear()

    def servoL(self, pose, duration=0.1):
        target_time = time.time() + float(duration)
        self.schedule_waypoints(
            np.asarray(pose, dtype=np.float64).reshape(1, 6),
            np.asarray([target_time], dtype=np.float64),
        )

    def schedule_waypoint(self, pose, target_time):
        self.schedule_waypoints(
            np.asarray(pose, dtype=np.float64).reshape(1, 6),
            np.asarray([target_time], dtype=np.float64),
        )

    def schedule_waypoints(self, poses, target_times):
        if not self.is_alive():
            raise RuntimeError("FrankyChunkController is not running")
        poses = np.asarray(poses, dtype=np.float64)
        target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
        if poses.ndim != 2 or poses.shape[1] != 6:
            raise ValueError(f"poses must have shape (N, 6), got {poses.shape}")
        if len(poses) != len(target_times):
            raise ValueError(
                f"poses/target_times length mismatch: {len(poses)} vs {len(target_times)}"
            )
        if len(poses) == 0:
            return
        if len(poses) > self.max_chunk_size:
            raise ValueError(
                f"chunk has {len(poses)} waypoints; maximum is {self.max_chunk_size}"
            )
        if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(target_times)):
            raise ValueError("poses and target_times must be finite")
        if np.any(np.diff(target_times) <= 0.0):
            raise ValueError("target_times must be strictly increasing")

        padded_poses = np.zeros((self.max_chunk_size, 6), dtype=np.float64)
        padded_times = np.zeros((self.max_chunk_size,), dtype=np.float64)
        padded_poses[: len(poses)] = poses
        padded_times[: len(target_times)] = target_times
        self.input_queue.put(
            {
                "cmd": Command.EXECUTE_CHUNK.value,
                "count": int(len(poses)),
                "target_poses": padded_poses,
                "target_times": padded_times,
            }
        )

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def _tip_to_flange(self, tip_poses):
        return np.stack(
            [
                mat_to_pose(pose_to_mat(pose) @ self.tx_tip_flange)
                for pose in np.asarray(tip_poses, dtype=np.float64)
            ],
            axis=0,
        )

    def _flange_to_tip(self, flange_pose):
        return mat_to_pose(pose_to_mat(flange_pose) @ self.tx_flange_tip)

    def _put_state(self, state):
        receive_time = time.time()
        self.ring_buffer.put(
            {
                "ActualTCPPose": self._flange_to_tip(
                    np.asarray(state["ee_pose"], dtype=np.float64)
                ),
                "ActualQ": np.asarray(state["joint_positions"], dtype=np.float64),
                "ActualQd": np.asarray(state["joint_velocities"], dtype=np.float64),
                "robot_receive_timestamp": receive_time,
                "robot_timestamp": receive_time - self.receive_latency,
            }
        )

    @staticmethod
    def _validate_robot_state(state):
        if "is_in_control" not in state or "has_errors" not in state:
            raise RuntimeError(
                "Franky server state is missing is_in_control/has_errors. "
                "Restart the updated launch_franky_interface_server.py on port 4243."
            )
        motion_error = state.get("last_motion_error")
        if motion_error:
            raise RuntimeError(
                "Franky asynchronous motion failed: "
                f"{motion_error}; current_errors={state.get('current_errors')}; "
                f"last_motion_errors={state.get('last_motion_errors')}; "
                f"control_command_success_rate="
                f"{state.get('control_command_success_rate')}"
            )
        if bool(state["has_errors"]):
            raise RuntimeError(
                "Franka reported has_errors=True (robot is in an error/Reflex state). "
                f"current_errors={state.get('current_errors')}; "
                f"last_motion_errors={state.get('last_motion_errors')}; "
                "no pending policy chunk will be sent."
            )

    def _latest_command(self):
        try:
            commands = self.input_queue.get_all()
        except Empty:
            return None
        latest = None
        for idx, cmd in enumerate(commands["cmd"]):
            if int(cmd) == Command.STOP.value:
                return {"cmd": Command.STOP.value}
            if int(cmd) == Command.EXECUTE_CHUNK.value:
                latest = {
                    key: value[idx]
                    for key, value in commands.items()
                }
        return latest

    def _execute_chunk(self, client, command):
        count = int(command["count"])
        poses = np.asarray(command["target_poses"][:count], dtype=np.float64)
        target_times = np.asarray(command["target_times"][:count], dtype=np.float64)
        now = time.time()
        future = target_times > now + 1e-3
        if np.any(future):
            poses = poses[future]
            target_times = target_times[future]
        else:
            poses = poses[-1:]
            target_times = np.asarray(
                [now + max(0.05, self.min_segment_duration)], dtype=np.float64
            )

        durations = np.diff(np.concatenate([[now], target_times]))
        durations = np.maximum(durations, self.min_segment_duration)
        flange_poses = self._tip_to_flange(poses)
        max_linear = self.max_pos_speed if np.isfinite(self.max_pos_speed) else None
        max_angular = self.max_rot_speed if np.isfinite(self.max_rot_speed) else None
        result = client.execute_cartesian_waypoints(
            flange_poses,
            durations,
            max_linear_velocity=max_linear,
            max_angular_velocity=max_angular,
        )
        if self.verbose:
            print(
                "[FrankyChunkController] submitted "
                f"{len(flange_poses)} waypoints, result={result}",
                flush=True,
            )

    def run(self):
        client = None
        try:
            client = FrankyRpcClient(
                self.robot_ip,
                self.robot_port,
                timeout=self.rpc_timeout,
            )

            if not self.read_only:
                recovery = client.recover_from_errors()
                if isinstance(recovery, dict) and recovery.get("recovered", False):
                    print(
                        "[FrankyChunkController] Startup error recovery completed",
                        flush=True,
                    )

            if self.joints_init is not None and not self.read_only:
                print("[FrankyChunkController] Moving to joints_init", flush=True)
                client.move_to_joint_positions(
                    self.joints_init,
                    self.joints_init_duration,
                )
                print("[FrankyChunkController] joints_init done", flush=True)

            state = client.get_robot_state()
            self._validate_robot_state(state)
            self._put_state(state)
            self.ready_event.set()
            print(
                f"[FrankyChunkController] Ready: state_hz={self.frequency:.1f}, "
                f"max_chunk={self.max_chunk_size}",
                flush=True,
            )

            dt = 1.0 / self.frequency
            next_cycle = time.monotonic()
            keep_running = True
            pending_chunk = None
            while keep_running:
                command = self._latest_command()
                if command is not None:
                    if int(command["cmd"]) == Command.STOP.value:
                        keep_running = False
                    elif not self.read_only:
                        # Latest policy inference wins, but it must not preempt
                        # the Franky motion currently in progress.
                        pending_chunk = command

                if keep_running:
                    state = client.get_robot_state()
                    self._validate_robot_state(state)
                    self._put_state(state)

                    if (
                        pending_chunk is not None
                        and not self.read_only
                        and not bool(state["is_in_control"])
                    ):
                        self._execute_chunk(client, pending_chunk)
                        pending_chunk = None

                    next_cycle += dt
                    precise_wait(next_cycle, time_func=time.monotonic)
        finally:
            if client is not None:
                if not self.read_only:
                    try:
                        client.terminate_current_policy()
                    except Exception as exc:
                        print(
                            f"[FrankyChunkController] shutdown warning: {exc}",
                            flush=True,
                        )
                try:
                    client.close()
                except Exception:
                    pass
