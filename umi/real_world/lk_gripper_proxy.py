"""LkMotor gripper async proxy, ported from omniumi.drivers.gripper.proxy.

Wraps the omniumi-style GripperController (LkMotor impedance gripper) in a
subprocess; exposes WSGController-shaped API so BimanualUmiEnv treats it the
same as WSG:
    start / stop / start_wait / stop_wait / is_ready / __enter__ / __exit__
    schedule_waypoint(pos: float, target_time: float)
    get_state(k=None) / get_all_state()

Unit convention:
    All gripper positions exposed by this proxy (both schedule_waypoint
    inputs and gripper_position state outputs) are in **radians**, matching
    the policy training convention. The upstream GripperController.move_gripper
    `width_m` parameter is also rad despite its name (see lk_gripper_controller).
    The `gripper_position` field name in get_all_state() output is kept for
    drop-in compatibility with BimanualUmiEnv's interpolation logic, even
    though the value is rad.

Differences from omniumi:
- Adds a small 1D linear interpolator in the subprocess worker so a sparse
  sequence of (target_time, pos) waypoints becomes a continuous target
  stream fed to GripperController.move_gripper, matching how RTDE/
  WSGController consume policy chunks.
- Replaces loguru with stdlib logging, replaces omniumi.core.timesync
  with umi.common.timesync, drops point-in-time / interpolated_at
  helpers (not used by BimanualUmiEnv).
"""
import bisect
import collections
import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _WaypointInterpolator:
    """Forward-extending 1D linear interpolator over (t_wall, pos_rad) pairs.

    Newer schedule() calls truncate previously-scheduled waypoints whose
    target_time is later than the new one (replan semantics: most recent
    plan wins beyond its earliest target_time).
    """

    def __init__(self, t0_wall: float, pos0_rad: float, max_history_s: float = 5.0):
        self._times: List[float] = [t0_wall]
        self._poses: List[float] = [pos0_rad]
        self._max_history_s = max_history_s

    def schedule(self, t_wall: float, pos_rad: float) -> None:
        if t_wall <= self._times[-1]:
            idx = bisect.bisect_left(self._times, t_wall)
            self._times = self._times[:idx]
            self._poses = self._poses[:idx]
        self._times.append(t_wall)
        self._poses.append(pos_rad)

    def call(self, t_wall: float) -> float:
        if t_wall <= self._times[0]:
            return self._poses[0]
        if t_wall >= self._times[-1]:
            return self._poses[-1]
        idx = bisect.bisect_right(self._times, t_wall) - 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        p0, p1 = self._poses[idx], self._poses[idx + 1]
        return p0 + (p1 - p0) * (t_wall - t0) / (t1 - t0)

    def prune(self, t_wall_now: float) -> None:
        cutoff = t_wall_now - self._max_history_s
        if self._times[0] >= cutoff:
            return
        idx = bisect.bisect_left(self._times, cutoff)
        # 保留前一个 anchor 避免插值首点空缺
        idx = max(0, idx - 1)
        if idx > 0:
            self._times = self._times[idx:]
            self._poses = self._poses[idx:]


class LkGripperProxy:
    def __init__(
        self,
        config: dict,
        receive_latency: float = 0.0,
        state_buffer_capacity: int = 600,
        worker_loop_hz: float = 60.0,
        allow_repeat_frames: bool = True,
        launch_timeout: float = 30.0,
    ):
        """
        Args:
            config: GripperController kwargs.
                Single motor:   {'mode': 'single', 'port1': ..., 'motor1_id': ...}
                Dual motor:     {'mode': 'dual',   'port1': ..., 'motor1_id': ...,
                                                   'port2': ..., 'motor2_id': ...}
            receive_latency: seconds; subtracted from capture timestamp to get
                gripper_timestamp (alignment field).
            state_buffer_capacity: 主进程 in-process buffer 容量,worker_loop_hz * 10s 量级即可。
            worker_loop_hz: 子进程调度+状态上报频率(注意 GripperController 内部
                控制循环是 120Hz,与这个独立)。
            allow_repeat_frames: async_read() 没有新帧时是否返回上一帧而非超时(保留 omniumi 行为)。
            launch_timeout: start_wait() 等待 ready 的最长秒数。
        """
        self._config = dict(config)
        self._mode = self._config.get('mode', 'single')
        self.receive_latency = receive_latency
        self.worker_loop_hz = worker_loop_hz
        self.allow_repeat_frames = allow_repeat_frames
        self.launch_timeout = launch_timeout

        self._command_queue: mp.Queue = mp.Queue()
        self._state_queue: mp.Queue = mp.Queue(maxsize=1)
        self._startup_queue: mp.Queue = mp.Queue(maxsize=1)

        self._process: Optional[mp.Process] = None
        self._startup_complete = False

        self._state_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._state_lock = threading.Lock()
        self._latest_state_dict: Optional[Dict[str, float]] = None
        self._latest_frame_id = 0
        self._new_state_event = threading.Event()

        # 主进程内 deque 缓冲(进程内访问,无 SHM)。BimanualUmiEnv.get_obs 调
        # get_all_state() 时从这里 stack 出 dict-of-arrays。
        self._state_buffer: collections.deque = collections.deque(maxlen=state_buffer_capacity)
        self._buffer_lock = threading.Lock()

        # 单电机时把第 4-6 维(motor2 数据)置 0,与 GripperController.get_current_gripper_states 一致
        if self._mode == 'dual':
            self._last_known_states = [0.0] * 6
        else:
            self._last_known_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ====== UMI lifecycle API (matches WSGController) ======
    @property
    def is_ready(self) -> bool:
        return self.is_connected

    @property
    def is_connected(self) -> bool:
        return (
            self._startup_complete
            and self._process is not None
            and self._process.is_alive()
            and self._state_thread is not None
            and self._state_thread.is_alive()
        )

    def start(self, wait: bool = True):
        if self.is_connected:
            logger.warning("LkGripperProxy is already running")
            return

        self._startup_complete = False
        self._new_state_event.clear()
        with self._buffer_lock:
            self._state_buffer.clear()
        with self._state_lock:
            self._latest_state_dict = None
            self._latest_frame_id = 0
        _drain_queue(self._command_queue)
        _drain_queue(self._state_queue)
        _drain_queue(self._startup_queue)

        self._process = mp.Process(
            target=_gripper_worker,
            args=(
                self._command_queue,
                self._state_queue,
                self._startup_queue,
                self._config,
                self.worker_loop_hz,
            ),
            daemon=True,
            name=f"LkGripper({self._config.get('mode', 'single')})",
        )
        self._process.start()
        logger.info("Gripper process started, pid=%s", self._process.pid)

        if wait:
            self.start_wait()

    def start_wait(self):
        try:
            status, detail = self._startup_queue.get(timeout=self.launch_timeout)
        except queue.Empty as exc:
            self.stop(wait=True)
            raise RuntimeError("Gripper startup timed out") from exc

        if status != 'ready':
            self.stop(wait=True)
            raise RuntimeError(f"Failed to start gripper controller: {detail}")

        self._stop_event.clear()
        self._state_thread = threading.Thread(
            target=self._state_reading_loop,
            name="lk_gripper_state_reader",
            daemon=True,
        )
        self._state_thread.start()
        self._startup_complete = True

    def stop(self, wait: bool = True):
        logger.info("Stopping LkGripperProxy")
        self._startup_complete = False

        if self._state_thread and self._state_thread.is_alive():
            self._stop_event.set()
            self._state_thread.join(timeout=2.0)
            if self._state_thread.is_alive():
                logger.warning("State reading thread did not stop gracefully")
        self._state_thread = None

        if self._process and self._process.is_alive():
            try:
                self._command_queue.put(('shutdown', None))
            except Exception:
                pass
            if wait:
                self.stop_wait()

    def stop_wait(self, timeout: float = 5.0):
        if self._process is None:
            return
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            logger.warning("Gripper process did not terminate; forcing")
            self._process.terminate()
            self._process.join(timeout=2)
        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # ====== UMI command API ======
    def schedule_waypoint(self, pos: float, target_time: float,
                          force_limit_nm: float = 1.5) -> None:
        """Schedule the gripper to reach `pos` (rad) by `target_time` (wall clock)."""
        self._command_queue.put((
            'schedule_waypoint',
            (float(pos), float(target_time), float(force_limit_nm)),
        ))

    def move_gripper(self, pos: float, force_limit_nm: float = 1.5) -> None:
        """Fire-and-forget: drive towards `pos` (rad) immediately."""
        self._command_queue.put((
            'move_gripper_immediate',
            (float(pos), float(force_limit_nm)),
        ))

    def stop_gripper(self) -> None:
        self._command_queue.put(('stop_gripper', ()))

    def restart_put(self, start_time: float) -> None:
        """No-op for LkGripper (no step_idx accumulator like WSG); kept for
        BimanualUmiEnv.start_episode compatibility."""
        pass

    # ====== UMI state-read API (matches WSGController) ======
    def get_state(self, k: Optional[int] = None, out=None) -> Dict[str, np.ndarray]:
        if k is None:
            with self._state_lock:
                latest = self._latest_state_dict
            if latest is None:
                return _empty_state_dict()
            return {k_: np.asarray([v]) for k_, v in latest.items()}
        with self._buffer_lock:
            snap = list(self._state_buffer)[-k:]
        return _stack_state_dicts(snap, expected_len=k)

    def get_all_state(self) -> Dict[str, np.ndarray]:
        with self._buffer_lock:
            snap = list(self._state_buffer)
        return _stack_state_dicts(snap)

    def get_current_gripper_states(self) -> List[float]:
        with self._state_lock:
            return list(self._last_known_states)

    # ====== internals ======
    def _state_reading_loop(self) -> None:
        frame_id = 0
        poll_interval = 1.0 / (self.worker_loop_hz * 2.0)
        while not self._stop_event.is_set():
            try:
                packet = self._state_queue.get(timeout=poll_interval)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("state queue read error: %s", e)
                time.sleep(0.1)
                continue

            frame_id += 1
            receive_time = time.time()
            states_list = packet['states']
            pos_rad = packet['pos_rad']
            measure_time = packet['capture_time']

            state_dict = {
                'gripper_position': pos_rad,
                'gripper_velocity': states_list[1],
                'gripper_force': states_list[2],
                'gripper_measure_timestamp': measure_time,
                'gripper_receive_timestamp': receive_time,
                'gripper_timestamp': measure_time - self.receive_latency,
                'gripper_frame_id': frame_id,
            }

            with self._state_lock:
                self._latest_state_dict = state_dict
                self._latest_frame_id = frame_id
                self._last_known_states = list(states_list)
            with self._buffer_lock:
                self._state_buffer.append(state_dict)
            self._new_state_event.set()


# ---- helpers ----

_STATE_KEYS = (
    'gripper_position',
    'gripper_velocity',
    'gripper_force',
    'gripper_measure_timestamp',
    'gripper_receive_timestamp',
    'gripper_timestamp',
    'gripper_frame_id',
)


def _empty_state_dict() -> Dict[str, np.ndarray]:
    return {k: np.zeros(0, dtype=np.float64) for k in _STATE_KEYS}


def _stack_state_dicts(snap, expected_len: Optional[int] = None) -> Dict[str, np.ndarray]:
    if not snap:
        if expected_len is None:
            return _empty_state_dict()
        return {k: np.zeros(expected_len, dtype=np.float64) for k in _STATE_KEYS}
    out: Dict[str, np.ndarray] = {}
    for key in _STATE_KEYS:
        out[key] = np.asarray([s[key] for s in snap])
    return out


def _drain_queue(q: mp.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break


# ---- subprocess worker ----

def _gripper_worker(
    command_queue: mp.Queue,
    state_queue: mp.Queue,
    startup_queue: mp.Queue,
    config: dict,
    loop_hz: float,
):
    # 子进程内 logger 单独配置一下,避免主进程的 handler 复制带来重复输出
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [gripper-worker] %(levelname)s %(message)s")

    try:
        from umi.real_world.lk_gripper_controller import GripperController
        gripper = GripperController(**config)
        gripper.start()
        startup_queue.put(('ready', None))
    except Exception as e:
        logger.error("Failed to initialize GripperController: %s\n%s", e, traceback.format_exc())
        try:
            startup_queue.put(('error', str(e)))
        except Exception:
            pass
        return

    interp: Optional[_WaypointInterpolator] = None
    last_force_limit = 1.5
    dt = 1.0 / loop_hz
    running = True

    try:
        while running:
            loop_start = time.perf_counter()

            # 1) 把队列里所有 pending 命令清空再决定行为
            while True:
                try:
                    command, args = command_queue.get_nowait()
                except queue.Empty:
                    break
                if command == 'shutdown':
                    running = False
                    break
                if command == 'schedule_waypoint':
                    pos_rad, target_time, force_limit = args
                    if interp is None:
                        interp = _WaypointInterpolator(
                            t0_wall=time.time(),
                            pos0_rad=gripper.current_position_rad,
                        )
                    interp.schedule(target_time, pos_rad)
                    last_force_limit = force_limit
                elif command == 'move_gripper_immediate':
                    pos_rad, force_limit = args
                    if interp is None:
                        interp = _WaypointInterpolator(
                            t0_wall=time.time(),
                            pos0_rad=gripper.current_position_rad,
                        )
                    interp.schedule(time.time(), pos_rad)
                    last_force_limit = force_limit
                elif command == 'stop_gripper':
                    gripper.stop_gripper()
                    interp = None
                else:
                    logger.warning("unknown command in worker: %s", command)

            if not running:
                break

            # 2) feed interpolated target to gripper if we have a plan
            if interp is not None:
                now = time.time()
                interp.prune(now)
                target_pos_rad = interp.call(now)
                # upstream GripperController.move_gripper 的 width_m 参数实际当 rad 用
                gripper.move_gripper(width_m=target_pos_rad, force_limit_nm=last_force_limit)

            # 3) 上报状态
            states = gripper.get_current_gripper_states()
            packet = {
                'states': list(states),
                'pos_rad': states[0],
                'capture_time': time.time(),
            }
            try:
                state_queue.put_nowait(packet)
            except queue.Full:
                try:
                    state_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    state_queue.put_nowait(packet)
                except queue.Full:
                    pass

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        logger.info("Gripper worker interrupted")
    except Exception as e:
        logger.error("Gripper worker fatal: %s\n%s", e, traceback.format_exc())
    finally:
        try:
            gripper.stop()
        except Exception:
            pass
        logger.info("Gripper worker exited")
