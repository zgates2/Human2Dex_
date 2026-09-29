import enum
import multiprocessing as mp
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Optional

import numpy as np

from umi.real_world.force_sensor import ForceSensorRS485Driver
from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    SHUTDOWN = 0
    TARE = 1


class ForceSensorController(mp.Process):
    """Cross-process wrapper around ForceSensorRS485Driver.

    Mirrors WSGController's lifecycle and SHM API so BimanualUmiEnv can treat
    force sensors and grippers symmetrically:
        get_state(k=None, out=None) -> latest single frame (k=None) or last-k.
        get_all_state()             -> all buffered frames as dict-of-arrays.

    Ring-buffer schema:
        wrench                     : (6,) float32  [Fx,Fy,Fz,Mx,My,Mz]
        force_capture_timestamp    : wall-clock seconds at frame parse
        force_receive_timestamp    : wall-clock seconds at SHM push
        force_timestamp            : capture - receive_latency (used for alignment)
        force_frame_id             : monotonically increasing
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        port: str,
        baudrate: int = 1000000,
        device_id: int = 0x01,
        frequency: float = 500.0,
        get_max_k: Optional[int] = None,
        command_queue_size: int = 64,
        launch_timeout: float = 5.0,
        receive_latency: float = 0.0,
        zero_on_start: bool = False,
        verbose: bool = False,
    ):
        super().__init__(name="ForceSensorController")
        self.port = port
        self.baudrate = baudrate
        self.device_id = device_id
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.zero_on_start = zero_on_start
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        example_cmd = {
            'cmd': Command.SHUTDOWN.value,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_cmd,
            buffer_size=command_queue_size,
        )

        example_ring = {
            'wrench': np.zeros(6, dtype=np.float32),
            'force_capture_timestamp': time.time(),
            'force_receive_timestamp': time.time(),
            'force_timestamp': time.time(),
            'force_frame_id': 0,
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_ring,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    def start(self, wait: bool = True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[ForceSensorController] spawned at pid={self.pid}")

    def stop(self, wait: bool = True):
        self.input_queue.put({'cmd': Command.SHUTDOWN.value})
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def tare(self):
        self.input_queue.put({'cmd': Command.TARE.value})

    def get_state(self, k: Optional[int] = None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        driver = ForceSensorRS485Driver(
            port=self.port,
            baudrate=self.baudrate,
            device_id=self.device_id,
            enable_crc_check=True,
        )
        if not driver.open():
            raise RuntimeError(
                f"ForceSensorController: failed to open {self.port} @ {self.baudrate}"
            )

        # 等驱动 producer 线程稳定地写出第一帧后再发零位指令，否则零位包可能
        # 在 ADC 还没就绪时被丢弃。
        time.sleep(0.5)
        if self.zero_on_start:
            driver.set_zero_calibration(auto=True)
            if self.verbose:
                print("[ForceSensorController] sent auto zero calibration")

        # monotonic → wall 偏移在子进程启动时取一次，运行期间不再刷新；
        # 见 umi/common/timesync.monotonic_to_wall_offset_s。
        wall_minus_monotonic = time.time() - time.monotonic()
        frame_id = 0
        last_pushed_capture_ns = -1

        keep_running = True
        try:
            while keep_running:
                latest = driver.get_last_frame()
                if latest is not None:
                    capture_monotonic_ns, sensor_tuple = latest
                    if capture_monotonic_ns != last_pushed_capture_ns:
                        last_pushed_capture_ns = capture_monotonic_ns
                        capture_wall = capture_monotonic_ns * 1e-9 + wall_minus_monotonic
                        receive_wall = time.time()
                        frame_id += 1
                        self.ring_buffer.put({
                            'wrench': np.asarray(sensor_tuple, dtype=np.float32),
                            'force_capture_timestamp': capture_wall,
                            'force_receive_timestamp': receive_wall,
                            'force_timestamp': capture_wall - self.receive_latency,
                            'force_frame_id': frame_id,
                        })
                        if not self.ready_event.is_set():
                            self.ready_event.set()

                try:
                    commands = self.input_queue.get_all()
                    cmds = commands['cmd']
                except Empty:
                    cmds = []
                for cmd in cmds:
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    if cmd == Command.TARE.value:
                        driver.set_zero_calibration(auto=True)

                # 500Hz 上限 → 2ms,这里 0.5ms 是 4x 余量,不会 starve 队列也不会过度 spin
                time.sleep(0.0005)
        finally:
            driver.close()
            if self.verbose:
                stats = driver.get_frame_statistics()
                if stats is not None:
                    print(f"[ForceSensorController] final stats: {stats}")
