from __future__ import annotations

import json
import pathlib
import subprocess
import threading
import time
from collections import deque
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np


class L515EgoCamera:
    """Single-camera adapter for the training-side L515 ego RGB stream.

    The RealSense SDK lives in a separate worker Python process.  The parent
    inference process only receives RGB frames through shared memory and
    exposes the same camera.get()/get_vis() interface as MultiMvsCamera.
    """

    def __init__(
        self,
        *,
        serial: str,
        worker_python: str,
        worker_script: str | pathlib.Path,
        input_resolution=(960, 540),
        output_resolution=(224, 224),
        capture_fps: int = 30,
        warmup_frames: int = 30,
        timeout_ms: int = 5000,
        shm_slots: int = 4,
        max_history: Optional[int] = None,
        rotate_180: bool = False,
    ):
        import cv2
        from diffusion_policy.common.cv2_util import get_image_transform

        self.serial = str(serial)
        self.worker_python = str(pathlib.Path(worker_python).expanduser())
        self.worker_script = str(pathlib.Path(worker_script).expanduser().resolve())
        self.input_resolution = tuple(int(v) for v in input_resolution)
        self.output_resolution = tuple(int(v) for v in output_resolution)
        self.capture_fps = int(capture_fps)
        self.warmup_frames = int(warmup_frames)
        self.timeout_ms = int(timeout_ms)
        self.shm_slots = max(2, int(shm_slots))
        self.rotate_180 = bool(rotate_180)
        if len(self.input_resolution) != 2 or min(self.input_resolution) <= 0:
            raise ValueError(f"invalid L515 input_resolution={self.input_resolution!r}")
        if len(self.output_resolution) != 2 or min(self.output_resolution) <= 0:
            raise ValueError(f"invalid L515 output_resolution={self.output_resolution!r}")
        if self.capture_fps <= 0:
            raise ValueError(f"capture_fps must be > 0, got {capture_fps!r}")

        input_w, input_h = self.input_resolution
        self._frame_shape = (input_h, input_w, 3)
        self._frame_bytes = int(np.prod(self._frame_shape)) * np.dtype(np.uint8).itemsize
        self._shm = SharedMemory(
            create=True,
            size=self._frame_bytes * self.shm_slots,
        )
        output_w, output_h = self.output_resolution
        self._vis_shape = (output_h, output_w, 3)
        self._vis_bytes = int(np.prod(self._vis_shape)) * np.dtype(np.uint8).itemsize
        # MultiCameraVisualizer runs in a forked process.  Keep the newest
        # transformed RGB frame in a dedicated shared-memory buffer so that
        # the preview process sees live frames even though its Python object
        # state is a fork-time snapshot.
        self._vis_shm = SharedMemory(create=True, size=self._vis_bytes)
        np.ndarray(self._vis_shape, dtype=np.uint8, buffer=self._vis_shm.buf)[:] = 0
        self._transform = get_image_transform(
            input_res=self.input_resolution,
            output_res=self.output_resolution,
            # The worker requests rs.format.rgb8, so no channel swap is needed.
            bgr_to_rgb=False,
        )
        # Keep cv2 imported in the parent so a missing OpenCV error is raised at
        # construction time rather than after hardware has started.
        self._cv2 = cv2

        history_len = (
            max_history
            if max_history is not None
            else max(64, int(round(self.capture_fps * 5.0)))
        )
        self._history = deque(maxlen=max(8, int(history_len)))
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._first_frame = threading.Event()
        self._stop = threading.Event()
        self._started = False
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: deque[str] = deque(maxlen=40)
        self._startup_error: str | None = None
        self._last_frame_id: int | None = None

    @property
    def n_cameras(self) -> int:
        return 1

    @property
    def is_ready(self) -> bool:
        return (
            self._started
            and self._ready.is_set()
            and self._first_frame.is_set()
            and self._proc is not None
            and self._proc.poll() is None
        )

    def start(self, wait: bool = True, put_start_time: float | None = None):
        del put_start_time
        if self._started:
            if wait:
                self.start_wait()
            return
        if not pathlib.Path(self.worker_python).is_file():
            raise FileNotFoundError(
                f"L515 worker Python not found: {self.worker_python}"
            )
        if not pathlib.Path(self.worker_script).is_file():
            raise FileNotFoundError(
                f"L515 worker script not found: {self.worker_script}"
            )

        cmd = [
            self.worker_python,
            "-u",
            self.worker_script,
            "--serial",
            self.serial,
            "--width",
            str(self.input_resolution[0]),
            "--height",
            str(self.input_resolution[1]),
            "--fps",
            str(self.capture_fps),
            "--warmup-frames",
            str(self.warmup_frames),
            "--timeout-ms",
            str(self.timeout_ms),
            "--shm-name",
            self._shm.name,
            "--shm-slots",
            str(self.shm_slots),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._started = True
        self._stop.clear()
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            name="L515EgoCamera-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            name="L515EgoCamera-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        if wait:
            self.start_wait()

    def start_wait(self, timeout: float = 30.0):
        deadline = time.monotonic() + float(timeout)
        while not self._ready.is_set():
            if self._startup_error is not None:
                raise RuntimeError(self._startup_error)
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(self._worker_failure_message())
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "L515 ego worker did not become ready within "
                    f"{float(timeout):.1f}s"
                )
            time.sleep(0.02)

        while not self._first_frame.is_set():
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(self._worker_failure_message())
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "L515 ego worker became ready but produced no RGB frame "
                    f"within {float(timeout):.1f}s"
                )
            time.sleep(0.01)

    def stop(self, wait: bool = True):
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            if wait:
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        if wait:
            for thread in (self._stdout_thread, self._stderr_thread):
                if thread is not None:
                    thread.join(timeout=1.0)
        self._proc = None
        self._started = False
        self._ready.clear()
        self._first_frame.clear()
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
        try:
            self._vis_shm.close()
        finally:
            try:
                self._vis_shm.unlink()
            except FileNotFoundError:
                pass

    def stop_wait(self):
        self.stop(wait=True)

    def start_recording(self, video_path, start_time: float = -1):
        # InferenceEpisodeRecorder saves policy-aligned RGB images itself.
        # Keep this method as a no-op because BimanualUmiEnv calls it for every
        # injected camera backend.
        del video_path, start_time

    def stop_recording(self):
        return None

    def restart_put(self, start_time: float):
        del start_time

    def get(self, k: int | None = None, out=None):
        del out
        requested = 1 if k is None else max(1, int(k))
        with self._lock:
            frames = list(self._history)
        if not frames:
            raise RuntimeError("L515 ego camera has not produced an RGB frame")
        if len(frames) < requested:
            frames = [frames[0]] * (requested - len(frames)) + frames
        else:
            frames = frames[-requested:]

        result = {
            "color": np.stack([frame["color"] for frame in frames], axis=0),
            "timestamp": np.asarray(
                [frame["timestamp"] for frame in frames], dtype=np.float64
            ),
            "camera_capture_timestamp": np.asarray(
                [frame["camera_capture_timestamp"] for frame in frames],
                dtype=np.float64,
            ),
            "camera_receive_timestamp": np.asarray(
                [frame["camera_receive_timestamp"] for frame in frames],
                dtype=np.float64,
            ),
            "camera_frame_id": np.asarray(
                [frame["camera_frame_id"] for frame in frames], dtype=np.int64
            ),
            "step_idx": np.asarray(
                [frame["step_idx"] for frame in frames], dtype=np.int64
            ),
        }
        return {0: result}

    def get_vis(self, out=None):
        del out
        # This method is called by MultiCameraVisualizer in another process.
        # Read directly from shared memory rather than the parent's deque.
        image = np.ndarray(
            self._vis_shape,
            dtype=np.uint8,
            buffer=self._vis_shm.buf,
        ).copy()
        return {"color": image[None, ...]}

    def close(self):
        self.stop(wait=True)

    def _read_stdout(self):
        assert self._proc is not None
        assert self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = payload.get("type")
            if kind == "ready":
                self._ready.set()
                continue
            if kind == "error":
                self._startup_error = str(payload.get("error", "unknown L515 worker error"))
                self._ready.set()
                continue
            if kind != "frame":
                continue

            slot = int(payload["slot"])
            frame_id = int(payload["frame_id"])
            timestamp = float(payload["timestamp"])
            if frame_id == self._last_frame_id:
                continue
            if slot < 0 or slot >= self.shm_slots:
                self._startup_error = f"L515 worker returned invalid shared-memory slot {slot}"
                self._ready.set()
                continue
            offset = slot * self._frame_bytes
            with self._lock:
                view = np.ndarray(
                    self._frame_shape,
                    dtype=np.uint8,
                    buffer=self._shm.buf,
                    offset=offset,
                )
                image = view.copy()
            if self.rotate_180:
                image = image[::-1, ::-1]
            image = np.ascontiguousarray(self._transform(image))
            frame = {
                "color": image,
                "timestamp": timestamp,
                "camera_capture_timestamp": timestamp,
                "camera_receive_timestamp": time.time(),
                "camera_frame_id": frame_id,
                "step_idx": frame_id,
            }
            with self._lock:
                self._history.append(frame)
                vis_view = np.ndarray(
                    self._vis_shape,
                    dtype=np.uint8,
                    buffer=self._vis_shm.buf,
                )
                vis_view[...] = image
            self._last_frame_id = frame_id
            self._first_frame.set()

    def _read_stderr(self):
        assert self._proc is not None
        assert self._proc.stderr is not None
        for raw_line in self._proc.stderr:
            line = raw_line.strip()
            if line:
                self._stderr_lines.append(line)

    def _worker_failure_message(self) -> str:
        code = None if self._proc is None else self._proc.returncode
        detail = " | ".join(self._stderr_lines)
        if self._startup_error:
            detail = f"{self._startup_error} | {detail}" if detail else self._startup_error
        return f"L515 ego worker exited (returncode={code}). {detail}".strip()
