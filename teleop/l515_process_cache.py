from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, replace
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np


_MP_CTX = mp.get_context("spawn")
_SHM_SLOTS = 8
_HISTORY_SIZE = 64


@dataclass(frozen=True)
class L515ProcessConfig:
    name: str
    serial: str
    rgb_width: int = 960
    rgb_height: int = 540
    depth_width: int = 640
    depth_height: int = 480
    rgb_fps: int = 60
    depth_fps: int = 30
    align_to: str = "color"
    timeout_ms: int = 5000
    warmup_frames: int = 90
    strict_global_time: bool = True
    auto_exposure: bool = True
    auto_exposure_priority: float = 0.0
    auto_white_balance: bool = True
    startup_timeout_s: float = 30.0

    @property
    def depth_output_width(self) -> int:
        if self.align_to == "color":
            return int(self.rgb_width)
        return int(self.depth_width)

    @property
    def depth_output_height(self) -> int:
        if self.align_to == "color":
            return int(self.rgb_height)
        return int(self.depth_height)


@dataclass
class L515ProcessFrame:
    color_rgb: np.ndarray | None
    depth_z16: np.ndarray | None
    color_frame_number: int | None
    depth_frame_number: int | None
    color_timestamp_ms: float | None
    depth_timestamp_ms: float | None
    color_timestamp_domain: str | None
    depth_timestamp_domain: str | None
    color_capture_ns: int | None
    depth_capture_ns: int | None
    color_intrinsics: dict[str, Any] | None
    depth_intrinsics: dict[str, Any] | None
    color_metadata: dict[str, float]
    depth_metadata: dict[str, float]
    depth_scale_m_per_unit: float
    sync_target_ns: int | None = None
    color_sync_delta_ns: int | None = None
    depth_sync_delta_ns: int | None = None
    selection_mode: str | None = None


def _empty_frame(depth_scale: float = math.nan) -> L515ProcessFrame:
    return L515ProcessFrame(
        color_rgb=None,
        depth_z16=None,
        color_frame_number=None,
        depth_frame_number=None,
        color_timestamp_ms=None,
        depth_timestamp_ms=None,
        color_timestamp_domain=None,
        depth_timestamp_domain=None,
        color_capture_ns=None,
        depth_capture_ns=None,
        color_intrinsics=None,
        depth_intrinsics=None,
        color_metadata={},
        depth_metadata={},
        depth_scale_m_per_unit=float(depth_scale),
    )


class L515ProcessCache:
    """Spawn-isolated L515 latest-frame cache.

    The parent process never imports pyrealsense2. Each L515 runs in a fresh
    Python process and writes color/depth images into double-buffered shared
    memory. Only small metadata dictionaries cross the multiprocessing queue.
    """

    def __init__(self, config: L515ProcessConfig):
        self.config = config
        self._color_shape = (
            int(config.rgb_height),
            int(config.rgb_width),
            3,
        )
        self._depth_shape = (
            int(config.depth_output_height),
            int(config.depth_output_width),
        )
        self._color_bytes = int(np.prod(self._color_shape)) * np.dtype(np.uint8).itemsize
        self._depth_bytes = int(np.prod(self._depth_shape)) * np.dtype(np.uint16).itemsize
        self._color_shm: SharedMemory | None = None
        self._depth_shm: SharedMemory | None = None
        self._lock = _MP_CTX.Lock()
        self._stop_event = _MP_CTX.Event()
        self._frame_queue = _MP_CTX.Queue(maxsize=16)
        self._startup_queue = _MP_CTX.Queue(maxsize=1)
        self._process: mp.process.BaseProcess | None = None
        self._latest_meta: dict[str, Any] | None = None
        self._startup_meta: dict[str, Any] | None = None
        self._history: deque[L515ProcessFrame] = deque(maxlen=_HISTORY_SIZE)
        self._frames_read = 0
        self._started = False

    @property
    def is_connected(self) -> bool:
        return (
            self._started
            and self._process is not None
            and self._process.is_alive()
        )

    def start(self) -> None:
        if self._started:
            return
        self._color_shm = SharedMemory(create=True, size=self._color_bytes * _SHM_SLOTS)
        self._depth_shm = SharedMemory(create=True, size=self._depth_bytes * _SHM_SLOTS)
        self._stop_event.clear()
        self._process = _MP_CTX.Process(
            target=_l515_worker,
            name=f"{self.config.name}_l515_worker",
            args=(
                asdict(self.config),
                self._color_shm.name,
                self._depth_shm.name,
                self._color_shape,
                self._depth_shape,
                self._color_bytes,
                self._depth_bytes,
                self._lock,
                self._stop_event,
                self._frame_queue,
                self._startup_queue,
            ),
            daemon=True,
        )
        self._process.start()
        try:
            status, detail = self._startup_queue.get(
                timeout=float(self.config.startup_timeout_s)
            )
        except queue.Empty as exc:
            self.stop()
            raise RuntimeError(
                f"L515 {self.config.name} worker startup timed out"
            ) from exc
        if status != "ok":
            self.stop()
            raise RuntimeError(f"L515 {self.config.name} worker failed: {detail}")
        self._startup_meta = dict(detail)
        latest_meta = self._startup_meta.get("latestMeta")
        if isinstance(latest_meta, dict):
            self._latest_meta = dict(latest_meta)
            self._frames_read = int(self._latest_meta.get("framesRead", 0))
            self._append_history_from_meta(self._latest_meta)
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=max(1.0, float(self.config.timeout_ms) / 1000.0))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._process = None
        self._started = False
        for shm in (self._color_shm, self._depth_shm):
            if shm is None:
                continue
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self._color_shm = None
        self._depth_shm = None

    def latest(self, copy_images: bool = True) -> L515ProcessFrame:
        self._drain_queue()
        if not self._history:
            depth_scale = (
                math.nan
                if self._startup_meta is None
                else float(self._startup_meta.get("depthScaleMPerUnit", math.nan))
            )
            return _empty_frame(depth_scale)
        return self._clone_frame(self._history[-1], copy_images=copy_images)

    def nearest(
        self,
        target_ns: int,
        copy_images: bool = True,
        max_delta_ns: int | None = None,
    ) -> L515ProcessFrame:
        self._drain_queue()
        if not self._history:
            depth_scale = (
                math.nan
                if self._startup_meta is None
                else float(self._startup_meta.get("depthScaleMPerUnit", math.nan))
            )
            return _empty_frame(depth_scale)
        target = int(target_ns)
        candidates = [
            frame
            for frame in self._history
            if self._frame_capture_ns(frame) is not None
        ]
        if not candidates:
            return self._clone_frame(self._history[-1], copy_images=copy_images)
        selected = min(
            candidates,
            key=lambda frame: abs(int(self._frame_capture_ns(frame)) - target),
        )
        selected_ns = self._frame_capture_ns(selected)
        if (
            max_delta_ns is not None
            and selected_ns is not None
            and abs(int(selected_ns) - target) > int(max_delta_ns)
        ):
            return _empty_frame(selected.depth_scale_m_per_unit)
        result = self._clone_frame(selected, copy_images=copy_images)
        color_delta = (
            None
            if result.color_capture_ns is None
            else int(result.color_capture_ns) - target
        )
        depth_delta = (
            None
            if result.depth_capture_ns is None
            else int(result.depth_capture_ns) - target
        )
        return replace(
            result,
            sync_target_ns=target,
            color_sync_delta_ns=color_delta,
            depth_sync_delta_ns=depth_delta,
            selection_mode="nearest_capture_ns",
        )

    def _frame_from_meta(
        self,
        meta: dict[str, Any],
        copy_images: bool,
    ) -> L515ProcessFrame:
        color_rgb = self._copy_slot(
            self._color_shm,
            self._color_shape,
            np.uint8,
            self._color_bytes,
            int(meta.get("colorSlot", -1)),
            copy_images,
        )
        depth_z16 = self._copy_slot(
            self._depth_shm,
            self._depth_shape,
            np.uint16,
            self._depth_bytes,
            int(meta.get("depthSlot", -1)),
            copy_images,
        )
        return L515ProcessFrame(
            color_rgb=color_rgb,
            depth_z16=depth_z16,
            color_frame_number=meta.get("colorFrameId"),
            depth_frame_number=meta.get("depthFrameId"),
            color_timestamp_ms=meta.get("colorTimestampMs"),
            depth_timestamp_ms=meta.get("depthTimestampMs"),
            color_timestamp_domain=meta.get("colorTimestampDomain"),
            depth_timestamp_domain=meta.get("depthTimestampDomain"),
            color_capture_ns=meta.get("colorCaptureNs"),
            depth_capture_ns=meta.get("depthCaptureNs"),
            color_intrinsics=meta.get("colorIntrinsics"),
            depth_intrinsics=meta.get("depthIntrinsics"),
            color_metadata=dict(meta.get("colorMetadata") or {}),
            depth_metadata=dict(meta.get("depthMetadata") or {}),
            depth_scale_m_per_unit=float(meta.get("depthScaleMPerUnit", math.nan)),
        )

    def _clone_frame(
        self,
        frame: L515ProcessFrame,
        copy_images: bool,
    ) -> L515ProcessFrame:
        color_rgb = frame.color_rgb
        depth_z16 = frame.depth_z16
        if copy_images:
            color_rgb = None if color_rgb is None else color_rgb.copy()
            depth_z16 = None if depth_z16 is None else depth_z16.copy()
        return replace(frame, color_rgb=color_rgb, depth_z16=depth_z16)

    def _frame_capture_ns(self, frame: L515ProcessFrame) -> int | None:
        values = [
            int(value)
            for value in (frame.color_capture_ns, frame.depth_capture_ns)
            if value is not None
        ]
        if not values:
            return None
        return int(round(sum(values) / len(values)))

    def _append_history_from_meta(self, meta: dict[str, Any]) -> None:
        frame = self._frame_from_meta(meta, copy_images=True)
        frame_key = (frame.color_frame_number, frame.depth_frame_number)
        if self._history:
            last = self._history[-1]
            if frame_key == (last.color_frame_number, last.depth_frame_number):
                self._history[-1] = frame
                return
        self._history.append(frame)

    def metadata(self) -> dict[str, Any]:
        self._drain_queue()
        latest = self.latest(copy_images=False)
        startup = self._startup_meta or {}
        return {
            "name": self.config.name,
            "serial": self.config.serial,
            "backend": "spawn_shared_memory",
            "rgbWidth": int(self.config.rgb_width),
            "rgbHeight": int(self.config.rgb_height),
            "depthWidth": int(self.config.depth_width),
            "depthHeight": int(self.config.depth_height),
            "depthOutputWidth": int(self.config.depth_output_width),
            "depthOutputHeight": int(self.config.depth_output_height),
            "rgbFps": int(self.config.rgb_fps),
            "depthFps": int(self.config.depth_fps),
            "alignTo": self.config.align_to,
            "strictGlobalTime": bool(self.config.strict_global_time),
            "autoExposure": bool(self.config.auto_exposure),
            "autoExposurePriority": float(self.config.auto_exposure_priority),
            "autoWhiteBalance": bool(self.config.auto_white_balance),
            "depthScaleMPerUnit": float(latest.depth_scale_m_per_unit),
            "colorIntrinsics": latest.color_intrinsics or startup.get("colorIntrinsics"),
            "depthIntrinsics": latest.depth_intrinsics or startup.get("depthIntrinsics"),
            "lastColorTimestampDomain": latest.color_timestamp_domain,
            "lastDepthTimestampDomain": latest.depth_timestamp_domain,
            "framesRead": int(self._frames_read),
            "isConnected": bool(self.is_connected),
        }

    def _drain_queue(self) -> None:
        while True:
            try:
                meta = self._frame_queue.get_nowait()
            except queue.Empty:
                return
            self._latest_meta = dict(meta)
            self._frames_read = int(self._latest_meta.get("framesRead", self._frames_read))
            self._append_history_from_meta(self._latest_meta)

    def _copy_slot(
        self,
        shm: SharedMemory | None,
        shape: tuple[int, ...],
        dtype,
        slot_bytes: int,
        slot: int,
        copy_images: bool,
    ) -> np.ndarray | None:
        if shm is None or slot < 0:
            return None
        offset = int(slot) * int(slot_bytes)
        with self._lock:
            view = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=offset)
            if copy_images:
                return view.copy()
            return view


def _queue_latest(q, item: dict[str, Any]) -> None:
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def _l515_worker(
    config_dict: dict[str, Any],
    color_shm_name: str,
    depth_shm_name: str,
    color_shape: tuple[int, int, int],
    depth_shape: tuple[int, int],
    color_bytes: int,
    depth_bytes: int,
    lock,
    stop_event,
    frame_queue,
    startup_queue,
) -> None:
    color_shm = None
    depth_shm = None
    pipeline = None
    try:
        import pyrealsense2 as rs

        cfg = L515ProcessConfig(**config_dict)
        color_shm = SharedMemory(name=color_shm_name)
        depth_shm = SharedMemory(name=depth_shm_name)
        pipeline = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(str(cfg.serial))
        rs_cfg.enable_stream(
            rs.stream.color,
            int(cfg.rgb_width),
            int(cfg.rgb_height),
            rs.format.rgb8,
            int(cfg.rgb_fps),
        )
        rs_cfg.enable_stream(
            rs.stream.depth,
            int(cfg.depth_width),
            int(cfg.depth_height),
            rs.format.z16,
            int(cfg.depth_fps),
        )
        resolved = rs_cfg.resolve(rs.pipeline_wrapper(pipeline))
        _configure_sensor_options(rs, resolved.get_device(), cfg)
        profile = pipeline.start(rs_cfg)
        device = profile.get_device()
        _configure_sensor_options(rs, device, cfg)
        depth_scale = float(device.first_depth_sensor().get_depth_scale())
        align = _make_align(rs, cfg.align_to)
        meta = {
            "colorSlot": -1,
            "depthSlot": -1,
            "colorFrameId": None,
            "depthFrameId": None,
            "colorTimestampMs": None,
            "depthTimestampMs": None,
            "colorTimestampDomain": None,
            "depthTimestampDomain": None,
            "colorCaptureNs": None,
            "depthCaptureNs": None,
            "colorIntrinsics": None,
            "depthIntrinsics": None,
            "colorMetadata": {},
            "depthMetadata": {},
            "depthScaleMPerUnit": depth_scale,
            "framesRead": 0,
        }
        ready_at = _warmup(
            rs,
            pipeline,
            align,
            cfg,
            meta,
            color_shm,
            depth_shm,
            color_shape,
            depth_shape,
            color_bytes,
            depth_bytes,
            lock,
        )
        if ready_at is None and cfg.strict_global_time:
            raise RuntimeError(
                "global_time not ready; "
                f"last domains={(meta.get('colorTimestampDomain'), meta.get('depthTimestampDomain'))}"
            )
        startup_queue.put(
            (
                "ok",
                {
                    "depthScaleMPerUnit": depth_scale,
                    "colorIntrinsics": meta.get("colorIntrinsics"),
                    "depthIntrinsics": meta.get("depthIntrinsics"),
                    "globalTimeReadyFrame": ready_at,
                    "latestMeta": dict(meta),
                },
            )
        )
        color_slot = int(meta.get("colorSlot", -1))
        depth_slot = int(meta.get("depthSlot", -1))
        last_color_frame = meta.get("colorFrameId")
        last_depth_frame = meta.get("depthFrameId")
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(int(cfg.timeout_ms))
            host_ns = time.monotonic_ns()
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            changed = False
            if color_frame:
                frame_id = int(color_frame.get_frame_number())
                if frame_id != last_color_frame:
                    color_slot = (color_slot + 1) % _SHM_SLOTS
                    color_arr = np.asanyarray(color_frame.get_data())
                    _write_slot(
                        color_shm,
                        color_shape,
                        np.uint8,
                        color_bytes,
                        color_slot,
                        color_arr,
                        lock,
                    )
                    last_color_frame = frame_id
                    meta.update(
                        {
                            "colorSlot": color_slot,
                            "colorFrameId": frame_id,
                            "colorTimestampMs": float(color_frame.get_timestamp()),
                            "colorTimestampDomain": _timestamp_domain_name(
                                rs,
                                color_frame.get_frame_timestamp_domain(),
                            ),
                            "colorCaptureNs": int(host_ns),
                            "colorIntrinsics": _intrinsics_from_frame(rs, color_frame),
                            "colorMetadata": _frame_metadata(rs, color_frame),
                        }
                    )
                    changed = True
            if depth_frame:
                frame_id = int(depth_frame.get_frame_number())
                if frame_id != last_depth_frame:
                    depth_slot = (depth_slot + 1) % _SHM_SLOTS
                    depth_arr = np.asanyarray(depth_frame.get_data())
                    _write_slot(
                        depth_shm,
                        depth_shape,
                        np.uint16,
                        depth_bytes,
                        depth_slot,
                        depth_arr,
                        lock,
                    )
                    last_depth_frame = frame_id
                    meta.update(
                        {
                            "depthSlot": depth_slot,
                            "depthFrameId": frame_id,
                            "depthTimestampMs": float(depth_frame.get_timestamp()),
                            "depthTimestampDomain": _timestamp_domain_name(
                                rs,
                                depth_frame.get_frame_timestamp_domain(),
                            ),
                            "depthCaptureNs": int(host_ns),
                            "depthIntrinsics": _intrinsics_from_frame(rs, depth_frame),
                            "depthMetadata": _frame_metadata(rs, depth_frame),
                        }
                    )
                    changed = True
            if changed:
                meta["framesRead"] = int(meta["framesRead"]) + 1
                _queue_latest(frame_queue, dict(meta))
    except Exception:
        detail = traceback.format_exc()
        try:
            startup_queue.put_nowait(("error", detail))
        except Exception:
            pass
    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        for shm in (color_shm, depth_shm):
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass


def _make_align(rs, align_to: str):
    if align_to == "color":
        return rs.align(rs.stream.color)
    if align_to == "depth":
        return rs.align(rs.stream.depth)
    if align_to == "none":
        return None
    raise ValueError(f"Unsupported align_to: {align_to}")


def _set_option_if_supported(rs, sensor, option, value: float) -> None:
    if sensor.supports(option):
        sensor.set_option(option, float(value))


def _configure_sensor_options(rs, device, cfg: L515ProcessConfig) -> None:
    for sensor in device.query_sensors():
        _set_option_if_supported(rs, sensor, rs.option.global_time_enabled, 1.0)
        if cfg.auto_exposure:
            _set_option_if_supported(rs, sensor, rs.option.enable_auto_exposure, 1.0)
            if hasattr(rs.option, "auto_exposure_priority"):
                _set_option_if_supported(
                    rs,
                    sensor,
                    rs.option.auto_exposure_priority,
                    float(cfg.auto_exposure_priority),
                )
        if cfg.auto_white_balance:
            _set_option_if_supported(rs, sensor, rs.option.enable_auto_white_balance, 1.0)


def _warmup(
    rs,
    pipeline,
    align,
    cfg: L515ProcessConfig,
    meta: dict[str, Any],
    color_shm: SharedMemory,
    depth_shm: SharedMemory,
    color_shape: tuple[int, int, int],
    depth_shape: tuple[int, int],
    color_bytes: int,
    depth_bytes: int,
    lock,
) -> int | None:
    ready_at = None
    color_slot = -1
    depth_slot = -1
    last_color_frame = None
    last_depth_frame = None
    for idx in range(max(0, int(cfg.warmup_frames))):
        frames = pipeline.wait_for_frames(int(cfg.timeout_ms))
        host_ns = time.monotonic_ns()
        if align is not None:
            frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame:
            frame_id = int(color_frame.get_frame_number())
            if frame_id != last_color_frame:
                color_slot = (color_slot + 1) % _SHM_SLOTS
                _write_slot(
                    color_shm,
                    color_shape,
                    np.uint8,
                    color_bytes,
                    color_slot,
                    np.asanyarray(color_frame.get_data()),
                    lock,
                )
                last_color_frame = frame_id
                meta.update(
                    {
                        "colorSlot": color_slot,
                        "colorFrameId": frame_id,
                        "colorTimestampMs": float(color_frame.get_timestamp()),
                        "colorTimestampDomain": _timestamp_domain_name(
                            rs,
                            color_frame.get_frame_timestamp_domain(),
                        ),
                        "colorCaptureNs": int(host_ns),
                        "colorIntrinsics": _intrinsics_from_frame(rs, color_frame),
                        "colorMetadata": _frame_metadata(rs, color_frame),
                    }
                )
        if depth_frame:
            frame_id = int(depth_frame.get_frame_number())
            if frame_id != last_depth_frame:
                depth_slot = (depth_slot + 1) % _SHM_SLOTS
                _write_slot(
                    depth_shm,
                    depth_shape,
                    np.uint16,
                    depth_bytes,
                    depth_slot,
                    np.asanyarray(depth_frame.get_data()),
                    lock,
                )
                last_depth_frame = frame_id
                meta.update(
                    {
                        "depthSlot": depth_slot,
                        "depthFrameId": frame_id,
                        "depthTimestampMs": float(depth_frame.get_timestamp()),
                        "depthTimestampDomain": _timestamp_domain_name(
                            rs,
                            depth_frame.get_frame_timestamp_domain(),
                        ),
                        "depthCaptureNs": int(host_ns),
                        "depthIntrinsics": _intrinsics_from_frame(rs, depth_frame),
                        "depthMetadata": _frame_metadata(rs, depth_frame),
                    }
                )
        if (
            meta.get("colorTimestampDomain") == "global_time"
            and meta.get("depthTimestampDomain") == "global_time"
            and ready_at is None
        ):
            ready_at = idx + 1
    return ready_at


def _write_slot(
    shm: SharedMemory,
    shape: tuple[int, ...],
    dtype,
    slot_bytes: int,
    slot: int,
    src: np.ndarray,
    lock,
) -> None:
    arr = np.asarray(src)
    expected_shape = tuple(shape)
    if arr.shape != expected_shape:
        raise RuntimeError(f"unexpected frame shape {arr.shape}, expected {expected_shape}")
    offset = int(slot) * int(slot_bytes)
    with lock:
        dst = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=offset)
        np.copyto(dst, arr, casting="safe")


def _timestamp_domain_name(rs, domain) -> str:
    for name in ("hardware_clock", "system_time", "global_time"):
        if hasattr(rs.timestamp_domain, name) and domain == getattr(rs.timestamp_domain, name):
            return name
    return str(domain)


def _distortion_model_name(rs, model) -> str:
    for name in (
        "none",
        "modified_brown_conrady",
        "inverse_brown_conrady",
        "brown_conrady",
        "kannala_brandt4",
    ):
        if hasattr(rs.distortion, name) and model == getattr(rs.distortion, name):
            return name
    return str(model)


def _intrinsics_from_frame(rs, frame) -> dict[str, Any]:
    intr = frame.profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": _distortion_model_name(rs, intr.model),
        "coeffs": [float(v) for v in intr.coeffs],
    }


def _frame_metadata(rs, frame) -> dict[str, float]:
    result: dict[str, float] = {}
    names = (
        "frame_counter",
        "frame_timestamp",
        "sensor_timestamp",
        "backend_timestamp",
        "time_of_arrival",
        "actual_exposure",
        "gain_level",
    )
    for name in names:
        if not hasattr(rs.frame_metadata_value, name):
            continue
        key = getattr(rs.frame_metadata_value, name)
        try:
            if frame.supports_frame_metadata(key):
                result[name] = float(frame.get_frame_metadata(key))
        except Exception:
            continue
    return result
