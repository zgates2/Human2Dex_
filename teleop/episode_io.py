#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared episode I/O, message building, and terminal helpers."""

from __future__ import annotations

import pickle
import queue
import re
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mvs_cpp import RGBCachedFrame


URDF_UPPER: dict[str, float] = {
    "thumb_cmc_pitch": 0.58,
    "thumb_cmc_yaw": 1.36,
    "index_mcp_pitch": 1.60,
    "middle_mcp_pitch": 1.60,
    "ring_mcp_pitch": 1.60,
    "pinky_mcp_pitch": 1.60,
}

LINKER_JOINTS: list[str] = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

WUJI_JOINTS: list[str] = [
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]


def angles_rad_to_cmd(angles_rad: np.ndarray) -> np.ndarray:
    """Convert six Linker O6 radians into the current uint8 command space."""
    cmds = np.zeros(len(LINKER_JOINTS), dtype=np.uint8)
    for i, (name, q) in enumerate(zip(LINKER_JOINTS, angles_rad)):
        upper = URDF_UPPER[name]
        q_c = float(np.clip(q, 0.0, upper))
        cmds[i] = int(np.clip(round(250.0 * (1.0 - q_c / upper)), 0, 255))
    return cmds


def rgb_image_path_for_frame(pkl_path: Path, frame_idx: int) -> Path:
    return pkl_path.parent / "images" / f"{pkl_path.stem}_rgb_{frame_idx:06d}.jpg"


def wrist_rgb_image_path_for_frame(pkl_path: Path, frame_idx: int) -> Path:
    return pkl_path.parent / "wrist" / f"{pkl_path.stem}_rgb_{frame_idx:06d}.jpg"


def pico_first_view_image_path_for_frame(pkl_path: Path, frame_idx: int) -> Path:
    return (
        pkl_path.parent
        / "pico_first_view"
        / f"{pkl_path.stem}_first_view_{frame_idx:06d}.jpg"
    )


def rgb_image_relpath(image_path: Path) -> str:
    return str(Path(image_path.parent.name) / image_path.name)


def bundle_relpath(pkl_path: Path, path: Path) -> str:
    return str(path.relative_to(pkl_path.parent))


def safe_camera_dir_name(camera_name: str) -> str:
    name = str(camera_name).strip().replace("/", "_").replace("\\", "_")
    if not name or name in {".", ".."}:
        raise ValueError("camera name 不能为空，也不能是 . 或 ..")
    return name


def l515_rgb_image_path_for_frame(
    pkl_path: Path,
    camera_name: str,
    frame_idx: int,
) -> Path:
    cam = safe_camera_dir_name(camera_name)
    return pkl_path.parent / "l515" / cam / "rgb" / f"{pkl_path.stem}_{cam}_rgb_{frame_idx:06d}.jpg"


def l515_depth_image_path_for_frame(
    pkl_path: Path,
    camera_name: str,
    depth_frame_number: int,
) -> Path:
    cam = safe_camera_dir_name(camera_name)
    return pkl_path.parent / "l515" / cam / "depth" / f"{pkl_path.stem}_{cam}_depth_{depth_frame_number:06d}.png"


def write_rgb_jpg(image: np.ndarray, path: Path, quality: int = 95) -> None:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"RGB 图像期望 shape=(H,W,3)，收到 {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # noqa: WPS433

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(
            str(path),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise IOError(f"cv2.imwrite 返回失败: {path}")
    except ImportError:
        from PIL import Image  # noqa: WPS433

        Image.fromarray(arr).save(path, format="JPEG", quality=int(quality))


def write_depth_png(depth_z16: np.ndarray, path: Path) -> None:
    arr = np.asarray(depth_z16)
    if arr.ndim != 2:
        raise ValueError(f"Depth 图像期望 shape=(H,W)，收到 {arr.shape}")
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    else:
        arr = arr.copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # noqa: WPS433

        ok = cv2.imwrite(str(path), arr)
        if not ok:
            raise IOError(f"cv2.imwrite 返回失败: {path}")
    except ImportError:
        from PIL import Image  # noqa: WPS433

        Image.fromarray(arr, mode="I;16").save(path, format="PNG")


@dataclass
class _JpgWriteTask:
    image: np.ndarray
    path: Path


@dataclass
class _DepthPngWriteTask:
    depth_z16: np.ndarray
    path: Path


class AsyncJpgWriter:
    """Background jpg writer so image encoding does not block sampling."""

    def __init__(
        self,
        quality: int = 95,
        max_queue: int = 512,
        num_workers: int = 2,
    ):
        self.quality = int(quality)
        self.max_queue = max(1, int(max_queue))
        self.num_workers = max(1, int(num_workers))
        self._queue: queue.Queue[_JpgWriteTask | None] = queue.Queue(
            maxsize=self.max_queue
        )
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"AsyncJpgWriter-{idx}",
                daemon=True,
            )
            for idx in range(self.num_workers)
        ]
        self._lock = threading.Lock()
        self._started = False
        self._stopped = False
        self.queued = 0
        self.written = 0
        self.blocked_puts = 0
        self.blocked_put_ns = 0
        self.errors: list[tuple[str, str]] = []

    def start(self) -> None:
        if self._stopped:
            raise RuntimeError("AsyncJpgWriter 已 stop，不能再次 start")
        if not self._started:
            self._started = True
            for thread in self._threads:
                thread.start()

    def wait_empty(self) -> None:
        if self._started:
            self._queue.join()

    def enqueue(self, image: np.ndarray, path: Path) -> None:
        if not self._started:
            raise RuntimeError("AsyncJpgWriter 尚未 start")

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.ascontiguousarray(arr)

        task = _JpgWriteTask(image=arr, path=path)
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            t0 = time.monotonic_ns()
            self._queue.put(task)
            blocked_ns = time.monotonic_ns() - t0
            with self._lock:
                self.blocked_puts += 1
                self.blocked_put_ns += blocked_ns
        with self._lock:
            self.queued += 1

    def stop(self, timeout_s: float = 10.0) -> bool:
        if not self._started:
            return True
        if self._stopped:
            return True
        self._queue.join()
        for _ in self._threads:
            self._queue.put(None)

        deadline = time.monotonic() + max(float(timeout_s), 0.1)
        for thread in self._threads:
            remaining = max(deadline - time.monotonic(), 0.1)
            thread.join(timeout=remaining)

        alive = sum(1 for thread in self._threads if thread.is_alive())
        if alive:
            print(f"[Warn] {alive} 个 RGB jpg 写入线程未能在超时内退出")
            return False
        self._stopped = True
        return True

    def pending(self) -> int:
        return int(self._queue.qsize())

    def alive_workers(self) -> int:
        return sum(1 for thread in self._threads if thread.is_alive())

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                write_rgb_jpg(task.image, task.path, quality=self.quality)
                with self._lock:
                    self.written += 1
            except Exception as exc:
                path = "" if task is None else str(task.path)
                with self._lock:
                    self.errors.append((path, repr(exc)))
            finally:
                self._queue.task_done()


class AsyncDepthPngWriter:
    """Background depth png writer for RealSense z16 frames."""

    def __init__(self, max_queue: int = 256):
        self._queue: queue.Queue[_DepthPngWriteTask | None] = queue.Queue(
            maxsize=max(1, int(max_queue))
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self.queued = 0
        self.written = 0
        self.blocked_puts = 0
        self.blocked_put_ns = 0
        self.errors: list[tuple[str, str]] = []

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def wait_empty(self) -> None:
        if self._started:
            self._queue.join()

    def enqueue(self, depth_z16: np.ndarray, path: Path) -> None:
        if not self._started:
            raise RuntimeError("AsyncDepthPngWriter 尚未 start")
        arr = np.asarray(depth_z16)
        if arr.dtype != np.uint16:
            arr = np.clip(arr, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        else:
            arr = arr.copy()
        task = _DepthPngWriteTask(depth_z16=arr, path=path)
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            t0 = time.monotonic_ns()
            self._queue.put(task)
            self.blocked_puts += 1
            self.blocked_put_ns += time.monotonic_ns() - t0
        self.queued += 1

    def stop(self, timeout_s: float = 10.0) -> bool:
        if not self._started:
            return True
        self._queue.join()
        self._queue.put(None)
        self._thread.join(timeout=max(float(timeout_s), 0.1))
        if self._thread.is_alive():
            print("[Warn] Depth png 写入线程未能在超时内退出")
            return False
        return True

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                write_depth_png(task.depth_z16, task.path)
                self.written += 1
            except Exception as exc:
                path = "" if task is None else str(task.path)
                self.errors.append((path, repr(exc)))
            finally:
                self._queue.task_done()


@dataclass
class L515EpisodeSaveState:
    last_depth_frame_number: dict[str, int]
    last_depth_relpath: dict[str, str]


def writer_stats(writer) -> dict | None:
    if writer is None:
        return None
    return {
        "async": True,
        "quality": int(getattr(writer, "quality", 95)),
        "workers": int(getattr(writer, "num_workers", 1)),
        "aliveWorkers": int(
            writer.alive_workers() if hasattr(writer, "alive_workers") else 0
        ),
        "maxQueue": int(getattr(writer, "max_queue", 0)),
        "pending": int(writer.pending() if hasattr(writer, "pending") else 0),
        "queued": int(writer.queued),
        "written": int(writer.written),
        "blockedPuts": int(writer.blocked_puts),
        "blockedPutMs": float(writer.blocked_put_ns / 1e6),
        "errorCount": len(writer.errors),
    }


def writer_errors(writer) -> list[dict]:
    if writer is None or not writer.errors:
        return []
    return [
        {"path": path, "error": error}
        for path, error in writer.errors[:20]
    ]


def rgb_fields(
    rgb_frame: RGBCachedFrame | None,
    sample_ns: int,
    rgb_image_path: Path | None = None,
    jpg_writer: AsyncJpgWriter | None = None,
    source_receive_ns: int | None = None,
    rgb_missing_reason: str | None = None,
) -> dict:
    if (
        rgb_frame is None
        or rgb_frame.capture_ns is None
        or rgb_frame.image is None
    ):
        return {
            "rgbImage": None,
            "rgbFrameId": None,
            "rgbCaptureNs": None,
            "rgbDeviceTimestampTicks": None,
            "rgbHostTimestampMs": None,
            "rgbSyncTargetNs": None,
            "rgbSyncDeltaNs": None,
            "rgbSelectionMode": None,
            "rgbAlignResidualNs": None,
            "rgbToPicoReceiveDeltaNs": None,
            "absRgbToPicoReceiveDeltaNs": None,
            "rgbMissingReason": rgb_missing_reason,
            "rgbQualityWarning": None,
            "rgbFrameFilled": False,
            "rgbFrameFillReason": None,
            "rgbFrameRepeated": None,
            "rgbFrameGap": None,
            "rgbFramesRead": None,
        }

    image_name = None
    if rgb_image_path is not None:
        if jpg_writer is None:
            write_rgb_jpg(rgb_frame.image, rgb_image_path)
        else:
            jpg_writer.enqueue(rgb_frame.image, rgb_image_path)
        image_name = rgb_image_relpath(rgb_image_path)

    return {
        "rgbImage": image_name,
        "rgbFrameId": rgb_frame.frame_id,
        "rgbCaptureNs": int(rgb_frame.capture_ns),
        "rgbDeviceTimestampTicks": (
            None
            if getattr(rgb_frame, "device_timestamp_ticks", None) is None
            else int(rgb_frame.device_timestamp_ticks)
        ),
        "rgbHostTimestampMs": (
            None
            if getattr(rgb_frame, "host_timestamp_ms", None) is None
            else int(rgb_frame.host_timestamp_ms)
        ),
        "rgbSyncTargetNs": (
            None
            if getattr(rgb_frame, "sync_target_ns", None) is None
            else int(rgb_frame.sync_target_ns)
        ),
        "rgbSyncDeltaNs": (
            None
            if getattr(rgb_frame, "sync_delta_ns", None) is None
            else int(rgb_frame.sync_delta_ns)
        ),
        "rgbSelectionMode": getattr(rgb_frame, "selection_mode", None),
        "rgbAlignResidualNs": int(sample_ns - rgb_frame.capture_ns),
        "rgbToPicoReceiveDeltaNs": (
            None
            if source_receive_ns is None or int(source_receive_ns) == 0
            else int(rgb_frame.capture_ns) - int(source_receive_ns)
        ),
        "absRgbToPicoReceiveDeltaNs": (
            None
            if source_receive_ns is None or int(source_receive_ns) == 0
            else abs(int(rgb_frame.capture_ns) - int(source_receive_ns))
        ),
        "rgbMissingReason": None,
        "rgbQualityWarning": rgb_missing_reason,
        "rgbFrameFilled": getattr(rgb_frame, "selection_mode", None)
        == "mvs_master_reused_previous",
        "rgbFrameFillReason": (
            rgb_missing_reason
            if getattr(rgb_frame, "selection_mode", None) == "mvs_master_reused_previous"
            else None
        ),
        "rgbFrameRepeated": getattr(rgb_frame, "is_repeated", None),
        "rgbFrameGap": getattr(rgb_frame, "frame_gap", None),
        "rgbFramesRead": getattr(rgb_frame, "frames_read", None),
    }


def pico_first_view_fields(
    *,
    frame,
    enabled: bool,
    sample_ns: int,
    image_path: Path | None = None,
    jpg_writer: AsyncJpgWriter | None = None,
    missing_reason: str | None = None,
) -> dict:
    if not enabled:
        return {}

    if (
        frame is None
        or getattr(frame, "capture_ns", None) is None
        or getattr(frame, "image", None) is None
    ):
        return {
            "picoFirstViewImage": None,
            "picoFirstViewFrameId": None,
            "picoFirstViewCaptureNs": None,
            "picoFirstViewAlignResidualNs": None,
            "picoFirstViewMissingReason": missing_reason,
        }

    image_name = None
    if image_path is not None:
        if jpg_writer is None:
            write_rgb_jpg(frame.image, image_path)
        else:
            jpg_writer.enqueue(frame.image, image_path)
        image_name = rgb_image_relpath(image_path)

    return {
        "picoFirstViewImage": image_name,
        "picoFirstViewFrameId": frame.frame_id,
        "picoFirstViewCaptureNs": int(frame.capture_ns),
        "picoFirstViewAlignResidualNs": int(sample_ns - frame.capture_ns),
        "picoFirstViewMissingReason": missing_reason,
    }


def sample_quality_fields(quality_payload: dict | None) -> dict:
    payload = {} if quality_payload is None else dict(quality_payload)
    flags = payload.get("flags", [])
    if flags is None:
        flags = []
    return {
        "sampleQuality": payload,
        "qualityFlags": list(flags),
    }


def pico_quality_fields(
    raw26x7: np.ndarray | None,
    pico_active: int | None,
    valid_mask: np.ndarray | list[bool] | None,
) -> dict:
    raw = None if raw26x7 is None else np.asarray(raw26x7, dtype=np.float32)
    if raw is not None and raw.size == 26 * 7:
        raw = raw.reshape(26, 7)
    return {
        "raw26x7": raw,
        "picoActive": None if pico_active is None else int(pico_active),
        "valid_mask": (
            None
            if valid_mask is None
            else np.asarray(valid_mask, dtype=bool).reshape(21)
        ),
    }


def timestamp_ns_from_ms(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(float(value) * 1_000_000.0))


def align_residual(sample_ns: int, capture_ns: int | None) -> int | None:
    if capture_ns is None:
        return None
    return int(sample_ns - int(capture_ns))


def empty_l515_camera_fields() -> dict:
    return {
        "rgbImage": None,
        "depthImage": None,
        "colorFrameId": None,
        "depthFrameId": None,
        "colorTimestampNs": None,
        "depthTimestampNs": None,
        "colorTimestampDomain": None,
        "depthTimestampDomain": None,
        "colorCaptureNs": None,
        "depthCaptureNs": None,
        "syncTargetNs": None,
        "colorSyncDeltaNs": None,
        "depthSyncDeltaNs": None,
        "selectionMode": None,
        "colorAlignResidualNs": None,
        "depthAlignResidualNs": None,
        "depthIsNewFrame": False,
        "depthScaleMPerUnit": None,
        "colorMetadata": {},
        "depthMetadata": {},
    }


def l515_camera_fields(
    *,
    camera_name: str,
    frame,
    sample_ns: int,
    pkl_path: Path | None,
    frame_idx: int,
    jpg_writer: AsyncJpgWriter | None,
    depth_writer: AsyncDepthPngWriter | None,
    save_state: L515EpisodeSaveState,
) -> dict:
    if frame is None or (frame.color_rgb is None and frame.depth_z16 is None):
        return empty_l515_camera_fields()

    rgb_relpath = None
    if pkl_path is not None and frame.color_rgb is not None:
        rgb_path = l515_rgb_image_path_for_frame(pkl_path, camera_name, frame_idx)
        if jpg_writer is None:
            write_rgb_jpg(frame.color_rgb, rgb_path)
        else:
            jpg_writer.enqueue(frame.color_rgb, rgb_path)
        rgb_relpath = bundle_relpath(pkl_path, rgb_path)

    depth_relpath = None
    depth_is_new = False
    depth_frame_number = frame.depth_frame_number
    if depth_frame_number is not None:
        last_depth_number = save_state.last_depth_frame_number.get(camera_name)
        if last_depth_number == int(depth_frame_number):
            depth_relpath = save_state.last_depth_relpath.get(camera_name)
        elif pkl_path is not None and frame.depth_z16 is not None:
            depth_path = l515_depth_image_path_for_frame(
                pkl_path,
                camera_name,
                int(depth_frame_number),
            )
            if depth_writer is None:
                write_depth_png(frame.depth_z16, depth_path)
            else:
                depth_writer.enqueue(frame.depth_z16, depth_path)
            depth_relpath = bundle_relpath(pkl_path, depth_path)
            save_state.last_depth_frame_number[camera_name] = int(depth_frame_number)
            save_state.last_depth_relpath[camera_name] = depth_relpath
            depth_is_new = True
        else:
            depth_relpath = save_state.last_depth_relpath.get(camera_name)

    return {
        "rgbImage": rgb_relpath,
        "depthImage": depth_relpath,
        "colorFrameId": frame.color_frame_number,
        "depthFrameId": frame.depth_frame_number,
        "colorTimestampNs": timestamp_ns_from_ms(frame.color_timestamp_ms),
        "depthTimestampNs": timestamp_ns_from_ms(frame.depth_timestamp_ms),
        "colorTimestampDomain": frame.color_timestamp_domain,
        "depthTimestampDomain": frame.depth_timestamp_domain,
        "colorCaptureNs": frame.color_capture_ns,
        "depthCaptureNs": frame.depth_capture_ns,
        "syncTargetNs": (
            None
            if getattr(frame, "sync_target_ns", None) is None
            else int(frame.sync_target_ns)
        ),
        "colorSyncDeltaNs": (
            None
            if getattr(frame, "color_sync_delta_ns", None) is None
            else int(frame.color_sync_delta_ns)
        ),
        "depthSyncDeltaNs": (
            None
            if getattr(frame, "depth_sync_delta_ns", None) is None
            else int(frame.depth_sync_delta_ns)
        ),
        "selectionMode": getattr(frame, "selection_mode", None),
        "colorAlignResidualNs": align_residual(sample_ns, frame.color_capture_ns),
        "depthAlignResidualNs": align_residual(sample_ns, frame.depth_capture_ns),
        "depthIsNewFrame": bool(depth_is_new),
        "depthScaleMPerUnit": float(frame.depth_scale_m_per_unit),
        "colorMetadata": dict(frame.color_metadata),
        "depthMetadata": dict(frame.depth_metadata),
    }


def l515_fields(
    *,
    l515_frames: dict[str, object] | None,
    sample_ns: int,
    pkl_path: Path | None,
    frame_idx: int,
    jpg_writer: AsyncJpgWriter | None,
    depth_writer: AsyncDepthPngWriter | None,
    save_state: L515EpisodeSaveState,
) -> dict:
    if not l515_frames:
        return {}
    return {
        "l515": {
            name: l515_camera_fields(
                camera_name=name,
                frame=frame,
                sample_ns=sample_ns,
                pkl_path=pkl_path,
                frame_idx=frame_idx,
                jpg_writer=jpg_writer,
                depth_writer=depth_writer,
                save_state=save_state,
            )
            for name, frame in l515_frames.items()
        }
    }


def build_message(
    o6_angles: np.ndarray,
    wuji_qpos: np.ndarray,
    pts21_mano: np.ndarray,
    wrist_pose_6d: np.ndarray,
    sdk_ts_ns: int,
    sample_ns: int,
    source_receive_ns: int,
    rgb_frame: RGBCachedFrame | None = None,
    rgb_image_path: Path | None = None,
    jpg_writer: AsyncJpgWriter | None = None,
    l515_payload: dict | None = None,
    raw26x7: np.ndarray | None = None,
    pico_active: int | None = None,
    valid_mask: np.ndarray | list[bool] | None = None,
    rgb_missing_reason: str | None = None,
    quality_payload: dict | None = None,
    pico_first_view_frame=None,
    pico_first_view_enabled: bool = False,
    pico_first_view_image_path: Path | None = None,
    pico_first_view_missing_reason: str | None = None,
) -> dict:
    o6_cmd = angles_rad_to_cmd(o6_angles)
    wuji_cmd = np.asarray(wuji_qpos, dtype=np.float32).reshape(5, 4)
    trajectory_pose = np.asarray(wrist_pose_6d, dtype=np.float32).reshape(6)
    rgb = rgb_fields(
        rgb_frame=rgb_frame,
        sample_ns=sample_ns,
        source_receive_ns=source_receive_ns,
        rgb_missing_reason=rgb_missing_reason,
        rgb_image_path=rgb_image_path,
        jpg_writer=jpg_writer,
    )
    pico_quality = pico_quality_fields(raw26x7, pico_active, valid_mask)
    sample_quality = sample_quality_fields(quality_payload)
    pico_first_view = pico_first_view_fields(
        frame=pico_first_view_frame,
        enabled=pico_first_view_enabled,
        sample_ns=sample_ns,
        image_path=pico_first_view_image_path,
        jpg_writer=jpg_writer,
        missing_reason=pico_first_view_missing_reason,
    )

    return {
        "timestamp": time.time(),
        "mainClockMonotonicNs": sdk_ts_ns,
        "sampleClockNs": int(sample_ns),
        "sourceReceiveNs": int(source_receive_ns),
        "sourceAgeNs": int(sample_ns - source_receive_ns),
        "o6_command": o6_cmd,
        "wuji_command": wuji_cmd,
        "trajectoryPose": trajectory_pose,
        "pts21_mano": pts21_mano.astype(np.float32),
        **pico_quality,
        **rgb,
        **pico_first_view,
        **sample_quality,
        **(l515_payload or {}),
    }


def build_invalid_message(
    sdk_ts_ns: int,
    sample_ns: int,
    source_receive_ns: int,
    rgb_frame: RGBCachedFrame | None = None,
    rgb_image_path: Path | None = None,
    jpg_writer: AsyncJpgWriter | None = None,
    l515_payload: dict | None = None,
    raw26x7: np.ndarray | None = None,
    pico_active: int | None = None,
    valid_mask: np.ndarray | list[bool] | None = None,
    rgb_missing_reason: str | None = None,
    quality_payload: dict | None = None,
    pico_first_view_frame=None,
    pico_first_view_enabled: bool = False,
    pico_first_view_image_path: Path | None = None,
    pico_first_view_missing_reason: str | None = None,
) -> dict:
    rgb = rgb_fields(
        rgb_frame=rgb_frame,
        sample_ns=sample_ns,
        source_receive_ns=source_receive_ns,
        rgb_missing_reason=rgb_missing_reason,
        rgb_image_path=rgb_image_path,
        jpg_writer=jpg_writer,
    )
    pico_quality = pico_quality_fields(raw26x7, pico_active, valid_mask)
    sample_quality = sample_quality_fields(quality_payload)
    pico_first_view = pico_first_view_fields(
        frame=pico_first_view_frame,
        enabled=pico_first_view_enabled,
        sample_ns=sample_ns,
        image_path=pico_first_view_image_path,
        jpg_writer=jpg_writer,
        missing_reason=pico_first_view_missing_reason,
    )
    return {
        "timestamp": time.time(),
        "mainClockMonotonicNs": sdk_ts_ns,
        "sampleClockNs": int(sample_ns),
        "sourceReceiveNs": int(source_receive_ns),
        "sourceAgeNs": (
            None
            if source_receive_ns == 0
            else int(sample_ns - source_receive_ns)
        ),
        "o6_command": None,
        "wuji_command": None,
        "trajectoryPose": None,
        "pts21_mano": None,
        **pico_quality,
        **rgb,
        **pico_first_view,
        **sample_quality,
        **(l515_payload or {}),
    }


def save_pkl(messages: list[dict], path: Path, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "formatVersion": 4,
                "metadata": metadata or {},
                "messages": messages,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    total_frames = len(messages)
    size_kb = path.stat().st_size / 1024
    print(f"[Save] 已写入 {total_frames} 帧 → {path}  ({size_kb:.1f} KB)")


_EPISODE_NAME_RE = re.compile(r"^episode_(\d+)$")


def existing_episode_indices(task_dir: Path) -> set[int]:
    task_dir = Path(task_dir)
    if not task_dir.exists():
        return set()
    indices: set[int] = set()
    for child in task_dir.iterdir():
        name = child.stem if child.is_file() and child.suffix == ".pkl" else child.name
        match = _EPISODE_NAME_RE.fullmatch(name)
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def next_episode_index(task_dir: Path) -> int:
    used = existing_episode_indices(task_dir)
    episode_index = 1
    while episode_index in used:
        episode_index += 1
    return episode_index


def episode_pkl_path(task_dir: Path, episode_index: int | None = None) -> Path:
    # episode_index is accepted for backward-compatible call sites. Numbering is
    # derived from the task directory so separate program runs share one sequence.
    selected_index = next_episode_index(task_dir)
    episode_name = f"episode_{selected_index:04d}"
    return Path(task_dir) / episode_name / f"{episode_name}.pkl"


def episode_index_from_path(path: Path) -> int:
    for candidate in (Path(path).parent.name, Path(path).stem):
        match = _EPISODE_NAME_RE.fullmatch(candidate)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"无法从 episode 路径解析编号: {path}")


class KeypressWatcher:
    """Non-blocking single-key watcher for terminal controls."""

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.events: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._old_termios = None
        self._enabled = False

    def start(self) -> None:
        if not sys.stdin.isatty():
            print("[Warn] stdin 不是 TTY，交互按键不可用；请用 Ctrl+C 退出")
            return
        self._old_termios = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._enabled = True
        self._thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._enabled and self._old_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
        self._enabled = False

    def _run(self) -> None:
        while not self.stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                self.stop_event.set()
                return
            self.events.put(ch.lower())


def tty_style(text: str, ansi_code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{ansi_code}m{text}\033[0m"


def record_button(text: str, color: str) -> str:
    color_code = {"green": "1;97;42", "red": "1;97;41"}[color]
    return tty_style(f"  {text}  ", color_code)


def record_symbol(symbol: str, color: str) -> str:
    color_code = {"green": "1;32", "red": "1;31"}[color]
    return tty_style(symbol, color_code)
