from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .backend import (
    MVSCPPNoDataError,
    get_mvs_usb_link_info,
    open_mvs_cpp_device,
)
from .mvs_proxy import MVSAsyncProxy, MVSWorkerConfig


_MIN_HISTORY_SIZE = 64
_HISTORY_SECONDS = 5.0


def _compute_center_crop_rect(
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
) -> tuple[int, int, int, int]:
    """Return a centered crop rectangle matching the requested output aspect."""
    if input_width <= 0 or input_height <= 0:
        raise ValueError(
            f"MVS input resolution must be positive, got {input_width}x{input_height}"
        )
    if output_width <= 0 or output_height <= 0:
        raise ValueError(
            f"MVS output resolution must be positive, got {output_width}x{output_height}"
        )

    input_aspect = input_width / input_height
    output_aspect = output_width / output_height
    if abs(input_aspect - output_aspect) < 1e-6:
        return 0, 0, input_width, input_height

    if input_aspect > output_aspect:
        crop_height = input_height
        crop_width = int(round(crop_height * output_aspect))
        crop_x = max(0, (input_width - crop_width) // 2)
        crop_y = 0
    else:
        crop_width = input_width
        crop_height = int(round(crop_width / output_aspect))
        crop_x = 0
        crop_y = max(0, (input_height - crop_height) // 2)

    return crop_x, crop_y, min(crop_width, input_width), min(crop_height, input_height)


def _fit_crop_rect_to_bounds(
    rect: tuple[int, int, int, int],
    *,
    bounds_width: int,
    bounds_height: int,
) -> tuple[int, int, int, int]:
    x, y, crop_w, crop_h = rect
    center_x = float(x) + float(crop_w) / 2.0
    center_y = float(y) + float(crop_h) / 2.0
    scale = min(
        1.0,
        float(bounds_width) / max(1.0, float(crop_w)),
        float(bounds_height) / max(1.0, float(crop_h)),
    )
    crop_w = max(1, int(round(float(crop_w) * scale)))
    crop_h = max(1, int(round(float(crop_h) * scale)))
    x = int(round(center_x - float(crop_w) / 2.0))
    y = int(round(center_y - float(crop_h) / 2.0))
    x = max(0, min(x, int(bounds_width) - crop_w))
    y = max(0, min(y, int(bounds_height) - crop_h))
    return x, y, crop_w, crop_h


def _probe_gray(frames: list[np.ndarray], max_dim: int) -> tuple[np.ndarray, float] | None:
    if not frames:
        return None
    src_h, src_w = frames[0].shape[:2]
    target = max(32, int(max_dim))
    longest = max(src_h, src_w)
    if longest > target:
        scale = longest / float(target)
        new_w = max(8, int(round(src_w / scale)))
        new_h = max(8, int(round(src_h / scale)))
    else:
        scale = 1.0
        new_w, new_h = src_w, src_h

    try:
        import cv2
    except Exception as exc:
        print(f"[Warn] MVS auto-crop: opencv import failed ({exc}); skip detection")
        return None

    grays = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] < 3:
            continue
        channel_max = np.max(frame[:, :, :3], axis=2).astype(np.uint8)
        if scale != 1.0:
            channel_max = cv2.resize(
                channel_max,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )
        grays.append(channel_max)
    if not grays:
        return None
    if len(grays) == 1:
        return grays[0], scale
    return np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8), scale


def _detect_inscribed_circle_sensor_crop(
    frames: list[np.ndarray],
    config: "MVSConfig",
) -> tuple[int, int, int, int] | None:
    if len(frames) < max(1, int(config.auto_crop_min_probe_frames)):
        return None

    probe = _probe_gray(frames, max_dim=int(config.auto_crop_detect_max_dim))
    if probe is None:
        return None
    gray_small, scale = probe

    try:
        import cv2
    except Exception as exc:
        print(f"[Warn] MVS auto-crop: opencv import failed ({exc}); skip detection")
        return None

    threshold = int(config.auto_crop_black_threshold)
    valid = (gray_small > threshold).astype(np.uint8)
    valid_ratio = float(np.mean(valid))
    if valid_ratio < 0.05:
        print(
            f"[Warn] MVS auto-crop: valid coverage too small "
            f"({valid_ratio:.4f}); using fallback crop"
        )
        return None

    kernel_size = max(3, min(valid.shape[:2]) // 80)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel, iterations=1)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    largest_area = float(cv2.contourArea(largest))
    if largest_area <= 0:
        return None

    bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(largest)
    margin = int(config.auto_crop_margin_px)
    bbox_x = int(round(float(bbox_x) * scale - float(margin)))
    bbox_y = int(round(float(bbox_y) * scale - float(margin)))
    bbox_w = int(round(float(bbox_w) * scale + 2.0 * float(margin)))
    bbox_h = int(round(float(bbox_h) * scale + 2.0 * float(margin)))
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    output_aspect = float(config.width) / float(config.height)
    bbox_cx = float(bbox_x) + float(bbox_w) / 2.0
    bbox_cy = float(bbox_y) + float(bbox_h) / 2.0
    if output_aspect >= 1.0:
        crop_w = max(float(bbox_w), float(bbox_h) * output_aspect)
        crop_h = crop_w / output_aspect
    else:
        crop_h = max(float(bbox_h), float(bbox_w) / output_aspect)
        crop_w = crop_h * output_aspect

    rect = (
        int(round(bbox_cx - crop_w / 2.0)),
        int(round(bbox_cy - crop_h / 2.0)),
        max(1, int(round(crop_w))),
        max(1, int(round(crop_h))),
    )
    sensor_rect = _fit_crop_rect_to_bounds(
        rect,
        bounds_width=int(config.input_width),
        bounds_height=int(config.input_height),
    )

    _, _, crop_w_i, crop_h_i = sensor_rect
    min_ratio = max(0.05, min(1.0, float(config.auto_crop_min_size_ratio)))
    min_sensor_side = min(int(config.input_width), int(config.input_height))
    if crop_w_i < min_sensor_side * min_ratio or crop_h_i < min_sensor_side * min_ratio:
        print(
            f"[Warn] MVS auto-crop: detected crop too small "
            f"(rect={crop_w_i}x{crop_h_i}, sensor={config.input_width}x{config.input_height}, "
            f"min_ratio={min_ratio:.2f}, valid_ratio={valid_ratio:.3f}, "
            f"area={largest_area:.1f}); using fallback crop"
        )
        return None

    print(
        f"[Init] MVS auto-crop: contour_bbox=({bbox_x},{bbox_y},{bbox_w},{bbox_h}) "
        f"valid_ratio={valid_ratio:.3f}"
    )
    return sensor_rect


@dataclass(frozen=True)
class MVSConfig:
    """Runtime config for the MVS RGB latest cache."""

    serial: str
    fps: float = 30.0
    width: int = 480
    height: int = 480
    input_width: int = 1440
    input_height: int = 1080
    center_crop: bool = True
    timeout_ms: int = 100
    warmup_sec: float = 1.0
    exposure_time_us: float = 15000.0
    gain_auto: str = "continuous"
    gain_db: float | None = None
    balance_white_auto: str = "continuous"
    rotate_180: bool = True
    image_node_num: int = 1
    frame_pool_size: int = 4
    auto_crop_black_border: bool = True
    auto_crop_probe_frames: int = 4
    auto_crop_min_probe_frames: int = 2
    auto_crop_warmup_frames: int = 2
    auto_crop_black_threshold: int = 12
    auto_crop_margin_px: int = 0
    auto_crop_min_size_ratio: float = 0.80
    auto_crop_reopen_delay_ms: int = 200
    auto_crop_detect_max_dim: int = 720
    use_subprocess: bool = True
    use_sensor_crop_roi: bool = True
    backend_read_timeout_ms: int | None = None
    startup_timeout_s: float = 30.0


@dataclass
class RGBCachedFrame:
    """RGB frame currently held by the MVS latest cache."""

    image: np.ndarray | None
    frame_id: int | None
    capture_ns: int | None
    device_timestamp_ticks: int | None = None
    host_timestamp_ms: int | None = None
    is_repeated: bool | None = None
    frame_gap: int | None = None
    frames_read: int | None = None
    sync_target_ns: int | None = None
    sync_delta_ns: int | None = None
    selection_mode: str | None = None


class LatestMVSCache:
    """
    Background MVS reader that keeps only the newest RGB frame.

    The capture timestamp stored in messages is the Python host monotonic time
    when the frame enters this cache. This keeps rgbAlignResidualNs comparable
    with sampleClockNs, which also comes from time.monotonic_ns().
    """

    def __init__(self, config: MVSConfig):
        self.config = config
        self.period = 1.0 / max(float(config.fps), 1.0)
        self._lock = threading.Lock()
        self._frame_available = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._latest = RGBCachedFrame(None, None, None)
        history_size = max(
            _MIN_HISTORY_SIZE,
            int(round(max(float(config.fps), 1.0) * _HISTORY_SECONDS)),
        )
        self._history: deque[RGBCachedFrame] = deque(maxlen=history_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._device = None
        self._proxy: MVSAsyncProxy | None = None
        self._frames_read = 0
        self._last_returned_frame_id: int | None = None
        self._latest_returns = 0
        self._repeated_returns = 0
        self._skipped_return_frames = 0

    @property
    def using_subprocess(self) -> bool:
        return self._proxy is not None

    def stats(self) -> dict:
        with self._lock:
            return {
                "framesRead": int(self._frames_read),
                "latestReturns": int(self._latest_returns),
                "repeatedReturns": int(self._repeated_returns),
                "skippedReturnFrames": int(self._skipped_return_frames),
                "lastFrameId": self._latest.frame_id,
                "lastCaptureNs": self._latest.capture_ns,
                "historySize": len(self._history),
                "historyMaxlen": int(self._history.maxlen or 0),
                "backend": "subprocess" if self.using_subprocess else "in_process",
            }

    def start(self) -> None:
        self._open_device()
        self._warmup()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.config.timeout_ms / 1000.0 + 0.5))
        if self._proxy is not None:
            try:
                self._proxy.stop()
            finally:
                self._proxy = None
        if self._device is not None:
            try:
                self._device.close()
            finally:
                self._device = None

    def _clone_frame_locked(
        self,
        frame: RGBCachedFrame,
        copy_image: bool,
        count_return: bool = True,
    ) -> RGBCachedFrame:
        if frame.image is None:
            image = None
        elif copy_image:
            image = frame.image.copy()
        else:
            image = frame.image
        frame_id = frame.frame_id
        is_repeated = None
        frame_gap = None
        if frame_id is not None and count_return:
            frame_id_int = int(frame_id)
            if self._last_returned_frame_id is None:
                is_repeated = False
                frame_gap = 0
            else:
                delta = frame_id_int - self._last_returned_frame_id
                is_repeated = delta == 0
                frame_gap = max(0, delta - 1)
            self._last_returned_frame_id = frame_id_int
            self._latest_returns += 1
            if is_repeated:
                self._repeated_returns += 1
            if frame_gap:
                self._skipped_return_frames += int(frame_gap)
        return RGBCachedFrame(
            image=image,
            frame_id=frame_id,
            capture_ns=frame.capture_ns,
            device_timestamp_ticks=frame.device_timestamp_ticks,
            host_timestamp_ms=frame.host_timestamp_ms,
            is_repeated=is_repeated,
            frame_gap=frame_gap,
            frames_read=self._frames_read,
            sync_target_ns=frame.sync_target_ns,
            sync_delta_ns=frame.sync_delta_ns,
            selection_mode=frame.selection_mode,
        )

    def _copy_latest_locked(
        self,
        copy_image: bool,
        count_return: bool = True,
    ) -> RGBCachedFrame:
        return self._clone_frame_locked(
            self._latest,
            copy_image=copy_image,
            count_return=count_return,
        )

    def _append_history_locked(self, frame: RGBCachedFrame) -> None:
        if self._history and self._history[-1].frame_id == frame.frame_id:
            self._history[-1] = frame
            return
        self._history.append(frame)

    def current_frame_id(self) -> int | None:
        with self._lock:
            if self._latest.frame_id is None:
                return None
            return int(self._latest.frame_id)

    def reset_return_tracking(self) -> None:
        with self._lock:
            self._last_returned_frame_id = None
            self._latest_returns = 0
            self._repeated_returns = 0
            self._skipped_return_frames = 0

    def latest(
        self,
        copy_image: bool = True,
        count_return: bool = True,
    ) -> RGBCachedFrame:
        with self._lock:
            return self._copy_latest_locked(
                copy_image=copy_image,
                count_return=count_return,
            )

    def nearest(
        self,
        target_ns: int,
        copy_image: bool = True,
        max_delta_ns: int | None = None,
        count_return: bool = True,
    ) -> RGBCachedFrame:
        target = int(target_ns)
        with self._lock:
            candidates = [
                frame
                for frame in self._history
                if frame.capture_ns is not None
            ]
            if not candidates:
                return self._copy_latest_locked(
                    copy_image=copy_image,
                    count_return=count_return,
                )
            selected = min(
                candidates,
                key=lambda frame: abs(int(frame.capture_ns) - target),
            )
            if (
                max_delta_ns is not None
                and selected.capture_ns is not None
                and abs(int(selected.capture_ns) - target) > int(max_delta_ns)
            ):
                return RGBCachedFrame(None, None, None)
            synced = replace(
                selected,
                sync_target_ns=target,
                sync_delta_ns=(
                    None
                    if selected.capture_ns is None
                    else int(selected.capture_ns) - target
                ),
                selection_mode="nearest_capture_ns",
            )
            return self._clone_frame_locked(
                synced,
                copy_image=copy_image,
                count_return=count_return,
            )

    def wait_next(
        self,
        last_frame_id: int | None,
        timeout_s: float,
        copy_image: bool = True,
        count_return: bool = True,
    ) -> RGBCachedFrame:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._frame_available:
            while not self._stop.is_set():
                frame_id = self._latest.frame_id
                if frame_id is not None and (
                    last_frame_id is None or int(frame_id) != int(last_frame_id)
                ):
                    return self._copy_latest_locked(
                        copy_image=copy_image,
                        count_return=count_return,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._frame_available.wait(timeout=remaining)
        raise TimeoutError(
            f"MVS cache returned no new frame within {timeout_s:.3f}s"
        )

    def _open_device(self) -> None:
        crop_x = crop_y = 0
        crop_width = crop_height = None
        crop_source = "full sensor"

        if self.config.auto_crop_black_border and not self.config.use_subprocess:
            sensor_rect = self._resolve_auto_crop_rect()
            if sensor_rect is not None:
                crop_x, crop_y, crop_width, crop_height = sensor_rect
                crop_source = "auto-crop"

        if crop_width is None and crop_height is None and self.config.center_crop:
            crop_x, crop_y, crop_width, crop_height = _compute_center_crop_rect(
                input_width=self.config.input_width,
                input_height=self.config.input_height,
                output_width=self.config.width,
                output_height=self.config.height,
            )
            crop_source = "center-crop fallback"

        print(
            f"[Init] 连接 MVS 相机 serial={self.config.serial} "
            f"fps={self.config.fps:g} sensor={self.config.input_width}x{self.config.input_height} "
            f"crop=({crop_x},{crop_y},{crop_width},{crop_height}) "
            f"output={self.config.width}x{self.config.height} source={crop_source}"
        )
        if self.config.use_subprocess:
            self._proxy = self._open_proxy_with_crop(
                sensor_width=self.config.input_width,
                sensor_height=self.config.input_height,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_width=crop_width,
                crop_height=crop_height,
                output_width=self.config.width,
                output_height=self.config.height,
                rotate_180=self.config.rotate_180,
            )
            print("[Init] MVS backend: subprocess proxy")
        else:
            self._device = self._open_device_with_crop(
                sensor_width=self.config.input_width,
                sensor_height=self.config.input_height,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_width=crop_width,
                crop_height=crop_height,
                output_width=self.config.width,
                output_height=self.config.height,
                rotate_180=self.config.rotate_180,
            )
            try:
                runtime = self._device.get_runtime_config()
                print(f"[Init] MVS runtime config: {runtime}")
            except Exception as exc:
                print(f"[Warn] 读取 MVS runtime config 失败: {exc}")

        usb_link = get_mvs_usb_link_info(self.config.serial)
        if usb_link is None:
            print("[Warn] 无法从 sysfs 读取 MVS USB 链路速度")
        elif usb_link.is_usb2_fallback:
            print(f"[Warn] MVS USB 链路可能降级: {usb_link.summary()}")
        else:
            print(f"[Init] MVS USB 链路: {usb_link.summary()}")

    def _open_device_with_crop(
        self,
        *,
        sensor_width: int,
        sensor_height: int,
        crop_x: int,
        crop_y: int,
        crop_width: int | None,
        crop_height: int | None,
        output_width: int,
        output_height: int,
        rotate_180: bool,
    ):
        return open_mvs_cpp_device(
            self.config.serial,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            output_width=output_width,
            output_height=output_height,
            image_node_num=self.config.image_node_num,
            frame_pool_size=self.config.frame_pool_size,
            rotate_180=rotate_180,
            acquisition_frame_rate_enable=True,
            acquisition_frame_rate_fps=float(self.config.fps),
            exposure_auto="off",
            exposure_time_us=self.config.exposure_time_us,
            gain_auto=self.config.gain_auto,
            gain_db=self.config.gain_db,
            balance_white_auto=self.config.balance_white_auto,
        )

    def _open_proxy_with_crop(
        self,
        *,
        sensor_width: int,
        sensor_height: int,
        crop_x: int,
        crop_y: int,
        crop_width: int | None,
        crop_height: int | None,
        output_width: int,
        output_height: int,
        rotate_180: bool,
    ) -> MVSAsyncProxy:
        worker_config = MVSWorkerConfig(
            serial=self.config.serial,
            output_width=output_width,
            output_height=output_height,
            fps=max(1, int(round(float(self.config.fps)))),
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            use_sensor_crop_roi=bool(self.config.use_sensor_crop_roi),
            image_node_num=self.config.image_node_num,
            frame_pool_size=self.config.frame_pool_size,
            rotate_180=rotate_180,
            acquisition_frame_rate_enable=True,
            exposure_auto="off",
            exposure_time_us=self.config.exposure_time_us,
            gain_auto=self.config.gain_auto,
            gain_db=self.config.gain_db,
            balance_white_auto=self.config.balance_white_auto,
            backend_read_timeout_ms=(
                int(self.config.backend_read_timeout_ms)
                if self.config.backend_read_timeout_ms is not None
                else max(50, int(self.config.timeout_ms))
            ),
            auto_crop_black_border=bool(self.config.auto_crop_black_border),
            auto_crop_probe_frames=int(self.config.auto_crop_probe_frames),
            auto_crop_min_probe_frames=int(self.config.auto_crop_min_probe_frames),
            auto_crop_warmup_frames=int(self.config.auto_crop_warmup_frames),
            auto_crop_black_threshold=int(self.config.auto_crop_black_threshold),
            auto_crop_margin_px=int(self.config.auto_crop_margin_px),
            auto_crop_min_size_ratio=float(self.config.auto_crop_min_size_ratio),
            auto_crop_reopen_delay_ms=int(self.config.auto_crop_reopen_delay_ms),
            auto_crop_detect_max_dim=int(self.config.auto_crop_detect_max_dim),
        )
        proxy = MVSAsyncProxy(
            worker_config,
            camera_name=f"mvs_{self.config.serial}",
            startup_timeout_s=float(self.config.startup_timeout_s),
        )
        proxy.start()
        return proxy

    def _resolve_auto_crop_rect(self) -> tuple[int, int, int, int] | None:
        probe_device = None
        try:
            print(
                "[Init] MVS auto-crop probing full sensor fisheye circle "
                "from live frames..."
            )
            probe_device = self._open_device_with_crop(
                sensor_width=self.config.input_width,
                sensor_height=self.config.input_height,
                crop_x=0,
                crop_y=0,
                crop_width=int(self.config.input_width),
                crop_height=int(self.config.input_height),
                output_width=int(self.config.input_width),
                output_height=int(self.config.input_height),
                rotate_180=False,
            )
            frames = self._capture_probe_frames(probe_device)
            sensor_rect = _detect_inscribed_circle_sensor_crop(frames, self.config)
            if sensor_rect is None:
                print(
                    "[Warn] MVS auto-crop could not locate the fisheye circle; "
                    "using fallback crop"
                )
                return None
            print(
                f"[Init] MVS auto-crop sensor_crop={sensor_rect} "
                f"output={self.config.width}x{self.config.height}"
            )
            return sensor_rect
        except Exception as exc:
            print(
                f"[Warn] MVS auto-crop probe failed ({type(exc).__name__}: {exc}); "
                "using fallback crop"
            )
            return None
        finally:
            if probe_device is not None:
                try:
                    probe_device.close()
                except Exception as exc:
                    print(f"[Warn] MVS auto-crop probe close failed: {exc}")
            delay_s = max(0, int(self.config.auto_crop_reopen_delay_ms)) / 1000.0
            if delay_s > 0:
                time.sleep(delay_s)

    def _capture_probe_frames(self, device) -> list[np.ndarray]:
        wanted = max(1, int(self.config.auto_crop_probe_frames))
        warmup = max(0, int(self.config.auto_crop_warmup_frames))
        timeout_ms = max(50, int(self.config.timeout_ms))
        deadline = time.monotonic() + max(
            2.0,
            float(wanted + warmup) * float(timeout_ms) / 1000.0 + 1.0,
        )

        discarded = 0
        while discarded < warmup and time.monotonic() < deadline:
            try:
                native = device.read_frame(timeout_ms=timeout_ms)
            except MVSCPPNoDataError:
                continue
            try:
                discarded += 1
            finally:
                native.release()

        frames: list[np.ndarray] = []
        while len(frames) < wanted and time.monotonic() < deadline:
            try:
                native = device.read_frame(timeout_ms=timeout_ms)
            except MVSCPPNoDataError:
                continue
            try:
                frames.append(native.copy_array())
            finally:
                native.release()

        print(
            f"[Init] MVS auto-crop probe warmup={discarded} "
            f"captured={len(frames)}/{wanted}"
        )
        return frames

    def _read_once(self) -> RGBCachedFrame:
        if self._proxy is not None:
            deadline = time.monotonic() + max(0.001, float(self.config.timeout_ms) / 1000.0)
            with self._lock:
                last_frame_id = self._latest.frame_id
            capture = None
            while time.monotonic() < deadline and not self._stop.is_set():
                capture = self._proxy.latest_capture_if_new(last_frame_id)
                if capture is not None:
                    break
                time.sleep(0.001)
            if capture is None:
                raise TimeoutError(
                    f"MVS proxy returned no new frame within {self.config.timeout_ms}ms"
                )
            if capture.image is None:
                raise RuntimeError("MVS proxy returned no image")
            return RGBCachedFrame(
                image=np.asarray(capture.image, dtype=np.uint8),
                frame_id=int(capture.frame_id),
                capture_ns=int(capture.timestamp_ns),
                device_timestamp_ticks=int(capture.device_timestamp_ticks),
                host_timestamp_ms=int(capture.host_timestamp_ms),
            )

        if self._device is None:
            raise RuntimeError("MVS device is not open")
        native = self._device.read_frame(timeout_ms=self.config.timeout_ms)
        try:
            return RGBCachedFrame(
                image=native.copy_array(),
                frame_id=int(native.frame_id),
                capture_ns=int(native.timestamp_ns),
                device_timestamp_ticks=int(native.device_timestamp_ticks),
                host_timestamp_ms=int(native.host_timestamp_ms),
            )
        finally:
            native.release()

    def _warmup(self) -> None:
        if self.config.warmup_sec <= 0:
            return
        print(f"[Init] MVS warmup {self.config.warmup_sec:.1f}s ...")
        deadline = time.perf_counter() + self.config.warmup_sec
        got_frame = False
        while time.perf_counter() < deadline and not self._stop.is_set():
            try:
                frame = self._read_once()
                with self._frame_available:
                    self._latest = frame
                    self._append_history_locked(frame)
                    self._frame_available.notify_all()
                got_frame = True
            except Exception:
                time.sleep(0.02)
        if not got_frame:
            print("[Warn] MVS warmup 期间没有读到 RGB 帧，后台线程会继续尝试")

    def _run(self) -> None:
        miss_count = 0
        warning_interval = max(int(round(float(self.config.fps))), 1)
        while not self._stop.is_set():
            try:
                frame = self._read_once()
                with self._frame_available:
                    self._latest = frame
                    self._append_history_locked(frame)
                    self._frames_read += 1
                    self._frame_available.notify_all()
                miss_count = 0
            except MVSCPPNoDataError:
                miss_count += 1
                if miss_count % warning_interval == 0:
                    print(
                        f"[Warn] MVS 已连续 {miss_count} 次未读到 RGB 帧 "
                        f"({miss_count/max(self.config.fps, 1.0):.1f}s)"
                    )
            except Exception as exc:
                miss_count += 1
                if miss_count % warning_interval == 0:
                    print(f"[Warn] MVS 缓存线程读取失败: {exc}")
