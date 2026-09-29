from __future__ import annotations

import sys
if __package__ in {None, ""}:
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "mvs_cpp"

import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass, asdict, replace
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple

import numpy as np
from loguru import logger


# spawn 而不是 fork:子进程从干净的 Python 解释器启动,不会继承主进程内任何
# 已经初始化的 SDK / 线程 / 全局状态(包括 Synexens SDK)。这是把 MVS SDK 与
# Synexens SDK 在 Python 进程层面隔离的关键 —— 同进程时 mvs_cpp (nanobind)
# 加载会污染 Synexens 的 USB streaming 状态,反之亦然。
_MP_CTX = mp.get_context("spawn")

# 双缓冲:子进程在 (latest_idx + 1) % 2 写新帧,主进程从 latest_idx 读最新帧。
# 只要 metadata 更新是原子的,主进程读和子进程写就不会撞车。
_SHM_SLOTS = 2

_DEFAULT_STARTUP_TIMEOUT_S = 30.0
_DEFAULT_BACKEND_READ_TIMEOUT_MS = 500


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class MVSWorkerConfig:
    """子进程启动 mvs_cpp.Device 需要的全部配置。

    必须是 picklable 的纯数值类型(不能含 callable),因为 spawn 子进程会
    pickle 这个对象传过去。
    """

    serial: str
    output_width: int
    output_height: int
    fps: int

    sensor_width: Optional[int] = None
    sensor_height: Optional[int] = None
    offset_x: int = 0
    offset_y: int = 0
    crop_x: int = 0
    crop_y: int = 0
    crop_width: Optional[int] = None
    crop_height: Optional[int] = None
    use_sensor_crop_roi: bool = True
    sensor_roi_align: int = 4

    image_node_num: int = 1
    frame_pool_size: int = 4
    rotate_180: bool = True
    acquisition_frame_rate_enable: bool = True

    exposure_auto: str = "off"
    exposure_time_us: float = 15000.0
    gain_auto: str = "continuous"
    gain_db: Optional[float] = None
    balance_white_auto: str = "continuous"
    black_level_enable: bool = True
    black_level: int = 240
    brightness: Optional[int] = None

    backend_read_timeout_ms: int = _DEFAULT_BACKEND_READ_TIMEOUT_MS

    # 鱼眼自动裁剪：目标是把鱼眼像圈完整内接到正方形（或 output_aspect 矩形）里，
    # 允许四角保留原始黑色 — 不去除黑边，而是“以圆为基准”重置 sensor crop。
    auto_crop_black_border: bool = True
    auto_crop_probe_frames: int = 4
    auto_crop_min_probe_frames: int = 2
    auto_crop_warmup_frames: int = 2
    auto_crop_black_threshold: int = 12
    auto_crop_margin_px: int = 0
    auto_crop_min_size_ratio: float = 0.80
    auto_crop_reopen_delay_ms: int = 200
    auto_crop_detect_max_dim: int = 720

    @property
    def slot_size_bytes(self) -> int:
        return int(self.output_height) * int(self.output_width) * 3


@dataclass(frozen=True)
class MVSProxyCapture:
    image: Optional[np.ndarray]
    frame_id: int
    timestamp_ns: int
    device_timestamp_ticks: int = 0
    host_timestamp_ms: int = 0

    def __iter__(self):
        # Backward compatible with older callers that unpack
        # (image, frame_id, timestamp_ns).
        yield self.image
        yield self.frame_id
        yield self.timestamp_ns


class MVSAsyncProxy:
    """主进程访问点:封装一个跑 mvs_cpp.Device 的子进程。

    - start():spawn 子进程,等子进程完成 SDK 初始化(StartGrabbing 之后)
    - stop():通知子进程退出 + join + 清理 shared memory
    - latest_capture():从 shared memory 拷一份当前最新帧
    - wait_next_capture(timeout_ms):等下一帧出现(用 mp.Event 跨进程)
    """

    def __init__(
        self,
        worker_config: MVSWorkerConfig,
        camera_name: str = "mvs_camera",
        startup_timeout_s: float = _DEFAULT_STARTUP_TIMEOUT_S,
        worker_target=None,
    ):
        self._config = worker_config
        self._camera_name = camera_name
        self._startup_timeout_s = float(startup_timeout_s)
        # worker_target 必须是 module-level 函数(spawn 子进程要 pickle 它)。
        # 测试时可以注入 fake worker。
        self._worker_target = worker_target if worker_target is not None else _mvs_proxy_worker

        self._height = int(worker_config.output_height)
        self._width = int(worker_config.output_width)
        self._slot_size = int(worker_config.slot_size_bytes)
        self._total_size = self._slot_size * _SHM_SLOTS

        self._shm: Optional[SharedMemory] = None
        self._process: Optional[mp.process.BaseProcess] = None

        # 跨进程同步原语:全部从 spawn context 创建,确保 worker pickle/unpickle 时
        # 重新解析正确的 context。
        self._stop_event = _MP_CTX.Event()
        self._new_frame_event = _MP_CTX.Event()
        self._metadata_lock = _MP_CTX.Lock()

        # latest_idx = -1 表示尚未有任何帧;>=0 是最近写完的 slot 编号
        self._latest_idx = _MP_CTX.Value("i", -1)
        self._frame_id = _MP_CTX.Value("q", 0)         # int64
        self._timestamp_ns = _MP_CTX.Value("Q", 0)      # uint64
        self._device_timestamp_ticks = _MP_CTX.Value("Q", 0)  # uint64
        self._host_timestamp_ms = _MP_CTX.Value("q", 0)  # int64
        self._alive = _MP_CTX.Value("b", 0)             # bool flag

        self._startup_queue: mp.Queue = _MP_CTX.Queue(maxsize=1)

        self._started = False

    @property
    def is_connected(self) -> bool:
        return (
            self._started
            and self._process is not None
            and self._process.is_alive()
            and bool(self._alive.value)
        )

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return (self._width, self._height)

    def start(self) -> None:
        if self._started:
            logger.warning(f"{self._camera_name}: proxy already started")
            return

        logger.info(f"{self._camera_name}: creating shared memory ({self._total_size} bytes)...")
        self._shm = SharedMemory(create=True, size=self._total_size)

        # 重置同步状态
        self._stop_event.clear()
        self._new_frame_event.clear()
        with self._metadata_lock:
            self._latest_idx.value = -1
            self._frame_id.value = 0
            self._timestamp_ns.value = 0
            self._device_timestamp_ticks.value = 0
            self._host_timestamp_ms.value = 0
            self._alive.value = 0
        # drain startup queue
        try:
            while True:
                self._startup_queue.get_nowait()
        except queue.Empty:
            pass

        logger.info(
            f"{self._camera_name}: spawning MVS worker process (serial={self._config.serial})..."
        )
        self._process = _MP_CTX.Process(
            target=self._worker_target,
            name=f"{self._camera_name}_worker",
            args=(
                self._config,
                self._shm.name,
                self._slot_size,
                self._stop_event,
                self._new_frame_event,
                self._metadata_lock,
                self._latest_idx,
                self._frame_id,
                self._timestamp_ns,
                self._device_timestamp_ticks,
                self._host_timestamp_ms,
                self._alive,
                self._startup_queue,
                self._camera_name,
            ),
            daemon=True,
        )
        self._process.start()

        try:
            status, detail = self._startup_queue.get(timeout=self._startup_timeout_s)
        except queue.Empty as exc:
            self._cleanup()
            raise RuntimeError(
                f"{self._camera_name}: MVS worker startup timed out after "
                f"{self._startup_timeout_s:.1f}s"
            ) from exc

        if status != "ready":
            self._cleanup()
            raise RuntimeError(
                f"{self._camera_name}: MVS worker startup failed: {detail}"
            )

        self._started = True
        logger.info(f"{self._camera_name}: MVS proxy ready")

    def stop(self) -> None:
        if not self._started and self._process is None:
            return

        logger.info(f"{self._camera_name}: stopping MVS proxy...")
        self._stop_event.set()
        self._cleanup()
        self._started = False
        logger.info(f"{self._camera_name}: MVS proxy stopped")

    def _cleanup(self) -> None:
        if self._process is not None:
            try:
                self._process.join(timeout=5.0)
                if self._process.is_alive():
                    logger.warning(
                        f"{self._camera_name}: worker did not exit gracefully; terminating"
                    )
                    self._process.terminate()
                    self._process.join(timeout=2.0)
            except Exception as exc:
                logger.warning(f"{self._camera_name}: error joining worker: {exc}")
            self._process = None

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(f"{self._camera_name}: error releasing shared memory: {exc}")
            self._shm = None

        with self._metadata_lock:
            self._alive.value = 0
            self._latest_idx.value = -1

    def latest_capture(self) -> MVSProxyCapture:
        """返回最近一帧；旧调用方仍可按 (image, frame_id, timestamp_ns) 解包。"""
        if self._shm is None:
            return MVSProxyCapture(None, 0, 0)
        with self._metadata_lock:
            idx = int(self._latest_idx.value)
            fid = int(self._frame_id.value)
            ts = int(self._timestamp_ns.value)
            device_ts = int(self._device_timestamp_ticks.value)
            host_ts_ms = int(self._host_timestamp_ms.value)
        if idx < 0:
            return MVSProxyCapture(None, 0, 0)
        view = np.ndarray(
            (self._height, self._width, 3),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=idx * self._slot_size,
        )
        # 必须 copy:子进程很快会写下一帧,view 本身不安全
        return MVSProxyCapture(view.copy(), fid, ts, device_ts, host_ts_ms)

    def latest_capture_if_new(self, last_frame_id: int | None) -> MVSProxyCapture | None:
        """Return a copied latest frame only if the worker frame id changed.

        This avoids relying on multiprocessing.Event edge semantics. The worker
        continuously publishes the latest frame id in shared metadata; polling
        that id is more robust for a latest-frame cache because missed event
        notifications should not suppress the newest frame.
        """
        if self._shm is None:
            return None
        with self._metadata_lock:
            idx = int(self._latest_idx.value)
            fid = int(self._frame_id.value)
            ts = int(self._timestamp_ns.value)
            device_ts = int(self._device_timestamp_ticks.value)
            host_ts_ms = int(self._host_timestamp_ms.value)
        if idx < 0:
            return None
        if last_frame_id is not None and fid == int(last_frame_id):
            return None
        view = np.ndarray(
            (self._height, self._width, 3),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=idx * self._slot_size,
        )
        return MVSProxyCapture(view.copy(), fid, ts, device_ts, host_ts_ms)

    def wait_next_capture(self, timeout_ms: float = 200) -> MVSProxyCapture:
        """阻塞等到下一帧出现。

        如果超时或没有数据,抛 TimeoutError / RuntimeError。
        """
        deadline_s = max(0.001, float(timeout_ms) / 1000.0)
        if not self._new_frame_event.wait(timeout=deadline_s):
            raise TimeoutError(
                f"{self._camera_name}: Timeout waiting for next capture after {timeout_ms}ms."
            )
        # 注意:多个 reader 共享 event,clear 时机要谨慎。这里采用"读后 clear"
        # 模式:wait_next_capture 假设单一调用者(_processing_loop)。其它从动 API
        # (latest_capture)不应该依赖 new_frame_event。
        with self._metadata_lock:
            idx = int(self._latest_idx.value)
            fid = int(self._frame_id.value)
            ts = int(self._timestamp_ns.value)
            device_ts = int(self._device_timestamp_ticks.value)
            host_ts_ms = int(self._host_timestamp_ms.value)
            self._new_frame_event.clear()
        if idx < 0 or self._shm is None:
            raise RuntimeError(
                f"{self._camera_name}: no frame data available even though event fired"
            )
        view = np.ndarray(
            (self._height, self._width, 3),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=idx * self._slot_size,
        )
        return MVSProxyCapture(view.copy(), fid, ts, device_ts, host_ts_ms)

    def __enter__(self) -> "MVSAsyncProxy":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


def _open_device_from_config(config: MVSWorkerConfig, open_mvs_cpp_device):
    sensor_width = config.sensor_width
    sensor_height = config.sensor_height
    offset_x = int(config.offset_x)
    offset_y = int(config.offset_y)
    crop_x = int(config.crop_x)
    crop_y = int(config.crop_y)
    crop_width = config.crop_width
    crop_height = config.crop_height

    if (
        config.use_sensor_crop_roi
        and crop_width is not None
        and crop_height is not None
        and int(crop_width) > 0
        and int(crop_height) > 0
    ):
        align = max(1, int(config.sensor_roi_align))
        sensor_width = max(align, (int(crop_width) // align) * align)
        sensor_height = max(align, (int(crop_height) // align) * align)
        offset_x += crop_x
        offset_y += crop_y
        offset_x = (offset_x // align) * align
        offset_y = (offset_y // align) * align
        crop_x = 0
        crop_y = 0
        crop_width = sensor_width
        crop_height = sensor_height

    return open_mvs_cpp_device(
        serial=config.serial,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        offset_x=offset_x,
        offset_y=offset_y,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_width=crop_width,
        crop_height=crop_height,
        output_width=config.output_width,
        output_height=config.output_height,
        image_node_num=config.image_node_num,
        frame_pool_size=config.frame_pool_size,
        rotate_180=config.rotate_180,
        acquisition_frame_rate_enable=config.acquisition_frame_rate_enable,
        acquisition_frame_rate_fps=float(config.fps),
        exposure_auto=config.exposure_auto,
        exposure_time_us=config.exposure_time_us,
        gain_auto=config.gain_auto,
        gain_db=config.gain_db,
        balance_white_auto=config.balance_white_auto,
        black_level_enable=config.black_level_enable,
        black_level=config.black_level,
        brightness=config.brightness,
    )


def _capture_probe_frames(
    device,
    config: MVSWorkerConfig,
    no_data_error,
    camera_name: str,
) -> list[np.ndarray]:
    """抓若干 probe 帧用于鱼眼像圈检测。

    先丢弃 auto_crop_warmup_frames 帧让 AE/AWB 有时间收敛，再开始累计样本。
    """
    wanted = max(1, int(config.auto_crop_probe_frames))
    warmup = max(0, int(config.auto_crop_warmup_frames))
    timeout_ms = max(50, int(config.backend_read_timeout_ms))
    deadline = time.monotonic() + max(
        2.0, (wanted + warmup) * timeout_ms / 1000.0 + 1.0
    )

    discarded = 0
    while discarded < warmup and time.monotonic() < deadline:
        try:
            native = device.read_frame(timeout_ms=timeout_ms)
        except no_data_error:
            continue
        native.release()
        discarded += 1

    frames: list[np.ndarray] = []
    while len(frames) < wanted and time.monotonic() < deadline:
        try:
            native = device.read_frame(timeout_ms=timeout_ms)
        except no_data_error:
            continue
        try:
            frames.append(np.array(native.as_array(), dtype=np.uint8, copy=True))
        finally:
            native.release()

    logger.info(
        f"{camera_name}: auto-crop probe warmup={discarded} captured={len(frames)}/{wanted}"
    )
    return frames


def _crop_rect_around_circle(
    center_x: float,
    center_y: float,
    radius: float,
    output_aspect: float,
    margin_px: int,
) -> Optional[Tuple[int, int, int, int]]:
    """围绕检测到的鱼眼圆心,算出 output 空间的外接 crop 矩形。

    当 output 是正方形(aspect=1)时,返回 2r × 2r 的正方形;此时把这块从 sensor
    抓出来 resize 到 output 尺寸,圆刚好内接,四角是原始黑色。
    margin_px 可为负数,用于去掉镜头外圈的黑环。
    """
    if radius <= 0 or output_aspect <= 0:
        return None

    radius = max(1.0, float(radius) + float(margin_px))
    diameter = 2.0 * radius
    if output_aspect >= 1.0:
        crop_h = diameter
        crop_w = diameter * output_aspect
    else:
        crop_w = diameter
        crop_h = diameter / output_aspect

    x = int(round(float(center_x) - crop_w / 2.0))
    y = int(round(float(center_y) - crop_h / 2.0))
    return x, y, max(1, int(round(crop_w))), max(1, int(round(crop_h)))


def _probe_gray(
    frames: list[np.ndarray],
    max_dim: int,
) -> Optional[Tuple[np.ndarray, float]]:
    """合成一张降采样的灰度图供检测使用。

    每帧取通道最大值;先按 max_dim 缩到小图再多帧 median —— 大头操作发生在缩
    小后的图上,速度比在原尺寸上做 median 快得多。
    返回 (small_gray, scale),scale = original_dim / small_dim,后续把检测结果
    乘以 scale 即可还原到原图坐标。
    """
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
        logger.warning(f"auto-crop: opencv import failed ({exc}); skip detection")
        return None

    grays = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] < 3:
            continue
        chan_max = np.max(frame[:, :, :3], axis=2).astype(np.uint8)
        if scale != 1.0:
            chan_max = cv2.resize(chan_max, (new_w, new_h), interpolation=cv2.INTER_AREA)
        grays.append(chan_max)
    if not grays:
        return None
    if len(grays) == 1:
        return grays[0], scale
    return np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8), scale


def _make_auto_crop_probe_config(config: MVSWorkerConfig) -> MVSWorkerConfig:
    """打开全幅 sensor 帧用于启动检测。

    probe 阶段不 crop、不 resize、不 rotate,这样检测结果天然就是 sensor 坐标。
    正式采集仍然使用原来的 output_width/output_height,collector 侧 shape 不变。
    """
    sensor_w = int(config.sensor_width or config.crop_width or config.output_width)
    sensor_h = int(config.sensor_height or config.crop_height or config.output_height)
    return replace(
        config,
        crop_x=0,
        crop_y=0,
        crop_width=sensor_w,
        crop_height=sensor_h,
        output_width=sensor_w,
        output_height=sensor_h,
        rotate_180=False,
    )


def _border_connected_black_mask(
    gray: np.ndarray,
    threshold: int,
) -> Optional[np.ndarray]:
    """只提取与图像边界连通的近黑区域。

    中间的黑色物体不参与裁剪;镜头黑边/黑环通常与边界连通,会被保留下来。
    """
    try:
        import cv2
    except Exception as exc:
        logger.warning(f"auto-crop: opencv import failed ({exc}); skip detection")
        return None

    height, width = gray.shape[:2]
    if height <= 2 or width <= 2:
        return None

    black = (gray <= int(threshold)).astype(np.uint8)
    kernel_size = max(3, min(height, width) // 80)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel, iterations=1)

    filled = black.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    for x in range(width):
        if filled[0, x] == 1:
            cv2.floodFill(filled, flood_mask, (x, 0), 2)
        if filled[height - 1, x] == 1:
            cv2.floodFill(filled, flood_mask, (x, height - 1), 2)
    for y in range(height):
        if filled[y, 0] == 1:
            cv2.floodFill(filled, flood_mask, (0, y), 2)
        if filled[y, width - 1] == 1:
            cv2.floodFill(filled, flood_mask, (width - 1, y), 2)
    return filled == 2


def _fit_crop_rect_to_bounds(
    rect: Tuple[int, int, int, int],
    *,
    bounds_width: int,
    bounds_height: int,
) -> Tuple[int, int, int, int]:
    x, y, crop_w, crop_h = rect
    center_x = float(x) + (float(crop_w) / 2.0)
    center_y = float(y) + (float(crop_h) / 2.0)
    scale = min(
        1.0,
        float(bounds_width) / max(1.0, float(crop_w)),
        float(bounds_height) / max(1.0, float(crop_h)),
    )
    crop_w = max(1, int(round(float(crop_w) * scale)))
    crop_h = max(1, int(round(float(crop_h) * scale)))
    x = int(round(center_x - (float(crop_w) / 2.0)))
    y = int(round(center_y - (float(crop_h) / 2.0)))
    x = max(0, min(x, int(bounds_width) - crop_w))
    y = max(0, min(y, int(bounds_height) - crop_h))
    return x, y, crop_w, crop_h


def _detect_inscribed_circle_sensor_crop(
    frames: list[np.ndarray],
    config: MVSWorkerConfig,
    *,
    output_aspect: float,
) -> Optional[Tuple[int, int, int, int]]:
    """在全幅 sensor probe 帧上找最大有效轮廓,返回 sensor crop。

    算法按常见无标定裁黑边流程:
    1. 多帧 median 得到灰度图;
    2. 阈值二值化得到非黑有效区域;
    3. 形态学 close/open 平滑边界、消掉小噪声;
    4. 找最大外部轮廓,用其 bbox 推出符合 output aspect 的 sensor crop。
    """
    if len(frames) < max(1, int(config.auto_crop_min_probe_frames)):
        return None

    probe = _probe_gray(frames, max_dim=int(config.auto_crop_detect_max_dim))
    if probe is None:
        return None
    gray_small, scale = probe

    try:
        import cv2
    except Exception as exc:
        logger.warning(f"auto-crop: opencv import failed ({exc}); skip detection")
        return None

    threshold = int(config.auto_crop_black_threshold)
    valid = (gray_small > threshold).astype(np.uint8)
    valid_ratio = float(np.mean(valid))
    if valid_ratio < 0.05:
        logger.warning(
            f"auto-crop: valid coverage too small ({valid_ratio:.4f}); "
            "skip detection"
        )
        return None

    kernel_size = max(3, min(valid.shape[:2]) // 80)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
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
    bbox_x = int(round((float(bbox_x) * scale) - float(margin)))
    bbox_y = int(round((float(bbox_y) * scale) - float(margin)))
    bbox_w = int(round(float(bbox_w) * scale + (2.0 * float(margin))))
    bbox_h = int(round(float(bbox_h) * scale + (2.0 * float(margin))))
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    bbox_cx = float(bbox_x) + (float(bbox_w) / 2.0)
    bbox_cy = float(bbox_y) + (float(bbox_h) / 2.0)
    if output_aspect >= 1.0:
        crop_w = max(float(bbox_w), float(bbox_h) * output_aspect)
        crop_h = crop_w / output_aspect
    else:
        crop_h = max(float(bbox_h), float(bbox_w) / output_aspect)
        crop_w = crop_h * output_aspect
    rect = (
        int(round(bbox_cx - (crop_w / 2.0))),
        int(round(bbox_cy - (crop_h / 2.0))),
        max(1, int(round(crop_w))),
        max(1, int(round(crop_h))),
    )

    sensor_w = int(config.output_width)
    sensor_h = int(config.output_height)
    sensor_rect = _fit_crop_rect_to_bounds(
        rect,
        bounds_width=sensor_w,
        bounds_height=sensor_h,
    )

    _, _, crop_w, crop_h = sensor_rect
    min_ratio = max(0.05, min(1.0, float(config.auto_crop_min_size_ratio)))
    min_sensor_side = min(sensor_w, sensor_h)
    if crop_w < min_sensor_side * min_ratio or crop_h < min_sensor_side * min_ratio:
        logger.warning(
            f"auto-crop: detected contour crop too small "
            f"(rect={crop_w}x{crop_h} vs sensor={sensor_w}x{sensor_h}, "
            f"min_ratio={min_ratio:.2f}, valid_ratio={valid_ratio:.3f}, "
            f"area={largest_area:.1f})"
        )
        return None

    logger.info(
        f"auto-crop: contour_bbox=({bbox_x},{bbox_y},{bbox_w},{bbox_h}) "
        f"valid_ratio={valid_ratio:.3f}"
    )
    return sensor_rect


def _resolve_auto_crop_config(
    config: MVSWorkerConfig,
    *,
    open_mvs_cpp_device,
    no_data_error,
    camera_name: str,
) -> MVSWorkerConfig:
    """探测鱼眼像圈,把 sensor crop 调整为"最大有效轮廓外接矩形"。

    任一环节失败都 fallback 到原 config,只发 warning。
    """
    if not config.auto_crop_black_border:
        return config

    probe_device = None
    try:
        logger.info(
            f"{camera_name}: auto-crop probing full sensor fisheye circle "
            "from live MVS frames..."
        )
        probe_config = _make_auto_crop_probe_config(config)
        probe_device = _open_device_from_config(probe_config, open_mvs_cpp_device)
        frames = _capture_probe_frames(
            probe_device,
            config=probe_config,
            no_data_error=no_data_error,
            camera_name=camera_name,
        )
        sensor_rect = _detect_inscribed_circle_sensor_crop(
            frames,
            probe_config,
            output_aspect=float(config.output_width) / float(config.output_height),
        )
        if sensor_rect is None:
            logger.warning(
                f"{camera_name}: auto-crop could not locate the fisheye circle; "
                "using configured MVS crop"
            )
            return config

        logger.info(
            f"{camera_name}: auto-crop sensor_crop={sensor_rect} "
            f"output={config.output_width}x{config.output_height}"
        )
        return replace(
            config,
            crop_x=sensor_rect[0],
            crop_y=sensor_rect[1],
            crop_width=sensor_rect[2],
            crop_height=sensor_rect[3],
        )
    except Exception as exc:
        logger.warning(
            f"{camera_name}: auto-crop probe failed ({type(exc).__name__}: {exc}); "
            "using configured MVS crop"
        )
        return config
    finally:
        if probe_device is not None:
            try:
                probe_device.close()
            except Exception as exc:
                logger.warning(f"{camera_name}: auto-crop probe close failed: {exc}")
            delay_s = max(0, int(config.auto_crop_reopen_delay_ms)) / 1000.0
            if delay_s > 0:
                time.sleep(delay_s)


def _mvs_proxy_worker(
    config: MVSWorkerConfig,
    shm_name: str,
    slot_size: int,
    stop_event,
    new_frame_event,
    metadata_lock,
    latest_idx,
    frame_id_value,
    timestamp_ns_value,
    device_timestamp_ticks_value,
    host_timestamp_ms_value,
    alive_value,
    startup_queue,
    camera_name: str,
) -> None:
    """子进程入口:打开 MVS Device,持续读帧写共享内存。

    主进程通过共享内存 + Value 拿数据,所以这里只跟 mvs_cpp / numpy 交互。
    """
    # 子进程内重新配置 logger(spawn 起来的 Python 解释器是新的)
    logger.info(f"{camera_name}: worker process started (pid={mp.current_process().pid})")

    shm: Optional[SharedMemory] = None
    device = None
    try:
        # 1. attach 到主进程已经创建的 shared memory
        shm = SharedMemory(name=shm_name, create=False)

        # 2. 子进程内 import + open mvs_cpp.Device
        from mvs_cpp.backend import (
            open_mvs_cpp_device,
            MVSCPPNoDataError,
        )

        config = _resolve_auto_crop_config(
            config,
            open_mvs_cpp_device=open_mvs_cpp_device,
            no_data_error=MVSCPPNoDataError,
            camera_name=camera_name,
        )

        # cpp 后处理路径开关:这里默认开,因为我们已经把 crop_*/output_* 都
        # 传进去了,SDK 会输出 (output_height, output_width, 3) 的 RGB。
        try:
            device = _open_device_from_config(config, open_mvs_cpp_device)
        except Exception as exc:
            if not config.use_sensor_crop_roi:
                raise
            logger.warning(
                f"{camera_name}: hardware ROI crop failed ({type(exc).__name__}: {exc}); "
                "falling back to software crop"
            )
            device = _open_device_from_config(
                replace(config, use_sensor_crop_roi=False),
                open_mvs_cpp_device,
            )

        # 验证帧 shape 跟主进程预分配的 slot_size 一致
        actual_w, actual_h = device.frame_shape
        expected_size = int(actual_h) * int(actual_w) * 3
        if expected_size != slot_size:
            raise RuntimeError(
                f"{camera_name}: frame shape mismatch: expected slot_size={slot_size} "
                f"but device produces {actual_w}x{actual_h}x3 = {expected_size}"
            )

        with metadata_lock:
            alive_value.value = 1

        startup_queue.put(("ready", None))
        logger.info(f"{camera_name}: worker ready, entering capture loop")

    except Exception as exc:
        logger.error(f"{camera_name}: worker startup failed: {exc}")
        logger.error(traceback.format_exc())
        try:
            startup_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        # 清理已创建资源
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
        return

    # 抓帧循环
    next_idx = 0
    consecutive_errors = 0
    fid_counter = 0  # worker 自增计数,比 SDK 的 nFrameNum 更可靠(SDK 内部 buffer
                     # 循环时 nFrameNum 会从 0 重启)
    try:
        while not stop_event.is_set():
            try:
                native_frame = device.read_frame(timeout_ms=config.backend_read_timeout_ms)
            except MVSCPPNoDataError:
                # 没数据,继续轮询(SDK 内部可能临时没有 frame buffer)
                consecutive_errors = 0
                continue
            except Exception as exc:
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 30 == 0:
                    logger.warning(
                        f"{camera_name}: read_frame error #{consecutive_errors}: {exc}"
                    )
                if consecutive_errors >= 100:
                    logger.error(
                        f"{camera_name}: too many consecutive read errors, exiting worker"
                    )
                    break
                time.sleep(0.05)
                continue

            consecutive_errors = 0
            fid_counter += 1

            try:
                # 复制到 shared memory 的 next_idx slot
                src = native_frame.as_array()
                dst = np.ndarray(
                    (config.output_height, config.output_width, 3),
                    dtype=np.uint8,
                    buffer=shm.buf,
                    offset=next_idx * slot_size,
                )
                np.copyto(dst, src)
                ts = int(native_frame.timestamp_ns)
                device_ts = int(native_frame.device_timestamp_ticks)
                host_ts_ms = int(native_frame.host_timestamp_ms)
            finally:
                native_frame.release()

            # 原子更新 metadata
            with metadata_lock:
                latest_idx.value = next_idx
                frame_id_value.value = fid_counter
                timestamp_ns_value.value = ts
                device_timestamp_ticks_value.value = device_ts
                host_timestamp_ms_value.value = host_ts_ms
            new_frame_event.set()

            next_idx = (next_idx + 1) % _SHM_SLOTS

    except Exception as exc:
        logger.error(f"{camera_name}: worker loop crashed: {exc}")
        logger.error(traceback.format_exc())
    finally:
        with metadata_lock:
            alive_value.value = 0
        try:
            if device is not None:
                device.close()
        except Exception as exc:
            logger.warning(f"{camera_name}: device.close() failed: {exc}")
        try:
            if shm is not None:
                shm.close()
        except Exception:
            pass
        logger.info(f"{camera_name}: worker process exiting")


# ---------------- helpers used by mvs.py for config conversion ----------------


def build_worker_config_from_camcontroller(
    cam_controller_config,
    *,
    enable_cpp_postprocess: bool,
    backend_read_timeout_ms: int = _DEFAULT_BACKEND_READ_TIMEOUT_MS,
) -> MVSWorkerConfig:
    """从 MVSCamControllerConfig 提取出可 pickle 的 MVSWorkerConfig。

    enable_cpp_postprocess=True 时,把 crop_*/output_* 都传给 SDK,让 SDK
    内部 crop+resize,worker 直接拿 (height, width, 3) 的成品 RGB。
    enable_cpp_postprocess=False 暂不支持(第一版)。
    """
    if not enable_cpp_postprocess:
        raise NotImplementedError(
            "MVSAsyncProxy currently requires enable_cpp_postprocess=True; "
            "configure crop_func/img_transform_func to identity."
        )
    return MVSWorkerConfig(
        serial=str(cam_controller_config.serial),
        output_width=int(cam_controller_config.width),
        output_height=int(cam_controller_config.height),
        fps=int(cam_controller_config.fps),
        sensor_width=cam_controller_config.sensor_width,
        sensor_height=cam_controller_config.sensor_height,
        offset_x=int(cam_controller_config.offset_x),
        offset_y=int(cam_controller_config.offset_y),
        crop_x=int(cam_controller_config.crop_x),
        crop_y=int(cam_controller_config.crop_y),
        crop_width=cam_controller_config.crop_width,
        crop_height=cam_controller_config.crop_height,
        use_sensor_crop_roi=True,
        sensor_roi_align=4,
        image_node_num=int(cam_controller_config.image_node_num),
        frame_pool_size=int(cam_controller_config.frame_pool_size),
        rotate_180=bool(cam_controller_config.rotate_180),
        acquisition_frame_rate_enable=bool(
            cam_controller_config.acquisition_frame_rate_enable
        ),
        exposure_auto=str(cam_controller_config.exposure_auto),
        exposure_time_us=float(cam_controller_config.exposure_time_us),
        gain_auto=str(cam_controller_config.gain_auto),
        gain_db=(
            None
            if getattr(cam_controller_config, "gain_db", None) is None
            else float(cam_controller_config.gain_db)
        ),
        balance_white_auto=str(cam_controller_config.balance_white_auto),
        black_level_enable=bool(cam_controller_config.black_level_enable),
        black_level=int(cam_controller_config.black_level),
        brightness=(
            None
            if getattr(cam_controller_config, "brightness", None) is None
            else int(cam_controller_config.brightness)
        ),
        backend_read_timeout_ms=int(backend_read_timeout_ms),
        auto_crop_black_border=_env_bool("DEXUMI_MVS_AUTO_CROP_BLACK_BORDER", True),
        auto_crop_probe_frames=_env_int("DEXUMI_MVS_AUTO_CROP_PROBE_FRAMES", 4),
        auto_crop_min_probe_frames=_env_int("DEXUMI_MVS_AUTO_CROP_MIN_PROBE_FRAMES", 2),
        auto_crop_warmup_frames=_env_int("DEXUMI_MVS_AUTO_CROP_WARMUP_FRAMES", 2),
        auto_crop_black_threshold=_env_int("DEXUMI_MVS_AUTO_CROP_BLACK_THRESHOLD", 12),
        auto_crop_margin_px=_env_int("DEXUMI_MVS_AUTO_CROP_MARGIN_PX", 0),
        auto_crop_min_size_ratio=_env_float("DEXUMI_MVS_AUTO_CROP_MIN_SIZE_RATIO", 0.80),
        auto_crop_reopen_delay_ms=_env_int("DEXUMI_MVS_AUTO_CROP_REOPEN_DELAY_MS", 200),
        auto_crop_detect_max_dim=_env_int("DEXUMI_MVS_AUTO_CROP_DETECT_MAX_DIM", 720),
    )


def main() -> None:
    """单文件烟雾测试:启动 proxy → 抓 30 帧 → 打印 frame_id/timestamp → 停止。

    用法:python -m mvs_cpp.mvs_proxy <serial>
    """
    import argparse

    parser = argparse.ArgumentParser(description="MVSAsyncProxy 单文件自测")
    parser.add_argument("serial", nargs="?", default=None, help="MVS 相机序列号 (留空则自动选第一个)")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    serial = args.serial
    if serial is None:
        from mvs_cpp.backend import list_mvs_device_serials

        serials = list_mvs_device_serials()
        if not serials:
            print("No MVS device found")
            return
        serial = serials[0]
        print(f"using first available serial: {serial}")

    config = MVSWorkerConfig(
        serial=serial,
        output_width=args.width,
        output_height=args.height,
        fps=args.fps,
    )
    proxy = MVSAsyncProxy(config, camera_name="mvs_smoke")
    try:
        proxy.start()
        for i in range(args.frames):
            try:
                img, fid, ts = proxy.wait_next_capture(timeout_ms=1000)
            except TimeoutError as exc:
                print(f"[{i}] timeout: {exc}")
                continue
            print(f"[{i}] frame_id={fid} ts_ns={ts} shape={img.shape}")
    finally:
        proxy.stop()


if __name__ == "__main__":
    main()
