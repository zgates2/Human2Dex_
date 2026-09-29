#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public sensor runtime instances used by collection entrypoints."""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import threading
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from collect_config import parse_resolution_config


@dataclass
class PicoCachedFrame:
    """One PICO frame held by the background latest cache."""

    raw26x7: np.ndarray | None
    active: int
    sdk_ts_ns: int
    receive_ns: int


class LatestPicoCache:
    """Background PICO reader that keeps the newest hand frame."""

    def __init__(self, reader, poll_hz: float):
        self.reader = reader
        self.period = 1.0 / max(float(poll_hz), 1.0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = PicoCachedFrame(None, 0, 0, 0)
        history_size = max(64, int(round(max(float(poll_hz), 1.0) * 5.0)))
        self._history: deque[PicoCachedFrame] = deque(maxlen=history_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._frames_read = 0
        self._read_errors = 0
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> bool:
        if not self._started:
            return True
        self._stop.set()
        self._thread.join(timeout=max(float(timeout_s), 0.1))
        if self._thread.is_alive():
            print(
                "[Warn] PICO 缓存线程未能在超时内退出；"
                "跳过 reader.close() 以避免与仍在进行的 SDK 读取竞争"
            )
            return False
        return True

    def latest(self) -> PicoCachedFrame:
        with self._lock:
            return self._clone_frame(self._latest)

    def nearest(self, target_ns: int) -> PicoCachedFrame:
        target = int(target_ns)
        with self._lock:
            candidates = [
                frame
                for frame in self._history
                if int(frame.receive_ns) != 0
            ]
            if not candidates:
                return self._clone_frame(self._latest)
            selected = min(
                candidates,
                key=lambda frame: abs(int(frame.receive_ns) - target),
            )
            return self._clone_frame(selected)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "framesRead": int(self._frames_read),
                "readErrors": int(self._read_errors),
                "lastSdkTimestampNs": int(self._latest.sdk_ts_ns),
                "lastReceiveNs": int(self._latest.receive_ns),
                "historySize": len(self._history),
            }

    def _clone_frame(self, frame: PicoCachedFrame) -> PicoCachedFrame:
        raw = None if frame.raw26x7 is None else frame.raw26x7.copy()
        return PicoCachedFrame(
            raw26x7=raw,
            active=frame.active,
            sdk_ts_ns=frame.sdk_ts_ns,
            receive_ns=frame.receive_ns,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                raw26x7, active = self.reader.read_raw()
                sdk_ts_ns = self.reader.get_timestamp_ns()
                if raw26x7 is not None:
                    raw26x7 = np.asarray(raw26x7, dtype=np.float64)
                frame = PicoCachedFrame(
                    raw26x7=raw26x7,
                    active=int(active),
                    sdk_ts_ns=int(sdk_ts_ns),
                    receive_ns=time.monotonic_ns(),
                )
                with self._lock:
                    self._latest = frame
                    self._history.append(frame)
                    self._frames_read += 1
            except Exception as exc:
                with self._lock:
                    self._read_errors += 1
                print(f"[Warn] PICO 缓存线程读取失败: {exc}")

            sleep_for = self.period - (time.perf_counter() - t0)
            if sleep_for > 0:
                self._stop.wait(sleep_for)


class PicoSensor:
    """Lifecycle wrapper around PicoHandReader and LatestPicoCache."""

    def __init__(self, hand: str, poll_hz: float):
        from pico_hand import PicoHandReader

        self.hand = str(hand)
        self.poll_hz = float(poll_hz)
        self.reader = PicoHandReader(hand=self.hand)
        self.cache = LatestPicoCache(self.reader, poll_hz=self.poll_hz)

    def start(self) -> None:
        self.cache.start()

    def latest(self) -> PicoCachedFrame:
        return self.cache.latest()

    def nearest(self, target_ns: int) -> PicoCachedFrame:
        return self.cache.nearest(target_ns=target_ns)

    def stop(self, timeout_s: float = 3.0) -> bool:
        return self.cache.stop(timeout_s=timeout_s)

    def close(self) -> None:
        self.reader.close()

    def metadata(self) -> dict[str, Any]:
        return {
            "hand": self.hand,
            "pollHz": self.poll_hz,
            "stats": self.cache.stats(),
        }


@dataclass
class PicoFirstViewCachedFrame:
    """One first-view RGB frame held by the background camera cache."""

    image: np.ndarray | None
    frame_id: int | None
    capture_ns: int | None


class LatestPicoFirstViewCache:
    """Background cache for a Pico first-view stream exposed as a video device."""

    def __init__(
        self,
        device: object,
        fps: float,
        width: int | None = None,
        height: int | None = None,
    ):
        self.device = device
        self.fps = max(float(fps), 1.0)
        self.width = width
        self.height = height
        self.period = 1.0 / self.fps
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = PicoFirstViewCachedFrame(None, None, None)
        history_size = max(64, int(round(self.fps * 5.0)))
        self._history: deque[PicoFirstViewCachedFrame] = deque(maxlen=history_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._capture = None
        self._cv2 = None
        self._frames_read = 0
        self._read_errors = 0
        self._started = False
        self._next_frame_id = 0
        self._browser_process = None
        self._browser_port: int | None = None
        self._browser_profile_dir: str | None = None
        self._browser_ws_url: str | None = None
        self._source_url: str | None = None

    def start(self) -> None:
        if self._started:
            return

        if self._is_web_page_source(self.device):
            self._start_web_page()
        else:
            self._start_video_device()

        self._started = True
        self._thread.start()

    def _start_video_device(self) -> None:
        import cv2  # noqa: WPS433

        source = self.device
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())
        use_v4l2 = isinstance(source, int) or (
            isinstance(source, str) and source.startswith("/dev/video")
        )
        backend = cv2.CAP_V4L2 if use_v4l2 else cv2.CAP_ANY
        capture = cv2.VideoCapture(source, backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"无法打开 Pico 第一视角视频源: {self.device!r}"
            )
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        capture.set(cv2.CAP_PROP_FPS, float(self.fps))
        self._cv2 = cv2
        self._capture = capture

    @staticmethod
    def _is_web_page_source(source: object) -> bool:
        if not isinstance(source, str):
            return False
        value = source.strip().lower()
        if value.startswith("touping.picoxr.com/"):
            return True
        parsed = urlparse(value)
        return parsed.hostname in {"touping.picoxr.com", "www.touping.picoxr.com"}

    @staticmethod
    def _normalize_web_url(source: object) -> str:
        value = str(source).strip()
        if value.startswith("touping.picoxr.com/"):
            return f"https://{value}"
        return value

    @staticmethod
    def _find_chrome() -> str:
        candidates = [
            shutil.which("google-chrome"),
            shutil.which("chromium"),
            "/opt/google/chrome/chrome",
        ]
        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        raise RuntimeError("未找到 Chrome/Chromium，无法打开 Pico 投屏网页")

    @staticmethod
    def _reserve_local_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def _start_web_page(self) -> None:
        import cv2  # noqa: WPS433

        self._cv2 = cv2
        self._source_url = self._normalize_web_url(self.device)
        chrome = self._find_chrome()
        self._browser_port = self._reserve_local_port()
        self._browser_profile_dir = tempfile.mkdtemp(prefix="dexumi_pico_cast_")
        command = [
            chrome,
            f"--remote-debugging-port={self._browser_port}",
            f"--user-data-dir={self._browser_profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-crash-reporter",
            "--autoplay-policy=no-user-gesture-required",
            f"--window-size={self.width or 1280},{self.height or 720}",
            self._source_url,
        ]
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            command.insert(1, "--headless=new")
        try:
            self._browser_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._browser_ws_url = self._wait_for_page_target()
        except Exception:
            self._stop_browser()
            raise

    def _wait_for_page_target(self) -> str:
        assert self._browser_port is not None
        endpoint = f"http://127.0.0.1:{self._browser_port}/json/list"
        deadline = time.monotonic() + 15.0
        last_error = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(endpoint, timeout=1.0) as response:
                    targets = json.loads(response.read().decode("utf-8"))
                pages = [
                    target for target in targets if target.get("type") == "page"
                ]
                if pages and pages[0].get("webSocketDebuggerUrl"):
                    return str(pages[0]["webSocketDebuggerUrl"])
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(0.1)
        detail = "" if last_error is None else f": {last_error}"
        raise RuntimeError(f"Chrome 投屏网页未能启动{detail}")

    @staticmethod
    def _cdp_call(websocket, sequence_id: int, method: str, params: dict | None = None):
        websocket.send(
            json.dumps(
                {
                    "id": int(sequence_id),
                    "method": method,
                    "params": params or {},
                }
            )
        )
        while True:
            payload = json.loads(websocket.recv(timeout=5.0))
            if payload.get("id") == int(sequence_id):
                if "error" in payload:
                    raise RuntimeError(f"Chrome DevTools {method} 失败: {payload['error']}")
                return payload.get("result", {})

    def _run_web_page(self) -> None:
        if self._browser_ws_url is None:
            raise RuntimeError("Chrome 投屏网页缺少 DevTools WebSocket 地址")
        try:
            from websockets.sync.client import connect  # noqa: WPS433
        except ImportError as exc:
            raise RuntimeError(
                "采集网页投屏需要 websockets；请在 pico 环境安装 websockets"
            ) from exc

        with connect(self._browser_ws_url, open_timeout=10.0) as websocket:
            sequence_id = 0
            sequence_id += 1
            self._cdp_call(websocket, sequence_id, "Page.enable")
            sequence_id += 1
            self._cdp_call(websocket, sequence_id, "Page.bringToFront")
            while not self._stop.is_set():
                t0 = time.perf_counter()
                sequence_id += 1
                result = self._cdp_call(
                    websocket,
                    sequence_id,
                    "Page.captureScreenshot",
                    {"format": "jpeg", "quality": 90, "fromSurface": True},
                )
                encoded = result.get("data")
                if encoded:
                    encoded_bytes = base64.b64decode(encoded)
                    bgr = self._cv2.imdecode(
                        np.frombuffer(encoded_bytes, dtype=np.uint8),
                        self._cv2.IMREAD_COLOR,
                    )
                    if bgr is not None:
                        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
                        frame = PicoFirstViewCachedFrame(
                            image=np.asarray(rgb, dtype=np.uint8),
                            frame_id=int(self._next_frame_id),
                            capture_ns=time.monotonic_ns(),
                        )
                        self._next_frame_id += 1
                        with self._lock:
                            self._latest = frame
                            self._history.append(frame)
                            self._frames_read += 1
                    else:
                        with self._lock:
                            self._read_errors += 1
                else:
                    with self._lock:
                        self._read_errors += 1

                sleep_for = self.period - (time.perf_counter() - t0)
                if sleep_for > 0:
                    self._stop.wait(sleep_for)

    def _stop_browser(self) -> None:
        if self._browser_process is not None:
            if self._browser_process.poll() is None:
                self._browser_process.terminate()
                try:
                    self._browser_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._browser_process.kill()
            self._browser_process = None
        if self._browser_profile_dir is not None:
            shutil.rmtree(self._browser_profile_dir, ignore_errors=True)
            self._browser_profile_dir = None

    def stop(self, timeout_s: float = 3.0) -> bool:
        if not self._started:
            return True
        self._stop.set()
        self._thread.join(timeout=max(float(timeout_s), 0.1))
        if self._thread.is_alive():
            print("[Warn] Pico 第一视角相机线程未能在超时内退出")
            self._stop_browser()
            return False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._stop_browser()
        return True

    def latest(self) -> PicoFirstViewCachedFrame:
        with self._lock:
            return self._clone_frame(self._latest)

    def nearest(self, target_ns: int) -> PicoFirstViewCachedFrame:
        target = int(target_ns)
        with self._lock:
            candidates = [
                frame for frame in self._history if frame.capture_ns is not None
            ]
            if not candidates:
                return self._clone_frame(self._latest)
            selected = min(
                candidates,
                key=lambda frame: abs(int(frame.capture_ns) - target),
            )
            return self._clone_frame(selected)

    def stats(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "framesRead": int(self._frames_read),
                "readErrors": int(self._read_errors),
                "lastFrameId": self._latest.frame_id,
                "lastCaptureNs": self._latest.capture_ns,
                "historySize": len(self._history),
            }

    @staticmethod
    def _clone_frame(frame: PicoFirstViewCachedFrame) -> PicoFirstViewCachedFrame:
        return PicoFirstViewCachedFrame(
            image=None if frame.image is None else frame.image.copy(),
            frame_id=frame.frame_id,
            capture_ns=frame.capture_ns,
        )

    def _run(self) -> None:
        if self._is_web_page_source(self.device):
            try:
                self._run_web_page()
            except Exception as exc:
                with self._lock:
                    self._read_errors += 1
                print(f"[Warn] Pico 投屏网页读取失败: {exc}")
            return

        assert self._capture is not None
        assert self._cv2 is not None
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, bgr = self._capture.read()
            if ok and bgr is not None:
                try:
                    if bgr.ndim == 2:
                        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_GRAY2RGB)
                    elif bgr.ndim == 3 and bgr.shape[2] == 4:
                        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGRA2RGB)
                    elif bgr.ndim == 3 and bgr.shape[2] == 3:
                        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
                    else:
                        raise ValueError(f"视频帧格式不支持: shape={bgr.shape}")
                    frame = PicoFirstViewCachedFrame(
                        image=np.asarray(rgb, dtype=np.uint8),
                        frame_id=int(self._next_frame_id),
                        capture_ns=time.monotonic_ns(),
                    )
                    self._next_frame_id += 1
                    with self._lock:
                        self._latest = frame
                        self._history.append(frame)
                        self._frames_read += 1
                except Exception:
                    with self._lock:
                        self._read_errors += 1
            else:
                with self._lock:
                    self._read_errors += 1

            sleep_for = self.period - (time.perf_counter() - t0)
            if sleep_for > 0:
                self._stop.wait(sleep_for)


class PicoFirstViewSensor:
    """Lifecycle wrapper for the optional Pico first-view video source."""

    def __init__(
        self,
        device: object,
        fps: float,
        width: int | None = None,
        height: int | None = None,
    ):
        self.device = device
        self.fps = float(fps)
        self.width = width
        self.height = height
        self.cache = LatestPicoFirstViewCache(
            device=device,
            fps=fps,
            width=width,
            height=height,
        )

    def start(self) -> None:
        self.cache.start()

    def latest(self) -> PicoFirstViewCachedFrame:
        return self.cache.latest()

    def nearest(self, target_ns: int) -> PicoFirstViewCachedFrame:
        return self.cache.nearest(target_ns=target_ns)

    def stop(self) -> bool:
        return self.cache.stop()

    def metadata(self) -> dict[str, Any]:
        return {
            "source": (
                "chrome_devtools_webpage"
                if LatestPicoFirstViewCache._is_web_page_source(self.device)
                else "opencv_video_capture"
            ),
            "device": self.device,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "stats": self.cache.stats(),
        }


class MVSSensor:
    """Lifecycle wrapper around mvs_cpp.LatestMVSCache."""

    def __init__(self, config):
        from mvs_cpp import LatestMVSCache

        self.cache = LatestMVSCache(config)

    @property
    def config(self):
        return self.cache.config

    @property
    def using_subprocess(self) -> bool:
        return bool(self.cache.using_subprocess)

    def start(self) -> None:
        self.cache.start()

    def stop(self) -> None:
        self.cache.stop()

    def reset_return_tracking(self) -> None:
        self.cache.reset_return_tracking()

    def latest(self, copy_image: bool = True, count_return: bool = True):
        return self.cache.latest(
            copy_image=copy_image,
            count_return=count_return,
        )

    def nearest(
        self,
        target_ns: int,
        copy_image: bool = True,
        max_delta_ns: int | None = None,
        count_return: bool = True,
    ):
        return self.cache.nearest(
            target_ns=target_ns,
            copy_image=copy_image,
            max_delta_ns=max_delta_ns,
            count_return=count_return,
        )

    def wait_next(
        self,
        last_frame_id: int | None,
        timeout_s: float,
        copy_image: bool = True,
        count_return: bool = True,
    ):
        return self.cache.wait_next(
            last_frame_id=last_frame_id,
            timeout_s=timeout_s,
            copy_image=copy_image,
            count_return=count_return,
        )

    def current_frame_id(self) -> int | None:
        return self.cache.current_frame_id()

    def stats(self) -> dict:
        return self.cache.stats()

    def metadata(self) -> dict:
        cfg = self.cache.config
        return {
            "serial": cfg.serial,
            "fps": float(cfg.fps),
            "backend": "subprocess" if self.cache.using_subprocess else "in_process",
            "stats": self.cache.stats(),
            "outputWidth": int(cfg.width),
            "outputHeight": int(cfg.height),
            "inputWidth": int(cfg.input_width),
            "inputHeight": int(cfg.input_height),
            "useSubprocess": bool(cfg.use_subprocess),
            "useSensorCropRoi": bool(cfg.use_sensor_crop_roi),
            "autoCropBlackBorder": bool(cfg.auto_crop_black_border),
            "centerCropFallback": bool(cfg.center_crop),
            "rotate180": bool(cfg.rotate_180),
            "exposureTimeUs": float(cfg.exposure_time_us),
            "gainAuto": str(cfg.gain_auto),
            "gainDb": None if cfg.gain_db is None else float(cfg.gain_db),
            "balanceWhiteAuto": str(cfg.balance_white_auto),
        }


def build_l515_process_config(raw: dict[str, Any]):
    from l515_process_cache import L515ProcessConfig

    rgb_width, rgb_height = parse_resolution_config(
        raw.get("rgb_resolution"),
        (int(raw.get("rgb_width", 960)), int(raw.get("rgb_height", 540))),
    )
    depth_width, depth_height = parse_resolution_config(
        raw.get("depth_resolution"),
        (int(raw.get("depth_width", 640)), int(raw.get("depth_height", 480))),
    )
    serial = raw.get("serial")
    if serial in (None, ""):
        raise ValueError("L515 config requires explicit serial")
    return L515ProcessConfig(
        name=str(raw["name"]),
        serial=str(serial),
        rgb_width=int(rgb_width),
        rgb_height=int(rgb_height),
        depth_width=int(depth_width),
        depth_height=int(depth_height),
        rgb_fps=int(raw.get("rgb_fps", 60)),
        depth_fps=int(raw.get("depth_fps", 30)),
        align_to=str(raw.get("align_to", "color")),
        timeout_ms=int(raw.get("timeout_ms", 5000)),
        warmup_frames=int(raw.get("warmup_frames", 90)),
        strict_global_time=bool(raw.get("strict_global_time", True)),
        auto_exposure=bool(raw.get("auto_exposure", True)),
        auto_exposure_priority=float(raw.get("auto_exposure_priority", 0.0)),
        auto_white_balance=bool(raw.get("auto_white_balance", True)),
        startup_timeout_s=float(raw.get("startup_timeout_s", 30.0)),
    )


class L515SensorGroup:
    """Lifecycle wrapper around multiple L515ProcessCache instances."""

    def __init__(self, camera_configs: list[dict[str, Any]] | None, stagger_sec: float = 0.0):
        self.camera_configs = list(camera_configs or [])
        self.stagger_sec = float(stagger_sec)
        self._caches: dict[str, object] = {}

    @property
    def caches(self) -> dict[str, object]:
        return self._caches

    def start(self) -> None:
        if not self.camera_configs:
            raise ValueError("collect_l515_mvs.py requires l515_cameras in YAML")

        from l515_process_cache import L515ProcessCache

        try:
            for idx, raw in enumerate(self.camera_configs):
                l515_config = build_l515_process_config(raw)
                if l515_config.name in self._caches:
                    raise ValueError(f"duplicate L515 camera name: {l515_config.name}")
                cache = L515ProcessCache(l515_config)
                print(
                    f"[Init] starting L515 worker name={l515_config.name} "
                    f"serial={l515_config.serial}"
                )
                cache.start()
                self._caches[l515_config.name] = cache
                meta = cache.metadata()
                print(
                    f"[Init] L515 ready name={l515_config.name} "
                    f"domains=({meta['lastColorTimestampDomain']},"
                    f"{meta['lastDepthTimestampDomain']}) "
                    f"depth_scale={meta['depthScaleMPerUnit']}"
                )
                if self.stagger_sec > 0 and idx + 1 < len(self.camera_configs):
                    time.sleep(self.stagger_sec)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for cache in self._caches.values():
            cache.stop()
        self._caches = {}

    def latest_all(self, copy_images: bool = True) -> dict[str, object]:
        return {
            name: cache.latest(copy_images=copy_images)
            for name, cache in self._caches.items()
        }

    def nearest_all(
        self,
        target_ns: int,
        copy_images: bool = True,
        max_delta_ns: int | None = None,
    ) -> dict[str, object]:
        return {
            name: cache.nearest(
                target_ns=int(target_ns),
                copy_images=copy_images,
                max_delta_ns=max_delta_ns,
            )
            for name, cache in self._caches.items()
        }

    def metadata(self) -> dict[str, Any]:
        return {
            name: cache.metadata()
            for name, cache in self._caches.items()
        }


def list_l515_cameras() -> None:
    from l515_camera import fallback_rs_enumerate_summary, list_l515_devices

    devices = list_l515_devices()
    if not devices:
        print("[L515] pyrealsense2 未发现相机")
        summary = fallback_rs_enumerate_summary()
        if summary:
            print("[L515] rs-enumerate-devices:")
            print(summary)
        return
    print(f"[L515] 发现 {len(devices)} 个 RealSense 设备:")
    for idx, dev in enumerate(devices):
        print(
            f"  {idx}: name={dev['name']} serial={dev['serial']} "
            f"product_line={dev['product_line']} firmware={dev['firmware']}"
        )
