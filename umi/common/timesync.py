from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional, Tuple


def now_monotonic_ns() -> int:
    return time.monotonic_ns()


@dataclass
class TimestampedSample:
    monotonic_ns: int
    payload: Any
    frame_id: int = 0
    extra: Optional[dict] = None


class RingBuffer:
    def __init__(self, capacity: int = 256, max_age_ns: int = 500_000_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        self._capacity = int(capacity)
        self._max_age_ns = int(max_age_ns)
        self._lock = threading.Lock()
        self._samples: Deque[TimestampedSample] = deque()
        self._timestamps: List[int] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def max_age_ns(self) -> int:
        return self._max_age_ns

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def push(self, sample: TimestampedSample) -> None:
        with self._lock:
            if self._samples and sample.monotonic_ns < self._samples[-1].monotonic_ns:
                # 出现回退（极少见，例如 driver 多线程争用）：丢弃，保持单调
                return
            self._samples.append(sample)
            self._timestamps.append(int(sample.monotonic_ns))
            self._evict_locked(reference_ns=int(sample.monotonic_ns))

    def _evict_locked(self, reference_ns: int) -> None:
        cutoff = reference_ns - self._max_age_ns
        while self._samples and self._samples[0].monotonic_ns < cutoff:
            self._samples.popleft()
            self._timestamps.pop(0)
        while len(self._samples) > self._capacity:
            self._samples.popleft()
            self._timestamps.pop(0)

    def latest(self) -> Optional[TimestampedSample]:
        with self._lock:
            if not self._samples:
                return None
            return self._samples[-1]

    def get_nearest(self, target_ns: int) -> Optional[TimestampedSample]:
        with self._lock:
            if not self._samples:
                return None
            idx = bisect.bisect_left(self._timestamps, int(target_ns))
            if idx == 0:
                return self._samples[0]
            if idx >= len(self._samples):
                return self._samples[-1]
            before = self._samples[idx - 1]
            after = self._samples[idx]
            if abs(before.monotonic_ns - target_ns) <= abs(after.monotonic_ns - target_ns):
                return before
            return after

    def get_neighbors(
        self, target_ns: int
    ) -> Tuple[Optional[TimestampedSample], Optional[TimestampedSample]]:
        with self._lock:
            if not self._samples:
                return None, None
            idx = bisect.bisect_left(self._timestamps, int(target_ns))
            before = self._samples[idx - 1] if idx > 0 else None
            after = self._samples[idx] if idx < len(self._samples) else None
            return before, after

    def snapshot(self) -> List[TimestampedSample]:
        with self._lock:
            return list(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
            self._timestamps.clear()


class DeviceHostClockCalibrator:
    # 用线性回归把 device 时戳映射到 host monotonic 域：
    # host_ns ≈ a * device_ts + b
    # device_ts 单位/起点不重要（厂商常见 ns / 100ns / tick），线性拟合自适应。

    def __init__(self, window_size: int = 256, min_samples: int = 30):
        if window_size < 4:
            raise ValueError("window_size too small (need >= 4)")
        if min_samples < 4:
            raise ValueError("min_samples too small (need >= 4)")
        self._window_size = int(window_size)
        self._min_samples = int(min_samples)
        self._lock = threading.Lock()
        self._device: Deque[int] = deque(maxlen=self._window_size)
        self._host: Deque[int] = deque(maxlen=self._window_size)
        self._slope: Optional[float] = None
        self._intercept: Optional[float] = None

    def reset(self) -> None:
        with self._lock:
            self._device.clear()
            self._host.clear()
            self._slope = None
            self._intercept = None

    def record(self, device_ts: int, host_monotonic_ns: int) -> None:
        with self._lock:
            self._device.append(int(device_ts))
            self._host.append(int(host_monotonic_ns))
            if len(self._device) >= self._min_samples:
                self._refit_locked()

    def _refit_locked(self) -> None:
        n = len(self._device)
        if n < self._min_samples:
            return
        mean_x = sum(self._device) / n
        mean_y = sum(self._host) / n
        num = 0.0
        den = 0.0
        for x, y in zip(self._device, self._host):
            dx = x - mean_x
            num += dx * (y - mean_y)
            den += dx * dx
        if den <= 0.0:
            return
        self._slope = num / den
        self._intercept = mean_y - self._slope * mean_x

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._slope is not None and self._intercept is not None

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._device)

    @property
    def coefficients(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            return self._slope, self._intercept

    def device_to_host_ns(self, device_ts: int) -> int:
        with self._lock:
            if self._slope is None or self._intercept is None:
                raise RuntimeError("Calibrator is not ready (need more samples).")
            return int(round(self._slope * float(device_ts) + self._intercept))

    def residual_stats_ns(self) -> Tuple[float, float, float]:
        with self._lock:
            if self._slope is None or self._intercept is None:
                return 0.0, 0.0, 0.0
            residuals = [
                float(y) - (self._slope * float(x) + self._intercept)
                for x, y in zip(self._device, self._host)
            ]
            if not residuals:
                return 0.0, 0.0, 0.0
            mean = sum(residuals) / len(residuals)
            variance = sum((r - mean) ** 2 for r in residuals) / len(residuals)
            return mean, variance ** 0.5, max(abs(r) for r in residuals)


def linear_interpolate(
    before: TimestampedSample,
    after: TimestampedSample,
    target_ns: int,
    payload_interpolator,
) -> Any:
    if after.monotonic_ns == before.monotonic_ns:
        return before.payload
    alpha = (int(target_ns) - before.monotonic_ns) / float(
        after.monotonic_ns - before.monotonic_ns
    )
    alpha = max(0.0, min(1.0, alpha))
    return payload_interpolator(before.payload, after.payload, alpha)


def monotonic_to_wall_offset_s() -> float:
    """Snapshot of (time.time() - time.monotonic()) to convert monotonic seconds
    into wall-clock seconds. Capture once at process startup; do not refresh
    while running because wall-clock can jump under NTP/manual changes and any
    in-flight timestamps would then be inconsistent."""
    return time.time() - time.monotonic()
