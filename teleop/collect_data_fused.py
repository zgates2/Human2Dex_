#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PICO + wrist MVS 模型自适应融合数据采集入口。

用途
----
只采集两类数据：
1. PICO 提供全局 wrist trajectoryPose 和一组 wrist-centered MANO 21 点。
2. wrist RGB checkpoint 实时预测另一组 MANO 21 点。
3. 自适应 Kalman-style 融合器按时序创新、骨长、同步误差和输入健康度逐关节
   分配权重；融合点生成 o6_command / wuji_command。
4. PKL 保留原始 PICO、视觉点、融合点、动态权重和诊断信息，便于离线验收。

不包含内容
--------
本脚本不启动深度相机，也不保存 depth。需要多相机/depth 时使用专用多相机采集脚本。

运行方法
--------


    conda activate pico

    cd /home/zjc/Desktop/human2dex

    python3 teleop/collect_data_fused.py --task-name pick_cube

交互控制
--------
    s        开始/暂停当前 episode
    Enter    录制中丢弃当前 episode，不保存 PKL/JPG
    q        退出
    Ctrl+C   退出；如果正在录制，会先保存当前 episode

保存结构
--------
配置中的 out 表示输出根目录，默认是 data/。同一个 task_name 永远写入同一个
task 目录，不会按运行次数再创建 demo 时间戳目录：

    <out>/<task_name>/episode_0001/episode_0001.pkl
    <out>/<task_name>/episode_0001/images/*.jpg
    <out>/<task_name>/episode_0002/episode_0002.pkl

episode 编号由 task 目录下已有 episode_XXXX 自动决定，选择最小缺失序号。
例如已有 episode_0001 和 episode_0003，下次会补 episode_0002。

代码结构
--------    
    collect_config.py   读取 YAML、解析输出目录、构造 MVS 配置
    sensor_runtime.py   PicoSensor / MVSSensor 的 start/latest/stop 实例 API
    episode_io.py       异步写图、PKL 保存、message 字段组装、键盘控制
    collect()           本文件的主采集编排：初始化传感器、采样、retarget、保存
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import shutil
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from collect_config import (
    CollectConfig,
    build_mvs_config,
    list_mvs_cameras,
    load_pico_mvs_config,
)
from episode_io import (
    AsyncJpgWriter,
    KeypressWatcher,
    LINKER_JOINTS,
    WUJI_JOINTS,
    build_invalid_message,
    build_message,
    episode_index_from_path,
    episode_pkl_path,
    pico_first_view_image_path_for_frame,
    record_button,
    record_symbol,
    rgb_image_path_for_frame,
    save_pkl,
    writer_errors,
    writer_stats,
)
from sensor_runtime import MVSSensor, PicoFirstViewSensor, PicoSensor
from hand_keypoint_fusion import AsyncFusedPipeline, FusedPicoWristRetargeter


DEFAULT_CONFIG_PATH = Path(__file__).with_name("collect_data_fused.yaml")
DEFAULT_FUSION_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "wrist" / "outputs" / "runs" / "ckeckpoints" / "wrist_3" / "best.pt"
)
DEFAULT_MANO_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "wrist" / "mano_v1_2" / "mano_v1_2" / "models"
)


def _rgb_frame_valid(rgb_frame) -> bool:
    return (
        rgb_frame is not None
        and getattr(rgb_frame, "capture_ns", None) is not None
        and getattr(rgb_frame, "image", None) is not None
    )


def _with_rgb_sync(rgb_frame, *, target_ns: int, selection_mode: str):
    if not _rgb_frame_valid(rgb_frame):
        return rgb_frame
    return replace(
        rgb_frame,
        sync_target_ns=int(target_ns),
        sync_delta_ns=int(rgb_frame.capture_ns) - int(target_ns),
        selection_mode=selection_mode,
    )


def _reused_rgb_frame(rgb_frame, *, target_ns: int):
    if not _rgb_frame_valid(rgb_frame):
        return rgb_frame
    return replace(
        rgb_frame,
        is_repeated=True,
        frame_gap=0 if getattr(rgb_frame, "frame_gap", None) is None else rgb_frame.frame_gap,
        sync_target_ns=int(target_ns),
        sync_delta_ns=int(rgb_frame.capture_ns) - int(target_ns),
        selection_mode="mvs_master_reused_previous",
    )


def _ns_summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }

    ordered = sorted(int(value) for value in values)

    def percentile(q: float) -> int:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * float(q)
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        if lo == hi:
            return ordered[lo]
        weight = position - lo
        return int(round(ordered[lo] * (1.0 - weight) + ordered[hi] * weight))

    return {
        "min": int(ordered[0]),
        "max": int(ordered[-1]),
        "mean": float(sum(ordered) / len(ordered)),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
    }


@dataclass
class EpisodeQualityStats:
    attempted_samples: int = 0
    valid_pico_samples: int = 0
    invalid_pico_samples: int = 0
    valid_rgb_samples: int = 0
    rgb_missing_samples: int = 0
    rgb_filled_samples: int = 0
    rgb_repeated_samples: int = 0
    rgb_low_fps_samples: int = 0
    rgb_quality_warning_samples: int = 0
    rgb_frame_gap_total: int = 0
    quality_flag_counts: dict[str, int] = field(default_factory=dict)
    source_age_ns: list[int] = field(default_factory=list)
    rgb_align_residual_ns: list[int] = field(default_factory=list)
    rgb_to_pico_receive_delta_ns: list[int] = field(default_factory=list)
    abs_rgb_to_pico_receive_delta_ns: list[int] = field(default_factory=list)

    def observe(
        self,
        *,
        valid_pico: bool,
        sample_ns: int,
        source_receive_ns: int,
        rgb_frame,
        quality_payload: dict | None = None,
    ) -> None:
        self.attempted_samples += 1
        for flag in (quality_payload or {}).get("flags", []):
            key = str(flag)
            self.quality_flag_counts[key] = self.quality_flag_counts.get(key, 0) + 1
        rgb_quality = (quality_payload or {}).get("rgb", {})
        if rgb_quality.get("lowFps") is True:
            self.rgb_low_fps_samples += 1
        if rgb_quality.get("qualityWarning") is not None:
            self.rgb_quality_warning_samples += 1
        if valid_pico:
            self.valid_pico_samples += 1
        else:
            self.invalid_pico_samples += 1

        if source_receive_ns != 0:
            self.source_age_ns.append(int(sample_ns) - int(source_receive_ns))

        if _rgb_frame_valid(rgb_frame):
            capture_ns = int(rgb_frame.capture_ns)
            self.valid_rgb_samples += 1
            if getattr(rgb_frame, "selection_mode", None) == "mvs_master_reused_previous":
                self.rgb_filled_samples += 1
            self.rgb_align_residual_ns.append(int(sample_ns) - capture_ns)
            if source_receive_ns != 0:
                delta_ns = capture_ns - int(source_receive_ns)
                self.rgb_to_pico_receive_delta_ns.append(delta_ns)
                self.abs_rgb_to_pico_receive_delta_ns.append(abs(delta_ns))
            if getattr(rgb_frame, "is_repeated", None) is True:
                self.rgb_repeated_samples += 1
            frame_gap = getattr(rgb_frame, "frame_gap", None)
            if frame_gap is not None:
                self.rgb_frame_gap_total += max(0, int(frame_gap))
        else:
            self.rgb_missing_samples += 1

    def as_metadata(self) -> dict:
        return {
            "attemptedSamples": int(self.attempted_samples),
            "validPicoSamples": int(self.valid_pico_samples),
            "invalidPicoSamples": int(self.invalid_pico_samples),
            "validRgbSamples": int(self.valid_rgb_samples),
            "rgbMissingSamples": int(self.rgb_missing_samples),
            "rgbFilledSamples": int(self.rgb_filled_samples),
            "rgbRepeatedSamples": int(self.rgb_repeated_samples),
            "rgbLowFpsSamples": int(self.rgb_low_fps_samples),
            "rgbQualityWarningSamples": int(self.rgb_quality_warning_samples),
            "rgbFrameGapTotal": int(self.rgb_frame_gap_total),
            "qualityFlagCounts": dict(sorted(self.quality_flag_counts.items())),
            "sourceAgeNs": _ns_summary(self.source_age_ns),
            "rgbAlignResidualNs": _ns_summary(self.rgb_align_residual_ns),
            "rgbToPicoReceiveDeltaNs": _ns_summary(
                self.rgb_to_pico_receive_delta_ns
            ),
            "absRgbToPicoReceiveDeltaNs": _ns_summary(
                self.abs_rgb_to_pico_receive_delta_ns
            ),
        }


@dataclass
class CollectUiState:
    recording: bool = False
    episode_index: int = 0
    frame_idx: int = 0
    saved_messages: int = 0
    valid_messages: int = 0
    discarded_episodes: int = 0
    valid_pico: bool = False
    rgb_valid: bool = False
    rgb_missing_reason: str | None = None
    rgb_frame_id: int | None = None
    rgb_frame_gap: int | None = None
    rgb_repeated: bool | None = None
    rgb_filled: bool = False
    rgb_dt_ms: float | None = None
    rgb_actual_fps: float = 0.0
    rgb_cache_fps: float = 0.0
    rgb_low_fps: bool = False
    rgb_align_ms: float | None = None
    rgb_to_pico_ms: float | None = None
    pico_quality_label: str = "MISSING"
    quality_flags: list[str] = field(default_factory=list)
    fps_loop: float = 0.0
    fps_saved: float = 0.0
    source_age_ms: float | None = None
    mean_bone_mm: float | None = None
    max_bone_mm: float | None = None
    trajectory_pose: np.ndarray | None = None
    trajectory_points: list[tuple[float, float, float]] = field(default_factory=list)
    writer_pending: int | None = None
    writer_written: int | None = None
    writer_blocked: int | None = None
    writer_errors: int | None = None
    status_text: str = "待机"
    alert_text: str | None = None


@dataclass
class CapturedFusionSample:
    frame_idx: int
    sdk_ts_ns: int
    sample_ns: int
    source_receive_ns: int
    raw26x7: np.ndarray
    pico_active: int
    rgb_frame: Any
    rgb_missing_reason: str | None
    rgb_image_path: Path | None
    pico_first_view_frame: Any
    pico_first_view_missing_reason: str | None
    pico_first_view_image_path: Path | None
    quality_payload: dict[str, Any]


MEDIAPIPE_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _bone_stats_mm(pts21) -> tuple[float | None, float | None]:
    try:
        pts = np.asarray(pts21, dtype=np.float32).reshape(21, 3)
    except Exception:
        return None, None
    lengths = [
        float(np.linalg.norm(pts[a] - pts[b]) * 1000.0)
        for a, b in MEDIAPIPE_BONES
    ]
    if not lengths:
        return None, None
    return float(np.mean(lengths)), float(np.max(lengths))


def _build_sample_quality(
    *,
    valid_pico: bool,
    raw26x7,
    active: int,
    rgb_valid: bool,
    rgb_frame,
    rgb_missing_reason: str | None,
    source_age_ms: float | None,
    rgb_to_pico_ms: float | None,
    rgb_dt_ms: float | None,
    rgb_actual_fps: float,
    rgb_low_fps: bool,
    rgb_pico_delta_large: bool,
    max_rgb_pico_delta_ms: float,
) -> dict:
    flags: list[str] = []
    if not valid_pico:
        if int(active) != 1:
            flags.append("pico_inactive")
        if raw26x7 is None:
            flags.append("pico_no_raw26x7")
        else:
            try:
                if np.asarray(raw26x7).shape != (26, 7):
                    flags.append("pico_bad_shape")
            except Exception:
                flags.append("pico_bad_shape")
    if not rgb_valid:
        flags.append("rgb_missing")
        if rgb_missing_reason:
            flags.append(str(rgb_missing_reason))
    else:
        if getattr(rgb_frame, "selection_mode", None) == "mvs_master_reused_previous":
            flags.append("rgb_filled")
        if getattr(rgb_frame, "is_repeated", None) is True:
            flags.append("rgb_repeated")
        frame_gap = getattr(rgb_frame, "frame_gap", None)
        if frame_gap is not None and int(frame_gap) > 0:
            flags.append("rgb_frame_gap")
        if rgb_low_fps:
            flags.append("rgb_low_fps")
        if rgb_pico_delta_large:
            flags.append("rgb_pico_delta_large")

    return {
        "ok": not flags,
        "flags": flags,
        "pico": {
            "valid": bool(valid_pico),
            "active": int(active),
            "rawShape": None if raw26x7 is None else list(np.asarray(raw26x7).shape),
            "sourceAgeMs": source_age_ms,
        },
        "rgb": {
            "valid": bool(rgb_valid),
            "actualFps": float(rgb_actual_fps),
            "dtMs": rgb_dt_ms,
            "lowFps": bool(rgb_low_fps),
            "frameFilled": bool(
                rgb_valid
                and getattr(rgb_frame, "selection_mode", None)
                == "mvs_master_reused_previous"
            ),
            "frameGap": None if rgb_frame is None else getattr(rgb_frame, "frame_gap", None),
            "repeated": None if rgb_frame is None else getattr(rgb_frame, "is_repeated", None),
            "missingReason": rgb_missing_reason if not rgb_valid else None,
            "qualityWarning": rgb_missing_reason if rgb_valid else None,
        },
        "alignment": {
            "rgbToPicoMs": rgb_to_pico_ms,
            "rgbPicoDeltaLarge": bool(rgb_pico_delta_large),
            "maxRgbPicoDeltaMs": float(max_rgb_pico_delta_ms),
        },
    }


def _status_color(ok: bool) -> tuple[int, int, int]:
    return (60, 220, 90) if ok else (30, 30, 230)


def _put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        str(text),
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _make_placeholder_rgb(width: int = 640, height: int = 480) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (28, 28, 28)
    _put_text(image, "NO RGB FRAME", (width // 2 - 120, height // 2), scale=0.9, color=(40, 40, 240), thickness=2)
    return image


def _draw_collect_preview(
    rgb_frame,
    state: CollectUiState,
    window_width: int = 960,
    window_height: int = 620,
) -> np.ndarray:
    if _rgb_frame_valid(rgb_frame):
        rgb = np.asarray(rgb_frame.image)
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            left = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            left = _make_placeholder_rgb()
    else:
        left = _make_placeholder_rgb()

    preview_h = window_height
    preview_w = min(640, max(320, window_width - 320))
    left = cv2.resize(left, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
    panel_w = window_width - preview_w
    panel = np.zeros((preview_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)

    rec_color = (40, 40, 240) if state.recording else (90, 90, 90)
    cv2.rectangle(left, (0, 0), (preview_w, 50), rec_color, -1)
    mode = "REC" if state.recording else "STANDBY"
    _put_text(left, f"{mode}  EP {state.episode_index:04d}", (16, 32), scale=0.82, color=(255, 255, 255), thickness=2)
    if state.alert_text:
        cv2.rectangle(left, (0, preview_h - 58), (preview_w, preview_h), (30, 30, 210), -1)
        _put_text(left, state.alert_text, (16, preview_h - 22), scale=0.72, color=(255, 255, 255), thickness=2)

    y = 34
    _put_text(panel, "DexUMI Collect", (16, y), scale=0.7, color=(255, 255, 255), thickness=2)
    y += 34
    _put_text(panel, f"s: start/pause   Enter: discard episode   q: quit", (16, y), scale=0.40, color=(190, 190, 190))
    y += 34

    rows = [
        ("loop fps", f"{state.fps_loop:.1f}", state.fps_loop > 0.8),
        ("saved fps", f"{state.fps_saved:.1f}", state.fps_saved > 0.8 or not state.recording),
        ("rgb cache fps", f"{state.rgb_cache_fps:.1f}", not state.recording or not state.rgb_low_fps),
        ("frame idx", str(state.frame_idx), True),
        ("saved", str(state.saved_messages), True),
        ("valid hand", str(state.valid_messages), state.valid_pico or not state.recording),
    ]
    for label, value, ok in rows:
        _put_text(panel, label, (18, y), color=(170, 170, 170))
        _put_text(panel, value, (150, y), color=_status_color(ok), thickness=2)
        y += 26

    y += 12
    cv2.line(panel, (16, y), (panel_w - 16, y), (70, 70, 70), 1)
    y += 28
    checks = [
        ("PICO hand", state.pico_quality_label, state.valid_pico),
        ("RGB frame", "OK" if state.rgb_valid else (state.rgb_missing_reason or "MISSING"), state.rgb_valid),
        ("RGB gap", str(state.rgb_frame_gap), not (state.rgb_frame_gap and state.rgb_frame_gap > 0)),
        ("RGB repeat", str(state.rgb_repeated), state.rgb_repeated is not True),
        ("RGB filled", str(state.rgb_filled), state.rgb_filled is not True),
        ("RGB id", str(state.rgb_frame_id), True),
    ]
    for label, value, ok in checks:
        _put_text(panel, label, (18, y), color=(170, 170, 170))
        _put_text(panel, value, (150, y), color=_status_color(ok), thickness=2)
        y += 26

    y += 10
    metrics = [
        ("source age", None if state.source_age_ms is None else f"{state.source_age_ms:.1f} ms"),
        ("rgb align", None if state.rgb_align_ms is None else f"{state.rgb_align_ms:.1f} ms"),
        ("rgb-pico", None if state.rgb_to_pico_ms is None else f"{state.rgb_to_pico_ms:.1f} ms"),
        ("rgb dt", None if state.rgb_dt_ms is None else f"{state.rgb_dt_ms:.1f} ms"),
        ("rgb sample fps", f"{state.rgb_actual_fps:.1f}"),
        ("mean bone", None if state.mean_bone_mm is None else f"{state.mean_bone_mm:.1f} mm"),
        ("max bone", None if state.max_bone_mm is None else f"{state.max_bone_mm:.1f} mm"),
    ]
    for label, value in metrics:
        _put_text(panel, label, (18, y), color=(170, 170, 170))
        _put_text(panel, "n/a" if value is None else value, (150, y), color=(230, 230, 230))
        y += 25

    if state.quality_flags:
        y += 4
        flag_text = ",".join(state.quality_flags[:4])
        _put_text(panel, "flags", (18, y), color=(170, 170, 170))
        _put_text(panel, flag_text, (105, y), scale=0.40, color=(40, 210, 255))
        y += 25

    if state.trajectory_pose is not None:
        traj = np.asarray(state.trajectory_pose, dtype=np.float32).reshape(6)
        y += 6
        _put_text(panel, "traj xyz", (18, y), color=(170, 170, 170))
        _put_text(panel, f"{traj[0]:+.2f} {traj[1]:+.2f} {traj[2]:+.2f}", (105, y), scale=0.45, color=(230, 230, 230))
        y += 25
        _put_text(panel, "traj rpy", (18, y), color=(170, 170, 170))
        _put_text(panel, f"{traj[3]:+.2f} {traj[4]:+.2f} {traj[5]:+.2f}", (105, y), scale=0.45, color=(230, 230, 230))
        y += 10

    if state.trajectory_points:
        box_x, box_y = 18, min(y + 8, preview_h - 140)
        box_w, box_h = panel_w - 36, 110
        cv2.rectangle(panel, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 50), 1)
        _put_text(panel, "wrist XY trajectory", (box_x + 8, box_y + 20), scale=0.42, color=(180, 180, 180))
        pts = np.asarray(state.trajectory_points, dtype=np.float32)
        xy = pts[:, :2]
        center = xy[-1]
        scale = max(float(np.max(np.abs(xy - center))), 0.05)
        px = box_x + box_w // 2 + ((xy[:, 0] - center[0]) / scale * (box_w * 0.42)).astype(np.int32)
        py = box_y + box_h // 2 - ((xy[:, 1] - center[1]) / scale * (box_h * 0.35)).astype(np.int32)
        for idx in range(1, len(px)):
            cv2.line(panel, (int(px[idx - 1]), int(py[idx - 1])), (int(px[idx]), int(py[idx])), (80, 200, 255), 1)
        cv2.circle(panel, (int(px[-1]), int(py[-1])), 4, (30, 240, 120), -1)
        y = box_y + box_h + 4

    y += 10
    writer_rows = [
        ("jpg pending", state.writer_pending),
        ("jpg written", state.writer_written),
        ("blocked puts", state.writer_blocked),
        ("jpg errors", state.writer_errors),
    ]
    for label, value in writer_rows:
        ok = value in (None, 0) if label in ("blocked puts", "jpg errors") else True
        _put_text(panel, label, (18, y), color=(170, 170, 170))
        _put_text(panel, "n/a" if value is None else str(value), (150, y), color=_status_color(ok))
        y += 25

    if state.status_text:
        _put_text(panel, state.status_text, (18, preview_h - 24), scale=0.48, color=(210, 210, 210))
    return np.concatenate([left, panel], axis=1)


def _build_metadata(
    config: CollectConfig,
    rgb_sensor: MVSSensor | None,
    pico_first_view_sensor: PicoFirstViewSensor | None,
    jpg_writer: AsyncJpgWriter | None,
    rotation_postmultiply,
    episode_index: int,
    quality_stats: EpisodeQualityStats | None,
    fusion_runtime: dict[str, Any] | None,
) -> dict:
    rgb_meta = None if rgb_sensor is None else rgb_sensor.metadata()
    image_shape = (
        None
        if rgb_sensor is None
        else [int(rgb_sensor.config.height), int(rgb_sensor.config.width), 3]
    )
    metadata = {
        "taskName": config.task_name,
        "configPath": (
            None if config.config_path is None else str(config.config_path)
        ),
        "collectionHz": float(config.hz),
        "cacheHz": max(float(config.cache_hz), float(config.hz)),
        "hand": config.hand,
        "episodeIndex": int(episode_index),
        "recordControl": {
            "mode": "interactive_s_toggle",
            "toggleKey": "s",
            "quitKey": "q",
            "debounceSeconds": float(config.key_debounce),
            "durationLimitSeconds": (
                None if config.duration is None else float(config.duration)
            ),
        },
        "trajectoryPose": {
            "shape": [6],
            "dtype": "float32",
            "representation": "[x, y, z, rx, ry, rz], rotation vector in radians",
            "origin": "PICO wrist point, raw26x7 index 1",
            "raw_local_axes": {
                "x": "right",
                "y": "up",
                "z": "backward",
            },
            "stored_local_axes": {
                "x": "up",
                "y": "right",
                "z": "forward",
            },
            "rotation_postmultiply": rotation_postmultiply.astype("float32"),
        },
        "pts21_mano": {
            "shape": [21, 3],
            "dtype": "float32",
            "source": "adaptive fused PICO + wrist RGB model, wrist-centered MANO frame",
        },
        "fusion": {
            "enabled": True,
            "algorithm": "adaptive_constant_velocity_kalman_with_bone_projection_v1",
            "checkpoint": str(getattr(config, "fusion_checkpoint", "")),
            "modelType": str(getattr(config, "fusion_model_type", "auto")),
            "device": str(getattr(config, "fusion_device", "auto")),
            "precision": str(getattr(config, "fusion_precision", "auto")),
            "picoStdMm": float(getattr(config, "fusion_pico_std_mm", 5.0)),
            "visionStdMm": float(getattr(config, "fusion_vision_std_mm", 14.0)),
            "innovationGateMm": float(getattr(config, "fusion_innovation_gate_mm", 22.0)),
            "processAccelMps2": float(getattr(config, "fusion_process_accel_mps2", 2.5)),
            "wujiMaxeval": int(getattr(config, "fusion_wuji_maxeval", 40)),
            "torchThreads": int(getattr(config, "fusion_torch_threads", 2)),
            "pipelineQueueSize": int(getattr(config, "fusion_pipeline_queue_size", 32)),
            "visionInferenceStride": int(getattr(config, "fusion_vision_stride", 2)),
            "cpuAffinity": str(getattr(config, "fusion_cpu_affinity", "0-11")),
            "runtime": dict(fusion_runtime or {}),
            "trajectoryPoseSource": "PICO only; wrist RGB model is local/wrist-centered",
            "safetyFallback": "PICO valid is required; RGB/model failure falls back to PICO-only local points",
            "pts21Field": "pts21_mano contains fused points",
            "sourceFields": ["pico_pts21_mano", "wrist_pts21_mano"],
            "weightFields": ["fusionPicoWeights", "fusionVisionWeights"],
        },
        "raw26x7": {
            "shape": [26, 7],
            "dtype": "float32",
            "source": "raw PICO hand state, preserved before retargeting",
        },
        "picoActive": {
            "dtype": "int",
            "meaning": "PICO active flag returned with raw26x7",
        },
        "valid_mask": {
            "shape": [21],
            "dtype": "bool",
            "meaning": "per-keypoint validity for pts21_mano; current valid frames are all True",
        },
        "sampleQuality": {
            "field": "sampleQuality",
            "flagsField": "qualityFlags",
            "meaning": (
                "per-sample machine-readable quality annotations for PICO "
                "validity, RGB availability, RGB FPS drops, frame filling, "
                "frame gaps, and RGB/PICO alignment warnings"
            ),
        },
        "o6_command": {
            "shape": [6],
            "dtype": "uint8",
            "range": [0, 255],
            "joints": list(LINKER_JOINTS),
        },
        "wuji_command": {
            "shape": [5, 4],
            "dtype": "float32",
            "unit": "radian",
            "flatShapeBeforeSave": [20],
            "layout": "qpos.reshape(5, 4), finger-major order",
            "joints": list(WUJI_JOINTS),
            "retargetingYaml": config.wuji_retargeting_yaml,
        },
        "rgb": {
            "source": "mvs_cpp" if rgb_sensor is not None else None,
            "enabled": rgb_sensor is not None,
            "storage": "jpg files saved in the bundle images/ subdirectory",
            "imageField": "rgbImage",
            "imageFieldValue": "relative path string or None",
            "syncMode": config.mvs_sync_mode,
            "captureFps": float(config.mvs_capture_fps or config.hz),
            "maxRgbPicoDeltaMs": float(config.max_rgb_pico_delta_ms),
            "maxRgbPicoDeltaMeaning": (
                "quality warning threshold in mvs_buffered/mvs_master mode; "
                "hard missing threshold only in nearest_pico_receive mode"
            ),
            "writer": writer_stats(jpg_writer),
            "writerErrors": writer_errors(jpg_writer),
            "imageShape": image_shape,
            "frameIdField": "rgbFrameId",
            "captureNsField": "rgbCaptureNs",
            "captureNsMeaning": "host time.monotonic_ns when the MVS backend acquired the frame",
            "syncTargetNsField": "rgbSyncTargetNs",
            "syncDeltaNsField": "rgbSyncDeltaNs",
            "syncDeltaNsMeaning": (
                "rgbCaptureNs - rgbSyncTargetNs; in mvs_buffered the target is "
                "the fixed collection sample time, in mvs_master the target is "
                "the selected MVS capture time or reuse target; PICO is chosen "
                "nearest to that target"
            ),
            "deviceTimestampTicksField": "rgbDeviceTimestampTicks",
            "hostTimestampMsField": "rgbHostTimestampMs",
            "alignResidualNsField": "rgbAlignResidualNs",
            "alignResidualNsMeaning": "sampleClockNs - rgbCaptureNs",
            "rgbToPicoReceiveDeltaNsField": "rgbToPicoReceiveDeltaNs",
            "rgbToPicoReceiveDeltaNsMeaning": "rgbCaptureNs - sourceReceiveNs",
            "absRgbToPicoReceiveDeltaNsField": "absRgbToPicoReceiveDeltaNs",
            "rgbMissingReasonField": "rgbMissingReason",
            "rgbQualityWarningField": "rgbQualityWarning",
            "rgbFrameFilledField": "rgbFrameFilled",
            "rgbFrameFillReasonField": "rgbFrameFillReason",
            "frameRepeatedField": "rgbFrameRepeated",
            "frameRepeatedMeaning": "True when this sample reused the same MVS frame as the previous sample",
            "frameGapField": "rgbFrameGap",
            "frameGapMeaning": "max(0, current rgbFrameId - previous returned rgbFrameId - 1)",
            "framesReadField": "rgbFramesRead",
            "mvs": rgb_meta,
        },
        "qualityStats": (
            EpisodeQualityStats().as_metadata()
            if quality_stats is None
            else quality_stats.as_metadata()
        ),
    }
    if pico_first_view_sensor is not None:
        metadata["picoFirstView"] = {
            **pico_first_view_sensor.metadata(),
            "enabled": True,
            "storage": "jpg files saved in the bundle pico_first_view/ subdirectory",
            "imageField": "picoFirstViewImage",
            "imageFieldValue": "relative path string or None",
            "frameIdField": "picoFirstViewFrameId",
            "captureNsField": "picoFirstViewCaptureNs",
            "alignResidualNsField": "picoFirstViewAlignResidualNs",
            "missingReasonField": "picoFirstViewMissingReason",
        }
    return metadata


def _save_episode(
    messages: list[dict],
    pkl_path: Path,
    config: CollectConfig,
    rgb_sensor: MVSSensor | None,
    pico_first_view_sensor: PicoFirstViewSensor | None,
    jpg_writer: AsyncJpgWriter | None,
    rotation_postmultiply,
    episode_index: int,
    quality_stats: EpisodeQualityStats | None,
    fusion_runtime: dict[str, Any] | None,
) -> bool:
    valid_count = sum(1 for m in messages if m["o6_command"] is not None)
    total_count = len(messages)
    drop_rate = (1 - valid_count / max(total_count, 1)) * 100.0
    print(
        f"[Done] 第 {episode_index} 段共采集 {total_count} 帧，"
        f"有效帧 {valid_count} 帧，丢弃率 {drop_rate:.1f}%"
    )

    if jpg_writer is not None:
        print("[Save] 等待 RGB jpg 队列写完...")
        jpg_writer.wait_empty()

    if valid_count == 0:
        print("[Warn] 本段无有效帧，不写入 PKL")
        return False

    if config.dry_run:
        print("[DryRun] 本段不写入文件")
        return False

    if jpg_writer is not None and jpg_writer.errors:
        print(
            f"[Warn] RGB jpg 写入失败 {len(jpg_writer.errors)} 帧，"
            "详情写入 metadata['rgb']['writerErrors']"
        )

    metadata = _build_metadata(
        config=config,
        rgb_sensor=rgb_sensor,
        pico_first_view_sensor=pico_first_view_sensor,
        jpg_writer=jpg_writer,
        rotation_postmultiply=rotation_postmultiply,
        episode_index=episode_index,
        quality_stats=quality_stats,
        fusion_runtime=fusion_runtime,
    )
    save_pkl(messages, pkl_path, metadata=metadata)
    return True


def collect(config: CollectConfig) -> None:
    from retargeter import (  # noqa: WPS433
        PICO_WRIST_TO_TARGET_TCP_ROT,
    )

    pico_sensor: PicoSensor | None = None
    rgb_sensor: MVSSensor | None = None
    pico_first_view_sensor: PicoFirstViewSensor | None = None
    retargeter: FusedPicoWristRetargeter | None = None
    pipeline: AsyncFusedPipeline | None = None
    key_watcher: KeypressWatcher | None = None
    stop_event = threading.Event()
    recording = False
    episode_messages: list[dict] = []
    preview_enabled = not bool(getattr(config, "no_preview", False))
    preview_window = "DexUMI collect"
    preview_state = CollectUiState()
    discarded_episodes = 0
    fps_loop_t0 = time.perf_counter()
    fps_loop_count = 0
    fps_saved_t0 = time.perf_counter()
    fps_saved_count = 0
    trajectory_history: deque[tuple[float, float, float]] = deque(maxlen=240)

    try:
        print("[Init] 初始化 PICO 数据采集...")
        pico_sensor = PicoSensor(
            hand=config.hand,
            poll_hz=max(float(config.cache_hz), float(config.hz)),
        )

        if config.pico_first_view_enabled:
            print(
                "[Init] 初始化 Pico 第一视角采集："
                f"device={config.pico_first_view_device!r}"
            )
            pico_first_view_sensor = PicoFirstViewSensor(
                device=config.pico_first_view_device,
                fps=float(config.pico_first_view_fps or config.hz),
                width=config.pico_first_view_width,
                height=config.pico_first_view_height,
            )
            pico_first_view_sensor.start()

        print("[Init] 加载 PICO + wrist RGB 融合模型和 retargeting...")
        retargeter = FusedPicoWristRetargeter(
            checkpoint=Path(getattr(config, "fusion_checkpoint")),
            mano_model_dir=Path(getattr(config, "fusion_mano_model_dir")),
            model_type=str(getattr(config, "fusion_model_type", "auto")),
            wrist_config=getattr(config, "fusion_wrist_config", None),
            dino_dir=getattr(config, "fusion_dino_dir", None),
            device=str(getattr(config, "fusion_device", "auto")),
            precision=str(getattr(config, "fusion_precision", "auto")),
            yaml_path=config.retargeting_yaml,
            wuji_yaml_path=config.wuji_retargeting_yaml,
            hand=config.hand,
            warmup_iters=int(getattr(config, "fusion_warmup_iters", 20)),
            pico_std_mm=float(getattr(config, "fusion_pico_std_mm", 5.0)),
            vision_std_mm=float(getattr(config, "fusion_vision_std_mm", 14.0)),
            innovation_gate_mm=float(getattr(config, "fusion_innovation_gate_mm", 22.0)),
            process_accel_mps2=float(getattr(config, "fusion_process_accel_mps2", 2.5)),
            wuji_maxeval=int(getattr(config, "fusion_wuji_maxeval", 40)),
            torch_threads=int(getattr(config, "fusion_torch_threads", 2)),
        )
        pipeline = AsyncFusedPipeline(
            retargeter,
            max_queue=int(getattr(config, "fusion_pipeline_queue_size", 32)),
            vision_stride=int(getattr(config, "fusion_vision_stride", 2)),
        )
        pipeline.start()
        print(
            f"[Init] fusion model_type={retargeter.model_type} "
            f"device={retargeter.device} precision={retargeter.precision} "
            f"async_pipeline_queue={pipeline.max_queue} "
            f"vision_stride={pipeline.vision_stride}"
        )

        try:
            rgb_sensor = MVSSensor(build_mvs_config(config))
            rgb_sensor.start()
        except Exception as exc:
            if not config.dry_run:
                raise
            print(f"[DryRun] MVS 初始化失败，降级为只跑 PICO: {exc}")
            if rgb_sensor is not None:
                rgb_sensor.stop()
            rgb_sensor = None

        pico_sensor.start()
        period = 1.0 / max(float(config.hz), 1.0)

        def _sigint(_sig, _frame):
            print("\n[Ctrl+C] 请求退出；若正在录制，会先保存当前段...")
            stop_event.set()

        signal.signal(signal.SIGINT, _sigint)
        key_watcher = KeypressWatcher(stop_event)
        key_watcher.start()

        print(
            "[Mode] 交互模式：按 s 开始/暂停保存；录制中按 Enter 丢弃当前 episode；"
            "按 q 或 Ctrl+C 退出；"
            f"s 键防误触间隔 {config.key_debounce:.2f}s"
        )
        if config.duration is not None:
            print(
                f"[Mode] duration={config.duration:.1f}s：每段达到时长后自动暂停保存，"
                "程序继续等待下一次 s"
            )
        if preview_enabled:
            try:
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(preview_window, 960, 620)
                print("[Mode] 预览窗口已开启：窗口内也可按 s / Enter / q")
            except Exception as exc:
                preview_enabled = False
                print(f"[Warn] OpenCV 预览窗口启动失败，已关闭预览: {exc}")

        episode_index = 0
        episode_pkl: Path | None = None
        jpg_writer: AsyncJpgWriter | None = None
        episode_quality_stats: EpisodeQualityStats | None = None
        frame_idx = 0
        miss_count = 0
        episode_start_perf = 0.0
        last_toggle_perf = 0.0
        warning_interval = max(int(round(float(config.hz))), 1)
        progress_interval = max(warning_interval * 5, 1)
        max_rgb_pico_delta_ns = int(
            round(max(float(config.max_rgb_pico_delta_ms), 0.0) * 1_000_000.0)
        )
        wait_timeout_s = max(0.2, 2.0 / max(float(config.hz), 1.0))
        last_rgb_frame_id: int | None = None
        last_good_rgb_frame = None
        last_rgb_capture_ns: int | None = None
        rgb_capture_intervals_ms: deque[float] = deque(maxlen=30)
        period_ns = int(round(period * 1_000_000_000.0))
        next_sample_target_ns: int | None = None
        rgb_cache_fps_t0 = time.perf_counter()
        rgb_cache_frames_t0 = 0
        completed_messages: dict[int, dict | None] = {}
        next_message_frame_idx = 0
        episode_valid_messages = 0
        pipeline_error_count = 0

        def _flush_completed_messages() -> None:
            nonlocal next_message_frame_idx
            nonlocal fps_saved_count
            nonlocal episode_valid_messages
            while next_message_frame_idx in completed_messages:
                msg = completed_messages.pop(next_message_frame_idx)
                next_message_frame_idx += 1
                if msg is None:
                    continue
                episode_messages.append(msg)
                fps_saved_count += 1
                if msg.get("o6_command") is not None:
                    episode_valid_messages += 1

        def _drain_pipeline_results() -> None:
            nonlocal pipeline_error_count
            assert pipeline is not None
            for completion in pipeline.drain():
                context = completion.context
                if not isinstance(context, CapturedFusionSample):
                    raise RuntimeError("pipeline completion 缺少 CapturedFusionSample")
                quality_payload = dict(context.quality_payload)
                quality_payload["flags"] = list(quality_payload.get("flags", []))
                if completion.error is not None or completion.result is None:
                    pipeline_error_count += 1
                    quality_payload["flags"].append("fusion_pipeline_failed")
                    msg = build_invalid_message(
                        sdk_ts_ns=context.sdk_ts_ns,
                        sample_ns=context.sample_ns,
                        source_receive_ns=context.source_receive_ns,
                        raw26x7=context.raw26x7,
                        pico_active=context.pico_active,
                        valid_mask=[False] * 21,
                        rgb_frame=context.rgb_frame,
                        rgb_missing_reason=context.rgb_missing_reason,
                        rgb_image_path=context.rgb_image_path,
                        pico_first_view_frame=context.pico_first_view_frame,
                        pico_first_view_enabled=config.pico_first_view_enabled,
                        pico_first_view_image_path=context.pico_first_view_image_path,
                        pico_first_view_missing_reason=context.pico_first_view_missing_reason,
                        jpg_writer=jpg_writer,
                        quality_payload=quality_payload,
                    )
                    msg["fusionPipelineError"] = completion.error
                    if pipeline_error_count <= 3 or pipeline_error_count % warning_interval == 0:
                        print(
                            f"[Warn] fusion pipeline frame={context.frame_idx} "
                            f"failed: {completion.error}"
                        )
                else:
                    retargeted = completion.result
                    wuji_qpos = retargeted.wuji_joint_radians
                    if wuji_qpos is None:
                        raise RuntimeError("Wuji retargeting 未返回 qpos")
                    msg = build_message(
                        o6_angles=retargeted.linker_joint_radians,
                        wuji_qpos=wuji_qpos,
                        pts21_mano=retargeted.pts21_mano,
                        wrist_pose_6d=retargeted.wrist_pose_6d,
                        sdk_ts_ns=context.sdk_ts_ns,
                        sample_ns=context.sample_ns,
                        source_receive_ns=context.source_receive_ns,
                        raw26x7=context.raw26x7,
                        pico_active=context.pico_active,
                        valid_mask=[True] * 21,
                        rgb_frame=context.rgb_frame,
                        rgb_missing_reason=context.rgb_missing_reason,
                        rgb_image_path=context.rgb_image_path,
                        pico_first_view_frame=context.pico_first_view_frame,
                        pico_first_view_enabled=config.pico_first_view_enabled,
                        pico_first_view_image_path=context.pico_first_view_image_path,
                        pico_first_view_missing_reason=context.pico_first_view_missing_reason,
                        jpg_writer=jpg_writer,
                        quality_payload=quality_payload,
                    )
                    msg.update(retargeted.message_fields())
                    if retargeted.wrist_inference_error is not None:
                        msg["qualityFlags"] = list(msg.get("qualityFlags", [])) + [
                            "wrist_inference_failed"
                        ]
                        msg.setdefault("sampleQuality", {}).setdefault("flags", []).append(
                            "wrist_inference_failed"
                        )
                    if context.frame_idx % 3 == 0:
                        preview_state.mean_bone_mm, preview_state.max_bone_mm = (
                            _bone_stats_mm(retargeted.pts21_mano)
                        )
                    preview_state.trajectory_pose = np.asarray(
                        retargeted.wrist_pose_6d,
                        dtype=np.float32,
                    ).reshape(6)
                    trajectory_history.append(
                        tuple(float(v) for v in preview_state.trajectory_pose[:3])
                    )
                    preview_state.trajectory_points = list(trajectory_history)
                completed_messages[context.frame_idx] = msg
            _flush_completed_messages()

        def _wait_and_drain_pipeline() -> dict[str, Any]:
            assert pipeline is not None
            pipeline.wait_empty()
            _drain_pipeline_results()
            if completed_messages:
                _flush_completed_messages()
            if next_message_frame_idx != frame_idx:
                raise RuntimeError(
                    "pipeline flush 后帧序号不连续: "
                    f"emitted={next_message_frame_idx} captured={frame_idx}"
                )
            return pipeline.stats()

        def _select_rgb_frame(source_receive_ns: int):
            if rgb_sensor is None:
                return None, "rgb_sensor_unavailable"
            if config.mvs_sync_mode in {"mvs_buffered", "mvs_master"}:
                return rgb_sensor.latest(copy_image=True), None
            if config.mvs_sync_mode == "nearest_pico_receive":
                rgb_frame = rgb_sensor.nearest(
                    target_ns=source_receive_ns,
                    copy_image=True,
                    max_delta_ns=max_rgb_pico_delta_ns,
                )
                if not _rgb_frame_valid(rgb_frame):
                    return rgb_frame, "no_rgb_within_max_delta"
                return rgb_frame, None

            rgb_frame = rgb_sensor.latest(copy_image=True)
            if not _rgb_frame_valid(rgb_frame):
                return rgb_frame, "no_rgb_frame_available"
            return rgb_frame, None

        def _start_episode() -> None:
            nonlocal recording
            nonlocal episode_index
            nonlocal episode_messages
            nonlocal episode_pkl
            nonlocal jpg_writer
            nonlocal episode_quality_stats
            nonlocal frame_idx
            nonlocal miss_count
            nonlocal episode_start_perf
            nonlocal discarded_episodes
            nonlocal fps_saved_t0
            nonlocal fps_saved_count
            nonlocal trajectory_history
            nonlocal last_rgb_frame_id
            nonlocal last_good_rgb_frame
            nonlocal last_rgb_capture_ns
            nonlocal rgb_capture_intervals_ms
            nonlocal next_sample_target_ns
            nonlocal rgb_cache_fps_t0
            nonlocal rgb_cache_frames_t0
            nonlocal completed_messages
            nonlocal next_message_frame_idx
            nonlocal episode_valid_messages
            nonlocal pipeline_error_count

            episode_pkl = episode_pkl_path(config.out, episode_index)
            episode_index = episode_index_from_path(episode_pkl)
            episode_messages = []
            episode_quality_stats = EpisodeQualityStats()
            frame_idx = 0
            miss_count = 0
            fps_saved_t0 = time.perf_counter()
            fps_saved_count = 0
            completed_messages = {}
            next_message_frame_idx = 0
            episode_valid_messages = 0
            pipeline_error_count = 0
            trajectory_history.clear()
            preview_state.trajectory_pose = None
            preview_state.trajectory_points = []
            last_rgb_frame_id = None
            last_good_rgb_frame = None
            last_rgb_capture_ns = None
            rgb_capture_intervals_ms.clear()
            next_sample_target_ns = None
            rgb_cache_fps_t0 = time.perf_counter()
            rgb_cache_frames_t0 = 0
            if rgb_sensor is not None:
                try:
                    rgb_sensor.reset_return_tracking()
                    rgb_cache_frames_t0 = int(rgb_sensor.stats().get("framesRead", 0))
                except Exception:
                    rgb_cache_frames_t0 = 0
            assert pipeline is not None
            pipeline.reset()
            episode_start_perf = time.perf_counter()
            if not config.dry_run:
                jpg_writer = AsyncJpgWriter(
                    quality=config.jpg_quality,
                    max_queue=config.jpg_writer_queue_size,
                    num_workers=config.jpg_writer_workers,
                )
                jpg_writer.start()
            else:
                jpg_writer = None
            recording = True
            rec_button = record_button("● REC START", "green")
            rec_symbol = record_symbol("●", "green")
            print(
                f"\n{rec_button} [Rec] {rec_symbol} 第 {episode_index} 段开始  "
                f"task={config.task_name} hz={config.hz} "
                f"cache_hz={config.cache_hz} out={episode_pkl}"
            )
            if jpg_writer is not None:
                print(
                    "[Rec] RGB JPG 异步写盘: "
                    f"workers={config.jpg_writer_workers} "
                    f"queue={config.jpg_writer_queue_size} "
                    f"quality={config.jpg_quality}"
                )

        def _stop_and_save_episode(reason: str) -> None:
            nonlocal recording
            nonlocal episode_messages
            nonlocal episode_pkl
            nonlocal jpg_writer
            nonlocal episode_quality_stats
            nonlocal frame_idx
            if not recording and not episode_messages:
                return
            recording = False
            stop_button = record_button("■ REC STOP", "red")
            stop_symbol = record_symbol("■", "red")
            print(
                f"\n{stop_button} [Rec] {stop_symbol} 第 {episode_index} 段暂停，"
                f"原因={reason}，正在排空异步流水线并保存..."
            )
            fusion_runtime = _wait_and_drain_pipeline()
            writer_to_close = jpg_writer
            jpg_writer = None

            if episode_pkl is not None:
                _save_episode(
                    messages=episode_messages,
                    pkl_path=episode_pkl,
                    config=config,
                    rgb_sensor=rgb_sensor,
                    pico_first_view_sensor=pico_first_view_sensor,
                    jpg_writer=writer_to_close,
                    rotation_postmultiply=PICO_WRIST_TO_TARGET_TCP_ROT,
                    episode_index=episode_index,
                    quality_stats=episode_quality_stats,
                    fusion_runtime=fusion_runtime,
                )
            if writer_to_close is not None:
                writer_to_close.stop()

            episode_messages = []
            episode_pkl = None
            episode_quality_stats = None
            frame_idx = 0
            while key_watcher is not None:
                try:
                    stale_ch = key_watcher.events.get_nowait()
                except queue.Empty:
                    break
                if stale_ch in ("q", "\x03", "\x04"):
                    stop_event.set()
            print("[Mode] 已回到待机；按 s 开始下一段，按 q 或 Ctrl+C 退出")

        def _discard_episode(reason: str) -> None:
            nonlocal recording
            nonlocal episode_messages
            nonlocal episode_pkl
            nonlocal jpg_writer
            nonlocal episode_quality_stats
            nonlocal frame_idx
            nonlocal miss_count
            nonlocal discarded_episodes
            nonlocal trajectory_history
            nonlocal completed_messages
            nonlocal next_message_frame_idx
            nonlocal episode_valid_messages
            nonlocal pipeline_error_count

            if not recording and not episode_messages:
                print("[Key] 当前未录制，Enter 丢弃请求已忽略")
                return

            recording = False
            assert pipeline is not None
            pipeline.wait_empty()
            pipeline.drain()
            discard_dir = None if episode_pkl is None else episode_pkl.parent
            writer_to_close = jpg_writer
            jpg_writer = None
            discarded_episodes += 1
            preview_state.alert_text = f"DISCARDED EPISODE {episode_index:04d}"
            print(
                f"\n[Discard] 丢弃第 {episode_index} 段，原因={reason}，"
                "不会保存 PKL"
            )

            if writer_to_close is not None:
                writer_to_close.stop()

            if discard_dir is not None and discard_dir.exists():
                try:
                    shutil.rmtree(discard_dir)
                    print(f"[Discard] 已删除 episode 目录: {discard_dir}")
                except Exception as exc:
                    print(f"[Warn] 删除 episode 目录失败 {discard_dir}: {exc}")

            episode_messages = []
            episode_pkl = None
            episode_quality_stats = None
            frame_idx = 0
            miss_count = 0
            completed_messages = {}
            next_message_frame_idx = 0
            episode_valid_messages = 0
            pipeline_error_count = 0
            trajectory_history.clear()
            preview_state.trajectory_pose = None
            preview_state.trajectory_points = []
            preview_state.writer_pending = None
            preview_state.writer_written = None
            preview_state.writer_blocked = None
            preview_state.writer_errors = None
            print("[Mode] 已回到待机；按 s 重新开始本段，按 q 或 Ctrl+C 退出")

        def _handle_control_key(ch: str) -> None:
            nonlocal last_toggle_perf
            if ch == "s":
                now = time.perf_counter()
                if now - last_toggle_perf < float(config.key_debounce):
                    print("[Key] 忽略过快的 s，防止误触")
                    return
                last_toggle_perf = now
                if recording:
                    _stop_and_save_episode("s_key")
                else:
                    _start_episode()
            elif ch in ("\n", "\r"):
                _discard_episode("enter_key")
            elif ch in ("q", "\x03", "\x04"):
                print("\n[Key] 请求退出")
                stop_event.set()
            elif ch not in (" "):
                print("[Key] 按 s 开始/暂停保存，Enter 丢弃当前 episode，按 q 退出")

        def _handle_key_events() -> None:
            while key_watcher is not None:
                try:
                    ch = key_watcher.events.get_nowait()
                except queue.Empty:
                    return
                _handle_control_key(ch)

        def _handle_preview_key() -> None:
            if not preview_enabled:
                return
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                return
            if key in (ord("s"), ord("S")):
                _handle_control_key("s")
            elif key in (13, 10):
                _handle_control_key("\n")
            elif key in (27, ord("q"), ord("Q")):
                _handle_control_key("q")

        def _refresh_preview(rgb_frame=None) -> None:
            if not preview_enabled:
                return
            preview_state.recording = recording
            preview_state.episode_index = int(episode_index)
            preview_state.frame_idx = int(frame_idx)
            preview_state.saved_messages = len(episode_messages)
            preview_state.valid_messages = int(episode_valid_messages)
            preview_state.discarded_episodes = int(discarded_episodes)
            preview_state.status_text = (
                f"out={episode_pkl}" if recording and episode_pkl is not None else "待机中，按 s 开始录制"
            )
            if jpg_writer is not None:
                stats = writer_stats(jpg_writer)
                if stats is not None:
                    preview_state.writer_pending = int(stats.get("pending", 0))
                    preview_state.writer_written = int(stats.get("written", 0))
                    preview_state.writer_blocked = int(stats.get("blockedPuts", 0))
                    preview_state.writer_errors = int(stats.get("errorCount", 0))
            else:
                preview_state.writer_pending = None
                preview_state.writer_written = None
                preview_state.writer_blocked = None
                preview_state.writer_errors = None
            if rgb_frame is None and rgb_sensor is not None:
                try:
                    rgb_frame = rgb_sensor.latest(
                        copy_image=True,
                        count_return=False,
                    )
                except Exception:
                    rgb_frame = None
            if not recording:
                preview_state.rgb_valid = _rgb_frame_valid(rgb_frame)
                preview_state.rgb_missing_reason = None if preview_state.rgb_valid else "no_rgb_frame_available"
                preview_state.rgb_frame_id = None if rgb_frame is None else rgb_frame.frame_id
                preview_state.rgb_frame_gap = None if rgb_frame is None else rgb_frame.frame_gap
                preview_state.rgb_repeated = None if rgb_frame is None else rgb_frame.is_repeated
            vis = _draw_collect_preview(rgb_frame, preview_state)
            cv2.imshow(preview_window, vis)
            _handle_preview_key()

        if config.duration is not None:
            _start_episode()
        else:
            print("  (待机中，按 s 开始录制)")

        while not stop_event.is_set():
            _handle_key_events()
            if stop_event.is_set():
                break
            if not recording:
                preview_state.valid_pico = False
                preview_state.rgb_valid = False
                preview_state.rgb_missing_reason = None
                preview_state.alert_text = None
                _refresh_preview()
                time.sleep(0.03 if preview_enabled else 0.05)
                continue

            t0 = time.perf_counter()
            sample_ns = time.monotonic_ns()
            fps_loop_count += 1
            fps_elapsed = time.perf_counter() - fps_loop_t0
            if fps_elapsed >= 1.0:
                preview_state.fps_loop = fps_loop_count / max(fps_elapsed, 1e-6)
                fps_loop_t0 = time.perf_counter()
                fps_loop_count = 0

            if config.duration is not None:
                if (t0 - episode_start_perf) >= config.duration:
                    _stop_and_save_episode("duration")
                    continue

            assert pico_sensor is not None
            rgb_missing_reason = None
            pico_first_view_missing_reason = None
            if config.mvs_sync_mode == "mvs_buffered" and rgb_sensor is not None:
                if next_sample_target_ns is None:
                    next_sample_target_ns = time.monotonic_ns()
                target_ns = int(next_sample_target_ns)
                next_sample_target_ns += period_ns
                now_ns = time.monotonic_ns()
                if next_sample_target_ns < now_ns - period_ns:
                    missed = (now_ns - next_sample_target_ns) // max(period_ns, 1) + 1
                    next_sample_target_ns += int(missed) * period_ns

                rgb_frame = rgb_sensor.nearest(
                    target_ns=target_ns,
                    copy_image=True,
                    max_delta_ns=None,
                    count_return=True,
                )
                if _rgb_frame_valid(rgb_frame):
                    last_good_rgb_frame = rgb_frame
                    rgb_frame = _with_rgb_sync(
                        rgb_frame,
                        target_ns=target_ns,
                        selection_mode="mvs_buffered_nearest_capture_ns",
                    )
                else:
                    rgb_frame = _reused_rgb_frame(
                        last_good_rgb_frame,
                        target_ns=target_ns,
                    )
                    rgb_missing_reason = (
                        "mvs_buffer_empty_reused_previous"
                        if _rgb_frame_valid(rgb_frame)
                        else "mvs_buffer_empty_no_previous_rgb"
                    )
                cached = pico_sensor.nearest(target_ns)
                sample_ns = target_ns
            elif config.mvs_sync_mode == "mvs_master" and rgb_sensor is not None:
                try:
                    rgb_frame = rgb_sensor.wait_next(
                        last_frame_id=last_rgb_frame_id,
                        timeout_s=wait_timeout_s,
                        copy_image=True,
                        count_return=True,
                    )
                    if rgb_frame.frame_id is not None:
                        last_rgb_frame_id = int(rgb_frame.frame_id)
                    last_good_rgb_frame = rgb_frame if _rgb_frame_valid(rgb_frame) else last_good_rgb_frame
                    target_ns = (
                        int(rgb_frame.capture_ns)
                        if _rgb_frame_valid(rgb_frame)
                        else time.monotonic_ns()
                    )
                    rgb_frame = _with_rgb_sync(
                        rgb_frame,
                        target_ns=target_ns,
                        selection_mode="mvs_master_capture_ns",
                    )
                except TimeoutError:
                    target_ns = time.monotonic_ns()
                    rgb_frame = _reused_rgb_frame(
                        last_good_rgb_frame,
                        target_ns=target_ns,
                    )
                    rgb_missing_reason = (
                        "mvs_timeout_reused_previous"
                        if _rgb_frame_valid(rgb_frame)
                        else "mvs_timeout_no_previous_rgb"
                    )
                cached = pico_sensor.nearest(target_ns)
                sample_ns = target_ns
            else:
                cached = pico_sensor.latest()
                source_receive_ns = cached.receive_ns
                rgb_frame, rgb_missing_reason = _select_rgb_frame(source_receive_ns)

            pico_first_view_frame = None
            if pico_first_view_sensor is not None:
                pico_first_view_frame = pico_first_view_sensor.nearest(sample_ns)
                if (
                    pico_first_view_frame.image is None
                    or pico_first_view_frame.capture_ns is None
                ):
                    pico_first_view_missing_reason = (
                        "no_pico_first_view_frame_available"
                    )

            raw26x7 = cached.raw26x7
            active = cached.active
            sdk_ts_ns = cached.sdk_ts_ns
            source_receive_ns = cached.receive_ns
            valid_pico = active == 1 and raw26x7 is not None and raw26x7.shape == (26, 7)
            rgb_valid = _rgb_frame_valid(rgb_frame)
            rgb_pico_delta_large = False
            if (
                rgb_valid
                and config.mvs_sync_mode in {"mvs_buffered", "mvs_master"}
                and source_receive_ns != 0
            ):
                delta_ns = abs(int(rgb_frame.capture_ns) - int(source_receive_ns))
                if delta_ns > max_rgb_pico_delta_ns:
                    rgb_pico_delta_large = True
                    if rgb_missing_reason is None:
                        rgb_missing_reason = "pico_rgb_delta_exceeds_warning_threshold"
            rgb_dt_ms = None
            if rgb_valid:
                capture_ns = int(rgb_frame.capture_ns)
                if last_rgb_capture_ns is not None:
                    rgb_dt_ms = (capture_ns - int(last_rgb_capture_ns)) / 1_000_000.0
                    if rgb_dt_ms >= 0:
                        rgb_capture_intervals_ms.append(float(rgb_dt_ms))
                last_rgb_capture_ns = capture_ns
            rgb_actual_fps = (
                1000.0 / (sum(rgb_capture_intervals_ms) / len(rgb_capture_intervals_ms))
                if rgb_capture_intervals_ms
                else 0.0
            )
            rgb_cache_elapsed = time.perf_counter() - rgb_cache_fps_t0
            if rgb_sensor is not None and rgb_cache_elapsed >= 1.0:
                try:
                    rgb_cache_frames_now = int(rgb_sensor.stats().get("framesRead", 0))
                    preview_state.rgb_cache_fps = (
                        (rgb_cache_frames_now - rgb_cache_frames_t0)
                        / max(rgb_cache_elapsed, 1e-6)
                    )
                    rgb_cache_frames_t0 = rgb_cache_frames_now
                    rgb_cache_fps_t0 = time.perf_counter()
                except Exception:
                    pass
            expected_capture_fps = float(config.mvs_capture_fps or config.hz)
            rgb_low_fps = bool(
                rgb_valid
                and (
                    (
                        rgb_dt_ms is not None
                        and rgb_dt_ms > (1000.0 / max(float(config.hz), 1.0)) * 1.5
                    )
                    or (
                        preview_state.rgb_cache_fps > 0.0
                        and preview_state.rgb_cache_fps < expected_capture_fps * 0.75
                    )
                )
            )
            source_age_ms = (
                None
                if source_receive_ns == 0
                else (sample_ns - int(source_receive_ns)) / 1_000_000.0
            )
            rgb_align_ms = (
                None
                if not rgb_valid
                else (sample_ns - int(rgb_frame.capture_ns)) / 1_000_000.0
            )
            rgb_to_pico_ms = (
                None
                if not rgb_valid or source_receive_ns == 0
                else (int(rgb_frame.capture_ns) - int(source_receive_ns)) / 1_000_000.0
            )
            quality_payload = _build_sample_quality(
                valid_pico=valid_pico,
                raw26x7=raw26x7,
                active=active,
                rgb_valid=rgb_valid,
                rgb_frame=rgb_frame,
                rgb_missing_reason=rgb_missing_reason,
                source_age_ms=source_age_ms,
                rgb_to_pico_ms=rgb_to_pico_ms,
                rgb_dt_ms=rgb_dt_ms,
                rgb_actual_fps=preview_state.rgb_cache_fps,
                rgb_low_fps=rgb_low_fps,
                rgb_pico_delta_large=rgb_pico_delta_large,
                max_rgb_pico_delta_ms=config.max_rgb_pico_delta_ms,
            )
            preview_state.valid_pico = bool(valid_pico)
            preview_state.rgb_valid = bool(rgb_valid)
            preview_state.rgb_missing_reason = rgb_missing_reason
            preview_state.rgb_frame_id = None if rgb_frame is None else rgb_frame.frame_id
            preview_state.rgb_frame_gap = None if rgb_frame is None else rgb_frame.frame_gap
            preview_state.rgb_repeated = None if rgb_frame is None else rgb_frame.is_repeated
            preview_state.rgb_filled = bool(quality_payload["rgb"]["frameFilled"])
            preview_state.rgb_dt_ms = rgb_dt_ms
            preview_state.rgb_actual_fps = rgb_actual_fps
            preview_state.rgb_low_fps = rgb_low_fps
            preview_state.pico_quality_label = "OK" if valid_pico else (
                "INACTIVE" if int(active) != 1 else "BAD DATA"
            )
            preview_state.quality_flags = list(quality_payload["flags"])
            preview_state.source_age_ms = source_age_ms
            preview_state.rgb_align_ms = rgb_align_ms
            preview_state.rgb_to_pico_ms = rgb_to_pico_ms
            alerts = []
            if not valid_pico:
                alerts.append("NO HAND KEYPOINTS")
            if not rgb_valid:
                alerts.append(f"RGB MISSING: {rgb_missing_reason}")
            elif getattr(rgb_frame, "frame_gap", 0):
                alerts.append(f"RGB FRAME GAP: {rgb_frame.frame_gap}")
            elif rgb_low_fps:
                alerts.append(f"RGB LOW FPS: {rgb_actual_fps:.1f}")
            elif getattr(rgb_frame, "is_repeated", False):
                alerts.append("RGB REPEATED")
            if rgb_pico_delta_large:
                alerts.append("RGB/PICO DELTA")
            if preview_state.rgb_filled:
                alerts.append("RGB FILLED")
            preview_state.alert_text = " | ".join(alerts) if alerts else None
            if episode_quality_stats is not None:
                episode_quality_stats.observe(
                    valid_pico=valid_pico,
                    sample_ns=sample_ns,
                    source_receive_ns=source_receive_ns,
                    rgb_frame=rgb_frame,
                    quality_payload=quality_payload,
                )

            _drain_pipeline_results()
            rgb_path = (
                None
                if config.dry_run or episode_pkl is None
                else rgb_image_path_for_frame(episode_pkl, frame_idx)
            )
            pico_first_view_path = (
                None
                if pico_first_view_sensor is None
                or config.dry_run
                or episode_pkl is None
                else pico_first_view_image_path_for_frame(episode_pkl, frame_idx)
            )
            if not valid_pico:
                miss_count += 1
                if miss_count % warning_interval == 0:
                    print(
                        f"[Warn] 已连续 {miss_count} 帧 ({miss_count/config.hz:.1f}s)"
                        " 未收到高质量手部数据"
                    )
                if config.keep_invalid or config.mvs_sync_mode in {"mvs_buffered", "mvs_master"}:
                    completed_messages[frame_idx] = build_invalid_message(
                        sdk_ts_ns=sdk_ts_ns,
                        sample_ns=sample_ns,
                        source_receive_ns=source_receive_ns,
                        raw26x7=raw26x7,
                        pico_active=active,
                        valid_mask=[False] * 21,
                        rgb_frame=rgb_frame,
                        rgb_missing_reason=rgb_missing_reason,
                        rgb_image_path=rgb_path,
                        pico_first_view_frame=pico_first_view_frame,
                        pico_first_view_enabled=config.pico_first_view_enabled,
                        pico_first_view_image_path=pico_first_view_path,
                        pico_first_view_missing_reason=pico_first_view_missing_reason,
                        jpg_writer=jpg_writer,
                        quality_payload=quality_payload,
                    )
                else:
                    completed_messages[frame_idx] = None
            else:
                miss_count = 0
                assert pipeline is not None
                raw_copy = np.asarray(raw26x7, dtype=np.float32).reshape(26, 7).copy()
                context = CapturedFusionSample(
                    frame_idx=frame_idx,
                    sdk_ts_ns=sdk_ts_ns,
                    sample_ns=sample_ns,
                    source_receive_ns=source_receive_ns,
                    raw26x7=raw_copy,
                    pico_active=int(active),
                    rgb_frame=rgb_frame,
                    rgb_missing_reason=rgb_missing_reason,
                    rgb_image_path=rgb_path,
                    pico_first_view_frame=pico_first_view_frame,
                    pico_first_view_missing_reason=pico_first_view_missing_reason,
                    pico_first_view_image_path=pico_first_view_path,
                    quality_payload=quality_payload,
                )
                pipeline.submit(
                    sequence_id=frame_idx,
                    raw26x7=raw_copy,
                    rgb=(rgb_frame.image if rgb_valid else None),
                    timestamp_ns=sample_ns,
                    rgb_repeated=bool(getattr(rgb_frame, "is_repeated", False)),
                    rgb_pico_delta_ms=rgb_to_pico_ms,
                    pico_source_age_ms=source_age_ms,
                    context=context,
                )

            _flush_completed_messages()

            # Count the completed sample before preview key handling can stop/save.
            completed_frame_idx = frame_idx
            frame_idx += 1

            saved_elapsed = time.perf_counter() - fps_saved_t0
            if saved_elapsed >= 1.0:
                preview_state.fps_saved = fps_saved_count / max(saved_elapsed, 1e-6)
                fps_saved_t0 = time.perf_counter()
                fps_saved_count = 0
            if completed_frame_idx % 3 == 0:
                _refresh_preview(rgb_frame)
            if frame_idx > 0 and frame_idx % progress_interval == 0:
                rec_symbol = record_symbol("●", "green")
                print(
                    f"[Rec] {rec_symbol} 第 {episode_index} 段 {frame_idx} 帧  "
                    f"有效={episode_valid_messages}  "
                    f"pipeline_pending={pipeline.stats()['submitted'] - pipeline.stats()['completed']}  "
                    f"耗时={time.perf_counter()-episode_start_perf:.1f}s"
                )

            elapsed = time.perf_counter() - t0
            sleep_for = (
                0.0
                if config.mvs_sync_mode == "mvs_master" and rgb_sensor is not None
                else period - elapsed
            )
            if sleep_for > 0.0:
                time.sleep(sleep_for)

    finally:
        save_error: Exception | None = None
        try:
            if recording or episode_messages:
                # Locals exist only after initialization reaches the loop setup.
                if "_stop_and_save_episode" in locals():
                    _stop_and_save_episode("exit")
        except Exception as exc:
            save_error = exc
            print(f"[Error] 退出时保存当前段失败: {exc}")
        finally:
            if key_watcher is not None:
                key_watcher.close()
            if pico_first_view_sensor is not None:
                pico_first_view_sensor.stop()
            if rgb_sensor is not None:
                rgb_sensor.stop()
            if pipeline is not None:
                pipeline.close()
            if retargeter is not None:
                retargeter.close()
            if pico_sensor is not None:
                pico_stopped = pico_sensor.stop()
                if pico_stopped:
                    pico_sensor.close()
                    print("[Exit] PICO 已释放")
                else:
                    print("[Exit] PICO reader 未关闭；缓存线程仍可能阻塞在 SDK 读取中")
            if preview_enabled:
                try:
                    cv2.destroyWindow(preview_window)
                except Exception:
                    pass
        if save_error is not None:
            raise save_error


def _parse_cpu_affinity(spec: str) -> set[int]:
    cores: set[int] = set()
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"CPU range 非法: {item}")
            cores.update(range(start, end + 1))
        else:
            cores.add(int(item))
    cpu_count = os.cpu_count() or 1
    return {core for core in cores if 0 <= core < cpu_count}


def _apply_cpu_affinity(spec: str) -> None:
    if not spec or spec.lower() in {"none", "off", "all"}:
        return
    if not hasattr(os, "sched_setaffinity"):
        print("[Warn] 当前系统不支持 sched_setaffinity，忽略 --cpu-affinity")
        return
    cores = _parse_cpu_affinity(spec)
    if not cores:
        raise ValueError(f"--cpu-affinity 未解析出有效 CPU: {spec}")
    try:
        os.sched_setaffinity(0, cores)
    except OSError as exc:
        available = set(os.sched_getaffinity(0))
        fallback = cores & available
        if not fallback:
            raise RuntimeError(
                f"无法设置 CPU affinity={sorted(cores)}，当前允许={sorted(available)}"
            ) from exc
        os.sched_setaffinity(0, fallback)
    print(f"[Runtime] CPU affinity={sorted(os.sched_getaffinity(0))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PICO + MVS 数据采集 → PKL。采集参数从 YAML 读取。"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"采集参数 YAML，默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--task-name",
        "-t",
        type=str,
        default=None,
        help="任务名称；不传则使用 YAML task_name；数据保存到 <out>/<task-name>/episode_XXXX/",
    )
    parser.add_argument(
        "--mvs-list",
        action="store_true",
        help="列出 MVS 相机序列号后退出",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="关闭 OpenCV 采集预览窗口，仅使用终端按键",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FUSION_CHECKPOINT)
    parser.add_argument("--model-type", choices=("auto", "stage1", "stage2"), default="auto")
    parser.add_argument("--mano-model-dir", type=Path, default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--wrist-config", type=Path, default=None)
    parser.add_argument("--dino-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--pico-std-mm", type=float, default=5.0)
    parser.add_argument("--vision-std-mm", type=float, default=14.0)
    parser.add_argument("--innovation-gate-mm", type=float, default=22.0)
    parser.add_argument("--process-accel-mps2", type=float, default=2.5)
    parser.add_argument("--wuji-maxeval", type=int, default=40)
    parser.add_argument(
        "--pipeline-queue-size",
        type=int,
        default=32,
        help="异步推理/重定向流水线缓冲帧数，默认 32",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=2,
        help="PyTorch CPU intra-op 线程数，默认 2，避免与重定向争抢 CPU",
    )
    parser.add_argument(
        "--vision-stride",
        type=int,
        default=2,
        help="每 N 个采集帧运行一次 wrist 模型；中间帧复用最近视觉点，默认 2",
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default="0-11",
        help="Linux CPU 亲和力；zjc 默认绑定 i9-13900H P-core 0-11；传 none 关闭",
    )
    args = parser.parse_args()

    if args.mvs_list:
        list_mvs_cameras()
        return

    try:
        _apply_cpu_affinity(args.cpu_affinity)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    cv2.setNumThreads(1)

    try:
        config = load_pico_mvs_config(
            config_path=Path(args.config).expanduser(),
            task_name=args.task_name,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if config.dry_run:
        print("[DryRun] 采集但不保存文件，按 Ctrl+C 退出")
    setattr(config, "no_preview", bool(args.no_preview))
    setattr(config, "fusion_checkpoint", args.checkpoint.expanduser().resolve())
    setattr(config, "fusion_model_type", args.model_type)
    setattr(config, "fusion_mano_model_dir", args.mano_model_dir.expanduser().resolve())
    setattr(config, "fusion_wrist_config", None if args.wrist_config is None else args.wrist_config.expanduser().resolve())
    setattr(config, "fusion_dino_dir", None if args.dino_dir is None else args.dino_dir.expanduser().resolve())
    setattr(config, "fusion_device", args.device)
    setattr(config, "fusion_precision", args.precision)
    setattr(config, "fusion_warmup_iters", int(args.warmup_iters))
    setattr(config, "fusion_pico_std_mm", float(args.pico_std_mm))
    setattr(config, "fusion_vision_std_mm", float(args.vision_std_mm))
    setattr(config, "fusion_innovation_gate_mm", float(args.innovation_gate_mm))
    setattr(config, "fusion_process_accel_mps2", float(args.process_accel_mps2))
    setattr(config, "fusion_wuji_maxeval", int(args.wuji_maxeval))
    setattr(config, "fusion_pipeline_queue_size", max(2, int(args.pipeline_queue_size)))
    setattr(config, "fusion_torch_threads", max(1, int(args.torch_threads)))
    setattr(config, "fusion_vision_stride", max(1, int(args.vision_stride)))
    setattr(config, "fusion_cpu_affinity", args.cpu_affinity)
    print(f"[Config] 已加载 {config.config_path}")
    print(f"[Config] task={config.task_name} out={config.out}")

    collect(config)


if __name__ == "__main__":
    main()
