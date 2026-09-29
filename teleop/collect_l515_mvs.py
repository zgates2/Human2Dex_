#!/usr/bin/env python3
"""
PICO + wrist MVS + 单个 L515 RGB 的同步采集入口。

用途
----
采集完整多传感器 episode：
1. PICO 手部数据：读取 26x7 原始状态，经过 retargeter 生成
   o6_command、wuji_command、trajectoryPose、pts21_mano。
2. wrist MVS RGB 相机：保存 wrist 视角 RGB 图像，并记录帧号和时间对齐信息。
3. 单个 L515：独立进程采集 RGB，保存 RGB jpg，并在 PKL 中保存帧号、
   时间戳、同步误差和图像相对路径。默认只选 YAML 中第一个 L515，
   也可以通过 --l515-camera 按名称选择。

运行方法
--------
    conda activate pico
    python3 collect_l515_mvs.py
    python3 collect_l515_mvs.py --task-name pick_cube        # 覆盖 YAML task_name
    python3 collect_l515_mvs.py --duration 10
    python3 collect_l515_mvs.py --config collect_l515_mvs.yaml
    python3 collect_l515_mvs.py --no-mvs

默认按命令行的 30 Hz 采样（可用 --hz 覆盖）；推荐使用 mvs_master_buffered，让 wrist MVS 的
采集时刻作为公共时间基准，再从 PICO/L515 缓存中选最近帧。

常用同步模式
------------
    fixed_rate              主循环按 hz 采样，取各传感器 latest frame
    fixed_rate_buffered     按固定采样时刻，从各相机历史缓存中选最近帧
    mvs_master              等待 wrist MVS 新帧，再取各传感器 latest frame
    mvs_master_buffered     以 wrist MVS capture time 为目标，从 L515 缓存选最近帧

交互控制
--------
    s        开始/暂停当前 episode
    Enter    录制中丢弃当前 episode，不保存 PKL/JPG
    q        退出
    Ctrl+C   退出；如果正在录制，会先保存当前 episode

实时预览
--------
默认打开 OpenCV 窗口：左上显示 wrist MVS RGB，右上显示所选 L515 RGB，左下显示
trajectoryPose 的 XY 手腕轨迹，右下显示录制状态、帧号和相机同步延迟。窗口内同样支持
s、Enter、q；无桌面环境时可加 --no-preview。

保存结构
--------
配置中的 out 或命令行 --out-root 表示输出根目录。同一个 task_name 永远写入同一个
task 目录，不会按运行次数再创建 demo 时间戳目录：

    <out>/<task_name>/episode_0001/episode_0001.pkl
    <out>/<task_name>/episode_0001/wrist/*.jpg
    <out>/<task_name>/episode_0001/l515/<camera_name>/rgb/*.jpg
    <out>/<task_name>/episode_0001/l515_intrinsics.json

episode 编号由 task 目录下已有 episode_XXXX 自动决定，选择最小缺失序号。
例如已有 episode_0001 和 episode_0003，下次会补 episode_0002。

代码结构
--------
    collect_config.py   读取 YAML、解析输出目录、构造 MVS 配置
    sensor_runtime.py   PicoSensor / MVSSensor / L515SensorGroup 实例 API
    episode_io.py       异步写图、PKL 保存、message 字段组装、键盘控制
    collect_cameras()   本文件的主采集编排：同步策略、采样、retarget、保存

"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import queue
import signal
import shutil
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the active collection env
    cv2 = None

from collect_config import (
    build_mvs_config,
    load_collect_config,
    task_output_dir,
)
from episode_io import (
    AsyncJpgWriter,
    KeypressWatcher,
    L515EpisodeSaveState,
    LINKER_JOINTS,
    WUJI_JOINTS,
    build_invalid_message,
    build_message,
    episode_pkl_path as make_episode_pkl_path,
    l515_fields,
    record_button,
    record_symbol,
    save_pkl,
    writer_errors,
    writer_stats,
    wrist_rgb_image_path_for_frame,
)
from sensor_runtime import L515SensorGroup, MVSSensor, PicoSensor


DEFAULT_CAMERA_CONFIG_PATH = Path(__file__).with_name("collect_l515_mvs.yaml")
DEXUMI_TELEOP_RETARGETER_PATH = Path(
    "/home/zjc/Desktop/human2dex/teleop/retargeter.py"
)


def _load_dexumi_retargeter_module():
    module_name = "dexumi_teleop_retargeter"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name,
        DEXUMI_TELEOP_RETARGETER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load retargeter from {DEXUMI_TELEOP_RETARGETER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _save_l515_intrinsics_json(path: Path, metadata: dict) -> None:
    l515_meta = metadata.get("l515") or {}
    cameras = l515_meta.get("cameras") or {}
    payload = {
        "format": "l515_intrinsics_v1",
        "sourcePkl": path.name,
        "cameras": {
            name: {
                "serial": cam.get("serial"),
                "alignTo": cam.get("alignTo"),
                "colorIntrinsics": cam.get("colorIntrinsics"),
            }
            for name, cam in cameras.items()
        },
    }
    out = path.parent / "l515_intrinsics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[Save] wrote L515 intrinsics -> {out}")


def _rgb_frame_valid(frame) -> bool:
    return (
        frame is not None
        and getattr(frame, "capture_ns", None) is not None
        and getattr(frame, "image", None) is not None
    )


def _l515_rgb_only_frames(frames: dict[str, object] | None) -> dict[str, object] | None:
    """Hide depth payloads before passing frames to the shared episode writer."""
    if not frames:
        return frames
    result = {}
    for name, frame in frames.items():
        result[name] = replace(
            frame,
            depth_z16=None,
            depth_frame_number=None,
            depth_timestamp_ms=None,
            depth_timestamp_domain=None,
            depth_capture_ns=None,
            depth_intrinsics=None,
            depth_metadata={},
        )
    return result


def _sample_quality(
    *, pico_frame, rgb_frame, l515_frames, sample_ns: int, max_delta_ms: float
) -> dict:
    raw = getattr(pico_frame, "raw26x7", None)
    active = int(getattr(pico_frame, "active", 0))
    valid_pico = active == 1 and raw is not None and getattr(raw, "shape", None) == (26, 7)
    flags = []
    if not valid_pico:
        flags.append("pico_invalid")
    if not _rgb_frame_valid(rgb_frame):
        flags.append("mvs_rgb_missing")
    mvs_delta_ms = None
    if _rgb_frame_valid(rgb_frame):
        mvs_delta_ms = (
            int(rgb_frame.capture_ns) - int(sample_ns)
        ) / 1_000_000.0
        if max_delta_ms > 0 and abs(mvs_delta_ms) > max_delta_ms:
            flags.append("mvs_sync_delta_large")
    l515_valid = False
    l515_delta_ms = None
    for frame in (l515_frames or {}).values():
        l515_valid = getattr(frame, "color_rgb", None) is not None
        capture_ns = getattr(frame, "color_capture_ns", None)
        if capture_ns is not None:
            l515_delta_ms = (int(capture_ns) - int(sample_ns)) / 1_000_000.0
        if not l515_valid:
            flags.append("l515_rgb_missing")
        break
    return {
        "ok": not flags,
        "flags": flags,
        "pico": {
            "valid": bool(valid_pico),
            "active": active,
            "rawShape": None if raw is None else list(getattr(raw, "shape", ())),
        },
        "rgb": {"valid": bool(_rgb_frame_valid(rgb_frame))},
        "alignment": {
            "mvsToSampleMs": mvs_delta_ms,
            "l515ToSampleMs": l515_delta_ms,
            "maxDeltaMs": float(max_delta_ms),
        },
        "l515": {"valid": bool(l515_valid), "colorToSampleMs": l515_delta_ms},
    }


def _select_one_l515(config, camera_name: str | None):
    cameras = list(config.l515_cameras or [])
    if not cameras:
        raise ValueError("collect_l515_mvs.py requires at least one l515_cameras entry")
    if camera_name is None:
        return [cameras[0]]
    for camera in cameras:
        if str(camera.get("name")) == str(camera_name) or str(camera.get("serial")) == str(camera_name):
            return [camera]
    choices = ", ".join(str(c.get("name")) for c in cameras)
    raise ValueError(f"unknown --l515-camera={camera_name!r}; choices: {choices}")


def _preview_image(image, *, title: str, width: int, height: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the preview window")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    if image is not None:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            canvas = cv2.resize(
                cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
    cv2.rectangle(canvas, (0, 0), (width, 34), (40, 40, 40), -1)
    cv2.putText(canvas, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (245, 245, 245), 2, cv2.LINE_AA)
    return canvas


def _draw_trajectory(points, *, width: int, height: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the preview window")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 22)
    margin = 42
    cv2.rectangle(canvas, (margin, 34), (width - 18, height - margin), (80, 80, 80), 1)
    cv2.putText(canvas, "trajectoryPose XY (m)", (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (245, 245, 245), 2, cv2.LINE_AA)
    if not points:
        cv2.putText(canvas, "waiting for valid PICO frame", (margin + 12, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
        return canvas
    xy = [(float(p[0]), float(p[1])) for p in points]
    cx, cy = xy[-1]
    scale = max(max(abs(x - cx) for x, _ in xy), max(abs(y - cy) for _, y in xy), 0.05)
    plot_w = width - margin - 18
    plot_h = height - 34 - margin
    projected = []
    for x, y in xy:
        px = margin + plot_w // 2 + int((x - cx) / scale * plot_w * 0.42)
        py = 34 + plot_h // 2 - int((y - cy) / scale * plot_h * 0.42)
        projected.append((max(margin, min(width - 18, px)), max(34, min(height - margin, py))))
    for a, b in zip(projected, projected[1:]):
        cv2.line(canvas, a, b, (80, 200, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, projected[-1], 5, (40, 240, 120), -1)
    cv2.putText(canvas, f"x={cx:+.3f} y={cy:+.3f} z={float(points[-1][2]):+.3f}",
                (margin + 8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def _draw_preview(
    *,
    rgb_frame,
    l515_frame,
    trajectory_points,
    recording: bool,
    episode_index: int,
    frame_idx: int,
    hz: float,
    l515_name: str,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the preview window")

    def _delta_text(frame, field: str) -> str:
        value = None if frame is None else getattr(frame, field, None)
        if value is None:
            return "--"
        return f"{float(value) / 1_000_000.0:+.1f} ms"

    cell_w, cell_h = 480, 300
    mvs_image = None if rgb_frame is None else getattr(rgb_frame, "image", None)
    l515_image = None if l515_frame is None else getattr(l515_frame, "color_rgb", None)
    top = np.hstack([
        _preview_image(mvs_image, title="wrist MVS RGB", width=cell_w, height=cell_h),
        _preview_image(l515_image, title=f"L515 RGB: {l515_name}", width=cell_w, height=cell_h),
    ])
    trajectory = _draw_trajectory(trajectory_points, width=cell_w, height=cell_h)
    panel = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)
    state = "REC" if recording else "STANDBY"
    color = (60, 60, 230) if recording else (100, 100, 100)
    cv2.putText(panel, f"{state}  EP {episode_index:04d}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2, cv2.LINE_AA)
    lines = [
        f"frame: {frame_idx}",
        f"target rate: {hz:.1f} Hz",
        f"MVS dt: {_delta_text(rgb_frame, 'sync_delta_ns')}",
        f"L515 dt: {_delta_text(l515_frame, 'color_sync_delta_ns')}",
        "keys: s start/stop | Enter discard | q quit",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(panel, line, (16, 78 + idx * 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([top, np.hstack([trajectory, panel])])


def _build_camera_metadata(
    *,
    config,
    rgb_sensor: MVSSensor | None,
    l515_group: L515SensorGroup,
    jpg_writer: AsyncJpgWriter | None,
    duration_s: float,
    sync_mode: str,
    sync_delay_ms: float,
    sync_max_delta_ms: float,
    drop_repeated_frames: bool,
    skipped_repeated_samples: int,
    skipped_invalid_pico_samples: int,
    rotation_postmultiply,
) -> dict:
    rgb_meta = None if rgb_sensor is None else rgb_sensor.metadata()
    l515_meta = l515_group.metadata()
    return {
        "format": "pico_l515_mvs_rgb_only",
        "taskName": config.task_name,
        "configPath": None if config.config_path is None else str(config.config_path),
        "collectionHz": float(config.hz),
        "cacheHz": max(float(config.cache_hz), float(config.hz)),
        "hand": config.hand,
        "l515Camera": {
            "name": (config.l515_cameras or [{}])[0].get("name"),
            "serial": (config.l515_cameras or [{}])[0].get("serial"),
        },
        "syncMode": str(sync_mode),
        "syncDelayMs": float(sync_delay_ms),
        "syncMaxDeltaMs": float(sync_max_delta_ms),
        "dropRepeatedFrames": bool(drop_repeated_frames),
        "skippedRepeatedSamples": int(skipped_repeated_samples),
        "skippedInvalidPicoSamples": int(skipped_invalid_pico_samples),
        "durationSeconds": float(duration_s),
        "clock": {
            "sampleClockNs": "time.monotonic_ns() in the parent sampling loop",
            "fixedRate": "sleep to config.hz, then read each latest sensor cache",
            "mvsMaster": "wait for a new MVS frame, then read L515 latest caches",
            "fixedRateBuffered": "sample target is fixed-rate host time; select nearest frames from each ring buffer after syncDelayMs",
            "mvsMasterBuffered": "sample target is MVS captureNs; select nearest L515 frames from ring buffers after syncDelayMs",
            "captureNs": "time.monotonic_ns() when each backend accepted the frame",
            "residualNs": "sampleClockNs - captureNs",
            "syncDeltaNs": "selected frame captureNs - syncTargetNs",
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
            "source": "PICO 26 points mapped to MediaPipe 21, wrist-centered MANO frame",
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
        "sampleQuality": {
            "field": "sampleQuality",
            "flagsField": "qualityFlags",
            "meaning": "PICO/MVS/L515 RGB availability and timestamp alignment",
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
            "storage": "jpg files saved in wrist/",
            "imageField": "rgbImage",
            "imageFieldValue": "relative path string or None",
            "mvsConfig": {
                "serial": (
                    None
                    if rgb_sensor is None
                    else str(rgb_sensor.config.serial)
                ),
                "captureFps": float(
                    config.mvs_capture_fps
                    or (config.hz if rgb_sensor is None else rgb_sensor.config.fps)
                ),
                "exposureAuto": "off",
                "exposureTimeUs": float(config.mvs_exposure_time_us),
                "gainAuto": str(config.mvs_gain_auto),
                "gainDb": float(config.mvs_gain_db),
            },
            "writer": writer_stats(jpg_writer),
            "writerErrors": writer_errors(jpg_writer),
            "mvs": rgb_meta,
        },
        "l515": {
            "enabled": bool(l515_meta),
            "backend": "spawn_shared_memory",
            "messageField": "l515",
            "storage": {
                "rgb": "jpg files saved under l515/<camera_name>/rgb/",
                "depth": "disabled by this collector",
            },
            "timestamp": {
                "colorTimestampNs": "RealSense frame.get_timestamp() in global_time domain, converted from ms to ns",
                "colorCaptureNs": "worker time.monotonic_ns() when color entered shared memory",
            },
            "rgbWriter": writer_stats(jpg_writer),
            "rgbWriterErrors": writer_errors(jpg_writer),
            "depthWriter": None,
            "depthWriterErrors": [],
            "cameras": l515_meta,
        },
    }


def _sleep_until_monotonic_ns(target_ns: int) -> None:
    while True:
        remaining_ns = int(target_ns) - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        time.sleep(min(remaining_ns / 1_000_000_000.0, 0.002))


def _with_rgb_sync(rgb_frame, target_ns: int | None, selection_mode: str):
    if rgb_frame is None or target_ns is None:
        return rgb_frame
    capture_ns = getattr(rgb_frame, "capture_ns", None)
    return replace(
        rgb_frame,
        sync_target_ns=int(target_ns),
        sync_delta_ns=(
            None
            if capture_ns is None
            else int(capture_ns) - int(target_ns)
        ),
        selection_mode=selection_mode,
    )


def _selected_frame_ids(rgb_frame, l515_frames: dict[str, object] | None) -> dict[str, int]:
    ids: dict[str, int] = {}
    rgb_id = getattr(rgb_frame, "frame_id", None)
    if rgb_id is not None:
        ids["rgb"] = int(rgb_id)
    for name, frame in (l515_frames or {}).items():
        color_id = getattr(frame, "color_frame_number", None)
        if color_id is not None:
            ids[f"{name}.color"] = int(color_id)
    return ids


def _has_repeated_selection(
    ids: dict[str, int],
    last_ids: dict[str, int],
) -> bool:
    # Skip only when ALL tracked camera frames are repeated, not just any one.
    # This prevents a single stalled camera from discarding valid frames from
    # other sensors.
    if not ids:
        return False
    return all(last_ids.get(key) == value for key, value in ids.items())


def _make_pico_camera_message(
    *,
    config,
    retargeter,
    pico_frame,
    pkl_path: Path,
    frame_idx: int,
    sample_ns: int,
    sync_target_ns: int | None,
    rgb_frame,
    l515_group: L515SensorGroup,
    l515_frames: dict[str, object] | None,
    jpg_writer: AsyncJpgWriter | None,
    save_state: L515EpisodeSaveState,
    sync_max_delta_ms: float,
) -> dict | None:
    rgb_image_path = (
        None
        if config.dry_run
        else wrist_rgb_image_path_for_frame(pkl_path, frame_idx)
    )

    raw26x7 = pico_frame.raw26x7
    active = pico_frame.active
    sdk_ts_ns = pico_frame.sdk_ts_ns
    source_receive_ns = pico_frame.receive_ns
    valid_pico = active == 1 and raw26x7 is not None and raw26x7.shape == (26, 7)

    # Always process image frames (MVS + L515) regardless of PICO validity,
    # so that camera data is never lost due to a bad gesture frame.
    if l515_frames is None:
        l515_frames = l515_group.latest_all(copy_images=True)
    l515_payload = l515_fields(
        l515_frames=_l515_rgb_only_frames(l515_frames),
        sample_ns=sample_ns,
        pkl_path=None if config.dry_run else pkl_path,
        frame_idx=frame_idx,
        jpg_writer=jpg_writer,
        depth_writer=None,
        save_state=save_state,
    )

    quality_payload = _sample_quality(
        pico_frame=pico_frame,
        rgb_frame=rgb_frame,
        l515_frames=l515_frames,
        sample_ns=sample_ns,
        max_delta_ms=sync_max_delta_ms,
    )

    if not valid_pico:
        # Build and return an invalid message that still carries the saved image paths.
        msg = build_invalid_message(
            sdk_ts_ns=sdk_ts_ns,
            sample_ns=sample_ns,
            source_receive_ns=source_receive_ns,
            rgb_frame=rgb_frame,
            rgb_image_path=rgb_image_path,
            jpg_writer=jpg_writer,
            l515_payload=l515_payload,
            raw26x7=raw26x7,
            pico_active=active,
            valid_mask=[False] * 21,
            quality_payload=quality_payload,
        )
        msg["syncTargetNs"] = None if sync_target_ns is None else int(sync_target_ns)
        return msg

    retargeted = retargeter.retarget(raw26x7)
    wuji_qpos = retargeted.wuji_joint_radians
    if wuji_qpos is None:
        raise RuntimeError("Wuji retargeting did not return qpos")
    msg = build_message(
        o6_angles=retargeted.linker_joint_radians,
        wuji_qpos=wuji_qpos,
        pts21_mano=retargeted.pts21_mano,
        wrist_pose_6d=retargeted.wrist_pose_6d,
        sdk_ts_ns=sdk_ts_ns,
        sample_ns=sample_ns,
        source_receive_ns=source_receive_ns,
        rgb_frame=rgb_frame,
        rgb_image_path=rgb_image_path,
        jpg_writer=jpg_writer,
        l515_payload=l515_payload,
        raw26x7=raw26x7,
        pico_active=active,
        valid_mask=[True] * 21,
        quality_payload=quality_payload,
    )
    msg["syncTargetNs"] = None if sync_target_ns is None else int(sync_target_ns)
    return msg


def collect_cameras(
    config,
    duration_s: float,
    l515_stagger_sec: float,
    l515_camera: str | None,
    no_mvs: bool,
    sync_mode: str,
    sync_delay_ms: float,
    sync_max_delta_ms: float,
    drop_repeated_frames: bool,
) -> Path | None:
    valid_sync_modes = {
        "fixed_rate",
        "mvs_master",
        "fixed_rate_buffered",
        "mvs_master_buffered",
    }
    if sync_mode not in valid_sync_modes:
        raise ValueError(f"unsupported sync mode: {sync_mode}")
    if no_mvs and sync_mode.startswith("mvs_master"):
        raise ValueError("--sync-mode mvs_master requires MVS to be enabled")

    dexumi_retargeter = _load_dexumi_retargeter_module()
    PICO_WRIST_TO_TARGET_TCP_ROT = dexumi_retargeter.PICO_WRIST_TO_TARGET_TCP_ROT
    PicoToLinkerO6Retargeter = dexumi_retargeter.PicoToLinkerO6Retargeter

    sync_delay_ns = max(0, int(round(float(sync_delay_ms) * 1_000_000.0)))
    max_delta_ns = (
        None
        if float(sync_max_delta_ms) <= 0
        else int(round(float(sync_max_delta_ms) * 1_000_000.0))
    )
    duration_limit_s = None if duration_s is None or float(duration_s) <= 0 else float(duration_s)

    # This collector intentionally records one L515 RGB stream only.
    config = replace(config, l515_cameras=_select_one_l515(config, l515_camera))
    selected_l515 = config.l515_cameras[0]
    print(
        f"[Config] selected L515 name={selected_l515.get('name')} "
        f"serial={selected_l515.get('serial')} (RGB only)"
    )

    print("[Init] starting PICO reader")
    pico_sensor = PicoSensor(
        hand=config.hand,
        poll_hz=max(float(config.cache_hz), float(config.hz)),
    )
    print("[Init] loading retargeters")
    retargeter = PicoToLinkerO6Retargeter(
        **(
            {"yaml_path": config.retargeting_yaml}
            if config.retargeting_yaml
            else {}
        ),
        wuji_yaml_path=config.wuji_retargeting_yaml,
        hand=config.hand,
    )
    pico_sensor.start()

    l515_group = L515SensorGroup(config.l515_cameras, stagger_sec=l515_stagger_sec)
    rgb_sensor: MVSSensor | None = None
    jpg_writer: AsyncJpgWriter | None = None
    key_watcher: KeypressWatcher | None = None
    saved_path: Path | None = None
    preview_enabled = not bool(getattr(config, "no_preview", False))
    if preview_enabled and cv2 is None:
        preview_enabled = False
        print("[Warn] OpenCV 未安装，禁用实时预览；如需预览请在采集环境安装 opencv-python")
    if preview_enabled and sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            preview_enabled = False
            print("[Warn] 未检测到 DISPLAY/WAYLAND_DISPLAY，禁用实时预览；有桌面会话时可直接启用")
    preview_window = "DexUMI MVS + L515 collector"
    trajectory_points: list[tuple[float, float, float]] = []

    try:
        l515_group.start()
        if not no_mvs:
            print("[Init] starting MVS cache")
            rgb_sensor = MVSSensor(build_mvs_config(config))
            rgb_sensor.start()

        period = 1.0 / max(float(config.hz), 1.0)
        period_ns = int(round(period * 1_000_000_000.0))
        frame_idx = 0
        progress_every = max(int(round(float(config.hz) * 5.0)), 1)
        wait_timeout_s = (
            0.25
            if rgb_sensor is None
            else max(0.25, float(rgb_sensor.config.timeout_ms) / 1000.0)
        )
        next_target_ns = time.monotonic_ns()
        last_rgb_frame_id = (
            None
            if rgb_sensor is None
            else rgb_sensor.current_frame_id()
        )
        recording = False
        episode_index = 0
        episode_messages: list[dict] = []
        episode_pkl_path: Path | None = None
        save_state = L515EpisodeSaveState({}, {})
        last_selected_ids: dict[str, int] = {}
        skipped_repeated_samples = 0
        skipped_invalid_pico_samples = 0
        invalid_pico_frame_count = 0
        episode_start_perf = 0.0
        last_toggle_perf = 0.0
        stop_event = threading.Event()

        def _sigint(_sig, _frame):
            print("\n[Ctrl+C] exit requested; saving current segment if recording...")
            stop_event.set()

        signal.signal(signal.SIGINT, _sigint)
        key_watcher = KeypressWatcher(stop_event)
        key_watcher.start()

        if preview_enabled:
            try:
                cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(preview_window, 960, 600)
            except Exception as exc:
                preview_enabled = False
                print(f"[Warn] OpenCV preview unavailable, continuing without preview: {exc}")

        print(
            "[Mode] interactive: press s to start/stop recording; "
            f"press Enter to discard; press q or Ctrl+C to exit; "
            f"debounce={config.key_debounce:.2f}s"
        )
        if duration_limit_s is not None:
            print(
                f"[Mode] duration={duration_limit_s:.1f}s: each segment auto-saves "
                "at the limit and returns to standby"
            )
        print("  (standby; press s to start recording)")

        def _start_episode() -> None:
            nonlocal recording
            nonlocal episode_index
            nonlocal episode_messages
            nonlocal episode_pkl_path
            nonlocal jpg_writer
            nonlocal save_state
            nonlocal frame_idx
            nonlocal last_selected_ids
            nonlocal skipped_repeated_samples
            nonlocal skipped_invalid_pico_samples
            nonlocal invalid_pico_frame_count
            nonlocal trajectory_points
            nonlocal episode_start_perf
            nonlocal next_target_ns
            nonlocal last_rgb_frame_id

            episode_index += 1
            episode_pkl_path = make_episode_pkl_path(config.out, episode_index)
            episode_messages = []
            save_state = L515EpisodeSaveState({}, {})
            frame_idx = 0
            trajectory_points = []
            last_selected_ids = {}
            skipped_repeated_samples = 0
            skipped_invalid_pico_samples = 0
            invalid_pico_frame_count = 0
            next_target_ns = time.monotonic_ns()
            last_rgb_frame_id = (
                None
                if rgb_sensor is None
                else rgb_sensor.current_frame_id()
            )
            retargeter.reset()
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
            rec_button = record_button("REC START", "green")
            rec_symbol = record_symbol("*", "green")
            print(
                f"\n{rec_button} [Rec] {rec_symbol} episode={episode_index} "
                f"task={config.task_name} hz={config.hz} cache_hz={config.cache_hz} "
                f"sync_mode={sync_mode} out={episode_pkl_path}"
            )

        # Maximum allowed invalid PICO gesture frames per episode before discard.
        INVALID_PICO_FRAME_LIMIT = 30

        def _stop_and_save_episode(reason: str) -> None:
            nonlocal recording
            nonlocal episode_messages
            nonlocal episode_pkl_path
            nonlocal jpg_writer
            nonlocal frame_idx
            nonlocal saved_path

            if not recording and not episode_messages:
                return
            recording = False
            stop_button = record_button("REC STOP", "red")
            stop_symbol = record_symbol("*", "red")
            print(
                f"\n{stop_button} [Rec] {stop_symbol} episode={episode_index} "
                f"stopped reason={reason}; saving..."
            )

            writer_to_close = jpg_writer
            jpg_writer = None
            if writer_to_close is not None:
                writer_to_close.stop()

            actual_duration_s = (
                0.0
                if episode_start_perf <= 0.0
                else max(time.perf_counter() - episode_start_perf, 0.0)
            )
            metadata = _build_camera_metadata(
                config=config,
                rgb_sensor=rgb_sensor,
                l515_group=l515_group,
                jpg_writer=writer_to_close,
                duration_s=actual_duration_s,
                sync_mode=sync_mode,
                sync_delay_ms=sync_delay_ms,
                sync_max_delta_ms=sync_max_delta_ms,
                drop_repeated_frames=drop_repeated_frames,
                skipped_repeated_samples=skipped_repeated_samples,
                skipped_invalid_pico_samples=skipped_invalid_pico_samples,
                rotation_postmultiply=PICO_WRIST_TO_TARGET_TCP_ROT,
            )
            valid_count = sum(
                1 for msg in episode_messages if msg["o6_command"] is not None
            )
            print(
                f"[Done] episode={episode_index} frames={len(episode_messages)} "
                f"valid={valid_count} invalid_pico={invalid_pico_frame_count} "
                f"invalid_skipped={skipped_invalid_pico_samples} "
                f"repeated_skipped={skipped_repeated_samples}"
            )
            # Discard episode if too many invalid PICO gesture frames were recorded.
            if invalid_pico_frame_count > INVALID_PICO_FRAME_LIMIT:
                print(
                    f"[Discard] episode={episode_index} discarded: "
                    f"invalid_pico_frames={invalid_pico_frame_count} > "
                    f"limit={INVALID_PICO_FRAME_LIMIT}; pkl NOT saved"
                )
            elif config.dry_run:
                print("[DryRun] not writing pkl/images")
            elif episode_pkl_path is not None:
                save_pkl(episode_messages, episode_pkl_path, metadata)
                _save_l515_intrinsics_json(episode_pkl_path, metadata)
                saved_path = episode_pkl_path

            episode_messages = []
            episode_pkl_path = None
            frame_idx = 0
            while key_watcher is not None:
                try:
                    stale_ch = key_watcher.events.get_nowait()
                except queue.Empty:
                    break
                if stale_ch in ("q", "\x03", "\x04"):
                    stop_event.set()
            print("[Mode] standby; press s to start next segment, q/Ctrl+C to exit")

        def _discard_episode(reason: str) -> None:
            nonlocal recording, episode_messages, episode_pkl_path, jpg_writer, frame_idx
            nonlocal trajectory_points
            if not recording and not episode_messages:
                return
            recording = False
            writer_to_close = jpg_writer
            jpg_writer = None
            if writer_to_close is not None:
                writer_to_close.stop()
            discard_dir = None if episode_pkl_path is None else episode_pkl_path.parent
            if discard_dir is not None and discard_dir.exists():
                shutil.rmtree(discard_dir)
            episode_messages = []
            episode_pkl_path = None
            frame_idx = 0
            trajectory_points = []
            print(f"[Discard] episode={episode_index} discarded ({reason})")

        def _handle_control_key(ch: str) -> None:
            nonlocal last_toggle_perf
            if ch == "s":
                now = time.perf_counter()
                if now - last_toggle_perf < float(config.key_debounce):
                    print("[Key] ignored rapid s press")
                    return
                last_toggle_perf = now
                if recording:
                    _stop_and_save_episode("s_key")
                else:
                    _start_episode()
            elif ch in ("\n", "\r"):
                _discard_episode("enter_key")
            elif ch in ("q", "\x03", "\x04"):
                print("\n[Key] exit requested")
                stop_event.set()
            elif ch != " ":
                print("[Key] press s to start/stop, Enter to discard, q to exit")

        def _handle_key_events() -> None:
            while key_watcher is not None:
                try:
                    ch = key_watcher.events.get_nowait()
                except queue.Empty:
                    return
                _handle_control_key(ch)

        def _refresh_preview(rgb_frame=None, l515_frames=None) -> None:
            nonlocal preview_enabled
            if not preview_enabled:
                return
            if rgb_frame is None and rgb_sensor is not None:
                try:
                    rgb_frame = rgb_sensor.latest(copy_image=True, count_return=False)
                except Exception:
                    rgb_frame = None
            if l515_frames is None:
                try:
                    l515_frames = l515_group.latest_all(copy_images=True)
                except Exception:
                    l515_frames = {}
            l515_frame = next(iter(l515_frames.values()), None)
            vis = _draw_preview(
                rgb_frame=rgb_frame,
                l515_frame=l515_frame,
                trajectory_points=trajectory_points,
                recording=recording,
                episode_index=episode_index,
                frame_idx=frame_idx,
                hz=float(config.hz),
                l515_name=str(config.l515_cameras[0].get("name", "l515")),
            )
            try:
                cv2.imshow(preview_window, vis)
                key = cv2.waitKey(1) & 0xFF
            except Exception as exc:
                preview_enabled = False
                print(f"[Warn] OpenCV 预览刷新失败，已禁用预览: {exc}")
                return
            if key == 255:
                return
            if key in (ord("s"), ord("S")):
                _handle_control_key("s")
            elif key in (13, 10):
                _handle_control_key("\n")
            elif key in (27, ord("q"), ord("Q")):
                _handle_control_key("q")

        def _record_one_sample() -> None:
            nonlocal frame_idx
            nonlocal next_target_ns
            nonlocal last_rgb_frame_id
            nonlocal skipped_repeated_samples
            nonlocal skipped_invalid_pico_samples
            nonlocal invalid_pico_frame_count

            assert episode_pkl_path is not None
            loop_t0 = time.perf_counter()
            sync_target_ns = None
            l515_frames = None
            selected_ids: dict[str, int] = {}

            if sync_mode in {"mvs_master", "mvs_master_buffered"}:
                assert rgb_sensor is not None
                try:
                    rgb_frame = rgb_sensor.wait_next(
                        last_frame_id=last_rgb_frame_id,
                        timeout_s=wait_timeout_s,
                        copy_image=True,
                    )
                except TimeoutError as exc:
                    print(f"[Warn] MVS master wait timed out: {exc}")
                    return
                if rgb_frame.frame_id is not None:
                    last_rgb_frame_id = int(rgb_frame.frame_id)
                sync_target_ns = (
                    int(rgb_frame.capture_ns)
                    if rgb_frame.capture_ns is not None
                    else time.monotonic_ns()
                )
                if sync_mode == "mvs_master_buffered":
                    _sleep_until_monotonic_ns(sync_target_ns + sync_delay_ns)
                    rgb_frame = _with_rgb_sync(
                        rgb_frame,
                        target_ns=sync_target_ns,
                        selection_mode="mvs_master_capture_ns",
                    )
                    l515_frames = l515_group.nearest_all(
                        target_ns=sync_target_ns,
                        copy_images=True,
                        max_delta_ns=max_delta_ns,
                    )
                    selected_ids = _selected_frame_ids(rgb_frame, l515_frames)
                    if (
                        drop_repeated_frames
                        and _has_repeated_selection(selected_ids, last_selected_ids)
                    ):
                        skipped_repeated_samples += 1
                        return
                else:
                    rgb_frame = _with_rgb_sync(
                        rgb_frame,
                        target_ns=sync_target_ns,
                        selection_mode="mvs_master_capture_ns",
                    )
                sample_ns = int(sync_target_ns)
            else:
                target_ns = (
                    next_target_ns
                    if sync_mode == "fixed_rate_buffered"
                    else time.monotonic_ns()
                )
                if sync_mode == "fixed_rate_buffered":
                    next_target_ns += period_ns
                    now_ns = time.monotonic_ns()
                    if next_target_ns < now_ns - period_ns:
                        missed = (now_ns - next_target_ns) // max(period_ns, 1) + 1
                        next_target_ns += int(missed) * period_ns
                sync_target_ns = int(target_ns)
                l515_frames = None
                selected_ids: dict[str, int] = {}
                if sync_mode == "fixed_rate_buffered":
                    _sleep_until_monotonic_ns(sync_target_ns + sync_delay_ns)
                    rgb_frame = (
                        None
                        if rgb_sensor is None
                        else rgb_sensor.nearest(
                            target_ns=sync_target_ns,
                            copy_image=True,
                            max_delta_ns=max_delta_ns,
                        )
                    )
                    l515_frames = l515_group.nearest_all(
                        target_ns=sync_target_ns,
                        copy_images=True,
                        max_delta_ns=max_delta_ns,
                    )
                    sample_ns = int(sync_target_ns)
                else:
                    sample_ns = int(target_ns)
                    rgb_frame = None if rgb_sensor is None else rgb_sensor.latest(copy_image=True)
                if rgb_frame is not None and sync_target_ns is not None:
                    rgb_frame = _with_rgb_sync(
                        rgb_frame,
                        target_ns=sync_target_ns,
                        selection_mode=(
                            "fixed_rate_latest"
                            if sync_mode == "fixed_rate"
                            else "fixed_rate_buffered_nearest_capture_ns"
                        ),
                    )
                if l515_frames is not None:
                    selected_ids = _selected_frame_ids(rgb_frame, l515_frames)
                    if (
                        drop_repeated_frames
                        and _has_repeated_selection(selected_ids, last_selected_ids)
                    ):
                        skipped_repeated_samples += 1
                        return

            pico_frame = (
                pico_sensor.nearest(int(sync_target_ns))
                if sync_target_ns is not None
                else pico_sensor.latest()
            )
            msg = _make_pico_camera_message(
                config=config,
                retargeter=retargeter,
                pico_frame=pico_frame,
                pkl_path=episode_pkl_path,
                frame_idx=frame_idx,
                sample_ns=sample_ns,
                sync_target_ns=sync_target_ns,
                rgb_frame=rgb_frame,
                l515_group=l515_group,
                l515_frames=l515_frames,
                jpg_writer=jpg_writer,
                save_state=save_state,
                sync_max_delta_ms=sync_max_delta_ms,
            )
            if msg is None:
                skipped_invalid_pico_samples += 1
            else:
                episode_messages.append(msg)
                last_selected_ids.update(selected_ids)
                trajectory = msg.get("trajectoryPose")
                if trajectory is not None:
                    try:
                        pose = np.asarray(trajectory, dtype=np.float32).reshape(6)
                    except (TypeError, ValueError):
                        pose = None
                    if pose is not None and np.all(np.isfinite(pose)):
                        trajectory_points.append(
                            (float(pose[0]), float(pose[1]), float(pose[2]))
                        )
                        if len(trajectory_points) > 300:
                            del trajectory_points[:-300]
                # Track invalid PICO gesture frames (saved to episode but no joint data).
                if msg.get("o6_command") is None:
                    invalid_pico_frame_count += 1
            _refresh_preview(rgb_frame, l515_frames)
            if frame_idx > 0 and frame_idx % progress_every == 0:
                valid = sum(
                    1 for msg_item in episode_messages
                    if msg_item["o6_command"] is not None
                )
                print(
                    f"[Rec] episode={episode_index} frames={frame_idx} "
                    f"valid={valid} elapsed={time.perf_counter()-episode_start_perf:.1f}s"
                )
            frame_idx += 1
            if sync_mode in {"fixed_rate", "fixed_rate_buffered"}:
                sleep_for = period - (time.perf_counter() - loop_t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        while not stop_event.is_set():
            _handle_key_events()
            if stop_event.is_set():
                break
            if not recording:
                _refresh_preview()
                time.sleep(0.03)
                continue
            if (
                duration_limit_s is not None
                and (time.perf_counter() - episode_start_perf) >= duration_limit_s
            ):
                _stop_and_save_episode("duration")
                continue
            _record_one_sample()
        return saved_path
    finally:
        save_error: Exception | None = None
        try:
            if 'recording' in locals() and (recording or episode_messages):
                _stop_and_save_episode("exit")
        except Exception as exc:
            save_error = exc
            print(f"[Error] failed to save current segment during exit: {exc}")
        if key_watcher is not None:
            key_watcher.close()
        if rgb_sensor is not None:
            rgb_sensor.stop()
        pico_stopped = pico_sensor.stop()
        if pico_stopped:
            pico_sensor.close()
        l515_group.stop()
        if preview_enabled and cv2 is not None:
            try:
                cv2.destroyWindow(preview_window)
            except Exception:
                pass
        if save_error is not None:
            raise save_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PICO + wrist MVS + one RealSense L515 RGB collection."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG_PATH,
        help=f"camera config YAML, default {DEFAULT_CAMERA_CONFIG_PATH}",
    )
    parser.add_argument(
        "--task-name",
        "-t",
        type=str,
        default=None,
        help="task folder name under the output root; defaults to YAML task_name",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="recording duration in seconds; overrides YAML duration",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="采样频率，默认固定为 30 Hz",
    )
    parser.add_argument(
        "--mvs-exposure-time-us",
        type=float,
        default=None,
        help="覆盖 YAML 中的 MVS 固定曝光时间，单位微秒",
    )
    parser.add_argument(
        "--mvs-capture-fps",
        type=float,
        default=None,
        help="覆盖 YAML 中的 MVS 相机采集帧率；默认使用 mvs_capture_fps 或 --hz",
    )
    parser.add_argument(
        "--mvs-gain-auto",
        choices=("off", "once", "continuous"),
        default=None,
        help="覆盖 YAML 中的 MVS 增益模式",
    )
    parser.add_argument(
        "--mvs-gain-db",
        type=float,
        default=None,
        help="覆盖 YAML 中的 MVS 固定增益，单位 dB",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="output root directory; records to out-root/task/episode_XXXX/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="collect without writing files",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="disable the OpenCV camera/trajectory preview window",
    )
    parser.add_argument(
        "--l515-stagger-sec",
        type=float,
        default=2.0,
        help="delay between starting L515 workers",
    )
    parser.add_argument(
        "--l515-camera",
        type=str,
        default=None,
        help="L515 camera name or serial; default is the first configured camera",
    )
    parser.add_argument(
        "--no-mvs",
        action="store_true",
        help="start only the configured L515 cameras",
    )
    parser.add_argument(
        "--sync-mode",
        choices=(
            "fixed_rate",
            "mvs_master",
            "fixed_rate_buffered",
            "mvs_master_buffered",
        ),
        default="mvs_master_buffered",
        help="mvs_master_buffered is recommended: MVS capture time is the common sample clock",
    )
    parser.add_argument(
        "--sync-delay-ms",
        type=float,
        default=35.0,
        help="buffered modes wait this long after sync target before nearest-neighbor selection",
    )
    parser.add_argument(
        "--sync-max-delta-ms",
        type=float,
        default=75.0,
        help="maximum accepted nearest-neighbor capture delta; <=0 disables the limit",
    )
    parser.add_argument(
        "--drop-repeated-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="buffered modes skip a sample when any selected camera frame id repeats",
    )
    args = parser.parse_args()
    try:
        config = load_collect_config(
            config_path=Path(args.config).expanduser(),
            task_name=args.task_name,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.duration is not None:
        config.duration = float(args.duration)
    if args.hz <= 0:
        parser.error("--hz must be positive")
    config.hz = float(args.hz)
    if args.mvs_exposure_time_us is not None:
        if not math.isfinite(args.mvs_exposure_time_us) or args.mvs_exposure_time_us <= 0:
            parser.error("--mvs-exposure-time-us must be a positive finite number")
        config.mvs_exposure_time_us = float(args.mvs_exposure_time_us)
    if args.mvs_capture_fps is not None:
        if not math.isfinite(args.mvs_capture_fps) or args.mvs_capture_fps <= 0:
            parser.error("--mvs-capture-fps must be a positive finite number")
        config.mvs_capture_fps = float(args.mvs_capture_fps)
    if args.mvs_gain_auto is not None:
        config.mvs_gain_auto = str(args.mvs_gain_auto)
    if args.mvs_gain_db is not None:
        if not math.isfinite(args.mvs_gain_db) or args.mvs_gain_db < 0:
            parser.error("--mvs-gain-db must be a finite number >= 0")
        config.mvs_gain_db = float(args.mvs_gain_db)
    if config.duration is None:
        config.duration = 10.0
    if args.out_root is not None:
        config.out = task_output_dir(
            Path(args.out_root).expanduser(),
            config.task_name,
        )
    if args.dry_run:
        config.dry_run = True
    if args.no_preview:
        config.no_preview = True
    print(f"[Config] loaded {config.config_path}")
    print(f"[Config] task={config.task_name} out={config.out}")
    print(
        f"[Config] MVS capture_fps={float(config.mvs_capture_fps or config.hz):.1f} "
        f"exposure_time_us={float(config.mvs_exposure_time_us):.1f} "
        f"gain_auto={config.mvs_gain_auto} gain_db={float(config.mvs_gain_db):.2f}"
    )
    collect_cameras(
        config,
        duration_s=float(config.duration),
        l515_stagger_sec=float(args.l515_stagger_sec),
        l515_camera=args.l515_camera,
        no_mvs=bool(args.no_mvs),
        sync_mode=str(args.sync_mode),
        sync_delay_ms=float(args.sync_delay_ms),
        sync_max_delta_ms=float(args.sync_max_delta_ms),
        drop_repeated_frames=bool(args.drop_repeated_frames),
    )


if __name__ == "__main__":
    main()
