#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared collection configuration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CollectConfig:
    """Runtime config loaded from YAML plus the task name from CLI."""

    task_name: str
    out: Path
    hz: float = 60.0
    cache_hz: float = 90.0
    hand: str = "right"
    retargeting_yaml: str | None = None
    wuji_retargeting_yaml: str | None = None
    duration: float | None = None
    key_debounce: float = 0.8
    keep_invalid: bool = False
    mvs_serial: str | None = None
    mvs_resolution: tuple[int, int] = (480, 480)
    mvs_input_resolution: tuple[int, int] = (1440, 1080)
    mvs_auto_crop_black_border: bool = False
    mvs_center_crop: bool = True
    mvs_use_sensor_crop_roi: bool = True
    mvs_exposure_time_us: float = 15000.0
    mvs_gain_auto: str = "off"
    mvs_gain_db: float = 10.0
    mvs_capture_fps: float | None = None
    mvs_sync_mode: str = "mvs_buffered"
    max_rgb_pico_delta_ms: float = 20.0
    jpg_quality: int = 95
    jpg_writer_workers: int = 2
    jpg_writer_queue_size: int = 512
    pico_first_view_enabled: bool = False
    pico_first_view_device: object | None = None
    pico_first_view_width: int | None = None
    pico_first_view_height: int | None = None
    pico_first_view_fps: float | None = None
    l515_cameras: list[dict[str, Any]] | None = None
    dry_run: bool = False
    config_path: Path | None = None


YAML_CONFIG_DEFAULTS: dict[str, object] = {
    "task_name": None,
    "out": None,
    "hz": 60.0,
    "cache_hz": 90.0,
    "hand": "right",
    "retargeting_yaml": None,
    "wuji_retargeting_yaml": "wuji_retargeting/config/adaptive_analytical_pico.yaml",
    "duration": None,
    "key_debounce": 0.8,
    "keep_invalid": False,
    "mvs_serial": None,
    "mvs_resolution": [480, 480],
    "mvs_input_resolution": [1440, 1080],
    "mvs_auto_crop_black_border": False,
    "mvs_center_crop": True,
    "mvs_use_sensor_crop_roi": True,
    "mvs_exposure_time_us": 15000.0,
    "mvs_gain_auto": "off",
    "mvs_gain_db": 10.0,
    "mvs_capture_fps": None,
    "mvs_sync_mode": "mvs_buffered",
    "max_rgb_pico_delta_ms": 20.0,
    "jpg_quality": 95,
    "jpg_writer_workers": 2,
    "jpg_writer_queue_size": 512,
    "pico_first_view_enabled": False,
    "pico_first_view_device": None,
    "pico_first_view_width": None,
    "pico_first_view_height": None,
    "pico_first_view_fps": None,
    "l515_cameras": [],
    "dry_run": False,
}


def load_yaml_mapping(path: Path) -> dict:
    try:
        import yaml  # noqa: WPS433
    except ImportError as exc:
        raise SystemExit("缺少 PyYAML，无法读取采集配置文件") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"采集配置必须是 YAML mapping: {path}")
    return data


def safe_task_dir_name(task_name: str) -> str:
    name = str(task_name).strip().replace("/", "_").replace("\\", "_")
    if not name or name in {".", ".."}:
        raise ValueError("--task-name 不能为空，也不能是 . 或 ..")
    return name


def normalize_l515_camera_configs(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("配置 l515_cameras 必须是 list")
    result: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"l515_cameras[{idx}] 必须是 YAML mapping")
        camera = dict(item)
        if camera.get("enabled", True) is False:
            continue
        if not camera.get("name"):
            camera["name"] = f"l515_{idx}"
        result.append(camera)
    return result


def parse_resolution_config(
    value: object,
    default: tuple[int, int],
) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = value.lower().replace(",", "x").split("x")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"resolution 必须是 WIDTHxHEIGHT 或 [WIDTH, HEIGHT]，收到 {value!r}")
    if len(parts) != 2:
        raise ValueError(f"resolution 必须包含两个数值，收到 {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"resolution 必须为正数，收到 {value!r}")
    return width, height


def default_out() -> Path:
    return Path("data")


def output_root_from_out(out: Path) -> Path:
    """Treat legacy PKL-style out values as an output root."""
    out = Path(out)
    if out.suffix.lower() == ".pkl":
        return out.parent
    return out


def task_output_dir(out: Path, task_name: str) -> Path:
    return output_root_from_out(out) / safe_task_dir_name(task_name)


def task_bundle_pkl_path(out: Path, task_name: str) -> Path:
    """Backward-compatible alias: returns the task output directory."""
    return task_output_dir(out, task_name)


def load_collect_config(
    config_path: Path,
    task_name: str | None,
    *,
    allow_l515: bool = True,
) -> CollectConfig:
    raw = load_yaml_mapping(config_path)
    unknown = sorted(set(raw) - set(YAML_CONFIG_DEFAULTS))
    if unknown:
        raise ValueError(f"采集配置包含未知字段: {unknown}")

    values = dict(YAML_CONFIG_DEFAULTS)
    values.update(raw)

    hand = str(values["hand"])
    if hand not in {"right", "left"}:
        raise ValueError("配置 hand 只能是 right 或 left")

    mvs_sync_mode = str(values["mvs_sync_mode"])
    if mvs_sync_mode not in {"mvs_buffered", "mvs_master", "nearest_pico_receive", "latest"}:
        raise ValueError(
            "配置 mvs_sync_mode 只能是 mvs_buffered、mvs_master、nearest_pico_receive 或 latest"
        )
    mvs_gain_auto = str(values["mvs_gain_auto"]).strip().lower()
    if mvs_gain_auto not in {"off", "once", "continuous"}:
        raise ValueError("配置 mvs_gain_auto 只能是 off、once 或 continuous")
    mvs_gain_db = float(values["mvs_gain_db"])
    if not math.isfinite(mvs_gain_db) or mvs_gain_db < 0:
        raise ValueError("配置 mvs_gain_db 必须是 >= 0 的有限数值")
    jpg_quality = int(values["jpg_quality"])
    if jpg_quality < 1 or jpg_quality > 100:
        raise ValueError("配置 jpg_quality 必须在 1 到 100 之间")
    jpg_writer_workers = int(values["jpg_writer_workers"])
    if jpg_writer_workers < 1:
        raise ValueError("配置 jpg_writer_workers 必须 >= 1")
    jpg_writer_queue_size = int(values["jpg_writer_queue_size"])
    if jpg_writer_queue_size < 1:
        raise ValueError("配置 jpg_writer_queue_size 必须 >= 1")

    pico_first_view_enabled = bool(values["pico_first_view_enabled"])
    pico_first_view_device = values["pico_first_view_device"]
    if pico_first_view_enabled and pico_first_view_device is None:
        raise ValueError(
            "配置 pico_first_view_enabled=true 时必须设置 pico_first_view_device"
        )

    def _optional_positive_int(name: str) -> int | None:
        value = values[name]
        if value is None:
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError(f"配置 {name} 必须是正整数")
        return parsed

    pico_first_view_width = _optional_positive_int("pico_first_view_width")
    pico_first_view_height = _optional_positive_int("pico_first_view_height")
    pico_first_view_fps = (
        None
        if values["pico_first_view_fps"] is None
        else float(values["pico_first_view_fps"])
    )
    if pico_first_view_fps is not None and (
        not math.isfinite(pico_first_view_fps) or pico_first_view_fps <= 0
    ):
        raise ValueError("配置 pico_first_view_fps 必须是正数且为有限数值")

    resolved_task_name = task_name if task_name is not None else values["task_name"]
    if resolved_task_name is None:
        raise ValueError("采集时必须在 YAML task_name 或命令行 --task-name 中指定任务名称")
    resolved_task_name = str(resolved_task_name)

    out_value = values["out"]
    out = default_out() if out_value is None else Path(str(out_value)).expanduser()
    out = task_output_dir(out, resolved_task_name)

    l515_cameras = (
        normalize_l515_camera_configs(values["l515_cameras"])
        if allow_l515
        else []
    )

    return CollectConfig(
        task_name=resolved_task_name,
        out=out,
        hz=float(values["hz"]),
        cache_hz=float(values["cache_hz"]),
        hand=hand,
        retargeting_yaml=(
            None
            if values["retargeting_yaml"] is None
            else str(values["retargeting_yaml"])
        ),
        wuji_retargeting_yaml=(
            None
            if values["wuji_retargeting_yaml"] is None
            else str(values["wuji_retargeting_yaml"])
        ),
        duration=None if values["duration"] is None else float(values["duration"]),
        key_debounce=float(values["key_debounce"]),
        keep_invalid=bool(values["keep_invalid"]),
        mvs_serial=None if values["mvs_serial"] is None else str(values["mvs_serial"]),
        mvs_resolution=parse_resolution_config(values["mvs_resolution"], (480, 480)),
        mvs_input_resolution=parse_resolution_config(
            values["mvs_input_resolution"],
            (1440, 1080),
        ),
        mvs_auto_crop_black_border=bool(values["mvs_auto_crop_black_border"]),
        mvs_center_crop=bool(values["mvs_center_crop"]),
        mvs_use_sensor_crop_roi=bool(values["mvs_use_sensor_crop_roi"]),
        mvs_exposure_time_us=float(values["mvs_exposure_time_us"]),
        mvs_gain_auto=mvs_gain_auto,
        mvs_gain_db=mvs_gain_db,
        mvs_capture_fps=(
            None
            if values["mvs_capture_fps"] is None
            else float(values["mvs_capture_fps"])
        ),
        mvs_sync_mode=mvs_sync_mode,
        max_rgb_pico_delta_ms=float(values["max_rgb_pico_delta_ms"]),
        jpg_quality=jpg_quality,
        jpg_writer_workers=jpg_writer_workers,
        jpg_writer_queue_size=jpg_writer_queue_size,
        pico_first_view_enabled=pico_first_view_enabled,
        pico_first_view_device=pico_first_view_device,
        pico_first_view_width=pico_first_view_width,
        pico_first_view_height=pico_first_view_height,
        pico_first_view_fps=pico_first_view_fps,
        l515_cameras=l515_cameras,
        dry_run=bool(values["dry_run"]),
        config_path=config_path.resolve(),
    )


def load_pico_mvs_config(config_path: Path, task_name: str | None) -> CollectConfig:
    return load_collect_config(
        config_path=config_path,
        task_name=task_name,
        allow_l515=False,
    )


def list_mvs_cameras() -> None:
    from mvs_cpp import get_mvs_usb_link_info, list_mvs_device_serials

    serials = list(list_mvs_device_serials())
    if not serials:
        print("[MVS] 未发现相机")
        return
    print(f"[MVS] 发现 {len(serials)} 个相机:")
    for idx, serial in enumerate(serials):
        usb_link = get_mvs_usb_link_info(serial)
        usb_suffix = f"  usb={usb_link.summary()}" if usb_link is not None else ""
        print(f"  {idx}: {serial}{usb_suffix}")


def resolve_mvs_serial(config: CollectConfig) -> str:
    if config.mvs_serial:
        return str(config.mvs_serial)

    from mvs_cpp import list_mvs_device_serials

    serials = list(list_mvs_device_serials())
    if not serials:
        raise RuntimeError(
            "未发现 MVS 相机；请检查 MVS SDK/USB 连接，或在 YAML 中设置 mvs_serial"
        )
    serial = str(serials[0])
    print(f"[Init] 自动选择第一个 MVS 相机 serial={serial}")
    return serial


def build_mvs_config(config: CollectConfig):
    from mvs_cpp import MVSConfig

    width, height = config.mvs_resolution
    input_width, input_height = config.mvs_input_resolution
    return MVSConfig(
        serial=resolve_mvs_serial(config),
        fps=float(config.mvs_capture_fps or config.hz),
        width=int(width),
        height=int(height),
        input_width=int(input_width),
        input_height=int(input_height),
        center_crop=bool(config.mvs_center_crop),
        auto_crop_black_border=bool(config.mvs_auto_crop_black_border),
        exposure_time_us=float(config.mvs_exposure_time_us),
        gain_auto=str(config.mvs_gain_auto),
        gain_db=float(config.mvs_gain_db),
        use_subprocess=True,
        use_sensor_crop_roi=bool(config.mvs_use_sensor_crop_roi),
    )
