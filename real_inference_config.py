#!/usr/bin/env python3
"""Configuration, hand-runtime setup, and environment compatibility adapters."""

from __future__ import annotations

import pathlib
import sys
import threading
import time

import numpy as np
import yaml


ROOT = pathlib.Path(__file__).parent
HAND_BACKENDS = ("linker_o6", "wuji_hand")
DISABLED_HAND_VALUES = {"", "none", "null", "false", "off", "disable", "disabled"}

def load_yaml_mapping(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{yaml_path} must contain a YAML mapping")
    return cfg

def load_config(yaml_path: str, overrides: dict) -> dict:
    """加载 yaml 配置，并用命令行参数覆盖其中非 None 的字段。"""
    cfg = load_yaml_mapping(yaml_path)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg

def resolve_episode_record_fps(eval_cfg: dict, robot_cfg: dict, fallback: float = 30.0) -> float:
    cameras_cfg = robot_cfg.get("cameras", {}) if isinstance(robot_cfg, dict) else {}
    if not isinstance(cameras_cfg, dict):
        cameras_cfg = {}
    value = eval_cfg.get(
        "episode_record_fps",
        eval_cfg.get(
            "save_episode_fps",
            cameras_cfg.get("record_fps", cameras_cfg.get("capture_fps", fallback)),
        ),
    )
    if isinstance(value, (list, tuple)):
        if not value:
            value = fallback
        else:
            value = value[0]
    record_fps = float(value)
    if not np.isfinite(record_fps) or record_fps <= 0.0:
        raise ValueError(f"episode_record_fps must be finite and > 0, got {value!r}")
    return record_fps

def normalize_hand_name(value):
    if value is None or value is False:
        return None
    if isinstance(value, str):
        hand = value.strip().lower()
        if hand in DISABLED_HAND_VALUES:
            return None
        if hand == "wujihand":
            hand = "wuji_hand"
        if hand in HAND_BACKENDS:
            return hand
    raise ValueError(f"hand must be one of {HAND_BACKENDS} or none, got {value!r}")

def legacy_enabled_hand(*configs: dict):
    enabled = []
    for hand in HAND_BACKENDS:
        for cfg in configs:
            hand_cfg = cfg.get(hand, {}) if isinstance(cfg, dict) else {}
            if isinstance(hand_cfg, dict) and hand_cfg.get("enabled", False):
                enabled.append(hand)
                break
    if len(enabled) > 1:
        raise ValueError(f"Only one hand can be enabled, got {enabled}")
    return enabled[0] if enabled else None

def resolve_hand_backend(eval_cfg: dict, robot_cfg: dict):
    if "hand" in eval_cfg:
        return normalize_hand_name(eval_cfg.get("hand"))
    return legacy_enabled_hand(eval_cfg, robot_cfg)

def merged_hand_config(eval_cfg: dict, robot_cfg: dict, hand: str) -> dict:
    merged = {}
    for source in (robot_cfg.get(hand, {}), eval_cfg.get(hand, {})):
        if source is None:
            continue
        if not isinstance(source, dict):
            raise ValueError(f"{hand} config must be a mapping")
        merged.update(source)
    return merged

def open_o6_hand(o6_cfg: dict):
    dexumi_root = pathlib.Path(
        o6_cfg.get("dexumi_root", "/home/zjc/Desktop/human2dex")
    ).expanduser().resolve()
    o6_driver_dir = dexumi_root / "o6_right_hand"
    controller_path = o6_driver_dir / "controller.py"
    if not controller_path.exists():
        raise FileNotFoundError(
            f"O6 controller not found: {controller_path}. "
            "Set linker_o6.dexumi_root to the DexUMI repo root."
        )

    driver_dir = str(o6_driver_dir)
    if driver_dir not in sys.path:
        sys.path.insert(0, driver_dir)

    from controller import O6RightHand

    hand = O6RightHand(
        can_channel=o6_cfg.get("can_channel", "can0"),
        bitrate=int(o6_cfg.get("bitrate", 1_000_000)),
    )
    speed = o6_cfg.get("speed", None)
    if speed is not None:
        hand.set_speed(int(speed))
    torque = o6_cfg.get("torque", None)
    if torque is not None:
        hand.set_torque(int(torque))
    return hand

def _as_qpos20(value, name: str = "qpos") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (20,):
        raise ValueError(f"{name} must contain 20 values, got shape {np.asarray(value).shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr

def _import_wuji_runtime(wuji_cfg: dict):
    wuji_root = pathlib.Path(
        wuji_cfg.get("wuji_demo_root", wuji_cfg.get("wuji_root", "/home/zjc/Desktop/wuji_demo"))
    ).expanduser().resolve()
    if str(wuji_root) not in sys.path:
        sys.path.insert(0, str(wuji_root))
    from wuji_pico_hand.filters import LowPassFilter
    from wuji_pico_hand.hand_driver import WujiHandDriver

    return WujiHandDriver, LowPassFilter

def open_wuji_hand(wuji_cfg: dict, dry_run: bool):
    WujiHandDriver, LowPassFilter = _import_wuji_runtime(wuji_cfg)
    driver = WujiHandDriver(
        serial_number=wuji_cfg.get("serial_number"),
        dry_run=bool(dry_run),
        timeout=wuji_cfg.get("timeout"),
    )
    filt = LowPassFilter(float(wuji_cfg.get("filter_alpha", wuji_cfg.get("alpha", 0.5))))
    limits = None
    if bool(wuji_cfg.get("refresh_limits", True)) and not dry_run:
        limits = driver.refresh_limits()
    driver.enable()
    print(f"[INFO] Wuji hand {'dry-run ' if dry_run else ''}enabled.")
    if limits is not None:
        print("[INFO] Wuji lower:", np.round(limits.lower, 4).tolist())
        print("[INFO] Wuji upper:", np.round(limits.upper, 4).tolist())
    return driver, filt

class NoOpScalarGripper:
    """BimanualUmiEnv 的 1D gripper 占位对象；O6 控制走单独 CAN。"""

    def __init__(self, init_pos: float = 0.0):
        self._pos = float(init_pos)
        self._ready = False
        now = time.time()
        self._timestamps = [now - 1e-3, now]
        self._positions = [self._pos, self._pos]
        self._lock = threading.Lock()

    @property
    def is_ready(self):
        return self._ready

    def start(self, wait=True):
        self._ready = True

    def start_wait(self):
        self._ready = True

    def stop(self, wait=True):
        self._ready = False

    def stop_wait(self):
        self._ready = False

    def schedule_waypoint(self, pos: float, target_time: float, *_, **__):
        with self._lock:
            self._pos = float(pos)
            self._timestamps.append(float(target_time))
            self._positions.append(self._pos)
            self._timestamps = self._timestamps[-200:]
            self._positions = self._positions[-200:]

    def get_state(self):
        return {
            "gripper_position": float(self._pos),
            "gripper_timestamp": time.time(),
        }

    def get_all_state(self):
        with self._lock:
            now = time.time()
            return {
                "gripper_timestamp": np.asarray(self._timestamps + [now], dtype=np.float64),
                "gripper_position": np.asarray(self._positions + [self._pos], dtype=np.float32),
            }
