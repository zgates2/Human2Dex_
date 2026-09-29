#!/usr/bin/env python3
"""Add wrist-camera model predictions to recorded DexUMI PKL episodes."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
TELEOP_ROOT = REPO_ROOT / "teleop"
WRIST_ROOT = REPO_ROOT / "wrist"
for _path in (REPO_ROOT, TELEOP_ROOT, WRIST_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from wrist_pose.transforms import WristImageTransform, read_rgb  # noqa: E402
from wrist_pose.stage2_projection import (  # noqa: E402
    ProjectionHead,
    ProjectionHeadInfo,
    forward_projection_from_stage1_outputs,
    load_projection_head,
    transform_model_uv_to_original,
    uv_valid_mask,
)


DEFAULT_CHECKPOINT = REPO_ROOT / "wrist" / "outputs" / "stage2" / "runs" / "stage2_mano_a100x8" / "checkpoints" / "best.pt"
DEFAULT_LINKER_YAML = REPO_ROOT / "linker_o6" / "linker_o6" / "linker_o6.yml"
DEFAULT_LINKER_URDF_DIR = REPO_ROOT / "linker_o6"
DEFAULT_WUJI_YAML = REPO_ROOT / "wuji_retargeting" / "config" / "adaptive_analytical_pico.yaml"

NUMPY2_CORE_PREFIX = "numpy._core"
NUMPY1_CORE_PREFIX = "numpy.core"


def ensure_model_imports() -> None:
    global WristDinoPoseModel
    global WristDinoManoPoseModel
    global expand20_to21
    global infer_pose_output_dim_from_state_dict
    global load_model_state_compatible
    global load_yaml
    global resolve_config_paths

    if "WristDinoPoseModel" in globals():
        return
    from wrist_pose.model import WristDinoPoseModel as _WristDinoPoseModel  # noqa: WPS433
    from wrist_pose.model import expand20_to21 as _expand20_to21  # noqa: WPS433
    from wrist_pose.stage2_model import WristDinoManoPoseModel as _WristDinoManoPoseModel  # noqa: WPS433
    from wrist_pose.stage2_model import infer_pose_output_dim_from_state_dict as _infer_pose_output_dim_from_state_dict  # noqa: WPS433
    from wrist_pose.utils import load_model_state_compatible as _load_model_state_compatible  # noqa: WPS433
    from wrist_pose.utils import load_yaml as _load_yaml  # noqa: WPS433
    from wrist_pose.utils import resolve_config_paths as _resolve_config_paths  # noqa: WPS433

    WristDinoPoseModel = _WristDinoPoseModel
    WristDinoManoPoseModel = _WristDinoManoPoseModel
    expand20_to21 = _expand20_to21
    infer_pose_output_dim_from_state_dict = _infer_pose_output_dim_from_state_dict
    load_model_state_compatible = _load_model_state_compatible
    load_yaml = _load_yaml
    resolve_config_paths = _resolve_config_paths


def ensure_retarget_imports() -> None:
    global LinkerO6Retargeter
    global WujiRetargeter
    global angles_rad_to_cmd

    if "LinkerO6Retargeter" in globals():
        return
    from episode_io import angles_rad_to_cmd as _angles_rad_to_cmd  # noqa: WPS433
    from retargeter import LinkerO6Retargeter as _LinkerO6Retargeter  # noqa: WPS433
    from wuji_retargeting import Retargeter as _WujiRetargeter  # noqa: WPS433

    LinkerO6Retargeter = _LinkerO6Retargeter
    WujiRetargeter = _WujiRetargeter
    angles_rad_to_cmd = _angles_rad_to_cmd


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == NUMPY2_CORE_PREFIX or module.startswith(f"{NUMPY2_CORE_PREFIX}."):
                compat_module = NUMPY1_CORE_PREFIX + module[len(NUMPY2_CORE_PREFIX) :]
                return super().find_class(compat_module, name)
            raise


def install_numpy_pickle_compat() -> list[str]:
    try:
        core = importlib.import_module("numpy._core")
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            core = importlib.import_module("numpy.core")
    aliases = {
        "numpy._core": core,
        "numpy._core.multiarray": getattr(core, "multiarray", None),
        "numpy._core.numeric": getattr(core, "numeric", None),
    }
    added = []
    for name, module in aliases.items():
        if module is not None and name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    return added


def read_pkl(path: Path) -> dict[str, Any]:
    added = install_numpy_pickle_compat()
    try:
        with path.open("rb") as f:
            data = NumpyCompatUnpickler(f).load()
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict with list field 'messages': {path}")
    return data


def atomic_write_pkl(data: dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def iter_pkl_paths(data_root: Path) -> list[Path]:
    return sorted(path for path in data_root.glob("**/*.pkl") if path.is_file())


def valid_pts21(value: Any) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return False
    return arr.shape == (21, 3) and np.all(np.isfinite(arr))


def valid_o6_command(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.shape == (6,) and np.all(np.isfinite(arr))


def valid_o6_radians(value: Any) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return False
    return arr.shape == (6,) and np.all(np.isfinite(arr))


def valid_wuji_command(value: Any) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return False
    return arr.shape == (5, 4) and np.all(np.isfinite(arr))


def valid_uv(value: Any, shape: tuple[int, int]) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return False
    return arr.shape == shape and np.all(np.isfinite(arr))


def valid_bool_mask(value: Any, length: int) -> bool:
    try:
        arr = np.asarray(value, dtype=bool)
    except Exception:
        return False
    return arr.shape == (length,)


def wrist_fields_complete(
    msg: dict[str, Any],
    mirror_standard_fields: bool = False,
    require_projection: bool = False,
) -> bool:
    complete = (
        msg.get("wrist_infer_error") is None
        and valid_pts21(msg.get("wrist_pts21_mano"))
        and valid_o6_command(msg.get("wrist_o6_command"))
        and valid_o6_radians(msg.get("wrist_o6_joint_radians"))
        and valid_wuji_command(msg.get("wrist_wuji_command"))
    )
    if mirror_standard_fields:
        complete = (
            complete
            and valid_o6_command(msg.get("o6_command"))
            and valid_wuji_command(msg.get("wuji_command"))
        )
    if require_projection:
        complete = (
            complete
            and msg.get("wrist_projection_error") is None
            and valid_uv(msg.get("wrist_uv21_rgb"), (21, 2))
            and valid_bool_mask(msg.get("wrist_uv21_valid"), 21)
            and valid_uv(msg.get("wrist_anchor_uv_rgb"), (6, 2))
            and valid_bool_mask(msg.get("wrist_anchor_valid"), 6)
        )
    return complete


def resolve_image_path(pkl_path: Path, msg: dict[str, Any], image_field: str) -> Path | None:
    rel_image = msg.get(image_field)
    if not isinstance(rel_image, str) or not rel_image:
        return None
    image_path = Path(rel_image)
    if image_path.is_absolute():
        return image_path
    return (pkl_path.parent / image_path).resolve()


@dataclass(frozen=True)
class FrameInput:
    frame_idx: int
    image_path: str


@dataclass(frozen=True)
class ChunkFrameInput:
    pkl_index: int
    frame_idx: int
    image_path: str


class WristFrameDataset(Dataset):
    def __init__(self, frames: list[FrameInput], transform: WristImageTransform) -> None:
        self.frames = frames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.frames[idx]
        try:
            rgb = read_rgb(item.image_path)
            transformed = self.transform(rgb)
            return {
                "ok": True,
                "frame_idx": int(item.frame_idx),
                "image": transformed.image,
                "scale": float(transformed.scale),
                "offset_xy": tuple(float(v) for v in transformed.offset_xy),
                "original_hw": tuple(int(v) for v in transformed.original_hw),
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "frame_idx": int(item.frame_idx),
                "image": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


def collate_wrist_frames(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = []
    frame_indices = []
    scales = []
    offsets = []
    original_hw = []
    errors = []
    for item in batch:
        frame_idx = int(item["frame_idx"])
        if item["ok"]:
            images.append(item["image"])
            frame_indices.append(frame_idx)
            scales.append(float(item["scale"]))
            offsets.append(tuple(item["offset_xy"]))
            original_hw.append(tuple(item["original_hw"]))
        else:
            errors.append((frame_idx, str(item["error"])))
    return {
        "frame_indices": frame_indices,
        "images": torch.stack(images, dim=0) if images else None,
        "scales": scales,
        "offsets": offsets,
        "original_hw": original_hw,
        "errors": errors,
    }


class ChunkWristFrameDataset(Dataset):
    def __init__(self, frames: list[ChunkFrameInput], transform: WristImageTransform) -> None:
        self.frames = frames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.frames[idx]
        try:
            rgb = read_rgb(item.image_path)
            transformed = self.transform(rgb)
            return {
                "ok": True,
                "pkl_index": int(item.pkl_index),
                "frame_idx": int(item.frame_idx),
                "image": transformed.image,
                "scale": float(transformed.scale),
                "offset_xy": tuple(float(v) for v in transformed.offset_xy),
                "original_hw": tuple(int(v) for v in transformed.original_hw),
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "pkl_index": int(item.pkl_index),
                "frame_idx": int(item.frame_idx),
                "image": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


def collate_chunk_wrist_frames(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = []
    frame_refs = []
    scales = []
    offsets = []
    original_hw = []
    errors = []
    for item in batch:
        ref = (int(item["pkl_index"]), int(item["frame_idx"]))
        if item["ok"]:
            images.append(item["image"])
            frame_refs.append(ref)
            scales.append(float(item["scale"]))
            offsets.append(tuple(item["offset_xy"]))
            original_hw.append(tuple(item["original_hw"]))
        else:
            errors.append((ref, str(item["error"])))
    return {
        "frame_refs": frame_refs,
        "images": torch.stack(images, dim=0) if images else None,
        "scales": scales,
        "offsets": offsets,
        "original_hw": original_hw,
        "errors": errors,
    }


def choose_precision(requested: str, device: torch.device) -> str:
    if device.type != "cuda" or requested == "fp32":
        return "fp32"
    if requested == "bf16":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp16"


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def state_dict_from_checkpoint(checkpoint: Any, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
        if tensor_values:
            return checkpoint
    raise ValueError(f"checkpoint does not contain a model state dict: {checkpoint_path}")


def detect_checkpoint_type(state_dict: dict[str, torch.Tensor], requested: str) -> str:
    if requested != "auto":
        return requested
    clean_keys = [key.removeprefix("module.") for key in state_dict.keys()]
    if any(key.startswith("pose_head.") or key.startswith("mano_layer.") for key in clean_keys):
        return "stage2"
    if any(key.startswith("head.") or key.startswith("pool.") for key in clean_keys):
        return "stage1"
    return "stage2"


def load_checkpoint_config(
    checkpoint: Any,
    config_path: Path | None,
    default_config: Path,
) -> dict[str, Any]:
    if config_path is not None:
        return load_yaml(config_path)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"]
    return load_yaml(default_config)


def load_any_wrist_model(
    checkpoint_path: Path,
    config_path: Path | None,
    dino_dir: Path | None,
    mano_model_dir: Path | None,
    device: torch.device,
    requested_model_type: str,
) -> tuple[torch.nn.Module, WristImageTransform, dict[str, Any], str]:
    ensure_model_imports()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state_dict_from_checkpoint(checkpoint, checkpoint_path)
    model_type = detect_checkpoint_type(state_dict, requested_model_type)

    default_config = WRIST_ROOT / "configs" / ("stage2_mano.yaml" if model_type == "stage2" else "baseline.yaml")
    cfg = load_checkpoint_config(checkpoint, config_path, default_config)
    resolve_config_paths(cfg, REPO_ROOT)
    if dino_dir is not None:
        cfg.setdefault("model", {})["dino_dir"] = str(dino_dir.expanduser().resolve())

    image_size = int(cfg["data"].get("image_size", 448))
    model_cfg = cfg.get("model", {})
    if model_type == "stage2":
        mano_cfg = cfg.setdefault("mano", {})
        if mano_model_dir is not None:
            mano_cfg["model_dir"] = str(mano_model_dir.expanduser().resolve())
        pose_output_dim = infer_pose_output_dim_from_state_dict(state_dict)
        model = WristDinoManoPoseModel(
            dino_dir=Path(model_cfg["dino_dir"]),
            mano_model_dir=Path(mano_cfg["model_dir"]),
            image_size=image_size,
            feature_dim=int(model_cfg.get("feature_dim", 384)),
            patch_size=int(model_cfg.get("patch_size", 16)),
            head_dropout=float(model_cfg.get("head_dropout", 0.1)),
            decoder_layers=int(model_cfg.get("decoder_layers", 2)),
            decoder_heads=int(model_cfg.get("decoder_heads", 6)),
            mano_side=str(mano_cfg.get("side", "right")),
            mano_use_pca=bool(mano_cfg.get("use_pca", False)),
            mano_flat_hand_mean=bool(mano_cfg.get("flat_hand_mean", False)),
            mano_output_scale=float(mano_cfg.get("output_scale", 1.0)),
            pose_output_dim=pose_output_dim,
        )
    elif model_type == "stage1":
        model = WristDinoPoseModel(
            dino_dir=Path(model_cfg["dino_dir"]),
            image_size=image_size,
            feature_dim=int(model_cfg.get("feature_dim", 384)),
            patch_size=int(model_cfg.get("patch_size", 16)),
            pooling_dropout=float(model_cfg.get("pooling_dropout", 0.0)),
            head_dropout=float(model_cfg.get("head_dropout", 0.1)),
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")

    load_model_state_compatible(model, state_dict, strict=True)
    model.to(device).eval()
    transform = WristImageTransform(image_size=image_size, train=False)
    return model, transform, cfg, model_type


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    precision: str,
    model_type: str,
    channels_last: bool,
    projection_head: ProjectionHead | None = None,
    projection_info: ProjectionHeadInfo | None = None,
    image_size: int | None = None,
) -> dict[str, np.ndarray]:
    if channels_last and device.type == "cuda":
        images = images.to(device=device, non_blocking=True, memory_format=torch.channels_last)
    else:
        images = images.to(device=device, non_blocking=True)
    with autocast_context(device, precision):
        outputs = model(images, return_tokens=projection_head is not None) if model_type == "stage2" else model(images)
        if model_type == "stage2":
            joints21 = outputs["joints21"]
        else:
            if projection_head is not None:
                raise RuntimeError("projection head requires a stage2 MANO wrist model")
            joints21 = expand20_to21(outputs["joints20"])
        result = {"pts21": joints21.detach().cpu().float().numpy().astype(np.float32)}
        if projection_head is not None:
            if projection_info is None:
                raise RuntimeError("projection_info is required when projection_head is set")
            projected = forward_projection_from_stage1_outputs(
                outputs=outputs,
                head=projection_head,
                labels=projection_info.labels,
                image_size=int(image_size or images.shape[-1]),
                min_scale=float(projection_info.min_scale),
                max_scale=float(projection_info.max_scale),
            )
            result["uv21_model"] = projected["uv21"].detach().cpu().float().numpy().astype(np.float32)
            result["anchor_uv_model"] = projected["anchor_uv"].detach().cpu().float().numpy().astype(np.float32)
            result["proj_scale"] = projected["proj_scale"].detach().cpu().float().numpy().astype(np.float32)
    return result


def retarget_wuji_mano_points(retargeter: WujiRetargeter, pts21_mano: Any) -> np.ndarray:
    points = np.asarray(pts21_mano, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"wrist_pts21_mano shape must be (21, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("wrist_pts21_mano contains NaN or Inf")

    keypoints = points.copy()
    rotation_xyz = getattr(retargeter, "rotation_xyz", {}) or {}
    has_rotation = any(float(rotation_xyz.get(axis, 0.0)) != 0.0 for axis in ("x", "y", "z"))
    if has_rotation:
        keypoints = retargeter._apply_rotation(keypoints)
    if getattr(retargeter, "_has_offset", False):
        keypoints = retargeter._apply_offset(keypoints)

    qpos = np.asarray(retargeter.optimizer.solve(keypoints), dtype=np.float32).reshape(-1)
    qpos = np.asarray(retargeter.lp_filter.next(qpos), dtype=np.float32).reshape(-1)
    if qpos.shape != (20,):
        raise RuntimeError(f"Wuji retargeter returned {qpos.shape}, expected (20,)")
    if not np.all(np.isfinite(qpos)):
        raise RuntimeError("Wuji retargeter returned NaN or Inf")
    return qpos.reshape(5, 4).astype(np.float32, copy=False)


def reset_wuji_retargeter(retargeter: WujiRetargeter) -> None:
    if hasattr(retargeter, "reset"):
        retargeter.reset()
        return
    optimizer = getattr(retargeter, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "last_qpos"):
        optimizer.last_qpos = None
    lp_filter = getattr(retargeter, "lp_filter", None)
    if lp_filter is not None:
        if hasattr(lp_filter, "reset"):
            lp_filter.reset()
        elif hasattr(lp_filter, "is_init"):
            lp_filter.is_init = False


def set_wrist_fields_error(msg: dict[str, Any], error: str, overwrite: bool) -> bool:
    changed = False
    updates = {
        "wrist_pts21_mano": None,
        "wrist_o6_joint_radians": None,
        "wrist_o6_command": None,
        "wrist_wuji_command": None,
        "wrist_wuji_error": None,
        "wrist_uv21_rgb": None,
        "wrist_uv21_valid": None,
        "wrist_anchor_uv_rgb": None,
        "wrist_anchor_valid": None,
        "wrist_projection_error": error,
        "wrist_infer_error": error,
    }
    for key, value in updates.items():
        if overwrite or key not in msg or msg.get(key) is not value:
            msg[key] = value
            changed = True
    return changed


def set_wrist_fields_success(
    msg: dict[str, Any],
    pts21: np.ndarray,
    o6_radians: np.ndarray,
    o6_command: np.ndarray,
    wuji_command: np.ndarray | None,
    projection: dict[str, Any] | None,
    overwrite: bool,
    mirror_standard_fields: bool,
) -> bool:
    changed = False
    updates = {
        "wrist_pts21_mano": np.asarray(pts21, dtype=np.float32).reshape(21, 3),
        "wrist_o6_joint_radians": np.asarray(o6_radians, dtype=np.float32).reshape(6),
        "wrist_o6_command": np.asarray(o6_command, dtype=np.uint8).reshape(6),
        "wrist_infer_error": None,
    }
    # ``prediction`` is present whenever RGB -> MANO21 inference succeeded.
    # It only contains these 2D keys when a projection head was requested.
    # Do not confuse a valid 3D-only prediction with a malformed projection.
    projection_keys = (
        "uv21_rgb",
        "uv21_valid",
        "anchor_uv_rgb",
        "anchor_valid",
    )
    if projection is not None and all(key in projection for key in projection_keys):
        updates.update(
            {
                "wrist_uv21_rgb": np.asarray(projection["uv21_rgb"], dtype=np.float32).reshape(21, 2),
                "wrist_uv21_valid": np.asarray(projection["uv21_valid"], dtype=bool).reshape(21),
                "wrist_anchor_uv_rgb": np.asarray(projection["anchor_uv_rgb"], dtype=np.float32).reshape(-1, 2),
                "wrist_anchor_valid": np.asarray(projection["anchor_valid"], dtype=bool).reshape(-1),
                "wrist_projection_error": None,
            }
        )
    if wuji_command is None:
        updates["wrist_wuji_command"] = None
        updates["wrist_wuji_error"] = "skipped"
    else:
        updates["wrist_wuji_command"] = np.asarray(wuji_command, dtype=np.float32).reshape(5, 4)
        updates["wrist_wuji_error"] = None
    if mirror_standard_fields:
        updates["o6_command"] = np.asarray(o6_command, dtype=np.uint8).reshape(6)
        updates["wuji_command"] = None if wuji_command is None else np.asarray(wuji_command, dtype=np.float32).reshape(5, 4)
    for key, value in updates.items():
        if overwrite or key not in msg or msg.get(key) is not value:
            msg[key] = value
            changed = True
    return changed


def infer_needed_frames(
    pkl_path: Path,
    messages: list[Any],
    image_field: str,
    overwrite: bool,
    mirror_standard_fields: bool,
    use_existing_wrist_pts21: bool,
    require_projection: bool,
    limit_frames: int | None,
    stats: Counter[str],
) -> tuple[list[FrameInput], dict[int, str]]:
    frames: list[FrameInput] = []
    errors: dict[int, str] = {}
    max_count = len(messages) if limit_frames is None else min(len(messages), int(limit_frames))
    for frame_idx in range(max_count):
        msg = messages[frame_idx]
        if not isinstance(msg, dict):
            stats["message_not_dict"] += 1
            continue
        if not overwrite and wrist_fields_complete(
            msg,
            mirror_standard_fields=mirror_standard_fields,
            require_projection=require_projection,
        ):
            stats["already_complete"] += 1
            continue
        if use_existing_wrist_pts21 and valid_pts21(msg.get("wrist_pts21_mano")) and not require_projection:
            stats["use_existing_wrist_pts21"] += 1
            continue
        image_path = resolve_image_path(pkl_path, msg, image_field)
        if image_path is None:
            errors[frame_idx] = f"missing image field: {image_field}"
            stats["missing_image_field"] += 1
            continue
        if not image_path.is_file():
            errors[frame_idx] = f"missing image file: {image_path}"
            stats["missing_image_file"] += 1
            continue
        frames.append(FrameInput(frame_idx=frame_idx, image_path=str(image_path)))
    return frames, errors


def run_prediction_loader(
    model: torch.nn.Module,
    transform: WristImageTransform,
    model_type: str,
    device: torch.device,
    precision: str,
    frames: list[FrameInput],
    batch_size: int,
    num_workers: int,
    channels_last: bool,
    projection_head: ProjectionHead | None,
    projection_info: ProjectionHeadInfo | None,
    stats: Counter[str],
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    predictions: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}
    if not frames:
        return predictions, errors

    loader = DataLoader(
        WristFrameDataset(frames, transform),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_wrist_frames,
        persistent_workers=max(0, int(num_workers)) > 0,
    )
    for batch in loader:
        for frame_idx, error in batch["errors"]:
            errors[int(frame_idx)] = error
            stats["image_read_error"] += 1
        images = batch["images"]
        if images is None:
            continue
        frame_indices = [int(v) for v in batch["frame_indices"]]
        try:
            pred_batch = predict_batch(
                model=model,
                images=images,
                device=device,
                precision=precision,
                model_type=model_type,
                channels_last=channels_last,
                projection_head=projection_head,
                projection_info=projection_info,
                image_size=transform.image_size,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for frame_idx in frame_indices:
                errors[frame_idx] = error
                stats["model_infer_error"] += 1
            continue
        for local_idx, frame_idx in enumerate(frame_indices):
            pred: dict[str, Any] = {
                "pts21": pred_batch["pts21"][local_idx].astype(np.float32, copy=False),
            }
            if "uv21_model" in pred_batch:
                uv21 = transform_model_uv_to_original(
                    pred_batch["uv21_model"][local_idx],
                    scale=float(batch["scales"][local_idx]),
                    offset_xy=batch["offsets"][local_idx],
                )
                anchor_uv = transform_model_uv_to_original(
                    pred_batch["anchor_uv_model"][local_idx],
                    scale=float(batch["scales"][local_idx]),
                    offset_xy=batch["offsets"][local_idx],
                )
                pred.update(
                    {
                        "uv21_rgb": uv21,
                        "uv21_valid": uv_valid_mask(uv21, batch["original_hw"][local_idx]),
                        "anchor_uv_rgb": anchor_uv,
                        "anchor_valid": uv_valid_mask(anchor_uv, batch["original_hw"][local_idx]),
                    }
                )
            predictions[frame_idx] = pred
            stats["predicted"] += 1
    return predictions, errors


def run_chunk_prediction_loader(
    model: torch.nn.Module,
    transform: WristImageTransform,
    model_type: str,
    device: torch.device,
    precision: str,
    frames: list[ChunkFrameInput],
    batch_size: int,
    num_workers: int,
    channels_last: bool,
    projection_head: ProjectionHead | None,
    projection_info: ProjectionHeadInfo | None,
    stats: Counter[str],
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[int, dict[int, str]]]:
    predictions: dict[int, dict[int, dict[str, Any]]] = {}
    errors: dict[int, dict[int, str]] = {}
    if not frames:
        return predictions, errors

    loader = DataLoader(
        ChunkWristFrameDataset(frames, transform),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=device.type == "cuda",
        collate_fn=collate_chunk_wrist_frames,
        persistent_workers=max(0, int(num_workers)) > 0,
        prefetch_factor=2 if max(0, int(num_workers)) > 0 else None,
    )
    for batch in loader:
        for (pkl_index, frame_idx), error in batch["errors"]:
            errors.setdefault(int(pkl_index), {})[int(frame_idx)] = error
            stats["image_read_error"] += 1
        images = batch["images"]
        if images is None:
            continue
        frame_refs = [(int(pkl_index), int(frame_idx)) for pkl_index, frame_idx in batch["frame_refs"]]
        try:
            pred_batch = predict_batch(
                model=model,
                images=images,
                device=device,
                precision=precision,
                model_type=model_type,
                channels_last=channels_last,
                projection_head=projection_head,
                projection_info=projection_info,
                image_size=transform.image_size,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            for pkl_index, frame_idx in frame_refs:
                errors.setdefault(pkl_index, {})[frame_idx] = error
                stats["model_infer_error"] += 1
            continue
        for local_idx, (pkl_index, frame_idx) in enumerate(frame_refs):
            pred: dict[str, Any] = {
                "pts21": pred_batch["pts21"][local_idx].astype(np.float32, copy=False),
            }
            if "uv21_model" in pred_batch:
                uv21 = transform_model_uv_to_original(
                    pred_batch["uv21_model"][local_idx],
                    scale=float(batch["scales"][local_idx]),
                    offset_xy=batch["offsets"][local_idx],
                )
                anchor_uv = transform_model_uv_to_original(
                    pred_batch["anchor_uv_model"][local_idx],
                    scale=float(batch["scales"][local_idx]),
                    offset_xy=batch["offsets"][local_idx],
                )
                pred.update(
                    {
                        "uv21_rgb": uv21,
                        "uv21_valid": uv_valid_mask(uv21, batch["original_hw"][local_idx]),
                        "anchor_uv_rgb": anchor_uv,
                        "anchor_valid": uv_valid_mask(anchor_uv, batch["original_hw"][local_idx]),
                    }
                )
            predictions.setdefault(pkl_index, {})[frame_idx] = pred
            stats["predicted"] += 1
    return predictions, errors


def apply_predictions_to_episode(
    pkl_path: Path,
    data: dict[str, Any],
    predictions: dict[int, dict[str, Any]],
    frame_errors: dict[int, str],
    linker: LinkerO6Retargeter,
    wuji: WujiRetargeter | None,
    overwrite: bool,
    limit_frames: int | None,
    skip_wuji: bool,
    mirror_standard_fields: bool,
    use_existing_wrist_pts21: bool,
    require_projection: bool,
    stats: Counter[str],
) -> bool:
    changed = False
    messages = data["messages"]
    max_count = len(messages) if limit_frames is None else min(len(messages), int(limit_frames))
    linker.reset()
    if wuji is not None:
        reset_wuji_retargeter(wuji)

    for frame_idx in range(max_count):
        msg = messages[frame_idx]
        if not isinstance(msg, dict):
            continue

        prediction: dict[str, Any] | None = None
        pts21: np.ndarray | None = None
        should_write = overwrite or not wrist_fields_complete(
            msg,
            mirror_standard_fields=mirror_standard_fields,
            require_projection=require_projection,
        )
        if frame_idx in predictions:
            prediction = predictions[frame_idx]
            pts21 = np.asarray(prediction["pts21"], dtype=np.float32).reshape(21, 3)
        elif valid_pts21(msg.get("wrist_pts21_mano")) and (use_existing_wrist_pts21 or not overwrite):
            pts21 = np.asarray(msg["wrist_pts21_mano"], dtype=np.float32).reshape(21, 3)
            if not use_existing_wrist_pts21 and not overwrite:
                should_write = False
        elif frame_idx in frame_errors:
            if should_write:
                changed |= set_wrist_fields_error(msg, frame_errors[frame_idx], overwrite=True)
                stats["wrote_error"] += 1
            continue
        else:
            continue

        try:
            o6_radians = np.asarray(linker.retarget(pts21), dtype=np.float32).reshape(6)
            o6_command = angles_rad_to_cmd(o6_radians)
            if skip_wuji:
                wuji_command = None
                stats["wuji_skipped"] += 1
            else:
                if wuji is None:
                    raise RuntimeError("Wuji retargeter is not initialized")
                wuji_command = retarget_wuji_mano_points(wuji, pts21)
        except Exception as exc:
            if should_write:
                changed |= set_wrist_fields_error(
                    msg,
                    f"retarget_error: {type(exc).__name__}: {exc}",
                    overwrite=True,
                )
                stats["retarget_error"] += 1
            continue

        if should_write:
            changed |= set_wrist_fields_success(
                msg=msg,
                pts21=pts21,
                o6_radians=o6_radians,
                o6_command=o6_command,
                wuji_command=wuji_command,
                projection=prediction,
                overwrite=True,
                mirror_standard_fields=mirror_standard_fields,
            )
            stats["wrote_success"] += 1

    if changed:
        metadata = data.setdefault("metadata", {})
        metadata["wrist_prediction"] = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "script": "tools/add_wrist_predictions_to_pkl.py",
            "source_pkl": str(pkl_path),
            "fields": [
                "wrist_pts21_mano",
                "wrist_o6_joint_radians",
                "wrist_o6_command",
                "wrist_wuji_command",
                "wrist_wuji_error",
                "wrist_uv21_rgb",
                "wrist_uv21_valid",
                "wrist_anchor_uv_rgb",
                "wrist_anchor_valid",
                "wrist_projection_error",
                "wrist_infer_error",
            ],
            "mirror_standard_fields": bool(mirror_standard_fields),
        }
        if mirror_standard_fields:
            metadata["wrist_prediction"]["fields"].extend(["o6_command", "wuji_command"])
    return changed


def finish_pkl_result(
    pkl_path: Path,
    data: dict[str, Any],
    changed: bool,
    model_type: str,
    precision: str,
    options: dict[str, Any],
    stats: Counter[str],
    start_time: float,
) -> dict[str, Any]:
    if changed and not bool(options["dry_run"]):
        meta = data.setdefault("metadata", {}).setdefault("wrist_prediction", {})
        meta.update(
            {
                "checkpoint": str(options["checkpoint"]),
                "config": None if options["config"] is None else str(options["config"]),
                "model_type": model_type,
                "precision": precision,
                "image_field": str(options["image_field"]),
                "overwrite_wrist": bool(options["overwrite_wrist"]),
                "mirror_standard_fields": bool(options["mirror_standard_fields"]),
                "retarget_existing_wrist_only": bool(options["retarget_existing_wrist_only"]),
                "write_wrist_projection": bool(options["write_wrist_projection"]),
                "projection_head_checkpoint": options.get("projection_head_checkpoint"),
                "pipeline_mode": str(options.get("pipeline_mode", "episode")),
                "stats": dict(stats),
            }
        )
        write_start = time.perf_counter()
        atomic_write_pkl(data, pkl_path)
        stats["time_write_pkl_s"] += time.perf_counter() - write_start
        stats["files_written"] += 1
    elif changed:
        stats["dry_run_would_write"] += 1
    else:
        stats["files_unchanged"] += 1

    messages = data.get("messages", [])
    elapsed = time.perf_counter() - start_time
    return {
        "pkl": str(pkl_path),
        "frames": len(messages) if isinstance(messages, list) else 0,
        "changed": bool(changed),
        "elapsed_s": elapsed,
        "stats": dict(stats),
    }


def process_one_pkl(
    pkl_path: Path,
    model: torch.nn.Module | None,
    transform: WristImageTransform | None,
    model_type: str,
    device: torch.device,
    precision: str,
    linker: LinkerO6Retargeter,
    wuji: WujiRetargeter,
    projection_head: ProjectionHead | None,
    projection_info: ProjectionHeadInfo | None,
    options: dict[str, Any],
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    t0 = time.perf_counter()
    read_start = time.perf_counter()
    data = read_pkl(pkl_path)
    stats["time_read_pkl_s"] += time.perf_counter() - read_start
    messages = data["messages"]
    frames, pre_errors = infer_needed_frames(
        pkl_path=pkl_path,
        messages=messages,
        image_field=str(options["image_field"]),
        overwrite=bool(options["overwrite_wrist"]),
        mirror_standard_fields=bool(options["mirror_standard_fields"]),
        use_existing_wrist_pts21=bool(options["retarget_existing_wrist_only"]),
        require_projection=bool(options["write_wrist_projection"]),
        limit_frames=options["limit_frames"],
        stats=stats,
    )
    if bool(options["retarget_existing_wrist_only"]):
        predictions: dict[int, dict[str, Any]] = {}
        load_or_infer_errors: dict[int, str] = {}
    else:
        if model is None or transform is None:
            raise RuntimeError("model and transform are required unless --retarget-existing-wrist-only is set")
        infer_start = time.perf_counter()
        predictions, load_or_infer_errors = run_prediction_loader(
            model=model,
            transform=transform,
            model_type=model_type,
            device=device,
            precision=precision,
            frames=frames,
            batch_size=int(options["batch_size"]),
            num_workers=int(options["num_workers"]),
            channels_last=bool(options["channels_last"]),
            projection_head=projection_head,
            projection_info=projection_info,
            stats=stats,
        )
        stats["time_infer_loader_s"] += time.perf_counter() - infer_start
    frame_errors = {**pre_errors, **load_or_infer_errors}
    apply_start = time.perf_counter()
    changed = apply_predictions_to_episode(
        pkl_path=pkl_path,
        data=data,
        predictions=predictions,
        frame_errors=frame_errors,
        linker=linker,
        wuji=wuji,
        overwrite=bool(options["overwrite_wrist"]),
        limit_frames=options["limit_frames"],
        skip_wuji=bool(options["skip_wuji"]),
        mirror_standard_fields=bool(options["mirror_standard_fields"]),
        use_existing_wrist_pts21=bool(options["retarget_existing_wrist_only"]),
        require_projection=bool(options["write_wrist_projection"]),
        stats=stats,
    )
    stats["time_retarget_s"] += time.perf_counter() - apply_start
    return finish_pkl_result(
        pkl_path=pkl_path,
        data=data,
        changed=changed,
        model_type=model_type,
        precision=precision,
        options=options,
        stats=stats,
        start_time=t0,
    )


def process_pkl_chunk(
    pkl_paths: list[str],
    model: torch.nn.Module | None,
    transform: WristImageTransform | None,
    model_type: str,
    device: torch.device,
    precision: str,
    linker: LinkerO6Retargeter,
    wuji: WujiRetargeter | None,
    projection_head: ProjectionHead | None,
    projection_info: ProjectionHeadInfo | None,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    chunk_stats: Counter[str] = Counter()
    episodes: list[dict[str, Any]] = []
    chunk_frames: list[ChunkFrameInput] = []

    for pkl_str in pkl_paths:
        pkl_path = Path(pkl_str)
        stats: Counter[str] = Counter()
        t0 = time.perf_counter()
        read_start = time.perf_counter()
        data = read_pkl(pkl_path)
        stats["time_read_pkl_s"] += time.perf_counter() - read_start
        messages = data["messages"]
        frames, pre_errors = infer_needed_frames(
            pkl_path=pkl_path,
            messages=messages,
            image_field=str(options["image_field"]),
            overwrite=bool(options["overwrite_wrist"]),
            mirror_standard_fields=bool(options["mirror_standard_fields"]),
            use_existing_wrist_pts21=bool(options["retarget_existing_wrist_only"]),
            require_projection=bool(options["write_wrist_projection"]),
            limit_frames=options["limit_frames"],
            stats=stats,
        )
        pkl_index = len(episodes)
        for frame in frames:
            chunk_frames.append(
                ChunkFrameInput(
                    pkl_index=pkl_index,
                    frame_idx=frame.frame_idx,
                    image_path=frame.image_path,
                )
            )
        episodes.append(
            {
                "pkl_path": pkl_path,
                "data": data,
                "pre_errors": pre_errors,
                "predictions": {},
                "load_or_infer_errors": {},
                "stats": stats,
                "start_time": t0,
            }
        )

    if bool(options["retarget_existing_wrist_only"]):
        predictions_by_pkl: dict[int, dict[int, dict[str, Any]]] = {}
        errors_by_pkl: dict[int, dict[int, str]] = {}
    else:
        if model is None or transform is None:
            raise RuntimeError("model and transform are required unless --retarget-existing-wrist-only is set")
        infer_start = time.perf_counter()
        predictions_by_pkl, errors_by_pkl = run_chunk_prediction_loader(
            model=model,
            transform=transform,
            model_type=model_type,
            device=device,
            precision=precision,
            frames=chunk_frames,
            batch_size=int(options["batch_size"]),
            num_workers=int(options["num_workers"]),
            channels_last=bool(options["channels_last"]),
            projection_head=projection_head,
            projection_info=projection_info,
            stats=chunk_stats,
        )
        chunk_stats["time_infer_loader_s"] += time.perf_counter() - infer_start

    results: list[dict[str, Any]] = []
    for pkl_index, episode in enumerate(episodes):
        stats = episode["stats"]
        predictions = predictions_by_pkl.get(pkl_index, {})
        stats["predicted"] += len(predictions)
        stats["load_or_infer_error"] += len(errors_by_pkl.get(pkl_index, {}))
        if pkl_index == 0:
            stats["chunk_pkls"] += len(episodes)
            stats["chunk_frames_queued"] += len(chunk_frames)
            stats["time_infer_loader_s"] += chunk_stats.get("time_infer_loader_s", 0)
            stats["image_read_error"] += chunk_stats.get("image_read_error", 0)
            stats["model_infer_error"] += chunk_stats.get("model_infer_error", 0)
        frame_errors = {
            **episode["pre_errors"],
            **errors_by_pkl.get(pkl_index, {}),
        }
        apply_start = time.perf_counter()
        changed = apply_predictions_to_episode(
            pkl_path=episode["pkl_path"],
            data=episode["data"],
            predictions=predictions,
            frame_errors=frame_errors,
            linker=linker,
            wuji=wuji,
            overwrite=bool(options["overwrite_wrist"]),
            limit_frames=options["limit_frames"],
            skip_wuji=bool(options["skip_wuji"]),
            mirror_standard_fields=bool(options["mirror_standard_fields"]),
            use_existing_wrist_pts21=bool(options["retarget_existing_wrist_only"]),
            require_projection=bool(options["write_wrist_projection"]),
            stats=stats,
        )
        stats["time_retarget_s"] += time.perf_counter() - apply_start
        results.append(
            finish_pkl_result(
                pkl_path=episode["pkl_path"],
                data=episode["data"],
                changed=changed,
                model_type=model_type,
                precision=precision,
                options=options,
                stats=stats,
                start_time=episode["start_time"],
            )
        )
    return results


def worker_main(
    rank: int,
    device_name: str,
    pkl_paths: list[str],
    options: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    try:
        ensure_retarget_imports()
        device = torch.device(device_name)
        if device.type == "cuda" and not bool(options["retarget_existing_wrist_only"]):
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        precision = choose_precision(str(options["precision"]), device)
        if bool(options["retarget_existing_wrist_only"]):
            model = None
            transform = None
            model_type = "existing_wrist_pts21"
            projection_head = None
            projection_info = None
        else:
            model, transform, cfg, model_type = load_any_wrist_model(
                checkpoint_path=Path(options["checkpoint"]),
                config_path=None if options["config"] is None else Path(options["config"]),
                dino_dir=None if options["dino_dir"] is None else Path(options["dino_dir"]),
                mano_model_dir=None if options["mano_model_dir"] is None else Path(options["mano_model_dir"]),
                device=device,
                requested_model_type=str(options["model_type"]),
            )
            if bool(options["channels_last"]) and device.type == "cuda":
                model.to(memory_format=torch.channels_last)
            projection_head = None
            projection_info = None
            if bool(options["write_wrist_projection"]):
                if model_type != "stage2":
                    raise RuntimeError("--write-wrist-projection requires --model-type stage2/auto resolving to stage2")
                feature_dim = int(cfg.get("model", {}).get("feature_dim", 384))
                projection_head, projection_info = load_projection_head(
                    checkpoint_path=Path(options["projection_head_checkpoint"]),
                    feature_dim=feature_dim,
                    device=device,
                )
        linker = LinkerO6Retargeter(
            yaml_path=str(options["linker_yaml"]),
            urdf_dir=str(options["linker_urdf_dir"]),
            hand=str(options["hand"]),
        )
        wuji = None
        if not bool(options["skip_wuji"]):
            wuji = WujiRetargeter.from_yaml(str(options["wuji_yaml"]), hand_side=str(options["hand"]))

        aggregate: Counter[str] = Counter()
        log_every = int(options.get("log_every", 1))
        if str(options.get("pipeline_mode", "chunk")) == "chunk":
            results = process_pkl_chunk(
                pkl_paths=pkl_paths,
                model=model,
                transform=transform,
                model_type=model_type,
                device=device,
                precision=precision,
                linker=linker,
                wuji=wuji,
                projection_head=projection_head,
                projection_info=projection_info,
                options=options,
            )
        else:
            results = []
            for pkl_str in pkl_paths:
                results.append(
                    process_one_pkl(
                        pkl_path=Path(pkl_str),
                        model=model,
                        transform=transform,
                        model_type=model_type,
                        device=device,
                        precision=precision,
                        linker=linker,
                        wuji=wuji,
                        projection_head=projection_head,
                        projection_info=projection_info,
                        options=options,
                    )
                )

        for idx, result in enumerate(results, start=1):
            aggregate.update(result["stats"])
            should_log = log_every > 0 and (idx == 1 or idx == len(results) or idx % log_every == 0)
            if should_log:
                print(
                    f"[worker {rank} {device_name}] {idx}/{len(results)} "
                    f"{Path(result['pkl']).name} changed={result['changed']} "
                    f"predicted={result['stats'].get('predicted', 0)} "
                    f"wrote={result['stats'].get('wrote_success', 0)} "
                    f"errors={result['stats'].get('wrote_error', 0) + result['stats'].get('retarget_error', 0)} "
                    f"elapsed={result['elapsed_s']:.2f}s",
                    flush=True,
                )
        result_queue.put({"rank": rank, "device": device_name, "ok": True, "stats": dict(aggregate)})
    except Exception as exc:
        result_queue.put(
            {
                "rank": rank,
                "device": device_name,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise


def parse_devices(value: str) -> list[str]:
    raw = value.strip()
    if raw == "auto":
        if torch.cuda.is_available():
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
        return ["cpu"]
    devices = [part.strip() for part in raw.split(",") if part.strip()]
    if not devices:
        raise ValueError("--devices resolved to an empty list")
    if devices == ["cpu"]:
        return ["cpu"]
    return devices


def partition_paths(paths: list[Path], num_parts: int) -> list[list[str]]:
    chunks: list[list[str]] = [[] for _ in range(num_parts)]
    for idx, path in enumerate(paths):
        chunks[idx % num_parts].append(str(path))
    return chunks


def queue_results(result_queue: mp.Queue, expected: int) -> list[dict[str, Any]]:
    results = []
    deadline = time.monotonic() + 1.0
    while len(results) < expected:
        try:
            results.append(result_queue.get(timeout=1.0))
            deadline = time.monotonic() + 1.0
        except Empty:
            if time.monotonic() > deadline:
                break
    return results


def positive_int(value: str) -> int:
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append wrist-camera predictions and retargeted commands to DexUMI PKL episodes.",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-type", choices=("auto", "stage1", "stage2"), default="auto")
    parser.add_argument("--projection-head-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--write-wrist-projection",
        action="store_true",
        help="Also write wrist_uv21_rgb / wrist_anchor_uv_rgb using a trained projection head.",
    )
    parser.add_argument("--image-field", type=str, default="rgbImage")
    parser.add_argument("--devices", type=str, default="auto", help="auto, cpu, or comma-separated CUDA devices")
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument(
        "--pipeline-mode",
        choices=("chunk", "episode"),
        default="chunk",
        help=(
            "chunk builds one DataLoader over all frames assigned to each GPU worker; "
            "episode keeps the old per-PKL DataLoader behavior."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print one worker progress line every N PKLs. Use 0 to disable per-PKL logs.",
    )
    parser.add_argument("--limit-pkls", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-wrist", action="store_true")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--dino-dir", type=Path, default=None)
    parser.add_argument("--mano-model-dir", type=Path, default=None)
    parser.add_argument("--linker-yaml", type=Path, default=DEFAULT_LINKER_YAML)
    parser.add_argument("--linker-urdf-dir", type=Path, default=DEFAULT_LINKER_URDF_DIR)
    parser.add_argument("--wuji-yaml", type=Path, default=DEFAULT_WUJI_YAML)
    parser.add_argument("--skip-wuji", action="store_true", help="Do not initialize or write wrist_wuji_command.")
    parser.add_argument(
        "--mirror-standard-fields",
        action="store_true",
        help=(
            "Also write wrist prediction commands to standard fields "
            "o6_command and wuji_command. Use this when downstream code expects "
            "the normal collection fields instead of wrist_* fields."
        ),
    )
    parser.add_argument(
        "--retarget-existing-wrist-only",
        action="store_true",
        help=(
            "Do not load the wrist model or run image inference. Reuse existing "
            "wrist_pts21_mano in each PKL and only regenerate retargeted command fields."
        ),
    )
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.set_defaults(channels_last=True)
    return parser


def options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    options = vars(args).copy()
    for key in (
        "checkpoint",
        "projection_head_checkpoint",
        "config",
        "data_root",
        "dino_dir",
        "mano_model_dir",
        "linker_yaml",
        "linker_urdf_dir",
        "wuji_yaml",
    ):
        value = options.get(key)
        options[key] = None if value is None else str(Path(value).expanduser().resolve())
    if options["projection_head_checkpoint"] is not None:
        options["write_wrist_projection"] = True
    if options["limit_frames"] is not None and int(options["limit_frames"]) <= 0:
        raise ValueError("--limit-frames must be positive")
    if options["limit_pkls"] is not None and int(options["limit_pkls"]) <= 0:
        raise ValueError("--limit-pkls must be positive")
    options["num_workers"] = max(0, int(options["num_workers"]))
    options["log_every"] = max(0, int(options["log_every"]))
    return options


def validate_inputs(options: dict[str, Any]) -> None:
    required_files = ["linker_yaml"]
    if not bool(options["retarget_existing_wrist_only"]):
        required_files.append("checkpoint")
    if bool(options["write_wrist_projection"]):
        if bool(options["retarget_existing_wrist_only"]):
            raise ValueError("--write-wrist-projection cannot be combined with --retarget-existing-wrist-only")
        if options["projection_head_checkpoint"] is None:
            raise ValueError("--write-wrist-projection requires --projection-head-checkpoint")
        required_files.append("projection_head_checkpoint")
    if not bool(options["skip_wuji"]):
        required_files.append("wuji_yaml")
    for key in required_files:
        path = Path(options[key])
        if not path.is_file():
            raise FileNotFoundError(f"{key} not found: {path}")
    if (
        not bool(options["retarget_existing_wrist_only"])
        and options["config"] is not None
        and not Path(options["config"]).is_file()
    ):
        raise FileNotFoundError(f"config not found: {options['config']}")
    if (
        not bool(options["retarget_existing_wrist_only"])
        and options["dino_dir"] is not None
        and not Path(options["dino_dir"]).is_dir()
    ):
        raise FileNotFoundError(f"dino_dir not found: {options['dino_dir']}")
    if (
        not bool(options["retarget_existing_wrist_only"])
        and options["mano_model_dir"] is not None
        and not Path(options["mano_model_dir"]).is_dir()
    ):
        raise FileNotFoundError(f"mano_model_dir not found: {options['mano_model_dir']}")
    if not Path(options["linker_urdf_dir"]).is_dir():
        raise FileNotFoundError(f"linker_urdf_dir not found: {options['linker_urdf_dir']}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    options = options_from_args(args)
    data_root = Path(options["data_root"])
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    validate_inputs(options)

    pkl_paths = iter_pkl_paths(data_root)
    if options["limit_pkls"] is not None:
        pkl_paths = pkl_paths[: int(options["limit_pkls"])]
    if not pkl_paths:
        print(f"No PKL files found under {data_root}")
        return 0

    devices = parse_devices(str(args.devices))
    chunks = partition_paths(pkl_paths, len(devices))
    active = [(device, chunk) for device, chunk in zip(devices, chunks) if chunk]
    print(
        json.dumps(
            {
                "data_root": str(data_root),
                "pkl_files": len(pkl_paths),
                "devices": [device for device, _ in active],
                "dry_run": bool(options["dry_run"]),
                "overwrite_wrist": bool(options["overwrite_wrist"]),
                "write_wrist_projection": bool(options["write_wrist_projection"]),
                "projection_head_checkpoint": options.get("projection_head_checkpoint"),
                "batch_size": int(options["batch_size"]),
                "num_workers_per_device": int(options["num_workers"]),
                "pipeline_mode": str(options["pipeline_mode"]),
                "log_every": int(options["log_every"]),
            },
            indent=2,
        ),
        flush=True,
    )

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    processes = []
    start = time.perf_counter()
    for rank, (device, chunk) in enumerate(active):
        process = ctx.Process(
            target=worker_main,
            args=(rank, device, chunk, options, result_queue),
            daemon=False,
        )
        process.start()
        processes.append(process)

    results = []
    while any(process.is_alive() for process in processes):
        try:
            results.append(result_queue.get(timeout=1.0))
        except Empty:
            pass
    for process in processes:
        process.join()
    while True:
        try:
            results.append(result_queue.get_nowait())
        except Empty:
            break

    total: Counter[str] = Counter()
    failed = False
    for result in results:
        if result.get("ok"):
            total.update(result.get("stats", {}))
        else:
            failed = True
            print(f"[ERROR] worker {result.get('rank')} {result.get('device')}: {result.get('error')}")
    for process in processes:
        if process.exitcode != 0:
            failed = True
            print(f"[ERROR] worker process pid={process.pid} exitcode={process.exitcode}")

    summary = {
        "elapsed_s": time.perf_counter() - start,
        "workers": len(active),
        "pkl_files": len(pkl_paths),
        "stats": dict(total),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
