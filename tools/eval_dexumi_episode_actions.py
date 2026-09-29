#!/usr/bin/env python3
"""Offline action evaluation for one raw DexUMI PKL episode.

The script reads a DexUMI collection directory such as
/home/zjc/Desktop/human2dex/data/pick_2, selects one PKL episode, runs a trained
policy checkpoint on sliding observation windows, compares predicted actions
against GT actions reconstructed from the PKL, and writes metrics plus plots.

Example:
  conda run -n umi204 python tools/eval_dexumi_episode_actions.py \
      --ckpt /home/zjc/Desktop/human2dex/ckpt/linker_o6/pick_7_fused/latest.ckpt \
      --data-dir /home/zjc/Desktop/human2dex/data/pick_7_fused \
      --episode episode_0013
      --output-dir /home/zjc/Desktop/human2dex/data

If diffusion_policy is not installed as a package, set DEX_POLICY_REPO to the
training/inference checkout that contains diffusion_policy/ and umi/. The
script also uses this unified human2dex checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _add_policy_repo_to_path() -> None:
    """Add the training checkout that owns diffusion_policy/ and umi/."""

    candidates: list[Path] = []
    env_repo = os.environ.get("DEX_POLICY_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser())
    candidates.extend(
        [
            ROOT_DIR.parent / "universal_manipulation_interface",
        ]
    )

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "diffusion_policy").is_dir() and (candidate / "umi").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


_add_policy_repo_to_path()

try:
    import cv2
    import dill
    import hydra
    import torch
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from omegaconf import OmegaConf

    from diffusion_policy.common.cv2_util import get_image_transform
    from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    from umi.common.pose_util import (
        mat_to_pose,
        mat_to_pose10d,
        pose10d_to_mat,
        pose_to_mat,
    )
except ImportError as exc:  # pragma: no cover - exercised only in wrong envs.
    raise SystemExit(
        "Missing runtime dependency. Run this script in the training/inference "
        "environment that has torch, hydra, cv2, matplotlib and this repo on "
        f"PYTHONPATH. Original error: {exc}"
    ) from exc


OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_CKPT = Path("ckpt/linker_o6/pick_sponge/latest.ckpt")
DEFAULT_DATA_DIR = Path("/home/zjc/Desktop/human2dex/data/pick_7_fused")

FIELD_ALIASES = {
    "camera0_rgb": ("camera0_rgb", "camera_0", "rgb", "rgbImage"),
    "camera_0": ("camera_0", "camera0_rgb", "rgb", "rgbImage"),
    "robot0_eef_pos": ("robot0_eef_pos", "trajectoryPose", "trajectory_pose", "robot_eef_pose"),
    "robot0_eef_rot_axis_angle": (
        "robot0_eef_rot_axis_angle",
        "trajectoryPose",
        "trajectory_pose",
        "robot_eef_pose",
    ),
    "robot0_eef_pos_wrt_start": ("robot0_eef_pos_wrt_start",),
    "robot0_eef_rot_axis_angle_wrt_start": ("robot0_eef_rot_axis_angle_wrt_start",),
    "robot0_gripper_width": ("robot0_gripper_width", "hand_command", "handCommand", "o6_command", "wuji_command"),
    "robot_eef_pose": ("robot_eef_pose", "trajectoryPose", "trajectory_pose"),
    "trajectoryPose": ("trajectoryPose", "trajectory_pose", "robot_eef_pose"),
    "trajectory_pose": ("trajectory_pose", "trajectoryPose", "robot_eef_pose"),
    "hand_command": ("hand_command", "handCommand", "o6_command"),
    "handCommand": ("handCommand", "hand_command", "o6_command"),
    "pts21_mano": ("pts21_mano",),
    "pts21_mano_flat": ("pts21_mano_flat", "pts21_mano"),
    "timestamp": ("timestamp",),
}


@dataclass
class LoadedPolicy:
    cfg: Any
    policy: Any
    device: torch.device
    ckpt_path: Path
    action_dim: int
    obs_horizon: int
    pose_repr: str
    action_down_sample_steps: int


@dataclass
class RawEpisode:
    name: str
    pkl_path: Path
    timestamp: np.ndarray
    rgb: np.ndarray
    pose6: np.ndarray
    pose_mat: np.ndarray
    o6_command: np.ndarray | None
    wuji_command: np.ndarray | None
    pts21_mano: np.ndarray | None

    def __len__(self) -> int:
        return int(self.pose6.shape[0])


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x runtimes to unpickle arrays written by NumPy 2.x."""

    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def _as_list(value: Any) -> list[Any]:
    return list(OmegaConf.to_container(value, resolve=True))


def _shape(attr: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in _as_list(attr["shape"]))


def _has_key(cfg_node: Any, key: str) -> bool:
    try:
        return key in cfg_node
    except TypeError:
        return hasattr(cfg_node, key)


def _obs_horizon_from_cfg(cfg: Any) -> int:
    if _has_key(cfg, "n_obs_steps"):
        return int(cfg.n_obs_steps)
    horizons = []
    for attr in cfg.task.shape_meta.obs.values():
        if _has_key(attr, "horizon"):
            horizons.append(int(attr.horizon))
    return max(horizons) if horizons else 1


def _action_down_sample_steps_from_cfg(cfg: Any) -> int:
    action_meta = cfg.task.shape_meta.action
    if _has_key(action_meta, "down_sample_steps"):
        value = action_meta.down_sample_steps
    elif _has_key(cfg.task, "action_down_sample_steps"):
        value = cfg.task.action_down_sample_steps
    else:
        value = 1
    if isinstance(value, int):
        return max(1, int(value))
    try:
        values = [int(x) for x in _as_list(value)]
        return max(1, values[0] if values else 1)
    except Exception:
        return 1


def _pose_repr_from_cfg(cfg: Any) -> str:
    if _has_key(cfg.task, "pose_repr") and _has_key(cfg.task.pose_repr, "obs_pose_repr"):
        return str(cfg.task.pose_repr.obs_pose_repr)
    return "rel"


def _disable_pretrained_download(cfg: Any) -> None:
    if _has_key(cfg, "policy") and _has_key(cfg.policy, "obs_encoder") and _has_key(cfg.policy.obs_encoder, "pretrained"):
        cfg.policy.obs_encoder.pretrained = False


def _resolve_device(device_arg: str, cfg: Any) -> torch.device:
    if device_arg == "auto":
        if _has_key(cfg, "training") and _has_key(cfg.training, "device"):
            requested = str(cfg.training.device)
        else:
            requested = "cuda"
    else:
        requested = device_arg
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA requested ({requested}) but unavailable; using CPU.")
        requested = "cpu"
    return torch.device(requested)


def _load_policy(ckpt_path: Path, device_arg: str, num_inference_steps: int | None) -> LoadedPolicy:
    ckpt = ckpt_path.expanduser()
    if ckpt.is_dir():
        ckpt = ckpt / "checkpoints" / "latest.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    _install_numpy_pickle_compat()
    payload = torch.load(ckpt.open("rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    _disable_pretrained_download(cfg)

    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    use_ema = bool(cfg.training.use_ema) if _has_key(cfg, "training") and _has_key(cfg.training, "use_ema") else False
    if use_ema and hasattr(workspace, "ema_model"):
        policy = workspace.ema_model
    elif hasattr(workspace, "model"):
        policy = workspace.model
    elif hasattr(workspace, "policy"):
        policy = workspace.policy
    else:
        raise RuntimeError("workspace has no ema_model, model, or policy attribute")

    if num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(num_inference_steps)

    device = _resolve_device(device_arg, cfg)
    policy.eval().to(device)

    action_shape = _shape(cfg.task.shape_meta.action)
    if len(action_shape) != 1:
        raise ValueError(f"expected flat action shape, got {action_shape}")
    action_dim = int(action_shape[0])
    if action_dim not in (12, 15, 26, 29):
        raise ValueError(
            f"unsupported checkpoint action_dim={action_dim}; supported DexUMI "
            "offline comparisons are 12/15 for Linker O6 and 26/29 for Wuji."
        )

    loaded = LoadedPolicy(
        cfg=cfg,
        policy=policy,
        device=device,
        ckpt_path=ckpt,
        action_dim=action_dim,
        obs_horizon=_obs_horizon_from_cfg(cfg),
        pose_repr=_pose_repr_from_cfg(cfg),
        action_down_sample_steps=_action_down_sample_steps_from_cfg(cfg),
    )

    print("checkpoint:", ckpt)
    print("workspace:", cfg._target_)
    print("policy:", cfg.policy._target_)
    print("obs_keys:", list(cfg.task.shape_meta.obs.keys()))
    print("action_dim:", loaded.action_dim)
    print("obs_horizon:", loaded.obs_horizon)
    print("pose_repr:", loaded.pose_repr)
    print("action_down_sample_steps:", loaded.action_down_sample_steps)
    if hasattr(policy, "num_inference_steps"):
        print("num_inference_steps:", policy.num_inference_steps)
    return loaded


def _episode_sort_key(path: Path) -> tuple[str, int, str]:
    parent = path.parent.name
    prefix = parent
    ep_idx = -1
    if "_ep" in parent:
        prefix, ep_text = parent.rsplit("_ep", 1)
        try:
            ep_idx = int(ep_text)
        except ValueError:
            ep_idx = -1
    return prefix, ep_idx, str(path)


def _find_pkl_paths(data_dir: Path) -> list[Path]:
    root = data_dir.expanduser()
    if root.is_file() and root.suffix == ".pkl":
        return [root]
    paths = sorted(root.glob("demo_*/*.pkl"), key=_episode_sort_key)
    if not paths:
        paths = sorted(root.rglob("*.pkl"), key=_episode_sort_key)
    return paths


def _select_pkl(data_dir: Path, episode: str | None) -> Path:
    root = data_dir.expanduser()
    if episode:
        ep_path = Path(episode).expanduser()
        if ep_path.is_file():
            return ep_path
        if ep_path.is_dir():
            paths = sorted(ep_path.glob("*.pkl"))
            if not paths:
                raise FileNotFoundError(f"no pkl found under episode dir: {ep_path}")
            return paths[0]

    paths = _find_pkl_paths(root)
    if not paths:
        raise FileNotFoundError(f"no pkl episodes found under: {root}")

    if episode is None:
        return paths[0]

    try:
        idx = int(episode)
    except ValueError:
        idx = None
    if idx is not None:
        if idx < 0 or idx >= len(paths):
            raise IndexError(f"episode index {idx} out of range [0, {len(paths)})")
        return paths[idx]

    matches = [
        path
        for path in paths
        if path.parent.name == episode or path.stem == episode or str(path.parent).endswith(episode)
    ]
    if not matches:
        raise FileNotFoundError(f"episode '{episode}' not found under {root}")
    return matches[0]


def _valid_array(value: Any, expected_shape: tuple[int, ...] | None, dtype: np.dtype) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if expected_shape is not None and arr.shape != expected_shape:
        return None
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        return None
    return arr.astype(dtype, copy=False)


def _pose7_xyzw_to_pose6(pose7: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    arr = np.asarray(pose7, dtype=np.float32).reshape(7)
    rotvec = R.from_quat(arr[3:7]).as_rotvec().astype(np.float32)
    return np.concatenate([arr[:3], rotvec], axis=0).astype(np.float32)


def _coerce_pose6(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape == (6,):
        pose = arr.astype(np.float32, copy=False)
    elif arr.shape == (7,):
        pose = _pose7_xyzw_to_pose6(arr)
    elif arr.shape == (4, 4):
        pose = mat_to_pose(arr.astype(np.float32, copy=False)).astype(np.float32)
    else:
        return None
    if not np.all(np.isfinite(pose)):
        return None
    return pose.astype(np.float32, copy=False)


def _read_rgb_from_msg(msg: dict[str, Any], pkl_dir: Path, skip_missing_images: bool) -> np.ndarray | None:
    rgb_value = msg.get("rgbImage")
    if rgb_value is None:
        return None
    if isinstance(rgb_value, np.ndarray):
        arr = np.asarray(rgb_value)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            return arr.astype(np.uint8, copy=False)
        return None

    rgb_path = pkl_dir / str(rgb_value)
    image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        if skip_missing_images:
            return None
        raise FileNotFoundError(f"failed to read image: {rgb_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _extract_o6_command(msg: dict[str, Any]) -> np.ndarray | None:
    return _valid_array(msg.get("o6_command", msg.get("handCommand")), (6,), np.float32)


def _extract_wuji_command(msg: dict[str, Any]) -> np.ndarray | None:
    value = msg.get("wuji_command")
    arr = _valid_array(value, (5, 4), np.float32)
    if arr is not None:
        return arr.reshape(-1)
    return _valid_array(value, (20,), np.float32)


def _load_pkl_messages(pkl_path: Path) -> list[dict[str, Any]]:
    _install_numpy_pickle_compat()
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict/messages: {pkl_path}")
    return [msg for msg in data["messages"] if isinstance(msg, dict)]


def _load_raw_episode(
        pkl_path: Path,
        action_dim: int,
        skip_missing_images: bool) -> RawEpisode:
    messages = _load_pkl_messages(pkl_path)
    need_o6 = action_dim in (12, 15)
    need_wuji = action_dim in (26, 29)

    timestamps: list[float] = []
    rgbs: list[np.ndarray] = []
    poses6: list[np.ndarray] = []
    poses_mat: list[np.ndarray] = []
    o6_commands: list[np.ndarray] = []
    wuji_commands: list[np.ndarray] = []
    pts21_values: list[np.ndarray] = []

    for msg in messages:
        pose6 = _coerce_pose6(msg.get("trajectoryPose"))
        rgb = _read_rgb_from_msg(msg, pkl_path.parent, skip_missing_images=skip_missing_images)
        o6 = _extract_o6_command(msg)
        wuji = _extract_wuji_command(msg)
        pts21 = _valid_array(msg.get("pts21_mano"), (21, 3), np.float32)

        if pose6 is None or rgb is None:
            continue
        if need_o6 and o6 is None:
            continue
        if need_wuji and wuji is None:
            continue

        timestamps.append(float(msg.get("timestamp", len(timestamps))))
        rgbs.append(rgb)
        poses6.append(pose6)
        poses_mat.append(pose_to_mat(pose6).astype(np.float32))
        if o6 is not None:
            o6_commands.append(o6.astype(np.float32, copy=False))
        if wuji is not None:
            wuji_commands.append(wuji.astype(np.float32, copy=False))
        if pts21 is not None:
            pts21_values.append(pts21.astype(np.float32, copy=False))
        else:
            pts21_values.append(np.zeros((21, 3), dtype=np.float32))

    if not poses6:
        raise ValueError(f"no valid frames in {pkl_path}")

    o6_arr = np.stack(o6_commands, axis=0) if len(o6_commands) == len(poses6) else None
    wuji_arr = np.stack(wuji_commands, axis=0) if len(wuji_commands) == len(poses6) else None
    pts21_arr = np.stack(pts21_values, axis=0) if len(pts21_values) == len(poses6) else None

    return RawEpisode(
        name=pkl_path.parent.name,
        pkl_path=pkl_path,
        timestamp=np.asarray(timestamps, dtype=np.float64),
        rgb=np.stack(rgbs, axis=0).astype(np.uint8),
        pose6=np.stack(poses6, axis=0).astype(np.float32),
        pose_mat=np.stack(poses_mat, axis=0).astype(np.float32),
        o6_command=o6_arr,
        wuji_command=wuji_arr,
        pts21_mano=pts21_arr,
    )


def _backend_from_action_dim(action_dim: int) -> str:
    if action_dim in (12, 15):
        return "o6"
    if action_dim in (26, 29):
        return "wuji"
    raise ValueError(f"unsupported action_dim={action_dim}")


def _key_horizon(attr: Any, default_horizon: int) -> int:
    if _has_key(attr, "horizon"):
        return int(attr.horizon)
    return int(default_horizon)


def _key_down_sample(attr: Any) -> int | list[int]:
    if not _has_key(attr, "down_sample_steps"):
        return 1
    value = attr.down_sample_steps
    if isinstance(value, int):
        return max(1, int(value))
    try:
        values = [int(x) for x in _as_list(value)]
        return values if values else 1
    except Exception:
        return 1


def _key_latency(attr: Any) -> int:
    if not _has_key(attr, "latency_steps"):
        return 0
    try:
        return int(round(float(attr.latency_steps)))
    except Exception:
        return 0


def _history_indices(current_idx: int, n_frames: int, attr: Any, default_horizon: int, is_rgb: bool) -> np.ndarray:
    horizon = _key_horizon(attr, default_horizon)
    down_sample = _key_down_sample(attr)
    latency = 0 if is_rgb else _key_latency(attr)

    if isinstance(down_sample, list):
        raw = [current_idx + latency]
        for i in range(max(0, horizon - 1)):
            step = down_sample[min(i, len(down_sample) - 1)]
            raw.append(current_idx - step + latency)
        idxs = np.asarray(raw[:horizon], dtype=np.int64)[::-1]
    else:
        idxs = np.asarray(
            [current_idx - i * int(down_sample) + latency for i in range(horizon)],
            dtype=np.int64,
        )[::-1]
    return np.clip(idxs, 0, n_frames - 1)


def _pose10d_sequence(
        episode: RawEpisode,
        idxs: np.ndarray,
        pose_repr: str,
        base_mode: str = "last") -> np.ndarray:
    pose_mats = episode.pose_mat[idxs]
    if base_mode == "start":
        base_pose_mat = episode.pose_mat[0]
        rep = "relative" if pose_repr == "rel" else pose_repr
    else:
        base_pose_mat = pose_mats[-1]
        rep = pose_repr
    converted = convert_pose_mat_rep(
        pose_mats,
        base_pose_mat=base_pose_mat,
        pose_rep=rep,
        backward=False,
    )
    return mat_to_pose10d(converted).astype(np.float32)


def _format_rgb_sequence(frames: np.ndarray, attr: Any) -> np.ndarray:
    c_out, h_out, w_out = _shape(attr)
    if c_out != 3:
        raise ValueError(f"only RGB image obs are supported, got shape={_shape(attr)}")
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB sequence, got {arr.shape}")
    _, h_in, w_in, _ = arr.shape
    if h_in != h_out or w_in != w_out:
        transform = get_image_transform(
            input_res=(w_in, h_in),
            output_res=(w_out, h_out),
            bgr_to_rgb=False,
        )
        arr = np.stack([transform(frame) for frame in arr], axis=0)
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return np.moveaxis(arr, -1, 1).astype(np.float32)


def _hand_sequence_for_shape(episode: RawEpisode, idxs: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 1:
        raise ValueError(f"hand obs expects a flat shape, got {shape}")
    dim = shape[0]
    if dim == 6:
        if episode.o6_command is None:
            raise ValueError("checkpoint requires 6D O6 hand obs, but episode has no handCommand/o6_command")
        return episode.o6_command[idxs].astype(np.float32)
    if dim == 20:
        if episode.wuji_command is None:
            raise ValueError("checkpoint requires 20D Wuji hand obs, but episode has no wuji_command")
        return episode.wuji_command[idxs].astype(np.float32)
    if dim == 1:
        if episode.o6_command is None:
            return np.zeros((len(idxs), 1), dtype=np.float32)
        return episode.o6_command[idxs, :1].astype(np.float32)
    raise ValueError(f"unsupported robot0_gripper_width shape {shape}; expected [6] or [20]")


def _obs_value_for_key(episode: RawEpisode, key: str, attr: Any, idxs: np.ndarray, loaded: LoadedPolicy) -> np.ndarray:
    obs_type = attr.get("type", "low_dim")
    shape = _shape(attr)

    if obs_type == "rgb":
        return _format_rgb_sequence(episode.rgb[idxs], attr)

    if key == "robot0_eef_pos":
        return _pose10d_sequence(episode, idxs, loaded.pose_repr)[:, :3]
    if key == "robot0_eef_rot_axis_angle":
        return _pose10d_sequence(episode, idxs, loaded.pose_repr)[:, 3:]
    if key == "robot0_eef_pos_wrt_start":
        return _pose10d_sequence(episode, idxs, loaded.pose_repr, base_mode="start")[:, :3]
    if key == "robot0_eef_rot_axis_angle_wrt_start":
        return _pose10d_sequence(episode, idxs, loaded.pose_repr, base_mode="start")[:, 3:]
    if key == "robot0_gripper_width":
        return _hand_sequence_for_shape(episode, idxs, shape)

    if key in ("robot_eef_pose", "trajectoryPose", "trajectory_pose"):
        if shape == (6,):
            return episode.pose6[idxs].astype(np.float32)
        if shape == (9,):
            return _pose10d_sequence(episode, idxs, loaded.pose_repr).astype(np.float32)
        raise ValueError(f"{key} shape {shape} is not supported")

    if key in ("hand_command", "handCommand"):
        return _hand_sequence_for_shape(episode, idxs, shape)

    if key == "pts21_mano":
        if episode.pts21_mano is None:
            return np.zeros((len(idxs),) + shape, dtype=np.float32)
        return episode.pts21_mano[idxs].astype(np.float32)
    if key == "pts21_mano_flat":
        if episode.pts21_mano is None:
            return np.zeros((len(idxs),) + shape, dtype=np.float32)
        return episode.pts21_mano[idxs].reshape(len(idxs), -1).astype(np.float32)
    if key == "timestamp":
        values = episode.timestamp[idxs].astype(np.float32)
        return values.reshape((len(idxs),) + shape)

    # Ignore-by-policy keys may still be required by the normalizer. Fill a
    # correctly shaped zero sequence instead of dropping the key.
    return np.zeros((len(idxs),) + shape, dtype=np.float32)


def _make_obs_dict(episode: RawEpisode, loaded: LoadedPolicy, current_idx: int) -> dict[str, np.ndarray]:
    obs_dict: dict[str, np.ndarray] = {}
    for key, attr in loaded.cfg.task.shape_meta.obs.items():
        key = str(key)
        is_rgb = attr.get("type", "low_dim") == "rgb"
        idxs = _history_indices(
            current_idx=current_idx,
            n_frames=len(episode),
            attr=attr,
            default_horizon=loaded.obs_horizon,
            is_rgb=is_rgb,
        )
        value = _obs_value_for_key(episode, key, attr, idxs, loaded)
        shape = _shape(attr)
        if is_rgb:
            expected_tail = (shape[0], shape[1], shape[2])
        else:
            expected_tail = shape
        if value.shape[1:] != expected_tail:
            raise ValueError(f"obs {key} expected T{expected_tail}, got {value.shape}")
        obs_dict[key] = value.astype(np.float32, copy=False)
    return obs_dict


def _predict_action(loaded: LoadedPolicy, obs_np: dict[str, np.ndarray]) -> np.ndarray:
    obs_torch = dict_apply(
        obs_np,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(loaded.device),
    )
    with torch.no_grad():
        result = loaded.policy.predict_action(obs_torch)
    if "action_pred" in result:
        action = result["action_pred"]
    elif "action" in result:
        action = result["action"]
    else:
        raise KeyError(f"policy result has no action/action_pred key: {list(result.keys())}")
    return action[0].detach().to("cpu").numpy().astype(np.float32)


def _action_base_pose_mat(episode: RawEpisode, loaded: LoadedPolicy, current_idx: int) -> np.ndarray:
    obs_meta = loaded.cfg.task.shape_meta.obs
    if "robot0_eef_pos" in obs_meta and "robot0_eef_rot_axis_angle" in obs_meta:
        pos_idxs = _history_indices(
            current_idx=current_idx,
            n_frames=len(episode),
            attr=obs_meta["robot0_eef_pos"],
            default_horizon=loaded.obs_horizon,
            is_rgb=False,
        )
        rot_idxs = _history_indices(
            current_idx=current_idx,
            n_frames=len(episode),
            attr=obs_meta["robot0_eef_rot_axis_angle"],
            default_horizon=loaded.obs_horizon,
            is_rgb=False,
        )
        pose6 = np.concatenate([
            episode.pose6[pos_idxs[-1], :3],
            episode.pose6[rot_idxs[-1], 3:6],
        ])
        return pose_to_mat(pose6).astype(np.float32)
    return episode.pose_mat[current_idx]


def _gt_action(episode: RawEpisode, loaded: LoadedPolicy, current_idx: int, target_idx: int) -> np.ndarray:
    if loaded.action_dim in (12, 26):
        pose_part = episode.pose6[target_idx]
    else:
        base_pose_mat = _action_base_pose_mat(episode, loaded, current_idx)
        converted = convert_pose_mat_rep(
            episode.pose_mat[target_idx:target_idx + 1],
            base_pose_mat=base_pose_mat,
            pose_rep=loaded.pose_repr,
            backward=False,
        )
        pose_part = mat_to_pose10d(converted)[0].astype(np.float32)

    if loaded.action_dim in (12, 15):
        if episode.o6_command is None:
            raise ValueError("O6 action comparison requires handCommand/o6_command")
        hand_part = episode.o6_command[target_idx]
    else:
        if episode.wuji_command is None:
            raise ValueError("Wuji action comparison requires wuji_command")
        hand_part = episode.wuji_command[target_idx]
    return np.concatenate([pose_part.astype(np.float32), hand_part.astype(np.float32)], axis=0)


def _absolute_pos_from_action(
        action: np.ndarray,
        episode: RawEpisode,
        loaded: LoadedPolicy,
        current_idx: int) -> np.ndarray:
    if loaded.action_dim in (12, 26):
        return np.asarray(action[:3], dtype=np.float32)
    rel_mat = pose10d_to_mat(np.asarray(action[:9], dtype=np.float32)[None])[0]
    base_pose_mat = _action_base_pose_mat(episode, loaded, current_idx)
    abs_mat = convert_pose_mat_rep(
        rel_mat[None],
        base_pose_mat=base_pose_mat,
        pose_rep=loaded.pose_repr,
        backward=True,
    )[0]
    return abs_mat[:3, 3].astype(np.float32)


def _evaluate_episode(
        episode: RawEpisode,
        loaded: LoadedPolicy,
        horizon_step: int,
        start_index: int,
        max_steps: int | None) -> dict[str, np.ndarray | list[float]]:
    backend = _backend_from_action_dim(loaded.action_dim)
    print(f"episode: {episode.name}")
    print(f"pkl: {episode.pkl_path}")
    print(f"frames: {len(episode)}")
    print(f"hand_backend: {backend}")

    pred_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    pred_abs_pos: list[np.ndarray] = []
    gt_abs_pos: list[np.ndarray] = []
    current_indices: list[int] = []
    target_indices: list[int] = []
    timestamps: list[float] = []
    latencies: list[float] = []

    start = max(0, int(start_index))
    stop = len(episode)
    if max_steps is not None:
        stop = min(stop, start + int(max_steps))

    for current_idx in range(start, stop):
        obs_np = _make_obs_dict(episode, loaded, current_idx)
        t0 = time.time()
        pred_seq = _predict_action(loaded, obs_np)
        latencies.append(time.time() - t0)

        if pred_seq.ndim != 2 or pred_seq.shape[-1] != loaded.action_dim:
            raise ValueError(f"expected policy action shape Hx{loaded.action_dim}, got {pred_seq.shape}")
        if horizon_step < 0 or horizon_step >= pred_seq.shape[0]:
            raise IndexError(f"horizon_step={horizon_step} out of range for policy output {pred_seq.shape}")

        target_idx = current_idx + int(horizon_step) * loaded.action_down_sample_steps
        if target_idx >= len(episode):
            break

        pred = pred_seq[horizon_step].astype(np.float32)
        gt = _gt_action(episode, loaded, current_idx=current_idx, target_idx=target_idx)
        if gt.shape[-1] != loaded.action_dim:
            raise ValueError(f"GT action dim mismatch: got {gt.shape[-1]}, expected {loaded.action_dim}")

        pred_actions.append(pred)
        gt_actions.append(gt)
        pred_abs_pos.append(_absolute_pos_from_action(pred, episode, loaded, current_idx))
        gt_abs_pos.append(episode.pose6[target_idx, :3].astype(np.float32))
        current_indices.append(current_idx)
        target_indices.append(target_idx)
        timestamps.append(float(episode.timestamp[target_idx]))

    if not pred_actions:
        raise RuntimeError("no predictions were produced; check start/max/horizon settings")

    return {
        "pred_action": np.stack(pred_actions, axis=0),
        "gt_action": np.stack(gt_actions, axis=0),
        "pred_abs_pos": np.stack(pred_abs_pos, axis=0),
        "gt_abs_pos": np.stack(gt_abs_pos, axis=0),
        "current_indices": np.asarray(current_indices, dtype=np.int64),
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "latencies": np.asarray(latencies[: len(pred_actions)], dtype=np.float64),
    }


def _metrics_for_array(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _norm_stats(diff: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(diff, dtype=np.float64), axis=-1)
    return {
        "mean": float(np.mean(norms)),
        "rmse": float(np.sqrt(np.mean(norms ** 2))),
        "max": float(np.max(norms)),
    }


def _compute_metrics(results: dict[str, np.ndarray | list[float]], action_dim: int) -> dict[str, Any]:
    pred = np.asarray(results["pred_action"])
    gt = np.asarray(results["gt_action"])
    pred_abs_pos = np.asarray(results["pred_abs_pos"])
    gt_abs_pos = np.asarray(results["gt_abs_pos"])

    if action_dim in (15, 29):
        rot_slice = slice(3, 9)
        hand_slice = slice(9, action_dim)
        rotation_name = "rot6d"
    else:
        rot_slice = slice(3, 6)
        hand_slice = slice(6, action_dim)
        rotation_name = "rotvec"

    metrics = {
        "overall": _metrics_for_array(pred, gt),
        "arm_position_action_space": _metrics_for_array(pred[:, :3], gt[:, :3]),
        f"arm_rotation_{rotation_name}": _metrics_for_array(pred[:, rot_slice], gt[:, rot_slice]),
        "hand": _metrics_for_array(pred[:, hand_slice], gt[:, hand_slice]),
        "arm_abs_position_xyz": _metrics_for_array(pred_abs_pos, gt_abs_pos),
        "norm_errors": {
            "arm_position_action_space": _norm_stats(pred[:, :3] - gt[:, :3]),
            f"arm_rotation_{rotation_name}": _norm_stats(pred[:, rot_slice] - gt[:, rot_slice]),
            "hand": _norm_stats(pred[:, hand_slice] - gt[:, hand_slice]),
            "arm_abs_position_xyz": _norm_stats(pred_abs_pos - gt_abs_pos),
        },
    }
    return metrics


def _subplot_grid(n_dims: int) -> tuple[int, int]:
    if n_dims <= 3:
        return n_dims, 1
    cols = 3 if n_dims <= 9 else 4
    rows = int(np.ceil(n_dims / cols))
    return rows, cols


def _plot_dim_series(
        x: np.ndarray,
        pred: np.ndarray,
        gt: np.ndarray,
        labels: list[str],
        title: str,
        ylabel: str,
        output_path: Path) -> None:
    n_dims = pred.shape[1]
    rows, cols = _subplot_grid(n_dims)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.4 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n_dims:
            ax.axis("off")
            continue
        ax.plot(x, gt[:, i], label="GT", linewidth=1.4)
        ax.plot(x, pred[:, i], label="Pred", linewidth=1.2)
        ax.set_title(labels[i] if i < len(labels) else f"dim_{i}", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_errors(
        x: np.ndarray,
        pred: np.ndarray,
        gt: np.ndarray,
        pred_abs_pos: np.ndarray,
        gt_abs_pos: np.ndarray,
        action_dim: int,
        output_path: Path) -> None:
    if action_dim in (15, 29):
        rot_slice = slice(3, 9)
        hand_slice = slice(9, action_dim)
        rot_label = "rotation rot6d norm"
    else:
        rot_slice = slice(3, 6)
        hand_slice = slice(6, action_dim)
        rot_label = "rotation rotvec norm"

    series = [
        ("arm position action-space norm", np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1)),
        (rot_label, np.linalg.norm(pred[:, rot_slice] - gt[:, rot_slice], axis=-1)),
        ("hand norm", np.linalg.norm(pred[:, hand_slice] - gt[:, hand_slice], axis=-1)),
        ("absolute xyz norm", np.linalg.norm(pred_abs_pos - gt_abs_pos, axis=-1)),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 8), sharex=True)
    for ax, (label, values) in zip(axes, series):
        ax.plot(x, values, linewidth=1.4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("step")
    fig.suptitle("Action Error Norms")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_trajectory_3d(pred_abs_pos: np.ndarray, gt_abs_pos: np.ndarray, output_path: Path) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt_abs_pos[:, 0], gt_abs_pos[:, 1], gt_abs_pos[:, 2], label="GT", linewidth=1.8)
    ax.plot(pred_abs_pos[:, 0], pred_abs_pos[:, 1], pred_abs_pos[:, 2], label="Pred", linewidth=1.5)
    ax.scatter(gt_abs_pos[0, 0], gt_abs_pos[0, 1], gt_abs_pos[0, 2], marker="o", s=30, label="GT start")
    ax.scatter(gt_abs_pos[-1, 0], gt_abs_pos[-1, 1], gt_abs_pos[-1, 2], marker="x", s=35, label="GT end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("End-Effector XYZ Trajectory")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_outputs(
        output_dir: Path,
        episode: RawEpisode,
        loaded: LoadedPolicy,
        results: dict[str, np.ndarray | list[float]],
        metrics: dict[str, Any],
        args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(results["pred_action"])
    gt = np.asarray(results["gt_action"])
    pred_abs_pos = np.asarray(results["pred_abs_pos"])
    gt_abs_pos = np.asarray(results["gt_abs_pos"])
    x = np.arange(pred.shape[0], dtype=np.int64)

    np.savez_compressed(
        output_dir / "pred_gt_actions.npz",
        pred_action=pred,
        gt_action=gt,
        error=pred - gt,
        pred_abs_pos=pred_abs_pos,
        gt_abs_pos=gt_abs_pos,
        timestamps=np.asarray(results["timestamps"]),
        current_indices=np.asarray(results["current_indices"]),
        target_indices=np.asarray(results["target_indices"]),
        latencies=np.asarray(results["latencies"]),
    )

    summary = {
        "checkpoint": str(loaded.ckpt_path),
        "data_episode": str(episode.pkl_path),
        "episode_name": episode.name,
        "action_dim": loaded.action_dim,
        "hand_backend": _backend_from_action_dim(loaded.action_dim),
        "pose_repr": loaded.pose_repr,
        "obs_horizon": loaded.obs_horizon,
        "horizon_step": int(args.horizon_step),
        "action_down_sample_steps": loaded.action_down_sample_steps,
        "num_evaluated_steps": int(pred.shape[0]),
        "episode_frames": len(episode),
        "metrics": metrics,
        "mean_policy_latency_s": float(np.mean(np.asarray(results["latencies"]))),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_dim_series(
        x=x,
        pred=pred_abs_pos,
        gt=gt_abs_pos,
        labels=["x", "y", "z"],
        title="End-Effector Position: Prediction vs GT",
        ylabel="position",
        output_path=output_dir / "arm_position.png",
    )

    if loaded.action_dim in (15, 29):
        rot_slice = slice(3, 9)
        rot_labels = [f"rot6d_{i}" for i in range(6)]
        hand_slice = slice(9, loaded.action_dim)
    else:
        rot_slice = slice(3, 6)
        rot_labels = ["rx", "ry", "rz"]
        hand_slice = slice(6, loaded.action_dim)

    _plot_dim_series(
        x=x,
        pred=pred[:, rot_slice],
        gt=gt[:, rot_slice],
        labels=rot_labels,
        title="Arm Rotation Action: Prediction vs GT",
        ylabel="rotation action",
        output_path=output_dir / "arm_rotation.png",
    )

    hand_dim = pred[:, hand_slice].shape[1]
    if _backend_from_action_dim(loaded.action_dim) == "o6":
        hand_labels = [f"o6_{i}" for i in range(hand_dim)]
        hand_ylabel = "command"
    else:
        hand_labels = [f"wuji_{i}" for i in range(hand_dim)]
        hand_ylabel = "radians"
    _plot_dim_series(
        x=x,
        pred=pred[:, hand_slice],
        gt=gt[:, hand_slice],
        labels=hand_labels,
        title="Dexterous Hand Action: Prediction vs GT",
        ylabel=hand_ylabel,
        output_path=output_dir / "hand.png",
    )

    _plot_errors(
        x=x,
        pred=pred,
        gt=gt,
        pred_abs_pos=pred_abs_pos,
        gt_abs_pos=gt_abs_pos,
        action_dim=loaded.action_dim,
        output_path=output_dir / "error.png",
    )
    _plot_trajectory_3d(pred_abs_pos, gt_abs_pos, output_dir / "trajectory_3d.png")

    print("wrote:", output_dir)
    print("summary:", output_dir / "summary.json")
    print("plots: arm_position.png arm_rotation.png hand.png error.png trajectory_3d.png")


def _print_episode_list(data_dir: Path) -> None:
    paths = _find_pkl_paths(data_dir)
    for idx, path in enumerate(paths):
        print(f"{idx:04d} {path.parent.name} {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="checkpoint .ckpt or output dir")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="DexUMI raw collection dir or pkl")
    parser.add_argument("--episode", default="0", help="episode index, episode dir name, pkl file, or episode dir")
    parser.add_argument("--output-dir", default=None, help="directory for summary/npz/plots")
    parser.add_argument("--device", default="auto", help="torch device, or auto")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--horizon-step", type=int, default=0, help="which policy action horizon step to compare")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="limit evaluated windows for smoke tests")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--list-episodes", action="store_true", help="list discovered episodes and exit")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    if args.list_episodes:
        _print_episode_list(data_dir)
        return

    loaded = _load_policy(
        ckpt_path=Path(args.ckpt),
        device_arg=args.device,
        num_inference_steps=args.num_inference_steps,
    )
    pkl_path = _select_pkl(data_dir, args.episode)
    episode = _load_raw_episode(
        pkl_path=pkl_path,
        action_dim=loaded.action_dim,
        skip_missing_images=args.skip_missing_images,
    )
    results = _evaluate_episode(
        episode=episode,
        loaded=loaded,
        horizon_step=int(args.horizon_step),
        start_index=int(args.start_index),
        max_steps=args.max_steps,
    )
    metrics = _compute_metrics(results, loaded.action_dim)

    if args.output_dir is None:
        output_dir = Path("data/eval_dexumi_episode_actions") / episode.name
    else:
        output_dir = Path(args.output_dir).expanduser()
    _write_outputs(
        output_dir=output_dir,
        episode=episode,
        loaded=loaded,
        results=results,
        metrics=metrics,
        args=args,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
