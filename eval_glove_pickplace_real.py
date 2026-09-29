#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""glove_pickplace / WujiHand DP 权重的实际部署推理入口。

这个脚本只负责推理和真机执行，不包含训练、数据转换或标定逻辑。


glove_pickplace_20hz.sh 使用的是 Data-Scaling-Laws 原 UMI 配置:
    --config-name=train_diffusion_unet_timm_umi_workspace

checkpoint 内部的 shape_meta 不是 DexUMI 自定义的 6D action，而是:
    obs:
        camera0_rgb
        robot0_eef_pos
        robot0_eef_rot_axis_angle
        robot0_gripper_width

    action:
        15D = wrist pose10d(9D) + Linker O6 handCommand(6D)
        29D = wrist pose10d(9D) + WujiHand qpos target(20D)

部署时本脚本只真正执行手部 command:
    action_dim=15: pred_action[9:15] -> O6 六个关节 0~255 命令
    action_dim=29: pred_action[9:29] -> WujiHand 20 维关节位置

前 9 维 wrist pose10d 只打印/记录，不发送给机械臂。
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import dill
import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.pose_util import mat_to_pose10d

OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class LoadedPolicy:
    """把 checkpoint 加载后的运行时对象集中放在一起，避免函数参数过长。"""

    cfg: OmegaConf
    policy: Any
    device: torch.device
    obs_horizon: int
    action_dim: int
    hand_obs_dim: int
    backend: str


def _install_numpy_pickle_compat() -> None:
    """兼容不同 numpy 版本保存的 checkpoint/pkl。

    一些旧环境保存的 pickle 会引用 numpy._core；新旧 numpy 包路径不一致时，
    torch.load/pickle.load 可能找不到模块。这里做别名映射，不改变数据内容。
    """

    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def _as_list(value: Any) -> list[Any]:
    """把 OmegaConf 的 ListConfig 转成普通 Python list。"""

    return list(OmegaConf.to_container(value, resolve=True))


def _shape(attr: Any) -> tuple[int, ...]:
    """读取 shape_meta 里某个观测/action 的 shape。"""

    return tuple(int(x) for x in _as_list(attr["shape"]))


def _hand_obs_dim(cfg: OmegaConf) -> int:
    """读取 robot0_gripper_width 的 checkpoint 维度。"""

    obs_meta = cfg.task.shape_meta.obs
    if "robot0_gripper_width" not in obs_meta:
        raise RuntimeError("checkpoint shape_meta.obs missing robot0_gripper_width")
    shape = _shape(obs_meta["robot0_gripper_width"])
    if len(shape) != 1:
        raise RuntimeError(f"robot0_gripper_width must be 1D, got {shape}")
    return int(shape[0])


def _backend_from_action_dim(action_dim: int) -> str:
    """根据 checkpoint action 维度选择实际手部 backend。"""

    if action_dim == 15:
        return "o6"
    if action_dim == 29:
        return "wuji"
    raise RuntimeError(
        f"Unsupported action_dim={action_dim}; expected 15 for O6 or 29 for WujiHand."
    )


def _obs_horizon(cfg: OmegaConf) -> int:
    """从 checkpoint 的 shape_meta 中计算观测历史长度。

    训练配置里每个 obs key 都可能有自己的 horizon；实际组 batch 时需要缓存
    最大 horizon 的历史帧。
    """

    horizons = [
        int(attr.horizon)
        for attr in cfg.task.shape_meta.obs.values()
        if "horizon" in attr
    ]
    return max(horizons) if horizons else 1


def _disable_pretrained_download(cfg: OmegaConf) -> None:
    """禁止部署机重复下载 timm 预训练权重。

    训练好的 ViT/UNet 权重已经在 ckpt 里。部署时如果仍保持 pretrained=True，
    timm.create_model 可能先去 HuggingFace/网络下载 backbone 预训练权重；
    这一步对推理没有必要，还可能因为网络不可用导致部署失败。
    """

    if "policy" in cfg and "obs_encoder" in cfg.policy and "pretrained" in cfg.policy.obs_encoder:
        cfg.policy.obs_encoder.pretrained = False


def load_policy(ckpt_path: str, device_name: str, num_inference_steps: int | None) -> LoadedPolicy:
    """加载训练得到的 diffusion policy。

    关键流程:
    1. torch.load 读取 checkpoint payload。
    2. 从 payload["cfg"] 恢复训练时的 Hydra 配置。
    3. 根据 cfg._target_ 实例化 workspace/model。
    4. load_payload 把 model、ema_model、normalizer 等权重加载回来。
    5. 训练时如果 use_ema=True，推理使用 ema_model。

    该部署脚本按 checkpoint 自动选择 backend:
        action_dim=15 -> Linker O6
        action_dim=29 -> WujiHand
    """

    ckpt = Path(ckpt_path).expanduser()
    if ckpt.is_dir():
        ckpt = ckpt / "checkpoints" / "latest.ckpt"
    _install_numpy_pickle_compat()
    payload = torch.load(ckpt.open("rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    _disable_pretrained_download(cfg)

    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(num_inference_steps)

    device = torch.device(device_name)
    policy.eval().to(device)
    action_dim = int(cfg.task.shape_meta.action.shape[0])
    backend = _backend_from_action_dim(action_dim)
    hand_obs_dim = _hand_obs_dim(cfg)
    expected_hand_obs_dim = 6 if backend == "o6" else 20
    if hand_obs_dim != expected_hand_obs_dim:
        raise RuntimeError(
            f"checkpoint backend={backend} expects robot0_gripper_width shape "
            f"[{expected_hand_obs_dim}], got [{hand_obs_dim}]"
        )
    ignore_hand_obs = cfg.task.shape_meta.obs.robot0_gripper_width.get("ignore_by_policy", False)

    print("checkpoint:", ckpt)
    print("workspace:", cfg._target_)
    print("policy:", cfg.policy._target_)
    print("task:", cfg.task.name if "name" in cfg.task else "<unknown>")
    print("obs_keys:", list(cfg.task.shape_meta.obs.keys()))
    print("action_dim:", action_dim)
    print("hand_backend:", backend)
    print("robot0_gripper_width shape:", [hand_obs_dim], "ignore_by_policy:", bool(ignore_hand_obs))
    print("obs_horizon:", _obs_horizon(cfg))
    if hasattr(policy, "num_inference_steps"):
        print("num_inference_steps:", policy.num_inference_steps)

    return LoadedPolicy(
        cfg=cfg,
        policy=policy,
        device=device,
        obs_horizon=_obs_horizon(cfg),
        action_dim=action_dim,
        hand_obs_dim=hand_obs_dim,
        backend=backend,
    )


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """PICO 四元数 [qx, qy, qz, qw] 转旋转矩阵。

    scipy 和 PICO 都使用 xyzw 顺序，但很多机器人库使用 wxyz。这里手写转换，
    避免后续维护时把四元数顺序混掉。
    """

    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12 or not np.all(np.isfinite(q)):
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


PICO_WRIST_TO_TARGET_TCP_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
"""PICO wrist 局部坐标系到训练使用 TCP 坐标系的固定旋转。

collect_data.py 里也使用同一个定义:
    new_x = old_y
    new_y = old_x
    new_z = -old_z

这个矩阵只改变 wrist 局部轴定义，不包含 PICO world 到机器人 base 的外参。
"""


def wrist_pose_matrix_from_pico(raw26x7: np.ndarray) -> np.ndarray:
    """从 PICO 原始 26x7 手部数据里构造 wrist 4x4 位姿矩阵。

    raw26x7[1] 是 wrist，格式:
        [x, y, z, qx, qy, qz, qw]

    输出语义与 collect_data.py 保存的 trajectoryPose 一致:
        T_picoWorld_wristTcp, shape=(4, 4)
    """

    wrist_pose = np.asarray(raw26x7[1, :7], dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_xyzw_to_rotmat(wrist_pose[3:7]) @ PICO_WRIST_TO_TARGET_TCP_ROT
    mat[:3, 3] = wrist_pose[:3]
    return mat.astype(np.float32)


def _pose4x4_to_obs_pose(
        pose_mat: np.ndarray,
        start_pose_mat: np.ndarray | None,
        pose_repr: str) -> tuple[np.ndarray, np.ndarray]:
    """把 4x4 wrist pose 转成模型需要的低维 obs。

    训练时 UmiDataset.__getitem__ 会把 robot0_eef_pos 和
    robot0_eef_rot_axis_angle 转为 pose10d 表示，其中:
        robot0_eef_pos             -> pose10d 前 3 维
        robot0_eef_rot_axis_angle  -> pose10d 后 6 维

    当前 glove_pickplace 配置里 ignore_proprioception=True，网络实际不使用这些
    低维键；但是 policy.normalizer 仍要求 obs_dict 里存在这些 key，所以部署时
    必须补齐形状正确的数据。

    start_pose_mat 当前仅保留给将来扩展 wrt_start 配置；本 checkpoint 的实时
    低维 obs 按 UMI 逻辑以当前帧 pose 为基准。
    """

    pose_mat = np.asarray(pose_mat, dtype=np.float32).reshape(4, 4)
    if start_pose_mat is None:
        start_pose_mat = pose_mat.copy()
    rel_mat = convert_pose_mat_rep(
        pose_mat[None, ...],
        base_pose_mat=pose_mat,
        pose_rep=pose_repr,
        backward=False,
    )
    pose10d = mat_to_pose10d(rel_mat).astype(np.float32)[0]
    return pose10d[:3], pose10d[3:]


def _rgb_to_chw_seq(frames: np.ndarray, attr: Any) -> np.ndarray:
    """把相机图像从 THWC uint8 RGB 转成模型输入 TCHW float。

    采集/相机输出:
        frames: (T, H, W, 3), RGB, uint8, 0~255

    模型输入:
        (T, 3, 224, 224), float32, 0~1

    这里的目标分辨率不是写死的，而是从 checkpoint 的 shape_meta 读取。
    """

    shape = _shape(attr)
    c, out_h, out_w = shape
    if c != 3:
        raise ValueError(f"Only RGB obs is supported, got shape {shape}")
    arr = np.asarray(frames)
    if arr.ndim != 4:
        raise ValueError(f"expected THWC image sequence, got {arr.shape}")
    t, in_h, in_w, in_c = arr.shape
    if in_c != 3:
        raise ValueError(f"expected RGB images, got {in_c} channels")
    tf = get_image_transform(
        input_res=(in_w, in_h),
        output_res=(out_w, out_h),
        bgr_to_rgb=False,
    )
    out = np.stack([tf(x) for x in arr], axis=0)
    if out.dtype == np.uint8:
        out = out.astype(np.float32) / 255.0
    else:
        out = out.astype(np.float32)
    return np.moveaxis(out, -1, 1)


def format_obs_for_policy(obs_seq: dict[str, np.ndarray], cfg: OmegaConf) -> dict[str, np.ndarray]:
    """按照 checkpoint 的 shape_meta 把 obs_seq 整理成 policy.predict_action 输入。

    不能按我们自己的字段名随便组织 obs；policy 的 normalizer 和 obs_encoder 都
    是按训练 checkpoint 里的 shape_meta 建的，所以 key、shape、dtype 必须匹配。
    """

    result: dict[str, np.ndarray] = {}
    for key, attr in cfg.task.shape_meta.obs.items():
        key = str(key)
        if key not in obs_seq:
            raise KeyError(f"missing obs key {key}; available={sorted(obs_seq.keys())}")
        if attr.get("type", "low_dim") == "rgb":
            result[key] = _rgb_to_chw_seq(obs_seq[key], attr)
        else:
            arr = np.asarray(obs_seq[key], dtype=np.float32)
            expected = _shape(attr)
            if arr.shape[1:] != expected:
                raise ValueError(f"{key} expected T{expected}, got {arr.shape}")
            result[key] = arr
    return result


def predict_action(loaded: LoadedPolicy, obs_seq: dict[str, np.ndarray]) -> np.ndarray:
    """执行一次 diffusion policy 推理。

    输入 obs_seq:
        每个 key 是未加 batch 维度的时间序列，例如:
            camera0_rgb: (T, H, W, 3)
            robot0_eef_pos: (T, 3)

    进入模型前会变成:
        batch=1 的 torch tensor，例如 (1, T, 3, 224, 224)

    输出:
        action: (action_horizon, loaded.action_dim)
    """

    obs_np = format_obs_for_policy(obs_seq, loaded.cfg)
    obs_torch = dict_apply(
        obs_np,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(loaded.device),
    )
    with torch.no_grad():
        result = loaded.policy.predict_action(obs_torch)
    action = result["action"][0].detach().to("cpu").numpy()
    if action.shape[-1] != loaded.action_dim:
        raise RuntimeError(f"policy returned {action.shape}, expected last dim {loaded.action_dim}")
    return action


def action15_to_o6_command(action: np.ndarray) -> np.ndarray:
    """从模型 15D action 中取出 O6 手真正要执行的 6D 命令。

    模型输出维度解释:
        action[0:9]   wrist pose10d，当前脚本不执行
        action[9:15]  handCommand，对应 Linker O6 六个关节，范围 0~255

    输出会 round + clip，并转成 uint8，避免给 O6 发送非法值。
    """

    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] != 15:
        raise ValueError(f"expected 15D action, got {arr.shape}")
    return np.clip(np.rint(arr[9:15]), 0, 255).astype(np.uint8)


def action29_to_wuji_qpos(action: np.ndarray) -> np.ndarray:
    """从模型 29D action 中取出 WujiHand 20D 关节位置目标。

    模型输出维度解释:
        action[0:9]   wrist pose10d，当前脚本不执行
        action[9:29]  WujiHand 20 维 qpos target

    WujiHand 的 qpos 是连续关节位置，不能复用 O6 的 0~255 uint8 转换。
    """

    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] != 29:
        raise ValueError(f"expected 29D action, got {arr.shape}")
    qpos = arr[9:29].reshape(-1)
    if qpos.shape != (20,):
        raise ValueError(f"expected Wuji qpos shape (20,), got {qpos.shape}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Wuji qpos contains NaN or Inf")
    return qpos.astype(np.float64)


def _open_o6_hand(dexumi_root: Path, can_channel: str, bitrate: int, speed: int, torque: int | None):
    """打开 Linker O6 右手 CAN 控制器。

    O6 控制代码不在当前 Data-Scaling-Laws 仓库里，而在 human2dex/o6_right_hand。
    这里临时把该目录加入 sys.path，然后 import controller.O6RightHand。
    """

    root = dexumi_root.expanduser().resolve()
    sys.path.insert(0, str(root / "o6_right_hand"))
    from controller import O6RightHand

    hand = O6RightHand(can_channel=can_channel, bitrate=bitrate)
    hand.set_speed(speed)
    if torque is not None:
        hand.set_torque(torque)
    return hand


def _as_fixed_vector(value: Any, dim: int, name: str) -> np.ndarray:
    """校验并返回固定长度 float32 向量。"""

    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"{name} expected shape ({dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _import_wuji_runtime(wuji_root: Path):
    """从 wuji_demo 动态导入 WujiHandDriver 和 LowPassFilter。"""

    root = wuji_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wuji_pico_hand.filters import LowPassFilter
    from wuji_pico_hand.hand_driver import WujiHandDriver

    return WujiHandDriver, LowPassFilter


def _open_wuji_hand(args: argparse.Namespace):
    """初始化 WujiHand driver 和 qpos 低通滤波器。"""

    WujiHandDriver, LowPassFilter = _import_wuji_runtime(Path(args.wuji_root))
    driver = WujiHandDriver(
        serial_number=args.wuji_serial_number,
        dry_run=bool(args.dry_run),
        timeout=args.wuji_timeout,
    )
    filt = LowPassFilter(alpha=float(args.wuji_alpha))
    if args.dry_run:
        print("dry-run: WujiHand is not opened")
    else:
        limits = driver.refresh_limits()
        driver.enable()
        print("WujiHand enabled")
        if limits is not None:
            print("Wuji lower:", np.round(limits.lower, 4).tolist())
            print("Wuji upper:", np.round(limits.upper, 4).tolist())
    return driver, filt


def _build_mvs_source(args: argparse.Namespace):
    """启动 MVS 相机 latest-frame cache。

    LatestMVSCache 会在后台持续取最新帧；推理循环里只读 cache.latest()，避免
    相机阻塞直接拖慢 policy 推理周期。
    """

    root = Path(args.dexumi_root).expanduser().resolve()
    sys.path.insert(0, str(root / "teleop"))
    sys.path.insert(0, str(root / "teleop" / "mvs_cpp"))
    from mvs_cpp import LatestMVSCache, MVSConfig

    config = MVSConfig(
        serial=args.mvs_serial,
        fps=float(args.frequency),
    )
    cache = LatestMVSCache(config)
    cache.start()
    return cache


def _open_pico_and_retargeter(args: argparse.Namespace):
    """打开 PICO 手部读取器和 DexUMI retargeter。

    PICO 只在命令行显式加 --use-pico 时启用。O6/15D 路径会使用 PICO
    retarget 得到 handCommand；Wuji/29D 路径只复用 wrist pose，手部状态
    仍由 WujiHandDriver.read_positions() 提供。
    """

    root = Path(args.dexumi_root).expanduser().resolve()
    sys.path.insert(0, str(root / "teleop"))
    sys.path.insert(0, str(root))
    from pico_hand import OPERATOR2MANO_RIGHT, PICO2MEDIAPIPE, PicoHandReader, _estimate_wrist_frame
    from retargeter import LinkerO6Retargeter

    reader = PicoHandReader(hand=args.hand)
    retargeter = LinkerO6Retargeter(
        **({"yaml_path": args.retarget_yaml} if args.retarget_yaml else {}),
        hand=args.hand,
    )
    return reader, retargeter, PICO2MEDIAPIPE, OPERATOR2MANO_RIGHT, _estimate_wrist_frame


def _pico_hand_command_and_pose(
        reader: Any,
        retargeter: Any,
        pico2mediapipe: np.ndarray,
        operator2mano_right: np.ndarray,
        estimate_wrist_frame: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """读取一帧 PICO，并转换成 handCommand + wrist 4x4 pose。

    返回:
        hand_cmd:  (6,) float32, Linker O6 命令空间，0~255
        wrist_mat: (4,4) float32, 与 collect_data.py 的 trajectoryPose 一致

    如果 PICO 当前不是高质量 active 状态，返回 None，主循环会沿用上一帧有效值。
    """

    raw26x7, active = reader.read_raw()
    if int(active) != 1 or raw26x7 is None or np.asarray(raw26x7).shape != (26, 7):
        return None
    raw26x7 = np.asarray(raw26x7, dtype=np.float64)

    # PICO 原始是 26 个点；Dex-retargeting 使用 MediaPipe 21 点拓扑。
    pts26 = raw26x7[:, :3]
    pts21_world = pts26[pico2mediapipe]

    # 与 collect_data.py 保持一致:
    # 1. 以 wrist 为原点居中。
    # 2. 用 wrist/index_mcp/middle_mcp 估计手部局部坐标系。
    # 3. 转到 dex-retargeting 期望的 MANO 坐标系。
    centered = pts21_world - pts21_world[0:1, :]
    rot = estimate_wrist_frame(centered)
    pts21_mano = (centered @ rot @ operator2mano_right).astype(np.float32)
    angles = retargeter.retarget(pts21_mano)

    # 与 collect_data.py 的 _angles_rad_to_cmd 保持同一映射:
    #   250 附近: 接近张开
    #   0 附近:   接近最大屈曲
    urdf_upper = np.array([0.58, 1.36, 1.60, 1.60, 1.60, 1.60], dtype=np.float32)
    q = np.clip(np.asarray(angles, dtype=np.float32), 0.0, urdf_upper)
    hand_cmd = np.clip(np.rint(250.0 * (1.0 - q / urdf_upper)), 0, 255).astype(np.float32)
    wrist_mat = wrist_pose_matrix_from_pico(raw26x7)
    return hand_cmd, wrist_mat


def run_real(args: argparse.Namespace, loaded: LoadedPolicy) -> None:
    """真实部署主循环。

    每个周期做以下事情:
    1. 从 MVS 或 cv2 相机取最新 RGB 图像。
    2. 按 checkpoint backend 填 robot0_gripper_width:
           - O6/15D: PICO retarget、O6 get_state() 或 default 6D。
           - Wuji/29D: live 读 WujiHandDriver.read_positions()，失败沿用上一帧；
             dry-run 用 default_wuji_position 填 20D。
    3. 根据 checkpoint 的 obs_keys 组装 obs_buffers。
    4. 调 policy.predict_action，得到 (action_horizon, action_dim)。
    5. 取第一步 hand command，发送给对应手部 backend。

    注意:
    - 这里只执行 action 序列的第一步，没有做 action chunk 调度。
    - 如果需要像原 eval_real.py 那样一次执行多步，可在这里扩展 action queue。
    """

    o6_hand = None
    wuji_driver = None
    wuji_filter = None
    camera = None
    cv2_camera = None
    reader = None
    retargeter = None
    pico2mp = None
    op2mano = None
    estimate_frame = None
    start_pose_mat = None
    if loaded.backend == "o6":
        last_hand_obs = np.full(loaded.hand_obs_dim, float(args.default_hand_command), dtype=np.float32)
    else:
        last_hand_obs = np.full(loaded.hand_obs_dim, float(args.default_wuji_position), dtype=np.float32)
    last_wrist_mat = np.eye(4, dtype=np.float32)
    rows: list[dict[str, Any]] = []

    try:
        # 相机源:
        # - mvs: DexUMI 采集时使用的工业相机路径。
        # - cv2: 普通 USB 摄像头调试路径。
        if args.real_camera == "mvs":
            camera = _build_mvs_source(args)
            print("MVS camera started")
        else:
            cv2_camera = cv2.VideoCapture(int(args.camera_index))
            if not cv2_camera.isOpened():
                raise RuntimeError(f"failed to open cv2 camera index {args.camera_index}")
            print("cv2 camera started")

        # 按 checkpoint action_dim 选择手部 backend。O6 参数只在 15D 时使用；
        # Wuji 参数只在 29D 时使用。
        if loaded.backend == "o6":
            if args.dry_run:
                print("dry-run: CAN hand is not opened")
            else:
                o6_hand = _open_o6_hand(
                    dexumi_root=Path(args.dexumi_root),
                    can_channel=args.can_channel,
                    bitrate=args.bitrate,
                    speed=args.speed,
                    torque=args.torque,
                )
                if args.home_on_start:
                    o6_hand.home()
                    time.sleep(1.0)
        else:
            wuji_driver, wuji_filter = _open_wuji_hand(args)
            if not args.dry_run:
                try:
                    last_hand_obs = _as_fixed_vector(
                        wuji_driver.read_positions(),
                        loaded.hand_obs_dim,
                        "Wuji actual position",
                    )
                except Exception as exc:
                    print(f"Wuji read_positions failed at startup; using default qpos: {exc}")

        # PICO 只在显式指定时启用。Wuji/29D 的手部 proprioception 不依赖 PICO，
        # 而是直接从 WujiHandDriver.read_positions() 读取当前 qpos。
        if args.use_pico:
            reader, retargeter, pico2mp, op2mano, estimate_frame = _open_pico_and_retargeter(args)
            print("PICO source started")
        else:
            print("PICO disabled")

        # 每个 obs key 一个环形队列，长度等于 checkpoint 需要的 obs_horizon。
        # 当前训练脚本把 low_dim_obs_horizon/img_obs_horizon 都设在 cfg 中，
        # 这里不写死，直接读取 checkpoint。
        obs_buffers = {
            str(key): deque(maxlen=loaded.obs_horizon)
            for key in loaded.cfg.task.shape_meta.obs.keys()
        }
        pose_repr = loaded.cfg.task.pose_repr.obs_pose_repr
        period = 1.0 / max(float(args.frequency), 1e-3)
        step_idx = 0
        print("Press Ctrl+C to stop.")

        while args.max_steps is None or step_idx < int(args.max_steps):
            loop_t0 = time.time()

            # 读取一帧 RGB。模型训练时用 RGB，不是 BGR；cv2 路径需要手动转换。
            if args.real_camera == "mvs":
                cached = camera.latest()
                frame = None if cached is None else cached.image
            else:
                ok, bgr = cv2_camera.read()
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
            if frame is None:
                print("camera frame unavailable; skipping")
                time.sleep(period)
                continue

            # 可选的 PICO 状态更新。PICO 掉帧时不终止推理，继续沿用上一帧状态。
            if reader is not None:
                pico = _pico_hand_command_and_pose(reader, retargeter, pico2mp, op2mano, estimate_frame)
                if pico is not None:
                    pico_hand_cmd, last_wrist_mat = pico
                    if loaded.backend == "o6":
                        last_hand_obs = _as_fixed_vector(
                            pico_hand_cmd,
                            loaded.hand_obs_dim,
                            "PICO O6 hand command",
                        )
                    if start_pose_mat is None:
                        start_pose_mat = last_wrist_mat.copy()
            if loaded.backend == "o6" and reader is None and o6_hand is not None:
                # 没启用 PICO 时，用当前 O6 真实关节位置补 robot0_gripper_width。
                # 这个低维键在当前 checkpoint 里 ignore_by_policy=True，但 normalizer
                # 仍要求它存在。
                last_hand_obs = _as_fixed_vector(
                    o6_hand.get_state(),
                    loaded.hand_obs_dim,
                    "O6 hand state",
                )
            if loaded.backend == "wuji" and wuji_driver is not None and not args.dry_run:
                try:
                    last_hand_obs = _as_fixed_vector(
                        wuji_driver.read_positions(),
                        loaded.hand_obs_dim,
                        "Wuji actual position",
                    )
                except Exception as exc:
                    print(f"Wuji read_positions failed; reusing previous qpos: {exc}")

            pos_obs, rot_obs = _pose4x4_to_obs_pose(last_wrist_mat, start_pose_mat, pose_repr)

            # 严格按 checkpoint 的 shape_meta.obs 组装 obs。
            # 不存在的额外 key 不传；缺少的必需 key 必须补齐。
            for key, attr in loaded.cfg.task.shape_meta.obs.items():
                key = str(key)
                obs_type = attr.get("type", "low_dim")
                if obs_type == "rgb":
                    obs_buffers[key].append(frame)
                elif key == "robot0_eef_pos":
                    obs_buffers[key].append(pos_obs)
                elif key == "robot0_eef_rot_axis_angle":
                    obs_buffers[key].append(rot_obs)
                elif key == "robot0_gripper_width":
                    obs_buffers[key].append(last_hand_obs.astype(np.float32))
                else:
                    obs_buffers[key].append(np.zeros(_shape(attr), dtype=np.float32))

            if any(len(buf) < loaded.obs_horizon for buf in obs_buffers.values()):
                # 启动初期历史帧还没攒够，先不推理。
                time.sleep(max(0.0, period - (time.time() - loop_t0)))
                continue

            obs_seq = {key: np.stack(list(buf), axis=0) for key, buf in obs_buffers.items()}
            action = predict_action(loaded, obs_seq)
            first = action[0]
            wrist_pred = first[:9]

            # 只执行预测序列的第一步，腕部 9D 只打印/记录。
            if loaded.backend == "o6":
                cmd = action15_to_o6_command(first)
                if o6_hand is None:
                    print(step_idx, "dry-run cmd", cmd.tolist(), "wrist9", np.round(wrist_pred, 4).tolist())
                else:
                    o6_hand.move(cmd.tolist())
                    print(step_idx, "sent", cmd.tolist())
                rows.append({
                    "step": step_idx,
                    "time": time.time(),
                    "cmd": cmd.tolist(),
                    "pred15_first": first.astype(float).tolist(),
                })
            else:
                if wuji_driver is None or wuji_filter is None:
                    raise RuntimeError("Wuji backend selected but Wuji runtime is not initialized")
                qpos = action29_to_wuji_qpos(first)
                qpos = wuji_driver.clamp(qpos)
                qpos = wuji_filter.apply(qpos)
                sent_qpos = wuji_driver.send_positions(qpos)
                if args.dry_run:
                    print(step_idx, "dry-run qpos", np.round(sent_qpos, 4).tolist(), "wrist9", np.round(wrist_pred, 4).tolist())
                else:
                    print(step_idx, "sent qpos", np.round(sent_qpos, 4).tolist())
                rows.append({
                    "step": step_idx,
                    "time": time.time(),
                    "qpos": np.asarray(sent_qpos, dtype=float).tolist(),
                    "pred29_first": first.astype(float).tolist(),
                })
            step_idx += 1

            # 尽量维持固定推理频率。若模型推理超过 period，则不额外 sleep。
            time.sleep(max(0.0, period - (time.time() - loop_t0)))

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        # 无论正常退出还是 Ctrl+C，都尽量释放相机、CAN、PICO。
        if args.output:
            out = Path(args.output).expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print("wrote:", out)
        if o6_hand is not None:
            if args.home_on_exit:
                o6_hand.home()
                time.sleep(0.5)
            o6_hand.close()
        if wuji_driver is not None:
            wuji_driver.close()
        if camera is not None and hasattr(camera, "stop"):
            camera.stop()
        if cv2_camera is not None:
            cv2_camera.release()
        if reader is not None and hasattr(reader, "close"):
            reader.close()


def _read_rgb_from_pkl_value(value: Any, pkl_dir: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """读取 collect_data.py 保存的 rgbImage 字段。

    新版 collect_data.py 的 rgbImage 可以直接是 ndarray；某些转换流程也可能保存
    为图像文件路径。这里兼容两种情况。
    """

    out_h, out_w = shape_hw
    if isinstance(value, np.ndarray):
        rgb = value
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"rgbImage ndarray must be HWC RGB, got {rgb.shape}")
        if rgb.shape[:2] != (out_h, out_w):
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.uint8)
    if value:
        bgr = cv2.imread(str(pkl_dir / str(value)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(pkl_dir / str(value))
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise ValueError("missing rgbImage")


def _load_pkl_episode(pkl_path: Path, cfg: OmegaConf) -> dict[str, np.ndarray]:
    """把 collect_data.py 的 PKL 重放成当前 checkpoint 需要的 episode。

    只用于 offline 验证，不参与真实部署。

    输入 PKL 字段:
        rgbImage
        trajectoryPose  shape=(4,4)
        handCommand     shape=(6,) for O6 or shape=(20,) for Wuji

    输出 episode 字段会按 checkpoint 的 shape_meta.obs 命名，例如 camera0_rgb、
    robot0_eef_pos、robot0_eef_rot_axis_angle、robot0_gripper_width。
    """

    _install_numpy_pickle_compat()
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    messages = data["messages"]
    obs_meta = cfg.task.shape_meta.obs
    rgb_key = next(str(k) for k, v in obs_meta.items() if v.get("type", "low_dim") == "rgb")
    _, img_h, img_w = _shape(obs_meta[rgb_key])
    pose_repr = cfg.task.pose_repr.obs_pose_repr
    hand_obs_dim = _hand_obs_dim(cfg)

    rows = {str(k): [] for k in obs_meta.keys()}
    actions = []
    start_pose_mat = None
    for msg in messages:
        # 无效帧直接跳过，避免把 None 送进模型。
        if msg.get("handCommand") is None or msg.get("trajectoryPose") is None or msg.get("rgbImage") is None:
            continue
        hand_cmd = _as_fixed_vector(msg["handCommand"], hand_obs_dim, "handCommand")
        pose = np.asarray(msg["trajectoryPose"], dtype=np.float32)
        if pose.shape != (4, 4):
            continue
        if start_pose_mat is None:
            start_pose_mat = pose.copy()
        pos_obs, rot_obs = _pose4x4_to_obs_pose(pose, start_pose_mat, pose_repr)
        rgb = _read_rgb_from_pkl_value(msg["rgbImage"], pkl_path.parent, (img_h, img_w))
        for key, attr in obs_meta.items():
            key = str(key)
            if attr.get("type", "low_dim") == "rgb":
                rows[key].append(rgb)
            elif key == "robot0_eef_pos":
                rows[key].append(pos_obs)
            elif key == "robot0_eef_rot_axis_angle":
                rows[key].append(rot_obs)
            elif key == "robot0_gripper_width":
                rows[key].append(hand_cmd)
            else:
                rows[key].append(np.zeros(_shape(attr), dtype=np.float32))
        # 离线输出里的 action 仅用于对照打印。这里按 checkpoint 的手部维度
        # 构造 pose10d(9) + hand tail，得到 15D 或 29D。
        actions.append(np.concatenate([mat_to_pose10d(pose[None])[0], hand_cmd], axis=0).astype(np.float32))
    if not actions:
        raise RuntimeError(f"no valid frames in {pkl_path}")
    episode = {key: np.stack(values, axis=0) for key, values in rows.items()}
    episode["action"] = np.stack(actions, axis=0)
    return episode


def _load_zarr_episode(zarr_path: Path, episode_index: int | None) -> dict[str, np.ndarray]:
    """读取 replay_buffer.zarr 中一个 episode。

    该函数只做简单读取，不重新执行 UmiDataset 的采样/插值逻辑；因此主要用于
    快速检查 checkpoint 能否跑通，而不是严格复现训练 dataloader。
    """

    root_path = zarr_path.expanduser()
    if root_path.name != "replay_buffer.zarr":
        root_path = root_path / "replay_buffer.zarr"
    root = zarr.open(str(root_path), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    ep = 0 if episode_index is None else int(episode_index)
    start = 0 if ep == 0 else int(ends[ep - 1])
    end = int(ends[ep])
    return {str(key): np.asarray(arr[start:end]) for key, arr in root["data"].items()}


def _obs_from_episode(episode: dict[str, np.ndarray], cfg: OmegaConf, idx: int, obs_horizon: int) -> dict[str, np.ndarray]:
    """从离线 episode 中取以 idx 结尾的一段 obs 历史。

    如果 idx 前面的历史帧不足 obs_horizon，则用第一帧向前 padding，和常见
    sequence sampler 的边界处理一致。
    """

    end = idx + 1
    start = max(0, end - obs_horizon)
    out = {}
    for key in cfg.task.shape_meta.obs.keys():
        key = str(key)
        source_key = key
        if source_key not in episode and key == "camera0_rgb" and "camera_0" in episode:
            source_key = "camera_0"
        arr = np.asarray(episode[source_key])
        seq = arr[start:end]
        if len(seq) < obs_horizon:
            seq = np.concatenate([np.repeat(seq[:1], obs_horizon - len(seq), axis=0), seq], axis=0)
        out[key] = seq
    return out


def run_offline(args: argparse.Namespace, loaded: LoadedPolicy) -> None:
    """离线验证入口。

    作用:
        用已有 PKL 或 replay_buffer.zarr 跑 policy.predict_action，打印模型输出。

    不做:
        不连接相机，不打开 CAN/WujiHand，不发送真机命令。
    """

    if args.pkl:
        episode = _load_pkl_episode(Path(args.pkl).expanduser(), loaded.cfg)
        source = args.pkl
    elif args.zarr:
        episode = _load_zarr_episode(Path(args.zarr).expanduser(), args.episode_index)
        source = args.zarr
    else:
        raise SystemExit("offline mode requires --pkl or --zarr")

    total = len(next(iter(episode.values())))
    stop = total if args.max_steps is None else min(total, int(args.max_steps))
    rows = []
    print("offline source:", source)
    print("frames:", total, "running:", stop)
    for idx in range(stop):
        obs_seq = _obs_from_episode(episode, loaded.cfg, idx, loaded.obs_horizon)
        pred = predict_action(loaded, obs_seq)
        if loaded.backend == "o6":
            cmd = action15_to_o6_command(pred[0])
            row = {"index": idx, "cmd": cmd.tolist()}
            print(idx, "cmd", cmd.tolist(), "pred15", np.round(pred[0], 4).tolist())
        else:
            qpos = action29_to_wuji_qpos(pred[0])
            row = {"index": idx, "qpos": qpos.astype(float).tolist()}
            print(idx, "qpos", np.round(qpos, 4).tolist(), "pred29", np.round(pred[0], 4).tolist())
        for j, value in enumerate(pred[0]):
            row[f"pred_{j}"] = float(value)
        rows.append(row)

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("wrote:", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    # checkpoint 与推理模式。
    parser.add_argument("-i", "--input", required=True, help="checkpoint .ckpt or run dir containing checkpoints/latest.ckpt")
    parser.add_argument("--mode", choices=["real", "offline"], default="real")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--dexumi-root", default="/home/zjc/Desktop/human2dex")
    parser.add_argument("--frequency", "-f", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("-o", "--output", default=None)

    # O6 真机控制参数。--dry-run 时不会打开 CAN。
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--can-channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--speed", type=int, default=200)
    parser.add_argument("--torque", type=int, default=None)
    parser.add_argument("--home-on-start", action="store_true")
    parser.add_argument("--home-on-exit", action="store_true")
    parser.add_argument("--default-hand-command", type=float, default=250.0)

    # WujiHand 真机控制参数。只有 action_dim=29 时使用；--dry-run 不打开真手。
    parser.add_argument("--wuji-root", default="/home/zjc/Desktop/wuji_demo")
    parser.add_argument("--wuji-serial-number", default=None)
    parser.add_argument("--wuji-alpha", type=float, default=0.35)
    parser.add_argument("--wuji-timeout", type=float, default=None)
    parser.add_argument("--default-wuji-position", type=float, default=0.0)

    # PICO 默认关闭；需要调试低维状态时显式加 --use-pico。
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--use-pico", action="store_true")
    parser.add_argument("--retarget-yaml", default=None)

    # 图像源。真实 DexUMI 部署通常用 mvs；普通 USB 摄像头调试可用 cv2。
    parser.add_argument("--real-camera", choices=["mvs", "cv2"], default="mvs")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--mvs-serial", default=None)

    # 离线验证输入。
    parser.add_argument("--pkl", default=None, help="offline validation on collect_data.py PKL")
    parser.add_argument("--zarr", default=None, help="offline validation on replay_buffer.zarr")
    parser.add_argument("--episode-index", type=int, default=None)
    args = parser.parse_args()

    loaded = load_policy(args.input, args.device, args.num_inference_steps)
    if args.mode == "real":
        run_real(args, loaded)
    else:
        run_offline(args, loaded)


if __name__ == "__main__":
    main()
