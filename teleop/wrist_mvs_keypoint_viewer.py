#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时 MVS wrist RGB -> wrist pose checkpoint -> 3D 关键点骨架可视化。

用途
----
启动 `teleop/collect_data.py` 同一套 MVS 鱼眼相机接口，实时读取 wrist RGB，
加载 Stage1 或 Stage2 wrist pose checkpoint，预测 wrist-centered / hand-local 的
MediaPipe 21 点 3D 关键点，并用 OpenCV 可视化：

  - 左侧：实时 wrist RGB
  - 右侧：预测 3D skeleton 的 XZ / YZ / XY 三个正交视角

注意
----
当前没有 wrist camera 到 hand-local 坐标系的准确外参 `camera_T_hand`，所以本脚本
不会把 3D skeleton 强行投影回 RGB 图像上。右侧骨架视图是 hand-local 坐标系下的
诊断可视化。

运行示例
--------
  conda run -n pico python teleop/wrist_mvs_keypoint_viewer.py

指定权重：
  conda run -n pico python teleop/wrist_mvs_keypoint_viewer.py \
      --checkpoint /home/zjc/Desktop/human2dex/wrist/outputs/runs/ckeckpoints/test_1/best.pt

指定 Stage2 权重：
  conda run -n pico python teleop/wrist_mvs_keypoint_viewer.py \
      --checkpoint /home/zjc/Desktop/human2dex/wrist/outputs/runs/ckeckpoints/wrist_3/best.pt \
      --model-type stage2 \
      --mano-model-dir /home/zjc/Desktop/human2dex/wrist/mano_v1_2/mano_v1_2/models \
      --device cuda \
      --precision bf16

只检查模型加载和一次推理，不打开相机：
  conda run -n pico python teleop/wrist_mvs_keypoint_viewer.py --model-check

列出 MVS 相机：
  conda run -n pico python teleop/wrist_mvs_keypoint_viewer.py --mvs-list

  /home/zjc/miniconda3/envs/pico/bin/python \
  teleop/wrist_mvs_keypoint_viewer.py \
  --checkpoint wrist/outputs/runs/ckeckpoints/wrist_3/best.pt \
  --model-type stage2 \
  --mano-model-dir wrist/mano_v1_2/mano_v1_2/models \
  --collect-config /tmp/dex_collect_data_60.yaml \
  --device cuda \
  --precision bf16 \
  --hz 30 \
  --warmup-iters 20
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = REPO_ROOT / "teleop"
WRIST_ROOT = REPO_ROOT / "wrist"
for path in (REPO_ROOT, TELEOP_ROOT, WRIST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collect_config import build_mvs_config, list_mvs_cameras, load_pico_mvs_config  # noqa: E402
from sensor_runtime import MVSSensor  # noqa: E402
from wrist_pose.mano_layer import ManoDependencyError  # noqa: E402
from wrist_pose.model import WristDinoPoseModel, expand20_to21  # noqa: E402
from wrist_pose.stage2_infer import predict_pts21_mano_stage2  # noqa: E402
from wrist_pose.stage2_model import WristDinoManoPoseModel, infer_pose_output_dim_from_state_dict  # noqa: E402
from wrist_pose.transforms import WristImageTransform  # noqa: E402
from wrist_pose.utils import load_model_state_compatible, load_yaml, resolve_config_paths  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "wrist" / "outputs" / "runs" / "ckeckpoints" / "test_1" / "best.pt"
)
DEFAULT_COLLECT_CONFIG = REPO_ROOT / "teleop" / "collect_data.yaml"
DEFAULT_DINO_DIR = REPO_ROOT / "wrist" / "dino"

JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = {
    "thumb": (80, 180, 255),
    "index": (60, 220, 90),
    "middle": (255, 180, 60),
    "ring": (220, 90, 220),
    "pinky": (80, 140, 255),
    "wrist": (230, 230, 230),
}

BONE_COLORS = [
    FINGER_COLORS["thumb"], FINGER_COLORS["thumb"], FINGER_COLORS["thumb"], FINGER_COLORS["thumb"],
    FINGER_COLORS["index"], FINGER_COLORS["index"], FINGER_COLORS["index"], FINGER_COLORS["index"],
    FINGER_COLORS["middle"], FINGER_COLORS["middle"], FINGER_COLORS["middle"], FINGER_COLORS["middle"],
    FINGER_COLORS["ring"], FINGER_COLORS["ring"], FINGER_COLORS["ring"], FINGER_COLORS["ring"],
    FINGER_COLORS["pinky"], FINGER_COLORS["pinky"], FINGER_COLORS["pinky"], FINGER_COLORS["pinky"],
]


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境 torch.cuda.is_available() 为 False")
    return device


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


def checkpoint_config(checkpoint: dict[str, Any], config_path: Path | None) -> dict[str, Any]:
    if config_path is not None:
        return load_yaml(config_path)
    cfg = checkpoint.get("config")
    if isinstance(cfg, dict):
        return cfg
    return load_yaml(WRIST_ROOT / "configs" / "baseline.yaml")


def resolve_dino_dir(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"DINOv3 目录不存在: {path}")
        return path

    configured = Path(str(cfg.get("model", {}).get("dino_dir", ""))).expanduser()
    if configured.is_dir():
        return configured.resolve()
    if DEFAULT_DINO_DIR.is_dir():
        return DEFAULT_DINO_DIR.resolve()
    raise FileNotFoundError(f"找不到 DINOv3 目录，请用 --dino-dir 指定。默认: {DEFAULT_DINO_DIR}")


def detect_checkpoint_type(checkpoint: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint 中缺少 model state_dict")
    keys = [key.removeprefix("module.") for key in state_dict]
    cfg = checkpoint.get("config")
    if any(key.startswith(("pose_head.", "mano_layer.")) or key == "default_beta" for key in keys):
        return "stage2"
    if isinstance(cfg, dict) and isinstance(cfg.get("mano"), dict):
        return "stage2"
    return "stage1"


def require_transformers_if_needed(
    state_dict: dict[str, Any],
    allow_local_dino_fallback: bool,
) -> None:
    trained_with_transformers_wrapper = any(
        key.startswith("backbone.model.") or key.startswith("module.backbone.model.")
        for key in state_dict
    )
    has_transformers = importlib.util.find_spec("transformers") is not None
    if trained_with_transformers_wrapper and not has_transformers and not allow_local_dino_fallback:
        raise RuntimeError(
            "当前环境没有 transformers，不能可靠运行这份 checkpoint。\n"
            "这份权重是在 Transformers DINOv3 wrapper 下训练/评估的；"
            "本地 fallback 虽然可能能加载，但离线误差会明显变大，实时结果会看起来完全错误。\n"
            "请先安装依赖：\n"
            "  conda run -n pico python -m pip install -r wrist/requirements.txt\n"
            "仅做底层调试时才加 --allow-local-dino-fallback。"
        )


def load_stage1_wrist_model(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    config_path: Path | None,
    dino_dir: Path | None,
    device: torch.device,
    allow_local_dino_fallback: bool,
) -> tuple[WristDinoPoseModel, WristImageTransform, dict[str, Any]]:
    cfg = checkpoint_config(checkpoint, config_path)
    resolved_dino = resolve_dino_dir(cfg, dino_dir)
    cfg.setdefault("model", {})["dino_dir"] = str(resolved_dino)

    state_dict = checkpoint["model"]
    require_transformers_if_needed(state_dict, allow_local_dino_fallback)

    try:
        model = WristDinoPoseModel(
            dino_dir=resolved_dino,
            image_size=int(cfg["data"].get("image_size", 448)),
            feature_dim=int(cfg["model"].get("feature_dim", 384)),
            patch_size=int(cfg["model"].get("patch_size", 16)),
            pooling_dropout=float(cfg["model"].get("pooling_dropout", 0.0)),
            head_dropout=float(cfg["model"].get("head_dropout", 0.1)),
        )
    except ImportError as exc:
        raise RuntimeError(
            "加载本地 DINOv3 需要 safetensors。请安装：\n"
            "  conda run -n pico python -m pip install safetensors"
        ) from exc

    load_model_state_compatible(model, state_dict, strict=True)
    model.to(device).eval()
    transform = WristImageTransform(
        image_size=int(cfg["data"].get("image_size", 448)),
        train=False,
    )
    return model, transform, cfg


def load_stage2_wrist_model(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    config_path: Path | None,
    dino_dir: Path | None,
    mano_model_dir: Path | None,
    device: torch.device,
    allow_local_dino_fallback: bool,
) -> tuple[WristDinoManoPoseModel, WristImageTransform, dict[str, Any]]:
    if config_path is not None:
        cfg = load_yaml(config_path)
    else:
        cfg = checkpoint.get("config")
        if not isinstance(cfg, dict):
            cfg = load_yaml(WRIST_ROOT / "configs" / "stage2_mano.yaml")
    resolve_config_paths(cfg, REPO_ROOT)

    if dino_dir is not None:
        cfg.setdefault("model", {})["dino_dir"] = str(resolve_dino_dir(cfg, dino_dir))
    elif not Path(str(cfg.get("model", {}).get("dino_dir", ""))).is_dir():
        cfg.setdefault("model", {})["dino_dir"] = str(resolve_dino_dir(cfg, None))

    if mano_model_dir is not None:
        cfg.setdefault("mano", {})["model_dir"] = str(mano_model_dir.expanduser().resolve())

    state_dict = checkpoint["model"]
    require_transformers_if_needed(state_dict, allow_local_dino_fallback)
    pose_output_dim = infer_pose_output_dim_from_state_dict(state_dict)

    mano = cfg.get("mano", {})
    try:
        model = WristDinoManoPoseModel(
            dino_dir=Path(cfg["model"]["dino_dir"]),
            mano_model_dir=Path(mano["model_dir"]),
            image_size=int(cfg["data"].get("image_size", 448)),
            feature_dim=int(cfg["model"].get("feature_dim", 384)),
            patch_size=int(cfg["model"].get("patch_size", 16)),
            head_dropout=float(cfg["model"].get("head_dropout", 0.1)),
            decoder_layers=int(cfg["model"].get("decoder_layers", 2)),
            decoder_heads=int(cfg["model"].get("decoder_heads", 6)),
            mano_side=str(mano.get("side", "right")),
            mano_use_pca=bool(mano.get("use_pca", False)),
            mano_flat_hand_mean=bool(mano.get("flat_hand_mean", False)),
            mano_output_scale=float(mano.get("output_scale", 1.0)),
            pose_output_dim=pose_output_dim,
        )
    except ManoDependencyError as exc:
        raise RuntimeError(
            "Stage2 checkpoint 需要 manotorch 和 MANO 模型文件。\n"
            "请确认已安装依赖，并且 --mano-model-dir 或 config 中的 mano.model_dir "
            "指向包含 MANO_RIGHT.pkl/MANO_LEFT.pkl 的 models 目录。\n"
            f"原始错误: {exc}"
        ) from exc

    load_model_state_compatible(model, state_dict, strict=True)
    model.to(device).eval()
    transform = WristImageTransform(
        image_size=int(cfg["data"].get("image_size", 448)),
        train=False,
    )
    return model, transform, cfg


def load_wrist_model(
    checkpoint_path: Path,
    config_path: Path | None,
    dino_dir: Path | None,
    mano_model_dir: Path | None,
    device: torch.device,
    allow_local_dino_fallback: bool,
    model_type: str,
) -> tuple[Any, WristImageTransform, dict[str, Any], str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    detected = detect_checkpoint_type(checkpoint, requested=model_type)
    if detected == "stage1":
        model, transform, cfg = load_stage1_wrist_model(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            config_path=config_path,
            dino_dir=dino_dir,
            device=device,
            allow_local_dino_fallback=allow_local_dino_fallback,
        )
    elif detected == "stage2":
        model, transform, cfg = load_stage2_wrist_model(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            config_path=config_path,
            dino_dir=dino_dir,
            mano_model_dir=mano_model_dir,
            device=device,
            allow_local_dino_fallback=allow_local_dino_fallback,
        )
    else:
        raise ValueError(f"未知 model type: {detected}")
    return model, transform, cfg, detected


def predict_pts21_mano(
    model: Any,
    transform: WristImageTransform,
    rgb: np.ndarray,
    device: torch.device,
    precision: str,
    model_type: str,
) -> np.ndarray:
    if model_type == "stage2":
        return predict_pts21_mano_stage2(
            model=model,
            transform=transform,
            rgb=rgb,
            device=device,
            precision=precision,
        )
    image = transform(rgb).image.unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad(), autocast_context(device, precision):
        joints20 = model(image)["joints20"]
        joints21 = expand20_to21(joints20)
    return joints21[0].detach().cpu().float().numpy().astype(np.float32)


def start_mvs_sensor(args: argparse.Namespace) -> MVSSensor:
    config = load_pico_mvs_config(args.collect_config.expanduser(), task_name=args.task_name)
    if args.mvs_serial:
        config.mvs_serial = args.mvs_serial
    if args.camera_hz is not None:
        config.hz = float(args.camera_hz)
    if args.mvs_exposure_us is not None:
        config.mvs_exposure_time_us = float(args.mvs_exposure_us)
    if args.mvs_resolution is not None:
        config.mvs_resolution = tuple(int(v) for v in args.mvs_resolution)
    if args.mvs_input_resolution is not None:
        config.mvs_input_resolution = tuple(int(v) for v in args.mvs_input_resolution)
    if args.sensor_crop_roi is not None:
        config.mvs_use_sensor_crop_roi = bool(args.sensor_crop_roi)

    print(
        f"[Init] 启动 MVS: config={args.collect_config} "
        f"resolution={config.mvs_resolution} input={config.mvs_input_resolution} "
        f"fps={config.hz} exposure_us={config.mvs_exposure_time_us} "
        f"sensor_crop_roi={config.mvs_use_sensor_crop_roi}"
    )
    sensor = MVSSensor(build_mvs_config(config))
    sensor.start()
    return sensor


def project_points(
    points: np.ndarray,
    axes: tuple[int, int],
    center: tuple[int, int],
    px_per_m: float,
) -> np.ndarray:
    xy = points[:, axes].astype(np.float32)
    out = np.zeros((points.shape[0], 2), dtype=np.int32)
    out[:, 0] = np.round(center[0] + xy[:, 0] * px_per_m).astype(np.int32)
    out[:, 1] = np.round(center[1] - xy[:, 1] * px_per_m).astype(np.int32)
    return out


def draw_skeleton_view(
    canvas: np.ndarray,
    points: np.ndarray,
    origin_xy: tuple[int, int],
    size: int,
    axes: tuple[int, int],
    title: str,
    px_per_m: float,
) -> None:
    x0, y0 = origin_xy
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (55, 55, 55), 1)
    cv2.putText(
        canvas,
        title,
        (x0 + 8, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    center = (x0 + size // 2, y0 + size // 2)
    cv2.line(canvas, (center[0] - 20, center[1]), (center[0] + 20, center[1]), (80, 80, 80), 1)
    cv2.line(canvas, (center[0], center[1] - 20), (center[0], center[1] + 20), (80, 80, 80), 1)

    pts2d = project_points(points, axes=axes, center=center, px_per_m=px_per_m)
    for bone_idx, (a, b) in enumerate(BONES):
        pa = tuple(int(v) for v in pts2d[a])
        pb = tuple(int(v) for v in pts2d[b])
        cv2.line(canvas, pa, pb, BONE_COLORS[bone_idx], 2, cv2.LINE_AA)
    for idx, p in enumerate(pts2d):
        radius = 4 if idx == 0 else 3
        color = (40, 40, 255) if idx == 0 else (245, 245, 245)
        cv2.circle(canvas, tuple(int(v) for v in p), radius, color, -1, cv2.LINE_AA)


def bone_length_stats(points: np.ndarray) -> tuple[float, float]:
    lengths = [float(np.linalg.norm(points[b] - points[a])) for a, b in BONES]
    return float(np.mean(lengths) * 1000.0), float(np.max(lengths) * 1000.0)


def make_visualization(
    rgb: np.ndarray,
    pts21: np.ndarray,
    fps: float,
    frame_id: int | None,
    view_range_m: float,
    camera_dt_ms: float | None = None,
) -> np.ndarray:
    rgb_view = cv2.resize(rgb, (480, 480), interpolation=cv2.INTER_AREA)
    rgb_view = cv2.cvtColor(rgb_view, cv2.COLOR_RGB2BGR)

    right = np.zeros((480, 480, 3), dtype=np.uint8)
    right[:] = (22, 22, 22)
    panel = 220
    px_per_m = panel / max(2.0 * float(view_range_m), 1e-6)

    draw_skeleton_view(right, pts21, (10, 10), panel, (0, 2), "XZ front", px_per_m)
    draw_skeleton_view(right, pts21, (250, 10), panel, (1, 2), "YZ side", px_per_m)
    draw_skeleton_view(right, pts21, (10, 250), panel, (0, 1), "XY palm", px_per_m)

    mean_bone_mm, max_bone_mm = bone_length_stats(pts21)
    tips = pts21[[4, 8, 12, 16, 20]]
    tip_span_mm = float(np.max(np.linalg.norm(tips - pts21[0:1], axis=1)) * 1000.0)
    lines = [
        f"rgb_id: {frame_id}",
        f"fps: {fps:.1f}",
        f"cam_dt: {camera_dt_ms:.1f} ms" if camera_dt_ms is not None else "cam_dt: n/a",
        f"tip_span: {tip_span_mm:.1f} mm",
        f"mean_bone: {mean_bone_mm:.1f} mm",
        f"max_bone: {max_bone_mm:.1f} mm",
        "unit: meter",
    ]
    y = 272
    for line in lines:
        cv2.putText(
            right,
            line,
            (250, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 24

    cv2.putText(
        rgb_view,
        "MVS wrist RGB",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([rgb_view, right], axis=1)


def warmup_model(
    model: Any,
    transform: WristImageTransform,
    device: torch.device,
    precision: str,
    model_type: str,
    image_size: int,
    iters: int,
) -> None:
    if iters <= 0:
        return
    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    print(f"[Init] GPU/model warmup iters={iters}")
    for _ in range(int(iters)):
        predict_pts21_mano(
            model=model,
            transform=transform,
            rgb=dummy,
            device=device,
            precision=precision,
            model_type=model_type,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="实时 MVS wrist RGB 关键点骨架可视化")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wrist-config", type=Path, default=None, help="可选，覆盖 checkpoint 内保存的 config")
    parser.add_argument("--model-type", choices=("auto", "stage1", "stage2"), default="auto", help="checkpoint 类型，默认自动识别")
    parser.add_argument("--dino-dir", type=Path, default=None, help=f"DINOv3 目录，默认 {DEFAULT_DINO_DIR}")
    parser.add_argument("--mano-model-dir", type=Path, default=None, help="Stage2 MANO models 目录，可指向 .../wrist/mano/models")
    parser.add_argument("--collect-config", type=Path, default=DEFAULT_COLLECT_CONFIG)
    parser.add_argument("--task-name", type=str, default="wrist_keypoint_viewer")
    parser.add_argument("--mvs-serial", type=str, default=None, help="覆盖 MVS 相机 serial")
    parser.add_argument("--mvs-list", action="store_true", help="列出 MVS 相机后退出")
    parser.add_argument("--camera-hz", type=float, default=None, help="覆盖 MVS 相机 fps；不填使用 collect config")
    parser.add_argument("--mvs-exposure-us", type=float, default=None, help="覆盖 MVS 曝光时间 us")
    parser.add_argument("--mvs-resolution", type=int, nargs=2, default=None, metavar=("W", "H"), help="覆盖 MVS 输出分辨率")
    parser.add_argument("--mvs-input-resolution", type=int, nargs=2, default=None, metavar=("W", "H"), help="覆盖 MVS 传感器输入分辨率")
    parser.add_argument("--sensor-crop-roi", type=int, choices=(0, 1), default=None, help="覆盖 mvs_use_sensor_crop_roi，1 开启，0 关闭")
    parser.add_argument("--hz", type=float, default=30.0, help="推理/显示目标频率")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--view-range-m", type=float, default=0.18, help="骨架视图半宽，单位 meter")
    parser.add_argument("--window-name", type=str, default="wrist keypoints")
    parser.add_argument("--debug-every", type=int, default=30)
    parser.add_argument("--profile-every", type=int, default=0, help="每 N 帧打印 wait/predict/visualize/show 耗时；0 表示关闭")
    parser.add_argument("--display-every", type=int, default=1, help="每 N 帧刷新一次 OpenCV 窗口；推理仍按 --hz 跑")
    parser.add_argument("--no-display", action="store_true", help="不打开 OpenCV 窗口，只跑取图和推理，用于测最大处理 fps")
    parser.add_argument("--latest-only", action="store_true", help="不等待新帧，直接取 latest；用于判断 wait_next 是否是瓶颈")
    parser.add_argument("--no-sleep", action="store_true", help="不按 --hz sleep，用于测当前 pipeline 最大速度")
    parser.add_argument("--warmup-iters", type=int, default=10, help="启动相机前先跑 N 次模型 warmup")
    parser.add_argument("--fps-alpha", type=float, default=0.2, help="显示 FPS 的 EWMA 平滑系数")
    parser.add_argument("--model-check", action="store_true", help="只检查模型加载和一次假图推理，不启动 MVS")
    parser.add_argument(
        "--allow-local-dino-fallback",
        action="store_true",
        help="允许在没有 transformers 时使用本地 DINO fallback；仅建议调试使用",
    )
    args = parser.parse_args()

    if args.mvs_list:
        list_mvs_cameras()
        return 0

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {checkpoint}")

    device = choose_device(args.device)
    print(f"[Init] 加载 checkpoint: {checkpoint}")
    model, transform, cfg, loaded_model_type = load_wrist_model(
        checkpoint_path=checkpoint,
        config_path=args.wrist_config,
        dino_dir=args.dino_dir,
        mano_model_dir=args.mano_model_dir,
        device=device,
        allow_local_dino_fallback=args.allow_local_dino_fallback,
        model_type=args.model_type,
    )
    precision = choose_precision(args.precision, device)
    print(
        f"[Init] model_type={loaded_model_type} device={device} precision={precision} "
        f"image_size={cfg['data'].get('image_size', 448)}"
    )
    if args.model_check:
        dummy = np.zeros((480, 480, 3), dtype=np.uint8)
        pts21 = predict_pts21_mano(
            model=model,
            transform=transform,
            rgb=dummy,
            device=device,
            precision=precision,
            model_type=loaded_model_type,
        )
        mean_bone_mm, max_bone_mm = bone_length_stats(pts21)
        print(
            "[Check] 模型推理正常: "
            f"pts21_shape={pts21.shape} "
            f"range_m=({float(pts21.min()):.4f}, {float(pts21.max()):.4f}) "
            f"mean_bone={mean_bone_mm:.1f}mm max_bone={max_bone_mm:.1f}mm"
        )
        return 0

    warmup_model(
        model=model,
        transform=transform,
        device=device,
        precision=precision,
        model_type=loaded_model_type,
        image_size=int(cfg["data"].get("image_size", 448)),
        iters=int(args.warmup_iters),
    )
    sensor = start_mvs_sensor(args)
    period = 1.0 / max(float(args.hz), 1e-3)
    last_frame_id: int | None = None
    last_debug_frame_id: int | None = None
    last_capture_ns: int | None = None
    camera_dt_ms: float | None = None
    frame_idx = 0
    fps_t0 = time.perf_counter()
    fps_count = 0
    fps_value = 0.0
    fps_ema = 0.0
    last_loop_t: float | None = None
    fps_alpha = min(max(float(args.fps_alpha), 0.01), 1.0)
    display_every = max(1, int(args.display_every))
    if not args.no_display:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        print("[Run] 按 q 或 ESC 退出")
    else:
        print("[Run] no-display 模式，按 Ctrl+C 退出")
    try:
        while True:
            t0 = time.perf_counter()
            if last_loop_t is not None:
                loop_dt = max(t0 - last_loop_t, 1e-6)
                instant_fps = 1.0 / loop_dt
                fps_ema = instant_fps if fps_ema <= 0.0 else (1.0 - fps_alpha) * fps_ema + fps_alpha * instant_fps
            last_loop_t = t0
            if args.latest_only:
                frame = sensor.latest(copy_image=True)
            else:
                try:
                    frame = sensor.wait_next(
                        last_frame_id=last_frame_id,
                        timeout_s=max(0.2, period * 2.0),
                        copy_image=True,
                    )
                except TimeoutError:
                    frame = sensor.latest(copy_image=True)
            t_after_wait = time.perf_counter()

            if frame.frame_id is not None:
                last_frame_id = int(frame.frame_id)
            if frame.image is None:
                time.sleep(min(period, 0.02))
                continue
            if frame.capture_ns is not None:
                capture_ns = int(frame.capture_ns)
                if last_capture_ns is not None:
                    camera_dt_ms = (capture_ns - last_capture_ns) / 1_000_000.0
                last_capture_ns = capture_ns

            pts21 = predict_pts21_mano(
                model=model,
                transform=transform,
                rgb=frame.image,
                device=device,
                precision=precision,
                model_type=loaded_model_type,
            )
            t_after_predict = time.perf_counter()

            fps_count += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                fps_value = fps_count / max(now - fps_t0, 1e-6)
                fps_t0 = now
                fps_count = 0

            t_after_vis = t_after_predict
            t_after_show = t_after_predict
            if not args.no_display and frame_idx % display_every == 0:
                vis = make_visualization(
                    rgb=frame.image,
                    pts21=pts21,
                    fps=fps_ema if fps_ema > 0.0 else fps_value,
                    frame_id=frame.frame_id,
                    view_range_m=args.view_range_m,
                    camera_dt_ms=camera_dt_ms,
                )
                t_after_vis = time.perf_counter()
                cv2.imshow(args.window_name, vis)
                key = cv2.waitKey(1)
                t_after_show = time.perf_counter()
                if key in (27, ord("q")):
                    break

            if args.debug_every > 0 and frame_idx % int(args.debug_every) == 0:
                mean_bone_mm, max_bone_mm = bone_length_stats(pts21)
                frame_delta = (
                    None
                    if last_debug_frame_id is None or frame.frame_id is None
                    else int(frame.frame_id) - int(last_debug_frame_id)
                )
                if frame.frame_id is not None:
                    last_debug_frame_id = int(frame.frame_id)
                print(
                    f"frame={frame_idx} rgb_id={frame.frame_id} "
                    f"fps_ema={fps_ema:.1f} fps_1s={fps_value:.1f} "
                    f"cam_dt={camera_dt_ms if camera_dt_ms is not None else -1.0:.1f}ms "
                    f"rgb_delta={frame_delta} frame_gap={frame.frame_gap} "
                    f"mean_bone={mean_bone_mm:.1f}mm max_bone={max_bone_mm:.1f}mm"
                )
            if args.profile_every > 0 and frame_idx % int(args.profile_every) == 0:
                print(
                    "[Profile] "
                    f"wait={(t_after_wait - t0) * 1000.0:.1f}ms "
                    f"predict={(t_after_predict - t_after_wait) * 1000.0:.1f}ms "
                    f"visualize={(t_after_vis - t_after_predict) * 1000.0:.1f}ms "
                    f"show={(t_after_show - t_after_vis) * 1000.0:.1f}ms"
                )

            sleep_for = period - (time.perf_counter() - t0)
            if sleep_for > 0 and not args.no_sleep:
                time.sleep(sleep_for)
            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C")
    finally:
        sensor.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[Exit] MVS 已释放")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
