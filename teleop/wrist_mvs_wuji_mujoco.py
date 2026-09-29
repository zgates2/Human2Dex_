#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrist MVS RGB -> DINO wrist pose -> Wuji retargeting -> MuJoCo 实时可视化。

用途
----
这个脚本用于检查第一阶段 wrist RGB 关键点模型在实时相机流上的控制效果：

  1. 使用 `teleop/collect_data.py` 同一套 MVS 鱼眼相机接口读取 wrist RGB。
  2. 加载 wrist pose checkpoint，预测 wrist-centered / MANO frame 的 21 个关键点。
  3. 复用 `tools/add_wuji_command_to_pick_sponge.py` 的 pts21_mano -> Wuji qpos
     重定向逻辑，输出 `finger1_joint1 ... finger5_joint4` 共 20 个关节弧度。
  4. 将 qpos 写入 MuJoCo 中的 Wuji hand 模型，实时可视化控制情况。

常用命令
--------
  conda run -n pico python teleop/wrist_mvs_wuji_mujoco.py \
      --checkpoint /home/zjc/Desktop/human2dex/wrist/outputs/runs/ckeckpoints/wrist_3/best.pt

指定采集配置和显示相机画面：
  conda run -n pico python teleop/wrist_mvs_wuji_mujoco.py \
      --checkpoint /home/zjc/Desktop/human2dex/wrist/outputs/runs/ckeckpoints/wrist_3/best.pt \
      --collect-config teleop/collect_data.yaml \
      --show-rgb

说明
----
仓库中的 Wuji URDF 引用了 STL mesh。如果 mesh 文件缺失，脚本会自动退化到一个
简化 MuJoCo 手模型：关节名和 qpos 顺序保持一致，但几何只是 capsule/box，用于
检查控制趋势，不代表真实外观。
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TELEOP_ROOT = REPO_ROOT / "teleop"
WRIST_ROOT = REPO_ROOT / "wrist"
for path in (REPO_ROOT, TELEOP_ROOT, WRIST_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collect_config import build_mvs_config, load_pico_mvs_config  # noqa: E402
from sensor_runtime import MVSSensor  # noqa: E402
from wrist_pose.model import WristDinoPoseModel, expand20_to21  # noqa: E402
from wrist_pose.transforms import WristImageTransform  # noqa: E402
from wrist_pose.utils import load_model_state_compatible, load_yaml  # noqa: E402
from wuji_retargeting import Retargeter  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "wrist" / "outputs" / "runs" / "ckeckpoints" / "test_1" / "best.pt"
)
DEFAULT_COLLECT_CONFIG = REPO_ROOT / "teleop" / "collect_data.yaml"
DEFAULT_WUJI_YAML = (
    REPO_ROOT / "wuji_retargeting" / "config" / "adaptive_analytical_pico.yaml"
)
DEFAULT_WUJI_URDF = (
    REPO_ROOT / "wuji_retargeting" / "wuji_hand_description" / "urdf" / "right.urdf"
)
WUJI_JOINT_NAMES = [
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
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


def load_checkpoint_config(checkpoint: dict[str, Any], fallback_config: Path | None) -> dict[str, Any]:
    cfg = checkpoint.get("config")
    if isinstance(cfg, dict):
        return cfg
    if fallback_config is not None:
        return load_yaml(fallback_config)
    return load_yaml(WRIST_ROOT / "configs" / "baseline.yaml")


def resolve_dino_dir(cfg: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    configured = Path(str(cfg.get("model", {}).get("dino_dir", ""))).expanduser()
    if configured.is_dir():
        return configured.resolve()
    local = WRIST_ROOT / "dino"
    if local.is_dir():
        return local.resolve()
    raise FileNotFoundError(
        "找不到 DINOv3 目录。请用 --dino-dir 指定，例如 wrist/dino"
    )


def load_wrist_model(
    checkpoint_path: Path,
    config_path: Path | None,
    dino_dir: Path | None,
    device: torch.device,
) -> tuple[WristDinoPoseModel, WristImageTransform, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = load_checkpoint_config(checkpoint, config_path)
    resolved_dino = resolve_dino_dir(cfg, dino_dir)
    cfg.setdefault("model", {})["dino_dir"] = str(resolved_dino)

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
            "加载本地 DINOv3 需要 safetensors。请在运行环境中安装：\n"
            "  conda run -n pico python -m pip install safetensors"
        ) from exc
    load_model_state_compatible(model, checkpoint["model"], strict=True)
    model.to(device).eval()
    transform = WristImageTransform(
        image_size=int(cfg["data"].get("image_size", 448)),
        train=False,
    )
    return model, transform, cfg


def predict_pts21_mano(
    model: WristDinoPoseModel,
    transform: WristImageTransform,
    rgb: np.ndarray,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    image = transform(rgb).image.unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad(), autocast_context(device, precision):
        joints20 = model(image)["joints20"]
        joints21 = expand20_to21(joints20)
    return joints21[0].detach().cpu().float().numpy().astype(np.float32)


def retarget_mano_points(retargeter: Retargeter, pts21_mano: np.ndarray) -> np.ndarray:
    """
    pts21_mano -> Wuji qpos。

    注意：`Retargeter.retarget()` 适合 raw MediaPipe 点，会先做 wrist-frame/MANO
    转换。wrist 模型已经直接输出 pts21_mano，所以这里复用离线脚本里的做法：
    不再调用 apply_mediapipe_transformations，直接进入 Wuji optimizer。
    """
    points = np.asarray(pts21_mano, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"pts21_mano shape must be (21, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("pts21_mano contains NaN or Inf")

    keypoints = points.copy()
    rotation_xyz = getattr(retargeter, "rotation_xyz", {}) or {}
    has_rotation = any(
        float(rotation_xyz.get(axis, 0.0)) != 0.0
        for axis in ("x", "y", "z")
    )
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
    return qpos


def parse_joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    if not urdf_path.is_file():
        return {}
    root = ET.parse(str(urdf_path)).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in WUJI_JOINT_NAMES:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits[name] = (
            float(limit.attrib.get("lower", "-2.0")),
            float(limit.attrib.get("upper", "2.0")),
        )
    return limits


def mj_range(joint_name: str, limits: dict[str, tuple[float, float]]) -> str:
    lo, hi = limits.get(joint_name, (-1.5, 1.8))
    return f'{lo:.6f} {hi:.6f}'


def simple_finger_xml(
    finger: int,
    base_x: float,
    base_y: float,
    base_z: float,
    yaw: float,
    lengths: tuple[float, float, float, float],
    limits: dict[str, tuple[float, float]],
) -> str:
    rgba = "0.86 0.88 0.90 1"
    lines = [
        f'<body name="finger{finger}_base" pos="{base_x:.4f} {base_y:.4f} {base_z:.4f}" euler="0 0 {yaw:.4f}">',
    ]
    indent = "  "
    for joint in range(1, 5):
        name = f"finger{finger}_joint{joint}"
        axis = "0 1 0" if joint == 1 else "1 0 0"
        length = lengths[joint - 1]
        lines.append(
            f'{indent}<joint name="{name}" type="hinge" axis="{axis}" '
            f'range="{mj_range(name, limits)}" damping="0.04"/>'
        )
        lines.append(
            f'{indent}<geom name="finger{finger}_link{joint}_geom" type="capsule" '
            f'fromto="0 0 0 0 0 {length:.4f}" size="0.0055" rgba="{rgba}"/>'
        )
        if joint < 4:
            lines.append(
                f'{indent}<body name="finger{finger}_link{joint + 1}" pos="0 0 {length:.4f}">'
            )
            indent += "  "
    lines.extend(f"{'  ' * depth}</body>" for depth in range(4, 0, -1))
    return "\n".join(lines)


def build_simple_wuji_model(urdf_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    limits = parse_joint_limits(urdf_path)
    fingers = [
        simple_finger_xml(1, -0.038, -0.004, 0.035, -0.65, (0.025, 0.026, 0.020, 0.015), limits),
        simple_finger_xml(2, -0.026, 0.000, 0.062, -0.18, (0.033, 0.026, 0.020, 0.016), limits),
        simple_finger_xml(3, -0.006, 0.000, 0.066, -0.04, (0.036, 0.029, 0.022, 0.017), limits),
        simple_finger_xml(4, 0.014, 0.000, 0.062, 0.12, (0.034, 0.027, 0.020, 0.016), limits),
        simple_finger_xml(5, 0.033, 0.000, 0.055, 0.28, (0.029, 0.023, 0.018, 0.014), limits),
    ]
    xml = f"""
<mujoco model="simple_wuji_hand">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <light name="key" pos="0 -0.8 0.8" dir="0 1 -1"/>
    <camera name="front" pos="0 -0.45 0.18" xyaxes="1 0 0 0 0.35 1"/>
    <body name="palm" pos="0 0 0">
      <geom name="palm_geom" type="box" pos="0 0 0.035" size="0.048 0.014 0.036" rgba="0.55 0.60 0.68 1"/>
      {' '.join(fingers)}
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def load_wuji_mujoco_model(urdf_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, str]:
    try:
        from load_in_mujoco import load_hand  # noqa: WPS433

        model, data = load_hand(
            urdf_path,
            fix_mimic=True,
            add_actuators=False,
            actuator_kp=5.0,
        )
        return model, data, "urdf"
    except Exception as exc:
        print(f"[Warn] Wuji URDF 加载失败，使用简化 MuJoCo 模型: {exc}")
        model, data = build_simple_wuji_model(urdf_path)
        return model, data, "simple"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_wuji_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos20: np.ndarray,
) -> None:
    qpos = np.asarray(qpos20, dtype=np.float64).reshape(20)
    for name, value in zip(WUJI_JOINT_NAMES, qpos):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        qi = int(model.jnt_qposadr[jid])
        lo, hi = model.jnt_range[jid]
        data.qpos[qi] = clamp(float(value), float(lo), float(hi)) if lo < hi else float(value)
    mujoco.mj_forward(model, data)


def format_status(frame_idx: int, fps: float, mpj: np.ndarray, qpos: np.ndarray, mode: str) -> str:
    tips = mpj[[4, 8, 12, 16, 20]]
    span = float(np.max(np.linalg.norm(tips - mpj[0:1], axis=1)) * 1000.0)
    return (
        f"frame={frame_idx} fps={fps:.1f} mode={mode} "
        f"tip_span={span:.1f}mm qpos=[{qpos.min():+.2f},{qpos.max():+.2f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wrist RGB checkpoint 实时驱动 Wuji hand MuJoCo 可视化"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--wrist-config", type=Path, default=None, help="可选，覆盖 checkpoint 内 config")
    parser.add_argument("--dino-dir", type=Path, default=None, help="本地 DINOv3 目录，默认自动回退到 wrist/dino")
    parser.add_argument("--collect-config", type=Path, default=DEFAULT_COLLECT_CONFIG)
    parser.add_argument("--task-name", type=str, default=None, help="覆盖 collect_data.yaml 的 task_name，仅用于加载配置")
    parser.add_argument("--mvs-serial", type=str, default=None, help="覆盖 MVS 相机 serial")
    parser.add_argument("--wuji-yaml", type=Path, default=DEFAULT_WUJI_YAML)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_WUJI_URDF)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--hz", type=float, default=15.0, help="推理/可视化目标频率")
    parser.add_argument("--qpos-ema", type=float, default=0.0, help="额外 qpos EMA 平滑系数，0 表示关闭")
    parser.add_argument("--show-rgb", action="store_true", help="同时用 OpenCV 显示 wrist RGB")
    parser.add_argument("--debug-every", type=int, default=30)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device = choose_device(args.device)
    print(f"[Init] 加载 wrist 模型: {checkpoint}")
    wrist_model, image_transform, wrist_cfg = load_wrist_model(
        checkpoint_path=checkpoint,
        config_path=args.wrist_config,
        dino_dir=args.dino_dir,
        device=device,
    )
    precision = choose_precision(args.precision, device)
    print(
        f"[Init] wrist model device={device} precision={precision} "
        f"image_size={wrist_cfg['data'].get('image_size', 448)}"
    )

    print(f"[Init] 加载 Wuji retargeter: {args.wuji_yaml}")
    wuji_retargeter = Retargeter.from_yaml(str(args.wuji_yaml), hand_side=args.hand)

    print(f"[Init] 加载 MuJoCo Wuji hand: {args.urdf}")
    mj_model, mj_data, mj_mode = load_wuji_mujoco_model(args.urdf)
    mj_model.opt.gravity[:] = 0
    print(f"[Init] MuJoCo model={mj_mode} nq={mj_model.nq} njnt={mj_model.njnt}")

    print(f"[Init] 加载 MVS 配置: {args.collect_config}")
    collect_cfg = load_pico_mvs_config(args.collect_config.expanduser(), args.task_name)
    if args.mvs_serial:
        collect_cfg.mvs_serial = args.mvs_serial
    collect_cfg.hz = float(max(args.hz, 1.0))
    rgb_sensor = MVSSensor(build_mvs_config(collect_cfg))

    period = 1.0 / max(float(args.hz), 1.0)
    last_frame_id = None
    last_qpos = np.zeros(20, dtype=np.float32)
    qpos_ema = float(np.clip(args.qpos_ema, 0.0, 0.99))
    frame_idx = 0
    fps_t0 = time.perf_counter()
    fps_count = 0
    fps_value = 0.0

    print("[Run] 启动 MVS 和 MuJoCo viewer，关闭 viewer 或 Ctrl+C 退出")
    rgb_sensor.start()
    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 0.35

            while viewer.is_running():
                loop_t0 = time.perf_counter()
                try:
                    frame = rgb_sensor.wait_next(
                        last_frame_id=last_frame_id,
                        timeout_s=max(0.2, period * 2.0),
                        copy_image=True,
                    )
                except TimeoutError:
                    frame = rgb_sensor.latest(copy_image=True)
                if frame.frame_id is not None:
                    last_frame_id = int(frame.frame_id)
                if frame.image is None:
                    viewer.sync()
                    time.sleep(min(period, 0.02))
                    continue

                pts21_mano = predict_pts21_mano(
                    model=wrist_model,
                    transform=image_transform,
                    rgb=frame.image,
                    device=device,
                    precision=precision,
                )
                qpos = retarget_mano_points(wuji_retargeter, pts21_mano)
                if qpos_ema > 0.0 and frame_idx > 0:
                    qpos = qpos_ema * last_qpos + (1.0 - qpos_ema) * qpos
                last_qpos = qpos.astype(np.float32, copy=True)

                apply_wuji_qpos(mj_model, mj_data, last_qpos)
                viewer.sync()

                fps_count += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    fps_value = fps_count / max(now - fps_t0, 1e-6)
                    fps_t0 = now
                    fps_count = 0

                if args.show_rgb:
                    bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        bgr,
                        format_status(frame_idx, fps_value, pts21_mano, last_qpos, mj_mode),
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("wrist RGB", bgr)
                    if cv2.waitKey(1) in (27, ord("q")):
                        break

                if args.debug_every > 0 and frame_idx % int(args.debug_every) == 0:
                    print(format_status(frame_idx, fps_value, pts21_mano, last_qpos, mj_mode))

                elapsed = time.perf_counter() - loop_t0
                sleep_for = period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                frame_idx += 1
    except KeyboardInterrupt:
        print("\n[Exit] Ctrl+C")
    finally:
        rgb_sensor.stop()
        if args.show_rgb:
            cv2.destroyAllWindows()
        print("[Exit] MVS 已释放")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
