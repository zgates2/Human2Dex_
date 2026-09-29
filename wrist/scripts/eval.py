#!/usr/bin/env python3
"""
评估训练好的 wrist pose checkpoint。

用途
----
加载 `train.py` 保存的 `best.pt` 或 `last.pt`，在 train/val/test split 上计算
3D 手部关键点指标。模型输出 20 个非 wrist joints，评估前会自动拼回固定 wrist
[0, 0, 0] 得到完整 21 x 3。

默认指标
--------
  - MPJPE mm
  - PCK@10mm
  - PCK@20mm
  - Per-joint MPJPE mm
  - Bone Length Error mm
  - PA-MPJPE mm

注意：PA-MPJPE 只作为辅助指标，不建议用来选择 best checkpoint，因为它会掩盖
绝对尺度和姿态错误。

常用命令
--------
评估 test split：
  python wrist/scripts/eval.py \
      --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
      --split test

评估 val split 并限制帧数：
  python wrist/scripts/eval.py \
      --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
      --split val \
      --limit-frames 1024

指定输出路径：
  python wrist/scripts/eval.py \
      --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
      --split test \
      --output wrist/outputs/runs/<run_name>/test_metrics.json

关键参数
--------
  --checkpoint   必填，模型 checkpoint 路径
  --config       可选，默认使用 checkpoint 内保存的 config
  --split        train / val / test
  --index-path   可选，手动指定 JSONL index
  --output       可选，metrics JSON 输出路径
  --batch-size   可选，覆盖评估 batch size
  --limit-frames 可选，只评估前 N 帧
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.dataset import WristPoseDataset  # noqa: E402
from wrist_pose.losses import WristPoseLoss  # noqa: E402
from wrist_pose.metrics import PoseMetricAccumulator  # noqa: E402
from wrist_pose.model import WristDinoPoseModel  # noqa: E402
from wrist_pose.utils import load_model_state_compatible, load_yaml, resolve_config_paths, save_json  # noqa: E402


def choose_precision(requested: str) -> str:
    if requested == "fp32" or not torch.cuda.is_available():
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


def load_config(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if args.config is not None:
        return load_yaml(args.config)
    cfg = checkpoint.get("config")
    if isinstance(cfg, dict):
        return cfg
    return load_yaml(WRIST_ROOT / "configs" / "baseline.yaml")


def make_model(cfg: dict[str, Any], device: torch.device) -> WristDinoPoseModel:
    model = WristDinoPoseModel(
        dino_dir=Path(cfg["model"]["dino_dir"]),
        image_size=int(cfg["data"]["image_size"]),
        feature_dim=int(cfg["model"].get("feature_dim", 384)),
        patch_size=int(cfg["model"].get("patch_size", 16)),
        pooling_dropout=float(cfg["model"].get("pooling_dropout", 0.0)),
        head_dropout=float(cfg["model"].get("head_dropout", 0.1)),
    )
    return model.to(device)


@torch.no_grad()
def evaluate(
    model: WristDinoPoseModel,
    loader: DataLoader,
    criterion: WristPoseLoss,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    metric = PoseMetricAccumulator()
    loss_totals = {
        "loss": 0.0,
        "joint_loss": 0.0,
        "bone_length_loss": 0.0,
        "bone_direction_loss": 0.0,
    }
    steps = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        gt = batch["joints20"].to(device, non_blocking=True)
        valid = batch["valid20"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            pred = model(image)["joints20"]
            losses = criterion(pred, gt, valid)
        metric.update(pred, gt, valid)
        for key in loss_totals:
            loss_totals[key] += float(losses[key].detach().item())
        steps += 1

    metrics = metric.compute()
    metrics.update({key: value / max(1, steps) for key, value in loss_totals.items()})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(args, checkpoint)
    resolve_config_paths(cfg, REPO_ROOT)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    precision = choose_precision(str(cfg["train"].get("precision", "auto")))

    index_path = args.index_path or (Path(cfg["data"]["index_dir"]) / f"{args.split}.jsonl")
    dataset = WristPoseDataset(
        index_path=index_path,
        image_size=int(cfg["data"]["image_size"]),
        train=False,
    )
    if args.limit_frames is not None:
        dataset = Subset(dataset, list(range(min(int(args.limit_frames), len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    model = make_model(cfg, device)
    load_model_state_compatible(model, checkpoint["model"], strict=True)
    criterion = WristPoseLoss(
        joint_beta=float(cfg["loss"]["joint_beta"]),
        bone_length_weight=float(cfg["loss"]["bone_length_weight"]),
        bone_direction_weight=float(cfg["loss"]["bone_direction_weight"]),
    ).to(device)

    metrics = evaluate(model, loader, criterion, device, precision)
    output = args.output or (args.checkpoint.parent.parent / f"{args.split}_metrics.json")
    save_json(output, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"saved metrics -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
