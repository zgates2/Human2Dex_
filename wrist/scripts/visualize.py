#!/usr/bin/env python3
"""
生成 wrist pose 预测可视化。

用途
----
加载训练好的 checkpoint，从指定 split 中取样本，输出诊断图：
  - RGB 图像
  - GT 3D skeleton
  - Pred 3D skeleton
  - Pred vs GT 同坐标系对比
  - 可选 Attention heatmap

当前不会把 3D skeleton 强行投影回 RGB 图像上作为准确 overlay。原因是项目目前
只有 wrist camera intrinsics，没有 hand-local 到 camera 的准确外参 `camera_T_hand`。
如果没有外参，脚本会写出 `projection_note.txt`，提示 projection skipped。

常用命令
--------
生成 val 可视化：
  python wrist/scripts/visualize.py \
      --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
      --split val

同时输出 attention heatmap：
  python wrist/scripts/visualize.py \
      --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
      --split val \
      --attention

指定输出目录和样本数：
  python wrist/scripts/visualize.py \
      --checkpoint wrist/outputs/runs/<run_name>/checkpoints/best.pt \
      --split test \
      --num-samples 32 \
      --output-dir wrist/outputs/runs/<run_name>/visualizations/test

关键参数
--------
  --checkpoint   必填，模型 checkpoint 路径
  --config       可选，默认使用 checkpoint 内保存的 config
  --split        train / val / test
  --index-path   可选，手动指定 JSONL index
  --output-dir   可选，图片输出目录
  --num-samples  生成多少个样本
  --batch-size   可视化推理 batch size
  --attention    是否保存 Attention heatmap
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
from wrist_pose.model import WristDinoPoseModel  # noqa: E402
from wrist_pose.utils import load_model_state_compatible, load_yaml, resolve_config_paths  # noqa: E402
from wrist_pose.visualize import projection_skipped_message, save_batch_visuals  # noqa: E402


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


def has_camera_extrinsic(config: dict[str, Any]) -> bool:
    projection = config.get("projection", {})
    if not isinstance(projection, dict):
        return False
    return projection.get("camera_T_hand") is not None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attention", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(args, checkpoint)
    resolve_config_paths(cfg, REPO_ROOT)
    precision = choose_precision(str(cfg["train"].get("precision", "auto")))

    index_path = args.index_path or (Path(cfg["data"]["index_dir"]) / f"{args.split}.jsonl")
    dataset = WristPoseDataset(index_path=index_path, image_size=int(cfg["data"]["image_size"]), train=False)
    dataset = Subset(dataset, list(range(min(int(args.num_samples), len(dataset)))))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    model = make_model(cfg, device)
    load_model_state_compatible(model, checkpoint["model"], strict=True)
    model.eval()

    output_dir = args.output_dir or (args.checkpoint.parent.parent / "visualizations" / args.split)
    output_dir.mkdir(parents=True, exist_ok=True)
    message = projection_skipped_message(has_camera_extrinsic(cfg))
    if message is not None:
        print(message)
        (output_dir / "projection_note.txt").write_text(message + "\n", encoding="utf-8")

    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            image = batch["image"].to(device)
            with autocast_context(device, precision):
                outputs = model(image, return_attention=args.attention)
            pred20 = outputs["joints20"].detach().cpu()
            attention = outputs.get("attention")
            if attention is not None:
                attention = attention.detach().cpu()
            save_batch_visuals(
                {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                pred20,
                output_dir,
                prefix=f"{args.split}_{batch_idx:03d}",
                attention=attention,
                max_items=int(args.batch_size),
                image_size=int(cfg["data"]["image_size"]),
            )
            saved += int(pred20.shape[0])

    metadata = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "index_path": str(index_path),
        "saved": saved,
        "projection_note": message,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
