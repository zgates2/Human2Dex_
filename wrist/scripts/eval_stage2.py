#!/usr/bin/env python3
"""评估 Stage2 MANO pose checkpoint。"""

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

from wrist_pose.stage2_dataset import Stage2ManoDataset  # noqa: E402
from wrist_pose.stage2_losses import Stage2ManoLoss  # noqa: E402
from wrist_pose.stage2_metrics import Joint21MetricAccumulator  # noqa: E402
from wrist_pose.stage2_infer import load_stage2_model  # noqa: E402
from wrist_pose.utils import resolve_config_paths, save_json  # noqa: E402


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


def ensure_split_index(cfg: dict[str, Any], split: str, checkpoint: dict[str, Any], checkpoint_path: Path) -> Path:
    if "split_paths" in checkpoint and split in checkpoint["split_paths"]:
        return Path(checkpoint["split_paths"][split])
    run_split = checkpoint_path.parent.parent / "splits" / f"{split}.jsonl"
    if run_split.is_file():
        return run_split
    raise FileNotFoundError(
        f"cannot find split index for {split!r}. Expected {run_split}. "
        "Pass --index-path explicitly if you want to evaluate a custom jsonl."
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, precision: str) -> dict[str, Any]:
    metric = Joint21MetricAccumulator()
    totals = {"loss": 0.0, "pose_loss": 0.0, "joint_loss": 0.0, "bone_loss": 0.0, "pose_prior": 0.0}
    steps = 0
    model.eval()
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        joints21 = batch["joints21"].to(device, non_blocking=True)
        valid21 = batch["valid21"].to(device, non_blocking=True)
        pose = batch["mano_pose"].to(device, non_blocking=True)
        beta = batch["mano_beta"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            outputs = model(image, mano_beta=beta)
            losses = criterion(outputs, pose, joints21, valid21)
        metric.update(outputs["joints21"], joints21, valid21)
        for key in totals:
            totals[key] += float(losses[key].detach().item())
        steps += 1
    metrics = metric.compute()
    metrics.update({key: value / max(1, steps) for key, value in totals.items()})
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
    model, _, cfg = load_stage2_model(args.checkpoint, REPO_ROOT, device, config_path=args.config)
    resolve_config_paths(cfg, REPO_ROOT)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    precision = choose_precision(str(cfg["train"].get("precision", "auto")))
    index_path = args.index_path or ensure_split_index(cfg, args.split, checkpoint, args.checkpoint)
    dataset = Stage2ManoDataset(index_path=index_path, image_size=int(cfg["data"]["image_size"]), train=False)
    if args.limit_frames is not None:
        dataset = Subset(dataset, list(range(min(int(args.limit_frames), len(dataset)))))
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["train"].get("num_workers", 4)), pin_memory=torch.cuda.is_available())
    criterion = Stage2ManoLoss(
        pose_weight=float(cfg["loss"].get("pose_weight", 1.0)),
        joint_weight=float(cfg["loss"].get("joint_weight", 1.0)),
        bone_weight=float(cfg["loss"].get("bone_weight", 0.1)),
        pose_prior_weight=float(cfg["loss"].get("pose_prior_weight", 0.01)),
        joint_beta=float(cfg["loss"].get("joint_beta", 0.005)),
    ).to(device)
    metrics = evaluate(model, loader, criterion, device, precision)
    output = args.output or (args.checkpoint.parent.parent / f"{args.split}_metrics.json")
    save_json(output, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"saved metrics -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
