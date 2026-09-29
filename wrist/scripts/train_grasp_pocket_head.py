#!/usr/bin/env python3
"""Train the RGB-only 2D functional grasp-pocket head.

The supervision is one manually clicked ``grasp_pocket`` per image.  The head
only consumes frozen DINO patch tokens: it intentionally never receives MANO,
PICO, or a projected 21-point skeleton.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wrist_pose.stage2_infer import load_stage2_model  # noqa: E402
from wrist_pose.stage2_projection import ProjectionHead, decode_direct_pocket  # noqa: E402
from train_stage2_projection_head import (  # noqa: E402
    DEFAULT_STAGE1_CHECKPOINT,
    ProjectionAnnotationDataset,
    atomic_write_json,
    choose_precision,
    freeze_stage1,
    load_annotation_records,
    make_loader,
    set_seed,
    split_records,
    summarize_records,
    to_jsonable,
)

LABELS = ["grasp_pocket"]


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def resolve_device(value: str) -> torch.device:
    device = torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def forward_direct(stage1: nn.Module, head: ProjectionHead, image: torch.Tensor, image_size: int, device: torch.device, precision: str) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with autocast_context(device, precision):
            outputs = stage1(image, return_tokens=True)
    raw = head(outputs["patch_tokens"].detach().float(), None)
    return decode_direct_pocket(raw, image_size), raw


def point_loss(uv: torch.Tensor, confidence: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, beta: float, confidence_weight: float, confidence_radius: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = valid[:, 0]
    if not bool(mask.any()):
        zero = uv.sum() * 0.0
        return zero, zero, zero
    delta = uv[mask] - target[mask, 0]
    loc = torch.nn.functional.smooth_l1_loss(delta, torch.zeros_like(delta), beta=float(beta))
    error = torch.linalg.norm(delta.detach(), dim=-1)
    confidence_target = torch.exp(-error / float(confidence_radius)).clamp(0.02, 0.98)
    conf = torch.nn.functional.binary_cross_entropy(confidence[mask], confidence_target)
    return loc + float(confidence_weight) * conf, loc, conf


@torch.no_grad()
def evaluate(stage1: nn.Module, head: ProjectionHead, loader: Any, image_size: int, device: torch.device, precision: str, args: argparse.Namespace) -> dict[str, Any]:
    errors: list[float] = []
    confidences: list[float] = []
    losses: list[float] = []
    per_episode: dict[str, list[float]] = {}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target_uv"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        (uv, confidence), _raw = forward_direct(stage1, head, image, image_size, device, precision)
        loss, _loc, _conf = point_loss(uv, confidence, target, valid, args.loss_beta, args.confidence_weight, args.confidence_radius)
        losses.append(float(loss.item()))
        values = torch.linalg.norm(uv - target[:, 0], dim=-1).detach().cpu().numpy()
        mask = valid[:, 0].detach().cpu().numpy()
        conf_np = confidence.detach().cpu().numpy()
        episodes = list(batch["episode"])
        for idx, is_valid in enumerate(mask):
            if not is_valid:
                continue
            value = float(values[idx])
            errors.append(value)
            confidences.append(float(conf_np[idx]))
            per_episode.setdefault(str(episodes[idx]), []).append(value)
    arr = np.asarray(errors, dtype=np.float64)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "n_points": int(arr.size),
        "mean_px": float(np.mean(arr)) if arr.size else float("nan"),
        "median_px": float(np.median(arr)) if arr.size else float("nan"),
        "p90_px": float(np.percentile(arr, 90)) if arr.size else float("nan"),
        "max_px": float(np.max(arr)) if arr.size else float("nan"),
        "mean_confidence": float(np.mean(confidences)) if confidences else float("nan"),
        "per_episode": {key: {"n": len(value), "median_px": float(np.median(value))} for key, value in sorted(per_episode.items())},
    }


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def save_overlays(stage1: nn.Module, head: ProjectionHead, dataset: ProjectionAnnotationDataset, indices: list[int], output: Path, image_size: int, device: torch.device, precision: str, limit: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if not indices:
        return
    loader = make_loader(dataset, indices[:limit], batch_size=min(32, limit), shuffle=False, num_workers=0)
    saved = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        (uv, confidence), _raw = forward_direct(stage1, head, image, image_size, device, precision)
        uv_np, conf_np = uv.detach().cpu().numpy(), confidence.detach().cpu().numpy()
        target_np = batch["target_uv"].numpy()
        valid_np = batch["valid"].numpy()
        scales, offsets = batch["scale"].numpy(), batch["offset_xy"].numpy()
        for idx, image_path in enumerate(batch["image_path"]):
            if not valid_np[idx, 0]:
                continue
            pred = (uv_np[idx] - offsets[idx]) / scales[idx]
            target = (target_np[idx, 0] - offsets[idx]) / scales[idx]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            cv2.drawMarker(bgr, tuple(np.rint(target).astype(int)), (40, 220, 40), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(bgr, tuple(np.rint(pred).astype(int)), 6, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(bgr, f"label=green x  pred=yellow  conf={conf_np[idx]:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
            record = dataset.records[int(batch["record_index"][idx])]
            cv2.imwrite(str(output / f"{saved:04d}_{record.episode}_frame{record.frame_id:06d}.jpg"), bgr)
            saved += 1


def save_checkpoint(path: Path, head: ProjectionHead, args: argparse.Namespace, metrics: dict[str, Any], epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "projection_head": head.state_dict(), "stage1_checkpoint": str(args.stage1_checkpoint), "epoch": int(epoch),
        "metrics": metrics, "labels": LABELS, "projection_head_type": "direct_pocket", "residual_scale": 0.0,
        "projection_model": "grasp_pocket_uv = sigmoid(MLP(mean(frozen_DINO_patch_tokens))) * image_size",
        "config": to_jsonable(vars(args)),
    }, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=WRIST_ROOT / "outputs" / "grasp_pocket_head")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--split-by", choices=("random", "episode", "source"), default="episode")
    parser.add_argument("--val-episode", action="append", default=[])
    parser.add_argument("--val-source", action="append", default=[])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-beta", type=float, default=4.0)
    parser.add_argument("--confidence-weight", type=float, default=0.10)
    parser.add_argument("--confidence-radius", type=float, default=24.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--num-overlays", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_annotation_records(args.annotations, labels=LABELS, min_visible=1)
    train_idx, val_idx, split = split_records(records, split_by=args.split_by, val_ratio=args.val_ratio, seed=args.seed, val_episodes=args.val_episode, val_sources=args.val_source)
    summary = {"all": summarize_records(records, LABELS), "train": summarize_records(records, LABELS, train_idx), "val": summarize_records(records, LABELS, val_idx), "split": split}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    set_seed(args.seed)
    device = resolve_device(args.device)
    precision = choose_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_dir = args.output_dir.expanduser().resolve() / (args.run_name or datetime.now().strftime("pocket_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "config.json", to_jsonable(vars(args)))
    atomic_write_json(run_dir / "split.json", to_jsonable(summary))
    stage1, _transform, cfg = load_stage2_model(checkpoint_path=args.stage1_checkpoint, repo_root=REPO_ROOT, device=device, config_path=args.config)
    freeze_stage1(stage1)
    image_size = int(cfg["data"]["image_size"])
    dataset = ProjectionAnnotationDataset(records, LABELS, image_size)
    train_loader = make_loader(dataset, train_idx, args.batch_size, True, args.num_workers)
    val_loader = make_loader(dataset, val_idx, args.batch_size, False, args.num_workers)
    head = ProjectionHead(feature_dim=int(cfg["model"].get("feature_dim", 384)), projection_head_type="direct_pocket", dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_metrics: dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        head.train(); total = loc_total = conf_total = 0.0; steps = 0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True); target = batch["target_uv"].to(device, non_blocking=True); valid = batch["valid"].to(device, non_blocking=True)
            (uv, confidence), _raw = forward_direct(stage1, head, image, image_size, device, precision)
            loss, loc, conf = point_loss(uv, confidence, target, valid, args.loss_beta, args.confidence_weight, args.confidence_radius)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip); optimizer.step()
            total += float(loss.item()); loc_total += float(loc.item()); conf_total += float(conf.item()); steps += 1
        val = evaluate(stage1, head, val_loader, image_size, device, precision, args)
        row = {"epoch": epoch, "train_loss": total / max(1, steps), "train_location_loss": loc_total / max(1, steps), "train_confidence_loss": conf_total / max(1, steps), "val_loss": val["loss"], "val_mean_px": val["mean_px"], "val_median_px": val["median_px"], "val_p90_px": val["p90_px"], "val_n": val["n_points"]}
        append_row(run_dir / "metrics.csv", row)
        print(f"epoch {epoch:04d}: train={row['train_loss']:.4f} val_median={row['val_median_px']:.3f}px val_p90={row['val_p90_px']:.3f}px n={row['val_n']}")
        save_checkpoint(run_dir / "checkpoints" / "last.pt", head, args, val, epoch)
        if float(val["median_px"]) < best:
            best, best_metrics = float(val["median_px"]), val
            save_checkpoint(run_dir / "checkpoints" / "best.pt", head, args, val, epoch)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(run_dir / "checkpoints" / f"epoch{epoch:04d}.pt", head, args, val, epoch)
    atomic_write_json(run_dir / "summary.json", {"best_median_px": best, "best_metrics": best_metrics})
    save_overlays(stage1, head, dataset, val_idx, run_dir / "overlays", image_size, device, precision, args.num_overlays)
    print(f"best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
