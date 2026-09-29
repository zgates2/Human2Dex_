#!/usr/bin/env python3
"""
训练 Stage2 wrist RGB -> MANO pose -> 21 joints baseline。

常用命令
--------
overfit sanity check：
  python wrist/scripts/train_stage2.py \
      --config wrist/configs/stage2_mano.yaml \
      --run-name stage2_overfit_128 \
      --overfit-frames 128 \
      --one-stage \
      --epochs 2

正式训练：
/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  wrist/scripts/train_stage2.py \
  --config wrist/configs/stage2_mano.yaml \
  --run-name stage2_mano_onestage_v1 \
  --one-stage \
  --epochs 130 \
  --batch-size 64 \
  --num-workers 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.mano_layer import ManoDependencyError  # noqa: E402
from wrist_pose.stage2_dataset import Stage2ManoDataset  # noqa: E402
from wrist_pose.stage2_losses import Stage2ManoLoss  # noqa: E402
from wrist_pose.stage2_metrics import (  # noqa: E402
    Joint21MetricAccumulator,
    compute_metrics_from_state,
    metric_state_tensor,
)
from wrist_pose.stage2_model import WristDinoManoPoseModel  # noqa: E402
from wrist_pose.utils import (  # noqa: E402
    WarmupCosineScheduler,
    append_csv,
    load_yaml,
    now_run_name,
    read_jsonl,
    resolve_config_paths,
    save_json,
    set_seed,
    worker_init_fn,
    write_jsonl,
)


@dataclass(frozen=True)
class DistEnv:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class ShardedEvalSampler(Sampler[int]):
    def __init__(self, dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if len(self.dataset) <= self.rank:
            return 0
        return int(math.ceil((len(self.dataset) - self.rank) / self.world_size))


def setup_distributed() -> DistEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistEnv(distributed, rank, local_rank, world_size, device)


def cleanup_distributed(dist_env: DistEnv) -> None:
    if dist_env.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def print_main(dist_env: DistEnv, *args, **kwargs) -> None:
    if dist_env.is_main:
        print(*args, **kwargs)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(model: WristDinoManoPoseModel, dist_env: DistEnv):
    if not dist_env.distributed:
        return model
    return DDP(
        model,
        device_ids=[dist_env.local_rank],
        output_device=dist_env.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def configure_torch_for_gpu(cfg: dict[str, Any], dist_env: DistEnv) -> None:
    if dist_env.device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
    torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))
    try:
        torch.set_float32_matmul_precision(str(cfg["train"].get("matmul_precision", "high")))
    except Exception:
        pass


def split_fit_rows(fits_path: Path, output_dir: Path, seed: int) -> dict[str, Path]:
    rows = read_jsonl(fits_path)
    rows = [row for row in rows if bool(row.get("fit_valid", True))]
    if not rows:
        raise RuntimeError(f"no fit_valid rows in {fits_path}")
    by_seq: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_seq.setdefault(str(row.get("sequence_id", "")), []).append(row)
    seqs = sorted(by_seq)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(seqs), generator=g).tolist()
    seqs = [seqs[i] for i in perm]
    n = len(seqs)
    n_train = max(1, int(round(n * 0.8)))
    n_val = max(1, int(round(n * 0.1))) if n >= 3 else 0
    if n_train + n_val >= n and n >= 3:
        n_train = n - 2
        n_val = 1
    splits = {
        "train": seqs[:n_train],
        "val": seqs[n_train : n_train + n_val],
        "test": seqs[n_train + n_val :],
    }
    if not splits["val"]:
        splits["val"] = splits["train"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, split_seqs in splits.items():
        split_rows = [row for seq in split_seqs for row in by_seq[seq]]
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, split_rows)
        paths[name] = path
    return paths


def compute_mean_beta(index_path: Path) -> torch.Tensor:
    rows = read_jsonl(index_path)
    betas = []
    for row in rows:
        beta = row.get("mano_beta")
        if beta is None:
            continue
        tensor = torch.as_tensor(beta, dtype=torch.float32).reshape(-1)
        if tensor.numel() == 10 and torch.isfinite(tensor).all():
            betas.append(tensor)
    if not betas:
        return torch.zeros(10, dtype=torch.float32)
    return torch.stack(betas, dim=0).mean(dim=0)


def make_loader(
    index_path: Path,
    cfg: dict[str, Any],
    train: bool,
    dist_env: DistEnv,
    overfit_frames: int | None = None,
) -> DataLoader:
    ds = Stage2ManoDataset(index_path=index_path, image_size=int(cfg["data"]["image_size"]), train=train)
    if overfit_frames is not None:
        count = min(int(overfit_frames), len(ds))
        ds = Subset(ds, list(range(count)))
    batch_size = int(cfg["train"]["batch_size"])
    drop_last = bool(train and overfit_frames is None and len(ds) >= batch_size * max(1, dist_env.world_size))
    sampler = None
    if dist_env.distributed:
        if train:
            sampler = DistributedSampler(
                ds,
                num_replicas=dist_env.world_size,
                rank=dist_env.rank,
                shuffle=True,
                drop_last=drop_last,
            )
        else:
            sampler = ShardedEvalSampler(ds, rank=dist_env.rank, world_size=dist_env.world_size)
    num_workers = int(cfg["train"].get("num_workers", 4))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": bool(train and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": bool(num_workers > 0 and cfg["train"].get("persistent_workers", True)),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 4))
    return DataLoader(ds, **kwargs)


def make_model(cfg: dict[str, Any], device: torch.device) -> WristDinoManoPoseModel:
    mano = cfg["mano"]
    try:
        model = WristDinoManoPoseModel(
            dino_dir=Path(cfg["model"]["dino_dir"]),
            mano_model_dir=Path(mano["model_dir"]),
            image_size=int(cfg["data"]["image_size"]),
            feature_dim=int(cfg["model"].get("feature_dim", 384)),
            patch_size=int(cfg["model"].get("patch_size", 16)),
            head_dropout=float(cfg["model"].get("head_dropout", 0.1)),
            decoder_layers=int(cfg["model"].get("decoder_layers", 2)),
            decoder_heads=int(cfg["model"].get("decoder_heads", 6)),
            mano_side=str(mano.get("side", "right")),
            mano_use_pca=bool(mano.get("use_pca", False)),
            mano_flat_hand_mean=bool(mano.get("flat_hand_mean", False)),
            mano_output_scale=float(mano.get("output_scale", 1.0)),
        )
    except ManoDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    return model.to(device)


def make_optimizer(model: WristDinoManoPoseModel, cfg: dict[str, Any], stage: str) -> torch.optim.Optimizer:
    weight_decay = float(cfg["train"].get("weight_decay", 0.05))
    if stage == "a":
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["train"]["head_lr"]), weight_decay=weight_decay)
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name.removeprefix("module.")
        if clean.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    groups = [
        {"params": head_params, "lr": float(cfg["train"]["head_lr"])},
        {"params": backbone_params, "lr": float(cfg["train"]["backbone_lr"])},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=weight_decay)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def choose_precision(requested: str) -> str:
    if requested == "fp32" or not torch.cuda.is_available():
        return "fp32"
    if requested == "bf16":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp16"


def move_batch(batch: dict[str, Any], device: torch.device, channels_last: bool) -> dict[str, Any]:
    image = batch["image"].to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        image = image.contiguous(memory_format=torch.channels_last)
    return {
        "image": image,
        "joints21": batch["joints21"].to(device, non_blocking=True),
        "valid21": batch["valid21"].to(device, non_blocking=True),
        "mano_pose": batch["mano_pose"].to(device, non_blocking=True),
        "global_orient": batch["global_orient"].to(device, non_blocking=True),
        "mano_beta": batch["mano_beta"].to(device, non_blocking=True),
    }


def reduce_loss_totals(totals: dict[str, float], steps: int, device: torch.device, dist_env: DistEnv) -> dict[str, float]:
    keys = sorted(totals)
    tensor = torch.tensor([float(totals[k]) for k in keys] + [float(steps)], dtype=torch.float64, device=device)
    if dist_env.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_steps = max(1.0, float(tensor[-1].item()))
    return {key: float(tensor[i].item() / global_steps) for i, key in enumerate(keys)}


def train_one_epoch(
    model,
    loader: DataLoader,
    criterion: Stage2ManoLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    device: torch.device,
    precision: str,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler | None,
    dist_env: DistEnv,
    channels_last: bool,
    grad_accum_steps: int,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "pose_loss": 0.0, "joint_loss": 0.0, "bone_loss": 0.0, "pose_prior": 0.0}
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(1, int(grad_accum_steps))
    for batch_idx, raw_batch in enumerate(loader):
        batch = move_batch(raw_batch, device, channels_last)
        do_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))
        sync_context = model.no_sync() if isinstance(model, DDP) and not do_step else contextlib.nullcontext()
        with sync_context:
            with autocast_context(device, precision):
                outputs = model(batch["image"], mano_beta=batch["mano_beta"])
                losses = criterion(
                    outputs,
                    batch["mano_pose"],
                    batch["joints21"],
                    batch["valid21"],
                    gt_global_orient=batch["global_orient"],
                )
                loss = losses["loss"] / float(grad_accum_steps)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if do_step:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += float(losses[key].detach().item())
        steps += 1
    return reduce_loss_totals(totals, steps, device, dist_env)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    criterion: Stage2ManoLoss,
    device: torch.device,
    precision: str,
    dist_env: DistEnv,
    channels_last: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    metric = Joint21MetricAccumulator()
    totals = {"loss": 0.0, "pose_loss": 0.0, "joint_loss": 0.0, "bone_loss": 0.0, "pose_prior": 0.0}
    steps = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device, channels_last)
        with autocast_context(device, precision):
            outputs = model(batch["image"], mano_beta=batch["mano_beta"])
            losses = criterion(
                outputs,
                batch["mano_pose"],
                batch["joints21"],
                batch["valid21"],
                gt_global_orient=batch["global_orient"],
            )
        metric.update(outputs["joints21"], batch["joints21"], batch["valid21"])
        for key in totals:
            totals[key] += float(losses[key].detach().item())
        steps += 1
    loss_metrics = reduce_loss_totals(totals, steps, device, dist_env)
    state = metric_state_tensor(metric, device)
    if dist_env.distributed:
        dist.all_reduce(state, op=dist.ReduceOp.SUM)
    return compute_metrics_from_state(state), loss_metrics


def save_checkpoint(path: Path, model, cfg: dict[str, Any], epoch: int, stage: str, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "config": cfg,
            "epoch": epoch,
            "stage": stage,
            "metrics": metrics,
        },
        path,
    )


def run_stage(
    stage: str,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Stage2ManoLoss,
    cfg: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    precision: str,
    start_epoch: int,
    epochs: int,
    best_mpjpe: float,
    dist_env: DistEnv,
) -> tuple[Any, int, float]:
    if epochs <= 0:
        return model, start_epoch, best_mpjpe
    raw_model = unwrap_model(model)
    if stage == "a":
        raw_model.freeze_backbone()
        print_main(dist_env, "Stage A: backbone frozen")
    elif stage in {"one_stage", "one"}:
        n = raw_model.unfreeze_backbone_last_blocks(int(cfg["train"].get("unfreeze_last_blocks", 4)))
        print_main(dist_env, f"One-stage: unfroze {n} backbone blocks from epoch 1")
    else:
        n = raw_model.unfreeze_backbone_last_blocks(int(cfg["train"].get("unfreeze_last_blocks", 4)))
        print_main(dist_env, f"Stage B: unfroze {n} backbone blocks")
    model = wrap_ddp(raw_model, dist_env)
    optimizer = make_optimizer(model, cfg, stage)
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    total_steps = max(1, math.ceil(len(train_loader) / grad_accum_steps) * epochs)
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 5))
    scheduler = WarmupCosineScheduler(optimizer, total_steps=total_steps, warmup_steps=min(total_steps, warmup_epochs * math.ceil(len(train_loader) / grad_accum_steps)))
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    channels_last = bool(cfg["train"].get("channels_last", True))

    for _ in range(epochs):
        epoch = start_epoch + 1
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_losses = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, precision, grad_clip, scaler, dist_env, channels_last, grad_accum_steps)
        val_metrics, val_losses = evaluate(model, val_loader, criterion, device, precision, dist_env, channels_last)
        row = {
            "epoch": epoch,
            "stage": stage,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_losses.items()},
            "val_mpjpe_mm": val_metrics["mpjpe_mm"],
            "val_pck_10mm": val_metrics["pck_10mm"],
            "val_pck_20mm": val_metrics["pck_20mm"],
            "val_fingertip_mpjpe_mm": val_metrics["fingertip_mpjpe_mm"],
            "val_bone_length_error_mm": val_metrics["bone_length_error_mm"],
        }
        if dist_env.is_main:
            append_csv(run_dir / "metrics.csv", row)
            save_json(run_dir / "metrics.json", {"last": row, "val": val_metrics})
            save_checkpoint(run_dir / "checkpoints" / "last.pt", model, cfg, epoch, stage, val_metrics)
            if val_metrics["mpjpe_mm"] < best_mpjpe:
                best_mpjpe = float(val_metrics["mpjpe_mm"])
                save_checkpoint(run_dir / "checkpoints" / "best.pt", model, cfg, epoch, stage, val_metrics)
        print_main(
            dist_env,
            f"epoch={epoch} stage={stage} train_loss={train_losses['loss']:.6f} "
            f"val_mpjpe={val_metrics['mpjpe_mm']:.2f}mm fingertip={val_metrics['fingertip_mpjpe_mm']:.2f}mm",
        )
        start_epoch = epoch
    return model, start_epoch, best_mpjpe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=WRIST_ROOT / "configs" / "stage2_mano.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overfit-frames", type=int, default=None)
    parser.add_argument("--training-mode", choices=("one_stage", "two_stage"), default=None)
    parser.add_argument("--one-stage", action="store_true", help="Shortcut for --training-mode one_stage.")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs for one-stage training.")
    parser.add_argument("--epochs-stage-a", type=int, default=None)
    parser.add_argument("--epochs-stage-b", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    args = parser.parse_args()

    dist_env = setup_distributed()
    cfg = load_yaml(args.config)
    resolve_config_paths(cfg, REPO_ROOT)
    if args.one_stage:
        cfg["train"]["training_mode"] = "one_stage"
    if args.training_mode is not None:
        cfg["train"]["training_mode"] = args.training_mode
    if args.epochs is not None:
        cfg["train"]["epochs_one_stage"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = int(args.num_workers)
    if args.grad_accum_steps is not None:
        cfg["train"]["grad_accum_steps"] = int(args.grad_accum_steps)
    if args.epochs_stage_a is not None:
        cfg["train"]["epochs_stage_a"] = int(args.epochs_stage_a)
    if args.epochs_stage_b is not None:
        cfg["train"]["epochs_stage_b"] = int(args.epochs_stage_b)
    training_mode = str(cfg["train"].get("training_mode", "two_stage"))
    if training_mode not in {"one_stage", "two_stage"}:
        raise SystemExit(f"Unsupported train.training_mode={training_mode!r}; expected one_stage or two_stage")
    configure_torch_for_gpu(cfg, dist_env)
    set_seed(int(cfg.get("seed", 42)) + dist_env.rank)

    run_name = args.run_name or now_run_name("stage2_mano")
    if dist_env.distributed:
        payload = [run_name if dist_env.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        run_name = payload[0]
    run_dir = Path(cfg["output"]["run_dir"]) / run_name
    split_dir = run_dir / "splits"
    if dist_env.is_main:
        paths = split_fit_rows(Path(cfg["data"]["stage2_fits"]), split_dir, seed=int(cfg.get("seed", 42)))
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "config.resolved.json", cfg)
        save_json(run_dir / "splits.json", {k: str(v) for k, v in paths.items()})
    if dist_env.distributed:
        dist.barrier()
    paths = {"train": split_dir / "train.jsonl", "val": split_dir / "val.jsonl"}

    device = dist_env.device
    precision = choose_precision(str(cfg["train"].get("precision", "auto")))
    print_main(dist_env, f"device={device} precision={precision} run_dir={run_dir}")
    train_loader = make_loader(paths["train"], cfg, train=True, dist_env=dist_env, overfit_frames=args.overfit_frames)
    val_loader = make_loader(paths["train"] if args.overfit_frames is not None else paths["val"], cfg, train=False, dist_env=dist_env, overfit_frames=args.overfit_frames)
    print_main(dist_env, f"train_batches_per_rank={len(train_loader)} val_batches_per_rank={len(val_loader)}")

    model = make_model(cfg, device)
    mean_beta = compute_mean_beta(paths["train"]).to(device)
    model.set_default_beta(mean_beta)
    print_main(dist_env, f"default_beta(mean train beta)={mean_beta.detach().cpu().tolist()}")
    if bool(cfg["train"].get("channels_last", True)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = Stage2ManoLoss(
        pose_weight=float(cfg["loss"].get("pose_weight", 1.0)),
        joint_weight=float(cfg["loss"].get("joint_weight", 1.0)),
        bone_weight=float(cfg["loss"].get("bone_weight", 0.1)),
        pose_prior_weight=float(cfg["loss"].get("pose_prior_weight", 0.01)),
        joint_beta=float(cfg["loss"].get("joint_beta", 0.005)),
    ).to(device)
    print_main(dist_env, f"training_mode={training_mode}")
    if training_mode == "one_stage":
        default_epochs = int(cfg["train"].get("epochs_stage_a", 0)) + int(cfg["train"].get("epochs_stage_b", 0))
        epochs_one_stage = int(cfg["train"].get("epochs_one_stage", default_epochs))
        model, epoch, best = run_stage(
            "one_stage",
            model,
            train_loader,
            val_loader,
            criterion,
            cfg,
            run_dir,
            device,
            precision,
            0,
            epochs_one_stage,
            float("inf"),
            dist_env,
        )
    else:
        model, epoch, best = run_stage("a", model, train_loader, val_loader, criterion, cfg, run_dir, device, precision, 0, int(cfg["train"]["epochs_stage_a"]), float("inf"), dist_env)
        model, epoch, best = run_stage("b", model, train_loader, val_loader, criterion, cfg, run_dir, device, precision, epoch, int(cfg["train"]["epochs_stage_b"]), best, dist_env)
    print_main(dist_env, json.dumps({"run_dir": str(run_dir), "best_mpjpe_mm": best}, indent=2))
    cleanup_distributed(dist_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
