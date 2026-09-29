#!/usr/bin/env python3
"""
训练 wrist RGB -> 3D hand pose baseline。

任务
----
输入单帧 wrist RGB 图像，输出 MediaPipe 21 点中除 wrist 外的 20 个 3D 关键点。
第 0 个 wrist 点固定为 [0, 0, 0]，评估和可视化时再拼回完整 21 x 3。

模型结构
--------
  RGB 448x448
    -> DINOv3 ViT-S/16 backbone
    -> 28x28 patch tokens
    -> Attention Pooling
    -> MLP Head
    -> 20x3 joints

训练阶段
--------
Stage A:
  冻结 DINOv3 backbone，只训练 Attention Pooling + MLP Head。

Stage B:
  解冻 DINOv3 最后 4 个 transformer blocks，继续 fine-tune。

单卡 / CPU 用法
--------------
  python wrist/scripts/train.py \
      --config wrist/configs/baseline.yaml \
      --run-name debug_run

overfit sanity check：
  python wrist/scripts/train.py \
      --config wrist/configs/baseline.yaml \
      --run-name overfit_256 \
      --overfit-frames 256 \
      --epochs-stage-a 2 \
      --epochs-stage-b 0

8 卡 A100 DDP 用法
-----------------
  torchrun --standalone --nproc_per_node=8 \
      wrist/scripts/train.py \
      --config wrist/configs/a100_8gpu.yaml \
      --run-name dinov3_vits16_a100x8

覆盖每卡 batch size：
  torchrun --standalone --nproc_per_node=8 \
      wrist/scripts/train.py \
      --config wrist/configs/a100_8gpu.yaml \
      --run-name dinov3_vits16_a100x8_bs96 \
      --batch-size 96 \
      --num-workers 8

重要参数
--------
  --config            YAML 配置路径
  --run-name          输出 run 名称
  --overfit-frames    只取前 N 帧训练和验证，用于快速排查训练链路
  --epochs-stage-a    覆盖 Stage A epoch 数
  --epochs-stage-b    覆盖 Stage B epoch 数
  --batch-size        每张 GPU 的 batch size，不是总 batch
  --num-workers       每个进程的 DataLoader worker 数
  --grad-accum-steps  梯度累积步数

输出
----
  wrist/outputs/runs/<run_name>/checkpoints/best.pt
  wrist/outputs/runs/<run_name>/checkpoints/last.pt
  wrist/outputs/runs/<run_name>/metrics.csv
  wrist/outputs/runs/<run_name>/metrics.json

多卡行为
--------
  - 使用 torchrun / DDP
  - train/val 数据按 rank 分片
  - 验证指标通过 all-reduce 汇总
  - 只有 rank 0 写 checkpoint、metrics 和可视化
  - A100 推荐 bf16 + TF32 + channels-last
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

from wrist_pose.constants import JOINT_NAMES  # noqa: E402
from wrist_pose.dataset import WristPoseDataset  # noqa: E402
from wrist_pose.losses import WristPoseLoss  # noqa: E402
from wrist_pose.metrics import PoseMetricAccumulator  # noqa: E402
from wrist_pose.model import WristDinoPoseModel  # noqa: E402
from wrist_pose.utils import (  # noqa: E402
    WarmupCosineScheduler,
    append_csv,
    load_yaml,
    now_run_name,
    resolve_config_paths,
    save_json,
    set_seed,
    worker_init_fn,
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
    """Non-padding distributed eval sampler so validation metrics stay exact."""

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
    return DistEnv(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed(dist_env: DistEnv) -> None:
    if dist_env.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def print_main(dist_env: DistEnv, *args, **kwargs) -> None:
    if dist_env.is_main:
        print(*args, **kwargs)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(model: WristDinoPoseModel, dist_env: DistEnv):
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


def make_loader(
    index_path: Path,
    cfg: dict[str, Any],
    train: bool,
    dist_env: DistEnv,
    overfit_frames: int | None = None,
) -> DataLoader:
    ds = WristPoseDataset(
        index_path=index_path,
        image_size=int(cfg["data"]["image_size"]),
        train=train,
    )
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
    loader_kwargs = {
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
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 4))
    return DataLoader(
        ds,
        **loader_kwargs,
    )


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


def make_optimizer(model: WristDinoPoseModel, cfg: dict[str, Any], stage: str) -> torch.optim.Optimizer:
    weight_decay = float(cfg["train"].get("weight_decay", 0.05))
    if stage == "a":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["train"]["head_lr"]),
            weight_decay=weight_decay,
        )

    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = name.removeprefix("module.")
        if clean_name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    groups = [
        {"params": head_params, "lr": float(cfg["train"]["head_lr"])},
        {"params": backbone_params, "lr": float(cfg["train"]["backbone_lr"])},
    ]
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


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


def move_batch_to_device(batch: dict[str, Any], device: torch.device, channels_last: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = batch["image"].to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        image = image.contiguous(memory_format=torch.channels_last)
    gt = batch["joints20"].to(device, non_blocking=True)
    valid = batch["valid20"].to(device, non_blocking=True)
    return image, gt, valid


def reduce_loss_totals(totals: dict[str, float], steps: int, device: torch.device, dist_env: DistEnv) -> dict[str, float]:
    keys = sorted(totals)
    tensor = torch.tensor(
        [float(totals[key]) for key in keys] + [float(steps)],
        dtype=torch.float64,
        device=device,
    )
    if dist_env.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_steps = max(1.0, float(tensor[-1].item()))
    return {key: float(tensor[i].item() / global_steps) for i, key in enumerate(keys)}


def metric_state_tensor(metric: PoseMetricAccumulator, device: torch.device) -> torch.Tensor:
    values = [
        metric.total_error_m,
        float(metric.total_count),
        float(metric.pck10_count),
        float(metric.pck20_count),
        *metric.per_joint_error_m.tolist(),
        *metric.per_joint_count.astype("float64").tolist(),
        metric.bone_length_error_m,
        float(metric.bone_count),
        metric.pa_error_m,
        float(metric.pa_count),
    ]
    return torch.tensor(values, dtype=torch.float64, device=device)


def compute_metrics_from_state(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().cpu().numpy()
    idx = 0
    total_error_m = float(values[idx]); idx += 1
    total_count = int(round(float(values[idx]))); idx += 1
    pck10_count = int(round(float(values[idx]))); idx += 1
    pck20_count = int(round(float(values[idx]))); idx += 1
    per_joint_error_m = values[idx : idx + 21].astype("float64"); idx += 21
    per_joint_count = values[idx : idx + 21].astype("float64"); idx += 21
    bone_length_error_m = float(values[idx]); idx += 1
    bone_count = int(round(float(values[idx]))); idx += 1
    pa_error_m = float(values[idx]); idx += 1
    pa_count = int(round(float(values[idx]))); idx += 1

    def safe_div(num: float, denom: float) -> float:
        return float(num) / float(denom) if denom else float("nan")

    return {
        "mpjpe_mm": safe_div(total_error_m, total_count) * 1000.0,
        "pck_10mm": safe_div(pck10_count, total_count),
        "pck_20mm": safe_div(pck20_count, total_count),
        "bone_length_error_mm": safe_div(bone_length_error_m, bone_count) * 1000.0,
        "pa_mpjpe_mm": safe_div(pa_error_m, pa_count) * 1000.0,
        "num_valid_joints": total_count,
        "per_joint_mpjpe_mm": {
            name: safe_div(float(err), float(count)) * 1000.0
            for name, err, count in zip(JOINT_NAMES, per_joint_error_m, per_joint_count)
        },
    }


def train_one_epoch(
    model: WristDinoPoseModel,
    loader: DataLoader,
    criterion: WristPoseLoss,
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
    totals: dict[str, float] = {"loss": 0.0, "joint_loss": 0.0, "bone_length_loss": 0.0, "bone_direction_loss": 0.0}
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(1, int(grad_accum_steps))
    for batch_idx, batch in enumerate(loader):
        image, gt, valid = move_batch_to_device(batch, device, channels_last=channels_last)
        do_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(loader))
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and not do_step
            else contextlib.nullcontext()
        )
        with sync_context:
            with autocast_context(device, precision):
                pred = model(image)["joints20"]
                losses = criterion(pred, gt, valid)
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
    return reduce_loss_totals(totals, steps, device=device, dist_env=dist_env)


@torch.no_grad()
def evaluate(
    model: WristDinoPoseModel,
    loader: DataLoader,
    criterion: WristPoseLoss,
    device: torch.device,
    precision: str,
    dist_env: DistEnv,
    channels_last: bool,
    capture_visuals: bool,
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any] | None]:
    model.eval()
    metric = PoseMetricAccumulator()
    totals: dict[str, float] = {"loss": 0.0, "joint_loss": 0.0, "bone_length_loss": 0.0, "bone_direction_loss": 0.0}
    steps = 0
    first_batch = None
    for batch in loader:
        image, gt, valid = move_batch_to_device(batch, device, channels_last=channels_last)
        with autocast_context(device, precision):
            outputs = model(image, return_attention=capture_visuals and first_batch is None)
            pred = outputs["joints20"]
            losses = criterion(pred, gt, valid)
        metric.update(pred, gt, valid)
        for key in totals:
            totals[key] += float(losses[key].detach().item())
        steps += 1
        if capture_visuals and first_batch is None:
            first_batch = {
                "batch": {k: v.cpu() if torch.is_tensor(v) else v for k, v in batch.items()},
                "pred20": pred.detach().cpu(),
                "attention": outputs.get("attention", None).detach().cpu() if outputs.get("attention", None) is not None else None,
            }
    loss_metrics = reduce_loss_totals(totals, steps, device=device, dist_env=dist_env)
    metric_tensor = metric_state_tensor(metric, device=device)
    if dist_env.distributed:
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
    return compute_metrics_from_state(metric_tensor), loss_metrics, first_batch


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


def maybe_save_visuals(
    vis_batch: dict[str, Any],
    run_dir: Path,
    epoch: int,
    cfg: dict[str, Any],
) -> None:
    try:
        from wrist_pose.visualize import save_batch_visuals
    except ImportError as exc:
        print(f"visualization skipped: {exc}")
        return
    save_batch_visuals(
        vis_batch["batch"],
        vis_batch["pred20"],
        run_dir / "visuals" / f"epoch_{epoch:04d}",
        prefix="val",
        attention=vis_batch["attention"],
        max_items=int(cfg["eval"].get("num_visuals", 8)),
        image_size=int(cfg["data"]["image_size"]),
    )


def run_stage(
    stage: str,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: WristPoseLoss,
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
    else:
        n = raw_model.unfreeze_backbone_last_blocks(int(cfg["train"].get("unfreeze_last_blocks", 4)))
        print_main(dist_env, f"Stage B: unfroze {n} backbone blocks")
    model = wrap_ddp(raw_model, dist_env)

    optimizer = make_optimizer(model, cfg, stage=stage)
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    total_steps = max(1, math.ceil(len(train_loader) / grad_accum_steps) * epochs)
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 5))
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=min(total_steps, max(0, warmup_epochs * math.ceil(len(train_loader) / grad_accum_steps))),
    )
    use_scaler = device.type == "cuda" and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))
    channels_last = bool(cfg["train"].get("channels_last", True))

    for local_epoch in range(epochs):
        epoch = start_epoch + 1
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            precision=precision,
            grad_clip=grad_clip,
            scaler=scaler,
            dist_env=dist_env,
            channels_last=channels_last,
            grad_accum_steps=grad_accum_steps,
        )
        val_metrics, val_losses, vis_batch = evaluate(
            model,
            val_loader,
            criterion,
            device,
            precision,
            dist_env=dist_env,
            channels_last=channels_last,
            capture_visuals=dist_env.is_main,
        )
        row = {
            "epoch": epoch,
            "stage": stage,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_losses.items()},
            "val_mpjpe_mm": val_metrics["mpjpe_mm"],
            "val_pck_10mm": val_metrics["pck_10mm"],
            "val_pck_20mm": val_metrics["pck_20mm"],
            "val_bone_length_error_mm": val_metrics["bone_length_error_mm"],
            "val_pa_mpjpe_mm": val_metrics["pa_mpjpe_mm"],
        }
        if dist_env.is_main:
            append_csv(run_dir / "metrics.csv", row)
            save_json(run_dir / "metrics.json", {"last": row, "val": val_metrics})
            save_checkpoint(run_dir / "checkpoints" / "last.pt", model, cfg, epoch, stage, val_metrics)

        if dist_env.is_main and val_metrics["mpjpe_mm"] < best_mpjpe:
            best_mpjpe = float(val_metrics["mpjpe_mm"])
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, cfg, epoch, stage, val_metrics)

        if dist_env.is_main and vis_batch is not None and int(cfg["eval"].get("save_visuals_every", 1)) > 0:
            if epoch % int(cfg["eval"].get("save_visuals_every", 1)) == 0:
                maybe_save_visuals(vis_batch, run_dir, epoch, cfg)

        print_main(
            dist_env,
            f"epoch={epoch} stage={stage} "
            f"train_loss={train_losses['loss']:.6f} "
            f"val_mpjpe={val_metrics['mpjpe_mm']:.2f}mm "
            f"pck10={val_metrics['pck_10mm']:.3f} pck20={val_metrics['pck_20mm']:.3f} "
            f"world_size={dist_env.world_size}"
        )
        start_epoch = epoch
    return model, start_epoch, best_mpjpe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=WRIST_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overfit-frames", type=int, default=None)
    parser.add_argument("--epochs-stage-a", type=int, default=None)
    parser.add_argument("--epochs-stage-b", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-GPU batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per process.")
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    args = parser.parse_args()

    dist_env = setup_distributed()
    cfg = load_yaml(args.config)
    resolve_config_paths(cfg, REPO_ROOT)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = int(args.num_workers)
    if args.grad_accum_steps is not None:
        cfg["train"]["grad_accum_steps"] = int(args.grad_accum_steps)
    configure_torch_for_gpu(cfg, dist_env)
    set_seed(int(cfg.get("seed", 42)) + dist_env.rank)
    if args.epochs_stage_a is not None:
        cfg["train"]["epochs_stage_a"] = int(args.epochs_stage_a)
    if args.epochs_stage_b is not None:
        cfg["train"]["epochs_stage_b"] = int(args.epochs_stage_b)

    run_name = args.run_name or now_run_name()
    if dist_env.distributed:
        payload = [run_name if dist_env.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        run_name = payload[0]
    run_dir = Path(cfg["output"]["run_dir"]) / run_name
    if dist_env.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "config.resolved.json", cfg)
    if dist_env.distributed:
        dist.barrier()

    device = dist_env.device
    precision = choose_precision(str(cfg["train"].get("precision", "auto")))
    effective_batch = (
        int(cfg["train"]["batch_size"])
        * dist_env.world_size
        * max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    )
    print_main(
        dist_env,
        f"device={device} precision={precision} run_dir={run_dir} "
        f"world_size={dist_env.world_size} per_gpu_batch={cfg['train']['batch_size']} "
        f"effective_batch={effective_batch}",
    )

    train_index = Path(cfg["data"]["index_dir"]) / "train.jsonl"
    val_index = Path(cfg["data"]["index_dir"]) / "val.jsonl"
    train_loader = make_loader(train_index, cfg, train=True, dist_env=dist_env, overfit_frames=args.overfit_frames)
    val_loader = make_loader(
        train_index if args.overfit_frames is not None else val_index,
        cfg,
        train=False,
        dist_env=dist_env,
        overfit_frames=args.overfit_frames,
    )
    print_main(dist_env, f"train_batches_per_rank={len(train_loader)} val_batches_per_rank={len(val_loader)}")

    model = make_model(cfg, device)
    if bool(cfg["train"].get("channels_last", True)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = WristPoseLoss(
        joint_beta=float(cfg["loss"]["joint_beta"]),
        bone_length_weight=float(cfg["loss"]["bone_length_weight"]),
        bone_direction_weight=float(cfg["loss"]["bone_direction_weight"]),
    ).to(device)

    model, epoch, best = run_stage(
        "a",
        model,
        train_loader,
        val_loader,
        criterion,
        cfg,
        run_dir,
        device,
        precision,
        start_epoch=0,
        epochs=int(cfg["train"]["epochs_stage_a"]),
        best_mpjpe=float("inf"),
        dist_env=dist_env,
    )
    model, epoch, best = run_stage(
        "b",
        model,
        train_loader,
        val_loader,
        criterion,
        cfg,
        run_dir,
        device,
        precision,
        start_epoch=epoch,
        epochs=int(cfg["train"]["epochs_stage_b"]),
        best_mpjpe=best,
        dist_env=dist_env,
    )
    print_main(dist_env, json.dumps({"run_dir": str(run_dir), "best_mpjpe_mm": best}, indent=2))
    cleanup_distributed(dist_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
