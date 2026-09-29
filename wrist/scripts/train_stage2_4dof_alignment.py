#!/usr/bin/env python3
"""Train a 4DoF 2D alignment head on top of an existing Stage2 wrist model.

This script consumes the JSON produced by
`tools/click_label_visible_hand_points.py` and learns where the Stage2
wrist-local MANO skeleton should be placed in the image.

It does not train a diffusion policy and does not modify PKL/Zarr data.

Default model:
  RGB -> frozen Stage2 DINO/MANO -> joints21 + patch tokens
  patch tokens -> small MLP -> theta, scale, tx, ty
  joints21 hand-local xy -> 4DoF projection -> image uv

Supervision:
  visible palm_center + five fingertips from click annotations.

/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python \
  tools/click_label_visible_hand_points.py \
  --input /share/project/liyuanyuan/data/dexglove_data/annotations/selected_images.txt \
  --output /share/project/liyuanyuan/data/dexglove_data/annotations/visible_hand_points_v1.json \
  --host 127.0.0.1 \
  --port 8899

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.constants import HAND_BONES  # noqa: E402
from wrist_pose.stage2_infer import load_stage2_model  # noqa: E402
from wrist_pose.transforms import WristImageTransform, read_rgb  # noqa: E402


DEFAULT_CHECKPOINT = (
    WRIST_ROOT
    / "outputs"
    / "test_3"
    / "runs"
    / "stage2_test_3"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_LABELS = (
    "palm_center",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)
TIP_LABEL_TO_JOINT = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
PALM_JOINTS = (0, 1, 5, 9, 13, 17)
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
POINT_COLORS_BGR = {
    "palm_center": (80, 80, 255),
    "thumb_tip": (80, 180, 255),
    "index_tip": (80, 255, 120),
    "middle_tip": (255, 210, 80),
    "ring_tip": (255, 120, 180),
    "pinky_tip": (180, 120, 255),
}
JOINT_COLORS_BGR = [
    (245, 245, 245),
    *((80, 180, 255) for _ in range(4)),
    *((80, 255, 120) for _ in range(4)),
    *((255, 210, 80) for _ in range(4)),
    *((255, 120, 180) for _ in range(4)),
    *((180, 120, 255) for _ in range(4)),
]


@dataclass(frozen=True)
class AnnotationRecord:
    frame_id: int
    episode: str
    image: Path
    image_rel: str
    dataset: str
    source_group: str
    annotation_file: Path
    target_uv_original: np.ndarray
    valid: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def parse_labels(value: str | None) -> list[str]:
    if value is None or value == "":
        return list(DEFAULT_LABELS)
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("labels cannot be empty")
    missing = [label for label in labels if label not in DEFAULT_LABELS]
    if missing:
        raise argparse.ArgumentTypeError(
            f"unsupported labels: {missing}; expected subset/order of {DEFAULT_LABELS}"
        )
    return labels


def parse_source_axes(value: str) -> tuple[int, int]:
    parts = [item.strip().lower() for item in value.replace(",", " ").split() if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--source-axes expects two axes, e.g. x,y or x,z")
    try:
        axes = tuple(AXIS_TO_INDEX[item] for item in parts)
    except KeyError as exc:
        raise argparse.ArgumentTypeError("axes must be selected from x,y,z") from exc
    if axes[0] == axes[1]:
        raise argparse.ArgumentTypeError("source axes must be different")
    return axes  # type: ignore[return-value]


def parse_source_signs(value: str) -> tuple[float, float]:
    parts = [item.strip().lower() for item in value.replace(",", " ").split() if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--source-signs expects two signs, e.g. 1,1 or -1,1")
    signs = []
    for item in parts:
        if item in {"1", "+1", "pos", "+"}:
            signs.append(1.0)
        elif item in {"-1", "neg", "-"}:
            signs.append(-1.0)
        else:
            raise argparse.ArgumentTypeError("--source-signs values must be 1 or -1")
    return signs[0], signs[1]


def parse_csv_values(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def infer_dataset_key(image: Path, episode: str, annotation_input: str | None) -> str:
    parts = list(image.parts)
    if episode in parts:
        idx = parts.index(episode)
        if idx >= 1:
            parent = parts[idx - 1]
            if parent not in {"images", "calibration_episodes", "calibration", "episodes"}:
                return parent
            if idx >= 2:
                return parts[idx - 2]
    if annotation_input:
        input_path = Path(annotation_input).expanduser()
        if input_path.name:
            return input_path.name
    if image.parent.name == "images" and image.parent.parent.name:
        return image.parent.parent.name
    return image.parent.name


def infer_source_group(
    *,
    image: Path,
    episode: str,
    dataset: str,
    raw: dict[str, Any],
    payload: dict[str, Any],
) -> str:
    for key in ("source_group", "mount_id", "camera_mount_id", "subject_id", "operator_id"):
        value = raw.get(key, payload.get(key))
        if value not in (None, ""):
            return str(value)
    annotation_input = payload.get("input")
    source = infer_dataset_key(
        image=image,
        episode=episode,
        annotation_input=str(annotation_input) if annotation_input not in (None, "") else None,
    )
    return f"{source}/{episode}"


def load_annotation_records(
    annotations_paths: Iterable[Path],
    labels: list[str],
    min_visible: int,
) -> list[AnnotationRecord]:
    records_by_image: dict[str, AnnotationRecord] = {}
    records: list[AnnotationRecord] = []
    for annotations_path in annotations_paths:
        annotations_path = annotations_path.expanduser().resolve()
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise ValueError(f"annotation JSON missing frames list: {annotations_path}")
        annotation_input = payload.get("input")

        for raw in frames:
            if not isinstance(raw, dict):
                continue
            points = raw.get("points")
            if not isinstance(points, dict):
                points = {}
            uv = np.zeros((len(labels), 2), dtype=np.float32)
            valid = np.zeros((len(labels),), dtype=bool)
            for idx, label in enumerate(labels):
                point = points.get(label)
                if not isinstance(point, dict) or not bool(point.get("visible", False)):
                    continue
                x = point.get("x")
                y = point.get("y")
                if x is None or y is None:
                    continue
                x_f = float(x)
                y_f = float(y)
                if not (np.isfinite(x_f) and np.isfinite(y_f)):
                    continue
                uv[idx] = (x_f, y_f)
                valid[idx] = True
            if int(valid.sum()) < int(min_visible):
                continue
            image = Path(str(raw.get("image", ""))).expanduser()
            if not image.is_file():
                raise FileNotFoundError(f"annotated image does not exist: {image}")
            image = image.resolve()
            episode = str(raw.get("episode", image.parent.parent.name))
            dataset = str(
                raw.get("dataset")
                or payload.get("dataset")
                or infer_dataset_key(
                    image=image,
                    episode=episode,
                    annotation_input=str(annotation_input) if annotation_input not in (None, "") else None,
                )
            )
            source_group = infer_source_group(
                image=image,
                episode=episode,
                dataset=dataset,
                raw=raw,
                payload=payload,
            )
            record = AnnotationRecord(
                frame_id=int(raw.get("frame_id", len(records))),
                episode=episode,
                image=image,
                image_rel=str(raw.get("image_rel", image.name)),
                dataset=dataset,
                source_group=source_group,
                annotation_file=annotations_path,
                target_uv_original=uv,
                valid=valid,
            )
            records_by_image[str(image)] = record

    records = list(records_by_image.values())
    if not records:
        raise ValueError(
            "no usable annotated frames; "
            f"min_visible={min_visible}, labels={labels}"
        )
    return records


class AlignmentAnnotationDataset(Dataset):
    def __init__(
        self,
        records: list[AnnotationRecord],
        labels: list[str],
        image_size: int,
    ) -> None:
        self.records = list(records)
        self.labels = list(labels)
        self.transform = WristImageTransform(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        rgb = read_rgb(str(record.image))
        result = self.transform(rgb)
        target = record.target_uv_original.copy()
        valid = record.valid.copy()
        scale = float(result.scale)
        offset_x, offset_y = result.offset_xy
        target_transformed = target.copy()
        target_transformed[:, 0] = target_transformed[:, 0] * scale + float(offset_x)
        target_transformed[:, 1] = target_transformed[:, 1] * scale + float(offset_y)
        return {
            "image": result.image,
            "target_uv": torch.from_numpy(target_transformed).float(),
            "valid": torch.from_numpy(valid).bool(),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "offset_xy": torch.tensor([offset_x, offset_y], dtype=torch.float32),
            "original_hw": torch.tensor(result.original_hw, dtype=torch.float32),
            "record_index": torch.tensor(idx, dtype=torch.long),
            "image_path": str(record.image),
            "image_rel": record.image_rel,
            "episode": record.episode,
            "dataset": record.dataset,
            "source_group": record.source_group,
        }


class Alignment4DoFHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        init_scale: float = 1800.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.zero_()
            final.bias[1] = math.log(max(float(init_scale), 1e-3))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        features = patch_tokens.mean(dim=1)
        return self.net(features)


def decode_4dof(
    raw: torch.Tensor,
    image_size: int,
    max_theta_rad: float,
    min_scale: float,
    max_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = float(max_theta_rad) * torch.tanh(raw[:, 0])
    log_scale = raw[:, 1].clamp(math.log(float(min_scale)), math.log(float(max_scale)))
    scale = torch.exp(log_scale)
    tx = float(image_size) * torch.sigmoid(raw[:, 2])
    ty = float(image_size) * torch.sigmoid(raw[:, 3])
    translation = torch.stack([tx, ty], dim=-1)
    return theta, scale, translation


def apply_4dof(
    points2d: torch.Tensor,
    theta: torch.Tensor,
    scale: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    rot = torch.stack(
        [
            torch.stack([cos_t, -sin_t], dim=-1),
            torch.stack([sin_t, cos_t], dim=-1),
        ],
        dim=-2,
    )
    rotated = torch.matmul(points2d, rot.transpose(-1, -2))
    return rotated * scale[:, None, None] + translation[:, None, :]


def make_alignment_anchors(
    joints21: torch.Tensor,
    labels: list[str],
    source_axes: tuple[int, int],
    source_signs: tuple[float, float],
) -> torch.Tensor:
    signs = torch.tensor(source_signs, dtype=joints21.dtype, device=joints21.device)
    anchors = []
    for label in labels:
        if label == "palm_center":
            point = joints21[:, PALM_JOINTS, :].mean(dim=1)
        else:
            point = joints21[:, TIP_LABEL_TO_JOINT[label], :]
        anchors.append(point[:, list(source_axes)] * signs)
    return torch.stack(anchors, dim=1)


def select_source_axes(
    joints21: torch.Tensor,
    source_axes: tuple[int, int],
    source_signs: tuple[float, float],
) -> torch.Tensor:
    signs = torch.tensor(source_signs, dtype=joints21.dtype, device=joints21.device)
    return joints21[:, :, list(source_axes)] * signs


def visible_smooth_l1(
    pred_uv: torch.Tensor,
    target_uv: torch.Tensor,
    valid: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    diff = pred_uv - target_uv
    mask = valid.unsqueeze(-1).expand_as(diff)
    if not bool(mask.any()):
        return pred_uv.sum() * 0.0
    return torch.nn.functional.smooth_l1_loss(diff[mask], torch.zeros_like(diff[mask]), beta=float(beta))


def compute_error_metrics(errors_by_label: dict[str, list[float]]) -> dict[str, Any]:
    all_errors = [value for values in errors_by_label.values() for value in values]
    if not all_errors:
        return {"n_points": 0}
    arr = np.asarray(all_errors, dtype=np.float64)
    metrics: dict[str, Any] = {
        "n_points": int(arr.size),
        "mean_px": float(np.mean(arr)),
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90)),
        "max_px": float(np.max(arr)),
        "per_label": {},
    }
    for label, values in errors_by_label.items():
        if not values:
            continue
        label_arr = np.asarray(values, dtype=np.float64)
        metrics["per_label"][label] = {
            "n": int(label_arr.size),
            "median_px": float(np.median(label_arr)),
            "p90_px": float(np.percentile(label_arr, 90)),
        }
    return metrics


def compute_error_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n_points": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_points": int(arr.size),
        "mean_px": float(np.mean(arr)),
        "median_px": float(np.median(arr)),
        "p90_px": float(np.percentile(arr, 90)),
        "max_px": float(np.max(arr)),
    }


def choose_precision(requested: str, device: torch.device) -> str:
    if requested == "fp32" or device.type != "cuda":
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


def configure_trainable_stage2(stage2: nn.Module, mode: str, last_blocks: int) -> None:
    for param in stage2.parameters():
        param.requires_grad = False
    stage2.eval()
    if mode == "none":
        return
    if mode in {"pose_head", "last_blocks"}:
        for param in stage2.pose_head.parameters():
            param.requires_grad = True
        stage2.train()
    if mode == "last_blocks":
        stage2.unfreeze_backbone_last_blocks(int(last_blocks))
        stage2.train()


def forward_alignment(
    stage2: nn.Module,
    head: Alignment4DoFHead,
    image: torch.Tensor,
    labels: list[str],
    source_axes: tuple[int, int],
    source_signs: tuple[float, float],
    image_size: int,
    max_theta_rad: float,
    min_scale: float,
    max_scale: float,
    train_stage2: bool,
    precision: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    grad_enabled = bool(train_stage2)
    with torch.set_grad_enabled(grad_enabled):
        with autocast_context(device, precision):
            outputs = stage2(image, return_tokens=True)
    patch_tokens = outputs["patch_tokens"]
    joints21 = outputs["joints21"]
    if not train_stage2:
        patch_tokens = patch_tokens.detach()
        joints21 = joints21.detach()
    raw = head(patch_tokens)
    theta, scale, translation = decode_4dof(
        raw,
        image_size=image_size,
        max_theta_rad=max_theta_rad,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    anchor_source = make_alignment_anchors(joints21, labels, source_axes, source_signs)
    pred_anchor_uv = apply_4dof(anchor_source, theta, scale, translation)
    full_source = select_source_axes(joints21, source_axes, source_signs)
    pred_full_uv = apply_4dof(full_source, theta, scale, translation)
    return {
        "raw": raw,
        "theta": theta,
        "scale": scale,
        "translation": translation,
        "joints21": joints21,
        "pred_anchor_uv": pred_anchor_uv,
        "pred_full_uv": pred_full_uv,
    }


def train_one_epoch(
    stage2: nn.Module,
    head: Alignment4DoFHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    labels: list[str],
    source_axes: tuple[int, int],
    source_signs: tuple[float, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    head.train()
    if args.train_stage2 == "none":
        stage2.eval()
    else:
        stage2.train()
    totals = {"loss": 0.0, "reproj_loss": 0.0, "scale_reg": 0.0, "theta_reg": 0.0}
    steps = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        pred = forward_alignment(
            stage2,
            head,
            image,
            labels,
            source_axes,
            source_signs,
            int(args.image_size),
            float(args.max_theta_rad),
            float(args.min_scale),
            float(args.max_scale),
            train_stage2=args.train_stage2 != "none",
            precision=precision,
            device=device,
        )
        reproj_loss = visible_smooth_l1(
            pred["pred_anchor_uv"],
            target_uv,
            valid,
            beta=float(args.loss_beta),
        )
        scale_reg = torch.mean((torch.log(pred["scale"]) - math.log(float(args.init_scale))) ** 2)
        theta_reg = torch.mean(pred["theta"] ** 2)
        loss = (
            reproj_loss
            + float(args.scale_reg_weight) * scale_reg
            + float(args.theta_reg_weight) * theta_reg
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            float(args.grad_clip),
        )
        optimizer.step()
        totals["loss"] += float(loss.detach().item())
        totals["reproj_loss"] += float(reproj_loss.detach().item())
        totals["scale_reg"] += float(scale_reg.detach().item())
        totals["theta_reg"] += float(theta_reg.detach().item())
        steps += 1
    steps = max(1, steps)
    return {key: value / steps for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    stage2: nn.Module,
    head: Alignment4DoFHead,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    labels: list[str],
    source_axes: tuple[int, int],
    source_signs: tuple[float, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage2.eval()
    head.eval()
    losses = []
    errors_by_label = {label: [] for label in labels}
    errors_by_episode: dict[str, list[float]] = {}
    errors_by_source: dict[str, list[float]] = {}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        pred = forward_alignment(
            stage2,
            head,
            image,
            labels,
            source_axes,
            source_signs,
            int(args.image_size),
            float(args.max_theta_rad),
            float(args.min_scale),
            float(args.max_scale),
            train_stage2=False,
            precision=precision,
            device=device,
        )
        loss = visible_smooth_l1(
            pred["pred_anchor_uv"],
            target_uv,
            valid,
            beta=float(args.loss_beta),
        )
        losses.append(float(loss.detach().item()))
        err = torch.linalg.norm(pred["pred_anchor_uv"] - target_uv, dim=-1)
        err_np = err.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        for label_idx, label in enumerate(labels):
            values = err_np[:, label_idx][valid_np[:, label_idx]]
            errors_by_label[label].extend(float(value) for value in values)
        episodes = list(batch.get("episode", []))
        sources = list(batch.get("source_group", []))
        for sample_idx in range(err_np.shape[0]):
            sample_values = [
                float(value)
                for value in err_np[sample_idx][valid_np[sample_idx]]
            ]
            if not sample_values:
                continue
            if sample_idx < len(episodes):
                errors_by_episode.setdefault(str(episodes[sample_idx]), []).extend(sample_values)
            if sample_idx < len(sources):
                errors_by_source.setdefault(str(sources[sample_idx]), []).extend(sample_values)
    metrics = compute_error_metrics(errors_by_label)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["per_episode"] = {
        key: compute_error_summary(values)
        for key, values in sorted(errors_by_episode.items())
    }
    metrics["per_source"] = {
        key: compute_error_summary(values)
        for key, values in sorted(errors_by_source.items())
    }
    return metrics


def summarize_records(
    records: Sequence[AnnotationRecord],
    labels: list[str],
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    if indices is None:
        selected = list(records)
    else:
        selected = [records[int(idx)] for idx in indices]
    episode_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    label_counts = {label: 0 for label in labels}
    visible_points = 0
    for record in selected:
        episode_counts[record.episode] = episode_counts.get(record.episode, 0) + 1
        dataset_counts[record.dataset] = dataset_counts.get(record.dataset, 0) + 1
        source_counts[record.source_group] = source_counts.get(record.source_group, 0) + 1
        visible_points += int(record.valid.sum())
        for idx, label in enumerate(labels):
            if bool(record.valid[idx]):
                label_counts[label] += 1
    return {
        "frames": len(selected),
        "visible_points": int(visible_points),
        "episodes": dict(sorted(episode_counts.items())),
        "datasets": dict(sorted(dataset_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "label_visible_counts": label_counts,
    }


def random_split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if n == 1:
        return indices, indices
    val_count = max(1, int(round(n * float(val_ratio)))) if val_ratio > 0 else 0
    val_count = min(val_count, n - 1) if n > 1 else val_count
    val_indices = indices[:val_count]
    train_indices = indices[val_count:] or indices
    return train_indices, val_indices or train_indices


def group_split_indices(
    records: Sequence[AnnotationRecord],
    group_key: str,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        key = getattr(record, group_key)
        groups.setdefault(str(key), []).append(idx)
    if len(groups) < 2:
        raise ValueError(
            f"--split-by {group_key} needs at least 2 groups; found {len(groups)}. "
            "Use --split-by random for a smoke test, or add another episode/source for real validation."
        )
    group_names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    target_val_frames = max(1, int(round(len(records) * float(val_ratio)))) if val_ratio > 0 else 0
    val_groups: set[str] = set()
    val_count = 0
    for name in group_names:
        if len(val_groups) >= len(group_names) - 1:
            break
        val_groups.add(name)
        val_count += len(groups[name])
        if val_count >= target_val_frames:
            break
    val_indices = [idx for name in group_names if name in val_groups for idx in groups[name]]
    train_indices = [idx for name in group_names if name not in val_groups for idx in groups[name]]
    if not train_indices or not val_indices:
        raise ValueError(f"invalid grouped split: train={len(train_indices)} val={len(val_indices)}")
    return train_indices, val_indices


def split_records(
    records: Sequence[AnnotationRecord],
    *,
    split_by: str,
    val_ratio: float,
    seed: int,
    val_episodes: Sequence[str] | None = None,
    val_sources: Sequence[str] | None = None,
) -> tuple[list[int], list[int], dict[str, Any]]:
    val_episode_set = set(parse_csv_values(val_episodes))
    val_source_set = set(parse_csv_values(val_sources))
    if val_episode_set or val_source_set:
        val_indices = [
            idx
            for idx, record in enumerate(records)
            if record.episode in val_episode_set or record.source_group in val_source_set
        ]
        val_index_set = set(val_indices)
        train_indices = [idx for idx, _record in enumerate(records) if idx not in val_index_set]
        if not val_indices:
            raise ValueError(
                f"manual validation split selected 0 frames; "
                f"val_episode={sorted(val_episode_set)} val_source={sorted(val_source_set)}"
            )
        if not train_indices:
            raise ValueError("manual validation split leaves 0 training frames")
        summary = {
            "mode": "manual",
            "val_episode": sorted(val_episode_set),
            "val_source": sorted(val_source_set),
        }
        return train_indices, val_indices, summary

    if split_by == "random":
        train_indices, val_indices = random_split_indices(len(records), val_ratio, seed)
        return train_indices, val_indices, {"mode": "random", "val_ratio": float(val_ratio)}
    if split_by == "episode":
        train_indices, val_indices = group_split_indices(records, "episode", val_ratio, seed)
        return train_indices, val_indices, {"mode": "episode", "val_ratio": float(val_ratio)}
    if split_by == "source":
        train_indices, val_indices = group_split_indices(records, "source_group", val_ratio, seed)
        return train_indices, val_indices, {"mode": "source", "val_ratio": float(val_ratio)}
    raise ValueError(f"unsupported split_by: {split_by}")


def make_loader(
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def draw_overlay(
    image_bgr: np.ndarray,
    pred_uv21: np.ndarray,
    target_uv: np.ndarray,
    valid: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    valid21 = (
        np.isfinite(pred_uv21).all(axis=1)
        & (pred_uv21[:, 0] >= 0)
        & (pred_uv21[:, 0] < w)
        & (pred_uv21[:, 1] >= 0)
        & (pred_uv21[:, 1] < h)
    )
    for a, b in HAND_BONES:
        if not (valid21[a] and valid21[b]):
            continue
        pa = tuple(np.round(pred_uv21[a]).astype(int))
        pb = tuple(np.round(pred_uv21[b]).astype(int))
        cv2.line(out, pa, pb, JOINT_COLORS_BGR[b], 1, cv2.LINE_AA)
    for j, uv in enumerate(pred_uv21):
        if not valid21[j]:
            continue
        cv2.circle(out, tuple(np.round(uv).astype(int)), 3, JOINT_COLORS_BGR[j], -1, cv2.LINE_AA)

    for idx, label in enumerate(labels):
        if not valid[idx]:
            continue
        p = tuple(np.round(target_uv[idx]).astype(int))
        color = POINT_COLORS_BGR[label]
        cv2.drawMarker(out, p, (255, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.circle(out, p, 5, color, 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def save_validation_overlays(
    stage2: nn.Module,
    head: Alignment4DoFHead,
    dataset: AlignmentAnnotationDataset,
    indices: list[int],
    output_dir: Path,
    device: torch.device,
    precision: str,
    labels: list[str],
    source_axes: tuple[int, int],
    source_signs: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen = indices[: int(args.num_overlays)]
    if not chosen:
        return
    loader = make_loader(dataset, chosen, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    saved = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        pred = forward_alignment(
            stage2,
            head,
            image,
            labels,
            source_axes,
            source_signs,
            int(args.image_size),
            float(args.max_theta_rad),
            float(args.min_scale),
            float(args.max_scale),
            train_stage2=False,
            precision=precision,
            device=device,
        )
        pred_full_uv = pred["pred_full_uv"].detach().cpu().numpy()
        target_uv_t = batch["target_uv"].numpy()
        valid = batch["valid"].numpy()
        scales = batch["scale"].numpy()
        offsets = batch["offset_xy"].numpy()
        image_paths = list(batch["image_path"])
        record_indices = batch["record_index"].numpy()
        for local_idx, image_path in enumerate(image_paths):
            inv_pred = pred_full_uv[local_idx].copy()
            inv_target = target_uv_t[local_idx].copy()
            inv_pred[:, 0] = (inv_pred[:, 0] - offsets[local_idx, 0]) / scales[local_idx]
            inv_pred[:, 1] = (inv_pred[:, 1] - offsets[local_idx, 1]) / scales[local_idx]
            inv_target[:, 0] = (inv_target[:, 0] - offsets[local_idx, 0]) / scales[local_idx]
            inv_target[:, 1] = (inv_target[:, 1] - offsets[local_idx, 1]) / scales[local_idx]
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            overlay = draw_overlay(
                image_bgr=image_bgr,
                pred_uv21=inv_pred,
                target_uv=inv_target,
                valid=valid[local_idx],
                labels=labels,
            )
            record = dataset.records[int(record_indices[local_idx])]
            out_path = output_dir / f"{saved:04d}_{record.episode}_frame{record.frame_id:06d}.jpg"
            cv2.imwrite(str(out_path), overlay)
            saved += 1


def save_checkpoint(
    path: Path,
    stage2: nn.Module,
    head: Alignment4DoFHead,
    args: argparse.Namespace,
    labels: list[str],
    source_axes: tuple[int, int],
    source_signs: tuple[float, float],
    metrics: dict[str, Any],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage2_model": stage2.state_dict(),
            "alignment_head": head.state_dict(),
            "checkpoint_source": str(args.checkpoint),
            "epoch": int(epoch),
            "metrics": metrics,
            "labels": labels,
            "anchor_definition": {
                "palm_center": list(PALM_JOINTS),
                "tips": TIP_LABEL_TO_JOINT,
                "source_axes": list(source_axes),
                "source_signs": list(source_signs),
                "model": "uv = scale * R(theta) * (joints21[source_axes] * source_signs) + translation",
            },
            "config": vars(args),
        },
        path,
    )


def append_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        action="append",
        default=None,
        help="JSON from tools/click_label_visible_hand_points.py. Repeat to merge multiple annotation files.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Existing Stage2 checkpoint")
    parser.add_argument("--config", type=Path, default=None, help="Optional Stage2 config override")
    parser.add_argument("--output-dir", type=Path, default=WRIST_ROOT / "outputs" / "stage2_4dof_alignment")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--labels", type=parse_labels, default=list(DEFAULT_LABELS))
    parser.add_argument("--min-visible", type=int, default=3)
    parser.add_argument("--source-axes", type=parse_source_axes, default=parse_source_axes("x,y"))
    parser.add_argument(
        "--source-signs",
        type=parse_source_signs,
        default=parse_source_signs("1,1"),
        help=(
            "Per-axis signs applied after --source-axes before 4DoF projection. "
            "Use this to test hand-local axis direction/reflection, e.g. 1,1 or -1,1."
        ),
    )
    parser.add_argument("--train-stage2", choices=("none", "pose_head", "last_blocks"), default="none")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--split-by",
        choices=("random", "episode", "source"),
        default="random",
        help=(
            "Validation split mode. Use episode/source for realistic checks when frames are temporally redundant."
        ),
    )
    parser.add_argument(
        "--val-episode",
        action="append",
        default=[],
        help="Force one or more episode names into validation. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--val-source",
        action="append",
        default=[],
        help="Force one or more source_group names into validation. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--align-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-beta", type=float, default=4.0)
    parser.add_argument("--init-scale", type=float, default=1800.0)
    parser.add_argument("--min-scale", type=float, default=100.0)
    parser.add_argument("--max-scale", type=float, default=8000.0)
    parser.add_argument("--max-theta-rad", type=float, default=math.pi)
    parser.add_argument("--scale-reg-weight", type=float, default=0.001)
    parser.add_argument("--theta-reg-weight", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--num-overlays", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--print-records", type=int, default=0, help="Print the first N usable records and exit/dry-run context.")
    parser.add_argument("--dry-run", action="store_true", help="Only parse annotation JSON and print split sizes")
    parser.add_argument("--self-test", action="store_true", help="Run a small synthetic geometry/loss test and exit")
    return parser


def run_self_test() -> int:
    set_seed(0)
    joints = torch.randn(2, 21, 3) * 0.05
    labels = list(DEFAULT_LABELS)
    axes = (0, 1)
    signs = (1.0, -1.0)
    anchors = make_alignment_anchors(joints, labels, axes, signs)
    raw = torch.tensor([[0.1, math.log(1200.0), 0.0, 0.0], [-0.2, math.log(1500.0), 0.3, -0.4]])
    theta, scale, translation = decode_4dof(raw, image_size=448, max_theta_rad=math.pi, min_scale=100.0, max_scale=8000.0)
    pred = apply_4dof(anchors, theta, scale, translation)
    target = pred.detach() + torch.randn_like(pred) * 0.5
    valid = torch.ones(2, len(labels), dtype=torch.bool)
    loss = visible_smooth_l1(pred, target, valid, beta=4.0)
    if not torch.isfinite(loss):
        raise RuntimeError("self-test loss is not finite")
    print({"self_test": "ok", "loss": float(loss.item()), "pred_shape": list(pred.shape)})
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return run_self_test()
    if args.annotations is None:
        raise SystemExit("--annotations is required unless --self-test is used")

    labels = list(args.labels)
    annotation_paths = list(args.annotations)
    records = load_annotation_records(annotation_paths, labels=labels, min_visible=int(args.min_visible))
    train_indices, val_indices, split_summary = split_records(
        records,
        split_by=str(args.split_by),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        val_episodes=args.val_episode,
        val_sources=args.val_source,
    )
    record_summary = summarize_records(records, labels)
    train_summary = summarize_records(records, labels, train_indices)
    val_summary = summarize_records(records, labels, val_indices)
    print(
        json.dumps(
            {
                "annotations": [str(path) for path in annotation_paths],
                "usable_frames": len(records),
                "train_frames": len(train_indices),
                "val_frames": len(val_indices),
                "split": split_summary,
                "labels": labels,
                "min_visible": int(args.min_visible),
                "source_axes": list(args.source_axes),
                "source_signs": list(args.source_signs),
                "all_summary": record_summary,
                "train_summary": train_summary,
                "val_summary": val_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if int(args.print_records) > 0:
        limit = min(int(args.print_records), len(records))
        for idx, record in enumerate(records[:limit]):
            print(
                f"{idx:05d}\t"
                f"dataset={record.dataset}\t"
                f"episode={record.episode}\t"
                f"source={record.source_group}\t"
                f"visible={int(record.valid.sum())}\t"
                f"image={record.image}"
            )
    if args.dry_run:
        return 0

    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    precision = choose_precision(str(args.precision), device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_name = args.run_name or datetime.now().strftime("align4dof_%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "config.json",
        to_jsonable(
            {
                **vars(args),
                "labels": labels,
                "source_axes": list(args.source_axes),
                "source_signs": list(args.source_signs),
            }
        ),
    )
    atomic_write_json(
        run_dir / "split.json",
        to_jsonable(
            {
                "annotations": annotation_paths,
                "split": split_summary,
                "all_summary": record_summary,
                "train_summary": train_summary,
                "val_summary": val_summary,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_records": [
                    {
                        "idx": int(idx),
                        "image": records[int(idx)].image,
                        "image_rel": records[int(idx)].image_rel,
                        "dataset": records[int(idx)].dataset,
                        "episode": records[int(idx)].episode,
                        "source_group": records[int(idx)].source_group,
                        "visible": int(records[int(idx)].valid.sum()),
                    }
                    for idx in train_indices
                ],
                "val_records": [
                    {
                        "idx": int(idx),
                        "image": records[int(idx)].image,
                        "image_rel": records[int(idx)].image_rel,
                        "dataset": records[int(idx)].dataset,
                        "episode": records[int(idx)].episode,
                        "source_group": records[int(idx)].source_group,
                        "visible": int(records[int(idx)].valid.sum()),
                    }
                    for idx in val_indices
                ],
            }
        ),
    )

    print(f"loading Stage2 checkpoint: {args.checkpoint}")
    stage2, _, cfg = load_stage2_model(
        checkpoint_path=args.checkpoint,
        repo_root=REPO_ROOT,
        device=device,
        config_path=args.config,
    )
    image_size = int(cfg["data"]["image_size"])
    args.image_size = image_size
    configure_trainable_stage2(stage2, str(args.train_stage2), int(args.unfreeze_last_blocks))

    dataset = AlignmentAnnotationDataset(records=records, labels=labels, image_size=image_size)
    train_loader = make_loader(dataset, train_indices, int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = make_loader(dataset, val_indices, int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))

    head = Alignment4DoFHead(
        feature_dim=int(cfg["model"].get("feature_dim", 384)),
        hidden_dim=256,
        dropout=0.1,
        init_scale=float(args.init_scale),
    ).to(device)

    param_groups = [{"params": head.parameters(), "lr": float(args.align_lr)}]
    stage2_params = [param for param in stage2.parameters() if param.requires_grad]
    if stage2_params:
        param_groups.append({"params": stage2_params, "lr": float(args.stage2_lr)})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(args.weight_decay))

    best_metric = float("inf")
    best_metrics: dict[str, Any] = {}
    print(
        f"run_dir={run_dir}\n"
        f"device={device} precision={precision} image_size={image_size} "
        f"train_stage2={args.train_stage2} trainable_stage2_params={sum(p.numel() for p in stage2_params)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            stage2=stage2,
            head=head,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            precision=precision,
            labels=labels,
            source_axes=args.source_axes,
            source_signs=args.source_signs,
            args=args,
        )
        val_metrics = evaluate(
            stage2=stage2,
            head=head,
            loader=val_loader,
            device=device,
            precision=precision,
            labels=labels,
            source_axes=args.source_axes,
            source_signs=args.source_signs,
            args=args,
        )
        current = float(val_metrics.get("median_px", float("inf")))
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reproj_loss": train_metrics["reproj_loss"],
            "val_loss": val_metrics.get("loss", float("nan")),
            "val_mean_px": val_metrics.get("mean_px", float("nan")),
            "val_median_px": val_metrics.get("median_px", float("nan")),
            "val_p90_px": val_metrics.get("p90_px", float("nan")),
            "val_max_px": val_metrics.get("max_px", float("nan")),
            "val_n_points": val_metrics.get("n_points", 0),
        }
        append_log(run_dir / "metrics.csv", row)
        print(
            f"epoch {epoch:04d}: "
            f"train_loss={row['train_loss']:.4f} "
            f"val_median={row['val_median_px']:.3f}px "
            f"val_p90={row['val_p90_px']:.3f}px "
            f"val_n={row['val_n_points']}"
        )
        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            stage2,
            head,
            args,
            labels,
            args.source_axes,
            args.source_signs,
            val_metrics,
            epoch,
        )
        if current < best_metric:
            best_metric = current
            best_metrics = val_metrics
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                stage2,
                head,
                args,
                labels,
                args.source_axes,
                args.source_signs,
                val_metrics,
                epoch,
            )
        if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch{epoch:04d}.pt",
                stage2,
                head,
                args,
                labels,
                args.source_axes,
                args.source_signs,
                val_metrics,
                epoch,
            )

    atomic_write_json(run_dir / "summary.json", {"best_metric": best_metric, "best_metrics": best_metrics})
    save_validation_overlays(
        stage2=stage2,
        head=head,
        dataset=dataset,
        indices=val_indices,
        output_dir=run_dir / "overlays",
        device=device,
        precision=precision,
        labels=labels,
        source_axes=args.source_axes,
        source_signs=args.source_signs,
        args=args,
    )
    print(f"done: {run_dir}")
    if int(args.epochs) > 0:
        print(f"best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")
    else:
        print("epochs=0: no training checkpoint was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
