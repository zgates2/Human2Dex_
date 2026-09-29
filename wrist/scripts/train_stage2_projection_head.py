#!/usr/bin/env python3
"""Train a frozen-Stage1 3D-to-RGB projection head for wrist MANO21.

This is the intended Stage 2 after a wrist-view MANO21 model has been trained:

  RGB -> frozen Stage1 DINOv3 + Query MANO decoder -> joints21_3d
  patch tokens + joints21_3d -> projection head -> weak-perspective camera
  camera projects all 21 joints back to RGB

Only the projection head is trained here. The Stage1 MANO model is frozen so
the 21-point hand structure cannot be damaged by sparse 2D annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
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


DEFAULT_STAGE1_CHECKPOINT = (
    WRIST_ROOT
    / "outputs"
    / "test_3"
    / "runs"
    / "stage2_test_3"
    / "checkpoints"
    / "best.pt"
)
FINGERTIP6_LABELS = (
    "palm_center",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)
MANO21_LABELS = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
DEFAULT_LABELS = FINGERTIP6_LABELS
LABEL_TO_JOINT = {label: idx for idx, label in enumerate(MANO21_LABELS)}
TIP_LABEL_TO_JOINT = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
PALM_JOINTS = (0, 1, 5, 9, 13, 17)
SUPPORTED_LABELS = tuple(dict.fromkeys((*FINGERTIP6_LABELS, *MANO21_LABELS)))
PROJECTION_HEAD_TYPES = ("weak", "residual2d")
POINT_COLORS_BGR = {
    "palm_center": (80, 80, 255),
    "wrist": (245, 245, 245),
    "thumb_cmc": (80, 180, 255),
    "thumb_mcp": (80, 180, 255),
    "thumb_ip": (80, 180, 255),
    "thumb_tip": (80, 180, 255),
    "index_mcp": (80, 255, 120),
    "index_pip": (80, 255, 120),
    "index_dip": (80, 255, 120),
    "index_tip": (80, 255, 120),
    "middle_mcp": (255, 210, 80),
    "middle_pip": (255, 210, 80),
    "middle_dip": (255, 210, 80),
    "middle_tip": (255, 210, 80),
    "ring_mcp": (255, 120, 180),
    "ring_pip": (255, 120, 180),
    "ring_dip": (255, 120, 180),
    "ring_tip": (255, 120, 180),
    "pinky_mcp": (180, 120, 255),
    "pinky_pip": (180, 120, 255),
    "pinky_dip": (180, 120, 255),
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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
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
    value = str(value).strip()
    if value in {"fingertips6", "tips6"}:
        return list(FINGERTIP6_LABELS)
    if value in {"mano21", "all21", "full21"}:
        return list(MANO21_LABELS)
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("labels cannot be empty")
    missing = [label for label in labels if label not in SUPPORTED_LABELS]
    if missing:
        raise argparse.ArgumentTypeError(f"unsupported labels: {missing}; expected subset/profile of {SUPPORTED_LABELS}")
    return labels


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


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


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
            record = AnnotationRecord(
                frame_id=int(raw.get("frame_id", len(records_by_image))),
                episode=episode,
                image=image,
                image_rel=str(raw.get("image_rel", image.name)),
                dataset=dataset,
                source_group=infer_source_group(image=image, episode=episode, raw=raw, payload=payload),
                annotation_file=annotations_path,
                target_uv_original=uv,
                valid=valid,
            )
            records_by_image[str(image)] = record
    records = list(records_by_image.values())
    if not records:
        raise ValueError(f"no usable annotated frames; min_visible={min_visible}, labels={labels}")
    return records


class ProjectionAnnotationDataset(Dataset):
    def __init__(self, records: list[AnnotationRecord], labels: list[str], image_size: int) -> None:
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
        target[:, 0] = target[:, 0] * float(result.scale) + float(result.offset_xy[0])
        target[:, 1] = target[:, 1] * float(result.scale) + float(result.offset_xy[1])
        return {
            "image": result.image,
            "target_uv": torch.from_numpy(target).float(),
            "valid": torch.from_numpy(record.valid.copy()).bool(),
            "scale": torch.tensor(float(result.scale), dtype=torch.float32),
            "offset_xy": torch.tensor(result.offset_xy, dtype=torch.float32),
            "record_index": torch.tensor(idx, dtype=torch.long),
            "image_path": str(record.image),
            "image_rel": record.image_rel,
            "episode": record.episode,
            "dataset": record.dataset,
            "source_group": record.source_group,
        }


class ProjectionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dim: int = 512,
        joint_embed_dim: int = 128,
        dropout: float = 0.1,
        init_scale: float = 140.0,
        projection_head_type: str = "residual2d",
        residual_scale: float = 32.0,
    ) -> None:
        super().__init__()
        projection_head_type = str(projection_head_type)
        if projection_head_type not in PROJECTION_HEAD_TYPES:
            raise ValueError(f"unsupported projection_head_type={projection_head_type!r}; expected {PROJECTION_HEAD_TYPES}")
        self.projection_head_type = projection_head_type
        self.residual_scale = float(residual_scale)
        output_dim = 9 + (21 * 2 if projection_head_type == "residual2d" else 0)
        self.joint_encoder = nn.Sequential(
            nn.LayerNorm(21 * 3),
            nn.Linear(21 * 3, 256),
            nn.GELU(),
            nn.Linear(256, joint_embed_dim),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim + joint_embed_dim),
            nn.Linear(feature_dim + joint_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.zero_()
            final.bias[:6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            final.bias[6] = math.log(max(float(init_scale), 1e-4))

    def forward(self, patch_tokens: torch.Tensor, joints21_norm: torch.Tensor) -> torch.Tensor:
        image_feature = patch_tokens.mean(dim=1)
        joint_feature = self.joint_encoder(joints21_norm.flatten(1))
        return self.net(torch.cat([image_feature, joint_feature], dim=-1))


def split_projection_output(
    raw: torch.Tensor,
    projection_head_type: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    projection_head_type = str(projection_head_type)
    if projection_head_type == "weak":
        return raw[:, :9], None
    if projection_head_type == "residual2d":
        if raw.shape[-1] < 9 + 21 * 2:
            raise ValueError(f"residual2d head expected at least 51 dims, got {raw.shape[-1]}")
        return raw[:, :9], raw[:, 9 : 9 + 21 * 2].reshape(raw.shape[0], 21, 2)
    raise ValueError(f"unsupported projection_head_type={projection_head_type!r}")


def apply_residual_2d(
    uv21_weak: torch.Tensor,
    residual_raw: torch.Tensor | None,
    residual_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual_raw is None:
        delta_uv = torch.zeros_like(uv21_weak)
        return uv21_weak, delta_uv
    delta_uv = torch.tanh(residual_raw.float()) * float(residual_scale)
    return uv21_weak + delta_uv.to(dtype=uv21_weak.dtype), delta_uv


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def normalize_joints21(joints21: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    center = joints21[:, PALM_JOINTS, :].mean(dim=1, keepdim=True)
    centered = joints21 - center
    tips = centered[:, [4, 8, 12, 16, 20], :]
    hand_scale = torch.linalg.norm(tips, dim=-1).mean(dim=1, keepdim=True).clamp_min(1e-4)
    normalized = centered / hand_scale[:, None, :]
    return normalized, center, hand_scale


def decode_projection(
    raw: torch.Tensor,
    image_size: int,
    min_scale: float,
    max_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot = rotation_6d_to_matrix(raw[:, :6])
    log_scale = raw[:, 6].clamp(math.log(float(min_scale)), math.log(float(max_scale)))
    scale = torch.exp(log_scale)
    tx = float(image_size) * torch.sigmoid(raw[:, 7])
    ty = float(image_size) * torch.sigmoid(raw[:, 8])
    translation = torch.stack([tx, ty], dim=-1)
    return rot, scale, translation


def project_weak_perspective(
    joints21_norm: torch.Tensor,
    rot: torch.Tensor,
    scale: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    rotated = torch.matmul(joints21_norm, rot.transpose(-1, -2))
    return rotated[:, :, :2] * scale[:, None, None] + translation[:, None, :]


def make_anchor_uv(uv21: torch.Tensor, labels: list[str]) -> torch.Tensor:
    anchors = []
    for label in labels:
        if label == "palm_center":
            anchors.append(uv21[:, PALM_JOINTS, :].mean(dim=1))
        elif label in LABEL_TO_JOINT:
            anchors.append(uv21[:, LABEL_TO_JOINT[label], :])
        else:
            raise KeyError(f"unsupported projection label: {label}")
    return torch.stack(anchors, dim=1)


def visible_smooth_l1(pred_uv: torch.Tensor, target_uv: torch.Tensor, valid: torch.Tensor, beta: float) -> torch.Tensor:
    diff = pred_uv - target_uv
    mask = valid.unsqueeze(-1).expand_as(diff)
    if not bool(mask.any()):
        return pred_uv.sum() * 0.0
    return torch.nn.functional.smooth_l1_loss(diff[mask], torch.zeros_like(diff[mask]), beta=float(beta))


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


def compute_error_metrics(errors_by_label: dict[str, list[float]]) -> dict[str, Any]:
    all_errors = [v for values in errors_by_label.values() for v in values]
    metrics = compute_error_summary(all_errors)
    metrics["per_label"] = {
        label: compute_error_summary(values)
        for label, values in errors_by_label.items()
        if values
    }
    return metrics


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


def freeze_stage1(stage1: nn.Module) -> None:
    for param in stage1.parameters():
        param.requires_grad = False
    stage1.eval()


def forward_projection(
    stage1: nn.Module,
    head: ProjectionHead,
    image: torch.Tensor,
    labels: list[str],
    image_size: int,
    min_scale: float,
    max_scale: float,
    precision: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        with autocast_context(device, precision):
            outputs = stage1(image, return_tokens=True)
    patch_tokens = outputs["patch_tokens"].detach()
    joints21 = outputs["joints21"].detach()
    joints21_norm, _center, hand_scale = normalize_joints21(joints21.float())
    raw = head(patch_tokens.float(), joints21_norm)
    base_raw, residual_raw = split_projection_output(
        raw,
        projection_head_type=getattr(head, "projection_head_type", "weak"),
    )
    rot, proj_scale, translation = decode_projection(base_raw, image_size=image_size, min_scale=min_scale, max_scale=max_scale)
    uv21_weak = project_weak_perspective(joints21_norm, rot, proj_scale, translation)
    uv21, delta_uv = apply_residual_2d(
        uv21_weak,
        residual_raw,
        residual_scale=float(getattr(head, "residual_scale", 32.0)),
    )
    anchor_uv = make_anchor_uv(uv21, labels)
    return {
        "raw": raw,
        "base_raw": base_raw,
        "residual_raw": residual_raw,
        "rot": rot,
        "proj_scale": proj_scale,
        "translation": translation,
        "hand_scale": hand_scale,
        "joints21": joints21,
        "joints21_norm": joints21_norm,
        "uv21_weak": uv21_weak,
        "delta_uv": delta_uv,
        "uv21": uv21,
        "anchor_uv": anchor_uv,
    }


def train_one_epoch(
    stage1: nn.Module,
    head: ProjectionHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    labels: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    stage1.eval()
    head.train()
    totals = {"loss": 0.0, "reproj_loss": 0.0, "scale_reg": 0.0, "residual_reg": 0.0}
    steps = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        pred = forward_projection(
            stage1,
            head,
            image,
            labels,
            int(args.image_size),
            float(args.min_proj_scale),
            float(args.max_proj_scale),
            precision,
            device,
        )
        reproj_loss = visible_smooth_l1(pred["anchor_uv"], target_uv, valid, beta=float(args.loss_beta))
        scale_reg = torch.mean((torch.log(pred["proj_scale"]) - math.log(float(args.init_proj_scale))) ** 2)
        residual_reg = torch.mean(pred["delta_uv"].float() ** 2)
        loss = (
            reproj_loss
            + float(args.scale_reg_weight) * scale_reg
            + float(args.residual_reg_weight) * residual_reg
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.grad_clip))
        optimizer.step()
        totals["loss"] += float(loss.detach().item())
        totals["reproj_loss"] += float(reproj_loss.detach().item())
        totals["scale_reg"] += float(scale_reg.detach().item())
        totals["residual_reg"] += float(residual_reg.detach().item())
        steps += 1
    steps = max(1, steps)
    return {key: value / steps for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    stage1: nn.Module,
    head: ProjectionHead,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    labels: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage1.eval()
    head.eval()
    losses = []
    errors_by_label = {label: [] for label in labels}
    weak_errors_by_label = {label: [] for label in labels}
    errors_by_episode: dict[str, list[float]] = {}
    errors_by_source: dict[str, list[float]] = {}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        pred = forward_projection(
            stage1,
            head,
            image,
            labels,
            int(args.image_size),
            float(args.min_proj_scale),
            float(args.max_proj_scale),
            precision,
            device,
        )
        loss = visible_smooth_l1(pred["anchor_uv"], target_uv, valid, beta=float(args.loss_beta))
        losses.append(float(loss.detach().item()))
        err = torch.linalg.norm(pred["anchor_uv"] - target_uv, dim=-1)
        weak_anchor_uv = make_anchor_uv(pred["uv21_weak"], labels)
        weak_err = torch.linalg.norm(weak_anchor_uv - target_uv, dim=-1)
        err_np = err.detach().cpu().numpy()
        weak_err_np = weak_err.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        for label_idx, label in enumerate(labels):
            values = err_np[:, label_idx][valid_np[:, label_idx]]
            errors_by_label[label].extend(float(value) for value in values)
            weak_values = weak_err_np[:, label_idx][valid_np[:, label_idx]]
            weak_errors_by_label[label].extend(float(value) for value in weak_values)
        episodes = list(batch.get("episode", []))
        sources = list(batch.get("source_group", []))
        for sample_idx in range(err_np.shape[0]):
            sample_values = [float(v) for v in err_np[sample_idx][valid_np[sample_idx]]]
            if sample_idx < len(episodes):
                errors_by_episode.setdefault(str(episodes[sample_idx]), []).extend(sample_values)
            if sample_idx < len(sources):
                errors_by_source.setdefault(str(sources[sample_idx]), []).extend(sample_values)
    metrics = compute_error_metrics(errors_by_label)
    weak_metrics = compute_error_metrics(weak_errors_by_label)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["weak_mean_px"] = weak_metrics.get("mean_px", float("nan"))
    metrics["weak_median_px"] = weak_metrics.get("median_px", float("nan"))
    metrics["weak_p90_px"] = weak_metrics.get("p90_px", float("nan"))
    metrics["weak_per_label"] = weak_metrics.get("per_label", {})
    metrics["per_episode"] = {k: compute_error_summary(v) for k, v in sorted(errors_by_episode.items())}
    metrics["per_source"] = {k: compute_error_summary(v) for k, v in sorted(errors_by_source.items())}
    return metrics


def summarize_records(records: Sequence[AnnotationRecord], labels: list[str], indices: Sequence[int] | None = None) -> dict[str, Any]:
    selected = list(records) if indices is None else [records[int(idx)] for idx in indices]
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


def group_split_indices(records: Sequence[AnnotationRecord], group_key: str, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    groups: dict[str, list[int]] = {}
    for idx, record in enumerate(records):
        groups.setdefault(str(getattr(record, group_key)), []).append(idx)
    if len(groups) < 2:
        raise ValueError(f"--split-by {group_key} needs at least 2 groups; found {len(groups)}. Use random for smoke tests.")
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    target = max(1, int(round(len(records) * float(val_ratio)))) if val_ratio > 0 else 0
    val_names: set[str] = set()
    count = 0
    for name in names:
        if len(val_names) >= len(names) - 1:
            break
        val_names.add(name)
        count += len(groups[name])
        if count >= target:
            break
    val_indices = [idx for name in names if name in val_names for idx in groups[name]]
    train_indices = [idx for name in names if name not in val_names for idx in groups[name]]
    if not train_indices or not val_indices:
        raise ValueError(f"invalid split: train={len(train_indices)} val={len(val_indices)}")
    return train_indices, val_indices


def split_records(
    records: Sequence[AnnotationRecord],
    *,
    split_by: str,
    val_ratio: float,
    seed: int,
    val_episodes: Sequence[str] | None,
    val_sources: Sequence[str] | None,
) -> tuple[list[int], list[int], dict[str, Any]]:
    val_episode_set = set(parse_csv_values(val_episodes))
    val_source_set = set(parse_csv_values(val_sources))
    if val_episode_set or val_source_set:
        val_indices = [
            idx
            for idx, record in enumerate(records)
            if record.episode in val_episode_set or record.source_group in val_source_set
        ]
        val_set = set(val_indices)
        train_indices = [idx for idx, _ in enumerate(records) if idx not in val_set]
        if not val_indices:
            raise ValueError("manual validation split selected 0 frames")
        if not train_indices:
            raise ValueError("manual validation split leaves 0 training frames")
        return train_indices, val_indices, {"mode": "manual", "val_episode": sorted(val_episode_set), "val_source": sorted(val_source_set)}
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


def make_loader(dataset: Dataset, indices: list[int], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers > 0),
    )


def load_projection_head_initialization(head: ProjectionHead, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("projection_head"), dict):
        raise ValueError(f"init projection checkpoint does not contain projection_head: {path}")
    source = checkpoint["projection_head"]
    target = head.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    copied: list[str] = []
    expanded: list[str] = []
    skipped: dict[str, list[int]] = {}
    for key, value in source.items():
        if key not in target:
            continue
        src = value.detach().cpu()
        dst = target[key].detach().cpu()
        if tuple(src.shape) == tuple(dst.shape):
            adapted[key] = src
            copied.append(key)
            continue
        if key in {"net.7.weight", "net.8.weight"} and src.ndim == 2 and dst.ndim == 2 and src.shape[0] == 9 and dst.shape[0] >= 9 and src.shape[1] == dst.shape[1]:
            merged = dst.clone()
            merged[:9].copy_(src)
            adapted[key] = merged
            expanded.append(key)
            continue
        if key in {"net.7.bias", "net.8.bias"} and src.ndim == 1 and dst.ndim == 1 and src.shape[0] == 9 and dst.shape[0] >= 9:
            merged = dst.clone()
            merged[:9].copy_(src)
            adapted[key] = merged
            expanded.append(key)
            continue
        skipped[key] = [int(x) for x in src.shape]
    missing, unexpected = head.load_state_dict(adapted, strict=False)
    head.to(device)
    return {
        "checkpoint": str(path),
        "copied": copied,
        "expanded": expanded,
        "skipped": skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "source_projection_head_type": checkpoint.get("projection_head_type")
        or (checkpoint.get("config", {}) or {}).get("projection_head_type")
        or "weak",
    }


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
        cv2.line(out, pa, pb, JOINT_COLORS_BGR[b], 2, cv2.LINE_AA)
    for j, uv in enumerate(pred_uv21):
        if not valid21[j]:
            continue
        cv2.circle(out, tuple(np.round(uv).astype(int)), 4, JOINT_COLORS_BGR[j], -1, cv2.LINE_AA)
    for idx, label in enumerate(labels):
        if not valid[idx]:
            continue
        p = tuple(np.round(target_uv[idx]).astype(int))
        color = POINT_COLORS_BGR[label]
        cv2.drawMarker(out, p, (255, 255, 255), cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.circle(out, p, 6, color, 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def save_validation_overlays(
    stage1: nn.Module,
    head: ProjectionHead,
    dataset: ProjectionAnnotationDataset,
    indices: list[int],
    output_dir: Path,
    device: torch.device,
    precision: str,
    labels: list[str],
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
        pred = forward_projection(
            stage1,
            head,
            image,
            labels,
            int(args.image_size),
            float(args.min_proj_scale),
            float(args.max_proj_scale),
            precision,
            device,
        )
        uv21 = pred["uv21"].detach().cpu().numpy()
        target_uv = batch["target_uv"].numpy()
        valid = batch["valid"].numpy()
        scales = batch["scale"].numpy()
        offsets = batch["offset_xy"].numpy()
        image_paths = list(batch["image_path"])
        record_indices = batch["record_index"].numpy()
        for local_idx, image_path in enumerate(image_paths):
            inv_uv21 = uv21[local_idx].copy()
            inv_target = target_uv[local_idx].copy()
            inv_uv21[:, 0] = (inv_uv21[:, 0] - offsets[local_idx, 0]) / scales[local_idx]
            inv_uv21[:, 1] = (inv_uv21[:, 1] - offsets[local_idx, 1]) / scales[local_idx]
            inv_target[:, 0] = (inv_target[:, 0] - offsets[local_idx, 0]) / scales[local_idx]
            inv_target[:, 1] = (inv_target[:, 1] - offsets[local_idx, 1]) / scales[local_idx]
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                continue
            overlay = draw_overlay(image_bgr, inv_uv21, inv_target, valid[local_idx], labels)
            record = dataset.records[int(record_indices[local_idx])]
            out_path = output_dir / f"{saved:04d}_{record.episode}_frame{record.frame_id:06d}.jpg"
            cv2.imwrite(str(out_path), overlay)
            saved += 1


def save_checkpoint(
    path: Path,
    head: ProjectionHead,
    args: argparse.Namespace,
    labels: list[str],
    metrics: dict[str, Any],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projection_head": head.state_dict(),
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "epoch": int(epoch),
            "metrics": metrics,
            "labels": labels,
            "projection_head_type": str(args.projection_head_type),
            "residual_scale": float(args.residual_scale),
            "projection_model": "uv = weak_perspective(normalize(joints21_3d)) + tanh(delta_uv) * residual_scale",
            "config": to_jsonable(vars(args)),
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
    parser.add_argument("--annotations", type=Path, action="append", default=None, help="JSON from click_label_visible_hand_points.py. Repeatable.")
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument(
        "--init-projection-head-checkpoint",
        type=Path,
        default=None,
        help="Optional weak/residual projection checkpoint used to initialize this projection head before training.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional Stage1 config override.")
    parser.add_argument("--output-dir", type=Path, default=WRIST_ROOT / "outputs" / "stage2_projection_head")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--labels", type=parse_labels, default=list(DEFAULT_LABELS))
    parser.add_argument("--min-visible", type=int, default=3)
    parser.add_argument("--split-by", choices=("random", "episode", "source"), default="random")
    parser.add_argument("--val-episode", action="append", default=[])
    parser.add_argument("--val-source", action="append", default=[])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-beta", type=float, default=4.0)
    parser.add_argument("--init-proj-scale", type=float, default=140.0)
    parser.add_argument("--min-proj-scale", type=float, default=10.0)
    parser.add_argument("--max-proj-scale", type=float, default=800.0)
    parser.add_argument("--scale-reg-weight", type=float, default=0.001)
    parser.add_argument(
        "--projection-head-type",
        choices=PROJECTION_HEAD_TYPES,
        default="residual2d",
        help="weak keeps the old 9-DoF weak-perspective head; residual2d adds a learned per-joint 2D correction after weak projection.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=32.0,
        help="Max pixel magnitude of tanh-bounded per-joint residual correction in model-input coordinates.",
    )
    parser.add_argument(
        "--residual-reg-weight",
        type=float,
        default=1e-4,
        help="L2 regularization weight on pixel residuals; keep small so it prevents drift without blocking correction.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--num-overlays", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--print-records", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_self_test() -> int:
    set_seed(0)
    joints = torch.randn(2, 21, 3)
    joints_norm, _center, _scale = normalize_joints21(joints)
    raw = torch.zeros(2, 9 + 21 * 2)
    raw[:, :6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    raw[:, 6] = math.log(120.0)
    base_raw, residual_raw = split_projection_output(raw, "residual2d")
    rot, scale, t = decode_projection(base_raw, image_size=448, min_scale=10.0, max_scale=800.0)
    uv21_weak = project_weak_perspective(joints_norm, rot, scale, t)
    uv21, delta_uv = apply_residual_2d(uv21_weak, residual_raw, residual_scale=32.0)
    anchors = make_anchor_uv(uv21, list(DEFAULT_LABELS))
    valid = torch.ones(2, len(DEFAULT_LABELS), dtype=torch.bool)
    loss = visible_smooth_l1(anchors, anchors.detach() + 0.1, valid, beta=4.0)
    if not torch.isfinite(loss):
        raise RuntimeError("self-test loss is not finite")
    if float(delta_uv.abs().max()) != 0.0:
        raise RuntimeError("zero-initialized residual should produce zero delta in self-test")
    print({"self_test": "ok", "uv21_shape": list(uv21.shape), "loss": float(loss.item())})
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
    dry_info = {
        "annotations": [str(path) for path in annotation_paths],
        "usable_frames": len(records),
        "train_frames": len(train_indices),
        "val_frames": len(val_indices),
        "split": split_summary,
        "labels": labels,
        "stage1_checkpoint": str(args.stage1_checkpoint),
        "init_projection_head_checkpoint": str(args.init_projection_head_checkpoint) if args.init_projection_head_checkpoint else None,
        "all_summary": record_summary,
        "train_summary": train_summary,
        "val_summary": val_summary,
    }
    print(json.dumps(dry_info, indent=2, ensure_ascii=False))
    if int(args.print_records) > 0:
        for idx, record in enumerate(records[: int(args.print_records)]):
            print(f"{idx:05d}\tepisode={record.episode}\tsource={record.source_group}\tvisible={int(record.valid.sum())}\timage={record.image}")
    if args.dry_run:
        return 0

    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    precision = choose_precision(str(args.precision), device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_name = args.run_name or datetime.now().strftime("projection_%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "config.json", to_jsonable({**vars(args), "labels": labels}))
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
            }
        ),
    )

    print(f"loading frozen Stage1 checkpoint: {args.stage1_checkpoint}")
    stage1, _transform, cfg = load_stage2_model(
        checkpoint_path=args.stage1_checkpoint,
        repo_root=REPO_ROOT,
        device=device,
        config_path=args.config,
    )
    freeze_stage1(stage1)
    image_size = int(cfg["data"]["image_size"])
    args.image_size = image_size

    dataset = ProjectionAnnotationDataset(records=records, labels=labels, image_size=image_size)
    train_loader = make_loader(dataset, train_indices, int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = make_loader(dataset, val_indices, int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    head = ProjectionHead(
        feature_dim=int(cfg["model"].get("feature_dim", 384)),
        hidden_dim=512,
        joint_embed_dim=128,
        dropout=0.1,
        init_scale=float(args.init_proj_scale),
        projection_head_type=str(args.projection_head_type),
        residual_scale=float(args.residual_scale),
    ).to(device)
    init_summary = None
    if args.init_projection_head_checkpoint is not None:
        init_summary = load_projection_head_initialization(head, args.init_projection_head_checkpoint, device=device)
        atomic_write_json(run_dir / "projection_head_init.json", to_jsonable(init_summary))
        print(
            "initialized projection head from "
            f"{init_summary['checkpoint']} "
            f"(copied={len(init_summary['copied'])}, expanded={init_summary['expanded']})"
        )
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_metric = float("inf")
    best_metrics: dict[str, Any] = {}
    print(f"run_dir={run_dir}\ndevice={device} precision={precision} image_size={image_size} frozen_stage1=True")
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(stage1, head, train_loader, optimizer, device, precision, labels, args)
        val_metrics = evaluate(stage1, head, val_loader, device, precision, labels, args)
        current = float(val_metrics.get("median_px", float("inf")))
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reproj_loss": train_metrics["reproj_loss"],
            "train_residual_reg": train_metrics["residual_reg"],
            "val_loss": val_metrics.get("loss", float("nan")),
            "val_mean_px": val_metrics.get("mean_px", float("nan")),
            "val_median_px": val_metrics.get("median_px", float("nan")),
            "val_p90_px": val_metrics.get("p90_px", float("nan")),
            "val_max_px": val_metrics.get("max_px", float("nan")),
            "val_weak_median_px": val_metrics.get("weak_median_px", float("nan")),
            "val_weak_p90_px": val_metrics.get("weak_p90_px", float("nan")),
            "val_n_points": val_metrics.get("n_points", 0),
        }
        append_log(run_dir / "metrics.csv", row)
        print(
            f"epoch {epoch:04d}: train_loss={row['train_loss']:.4f} "
            f"val_median={row['val_median_px']:.3f}px val_p90={row['val_p90_px']:.3f}px "
            f"weak_median={row['val_weak_median_px']:.3f}px "
            f"val_n={row['val_n_points']}"
        )
        save_checkpoint(run_dir / "checkpoints" / "last.pt", head, args, labels, val_metrics, epoch)
        if current < best_metric:
            best_metric = current
            best_metrics = val_metrics
            save_checkpoint(run_dir / "checkpoints" / "best.pt", head, args, labels, val_metrics, epoch)
        if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
            save_checkpoint(run_dir / "checkpoints" / f"epoch{epoch:04d}.pt", head, args, labels, val_metrics, epoch)

    atomic_write_json(run_dir / "summary.json", {"best_metric": best_metric, "best_metrics": best_metrics})
    save_validation_overlays(stage1, head, dataset, val_indices, run_dir / "overlays", device, precision, labels, args)
    print(f"done: {run_dir}")
    if int(args.epochs) > 0:
        print(f"best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}")
    else:
        print("epochs=0: no training checkpoint was written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
