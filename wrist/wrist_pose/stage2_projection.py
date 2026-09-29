"""Projection-head utilities for wrist MANO21 -> RGB-space 2D skeletons.

This module intentionally contains only the reusable projection math.  Training
scripts, offline PKL inference, and augmentation rendering should all use the
same definitions so a projection-head checkpoint has one stable contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


FINGERTIP6_LABELS = (
    "palm_center",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)
GRASP_POCKET_LABELS = ("grasp_pocket",)
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
SUPPORTED_LABELS = tuple(dict.fromkeys((*FINGERTIP6_LABELS, *MANO21_LABELS, *GRASP_POCKET_LABELS)))
PROJECTION_HEAD_TYPES = ("weak", "residual2d", "direct_pocket")


@dataclass(frozen=True)
class ProjectionHeadInfo:
    checkpoint: str
    labels: list[str]
    min_scale: float
    max_scale: float
    init_scale: float
    projection_head_type: str
    residual_scale: float
    epoch: int | None
    metrics: dict[str, Any]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dim: int = 512,
        joint_embed_dim: int = 128,
        dropout: float = 0.1,
        init_scale: float = 140.0,
        projection_head_type: str = "weak",
        residual_scale: float = 32.0,
    ) -> None:
        super().__init__()
        projection_head_type = str(projection_head_type)
        if projection_head_type not in PROJECTION_HEAD_TYPES:
            raise ValueError(f"unsupported projection_head_type={projection_head_type!r}; expected {PROJECTION_HEAD_TYPES}")
        self.projection_head_type = projection_head_type
        self.residual_scale = float(residual_scale)
        direct_pocket = projection_head_type == "direct_pocket"
        output_dim = 3 if direct_pocket else 9 + (21 * 2 if projection_head_type == "residual2d" else 0)
        self.joint_encoder = None if direct_pocket else nn.Sequential(
            nn.LayerNorm(21 * 3),
            nn.Linear(21 * 3, 256),
            nn.GELU(),
            nn.Linear(256, joint_embed_dim),
            nn.GELU(),
        )
        input_dim = feature_dim if direct_pocket else feature_dim + joint_embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
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
            if direct_pocket:
                # Pixel centre, with deliberately unconfident initial probability.
                final.bias[2] = -1.0
            else:
                final.bias[:6] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                final.bias[6] = math.log(max(float(init_scale), 1e-4))

    def forward(self, patch_tokens: torch.Tensor, joints21_norm: torch.Tensor | None = None) -> torch.Tensor:
        image_feature = patch_tokens.mean(dim=1)
        if self.projection_head_type == "direct_pocket":
            return self.net(image_feature)
        if joints21_norm is None or self.joint_encoder is None:
            raise ValueError("weak/residual projection head requires normalized 21-point joints")
        joint_feature = self.joint_encoder(joints21_norm.flatten(1))
        return self.net(torch.cat([image_feature, joint_feature], dim=-1))


def infer_projection_head_type(state_dict: dict[str, torch.Tensor], config: dict[str, Any]) -> str:
    value = config.get("projection_head_type") or config.get("head_type")
    if value:
        value = str(value)
        if value not in PROJECTION_HEAD_TYPES:
            raise ValueError(f"unsupported projection_head_type in checkpoint: {value!r}")
        return value
    bias = state_dict.get("net.7.bias")
    if bias is None:
        bias = state_dict.get("net.8.bias")
    if bias is not None and int(bias.numel()) == 9 + 21 * 2:
        return "residual2d"
    if bias is not None and int(bias.numel()) == 3:
        return "direct_pocket"
    return "weak"


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


def decode_direct_pocket(raw: torch.Tensor, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode RGB-only functional pocket location and a learned confidence.

    The point is directly supervised in image pixels.  It does not use MANO or
    PICO joints, precisely so their geometric bias cannot move the G/L anchor.
    """
    if raw.ndim != 2 or raw.shape[-1] != 3:
        raise ValueError(f"direct_pocket head expects [B,3], got {tuple(raw.shape)}")
    uv = torch.sigmoid(raw[:, :2]) * float(image_size)
    confidence = torch.sigmoid(raw[:, 2])
    return uv, confidence


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


def parse_projection_labels(value: str | Sequence[str] | None) -> list[str]:
    if value is None or value == "":
        return list(DEFAULT_LABELS)
    if isinstance(value, str):
        text = value.strip()
        if text in {"fingertips6", "tips6"}:
            return list(FINGERTIP6_LABELS)
        if text in {"mano21", "all21", "full21"}:
            return list(MANO21_LABELS)
        labels = [item.strip() for item in text.split(",") if item.strip()]
    else:
        labels = [str(item).strip() for item in value if str(item).strip()]
        if labels == ["mano21"] or labels == ["all21"] or labels == ["full21"]:
            return list(MANO21_LABELS)
        if labels == ["fingertips6"] or labels == ["tips6"]:
            return list(FINGERTIP6_LABELS)
    if not labels:
        raise ValueError("projection labels cannot be empty")
    missing = [label for label in labels if label not in SUPPORTED_LABELS]
    if missing:
        raise ValueError(f"unsupported projection labels: {missing}; expected subset/profile of {SUPPORTED_LABELS}")
    return labels


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


def make_anchor_uv(uv21: torch.Tensor, labels: Sequence[str]) -> torch.Tensor:
    anchors = []
    for label in labels:
        label = str(label)
        if label == "palm_center":
            anchors.append(uv21[:, PALM_JOINTS, :].mean(dim=1))
        elif label in LABEL_TO_JOINT:
            anchors.append(uv21[:, LABEL_TO_JOINT[label], :])
        else:
            raise KeyError(f"unsupported projection label: {label}")
    return torch.stack(anchors, dim=1)


def forward_projection_from_stage1_outputs(
    outputs: dict[str, torch.Tensor],
    head: ProjectionHead,
    labels: Sequence[str],
    image_size: int,
    min_scale: float,
    max_scale: float,
) -> dict[str, torch.Tensor]:
    if "patch_tokens" not in outputs:
        raise KeyError("Stage1 outputs must include patch_tokens; call model(..., return_tokens=True)")
    patch_tokens = outputs["patch_tokens"].detach()
    if getattr(head, "projection_head_type", "weak") == "direct_pocket":
        raw = head(patch_tokens.float(), None)
        pocket_uv, pocket_confidence = decode_direct_pocket(raw, image_size=int(image_size))
        return {
            "raw": raw,
            "grasp_pocket_uv": pocket_uv,
            "grasp_pocket_confidence": pocket_confidence,
        }
    joints21 = outputs["joints21"].detach()
    joints21_norm, _center, hand_scale = normalize_joints21(joints21.float())
    raw = head(patch_tokens.float(), joints21_norm)
    base_raw, residual_raw = split_projection_output(
        raw,
        projection_head_type=getattr(head, "projection_head_type", "weak"),
    )
    rot, proj_scale, translation = decode_projection(
        base_raw,
        image_size=int(image_size),
        min_scale=float(min_scale),
        max_scale=float(max_scale),
    )
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


def transform_model_uv_to_original(
    uv: np.ndarray,
    scale: float,
    offset_xy: Sequence[float],
) -> np.ndarray:
    arr = np.asarray(uv, dtype=np.float32).copy()
    offset = np.asarray(offset_xy, dtype=np.float32).reshape(1, 2)
    arr[:, 0:2] = (arr[:, 0:2] - offset) / max(float(scale), 1e-8)
    return arr.astype(np.float32, copy=False)


def uv_valid_mask(uv: np.ndarray, original_hw: Sequence[int] | None = None) -> np.ndarray:
    arr = np.asarray(uv, dtype=np.float32)
    valid = np.isfinite(arr).all(axis=-1)
    if original_hw is not None:
        h, w = int(original_hw[0]), int(original_hw[1])
        valid = valid & (arr[..., 0] >= 0) & (arr[..., 0] < w) & (arr[..., 1] >= 0) & (arr[..., 1] < h)
    return valid.astype(bool, copy=False)


def load_projection_head(
    checkpoint_path: str | Path,
    feature_dim: int,
    device: torch.device,
) -> tuple[ProjectionHead, ProjectionHeadInfo]:
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("projection_head"), dict):
        raise ValueError(f"projection checkpoint does not contain projection_head: {path}")
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    labels = parse_projection_labels(checkpoint.get("labels") or config.get("labels"))
    init_scale = float(config.get("init_proj_scale", 140.0))
    min_scale = float(config.get("min_proj_scale", 10.0))
    max_scale = float(config.get("max_proj_scale", 800.0))
    projection_head_type = infer_projection_head_type(checkpoint["projection_head"], config)
    residual_scale = float(config.get("residual_scale", 32.0))
    head = ProjectionHead(
        feature_dim=int(feature_dim),
        init_scale=init_scale,
        projection_head_type=projection_head_type,
        residual_scale=residual_scale,
    )
    head.load_state_dict(checkpoint["projection_head"], strict=True)
    head.to(device).eval()
    info = ProjectionHeadInfo(
        checkpoint=str(path),
        labels=labels,
        min_scale=min_scale,
        max_scale=max_scale,
        init_scale=init_scale,
        projection_head_type=projection_head_type,
        residual_scale=residual_scale,
        epoch=int(checkpoint["epoch"]) if checkpoint.get("epoch") is not None else None,
        metrics=checkpoint.get("metrics") if isinstance(checkpoint.get("metrics"), dict) else {},
    )
    return head, info
