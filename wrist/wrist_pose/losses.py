"""Masked losses for wrist-centered 3D hand pose."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import HAND_BONES
from .model import expand20_to21


class WristPoseLoss(nn.Module):
    def __init__(
        self,
        joint_beta: float = 0.005,
        bone_length_weight: float = 0.1,
        bone_direction_weight: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.joint_beta = float(joint_beta)
        self.bone_length_weight = float(bone_length_weight)
        self.bone_direction_weight = float(bone_direction_weight)
        self.eps = float(eps)
        self.register_buffer("bones", torch.tensor(HAND_BONES, dtype=torch.long), persistent=False)

    def forward(
        self,
        pred20: torch.Tensor,
        gt20: torch.Tensor,
        valid20: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid20 = valid20.bool()
        joint_loss = masked_smooth_l1(pred20, gt20, valid20, beta=self.joint_beta)

        pred21 = expand20_to21(pred20)
        gt21 = expand20_to21(gt20)
        root_valid = torch.ones(
            valid20.shape[0],
            1,
            dtype=torch.bool,
            device=valid20.device,
        )
        valid21 = torch.cat([root_valid, valid20], dim=1)

        bone_length_loss = self._bone_length_loss(pred21, gt21, valid21)
        bone_direction_loss = self._bone_direction_loss(pred21, gt21, valid21)
        total = (
            joint_loss
            + self.bone_length_weight * bone_length_loss
            + self.bone_direction_weight * bone_direction_loss
        )
        return {
            "loss": total,
            "joint_loss": joint_loss.detach(),
            "bone_length_loss": bone_length_loss.detach(),
            "bone_direction_loss": bone_direction_loss.detach(),
        }

    def _bone_mask(self, valid21: torch.Tensor) -> torch.Tensor:
        src = self.bones[:, 0]
        dst = self.bones[:, 1]
        return valid21[:, src] & valid21[:, dst]

    def _bone_length_loss(
        self,
        pred21: torch.Tensor,
        gt21: torch.Tensor,
        valid21: torch.Tensor,
    ) -> torch.Tensor:
        src = self.bones[:, 0]
        dst = self.bones[:, 1]
        pred_vec = pred21[:, dst] - pred21[:, src]
        gt_vec = gt21[:, dst] - gt21[:, src]
        pred_len = torch.linalg.norm(pred_vec, dim=-1)
        gt_len = torch.linalg.norm(gt_vec, dim=-1)
        mask = self._bone_mask(valid21)
        return masked_smooth_l1(pred_len.unsqueeze(-1), gt_len.unsqueeze(-1), mask, beta=self.joint_beta)

    def _bone_direction_loss(
        self,
        pred21: torch.Tensor,
        gt21: torch.Tensor,
        valid21: torch.Tensor,
    ) -> torch.Tensor:
        src = self.bones[:, 0]
        dst = self.bones[:, 1]
        pred_vec = pred21[:, dst] - pred21[:, src]
        gt_vec = gt21[:, dst] - gt21[:, src]
        pred_len = torch.linalg.norm(pred_vec, dim=-1, keepdim=True)
        gt_len = torch.linalg.norm(gt_vec, dim=-1, keepdim=True)
        mask = self._bone_mask(valid21) & (gt_len.squeeze(-1) > self.eps)
        pred_dir = pred_vec / pred_len.clamp_min(self.eps)
        gt_dir = gt_vec / gt_len.clamp_min(self.eps)
        cosine = torch.sum(pred_dir * gt_dir, dim=-1).clamp(-1.0, 1.0)
        loss = 1.0 - cosine
        if not torch.any(mask):
            return loss.sum() * 0.0
        return loss[mask].mean()


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=loss.dtype)
    denom = mask.sum() * loss.shape[-1]
    if denom <= 0:
        return loss.sum() * 0.0
    return (loss * mask).sum() / denom.clamp_min(1.0)
