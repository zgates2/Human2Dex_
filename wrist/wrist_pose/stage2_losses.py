"""Losses for Stage 2 MANO pose baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .constants import HAND_BONES
from .losses import masked_smooth_l1


class Stage2ManoLoss(nn.Module):
    def __init__(
        self,
        pose_weight: float = 1.0,
        joint_weight: float = 1.0,
        bone_weight: float = 0.1,
        pose_prior_weight: float = 0.01,
        joint_beta: float = 0.005,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.pose_weight = float(pose_weight)
        self.joint_weight = float(joint_weight)
        self.bone_weight = float(bone_weight)
        self.pose_prior_weight = float(pose_prior_weight)
        self.joint_beta = float(joint_beta)
        self.eps = float(eps)
        self.register_buffer("bones", torch.tensor(HAND_BONES, dtype=torch.long), persistent=False)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        gt_pose: torch.Tensor,
        gt_joints21: torch.Tensor,
        valid21: torch.Tensor,
        gt_global_orient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_pose = outputs["mano_pose"]
        pred_joints21 = outputs["joints21"]
        valid21 = valid21.bool()

        if gt_global_orient is not None and "global_orient" in outputs:
            pred_pose_for_loss = torch.cat([outputs["global_orient"], pred_pose], dim=1)
            gt_pose_for_loss = torch.cat([gt_global_orient, gt_pose], dim=1)
        else:
            pred_pose_for_loss = pred_pose
            gt_pose_for_loss = gt_pose
        pose_loss = F.smooth_l1_loss(pred_pose_for_loss, gt_pose_for_loss, beta=0.05, reduction="mean")
        joint_loss = masked_smooth_l1(pred_joints21, gt_joints21, valid21, beta=self.joint_beta)
        bone_loss = self._bone_length_loss(pred_joints21, gt_joints21, valid21)
        pose_prior = torch.mean(pred_pose**2)
        total = (
            self.pose_weight * pose_loss
            + self.joint_weight * joint_loss
            + self.bone_weight * bone_loss
            + self.pose_prior_weight * pose_prior
        )
        return {
            "loss": total,
            "pose_loss": pose_loss.detach(),
            "joint_loss": joint_loss.detach(),
            "bone_loss": bone_loss.detach(),
            "pose_prior": pose_prior.detach(),
        }

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
        mask = valid21[:, src] & valid21[:, dst]
        return masked_smooth_l1(pred_len.unsqueeze(-1), gt_len.unsqueeze(-1), mask, beta=self.joint_beta)
