"""Evaluation metrics for 20 predicted joints and 21-joint visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .constants import HAND_BONES, JOINT_NAMES
from .model import expand20_to21


@dataclass
class PoseMetricAccumulator:
    total_error_m: float = 0.0
    total_count: int = 0
    pck10_count: int = 0
    pck20_count: int = 0
    per_joint_error_m: np.ndarray = field(default_factory=lambda: np.zeros(21, dtype=np.float64))
    per_joint_count: np.ndarray = field(default_factory=lambda: np.zeros(21, dtype=np.int64))
    bone_length_error_m: float = 0.0
    bone_count: int = 0
    pa_error_m: float = 0.0
    pa_count: int = 0

    @torch.no_grad()
    def update(self, pred20: torch.Tensor, gt20: torch.Tensor, valid20: torch.Tensor) -> None:
        pred20 = pred20.detach().float().cpu()
        gt20 = gt20.detach().float().cpu()
        valid20 = valid20.detach().bool().cpu()
        err20 = torch.linalg.norm(pred20 - gt20, dim=-1)
        valid_err = err20[valid20]
        if valid_err.numel() > 0:
            self.total_error_m += float(valid_err.sum().item())
            self.total_count += int(valid_err.numel())
            self.pck10_count += int((valid_err < 0.010).sum().item())
            self.pck20_count += int((valid_err < 0.020).sum().item())

        pred21 = expand20_to21(pred20)
        gt21 = expand20_to21(gt20)
        root_valid = torch.ones(valid20.shape[0], 1, dtype=torch.bool)
        valid21 = torch.cat([root_valid, valid20], dim=1)

        err21 = torch.linalg.norm(pred21 - gt21, dim=-1).numpy()
        valid21_np = valid21.numpy()
        for joint_idx in range(21):
            mask = valid21_np[:, joint_idx]
            if np.any(mask):
                self.per_joint_error_m[joint_idx] += float(err21[mask, joint_idx].sum())
                self.per_joint_count[joint_idx] += int(np.count_nonzero(mask))

        self._update_bone_length(pred21.numpy(), gt21.numpy(), valid21_np)
        self._update_pa_mpjpe(pred21.numpy(), gt21.numpy(), valid21_np)

    def _update_bone_length(self, pred21: np.ndarray, gt21: np.ndarray, valid21: np.ndarray) -> None:
        for src, dst in HAND_BONES:
            mask = valid21[:, src] & valid21[:, dst]
            if not np.any(mask):
                continue
            pred_len = np.linalg.norm(pred21[mask, dst] - pred21[mask, src], axis=-1)
            gt_len = np.linalg.norm(gt21[mask, dst] - gt21[mask, src], axis=-1)
            err = np.abs(pred_len - gt_len)
            self.bone_length_error_m += float(err.sum())
            self.bone_count += int(err.shape[0])

    def _update_pa_mpjpe(self, pred21: np.ndarray, gt21: np.ndarray, valid21: np.ndarray) -> None:
        for pred, gt, mask in zip(pred21, gt21, valid21):
            if np.count_nonzero(mask) < 3:
                continue
            aligned = procrustes_align(pred[mask], gt[mask])
            err = np.linalg.norm(aligned - gt[mask], axis=-1)
            self.pa_error_m += float(err.sum())
            self.pa_count += int(err.shape[0])

    def compute(self) -> dict[str, Any]:
        mpjpe_mm = safe_div(self.total_error_m, self.total_count) * 1000.0
        pck10 = safe_div(self.pck10_count, self.total_count)
        pck20 = safe_div(self.pck20_count, self.total_count)
        per_joint = {}
        for name, total, count in zip(JOINT_NAMES, self.per_joint_error_m, self.per_joint_count):
            per_joint[name] = safe_div(float(total), int(count)) * 1000.0
        return {
            "mpjpe_mm": mpjpe_mm,
            "pck_10mm": pck10,
            "pck_20mm": pck20,
            "bone_length_error_mm": safe_div(self.bone_length_error_m, self.bone_count) * 1000.0,
            "pa_mpjpe_mm": safe_div(self.pa_error_m, self.pa_count) * 1000.0,
            "num_valid_joints": self.total_count,
            "per_joint_mpjpe_mm": per_joint,
        }


def safe_div(num: float, denom: int | float) -> float:
    return float(num) / float(denom) if denom else float("nan")


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Similarity-align source to target using Umeyama alignment."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    src_mean = source.mean(axis=0, keepdims=True)
    tgt_mean = target.mean(axis=0, keepdims=True)
    src = source - src_mean
    tgt = target - tgt_mean
    src_var = np.sum(src**2)
    if src_var < 1e-12:
        return source.copy()
    cov = src.T @ tgt
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    scale = float(np.sum(s) / src_var)
    return scale * src @ r.T + tgt_mean
