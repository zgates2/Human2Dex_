"""Metrics for Stage 2 direct 21-joint MANO outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .constants import FINGERTIP_INDICES, HAND_BONES, JOINT_NAMES
from .metrics import procrustes_align, safe_div


@dataclass
class Joint21MetricAccumulator:
    total_error_m: float = 0.0
    total_count: int = 0
    pck10_count: int = 0
    pck20_count: int = 0
    per_joint_error_m: np.ndarray = field(default_factory=lambda: np.zeros(21, dtype=np.float64))
    per_joint_count: np.ndarray = field(default_factory=lambda: np.zeros(21, dtype=np.int64))
    fingertip_error_m: float = 0.0
    fingertip_count: int = 0
    bone_length_error_m: float = 0.0
    bone_count: int = 0
    pa_error_m: float = 0.0
    pa_count: int = 0

    @torch.no_grad()
    def update(self, pred21: torch.Tensor, gt21: torch.Tensor, valid21: torch.Tensor) -> None:
        pred21 = pred21.detach().float().cpu()
        gt21 = gt21.detach().float().cpu()
        valid21 = valid21.detach().bool().cpu()
        err21_t = torch.linalg.norm(pred21 - gt21, dim=-1)
        valid_err = err21_t[valid21]
        if valid_err.numel() > 0:
            self.total_error_m += float(valid_err.sum().item())
            self.total_count += int(valid_err.numel())
            self.pck10_count += int((valid_err < 0.010).sum().item())
            self.pck20_count += int((valid_err < 0.020).sum().item())

        tip_mask = valid21[:, FINGERTIP_INDICES]
        tip_err = err21_t[:, FINGERTIP_INDICES][tip_mask]
        if tip_err.numel() > 0:
            self.fingertip_error_m += float(tip_err.sum().item())
            self.fingertip_count += int(tip_err.numel())

        err21 = err21_t.numpy()
        valid_np = valid21.numpy()
        pred_np = pred21.numpy()
        gt_np = gt21.numpy()
        for joint_idx in range(21):
            mask = valid_np[:, joint_idx]
            if np.any(mask):
                self.per_joint_error_m[joint_idx] += float(err21[mask, joint_idx].sum())
                self.per_joint_count[joint_idx] += int(np.count_nonzero(mask))
        for src, dst in HAND_BONES:
            mask = valid_np[:, src] & valid_np[:, dst]
            if not np.any(mask):
                continue
            pred_len = np.linalg.norm(pred_np[mask, dst] - pred_np[mask, src], axis=-1)
            gt_len = np.linalg.norm(gt_np[mask, dst] - gt_np[mask, src], axis=-1)
            err = np.abs(pred_len - gt_len)
            self.bone_length_error_m += float(err.sum())
            self.bone_count += int(err.shape[0])
        for pred, gt, mask in zip(pred_np, gt_np, valid_np):
            if np.count_nonzero(mask) < 3:
                continue
            aligned = procrustes_align(pred[mask], gt[mask])
            err = np.linalg.norm(aligned - gt[mask], axis=-1)
            self.pa_error_m += float(err.sum())
            self.pa_count += int(err.shape[0])

    def compute(self) -> dict[str, Any]:
        per_joint = {}
        for name, total, count in zip(JOINT_NAMES, self.per_joint_error_m, self.per_joint_count):
            per_joint[name] = safe_div(float(total), int(count)) * 1000.0
        return {
            "mpjpe_mm": safe_div(self.total_error_m, self.total_count) * 1000.0,
            "pck_10mm": safe_div(self.pck10_count, self.total_count),
            "pck_20mm": safe_div(self.pck20_count, self.total_count),
            "fingertip_mpjpe_mm": safe_div(self.fingertip_error_m, self.fingertip_count) * 1000.0,
            "bone_length_error_mm": safe_div(self.bone_length_error_m, self.bone_count) * 1000.0,
            "pa_mpjpe_mm": safe_div(self.pa_error_m, self.pa_count) * 1000.0,
            "num_valid_joints": self.total_count,
            "per_joint_mpjpe_mm": per_joint,
        }


def metric_state_tensor(metric: Joint21MetricAccumulator, device: torch.device) -> torch.Tensor:
    values = [
        metric.total_error_m,
        float(metric.total_count),
        float(metric.pck10_count),
        float(metric.pck20_count),
        *metric.per_joint_error_m.tolist(),
        *metric.per_joint_count.astype("float64").tolist(),
        metric.fingertip_error_m,
        float(metric.fingertip_count),
        metric.bone_length_error_m,
        float(metric.bone_count),
        metric.pa_error_m,
        float(metric.pa_count),
    ]
    return torch.tensor(values, dtype=torch.float64, device=device)


def compute_metrics_from_state(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().cpu().numpy()
    idx = 0
    metric = Joint21MetricAccumulator()
    metric.total_error_m = float(values[idx]); idx += 1
    metric.total_count = int(round(float(values[idx]))); idx += 1
    metric.pck10_count = int(round(float(values[idx]))); idx += 1
    metric.pck20_count = int(round(float(values[idx]))); idx += 1
    metric.per_joint_error_m = values[idx : idx + 21].astype("float64"); idx += 21
    metric.per_joint_count = values[idx : idx + 21].astype("int64"); idx += 21
    metric.fingertip_error_m = float(values[idx]); idx += 1
    metric.fingertip_count = int(round(float(values[idx]))); idx += 1
    metric.bone_length_error_m = float(values[idx]); idx += 1
    metric.bone_count = int(round(float(values[idx]))); idx += 1
    metric.pa_error_m = float(values[idx]); idx += 1
    metric.pa_count = int(round(float(values[idx]))); idx += 1
    return metric.compute()
