#!/usr/bin/env python3
"""
离线拟合 MANO pose。

用途
----
读取 `stage2_index.jsonl` 中的 pts21_mano，使用 manotorch MANO layer 拟合
hand pose 和 sequence 共享 beta，输出 `stage2_mano_fits.jsonl`。

常用命令
--------
检查 MANO 环境：
  python wrist/scripts/fit_mano.py --check-mano

小样本拟合：
  python wrist/scripts/fit_mano.py --limit-frames 256
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.constants import HAND_BONES  # noqa: E402
from wrist_pose.mano_layer import ManoDependencyError, ManoTorchLayer, check_mano_backend  # noqa: E402
from wrist_pose.utils import load_yaml, read_jsonl, resolve_config_paths, save_json, write_jsonl  # noqa: E402


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA, but torch.cuda.is_available() is False")
    return device


def masked_joint_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, beta: float) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, gt, beta=beta, reduction="none")
    mask = valid.unsqueeze(-1).to(loss.dtype)
    denom = mask.sum() * 3.0
    return (loss * mask).sum() / denom.clamp_min(1.0)


def bone_length_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, beta: float) -> torch.Tensor:
    bones = torch.tensor(HAND_BONES, dtype=torch.long, device=pred.device)
    src = bones[:, 0]
    dst = bones[:, 1]
    pred_len = torch.linalg.norm(pred[:, dst] - pred[:, src], dim=-1)
    gt_len = torch.linalg.norm(gt[:, dst] - gt[:, src], dim=-1)
    mask = valid[:, src] & valid[:, dst]
    loss = F.smooth_l1_loss(pred_len.unsqueeze(-1), gt_len.unsqueeze(-1), beta=beta, reduction="none")
    mask_f = mask.unsqueeze(-1).to(loss.dtype)
    return (loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def optimize_sequence(
    rows: list[dict[str, Any]],
    mano_layer: ManoTorchLayer,
    device: torch.device,
    steps: int,
    lr_pose: float,
    lr_beta: float,
    joint_beta: float,
    fit_error_threshold_mm: float,
    optimize_beta: bool,
) -> list[dict[str, Any]]:
    gt = torch.tensor([row["pts21_mano"] for row in rows], dtype=torch.float32, device=device)
    valid = torch.tensor([row.get("valid_mask", [True] * 21) for row in rows], dtype=torch.bool, device=device)
    n = gt.shape[0]
    full_pose = torch.zeros(n, 48, dtype=torch.float32, device=device, requires_grad=True)
    beta = torch.zeros(1, 10, dtype=torch.float32, device=device, requires_grad=optimize_beta)
    params = [{"params": [full_pose], "lr": lr_pose}]
    if optimize_beta:
        params.append({"params": [beta], "lr": lr_beta})
    optimizer = torch.optim.AdamW(params, weight_decay=0.0)

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        beta_batch = beta.expand(n, -1)
        out = mano_layer(full_pose, beta_batch)
        pred = out["joints21"]
        loss_joint = masked_joint_loss(pred, gt, valid, beta=joint_beta)
        loss_bone = bone_length_loss(pred, gt, valid, beta=joint_beta)
        loss_pose = torch.mean(full_pose**2)
        loss_shape = torch.mean(beta**2)
        loss = loss_joint + 0.1 * loss_bone + 0.001 * loss_pose + 0.001 * loss_shape
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        beta_batch = beta.expand(n, -1)
        out = mano_layer(full_pose, beta_batch)
        pred = out["joints21"]
        per_joint = torch.linalg.norm(pred - gt, dim=-1)
        fit_error = (per_joint * valid.to(per_joint.dtype)).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        full_pose_np = full_pose.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()[0]
        pred_np = pred.detach().cpu().numpy()

    fitted = []
    for row, row_pose, row_pred, row_error_m in zip(rows, full_pose_np, pred_np, fit_error.detach().cpu().numpy()):
        err_mm = float(row_error_m * 1000.0)
        out_row = dict(row)
        out_row["global_orient"] = row_pose[:3].astype("float32").tolist()
        out_row["mano_pose"] = row_pose[3:].astype("float32").tolist()
        out_row["mano_full_pose"] = row_pose.astype("float32").tolist()
        out_row["mano_beta"] = beta_np.astype("float32").tolist()
        out_row["mano_joints21_fit"] = row_pred.astype("float32").tolist()
        out_row["fit_error_mm"] = err_mm
        out_row["fit_valid"] = bool(err_mm <= fit_error_threshold_mm)
        fitted.append(out_row)
    return fitted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=WRIST_ROOT / "configs" / "stage2_mano.yaml")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr-pose", type=float, default=0.03)
    parser.add_argument("--lr-beta", type=float, default=0.003)
    parser.add_argument("--no-optimize-beta", action="store_true")
    parser.add_argument("--check-mano", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    resolve_config_paths(cfg, REPO_ROOT)
    mano_cfg = cfg.get("mano", {})
    try:
        backend_info = check_mano_backend(mano_cfg["model_dir"], side=str(mano_cfg.get("side", "right")))
        if args.check_mano:
            print(json.dumps(backend_info, indent=2))
            return 0
    except ManoDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    index_path = args.index or Path(cfg["data"]["stage2_index"])
    output_path = args.output or Path(cfg["data"]["stage2_fits"])
    rows = read_jsonl(index_path)
    if args.limit_frames is not None:
        rows = rows[: int(args.limit_frames)]
    if not rows:
        raise SystemExit(f"no rows found in {index_path}")

    device = choose_device(args.device)
    try:
        mano_layer = ManoTorchLayer(
            model_dir=mano_cfg["model_dir"],
            side=str(mano_cfg.get("side", "right")),
            use_pca=bool(mano_cfg.get("use_pca", False)),
            flat_hand_mean=bool(mano_cfg.get("flat_hand_mean", False)),
            output_scale=float(mano_cfg.get("output_scale", 1.0)),
        ).to(device)
    except ManoDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    mano_layer.eval()
    for param in mano_layer.parameters():
        param.requires_grad = False

    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row.get("sequence_id", ""))].append(row)

    all_fitted = []
    for sequence_id, seq_rows in by_sequence.items():
        print(f"[fit] sequence={sequence_id} frames={len(seq_rows)}")
        all_fitted.extend(
            optimize_sequence(
                rows=seq_rows,
                mano_layer=mano_layer,
                device=device,
                steps=args.steps,
                lr_pose=args.lr_pose,
                lr_beta=args.lr_beta,
                joint_beta=float(cfg["loss"].get("joint_beta", 0.005)),
                fit_error_threshold_mm=float(mano_cfg.get("fit_error_threshold_mm", 20.0)),
                optimize_beta=not args.no_optimize_beta,
            )
        )

    write_jsonl(output_path, all_fitted)
    errors = [float(row["fit_error_mm"]) for row in all_fitted]
    report = {
        "index": str(index_path),
        "output": str(output_path),
        "num_frames": len(all_fitted),
        "num_fit_valid": int(sum(bool(row["fit_valid"]) for row in all_fitted)),
        "fit_error_mm": {
            "mean": float(sum(errors) / max(1, len(errors))),
            "p50": float(torch.quantile(torch.tensor(errors), 0.50).item()) if errors else None,
            "p90": float(torch.quantile(torch.tensor(errors), 0.90).item()) if errors else None,
            "p95": float(torch.quantile(torch.tensor(errors), 0.95).item()) if errors else None,
        },
    }
    save_json(output_path.with_name("fit_report.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
