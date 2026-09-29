"""Stage 2 inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .stage2_model import WristDinoManoPoseModel, infer_pose_output_dim_from_state_dict
from .transforms import WristImageTransform
from .utils import load_model_state_compatible, load_yaml, resolve_config_paths


def load_stage2_model(
    checkpoint_path: str | Path,
    repo_root: str | Path,
    device: torch.device,
    config_path: str | Path | None = None,
) -> tuple[WristDinoManoPoseModel, WristImageTransform, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path is not None:
        cfg = load_yaml(config_path)
    else:
        cfg = checkpoint.get("config")
        if not isinstance(cfg, dict):
            cfg = load_yaml(Path(repo_root) / "wrist" / "configs" / "stage2_mano.yaml")
    resolve_config_paths(cfg, repo_root)
    mano = cfg["mano"]
    pose_output_dim = infer_pose_output_dim_from_state_dict(checkpoint["model"])
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
        pose_output_dim=pose_output_dim,
    )
    load_model_state_compatible(model, checkpoint["model"], strict=True)
    model.to(device).eval()
    transform = WristImageTransform(image_size=int(cfg["data"]["image_size"]), train=False)
    return model, transform, cfg


@torch.no_grad()
def predict_pts21_mano_stage2(
    model: WristDinoManoPoseModel,
    transform: WristImageTransform,
    rgb: np.ndarray,
    device: torch.device,
    precision: str = "fp32",
) -> np.ndarray:
    image = transform(rgb).image.unsqueeze(0).to(device, non_blocking=True)
    if device.type == "cuda" and precision != "fp32":
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        ctx = torch.autocast(device_type="cpu", enabled=False)
    with ctx:
        joints21 = model(image)["joints21"]
    return joints21[0].detach().cpu().float().numpy().astype(np.float32)
