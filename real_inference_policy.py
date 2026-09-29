#!/usr/bin/env python3
"""Checkpoint validation and policy construction."""

from __future__ import annotations

import pathlib
import zipfile

import dill
import hydra
import torch

from real_inference_actions import disable_pretrained_download
from real_inference_config import ROOT

def validate_checkpoint_metadata(ckpt_path: str) -> pathlib.Path:
    """Fail early when a PyTorch zip checkpoint is missing or corrupted."""
    path = pathlib.Path(ckpt_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Checkpoint is empty: {path}")

    try:
        with zipfile.ZipFile(path) as archive:
            data_member = next(
                (
                    name
                    for name in archive.namelist()
                    if name == "data.pkl" or name.endswith("/data.pkl")
                ),
                None,
            )
            if data_member is None:
                return path
            with archive.open(data_member) as f:
                while f.read(1024 * 1024):
                    pass
    except zipfile.BadZipFile:
        raise RuntimeError(
            f"Checkpoint appears corrupted: {path}. "
            f"Re-copy or regenerate it, then verify with: unzip -t {path}"
        ) from None
    return path

def load_policy(ckpt_path: str, device: torch.device):
    """从 checkpoint 加载策略模型。"""
    ckpt_path = validate_checkpoint_metadata(ckpt_path)
    try:
        payload = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
    except EOFError:
        raise RuntimeError(
            f"Failed to load checkpoint payload from {ckpt_path}. "
            "The file is likely incomplete or corrupted; re-copy or regenerate it."
        ) from None
    cfg = payload["cfg"]
    disable_pretrained_download(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(ROOT / "tmp_ws"))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model
    policy.eval().to(device)
    return policy, cfg
