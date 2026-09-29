"""DINOv3 wrist RGB -> MANO pose -> MANO joints model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .mano_layer import ManoTorchLayer
from .model import WristDinoPoseModel, find_transformer_blocks


def infer_pose_output_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    suffixes = (
        "pose_head.head.8.weight",
        "module.pose_head.head.8.weight",
    )
    for suffix in suffixes:
        tensor = state_dict.get(suffix)
        if tensor is not None:
            dim = int(tensor.shape[0])
            if dim in (45, 48):
                return dim
    for key, tensor in state_dict.items():
        clean = key.removeprefix("module.")
        if clean == "pose_head.head.8.weight":
            dim = int(tensor.shape[0])
            if dim in (45, 48):
                return dim
    return 48


class ManoPoseHead(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        num_queries: int = 6,
        decoder_layers: int = 2,
        decoder_heads: int = 6,
        dropout: float = 0.1,
        pose_dim: int = 48,
    ) -> None:
        super().__init__()
        if pose_dim not in (45, 48):
            raise ValueError(f"pose_dim must be 45 or 48, got {pose_dim}")
        self.pose_dim = int(pose_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=decoder_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(num_queries * dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.pose_dim),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch = patch_tokens.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, patch_tokens)
        decoded = self.norm(decoded)
        return self.head(decoded.flatten(1))


class WristDinoManoPoseModel(nn.Module):
    def __init__(
        self,
        dino_dir: str | Path,
        mano_model_dir: str | Path,
        image_size: int = 448,
        feature_dim: int = 384,
        patch_size: int = 16,
        head_dropout: float = 0.1,
        decoder_layers: int = 2,
        decoder_heads: int = 6,
        mano_side: str = "right",
        mano_use_pca: bool = False,
        mano_flat_hand_mean: bool = False,
        mano_output_scale: float = 1.0,
        pose_output_dim: int = 48,
    ) -> None:
        super().__init__()
        self.dino_dir = str(dino_dir)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.feature_dim = int(feature_dim)
        self.backbone = WristDinoPoseModel._load_backbone(self.dino_dir)
        self.backbone_frozen = False
        self.pose_head = ManoPoseHead(
            dim=feature_dim,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            dropout=head_dropout,
            pose_dim=pose_output_dim,
        )
        self.mano_layer = ManoTorchLayer(
            model_dir=mano_model_dir,
            side=mano_side,
            use_pca=mano_use_pca,
            flat_hand_mean=mano_flat_hand_mean,
            output_scale=mano_output_scale,
        )
        self.register_buffer("default_beta", torch.zeros(10), persistent=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        mano_beta: torch.Tensor | None = None,
        return_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.backbone_frozen and torch.is_grad_enabled():
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
        patch_tokens = self.extract_patch_tokens(outputs)
        mano_output = self.pose_head(patch_tokens)
        if mano_output.shape[1] == 48:
            mano_full_pose = mano_output
            global_orient = mano_full_pose[:, :3]
            mano_pose = mano_full_pose[:, 3:]
        elif mano_output.shape[1] == 45:
            global_orient = torch.zeros(
                mano_output.shape[0],
                3,
                dtype=mano_output.dtype,
                device=mano_output.device,
            )
            mano_pose = mano_output
            mano_full_pose = torch.cat([global_orient, mano_pose], dim=1)
        else:
            raise ValueError(f"MANO pose head must output 45 or 48 dims, got {mano_output.shape[1]}")
        if mano_beta is None:
            mano_beta = self.default_beta.unsqueeze(0).expand(pixel_values.shape[0], -1)
        mano = self.mano_layer(mano_pose, mano_beta, global_orient=global_orient)
        result = {
            "mano_pose": mano_pose,
            "global_orient": global_orient,
            "mano_full_pose": mano_full_pose,
            "mano_beta": mano_beta,
            "joints21": mano["joints21"],
            "vertices": mano["vertices"],
        }
        if return_tokens:
            result["patch_tokens"] = patch_tokens
        return result

    def extract_patch_tokens(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state"):
            tokens = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            tokens = outputs["last_hidden_state"]
        elif isinstance(outputs, (tuple, list)) and outputs:
            tokens = outputs[0]
        else:
            raise ValueError("backbone outputs do not contain last_hidden_state")
        expected = (self.image_size // self.patch_size) ** 2
        if tokens.shape[1] < expected:
            raise ValueError(
                f"not enough tokens for {self.image_size} input: got {tokens.shape[1]}, "
                f"expected at least {expected}"
            )
        return tokens[:, -expected:, :]

    def set_default_beta(self, beta: torch.Tensor) -> None:
        beta = beta.detach().float().reshape(10).to(self.default_beta.device)
        self.default_beta.copy_(beta)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone_frozen = True
        self.backbone.eval()

    def unfreeze_backbone_last_blocks(self, num_blocks: int = 4) -> int:
        self.freeze_backbone()
        blocks = find_transformer_blocks(self.backbone)
        if not blocks:
            self.backbone_frozen = True
            return 0
        selected = blocks[-int(num_blocks) :]
        for block in selected:
            for param in block.parameters():
                param.requires_grad = True
        self.backbone_frozen = False
        return len(selected)
