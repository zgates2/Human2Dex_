"""DINOv3 ViT-S/16 + attention pooling + 20-joint regression head."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
from torch import nn

from .head import PoseRegressionHead
from .local_dinov3 import LocalDinoV3ViT
from .pooling import AttentionPooling


class WristDinoPoseModel(nn.Module):
    def __init__(
        self,
        dino_dir: str | Path,
        image_size: int = 448,
        feature_dim: int = 384,
        patch_size: int = 16,
        pooling_dropout: float = 0.0,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dino_dir = str(dino_dir)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.feature_dim = int(feature_dim)
        self.backbone = self._load_backbone(self.dino_dir)
        self.backbone_frozen = False
        self.pool = AttentionPooling(feature_dim, dropout=pooling_dropout)
        self.head = PoseRegressionHead(feature_dim, dropout=head_dropout, num_joints=20)

    @staticmethod
    def _load_backbone(dino_dir: str):
        try:
            from transformers import AutoModel
        except ImportError:
            AutoModel = None

        if AutoModel is not None:
            try:
                return AutoModel.from_pretrained(
                    dino_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            except Exception as exc:
                warnings.warn(
                    "Transformers could not load local DINOv3; using the local "
                    f"ViT-S/16 fallback. Original error: {exc}",
                    RuntimeWarning,
                )
        else:
            warnings.warn(
                "transformers is not installed; using the local DINOv3 ViT-S/16 fallback.",
                RuntimeWarning,
            )
        return LocalDinoV3ViT.from_pretrained(dino_dir)

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.backbone_frozen and torch.is_grad_enabled():
            self.backbone.eval()
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
        patch_tokens = self.extract_patch_tokens(outputs)
        pooled, attention = self.pool(patch_tokens)
        joints20 = self.head(pooled)
        result = {"joints20": joints20}
        if return_attention:
            result["attention"] = attention
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


def expand20_to21(joints20: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(
        joints20.shape[0],
        1,
        3,
        dtype=joints20.dtype,
        device=joints20.device,
    )
    return torch.cat([zeros, joints20], dim=1)


def find_transformer_blocks(module: nn.Module) -> list[nn.Module]:
    candidates = [
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("vit", "encoder", "layer"),
        ("vit", "encoder", "layers"),
        ("dinov3_vit", "encoder", "layer"),
        ("dinov3_vit", "encoder", "layers"),
        ("model", "encoder", "layer"),
        ("model", "encoder", "layers"),
        ("backbone", "encoder", "layer"),
        ("backbone", "encoder", "layers"),
        ("layer",),
        ("layers",),
        ("blocks",),
    ]
    for path in candidates:
        obj: Any = module
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok and isinstance(obj, (nn.ModuleList, list, tuple)) and len(obj) > 0:
            return list(obj)

    best: list[nn.Module] = []
    for _, child in module.named_modules():
        if isinstance(child, nn.ModuleList) and len(child) > len(best):
            if all(isinstance(item, nn.Module) for item in child):
                best = list(child)
    return best
