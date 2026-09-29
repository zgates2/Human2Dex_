"""Local DINOv3 ViT fallback for the bundled ViT-S/16 safetensors weights.

This is used only when the installed Transformers version cannot load
``model_type=dinov3_vit``. It matches the local weight names and keeps the
baseline runnable in existing project environments.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class LocalDinoV3ViT(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = int(config["hidden_size"])
        self.patch_size = int(config["patch_size"])
        self.num_register_tokens = int(config.get("num_register_tokens", 4))
        self.num_heads = int(config["num_attention_heads"])
        self.num_layers = int(config["num_hidden_layers"])
        self.rope_theta = float(config.get("rope_theta", 100.0))
        self.layer_norm_eps = float(config.get("layer_norm_eps", 1e-5))

        self.embeddings = DinoV3Embeddings(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            num_channels=int(config.get("num_channels", 3)),
            num_register_tokens=self.num_register_tokens,
        )
        self.layer = nn.ModuleList(
            [
                DinoV3Block(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    intermediate_size=int(config["intermediate_size"]),
                    layer_norm_eps=self.layer_norm_eps,
                    rope_theta=self.rope_theta,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    @classmethod
    def from_pretrained(cls, dino_dir: str | Path) -> "LocalDinoV3ViT":
        dino_dir = Path(dino_dir)
        with (dino_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(config)
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required for the local DINOv3 fallback") from exc
        state = load_file(str(dino_dir / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = {"embeddings.mask_token"}
        missing = [key for key in missing if key not in allowed_missing]
        if missing or unexpected:
            raise RuntimeError(
                "local DINOv3 fallback weight mismatch: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        return model

    def forward(self, pixel_values: torch.Tensor, **_: Any) -> SimpleNamespace:
        x, grid_hw = self.embeddings(pixel_values)
        for block in self.layer:
            x = block(x, grid_hw=grid_hw, num_special_tokens=1 + self.num_register_tokens)
        x = self.norm(x)
        return SimpleNamespace(last_hidden_state=x)


class DinoV3Embeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        num_channels: int,
        num_register_tokens: int,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, hidden_size))
        self.patch_embeddings = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        patches = self.patch_embeddings(pixel_values)
        grid_hw = (int(patches.shape[-2]), int(patches.shape[-1]))
        patches = patches.flatten(2).transpose(1, 2)
        batch_size = pixel_values.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        registers = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls, registers, patches], dim=1), grid_hw


class DinoV3Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = DinoV3Attention(hidden_size, num_heads, rope_theta)
        self.layer_scale1 = LayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = DinoV3MLP(hidden_size, intermediate_size)
        self.layer_scale2 = LayerScale(hidden_size)

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int], num_special_tokens: int) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x), grid_hw, num_special_tokens))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DinoV3Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, rope_theta: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, grid_hw: tuple[int, int], num_special_tokens: int) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q = self.q_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = apply_2d_rope(q, k, grid_hw, num_special_tokens, self.rope_theta)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).reshape(batch, tokens, self.hidden_size)
        return self.o_proj(attn)


class DinoV3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x), approximate="none"))


class LayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    grid_hw: tuple[int, int],
    num_special_tokens: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_count = grid_hw[0] * grid_hw[1]
    if patch_count <= 0:
        return q, k
    q_special, q_patch = q[:, :, :num_special_tokens], q[:, :, num_special_tokens:]
    k_special, k_patch = k[:, :, :num_special_tokens], k[:, :, num_special_tokens:]
    if q_patch.shape[2] != patch_count:
        return q, k

    cos, sin = rope_2d_cache(
        h=grid_hw[0],
        w=grid_hw[1],
        dim=q.shape[-1],
        theta=theta,
        device=q.device,
        dtype=q.dtype,
    )
    q_patch = rotate_with_cache(q_patch, cos, sin)
    k_patch = rotate_with_cache(k_patch, cos, sin)
    return torch.cat([q_special, q_patch], dim=2), torch.cat([k_special, k_patch], dim=2)


def rope_2d_cache(
    h: int,
    w: int,
    dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim % 4 != 0:
        raise ValueError("2D RoPE requires head_dim divisible by 4")
    half = dim // 2
    quarter = half // 2
    freqs = 1.0 / (theta ** (torch.arange(0, quarter, device=device, dtype=torch.float32) / max(1, quarter)))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y_angles = yy.reshape(-1, 1) * freqs.reshape(1, -1)
    x_angles = xx.reshape(-1, 1) * freqs.reshape(1, -1)
    angles = torch.cat([y_angles, x_angles], dim=1)
    angles = torch.repeat_interleave(angles, repeats=2, dim=1)
    cos = torch.cos(angles).to(dtype=dtype)[None, None, :, :]
    sin = torch.sin(angles).to(dtype=dtype)[None, None, :, :]
    return cos, sin


def rotate_with_cache(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)
