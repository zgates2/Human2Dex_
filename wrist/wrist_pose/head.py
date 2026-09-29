"""Regression head for 20 non-wrist joints."""

from __future__ import annotations

import torch
from torch import nn


class PoseRegressionHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 384,
        hidden_dims: tuple[int, int] = (1024, 512),
        dropout: float = 0.1,
        num_joints: int = 20,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            last = hidden
        layers.append(nn.Linear(last, num_joints * 3))
        self.net = nn.Sequential(*layers)
        self.num_joints = int(num_joints)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.net(features)
        return out.reshape(features.shape[0], self.num_joints, 3)
