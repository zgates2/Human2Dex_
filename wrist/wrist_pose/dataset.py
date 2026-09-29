"""Dataset for DexUMI wrist RGB to hand-local 3D pseudo-GT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import WristImageTransform, read_rgb
from .utils import read_jsonl


class WristPoseDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        image_size: int = 448,
        train: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.rows = read_jsonl(self.index_path)
        if max_samples is not None:
            self.rows = self.rows[: int(max_samples)]
        self.transform = WristImageTransform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        image_path = str(row["image_path"])
        rgb = read_rgb(image_path)
        result = self.transform(rgb)

        joints21 = np.asarray(row["pts21_mano"], dtype=np.float32)
        if joints21.shape != (21, 3):
            raise ValueError(f"pts21_mano must be (21, 3), got {joints21.shape}")
        valid21 = np.asarray(row.get("valid_mask", [True] * 21), dtype=bool)
        if valid21.shape != (21,):
            raise ValueError(f"valid_mask must be (21,), got {valid21.shape}")

        return {
            "image": result.image,
            "joints20": torch.from_numpy(joints21[1:].copy()).float(),
            "joints21": torch.from_numpy(joints21.copy()).float(),
            "valid20": torch.from_numpy(valid21[1:].copy()).bool(),
            "valid21": torch.from_numpy(valid21.copy()).bool(),
            "image_path": image_path,
            "sequence_id": row.get("sequence_id", ""),
            "frame_idx": int(row.get("frame_idx", idx)),
            "timestamp": float(row.get("timestamp", 0.0)),
            "scale": float(result.scale),
            "offset_xy": torch.tensor(result.offset_xy, dtype=torch.float32),
            "original_hw": torch.tensor(result.original_hw, dtype=torch.int64),
        }
