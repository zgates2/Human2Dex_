"""Dataset for Stage 2 wrist RGB -> MANO pose training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import WristImageTransform, read_rgb
from .utils import read_jsonl


class Stage2ManoDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        image_size: int = 448,
        train: bool = False,
        max_samples: int | None = None,
        require_fit_valid: bool = True,
    ) -> None:
        self.index_path = Path(index_path)
        rows = read_jsonl(self.index_path)
        if require_fit_valid:
            rows = [row for row in rows if bool(row.get("fit_valid", True))]
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.rows = rows
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

        pose_value = row.get("mano_full_pose", row["mano_pose"])
        mano_pose = np.asarray(pose_value, dtype=np.float32)
        global_orient = None
        if mano_pose.shape == (16, 3):
            mano_pose = mano_pose.reshape(48)
        if mano_pose.shape == (48,):
            global_orient = mano_pose[:3]
            mano_pose = mano_pose[3:]
        elif mano_pose.shape == (15, 3):
            mano_pose = mano_pose.reshape(45)
        if mano_pose.shape != (45,):
            raise ValueError(f"mano_pose must be (45,), (48,), (15, 3), or (16, 3), got {mano_pose.shape}")
        if global_orient is None:
            global_orient = np.asarray(row.get("global_orient", np.zeros(3)), dtype=np.float32)
        if global_orient.shape != (3,):
            raise ValueError(f"global_orient must be (3,), got {global_orient.shape}")

        mano_beta = np.asarray(row.get("mano_beta", np.zeros(10)), dtype=np.float32)
        if mano_beta.shape != (10,):
            raise ValueError(f"mano_beta must be (10,), got {mano_beta.shape}")

        fitted = np.asarray(row.get("mano_joints21_fit", joints21), dtype=np.float32)
        if fitted.shape != (21, 3):
            raise ValueError(f"mano_joints21_fit must be (21, 3), got {fitted.shape}")

        return {
            "image": result.image,
            "joints21": torch.from_numpy(joints21.copy()).float(),
            "valid21": torch.from_numpy(valid21.copy()).bool(),
            "mano_pose": torch.from_numpy(mano_pose.copy()).float(),
            "global_orient": torch.from_numpy(global_orient.copy()).float(),
            "mano_full_pose": torch.from_numpy(np.concatenate([global_orient, mano_pose]).copy()).float(),
            "mano_beta": torch.from_numpy(mano_beta.copy()).float(),
            "mano_joints21_fit": torch.from_numpy(fitted.copy()).float(),
            "fit_error_mm": float(row.get("fit_error_mm", 0.0)),
            "image_path": image_path,
            "sequence_id": row.get("sequence_id", ""),
            "frame_idx": int(row.get("frame_idx", idx)),
            "timestamp": float(row.get("timestamp", 0.0)),
            "scale": float(result.scale),
            "offset_xy": torch.tensor(result.offset_xy, dtype=torch.float32),
            "original_hw": torch.tensor(result.original_hw, dtype=torch.int64),
        }
