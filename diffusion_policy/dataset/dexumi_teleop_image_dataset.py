from __future__ import annotations

import copy
import os
from typing import Dict

import cv2
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _downsample_mask(mask: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    if max_n is None or int(max_n) <= 0:
        return mask
    true_idxs = np.flatnonzero(mask)
    if len(true_idxs) <= int(max_n):
        return mask
    rng = np.random.default_rng(seed)
    keep = rng.choice(true_idxs, size=int(max_n), replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out[keep] = True
    return out


class DexUmiTeleopImageDataset(BaseImageDataset):
    """Dataset for DexUMI replay_buffer.zarr produced by tools/convert_pkl_to_training_zarr.py."""

    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            horizon=16,
            pad_before=1,
            pad_after=15,
            n_obs_steps=None,
            n_latency_steps=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            episode_mask=None,
        ):
        super().__init__()
        self.shape_meta = shape_meta
        self.dataset_path = os.path.expanduser(dataset_path)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else int(pad_before) + 1
        self.n_latency_steps = int(n_latency_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.max_train_episodes = max_train_episodes

        zarr_path = os.path.join(self.dataset_path, "replay_buffer.zarr")
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

        self.rgb_keys = []
        self.lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        required = set(self.rgb_keys + self.lowdim_keys + ["action"])
        missing = sorted(required - set(self.replay_buffer.keys()))
        if missing:
            raise KeyError(f"{zarr_path} missing required fields: {missing}")

        if episode_mask is None:
            val_mask = get_val_mask(
                n_episodes=self.replay_buffer.n_episodes,
                val_ratio=self.val_ratio,
                seed=self.seed,
            )
            train_mask = _downsample_mask(~val_mask, max_train_episodes, seed=self.seed)
            self.val_mask = val_mask
            self.episode_mask = train_mask
        else:
            self.episode_mask = np.asarray(episode_mask, dtype=bool)
            self.val_mask = ~self.episode_mask

        self.indices = self._build_indices()

    def _build_indices(self) -> list[tuple[int, int, int]]:
        episode_ends = self.replay_buffer.episode_ends[:]
        indices = []
        for ep_idx, ep_end in enumerate(episode_ends):
            if not self.episode_mask[ep_idx]:
                continue
            ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            ep_end = int(ep_end)
            for current_idx in range(ep_start, ep_end):
                if current_idx + self.n_latency_steps >= ep_end:
                    continue
                indices.append((current_idx, ep_start, ep_end))
        return indices

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.episode_mask = self.val_mask.copy()
        val_set.val_mask = ~self.val_mask
        val_set.indices = val_set._build_indices()
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"])
        for key in self.lowdim_keys:
            arr = self.replay_buffer[key]
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr[:], axis=-1)
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(arr)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.indices)

    def _slice_with_edge_pad(self, arr, indices: np.ndarray) -> np.ndarray:
        return np.asarray(arr[indices])

    def _obs_indices(self, current_idx: int, ep_start: int) -> np.ndarray:
        idxs = np.arange(current_idx - self.n_obs_steps + 1, current_idx + 1)
        return np.clip(idxs, ep_start, current_idx).astype(np.int64)

    def _action_indices(self, current_idx: int, ep_end: int) -> np.ndarray:
        start = current_idx + self.n_latency_steps
        idxs = np.arange(start, start + self.horizon)
        return np.clip(idxs, start, ep_end - 1).astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        cv2.setNumThreads(1)
        current_idx, ep_start, ep_end = self.indices[idx]
        obs_idxs = self._obs_indices(current_idx, ep_start)
        action_idxs = self._action_indices(current_idx, ep_end)

        obs_dict = {}
        for key in self.rgb_keys:
            data = self._slice_with_edge_pad(self.replay_buffer[key], obs_idxs)
            obs_dict[key] = np.moveaxis(data, -1, 1).astype(np.float32) / 255.0
        for key in self.lowdim_keys:
            obs_dict[key] = self._slice_with_edge_pad(
                self.replay_buffer[key],
                obs_idxs,
            ).astype(np.float32)

        action = self._slice_with_edge_pad(
            self.replay_buffer["action"],
            action_idxs,
        ).astype(np.float32)

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
