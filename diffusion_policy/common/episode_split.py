"""Episode-level split helpers that keep augmented variants together."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_AUGMENTATION_SUFFIX_PATTERN = r"_hand_aug(?:[0-9]+)?$"


def manifest_path_for_dataset(dataset_path: str | Path) -> Path:
    return Path(dataset_path).expanduser().resolve().with_name("manifest.json")


def canonical_episode_group(
    pkl_path: str | Path,
    suffix_pattern: str = DEFAULT_AUGMENTATION_SUFFIX_PATTERN,
) -> str:
    """Return a stable group key shared by an episode and its augmentations."""
    episode_dir = Path(pkl_path).expanduser().parent
    base_name = re.sub(suffix_pattern, "", episode_dir.name)
    return str(episode_dir.parent / base_name)


def load_manifest_episode_groups(
    dataset_path: str | Path,
    n_episodes: int,
    suffix_pattern: str = DEFAULT_AUGMENTATION_SUFFIX_PATTERN,
) -> list[str]:
    manifest_path = manifest_path_for_dataset(dataset_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Grouped validation requires the converter manifest: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list):
        raise ValueError(f"Manifest does not contain an episodes list: {manifest_path}")
    if len(episodes) != int(n_episodes):
        raise ValueError(
            "Manifest/Zarr episode count mismatch: "
            f"manifest={len(episodes)} zarr={n_episodes} path={manifest_path}"
        )

    groups = []
    for index, episode in enumerate(episodes):
        pkl_path = episode.get("pkl") if isinstance(episode, dict) else None
        if not pkl_path:
            raise ValueError(f"Manifest episode {index} is missing pkl: {manifest_path}")
        groups.append(canonical_episode_group(pkl_path, suffix_pattern=suffix_pattern))
    return groups


def get_grouped_val_mask(
    group_ids: Sequence[str],
    val_ratio: float,
    seed: int = 0,
) -> np.ndarray:
    """Sample validation groups and assign every member of each group together."""
    groups = np.asarray([str(value) for value in group_ids], dtype=object)
    val_mask = np.zeros(len(groups), dtype=bool)
    if val_ratio <= 0 or len(groups) == 0:
        return val_mask
    if val_ratio >= 1:
        raise ValueError(f"val_ratio must be less than 1, got {val_ratio}")

    unique_groups = list(dict.fromkeys(groups.tolist()))
    if len(unique_groups) <= 1:
        return val_mask
    n_val_groups = min(
        max(1, round(len(unique_groups) * float(val_ratio))),
        len(unique_groups) - 1,
    )
    rng = np.random.default_rng(seed=seed)
    selected_indices = rng.choice(len(unique_groups), size=n_val_groups, replace=False)
    selected_groups = {unique_groups[int(index)] for index in selected_indices}
    return np.asarray([group in selected_groups for group in groups], dtype=bool)
