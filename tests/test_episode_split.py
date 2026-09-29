import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from diffusion_policy.common.episode_split import (
    canonical_episode_group,
    get_grouped_val_mask,
    load_manifest_episode_groups,
)


class EpisodeSplitTest(unittest.TestCase):
    def test_canonical_group_strips_augmented_suffixes(self):
        original = "/data/task/episode_0001/episode_0001.pkl"
        augmented = "/data/task/episode_0001_hand_aug/episode_0001.pkl"
        variant = "/data/task/episode_0001_hand_aug03/episode_0001.pkl"
        self.assertEqual(canonical_episode_group(original), canonical_episode_group(augmented))
        self.assertEqual(canonical_episode_group(original), canonical_episode_group(variant))

    def test_grouped_mask_never_splits_a_group(self):
        groups = [f"episode_{index:04d}" for index in range(20) for _ in range(2)]
        mask = get_grouped_val_mask(groups, val_ratio=0.2, seed=42)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), len(mask))
        for index in range(0, len(mask), 2):
            self.assertEqual(bool(mask[index]), bool(mask[index + 1]))
        np.testing.assert_array_equal(mask, get_grouped_val_mask(groups, 0.2, seed=42))

    def test_manifest_order_and_count_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_path = root / "dataset.zarr.zip"
            manifest = {
                "episodes": [
                    {"pkl": "/data/task/episode_0001/episode_0001.pkl"},
                    {"pkl": "/data/task/episode_0001_hand_aug/episode_0001.pkl"},
                ]
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            groups = load_manifest_episode_groups(dataset_path, n_episodes=2)
            self.assertEqual(groups[0], groups[1])
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                load_manifest_episode_groups(dataset_path, n_episodes=3)


if __name__ == "__main__":
    unittest.main()
