import unittest

import torch

from diffusion_policy.model.common.lowdim_obs_augmentation import (
    apply_lowdim_obs_augmentation,
)


class LowDimObservationAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.hand = torch.tensor(
            [
                [[-0.8, -0.4, 0.0], [-0.7, -0.3, 0.1]],
                [[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]],
            ],
            dtype=torch.float32,
        )
        self.image = torch.ones((2, 1, 3, 4, 4), dtype=torch.float32)
        self.obs = {
            "robot0_gripper_width": self.hand,
            "camera0_rgb": self.image,
        }

    def test_disabled_returns_original_mapping(self):
        output = apply_lowdim_obs_augmentation(self.obs, {"enabled": False})
        self.assertIs(output, self.obs)

    def test_dropout_is_sample_wide_and_selected_key_only(self):
        output = apply_lowdim_obs_augmentation(
            self.obs,
            {
                "enabled": True,
                "keys": ["robot0_gripper_width"],
                "dropout_prob": 1.0,
                "dropout_fill": 0.0,
                "clip_abs": None,
            },
        )
        self.assertTrue(torch.equal(output["robot0_gripper_width"], torch.zeros_like(self.hand)))
        self.assertIs(output["camera0_rgb"], self.image)
        self.assertTrue(torch.equal(self.obs["robot0_gripper_width"], self.hand))

    def test_per_dimension_bias_is_constant_across_horizon(self):
        generator = torch.Generator().manual_seed(7)
        output = apply_lowdim_obs_augmentation(
            self.obs,
            {
                "enabled": True,
                "per_dimension_bias_std": 0.2,
                "clip_abs": None,
            },
            generator=generator,
        )
        delta = output["robot0_gripper_width"] - self.hand
        torch.testing.assert_close(delta[:, 0], delta[:, 1])

    def test_element_noise_is_reproducible_with_generator(self):
        config = {
            "enabled": True,
            "element_noise_std": 0.1,
            "clip_abs": None,
        }
        first = apply_lowdim_obs_augmentation(
            self.obs, config, generator=torch.Generator().manual_seed(11)
        )["robot0_gripper_width"]
        second = apply_lowdim_obs_augmentation(
            self.obs, config, generator=torch.Generator().manual_seed(11)
        )["robot0_gripper_width"]
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, self.hand))

    def test_clip_and_invalid_probability(self):
        output = apply_lowdim_obs_augmentation(
            self.obs,
            {
                "enabled": True,
                "per_dimension_bias_std": 100.0,
                "clip_abs": 0.5,
            },
            generator=torch.Generator().manual_seed(3),
        )["robot0_gripper_width"]
        self.assertLessEqual(float(output.abs().max()), 0.5)
        with self.assertRaisesRegex(ValueError, "dropout_prob"):
            apply_lowdim_obs_augmentation(
                self.obs,
                {"enabled": True, "dropout_prob": 1.1},
            )


if __name__ == "__main__":
    unittest.main()
