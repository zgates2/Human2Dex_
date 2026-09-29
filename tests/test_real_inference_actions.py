import unittest

import numpy as np

from real_inference_actions import (
    scale_franka_negative_x_delta,
    scale_franka_negative_y_delta,
    scale_franka_z_delta,
    validate_hand_checkpoint_compatibility,
)


class FrankaZDeltaGainTest(unittest.TestCase):
    def test_scales_up_and_down_motion_around_current_z(self):
        actions = np.asarray([
            [0.1, 0.2, 0.490, 0.0, 0.0, 0.0, 0.04],
            [0.1, 0.2, 0.500, 0.0, 0.0, 0.0, 0.04],
            [0.1, 0.2, 0.520, 0.0, 0.0, 0.0, 0.04],
        ], dtype=np.float64)

        scaled = scale_franka_z_delta(actions, current_z=0.500, gain=1.5)

        np.testing.assert_allclose(scaled[:, 2], [0.485, 0.500, 0.530])
        np.testing.assert_allclose(scaled[:, [0, 1, 3, 4, 5, 6]], actions[:, [0, 1, 3, 4, 5, 6]])

    def test_rejects_invalid_gain(self):
        with self.assertRaises(ValueError):
            scale_franka_z_delta(np.zeros((1, 7)), current_z=0.5, gain=0.0)


class FrankaNegativeXDeltaGainTest(unittest.TestCase):
    def test_scales_only_negative_x_motion_around_current_x(self):
        actions = np.asarray([
            [0.480, 0.2, 0.5, 0.0, 0.0, 0.0, 0.04],
            [0.500, 0.2, 0.5, 0.0, 0.0, 0.0, 0.04],
            [0.520, 0.2, 0.5, 0.0, 0.0, 0.0, 0.04],
        ], dtype=np.float64)

        scaled = scale_franka_negative_x_delta(actions, current_x=0.500, gain=1.5)

        np.testing.assert_allclose(scaled[:, 0], [0.470, 0.500, 0.520])
        np.testing.assert_allclose(scaled[:, 1:], actions[:, 1:])

    def test_rejects_invalid_gain(self):
        with self.assertRaises(ValueError):
            scale_franka_negative_x_delta(np.zeros((1, 7)), current_x=0.5, gain=0.0)


class FrankaNegativeYDeltaGainTest(unittest.TestCase):
    def test_scales_only_negative_y_motion_around_current_y(self):
        actions = np.asarray([
            [0.1, 0.480, 0.5, 0.0, 0.0, 0.0, 0.04],
            [0.1, 0.500, 0.5, 0.0, 0.0, 0.0, 0.04],
            [0.1, 0.520, 0.5, 0.0, 0.0, 0.0, 0.04],
        ], dtype=np.float64)

        scaled = scale_franka_negative_y_delta(actions, current_y=0.500, gain=1.5)

        np.testing.assert_allclose(scaled[:, 1], [0.470, 0.500, 0.520])
        np.testing.assert_allclose(scaled[:, [0, 2, 3, 4, 5, 6]], actions[:, [0, 2, 3, 4, 5, 6]])

    def test_rejects_invalid_gain(self):
        with self.assertRaises(ValueError):
            scale_franka_negative_y_delta(np.zeros((1, 7)), current_y=0.5, gain=0.0)


class HandCheckpointCompatibilityTest(unittest.TestCase):
    def _shape_meta(self, gripper_shape):
        return {
            "obs": {
                "robot0_gripper_width": {
                    "shape": list(gripper_shape),
                },
            },
        }

    def test_rejects_o6_checkpoint_for_wuji_hand(self):
        with self.assertRaisesRegex(ValueError, "This looks like a Linker O6 checkpoint"):
            validate_hand_checkpoint_compatibility(
                hand_backend="wuji_hand",
                action_dim=15,
                shape_meta=self._shape_meta((6,)),
                pts21_mode=False,
                task_name="linker_o6_open_drawer_raw_no_processing",
            )

    def test_allows_wuji_direct_checkpoint(self):
        warnings = validate_hand_checkpoint_compatibility(
            hand_backend="wuji_hand",
            action_dim=29,
            shape_meta=self._shape_meta((20,)),
            pts21_mode=False,
            task_name="wuji_open_drawer",
        )

        self.assertEqual(warnings, [])

    def test_warns_for_wuji_command_head(self):
        warnings = validate_hand_checkpoint_compatibility(
            hand_backend="wuji_hand",
            action_dim=10,
            shape_meta=self._shape_meta((20,)),
            pts21_mode=False,
            task_name="wuji_with_command_head",
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("result['wuji_command']", warnings[0])


if __name__ == "__main__":
    unittest.main()
