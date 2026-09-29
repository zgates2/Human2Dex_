import unittest

import numpy as np

from real_inference_hands import _o6_command_bias, _wuji_command_bias


class O6CommandBiasTest(unittest.TestCase):
    def test_values_bias_is_six_dimensional(self):
        bias = _o6_command_bias({
            "command_bias": {
                "values": [-8, -8, -8, -8, -8, -8],
            },
        })

        np.testing.assert_allclose(bias, [-8, -8, -8, -8, -8, -8])

    def test_joint_and_index_biases_are_additive(self):
        bias = _o6_command_bias({
            "command_bias": {
                "values": [0, 0, 0, 0, 0, 0],
                "joints": {
                    "thumb_cmc_pitch": -3,
                    "index": -5,
                },
                "indices": {
                    "5": -2,
                },
            },
        })

        np.testing.assert_allclose(bias, [-3, 0, -5, 0, 0, -2])

    def test_disabled_or_empty_bias_returns_none(self):
        self.assertIsNone(_o6_command_bias({}))
        self.assertIsNone(_o6_command_bias({"command_bias": {"enabled": False}}))
        self.assertIsNone(_o6_command_bias({"command_bias": {"joints": {}}}))

    def test_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            _o6_command_bias({"command_bias": {"values": [0, 1]}})


class WujiCommandBiasTest(unittest.TestCase):
    def test_finger_bias_targets_index_row(self):
        bias = _wuji_command_bias({
            "command_bias": {
                "fingers": {
                    "index": [0.0, 0.0, 0.20, 0.16],
                },
            },
        })

        expected = np.zeros(20, dtype=np.float64)
        expected[4:8] = [0.0, 0.0, 0.20, 0.16]
        np.testing.assert_allclose(bias, expected)

    def test_full_vector_and_index_biases_are_additive(self):
        bias = _wuji_command_bias({
            "command_bias": {
                "values": np.ones((5, 4)).tolist(),
                "indices": {"6": 0.5},
            },
        })

        expected = np.ones(20, dtype=np.float64)
        expected[6] += 0.5
        np.testing.assert_allclose(bias, expected)

    def test_disabled_or_empty_bias_returns_none(self):
        self.assertIsNone(_wuji_command_bias({}))
        self.assertIsNone(_wuji_command_bias({"command_bias": {"enabled": False}}))
        self.assertIsNone(_wuji_command_bias({"command_bias": {"fingers": {}}}))

    def test_rejects_bad_finger_shape(self):
        with self.assertRaises(ValueError):
            _wuji_command_bias({
                "command_bias": {
                    "fingers": {
                        "index": [0.0, 0.1],
                    },
                },
            })


if __name__ == "__main__":
    unittest.main()
