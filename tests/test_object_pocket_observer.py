import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from real_inference_object_pocket import (
    RuntimeObjectPocketObserver,
    _ObjectMemory,
    normalize_object_specs,
)


class ObjectPocketObserverTest(unittest.TestCase):
    def test_legacy_single_object_config_is_preserved(self):
        specs = normalize_object_specs({
            "prompt": "sponge",
            "reference_scale_px": 20.0,
            "reference_area_px2": 400.0,
        })

        self.assertEqual([item["name"] for item in specs], ["object"])
        self.assertEqual(specs[0]["prompt"], "sponge")
        self.assertEqual(specs[0]["reference_scale_px"], 20.0)

    def test_multi_object_order_and_overrides_are_stable(self):
        specs = normalize_object_specs({
            "min_area": 10,
            "objects": [
                {
                    "name": "green_cup",
                    "prompt": "green disposable plastic cup",
                    "reference_scale_px": 30.0,
                    "reference_area_px2": 900.0,
                },
                {
                    "name": "red_cup",
                    "prompt": "red disposable plastic cup",
                    "min_area": 20,
                    "reference_scale_px": 40.0,
                    "reference_area_px2": 1600.0,
                },
            ],
        })

        self.assertEqual([item["name"] for item in specs], ["green_cup", "red_cup"])
        self.assertEqual([item["min_area"] for item in specs], [10, 20])

    def test_shape_must_match_five_values_per_object(self):
        observer = RuntimeObjectPocketObserver(
            {
                "enabled": True,
                "objects": [
                    {"name": "a", "prompt": "a"},
                    {"name": "b", "prompt": "b"},
                ],
            },
            {"obs": {"objectPocketObs": {"shape": [5]}}},
            "linker_o6",
        )

        self.assertFalse(observer.enabled)
        self.assertIn("does not match", observer._disabled_reason)

    @mock.patch.object(RuntimeObjectPocketObserver, "_start_worker")
    @mock.patch("real_inference_object_pocket._load_hand_runtime_modules")
    def test_wuji_backend_is_enabled_with_20d_kinematics(
        self, load_modules, start_worker
    ):
        kinematics_cls = mock.Mock()
        load_modules.return_value = {
            "fisheye_unit_rays_to_pixels": mock.Mock(),
            "camera_points": mock.Mock(),
            "compute_grasp_pocket": mock.Mock(),
            "load_calibration": mock.Mock(
                return_value=SimpleNamespace(urdf_path="/tmp/wuji.urdf")
            ),
            "load_grasp_config": mock.Mock(
                return_value={"a": 0.45, "b": 0.25, "normal_sign": -1.0}
            ),
            "Kinematics": kinematics_cls,
            "expected_state_dim": 20,
            "default_calibration": "/tmp/camera_from_wuji_base.json",
        }

        observer = RuntimeObjectPocketObserver(
            {"enabled": True},
            {"obs": {"objectPocketObs": {"shape": [5]}}},
            "wuji_hand",
        )

        self.assertTrue(observer.enabled)
        self.assertEqual(observer.hand_backend, "wuji_hand")
        self.assertEqual(observer.expected_state_dim, 20)
        self.assertEqual(
            str(observer.calibration_path), "/tmp/camera_from_wuji_base.json"
        )
        load_modules.assert_called_once_with("wuji_hand")
        kinematics_cls.assert_called_once_with("/tmp/wuji.urdf")
        start_worker.assert_called_once_with()

    def test_wuji_pocket_accepts_flat_or_five_by_four_state(self):
        observer = RuntimeObjectPocketObserver.__new__(RuntimeObjectPocketObserver)
        observer.expected_state_dim = 20
        observer.kinematics = mock.Mock(
            points21=mock.Mock(return_value=np.zeros((21, 3), dtype=np.float64))
        )
        observer.grasp_config = {"a": 0.45, "b": 0.25, "normal_sign": -1.0}
        observer._compute_grasp_pocket = mock.Mock(
            return_value=SimpleNamespace(
                point3d=np.asarray([0.0, 0.0, 0.1], dtype=np.float64)
            )
        )
        observer._camera_points = mock.Mock(
            return_value=np.asarray([[0.0, 0.0, 0.1]], dtype=np.float64)
        )
        observer._fisheye_unit_rays_to_pixels = mock.Mock(
            return_value=np.asarray([[112.0, 96.0]], dtype=np.float64)
        )
        observer.calibration = SimpleNamespace(K=np.eye(3), D=np.zeros(4))

        for state in (np.zeros(20), np.zeros((5, 4))):
            uv, confidence = observer._pocket_uv(state)
            np.testing.assert_allclose(uv, [112.0, 96.0])
            self.assertEqual(confidence, 1.0)

        uv, confidence = observer._pocket_uv(np.zeros(6))
        self.assertIsNone(uv)
        self.assertEqual(confidence, 0.0)

    def test_vector_concatenates_objects_in_config_order(self):
        observer = RuntimeObjectPocketObserver.__new__(RuntimeObjectPocketObserver)
        observer.objects = normalize_object_specs({
            "objects": [
                {
                    "name": "a",
                    "prompt": "a",
                    "reference_scale_px": 10.0,
                    "reference_area_px2": 100.0,
                },
                {
                    "name": "b",
                    "prompt": "b",
                    "reference_scale_px": 20.0,
                    "reference_area_px2": 400.0,
                },
            ]
        })
        now = time.monotonic()
        observer._memories = {
            "a": _ObjectMemory(
                uv=np.asarray([20.0, 30.0]),
                area=100.0,
                confidence=0.8,
                updated_time=now,
            ),
            "b": _ObjectMemory(
                uv=np.asarray([30.0, 50.0]),
                area=800.0,
                confidence=0.6,
                updated_time=now,
            ),
        }
        observer.combine_pocket_confidence = True

        vector = observer._current_obs_vector(
            np.asarray([10.0, 10.0]), pocket_conf=0.5
        )

        self.assertEqual(vector.shape, (10,))
        np.testing.assert_allclose(vector[:5], [1.0, 2.0, 0.0, 0.5, 1.0], atol=1e-3)
        np.testing.assert_allclose(
            vector[5:], [1.0, 2.0, np.log(2.0), 0.5, 1.0], atol=1e-3
        )


if __name__ == "__main__":
    unittest.main()
