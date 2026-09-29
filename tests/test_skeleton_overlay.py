import unittest
from unittest import mock

import numpy as np

try:
    from real_inference_skeleton_overlay import O6SkeletonOverlayRenderer
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    O6SkeletonOverlayRenderer = None


@unittest.skipIf(O6SkeletonOverlayRenderer is None, "OpenCV is not installed")
class SkeletonOverlayTest(unittest.TestCase):
    def test_projectable_mask_keeps_positive_depth_out_of_frame_points(self):
        uv = np.asarray(
            [
                [-8.0, 4.0],
                [20.0, 7.0],
                [5.0, -3.0],
                [5.0, 6.0],
                [np.nan, 1.0],
            ],
            dtype=np.float64,
        )
        depth = np.asarray([0.2, 0.3, 0.4, -0.1, 0.5], dtype=np.float64)

        valid = O6SkeletonOverlayRenderer._projectable_uv_mask(uv, depth)

        np.testing.assert_array_equal(
            valid,
            np.asarray([True, True, True, False, False], dtype=bool),
        )

    def test_clip_uv_to_image_maps_to_nearest_pixel_inside_frame(self):
        uv = np.asarray(
            [
                [-8.0, 4.0],
                [20.0, 7.0],
                [5.0, -3.0],
                [5.0, 99.0],
            ],
            dtype=np.float64,
        )

        clipped = O6SkeletonOverlayRenderer._clip_uv_to_image(
            uv,
            width=10,
            height=8,
        )

        np.testing.assert_allclose(
            clipped,
            np.asarray(
                [
                    [0.0, 4.0],
                    [9.0, 7.0],
                    [5.0, 0.0],
                    [5.0, 7.0],
                ],
                dtype=np.float64,
            ),
        )

    def test_overlay_draws_out_of_frame_point_on_nearest_border(self):
        renderer = O6SkeletonOverlayRenderer.__new__(O6SkeletonOverlayRenderer)
        renderer.draw_wrist = False
        renderer.line_thickness = 1
        renderer.point_radius = 2.0
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        points21 = np.zeros((21, 3), dtype=np.float64)
        uv21 = np.zeros((21, 2), dtype=np.float64)
        depth21 = np.full(21, -1.0, dtype=np.float64)
        uv21[1] = [-6.0, 4.0]
        depth21[1] = 0.2

        out = renderer._overlay_one(image, points21, uv21, depth21)

        self.assertGreater(int(out[:, 0, :].sum()), 0)

    def test_overlay_still_drops_points_behind_camera(self):
        renderer = O6SkeletonOverlayRenderer.__new__(O6SkeletonOverlayRenderer)
        renderer.draw_wrist = False
        renderer.line_thickness = 1
        renderer.point_radius = 2.0
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        points21 = np.zeros((21, 3), dtype=np.float64)
        uv21 = np.zeros((21, 2), dtype=np.float64)
        depth21 = np.full(21, -1.0, dtype=np.float64)
        uv21[1] = [-6.0, 4.0]

        out = renderer._overlay_one(image, points21, uv21, depth21)

        self.assertEqual(int(out.sum()), 0)

    def test_wrist_estimate_falls_back_to_2d_when_3d_projection_is_behind_camera(self):
        renderer = O6SkeletonOverlayRenderer.__new__(O6SkeletonOverlayRenderer)
        renderer.wrist_extension_ratio = 1.0
        renderer.rvec = np.zeros(3, dtype=np.float64)
        renderer.tvec = np.zeros(3, dtype=np.float64)
        renderer.K = np.eye(3, dtype=np.float64)
        renderer.D = np.zeros((4, 1), dtype=np.float64)
        points21 = np.zeros((21, 3), dtype=np.float64)
        uv21 = np.zeros((21, 2), dtype=np.float64)
        palm_indices = (1, 5, 9, 13, 17)
        tip_indices = (4, 8, 12, 16, 20)
        uv21[list(palm_indices)] = np.asarray([[5.0, 5.0]] * 5)
        uv21[list(tip_indices)] = np.asarray([[5.0, 1.0]] * 5)
        valid = np.zeros(21, dtype=bool)
        valid[list(palm_indices)] = True
        valid[list(tip_indices)] = True

        with mock.patch(
            "real_inference_skeleton_overlay.project_fisheye",
            return_value=(np.asarray([[1000.0, 1000.0]]), np.asarray([-1.0])),
        ):
            wrist_uv = renderer._estimate_wrist_uv(points21, uv21, valid)

        np.testing.assert_allclose(wrist_uv, np.asarray([5.0, 9.0]))

    def test_overlay_draws_wrist_from_2d_fallback_on_nearest_border(self):
        renderer = O6SkeletonOverlayRenderer.__new__(O6SkeletonOverlayRenderer)
        renderer.draw_wrist = True
        renderer.line_thickness = 1
        renderer.point_radius = 2.0
        renderer.wrist_extension_ratio = 1.0
        renderer.rvec = np.zeros(3, dtype=np.float64)
        renderer.tvec = np.zeros(3, dtype=np.float64)
        renderer.K = np.eye(3, dtype=np.float64)
        renderer.D = np.zeros((4, 1), dtype=np.float64)
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        points21 = np.zeros((21, 3), dtype=np.float64)
        uv21 = np.zeros((21, 2), dtype=np.float64)
        depth21 = np.full(21, -1.0, dtype=np.float64)
        palm_indices = (1, 5, 9, 13, 17)
        tip_indices = (4, 8, 12, 16, 20)
        uv21[list(palm_indices)] = np.asarray([[5.0, 4.0]] * 5)
        uv21[list(tip_indices)] = np.asarray([[5.0, -3.0]] * 5)
        depth21[list(palm_indices)] = 0.2
        depth21[list(tip_indices)] = 0.2

        with mock.patch(
            "real_inference_skeleton_overlay.project_fisheye",
            return_value=(np.asarray([[1000.0, 1000.0]]), np.asarray([-1.0])),
        ):
            out = renderer._overlay_one(image, points21, uv21, depth21)

        self.assertGreater(int(out[-1, :, :].sum()), 0)


if __name__ == "__main__":
    unittest.main()
