import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference_episode_recorder import InferenceEpisodeRecorder


class _FakeCamera:
    def __init__(self, data):
        self._data = data

    def get(self, k):
        return {0: self._data}


class _FakeRobot:
    def __init__(self, state):
        self._state = state

    def get_all_state(self):
        return self._state


class _FakeEnv:
    def __init__(self, camera_data, robot_state):
        self.camera = _FakeCamera(camera_data)
        self.robots = [_FakeRobot(robot_state)]


class InferenceEpisodeRecorderTest(unittest.TestCase):
    def test_o6_measured_state_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = InferenceEpisodeRecorder(
                output_dir=tmp_dir,
                frequency=30.0,
                hand_backend="linker_o6",
                jpg_workers=1,
            )
            pkl_path = recorder.start_episode({"checkpoint": "test.ckpt"})
            recorder._append_message(
                target_timestamp=1.0,
                trajectory_pose=np.zeros(6, dtype=np.float32),
                command=np.asarray([10, 20, 30, 40, 50, 60], dtype=np.uint8),
                measured_hand_state=np.asarray([9, 19, 29, 39, 49, 59], dtype=np.uint8),
                image=np.zeros((16, 16, 3), dtype=np.uint8),
                frame_id=1,
                capture_timestamp=1.0,
                align_delta_s=0.0,
            )
            result = recorder.finish_episode(reason="test")
            self.assertEqual(result, pkl_path)
            with Path(result).open("rb") as stream:
                payload = pickle.load(stream)
            message = payload["messages"][0]
            np.testing.assert_array_equal(
                message["o6_measured_state"],
                np.asarray([9, 19, 29, 39, 49, 59], dtype=np.uint8),
            )
            stats = payload["metadata"]["stats"]
            self.assertEqual(stats["savedMeasuredHandStates"], 1)
            provenance = payload["metadata"]["provenance"]
            self.assertIn("hostname", provenance)
            self.assertIn("pythonExecutable", provenance)
            self.assertIn("gitStatusShort", provenance)
            checkpoint = provenance["artifacts"]["checkpoint"]
            self.assertEqual(checkpoint["path"], "test.ckpt")
            self.assertFalse(checkpoint["exists"])

    def test_fixed_record_fps_resamples_command_segment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = InferenceEpisodeRecorder(
                output_dir=tmp_dir,
                frequency=10.0,
                record_fps=30.0,
                hand_backend="linker_o6",
                jpg_workers=1,
            )
            pkl_path = recorder.start_episode({"checkpoint": "test.ckpt"})
            camera_ts = 1.0 + np.arange(4, dtype=np.float64) / 30.0
            camera_images = np.zeros((4, 12, 12, 3), dtype=np.uint8)
            camera_images[:, :, :, 0] = np.arange(4, dtype=np.uint8)[:, None, None]
            camera_data = {
                "timestamp": camera_ts,
                "camera_capture_timestamp": camera_ts,
                "camera_frame_id": np.arange(4, dtype=np.int64),
                "color": camera_images,
            }
            robot_state = {
                "robot_timestamp": np.asarray([0.9, 1.2], dtype=np.float64),
                "ActualTCPPose": np.asarray(
                    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
            }
            env = _FakeEnv(camera_data, robot_state)

            recorder.record_segment(
                env=env,
                timestamps=np.asarray([1.0, 1.1], dtype=np.float64),
                hand_commands=np.asarray(
                    [[10, 10, 10, 10, 10, 10],
                     [20, 20, 20, 20, 20, 20]],
                    dtype=np.uint8,
                ),
                measured_hand_states=np.asarray(
                    [[9, 9, 9, 9, 9, 9],
                     [19, 19, 19, 19, 19, 19]],
                    dtype=np.uint8,
                ),
            )
            result = recorder.finish_episode(env=env, reason="test")
            self.assertEqual(result, pkl_path)
            with Path(result).open("rb") as stream:
                payload = pickle.load(stream)
            messages = payload["messages"]
            self.assertEqual(len(messages), 4)
            timestamps = np.asarray([msg["timestamp"] for msg in messages])
            np.testing.assert_allclose(
                np.diff(timestamps),
                np.full(3, 1.0 / 30.0),
                atol=1e-6,
            )
            np.testing.assert_array_equal(
                messages[0]["hand_command"],
                np.full(6, 10, dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                messages[-1]["hand_command"],
                np.full(6, 20, dtype=np.uint8),
            )
            metadata = payload["metadata"]
            self.assertEqual(metadata["collectionHz"], 30.0)
            self.assertEqual(metadata["controlHz"], 10.0)
            self.assertEqual(metadata["recordHz"], 30.0)


if __name__ == "__main__":
    unittest.main()
