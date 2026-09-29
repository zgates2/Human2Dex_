import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LabelInferenceEpisodeTest(unittest.TestCase):
    def test_annotations_are_append_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            episode_dir = Path(tmp_dir) / "episode_0001"
            episode_dir.mkdir()
            pkl_path = episode_dir / "episode_0001.pkl"
            with pkl_path.open("wb") as stream:
                pickle.dump({
                    "formatVersion": 4,
                    "metadata": {
                        "checkpoint": "test.ckpt",
                        "recording": {"finishReason": "test"},
                    },
                    "messages": [{}, {}],
                }, stream)

            candidates = (
                Path(__file__).with_name("label_inference_episode.py"),
                Path(__file__).resolve().parents[1] / "tools" / "label_inference_episode.py",
            )
            script = next((path for path in candidates if path.is_file()), candidates[0])
            base = [
                sys.executable,
                str(script),
                str(episode_dir),
                "--outcome",
                "operator_stop_before_outcome",
            ]
            subprocess.run(base, check=True, capture_output=True, text=True)
            subprocess.run(base + ["--notes", "second label"], check=True, capture_output=True, text=True)

            with (episode_dir / "episode_outcome.json").open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            self.assertEqual(len(payload["annotations"]), 2)
            self.assertEqual(payload["latest"]["notes"], "second label")
            self.assertEqual(payload["latest"]["episodeSummary"]["frameCount"], 2)


if __name__ == "__main__":
    unittest.main()
