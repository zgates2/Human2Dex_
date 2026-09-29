#!/usr/bin/env python3
"""Append MediaPipe Hand Landmarker 21-point RGB landmarks to DexUMI PKLs.

This is intended as a QC / pseudo-labeling tool for human-hand wrist RGB data.
It writes additive `mediapipe_*` fields and optionally renders skeleton overlays.
Run it on a copied dataset first if you do not want to mutate source PKLs.

PY=/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python
RAW_ROOT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/wrist_test_4
RUN_ROOT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_mix_5/mediapipe_test_$(date +%Y%m%d_%H%M%S)
TEST_ROOT=${RUN_ROOT}/episodes
OVERLAY_ROOT=${RUN_ROOT}/overlays
MODEL=/home/zjc/Desktop/human2dex/wrist/models/mediapipe/hand_landmarker.task

mkdir -p ${TEST_ROOT} ${OVERLAY_ROOT}

for ep in episode_0001 episode_0097 episode_0245; do
  cp -a ${RAW_ROOT}/${ep} ${TEST_ROOT}/
done

${PY} tools/add_mediapipe_hands_to_pkl.py \
  --data-root ${TEST_ROOT} \
  --model-asset-path ${MODEL} \
  --limit-frames 160 \
  --overlay-output-dir ${OVERLAY_ROOT} \
  --max-overlays-per-episode 40 \
  --min-detection-confidence 0.25 \
  --min-presence-confidence 0.25 \
  --min-tracking-confidence 0.25 \
  --overwrite

echo "overlay 输出目录: ${OVERLAY_ROOT}"

"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HAND_BONES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
JOINT_COLORS_RGB = [
    (245, 245, 245),
    *((255, 180, 80) for _ in range(4)),
    *((80, 255, 120) for _ in range(4)),
    *((80, 210, 255) for _ in range(4)),
    *((255, 120, 180) for _ in range(4)),
    *((180, 120, 255) for _ in range(4)),
]


def natural_key(path: Path) -> list[Any]:
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def list_episode_dirs(root: Path, episodes: Sequence[str] | None = None) -> list[Path]:
    root = Path(root)
    requested = set(episodes or [])
    if (root / "images").is_dir():
        if requested and root.name not in requested:
            raise FileNotFoundError(f"requested episode not matched by input episode dir: {requested}")
        return [root]
    dirs = [
        p
        for p in sorted(root.iterdir(), key=natural_key)
        if p.is_dir() and (p / "images").is_dir() and (not requested or p.name in requested)
    ]
    if requested:
        found = {p.name for p in dirs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"episodes not found under {root}: {missing}")
    return dirs


def first_pkl(episode_dir: Path) -> Path | None:
    paths = sorted(Path(episode_dir).glob("*.pkl"), key=natural_key)
    return paths[0] if paths else None


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load PKLs written by either NumPy 1.x or NumPy 2.x.

    Some existing PKLs reference ``numpy._core.*`` modules created by NumPy 2.x.
    The sam3 environment currently keeps NumPy 1.26 for MediaPipe compatibility,
    where those private module paths do not exist.  Remapping only these NumPy
    internals preserves normal pickle behavior for all project objects.
    """

    MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.numerictypes": "numpy.core.numerictypes",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(self.MODULE_ALIASES.get(module, module), name)


def read_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        try:
            data = pickle.load(handle)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            handle.seek(0)
            data = NumpyCompatUnpickler(handle).load()
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict/messages: {path}")
    return data


def atomic_write_pkl(data: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def resolve_image_path(pkl_path: Path, msg: dict[str, Any], image_field: str) -> Path | None:
    value = msg.get(image_field)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    episode_dir = pkl_path.parent
    candidates = [
        episode_dir / path,
        episode_dir / "images" / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def clear_fields(msg: dict[str, Any], error: str) -> None:
    msg["mediapipe_uv21_rgb"] = None
    msg["mediapipe_uv21_valid"] = np.zeros(21, dtype=bool)
    msg["mediapipe_world_landmarks"] = None
    msg["mediapipe_handedness"] = None
    msg["mediapipe_score"] = 0.0
    msg["mediapipe_error"] = error


def draw_skeleton_rgb(image_rgb: np.ndarray, uv21: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from PIL import ImageDraw

    out = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    width, height = out.size
    arr = np.asarray(uv21, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    mask &= np.isfinite(arr).all(axis=1)
    mask &= arr[:, 0] >= 0
    mask &= arr[:, 0] < width
    mask &= arr[:, 1] >= 0
    mask &= arr[:, 1] < height
    for a, b in HAND_BONES:
        if mask[a] and mask[b]:
            draw.line(
                [tuple(map(float, arr[a])), tuple(map(float, arr[b]))],
                fill=JOINT_COLORS_RGB[b],
                width=2,
            )
    for i, uv in enumerate(arr):
        if not mask[i]:
            continue
        x, y = float(uv[0]), float(uv[1])
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=JOINT_COLORS_RGB[i])
    return np.asarray(out, dtype=np.uint8)


def load_mediapipe_detector(args):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"failed to import mediapipe: {type(exc).__name__}: {exc}") from exc

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(args.model_asset_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=int(args.num_hands),
        min_hand_detection_confidence=float(args.min_detection_confidence),
        min_hand_presence_confidence=float(args.min_presence_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )
    return mp, vision.HandLandmarker.create_from_options(options)


def detect_one(mp, detector, image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return {
            "ok": False,
            "error": "no_hand_detected",
            "rgb": rgb,
            "uv21": None,
            "valid": np.zeros(21, dtype=bool),
            "world": None,
            "handedness": None,
            "score": 0.0,
        }

    best_idx = 0
    best_score = -1.0
    for idx, categories in enumerate(result.handedness or []):
        if categories:
            score = float(categories[0].score)
            if score > best_score:
                best_score = score
                best_idx = idx
    if best_score < 0:
        best_score = 1.0

    landmarks = result.hand_landmarks[best_idx]
    uv = np.asarray([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    valid = np.asarray(
        [
            np.isfinite(lm.x)
            and np.isfinite(lm.y)
            and (-0.1 <= float(lm.x) <= 1.1)
            and (-0.1 <= float(lm.y) <= 1.1)
            for lm in landmarks
        ],
        dtype=bool,
    )
    world = None
    if result.hand_world_landmarks and best_idx < len(result.hand_world_landmarks):
        world = np.asarray(
            [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[best_idx]],
            dtype=np.float32,
        )
    handedness = None
    if result.handedness and best_idx < len(result.handedness) and result.handedness[best_idx]:
        category = result.handedness[best_idx][0]
        handedness = str(category.category_name)
        best_score = float(category.score)

    return {
        "ok": True,
        "error": None,
        "rgb": rgb,
        "uv21": uv,
        "valid": valid,
        "world": world,
        "handedness": handedness,
        "score": float(best_score),
    }


def process_pkl(args, mp, detector, pkl_path: Path, overlay_episode_dir: Path | None) -> dict[str, Any]:
    data = read_pkl(pkl_path)
    messages = data.get("messages", [])
    max_count = len(messages) if args.limit_frames is None else min(len(messages), int(args.limit_frames))
    stats = Counter()
    overlay_episode_dir.mkdir(parents=True, exist_ok=True) if overlay_episode_dir else None

    overlay_indices: set[int] = set()
    if overlay_episode_dir and max_count > 0 and int(args.max_overlays_per_episode) > 0:
        n = min(max_count, int(args.max_overlays_per_episode))
        overlay_indices = set(np.linspace(0, max_count - 1, n).round().astype(int).tolist())

    for frame_idx, msg in enumerate(messages[:max_count]):
        if not isinstance(msg, dict):
            stats["message_not_dict"] += 1
            continue
        if (
            not args.overwrite
            and msg.get("mediapipe_error") is None
            and isinstance(msg.get("mediapipe_uv21_rgb"), np.ndarray)
        ):
            stats["already_complete"] += 1
            continue
        image_path = resolve_image_path(pkl_path, msg, str(args.image_field))
        if image_path is None:
            clear_fields(msg, f"missing_image_field:{args.image_field}")
            stats["missing_image_field"] += 1
            continue
        if not image_path.is_file():
            clear_fields(msg, f"missing_image_file:{image_path}")
            stats["missing_image_file"] += 1
            continue
        try:
            pred = detect_one(mp, detector, image_path)
        except Exception as exc:
            clear_fields(msg, f"{type(exc).__name__}:{exc}")
            stats["detect_error"] += 1
            continue

        if not pred["ok"]:
            clear_fields(msg, str(pred["error"]))
            stats[str(pred["error"])] += 1
        else:
            msg["mediapipe_uv21_rgb"] = np.asarray(pred["uv21"], dtype=np.float32).reshape(21, 2)
            msg["mediapipe_uv21_valid"] = np.asarray(pred["valid"], dtype=bool).reshape(21)
            msg["mediapipe_world_landmarks"] = (
                None
                if pred["world"] is None
                else np.asarray(pred["world"], dtype=np.float32).reshape(21, 3)
            )
            msg["mediapipe_handedness"] = pred["handedness"]
            msg["mediapipe_score"] = float(pred["score"])
            msg["mediapipe_error"] = None
            stats["detected"] += 1

        if overlay_episode_dir and frame_idx in overlay_indices:
            uv = msg.get("mediapipe_uv21_rgb")
            valid = msg.get("mediapipe_uv21_valid")
            if isinstance(uv, np.ndarray) and isinstance(valid, np.ndarray):
                overlay = draw_skeleton_rgb(pred["rgb"], uv, valid)
            else:
                overlay = pred.get("rgb")
            if overlay is not None:
                out_name = f"{frame_idx:06d}_{image_path.stem}_mediapipe.jpg"
                Image.fromarray(np.asarray(overlay, dtype=np.uint8)).save(
                    overlay_episode_dir / out_name,
                    quality=int(args.jpeg_quality),
                )
                stats["overlays"] += 1

    metadata = data.setdefault("metadata", {})
    metadata["mediapipe_hands"] = {
        "model_asset_path": str(args.model_asset_path),
        "image_field": str(args.image_field),
        "num_hands": int(args.num_hands),
        "limit_frames": args.limit_frames,
        "fields": [
            "mediapipe_uv21_rgb",
            "mediapipe_uv21_valid",
            "mediapipe_world_landmarks",
            "mediapipe_handedness",
            "mediapipe_score",
            "mediapipe_error",
        ],
    }
    if not args.dry_run:
        atomic_write_pkl(data, pkl_path)
    return {"pkl": str(pkl_path), "frames": int(max_count), **dict(stats)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--episode", action="append", default=[], help="Episode name. Repeatable.")
    parser.add_argument("--model-asset-path", type=Path, required=True)
    parser.add_argument("--image-field", default="rgbImage")
    parser.add_argument("--limit-pkls", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.35)
    parser.add_argument("--min-presence-confidence", type=float, default=0.35)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.35)
    parser.add_argument("--overlay-output-dir", type=Path, default=None)
    parser.add_argument("--max-overlays-per-episode", type=int, default=40)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.model_asset_path = args.model_asset_path.expanduser().resolve()
    if not args.model_asset_path.is_file():
        raise FileNotFoundError(args.model_asset_path)

    episodes = list_episode_dirs(args.data_root, args.episode)
    if args.limit_pkls is not None:
        episodes = episodes[: int(args.limit_pkls)]
    if not episodes:
        raise SystemExit(f"no episodes selected from {args.data_root}")

    print(f"selected episodes: {len(episodes)}")
    print(f"model: {args.model_asset_path}")
    print(f"overlay: {args.overlay_output_dir}")
    if args.dry_run:
        for ep in episodes:
            print(ep)
        return 0

    mp, detector = load_mediapipe_detector(args)
    all_stats = []
    try:
        for ep in episodes:
            pkl_path = first_pkl(ep)
            if pkl_path is None:
                row = {"episode": ep.name, "error": "no_pkl"}
                print(json.dumps(row, ensure_ascii=False))
                all_stats.append(row)
                continue
            overlay_ep = args.overlay_output_dir / ep.name if args.overlay_output_dir else None
            row = process_pkl(args, mp, detector, pkl_path, overlay_ep)
            row["episode"] = ep.name
            print(json.dumps(row, ensure_ascii=False))
            all_stats.append(row)
    finally:
        detector.close()

    if args.overlay_output_dir:
        args.overlay_output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "data_root": str(args.data_root),
            "episodes": [ep.name for ep in episodes],
            "model_asset_path": str(args.model_asset_path),
            "stats": all_stats,
        }
        (args.overlay_output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"summary: {args.overlay_output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
