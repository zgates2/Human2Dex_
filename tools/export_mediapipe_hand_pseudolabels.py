#!/usr/bin/env python3
"""Export MediaPipe hand landmarks as MANO21-compatible pseudo annotations.

The output JSON follows the format used by tools/click_label_visible_hand_points.py
and wrist/scripts/train_stage2_projection_head.py:

  --labels mano21
  points[label] = {"x": pixel_x, "y": pixel_y, "visible": true}

This script does not modify source PKLs or images.  It is intended to generate
automatic pseudo labels plus visual overlays for manual screening/editing.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MANO21_LABELS = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

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

HAND_BONES_LABELS = tuple((MANO21_LABELS[a], MANO21_LABELS[b]) for a, b in HAND_BONES)

JOINT_COLORS_RGB = [
    (245, 245, 245),
    *((255, 180, 80) for _ in range(4)),
    *((80, 255, 120) for _ in range(4)),
    *((80, 210, 255) for _ in range(4)),
    *((255, 120, 180) for _ in range(4)),
    *((180, 120, 255) for _ in range(4)),
]

LABEL_COLORS = {
    label: "#{:02x}{:02x}{:02x}".format(*JOINT_COLORS_RGB[idx])
    for idx, label in enumerate(MANO21_LABELS)
}


@dataclass(frozen=True)
class FrameItem:
    episode: str
    frame_id: int
    image: Path
    image_rel: str
    dataset: str
    source_group: str


def natural_key(path: Path | str) -> list[Any]:
    name = Path(path).name
    parts = re.split(r"(\d+)", name)
    return [int(part) if part.isdigit() else part for part in parts]


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load PKLs written by either NumPy 1.x or NumPy 2.x."""

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


def first_pkl(episode_dir: Path) -> Path | None:
    paths = sorted(Path(episode_dir).glob("*.pkl"), key=natural_key)
    return paths[0] if paths else None


def episode_dirs(root: Path, episodes: Sequence[str] | None, limit_episodes: int | None) -> list[Path]:
    root = root.expanduser().resolve()
    requested = set(episodes or [])
    if (root / "images").is_dir():
        if requested and root.name not in requested:
            raise FileNotFoundError(f"input episode dir is {root.name}, not in requested episodes {sorted(requested)}")
        dirs = [root]
    elif root.is_dir():
        dirs = [
            p
            for p in sorted(root.iterdir(), key=natural_key)
            if p.is_dir() and (p / "images").is_dir() and (not requested or p.name in requested)
        ]
    else:
        raise FileNotFoundError(root)
    if requested:
        found = {p.name for p in dirs}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"episodes not found under {root}: {missing}")
    if limit_episodes is not None:
        dirs = dirs[: int(limit_episodes)]
    return dirs


def image_rel_to_episode(episode_dir: Path, image: Path) -> str:
    try:
        return image.resolve().relative_to(episode_dir.resolve()).as_posix()
    except ValueError:
        return image.name


def dataset_key(data_root: Path, episode_dir: Path) -> str:
    try:
        parent = episode_dir.resolve().parent
        return parent.name or data_root.resolve().name
    except Exception:
        return data_root.name


def resolve_image_from_msg(pkl_path: Path, msg: dict[str, Any], image_field: str) -> Path | None:
    value = msg.get(image_field)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    episode_dir = pkl_path.parent
    candidates = [episode_dir / path, episode_dir / "images" / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def discover_frames_from_pkl(args: argparse.Namespace) -> list[FrameItem]:
    frames: list[FrameItem] = []
    for ep_dir in episode_dirs(args.data_root, args.episode, args.limit_episodes):
        pkl_path = first_pkl(ep_dir)
        if pkl_path is None:
            continue
        data = read_pkl(pkl_path)
        messages = data.get("messages", [])
        start = max(0, int(args.start))
        stride = max(1, int(args.stride))
        selected = list(range(start, len(messages), stride))
        if args.limit_frames_per_episode is not None:
            selected = selected[: int(args.limit_frames_per_episode)]
        seen_images: set[str] = set()
        for msg_idx in selected:
            msg = messages[msg_idx]
            if not isinstance(msg, dict):
                continue
            image = resolve_image_from_msg(pkl_path, msg, str(args.image_field))
            if image is None or not image.is_file():
                continue
            image_key = str(image)
            if image_key in seen_images:
                continue
            seen_images.add(image_key)
            dset = dataset_key(args.data_root, ep_dir)
            frames.append(
                FrameItem(
                    episode=ep_dir.name,
                    frame_id=int(msg_idx),
                    image=image,
                    image_rel=image_rel_to_episode(ep_dir, image),
                    dataset=dset,
                    source_group=f"{dset}/{ep_dir.name}",
                )
            )
    return frames


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        [p.resolve() for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def discover_frames_from_images(args: argparse.Namespace) -> list[FrameItem]:
    frames: list[FrameItem] = []
    for ep_dir in episode_dirs(args.data_root, args.episode, args.limit_episodes):
        images = list_images(ep_dir / "images")
        start = max(0, int(args.start))
        stride = max(1, int(args.stride))
        images = images[start::stride]
        if args.limit_frames_per_episode is not None:
            images = images[: int(args.limit_frames_per_episode)]
        dset = dataset_key(args.data_root, ep_dir)
        for local_idx, image in enumerate(images):
            frames.append(
                FrameItem(
                    episode=ep_dir.name,
                    frame_id=int(local_idx),
                    image=image,
                    image_rel=image_rel_to_episode(ep_dir, image),
                    dataset=dset,
                    source_group=f"{dset}/{ep_dir.name}",
                )
            )
    return frames


def load_mediapipe_detector(args: argparse.Namespace):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as exc:
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
        return {"ok": False, "error": "no_hand_detected", "rgb": rgb}

    best_idx = 0
    best_score = -1.0
    for idx, categories in enumerate(result.handedness or []):
        if categories:
            score = float(categories[0].score)
            if score > best_score:
                best_idx = idx
                best_score = score
    if best_score < 0:
        best_score = 1.0

    landmarks = result.hand_landmarks[best_idx]
    uv21 = np.asarray([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
    in_bounds = (
        np.isfinite(uv21).all(axis=1)
        & (uv21[:, 0] >= 0)
        & (uv21[:, 0] < w)
        & (uv21[:, 1] >= 0)
        & (uv21[:, 1] < h)
    )
    bbox_wh = uv21.max(axis=0) - uv21.min(axis=0)
    bbox_area_ratio = float((bbox_wh[0] * bbox_wh[1]) / max(1, w * h))
    in_bounds_ratio = float(in_bounds.mean())

    handedness = None
    if result.handedness and best_idx < len(result.handedness) and result.handedness[best_idx]:
        category = result.handedness[best_idx][0]
        handedness = str(category.category_name)
        best_score = float(category.score)

    world = None
    if result.hand_world_landmarks and best_idx < len(result.hand_world_landmarks):
        world = np.asarray(
            [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[best_idx]],
            dtype=np.float32,
        )

    return {
        "ok": True,
        "error": None,
        "rgb": rgb,
        "uv21": uv21,
        "valid": in_bounds,
        "score": float(best_score),
        "handedness": handedness,
        "bbox_area_ratio": bbox_area_ratio,
        "in_bounds_ratio": in_bounds_ratio,
        "world": world,
    }


def quality_pass(pred: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if not pred.get("ok"):
        return False, str(pred.get("error", "detect_failed"))
    if float(pred.get("score", 0.0)) < float(args.min_score):
        return False, "low_score"
    if float(pred.get("in_bounds_ratio", 0.0)) < float(args.min_in_bounds_ratio):
        return False, "out_of_bounds"
    area = float(pred.get("bbox_area_ratio", 0.0))
    if area < float(args.min_bbox_area_ratio):
        return False, "bbox_too_small"
    if area > float(args.max_bbox_area_ratio):
        return False, "bbox_too_large"
    return True, "accepted"


def points_from_uv(uv21: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, Any]]:
    points: dict[str, dict[str, Any]] = {}
    for idx, label in enumerate(MANO21_LABELS):
        if bool(valid[idx]):
            points[label] = {
                "x": round(float(uv21[idx, 0]), 3),
                "y": round(float(uv21[idx, 1]), 3),
                "visible": True,
            }
        else:
            points[label] = {"x": None, "y": None, "visible": False}
    return points


def draw_overlay(image_rgb: np.ndarray, uv21: np.ndarray | None, valid: np.ndarray | None, text: str) -> Image.Image:
    out = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    if uv21 is not None and valid is not None:
        arr = np.asarray(uv21, dtype=np.float32)
        mask = np.asarray(valid, dtype=bool).reshape(21)
        for a, b in HAND_BONES:
            if mask[a] and mask[b]:
                draw.line(
                    [tuple(map(float, arr[a])), tuple(map(float, arr[b]))],
                    fill=JOINT_COLORS_RGB[b],
                    width=2,
                )
        for idx, uv in enumerate(arr):
            if not mask[idx]:
                continue
            x, y = float(uv[0]), float(uv[1])
            r = 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=JOINT_COLORS_RGB[idx])
    draw.rectangle((0, 0, min(out.width, 430), 22), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255))
    return out


def evenly_spaced_indices(n: int, k: int) -> set[int]:
    if n <= 0 or k <= 0:
        return set()
    if k >= n:
        return set(range(n))
    return set(np.linspace(0, n - 1, k).round().astype(int).tolist())


def make_contact_sheet(image_paths: list[Path], output_path: Path, thumb_size: int = 160, cols: int = 6) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        try:
            im = Image.open(path).convert("RGB").resize((thumb_size, thumb_size))
        except Exception:
            continue
        thumbs.append((path, im))
    if not thumbs:
        return
    rows = int(math.ceil(len(thumbs) / float(cols)))
    cell_h = thumb_size + 24
    sheet = Image.new("RGB", (cols * thumb_size, rows * cell_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for idx, (path, im) in enumerate(thumbs):
        x = (idx % cols) * thumb_size
        y = (idx // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + thumb_size + 4), path.name[:28], fill=(230, 230, 230))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root or one episode dir.")
    parser.add_argument("--output", type=Path, required=True, help="Output annotation JSON.")
    parser.add_argument("--model-asset-path", type=Path, required=True)
    parser.add_argument("--episode", action="append", default=[], help="Episode name. Repeat for multiple episodes.")
    parser.add_argument("--source", choices=("pkl", "images"), default="pkl", help="Use PKL rgbImage paths or image directory listing.")
    parser.add_argument("--image-field", default="rgbImage")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-frames-per-episode", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.25)
    parser.add_argument("--min-presence-confidence", type=float, default=0.25)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.25)
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--min-in-bounds-ratio", type=float, default=0.90)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.65)
    parser.add_argument("--overlay-output-dir", type=Path, default=None)
    parser.add_argument("--max-overlays-per-episode", type=int, default=80, help="0 means save all processed overlays.")
    parser.add_argument("--save-rejected-overlays", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.model_asset_path = args.model_asset_path.expanduser().resolve()
    if not args.model_asset_path.is_file():
        raise FileNotFoundError(args.model_asset_path)

    frames = discover_frames_from_pkl(args) if args.source == "pkl" else discover_frames_from_images(args)
    if not frames:
        raise SystemExit(f"no frames selected from {args.data_root}")

    print(f"selected frames: {len(frames)}")
    print(f"first image: {frames[0].image}")
    print(f"output: {args.output}")
    if args.dry_run:
        for frame in frames[:20]:
            print(f"{frame.episode}\t{frame.frame_id}\t{frame.image}")
        if len(frames) > 20:
            print(f"... {len(frames) - 20} more")
        return 0

    mp, detector = load_mediapipe_detector(args)
    now = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    summary_by_episode: dict[str, dict[str, int]] = {}
    overlay_paths_by_episode: dict[str, list[Path]] = {}
    overlay_slots_by_episode: dict[str, set[int]] = {}
    frames_by_episode: dict[str, list[int]] = {}

    for idx, frame in enumerate(frames):
        frames_by_episode.setdefault(frame.episode, []).append(idx)
    for ep, indices in frames_by_episode.items():
        max_overlays = len(indices) if int(args.max_overlays_per_episode) == 0 else int(args.max_overlays_per_episode)
        local_slots = evenly_spaced_indices(len(indices), max_overlays)
        overlay_slots_by_episode[ep] = {indices[i] for i in local_slots}

    try:
        for global_idx, frame in enumerate(frames):
            ep_summary = summary_by_episode.setdefault(frame.episode, {"processed": 0, "accepted": 0, "rejected": 0})
            ep_summary["processed"] += 1
            pred = detect_one(mp, detector, frame.image)
            keep, reason = quality_pass(pred, args)
            score = float(pred.get("score", 0.0)) if pred.get("ok") else 0.0
            handedness = pred.get("handedness")

            should_overlay = (
                args.overlay_output_dir is not None
                and global_idx in overlay_slots_by_episode.get(frame.episode, set())
                and (keep or bool(args.save_rejected_overlays))
            )
            overlay_rel = None
            if should_overlay:
                overlay_ep_dir = args.overlay_output_dir / frame.episode
                overlay_ep_dir.mkdir(parents=True, exist_ok=True)
                overlay_name = f"{frame.frame_id:06d}_{frame.image.stem}_{reason}.jpg"
                overlay_path = overlay_ep_dir / overlay_name
                if pred.get("ok"):
                    overlay = draw_overlay(
                        pred["rgb"],
                        pred.get("uv21"),
                        pred.get("valid"),
                        f"{reason} score={score:.3f} {handedness or ''}",
                    )
                else:
                    with Image.open(frame.image) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    overlay = draw_overlay(rgb, None, None, reason)
                overlay.save(overlay_path, quality=95)
                overlay_paths_by_episode.setdefault(frame.episode, []).append(overlay_path)
                overlay_rel = str(overlay_path)

            if keep:
                ep_summary["accepted"] += 1
                valid = np.asarray(pred["valid"], dtype=bool).reshape(21)
                records.append(
                    {
                        "frame_id": int(frame.frame_id),
                        "episode": frame.episode,
                        "dataset": frame.dataset,
                        "source_group": frame.source_group,
                        "image": str(frame.image),
                        "image_rel": frame.image_rel,
                        "points": points_from_uv(np.asarray(pred["uv21"], dtype=np.float32), valid),
                        "notes": f"mediapipe pseudo label; score={score:.4f}; handedness={handedness}; review_status=unreviewed",
                        "pseudo_label": {
                            "method": "mediapipe_hand_landmarker",
                            "model_asset_path": str(args.model_asset_path),
                            "score": score,
                            "handedness": handedness,
                            "bbox_area_ratio": float(pred.get("bbox_area_ratio", 0.0)),
                            "in_bounds_ratio": float(pred.get("in_bounds_ratio", 0.0)),
                            "review_status": "unreviewed",
                            "overlay": overlay_rel,
                        },
                    }
                )
            else:
                ep_summary["rejected"] += 1
                rejected.append(
                    {
                        "episode": frame.episode,
                        "frame_id": int(frame.frame_id),
                        "image": str(frame.image),
                        "image_rel": frame.image_rel,
                        "reason": reason,
                        "score": score,
                        "handedness": handedness,
                        "overlay": overlay_rel,
                    }
                )
    finally:
        detector.close()

    payload = {
        "version": 1,
        "task": "visible_hand_points",
        "label_profile": "mano21",
        "created_at": now,
        "updated_at": now,
        "input": str(args.data_root),
        "output": str(args.output),
        "labels": list(MANO21_LABELS),
        "label_colors": LABEL_COLORS,
        "skeleton_edges": [list(edge) for edge in HAND_BONES_LABELS],
        "point_format": {
            "visible": {"x": "pixel x in original distorted RGB", "y": "pixel y in original distorted RGB", "visible": True},
            "occluded": {"x": None, "y": None, "visible": False},
        },
        "pseudo_label_source": {
            "method": "mediapipe_hand_landmarker",
            "model_asset_path": str(args.model_asset_path),
            "source": str(args.source),
            "image_field": str(args.image_field),
            "thresholds": {
                "min_score": float(args.min_score),
                "min_in_bounds_ratio": float(args.min_in_bounds_ratio),
                "min_bbox_area_ratio": float(args.min_bbox_area_ratio),
                "max_bbox_area_ratio": float(args.max_bbox_area_ratio),
            },
            "coordinate_contract": "raw/distorted RGB pixel coordinates; no undistortion applied",
        },
        "summary_by_episode": summary_by_episode,
        "frames": records,
    }
    write_json(args.output, payload)
    images_txt_path = args.output.with_suffix(".images.txt")
    images_txt_path.write_text(
        "\n".join(str(record["image"]) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    rejected_path = args.output.with_suffix(".rejected.json")
    write_json(rejected_path, {"created_at": now, "rejected": rejected, "summary_by_episode": summary_by_episode})

    if args.overlay_output_dir is not None:
        args.overlay_output_dir.mkdir(parents=True, exist_ok=True)
        for ep, paths in overlay_paths_by_episode.items():
            make_contact_sheet(paths, args.overlay_output_dir / f"{ep}_contact_sheet.jpg")
        gallery = {
            "annotation_json": str(args.output),
            "rejected_json": str(rejected_path),
            "summary_by_episode": summary_by_episode,
            "contact_sheets": [
                str(args.overlay_output_dir / f"{ep}_contact_sheet.jpg")
                for ep in sorted(overlay_paths_by_episode)
            ],
        }
        write_json(args.overlay_output_dir / "review_summary.json", gallery)
        print(f"review summary: {args.overlay_output_dir / 'review_summary.json'}")

    print(f"accepted frames: {len(records)}")
    print(f"rejected frames: {len(rejected)}")
    print(f"annotation: {args.output}")
    print(f"accepted image list: {images_txt_path}")
    print(f"rejected: {rejected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
