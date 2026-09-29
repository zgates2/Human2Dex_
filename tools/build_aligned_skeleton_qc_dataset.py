#!/usr/bin/env python3
"""Build a clean image dataset and an aligned hand-skeleton QC video.

This tool intentionally separates model inputs from visualization outputs:

* ``export`` creates a complete clean RGB image dataset from processed PKL
  episodes.  No skeleton or diagnostic graphics are baked into these images.
* ``annotate`` opens the existing browser MANO-21 annotator.  Predicted wrist
  points are pre-filled, so only visibly wrong joints need to be clicked.
* ``render`` interpolates the sparse manual corrections through time and
  creates polished skeleton/object/pocket diagnostic frames and MP4 videos.

Typical workflow::

    python tools/build_aligned_skeleton_qc_dataset.py export \
      --data-root /path/to/base_enriched_v1 \
      --output-root /path/to/aligned_skeleton_qc \
      --episode episode_0001 --episode episode_0010 \
      --annotation-stride 18

    python tools/build_aligned_skeleton_qc_dataset.py annotate \
      --output-root /path/to/aligned_skeleton_qc --port 8899

    python tools/build_aligned_skeleton_qc_dataset.py render \
      --output-root /path/to/aligned_skeleton_qc
"""

from __future__ import annotations

import argparse
import copy
import errno
import importlib.util
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANO21_LABELS = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
PALM_ARC = ((5, 9), (9, 13), (13, 17))

# RGB colors.  Conversion to OpenCV BGR happens only at draw time.
JOINT_COLORS_RGB = (
    [(245, 245, 245)]
    + [(255, 180, 80)] * 4
    + [(80, 255, 120)] * 4
    + [(80, 210, 255)] * 4
    + [(255, 120, 180)] * 4
    + [(180, 120, 255)] * 4
)
LABEL_COLORS_HEX = {
    name: "#%02x%02x%02x" % JOINT_COLORS_RGB[idx]
    for idx, name in enumerate(MANO21_LABELS)
}
OBJECT_COLORS_RGB = (
    (83, 214, 255),
    (255, 112, 154),
    (110, 238, 143),
    (186, 133, 255),
    (255, 194, 90),
    (90, 175, 255),
)
POCKET_RGB = (255, 224, 72)

_PIPELINE_COMMON = None
_PIPELINE_SKELETON = None


def load_original_pipeline_renderers():
    """Load the exact appearance and skeleton functions used by the pipeline."""
    global _PIPELINE_COMMON, _PIPELINE_SKELETON
    if _PIPELINE_COMMON is not None and _PIPELINE_SKELETON is not None:
        return _PIPELINE_COMMON, _PIPELINE_SKELETON
    pipeline_dir = Path(__file__).resolve().parents[1] / "glove_aug_pipeline"
    common_path = pipeline_dir / "common.py"
    skeleton_path = pipeline_dir / "wrist_skeleton_overlay.py"
    for module_path in (common_path, skeleton_path):
        if not module_path.is_file():
            raise FileNotFoundError(module_path)

    common_spec = importlib.util.spec_from_file_location("dexglove_qc_aug_common", common_path)
    skeleton_spec = importlib.util.spec_from_file_location("dexglove_qc_skeleton_overlay", skeleton_path)
    if common_spec is None or common_spec.loader is None or skeleton_spec is None or skeleton_spec.loader is None:
        raise ImportError("failed to create import specs for the original augmentation pipeline")
    common_module = importlib.util.module_from_spec(common_spec)
    skeleton_module = importlib.util.module_from_spec(skeleton_spec)
    sys.modules[common_spec.name] = common_module
    sys.modules[skeleton_spec.name] = skeleton_module
    common_spec.loader.exec_module(common_module)
    skeleton_spec.loader.exec_module(skeleton_module)
    _PIPELINE_COMMON = common_module
    _PIPELINE_SKELETON = skeleton_module
    return common_module, skeleton_module


def natural_key(value: str | Path) -> list[Any]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(value))]


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_episode_pkl(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        return payload["messages"], payload.get("metadata", {})
    if isinstance(payload, list):
        return payload, {}
    raise TypeError(f"unsupported episode PKL structure: {path} ({type(payload).__name__})")


def find_episode_pkl(episode_dir: Path) -> Path:
    preferred = episode_dir / f"{episode_dir.name}.pkl"
    if preferred.is_file():
        return preferred
    candidates = sorted(episode_dir.glob("*.pkl"), key=natural_key)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no PKL found under {episode_dir}")
    raise RuntimeError(f"multiple PKLs under {episode_dir}; cannot choose: {candidates}")


def discover_episode_dirs(data_root: Path, requested: Sequence[str], max_episodes: int | None) -> list[Path]:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if (data_root / f"{data_root.name}.pkl").is_file() or list(data_root.glob("*.pkl")):
        candidates = [data_root]
    else:
        candidates = [p for p in data_root.iterdir() if p.is_dir() and list(p.glob("*.pkl"))]
        candidates.sort(key=natural_key)
    if requested:
        by_name = {p.name: p for p in candidates}
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise FileNotFoundError(f"episodes not found: {', '.join(missing)}")
        candidates = [by_name[name] for name in requested]
    if max_episodes is not None:
        candidates = candidates[: max(0, int(max_episodes))]
    if not candidates:
        raise RuntimeError(f"no episode directories discovered under {data_root}")
    return candidates


def resolve_image_path(episode_dir: Path, image_field: Any) -> Path | None:
    if not isinstance(image_field, (str, os.PathLike)) or not str(image_field):
        return None
    path = Path(image_field)
    if not path.is_absolute():
        path = episode_dir / path
    return path.resolve()


def safe_link_or_copy(source: Path, target: Path, mode: str, overwrite: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(f"output already exists: {target}; pass --overwrite to replace it")
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
        return "copy"
    if mode == "symlink":
        target.symlink_to(source)
        return "symlink"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK}:
            raise
        shutil.copy2(source, target)
        return "copy_fallback"


def as_uv(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 2 or not np.isfinite(arr[:2]).all():
        return None
    return [float(arr[0]), float(arr[1])]


def as_uv21(value: Any) -> list[list[float]] | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(21, 2)
    except (TypeError, ValueError):
        return None
    result: list[list[float]] = []
    for point in arr:
        if np.isfinite(point).all():
            result.append([float(point[0]), float(point[1])])
        else:
            result.append([float("nan"), float("nan")])
    return result


def as_valid21(value: Any, uv21: list[list[float]] | None) -> list[bool]:
    if uv21 is None:
        return [False] * 21
    try:
        arr = np.asarray(value, dtype=bool).reshape(21)
        return [bool(x) and bool(np.isfinite(uv21[i]).all()) for i, x in enumerate(arr)]
    except (TypeError, ValueError):
        return [bool(np.isfinite(point).all()) for point in uv21]


def discover_object_names(message: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    pattern = re.compile(r"^task_object_(.+)_policy_uv$")
    for key in message:
        match = pattern.match(key)
        if match:
            names.add(match.group(1))
    return sorted(names, key=natural_key)


def extract_objects(message: dict[str, Any], allowed_names: set[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = discover_object_names(message)
    for name in names:
        if allowed_names is not None and name not in allowed_names:
            continue
        prefix = f"task_object_{name}_"
        fields = {
            key[len(prefix):]: jsonable(value)
            for key, value in message.items()
            if key.startswith(prefix)
        }
        result[name] = fields
    return result


def annotation_points(uv21: list[list[float]] | None, valid21: list[bool]) -> dict[str, Any]:
    points: dict[str, Any] = {}
    for idx, label in enumerate(MANO21_LABELS):
        if uv21 is not None and valid21[idx] and np.isfinite(uv21[idx]).all():
            points[label] = {
                "x": round(float(uv21[idx][0]), 3),
                "y": round(float(uv21[idx][1]), 3),
                "visible": True,
            }
        else:
            points[label] = {"x": None, "y": None, "visible": False}
    return points


def select_annotation_records(records: list[dict[str, Any]], stride: int, max_per_episode: int | None) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[record["episode"]].append(record)
    selected: list[dict[str, Any]] = []
    for episode in sorted(by_episode, key=natural_key):
        episode_records = sorted(by_episode[episode], key=lambda r: int(r["frame_index"]))
        indices = list(range(0, len(episode_records), max(1, int(stride))))
        if episode_records and (not indices or indices[-1] != len(episode_records) - 1):
            indices.append(len(episode_records) - 1)
        if max_per_episode is not None and len(indices) > max_per_episode:
            indices = np.linspace(0, len(episode_records) - 1, int(max_per_episode)).round().astype(int).tolist()
            indices = sorted(set(indices))
        selected.extend(episode_records[i] for i in indices)
    return selected


def build_annotation_payload(output_root: Path, selected: list[dict[str, Any]]) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    frames: list[dict[str, Any]] = []
    for frame_id, record in enumerate(selected):
        image = Path(record["exported_image"]).resolve()
        frames.append({
            "frame_id": frame_id,
            "episode": record["episode"],
            "image": str(image),
            "image_rel": str(image.relative_to(output_root.resolve())),
            "points": annotation_points(record["skeleton"]["uv21"], record["skeleton"]["valid21"]),
            "notes": f"source_frame_index={record['frame_index']}; predicted points pre-filled",
        })
    return {
        "version": 1,
        "task": "visible_hand_points",
        "created_at": now,
        "updated_at": now,
        "input": str((output_root / "annotation_images.txt").resolve()),
        "output": str((output_root / "skeleton_keyframes.json").resolve()),
        "labels": list(MANO21_LABELS),
        "label_colors": LABEL_COLORS_HEX,
        "skeleton_edges": [[MANO21_LABELS[a], MANO21_LABELS[b]] for a, b in HAND_BONES],
        "point_format": {
            "visible": {"x": "pixel x", "y": "pixel y", "visible": True},
            "occluded": {"x": None, "y": None, "visible": False},
        },
        "frames": frames,
    }


def annotation_frame_key(frame: dict[str, Any]) -> tuple[str, str]:
    return str(frame.get("episode", "")), Path(str(frame.get("image", ""))).name


def preserve_existing_annotation_points(
    payload: dict[str, Any],
    previous: dict[str, Any] | None,
) -> int:
    if not previous:
        return 0
    previous_by_key = {
        annotation_frame_key(frame): frame
        for frame in previous.get("frames", [])
        if isinstance(frame, dict)
    }
    preserved = 0
    for frame in payload.get("frames", []):
        old = previous_by_key.get(annotation_frame_key(frame))
        if old is None or not isinstance(old.get("points"), dict):
            continue
        frame["points"] = old["points"]
        frame["notes"] = old.get("notes", frame.get("notes", ""))
        preserved += 1
    return preserved


def command_export(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.jsonl"
    annotation_path = output_root / "skeleton_keyframes.json"
    previous_annotations: dict[str, Any] | None = None
    if annotation_path.is_file() and args.overwrite:
        try:
            previous_annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"existing annotation JSON is invalid: {annotation_path}") from exc
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; use --overwrite to rebuild")

    requested_objects = set(args.object) if args.object else None
    episode_dirs = discover_episode_dirs(data_root, args.episode, args.max_episodes)
    records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    link_stats: defaultdict[str, int] = defaultdict(int)

    for episode_dir in episode_dirs:
        pkl_path = find_episode_pkl(episode_dir)
        messages, metadata = load_episode_pkl(pkl_path)
        if args.limit_frames_per_episode is not None:
            messages = messages[: max(0, int(args.limit_frames_per_episode))]
        exported = 0
        missing = 0
        object_names: set[str] = set()
        fps = float(metadata.get("collectionHz", 30.0) or 30.0)
        for frame_index, message in enumerate(messages):
            source_image = resolve_image_path(episode_dir, message.get(args.image_field))
            if source_image is None or not source_image.is_file():
                missing += 1
                if args.skip_missing:
                    continue
                raise FileNotFoundError(
                    f"missing {args.image_field} for {episode_dir.name} frame {frame_index}: {source_image}"
                )
            suffix = source_image.suffix.lower() if source_image.suffix.lower() in IMAGE_EXTENSIONS else ".jpg"
            exported_image = output_root / "images" / episode_dir.name / f"frame_{frame_index:06d}{suffix}"
            actual_mode = safe_link_or_copy(source_image, exported_image, args.link_mode, args.overwrite)
            link_stats[actual_mode] += 1

            uv21 = as_uv21(message.get("wrist_uv21_rgb"))
            valid21 = as_valid21(message.get("wrist_uv21_valid"), uv21)
            objects = extract_objects(message, requested_objects)
            object_names.update(objects)
            pocket_uv = as_uv(message.get("wrist_grasp_pocket_uv"))
            record = {
                "episode": episode_dir.name,
                "frame_index": frame_index,
                "fps": fps,
                "source_pkl": str(pkl_path),
                "source_image": str(source_image),
                "exported_image": str(exported_image.resolve()),
                "rgb_frame_id": jsonable(message.get("rgbFrameId")),
                "timestamp": jsonable(message.get("timestamp")),
                "skeleton": {
                    "uv21": uv21,
                    "valid21": valid21,
                    "error": jsonable(message.get("wrist_projection_error")),
                },
                "pocket": {
                    "uv": pocket_uv,
                    "valid": bool(message.get("wrist_grasp_pocket_valid", pocket_uv is not None)),
                    "confidence": jsonable(message.get("wrist_grasp_pocket_confidence")),
                    "source": jsonable(message.get("wrist_grasp_pocket_source")),
                    "error": jsonable(message.get("wrist_grasp_pocket_error")),
                },
                "objects": objects,
                "objectPocketObs": jsonable(message.get("objectPocketObs")),
                "objectPocketObsValid": jsonable(message.get("objectPocketObsValid")),
                "objectPocketObsError": jsonable(message.get("objectPocketObsError")),
            }
            records.append(record)
            exported += 1
        episode_summaries.append({
            "episode": episode_dir.name,
            "pkl": str(pkl_path),
            "messages": len(messages),
            "exported": exported,
            "missing": missing,
            "fps": fps,
            "objects": sorted(object_names, key=natural_key),
        })
        print(json.dumps(episode_summaries[-1], ensure_ascii=False))

    if not records:
        raise RuntimeError("no frames were exported")
    records.sort(key=lambda r: (natural_key(r["episode"]), int(r["frame_index"])))
    atomic_write_jsonl(manifest_path, records)

    selected = select_annotation_records(records, args.annotation_stride, args.annotation_max_frames_per_episode)
    if previous_annotations:
        record_by_key = {
            (str(record["episode"]), Path(record["exported_image"]).name): record
            for record in records
        }
        selected_keys = {
            (str(record["episode"]), Path(record["exported_image"]).name)
            for record in selected
        }
        for old_frame in previous_annotations.get("frames", []):
            key = annotation_frame_key(old_frame)
            record = record_by_key.get(key)
            if record is not None and key not in selected_keys:
                selected.append(record)
                selected_keys.add(key)
        selected.sort(key=lambda r: (natural_key(r["episode"]), int(r["frame_index"])))
    image_list_path = output_root / "annotation_images.txt"
    image_list_path.write_text("".join(f"{r['exported_image']}\n" for r in selected), encoding="utf-8")
    if not annotation_path.exists() or args.overwrite:
        annotation_payload = build_annotation_payload(output_root, selected)
        preserved_annotations = preserve_existing_annotation_points(annotation_payload, previous_annotations)
        atomic_write_json(annotation_path, annotation_payload)
    else:
        preserved_annotations = 0

    dataset_info = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "clean_images_have_skeleton": False,
        "manifest": str(manifest_path),
        "annotation_images": str(image_list_path),
        "skeleton_keyframes": str(annotation_path),
        "annotation_stride": int(args.annotation_stride),
        "frames": len(records),
        "annotation_frames": len(selected),
        "preserved_annotation_frames": preserved_annotations,
        "link_stats": dict(link_stats),
        "episodes": episode_summaries,
    }
    atomic_write_json(output_root / "dataset_info.json", dataset_info)
    print(f"clean image dataset: {output_root / 'images'}")
    print(f"manifest: {manifest_path}")
    print(f"annotation keyframes: {len(selected)} -> {annotation_path}")
    if preserved_annotations:
        print(f"preserved existing annotations: {preserved_annotations}")
    print("next: run the annotate subcommand and correct only visibly misaligned joints")
    return 0


def command_annotate(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    image_list = output_root / "annotation_images.txt"
    annotation_path = output_root / "skeleton_keyframes.json"
    if not image_list.is_file() or not annotation_path.is_file():
        raise FileNotFoundError("run export first; annotation_images.txt or skeleton_keyframes.json is missing")
    script = Path(__file__).resolve().parent / "click_label_visible_hand_points.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    command = [
        sys.executable,
        str(script),
        "--input", str(image_list),
        "--output", str(annotation_path),
        "--label-profile", "mano21",
        "--host", str(args.host),
        "--port", str(args.port),
    ]
    if args.open_browser:
        command.append("--open-browser")
    print("launching:")
    print(" ".join(command))
    print("Predictions are pre-filled. Select a wrong joint and click its correct location; use X for occluded joints.")
    return int(subprocess.run(command, check=False).returncode)


def rgb_to_bgr(color: Sequence[int]) -> tuple[int, int, int]:
    return int(color[2]), int(color[1]), int(color[0])


def parse_manual_annotations(
    annotation_path: Path,
    manifest_by_image: dict[str, dict[str, Any]],
    alignment_mode: str,
    sparse_edit_threshold_px: float,
) -> dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    result: defaultdict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for frame in payload.get("frames", []):
        image = str(Path(frame.get("image", "")).resolve())
        record = manifest_by_image.get(image)
        if record is None:
            continue
        predicted = record.get("skeleton", {}).get("uv21")
        predicted_valid = record.get("skeleton", {}).get("valid21", [False] * 21)
        if predicted is None:
            continue
        predicted_arr = np.asarray(predicted, dtype=np.float32).reshape(21, 2)
        correction = np.zeros((21, 2), dtype=np.float32)
        correction_valid = np.zeros(21, dtype=bool)
        manual_visible = np.zeros(21, dtype=bool)
        manual_arr = predicted_arr.copy()
        points = frame.get("points", {})
        for joint_idx, label in enumerate(MANO21_LABELS):
            point = points.get(label)
            if not isinstance(point, dict) or not bool(point.get("visible", False)):
                continue
            manual_visible[joint_idx] = True
            if not bool(predicted_valid[joint_idx]) or not np.isfinite(predicted_arr[joint_idx]).all():
                continue
            try:
                manual = np.array([float(point["x"]), float(point["y"])], dtype=np.float32)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(manual).all():
                manual_arr[joint_idx] = manual
                correction[joint_idx] = manual - predicted_arr[joint_idx]
                correction_valid[joint_idx] = True

        if alignment_mode == "sparse_affine":
            predicted_valid_arr = np.asarray(predicted_valid, dtype=bool).reshape(21)
            predicted_valid_arr &= np.isfinite(predicted_arr).all(axis=1)
            changed = correction_valid & (
                np.linalg.norm(correction, axis=1) > float(sparse_edit_threshold_px)
            )
            changed_indices = np.flatnonzero(changed)
            global_correction = np.zeros_like(correction)
            if len(changed_indices) == 1:
                global_correction[:] = correction[changed_indices[0]]
            elif len(changed_indices) >= 2:
                source = predicted_arr[changed_indices].reshape(-1, 1, 2)
                target = manual_arr[changed_indices].reshape(-1, 1, 2)
                transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
                if transform is not None and np.isfinite(transform).all():
                    homogeneous = np.concatenate(
                        [predicted_arr, np.ones((21, 1), dtype=np.float32)], axis=1
                    )
                    transformed = homogeneous @ np.asarray(transform, dtype=np.float32).T
                    global_correction = transformed - predicted_arr
                else:
                    global_correction[:] = np.median(correction[changed_indices], axis=0)
            if len(changed_indices) > 0:
                correction = global_correction
                # Manually clicked points remain exact after the global hand transform.
                correction[changed_indices] = manual_arr[changed_indices] - predicted_arr[changed_indices]
            else:
                correction[:] = 0.0
            correction_valid = predicted_valid_arr
        result[record["episode"]].append(
            (int(record["frame_index"]), correction, correction_valid, manual_visible)
        )
    for episode in result:
        result[episode].sort(key=lambda item: item[0])
    return dict(result)


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(values) < 2:
        return values
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    output = np.empty_like(values)
    for joint in range(values.shape[1]):
        for axis in range(2):
            output[:, joint, axis] = np.convolve(padded[:, joint, axis], kernel, mode="valid")
    return output


def smooth_uv_sequence(
    values: Sequence[Any],
    validity: Sequence[bool],
    *,
    median_window: int,
    gaussian_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(values)
    array = np.full((count, 2), np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for idx, (value, is_valid) in enumerate(zip(values, validity)):
        uv = as_uv(value)
        if bool(is_valid) and uv is not None:
            array[idx] = np.asarray(uv, dtype=np.float32)
            valid[idx] = True
    if not valid.any():
        return array, valid

    filtered = array.copy()
    radius = max(0, int(median_window) // 2)
    if radius > 0:
        for idx in np.flatnonzero(valid):
            start = max(0, int(idx) - radius)
            stop = min(count, int(idx) + radius + 1)
            neighbours = array[start:stop][valid[start:stop]]
            if len(neighbours):
                filtered[idx] = np.median(neighbours, axis=0)

    valid_indices = np.flatnonzero(valid).astype(np.float32)
    all_indices = np.arange(count, dtype=np.float32)
    filled = np.empty_like(filtered)
    for axis in range(2):
        filled[:, axis] = np.interp(all_indices, valid_indices, filtered[valid, axis])

    if gaussian_sigma > 0 and count > 1:
        radius = max(1, int(math.ceil(3.0 * gaussian_sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / float(gaussian_sigma)) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(filled, ((radius, radius), (0, 0)), mode="edge")
        for axis in range(2):
            filled[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    return filled, valid


def build_center_smoothed_records(
    records: list[dict[str, Any]],
    object_names: list[str],
    *,
    median_window: int,
    gaussian_sigma: float,
) -> list[dict[str, Any]]:
    output = copy.deepcopy(records)
    pocket_uv, pocket_valid = smooth_uv_sequence(
        [record.get("pocket", {}).get("uv") for record in records],
        [bool(record.get("pocket", {}).get("valid", False)) for record in records],
        median_window=median_window,
        gaussian_sigma=gaussian_sigma,
    )
    for idx, record in enumerate(output):
        if pocket_valid[idx]:
            record["pocket"]["uv"] = pocket_uv[idx].tolist()

    for name in object_names:
        object_uv, object_valid = smooth_uv_sequence(
            [record.get("objects", {}).get(name, {}).get("policy_uv") for record in records],
            [bool(record.get("objects", {}).get(name, {}).get("memory_valid", False)) for record in records],
            median_window=median_window,
            gaussian_sigma=gaussian_sigma,
        )
        for idx, record in enumerate(output):
            fields = record.get("objects", {}).get(name)
            if fields is None or not object_valid[idx]:
                continue
            fields["policy_uv"] = object_uv[idx].tolist()
            if pocket_valid[idx]:
                delta = object_uv[idx] - pocket_uv[idx]
                fields["pocket_delta_uv"] = delta.tolist()
                fields["pocket_distance_px"] = float(np.linalg.norm(delta))
    return output


def build_correction_track(
    frame_indices: np.ndarray,
    keyframes: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    sigma: float,
    max_correction_px: float | None,
) -> np.ndarray:
    corrections = np.zeros((len(frame_indices), 21, 2), dtype=np.float32)
    if not keyframes:
        return corrections
    for joint in range(21):
        xs: list[float] = []
        vals_x: list[float] = []
        vals_y: list[float] = []
        for frame_index, corr, valid, _manual_visible in keyframes:
            if valid[joint]:
                xs.append(float(frame_index))
                vals_x.append(float(corr[joint, 0]))
                vals_y.append(float(corr[joint, 1]))
        if not xs:
            continue
        corrections[:, joint, 0] = np.interp(frame_indices, xs, vals_x)
        corrections[:, joint, 1] = np.interp(frame_indices, xs, vals_y)
    corrections = gaussian_smooth(corrections, sigma)
    if max_correction_px is not None and max_correction_px > 0:
        norms = np.linalg.norm(corrections, axis=2, keepdims=True)
        scale = np.minimum(1.0, float(max_correction_px) / np.maximum(norms, 1e-6))
        corrections *= scale
    return corrections


def build_visibility_track(
    frame_indices: np.ndarray,
    keyframes: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    visibility = np.ones((len(frame_indices), 21), dtype=bool)
    if not keyframes:
        return visibility
    keyframe_indices = np.asarray([item[0] for item in keyframes], dtype=np.float32)
    keyframe_visibility = np.stack([item[3] for item in keyframes], axis=0).astype(np.float32)
    for joint in range(21):
        visibility[:, joint] = np.interp(
            frame_indices,
            keyframe_indices,
            keyframe_visibility[:, joint],
        ) >= 0.5
    return visibility


def draw_text_with_shadow(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (8, 10, 14), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 14,
) -> None:
    p0 = np.asarray(start, dtype=np.float32)
    p1 = np.asarray(end, dtype=np.float32)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1:
        return
    direction = (p1 - p0) / length
    position = 0.0
    while position < length:
        a = p0 + direction * position
        b = p0 + direction * min(position + dash, length)
        cv2.line(image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, thickness, cv2.LINE_AA)
        position += dash * 1.65


def draw_skeleton(image: np.ndarray, uv21: np.ndarray, valid21: np.ndarray, scale: int, draw_palm_arc: bool) -> None:
    dark = (12, 14, 19)
    white = (245, 245, 245)
    edges = list(HAND_BONES) + (list(PALM_ARC) if draw_palm_arc else [])
    for a, b in edges:
        if not (valid21[a] and valid21[b]):
            continue
        p0 = tuple(np.rint(uv21[a] * scale).astype(int))
        p1 = tuple(np.rint(uv21[b] * scale).astype(int))
        color = rgb_to_bgr(JOINT_COLORS_RGB[b])
        cv2.line(image, p0, p1, dark, max(4, int(round(3.4 * scale))), cv2.LINE_AA)
        cv2.line(image, p0, p1, white, max(3, int(round(2.4 * scale))), cv2.LINE_AA)
        cv2.line(image, p0, p1, color, max(2, int(round(1.35 * scale))), cv2.LINE_AA)
    for idx, point in enumerate(uv21):
        if not valid21[idx]:
            continue
        center = tuple(np.rint(point * scale).astype(int))
        color = rgb_to_bgr(JOINT_COLORS_RGB[idx])
        cv2.circle(image, center, max(5, 3 * scale), dark, -1, cv2.LINE_AA)
        cv2.circle(image, center, max(4, 2 * scale + 1), white, -1, cv2.LINE_AA)
        cv2.circle(image, center, max(2, int(round(1.35 * scale))), color, -1, cv2.LINE_AA)


def draw_pocket_and_objects(
    image: np.ndarray,
    record: dict[str, Any],
    object_names: list[str],
) -> None:
    pocket = record.get("pocket", {})
    pocket_uv = as_uv(pocket.get("uv"))
    pocket_valid = bool(pocket.get("valid", False)) and pocket_uv is not None
    if pocket_valid:
        p = tuple(np.rint(np.asarray(pocket_uv)).astype(int))
        yellow = rgb_to_bgr(POCKET_RGB)
        cv2.circle(image, p, 7, yellow, 2, cv2.LINE_AA)
        cv2.circle(image, p, 2, yellow, -1, cv2.LINE_AA)
        cv2.line(image, (p[0] - 10, p[1]), (p[0] + 10, p[1]), yellow, 1, cv2.LINE_AA)
        cv2.line(image, (p[0], p[1] - 10), (p[0], p[1] + 10), yellow, 1, cv2.LINE_AA)

    for object_index, name in enumerate(object_names):
        fields = record.get("objects", {}).get(name, {})
        uv = as_uv(fields.get("policy_uv"))
        memory_valid = bool(fields.get("memory_valid", False)) and uv is not None
        if not memory_valid:
            continue
        color = rgb_to_bgr(OBJECT_COLORS_RGB[object_index % len(OBJECT_COLORS_RGB)])
        obj = tuple(np.rint(np.asarray(uv)).astype(int))
        if pocket_valid:
            p = tuple(np.rint(np.asarray(pocket_uv)).astype(int))
            # Deliberately one colored layer only: no black/white halo around the dashed error line.
            draw_dashed_line(image, p, obj, color, 2, dash=7)
            delta = fields.get("pocket_delta_uv")
            if not isinstance(delta, (list, tuple)) or len(delta) < 2:
                delta = [float(obj[0] - p[0]), float(obj[1] - p[1])]
            distance = fields.get("pocket_distance_px")
            if distance is None:
                distance = float(np.linalg.norm(np.asarray(obj, dtype=np.float32) - np.asarray(p, dtype=np.float32)))
            midpoint = np.asarray([(p[0] + obj[0]) * 0.5, (p[1] + obj[1]) * 0.5], dtype=np.float32)
            direction = np.asarray([obj[0] - p[0], obj[1] - p[1]], dtype=np.float32)
            norm = max(float(np.linalg.norm(direction)), 1.0)
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32) / norm
            normal *= 18.0 if object_index % 2 == 0 else -18.0
            label_center = midpoint + normal
            error_text = (
                f"{compact_object_label(name)}  ({format_number(delta[0], 1)}, "
                f"{format_number(delta[1], 1)})  {format_number(distance, 1)}px"
            )
            text_size, _ = cv2.getTextSize(error_text, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            text_x = int(np.clip(label_center[0] - text_size[0] * 0.5, 4, max(4, image.shape[1] - text_size[0] - 4)))
            text_y = int(np.clip(label_center[1], text_size[1] + 4, image.shape[0] - 6))
            cv2.putText(
                image,
                error_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.circle(image, obj, 6, color, 2, cv2.LINE_AA)
        cv2.circle(image, obj, 2, color, -1, cv2.LINE_AA)


def rounded_rectangle(image: np.ndarray, xyxy: tuple[int, int, int, int], radius: int, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = xyxy
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)


def format_number(value: Any, digits: int = 2, fallback: str = "--") -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(numeric):
        return fallback
    return f"{numeric:.{digits}f}"


def first_error(*values: Any) -> str | None:
    for value in values:
        if value not in (None, "", False, []):
            return str(value)
    return None


def compact_object_label(name: str) -> str:
    tokens = [token for token in name.replace("-", "_").split("_") if token]
    generic = {"disposable", "plastic", "small", "task", "object"}
    compact = [token for token in tokens if token not in generic]
    if len(compact) > 3:
        compact = [compact[0], compact[-1]]
    return " ".join(compact or tokens)[:18]


def infer_hand_mask_root(output_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    info_path = output_root / "dataset_info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        base_root = Path(info["data_root"]).expanduser().resolve()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    candidate = base_root.parent / "sam3_hand_masks_v1" / "masks"
    return candidate if candidate.is_dir() else None


def load_hand_mask(mask_root: Path, record: dict[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    source_stem = Path(record["source_image"]).stem
    episode = str(record["episode"])
    candidates = (
        mask_root / episode / f"{source_stem}_mask.png",
        mask_root / episode / "images" / f"{source_stem}_mask.png",
    )
    mask_path = next((path for path in candidates if path.is_file()), None)
    if mask_path is None:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def make_original_appearance_style(common_module: Any, episode: str, seed: int, variant: int) -> Any:
    """Reproduce the exact extreme/mixed/normal settings used by stage 07."""
    return common_module.make_style(
        common_module.episode_seed(int(seed), episode),
        variant=int(variant),
        glove_mix_range=(0.82, 1.00),
        glove_brightness_range=(0.60, 1.45),
        glove_contrast_range=(0.55, 1.85),
        glove_texture_strength_range=(16.0, 34.0),
        glove_noise_strength_range=(8.0, 18.0),
        glove_rib_strength_range=(3.0, 12.0),
        glove_speckle_strength_range=(4.0, 14.0),
        glove_sheen_strength_range=(4.0, 18.0),
        glove_shade_contrast_range=(0.42, 2.10),
        background_brightness_range=(0.88, 1.12),
        background_contrast_range=(0.90, 1.16),
        background_hue_shift_range=(-0.04, 0.04),
        background_saturation_range=(0.90, 1.12),
        background_mix_range=(0.85, 1.0),
    )


def render_dashboard(
    height: int,
    width: int,
    record: dict[str, Any],
    object_names: list[str],
    correction: np.ndarray,
    keyframe_count: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), (17, 20, 27), dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, height), (60, 67, 81), 2, cv2.LINE_AA)
    margin = 20
    y = 34
    draw_text_with_shadow(panel, "HAND-OBJECT ALIGNMENT", (margin, y), 0.62, (245, 247, 252), 2)
    y += 25
    draw_text_with_shadow(
        panel,
        f"{record['episode']}   frame {int(record['frame_index']):06d}",
        (margin, y), 0.43, (156, 168, 188), 1,
    )
    y += 17
    cv2.line(panel, (margin, y), (width - margin, y), (55, 62, 75), 1, cv2.LINE_AA)
    y += 26

    pocket = record.get("pocket", {})
    pocket_valid = bool(pocket.get("valid", False)) and as_uv(pocket.get("uv")) is not None
    pocket_color = rgb_to_bgr(POCKET_RGB) if pocket_valid else (105, 112, 125)
    cv2.circle(panel, (margin + 6, y - 5), 6, pocket_color, -1, cv2.LINE_AA)
    draw_text_with_shadow(panel, "GRASP POCKET", (margin + 20, y), 0.5, (236, 238, 244), 1)
    y += 21
    uv = as_uv(pocket.get("uv"))
    uv_text = f"uv ({format_number(uv[0], 1)}, {format_number(uv[1], 1)})" if uv else "uv (--, --)"
    conf = format_number(pocket.get("confidence"), 2)
    draw_text_with_shadow(panel, f"{uv_text}   confidence {conf}", (margin, y), 0.40, (174, 184, 201), 1)
    y += 18
    mean_corr = float(np.linalg.norm(correction, axis=1).mean()) if correction.size else 0.0
    draw_text_with_shadow(panel, f"manual alignment {mean_corr:.1f}px   keyframes {keyframe_count}", (margin, y), 0.40, (174, 184, 201), 1)
    y += 18
    pocket_error = first_error(pocket.get("error"), record.get("skeleton", {}).get("error"))
    if pocket_error:
        draw_text_with_shadow(panel, f"error: {pocket_error[:40]}", (margin, y), 0.38, (95, 115, 255), 1)
        y += 18
    y += 7

    card_gap = 10
    remaining = max(1, len(object_names))
    card_height = max(104, min(145, (height - y - margin - card_gap * (remaining - 1)) // remaining))
    for object_index, name in enumerate(object_names):
        if y + 75 > height:
            break
        fields = record.get("objects", {}).get(name, {})
        color = rgb_to_bgr(OBJECT_COLORS_RGB[object_index % len(OBJECT_COLORS_RGB)])
        memory_valid = bool(fields.get("memory_valid", False))
        visible = bool(fields.get("policy_visible", fields.get("raw_visible", False)))
        rounded_rectangle(panel, (margin - 4, y - 14, width - margin, min(height - 8, y + card_height)), 12, (26, 31, 41))
        cv2.rectangle(panel, (margin - 4, y - 14), (margin, min(height - 8, y + card_height)), color, -1)
        display_name = name.replace("_", " ").upper()
        draw_text_with_shadow(panel, display_name[:35], (margin + 10, y + 5), 0.45, (238, 240, 246), 1)
        status = "LIVE" if visible else ("HOLD" if memory_valid else "INVALID")
        status_color = (112, 230, 150) if visible else ((92, 194, 255) if memory_valid else (105, 112, 125))
        draw_text_with_shadow(panel, status, (width - margin - 72, y + 5), 0.40, status_color, 1)
        delta = fields.get("pocket_delta_uv")
        if not isinstance(delta, (list, tuple)) or len(delta) < 2:
            delta = [None, None]
        line1 = (
            f"du {format_number(delta[0], 1):>6}   dv {format_number(delta[1], 1):>6}   "
            f"dist {format_number(fields.get('pocket_distance_px'), 1):>6}px"
        )
        draw_text_with_shadow(panel, line1, (margin + 10, y + 30), 0.40, (184, 192, 207), 1)
        line2 = (
            f"area {format_number(fields.get('policy_mask_area'), 0):>6}px2   "
            f"conf {format_number(fields.get('policy_confidence'), 2)}   age {fields.get('frames_since_seen', '--')}"
        )
        draw_text_with_shadow(panel, line2, (margin + 10, y + 51), 0.40, (184, 192, 207), 1)
        obs = fields.get("obs")
        if isinstance(obs, (list, tuple)) and len(obs) >= 3:
            line3 = f"obs dx {format_number(obs[0])}   dy {format_number(obs[1])}   log_area {format_number(obs[2])}"
            draw_text_with_shadow(panel, line3, (margin + 10, y + 72), 0.38, (147, 158, 177), 1)
        obj_error = first_error(fields.get("error"), fields.get("raw_error"))
        if obj_error:
            draw_text_with_shadow(panel, f"error: {obj_error[:42]}", (margin + 10, y + 93), 0.36, (95, 115, 255), 1)
        y += card_height + card_gap
    return panel


def encode_video_ffmpeg(frames_dir: Path, video_path: Path, fps: float) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:g}",
        "-i", str(frames_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video_path),
    ]
    subprocess.run(command, check=True)


def command_render(args: argparse.Namespace) -> int:
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.jsonl"
    annotation_path = output_root / "skeleton_keyframes.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run export first: {manifest_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    records = read_jsonl(manifest_path)
    if args.episode:
        requested = set(args.episode)
        records = [record for record in records if record["episode"] in requested]
    if not records:
        raise RuntimeError("no manifest records selected")
    manifest_by_image = {str(Path(r["exported_image"]).resolve()): r for r in records}
    manual_by_episode = parse_manual_annotations(
        annotation_path,
        manifest_by_image,
        alignment_mode=str(args.alignment_mode),
        sparse_edit_threshold_px=float(args.sparse_edit_threshold_px),
    )
    common_module, skeleton_module = load_original_pipeline_renderers()
    hand_mask_root = infer_hand_mask_root(output_root, args.hand_mask_root)
    if args.apply_hand_appearance and (hand_mask_root is None or not hand_mask_root.is_dir()):
        raise FileNotFoundError(
            "hand appearance augmentation is enabled but no SAM3 hand-mask root was found; "
            "pass --hand-mask-root or use --no-apply-hand-appearance"
        )
    if hand_mask_root is not None:
        print(f"hand masks: {hand_mask_root}")

    by_episode: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[record["episode"]].append(record)
    for episode in by_episode:
        by_episode[episode].sort(key=lambda r: int(r["frame_index"]))

    for episode in sorted(by_episode, key=natural_key):
        episode_records = by_episode[episode]
        appearance_style = make_original_appearance_style(
            common_module,
            episode,
            seed=int(args.appearance_seed),
            variant=int(args.appearance_variant),
        )
        frame_indices = np.asarray([int(r["frame_index"]) for r in episode_records], dtype=np.float32)
        keyframes = manual_by_episode.get(episode, [])
        correction_track = build_correction_track(
            frame_indices,
            keyframes,
            sigma=float(args.smooth_sigma),
            max_correction_px=args.max_correction_px,
        )
        visibility_track = build_visibility_track(frame_indices, keyframes)
        object_names = sorted(
            {name for record in episode_records for name in record.get("objects", {})},
            key=natural_key,
        )
        render_records = build_center_smoothed_records(
            episode_records,
            object_names,
            median_window=int(args.center_median_window),
            gaussian_sigma=float(args.center_smooth_sigma),
        )
        frames_dir = output_root / "rendered_frames" / episode
        if frames_dir.exists() and args.overwrite:
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        rendered_count = 0
        missing_hand_masks = 0
        for output_index, (record, correction, manual_visibility) in enumerate(
            zip(render_records, correction_track, visibility_track)
        ):
            source = Path(record["exported_image"])
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read image: {source}")
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if args.apply_hand_appearance:
                assert hand_mask_root is not None
                hand_mask = load_hand_mask(hand_mask_root, record, (h, w))
                if hand_mask is None:
                    missing_hand_masks += 1
                else:
                    image_rgb = common_module.apply_glove_augmentation(
                        image_rgb,
                        hand_mask,
                        appearance_style,
                        int(record["frame_index"]),
                        feather_radius=float(args.feather_radius),
                    )
            skeleton = record.get("skeleton", {})
            uv21 = skeleton.get("uv21")
            if uv21 is not None:
                uv = np.asarray(uv21, dtype=np.float32).reshape(21, 2) + correction
                valid = np.asarray(skeleton.get("valid21", [False] * 21), dtype=bool).reshape(21)
                valid &= np.isfinite(uv).all(axis=1)
                valid &= manual_visibility
                image_rgb = skeleton_module.draw_wrist_skeleton_rgb(
                    image_rgb,
                    uv,
                    valid,
                    line_width=int(args.skeleton_line_width),
                    point_radius=int(args.skeleton_point_radius),
                )
            canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            draw_pocket_and_objects(canvas, record, object_names)
            if canvas.shape[1] % 2:
                canvas = np.pad(canvas, ((0, 0), (0, 1), (0, 0)), mode="edge")
            if canvas.shape[0] % 2:
                canvas = np.pad(canvas, ((0, 1), (0, 0), (0, 0)), mode="edge")
            output_path = frames_dir / f"frame_{output_index:06d}.jpg"
            if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]):
                raise RuntimeError(f"failed to write {output_path}")
            rendered_count += 1
        fps = float(args.fps) if args.fps and args.fps > 0 else float(episode_records[0].get("fps", 30.0))
        video_path = output_root / "videos" / f"{episode}.mp4"
        encode_video_ffmpeg(frames_dir, video_path, fps)
        correction_norm = np.linalg.norm(correction_track, axis=2)
        summary = {
            "episode": episode,
            "frames": rendered_count,
            "fps": fps,
            "objects": object_names,
            "manual_keyframes": len(keyframes),
            "hand_mask_root": str(hand_mask_root) if hand_mask_root is not None else None,
            "appearance_variant": int(args.appearance_variant),
            "missing_hand_masks": missing_hand_masks,
            "center_median_window": int(args.center_median_window),
            "center_smooth_sigma": float(args.center_smooth_sigma),
            "mean_joint_correction_px": float(correction_norm.mean()),
            "p90_joint_correction_px": float(np.percentile(correction_norm, 90)),
            "video": str(video_path),
        }
        atomic_write_json(output_root / "videos" / f"{episode}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))
    print(f"rendered frames: {output_root / 'rendered_frames'}")
    print(f"videos: {output_root / 'videos'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Export clean images and prepare sparse MANO-21 annotations.")
    export.add_argument("--data-root", type=Path, required=True, help="Processed episode root, preferably base_enriched_v1 with clean RGB links.")
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--episode", action="append", default=[], help="Episode name. Repeat to select several episodes.")
    export.add_argument("--max-episodes", type=int, default=None)
    export.add_argument("--limit-frames-per-episode", type=int, default=None, help="Smoke-test limit; omit for the complete episode.")
    export.add_argument("--image-field", default="rgbImage")
    export.add_argument("--object", action="append", default=[], help="Object name without task_object_ prefix. Repeat; default discovers all objects.")
    export.add_argument("--annotation-stride", type=int, default=18, help="Prepare one correction keyframe every N exported frames.")
    export.add_argument("--annotation-max-frames-per-episode", type=int, default=None)
    export.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    export.add_argument("--skip-missing", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(func=command_export)

    annotate = subparsers.add_parser("annotate", help="Launch the existing browser annotator with predicted 21 points pre-filled.")
    annotate.add_argument("--output-root", type=Path, required=True)
    annotate.add_argument("--host", default="127.0.0.1")
    annotate.add_argument("--port", type=int, default=8899)
    annotate.add_argument("--open-browser", action="store_true")
    annotate.set_defaults(func=command_annotate)

    render = subparsers.add_parser("render", help="Interpolate manual corrections and render polished QC frames/videos.")
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--episode", action="append", default=[])
    render.add_argument("--smooth-sigma", type=float, default=2.0, help="Gaussian temporal smoothing in frames after correction interpolation.")
    render.add_argument(
        "--alignment-mode",
        choices=("sparse_affine", "per_joint"),
        default="sparse_affine",
        help="sparse_affine lets a few edited joints align the whole hand; per_joint only moves explicitly edited joints.",
    )
    render.add_argument(
        "--sparse-edit-threshold-px",
        type=float,
        default=0.75,
        help="A pre-filled point moved by more than this is considered manually edited.",
    )
    render.add_argument("--max-correction-px", type=float, default=100.0, help="Safety clamp for each joint correction; <=0 disables.")
    render.add_argument(
        "--hand-mask-root",
        type=Path,
        default=None,
        help="SAM3 hand masks directory. Default auto-detects ../sam3_hand_masks_v1/masks from dataset_info.json.",
    )
    render.add_argument(
        "--apply-hand-appearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the original stage-07 extreme/mixed/normal mask-guided appearance augmentation.",
    )
    render.add_argument("--appearance-seed", type=int, default=20260808)
    render.add_argument("--appearance-variant", type=int, default=0, choices=(0, 1, 2))
    render.add_argument("--feather-radius", type=float, default=4.0)
    render.add_argument(
        "--center-median-window",
        type=int,
        default=5,
        help="Odd temporal window used to suppress pocket/object center spikes; 1 disables median filtering.",
    )
    render.add_argument(
        "--center-smooth-sigma",
        type=float,
        default=3.0,
        help="Offline zero-phase Gaussian smoothing sigma in frames for pocket/object centers; 0 disables it.",
    )
    render.add_argument("--skeleton-line-width", type=int, default=2)
    render.add_argument("--skeleton-point-radius", type=int, default=4)
    render.add_argument("--jpeg-quality", type=int, default=95)
    render.add_argument("--fps", type=float, default=0.0, help="Override video FPS; 0 reads collectionHz from each episode.")
    render.add_argument("--overwrite", action="store_true")
    render.set_defaults(func=command_render)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
