#!/usr/bin/env python3
"""Human-side calibrated grasp-pocket views for Human2Dex training.

This is the training-side counterpart of
``Dex_Data-Scaling-Laws-Infer/scripts_real/fisheye_canonical_views.py``.
It deliberately uses raw PICO wrist poses, rather than ``pts21_mano``:
the latter is re-oriented on every frame and therefore has no fixed camera
extrinsic.  The pipeline is:

    raw26x7 PICO world points + raw wrist quaternion
      -> raw wrist-local points -> fitted camera<-raw_wrist
      -> human grasp pocket -> calibrated fisheye wide/local views.

Subcommands:
  fit-camera  fit camera<-raw_PICO_wrist from clicked visible hand points.
  preview     render raw overlay + wide G + local L without touching data.
  materialize copy a PKL dataset and add G/L image fields to the copy.

No robot, camera or policy is opened by this program.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from fisheye_canonical_views import (  # noqa: E402
    FisheyeCanonicalViewRenderer,
    fisheye_unit_rays_to_pixels,
    normalize,
)


# MediaPipe ordering used everywhere else in this repository.
PICO2MEDIAPIPE = np.asarray(
    [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25],
    dtype=np.int64,
)
MANO21_LABELS = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
LABEL_TO_MP_INDEX = {name: idx for idx, name in enumerate(MANO21_LABELS)}
FINGERTIP_LABELS = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
FINGER_COLORS_BGR = (
    (245, 245, 245),  # wrist
    (80, 180, 255),   # thumb
    (80, 255, 120),   # index
    (255, 210, 80),   # middle
    (255, 120, 180),  # ring
    (180, 120, 255),  # pinky
)


def _natural_key(path: Path) -> list[Any]:
    return [int(v) if v.isdigit() else v for v in re.split(r"(\d+)", path.as_posix())]


def _read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"Not a DexUMI PKL with messages: {path}")
    return value


def _atomic_pickle(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_structured(path: Path) -> dict[str, Any]:
    text = path.expanduser().resolve().read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("Reading YAML requires PyYAML in the selected environment") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _parse_size(text: str) -> tuple[int, int]:
    parts = str(text).lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT")
    width, height = (int(value) for value in parts)
    if width < 2 or height < 2:
        raise argparse.ArgumentTypeError("size values must be at least 2")
    return width, height


def _parse_pair(text: str) -> tuple[float, float]:
    values = [float(value.strip()) for value in str(text).split(",")]
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected two finite comma-separated values")
    return values[0], values[1]


def _quat_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid raw PICO wrist quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def raw_to_wrist_points(raw26x7: Any) -> np.ndarray:
    """Return 21 PICO points in raw wrist coordinates, in metres.

    ``raw26x7[1]`` provides the wrist position/orientation in PICO world.  This
    raw local coordinate frame is physically rigid to the wrist camera.  It is
    intentionally different from ``pts21_mano`` which is re-estimated from the
    current hand pose and thus is not rigid to the camera mount.
    """
    raw = np.asarray(raw26x7, dtype=np.float64)
    if raw.shape != (26, 7) or not np.all(np.isfinite(raw)):
        raise ValueError(f"raw26x7 must be finite (26,7), got {raw.shape}")
    wrist_world = raw[1, :3]
    world_from_wrist = _quat_xyzw_to_rotation(raw[1, 3:7])
    points_world = raw[PICO2MEDIAPIPE, :3]
    return (points_world - wrist_world[None, :]) @ world_from_wrist


def raw_palm_in_wrist(raw26x7: Any) -> np.ndarray:
    raw = np.asarray(raw26x7, dtype=np.float64)
    if raw.shape != (26, 7) or not np.all(np.isfinite(raw)):
        raise ValueError(f"raw26x7 must be finite (26,7), got {raw.shape}")
    wrist_world = raw[1, :3]
    world_from_wrist = _quat_xyzw_to_rotation(raw[1, 3:7])
    return (raw[0, :3] - wrist_world) @ world_from_wrist


def _rvec_tvec_to_matrix(rvec: Iterable[float], tvec: Iterable[float]) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def _project(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    uv, _ = cv2.fisheye.projectPoints(
        values,
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K,
        D,
    )
    return uv.reshape(-1, 2)


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, Any]]:
    data = _load_structured(path)
    K = np.asarray(data.get("camera_matrix", data.get("K")), dtype=np.float64)
    D = np.asarray(data.get("dist_coeffs", data.get("D")), dtype=np.float64).reshape(-1)
    if K.shape != (3, 3) or D.shape != (4,):
        raise ValueError("intrinsics must provide a 3x3 K/camera_matrix and four D/dist_coeffs values")
    width = int(data.get("image_width", data.get("image_size", [0, 0])[0]))
    height = int(data.get("image_height", data.get("image_size", [0, 0])[1]))
    if width < 2 or height < 2:
        raise ValueError("intrinsics must provide image_width/image_height")
    return K, D.reshape(4, 1), (width, height), data


def _load_camera_contract(path: Path) -> dict[str, Any]:
    value = _load_structured(path)
    if value.get("format") != "human_camera_from_pico_wrist_v1":
        raise ValueError(f"Not a human camera contract: {path}")
    intrinsics = value.get("intrinsics", {})
    transform = value.get("camera_from_pico_raw_wrist", {})
    K = np.asarray(intrinsics.get("K"), dtype=np.float64)
    D = np.asarray(intrinsics.get("D"), dtype=np.float64).reshape(-1)
    rvec = np.asarray(transform.get("rvec"), dtype=np.float64).reshape(-1)
    tvec = np.asarray(transform.get("tvec"), dtype=np.float64).reshape(-1)
    size = np.asarray(intrinsics.get("image_size"), dtype=np.int64).reshape(-1)
    if K.shape != (3, 3) or D.shape != (4,) or rvec.shape != (3,) or tvec.shape != (3,) or size.shape != (2,):
        raise ValueError(f"Malformed human camera contract: {path}")
    if not np.all(np.isfinite(np.r_[K.reshape(-1), D, rvec, tvec])):
        raise ValueError(f"Non-finite human camera contract: {path}")
    return {
        "path": str(path.expanduser().resolve()),
        "payload": value,
        "K": K,
        "D": D.reshape(4, 1),
        "image_size": (int(size[0]), int(size[1])),
        "rvec": rvec,
        "tvec": tvec,
        "rotation": _rvec_tvec_to_matrix(rvec, tvec)[:3, :3],
    }


def _pocket_geometry(points_wrist: np.ndarray, a: float, b: float, normal_sign: float) -> dict[str, np.ndarray | float]:
    points = np.asarray(points_wrist, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise ValueError("points_wrist must be finite (21,3)")
    palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
    across = points[17] - points[5]  # index MCP -> pinky MCP
    palm_width = float(np.linalg.norm(across))
    if palm_width < 1e-5:
        raise ValueError("human palm width is zero")
    finger_bases = np.mean(points[[5, 9, 13, 17]], axis=0)
    finger_tips = np.mean(points[[8, 12, 16, 20]], axis=0)
    finger_forward = normalize(finger_tips - finger_bases)
    normal = normalize(np.cross(finger_forward, across))
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isfinite(float(normal_sign)):
        raise ValueError("grasp-pocket parameters must be finite")
    pocket = (
        palm_center
        + float(a) * palm_width * float(normal_sign) * normal
        + float(b) * palm_width * finger_forward
    )
    return {
        "palm_center": palm_center,
        "palm_across": across,
        "palm_width": palm_width,
        "finger_forward": finger_forward,
        "palm_normal": normal,
        "pocket": pocket,
    }


def _frame_geometry(message: dict[str, Any], contract: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    points_wrist = raw_to_wrist_points(message.get("raw26x7"))
    pocket = _pocket_geometry(points_wrist, args.a, args.b, args.normal_sign)
    rotation = contract["rotation"]
    tvec = contract["tvec"]
    points_camera = points_wrist @ rotation.T + tvec[None, :]
    pocket_camera = np.asarray(pocket["pocket"], dtype=np.float64) @ rotation.T + tvec
    if not np.all(np.isfinite(pocket_camera)) or pocket_camera[2] <= 1e-5:
        raise ValueError("human grasp pocket is not in front of camera")
    uv = _project(points_camera, contract["rvec"], contract["tvec"], contract["K"], contract["D"])
    pocket_uv = _project(pocket["pocket"][None, :], contract["rvec"], contract["tvec"], contract["K"], contract["D"])[0]
    across_camera = normalize(rotation @ np.asarray(pocket["palm_across"], dtype=np.float64))
    return {
        **pocket,
        "points_wrist": points_wrist,
        "points_camera": points_camera,
        "pocket_camera": pocket_camera,
        "points_uv": uv,
        "pocket_uv": pocket_uv,
        "palm_across_camera": across_camera,
    }


def _uv_inside(uv: np.ndarray, width: int, height: int) -> bool:
    return bool(np.all(np.isfinite(uv)) and 0 <= uv[0] < width and 0 <= uv[1] < height)


def _draw_overlay(image: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    result = np.asarray(image).copy()
    uv = np.asarray(geometry["points_uv"], dtype=np.float64)
    height, width = result.shape[:2]
    for start, end in HAND_EDGES:
        if _uv_inside(uv[start], width, height) and _uv_inside(uv[end], width, height):
            finger = 1 if end <= 4 else 2 if end <= 8 else 3 if end <= 12 else 4 if end <= 16 else 5
            cv2.line(result, tuple(np.rint(uv[start]).astype(int)), tuple(np.rint(uv[end]).astype(int)), FINGER_COLORS_BGR[finger], 1, cv2.LINE_AA)
    for index, point in enumerate(uv):
        if not _uv_inside(point, width, height):
            continue
        finger = 0 if index == 0 else 1 if index <= 4 else 2 if index <= 8 else 3 if index <= 12 else 4 if index <= 16 else 5
        cv2.circle(result, tuple(np.rint(point).astype(int)), 3, FINGER_COLORS_BGR[finger], -1, cv2.LINE_AA)
    pocket_uv = np.asarray(geometry["pocket_uv"], dtype=np.float64)
    if _uv_inside(pocket_uv, width, height):
        cv2.circle(result, tuple(np.rint(pocket_uv).astype(int)), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result, tuple(np.rint(pocket_uv).astype(int)), 7, (32, 32, 32), 1, cv2.LINE_AA)
    return result


def _build_renderers(contract: dict[str, Any], args: argparse.Namespace) -> tuple[FisheyeCanonicalViewRenderer, FisheyeCanonicalViewRenderer]:
    return (
        FisheyeCanonicalViewRenderer(
            contract["K"], contract["D"], output_size=args.global_size,
            anchor_uv_ratio=args.anchor_uv_ratio, palm_across_sign=args.palm_across_sign, border=args.border_mode,
        ),
        FisheyeCanonicalViewRenderer(
            contract["K"], contract["D"], output_size=args.local_size, border=args.border_mode,
        ),
    )


def _render_views(image: np.ndarray, geometry: dict[str, Any], contract: dict[str, Any], args: argparse.Namespace, smoothing: dict[str, np.ndarray | float] | None, renderers: tuple[FisheyeCanonicalViewRenderer, FisheyeCanonicalViewRenderer]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    if image.shape[:2] != (contract["image_size"][1], contract["image_size"][0]):
        raise ValueError(f"image shape {image.shape[:2]} differs from contract image_size {contract['image_size']}")
    if smoothing is None or args.ema_alpha >= 1.0:
        pocket_camera = np.asarray(geometry["pocket_camera"], dtype=np.float64)
        palm_width = float(geometry["palm_width"])
        palm_across = np.asarray(geometry["palm_across_camera"], dtype=np.float64)
    else:
        alpha = float(args.ema_alpha)
        pocket_camera = (1.0 - alpha) * np.asarray(smoothing["pocket"], dtype=np.float64) + alpha * np.asarray(geometry["pocket_camera"], dtype=np.float64)
        palm_width = (1.0 - alpha) * float(smoothing["width"]) + alpha * float(geometry["palm_width"])
        palm_across = normalize((1.0 - alpha) * np.asarray(smoothing["across"], dtype=np.float64) + alpha * np.asarray(geometry["palm_across_camera"], dtype=np.float64))
    global_renderer, local_renderer = renderers
    global_view, _ = global_renderer.render_wide(image, pocket_camera, palm_across)
    local_view = local_renderer.render_local(
        image, pocket_camera, palm_width,
        width_in_palm=args.local_width_in_palm,
        height_in_palm=args.local_height_in_palm,
        rotation_deg=args.local_rotation_deg,
    )
    return global_view, local_view, {"pocket": pocket_camera, "width": palm_width, "across": palm_across}


def _find_pkls(root: Path) -> list[Path]:
    found = sorted(root.expanduser().resolve().rglob("*.pkl"), key=_natural_key)
    if not found:
        raise FileNotFoundError(f"No PKL files below {root}")
    return found


def _annotation_observations(annotation_path: Path, pkl_root: Path) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    annotation = _load_structured(annotation_path)
    frames = annotation.get("frames")
    if not isinstance(frames, list):
        raise ValueError("annotation must contain frames")
    requested_names = {Path(str(row.get("image", ""))).name for row in frames if row.get("image")}
    candidates: dict[str, list[tuple[Path, int, dict[str, Any]]]] = defaultdict(list)
    candidates_by_absolute_path: dict[str, list[tuple[Path, int, dict[str, Any]]]] = defaultdict(list)
    for pkl_path in _find_pkls(pkl_root):
        data = _read_pickle(pkl_path)
        for frame_index, message in enumerate(data["messages"]):
            if not isinstance(message, dict) or not message.get("rgbImage"):
                continue
            name = Path(str(message["rgbImage"])).name
            if name in requested_names:
                record = (pkl_path, frame_index, message)
                candidates[name].append(record)
                candidates_by_absolute_path[str((pkl_path.parent / str(message["rgbImage"])).resolve())].append(record)
    objects: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    frame_names: list[str] = []
    missing = 0
    ambiguous = 0
    for row in frames:
        image_name = Path(str(row.get("image", ""))).name
        absolute_image = str(Path(str(row.get("image", ""))).expanduser().resolve())
        options = candidates_by_absolute_path.get(absolute_image, []) or candidates.get(image_name, [])
        if not options:
            missing += 1
            continue
        if len(options) != 1:
            ambiguous += 1
            continue
        _, _, message = options[0]
        try:
            points = raw_to_wrist_points(message.get("raw26x7"))
            palm = raw_palm_in_wrist(message.get("raw26x7"))
        except ValueError:
            continue
        labels = row.get("points", {})
        if not isinstance(labels, dict):
            continue
        used = 0
        for label, point in labels.items():
            if not isinstance(point, dict) or not bool(point.get("visible", False)):
                continue
            if label == "palm_center":
                object_point = palm
            elif label in LABEL_TO_MP_INDEX:
                object_point = points[LABEL_TO_MP_INDEX[label]]
            elif label in FINGERTIP_LABELS:
                object_point = points[FINGERTIP_LABELS[label]]
            else:
                continue
            try:
                uv = np.asarray([float(point["x"]), float(point["y"])], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if np.all(np.isfinite(uv)):
                objects.append(object_point)
                pixels.append(uv)
                frame_names.append(image_name)
                used += 1
        if used == 0:
            missing += 1
    if ambiguous:
        raise ValueError(f"{ambiguous} annotated images have non-unique basenames under {pkl_root}; create a list from one dataset root or rename copies")
    if len(objects) < 12 or len(set(frame_names)) < 3:
        raise ValueError(f"Need >=12 clicked correspondences from >=3 frames; got points={len(objects)}, frames={len(set(frame_names))}, missing={missing}")
    return np.stack(objects), np.stack(pixels), frame_names, missing


def _camera_annotation_records(annotation_path: Path, pkl_root: Path) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    """Resolve annotated source images to their exact PKL messages for QA."""
    annotation = _load_structured(annotation_path)
    frames = annotation.get("frames")
    if not isinstance(frames, list):
        raise ValueError("annotation must contain frames")
    requested_names = {Path(str(row.get("image", ""))).name for row in frames if row.get("image")}
    by_path: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    by_name: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for pkl_path in _find_pkls(pkl_root):
        data = _read_pickle(pkl_path)
        for message in data["messages"]:
            if not isinstance(message, dict) or not message.get("rgbImage"):
                continue
            image_path = (pkl_path.parent / str(message["rgbImage"])).resolve()
            if image_path.name not in requested_names:
                continue
            record = (message, image_path)
            by_path[str(image_path)].append(record)
            by_name[image_path.name].append(record)
    records: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for row in frames:
        annotation_path_value = Path(str(row.get("image", ""))).expanduser().resolve()
        options = by_path.get(str(annotation_path_value), []) or by_name.get(annotation_path_value.name, [])
        if len(options) != 1:
            raise ValueError(f"Cannot uniquely resolve annotated image {annotation_path_value}; matches={len(options)}")
        message, image_path = options[0]
        records.append((row, message, image_path))
    return records


def _camera_label_point(label: str, message: dict[str, Any]) -> np.ndarray | None:
    points = raw_to_wrist_points(message.get("raw26x7"))
    if label == "palm_center":
        return raw_palm_in_wrist(message.get("raw26x7"))
    if label in LABEL_TO_MP_INDEX:
        return points[LABEL_TO_MP_INDEX[label]]
    if label in FINGERTIP_LABELS:
        return points[FINGERTIP_LABELS[label]]
    return None


def _pocket_annotation_observations(annotation_path: Path, pkl_root: Path) -> tuple[list[np.ndarray], np.ndarray, list[str], int]:
    """Read functional ``grasp_pocket`` clicks and match them to PKL frames."""
    annotation = _load_structured(annotation_path)
    frames = annotation.get("frames")
    if not isinstance(frames, list):
        raise ValueError("annotation must contain frames")
    requested_names = {Path(str(row.get("image", ""))).name for row in frames if row.get("image")}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates_by_absolute_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pkl_path in _find_pkls(pkl_root):
        data = _read_pickle(pkl_path)
        for message in data["messages"]:
            if isinstance(message, dict) and message.get("rgbImage"):
                name = Path(str(message["rgbImage"])).name
                if name in requested_names:
                    candidates[name].append(message)
                    candidates_by_absolute_path[str((pkl_path.parent / str(message["rgbImage"])).resolve())].append(message)
    point_sets: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    names: list[str] = []
    missing = 0
    ambiguous = 0
    for row in frames:
        image_name = Path(str(row.get("image", ""))).name
        absolute_image = str(Path(str(row.get("image", ""))).expanduser().resolve())
        choices = candidates_by_absolute_path.get(absolute_image, []) or candidates.get(image_name, [])
        if not choices:
            missing += 1
            continue
        if len(choices) != 1:
            ambiguous += 1
            continue
        point = (row.get("points") or {}).get("grasp_pocket")
        if not isinstance(point, dict) or not bool(point.get("visible", False)):
            continue
        try:
            uv = np.asarray([float(point["x"]), float(point["y"])], dtype=np.float64)
            wrist_points = raw_to_wrist_points(choices[0].get("raw26x7"))
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue
        if np.all(np.isfinite(uv)):
            point_sets.append(wrist_points)
            pixels.append(uv)
            names.append(image_name)
    if ambiguous:
        raise ValueError(f"{ambiguous} grasp-pocket images are ambiguous under {pkl_root}; use image copies from one source dataset")
    if len(point_sets) < 8 or len(set(names)) < 4:
        raise ValueError(f"Need at least 8 grasp-pocket clicks from 4 frames; got {len(point_sets)}")
    return point_sets, np.stack(pixels), names, missing


def _fit_transform(objects: np.ndarray, pixels: np.ndarray, K: np.ndarray, D: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:
        raise RuntimeError("fit-camera requires scipy.optimize.least_squares") from exc

    objects = np.asarray(objects, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)

    def residual(params: np.ndarray, indices: np.ndarray) -> np.ndarray:
        rvec, tvec = params[:3], params[3:]
        selected = objects[indices]
        rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        depth = selected @ rotation[2, :].reshape(3, 1) + tvec[2]
        try:
            projected = _project(selected, rvec, tvec, K, D)
            value = np.clip(projected - pixels[indices], -500.0, 500.0).reshape(-1)
        except cv2.error:
            value = np.full(indices.size * 2, 500.0, dtype=np.float64)
        depth_penalty = np.minimum(depth.reshape(-1) - 0.015, 0.0) * 2000.0
        return np.concatenate([value, depth_penalty])

    all_indices = np.arange(len(objects), dtype=np.int64)
    rng = np.random.default_rng(seed)
    starts = [
        np.asarray([0, 0, 0, 0, 0, 0.10], dtype=np.float64),
        np.asarray([math.pi, 0, 0, 0, 0, 0.10], dtype=np.float64),
        np.asarray([0, math.pi, 0, 0, 0, 0.10], dtype=np.float64),
        np.asarray([0, 0, math.pi, 0, 0, 0.10], dtype=np.float64),
        np.asarray([math.pi / 2, 0, 0, 0, 0, 0.10], dtype=np.float64),
        np.asarray([-math.pi / 2, 0, 0, 0, 0, 0.10], dtype=np.float64),
    ]
    # A pinhole PnP estimate ignores distortion but is usually in the right
    # basin; fisheye least-squares below then uses the correct camera model.
    try:
        ok, pnp_rvec, pnp_tvec = cv2.solvePnP(
            objects.reshape(-1, 1, 3), pixels.reshape(-1, 1, 2), K,
            np.zeros((4, 1), dtype=np.float64), flags=cv2.SOLVEPNP_EPNP,
        )
        if ok and float(pnp_tvec.reshape(-1)[2]) > 0.01:
            starts.append(np.r_[pnp_rvec.reshape(3), pnp_tvec.reshape(3)])
    except cv2.error:
        pass
    starts.extend(
        np.r_[rng.normal(0, 2.2, 3), [rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), rng.uniform(0.04, 0.25)]]
        for _ in range(14)
    )
    best = None
    for start in starts:
        result = least_squares(residual, start, args=(all_indices,), loss="soft_l1", f_scale=4.0, max_nfev=1800)
        score = float(np.median(np.linalg.norm(_project(objects, result.x[:3], result.x[3:], K, D) - pixels, axis=1)))
        if best is None or score < best[0]:
            best = (score, result.x)
    assert best is not None
    params = best[1]
    errors = np.linalg.norm(_project(objects, params[:3], params[3:], K, D) - pixels, axis=1)
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    threshold = max(5.0, median + 3.0 * max(1.4826 * mad, 1.0))
    inliers = errors <= threshold
    if int(inliers.sum()) >= 12:
        refined = least_squares(residual, params, args=(all_indices[inliers],), loss="soft_l1", f_scale=3.0, max_nfev=2400)
        params = refined.x
        errors = np.linalg.norm(_project(objects, params[:3], params[3:], K, D) - pixels, axis=1)
        inliers = errors <= threshold
    return params[:3], params[3:], inliers


def _metrics(errors: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(errors, dtype=np.float64)
    if not values.size:
        return {"n": 0, "median_px": None, "p90_px": None, "mean_px": None, "max_px": None}
    return {"n": int(values.size), "median_px": float(np.median(values)), "p90_px": float(np.percentile(values, 90)), "mean_px": float(np.mean(values)), "max_px": float(np.max(values))}


def run_fit_camera(args: argparse.Namespace) -> int:
    K, D, image_size, intrinsic_payload = _load_intrinsics(args.intrinsics)
    objects, pixels, frame_names, skipped = _annotation_observations(args.annotations, args.pkl_root)
    unique_frames = sorted(set(frame_names))
    rng = random.Random(args.seed)
    rng.shuffle(unique_frames)
    holdout_count = int(round(len(unique_frames) * args.holdout_ratio)) if len(unique_frames) >= 6 else 0
    holdout_names = set(unique_frames[:holdout_count])
    train_indices = np.asarray([i for i, name in enumerate(frame_names) if name not in holdout_names], dtype=np.int64)
    holdout_indices = np.asarray([i for i, name in enumerate(frame_names) if name in holdout_names], dtype=np.int64)
    if len(train_indices) < 12:
        train_indices = np.arange(len(objects), dtype=np.int64)
        holdout_indices = np.asarray([], dtype=np.int64)
    rvec, tvec, inlier_mask = _fit_transform(objects[train_indices], pixels[train_indices], K, D, args.seed)
    all_pred = _project(objects, rvec, tvec, K, D)
    all_errors = np.linalg.norm(all_pred - pixels, axis=1)
    train_errors = all_errors[train_indices]
    holdout_errors = all_errors[holdout_indices]
    contract = {
        "format": "human_camera_from_pico_wrist_v1",
        "status": "provisional_requires_preview_holdout",
        "description": "camera_from_pico_raw_wrist fitted from manually clicked PICO hand points; uses raw26x7[1] orientation, never pts21_mano.",
        "coordinate_contract": {
            "object": "PICO raw wrist-local frame: row=(world-wrist_world) @ R_world_from_raw_wrist",
            "transform": "x_camera = R_camera_from_raw_wrist @ x_raw_wrist + t_camera_from_raw_wrist",
            "image": "pixels must be the exact stored RGB geometry; image is not resized/cropped after fitting",
        },
        "intrinsics": {
            "source_path": str(args.intrinsics.expanduser().resolve()),
            "image_size": [int(image_size[0]), int(image_size[1])],
            "K": K.tolist(), "D": D.reshape(-1).tolist(),
            "source_keys": sorted(intrinsic_payload.keys()),
        },
        "camera_from_pico_raw_wrist": {
            "rvec": rvec.tolist(), "tvec": tvec.tolist(),
            "matrix": _rvec_tvec_to_matrix(rvec, tvec).tolist(),
        },
        "fit": {
            "annotation_path": str(args.annotations.expanduser().resolve()),
            "pkl_root": str(args.pkl_root.expanduser().resolve()),
            "image_match": "unique RGB basename under pkl_root",
            "correspondences": int(len(objects)),
            "annotated_frames": int(len(unique_frames)),
            "skipped_annotation_frames": int(skipped),
            "train_frames": int(len(set(frame_names[i] for i in train_indices))),
            "holdout_frames": int(len(holdout_names)),
            "inliers_on_train_fit": int(inlier_mask.sum()),
            "train": _metrics(train_errors),
            "holdout": _metrics(holdout_errors),
            "all": _metrics(all_errors),
        },
    }
    _atomic_json(args.output, contract)
    print(json.dumps(contract["fit"], ensure_ascii=False, indent=2))
    print(f"wrote={args.output}")
    return 0


def run_diagnose_camera(args: argparse.Namespace) -> int:
    """Render fitted full skeletons and clicked points on the same raw RGB."""
    contract = _load_camera_contract(args.camera_contract)
    records = _camera_annotation_records(args.annotations, args.pkl_root)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_label: dict[str, list[float]] = defaultdict(list)
    per_frame: list[dict[str, Any]] = []
    thumbnails: list[np.ndarray] = []
    for frame_index, (row, message, image_path) in enumerate(records):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        points_wrist = raw_to_wrist_points(message.get("raw26x7"))
        points_uv = _project(points_wrist, contract["rvec"], contract["tvec"], contract["K"], contract["D"])
        overlay = _draw_overlay(image, {"points_uv": points_uv, "pocket_uv": np.asarray([np.nan, np.nan])})
        label_details: dict[str, Any] = {}
        points = row.get("points", {}) or {}
        for label, click in points.items():
            if not isinstance(click, dict) or not bool(click.get("visible", False)):
                continue
            object_point = _camera_label_point(label, message)
            if object_point is None:
                continue
            clicked_uv = np.asarray([float(click["x"]), float(click["y"])], dtype=np.float64)
            predicted_uv = _project(object_point[None, :], contract["rvec"], contract["tvec"], contract["K"], contract["D"])[0]
            delta = predicted_uv - clicked_uv
            error = float(np.linalg.norm(delta))
            per_label[label].append(error)
            label_details[label] = {"clickedUv": clicked_uv.tolist(), "predictedUv": predicted_uv.tolist(), "deltaUv": delta.tolist(), "errorPx": error}
            cv2.drawMarker(overlay, tuple(np.rint(clicked_uv).astype(int)), (20, 20, 255), cv2.MARKER_TILTED_CROSS, 11, 1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(np.rint(predicted_uv).astype(int)), 4, (0, 255, 255), -1, cv2.LINE_AA)
        values = [detail["errorPx"] for detail in label_details.values()]
        if values:
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 27), (24, 24, 24), -1)
            cv2.putText(overlay, f"frame {frame_index:02d}: click=red x, prediction=yellow, median={np.median(values):.1f}px", (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
        target = output / f"{frame_index:03d}_{image_path.stem}.jpg"
        if not cv2.imwrite(str(target), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 96]):
            raise OSError(f"failed to write {target}")
        thumbnails.append(cv2.resize(overlay, (240, 240), interpolation=cv2.INTER_AREA))
        per_frame.append({"annotationImage": str(row.get("image")), "sourceImage": str(image_path), "overlay": str(target), "rgbPicoDeltaMs": None if message.get("rgbToPicoReceiveDeltaNs") is None else float(message["rgbToPicoReceiveDeltaNs"]) / 1e6, "absRgbPicoDeltaMs": None if message.get("absRgbToPicoReceiveDeltaNs") is None else float(message["absRgbToPicoReceiveDeltaNs"]) / 1e6, "rgbAlignResidualMs": None if message.get("rgbAlignResidualNs") is None else float(message["rgbAlignResidualNs"]) / 1e6, "errors": label_details})
    if thumbnails:
        columns = 6
        rows = (len(thumbnails) + columns - 1) // columns
        sheet = np.zeros((rows * 240, columns * 240, 3), dtype=np.uint8)
        for index, tile in enumerate(thumbnails):
            y, x = divmod(index, columns)
            sheet[y * 240:(y + 1) * 240, x * 240:(x + 1) * 240] = tile
        cv2.imwrite(str(output / "contact_sheet.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    payload = {
        "format": "human_camera_fit_diagnosis_v1",
        "cameraContract": contract["path"],
        "annotations": str(args.annotations.expanduser().resolve()),
        "pklRoot": str(args.pkl_root.expanduser().resolve()),
        "legend": {"red_x": "manual click", "yellow_dot": "fitted projection"},
        "contactSheet": str(output / "contact_sheet.jpg"),
        "perLabel": {label: _metrics(np.asarray(values, dtype=np.float64)) for label, values in sorted(per_label.items())},
        "frames": per_frame,
    }
    _atomic_json(output / "diagnosis.json", payload)
    print(json.dumps(payload["perLabel"], ensure_ascii=False, indent=2))
    print(f"output={output}")
    return 0


def run_fit_pocket(args: argparse.Namespace) -> int:
    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:
        raise RuntimeError("fit-pocket requires scipy.optimize.least_squares") from exc
    contract = _load_camera_contract(args.camera_contract)
    point_sets, clicked_uv, frame_names, skipped = _pocket_annotation_observations(args.annotations, args.pkl_root)
    point_sets_array = np.stack(point_sets)
    unique_frames = sorted(set(frame_names))
    rng = random.Random(args.seed)
    rng.shuffle(unique_frames)
    holdout_count = int(round(len(unique_frames) * args.holdout_ratio)) if len(unique_frames) >= 6 else 0
    holdout_names = set(unique_frames[:holdout_count])
    train_indices = np.asarray([i for i, name in enumerate(frame_names) if name not in holdout_names], dtype=np.int64)
    holdout_indices = np.asarray([i for i, name in enumerate(frame_names) if name in holdout_names], dtype=np.int64)
    if len(train_indices) < 4:
        train_indices = np.arange(len(point_sets), dtype=np.int64)
        holdout_indices = np.asarray([], dtype=np.int64)

    def predicted_uv(params: np.ndarray, indices: np.ndarray) -> np.ndarray:
        pockets = np.stack([
            _pocket_geometry(point_sets_array[index], params[0], params[1], args.normal_sign)["pocket"]
            for index in indices
        ])
        return _project(pockets, contract["rvec"], contract["tvec"], contract["K"], contract["D"])

    def residual(params: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return (predicted_uv(params, indices) - clicked_uv[indices]).reshape(-1)

    result = least_squares(
        residual,
        x0=np.asarray([args.initial_a, args.initial_b], dtype=np.float64),
        args=(train_indices,),
        bounds=(np.asarray([-2.0, -2.0]), np.asarray([2.0, 2.0])),
        loss="soft_l1", f_scale=3.0, max_nfev=2000,
    )
    a, b = (float(result.x[0]), float(result.x[1]))
    all_errors = np.linalg.norm(predicted_uv(result.x, np.arange(len(point_sets))) - clicked_uv, axis=1)
    output = {
        "format": "human_grasp_pocket_v1",
        "status": "provisional_requires_preview_holdout",
        "description": "Human functional grasp-pocket anchor fitted from manually clicked desired pocket pixels.",
        "camera_contract": contract["path"],
        "formula": "palm_center + a*palm_width*normal_sign*palm_normal + b*palm_width*finger_forward",
        "definitions": {
            "palm_center": "mean(wrist,index_mcp,middle_mcp,ring_mcp,pinky_mcp)",
            "palm_width": "norm(pinky_mcp-index_mcp)",
            "finger_forward": "mean(finger tips)-mean(finger MCPs)",
            "palm_normal": "normalize(cross(finger_forward, index_mcp_to_pinky_mcp))",
        },
        "a": a, "b": b, "normal_sign": float(args.normal_sign),
        "fit": {
            "annotations": str(args.annotations.expanduser().resolve()),
            "pkl_root": str(args.pkl_root.expanduser().resolve()),
            "clicks": int(len(point_sets)), "annotated_frames": int(len(unique_frames)),
            "skipped_annotation_frames": int(skipped),
            "train": _metrics(all_errors[train_indices]),
            "holdout": _metrics(all_errors[holdout_indices]),
            "all": _metrics(all_errors),
        },
    }
    _atomic_json(args.output, output)
    print(json.dumps(output["fit"], ensure_ascii=False, indent=2))
    print(f"wrote={args.output}")
    return 0


def run_fit_effective_pocket(args: argparse.Namespace) -> int:
    """Fit an *effective* PICO-wrist-to-camera map from direct pocket clicks.

    This is intentionally not advertised as a physical hand-eye calibration:
    PICO hand geometry remains a noisy latent parameterization.  Its purpose is
    to quantify the geometric prior against the one semantic pixel that G/L
    actually needs, while the RGB pocket head remains the final visual anchor.
    """
    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:
        raise RuntimeError("fit-effective-pocket requires scipy.optimize.least_squares") from exc
    K, D, image_size, intrinsic_payload = _load_intrinsics(args.intrinsics)
    point_sets, clicked_uv, frame_names, skipped = _pocket_annotation_observations(args.annotations, args.pkl_root)
    points = np.stack(point_sets)
    unique_frames = sorted(set(frame_names))
    rng = random.Random(args.seed)
    rng.shuffle(unique_frames)
    holdout_count = int(round(len(unique_frames) * args.holdout_ratio)) if len(unique_frames) >= 6 else 0
    holdout_names = set(unique_frames[:holdout_count])
    train_indices = np.asarray([index for index, name in enumerate(frame_names) if name not in holdout_names], dtype=np.int64)
    holdout_indices = np.asarray([index for index, name in enumerate(frame_names) if name in holdout_names], dtype=np.int64)
    if len(train_indices) < 6:
        train_indices = np.arange(len(points), dtype=np.int64)
        holdout_indices = np.asarray([], dtype=np.int64)

    def project(params: np.ndarray, indices: np.ndarray) -> np.ndarray:
        rvec, tvec, a, b = params[:3], params[3:6], float(params[6]), float(params[7])
        pockets = np.stack([
            _pocket_geometry(points[index], a, b, args.normal_sign)["pocket"]
            for index in indices
        ])
        return _project(pockets, rvec, tvec, K, D)

    def residual(params: np.ndarray, indices: np.ndarray) -> np.ndarray:
        try:
            predicted = project(params, indices)
            value = np.clip(predicted - clicked_uv[indices], -500.0, 500.0).reshape(-1)
        except (ValueError, cv2.error):
            value = np.full(int(indices.size) * 2, 500.0, dtype=np.float64)
        return value

    initial = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.12, args.initial_a, args.initial_b], dtype=np.float64)
    init_description = "default"
    if args.init_camera_contract is not None:
        contract = _load_camera_contract(args.init_camera_contract)
        initial[:3], initial[3:6] = contract["rvec"], contract["tvec"]
        init_description = f"camera contract: {contract['path']}"
    starts = [initial]
    if args.init_camera_contract is None:
        starts.extend([
            initial + np.asarray([math.pi, 0, 0, 0, 0, 0, 0, 0]),
            initial + np.asarray([0, math.pi, 0, 0, 0, 0, 0, 0]),
        ])
    lower = np.asarray([-2 * math.pi, -2 * math.pi, -2 * math.pi, -0.5, -0.5, 0.015, -2.0, -2.0])
    upper = np.asarray([2 * math.pi, 2 * math.pi, 2 * math.pi, 0.5, 0.5, 1.5, 2.0, 2.0])
    best: tuple[float, np.ndarray] | None = None
    for start in starts:
        result = least_squares(residual, start, args=(train_indices,), bounds=(lower, upper), loss="soft_l1", f_scale=3.0, max_nfev=4000)
        error = np.linalg.norm(project(result.x, train_indices) - clicked_uv[train_indices], axis=1)
        score = float(np.median(error))
        if best is None or score < best[0]:
            best = score, result.x
    assert best is not None
    params = best[1]
    all_indices = np.arange(len(points), dtype=np.int64)
    all_errors = np.linalg.norm(project(params, all_indices) - clicked_uv, axis=1)
    output = {
        "format": "human_effective_camera_from_pico_pocket_v1",
        "status": "effective_visual_alignment_only_not_physical_extrinsic",
        "description": "Jointly fitted from direct functional grasp-pocket pixels and PICO hand geometry. It is an RGB G/L geometric prior, not a physical camera-to-hand extrinsic claim.",
        "intrinsics": {"source_path": str(args.intrinsics.expanduser().resolve()), "image_size": list(image_size), "K": K.tolist(), "D": D.reshape(-1).tolist(), "source_keys": sorted(intrinsic_payload)},
        "camera_from_pico_raw_wrist": {"rvec": params[:3].tolist(), "tvec": params[3:6].tolist(), "matrix": _rvec_tvec_to_matrix(params[:3], params[3:6]).tolist()},
        "grasp_pocket": {"a": float(params[6]), "b": float(params[7]), "normal_sign": float(args.normal_sign), "formula": "PICO virtual pocket; direct click is the only optimization target"},
        "fit": {"annotations": str(args.annotations.expanduser().resolve()), "pkl_root": str(args.pkl_root.expanduser().resolve()), "initialization": init_description, "clicks": int(len(points)), "annotated_frames": int(len(unique_frames)), "skipped_annotation_frames": int(skipped), "train": _metrics(all_errors[train_indices]), "holdout": _metrics(all_errors[holdout_indices]), "all": _metrics(all_errors)},
    }
    _atomic_json(args.output, output)
    print(json.dumps(output["fit"], ensure_ascii=False, indent=2))
    print(f"wrote={args.output}")
    return 0


def _select_pkls(root: Path, episode_spec: str | None, limit: int | None, episode_name_regex: str | None = None) -> list[Path]:
    paths = _find_pkls(root)
    if episode_name_regex:
        expression = re.compile(episode_name_regex)
        paths = [path for path in paths if expression.fullmatch(path.parent.name)]
    wanted = set()
    if episode_spec:
        for token in episode_spec.split(","):
            token = token.strip()
            if token:
                wanted.add(token if token.startswith("episode_") else f"episode_{int(token):04d}")
    if wanted:
        paths = [path for path in paths if path.parent.name in wanted]
    if limit is not None:
        paths = paths[:int(limit)]
    if not paths:
        raise FileNotFoundError("No PKLs selected")
    return paths


def _canonical_metadata(contract: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": "human_canonical_views_v1",
        "renderer_sha256_must_match_deployment": "fisheye_canonical_views.py",
        "cameraContract": contract["path"],
        "sourceRgbField": args.source_image_field,
        "globalImageField": args.global_image_field,
        "localImageField": args.local_image_field,
        "graspPocket": {"a": float(args.a), "b": float(args.b), "normalSign": float(args.normal_sign)},
        "global": {"outputSize": list(args.global_size), "anchorUvRatio": list(args.anchor_uv_ratio), "palmAcrossSign": float(args.palm_across_sign), "borderMode": args.border_mode},
        "local": {"outputSize": list(args.local_size), "widthInPalm": float(args.local_width_in_palm), "heightInPalm": float(args.local_height_in_palm), "rotationDeg": float(args.local_rotation_deg), "borderMode": args.border_mode},
        "emaAlpha": float(args.ema_alpha),
    }


def _apply_pocket_config(args: argparse.Namespace) -> None:
    path = getattr(args, "pocket_config", None)
    if path is not None:
        value = _load_structured(path)
        if value.get("format") != "human_grasp_pocket_v1":
            raise ValueError(f"Not a human grasp-pocket config: {path}")
        for field in ("a", "b", "normal_sign"):
            given = getattr(args, field)
            expected = float(value[field])
            if given is not None and not math.isclose(float(given), expected, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(f"--{field.replace('_', '-')} conflicts with --pocket-config")
            setattr(args, field, expected)
    if args.a is None or args.b is None or args.normal_sign is None:
        raise ValueError("Specify --pocket-config, or all of --a --b --normal-sign")


def run_preview(args: argparse.Namespace) -> int:
    contract = _load_camera_contract(args.camera_contract)
    pkl_paths = _select_pkls(args.input, args.episodes, args.limit_episodes, args.episode_name_regex)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"format": "human_canonical_preview_v1", "contract": contract["path"], "settings": _canonical_metadata(contract, args), "frames": [], "counts": Counter()}
    renderers = _build_renderers(contract, args)
    for pkl_path in pkl_paths:
        data = _read_pickle(pkl_path)
        smoothed: dict[str, Any] | None = None
        saved = 0
        for frame_index, message in enumerate(data["messages"]):
            if saved >= args.max_frames_per_episode or frame_index % args.stride:
                continue
            if not isinstance(message, dict) or not message.get(args.source_image_field):
                continue
            source = pkl_path.parent / str(message[args.source_image_field])
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                report["counts"]["missing_image"] += 1
                continue
            try:
                geometry = _frame_geometry(message, contract, args)
                wide, local, smoothed = _render_views(image, geometry, contract, args, smoothed, renderers)
                raw = _draw_overlay(image, geometry)
                raw = cv2.resize(raw, args.global_size, interpolation=cv2.INTER_AREA)
                visual = np.concatenate([raw, wide, local], axis=1)
                episode_dir = output / pkl_path.parent.name
                episode_dir.mkdir(parents=True, exist_ok=True)
                target = episode_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(target), visual, [int(cv2.IMWRITE_JPEG_QUALITY), 96]):
                    raise OSError(f"could not write {target}")
                report["frames"].append({"pkl": str(pkl_path), "frameIndex": frame_index, "source": str(source), "overlay": str(target), "pocketUv": np.asarray(geometry["pocket_uv"]).tolist(), "pocketDepthM": float(np.asarray(geometry["pocket_camera"])[2]), "palmWidthM": float(geometry["palm_width"])})
                saved += 1
                report["counts"]["valid"] += 1
            except Exception as exc:
                report["counts"]["error"] += 1
                report["frames"].append({"pkl": str(pkl_path), "frameIndex": frame_index, "error": repr(exc)})
    report["counts"] = dict(report["counts"])
    _atomic_json(output / "summary.json", report)
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(f"output={output}")
    return 0


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def run_materialize(args: argparse.Namespace) -> int:
    contract = _load_camera_contract(args.camera_contract)
    source_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output must not exist: {output_root}")
    pkl_paths = _select_pkls(source_root, args.episodes, args.limit_episodes, args.episode_name_regex)
    selected_dirs = sorted({path.parent for path in pkl_paths}, key=_natural_key)
    output_root.mkdir(parents=True)
    report: dict[str, Any] = {"format": "human_canonical_materialize_v1", "sourceRoot": str(source_root), "outputRoot": str(output_root), "settings": _canonical_metadata(contract, args), "episodes": [], "counts": Counter()}
    renderers = _build_renderers(contract, args)
    for source_dir in selected_dirs:
        relative_dir = source_dir.relative_to(source_root)
        target_dir = output_root / relative_dir
        shutil.copytree(source_dir, target_dir, copy_function=_hardlink_or_copy)
        target_pkl = target_dir / next(path.name for path in pkl_paths if path.parent == source_dir)
        data = _read_pickle(target_pkl)
        smoothed: dict[str, Any] | None = None
        episode_counts: Counter[str] = Counter()
        for frame_index, message in enumerate(data["messages"]):
            if not isinstance(message, dict):
                episode_counts["not_dict"] += 1
                continue
            source_rel = message.get(args.source_image_field)
            if not source_rel:
                episode_counts["missing_source_field"] += 1
                continue
            image = cv2.imread(str(target_dir / str(source_rel)), cv2.IMREAD_COLOR)
            if image is None:
                episode_counts["missing_image"] += 1
                continue
            try:
                geometry = _frame_geometry(message, contract, args)
                wide, local, smoothed = _render_views(image, geometry, contract, args, smoothed, renderers)
                global_rel = Path("canonical_views") / "global" / f"frame_{frame_index:06d}.jpg"
                local_rel = Path("canonical_views") / "local" / f"frame_{frame_index:06d}.jpg"
                global_path, local_path = target_dir / global_rel, target_dir / local_rel
                global_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(global_path), wide, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]):
                    raise OSError(f"failed to write {global_path}")
                if not cv2.imwrite(str(local_path), local, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]):
                    raise OSError(f"failed to write {local_path}")
                message[args.global_image_field] = global_rel.as_posix()
                message[args.local_image_field] = local_rel.as_posix()
                message["humanGraspPocketWrist"] = np.asarray(geometry["pocket"], dtype=np.float32)
                message["humanGraspPocketCamera"] = np.asarray(geometry["pocket_camera"], dtype=np.float32)
                message["humanGraspPocketUV"] = np.asarray(geometry["pocket_uv"], dtype=np.float32)
                message["humanGraspPocketValid"] = True
                message["canonicalSourceFrameIndex"] = int(frame_index)
                episode_counts["valid"] += 1
            except Exception as exc:
                message[args.global_image_field] = None
                message[args.local_image_field] = None
                message["humanGraspPocketValid"] = False
                message["humanCanonicalViewError"] = repr(exc)
                episode_counts["error"] += 1
        metadata = data.setdefault("metadata", {})
        metadata["humanCanonicalViews"] = _canonical_metadata(contract, args)
        metadata["humanCanonicalViews"]["sourceDataset"] = str(source_root)
        metadata["humanCanonicalViews"]["episodeCounts"] = dict(episode_counts)
        _atomic_pickle(target_pkl, data)
        report["episodes"].append({"source": str(source_dir), "output": str(target_dir), "counts": dict(episode_counts)})
        report["counts"].update(episode_counts)
        print(f"{source_dir.name}: {dict(episode_counts)}", flush=True)
    report["counts"] = dict(report["counts"])
    _atomic_json(output_root / "human_canonical_views_manifest.json", report)
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(f"output={output_root}")
    return 0


def _add_view_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-contract", type=Path, required=True)
    parser.add_argument("--pocket-config", type=Path, default=None, help="human_grasp_pocket_v1 JSON/YAML; preferred after fitting")
    parser.add_argument("--a", type=float, default=None, help="human pocket normal offset, in palm-width units")
    parser.add_argument("--b", type=float, default=None, help="human pocket finger-forward offset, in palm-width units")
    parser.add_argument("--normal-sign", type=float, choices=(-1.0, 1.0), default=None)
    parser.add_argument("--source-image-field", default="rgbImage")
    parser.add_argument("--global-image-field", default="globalCanonicalImage")
    parser.add_argument("--local-image-field", default="localCanonicalImage")
    parser.add_argument("--global-size", type=_parse_size, default=(224, 224))
    parser.add_argument("--local-size", type=_parse_size, default=(224, 224))
    parser.add_argument("--anchor-uv-ratio", type=_parse_pair, default=(0.5, 0.35))
    parser.add_argument("--palm-across-sign", type=float, default=1.0)
    parser.add_argument("--border-mode", choices=("constant", "edge", "replicate", "reflect", "reflect101"), default="reflect101")
    parser.add_argument("--local-width-in-palm", type=float, default=2.5)
    parser.add_argument("--local-height-in-palm", type=float, default=2.5)
    parser.add_argument("--local-rotation-deg", type=float, default=0.0)
    parser.add_argument("--ema-alpha", type=float, default=0.3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-camera", help="fit camera<-raw PICO wrist from clicked 2D hand points")
    fit.add_argument("--annotations", type=Path, required=True)
    fit.add_argument("--pkl-root", type=Path, required=True)
    fit.add_argument("--intrinsics", type=Path, required=True, help="exact stored-RGB fisheye intrinsics")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--holdout-ratio", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=42)
    fit.set_defaults(func=run_fit_camera)
    diagnose = subparsers.add_parser("diagnose-camera", help="render prediction-versus-click overlays for an existing camera fit")
    diagnose.add_argument("--annotations", type=Path, required=True)
    diagnose.add_argument("--pkl-root", type=Path, required=True)
    diagnose.add_argument("--camera-contract", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.set_defaults(func=run_diagnose_camera)
    pocket = subparsers.add_parser("fit-pocket", help="fit human functional grasp pocket from clicked desired-pocket pixels")
    pocket.add_argument("--annotations", type=Path, required=True, help="click_label_visible_hand_points.py --label-profile grasp_pocket output")
    pocket.add_argument("--pkl-root", type=Path, required=True)
    pocket.add_argument("--camera-contract", type=Path, required=True)
    pocket.add_argument("--output", type=Path, required=True)
    pocket.add_argument("--normal-sign", type=float, choices=(-1.0, 1.0), required=True)
    pocket.add_argument("--initial-a", type=float, default=0.0)
    pocket.add_argument("--initial-b", type=float, default=0.35)
    pocket.add_argument("--holdout-ratio", type=float, default=0.2)
    pocket.add_argument("--seed", type=int, default=42)
    pocket.set_defaults(func=run_fit_pocket)
    effective = subparsers.add_parser("fit-effective-pocket", help="jointly fit an effective PICO-to-camera G/L prior from direct grasp_pocket clicks")
    effective.add_argument("--annotations", type=Path, required=True, help="click_label_visible_hand_points.py --label-profile grasp_pocket output")
    effective.add_argument("--pkl-root", type=Path, required=True)
    effective.add_argument("--intrinsics", type=Path, required=True)
    effective.add_argument("--output", type=Path, required=True)
    effective.add_argument("--init-camera-contract", type=Path, default=None, help="optional old PICO-image fit; only an optimizer initialization")
    effective.add_argument("--normal-sign", type=float, choices=(-1.0, 1.0), required=True)
    effective.add_argument("--initial-a", type=float, default=0.0)
    effective.add_argument("--initial-b", type=float, default=0.35)
    effective.add_argument("--holdout-ratio", type=float, default=0.2)
    effective.add_argument("--seed", type=int, default=42)
    effective.set_defaults(func=run_fit_effective_pocket)
    preview = subparsers.add_parser("preview", help="render raw overlay + wide/local views without modifying PKLs")
    preview.add_argument("--input", type=Path, required=True)
    preview.add_argument("--output", type=Path, required=True)
    preview.add_argument("--episodes", default=None, help="e.g. 79,80 or episode_0079_hand_aug")
    preview.add_argument("--limit-episodes", type=int, default=None)
    preview.add_argument("--episode-name-regex", default=None, help="optional full-match regex, e.g. '^episode_[0-9]{4}$' to exclude *_hand_aug")
    preview.add_argument("--max-frames-per-episode", type=int, default=40)
    preview.add_argument("--stride", type=int, default=4)
    _add_view_arguments(preview)
    preview.set_defaults(func=run_preview)
    materialize = subparsers.add_parser("materialize", help="copy input dataset and add calibrated global/local canonical images")
    materialize.add_argument("--input", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True, help="must be a new root; source PKLs are never changed")
    materialize.add_argument("--episodes", default=None)
    materialize.add_argument("--limit-episodes", type=int, default=None)
    materialize.add_argument("--episode-name-regex", default=None, help="optional full-match regex, e.g. '^episode_[0-9]{4}$' to exclude *_hand_aug")
    materialize.add_argument("--jpeg-quality", type=int, default=95)
    _add_view_arguments(materialize)
    materialize.set_defaults(func=run_materialize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if hasattr(args, "ema_alpha") and not 0.0 <= args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in [0,1]")
    if args.command in {"preview", "materialize"}:
        _apply_pocket_config(args)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
