#!/usr/bin/env python3
"""
 - 读取 episode_xxxx.pkl 里的 messages
  - 读取每帧 rgbImage
  - 读取手部状态：
      - O6 默认用 o6_measured_state，fallback 到 o6_command，要求 6 维
      - Wuji 默认用 wuji_measured_state，fallback 到 wuji_command，要求 20 维

  - 用 FK + 相机外参把手部 21 点骨架投影到图像上
  - 根据 configs/grasp_pocket_v1.json 计算抓取 pocket 点并画出来
  - 如果 episode 里有 objectPocketObsDebug，直接用里面记录的物体中心
  - 如果没有，并且没传 --no-object，会启动 SAM3 检测物体中心并缓存
  - 输出叠加视频、逐帧图片和 summary.json

    O6 示例：

  cd /home/zjc/Desktop/human2dex
  conda run -n l515_mvs310 python scripts_real/render_o6_inference_episode_video.py \
    --episode-dir data_local/franka_o6_eval/inference_episodes/episode_0004 \
    --overwrite

  Wuji 示例：

  cd /home/zjc/Desktop/human2dex
  conda run -n l515_mvs310 python scripts_real/render_wuji_inference_episode_video.py \
    --episode-dir data_local/wuji_cup/inference_episodes/episode_0001 \
    --overwrite
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_REAL = ROOT / "scripts_real"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_REAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_REAL))

from real_inference_skeleton_overlay import (  # noqa: E402
    O6SkeletonOverlayRenderer,
    WujiSkeletonOverlayRenderer,
)
from o6_fk21 import O6Kinematics  # noqa: E402
from o6_grasp_pocket_overlay import (  # noqa: E402
    compute_grasp_pocket,
    load_calibration,
    project_points,
    uv_valid,
)
from wuji_fk21 import WujiKinematics  # noqa: E402


DEFAULT_EPISODE_DIR = (
    ROOT / "data_local/franka_o6_eval/inference_episodes/episode_0004"
)
DEFAULT_O6_CALIBRATION = (
    ROOT
    / "data_local/o6_camera_calibration/mvs_DA9057802_o6_mount_v1"
    / "fifit_20260905_230324/camera_from_o6_base.json" # o6_v4
)
DEFAULT_WUJI_CALIBRATION = (
    ROOT
    / "data_local/wuji_camera_calibration/mvs_DA9057802_wuji_mount_v1"
    / "fit_20260814_231759/camera_from_wuji_base.json"
)
DEFAULT_POCKET_CONFIG = ROOT / "configs/grasp_pocket_v1.json"
DEFAULT_SAM3_PYTHON = Path("/home/zjc/miniconda3/envs/sam3/bin/python")
DEFAULT_SAM3_CODE = Path("/home/zjc/Desktop/human2dex/sam/sam3-code")
DEFAULT_SAM3_CHECKPOINT = Path("/home/zjc/Desktop/human2dex/sam/sam3/sam3.pt")
DEFAULT_OBJECT_PROMPTS = ("brown box", "egg box", "egg tray")

POCKET_BGR = (72, 224, 255)
OBJECT_BGR = (255, 214, 83)


@dataclass
class EpisodeFrame:
    sequence_index: int
    message_index: int
    message: dict[str, Any]
    state: np.ndarray
    image_path: Path


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def load_episode(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise TypeError(f"expected dict/messages PKL: {path}")
    return payload["messages"], dict(payload.get("metadata", {}))


def resolve_episode_pkl(episode_dir: Path) -> Path:
    preferred = episode_dir / f"{episode_dir.name}.pkl"
    if preferred.is_file():
        return preferred
    candidates = sorted(episode_dir.glob("*.pkl"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"cannot uniquely locate episode PKL under {episode_dir}")


def resolve_image(episode_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = episode_dir / path
    return path.resolve()


def valid_hand_state(value: Any, *, expected_dim: int, nonnegative: bool) -> np.ndarray | None:
    try:
        state = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if state.shape != (int(expected_dim),) or not np.isfinite(state).all():
        return None
    if nonnegative and np.any(state < 0):
        return None
    return state


def collect_frames(
    episode_dir: Path,
    messages: list[dict[str, Any]],
    *,
    state_field: str,
    fallback_state_field: str,
    expected_state_dim: int,
    nonnegative_state: bool,
    image_field: str,
    max_frames: int | None,
) -> list[EpisodeFrame]:
    frames: list[EpisodeFrame] = []
    for message_index, message in enumerate(messages):
        state = valid_hand_state(
            message.get(state_field),
            expected_dim=expected_state_dim,
            nonnegative=nonnegative_state,
        )
        if state is None:
            state = valid_hand_state(
                message.get(fallback_state_field),
                expected_dim=expected_state_dim,
                nonnegative=nonnegative_state,
            )
        image_path = resolve_image(episode_dir, message.get(image_field))
        if state is None or image_path is None or not image_path.is_file():
            continue
        frames.append(EpisodeFrame(
            sequence_index=len(frames),
            message_index=message_index,
            message=message,
            state=state,
            image_path=image_path,
        ))
        if max_frames is not None and len(frames) >= int(max_frames):
            break
    if not frames:
        raise RuntimeError(
            f"no valid image/hand-state frames found: "
            f"state_field={state_field!r}, fallback={fallback_state_field!r}, "
            f"expected_dim={expected_state_dim}"
        )
    return frames


def load_pocket_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    for key in ("a", "b", "normal_sign"):
        value = float(payload[key])
        if not math.isfinite(value):
            raise ValueError(f"invalid grasp pocket {key}={value}")
    return payload


def compute_pocket_tracks(
    frames: list[EpisodeFrame],
    calibration,
    pocket_config: dict[str, Any],
    *,
    kinematics_cls,
) -> tuple[list[list[float] | None], list[bool]]:
    kinematics = kinematics_cls(calibration.urdf_path)
    values: list[list[float] | None] = []
    valid: list[bool] = []
    for frame in frames:
        try:
            points21 = kinematics.points21(frame.state)
            pocket = compute_grasp_pocket(
                points21,
                a=float(pocket_config["a"]),
                b=float(pocket_config["b"]),
                normal_sign=float(pocket_config["normal_sign"]),
            )
            uv, depth = project_points(pocket.point3d[None, :], calibration)
            image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("image read failed")
            ok = uv_valid(uv[0], float(depth[0]), image.shape[1], image.shape[0])
            values.append(uv[0].tolist() if ok else None)
            valid.append(bool(ok))
        except Exception:
            values.append(None)
            valid.append(False)
    return values, valid


def as_uv(value: Any) -> np.ndarray | None:
    try:
        uv = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if uv.size < 2 or not np.isfinite(uv[:2]).all():
        return None
    return uv[:2].copy()


def smooth_uv_sequence(
    values: Sequence[Any],
    validity: Sequence[bool],
    *,
    median_window: int,
    gaussian_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(values)
    array = np.full((count, 2), np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)
    for index, (value, is_valid) in enumerate(zip(values, validity)):
        uv = as_uv(value)
        if bool(is_valid) and uv is not None:
            array[index] = uv
            valid[index] = True
    if not valid.any():
        return array, valid

    filtered = array.copy()
    radius = max(0, int(median_window) // 2)
    if radius:
        for index in np.flatnonzero(valid):
            start = max(0, int(index) - radius)
            stop = min(count, int(index) + radius + 1)
            neighbours = array[start:stop][valid[start:stop]]
            if len(neighbours):
                filtered[index] = np.median(neighbours, axis=0)

    indices = np.arange(count, dtype=np.float64)
    valid_indices = np.flatnonzero(valid).astype(np.float64)
    filled = np.empty_like(filtered)
    for axis in range(2):
        filled[:, axis] = np.interp(indices, valid_indices, filtered[valid, axis])

    if gaussian_sigma > 0 and count > 1:
        radius = max(1, int(math.ceil(3.0 * gaussian_sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / float(gaussian_sigma)) ** 2)
        kernel /= kernel.sum()
        padded = np.pad(filled, ((radius, radius), (0, 0)), mode="edge")
        for axis in range(2):
            filled[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")
    return filled, valid


class Sam3Worker:
    def __init__(self, args: argparse.Namespace):
        command = [
            str(Path(args.sam3_python).expanduser().resolve()),
            str(ROOT / "real_inference_object_pocket.py"),
            "--sam3-worker",
            "--sam3-code", str(Path(args.sam3_code).expanduser().resolve()),
            "--checkpoint", str(Path(args.sam3_checkpoint).expanduser().resolve()),
            "--version", str(args.sam3_version),
            "--prompt", str(args.object_prompt[0]),
            "--device", str(args.sam3_device),
            "--min-confidence", str(float(args.object_min_confidence)),
            "--min-area", str(int(args.object_min_area)),
            "--max-area-ratio", str(float(args.object_max_area_ratio)),
            "--jpeg-quality", "95",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        ready = self._read_json("ready")
        print(f"SAM3 ready: prompt={ready.get('prompt')}")

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            text = line.rstrip()
            if text:
                self._stderr_lines.append(text)
                if len(self._stderr_lines) > 100:
                    del self._stderr_lines[:-100]

    def _read_json(self, expected_type: str) -> dict[str, Any]:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                details = "\n".join(self._stderr_lines[-20:])
                raise RuntimeError(f"SAM3 worker exited unexpectedly\n{details}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == expected_type:
                return payload

    def detect(
        self,
        image_path: Path,
        prompts: Sequence[str],
        sequence: int,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        specs = [
            {
                "name": f"candidate_{index}",
                "prompt": prompt,
                "min_confidence": float(args.object_min_confidence),
                "min_area": int(args.object_min_area),
                "max_area_ratio": float(args.object_max_area_ratio),
            }
            for index, prompt in enumerate(prompts)
        ]
        request = {
            "type": "frame",
            "seq": int(sequence),
            "objects": specs,
            "image_jpeg_b64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        }
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._read_json("result")

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=5.0)
        except Exception:
            self.process.terminate()


def object_cache_config(args: argparse.Namespace, frames: list[EpisodeFrame]) -> dict[str, Any]:
    return {
        "version": 1,
        "episode_dir": str(Path(args.episode_dir).expanduser().resolve()),
        "frames": len(frames),
        "object_name": str(args.object_name),
        "prompts": list(args.object_prompt),
        "stride": int(args.object_stride),
        "min_confidence": float(args.object_min_confidence),
        "min_area": int(args.object_min_area),
        "max_area_ratio": float(args.object_max_area_ratio),
        "max_jump_px": float(args.object_max_jump_px),
    }


def choose_object_candidate(
    rows: Sequence[Any],
    previous_uv: np.ndarray | None,
    prompts: Sequence[str],
    max_jump_px: float,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not bool(row.get("visible", False)):
            continue
        uv = as_uv(row.get("uv"))
        if uv is None:
            continue
        candidate = dict(row)
        candidate["uv_array"] = uv
        candidate["prompt"] = prompts[index] if index < len(prompts) else row.get("prompt")
        candidate["distance_from_previous"] = (
            None if previous_uv is None else float(np.linalg.norm(uv - previous_uv))
        )
        candidates.append(candidate)
    if not candidates:
        return None
    if previous_uv is None:
        return max(candidates, key=lambda row: float(row.get("confidence", 0.0)))
    plausible = [
        row for row in candidates
        if float(row["distance_from_previous"]) <= float(max_jump_px)
    ]
    if not plausible:
        return None
    return max(
        plausible,
        key=lambda row: float(row.get("confidence", 0.0)) - 0.002 * float(row["distance_from_previous"]),
    )


def run_object_detection(
    frames: list[EpisodeFrame],
    cache_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    config = object_cache_config(args, frames)
    if cache_path.is_file() and not args.rerun_object:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("config") != config:
            raise RuntimeError(
                f"object cache config differs: {cache_path}; pass --rerun-object to rebuild"
            )
        rows = payload.get("frames", [])
        if len(rows) != len(frames):
            raise RuntimeError("object cache frame count mismatch; pass --rerun-object")
        print(f"reusing object cache: {cache_path}")
        return rows

    prompts = list(args.object_prompt)
    worker = Sam3Worker(args)
    memory_uv: np.ndarray | None = None
    memory_area = 0.0
    memory_confidence = 0.0
    rows: list[dict[str, Any]] = []
    detections = 0
    try:
        for frame in frames:
            source = "hold"
            chosen_serializable = None
            candidates_serializable: list[dict[str, Any]] = []
            if frame.sequence_index % max(1, int(args.object_stride)) == 0:
                response = worker.detect(
                    frame.image_path,
                    prompts,
                    frame.sequence_index,
                    args,
                )
                candidate_rows = response.get("objects", [])
                for candidate in candidate_rows if isinstance(candidate_rows, list) else []:
                    if isinstance(candidate, dict):
                        candidates_serializable.append(candidate)
                chosen = choose_object_candidate(
                    candidate_rows if isinstance(candidate_rows, list) else [],
                    memory_uv,
                    prompts,
                    float(args.object_max_jump_px),
                )
                if chosen is not None:
                    memory_uv = np.asarray(chosen.pop("uv_array"), dtype=np.float64)
                    memory_area = float(chosen.get("mask_area", 0.0) or 0.0)
                    memory_confidence = float(chosen.get("confidence", 0.0) or 0.0)
                    chosen_serializable = chosen
                    source = "sam3"
                    detections += 1
            rows.append({
                "sequence_index": frame.sequence_index,
                "message_index": frame.message_index,
                "image": str(frame.image_path),
                "name": str(args.object_name),
                "uv": None if memory_uv is None else memory_uv.tolist(),
                "mask_area": float(memory_area),
                "confidence": float(memory_confidence),
                "valid": memory_uv is not None,
                "source": source if memory_uv is not None else "invalid",
                "chosen": chosen_serializable,
                "candidates": candidates_serializable,
            })
            if frame.sequence_index % 10 == 0 or frame.sequence_index == len(frames) - 1:
                print(f"object detection {frame.sequence_index + 1}/{len(frames)} detections={detections}")
    finally:
        worker.close()
    atomic_write_json(cache_path, {"config": config, "frames": rows})
    print(f"object cache: {cache_path}")
    return rows


def recorded_object_rows(frames: list[EpisodeFrame], object_name: str) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    found = False
    last_uv: np.ndarray | None = None
    for frame in frames:
        debug = frame.message.get("objectPocketObsDebug")
        chosen = None
        if isinstance(debug, dict):
            for item in debug.get("objects", []):
                if isinstance(item, dict) and str(item.get("name")) == object_name:
                    chosen = item
                    break
        if chosen is not None:
            uv = as_uv(chosen.get("uv"))
            if uv is not None:
                last_uv = uv
                found = True
        rows.append({
            "sequence_index": frame.sequence_index,
            "message_index": frame.message_index,
            "name": object_name,
            "uv": None if last_uv is None else last_uv.tolist(),
            "mask_area": 0.0 if chosen is None else float(chosen.get("area", 0.0) or 0.0),
            "confidence": 0.0 if chosen is None else float(chosen.get("confidence", 0.0) or 0.0),
            "valid": last_uv is not None,
            "source": "recorded" if chosen is not None else ("hold" if last_uv is not None else "invalid"),
        })
    return rows if found else None


def draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 5,
) -> None:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1.0:
        return
    direction = (p1 - p0) / length
    position = 0.0
    while position < length:
        a = p0 + position * direction
        b = p0 + min(position + dash, length) * direction
        cv2.line(
            image,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            color,
            int(thickness),
            cv2.LINE_AA,
        )
        position += dash * 1.65


def draw_pocket_object(
    image: np.ndarray,
    pocket_uv: np.ndarray | None,
    object_uv: np.ndarray | None,
    object_name: str,
) -> None:
    h, w = image.shape[:2]
    pocket = None if pocket_uv is None else tuple(np.rint(pocket_uv).astype(int))
    obj = None if object_uv is None else tuple(np.rint(object_uv).astype(int))
    if pocket is not None:
        cv2.circle(image, pocket, 5, POCKET_BGR, 2, cv2.LINE_AA)
        cv2.circle(image, pocket, 2, POCKET_BGR, -1, cv2.LINE_AA)
        cv2.line(image, (pocket[0] - 8, pocket[1]), (pocket[0] + 8, pocket[1]), POCKET_BGR, 1, cv2.LINE_AA)
        cv2.line(image, (pocket[0], pocket[1] - 8), (pocket[0], pocket[1] + 8), POCKET_BGR, 1, cv2.LINE_AA)
    if obj is not None:
        cv2.circle(image, obj, 5, OBJECT_BGR, 2, cv2.LINE_AA)
        cv2.circle(image, obj, 2, OBJECT_BGR, -1, cv2.LINE_AA)
    if pocket is None or obj is None:
        return
    draw_dashed_line(image, pocket, obj, OBJECT_BGR, thickness=1, dash=5)
    delta = np.asarray(object_uv, dtype=np.float64) - np.asarray(pocket_uv, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    midpoint = 0.5 * (np.asarray(pocket, dtype=np.float64) + np.asarray(obj, dtype=np.float64))
    direction = np.asarray(obj, dtype=np.float64) - np.asarray(pocket, dtype=np.float64)
    norm = max(float(np.linalg.norm(direction)), 1.0)
    midpoint += np.asarray([-direction[1], direction[0]]) / norm * 11.0
    label = f"{object_name.replace('_', ' ')} ({delta[0]:.1f},{delta[1]:.1f}) {distance:.1f}px"
    font_scale = 0.25
    size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x = int(np.clip(midpoint[0] - 0.5 * size[0], 2, max(2, w - size[0] - 2)))
    y = int(np.clip(midpoint[1], size[1] + 2, h - 3))
    # Match the human QC video: colored text with no black outline or side panel.
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, OBJECT_BGR, 1, cv2.LINE_AA)


def encode_video(frames_dir: Path, video_path: Path, fps: float) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:g}",
        "-i", str(frames_dir / "frame_%06d.jpg"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(video_path),
    ], check=True)


def run(args: argparse.Namespace) -> int:
    hand = str(args.hand)
    if hand == "o6":
        hand_label = "o6"
        state_field = str(args.state_field or "o6_measured_state")
        fallback_state_field = "o6_command"
        expected_state_dim = 6
        nonnegative_state = True
        calibration_default = DEFAULT_O6_CALIBRATION
        kinematics_cls = O6Kinematics
        renderer_cls = O6SkeletonOverlayRenderer
    else:
        hand_label = "wuji"
        state_field = str(args.state_field or "wuji_measured_state")
        fallback_state_field = "wuji_command"
        expected_state_dim = 20
        nonnegative_state = False
        calibration_default = DEFAULT_WUJI_CALIBRATION
        kinematics_cls = WujiKinematics
        renderer_cls = WujiSkeletonOverlayRenderer

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else episode_dir / f"{hand_label}_object_pocket_video"
    )
    pkl_path = resolve_episode_pkl(episode_dir)
    messages, metadata = load_episode(pkl_path)
    frames = collect_frames(
        episode_dir,
        messages,
        state_field=state_field,
        fallback_state_field=fallback_state_field,
        expected_state_dim=expected_state_dim,
        nonnegative_state=nonnegative_state,
        image_field=str(args.image_field),
        max_frames=args.max_frames,
    )
    calibration_path = Path(args.calibration or calibration_default).expanduser().resolve()
    pocket_config_path = Path(args.grasp_pocket_config).expanduser().resolve()
    calibration = load_calibration(calibration_path)
    pocket_config = load_pocket_config(pocket_config_path)

    pocket_values, pocket_valid_raw = compute_pocket_tracks(
        frames,
        calibration,
        pocket_config,
        kinematics_cls=kinematics_cls,
    )
    pocket_smooth, pocket_valid = smooth_uv_sequence(
        pocket_values,
        pocket_valid_raw,
        median_window=int(args.center_median_window),
        gaussian_sigma=float(args.center_smooth_sigma),
    )

    object_rows = recorded_object_rows(frames, str(args.object_name))
    cache_path = output_dir / "object_detections.json"
    if object_rows is None and not args.no_object:
        output_dir.mkdir(parents=True, exist_ok=True)
        object_rows = run_object_detection(frames, cache_path, args)
    if object_rows is None:
        object_rows = [
            {"uv": None, "valid": False, "source": "disabled"}
            for _ in frames
        ]
    object_smooth, object_valid = smooth_uv_sequence(
        [row.get("uv") for row in object_rows],
        [bool(row.get("valid", False)) for row in object_rows],
        median_window=int(args.center_median_window),
        gaussian_sigma=float(args.center_smooth_sigma),
    )

    renderer = renderer_cls(
        calibration_path,
        rgb_keys=["camera0_rgb"],
        line_thickness=int(args.line_thickness),
        point_radius=float(args.point_radius),
        draw_wrist=bool(args.draw_wrist),
        wrist_extension_ratio=float(args.wrist_extension_ratio),
    )
    frames_dir = output_dir / "rendered_frames"
    if frames_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{frames_dir} exists; pass --overwrite")
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_summaries: list[dict[str, Any]] = []
    for output_index, frame in enumerate(frames):
        image_bgr = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"failed to read {frame.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        env = {"camera0_rgb": image_rgb[None, ...]}
        skeleton_rgb = renderer.apply(env, frame.state)["camera0_rgb"][0]
        rendered = cv2.cvtColor(np.asarray(skeleton_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        pocket_uv = pocket_smooth[output_index] if pocket_valid[output_index] else None
        object_uv = object_smooth[output_index] if object_valid[output_index] else None
        draw_pocket_object(rendered, pocket_uv, object_uv, str(args.object_name))
        output_path = frames_dir / f"frame_{output_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]):
            raise RuntimeError(f"failed to write {output_path}")
        delta = None
        distance = None
        if pocket_uv is not None and object_uv is not None:
            delta_array = object_uv - pocket_uv
            delta = delta_array.tolist()
            distance = float(np.linalg.norm(delta_array))
        frame_summaries.append({
            "output_index": output_index,
            "message_index": frame.message_index,
            "image": str(frame.image_path),
            "state_field": state_field,
            "pocket_uv": None if pocket_uv is None else pocket_uv.tolist(),
            "object_uv": None if object_uv is None else object_uv.tolist(),
            "object_source": object_rows[output_index].get("source"),
            "pocket_to_object_delta_uv": delta,
            "pocket_to_object_distance_px": distance,
            "rendered": str(output_path),
        })

    fps = float(args.fps) if float(args.fps) > 0 else float(metadata.get("collectionHz", 10.0) or 10.0)
    video_path = output_dir / f"{episode_dir.name}_{hand_label}_object_pocket.mp4"
    encode_video(frames_dir, video_path, fps)
    summary = {
        "format": "dexterous_hand_inference_episode_object_pocket_video_v2",
        "hand": hand_label,
        "episode_dir": str(episode_dir),
        "source_pkl": str(pkl_path),
        "frames": len(frames),
        "fps": fps,
        "state_field": state_field,
        "fallback_state_field": fallback_state_field,
        "image_field": str(args.image_field),
        "calibration": str(calibration_path),
        "grasp_pocket_config": str(pocket_config_path),
        "skeleton": {
            "line_thickness": int(args.line_thickness),
            "point_radius": float(args.point_radius),
            "draw_wrist": bool(args.draw_wrist),
            "wrist_extension_ratio": float(args.wrist_extension_ratio),
        },
        "object": {
            "enabled": not args.no_object,
            "name": str(args.object_name),
            "prompts": list(args.object_prompt),
            "stride": int(args.object_stride),
            "cache": str(cache_path),
        },
        "filter": {
            "median_window": int(args.center_median_window),
            "gaussian_sigma": float(args.center_smooth_sigma),
        },
        "video": str(video_path),
        "frame_records": frame_summaries,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "frames": len(frames),
        "fps": fps,
        "object_valid_frames": int(object_valid.sum()),
        "pocket_valid_frames": int(pocket_valid.sum()),
        "video": str(video_path),
    }, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hand",
        choices=("o6", "wuji"),
        default="o6",
        help="Select hand FK, measured-state field, renderer, and default calibration.",
    )
    parser.add_argument("--episode-dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Defaults to the latest checked O6/Wuji calibration for --hand.",
    )
    parser.add_argument("--grasp-pocket-config", type=Path, default=DEFAULT_POCKET_CONFIG)
    parser.add_argument(
        "--state-field",
        default=None,
        help="Defaults to o6_measured_state or wuji_measured_state for --hand.",
    )
    parser.add_argument("--image-field", default="rgbImage")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--point-radius", type=float, default=4.0)
    parser.add_argument("--draw-wrist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wrist-extension-ratio", type=float, default=0.85)
    parser.add_argument("--center-median-window", type=int, default=5)
    parser.add_argument("--center-smooth-sigma", type=float, default=3.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--object-name", default="egg_box")
    parser.add_argument("--object-prompt", action="append", default=None)
    parser.add_argument("--object-stride", type=int, default=2)
    parser.add_argument("--object-min-confidence", type=float, default=0.15)
    parser.add_argument("--object-min-area", type=int, default=12)
    parser.add_argument("--object-max-area-ratio", type=float, default=0.75)
    parser.add_argument("--object-max-jump-px", type=float, default=65.0)
    parser.add_argument("--no-object", action="store_true")
    parser.add_argument("--rerun-object", action="store_true")
    parser.add_argument("--sam3-python", type=Path, default=DEFAULT_SAM3_PYTHON)
    parser.add_argument("--sam3-code", type=Path, default=DEFAULT_SAM3_CODE)
    parser.add_argument("--sam3-checkpoint", type=Path, default=DEFAULT_SAM3_CHECKPOINT)
    parser.add_argument("--sam3-version", default="sam3")
    parser.add_argument("--sam3-device", default="0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.object_prompt is None:
        args.object_prompt = list(DEFAULT_OBJECT_PROMPTS)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
