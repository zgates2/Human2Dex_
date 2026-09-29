#!/usr/bin/env python3
"""Record real-policy inference as DexUMI-compatible PKL/JPG episodes."""

from __future__ import annotations

import hashlib
import os
import pickle
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from umi.common.interpolation_util import PoseInterpolator


_EPISODE_RE = re.compile(r"^episode_(\d+)$")
_SMALL_HASH_LIMIT = 16 * 1024 * 1024


def _safe_filename_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "rgb"


def _run_git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _artifact_snapshot(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result
    stat = path.stat()
    result.update({
        "sizeBytes": int(stat.st_size),
        "mtimeNs": int(stat.st_mtime_ns),
    })
    if stat.st_size <= _SMALL_HASH_LIMIT:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    else:
        result["sha256"] = None
        result["sha256Reason"] = "file_exceeds_inline_hash_limit"
    return result


def _runtime_provenance(metadata: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    dirty = _run_git(repo_root, "status", "--short")
    artifacts = {}
    for key in ("checkpoint", "robotConfig"):
        value = metadata.get(key)
        if value:
            artifacts[key] = _artifact_snapshot(value)
    return {
        "hostname": socket.gethostname(),
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version.split()[0],
        "repoRoot": str(repo_root),
        "gitHead": _run_git(repo_root, "rev-parse", "HEAD"),
        "gitBranch": _run_git(repo_root, "branch", "--show-current"),
        "gitDirty": bool(dirty),
        "gitStatusShort": [] if not dirty else dirty.splitlines(),
        "artifacts": artifacts,
        "capturedWallTime": time.time(),
    }


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape (H,W,3), got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        finite_max = float(np.nanmax(arr)) if arr.size else 0.0
        if finite_max <= 1.5:
            arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _next_episode_index(root: Path) -> int:
    used = set()
    if root.exists():
        for child in root.iterdir():
            match = _EPISODE_RE.fullmatch(child.name)
            if match is not None:
                used.add(int(match.group(1)))
    index = 1
    while index in used:
        index += 1
    return index


@dataclass
class _JpgTask:
    image: np.ndarray
    path: Path


class _AsyncJpgWriter:
    def __init__(self, quality: int = 95, workers: int = 2, max_queue: int = 512):
        self.quality = int(quality)
        self._queue: queue.Queue[Optional[_JpgTask]] = queue.Queue(
            maxsize=max(1, int(max_queue))
        )
        self._threads = [
            threading.Thread(target=self._run, name=f"InferenceJpgWriter-{idx}", daemon=True)
            for idx in range(max(1, int(workers)))
        ]
        self._started = False
        self._stopped = False
        self._lock = threading.Lock()
        self.queued = 0
        self.written = 0
        self.blocked_puts = 0
        self.errors: list[tuple[str, str]] = []

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for thread in self._threads:
            thread.start()

    def enqueue(self, image: np.ndarray, path: Path) -> None:
        if not self._started or self._stopped:
            raise RuntimeError("JPG writer is not running")
        task = _JpgTask(image=_as_rgb_uint8(image).copy(), path=Path(path))
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            with self._lock:
                self.blocked_puts += 1
            self._queue.put(task)
        with self._lock:
            self.queued += 1

    def wait_empty(self) -> None:
        if self._started:
            self._queue.join()

    def stop(self) -> None:
        if self._stopped:
            return
        self.wait_empty()
        self._stopped = True
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                task.path.parent.mkdir(parents=True, exist_ok=True)
                bgr = cv2.cvtColor(task.image, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(
                    str(task.path),
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
                )
                if not ok:
                    raise IOError(f"cv2.imwrite failed: {task.path}")
                with self._lock:
                    self.written += 1
            except Exception as exc:
                failed_path = "<shutdown>" if task is None else str(task.path)
                with self._lock:
                    self.errors.append((failed_path, repr(exc)))
            finally:
                self._queue.task_done()


class InferenceEpisodeRecorder:
    """Align camera and measured TCP buffers to submitted policy waypoints."""

    def __init__(
        self,
        output_dir: str | Path,
        frequency: float,
        hand_backend: str,
        *,
        camera_idx: int = 0,
        record_fps: Optional[float] = None,
        jpg_quality: int = 95,
        jpg_workers: int = 2,
        jpg_queue_size: int = 512,
    ):
        if hand_backend not in {"linker_o6", "wuji_hand"}:
            raise ValueError("episode recording requires hand=linker_o6 or hand=wuji_hand")
        self.root = Path(output_dir).expanduser() / "inference_episodes"
        self.frequency = float(frequency)
        self.record_fps = float(self.frequency if record_fps is None else record_fps)
        if not np.isfinite(self.record_fps) or self.record_fps <= 0.0:
            raise ValueError(f"record_fps must be finite and > 0, got {record_fps!r}")
        self.hand_backend = str(hand_backend)
        self.camera_idx = int(camera_idx)
        self.jpg_quality = int(jpg_quality)
        self.jpg_workers = int(jpg_workers)
        self.jpg_queue_size = int(jpg_queue_size)
        self.max_rgb_delta_s = max(0.05, 1.5 / max(self.record_fps, 1.0))

        self._episode_dir: Optional[Path] = None
        self._pkl_path: Optional[Path] = None
        self._episode_index: Optional[int] = None
        self._metadata: dict[str, Any] = {}
        self._messages: list[dict[str, Any]] = []
        self._pending: list[
            tuple[
                float,
                np.ndarray,
                Optional[np.ndarray],
                Optional[dict[str, np.ndarray]],
            ]
        ] = []
        self._writer: Optional[_AsyncJpgWriter] = None
        self._last_timestamp = -np.inf
        self._last_rgb_frame_id: Optional[int] = None
        self._started_wall_time: Optional[float] = None
        self._next_sample_timestamp: Optional[float] = None
        self._last_command_timestamp = -np.inf
        self._last_command: Optional[np.ndarray] = None
        self._last_measured_state: Optional[np.ndarray] = None
        self._last_policy_input_images: Optional[dict[str, np.ndarray]] = None
        self._stats = {
            "queuedActionCommands": 0,
            "queuedSamples": 0,
            "savedSamples": 0,
            "skippedDuplicateTimestamp": 0,
            "skippedOldBuffer": 0,
            "skippedRgbAlignment": 0,
            "bufferReadErrors": 0,
            "pendingDroppedAtFinish": 0,
            "queuedMeasuredHandStates": 0,
            "savedMeasuredHandStates": 0,
            "queuedPolicyInputImages": 0,
            "savedPolicyInputImages": 0,
        }

    @property
    def active(self) -> bool:
        return self._episode_dir is not None

    @property
    def pkl_path(self) -> Optional[Path]:
        return self._pkl_path

    def start_episode(self, metadata: Optional[dict[str, Any]] = None) -> Path:
        if self.active:
            raise RuntimeError("an inference episode is already active")
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            episode_index = _next_episode_index(self.root)
            episode_name = f"episode_{episode_index:04d}"
            episode_dir = self.root / episode_name
            try:
                episode_dir.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                continue

        self._episode_dir = episode_dir
        self._pkl_path = episode_dir / f"{episode_name}.pkl"
        self._episode_index = episode_index
        self._messages = []
        self._pending = []
        self._last_timestamp = -np.inf
        self._last_rgb_frame_id = None
        self._next_sample_timestamp = None
        self._last_command_timestamp = -np.inf
        self._last_command = None
        self._last_measured_state = None
        self._last_policy_input_images = None
        self._started_wall_time = time.time()
        self._stats = {key: 0 for key in self._stats}
        self._metadata = dict(metadata or {})
        self._metadata.setdefault(
            "provenance",
            _runtime_provenance(self._metadata),
        )
        self._writer = _AsyncJpgWriter(
            quality=self.jpg_quality,
            workers=self.jpg_workers,
            max_queue=self.jpg_queue_size,
        )
        self._writer.start()
        print(f"[EpisodeRecorder] recording to {self._pkl_path}")
        return self._pkl_path

    def record_segment(
        self,
        env: Any,
        timestamps: np.ndarray,
        hand_commands: np.ndarray,
        measured_hand_states: Optional[np.ndarray] = None,
        policy_obs: Optional[dict[str, Any]] = None,
        shape_meta: Optional[dict[str, Any]] = None,
    ) -> int:
        if not self.active:
            return 0
        ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        commands = self._normalize_commands(hand_commands)
        measured_states = self._normalize_measured_states(measured_hand_states)
        if len(ts) != len(commands):
            raise ValueError(
                f"record timestamps/hand command length mismatch: {len(ts)} vs {len(commands)}"
            )
        if measured_states is not None and len(measured_states) != len(commands):
            raise ValueError(
                "record measured-state/hand-command length mismatch: "
                f"{len(measured_states)} vs {len(commands)}"
            )
        policy_input_images = self._extract_policy_input_images(policy_obs, shape_meta)
        for index, (timestamp, command) in enumerate(zip(ts, commands)):
            if not np.isfinite(timestamp):
                continue
            if timestamp <= self._last_command_timestamp + 1e-6:
                self._stats["skippedDuplicateTimestamp"] += 1
                continue
            measured_state = None
            if measured_states is not None:
                measured_state = np.asarray(measured_states[index]).copy()
            if self._next_sample_timestamp is None:
                self._next_sample_timestamp = float(timestamp)
            if self._last_command is not None:
                self._queue_fixed_samples_until(
                    float(timestamp),
                    command=self._last_command,
                    measured_state=self._last_measured_state,
                    policy_input_images=self._last_policy_input_images,
                    include_end=False,
                )
            self._last_command = np.asarray(command).copy()
            self._last_measured_state = measured_state
            self._last_policy_input_images = self._copy_policy_images(policy_input_images)
            self._last_command_timestamp = float(timestamp)
            self._stats["queuedActionCommands"] += 1
            self._queue_fixed_samples_until(
                float(timestamp),
                command=self._last_command,
                measured_state=self._last_measured_state,
                policy_input_images=self._last_policy_input_images,
                include_end=True,
            )
        return self._flush_available(env)

    def finish_episode(self, env: Any = None, reason: str = "finished") -> Optional[Path]:
        if not self.active:
            return None
        if env is not None and self._pending:
            last_target = self._pending[-1][0]
            remaining = last_target - time.time()
            if 0.0 < remaining <= 0.25:
                time.sleep(remaining + 0.02)
            self._flush_available(env)

        self._stats["pendingDroppedAtFinish"] += len(self._pending)
        self._pending = []
        writer = self._writer
        if writer is not None:
            writer.stop()

        failed_paths = set()
        if writer is not None:
            failed_paths = {path for path, _ in writer.errors}
        valid_messages = []
        for message in self._messages:
            image_path = self._episode_dir / str(message["rgbImage"])
            policy_image_ok = True
            policy_image = message.get("policyInputRgbImage")
            if policy_image:
                policy_image_path = self._episode_dir / str(policy_image)
                policy_image_ok = (
                    str(policy_image_path) not in failed_paths
                    and policy_image_path.exists()
                )
            policy_images = message.get("policyInputRgbImages") or {}
            if isinstance(policy_images, dict):
                for relpath in policy_images.values():
                    if not relpath:
                        continue
                    policy_image_path = self._episode_dir / str(relpath)
                    policy_image_ok = (
                        policy_image_ok
                        and str(policy_image_path) not in failed_paths
                        and policy_image_path.exists()
                    )
            if (
                    str(image_path) not in failed_paths
                    and image_path.exists()
                    and policy_image_ok):
                valid_messages.append(message)
        self._messages = valid_messages
        self._stats["savedSamples"] = len(valid_messages)
        self._stats["savedPolicyInputImages"] = sum(
            len(message.get("policyInputRgbImages") or {})
            if isinstance(message.get("policyInputRgbImages") or {}, dict)
            else int(bool(message.get("policyInputRgbImage")))
            for message in valid_messages
        )

        result = None
        if self._messages:
            ended_wall_time = time.time()
            metadata = dict(self._metadata)
            metadata.update({
                "sourceScript": "eval_real_franka_o6.py",
                "episodeIndex": int(self._episode_index),
                "collectionHz": float(self.record_fps),
                "controlHz": float(self.frequency),
                "recordHz": float(self.record_fps),
                "handBackend": self.hand_backend,
                "picoFieldsRecorded": False,
                "recording": {
                    "sampleBasis": (
                        "fixed-rate clock starting at the first submitted policy "
                        "waypoint timestamp"
                    ),
                    "samplePeriodSeconds": 1.0 / float(self.record_fps),
                    "commandTiming": (
                        "hand commands are zero-order-held from the latest submitted "
                        "waypoint at or before each fixed-rate sample"
                    ),
                    "startedWallTime": self._started_wall_time,
                    "endedWallTime": ended_wall_time,
                    "finishReason": str(reason),
                },
                "trajectoryPose": {
                    "shape": [6],
                    "dtype": "float32",
                    "representation": "[x,y,z,rx,ry,rz], rotation vector in radians",
                    "source": "Franka ActualTCPPose in the configured policy TCP frame",
                },
                "handCommand": self._hand_metadata(),
                "rgb": {
                    "cameraIndex": self.camera_idx,
                    "storage": "jpg files saved in the bundle images/ subdirectory",
                    "imageField": "rgbImage",
                    "jpegQuality": self.jpg_quality,
                    "maxAlignmentDeltaMs": self.max_rgb_delta_s * 1000.0,
                    "writer": {
                        "queued": 0 if writer is None else writer.queued,
                        "written": 0 if writer is None else writer.written,
                        "blockedPuts": 0 if writer is None else writer.blocked_puts,
                        "errors": [] if writer is None else list(writer.errors),
                    },
                },
                "policyInputRgb": {
                    "cameraIndex": self.camera_idx,
                    "storage": "jpg files saved in the bundle policy_images/ subdirectory",
                    "legacyImageField": "policyInputRgbImage",
                    "imageField": "policyInputRgbImages",
                    "source": (
                        "latest frame from the processed policy observation after "
                        "hand observation patching, skeleton overlay, canonical views, "
                        "and view alignment"
                    ),
                    "keys": "RGB observation keys from checkpoint shape_meta when available",
                    "sameImageForActionsFromOneInferenceChunk": True,
                    "sameImageForSamplesFromOneInferenceChunk": True,
                },
                "stats": dict(self._stats),
            })
            payload = {
                "formatVersion": 4,
                "metadata": metadata,
                "messages": self._messages,
            }
            tmp_path = self._pkl_path.with_suffix(".pkl.tmp")
            with tmp_path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self._pkl_path)
            result = self._pkl_path
            print(
                f"[EpisodeRecorder] saved {len(self._messages)} frames -> {self._pkl_path}"
            )
        else:
            print("[EpisodeRecorder] no valid aligned frames; no PKL was written")
            shutil.rmtree(self._episode_dir, ignore_errors=True)

        self._reset_active_state()
        return result

    def _flush_available(self, env: Any) -> int:
        if not self._pending:
            return 0
        try:
            camera = self._read_camera_buffer(env)
            robot = env.robots[0].get_all_state()
            camera_ts = np.asarray(camera["timestamp"], dtype=np.float64).reshape(-1)
            robot_ts = np.asarray(robot["robot_timestamp"], dtype=np.float64).reshape(-1)
            robot_pose = np.asarray(robot["ActualTCPPose"], dtype=np.float64).reshape(-1, 6)
        except Exception as exc:
            self._stats["bufferReadErrors"] += 1
            print(f"[EpisodeRecorder] buffer read delayed: {exc}")
            return 0

        valid_camera = np.isfinite(camera_ts)
        valid_robot = np.isfinite(robot_ts) & np.all(np.isfinite(robot_pose), axis=1)
        camera_ts = camera_ts[valid_camera]
        robot_ts = robot_ts[valid_robot]
        robot_pose = robot_pose[valid_robot]
        camera_images = self._as_frame_array(camera["color"])[valid_camera]
        camera_frame_ids = self._optional_camera_array(
            camera, "camera_frame_id", len(valid_camera), valid_camera
        )
        camera_capture_ts = self._optional_camera_array(
            camera, "camera_capture_timestamp", len(valid_camera), valid_camera
        )
        if len(camera_ts) == 0 or len(robot_ts) == 0:
            return 0

        camera_order = np.argsort(camera_ts)
        camera_ts = camera_ts[camera_order]
        camera_images = camera_images[camera_order]
        if camera_frame_ids is not None:
            camera_frame_ids = camera_frame_ids[camera_order]
        if camera_capture_ts is not None:
            camera_capture_ts = camera_capture_ts[camera_order]

        robot_ts, unique_idx = np.unique(robot_ts, return_index=True)
        robot_pose = robot_pose[unique_idx]
        common_latest = min(float(camera_ts[-1]), float(robot_ts[-1]))
        ready_epsilon = max(1e-9, (1.0 / self.record_fps) * 1e-6)
        remaining = []
        ready: list[
            tuple[
                float,
                np.ndarray,
                Optional[np.ndarray],
                Optional[dict[str, np.ndarray]],
            ]
        ] = []
        for item in self._pending:
            if item[0] <= common_latest + ready_epsilon:
                ready.append(item)
            else:
                remaining.append(item)
        self._pending = remaining
        if not ready:
            return 0

        eligible = []
        for item in ready:
            if item[0] < robot_ts[0] or item[0] < camera_ts[0] - self.max_rgb_delta_s:
                self._stats["skippedOldBuffer"] += 1
            else:
                eligible.append(item)
        if not eligible:
            return 0

        target_ts = np.asarray([item[0] for item in eligible], dtype=np.float64)
        commands = [item[1] for item in eligible]
        measured_states = [item[2] for item in eligible]
        policy_images = [item[3] for item in eligible]
        pose_values = self._interpolate_pose(robot_ts, robot_pose, target_ts)
        saved = 0
        for target, command, measured_state, policy_image, pose in zip(
            target_ts, commands, measured_states, policy_images, pose_values
        ):
            camera_idx = int(np.argmin(np.abs(camera_ts - target)))
            rgb_delta_s = float(target - camera_ts[camera_idx])
            if abs(rgb_delta_s) > self.max_rgb_delta_s:
                self._stats["skippedRgbAlignment"] += 1
                continue
            frame_id = None
            if camera_frame_ids is not None:
                frame_id = int(camera_frame_ids[camera_idx])
            capture_ts = float(camera_ts[camera_idx])
            if camera_capture_ts is not None:
                capture_ts = float(camera_capture_ts[camera_idx])
            self._append_message(
                target_timestamp=float(target),
                trajectory_pose=pose,
                command=command,
                measured_hand_state=measured_state,
                image=camera_images[camera_idx],
                policy_input_images=policy_image,
                frame_id=frame_id,
                capture_timestamp=capture_ts,
                align_delta_s=rgb_delta_s,
            )
            saved += 1
        return saved

    def _read_camera_buffer(self, env: Any) -> dict[str, np.ndarray]:
        requested = max(8, len(self._pending) + 8)
        camera_collection = getattr(env.camera, "cameras", None)
        if camera_collection is not None:
            values = list(camera_collection.values())
            if self.camera_idx < len(values):
                ring = getattr(values[self.camera_idx], "ring_buffer", None)
                if ring is not None:
                    requested = min(
                        requested,
                        int(ring.count),
                        int(ring.get_max_k),
                    )
        requested = max(1, requested)
        camera_data = env.camera.get(k=requested)
        if self.camera_idx not in camera_data:
            raise KeyError(f"camera{self.camera_idx} is unavailable")
        return camera_data[self.camera_idx]

    @staticmethod
    def _copy_policy_images(
        policy_input_images: Optional[dict[str, np.ndarray]],
    ) -> Optional[dict[str, np.ndarray]]:
        if not policy_input_images:
            return None
        return {
            str(key): np.asarray(image).copy()
            for key, image in policy_input_images.items()
        }

    def _queue_fixed_samples_until(
        self,
        end_timestamp: float,
        *,
        command: np.ndarray,
        measured_state: Optional[np.ndarray],
        policy_input_images: Optional[dict[str, np.ndarray]],
        include_end: bool,
    ) -> None:
        if self._next_sample_timestamp is None:
            self._next_sample_timestamp = float(end_timestamp)
        period = 1.0 / self.record_fps
        epsilon = max(1e-9, period * 1e-6)
        while True:
            sample_timestamp = float(self._next_sample_timestamp)
            if include_end:
                should_queue = sample_timestamp <= float(end_timestamp) + epsilon
            else:
                should_queue = sample_timestamp < float(end_timestamp) - epsilon
            if not should_queue:
                break
            self._append_pending_sample(
                sample_timestamp,
                command=command,
                measured_state=measured_state,
                policy_input_images=policy_input_images,
            )
            self._next_sample_timestamp = sample_timestamp + period

    def _append_pending_sample(
        self,
        timestamp: float,
        *,
        command: np.ndarray,
        measured_state: Optional[np.ndarray],
        policy_input_images: Optional[dict[str, np.ndarray]],
    ) -> None:
        if timestamp <= self._last_timestamp + 1e-6:
            self._stats["skippedDuplicateTimestamp"] += 1
            return
        if measured_state is not None:
            self._stats["queuedMeasuredHandStates"] += 1
        if policy_input_images:
            self._stats["queuedPolicyInputImages"] += len(policy_input_images)
        self._pending.append(
            (
                float(timestamp),
                np.asarray(command).copy(),
                None if measured_state is None else np.asarray(measured_state).copy(),
                self._copy_policy_images(policy_input_images),
            )
        )
        self._last_timestamp = float(timestamp)
        self._stats["queuedSamples"] += 1

    @staticmethod
    def _as_frame_array(value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(f"camera color buffer has invalid shape {arr.shape}")
        return arr

    def _extract_policy_input_images(
        self,
        policy_obs: Optional[dict[str, Any]],
        shape_meta: Optional[dict[str, Any]],
    ) -> dict[str, np.ndarray]:
        if policy_obs is None:
            return {}
        keys = self._policy_rgb_keys(policy_obs, shape_meta)
        images: dict[str, np.ndarray] = {}
        for key in keys:
            if key not in policy_obs:
                continue
            frames = self._as_frame_array(policy_obs[key])
            images[key] = np.asarray(frames[-1]).copy()
        return images

    def _policy_rgb_keys(
        self,
        policy_obs: dict[str, Any],
        shape_meta: Optional[dict[str, Any]],
    ) -> list[str]:
        if shape_meta is not None:
            obs_meta = shape_meta.get("obs", {})
            keys = [
                str(key)
                for key, attr in obs_meta.items()
                if str(attr.get("type", "low_dim")) == "rgb"
            ]
            return [key for key in keys if key in policy_obs]
        default_key = f"camera{self.camera_idx}_rgb"
        if default_key in policy_obs:
            return [default_key]
        return sorted(
            str(key)
            for key, value in policy_obs.items()
            if str(key).endswith("_rgb")
            and np.asarray(value).ndim in (3, 4)
        )

    @staticmethod
    def _optional_camera_array(
        camera: dict[str, np.ndarray],
        key: str,
        original_len: int,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        if key not in camera:
            return None
        arr = np.asarray(camera[key]).reshape(-1)
        if len(arr) != original_len:
            return None
        return arr[mask]

    @staticmethod
    def _interpolate_pose(
        robot_ts: np.ndarray,
        robot_pose: np.ndarray,
        target_ts: np.ndarray,
    ) -> np.ndarray:
        if len(robot_ts) == 1:
            return np.repeat(robot_pose[:1], len(target_ts), axis=0)
        interpolator = PoseInterpolator(t=robot_ts, x=robot_pose)
        return np.asarray(interpolator(target_ts), dtype=np.float64).reshape(-1, 6)

    def _append_message(
        self,
        *,
        target_timestamp: float,
        trajectory_pose: np.ndarray,
        command: np.ndarray,
        measured_hand_state: Optional[np.ndarray],
        image: np.ndarray,
        policy_input_images: Optional[dict[str, np.ndarray]] = None,
        frame_id: Optional[int] = None,
        capture_timestamp: float = 0.0,
        align_delta_s: float = 0.0,
    ) -> None:
        message_idx = len(self._messages)
        image_name = f"{self._pkl_path.stem}_rgb_{message_idx:06d}.jpg"
        image_path = self._episode_dir / "images" / image_name
        self._writer.enqueue(image, image_path)
        relpath = str(Path("images") / image_name)
        policy_relpath = None
        policy_relpaths: dict[str, str] = {}
        for key, policy_input_image in (policy_input_images or {}).items():
            key_text = _safe_filename_component(key)
            policy_image_name = (
                f"{self._pkl_path.stem}_policy_{key_text}_{message_idx:06d}.jpg"
            )
            policy_image_path = (
                self._episode_dir / "policy_images" / key_text / policy_image_name
            )
            self._writer.enqueue(policy_input_image, policy_image_path)
            policy_relpaths[str(key)] = str(
                Path("policy_images") / key_text / policy_image_name
            )
        default_key = f"camera{self.camera_idx}_rgb"
        if default_key in policy_relpaths:
            policy_relpath = policy_relpaths[default_key]
        elif policy_relpaths:
            policy_relpath = policy_relpaths[sorted(policy_relpaths)[0]]
        repeated = frame_id is not None and frame_id == self._last_rgb_frame_id
        message: dict[str, Any] = {
            "timestamp": float(target_timestamp),
            "trajectoryPose": np.asarray(trajectory_pose, dtype=np.float32).reshape(6),
            "rgbImage": relpath,
            "policyInputRgbImage": policy_relpath,
            "policyInputRgbImages": policy_relpaths,
            "rgbFrameId": frame_id,
            "rgbCaptureTimestamp": float(capture_timestamp),
            "rgbAlignResidualNs": int(round(align_delta_s * 1e9)),
            "rgbFrameRepeated": bool(repeated),
        }
        if self.hand_backend == "linker_o6":
            o6_command = np.asarray(command, dtype=np.uint8).reshape(6)
            message["o6_command"] = o6_command
            message["hand_command"] = o6_command.copy()
            message["o6_measured_state"] = (
                None
                if measured_hand_state is None
                else np.asarray(measured_hand_state, dtype=np.uint8).reshape(6)
            )
            message["wuji_command"] = None
            message["wuji_measured_state"] = None
        else:
            message["o6_command"] = None
            message["hand_command"] = None
            message["o6_measured_state"] = None
            message["wuji_command"] = np.asarray(command, dtype=np.float32).reshape(5, 4)
            message["wuji_measured_state"] = (
                None
                if measured_hand_state is None
                else np.asarray(measured_hand_state, dtype=np.float32).reshape(5, 4)
            )
        if measured_hand_state is not None:
            self._stats["savedMeasuredHandStates"] += 1
        self._messages.append(message)
        self._last_rgb_frame_id = frame_id

    def _normalize_commands(self, commands: np.ndarray) -> np.ndarray:
        arr = np.asarray(commands)
        if self.hand_backend == "linker_o6":
            arr = arr.reshape(-1, 6)
            if not np.all(np.isfinite(arr)):
                raise ValueError("O6 hand commands contain NaN or Inf")
            return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        if arr.ndim >= 3 and arr.shape[-2:] == (5, 4):
            arr = arr.reshape(-1, 20)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, 20)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Wuji hand commands contain NaN or Inf")
        return arr

    def _normalize_measured_states(
        self, measured_states: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if measured_states is None:
            return None
        arr = np.asarray(measured_states)
        if self.hand_backend == "linker_o6":
            arr = np.asarray(arr, dtype=np.float64).reshape(-1, 6)
            if not np.all(np.isfinite(arr)):
                raise ValueError("O6 measured states contain NaN or Inf")
            return np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        if arr.ndim >= 3 and arr.shape[-2:] == (5, 4):
            arr = arr.reshape(-1, 20)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1, 20)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Wuji measured states contain NaN or Inf")
        return arr

    def _hand_metadata(self) -> dict[str, Any]:
        if self.hand_backend == "linker_o6":
            return {
                "fields": ["o6_command", "hand_command"],
                "measuredField": "o6_measured_state",
                "measuredStateTiming": "cached state immediately before the corresponding command",
                "shape": [6],
                "dtype": "uint8",
                "range": [0, 255],
                "source": "post-clip policy command submitted to the O6 hand",
            }
        return {
            "field": "wuji_command",
            "measuredField": "wuji_measured_state",
            "measuredStateTiming": "state immediately before the corresponding command when available",
            "shape": [5, 4],
            "dtype": "float32",
            "unit": "radian",
            "source": "post-clamp/filter command submitted to the Wuji hand",
        }

    def _reset_active_state(self) -> None:
        self._episode_dir = None
        self._pkl_path = None
        self._episode_index = None
        self._metadata = {}
        self._messages = []
        self._pending = []
        self._writer = None
        self._started_wall_time = None
        self._last_timestamp = -np.inf
        self._last_rgb_frame_id = None
        self._next_sample_timestamp = None
        self._last_command_timestamp = -np.inf
        self._last_command = None
        self._last_measured_state = None
        self._last_policy_input_images = None
