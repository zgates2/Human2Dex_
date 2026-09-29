#!/usr/bin/env python3
"""点击O6地标，适配一个相机<-O6_base变换，并渲染遮挡区域的21点骨架。
工作流被刻意拆分为两个命令：
1、``annotate`` 功能可选择多样且同步的O6帧，并打开一个小型窗口。```
 OpenCV点击用户界面。拟合/保留集成员资格在任何拟合之前即已确定。
2. ``fit`` 仅使用 ``split=fit`` 点击来估计一个刚性变换，并
 在未参与训练的RGB图像上渲染完整的MediaPipe风格21点正向运动学骨架。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from o6_fk21 import (
    ANCHOR_SETS,
    ANCHOR_TO_POINT21_INDEX,
    CALIBRATION_ANCHOR_TO_LINK,
    HAND21_BONES,
    HAND21_NAMES,
    O6Kinematics,
    POINT21_COLORS_BGR,
)


DEFAULT_URDF = "/home/zjc/Desktop/human2dex/linker_o6/linker_o6/right/linkerhand_o6_right.urdf"
DEFAULT_CAMERA_CONTRACT = (
    "/home/zjc/Desktop/human2dex/converted_data/params/"
    "camera_contract_mvs_DA9057801_o6_mount_v1.yaml"
)
DEFAULT_EPISODES_ROOT = (
    "/home/zjc/Desktop/human2dex/"
    "data_local/franka_o6_eval/inference_episodes"
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_structured_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"PyYAML is required to read camera contract {path}"
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_camera_contract(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    data = load_structured_file(path)
    camera = data.get("camera", {})
    mount = data.get("mount", {})
    intrinsics = data.get("intrinsics", {})
    policy = data.get("policy_image", {})
    kinematics = data.get("kinematics", {})

    model = str(camera.get("model", "")).lower()
    if model not in {"fisheye", "opencv_fisheye"}:
        raise ValueError(f"Contract camera model must be OpenCV fisheye, got {model!r}")
    resize = tuple(int(value) for value in policy.get("resize", []))
    if len(resize) != 2:
        raise ValueError("camera contract policy_image.resize must be [width, height]")
    K = np.asarray(policy.get("K"), dtype=np.float64).reshape(3, 3)
    D = np.asarray(policy.get("D"), dtype=np.float64).reshape(4, 1)
    mount_id = str(mount.get("camera_mount_id", ""))
    contract_id = str(data.get("contract_id", ""))
    if not mount_id or contract_id != mount_id:
        raise ValueError(
            f"contract_id {contract_id!r} must match camera_mount_id {mount_id!r}"
        )

    source_path = Path(str(intrinsics.get("source_path", ""))).expanduser()
    source_sha256 = str(intrinsics.get("source_sha256", ""))
    if source_path.is_file() and source_sha256:
        actual = file_sha256(source_path.resolve())
        if actual != source_sha256:
            raise ValueError(
                f"Intrinsics source hash changed: contract={source_sha256} actual={actual}"
            )

    urdf_path = Path(str(kinematics.get("urdf_path", DEFAULT_URDF))).expanduser().resolve()
    urdf_sha256 = str(kinematics.get("urdf_sha256", ""))
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    actual_urdf_sha256 = file_sha256(urdf_path)
    if urdf_sha256 and actual_urdf_sha256 != urdf_sha256:
        raise ValueError(
            f"URDF hash changed: contract={urdf_sha256} actual={actual_urdf_sha256}"
        )

    record = {
        "path": str(path),
        "sha256": file_sha256(path),
        "contract_id": contract_id,
        "camera_mount_id": mount_id,
        "camera_serial": camera.get("serial"),
        "camera_model": "fisheye",
        "policy_image": policy,
        "intrinsics_source_path": str(source_path),
        "intrinsics_source_sha256": source_sha256,
        "urdf_path": str(urdf_path),
        "urdf_sha256": actual_urdf_sha256,
    }
    return {
        "raw": data,
        "record": record,
        "K": K,
        "D": D,
        "image_size": resize,
        "camera_model": "fisheye",
        "urdf_path": urdf_path,
    }


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            import dill  # type: ignore

            return dill.load(handle)


def parse_episode_spec(spec: str) -> list[int]:
    result: list[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid episode range {token!r}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    unique = sorted(set(result))
    if not unique:
        raise ValueError("No episodes were selected")
    return unique


def episode_pkl_path(root: Path, episode_number: int) -> Path:
    name = f"episode_{episode_number:04d}"
    return root / name / f"{name}.pkl"


def resolve_image_path(episode_dir: Path, image_value: str) -> Path:
    path = Path(str(image_value))
    return path if path.is_absolute() else episode_dir / path


def collect_candidates(
    episodes_root: Path,
    episode_numbers: list[int],
    state_source: str,
    state_time_alignment: str,
    max_align_ms: float,
    include_repeated: bool,
) -> list[dict[str, Any]]:
    if state_time_alignment not in {"rgb", "message"}:
        raise ValueError(
            "state_time_alignment must be 'rgb' or 'message', got "
            f"{state_time_alignment!r}"
        )
    candidates: list[dict[str, Any]] = []
    for episode_number in episode_numbers:
        pkl_path = episode_pkl_path(episodes_root, episode_number)
        if not pkl_path.is_file():
            raise FileNotFoundError(pkl_path)
        payload = load_pickle(pkl_path)
        metadata = payload.get("metadata", {})
        if metadata.get("handBackend") != "linker_o6":
            raise ValueError(f"{pkl_path} is not a linker_o6 episode")
        messages = payload.get("messages", [])
        state_times = []
        state_values = []
        for message in messages:
            timestamp = message.get("timestamp")
            state = message.get(state_source)
            if timestamp is None or state is None:
                continue
            state_array = np.asarray(state, dtype=np.float64).reshape(-1)
            if (
                state_array.shape == (6,)
                and np.all(np.isfinite(state_array))
                and np.isfinite(float(timestamp))
            ):
                state_times.append(float(timestamp))
                state_values.append(state_array)
        if not state_times:
            continue
        state_times_array = np.asarray(state_times, dtype=np.float64)
        state_values_array = np.stack(state_values, axis=0)
        order = np.argsort(state_times_array)
        state_times_array = state_times_array[order]
        state_values_array = state_values_array[order]
        state_times_array, unique_indices = np.unique(
            state_times_array, return_index=True
        )
        state_values_array = state_values_array[unique_indices]

        for message_index, message in enumerate(messages):
            state = message.get(state_source)
            if state is None:
                continue
            state_array = np.asarray(state, dtype=np.float64).reshape(-1)
            if state_array.shape != (6,) or not np.all(np.isfinite(state_array)):
                continue
            if not include_repeated and bool(message.get("rgbFrameRepeated", False)):
                continue
            residual_ns = message.get("rgbAlignResidualNs")
            align_ms = None if residual_ns is None else float(residual_ns) * 1e-6
            if align_ms is not None and abs(align_ms) > max_align_ms:
                continue
            message_timestamp = message.get("timestamp")
            if message_timestamp is None or not np.isfinite(float(message_timestamp)):
                continue
            state_query_timestamp = float(message_timestamp)
            aligned_state = state_array.copy()
            if state_time_alignment == "rgb" and residual_ns is not None:
                # The recorder defines residual = target_timestamp - rgb_timestamp.
                # q_meas is sampled around target_timestamp, so interpolate the
                # measured-state sequence to the actual RGB timestamp.
                state_query_timestamp -= float(residual_ns) * 1e-9
                if (
                    len(state_times_array) < 2
                    or state_query_timestamp < state_times_array[0]
                    or state_query_timestamp > state_times_array[-1]
                ):
                    continue
                aligned_state = np.asarray(
                    [
                        np.interp(
                            state_query_timestamp,
                            state_times_array,
                            state_values_array[:, joint_index],
                        )
                        for joint_index in range(6)
                    ],
                    dtype=np.float64,
                )
            image_value = message.get("rgbImage")
            if not image_value:
                continue
            image_path = resolve_image_path(pkl_path.parent, str(image_value))
            if not image_path.is_file():
                continue
            candidates.append(
                {
                    "episode": f"episode_{episode_number:04d}",
                    "episode_number": int(episode_number),
                    "pkl_path": str(pkl_path),
                    "message_index": int(message_index),
                    "image_path": str(image_path),
                    "rgb_frame_id": message.get("rgbFrameId"),
                    "timestamp": message.get("timestamp"),
                    "rgb_capture_timestamp": message.get("rgbCaptureTimestamp"),
                    "rgb_align_ms": align_ms,
                    "state_source": state_source,
                    "state_time_alignment": state_time_alignment,
                    "state_query_timestamp": state_query_timestamp,
                    "state_interpolation_shift_ms": (
                        state_query_timestamp - float(message_timestamp)
                    ) * 1000.0,
                    "state_interpolation_max_delta": float(
                        np.max(np.abs(aligned_state - state_array))
                    ),
                    "o6_state_message": state_array.astype(float).tolist(),
                    "o6_state": aligned_state.astype(float).tolist(),
                }
            )
    if not candidates:
        raise RuntimeError("No eligible O6 frames matched the filters")
    return candidates


def state_feature(frame: dict[str, Any]) -> np.ndarray:
    state = np.asarray(frame["o6_state"], dtype=np.float64)
    return np.clip((250.0 - state) / 250.0, 0.0, 1.0)


def select_diverse_frames(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    count = min(int(count), len(candidates))
    features = np.stack([state_feature(frame) for frame in candidates], axis=0)
    mean = np.mean(features, axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(features - mean, axis=1)))
    selected = [first]
    min_distance = np.linalg.norm(features - features[first], axis=1)
    min_distance[first] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(min_distance))
        if index in selected:
            break
        selected.append(index)
        distance = np.linalg.norm(features - features[index], axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
    frames = [dict(candidates[index]) for index in selected]
    frames.sort(key=lambda frame: (frame["episode_number"], frame["message_index"]))
    return frames


def make_annotation_payload(args: argparse.Namespace) -> dict[str, Any]:
    contract = None
    if getattr(args, "camera_contract", None):
        contract = load_camera_contract(Path(args.camera_contract))
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    episode_numbers = parse_episode_spec(args.episodes)
    candidates = collect_candidates(
        episodes_root=episodes_root,
        episode_numbers=episode_numbers,
        state_source=args.state_source,
        state_time_alignment=getattr(args, "state_time_alignment", "rgb"),
        max_align_ms=float(args.max_align_ms),
        include_repeated=bool(args.include_repeated),
    )
    frames = select_diverse_frames(candidates, int(args.num_frames))
    for index, frame in enumerate(frames):
        frame["split"] = "holdout" if (index + 1) % int(args.holdout_every) == 0 else "fit"
        frame["annotations"] = {}
        frame["skipped_anchors"] = []
        frame["done"] = False
    first_image = cv2.imread(frames[0]["image_path"], cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(frames[0]["image_path"])
    selected_image_size = (int(first_image.shape[1]), int(first_image.shape[0]))
    if contract is not None and selected_image_size != contract["image_size"]:
        raise ValueError(
            f"Selected RGB size {selected_image_size} does not match contract policy "
            f"image size {contract['image_size']}"
        )
    urdf_path = Path(
        args.urdf or (contract["urdf_path"] if contract is not None else DEFAULT_URDF)
    ).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if contract is not None and file_sha256(urdf_path) != contract["record"]["urdf_sha256"]:
        raise ValueError("Selected URDF does not match the camera contract")
    anchor_names = list(ANCHOR_SETS[args.anchor_set])
    result = {
        "formatVersion": 2,
        "tool": "calibrate_o6_camera_extrinsic.py",
        "episodes_root": str(episodes_root),
        "episodes": episode_numbers,
        "urdf_path": str(urdf_path),
        "urdf_sha256": file_sha256(urdf_path),
        "o6_base_link": "hand_base_link",
        "state_source": args.state_source,
        "selection": {
            "method": "greedy_farthest_o6_state",
            "eligible_frames": len(candidates),
            "selected_frames": len(frames),
            "max_align_ms": float(args.max_align_ms),
            "include_repeated": bool(args.include_repeated),
            "state_time_alignment": getattr(args, "state_time_alignment", "rgb"),
            "holdout_every": int(args.holdout_every),
        },
        "anchor_set": args.anchor_set,
        "anchor_names": anchor_names,
        "frames": frames,
    }
    if contract is not None:
        result["camera_mount_id"] = contract["record"]["camera_mount_id"]
        result["camera_contract"] = contract["record"]
    return result


CANONICAL_HAND_UV = np.asarray(
    [
        [130, 225],
        [105, 190], [80, 165], [55, 142], [32, 120],
        [108, 175], [102, 130], [98, 83], [94, 40],
        [135, 168], [135, 116], [135, 65], [135, 24],
        [162, 174], [171, 128], [178, 84], [182, 47],
        [188, 185], [207, 150], [218, 116], [226, 85],
    ],
    dtype=np.int32,
)


class AnnotationApp:
    def __init__(
        self,
        payload: dict[str, Any],
        output_path: Path,
        display_scale: int,
        min_points: int,
    ) -> None:
        self.payload = payload
        self.output_path = output_path
        self.frames = payload["frames"]
        self.anchor_names = list(payload["anchor_names"])
        self.display_scale = max(1, int(display_scale))
        self.min_points = max(1, int(min_points))
        self.window = "O6 camera extrinsic annotation"
        self.frame_index = next(
            (index for index, frame in enumerate(self.frames) if not frame.get("done", False)),
            0,
        )
        self.notice = ""
        self.notice_color = (80, 220, 255)
        self._image: np.ndarray | None = None
        self._display_image_size = (0, 0)
        self._load_current_image()

    @property
    def frame(self) -> dict[str, Any]:
        return self.frames[self.frame_index]

    def save(self) -> None:
        atomic_write_json(self.output_path, self.payload)

    def _load_current_image(self) -> None:
        image = cv2.imread(self.frame["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.frame["image_path"])
        self._image = image
        self._display_image_size = (
            image.shape[1] * self.display_scale,
            image.shape[0] * self.display_scale,
        )

    def _current_anchor(self) -> str | None:
        annotations = self.frame.get("annotations", {})
        skipped = set(self.frame.get("skipped_anchors", []))
        for name in self.anchor_names:
            if name not in annotations and name not in skipped:
                return name
        return None

    def _set_notice(self, text: str, color: tuple[int, int, int] = (80, 220, 255)) -> None:
        self.notice = text
        self.notice_color = color

    def _advance(self, delta: int) -> None:
        self.frame_index = int(np.clip(self.frame_index + delta, 0, len(self.frames) - 1))
        self._load_current_image()
        self._set_notice("")

    def _mark_click(self, display_x: int, display_y: int) -> None:
        width, height = self._display_image_size
        if not (0 <= display_x < width and 0 <= display_y < height):
            return
        anchor = self._current_anchor()
        if anchor is None:
            self._set_notice("Frame complete; press N for next frame")
            return
        u = float(display_x) / self.display_scale
        v = float(display_y) / self.display_scale
        self.frame.setdefault("annotations", {})[anchor] = [u, v]
        skipped = self.frame.setdefault("skipped_anchors", [])
        if anchor in skipped:
            skipped.remove(anchor)
        self.frame["done"] = False
        self.save()

    def _skip_current(self) -> None:
        anchor = self._current_anchor()
        if anchor is None:
            return
        skipped = self.frame.setdefault("skipped_anchors", [])
        if anchor not in skipped:
            skipped.append(anchor)
        self.frame["done"] = False
        self.save()

    def _undo(self) -> None:
        annotations = self.frame.setdefault("annotations", {})
        skipped = self.frame.setdefault("skipped_anchors", [])
        completed = [
            name for name in self.anchor_names if name in annotations or name in skipped
        ]
        if not completed:
            return
        name = completed[-1]
        annotations.pop(name, None)
        if name in skipped:
            skipped.remove(name)
        self.frame["done"] = False
        self.save()

    def _finish_frame(self) -> bool:
        count = len(self.frame.get("annotations", {}))
        if count < self.min_points:
            self._set_notice(
                f"Need at least {self.min_points} clicks; currently {count}",
                (80, 80, 255),
            )
            return False
        self.frame["done"] = True
        self.save()
        return True

    def _mouse(self, event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mark_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._skip_current()

    def _render_reference(self, height: int) -> np.ndarray:
        panel = np.full((height, 330, 3), 28, dtype=np.uint8)
        diagram_offset = np.array([40, 35], dtype=np.int32)
        points = CANONICAL_HAND_UV + diagram_offset
        current = self._current_anchor()
        current_index = None if current is None else ANCHOR_TO_POINT21_INDEX[current]
        for a, b in HAND21_BONES:
            cv2.line(panel, tuple(points[a]), tuple(points[b]), (105, 105, 105), 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            color = POINT21_COLORS_BGR[index]
            radius = 7 if index == current_index else 4
            cv2.circle(panel, tuple(point), radius, color, -1, cv2.LINE_AA)
            if index == current_index:
                cv2.circle(panel, tuple(point), radius + 4, (255, 255, 255), 2, cv2.LINE_AA)

        y = 305
        frame = self.frame
        lines = [
            f"Frame {self.frame_index + 1}/{len(self.frames)}  [{frame['split'].upper()}]",
            f"{frame['episode']}  message={frame['message_index']}",
            f"align={frame.get('rgb_align_ms')} ms",
            f"q->rgb shift={frame.get('state_interpolation_shift_ms', 0.0):.2f} ms",
            f"Click: {current or 'DONE'}",
            "LMB click | RMB/S skip | U undo",
            "N next | P previous | R reset | Q save/quit",
            f"Clicks: {len(frame.get('annotations', {}))}  min={self.min_points}",
        ]
        for line in lines:
            cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (225, 225, 225), 1, cv2.LINE_AA)
            y += 23
        if self.notice:
            cv2.putText(panel, self.notice, (12, min(y + 8, height - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.notice_color, 1, cv2.LINE_AA)
        return panel

    def render(self) -> np.ndarray:
        assert self._image is not None
        width, height = self._display_image_size
        image = cv2.resize(self._image, (width, height), interpolation=cv2.INTER_NEAREST)
        annotations = self.frame.get("annotations", {})
        for name, uv in annotations.items():
            index = ANCHOR_TO_POINT21_INDEX[name]
            point = (int(round(uv[0] * self.display_scale)), int(round(uv[1] * self.display_scale)))
            color = POINT21_COLORS_BGR[index]
            cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(image, name, (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        reference = self._render_reference(height)
        return np.hstack([image, reference])

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse)
        try:
            while True:
                cv2.imshow(self.window, self.render())
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    self._skip_current()
                elif key in (8, 127, ord("u")):
                    self._undo()
                elif key == ord("r"):
                    self.frame["annotations"] = {}
                    self.frame["skipped_anchors"] = []
                    self.frame["done"] = False
                    self.save()
                elif key == ord("n"):
                    if self._finish_frame():
                        if self.frame_index == len(self.frames) - 1:
                            self._set_notice("All frames finished; press Q to exit")
                        else:
                            self._advance(1)
                elif key == ord("p"):
                    self._advance(-1)
        finally:
            self.save()
            cv2.destroyWindow(self.window)


def run_annotate(args: argparse.Namespace) -> None:
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if getattr(args, "camera_contract", None):
            current_contract = load_camera_contract(Path(args.camera_contract))
            saved_contract = payload.get("camera_contract")
            if saved_contract is not None and (
                saved_contract.get("sha256") != current_contract["record"]["sha256"]
                or saved_contract.get("camera_mount_id")
                != current_contract["record"]["camera_mount_id"]
            ):
                raise ValueError(
                    "Existing annotations belong to a different camera contract. "
                    "Use a new output path or --overwrite only for the intended mount."
                )
        print(f"[resume] loaded {output_path}")
    else:
        payload = make_annotation_payload(args)
        atomic_write_json(output_path, payload)
        split_counts = {
            split: sum(frame["split"] == split for frame in payload["frames"])
            for split in ("fit", "holdout")
        }
        print(
            f"[created] {output_path} frames={len(payload['frames'])} "
            f"fit={split_counts['fit']} holdout={split_counts['holdout']}"
        )
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is not set. Run this command in the zjc desktop terminal "
            "or an SSH session with X forwarding."
        )
    app = AnnotationApp(
        payload=payload,
        output_path=output_path,
        display_scale=int(args.display_scale),
        min_points=int(args.min_points),
    )
    app.run()


def load_intrinsics(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    data = load_structured_file(path)
    if "K" in data and "D" in data:
        K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["D"], dtype=np.float64).reshape(4, 1)
        if "image_width" in data and "image_height" in data:
            size = (int(data["image_width"]), int(data["image_height"]))
        elif "image_size" in data:
            raw_size = [int(value) for value in data["image_size"]]
            if len(raw_size) != 2:
                raise ValueError("image_size must contain width,height")
            size = (raw_size[0], raw_size[1])
        else:
            size = expected_size
        model = str(data.get("camera_model", "fisheye")).lower()
    elif data.get("intrinsic_type") == "FISHEYE" and "intrinsics" in data:
        intr = data["intrinsics"]
        focal = float(intr["focal_length"])
        K = np.array(
            [
                [focal, 0.0, float(intr["principal_pt_x"])],
                [0.0, focal, float(intr["principal_pt_y"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        D = np.asarray(
            [
                intr["radial_distortion_1"],
                intr["radial_distortion_2"],
                intr["radial_distortion_3"],
                intr["radial_distortion_4"],
            ],
            dtype=np.float64,
        ).reshape(4, 1)
        size = (int(data["image_width"]), int(data["image_height"]))
        model = "fisheye"
    else:
        raise ValueError(
            "Unsupported intrinsics JSON. Expected {K,D,image_width,image_height} "
            "or OpenCameraImuCalibration FISHEYE format."
        )
    if model != "fisheye":
        raise ValueError(f"Only the fisheye camera model is supported, got {model!r}")
    if size != expected_size:
        raise ValueError(
            f"Intrinsics resolution {size} does not match annotated RGB {expected_size}. "
            "Calibrate the exact post-processed policy image instead of rescaling blindly."
        )
    return {"K": K, "D": D, "image_size": size, "camera_model": model, "raw": data}


def project_fisheye(
    points_o6: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_o6, dtype=np.float64).reshape(-1, 3)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    uv, _ = cv2.fisheye.projectPoints(points.reshape(-1, 1, 3), rvec, tvec, K, D)
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (rotation @ points.T + tvec).T
    return uv.reshape(-1, 2), camera_points[:, 2]


def collect_correspondences(
    payload: dict[str, Any],
    split: str,
    kinematics: O6Kinematics,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(payload["frames"]):
        if frame.get("split") != split:
            continue
        anchors = kinematics.calibration_anchor_points(frame["o6_state"])
        for anchor_name, uv in frame.get("annotations", {}).items():
            if anchor_name not in anchors:
                continue
            object_points.append(anchors[anchor_name])
            image_points.append(np.asarray(uv, dtype=np.float64).reshape(2))
            metadata.append(
                {
                    "frame_index": frame_index,
                    "episode": frame["episode"],
                    "message_index": frame["message_index"],
                    "anchor": anchor_name,
                    "o6_state": np.asarray(
                        frame["o6_state"], dtype=np.float64
                    ).astype(float).tolist(),
                }
            )
    if object_points:
        return np.stack(object_points), np.stack(image_points), metadata
    return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64), metadata


def initialize_extrinsic(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(object_points) < 6:
        raise ValueError(f"At least 6 fit clicks are required, got {len(object_points)}")
    normalized = cv2.fisheye.undistortPoints(
        image_points.reshape(-1, 1, 2), K, D
    )
    identity = np.eye(3, dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        normalized,
        identity,
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        raise RuntimeError("Initial EPNP camera<-O6_base solve failed")
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        normalized,
        identity,
        None,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("Iterative camera<-O6_base initialization failed")
    return rvec.reshape(3), tvec.reshape(3)


def fit_extrinsic(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rvec0, tvec0 = initialize_extrinsic(object_points, image_points, K, D)

    def residual(parameters: np.ndarray) -> np.ndarray:
        uv, depth = project_fisheye(
            object_points, parameters[:3], parameters[3:6], K, D
        )
        pixel_residual = (uv - image_points).reshape(-1)
        # least_squares requires a fixed residual dimension throughout the
        # optimization.  Keep one depth term per clicked point even when all
        # points are already in front of the camera (the terms are then zero).
        depth_penalty = np.maximum(0.0, 0.005 - depth) * 1000.0
        return np.concatenate([pixel_residual, depth_penalty])

    result = least_squares(
        residual,
        np.concatenate([rvec0, tvec0]),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=3000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    rvec = result.x[:3]
    tvec = result.x[3:6]
    _, depth = project_fisheye(object_points, rvec, tvec, K, D)
    if float(np.mean(depth > 0.0)) < 0.95:
        raise RuntimeError("Fitted camera transform puts too many O6 points behind the camera")
    return rvec, tvec, {
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
    }


def error_statistics(errors: np.ndarray) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean_px": None, "median_px": None, "p90_px": None, "p95_px": None, "max_px": None}
    return {
        "count": int(len(values)),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


def evaluate_correspondences(
    object_points: np.ndarray,
    image_points: np.ndarray,
    metadata: list[dict[str, Any]],
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not len(object_points):
        return error_statistics(np.empty(0)), []
    predicted, depth = project_fisheye(object_points, rvec, tvec, K, D)
    errors = np.linalg.norm(predicted - image_points, axis=1)
    rows = []
    for item, gt, pred, error, z in zip(metadata, image_points, predicted, errors, depth):
        residual = pred - gt
        rows.append(
            {
                **item,
                "gt_uv": gt.astype(float).tolist(),
                "pred_uv": pred.astype(float).tolist(),
                "residual_uv": residual.astype(float).tolist(),
                "error_px": float(error),
                "depth_m": float(z),
            }
        )
    return error_statistics(errors), rows


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) < 4 or float(np.std(x)) <= 1e-9 or float(np.std(y)) <= 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correspondence_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        residuals = np.asarray(
            [row["residual_uv"] for row in group_rows], dtype=np.float64
        ).reshape(-1, 2)
        errors = np.asarray(
            [row["error_px"] for row in group_rows], dtype=np.float64
        )
        bias = np.mean(residuals, axis=0)
        residual_norm_sum = float(np.sum(np.linalg.norm(residuals, axis=1)))
        direction_consistency = (
            0.0
            if residual_norm_sum <= 1e-12
            else float(np.linalg.norm(np.sum(residuals, axis=0)) / residual_norm_sum)
        )
        states = np.asarray(
            [row["o6_state"] for row in group_rows], dtype=np.float64
        ).reshape(-1, 6)
        q_correlations = {}
        for joint_index in range(6):
            corr_u = _correlation(states[:, joint_index], residuals[:, 0])
            corr_v = _correlation(states[:, joint_index], residuals[:, 1])
            corr_error = _correlation(states[:, joint_index], errors)
            if corr_u is not None or corr_v is not None or corr_error is not None:
                q_correlations[str(joint_index)] = {
                    "residual_u": corr_u,
                    "residual_v": corr_v,
                    "error": corr_error,
                }
        return {
            "error": error_statistics(errors),
            "bias_uv_px": bias.astype(float).tolist(),
            "bias_norm_px": float(np.linalg.norm(bias)),
            "direction_consistency": direction_consistency,
            "q_correlation": q_correlations,
        }

    by_anchor: dict[str, list[dict[str, Any]]] = {}
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_anchor.setdefault(str(row["anchor"]), []).append(row)
        frame_key = f"{row['episode']}:message_{int(row['message_index']):06d}"
        by_frame.setdefault(frame_key, []).append(row)
    return {
        "by_anchor": {
            key: summarize(value) for key, value in sorted(by_anchor.items())
        },
        "by_frame": {
            key: summarize(value) for key, value in sorted(by_frame.items())
        },
    }


def transform_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def draw_cross(image: np.ndarray, uv: np.ndarray, color: tuple[int, int, int]) -> None:
    point = tuple(np.round(uv).astype(int))
    cv2.drawMarker(image, point, color, cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)


def render_frame(
    frame: dict[str, Any],
    kinematics: O6Kinematics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    image = cv2.imread(frame["image_path"], cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame["image_path"])
    points21 = kinematics.points21(frame["o6_state"])
    uv21, depth21 = project_fisheye(points21, rvec, tvec, K, D)
    overlay = image.copy()
    for a, b in HAND21_BONES:
        if depth21[a] <= 0.0 or depth21[b] <= 0.0:
            continue
        pa = tuple(np.round(uv21[a]).astype(int))
        pb = tuple(np.round(uv21[b]).astype(int))
        cv2.line(overlay, pa, pb, POINT21_COLORS_BGR[b], 2, cv2.LINE_AA)
    for index, uv in enumerate(uv21):
        if depth21[index] <= 0.0:
            continue
        point = tuple(np.round(uv).astype(int))
        cv2.circle(overlay, point, 3 if index else 4, POINT21_COLORS_BGR[index], -1, cv2.LINE_AA)

    anchors = kinematics.calibration_anchor_points(frame["o6_state"])
    clicked_errors = []
    for name, gt_uv in frame.get("annotations", {}).items():
        pred_uv, pred_depth = project_fisheye(
            np.asarray([anchors[name]]), rvec, tvec, K, D
        )
        gt = np.asarray(gt_uv, dtype=np.float64)
        if pred_depth[0] > 0.0:
            index = ANCHOR_TO_POINT21_INDEX[name]
            cv2.circle(overlay, tuple(np.round(pred_uv[0]).astype(int)), 6, POINT21_COLORS_BGR[index], 1, cv2.LINE_AA)
            draw_cross(overlay, gt, (255, 255, 255))
            clicked_errors.append(float(np.linalg.norm(pred_uv[0] - gt)))

    rendered = cv2.addWeighted(image, 0.55, overlay, 0.45, 0.0)
    median_error = None if not clicked_errors else float(np.median(clicked_errors))
    text = f"{frame['split'].upper()} {frame['episode']} m={frame['message_index']}"
    if median_error is not None:
        text += f" click-med={median_error:.2f}px"
    cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(rendered, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return rendered, {
        "episode": frame["episode"],
        "message_index": frame["message_index"],
        "split": frame["split"],
        "clicked_median_error_px": median_error,
        "projected_points_in_front": int(np.sum(depth21 > 0.0)),
    }


def make_contact_sheet(images: list[np.ndarray], columns: int = 4, scale: int = 2) -> np.ndarray | None:
    if not images:
        return None
    resized = [cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) for image in images]
    height, width = resized[0].shape[:2]
    rows = (len(resized) + columns - 1) // columns
    sheet = np.full((rows * height, columns * width, 3), 245, dtype=np.uint8)
    for index, image in enumerate(resized):
        row, column = divmod(index, columns)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return sheet


def run_fit(args: argparse.Namespace) -> None:
    annotations_path = Path(args.annotations).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with annotations_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    saved_contract = payload.get("camera_contract")
    contract_path_value = args.camera_contract or (
        saved_contract.get("path") if isinstance(saved_contract, dict) else None
    )
    contract = (
        load_camera_contract(Path(contract_path_value))
        if contract_path_value
        else None
    )
    if contract is not None and isinstance(saved_contract, dict):
        if (
            saved_contract.get("sha256") != contract["record"]["sha256"]
            or saved_contract.get("camera_mount_id")
            != contract["record"]["camera_mount_id"]
        ):
            raise ValueError(
                "Annotation camera contract no longer matches the selected contract"
            )

    urdf_path = Path(
        args.urdf
        or payload.get("urdf_path")
        or (contract["urdf_path"] if contract is not None else DEFAULT_URDF)
    ).expanduser().resolve()
    actual_urdf_sha256 = file_sha256(urdf_path)
    if payload.get("urdf_sha256") and payload["urdf_sha256"] != actual_urdf_sha256:
        raise ValueError("URDF changed after annotation")
    if contract is not None and (
        actual_urdf_sha256 != contract["record"]["urdf_sha256"]
    ):
        raise ValueError("Fit URDF does not match the camera contract")
    kinematics = O6Kinematics(urdf_path)

    first_image = cv2.imread(payload["frames"][0]["image_path"], cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(payload["frames"][0]["image_path"])
    image_size = (int(first_image.shape[1]), int(first_image.shape[0]))
    if args.intrinsics:
        intrinsics_path = Path(args.intrinsics).expanduser().resolve()
        intrinsics = load_intrinsics(intrinsics_path, image_size)
        intrinsics_source = {
            "kind": "explicit_file",
            "path": str(intrinsics_path),
            "sha256": file_sha256(intrinsics_path),
        }
    elif contract is not None:
        if image_size != contract["image_size"]:
            raise ValueError(
                f"Annotated RGB size {image_size} does not match contract policy "
                f"image size {contract['image_size']}"
            )
        intrinsics = contract
        intrinsics_path = Path(contract["record"]["intrinsics_source_path"])
        intrinsics_source = {
            "kind": "camera_contract_policy_image",
            "contract_path": contract["record"]["path"],
            "contract_sha256": contract["record"]["sha256"],
            "source_path": contract["record"]["intrinsics_source_path"],
            "source_sha256": contract["record"]["intrinsics_source_sha256"],
        }
    else:
        raise ValueError("Provide --camera-contract or legacy --intrinsics")
    K, D = intrinsics["K"], intrinsics["D"]

    fit_object, fit_image, fit_meta = collect_correspondences(payload, "fit", kinematics)
    hold_object, hold_image, hold_meta = collect_correspondences(payload, "holdout", kinematics)
    if len(hold_object) < 4:
        raise ValueError(
            f"At least 4 held-out clicks are required for validation, got {len(hold_object)}"
        )
    rvec, tvec, optimizer = fit_extrinsic(fit_object, fit_image, K, D)
    fit_stats, fit_rows = evaluate_correspondences(
        fit_object, fit_image, fit_meta, rvec, tvec, K, D
    )
    hold_stats, hold_rows = evaluate_correspondences(
        hold_object, hold_image, hold_meta, rvec, tvec, K, D
    )
    fit_diagnostics = correspondence_diagnostics(fit_rows)
    holdout_diagnostics = correspondence_diagnostics(hold_rows)
    systematic_anchor_failures = []
    for anchor, diagnostic in holdout_diagnostics["by_anchor"].items():
        if (
            diagnostic["error"]["count"] >= 3
            and diagnostic["bias_norm_px"] > 4.0
            and diagnostic["direction_consistency"] > 0.70
        ):
            systematic_anchor_failures.append(anchor)

    acceptance = {
        "median_threshold_px": 4.0,
        "p90_threshold_px": 8.0,
        "per_anchor_bias_threshold_px": 4.0,
        "direction_consistency_threshold": 0.70,
        "systematic_anchor_failures": systematic_anchor_failures,
        "passed": bool(
            hold_stats["median_px"] is not None
            and hold_stats["p90_px"] is not None
            and hold_stats["median_px"] <= 4.0
            and hold_stats["p90_px"] <= 8.0
            and not systematic_anchor_failures
        ),
    }
    calibration = {
        "formatVersion": 2,
        "camera_model": "fisheye",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "K": K.astype(float).tolist(),
        "D": D.reshape(-1).astype(float).tolist(),
        "transform_convention": "X_camera = R_camera_from_o6_base @ X_o6_base + t_camera_from_o6_base",
        "o6_base_link": kinematics.hand_base_link,
        "T_camera_from_o6_base": transform_matrix(rvec, tvec).astype(float).tolist(),
        "rvec_camera_from_o6_base": rvec.astype(float).tolist(),
        "tvec_camera_from_o6_base_m": tvec.astype(float).tolist(),
        "urdf_path": str(urdf_path),
        "urdf_sha256": actual_urdf_sha256,
        "annotations_path": str(annotations_path),
        "annotations_sha256": file_sha256(annotations_path),
        "intrinsics_path": str(intrinsics_path),
        "intrinsics_source": intrinsics_source,
        "camera_mount_id": (
            contract["record"]["camera_mount_id"] if contract is not None else None
        ),
        "camera_contract": contract["record"] if contract is not None else None,
        "policy_image_contract": (
            contract["record"]["policy_image"] if contract is not None else None
        ),
        "optimizer": optimizer,
        "metrics": {"fit": fit_stats, "holdout": hold_stats},
        "diagnostics": {
            "fit": fit_diagnostics,
            "holdout": holdout_diagnostics,
            "q_state_order": [
                "thumb_cmc_pitch",
                "thumb_cmc_yaw",
                "index_mcp_pitch",
                "middle_mcp_pitch",
                "ring_mcp_pitch",
                "pinky_mcp_pitch",
            ],
        },
        "acceptance": acceptance,
        "fit_correspondences": fit_rows,
        "holdout_correspondences": hold_rows,
    }
    calibration_path = output_dir / "camera_from_o6_base.json"
    atomic_write_json(calibration_path, calibration)

    holdout_dir = output_dir / "holdout_rendered"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    for stale_render in holdout_dir.glob("episode_*_message_*.jpg"):
        stale_render.unlink()
    contact_sheet_path = output_dir / "holdout_contact_sheet.jpg"
    if contact_sheet_path.exists():
        contact_sheet_path.unlink()
    rendered_images = []
    rendered_meta = []
    for frame in payload["frames"]:
        if frame.get("split") != "holdout":
            continue
        rendered, meta = render_frame(frame, kinematics, rvec, tvec, K, D)
        filename = f"{frame['episode']}_message_{int(frame['message_index']):06d}.jpg"
        cv2.imwrite(str(holdout_dir / filename), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rendered_images.append(rendered)
        rendered_meta.append(meta)
    contact_sheet = make_contact_sheet(rendered_images[: int(args.contact_sheet_frames)])
    if contact_sheet is not None:
        cv2.imwrite(str(contact_sheet_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    atomic_write_json(
        output_dir / "holdout_render_report.json",
        {
            "camera_mount_id": calibration["camera_mount_id"],
            "camera_contract_sha256": (
                contract["record"]["sha256"] if contract is not None else None
            ),
            "acceptance": acceptance,
            "diagnostics": holdout_diagnostics,
            "frames": rendered_meta,
        },
    )

    print(f"[saved] {calibration_path}")
    print(
        "[fit] "
        f"n={fit_stats['count']} median={fit_stats['median_px']:.3f}px "
        f"p90={fit_stats['p90_px']:.3f}px"
    )
    print(
        "[holdout] "
        f"n={hold_stats['count']} median={hold_stats['median_px']:.3f}px "
        f"p90={hold_stats['p90_px']:.3f}px max={hold_stats['max_px']:.3f}px"
    )
    print(
        f"[acceptance] passed={acceptance['passed']} "
        f"systematic_anchor_failures={systematic_anchor_failures}"
    )
    print(f"[rendered] {holdout_dir}")


def run_self_test(args: argparse.Namespace) -> None:
    kinematics = O6Kinematics(args.urdf)
    rng = np.random.default_rng(42)
    K = np.array([[76.0, 0.0, 111.0], [0.0, 75.0, 112.0], [0.0, 0.0, 1.0]])
    D = np.array([-0.015, 0.004, -0.001, 0.0002], dtype=np.float64).reshape(4, 1)
    true_rvec = np.array([0.12, -0.18, 0.05], dtype=np.float64)
    true_tvec = np.array([0.02, 0.09, 0.26], dtype=np.float64)
    fit_object = []
    fit_image = []
    hold_object = []
    hold_image = []
    anchor_names = ANCHOR_SETS["tips"]
    for frame_index in range(28):
        state = rng.uniform(25.0, 245.0, size=6)
        anchors = kinematics.calibration_anchor_points(state)
        object_points = np.stack([anchors[name] for name in anchor_names])
        image_points, depth = project_fisheye(object_points, true_rvec, true_tvec, K, D)
        if np.any(depth <= 0.0):
            raise RuntimeError("Synthetic setup generated points behind camera")
        image_points += rng.normal(0.0, 0.25, size=image_points.shape)
        if (frame_index + 1) % 4 == 0:
            hold_object.extend(object_points)
            hold_image.extend(image_points)
        else:
            fit_object.extend(object_points)
            fit_image.extend(image_points)
    fit_object_array = np.asarray(fit_object, dtype=np.float64)
    fit_image_array = np.asarray(fit_image, dtype=np.float64)
    hold_object_array = np.asarray(hold_object, dtype=np.float64)
    hold_image_array = np.asarray(hold_image, dtype=np.float64)
    rvec, tvec, _ = fit_extrinsic(fit_object_array, fit_image_array, K, D)
    predicted, _ = project_fisheye(hold_object_array, rvec, tvec, K, D)
    holdout_error = np.linalg.norm(predicted - hold_image_array, axis=1)
    rotation_true, _ = cv2.Rodrigues(true_rvec)
    rotation_fit, _ = cv2.Rodrigues(rvec)
    rotation_delta, _ = cv2.Rodrigues(rotation_fit @ rotation_true.T)
    rotation_error_deg = float(np.linalg.norm(rotation_delta) * 180.0 / np.pi)
    translation_error_mm = float(np.linalg.norm(tvec - true_tvec) * 1000.0)
    stats = error_statistics(holdout_error)
    diagnostic_rows = [
        {
            "episode": "synthetic",
            "message_index": index,
            "anchor": "index_tip",
            "o6_state": [250.0, 250.0, float(40 + index * 30), 250.0, 250.0, 250.0],
            "residual_uv": [5.0, 0.0],
            "error_px": 5.0,
        }
        for index in range(5)
    ]
    diagnostic_test = correspondence_diagnostics(diagnostic_rows)
    print(json.dumps({
        "holdout": stats,
        "rotation_error_deg": rotation_error_deg,
        "translation_error_mm": translation_error_mm,
        "diagnostic_bias_norm_px": diagnostic_test["by_anchor"]["index_tip"]["bias_norm_px"],
        "diagnostic_direction_consistency": diagnostic_test["by_anchor"]["index_tip"]["direction_consistency"],
    }, indent=2))
    if stats["median_px"] is None or stats["median_px"] > 0.8:
        raise RuntimeError("Synthetic held-out reprojection test failed")
    if rotation_error_deg > 1.0 or translation_error_mm > 5.0:
        raise RuntimeError("Synthetic camera transform recovery test failed")
    if (
        abs(diagnostic_test["by_anchor"]["index_tip"]["bias_norm_px"] - 5.0) > 1e-9
        or diagnostic_test["by_anchor"]["index_tip"]["direction_consistency"] < 0.99
    ):
        raise RuntimeError("Synthetic correspondence diagnostics test failed")


def run_contract_check(args: argparse.Namespace) -> None:
    contract = load_camera_contract(Path(args.camera_contract))
    record = contract["record"]
    print(json.dumps({
        "contract_id": record["contract_id"],
        "camera_mount_id": record["camera_mount_id"],
        "camera_serial": record["camera_serial"],
        "contract_sha256": record["sha256"],
        "intrinsics_source_sha256": record["intrinsics_source_sha256"],
        "urdf_sha256": record["urdf_sha256"],
        "policy_image_size": list(contract["image_size"]),
        "K": contract["K"].astype(float).tolist(),
        "D": contract["D"].reshape(-1).astype(float).tolist(),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="select frames and open the click UI")
    annotate.add_argument("--episodes-root", default=DEFAULT_EPISODES_ROOT)
    annotate.add_argument("--episodes", required=True, help="e.g. 34-35 or 22-28,34-35")
    annotate.add_argument("--urdf", default=None)
    annotate.add_argument(
        "--camera-contract",
        default=DEFAULT_CAMERA_CONTRACT,
        help="complete YAML camera/mount/image contract; pass an empty value for legacy mode",
    )
    annotate.add_argument("--output", required=True)
    annotate.add_argument("--state-source", default="o6_measured_state", choices=["o6_measured_state", "o6_command"])
    annotate.add_argument(
        "--state-time-alignment",
        choices=["rgb", "message"],
        default="rgb",
        help="interpolate q samples to each RGB timestamp (recommended) or use message q directly",
    )
    annotate.add_argument("--num-frames", type=int, default=32)
    annotate.add_argument("--holdout-every", type=int, default=4)
    annotate.add_argument("--anchor-set", choices=sorted(ANCHOR_SETS), default="tips")
    annotate.add_argument("--max-align-ms", type=float, default=25.0)
    annotate.add_argument("--include-repeated", action="store_true")
    annotate.add_argument("--display-scale", type=int, default=3)
    annotate.add_argument("--min-points", type=int, default=4)
    annotate.add_argument("--overwrite", action="store_true")
    annotate.set_defaults(func=run_annotate)

    fit = subparsers.add_parser("fit", help="fit camera<-O6_base and render held-out frames")
    fit.add_argument("--annotations", required=True)
    fit.add_argument(
        "--camera-contract",
        default=None,
        help="override the contract recorded in annotations",
    )
    fit.add_argument(
        "--intrinsics",
        default=None,
        help="legacy exact-resolution fisheye JSON/YAML; contract is preferred",
    )
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--urdf", default=None)
    fit.add_argument("--contact-sheet-frames", type=int, default=24)
    fit.set_defaults(func=run_fit)

    self_test = subparsers.add_parser("self-test", help="synthetic FK/projection/fitting test")
    self_test.add_argument("--urdf", default=DEFAULT_URDF)
    self_test.set_defaults(func=run_self_test)

    contract_check = subparsers.add_parser(
        "contract-check", help="validate and print the active camera contract"
    )
    contract_check.add_argument("--camera-contract", default=DEFAULT_CAMERA_CONTRACT)
    contract_check.set_defaults(func=run_contract_check)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
