#!/usr/bin/env python3
"""Runtime task-object observation for objectPocketObs checkpoints.

Main inference stays in the umi204 environment.  SAM3 is isolated in a worker
Python process so its dependencies do not pollute the robot-control runtime.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_O6_CALIBRATION = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "o6_camera_calibration/mvs_DA9057802_o6_mount_v1/fit_retry1/"
    "camera_from_o6_base.json"
)
DEFAULT_WUJI_CALIBRATION = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "wuji_camera_calibration/mvs_DA9057802_wuji_mount_v1/fit_20260814_231759/"
    "camera_from_wuji_base.json"
)
# Backward-compatible name for callers that imported the old O6-only default.
DEFAULT_CALIBRATION = DEFAULT_O6_CALIBRATION
DEFAULT_GRASP_POCKET_CONFIG = (
    "/home/zjc/Desktop/human2dex/configs/grasp_pocket_v1.json"
)


def _load_hand_runtime_modules(hand_backend: str):
    scripts_real = ROOT / "scripts_real"
    if str(scripts_real) not in sys.path:
        sys.path.insert(0, str(scripts_real))
    from fisheye_canonical_views import fisheye_unit_rays_to_pixels
    from o6_grasp_view import (
        camera_points,
        compute_grasp_pocket,
        load_calibration,
        load_grasp_config,
    )
    from o6_fk21 import O6Kinematics

    if hand_backend == "linker_o6":
        kinematics_cls = O6Kinematics
        expected_state_dim = 6
        default_calibration = DEFAULT_O6_CALIBRATION
    elif hand_backend == "wuji_hand":
        from wuji_fk21 import WujiKinematics

        kinematics_cls = WujiKinematics
        expected_state_dim = 20
        default_calibration = DEFAULT_WUJI_CALIBRATION
    else:
        raise ValueError(
            "object_pocket_obs supports hand=linker_o6 or hand=wuji_hand, "
            f"got {hand_backend!r}"
        )

    return {
        "fisheye_unit_rays_to_pixels": fisheye_unit_rays_to_pixels,
        "camera_points": camera_points,
        "compute_grasp_pocket": compute_grasp_pocket,
        "load_calibration": load_calibration,
        "load_grasp_config": load_grasp_config,
        "Kinematics": kinematics_cls,
        "expected_state_dim": expected_state_dim,
        "default_calibration": default_calibration,
    }


def _obs_horizon(shape_meta, key: str, default: int) -> int:
    try:
        return int(shape_meta["obs"][key].get("horizon", default))
    except Exception:
        return int(default)


def _latest_rgb_frame(env_obs: dict, key: str) -> Optional[np.ndarray]:
    if key not in env_obs:
        return None
    arr = np.asarray(env_obs[key])
    if arr.ndim == 4 and arr.shape[-1] == 3 and arr.shape[0] > 0:
        frame = arr[-1]
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        frame = arr
    else:
        return None
    if np.issubdtype(frame.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(frame)) <= 1.5 else 1.0
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _encode_rgb_jpeg_b64(image_rgb: np.ndarray, quality: int = 90) -> str:
    bgr = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("failed to JPEG-encode RGB frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _json_write(stream, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def _squeeze_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    return arr.astype(bool, copy=False)


def _to_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _mask_stats(mask: np.ndarray) -> tuple[list[float], int, list[int]]:
    binary = _squeeze_mask(mask)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return (
        [float(xs.mean()), float(ys.mean())],
        int(len(xs)),
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    )


def _pick_sam3_object(outputs: dict[str, Any], image_area: int, spec: dict[str, Any]) -> dict[str, Any]:
    obj_ids = _to_numpy(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    masks_raw = outputs.get("out_binary_masks", [])
    masks_arr = _to_numpy(masks_raw, dtype=bool)
    masks = [] if masks_arr.size == 0 else list(masks_arr)
    scores = _to_numpy(outputs.get("out_probs", []), dtype=np.float64).reshape(-1)
    if len(scores) != len(obj_ids):
        scores = np.ones(len(obj_ids), dtype=np.float64)

    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for index, obj_id in enumerate(obj_ids):
        if index >= len(masks):
            continue
        mask = _squeeze_mask(masks[index])
        area = int(np.count_nonzero(mask))
        score = float(scores[index]) if index < len(scores) else 1.0
        if area < int(spec["min_area"]):
            continue
        if area > float(spec["max_area_ratio"]) * int(image_area):
            continue
        if score < float(spec["min_confidence"]):
            continue
        candidates.append((score, area, int(obj_id), mask))
    if not candidates:
        return {"visible": False, "error": "no_plausible_sam3_mask"}

    score, area, obj_id, mask = max(candidates)
    try:
        uv, area2, bbox = _mask_stats(mask)
    except ValueError as exc:
        return {"visible": False, "error": str(exc)}
    return {
        "visible": True,
        "uv": uv,
        "mask_area": int(area2 or area),
        "confidence": float(np.clip(score, 0.0, 1.0)),
        "bbox_xyxy": bbox,
        "object_id": int(obj_id),
        "error": None,
    }


def normalize_object_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered task-object specs shared by deployment and training.

    The legacy single-object keys remain supported.  Multi-object checkpoints
    use ``objects`` in the exact order used to build the training-side
    ``objectPocketObs`` vector.
    """
    defaults = {
        "min_confidence": float(config.get("min_confidence", 0.25)),
        "min_area": int(config.get("min_area", 12)),
        "max_area_ratio": float(config.get("max_area_ratio", 0.35)),
        "reference_scale_px": float(config.get("reference_scale_px", 37.7)),
        "reference_area_px2": float(config.get("reference_area_px2", 1213.0)),
        "confidence_tau_seconds": max(
            float(config.get("confidence_tau_seconds", 1.0)), 1e-6
        ),
        # Training confidence is min(SAM3 decayed confidence,
        # human grasp-pocket confidence).  Deployment obtains the pocket from
        # O6 geometry, so use a fixed cap measured from the training dataset to
        # keep the confidence range compatible without running the human head.
        "pocket_confidence_cap": float(
            config.get("pocket_confidence_cap", 1.0)
        ),
        "log_area_clip": float(config.get("log_area_clip", 4.0)),
    }
    raw_objects = config.get("objects")
    if raw_objects is None:
        raw_objects = [
            {
                "name": config.get("name", "object"),
                "prompt": config.get("prompt", "blue cube"),
            }
        ]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise ValueError("object_pocket_obs.objects must be a non-empty list")

    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, dict):
            raise ValueError(f"object_pocket_obs.objects[{index}] must be a mapping")
        name = str(raw.get("name", f"object_{index}")).strip()
        prompt = str(raw.get("prompt", name)).strip()
        if not name or not prompt:
            raise ValueError(f"object_pocket_obs.objects[{index}] needs name and prompt")
        if name in names:
            raise ValueError(f"duplicate object_pocket_obs object name: {name}")
        names.add(name)
        spec = {"name": name, "prompt": prompt}
        for key, default in defaults.items():
            value = raw.get(key, default)
            spec[key] = int(value) if key == "min_area" else float(value)
        if spec["min_area"] <= 0:
            raise ValueError(f"{name}.min_area must be positive")
        if not 0.0 <= spec["min_confidence"] <= 1.0:
            raise ValueError(f"{name}.min_confidence must be in [0, 1]")
        if not 0.0 < spec["max_area_ratio"] <= 1.0:
            raise ValueError(f"{name}.max_area_ratio must be in (0, 1]")
        if spec["reference_scale_px"] <= 0.0 or spec["reference_area_px2"] <= 0.0:
            raise ValueError(f"{name} reference scale/area must be positive")
        if spec["confidence_tau_seconds"] <= 0.0 or spec["log_area_clip"] <= 0.0:
            raise ValueError(f"{name} confidence tau/log-area clip must be positive")
        if not 0.0 < spec["pocket_confidence_cap"] <= 1.0:
            raise ValueError(f"{name}.pocket_confidence_cap must be in (0, 1]")
        result.append(spec)
    return result


def _sam3_worker_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAM3 JSONL worker for deployment objectPocketObs")
    parser.add_argument("--sam3-code", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version", default="sam3")
    parser.add_argument("--prompt", default="blue cube")
    parser.add_argument("--device", default="0")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args(argv)

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device))
    sam3_code = str(pathlib.Path(args.sam3_code).expanduser().resolve())
    if sam3_code not in sys.path:
        sys.path.insert(0, sam3_code)

    import torch
    from sam3 import build_sam3_predictor

    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    predictor = build_sam3_predictor(
        version=str(args.version),
        checkpoint_path=str(pathlib.Path(args.checkpoint).expanduser()),
        compile=bool(args.compile),
        async_loading_frames=False,
    )
    default_spec = {
        "name": "object",
        "prompt": str(args.prompt),
        "min_confidence": float(args.min_confidence),
        "min_area": int(args.min_area),
        "max_area_ratio": float(args.max_area_ratio),
    }
    print(json.dumps({"type": "ready", "prompt": args.prompt}, ensure_ascii=False), flush=True)

    for line in sys.stdin:
        try:
            req = json.loads(line)
        except Exception:
            continue
        if req.get("type") == "shutdown":
            break
        seq = int(req.get("seq", -1))
        t0 = time.time()
        try:
            data = base64.b64decode(str(req["image_jpeg_b64"]))
            image_bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError("worker failed to decode JPEG")
            height, width = image_bgr.shape[:2]
            requested_objects = req.get("objects")
            if not isinstance(requested_objects, list) or not requested_objects:
                requested_objects = [{**default_spec, "prompt": str(req.get("prompt", args.prompt))}]
            rows: list[dict[str, Any]] = []
            with tempfile.TemporaryDirectory(prefix="sam3_live_object_") as tmp:
                frame_path = pathlib.Path(tmp) / "000000.jpg"
                frame_path.write_bytes(data)
                for index, requested in enumerate(requested_objects):
                    spec = dict(default_spec)
                    if isinstance(requested, dict):
                        spec.update(requested)
                    spec["name"] = str(spec.get("name", f"object_{index}"))
                    spec["prompt"] = str(spec.get("prompt", spec["name"]))
                    response = predictor.handle_request(
                        {"type": "start_session", "resource_path": tmp}
                    )
                    session_id = str(response["session_id"])
                    try:
                        response = predictor.handle_request({
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": spec["prompt"],
                        })
                        row = _pick_sam3_object(
                            response.get("outputs", {}), width * height, spec
                        )
                    except Exception as exc:
                        row = {
                            "visible": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    finally:
                        predictor.handle_request({
                            "type": "close_session",
                            "session_id": session_id,
                            "run_gc_collect": False,
                        })
                    row.update({"name": spec["name"], "prompt": spec["prompt"]})
                    rows.append(row)
            payload = {
                "type": "result",
                "seq": seq,
                "width": int(width),
                "height": int(height),
                "latency_s": float(time.time() - t0),
                "objects": rows,
            }
            if len(rows) == 1:
                payload.update(rows[0])
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({
                "type": "result",
                "seq": seq,
                "visible": False,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": float(time.time() - t0),
            }, ensure_ascii=False), flush=True)
    return 0


@dataclass
class _ObjectMemory:
    uv: Optional[np.ndarray] = None
    area: float = 0.0
    confidence: float = 0.0
    updated_time: float = 0.0
    seq: int = -1
    error: Optional[str] = None


class RuntimeObjectPocketObserver:
    def __init__(self, config: dict[str, Any], shape_meta, hand_backend: str | None):
        self.config = dict(config or {})
        self.hand_backend = hand_backend
        self.enabled = bool(self.config.get("enabled", False))
        self.output_key = str(self.config.get("output_key", "objectPocketObs"))
        self.source_rgb_key = str(self.config.get("source_rgb_key", "camera0_rgb"))
        self.reference_mode = str(
            self.config.get("reference_mode", "pocket")
        ).strip().lower()
        if self.reference_mode in {"absolute", "image_center", "absolute_image"}:
            self.reference_mode = "image_center"
        elif self.reference_mode in {"pocket", "grasp_pocket", "human2dex"}:
            self.reference_mode = "pocket"
        else:
            raise ValueError(
                "object_pocket_obs.reference_mode must be pocket or image_center, "
                f"got {self.reference_mode!r}"
            )
        self.objects = normalize_object_specs(self.config)
        self.combine_pocket_confidence = bool(
            self.config.get("combine_pocket_confidence", True)
        )
        self.reference_scale_space = str(
            self.config.get("reference_scale_space", "policy")
        ).strip().lower()
        self.reference_area_space = str(
            self.config.get("reference_area_space", "policy")
        ).strip().lower()
        if self.reference_scale_space not in {"policy", "source"}:
            raise ValueError("reference_scale_space must be policy or source")
        if self.reference_area_space not in {"policy", "source"}:
            raise ValueError("reference_area_space must be policy or source")
        rgb_shapes = [
            tuple(int(x) for x in attr.get("shape", ()))
            for attr in shape_meta.get("obs", {}).values()
            if attr.get("type", "low_dim") == "rgb"
        ]
        if rgb_shapes:
            _, self.policy_image_height, self.policy_image_width = rgb_shapes[0]
        else:
            self.policy_image_width = int(self.config.get("policy_image_width", 224))
            self.policy_image_height = int(self.config.get("policy_image_height", 224))
        configured_reference_size = self.config.get(
            "reference_image_size",
            [self.config.get("source_image_width", 480),
             self.config.get("source_image_height", 480)],
        )
        if not isinstance(configured_reference_size, (list, tuple)) or len(configured_reference_size) != 2:
            raise ValueError("reference_image_size must be [width, height]")
        self.reference_image_width = float(configured_reference_size[0])
        self.reference_image_height = float(configured_reference_size[1])
        if self.reference_image_width <= 0 or self.reference_image_height <= 0:
            raise ValueError("reference_image_size must be positive")
        self._source_image_size = (
            self.reference_image_width,
            self.reference_image_height,
        )
        self.tracker_hz = max(float(self.config.get("tracker_hz", 4.0)), 1e-3)
        self.jpeg_quality = int(self.config.get("jpeg_quality", 90))
        self._next_submit_time = 0.0
        self._pending = False
        self._seq = 0
        self._minimum_seq = 0
        self._memories = {spec["name"]: _ObjectMemory() for spec in self.objects}
        self._results: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._worker_ready = False
        self._disabled_reason: Optional[str] = None

        if self.output_key not in shape_meta["obs"]:
            self.enabled = False
            self._disabled_reason = f"shape_meta missing {self.output_key}"
            return
        shape = tuple(
            int(value)
            for value in shape_meta["obs"][self.output_key].get("shape", ())
        )
        expected_shape = (5 * len(self.objects),)
        if shape != expected_shape:
            self.enabled = False
            self._disabled_reason = (
                f"{self.output_key} shape {shape} does not match configured "
                f"object order {expected_shape}"
            )
            return
        if not self.enabled:
            self._disabled_reason = "disabled by config"
            return
        if self.reference_mode == "image_center":
            # Absolute baseline: object tracking is identical, but the policy
            # observation is measured from the final RGB image center.  No
            # hand kinematics, calibration, or Pocket computation is needed.
            self.expected_state_dim = None
            self._start_worker()
            return
        if hand_backend not in {"linker_o6", "wuji_hand"}:
            self.enabled = False
            self._disabled_reason = (
                "object_pocket_obs requires hand=linker_o6 or hand=wuji_hand"
            )
            return

        modules = _load_hand_runtime_modules(hand_backend)
        self._fisheye_unit_rays_to_pixels = modules["fisheye_unit_rays_to_pixels"]
        self._camera_points = modules["camera_points"]
        self._compute_grasp_pocket = modules["compute_grasp_pocket"]
        self.expected_state_dim = int(modules["expected_state_dim"])
        self.calibration_path = pathlib.Path(
            self.config.get("calibration", modules["default_calibration"])
        ).expanduser().resolve()
        self.grasp_pocket_config_path = pathlib.Path(
            self.config.get("grasp_pocket_config", DEFAULT_GRASP_POCKET_CONFIG)
        ).expanduser().resolve()
        self.calibration = modules["load_calibration"](self.calibration_path)
        self.grasp_config = modules["load_grasp_config"](self.grasp_pocket_config_path)
        self.kinematics = modules["Kinematics"](self.calibration.urdf_path)
        self._start_worker()

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "disabledReason": self._disabled_reason,
            "handBackend": self.hand_backend,
            "referenceMode": self.reference_mode,
            "expectedStateDim": getattr(self, "expected_state_dim", None),
            "calibration": (
                str(self.calibration_path)
                if getattr(self, "calibration_path", None)
                else None
            ),
            "sourceRgbKey": self.source_rgb_key,
            "outputKey": self.output_key,
            "policyImageSize": [self.policy_image_width, self.policy_image_height],
            "imageCenter": [
                self.policy_image_width * 0.5,
                self.policy_image_height * 0.5,
            ],
            "referenceScaleSpace": self.reference_scale_space,
            "referenceAreaSpace": self.reference_area_space,
            "trackerHz": self.tracker_hz,
            "objectOrder": [spec["name"] for spec in self.objects],
            "dimensionPerObject": 5,
            "objects": [dict(spec) for spec in self.objects],
        }

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and proc.poll() is None:
                _json_write(proc.stdin, {"type": "shutdown"})
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    def reset(self) -> None:
        """Clear cross-episode object memory without restarting the SAM3 model."""
        self._memories = {spec["name"]: _ObjectMemory() for spec in self.objects}
        self._minimum_seq = self._seq
        self._next_submit_time = 0.0

    def _start_worker(self) -> None:
        worker_python = str(self.config.get("worker_python", "/home/zjc/miniconda3/envs/sam3/bin/python"))
        sam3_code = str(self.config.get("sam3_code", "/home/zjc/Desktop/human2dex/sam/sam3-code"))
        checkpoint = str(self.config.get("checkpoint", "/home/zjc/Desktop/human2dex/sam/sam3/sam3.pt"))
        device = str(self.config.get("device", "0"))
        cmd = [
            worker_python,
            str(pathlib.Path(__file__).resolve()),
            "--sam3-worker",
            "--sam3-code", sam3_code,
            "--checkpoint", checkpoint,
            "--version", str(self.config.get("version", "sam3")),
            "--prompt", self.objects[0]["prompt"],
            "--device", device,
            "--min-confidence", str(float(self.objects[0]["min_confidence"])),
            "--min-area", str(int(self.objects[0]["min_area"])),
            "--max-area-ratio", str(float(self.objects[0]["max_area_ratio"])),
            "--jpeg-quality", str(self.jpeg_quality),
        ]
        if bool(self.config.get("compile", False)):
            cmd.append("--compile")
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.enabled = False
            self._disabled_reason = f"failed to start SAM3 worker: {exc}"
            print(f"[WARN] object_pocket_obs disabled: {self._disabled_reason}")
            return
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        print(
            "[INFO] object_pocket_obs enabled: "
            f"hand={self.hand_backend}, "
            f"objects={[spec['name'] for spec in self.objects]}, "
            f"tracker_hz={self.tracker_hz}, "
            f"worker={worker_python}, device={device}"
        )

    def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                print(f"[SAM3 stdout] {text}")
                continue
            if payload.get("type") == "ready":
                self._worker_ready = True
                print("[INFO] SAM3 object worker ready")
            elif payload.get("type") == "result":
                self._results.put(payload)

    def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            text = line.rstrip()
            if text:
                print(f"[SAM3 stderr] {text}")

    def _drain_results(self) -> None:
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            self._pending = False
            seq = int(item.get("seq", -1))
            if seq < self._minimum_seq:
                continue
            rows = item.get("objects")
            if not isinstance(rows, list):
                rows = [{**item, "name": self.objects[0]["name"]}]
            updated_time = time.monotonic()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", ""))
                memory = self._memories.get(name)
                if memory is None or seq < memory.seq:
                    continue
                if bool(row.get("visible", False)):
                    uv = np.asarray(
                        row.get("uv", [np.nan, np.nan]), dtype=np.float64
                    ).reshape(2)
                    area = float(row.get("mask_area", 0.0) or 0.0)
                    conf = float(row.get("confidence", 0.0) or 0.0)
                    if np.all(np.isfinite(uv)) and area > 0:
                        self._memories[name] = _ObjectMemory(
                            uv=uv,
                            area=area,
                            confidence=float(np.clip(conf, 0.0, 1.0)),
                            updated_time=updated_time,
                            seq=seq,
                            error=None,
                        )
                        continue
                memory.seq = seq
                memory.error = str(row.get("error") or "object_missing")

    def _submit_if_due(self, env_obs: dict) -> None:
        if not self.enabled or self._proc is None or self._proc.poll() is not None:
            return
        now = time.monotonic()
        if self._pending or now < self._next_submit_time:
            return
        image = _latest_rgb_frame(env_obs, self.source_rgb_key)
        if image is None:
            return
        self._source_image_size = (float(image.shape[1]), float(image.shape[0]))
        try:
            payload = {
                "type": "frame",
                "seq": int(self._seq),
                "objects": [
                    {
                        "name": spec["name"],
                        "prompt": spec["prompt"],
                        "min_confidence": spec["min_confidence"],
                        "min_area": spec["min_area"],
                        "max_area_ratio": spec["max_area_ratio"],
                    }
                    for spec in self.objects
                ],
                "image_jpeg_b64": _encode_rgb_jpeg_b64(image, quality=self.jpeg_quality),
            }
            assert self._proc.stdin is not None
            _json_write(self._proc.stdin, payload)
            self._pending = True
            self._seq += 1
            self._next_submit_time = now + 1.0 / self.tracker_hz
        except Exception as exc:
            self.enabled = False
            self._disabled_reason = f"SAM3 worker write failed: {exc}"
            print(f"[WARN] object_pocket_obs disabled: {self._disabled_reason}")

    def _pocket_uv(self, hand_state) -> tuple[Optional[np.ndarray], float]:
        if hand_state is None:
            return None, 0.0
        state = np.asarray(hand_state, dtype=np.float64).reshape(-1)
        if (
            state.shape != (self.expected_state_dim,)
            or not np.all(np.isfinite(state))
        ):
            return None, 0.0
        try:
            points = self.kinematics.points21(state)
            pocket = self._compute_grasp_pocket(
                points,
                a=float(self.grasp_config["a"]),
                b=float(self.grasp_config["b"]),
                normal_sign=float(self.grasp_config["normal_sign"]),
            )
            pocket_camera = self._camera_points(pocket.point3d[None, :], self.calibration)[0]
            if not np.all(np.isfinite(pocket_camera)) or float(pocket_camera[2]) <= 1e-6:
                return None, 0.0
            uv = self._fisheye_unit_rays_to_pixels(pocket_camera[None, :], self.calibration.K, self.calibration.D)[0]
            if not np.all(np.isfinite(uv)):
                return None, 0.0
            return np.asarray(uv, dtype=np.float64).reshape(2), 1.0
        except Exception:
            return None, 0.0

    def _current_obs_vector(self, pocket_uv: Optional[np.ndarray], pocket_conf: float) -> np.ndarray:
        blocks: list[np.ndarray] = []
        now = time.monotonic()
        source_w, source_h = self._source_image_size
        sx = float(self.policy_image_width) / max(source_w, 1.0)
        sy = float(self.policy_image_height) / max(source_h, 1.0)
        center_u = float(self.policy_image_width) * 0.5
        center_v = float(self.policy_image_height) * 0.5
        if self.reference_scale_space == "policy":
            scale_x_factor = 1.0
            scale_y_factor = 1.0
        else:
            scale_x_factor = float(self.policy_image_width) / self.reference_image_width
            scale_y_factor = float(self.policy_image_height) / self.reference_image_height
        if self.reference_area_space == "policy":
            area_factor = sx * sy
        else:
            area_factor = 1.0

        for spec in self.objects:
            memory = self._memories[spec["name"]]
            if memory.uv is None:
                blocks.append(np.zeros(5, dtype=np.float32))
                continue
            age = max(0.0, now - float(memory.updated_time))
            conf = float(memory.confidence) * math.exp(
                -age / float(spec["confidence_tau_seconds"])
            )
            conf = min(conf, float(spec["pocket_confidence_cap"]))
            if self.combine_pocket_confidence:
                conf = min(conf, float(np.clip(pocket_conf, 0.0, 1.0)))

            if self.reference_mode == "image_center":
                uv_final = np.asarray(
                    [float(memory.uv[0]) * sx, float(memory.uv[1]) * sy],
                    dtype=np.float64,
                )
                scale_x = float(spec["reference_scale_px"]) * scale_x_factor
                scale_y = float(spec["reference_scale_px"]) * scale_y_factor
                dx = float((uv_final[0] - center_u) / scale_x)
                dy = float((uv_final[1] - center_v) / scale_y)
                area_value = float(memory.area) * area_factor
                reference_area = float(spec["reference_area_px2"])
            else:
                if pocket_uv is None:
                    blocks.append(np.zeros(5, dtype=np.float32))
                    continue
                dx = float(
                    (memory.uv[0] - pocket_uv[0])
                    / float(spec["reference_scale_px"])
                )
                dy = float(
                    (memory.uv[1] - pocket_uv[1])
                    / float(spec["reference_scale_px"])
                )
                area_value = float(memory.area)
                reference_area = float(spec["reference_area_px2"])

            log_area = math.log(max(area_value, 1.0) / reference_area)
            log_area = float(
                np.clip(log_area, -spec["log_area_clip"], spec["log_area_clip"])
            )
            blocks.append(
                np.asarray([dx, dy, log_area, conf, 1.0], dtype=np.float32)
            )
        return np.concatenate(blocks, axis=0)

    def apply(self, env_obs: dict, hand_state) -> dict:
        if self.output_key not in self.config.get("_shape_obs_keys", {self.output_key}):
            return env_obs
        self._drain_results()
        self._submit_if_due(env_obs)
        if self.reference_mode == "pocket":
            pocket_uv, pocket_conf = self._pocket_uv(hand_state)
        else:
            pocket_uv, pocket_conf = None, 1.0
        vector = self._current_obs_vector(pocket_uv, pocket_conf)
        horizon = _obs_horizon(
            self._shape_meta,
            self.output_key,
            len(np.asarray(env_obs.get("timestamp", [0]))),
        ) if hasattr(self, "_shape_meta") else len(np.asarray(env_obs.get("timestamp", [0])))
        out = dict(env_obs)
        out[self.output_key] = np.repeat(vector[None, :], max(1, int(horizon)), axis=0)
        debug_objects = []
        for index, spec in enumerate(self.objects):
            memory = self._memories[spec["name"]]
            block = vector[5 * index: 5 * (index + 1)]
            debug_objects.append({
                "name": spec["name"],
                "prompt": spec["prompt"],
                "uv": None if memory.uv is None else memory.uv.tolist(),
                "area": float(memory.area),
                "confidence": float(block[3]),
                "valid": bool(block[4] > 0.5),
                "error": memory.error,
            })
        out["objectPocketObsDebug"] = {
            "objectOrder": [spec["name"] for spec in self.objects],
            "objects": debug_objects,
            "pocket_uv": None if pocket_uv is None else pocket_uv.tolist(),
        }
        return out

    def bind_shape_meta(self, shape_meta) -> "RuntimeObjectPocketObserver":
        self._shape_meta = shape_meta
        self.config["_shape_obs_keys"] = {str(k) for k in shape_meta["obs"].keys()}
        return self


def build_object_pocket_observer(cfg: dict, hand_backend: str | None, shape_meta):
    config = dict((cfg.get("object_pocket_obs", {}) or {}))
    observer = RuntimeObjectPocketObserver(config, shape_meta, hand_backend).bind_shape_meta(shape_meta)
    if observer.enabled:
        return observer
    if observer.output_key in shape_meta["obs"]:
        print(f"[WARN] object_pocket_obs not enabled; {observer.output_key} will use invalid zeros ({observer._disabled_reason})")
    return None


if __name__ == "__main__":
    if "--sam3-worker" in sys.argv:
        args = [item for item in sys.argv[1:] if item != "--sam3-worker"]
        raise SystemExit(_sam3_worker_main(args))
    raise SystemExit("This module is imported by real_inference_runner.py; use --sam3-worker internally.")
