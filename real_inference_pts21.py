#!/usr/bin/env python3
"""21x3 MANO keypoint action adapter for real Franka inference."""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import select
import subprocess
import sys
from typing import Optional

import numpy as np

from real_inference_actions import align_recording_hand_commands, patch_hand_obs
from real_inference_config import HAND_BACKENDS


DEFAULT_RETARGET_PYTHON = "/home/zjc/miniconda3/envs/l515_mvs310/bin/python"
DEFAULT_DEXUMI_ROOT = "/home/zjc/Desktop/human2dex"
PTS21_ACTION_DIM = 72
PTS21_FLAT_DIM = 63


def _as_points_sequence(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 2 and arr.shape[-1] == PTS21_FLAT_DIM:
        arr = arr.reshape(-1, 21, 3)
    if arr.ndim != 3 or arr.shape[-2:] != (21, 3):
        raise ValueError(f"{name} must be (H,63) or (H,21,3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")

    # The checkpoint was trained on wrist-local MANO points. Keep the runtime
    # representation local even if a sampled wrist channel drifts slightly.
    arr = arr - arr[:, :1, :]
    return arr.astype(np.float32, copy=False)


def _policy_pts21_mean(policy) -> np.ndarray:
    normalizer = getattr(policy, "normalizer", None)
    if normalizer is None or not hasattr(normalizer, "get_input_stats"):
        raise RuntimeError("Policy does not expose normalizer input statistics")
    stats = normalizer.get_input_stats()
    try:
        mean = stats["robot0_gripper_width"]["mean"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "Checkpoint is missing robot0_gripper_width normalizer statistics"
        ) from exc
    if hasattr(mean, "detach"):
        mean = mean.detach().cpu().numpy()
    points = _as_points_sequence(mean, name="checkpoint pts21 observation mean")
    if len(points) != 1:
        raise RuntimeError(f"Expected one pts21 mean vector, got {points.shape}")
    return points[0]


class _RetargetWorker:
    """Persistent retargeting subprocess using the existing DexUMI environment."""

    def __init__(self, backend: str, config: dict):
        if backend not in HAND_BACKENDS:
            raise ValueError(f"Unsupported hand backend: {backend!r}")
        self.backend = backend
        self.timeout_s = float(config.get("worker_timeout_s", 30.0))
        worker_python = str(config.get("worker_python", DEFAULT_RETARGET_PYTHON))
        dexumi_root = str(config.get("dexumi_root", DEFAULT_DEXUMI_ROOT))
        hand_side = str(config.get("hand_side", "right"))
        wuji_maxeval = int(config.get("wuji_maxeval", 40))
        command = [
            worker_python,
            str(pathlib.Path(__file__).resolve()),
            "--worker",
            "--backend", backend,
            "--dexumi-root", dexumi_root,
            "--hand-side", hand_side,
            "--wuji-maxeval", str(wuji_maxeval),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        ready = self._read_response("startup")
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"Unexpected pts21 retarget worker response: {ready}")

    def _read_response(self, operation: str) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("pts21 retarget worker stdout is unavailable")
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout_s)
        if not ready:
            self.process.kill()
            raise TimeoutError(
                f"pts21 retarget worker timed out during {operation} "
                f"after {self.timeout_s:.1f}s"
            )
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(
                f"pts21 retarget worker exited during {operation}, returncode={code}"
            )
        response = json.loads(line)
        if not response.get("ok", False):
            raise RuntimeError(
                f"pts21 retarget worker failed during {operation}: "
                f"{response.get('error', response)}"
            )
        return response

    def _request(self, payload: dict, operation: str) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"pts21 retarget worker is not running, returncode={self.process.returncode}"
            )
        if self.process.stdin is None:
            raise RuntimeError("pts21 retarget worker stdin is unavailable")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        return self._read_response(operation)

    def retarget(self, points: np.ndarray) -> np.ndarray:
        sequence = _as_points_sequence(points, name="policy pts21 action")
        response = self._request(
            {"op": "retarget", "points": sequence.tolist()},
            "retarget",
        )
        commands = np.asarray(response["commands"])
        expected_dim = 6 if self.backend == "linker_o6" else 20
        if commands.shape != (len(sequence), expected_dim):
            raise RuntimeError(
                f"Retarget worker returned {commands.shape}, "
                f"expected {(len(sequence), expected_dim)}"
            )
        if not np.all(np.isfinite(commands)):
            raise RuntimeError("Retarget worker returned NaN or Inf")
        if self.backend == "linker_o6":
            return np.clip(np.rint(commands), 0, 255).astype(np.uint8)
        return commands.astype(np.float64, copy=False)

    def reset(self) -> None:
        self._request({"op": "reset"}, "reset")

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            self._request({"op": "close"}, "close")
        except Exception:
            process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


class Pts21ActionAdapter:
    """Split 72D actions and retarget their 21x3 hand tail at deployment."""

    def __init__(self, *, policy, hand_backend: Optional[str], config: dict):
        self.hand_backend = hand_backend
        self.config = dict(config or {})
        self.initial_points = _policy_pts21_mean(policy)
        self.last_points = self.initial_points.copy()
        self.worker = None
        if hand_backend is not None:
            self.worker = _RetargetWorker(hand_backend, self.config)
        print(
            "[INFO] pts21 action adapter enabled: "
            f"72D = 9D arm + 21x3 MANO points, hand={hand_backend or 'none'}"
        )
        print(
            "[INFO] pts21 observation source = last executed target points; "
            "episode initialization = checkpoint training mean"
        )

    def patch_observation(self, env_obs: dict, shape_meta) -> dict:
        return patch_hand_obs(env_obs, self.last_points.reshape(-1), shape_meta)

    @staticmethod
    def split_arm(pred: np.ndarray, gripper_width: float) -> np.ndarray:
        arr = np.asarray(pred, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[-1] != PTS21_ACTION_DIM:
            raise ValueError(f"pts21 policy action must be (H,72), got {arr.shape}")
        dummy_gripper = np.full((len(arr), 1), gripper_width, dtype=np.float32)
        return np.concatenate([arr[:, :9], dummy_gripper], axis=-1)

    def split(self, pred: np.ndarray, gripper_width: float):
        arr = np.asarray(pred, dtype=np.float32)
        arm_action = self.split_arm(arr, gripper_width)
        points = _as_points_sequence(arr[:, 9:72], name="policy pts21 action")
        flat_points = points.reshape(len(points), PTS21_FLAT_DIM)
        hand_actions = None if self.worker is None else self.worker.retarget(points)
        return arm_action, hand_actions, flat_points

    @staticmethod
    def align_points(
            points: Optional[np.ndarray],
            kept_mask: np.ndarray,
            submitted_timestamps: np.ndarray) -> Optional[np.ndarray]:
        _, scheduled = align_recording_hand_commands(
            points,
            kept_mask,
            submitted_timestamps,
        )
        return scheduled

    def commit_executed(
            self,
            scheduled_points: Optional[np.ndarray],
            executed_count: int) -> None:
        if scheduled_points is None or executed_count <= 0:
            return
        index = min(int(executed_count), len(scheduled_points)) - 1
        self.last_points = _as_points_sequence(
            scheduled_points[index],
            name="executed pts21 target",
        )[0]

    def reset(self) -> None:
        self.last_points = self.initial_points.copy()
        if self.worker is not None:
            self.worker.reset()

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


def build_pts21_action_adapter(
        cfg: dict,
        *,
        policy,
        action_dim: int,
        hand_backend: Optional[str]) -> Optional[Pts21ActionAdapter]:
    mode = str(cfg.get("hand_action_mode", "direct_command")).strip().lower()
    if mode in ("", "direct", "direct_command", "robot_command"):
        return None
    if mode not in ("pts21", "pts21_mano", "mono_keypoints"):
        raise ValueError(f"Unsupported hand_action_mode={mode!r}")
    if int(action_dim) != PTS21_ACTION_DIM:
        raise ValueError(
            f"hand_action_mode=pts21_mano requires action_dim=72, got {action_dim}"
        )
    return Pts21ActionAdapter(
        policy=policy,
        hand_backend=hand_backend,
        config=cfg.get("pts21_retarget", {}),
    )


def _reset_wuji_backend(backend) -> None:
    lp_filter = getattr(backend, "lp_filter", None)
    if lp_filter is not None and hasattr(lp_filter, "reset"):
        lp_filter.reset()
    optimizer = getattr(backend, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "last_qpos"):
        optimizer.last_qpos = None


def _retarget_wuji_mano(backend, points: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(points, dtype=np.float64).reshape(21, 3).copy()
    rotation_xyz = getattr(backend, "rotation_xyz", {}) or {}
    if any(float(rotation_xyz.get(axis, 0.0)) != 0.0 for axis in ("x", "y", "z")):
        keypoints = backend._apply_rotation(keypoints)
    if getattr(backend, "_has_offset", False):
        keypoints = backend._apply_offset(keypoints)
    qpos = np.asarray(backend.optimizer.solve(keypoints), dtype=np.float32).reshape(-1)
    qpos = np.asarray(backend.lp_filter.next(qpos), dtype=np.float32).reshape(-1)
    if qpos.shape != (20,) or not np.all(np.isfinite(qpos)):
        raise RuntimeError(f"Wuji retargeter returned invalid qpos: {qpos.shape}")
    return qpos


def _worker_main(args) -> int:
    dexumi_root = pathlib.Path(args.dexumi_root).expanduser().resolve()
    teleop_dir = dexumi_root / "teleop"
    if not teleop_dir.is_dir():
        raise FileNotFoundError(f"DexUMI teleop directory not found: {teleop_dir}")
    sys.path.insert(0, str(teleop_dir))
    sys.path.insert(0, str(dexumi_root))

    with contextlib.redirect_stdout(sys.stderr):
        if args.backend == "linker_o6":
            from episode_io import angles_rad_to_cmd
            from retargeter import LinkerO6Retargeter

            backend = LinkerO6Retargeter(hand=args.hand_side)

            def convert_one(points):
                return angles_rad_to_cmd(backend.retarget(points))

            def reset_backend():
                backend.reset()
        else:
            from wuji_retargeting import Retargeter

            yaml_path = dexumi_root / "wuji_retargeting/config/adaptive_analytical_pico.yaml"
            backend = Retargeter.from_yaml(str(yaml_path), hand_side=args.hand_side)
            backend.optimizer.opt.set_maxeval(int(args.wuji_maxeval))

            def convert_one(points):
                return _retarget_wuji_mano(backend, points)

            def reset_backend():
                _reset_wuji_backend(backend)

    print(json.dumps({"ok": True, "status": "ready", "backend": args.backend}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "retarget":
                points = _as_points_sequence(request.get("points"), name="worker points")
                with contextlib.redirect_stdout(sys.stderr):
                    commands = np.stack([convert_one(item) for item in points], axis=0)
                response = {"ok": True, "commands": commands.tolist()}
            elif operation == "reset":
                with contextlib.redirect_stdout(sys.stderr):
                    reset_backend()
                response = {"ok": True, "status": "reset"}
            elif operation == "close":
                print(json.dumps({"ok": True, "status": "closed"}), flush=True)
                return 0
            else:
                raise ValueError(f"Unknown worker operation: {operation!r}")
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", choices=HAND_BACKENDS)
    parser.add_argument("--dexumi-root", default=DEFAULT_DEXUMI_ROOT)
    parser.add_argument("--hand-side", choices=["right", "left"], default="right")
    parser.add_argument("--wuji-maxeval", type=int, default=40)
    return parser


if __name__ == "__main__":
    worker_args = _build_worker_parser().parse_args()
    if not worker_args.worker or worker_args.backend is None:
        raise SystemExit("This module is an internal pts21 retarget worker")
    raise SystemExit(_worker_main(worker_args))
