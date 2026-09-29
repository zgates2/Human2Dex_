#!/usr/bin/env python3
"""Offline fisheye-corrected local views centered on O6 grasp_pocket_v1.

The generator is preview-only.  It reads saved inference episode PKLs, computes
O6 FK points and the grasp-pocket anchor, and renders a local camera-plane view
by projecting every output pixel through the calibrated fisheye model.  It does
not connect to a robot, load a policy, or modify source PKLs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_REAL = ROOT / "scripts_real"
if str(SCRIPTS_REAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_REAL))

from o6_grasp_pocket_overlay import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_EPISODES_ROOT,
    O6Kinematics,
    compute_grasp_pocket,
    draw_pocket,
    iter_episode_frames,
    load_calibration,
    parse_episode_spec,
    uv_valid,
)


DEFAULT_GRASP_POCKET_CONFIG = ROOT / "configs" / "grasp_pocket_v1.json"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_size(value: str) -> tuple[int, int]:
    text = str(value).lower().replace(" ", "")
    if "x" not in text:
        raise ValueError(f"Expected WIDTHxHEIGHT, got {value!r}")
    width, height = [int(part) for part in text.split("x", 1)]
    if width < 2 or height < 2:
        raise ValueError("Output size must be at least 2x2")
    return width, height


def load_grasp_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("name", "")) != "grasp_pocket_v1":
        raise ValueError(f"Expected grasp_pocket_v1 config, got {payload.get('name')!r}")
    for key in ("a", "b", "normal_sign"):
        value = float(payload[key])
        if not math.isfinite(value):
            raise ValueError(f"Invalid grasp config value {key}={value}")
    return payload


def camera_points(points_o6: np.ndarray, calibration) -> np.ndarray:
    points = np.asarray(points_o6, dtype=np.float64).reshape(-1, 3)
    rotation, _ = cv2.Rodrigues(np.asarray(calibration.rvec, dtype=np.float64).reshape(3, 1))
    translation = np.asarray(calibration.tvec, dtype=np.float64).reshape(1, 3)
    return (rotation @ points.T).T + translation


def project_camera_points(points_camera: np.ndarray, calibration) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    zero = np.zeros((3, 1), dtype=np.float64)
    uv, _ = cv2.fisheye.projectPoints(
        points.reshape(-1, 1, 3), zero, zero, calibration.K, calibration.D
    )
    return uv.reshape(-1, 2), points[:, 2].copy()


def _border_mode(name: str) -> int:
    modes = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "reflect101": cv2.BORDER_REFLECT_101,
    }
    key = str(name).lower()
    if key not in modes:
        raise ValueError(f"Unknown border mode {name!r}; use {sorted(modes)}")
    return modes[key]


def render_grasp_view(
    image_bgr: np.ndarray,
    pocket_camera: np.ndarray,
    palm_width_m: float,
    calibration,
    *,
    output_size: tuple[int, int],
    width_in_palm: float,
    height_in_palm: float,
    rotation_deg: float,
    border_mode: str,
) -> np.ndarray:
    """Render a square/rectangular camera-plane patch through fisheye projection.

    The output center is the 3D pocket.  The local plane is parallel to the
    camera image plane at the pocket depth; x/y are camera coordinates.  Thus
    radial fisheye distortion is applied by the calibrated camera model instead
    of being approximated by an ordinary pixel crop.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got {image.shape}")
    out_w, out_h = output_size
    pocket = np.asarray(pocket_camera, dtype=np.float64).reshape(3)
    palm_width = float(palm_width_m)
    if not np.all(np.isfinite(pocket)) or pocket[2] <= 1e-6:
        raise ValueError("Pocket must be finite and in front of the camera")
    if not math.isfinite(palm_width) or palm_width <= 1e-6:
        raise ValueError("Invalid palm width")

    view_width = palm_width * float(width_in_palm)
    view_height = palm_width * float(height_in_palm)
    x = ((np.arange(out_w, dtype=np.float64) + 0.5) / out_w - 0.5) * view_width
    y = ((np.arange(out_h, dtype=np.float64) + 0.5) / out_h - 0.5) * view_height
    xx, yy = np.meshgrid(x, y)
    theta = math.radians(float(rotation_deg))
    c, s = math.cos(theta), math.sin(theta)
    dx = c * xx - s * yy
    dy = s * xx + c * yy
    points_camera = np.empty((out_h * out_w, 3), dtype=np.float64)
    points_camera[:, 0] = pocket[0] + dx.reshape(-1)
    points_camera[:, 1] = pocket[1] + dy.reshape(-1)
    points_camera[:, 2] = pocket[2]
    source_uv, _ = project_camera_points(points_camera, calibration)
    map_x = source_uv[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = source_uv[:, 1].reshape(out_h, out_w).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=_border_mode(border_mode),
    )


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (24, 24, 24), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run(args: argparse.Namespace) -> int:
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_size = parse_size(args.output_size)
    grasp_cfg_path = Path(args.grasp_pocket_config).expanduser().resolve()
    grasp_cfg = load_grasp_config(grasp_cfg_path)
    calibration = load_calibration(Path(args.calibration))
    kinematics = O6Kinematics(calibration.urdf_path)
    episodes = parse_episode_spec(args.episodes)

    summary: dict[str, Any] = {
        "format": "o6_grasp_view_preview_v1",
        "grasp_pocket_config": str(grasp_cfg_path),
        "calibration": str(calibration.path),
        "episodes_root": str(episodes_root),
        "episodes": episodes,
        "state_field": args.state_field,
        "output_size": list(output_size),
        "width_in_palm": float(args.width_in_palm),
        "height_in_palm": float(args.height_in_palm),
        "rotation_deg": float(args.rotation_deg),
        "ema_alpha": float(args.ema_alpha),
        "frames": [],
        "counts": {"processed": 0, "valid": 0, "errors": 0},
    }
    last_pocket: dict[int, np.ndarray] = {}
    last_width: dict[int, float] = {}
    for episode, frame_index, message, state, image_path in iter_episode_frames(
        episodes_root,
        episodes,
        state_field=args.state_field,
        stride=args.stride,
        max_frames_per_episode=args.max_frames_per_episode,
        include_repeated_rgb=args.include_repeated_rgb,
    ):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            summary["counts"]["errors"] += 1
            continue
        record: dict[str, Any] = {"episode": f"episode_{episode:04d}", "frame_index": int(frame_index)}
        try:
            points_o6 = kinematics.points21(state)
            pocket = compute_grasp_pocket(
                points_o6,
                a=float(grasp_cfg["a"]),
                b=float(grasp_cfg["b"]),
                normal_sign=float(grasp_cfg["normal_sign"]),
            )
            pocket_camera = camera_points(pocket.point3d[None, :], calibration)[0]
            alpha = float(args.ema_alpha)
            if episode not in last_pocket or alpha >= 1.0:
                smooth_pocket = pocket_camera.copy()
                smooth_width = float(pocket.palm_width)
            else:
                alpha = max(0.0, min(1.0, alpha))
                smooth_pocket = (1.0 - alpha) * last_pocket[episode] + alpha * pocket_camera
                smooth_width = (1.0 - alpha) * last_width[episode] + alpha * float(pocket.palm_width)
            last_pocket[episode] = smooth_pocket
            last_width[episode] = smooth_width
            raw_uv, raw_depth = project_camera_points(pocket_camera[None, :], calibration)
            smooth_uv, smooth_depth = project_camera_points(smooth_pocket[None, :], calibration)
            valid = uv_valid(smooth_uv[0], float(smooth_depth[0]), image.shape[1], image.shape[0])
            if not valid:
                raise ValueError("smoothed pocket projects outside source image")
            grasp_view = render_grasp_view(
                image,
                smooth_pocket,
                smooth_width,
                calibration,
                output_size=output_size,
                width_in_palm=args.width_in_palm,
                height_in_palm=args.height_in_palm,
                rotation_deg=args.rotation_deg,
                border_mode=args.border_mode,
            )
            raw_overlay = draw_pocket(
                image,
                raw_uv[0],
                smooth_uv[0],
                raw_valid=uv_valid(raw_uv[0], float(raw_depth[0]), image.shape[1], image.shape[0]),
                smooth_valid=valid,
                radius=args.pocket_radius,
            )
            raw_overlay = label(raw_overlay, f"RAW ep{episode:04d} f{frame_index:04d} pocket_v1")
            view_labeled = label(grasp_view, f"GRASP_VIEW {args.width_in_palm:.2f}x{args.height_in_palm:.2f} palm")
            episode_view_dir = output_dir / "grasp_view" / f"episode_{episode:04d}"
            episode_compare_dir = output_dir / "compare" / f"episode_{episode:04d}"
            episode_view_dir.mkdir(parents=True, exist_ok=True)
            episode_compare_dir.mkdir(parents=True, exist_ok=True)
            name = f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(episode_view_dir / name), view_labeled, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            comparison = np.concatenate([raw_overlay, view_labeled], axis=1)
            cv2.imwrite(str(episode_compare_dir / name), comparison, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            record.update({
                "image": str(image_path),
                "grasp_view": str(episode_view_dir / name),
                "comparison": str(episode_compare_dir / name),
                "pocket_uv_raw": raw_uv[0].tolist(),
                "pocket_uv_smooth": smooth_uv[0].tolist(),
                "pocket_depth_m": float(smooth_depth[0]),
                "palm_width_m": float(smooth_width),
                "valid": True,
            })
            summary["counts"]["processed"] += 1
            summary["counts"]["valid"] += 1
        except Exception as exc:
            record["error"] = repr(exc)
            summary["counts"]["errors"] += 1
        summary["frames"].append(record)

    atomic_write_json(output_dir / "summary.json", summary)
    (output_dir / "grasp_pocket_config_used.json").write_text(
        json.dumps(grasp_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["counts"], ensure_ascii=False))
    print(f"output_dir={output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", default=DEFAULT_EPISODES_ROOT)
    parser.add_argument("--episodes", required=True, help="Episode list, e.g. 92,109")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--grasp-pocket-config", default=str(DEFAULT_GRASP_POCKET_CONFIG))
    parser.add_argument("--state-field", default="o6_measured_state")
    parser.add_argument("--output-size", default="224x224")
    parser.add_argument("--width-in-palm", type=float, default=2.5)
    parser.add_argument("--height-in-palm", type=float, default=2.5)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--border-mode", default="reflect101")
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--pocket-radius", type=float, default=7.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--include-repeated-rgb", action="store_true")
    parser.set_defaults(func=run)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
