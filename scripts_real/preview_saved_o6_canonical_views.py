#!/usr/bin/env python3
"""Offline G/L audit from an already saved O6 inference episode.

This does not open a camera or send any robot command.  It recomputes the
canonical views from each saved raw RGB and its recorded O6 measured state,
then writes raw|G|L contact sheets to a new output directory.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_inference_canonical_views import O6CanonicalViewsRenderer
from scripts_real.fisheye_canonical_views import fisheye_unit_rays_to_pixels


def _read_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"not an inference episode PKL: {path}")
    return value


def _episode_pickle(episode: Path) -> Path:
    paths = sorted(episode.glob("*.pkl"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one *.pkl in {episode}, found {len(paths)}")
    return paths[0]


def _draw_raw(image: np.ndarray, center_uv: np.ndarray, frame: int) -> np.ndarray:
    result = image.copy()
    center = tuple(np.rint(center_uv).astype(int))
    cv2.circle(result, center, 7, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.putText(result, f"raw #{frame:06d}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=80)
    args = parser.parse_args()
    if args.stride < 1 or args.max_frames < 1:
        raise ValueError("--stride and --max-frames must be >= 1")

    episode = args.episode.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    config = yaml.safe_load(args.config.expanduser().read_text(encoding="utf-8"))
    view_cfg = dict((config or {}).get("canonical_views", {}) or {})
    if not bool(view_cfg.get("enabled", False)):
        raise ValueError("canonical_views.enabled must be true")
    renderer = O6CanonicalViewsRenderer(view_cfg)
    payload = _read_pickle(_episode_pickle(episode))

    report = {"episode": str(episode), "frames": [], "renderer": renderer.metadata()}
    saved = 0
    for frame, message in enumerate(payload["messages"]):
        if saved >= args.max_frames:
            break
        if frame % args.stride:
            continue
        if not isinstance(message, dict) or message.get("o6_measured_state") is None:
            continue
        image_path = episode / str(message.get("rgbImage", ""))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        state = np.asarray(message["o6_measured_state"], dtype=np.float64).reshape(6)
        rendered = renderer.apply({renderer.source_rgb_key: image[None]}, state)
        global_view = np.asarray(rendered[renderer.global_output_key][0])
        local_view = np.asarray(rendered[renderer.local_output_key][0])
        center_uv = fisheye_unit_rays_to_pixels(
            renderer._last_pocket_camera[None, :], renderer.calibration.K, renderer.calibration.D
        )[0]
        raw = _draw_raw(image, center_uv, frame)
        raw = cv2.resize(raw, renderer.global_size, interpolation=cv2.INTER_AREA)
        montage = np.concatenate((raw, global_view, local_view), axis=1)
        target = output / f"frame_{frame:06d}.jpg"
        if not cv2.imwrite(str(target), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise IOError(f"could not write {target}")
        report["frames"].append({
            "frame": frame,
            "output": target.name,
            "pocketUv": center_uv.tolist(),
            "globalBlackRatio": float(np.mean(np.all(global_view == 0, axis=-1))),
            "localBlackRatio": float(np.mean(np.all(local_view == 0, axis=-1))),
        })
        saved += 1
    report["savedFrames"] = saved
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "savedFrames": saved}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
