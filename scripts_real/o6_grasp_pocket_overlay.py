#!/usr/bin/env python3
"""Offline O6 grasp-pocket center overlay and a/b calibration tools.

This script is intentionally offline-only:
- it reads saved inference episode PKLs and JPG frames;
- it reuses the existing O6 FK and fisheye projection utilities;
- it writes overlay JPGs and JSON summaries;
- it never connects to Franka/O6 and never loads a policy checkpoint.

The "grasp pocket" is a functional reference point:

    palm_center + a * palm_width * palm_normal
                + b * palm_width * finger_forward

The first version is intended for pick_6_pro/sponge-style power grasp analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_REAL = REPO_ROOT / "scripts_real"
if str(SCRIPTS_REAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_REAL))

from calibrate_o6_camera_extrinsic import project_fisheye  # noqa: E402
from o6_fk21 import HAND21_BONES, O6Kinematics, POINT21_COLORS_BGR  # noqa: E402


DEFAULT_CALIBRATION = (
    "/home/zjc/Desktop/human2dex/data_local/"
    "o6_camera_calibration/mvs_DA9057801_o6_mount_v1/fit_ep0003_v1/"
    "camera_from_o6_base.json"
)
DEFAULT_EPISODES_ROOT = (
    "/home/zjc/Desktop/human2dex/"
    "data_local/franka_o6_eval/inference_episodes"
)

PALM_INDICES = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
MCP_INDICES = np.asarray([5, 9, 13, 17], dtype=np.int64)
POWER_TIP_INDICES = np.asarray([8, 12, 16], dtype=np.int64)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
            import dill  # type: ignore

            return dill.load(handle)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


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
        raise ValueError("No episodes selected")
    return unique


def episode_pkl_path(root: Path, episode_number: int) -> Path:
    name = f"episode_{episode_number:04d}"
    return root / name / f"{name}.pkl"


def resolve_image_path(episode_dir: Path, image_value: Any) -> Path:
    path = Path(str(image_value))
    return path if path.is_absolute() else episode_dir / path


def as_state(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        state = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        return None
    return state


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9 and np.all(np.isfinite(vector)):
        return vector / norm
    if fallback is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return normalize(fallback, None)


@dataclass
class Calibration:
    path: Path
    K: np.ndarray
    D: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    urdf_path: Path
    raw: dict[str, Any]


def load_calibration(path: Path) -> Calibration:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return Calibration(
        path=path,
        K=np.asarray(raw["K"], dtype=np.float64).reshape(3, 3),
        D=np.asarray(raw["D"], dtype=np.float64).reshape(4, 1),
        rvec=np.asarray(raw["rvec_camera_from_o6_base"], dtype=np.float64).reshape(3),
        tvec=np.asarray(raw["tvec_camera_from_o6_base_m"], dtype=np.float64).reshape(3),
        urdf_path=Path(raw["urdf_path"]).expanduser().resolve(),
        raw=raw,
    )


@dataclass
class PocketResult:
    point3d: np.ndarray
    palm_center: np.ndarray
    palm_width: float
    palm_normal: np.ndarray
    finger_forward: np.ndarray


def compute_grasp_pocket(
    points21: np.ndarray,
    *,
    a: float,
    b: float,
    normal_sign: float,
) -> PocketResult:
    points = np.asarray(points21, dtype=np.float64).reshape(21, 3)
    palm_center = points[PALM_INDICES].mean(axis=0)
    mcp_center = points[MCP_INDICES].mean(axis=0)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if not np.isfinite(palm_width) or palm_width <= 1e-9:
        palm_width = float(np.mean([
            np.linalg.norm(points[5] - points[13]),
            np.linalg.norm(points[9] - points[17]),
        ]))
    if not np.isfinite(palm_width) or palm_width <= 1e-9:
        raise ValueError("Invalid palm width")

    finger_forward = normalize(points[POWER_TIP_INDICES].mean(axis=0) - mcp_center)
    palm_across = normalize(points[17] - points[5])
    palm_normal = normalize(np.cross(palm_across, finger_forward))
    palm_normal *= float(normal_sign)
    point3d = palm_center + float(a) * palm_width * palm_normal + float(b) * palm_width * finger_forward
    return PocketResult(
        point3d=point3d,
        palm_center=palm_center,
        palm_width=palm_width,
        palm_normal=palm_normal,
        finger_forward=finger_forward,
    )


def project_points(points3d: np.ndarray, calibration: Calibration) -> tuple[np.ndarray, np.ndarray]:
    return project_fisheye(points3d, calibration.rvec, calibration.tvec, calibration.K, calibration.D)


def uv_valid(uv: np.ndarray, depth: float, width: int, height: int) -> bool:
    uv = np.asarray(uv, dtype=np.float64).reshape(2)
    return (
        bool(np.all(np.isfinite(uv)))
        and np.isfinite(float(depth))
        and float(depth) > 0.0
        and 0.0 <= float(uv[0]) < float(width)
        and 0.0 <= float(uv[1]) < float(height)
    )


def draw_antialiased_point(
    image: np.ndarray,
    center: np.ndarray,
    radius: float,
    color_bgr: Iterable[int],
) -> None:
    height, width = image.shape[:2]
    x, y = [float(value) for value in np.asarray(center).reshape(2)]
    r = max(1.0, float(radius))
    x0 = max(0, int(math.floor(x - r - 1.0)))
    x1 = min(width - 1, int(math.ceil(x + r + 1.0)))
    y0 = max(0, int(math.floor(y - r - 1.0)))
    y1 = min(height - 1, int(math.ceil(y + r + 1.0)))
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    dist = np.sqrt((xx.astype(np.float64) - x) ** 2 + (yy.astype(np.float64) - y) ** 2)
    alpha = np.clip(r + 0.5 - dist, 0.0, 1.0)[..., None]
    if not np.any(alpha > 0):
        return
    patch = image[y0:y1 + 1, x0:x1 + 1].astype(np.float32, copy=False)
    color = np.asarray(list(color_bgr), dtype=np.float32).reshape(1, 1, 3)
    image[y0:y1 + 1, x0:x1 + 1] = np.clip(patch * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def draw_skeleton(
    image: np.ndarray,
    uv21: np.ndarray,
    depth21: np.ndarray,
    *,
    line_thickness: int = 2,
    point_radius: float = 4.0,
) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    valid = np.asarray([
        uv_valid(uv21[index], depth21[index], width, height)
        for index in range(21)
    ], dtype=bool)
    valid[0] = False  # O6 hand base often projects behind the camera; keep overlay readable.
    for a_idx, b_idx in HAND21_BONES:
        if not (valid[a_idx] and valid[b_idx]):
            continue
        pa = tuple(np.round(uv21[a_idx]).astype(int))
        pb = tuple(np.round(uv21[b_idx]).astype(int))
        cv2.line(out, pa, pb, POINT21_COLORS_BGR[b_idx], int(line_thickness), cv2.LINE_AA)
    for index, uv in enumerate(uv21):
        if not valid[index]:
            continue
        draw_antialiased_point(out, uv, float(point_radius), POINT21_COLORS_BGR[index])
    return out


def draw_pocket(
    image: np.ndarray,
    raw_uv: np.ndarray | None,
    smooth_uv: np.ndarray | None,
    *,
    raw_valid: bool,
    smooth_valid: bool,
    radius: float = 7.0,
) -> np.ndarray:
    out = image.copy()
    if raw_uv is not None and raw_valid:
        x, y = np.round(raw_uv).astype(int)
        cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    if smooth_uv is not None and smooth_valid:
        draw_antialiased_point(out, smooth_uv, radius, (0, 255, 255))
        x, y = np.round(smooth_uv).astype(int)
        cv2.circle(out, (x, y), int(round(radius + 4)), (0, 180, 255), 2, cv2.LINE_AA)
    return out


def iter_episode_frames(
    episodes_root: Path,
    episodes: list[int],
    *,
    state_field: str,
    stride: int,
    max_frames_per_episode: int | None,
    include_repeated_rgb: bool,
):
    for episode_number in episodes:
        pkl_path = episode_pkl_path(episodes_root, episode_number)
        if not pkl_path.is_file():
            raise FileNotFoundError(pkl_path)
        payload = load_pickle(pkl_path)
        messages = list(payload.get("messages", []))
        episode_dir = pkl_path.parent
        emitted = 0
        for frame_index, message in enumerate(messages):
            if frame_index % max(1, int(stride)) != 0:
                continue
            if not include_repeated_rgb and bool(message.get("rgbFrameRepeated", False)):
                continue
            state = as_state(message.get(state_field))
            if state is None:
                continue
            image_value = message.get("rgbImage")
            if image_value is None:
                continue
            image_path = resolve_image_path(episode_dir, image_value)
            if not image_path.is_file():
                continue
            yield episode_number, frame_index, message, state, image_path
            emitted += 1
            if max_frames_per_episode is not None and emitted >= max_frames_per_episode:
                break


def run_overlay(args: argparse.Namespace) -> int:
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    episodes = parse_episode_spec(args.episodes)
    calibration = load_calibration(Path(args.calibration))
    kinematics = O6Kinematics(calibration.urdf_path)

    summary: dict[str, Any] = {
        "calibration": str(calibration.path),
        "episodes_root": str(episodes_root),
        "episodes": episodes,
        "state_field": args.state_field,
        "a": args.a,
        "b": args.b,
        "normal_sign": args.normal_sign,
        "ema_alpha": args.ema_alpha,
        "frames": [],
        "counts": {"processed": 0, "pocket_valid": 0, "smooth_valid": 0, "errors": 0},
    }

    last_smooth_uv_by_episode: dict[int, np.ndarray] = {}
    for episode_number, frame_index, message, state, image_path in iter_episode_frames(
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
        height, width = image.shape[:2]
        try:
            points21 = kinematics.points21(state)
            uv21, depth21 = project_points(points21, calibration)
            pocket = compute_grasp_pocket(points21, a=args.a, b=args.b, normal_sign=args.normal_sign)
            pocket_uv, pocket_depth = project_points(pocket.point3d[None, :], calibration)
            raw_uv = pocket_uv[0]
            raw_valid = uv_valid(raw_uv, float(pocket_depth[0]), width, height)
            if raw_valid:
                if episode_number not in last_smooth_uv_by_episode or args.ema_alpha >= 1.0:
                    smooth_uv = raw_uv.copy()
                else:
                    alpha = float(args.ema_alpha)
                    smooth_uv = (
                        (1.0 - alpha) * last_smooth_uv_by_episode[episode_number]
                        + alpha * raw_uv
                    )
                last_smooth_uv_by_episode[episode_number] = smooth_uv
                smooth_valid = uv_valid(smooth_uv, float(pocket_depth[0]), width, height)
            else:
                smooth_uv = last_smooth_uv_by_episode.get(episode_number)
                smooth_valid = smooth_uv is not None and uv_valid(smooth_uv, 1.0, width, height)
        except Exception as exc:
            summary["counts"]["errors"] += 1
            summary["frames"].append({
                "episode": f"episode_{episode_number:04d}",
                "frame_index": frame_index,
                "error": repr(exc),
            })
            continue

        overlay = draw_skeleton(
            image,
            uv21,
            depth21,
            line_thickness=args.line_thickness,
            point_radius=args.point_radius,
        )
        overlay = draw_pocket(
            overlay,
            raw_uv,
            smooth_uv,
            raw_valid=raw_valid,
            smooth_valid=smooth_valid,
            radius=args.pocket_radius,
        )
        text = (
            f"ep{episode_number:04d} f{frame_index:04d} "
            f"a={args.a:.3f} b={args.b:.3f}"
        )
        cv2.putText(overlay, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        episode_out = output_dir / f"episode_{episode_number:04d}"
        episode_out.mkdir(parents=True, exist_ok=True)
        out_path = episode_out / f"frame_{frame_index:06d}_grasp_pocket.jpg"
        cv2.imwrite(str(out_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        summary["counts"]["processed"] += 1
        summary["counts"]["pocket_valid"] += int(bool(raw_valid))
        summary["counts"]["smooth_valid"] += int(bool(smooth_valid))
        summary["frames"].append({
            "episode": f"episode_{episode_number:04d}",
            "frame_index": int(frame_index),
            "image": str(image_path),
            "overlay": str(out_path),
            "timestamp": float(message.get("timestamp", float("nan"))),
            "rgb_frame_repeated": bool(message.get("rgbFrameRepeated", False)),
            "pocket_uv_raw": raw_uv.tolist(),
            "pocket_uv_smooth": None if smooth_uv is None else np.asarray(smooth_uv).tolist(),
            "pocket_depth_m": float(pocket_depth[0]),
            "pocket_valid": bool(raw_valid),
            "smooth_valid": bool(smooth_valid),
            "palm_width_m": float(pocket.palm_width),
        })

    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["counts"], ensure_ascii=False))
    print(f"output_dir={output_dir}")
    return 0


def run_make_template(args: argparse.Namespace) -> int:
    episodes_root = Path(args.episodes_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    frames = []
    for episode_number, frame_index, message, state, image_path in iter_episode_frames(
        episodes_root,
        parse_episode_spec(args.episodes),
        state_field=args.state_field,
        stride=args.stride,
        max_frames_per_episode=args.max_frames_per_episode,
        include_repeated_rgb=args.include_repeated_rgb,
    ):
        frames.append({
            "episode": f"episode_{episode_number:04d}",
            "frame_index": int(frame_index),
            "image": str(image_path),
            "timestamp": float(message.get("timestamp", float("nan"))),
            "target_uv": None,
            "weight": 1.0,
            "note": "",
        })
    payload = {
        "format": "o6_grasp_pocket_ab_clicks_v1",
        "instruction": "Fill target_uv=[u,v] for frames where the desired grasp pocket location is visible/known; leave null to skip.",
        "episodes_root": str(episodes_root),
        "state_field": args.state_field,
        "frames": frames,
    }
    atomic_write_json(output, payload)
    print(f"wrote {len(frames)} annotation slots to {output}")
    return 0


def run_annotate(args: argparse.Namespace) -> int:
    template_path = Path(args.template).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    frames = list(payload.get("frames", []))
    clicked = 0
    index = 0
    while index < len(frames):
        frame = frames[index]
        image = cv2.imread(str(frame["image"]), cv2.IMREAD_COLOR)
        if image is None:
            index += 1
            continue
        display = image.copy()
        target = frame.get("target_uv")
        if target is not None:
            draw_antialiased_point(display, np.asarray(target, dtype=np.float64), 6.0, (0, 255, 255))
        label = f"{index+1}/{len(frames)} {frame.get('episode')} frame={frame.get('frame_index')}"
        cv2.putText(display, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "left click=set, n=skip, u=undo, q=quit", (8, display.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        state = {"clicked": None}

        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                userdata["clicked"] = [float(x), float(y)]

        cv2.namedWindow("o6_grasp_pocket_annotate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("o6_grasp_pocket_annotate", on_mouse, state)
        while True:
            cv2.imshow("o6_grasp_pocket_annotate", display)
            key = cv2.waitKey(30) & 0xFF
            if state["clicked"] is not None:
                frame["target_uv"] = state["clicked"]
                clicked += 1
                index += 1
                break
            if key == ord("n"):
                index += 1
                break
            if key == ord("u"):
                frame["target_uv"] = None
                break
            if key == ord("q") or key == 27:
                atomic_write_json(output_path, payload)
                cv2.destroyAllWindows()
                print(f"saved partial annotations to {output_path}, clicked={clicked}")
                return 0
    cv2.destroyAllWindows()
    atomic_write_json(output_path, payload)
    print(f"saved annotations to {output_path}, clicked={clicked}")
    return 0


def load_click_frames(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = []
    for frame in payload.get("frames", []):
        target = frame.get("target_uv")
        if target is None:
            continue
        uv = np.asarray(target, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(uv)):
            continue
        item = dict(frame)
        item["target_uv"] = uv
        item["weight"] = float(item.get("weight", 1.0))
        frames.append(item)
    if not frames:
        raise ValueError(f"No valid target_uv annotations in {path}")
    return frames


def run_fit_ab(args: argparse.Namespace) -> int:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for fit-ab") from exc

    episodes_root = Path(args.episodes_root).expanduser().resolve()
    calibration = load_calibration(Path(args.calibration))
    kinematics = O6Kinematics(calibration.urdf_path)
    frames = load_click_frames(Path(args.annotations).expanduser().resolve())

    by_episode: dict[str, dict[int, dict[str, Any]]] = {}
    for frame in frames:
        by_episode.setdefault(str(frame["episode"]), {})[int(frame["frame_index"])] = frame

    samples = []
    for episode_name, frame_map in by_episode.items():
        pkl_path = episodes_root / episode_name / f"{episode_name}.pkl"
        if not pkl_path.is_file():
            raise FileNotFoundError(pkl_path)
        payload = load_pickle(pkl_path)
        messages = payload.get("messages", [])
        for frame_index, anno in frame_map.items():
            if frame_index < 0 or frame_index >= len(messages):
                raise IndexError(f"{episode_name} frame {frame_index} out of range")
            state = as_state(messages[frame_index].get(args.state_field))
            if state is None:
                raise ValueError(f"{episode_name} frame {frame_index} has no valid {args.state_field}")
            points21 = kinematics.points21(state)
            samples.append({
                "episode": episode_name,
                "frame_index": frame_index,
                "points21": points21,
                "target_uv": np.asarray(anno["target_uv"], dtype=np.float64).reshape(2),
                "weight": float(anno.get("weight", 1.0)),
            })
    if len(samples) < 2:
        raise ValueError("At least two annotated frames are recommended for a,b fitting")

    if str(args.normal_sign).lower() == "auto":
        signs = [1.0, -1.0]
    else:
        signs = [float(args.normal_sign)]

    fits = []
    for sign in signs:
        def residual(params: np.ndarray) -> np.ndarray:
            a, b = [float(v) for v in params]
            values = []
            for sample in samples:
                pocket = compute_grasp_pocket(sample["points21"], a=a, b=b, normal_sign=sign)
                uv, depth = project_points(pocket.point3d[None, :], calibration)
                err = (uv[0] - sample["target_uv"]) * math.sqrt(max(0.0, sample["weight"]))
                if not np.isfinite(depth[0]) or depth[0] <= 0:
                    err = err + 1000.0
                values.extend(err.tolist())
            return np.asarray(values, dtype=np.float64)

        result = least_squares(
            residual,
            x0=np.asarray([args.init_a, args.init_b], dtype=np.float64),
            bounds=([args.min_a, args.min_b], [args.max_a, args.max_b]),
            loss=args.loss,
            f_scale=args.f_scale,
            max_nfev=args.max_nfev,
        )
        raw_residual = residual(result.x).reshape(-1, 2)
        per_sample = np.linalg.norm(raw_residual, axis=1)
        fits.append({
            "normal_sign": sign,
            "a": float(result.x[0]),
            "b": float(result.x[1]),
            "cost": float(result.cost),
            "success": bool(result.success),
            "message": str(result.message),
            "median_px": float(np.median(per_sample)),
            "p90_px": float(np.percentile(per_sample, 90)),
            "max_px": float(np.max(per_sample)),
            "n": int(len(samples)),
        })

    best = min(fits, key=lambda item: (item["median_px"], item["p90_px"], item["cost"]))
    output = {
        "format": "o6_grasp_pocket_ab_fit_v1",
        "annotations": str(Path(args.annotations).expanduser().resolve()),
        "episodes_root": str(episodes_root),
        "calibration": str(calibration.path),
        "state_field": args.state_field,
        "best": best,
        "candidates": fits,
        "samples": [
            {
                "episode": sample["episode"],
                "frame_index": sample["frame_index"],
                "target_uv": sample["target_uv"].tolist(),
                "weight": sample["weight"],
            }
            for sample in samples
        ],
    }
    output_path = Path(args.output).expanduser().resolve()
    atomic_write_json(output_path, output)
    print(json.dumps(best, ensure_ascii=False))
    print(f"output={output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--episodes-root", default=DEFAULT_EPISODES_ROOT)
        sub.add_argument("--episodes", required=True, help="Episode list/range, e.g. 92 or 92,109 or 92-95.")
        sub.add_argument("--state-field", default="o6_measured_state")
        sub.add_argument("--include-repeated-rgb", action="store_true")
        sub.add_argument("--stride", type=int, default=1)
        sub.add_argument("--max-frames-per-episode", type=int, default=None)

    overlay = subparsers.add_parser("overlay", help="Render O6 skeleton + grasp-pocket overlay on saved episodes.")
    add_common(overlay)
    overlay.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    overlay.add_argument("--output-dir", required=True)
    overlay.add_argument("--a", type=float, default=0.657647381181536)
    overlay.add_argument("--b", type=float, default=0.47029222572866813)
    overlay.add_argument("--normal-sign", type=float, default=-1.0)
    overlay.add_argument("--ema-alpha", type=float, default=0.3)
    overlay.add_argument("--line-thickness", type=int, default=2)
    overlay.add_argument("--point-radius", type=float, default=4.0)
    overlay.add_argument("--pocket-radius", type=float, default=7.0)
    overlay.set_defaults(func=run_overlay)

    template = subparsers.add_parser("make-template", help="Create annotation JSON template for desired grasp-pocket clicks.")
    add_common(template)
    template.add_argument("--output", required=True)
    template.set_defaults(func=run_make_template)

    annotate = subparsers.add_parser("annotate", help="OpenCV click UI for filling target_uv in a template.")
    annotate.add_argument("--template", required=True)
    annotate.add_argument("--output", required=True)
    annotate.set_defaults(func=run_annotate)

    fit = subparsers.add_parser("fit-ab", help="Fit a,b from target_uv click annotations.")
    fit.add_argument("--episodes-root", default=DEFAULT_EPISODES_ROOT)
    fit.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    fit.add_argument("--annotations", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--state-field", default="o6_measured_state")
    fit.add_argument("--normal-sign", default="auto", help="'auto', '1', or '-1'.")
    fit.add_argument("--init-a", type=float, default=0.45)
    fit.add_argument("--init-b", type=float, default=0.25)
    fit.add_argument("--min-a", type=float, default=-1.5)
    fit.add_argument("--max-a", type=float, default=1.5)
    fit.add_argument("--min-b", type=float, default=-1.5)
    fit.add_argument("--max-b", type=float, default=2.5)
    fit.add_argument("--loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    fit.add_argument("--f-scale", type=float, default=8.0)
    fit.add_argument("--max-nfev", type=int, default=200)
    fit.set_defaults(func=run_fit_ab)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "normal_sign") and str(args.normal_sign).lower() != "auto":
        args.normal_sign = float(args.normal_sign)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
