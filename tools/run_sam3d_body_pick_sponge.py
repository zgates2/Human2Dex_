#!/usr/bin/env python3
"""Run official SAM 3D Body inference on a few pick_sponge images.

This script uses facebookresearch/sam-3d-body code under sam/sam-3d-body-code.
It does not use the glove augmentation pipeline. If an ROI json is provided, it
passes the annotated bbox directly to SAM3D as the top-down crop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SAM3D_CODE = REPO_ROOT / "sam" / "sam-3d-body-code"
if str(SAM3D_CODE) not in sys.path:
    sys.path.insert(0, str(SAM3D_CODE))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path):
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def list_episode_dirs(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "images").is_dir()],
        key=natural_key,
    )


def list_images(episode_dir: Path) -> List[Path]:
    return sorted(
        [p for p in (episode_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def load_roi(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {"episodes": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def padded_bbox(
    bbox: Sequence[float], width: int, height: int, padding: float
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = normalize_bbox(bbox)
    bw = x1 - x0
    bh = y1 - y0
    pad = max(bw, bh) * padding
    return (
        max(0.0, x0 - pad),
        max(0.0, y0 - pad),
        min(float(width - 1), x1 + pad),
        min(float(height - 1), y1 + pad),
    )


def pick_episodes(input_root: Path, roi_data: Dict[str, Any], requested: Sequence[str], limit: int) -> List[Path]:
    by_name = {p.name: p for p in list_episode_dirs(input_root)}
    if requested:
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise FileNotFoundError(f"Missing episodes: {', '.join(missing)}")
        return [by_name[name] for name in requested]

    annotated = [
        by_name[name]
        for name in sorted(roi_data.get("episodes", {}).keys())
        if name in by_name
    ]
    if annotated:
        return annotated[:limit]
    return list(by_name.values())[:limit]


def keypoint_names() -> List[str]:
    return (
        mhr70_pose_info["body_keypoint_names"]
        + mhr70_pose_info["foot_keypoint_names"]
        + mhr70_pose_info["left_hand_keypoint_names"]
        + mhr70_pose_info["right_hand_keypoint_names"]
        + mhr70_pose_info["extra_keypoint_names"]
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def summarize_person(person: Dict[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    k2d = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
    k3d = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
    left_slice = slice(23, 43)
    right_slice = slice(43, 63)
    summary = {
        "bbox": np.asarray(person["bbox"]).round(3).tolist(),
        "lhand_bbox": np.asarray(person.get("lhand_bbox", [])).round(3).tolist(),
        "rhand_bbox": np.asarray(person.get("rhand_bbox", [])).round(3).tolist(),
        "pred_keypoints_2d_shape": list(k2d.shape),
        "pred_keypoints_3d_shape": list(k3d.shape),
        "pred_vertices_shape": list(np.asarray(person["pred_vertices"]).shape),
        "pred_cam_t": np.asarray(person["pred_cam_t"]).round(6).tolist(),
        "focal_length": float(np.asarray(person["focal_length"]).reshape(-1)[0]),
        "left_hand_keypoints_2d": [
            {"index": i, "name": names[i], "xy": k2d[i].round(3).tolist()}
            for i in range(left_slice.start, left_slice.stop)
        ],
        "right_hand_keypoints_2d": [
            {"index": i, "name": names[i], "xy": k2d[i].round(3).tolist()}
            for i in range(right_slice.start, right_slice.stop)
        ],
        "hand_pose_params_shape": list(np.asarray(person["hand_pose_params"]).shape),
    }
    return summary


def draw_outputs(image_bgr: np.ndarray, outputs: Sequence[Dict[str, Any]], names: Sequence[str]) -> np.ndarray:
    out = image_bgr.copy()
    colors = {
        "bbox": (40, 230, 40),
        "left": (255, 80, 40),
        "right": (40, 80, 255),
        "body": (230, 230, 230),
    }
    for person in outputs:
        bbox = np.asarray(person["bbox"], dtype=np.float32)
        cv2.rectangle(out, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), colors["bbox"], 2)
        for box_name, color_key in [("lhand_bbox", "left"), ("rhand_bbox", "right")]:
            if box_name in person:
                box = np.asarray(person[box_name], dtype=np.float32)
                cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[color_key], 2)

        k2d = np.asarray(person["pred_keypoints_2d"], dtype=np.float32)
        for idx, xy in enumerate(k2d):
            x, y = float(xy[0]), float(xy[1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if 23 <= idx < 43:
                color = colors["left"]
                radius = 4
            elif 43 <= idx < 63:
                color = colors["right"]
                radius = 4
            else:
                color = colors["body"]
                radius = 2
            cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1, lineType=cv2.LINE_AA)

        for idx in list(range(23, 43, 4)) + list(range(43, 63, 4)):
            x, y = k2d[idx]
            if np.isfinite(x) and np.isfinite(y):
                cv2.putText(
                    out,
                    names[idx].replace("left_", "L_").replace("right_", "R_"),
                    (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return out


def save_npz(output_path: Path, outputs: Sequence[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat: Dict[str, Any] = {"num_people": np.array([len(outputs)], dtype=np.int32)}
    for pid, person in enumerate(outputs):
        for key in [
            "bbox",
            "lhand_bbox",
            "rhand_bbox",
            "pred_keypoints_2d",
            "pred_keypoints_3d",
            "pred_vertices",
            "pred_cam_t",
            "focal_length",
            "hand_pose_params",
            "body_pose_params",
            "shape_params",
            "scale_params",
        ]:
            if key in person and person[key] is not None:
                flat[f"person{pid}_{key}"] = np.asarray(person[key])
    np.savez_compressed(output_path, **flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "pick_sponge"))
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "pick_sponge_sam3d_body_test"))
    parser.add_argument("--checkpoint-path", default=str(REPO_ROOT / "sam" / "sam-3d-body-dinov3" / "model.ckpt"))
    parser.add_argument("--mhr-path", default=str(REPO_ROOT / "sam" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"))
    parser.add_argument("--roi", default=str(REPO_ROOT / "glove_aug_pipeline" / "roi_annotations.json"))
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--limit-episodes", type=int, default=1)
    parser.add_argument("--limit-frames", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bbox-mode", choices=["roi", "full"], default="roi")
    parser.add_argument("--bbox-padding", type=float, default=0.35)
    parser.add_argument("--inference-type", choices=["full", "body", "hand"], default="full")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    roi_data = load_roi(Path(args.roi) if args.roi else None)
    names = keypoint_names()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    print(f"Loading SAM3D Body on {device}...")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path,
        device=device,
        mhr_path=args.mhr_path,
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    episodes = pick_episodes(input_root, roi_data, args.episode, args.limit_episodes)
    records: List[Dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    for episode_dir in episodes:
        images = list_images(episode_dir)[:: max(1, args.stride)][: args.limit_frames]
        roi_item = roi_data.get("episodes", {}).get(episode_dir.name, {})
        for image_path in images:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            h, w = image_bgr.shape[:2]
            if args.bbox_mode == "roi" and roi_item.get("bbox"):
                bbox = padded_bbox(roi_item["bbox"], w, h, args.bbox_padding)
                bboxes = np.asarray([bbox], dtype=np.float32)
            elif args.bbox_mode == "full":
                bbox = (0.0, 0.0, float(w - 1), float(h - 1))
                bboxes = np.asarray([bbox], dtype=np.float32)
            else:
                bbox = (0.0, 0.0, float(w - 1), float(h - 1))
                bboxes = np.asarray([bbox], dtype=np.float32)

            print(f"Running {image_path} bbox={np.round(bboxes[0], 2).tolist()} inference={args.inference_type}")
            outputs = estimator.process_one_image(
                str(image_path),
                bboxes=bboxes,
                use_mask=False,
                inference_type=args.inference_type,
            )

            rel_dir = output_root / episode_dir.name
            rel_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem
            overlay_path = rel_dir / f"{stem}_sam3d_keypoints.jpg"
            npz_path = rel_dir / f"{stem}_sam3d_outputs.npz"

            overlay = draw_outputs(image_bgr, outputs, names)
            cv2.imwrite(str(overlay_path), overlay)
            save_npz(npz_path, outputs)

            records.append(
                {
                    "episode": episode_dir.name,
                    "image": str(image_path),
                    "bbox_mode": args.bbox_mode,
                    "input_bbox": np.round(bboxes[0], 3).tolist(),
                    "inference_type": args.inference_type,
                    "num_outputs": len(outputs),
                    "overlay": str(overlay_path),
                    "npz": str(npz_path),
                    "people": [summarize_person(person, names) for person in outputs],
                }
            )

    summary = {
        "input": str(input_root),
        "checkpoint_path": args.checkpoint_path,
        "mhr_path": args.mhr_path,
        "device": str(device),
        "note": "Official SAM 3D Body inference. Hand keypoints are MHR70 indices 23:43 left, 43:63 right.",
        "records": records,
    }
    summary_path = output_root / "summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    tmp_path.replace(summary_path)
    print(f"Wrote summary: {summary_path}")
    for record in records:
        print(f"{Path(record['image']).name}: outputs={record['num_outputs']} overlay={record['overlay']}")


if __name__ == "__main__":
    main()
