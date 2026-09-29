#!/usr/bin/env python3
"""Preview cavity-center fisheye alignment on saved RGB images."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_inference_config import load_yaml_mapping
from real_inference_view_alignment import ViewCanonicalizer


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_pair(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("point must look like x,y")
    return [float(parts[0]), float(parts[1])]


def _parse_size(value: str) -> list[int]:
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must look like width,height")
    width, height = int(parts[0]), int(parts[1])
    if width <= 1 or height <= 1:
        raise argparse.ArgumentTypeError("width/height must be > 1")
    return [width, height]


def _parse_circle(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("circle must look like cx,cy,radius")
    circle = [float(part) for part in parts]
    if circle[2] <= 0.0:
        raise argparse.ArgumentTypeError("circle radius must be > 0")
    return circle


def _parse_scales(value: str) -> list[float]:
    scales = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not scales or any(scale <= 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("scales must be positive")
    return scales


def _find_images(path: pathlib.Path, limit: int) -> list[pathlib.Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit > 0:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"no images found in {path}")
    return images


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _overlay_reflection(aligned_rgb: np.ndarray, reflected_mask: np.ndarray) -> np.ndarray:
    overlay = aligned_rgb.copy()
    if not np.any(reflected_mask):
        return overlay
    if np.issubdtype(overlay.dtype, np.floating):
        magenta = np.array([1.0, 0.0, 1.0], dtype=overlay.dtype)
    else:
        magenta = np.array([255, 0, 255], dtype=overlay.dtype)
    overlay[reflected_mask] = (
        0.55 * overlay[reflected_mask].astype(np.float32)
        + 0.45 * magenta.astype(np.float32)
    ).astype(overlay.dtype)
    return overlay


def _draw_debug(
        image_rgb: np.ndarray,
        debug: dict,
        label: str,
        *,
        draw_center: bool = True) -> np.ndarray:
    out = image_rgb.copy()
    circle = debug.get("outputCircle")
    if circle is not None:
        cx, cy, radius = circle
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(radius)), (255, 255, 0), 1, cv2.LINE_AA)
    if draw_center:
        center = debug.get("targetCenter")
        if center is not None:
            x, y = center
            cv2.drawMarker(out, (int(round(x)), int(round(y))), (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
    return _label(out, label)


def _make_contact_sheet(cells: list[np.ndarray], cols: int) -> np.ndarray:
    cols = max(1, int(cols))
    height = max(cell.shape[0] for cell in cells)
    width = max(cell.shape[1] for cell in cells)
    pad = 8
    normalized = []
    for cell in cells:
        if cell.shape[:2] != (height, width):
            cell = cv2.resize(cell, (width, height), interpolation=cv2.INTER_AREA)
        normalized.append(cell)
    rows = []
    for start in range(0, len(normalized), cols):
        row = normalized[start:start + cols]
        while len(row) < cols:
            row.append(np.full((height, width, 3), 255, dtype=np.uint8))
        pieces = []
        for index, cell in enumerate(row):
            if index:
                pieces.append(np.full((height, pad, 3), 255, dtype=np.uint8))
            pieces.append(cell)
        rows.append(np.hstack(pieces))
    pieces = []
    for index, row in enumerate(rows):
        if index:
            pieces.append(np.full((pad, row.shape[1], 3), 255, dtype=np.uint8))
        pieces.append(row)
    return np.vstack(pieces)


def _json_debug(debug: dict) -> dict:
    result = {}
    for key, value in debug.items():
        if isinstance(value, np.ndarray):
            continue
        result[key] = value
    return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eval_franka_pts21_config.yaml")
    parser.add_argument("--input", required=True, nargs="+", help="Image file(s) or directories")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--train-center", type=_parse_pair, default=None)
    parser.add_argument("--train-image-size", type=_parse_size, default=None)
    parser.add_argument("--deploy-center", type=_parse_pair, required=True)
    parser.add_argument("--deploy-circle", type=_parse_circle, default=None)
    parser.add_argument("--circle-threshold", type=float, default=None)
    parser.add_argument("--circle-margin-px", type=float, default=None)
    parser.add_argument(
        "--reflect-inner-margin-px",
        "--reflect-margin-px",
        type=float,
        default=None,
        dest="reflect_inner_margin_px",
    )
    parser.add_argument("--circle-mode", default=None)
    parser.add_argument("--scales", type=_parse_scales, default=_parse_scales("1.0,0.95,0.92,0.90"))
    parser.add_argument("--cols", type=int, default=4)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    cfg_path = pathlib.Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_yaml_mapping(str(cfg_path))
    base_alignment_cfg = dict(cfg.get("view_alignment", {}) or {})
    base_alignment_cfg.update({
        "enabled": True,
        "transform_type": "cavity_center",
        "deploy_center_px": args.deploy_center,
    })
    if args.train_center is not None:
        base_alignment_cfg["train_center_px"] = args.train_center
    if args.train_image_size is not None:
        base_alignment_cfg["train_image_size"] = args.train_image_size
    if args.deploy_circle is not None:
        base_alignment_cfg["deploy_circle_px"] = args.deploy_circle
    if args.circle_threshold is not None:
        base_alignment_cfg["circle_threshold"] = float(args.circle_threshold)
    if args.circle_margin_px is not None:
        base_alignment_cfg["circle_margin_px"] = float(args.circle_margin_px)
    if args.reflect_inner_margin_px is not None:
        base_alignment_cfg["reflect_inner_margin_px"] = float(args.reflect_inner_margin_px)
    if args.circle_mode is not None:
        base_alignment_cfg["circle_mode"] = str(args.circle_mode).replace("-", "_")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    aligned_dir = output_dir / "aligned"
    overlay_dir = output_dir / "reflected_overlay"
    compare_dir = output_dir / "compare"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[pathlib.Path] = []
    for raw in args.input:
        image_paths.extend(_find_images(pathlib.Path(raw).expanduser().resolve(), args.max_images))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    summary = {
        "config": str(cfg_path),
        "outputDir": str(output_dir),
        "scales": [float(scale) for scale in args.scales],
        "images": {},
    }
    all_cells = []
    for image_path in image_paths:
        raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            raise IOError(f"cv2.imread failed: {image_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        cells = [_label(raw_rgb, f"raw {image_path.name}")]
        image_record = {}
        for scale in args.scales:
            alignment_cfg = dict(base_alignment_cfg)
            alignment_cfg["scale"] = float(scale)
            canonicalizer = ViewCanonicalizer(alignment_cfg)
            aligned_rgb, debug = canonicalizer.apply_image_with_debug(raw_rgb)
            overlay_rgb = _overlay_reflection(aligned_rgb, debug["reflectedMask"])

            stem = f"{image_path.stem}_s{float(scale):.3f}"
            aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aligned_dir / f"{stem}.jpg"), aligned_bgr)
            cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), overlay_bgr)
            cells.append(
                _draw_debug(
                    aligned_rgb,
                    debug,
                    (
                        f"s={float(scale):.2f} refl="
                        f"{debug['reflectedAreaFractionInCircle']:.3f}"
                    ),
                )
            )
            image_record[f"{float(scale):.3f}"] = {
                "aligned": str(aligned_dir / f"{stem}.jpg"),
                "reflectedOverlay": str(overlay_dir / f"{stem}.jpg"),
                **_json_debug(debug),
            }

        sheet_rgb = _make_contact_sheet(cells, cols=args.cols)
        sheet_bgr = cv2.cvtColor(sheet_rgb, cv2.COLOR_RGB2BGR)
        sheet_path = compare_dir / f"{image_path.stem}_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet_bgr)
        image_record["contactSheet"] = str(sheet_path)
        summary["images"][str(image_path)] = image_record
        all_cells.append(sheet_rgb)

    if all_cells:
        max_w = max(cell.shape[1] for cell in all_cells)
        padded = []
        for cell in all_cells:
            if cell.shape[1] < max_w:
                pad = np.full((cell.shape[0], max_w - cell.shape[1], 3), 255, dtype=np.uint8)
                cell = np.hstack([cell, pad])
            padded.append(cell)
        combined = np.vstack(
            [
                part
                for index, cell in enumerate(padded)
                for part in (
                    ([np.full((8, max_w, 3), 255, dtype=np.uint8)] if index else [])
                    + [cell]
                )
            ]
        )
        combined_path = output_dir / "all_contact_sheets.jpg"
        cv2.imwrite(str(combined_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        summary["allContactSheets"] = str(combined_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
