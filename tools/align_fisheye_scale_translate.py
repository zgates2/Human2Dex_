#!/usr/bin/env python3
"""Align deployment fisheye frames to a marked training cavity center.

The script has three modes:

1. annotate: click one cavity center per image and save a marks JSON.
2. grid: generate coordinate-grid previews for headless/manual inspection.
3. apply: scale+translate deployment frames toward the training center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_calibration/fisheye_scale_translate_align"


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(value) for value in text.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must look like 224x224") from exc
    if width <= 1 or height <= 1:
        raise argparse.ArgumentTypeError("size values must be > 1")
    return width, height


def parse_pair(text: str) -> tuple[float, float]:
    try:
        x, y = (float(value.strip()) for value in text.split(",", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point must look like x,y") from exc
    return x, y


def parse_circle(text: str) -> tuple[float, float, float]:
    try:
        x, y, radius = (float(value.strip()) for value in text.split(",", 2))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("circle must look like cx,cy,r") from exc
    if radius <= 0.0:
        raise argparse.ArgumentTypeError("circle radius must be positive")
    return x, y, radius


def parse_scales(text: str) -> list[float]:
    try:
        scales = [float(value) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scales must look like 1.0,0.95,0.90") from exc
    if not scales or any(scale <= 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("all scales must be positive")
    return scales


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def abs_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def load_marks(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"images": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "images" not in data or not isinstance(data["images"], dict):
        raise ValueError(f"invalid marks JSON: {path}")
    return data


def save_marks(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def point_from_arg_or_marks(
    *,
    image_path: Path,
    image_size: tuple[int, int],
    marks: dict[str, Any],
    explicit: tuple[float, float] | None,
    units: str,
    name: str,
) -> tuple[float, float]:
    if explicit is not None:
        return resolve_point(explicit, image_size=image_size, units=units)
    record = marks.get("images", {}).get(abs_path(image_path))
    if record is not None and "center" in record:
        center = record["center"]
        return float(center[0]), float(center[1])
    raise ValueError(f"{name} missing for {image_path}; mark it or pass an explicit point")


def resolve_point(
    point: tuple[float, float],
    *,
    image_size: tuple[int, int],
    units: str,
) -> tuple[float, float]:
    width, height = image_size
    x, y = point
    if units == "ratio" or (units == "auto" and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return x * float(width - 1), y * float(height - 1)
    return x, y


def scale_point_to_output(
    point: tuple[float, float],
    *,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[float, float]:
    in_w, in_h = input_size
    out_w, out_h = output_size
    if in_w <= 1 or in_h <= 1:
        raise ValueError("input image is too small")
    x, y = point
    return x * float(out_w - 1) / float(in_w - 1), y * float(out_h - 1) / float(in_h - 1)


def estimate_fisheye_circle(
    image: np.ndarray,
    *,
    threshold: int = 8,
    margin_px: float = 0.0,
) -> tuple[float, float, float, dict[str, float]]:
    """Estimate the non-black fisheye boundary from the largest bright component."""
    max_chan = np.max(image[:, :, :3], axis=2).astype(np.uint8)
    mask = (max_chan > int(threshold)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = image.shape[:2]
        radius = min(width, height) / 2.0 - margin_px
        return (
            (width - 1) / 2.0,
            (height - 1) / 2.0,
            max(1.0, radius),
            {"mask_fraction": 0.0, "fallback": 1.0},
        )
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    radius = max(1.0, float(radius) + float(margin_px))
    stats = {
        "mask_fraction": float(np.mean(mask > 0)),
        "contour_area": float(cv2.contourArea(contour)),
        "fallback": 0.0,
    }
    return float(cx), float(cy), float(radius), stats


def circle_from_arg_marks_or_auto(
    *,
    image_path: Path,
    image: np.ndarray,
    marks: dict[str, Any],
    explicit: tuple[float, float, float] | None,
    units: str,
    threshold: int,
    margin_px: float,
) -> tuple[float, float, float, dict[str, float], str]:
    width, height = image.shape[1], image.shape[0]
    if explicit is not None:
        cx, cy = resolve_point((explicit[0], explicit[1]), image_size=(width, height), units=units)
        radius = explicit[2] * float(min(width, height)) if units == "ratio" else explicit[2]
        return cx, cy, radius, {"manual": 1.0}, "explicit"
    record = marks.get("images", {}).get(abs_path(image_path))
    if record is not None and "circle" in record:
        circle = record["circle"]
        return float(circle[0]), float(circle[1]), float(circle[2]), {"manual": 1.0}, "marks"
    cx, cy, radius, stats = estimate_fisheye_circle(image, threshold=threshold, margin_px=margin_px)
    return cx, cy, radius, stats, "auto"


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def draw_point_and_circle(
    image: np.ndarray,
    *,
    point: tuple[float, float] | None = None,
    circle: tuple[float, float, float] | None = None,
    label: str | None = None,
    grid_step: int | None = None,
) -> np.ndarray:
    result = image.copy()
    if grid_step is not None and grid_step > 0:
        height, width = result.shape[:2]
        for x in range(0, width, grid_step):
            cv2.line(result, (x, 0), (x, height - 1), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(result, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)
        for y in range(0, height, grid_step):
            cv2.line(result, (0, y), (width - 1, y), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.putText(result, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1)
    if circle is not None:
        cx, cy, radius = circle
        cv2.circle(
            result,
            (int(round(cx)), int(round(cy))),
            int(round(radius)),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if point is not None:
        x, y = point
        center = (int(round(x)), int(round(y)))
        cv2.drawMarker(result, center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.circle(result, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            result,
            f"{x:.1f},{y:.1f}",
            (min(result.shape[1] - 80, center[0] + 6), max(15, center[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if label:
        result = draw_label(result, label)
    return result


def resize_cell(image: np.ndarray, cell_size: tuple[int, int]) -> np.ndarray:
    width, height = cell_size
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, cell_size, interpolation=cv2.INTER_AREA)


def make_contact_sheet(cells: list[np.ndarray], *, cols: int, cell_size: tuple[int, int]) -> np.ndarray:
    cols = max(1, int(cols))
    pad = 8
    rows: list[np.ndarray] = []
    for start in range(0, len(cells), cols):
        row = [resize_cell(cell, cell_size) for cell in cells[start : start + cols]]
        while len(row) < cols:
            row.append(np.full((cell_size[1], cell_size[0], 3), 255, dtype=np.uint8))
        pieces: list[np.ndarray] = []
        for idx, cell in enumerate(row):
            if idx:
                pieces.append(np.full((cell_size[1], pad, 3), 255, dtype=np.uint8))
            pieces.append(cell)
        rows.append(np.hstack(pieces))
    pieces = []
    for idx, row in enumerate(rows):
        if idx:
            pieces.append(np.full((pad, row.shape[1], 3), 255, dtype=np.uint8))
        pieces.append(row)
    return np.vstack(pieces)


def align_scale_translate(
    image: np.ndarray,
    *,
    source_center: tuple[float, float],
    target_center: tuple[float, float],
    source_circle: tuple[float, float, float],
    output_size: tuple[int, int],
    scale: float,
    circle_mode: str,
    reference_circle: tuple[float, float, float] | None,
) -> tuple[np.ndarray, dict[str, float], tuple[float, float, float], np.ndarray]:
    src_h, src_w = image.shape[:2]
    out_w, out_h = output_size
    sx, sy = source_center
    tx = target_center[0] - float(scale) * sx
    ty = target_center[1] - float(scale) * sy

    yy, xx = np.indices((out_h, out_w), dtype=np.float32)
    inv_x = (xx - tx) / float(scale)
    inv_y = (yy - ty) / float(scale)
    src_cx, src_cy, src_radius = source_circle

    sample_x = inv_x.copy()
    sample_y = inv_y.copy()
    dx = sample_x - float(src_cx)
    dy = sample_y - float(src_cy)
    dist = np.sqrt(dx * dx + dy * dy)
    outside_source_circle = dist > float(src_radius)
    if np.any(outside_source_circle):
        period = max(1.0, 2.0 * float(src_radius))
        dist_mod = np.mod(dist, period)
        reflected_dist = np.where(dist_mod > float(src_radius), period - dist_mod, dist_mod)
        safe_dist = np.maximum(dist, 1e-6)
        sample_x = np.where(
            outside_source_circle,
            float(src_cx) + dx / safe_dist * reflected_dist,
            sample_x,
        )
        sample_y = np.where(
            outside_source_circle,
            float(src_cy) + dy / safe_dist * reflected_dist,
            sample_y,
        )

    warped = cv2.remap(
        image,
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    source_valid = (
        (inv_x >= 0.0)
        & (inv_x <= float(src_w - 1))
        & (inv_y >= 0.0)
        & (inv_y <= float(src_h - 1))
        & ((inv_x - src_cx) ** 2 + (inv_y - src_cy) ** 2 <= src_radius**2)
    )

    if circle_mode == "transformed-source":
        out_circle = (
            target_center[0] + float(scale) * (src_cx - sx),
            target_center[1] + float(scale) * (src_cy - sy),
            float(scale) * src_radius,
        )
    elif circle_mode == "fixed-source":
        out_circle = (
            src_cx * float(out_w - 1) / float(max(1, src_w - 1)),
            src_cy * float(out_h - 1) / float(max(1, src_h - 1)),
            src_radius * float(min(out_w, out_h)) / float(max(1, min(src_w, src_h))),
        )
    elif circle_mode == "fixed-reference":
        if reference_circle is None:
            raise ValueError("circle_mode=fixed-reference requires a reference circle")
        out_circle = reference_circle
    elif circle_mode == "fixed-output":
        out_circle = (
            target_center[0],
            target_center[1],
            float(min(out_w, out_h)) * 0.5,
        )
    else:
        raise ValueError(f"unknown circle mode: {circle_mode}")

    out_cx, out_cy, out_radius = out_circle
    output_circle_mask = (xx - out_cx) ** 2 + (yy - out_cy) ** 2 <= out_radius**2
    reflected_mask = output_circle_mask & ~source_valid
    result = warped.copy()
    result[~output_circle_mask] = 0

    circle_area = max(1, int(np.count_nonzero(output_circle_mask)))
    stats = {
        "scale": float(scale),
        "source_center_x": float(sx),
        "source_center_y": float(sy),
        "target_center_x": float(target_center[0]),
        "target_center_y": float(target_center[1]),
        "output_circle_cx": float(out_cx),
        "output_circle_cy": float(out_cy),
        "output_circle_radius": float(out_radius),
        "circle_area_fraction": float(np.mean(output_circle_mask)),
        "reflected_area_fraction_in_circle": float(np.count_nonzero(reflected_mask) / circle_area),
        "black_fraction": float(np.mean(np.all(result <= 8, axis=2))),
    }
    return result, stats, out_circle, reflected_mask


def command_grid(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells: list[np.ndarray] = []
    records: dict[str, Any] = {"images": {}}
    for raw_path in args.images:
        image_path = raw_path.expanduser().resolve()
        image = read_image(image_path)
        cx, cy, radius, stats = estimate_fisheye_circle(
            image,
            threshold=int(args.circle_threshold),
            margin_px=float(args.circle_margin_px),
        )
        circle = (cx, cy, radius)
        preview = draw_point_and_circle(
            image,
            circle=circle,
            label=f"{image_path.name} {image.shape[1]}x{image.shape[0]}",
            grid_step=int(args.grid_step),
        )
        out_path = output_dir / f"{image_path.stem}_grid.jpg"
        cv2.imwrite(str(out_path), preview)
        cells.append(preview)
        records["images"][str(image_path)] = {
            "size": [int(image.shape[1]), int(image.shape[0])],
            "circle": [float(circle[0]), float(circle[1]), float(circle[2])],
            "circle_stats": stats,
            "grid_preview": str(out_path),
        }
    sheet = make_contact_sheet(cells, cols=int(args.cols), cell_size=args.cell_size)
    sheet_path = output_dir / "coordinate_grid_contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    records["contact_sheet"] = str(sheet_path)
    save_marks(output_dir / "auto_circles.json", records)
    print(json.dumps(records, indent=2))


def command_annotate(args: argparse.Namespace) -> None:
    marks_path = args.marks_json.expanduser().resolve()
    marks = load_marks(marks_path)
    window = "click cavity center"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for raw_path in args.images:
        image_path = raw_path.expanduser().resolve()
        image = read_image(image_path)
        display = image.copy()
        cx, cy, radius, circle_stats = estimate_fisheye_circle(
            image,
            threshold=int(args.circle_threshold),
            margin_px=float(args.circle_margin_px),
        )
        circle = (cx, cy, radius)
        clicked: list[tuple[float, float]] = []

        def redraw() -> None:
            point = clicked[-1] if clicked else None
            view = draw_point_and_circle(
                display,
                point=point,
                circle=circle,
                label=f"{image_path.name}: click center, n/enter save, r reset, q quit",
                grid_step=int(args.grid_step) if args.grid_step > 0 else None,
            )
            cv2.imshow(window, view)

        def on_mouse(event, x, y, flags, userdata) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.append((float(x), float(y)))
                redraw()

        cv2.setMouseCallback(window, on_mouse)
        redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in {ord("n"), ord("s"), 13, 10}:
                if not clicked:
                    print(f"no center selected for {image_path}; skipping")
                    break
                center = clicked[-1]
                marks["images"][str(image_path)] = {
                    "size": [int(image.shape[1]), int(image.shape[0])],
                    "center": [float(center[0]), float(center[1])],
                    "circle": [float(circle[0]), float(circle[1]), float(circle[2])],
                    "circle_stats": circle_stats,
                }
                save_marks(marks_path, marks)
                print(f"saved {image_path}: center={center}")
                break
            if key == ord("r"):
                clicked.clear()
                redraw()
            if key == ord("q") or key == 27:
                save_marks(marks_path, marks)
                cv2.destroyWindow(window)
                print(f"saved marks to {marks_path}")
                return
    cv2.destroyWindow(window)
    save_marks(marks_path, marks)
    print(f"saved marks to {marks_path}")


def iter_deploy_images(paths: Iterable[Path]) -> list[Path]:
    result = []
    for path in paths:
        expanded = path.expanduser().resolve()
        if expanded.is_dir():
            result.extend(sorted(expanded.glob("*.jpg")))
        else:
            result.append(expanded)
    return result


def command_apply(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marks = load_marks(args.marks_json.expanduser().resolve() if args.marks_json else None)

    train_path = args.train_image.expanduser().resolve()
    train_image = read_image(train_path)
    output_size = args.output_size
    train_size = (train_image.shape[1], train_image.shape[0])
    train_center = point_from_arg_or_marks(
        image_path=train_path,
        image_size=train_size,
        marks=marks,
        explicit=args.train_center,
        units=args.center_units,
        name="train center",
    )
    target_center = scale_point_to_output(
        train_center,
        input_size=train_size,
        output_size=output_size,
    )
    train_cx, train_cy, train_radius, train_circle_source = estimate_fisheye_circle(
        train_image,
        threshold=int(args.circle_threshold),
        margin_px=float(args.circle_margin_px),
    )
    train_circle_raw = (train_cx, train_cy, train_radius)
    train_circle_output = (
        train_circle_raw[0] * float(output_size[0] - 1) / float(train_size[0] - 1),
        train_circle_raw[1] * float(output_size[1] - 1) / float(train_size[1] - 1),
        train_circle_raw[2] * float(min(output_size)) / float(min(train_size)),
    )

    reference_view = cv2.resize(train_image, output_size, interpolation=cv2.INTER_AREA)
    reference_view = draw_point_and_circle(
        reference_view,
        point=target_center,
        circle=train_circle_output,
        label=f"train ref center {target_center[0]:.1f},{target_center[1]:.1f}",
    )

    deploy_paths = iter_deploy_images(args.deploy_images)
    summary: dict[str, Any] = {
        "train_image": str(train_path),
        "train_size": list(train_size),
        "train_center": list(train_center),
        "target_center_output": list(target_center),
        "train_circle_output": list(train_circle_output),
        "train_circle_source": train_circle_source,
        "output_size": list(output_size),
        "circle_mode": args.circle_mode,
        "deploy": {},
    }
    all_sheet_cells: list[np.ndarray] = []

    for deploy_path in deploy_paths:
        image = read_image(deploy_path)
        deploy_size = (image.shape[1], image.shape[0])
        deploy_center = point_from_arg_or_marks(
            image_path=deploy_path,
            image_size=deploy_size,
            marks=marks,
            explicit=args.deploy_center,
            units=args.center_units,
            name="deploy center",
        )
        source_circle, circle_stats, circle_source = None, None, None
        circle_result = circle_from_arg_marks_or_auto(
            image_path=deploy_path,
            image=image,
            marks=marks,
            explicit=args.deploy_circle,
            units=args.circle_units,
            threshold=int(args.circle_threshold),
            margin_px=float(args.circle_margin_px),
        )
        source_circle = circle_result[:3]
        circle_stats = circle_result[3]
        circle_source = circle_result[4]
        deploy_record: dict[str, Any] = {
            "size": list(deploy_size),
            "center": list(deploy_center),
            "source_circle": [float(v) for v in source_circle],
            "source_circle_source": circle_source,
            "source_circle_stats": circle_stats,
            "outputs": {},
        }

        cells = [
            reference_view,
            draw_point_and_circle(
                image,
                point=deploy_center,
                circle=source_circle,
                label=f"deploy {deploy_path.name}",
            ),
        ]
        for scale in args.scales:
            aligned, stats, out_circle, reflected_mask = align_scale_translate(
                image,
                source_center=deploy_center,
                target_center=target_center,
                source_circle=source_circle,
                output_size=output_size,
                scale=float(scale),
                circle_mode=args.circle_mode,
                reference_circle=train_circle_output,
            )
            overlay = aligned.copy()
            overlay[reflected_mask] = (
                0.55 * overlay[reflected_mask].astype(np.float32)
                + 0.45 * np.array([255, 0, 255], dtype=np.float32)
            ).astype(np.uint8)
            out_name = f"{deploy_path.stem}_align_s{float(scale):.3f}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), aligned)
            reflected_path = output_dir / f"{deploy_path.stem}_align_s{float(scale):.3f}_reflected_overlay.jpg"
            cv2.imwrite(str(reflected_path), overlay)
            deploy_record["outputs"][f"{float(scale):.3f}"] = {
                "image": str(out_path),
                "reflected_overlay": str(reflected_path),
                **stats,
            }
            cells.append(
                draw_point_and_circle(
                    aligned,
                    point=target_center,
                    circle=out_circle,
                    label=(
                        f"s={float(scale):.2f} refl="
                        f"{stats['reflected_area_fraction_in_circle']:.3f}"
                    ),
                )
            )
        sheet = make_contact_sheet(cells, cols=int(args.cols), cell_size=output_size)
        sheet_path = output_dir / f"{deploy_path.stem}_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet)
        deploy_record["contact_sheet"] = str(sheet_path)
        summary["deploy"][str(deploy_path)] = deploy_record
        all_sheet_cells.append(sheet)

    if all_sheet_cells:
        max_w = max(cell.shape[1] for cell in all_sheet_cells)
        normalized = []
        for cell in all_sheet_cells:
            if cell.shape[1] == max_w:
                normalized.append(cell)
                continue
            pad = np.full((cell.shape[0], max_w - cell.shape[1], 3), 255, dtype=np.uint8)
            normalized.append(np.hstack([cell, pad]))
        combined = np.vstack(
            [
                part
                for index, cell in enumerate(normalized)
                for part in (
                    ([np.full((8, max_w, 3), 255, dtype=np.uint8)] if index else [])
                    + [cell]
                )
            ]
        )
        combined_path = output_dir / "all_contact_sheets.jpg"
        cv2.imwrite(str(combined_path), combined)
        summary["all_contact_sheets"] = str(combined_path)

    summary_path = output_dir / "align_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="click cavity centers and save marks JSON")
    annotate.add_argument("--marks-json", type=Path, required=True)
    annotate.add_argument("--circle-threshold", type=int, default=8)
    annotate.add_argument("--circle-margin-px", type=float, default=0.0)
    annotate.add_argument("--grid-step", type=int, default=20)
    annotate.add_argument("images", type=Path, nargs="+")
    annotate.set_defaults(func=command_annotate)

    grid = subparsers.add_parser("grid", help="write coordinate-grid previews")
    grid.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "grid")
    grid.add_argument("--circle-threshold", type=int, default=8)
    grid.add_argument("--circle-margin-px", type=float, default=0.0)
    grid.add_argument("--grid-step", type=int, default=20)
    grid.add_argument("--cell-size", type=parse_size, default=(224, 224))
    grid.add_argument("--cols", type=int, default=4)
    grid.add_argument("images", type=Path, nargs="+")
    grid.set_defaults(func=command_grid)

    apply = subparsers.add_parser("apply", help="apply scale+translate alignment")
    apply.add_argument("--train-image", type=Path, required=True)
    apply.add_argument("--deploy-images", type=Path, nargs="+", required=True)
    apply.add_argument("--marks-json", type=Path, default=None)
    apply.add_argument("--train-center", type=parse_pair, default=None)
    apply.add_argument("--deploy-center", type=parse_pair, default=None)
    apply.add_argument("--deploy-circle", type=parse_circle, default=None)
    apply.add_argument("--center-units", choices=["auto", "pixel", "ratio"], default="auto")
    apply.add_argument("--circle-units", choices=["pixel", "ratio"], default="pixel")
    apply.add_argument("--circle-threshold", type=int, default=8)
    apply.add_argument("--circle-margin-px", type=float, default=0.0)
    apply.add_argument(
        "--circle-mode",
        choices=["fixed-source", "transformed-source", "fixed-reference", "fixed-output"],
        default="fixed-source",
    )
    apply.add_argument("--output-size", type=parse_size, default=(224, 224))
    apply.add_argument("--scales", type=parse_scales, default=parse_scales("1.0,0.95,0.92,0.90,0.87"))
    apply.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "apply")
    apply.add_argument("--cols", type=int, default=4)
    apply.set_defaults(func=command_apply)

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
