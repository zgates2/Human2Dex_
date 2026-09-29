#!/usr/bin/env python3
"""在现有的棋盘图像上验证现有的OpenCV鱼眼校准效果。

该工具保持K/D固定。它根据每张图像估计一个板面姿态
未失真角点坐标，通过鱼眼镜头将棋盘投影回原位
模型，并报告了留出数据集的重投影误差，包括图像边缘误差。
它永远不会覆盖校准文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


DEFAULT_INTRINSICS = (
    "/home/zjc/Desktop/human2dex/converted_data/params/fisheye_calib.yaml"
)
DEFAULT_CONTRACT = (
    "/home/zjc/Desktop/human2dex/converted_data/params/"
    "camera_contract_mvs_DA9057801_o6_mount_v1.yaml"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def parse_pattern_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(value) for value in text.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("pattern size must look like 7x9") from exc
    if width < 2 or height < 2:
        raise argparse.ArgumentTypeError("pattern size must contain at least 2x2 corners")
    return width, height


def load_intrinsics(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    K_value = data.get("camera_matrix", data.get("K"))
    D_value = data.get("dist_coeffs", data.get("D"))
    if K_value is None or D_value is None:
        raise ValueError(f"{path} must contain camera_matrix/dist_coeffs or K/D")
    K = np.asarray(K_value, dtype=np.float64).reshape(3, 3)
    D = np.asarray(D_value, dtype=np.float64).reshape(4, 1)
    if "image_width" in data and "image_height" in data:
        size = (int(data["image_width"]), int(data["image_height"]))
    elif "image_size" in data:
        size = tuple(int(value) for value in data["image_size"])
    else:
        raise ValueError(f"{path} must record image_width/image_height")
    if len(size) != 2:
        raise ValueError("image size must be [width, height]")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
        raise ValueError("K/D contain non-finite values")
    return {"K": K, "D": D, "image_size": size, "raw": data}


def verify_contract(contract_path: Path, intrinsics_path: Path, intrinsics: dict[str, Any]) -> dict[str, Any]:
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError(f"Expected a YAML mapping in {contract_path}")
    source = contract.get("intrinsics", {})
    expected_sha = str(source.get("source_sha256", ""))
    actual_sha = file_sha256(intrinsics_path)
    if expected_sha and expected_sha != actual_sha:
        raise ValueError(
            f"Intrinsics hash differs from camera contract: {expected_sha} != {actual_sha}"
        )
    contract_K = np.asarray(source.get("K"), dtype=np.float64).reshape(3, 3)
    contract_D = np.asarray(source.get("D"), dtype=np.float64).reshape(4, 1)
    contract_size = tuple(int(value) for value in source.get("image_size", []))
    if contract_size != intrinsics["image_size"]:
        raise ValueError("Intrinsics resolution differs from camera contract")
    if not np.allclose(contract_K, intrinsics["K"], atol=1e-9):
        raise ValueError("K differs from camera contract")
    if not np.allclose(contract_D, intrinsics["D"], atol=1e-12):
        raise ValueError("D differs from camera contract")
    return {
        "path": str(contract_path),
        "sha256": file_sha256(contract_path),
        "contract_id": contract.get("contract_id"),
        "camera_mount_id": contract.get("mount", {}).get("camera_mount_id"),
    }


def board_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    width, height = pattern_size
    points = np.zeros((width * height, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    points[:, :2] *= float(square_size)
    return points


def find_chessboard(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    corners = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, detected = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        if found:
            corners = detected
    if corners is None:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, detected = cv2.findChessboardCorners(gray, pattern_size, flags)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                60,
                1e-4,
            )
            corners = cv2.cornerSubPix(gray, detected, (11, 11), (-1, -1), criteria)
    if corners is None:
        return None
    return np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)


def estimate_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = cv2.fisheye.undistortPoints(image_points, K, D)
    identity_K = np.eye(3, dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        object_points.reshape(-1, 1, 3),
        normalized,
        identity_K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed")
    projected, _ = cv2.fisheye.projectPoints(
        object_points.reshape(-1, 1, 3), rvec, tvec, K, D
    )
    return rvec.reshape(3), tvec.reshape(3), projected.reshape(-1, 2)


def statistics(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "mean_px": None,
            "median_px": None,
            "p90_px": None,
            "p95_px": None,
            "max_px": None,
        }
    return {
        "count": int(values.size),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


def radius_normalized(points: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)
    half_diagonal = float(np.linalg.norm(center))
    return np.linalg.norm(np.asarray(points) - center, axis=1) / max(half_diagonal, 1.0)


def draw_overlay(
    image: np.ndarray,
    observed: np.ndarray,
    projected: np.ndarray,
    errors: np.ndarray,
    label: str,
) -> np.ndarray:
    overlay = image.copy()
    for observed_uv, projected_uv, error in zip(observed, projected, errors):
        observed_pt = tuple(np.round(observed_uv).astype(int))
        projected_pt = tuple(np.round(projected_uv).astype(int))
        color = (0, 210, 0) if error <= 2.0 else ((0, 190, 255) if error <= 4.0 else (0, 0, 255))
        cv2.line(overlay, observed_pt, projected_pt, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, observed_pt, 3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(overlay, projected_pt, color, cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def run_validate(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    for stale in overlays_dir.glob("*.*"):
        if stale.is_file():
            stale.unlink()

    intrinsics_path = Path(args.intrinsics).expanduser().resolve()
    intrinsics = load_intrinsics(intrinsics_path)
    contract_record = None
    if args.camera_contract:
        contract_record = verify_contract(
            Path(args.camera_contract).expanduser().resolve(),
            intrinsics_path,
            intrinsics,
        )
    K, D = intrinsics["K"], intrinsics["D"]
    image_size = intrinsics["image_size"]
    object_points = board_points(args.pattern_size, args.square_size)

    image_paths = list_images(input_dir, args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in {input_dir}")
    if args.max_images is not None:
        image_paths = image_paths[: int(args.max_images)]

    per_image = []
    all_errors = []
    all_radius = []
    detected_count = 0
    edge_frame_count = 0
    quadrant_ids: set[int] = set()
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        row: dict[str, Any] = {"image_path": str(image_path), "detected": False}
        if image is None:
            row["reason"] = "imread_failed"
            per_image.append(row)
            continue
        actual_size = (int(image.shape[1]), int(image.shape[0]))
        row["image_size"] = list(actual_size)
        if actual_size != image_size:
            row["reason"] = f"resolution_mismatch_expected_{image_size[0]}x{image_size[1]}"
            per_image.append(row)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = find_chessboard(gray, args.pattern_size)
        if corners is None:
            row["reason"] = "corners_not_found"
            per_image.append(row)
            continue
        observed = corners.reshape(-1, 2)
        try:
            rvec, tvec, projected = estimate_board_pose(object_points, corners, K, D)
        except (cv2.error, RuntimeError) as exc:
            row["reason"] = f"pose_failed:{exc}"
            per_image.append(row)
            continue
        errors = np.linalg.norm(projected - observed, axis=1)
        radii = radius_normalized(observed, image_size)
        board_center = np.mean(observed, axis=0)
        board_center_radius = float(radius_normalized(board_center[None], image_size)[0])
        if board_center_radius >= args.edge_frame_radius:
            edge_frame_count += 1
        quadrant = int(board_center[0] >= image_size[0] / 2) + 2 * int(board_center[1] >= image_size[1] / 2)
        quadrant_ids.add(quadrant)

        detected_count += 1
        all_errors.append(errors)
        all_radius.append(radii)
        row.update({
            "detected": True,
            "rvec": rvec.astype(float).tolist(),
            "tvec": tvec.astype(float).tolist(),
            "board_center_uv": board_center.astype(float).tolist(),
            "board_center_radius_normalized": board_center_radius,
            "corner_radius_min": float(np.min(radii)),
            "corner_radius_max": float(np.max(radii)),
            "error": statistics(errors),
        })
        label = (
            f"{image_path.name} median={row['error']['median_px']:.2f}px "
            f"p90={row['error']['p90_px']:.2f}px"
        )
        cv2.imwrite(str(overlays_dir / f"{image_path.stem}_overlay.jpg"), draw_overlay(image, observed, projected, errors, label))
        per_image.append(row)

    errors = np.concatenate(all_errors) if all_errors else np.empty(0, dtype=np.float64)
    radii = np.concatenate(all_radius) if all_radius else np.empty(0, dtype=np.float64)
    center_errors = errors[radii < args.edge_corner_radius] if errors.size else errors
    edge_errors = errors[radii >= args.edge_corner_radius] if errors.size else errors
    aggregate = statistics(errors)
    regions = {
        "center": statistics(center_errors),
        "edge": statistics(edge_errors),
        "edge_corner_radius_threshold": float(args.edge_corner_radius),
    }

    reasons = []
    if detected_count < args.min_valid_images:
        reasons.append(f"valid_images {detected_count} < {args.min_valid_images}")
    if edge_frame_count < args.min_edge_frames:
        reasons.append(f"edge_frames {edge_frame_count} < {args.min_edge_frames}")
    if len(quadrant_ids) < args.min_quadrants:
        reasons.append(f"covered_quadrants {len(quadrant_ids)} < {args.min_quadrants}")
    if aggregate["median_px"] is None or aggregate["median_px"] > args.max_median_px:
        reasons.append(f"global_median > {args.max_median_px:.2f}px")
    if aggregate["p90_px"] is None or aggregate["p90_px"] > args.max_p90_px:
        reasons.append(f"global_p90 > {args.max_p90_px:.2f}px")
    if regions["edge"]["count"] == 0:
        reasons.append("no edge corners")
    elif regions["edge"]["p90_px"] > args.max_edge_p90_px:
        reasons.append(f"edge_p90 > {args.max_edge_p90_px:.2f}px")

    report = {
        "format_version": 1,
        "tool": "fisheye_validate.py",
        "input_dir": str(input_dir),
        "intrinsics": {
            "path": str(intrinsics_path),
            "sha256": file_sha256(intrinsics_path),
            "image_size": list(image_size),
            "K": K.astype(float).tolist(),
            "D": D.reshape(-1).astype(float).tolist(),
        },
        "camera_contract": contract_record,
        "board": {
            "pattern_size": list(args.pattern_size),
            "square_size": float(args.square_size),
        },
        "coverage": {
            "input_images": len(image_paths),
            "valid_images": detected_count,
            "edge_frames": edge_frame_count,
            "covered_quadrants": sorted(quadrant_ids),
        },
        "thresholds": {
            "min_valid_images": args.min_valid_images,
            "min_edge_frames": args.min_edge_frames,
            "min_quadrants": args.min_quadrants,
            "max_median_px": args.max_median_px,
            "max_p90_px": args.max_p90_px,
            "max_edge_p90_px": args.max_edge_p90_px,
        },
        "aggregate": aggregate,
        "regions": regions,
        "passed": not reasons,
        "failure_reasons": reasons,
        "per_image": per_image,
    }
    report_path = output_dir / "fisheye_validation_report.json"
    atomic_write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path),
        "valid_images": detected_count,
        "edge_frames": edge_frame_count,
        "aggregate": aggregate,
        "edge": regions["edge"],
        "passed": report["passed"],
        "failure_reasons": reasons,
    }, ensure_ascii=False, indent=2))


def run_self_test(args: argparse.Namespace) -> None:
    intrinsics_path = Path(args.intrinsics).expanduser().resolve()
    intrinsics = load_intrinsics(intrinsics_path)
    if args.camera_contract:
        verify_contract(
            Path(args.camera_contract).expanduser().resolve(),
            intrinsics_path,
            intrinsics,
        )
    K, D = intrinsics["K"], intrinsics["D"]
    object_points = board_points((7, 9), 0.025)
    rng = np.random.default_rng(13)
    recovered_errors = []
    for index in range(12):
        rvec = np.array([0.15 + index * 0.015, -0.25 + index * 0.035, 0.04], dtype=np.float64)
        tvec = np.array([-0.08 + index * 0.014, -0.06 + index * 0.008, 0.34 + index * 0.006], dtype=np.float64)
        projected, _ = cv2.fisheye.projectPoints(object_points.reshape(-1, 1, 3), rvec, tvec, K, D)
        observed = projected + rng.normal(0.0, 0.18, size=projected.shape)
        _, _, reprojection = estimate_board_pose(object_points, observed, K, D)
        recovered_errors.extend(np.linalg.norm(reprojection - observed.reshape(-1, 2), axis=1))
    stats = statistics(np.asarray(recovered_errors))
    print(json.dumps({"self_test": stats}, indent=2))
    if stats["median_px"] is None or stats["median_px"] > 0.5 or stats["p90_px"] > 0.8:
        raise RuntimeError("Synthetic fisheye validation self-test failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate fixed K/D on fresh images")
    validate.add_argument("--input-dir", required=True)
    validate.add_argument("--pattern-size", type=parse_pattern_size, required=True)
    validate.add_argument("--square-size", type=float, required=True, help="board square size; unit does not affect pixel validation")
    validate.add_argument("--output-dir", required=True)
    validate.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    validate.add_argument("--camera-contract", default=DEFAULT_CONTRACT)
    validate.add_argument("--recursive", action="store_true")
    validate.add_argument("--max-images", type=int, default=None)
    validate.add_argument("--edge-corner-radius", type=float, default=0.62)
    validate.add_argument("--edge-frame-radius", type=float, default=0.48)
    validate.add_argument("--min-valid-images", type=int, default=15)
    validate.add_argument("--min-edge-frames", type=int, default=4)
    validate.add_argument("--min-quadrants", type=int, default=4)
    validate.add_argument("--max-median-px", type=float, default=2.0)
    validate.add_argument("--max-p90-px", type=float, default=4.0)
    validate.add_argument("--max-edge-p90-px", type=float, default=6.0)
    validate.set_defaults(func=run_validate)

    self_test = subparsers.add_parser("self-test", help="synthetic projection/PnP test")
    self_test.add_argument("--intrinsics", default=DEFAULT_INTRINSICS)
    self_test.add_argument("--camera-contract", default=DEFAULT_CONTRACT)
    self_test.set_defaults(func=run_self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
