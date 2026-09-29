#!/usr/bin/env python3
"""05: Track configured task objects with SAM3 and backfill dense raw fields.

This is independent from the SAM3 hand-mask branch used by appearance
augmentation.  It never reads or writes the hand masks.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline_common import get, load_config, path, write_manifest


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def field_prefix(spec: dict[str, Any]) -> str:
    return str(spec.get("field_prefix") or f"task_object_{safe_name(str(spec['name']))}")


def read_pkl(source: Path) -> dict[str, Any]:
    try:
        importlib.import_module("numpy._core")
    except ImportError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.numeric", core.numeric)
    with source.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"Not a DexUMI PKL: {source}")
    return value


def atomic_pickle(value: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def resolve_image(pkl_path: Path, message: dict[str, Any], image_field: str) -> Path:
    value = message.get(image_field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {image_field} in {pkl_path}")
    candidate = Path(value)
    result = candidate if candidate.is_absolute() else pkl_path.parent / candidate
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def make_numeric_frame_dir(
    pkl_path: Path,
    messages: list[dict[str, Any]],
    image_field: str,
    temp_root: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path, list[Path]]:
    holder = tempfile.TemporaryDirectory(prefix=f"{pkl_path.parent.name}_", dir=temp_root)
    frame_dir = Path(holder.name)
    paths: list[Path] = []
    for index, message in enumerate(messages):
        source = resolve_image(pkl_path, message, image_field)
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        destination = frame_dir / f"{index:08d}{suffix}"
        os.symlink(source, destination)
        paths.append(source)
    return holder, frame_dir, paths


def choose_object(
    outputs: dict[str, Any],
    image_area: int,
    spec: dict[str, Any],
) -> tuple[int, float, int]:
    obj_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
    scores = np.asarray(outputs.get("out_probs", []), dtype=np.float64).reshape(-1)
    if len(scores) != len(obj_ids):
        scores = np.ones(len(obj_ids), dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for index, obj_id in enumerate(obj_ids):
        area = int(np.count_nonzero(masks[index])) if index < len(masks) else 0
        score = float(scores[index])
        if area < int(spec["seed_min_area"]):
            continue
        if area > float(spec["max_area_ratio"]) * image_area:
            continue
        if score < float(spec["min_confidence"]):
            continue
        candidates.append((score, area, int(obj_id)))
    if not candidates:
        raise RuntimeError("no plausible SAM3 seed mask")
    return max(candidates)


def find_seed(
    predictor: Any,
    session_id: str,
    frame_dir: Path,
    frame_count: int,
    spec: dict[str, Any],
) -> tuple[int, int, float, int, int]:
    stride = max(1, int(spec["prompt_search_stride"]))
    search_limit = frame_count
    configured_limit = int(spec.get("prompt_search_max_frames", 0))
    if configured_limit > 0:
        search_limit = min(search_limit, configured_limit)
    candidates = list(range(0, search_limit, stride))
    if candidates and candidates[-1] != search_limit - 1:
        candidates.append(search_limit - 1)
    if not candidates:
        raise RuntimeError("empty episode")
    for attempt, frame_index in enumerate(candidates, start=1):
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": str(spec["prompt"]),
            }
        )
        image = cv2.imread(str(sorted(frame_dir.iterdir())[frame_index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode seed frame {frame_index}")
        try:
            score, area, object_id = choose_object(
                response.get("outputs", {}),
                int(image.shape[0] * image.shape[1]),
                spec,
            )
            return frame_index, object_id, score, area, attempt
        except RuntimeError:
            predictor.handle_request({"type": "reset_session", "session_id": session_id})
    raise RuntimeError(
        f"prompt={spec['prompt']!r} found no seed in {len(candidates)} frames"
    )


def output_for_id(outputs: dict[str, Any], object_id: int) -> tuple[np.ndarray | None, float]:
    ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(ids == int(object_id))
    masks = np.asarray(outputs.get("out_binary_masks", []), dtype=bool)
    scores = np.asarray(outputs.get("out_probs", []), dtype=np.float64).reshape(-1)
    if len(matches) == 0:
        # A reset after repeated text-prompt attempts can renumber the single
        # tracked object between add_prompt and propagation. Each session here
        # intentionally contains exactly one task object, so this fallback is
        # unambiguous and avoids turning a valid track into 100% missing.
        if len(masks) > 0:
            index = int(np.argmax(scores[: len(masks)])) if len(scores) else 0
        else:
            return None, 0.0
    else:
        index = int(matches[0])
    if index >= len(masks):
        return None, 0.0
    return masks[index], float(scores[index]) if index < len(scores) else 1.0


def mask_stats(mask: np.ndarray) -> tuple[list[float], int, list[int]]:
    binary = np.asarray(mask, dtype=bool).squeeze()
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        raise ValueError("empty mask")
    return (
        [float(xs.mean()), float(ys.mean())],
        int(len(xs)),
        [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    )


def make_row_from_outputs(
    outputs: dict[str, Any],
    object_id: int,
    image_area: int,
    spec: dict[str, Any],
    seed_frame: int,
) -> dict[str, Any]:
    mask, confidence = output_for_id(outputs, object_id)
    center: list[float] | None = None
    area = 0
    bbox: list[int] | None = None
    error: str | None = None
    if mask is None or not np.any(mask):
        error = "object_missing"
    else:
        center, area, bbox = mask_stats(mask)
        if area < int(spec["min_area"]):
            error = "mask_too_small"
        elif area > float(spec["max_area_ratio"]) * int(image_area):
            error = "mask_too_large"
        elif confidence < float(spec["min_confidence"]):
            error = "low_confidence"
    visible = error is None
    return {
        "uv": center if visible else None,
        "mask_area": area if visible else 0,
        "confidence": float(np.clip(confidence, 0.0, 1.0)) if visible else 0.0,
        "visible": visible,
        "bbox_xyxy": bbox if visible else None,
        "seed_frame": int(seed_frame),
        "error": error,
        "reseeded": False,
    }


def jump_error(
    row: dict[str, Any],
    last_uv: tuple[float, float] | None,
    last_area: float,
    spec: dict[str, Any],
) -> str | None:
    if not bool(spec.get("jump_filter_enabled", True)):
        return None
    if not bool(row.get("visible", False)) or last_uv is None or last_area <= 0:
        return None
    uv = row.get("uv")
    if not isinstance(uv, list) or len(uv) != 2:
        return "jump_bad_uv"
    center_distance = float(np.hypot(float(uv[0]) - last_uv[0], float(uv[1]) - last_uv[1]))
    area = max(float(row.get("mask_area", 0.0) or 0.0), 1.0)
    center_threshold = max(
        float(spec.get("jump_center_px", 60.0)),
        float(spec.get("jump_center_area_sqrt_factor", 3.0)) * float(np.sqrt(max(last_area, area))),
    )
    if center_distance > center_threshold:
        return f"center_jump_{center_distance:.1f}px"
    ratio = area / max(last_area, 1.0)
    if ratio > float(spec.get("jump_area_ratio_max", 2.5)):
        return f"area_jump_x{ratio:.2f}"
    if ratio < float(spec.get("jump_area_ratio_min", 0.35)):
        return f"area_drop_x{ratio:.2f}"
    return None


def prompt_one_frame(
    predictor: Any,
    frame_dir: Path,
    frame_index: int,
    frame_count: int,
    spec: dict[str, Any],
) -> dict[str, Any] | None:
    session_id = str(
        predictor.handle_request(
            {"type": "start_session", "resource_path": str(frame_dir)}
        )["session_id"]
    )
    try:
        response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(frame_index),
                "text": str(spec["prompt"]),
            }
        )
        image = cv2.imread(str(sorted(frame_dir.iterdir())[frame_index]), cv2.IMREAD_COLOR)
        if image is None:
            return None
        confidence, area, object_id = choose_object(
            response.get("outputs", {}),
            int(image.shape[0] * image.shape[1]),
            spec,
        )
        row = make_row_from_outputs(
            response.get("outputs", {}),
            object_id,
            int(image.shape[0] * image.shape[1]),
            spec,
            frame_index,
        )
        if not bool(row.get("visible", False)):
            return None
        row["confidence"] = float(np.clip(confidence, 0.0, 1.0))
        row["mask_area"] = int(area)
        row["seed_frame"] = int(frame_index)
        row["reseeded"] = True
        row["reseed_source"] = "jump_prompt"
        row["error"] = None
        return row
    except Exception:
        return None
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id, "run_gc_collect": False}
        )


def invalidate_jump_row(row: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        **row,
        "uv": None,
        "mask_area": 0,
        "confidence": 0.0,
        "visible": False,
        "bbox_xyxy": None,
        "error": error,
        "reseeded": bool(row.get("reseeded", False)),
    }


def track_object(
    predictor: Any,
    frame_dir: Path,
    frame_count: int,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
    session_id = str(response["session_id"])
    try:
        seed_frame, object_id, seed_confidence, seed_area, attempts = find_seed(
            predictor, session_id, frame_dir, frame_count, spec
        )
        # Prompt search can touch many failed candidates. A completely fresh
        # tracking session is required; reset_session alone does not reliably
        # clear all propagation state for a late-found object.
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id, "run_gc_collect": False}
        )
        session_id = str(
            predictor.handle_request(
                {"type": "start_session", "resource_path": str(frame_dir)}
            )["session_id"]
        )
        seed_response = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": seed_frame,
                "text": str(spec["prompt"]),
            }
        )
        seed_image = cv2.imread(str(sorted(frame_dir.iterdir())[seed_frame]), cv2.IMREAD_COLOR)
        if seed_image is None:
            raise RuntimeError(f"Cannot decode clean seed frame {seed_frame}")
        seed_confidence, seed_area, object_id = choose_object(
            seed_response.get("outputs", {}),
            int(seed_image.shape[0] * seed_image.shape[1]),
            spec,
        )
        seed_mask, _ = output_for_id(seed_response.get("outputs", {}), object_id)
        if seed_mask is None or not np.any(seed_mask):
            raise RuntimeError("clean seed prompt returned no mask")
        seed_uv, seed_mask_area, seed_bbox = mask_stats(seed_mask)
        stream = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "both",
                "start_frame_index": seed_frame,
                "max_frame_num_to_track": frame_count,
                # Keep propagation permissive, then apply configured checks.
                "output_prob_thresh": float(spec.get("propagation_output_threshold", 0.0)),
            }
        )
        results = {
            int(item["frame_index"]): item.get("outputs", {})
            for item in stream
            if item.get("frame_index") is not None
            and 0 <= int(item["frame_index"]) < frame_count
        }
        frame_paths = sorted(frame_dir.iterdir())
        first_image = cv2.imread(str(frame_paths[seed_frame]), cv2.IMREAD_COLOR)
        if first_image is None:
            raise RuntimeError(f"Cannot decode frame {seed_frame}")
        image_area = int(first_image.shape[0] * first_image.shape[1])
        rows: list[dict[str, Any]] = []
        valid_count = 0
        jump_rejected = 0
        reseed_attempts = 0
        reseed_success = 0
        max_reseeds = max(0, int(spec.get("jump_reseed_max_attempts", 24)))
        last_uv: tuple[float, float] | None = None
        last_area = 0.0
        for frame_index in range(frame_count):
            row = make_row_from_outputs(
                results.get(frame_index, {}),
                object_id,
                image_area,
                spec,
                seed_frame,
            )
            error = jump_error(row, last_uv, last_area, spec)
            if error is not None:
                reseeded = None
                if reseed_attempts < max_reseeds:
                    reseed_attempts += 1
                    reseeded = prompt_one_frame(
                        predictor,
                        frame_dir,
                        frame_index,
                        frame_count,
                        spec,
                    )
                if reseeded is not None and jump_error(reseeded, last_uv, last_area, spec) is None:
                    row = reseeded
                    reseed_success += 1
                else:
                    jump_rejected += 1
                    suffix = "reseed_failed" if reseeded is None else "reseed_unstable"
                    row = invalidate_jump_row(row, f"{error}_{suffix}")
            if bool(row.get("visible", False)):
                valid_count += 1
                uv = row.get("uv")
                assert isinstance(uv, list) and len(uv) == 2
                last_uv = (float(uv[0]), float(uv[1]))
                last_area = float(row.get("mask_area", 0.0) or 0.0)
            rows.append(row)
        if not rows[seed_frame]["visible"]:
            rows[seed_frame] = {
                "uv": seed_uv,
                "mask_area": int(seed_mask_area),
                "confidence": float(np.clip(seed_confidence, 0.0, 1.0)),
                "visible": True,
                "bbox_xyxy": seed_bbox,
                "seed_frame": int(seed_frame),
                "error": None,
                "reseeded": False,
            }
            valid_count += 1
        summary = {
            "name": str(spec["name"]),
            "prompt": str(spec["prompt"]),
            "seed_frame": int(seed_frame),
            "seed_confidence": float(seed_confidence),
            "seed_area": int(seed_area),
            "prompt_attempts": int(attempts),
            "frames": frame_count,
            "visible_frames": valid_count,
            "jump_rejected_frames": int(jump_rejected),
            "jump_reseed_attempts": int(reseed_attempts),
            "jump_reseed_success": int(reseed_success),
        }
        return rows, summary
    finally:
        predictor.handle_request(
            {"type": "close_session", "session_id": session_id, "run_gc_collect": False}
        )


def fill_failed(messages: list[dict[str, Any]], spec: dict[str, Any], error: str) -> None:
    prefix = field_prefix(spec)
    for message in messages:
        message[f"{prefix}_raw_uv"] = None
        message[f"{prefix}_raw_mask_area"] = 0
        message[f"{prefix}_raw_confidence"] = 0.0
        message[f"{prefix}_raw_visible"] = False
        message[f"{prefix}_raw_bbox_xyxy"] = None
        message[f"{prefix}_raw_seed_frame"] = -1
        message[f"{prefix}_raw_error"] = error


def write_rows(messages: list[dict[str, Any]], spec: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    prefix = field_prefix(spec)
    for message, row in zip(messages, rows):
        for key, value in row.items():
            message[f"{prefix}_raw_{key}" if key not in {"seed_frame", "error"} else f"{prefix}_raw_{key}"] = value


def episode_complete(messages: list[dict[str, Any]], objects: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    return all(f"{field_prefix(spec)}_raw_visible" in messages[0] for spec in objects)


def process_episode(
    predictor: Any,
    pkl_path: Path,
    objects: list[dict[str, Any]],
    image_field: str,
    temp_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    data = read_pkl(pkl_path)
    messages = data["messages"]
    if not overwrite and episode_complete(messages, objects):
        return {"pkl": str(pkl_path), "status": "skipped", "frames": len(messages)}
    holder, frame_dir, _ = make_numeric_frame_dir(
        pkl_path, messages, image_field, temp_root
    )
    summaries: list[dict[str, Any]] = []
    try:
        for spec in objects:
            try:
                rows, summary = track_object(predictor, frame_dir, len(messages), spec)
                write_rows(messages, spec, rows)
                summaries.append(summary)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                fill_failed(messages, spec, error)
                summaries.append({"name": spec["name"], "prompt": spec["prompt"], "error": error})
        metadata = data.setdefault("metadata", {})
        metadata["taskObjectTrackingV1"] = {
            "objects": summaries,
            "image_field": image_field,
            "causal_note": "bidirectional raw masks; stage 06 suppresses all frames before seed",
        }
        atomic_pickle(data, pkl_path)
    finally:
        holder.cleanup()
    return {"pkl": str(pkl_path), "status": "written", "frames": len(messages), "objects": summaries}


def gpu_worker(
    worker_index: int,
    device: int,
    pkl_paths: list[str],
    compact: dict[str, Any],
    report_path: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/sam3_task_objects_gpu{device}")
    sam3_code = str(compact["sam3_code"])
    if sam3_code not in sys.path:
        sys.path.insert(0, sam3_code)
    import torch
    from sam3 import build_sam3_predictor

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    predictor = build_sam3_predictor(
        version=str(compact["version"]),
        checkpoint_path=str(compact["checkpoint"]),
        compile=bool(compact["compile"]),
        async_loading_frames=False,
    )
    temp_root = Path(str(compact["temp_root"]))
    temp_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, pkl_text in enumerate(pkl_paths, start=1):
        result = process_episode(
            predictor=predictor,
            pkl_path=Path(pkl_text),
            objects=compact["objects"],
            image_field=str(compact["image_field"]),
            temp_root=temp_root,
            overwrite=bool(compact["overwrite"]),
        )
        results.append(result)
        print(
            f"[05 gpu={device} worker={worker_index}] {index}/{len(pkl_paths)} "
            f"{Path(pkl_text).parent.name} {result['status']}",
            flush=True,
        )
    Path(report_path).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_objects(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = {
        "prompt_search_stride": int(get(config, "task_object_tracking.prompt_search_stride", 10)),
        "prompt_search_max_frames": int(get(config, "task_object_tracking.prompt_search_max_frames", 0)),
        "min_confidence": float(get(config, "task_object_tracking.min_confidence", 0.20)),
        "min_area": int(get(config, "task_object_tracking.min_area", 25)),
        "seed_min_area": int(get(config, "task_object_tracking.seed_min_area", 150)),
        "max_area_ratio": float(get(config, "task_object_tracking.max_area_ratio", 0.35)),
    }
    objects = get(config, "task_objects", required=True)
    if not isinstance(objects, list) or not objects:
        raise ValueError("task_objects must be a non-empty list")
    normalized = []
    for item in objects:
        if not isinstance(item, dict) or not item.get("name") or not item.get("prompt"):
            raise ValueError("Every task object needs name and prompt")
        normalized.append({**defaults, **item})
    prefixes = [field_prefix(item) for item in normalized]
    if len(prefixes) != len(set(prefixes)):
        raise ValueError(f"Duplicate task-object field prefixes: {prefixes}")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument(
        "--object",
        action="append",
        default=[],
        help="Only process a configured object name; repeatable, useful for targeted resume",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    base_root = path(config, "paths.base_root")
    reports_root = path(config, "paths.task_object_reports_root")
    pkls = sorted(base_root.rglob("*.pkl"))
    if args.limit_episodes is not None:
        pkls = pkls[: max(0, args.limit_episodes)]
    if not pkls:
        raise FileNotFoundError(f"No PKLs under {base_root}")
    devices = [int(item) for item in get(config, "parallel.task_object_devices", list(range(8)))]
    if not devices:
        raise ValueError("parallel.task_object_devices is empty")
    objects = normalize_objects(config)
    if args.object:
        requested = set(args.object)
        objects = [item for item in objects if str(item["name"]) in requested]
        missing = requested - {str(item["name"]) for item in objects}
        if missing:
            raise ValueError(f"Unknown configured objects: {sorted(missing)}")
    compact = {
        "sam3_code": str(path(config, "task_object_tracking.sam3_code")),
        "checkpoint": str(path(config, "task_object_tracking.checkpoint")),
        "version": str(get(config, "task_object_tracking.version", "sam3")),
        "compile": bool(get(config, "task_object_tracking.compile", False)),
        "image_field": str(get(config, "fields.source_image", "rgbImage")),
        "temp_root": str(get(config, "task_object_tracking.temp_root", "/tmp/human2dex_task_objects")),
        "objects": objects,
        "overwrite": bool(args.overwrite or get(config, "task_object_tracking.overwrite", False)),
    }
    print(json.dumps({"base_root": str(base_root), "reports_root": str(reports_root), "episodes": len(pkls), "devices": devices, "objects": objects}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    reports_root.mkdir(parents=True, exist_ok=True)
    shards = [pkls[index::len(devices)] for index in range(len(devices))]
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    report_paths: list[Path] = []
    for worker_index, (device, shard) in enumerate(zip(devices, shards)):
        if not shard:
            continue
        report = reports_root / f"worker_{worker_index:02d}_gpu_{device}.json"
        report_paths.append(report)
        process = context.Process(
            target=gpu_worker,
            args=(worker_index, device, [str(item) for item in shard], compact, str(report)),
        )
        process.start()
        processes.append(process)
    failures = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failures.append({"pid": process.pid, "exitcode": process.exitcode})
    if failures:
        raise RuntimeError(f"SAM3 task-object workers failed: {failures}")
    results = []
    for report in report_paths:
        results.extend(json.loads(report.read_text(encoding="utf-8")))
    summary = {"base_root": str(base_root), "episodes": len(pkls), "devices": devices, "objects": objects, "results": results}
    (reports_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(base_root, "05_track_task_objects", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
