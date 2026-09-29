#!/usr/bin/env python3
"""06: Build causal, deployment-matched object-to-pocket observations.

Dense SAM3 results from stage 05 are sampled at the configured deployment
tracker rate.  Between updates the last center/area are held, confidence
decays smoothly, and no future result is exposed before its causal seed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def finite_pair(value: Any) -> tuple[float, float] | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if array.shape != (2,) or not np.isfinite(array).all():
        return None
    return float(array[0]), float(array[1])


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def resolve_reference_areas(
    pkls: list[Path], objects: list[dict[str, Any]]
) -> dict[str, float]:
    values: dict[str, list[float]] = {str(item["name"]): [] for item in objects}
    unresolved = {
        str(item["name"])
        for item in objects
        if item.get("reference_area_px2") in {None, "auto", "auto_median"}
    }
    if unresolved:
        for pkl_path in pkls:
            messages = read_pkl(pkl_path)["messages"]
            for spec in objects:
                name = str(spec["name"])
                if name not in unresolved:
                    continue
                prefix = field_prefix(spec)
                for index, message in enumerate(messages):
                    seed = int_or_default(message.get(f"{prefix}_raw_seed_frame"), -1)
                    if index < seed or not bool(message.get(f"{prefix}_raw_visible", False)):
                        continue
                    area = float(message.get(f"{prefix}_raw_mask_area", 0.0) or 0.0)
                    if math.isfinite(area) and area > 0:
                        values[name].append(area)
    result: dict[str, float] = {}
    for spec in objects:
        name = str(spec["name"])
        configured = spec.get("reference_area_px2")
        if configured not in {None, "auto", "auto_median"}:
            area = float(configured)
        elif values[name]:
            area = float(np.median(np.asarray(values[name], dtype=np.float64)))
        elif spec.get("fallback_reference_area_px2") is not None:
            area = float(spec["fallback_reference_area_px2"])
        else:
            raise ValueError(f"Cannot resolve reference_area_px2 for {name}")
        if not math.isfinite(area) or area <= 0:
            raise ValueError(f"Invalid reference area for {name}: {area}")
        result[name] = area
    return result


def deterministic_seed(base_seed: int, pkl_path: Path) -> int:
    digest = hashlib.sha256(str(pkl_path).encode("utf-8")).digest()
    return int(base_seed + int.from_bytes(digest[:4], "little"))


def reference_scale_for(
    spec: dict[str, Any],
    obs_config: dict[str, Any],
    reference_area: float,
) -> float:
    configured = spec.get("reference_scale_px", obs_config["reference_scale_px"])
    if configured in {None, "auto", "auto_sqrt_area", "sqrt_reference_area"}:
        return float(math.sqrt(reference_area))
    return float(configured)


def process_pkl(
    pkl_text: str,
    objects: list[dict[str, Any]],
    reference_areas: dict[str, float],
    obs_config: dict[str, Any],
    fields: dict[str, str],
) -> dict[str, Any]:
    pkl_path = Path(pkl_text)
    data = read_pkl(pkl_path)
    messages = data["messages"]
    source_fps = float(obs_config["source_fps"])
    tracker_hz = float(obs_config["tracker_hz"])
    base_interval = source_fps / tracker_hz
    latency = max(0, int(obs_config["latency_frames"]))
    jitter = max(0, int(obs_config["jitter_frames"]))
    dropout = float(obs_config["dropout_prob"])
    tau = max(float(obs_config["confidence_tau_seconds"]), 1e-6)
    clip_log = float(obs_config["log_area_clip"])
    rng = np.random.default_rng(deterministic_seed(int(obs_config["seed"]), pkl_path))

    state = {
        str(spec["name"]): {
            "uv": None,
            "area": 0.0,
            "confidence": 0.0,
            "last_seen": None,
            "next_update": latency,
        }
        for spec in objects
    }
    memory_frames = {str(spec["name"]): 0 for spec in objects}
    valid_frames = 0

    for frame_index, message in enumerate(messages):
        pocket = finite_pair(message.get(fields["pocket_uv"]))
        pocket_valid = bool(message.get(fields["pocket_valid"], False)) and pocket is not None
        pocket_confidence = float(message.get(fields["pocket_confidence"], 0.0) or 0.0)
        combined: list[float] = []
        required_memory_valid = True

        for spec in objects:
            name = str(spec["name"])
            prefix = field_prefix(spec)
            current = state[name]
            policy_visible = False
            update_source_index = -1
            episode_seed = int_or_default(
                messages[0].get(f"{prefix}_raw_seed_frame") if messages else None,
                -1,
            )
            seed_due = (
                current["uv"] is None
                and episode_seed >= 0
                and frame_index >= episode_seed + latency
            )
            if seed_due or frame_index >= int(current["next_update"]):
                update_source_index = episode_seed if seed_due else max(0, frame_index - latency)
                raw = messages[update_source_index]
                seed_frame = int_or_default(raw.get(f"{prefix}_raw_seed_frame"), -1)
                raw_visible = bool(raw.get(f"{prefix}_raw_visible", False))
                raw_uv = finite_pair(raw.get(f"{prefix}_raw_uv"))
                raw_area = float(raw.get(f"{prefix}_raw_mask_area", 0.0) or 0.0)
                raw_conf = float(raw.get(f"{prefix}_raw_confidence", 0.0) or 0.0)
                dropped = bool(rng.random() < dropout)
                if (
                    update_source_index >= seed_frame >= 0
                    and raw_visible
                    and raw_uv is not None
                    and raw_area > 0
                    and not dropped
                ):
                    current["uv"] = raw_uv
                    current["area"] = raw_area
                    current["confidence"] = float(np.clip(raw_conf, 0.0, 1.0))
                    current["last_seen"] = frame_index
                    policy_visible = True
                interval = max(1, int(round(base_interval)))
                if jitter:
                    interval = max(1, interval + int(rng.integers(-jitter, jitter + 1)))
                current["next_update"] = frame_index + interval

            memory_valid = current["uv"] is not None and current["last_seen"] is not None
            if bool(spec.get("required_for_qc", True)):
                required_memory_valid = required_memory_valid and memory_valid
            if memory_valid:
                age_frames = frame_index - int(current["last_seen"])
                effective_confidence = float(current["confidence"]) * math.exp(
                    -(age_frames / source_fps) / tau
                )
                if bool(obs_config["combine_pocket_confidence"]):
                    effective_confidence = min(
                        effective_confidence,
                        float(np.clip(pocket_confidence, 0.0, 1.0)),
                    )
                held_uv = current["uv"]
                assert held_uv is not None
                delta = None if pocket is None else [held_uv[0] - pocket[0], held_uv[1] - pocket[1]]
                distance = None if delta is None else float(math.hypot(delta[0], delta[1]))
                frames_since_seen = age_frames
            else:
                effective_confidence = 0.0
                held_uv = None
                delta = None
                distance = None
                frames_since_seen = -1

            reference_area = float(reference_areas[name])
            reference_scale = reference_scale_for(spec, obs_config, reference_area)
            usable = memory_valid and pocket_valid and reference_scale > 0 and reference_area > 0
            if usable:
                assert delta is not None
                dx = float(delta[0] / reference_scale)
                dy = float(delta[1] / reference_scale)
                log_area = float(np.clip(math.log(max(float(current["area"]), 1.0) / reference_area), -clip_log, clip_log))
                object_obs = [dx, dy, log_area, effective_confidence, 1.0]
                memory_frames[name] += 1
                error = None
            else:
                object_obs = [0.0, 0.0, 0.0, 0.0, 0.0]
                error = "memory_unavailable" if not memory_valid else "pocket_invalid"

            message[f"{prefix}_policy_uv"] = None if held_uv is None else [float(held_uv[0]), float(held_uv[1])]
            message[f"{prefix}_policy_mask_area"] = float(current["area"]) if memory_valid else 0.0
            message[f"{prefix}_policy_confidence"] = float(effective_confidence)
            message[f"{prefix}_policy_visible"] = bool(policy_visible)
            message[f"{prefix}_memory_valid"] = bool(memory_valid)
            message[f"{prefix}_frames_since_seen"] = int(frames_since_seen)
            message[f"{prefix}_update_source_frame"] = int(update_source_index)
            message[f"{prefix}_pocket_delta_uv"] = delta
            message[f"{prefix}_pocket_distance_px"] = distance
            message[f"{prefix}_obs"] = object_obs
            message[f"{prefix}_error"] = error
            combined.extend(object_obs)

        combined_valid = bool(pocket_valid and required_memory_valid)
        message[fields["combined_obs"]] = combined
        message[fields["combined_valid"]] = combined_valid
        message[fields["combined_error"]] = None if combined_valid else "missing_pocket_or_object_memory"
        valid_frames += int(combined_valid)

    metadata = data.setdefault("metadata", {})
    metadata["objectPocketObservationV1"] = {
        "object_order": [str(spec["name"]) for spec in objects],
        "dimension_per_object": 5,
        "combined_field": fields["combined_obs"],
        "reference_areas_px2": reference_areas,
        "config": obs_config,
        "semantics": ["dx", "dy", "log_area_ratio", "confidence", "memory_valid"],
    }
    atomic_pickle(data, pkl_path)
    return {
        "pkl": str(pkl_path),
        "frames": len(messages),
        "combined_valid_frames": valid_frames,
        "memory_frames": memory_frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit-episodes", type=int, default=None)
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
    objects = get(config, "task_objects", required=True)
    if not isinstance(objects, list) or not objects:
        raise ValueError("task_objects must be a non-empty list")
    reference_areas = resolve_reference_areas(pkls, objects)
    obs_config = {
        "source_fps": float(get(config, "object_observation.source_fps", 30.0)),
        "tracker_hz": float(get(config, "object_observation.tracker_hz", 4.0)),
        "latency_frames": int(get(config, "object_observation.latency_frames", 0)),
        "jitter_frames": int(get(config, "object_observation.jitter_frames", 1)),
        "dropout_prob": float(get(config, "object_observation.dropout_prob", 0.0)),
        "confidence_tau_seconds": float(get(config, "object_observation.confidence_tau_seconds", 1.0)),
        "reference_scale_px": get(config, "object_observation.reference_scale_px", 37.7),
        "log_area_clip": float(get(config, "object_observation.log_area_clip", 4.0)),
        "combine_pocket_confidence": bool(get(config, "object_observation.combine_pocket_confidence", True)),
        "seed": int(get(config, "object_observation.seed", 20260807)),
    }
    if obs_config["source_fps"] <= 0 or obs_config["tracker_hz"] <= 0:
        raise ValueError("source_fps and tracker_hz must be positive")
    fields = {
        "pocket_uv": str(get(config, "fields.pocket_uv", "wrist_grasp_pocket_uv")),
        "pocket_valid": str(get(config, "fields.pocket_valid", "wrist_grasp_pocket_valid")),
        "pocket_confidence": str(get(config, "fields.pocket_confidence", "wrist_grasp_pocket_confidence")),
        "combined_obs": str(get(config, "fields.object_pocket_obs", "objectPocketObs")),
        "combined_valid": str(get(config, "fields.object_pocket_valid", "objectPocketObsValid")),
        "combined_error": str(get(config, "fields.object_pocket_error", "objectPocketObsError")),
    }
    workers = max(1, min(args.workers or int(get(config, "parallel.object_obs_workers", 32)), len(pkls)))
    print(json.dumps({"base_root": str(base_root), "episodes": len(pkls), "workers": workers, "objects": objects, "reference_areas_px2": reference_areas, "observation": obs_config, "fields": fields}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_pkl, str(item), objects, reference_areas, obs_config, fields) for item in pkls]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index == 1 or index % 25 == 0 or index == len(futures):
                print(f"[06] {index}/{len(futures)}", flush=True)
    summary = {"base_root": str(base_root), "episodes": len(pkls), "objects": objects, "reference_areas_px2": reference_areas, "observation": obs_config, "results": results}
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "object_pocket_obs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(base_root, "06_build_object_pocket_obs", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
