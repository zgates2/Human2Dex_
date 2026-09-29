#!/usr/bin/env python3
"""07: Validate G/L+labels and publish target-specific contiguous segments.

Invalid pocket frames are never silently centered or kept inside a policy
episode.  Instead each maximal valid run becomes a new, hard-link-backed
episode in ``qc_segments_root/{o6,wuji}``.  The source appearance/G-L data is
left untouched.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline_common import get, load_config, path, require_new_directory, write_manifest


def read_pkl(source: Path) -> dict[str, Any]:
    try:
        importlib.import_module("numpy._core")
    except ImportError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core); sys.modules.setdefault("numpy._core.numeric", core.numeric)
    with source.open("rb") as handle: value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"Not a DexUMI PKL: {source}")
    return value


def atomic_pickle(value: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass


def hardlink_or_copy(source: str, destination: str) -> str:
    try: os.link(source, destination)
    except OSError: shutil.copy2(source, destination)
    return destination


def finite_vector(value: Any, minimum_size: int) -> bool:
    try: array = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception: return False
    return bool(array.size >= minimum_size and np.isfinite(array).all())


def resolve_image(pkl_path: Path, message: dict[str, Any], field: str) -> Path | None:
    value = message.get(field)
    if not isinstance(value, str) or not value: return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else pkl_path.parent / candidate


def valid_frame(pkl_path: Path, message: Any, spec: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(message, dict): return False, ["message_not_dict"]
    reasons: list[str] = []
    if not bool(message.get(spec["pocket_valid"], False)): reasons.append("invalid_pocket")
    for field, reason in ((spec["global_image"], "missing_global"), (spec["local_image"], "missing_local")):
        image = resolve_image(pkl_path, message, field)
        if image is None or not image.is_file(): reasons.append(reason)
    if not finite_vector(message.get(spec["trajectory_field"]), spec["trajectory_dims"]): reasons.append("invalid_trajectory")
    if not finite_vector(message.get(spec["action_field"]), spec["action_dims"]): reasons.append("invalid_action")
    return not reasons, reasons


def contiguous_runs(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []; start: int | None = None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None: start = index
        elif not flag and start is not None:
            runs.append((start, index)); start = None
    return runs


def parent_episode(name: str, suffix_pattern: str) -> str:
    return re.sub(suffix_pattern, "", name) if suffix_pattern else name


def write_overlay(source_pkl: Path, message: dict[str, Any], spec: dict[str, Any], destination: Path) -> None:
    raw = resolve_image(source_pkl, message, spec["source_image"])
    global_image = resolve_image(source_pkl, message, spec["global_image"])
    local_image = resolve_image(source_pkl, message, spec["local_image"])
    images = [cv2.imread(str(item), cv2.IMREAD_COLOR) if item is not None else None for item in (raw, global_image, local_image)]
    if any(item is None for item in images): return
    height, width = images[1].shape[:2]
    panels = [cv2.resize(images[0], (width, height), interpolation=cv2.INTER_AREA), images[1], images[2]]
    labels = ("augmented RGB", "G wide", "L local")
    for panel, label in zip(panels, labels): cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.hconcat(panels), [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def worker(source_text: str, source_root_text: str, output_root_text: str, config: dict[str, Any]) -> dict[str, Any]:
    source_pkl = Path(source_text); source_root = Path(source_root_text); output_root = Path(output_root_text)
    cv2.setNumThreads(1)
    data = read_pkl(source_pkl); messages = data["messages"]
    qc = config["qc"]; fields = config["fields"]
    source_name = source_pkl.parent.name
    specs = {
        "o6": {"action_field": config["labels"]["o6_action_field"], "action_dims": 6},
        "wuji": {"action_field": config["labels"]["wuji_action_field"], "action_dims": 20},
    }
    common = {"pocket_valid": fields["pocket_valid"], "global_image": fields["global_image"], "local_image": fields["local_image"], "source_image": fields["source_image"], "trajectory_field": config["labels"]["trajectory_field"], "trajectory_dims": int(config["labels"]["trajectory_min_dims"])}
    result: dict[str, Any] = {"pkl": str(source_pkl), "source_episode": source_name, "targets": {}}
    for target, target_spec in specs.items():
        spec = {**common, **target_spec}
        flags: list[bool] = []; errors: Counter[str] = Counter()
        for message in messages:
            valid, reasons = valid_frame(source_pkl, message, spec); flags.append(valid); errors.update(reasons)
        runs = [(start, end) for start, end in contiguous_runs(flags) if end - start >= int(qc["min_segment_frames"])]
        target_summary = {"frames": len(messages), "valid_frames": int(sum(flags)), "discarded_frames": int(len(messages) - sum(flags)), "reasons": dict(errors), "segments": []}
        for segment_index, (start, end) in enumerate(runs):
            target_dir = output_root / target / f"{source_name}_seg_{segment_index:03d}"
            shutil.copytree(source_pkl.parent, target_dir, copy_function=hardlink_or_copy)
            target_pkl = target_dir / source_pkl.name
            segment_data = read_pkl(target_pkl)
            segment_data["messages"] = segment_data["messages"][start:end]
            metadata = segment_data.setdefault("metadata", {})
            metadata["human2dexPipelineSegment"] = {"source_pkl": str(source_pkl), "source_episode": source_name, "parent_episode": parent_episode(source_name, str(qc["parent_episode_suffix_regex"])), "target": target, "start_frame": start, "end_frame_exclusive": end, "length": end - start}
            atomic_pickle(segment_data, target_pkl)
            target_summary["segments"].append({"path": str(target_dir), "start": start, "end": end, "length": end - start})
            every = max(1, int(qc["overlays_per_segment"]))
            indices = np.linspace(start, end - 1, num=min(every, end - start), dtype=int).tolist()
            for frame in sorted(set(indices)):
                write_overlay(source_pkl, messages[frame], spec, output_root / "overlays" / target / f"{source_name}_seg_{segment_index:03d}_frame_{frame:06d}.jpg")
        result["targets"][target] = target_summary
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--workers", type=int, default=None); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    source = path(config, "paths.gl_root"); output = path(config, "paths.qc_segments_root")
    if not source.is_dir(): raise FileNotFoundError(f"Run 06 first: {source}")
    require_new_directory(output)
    pkls = sorted(source.rglob("*.pkl"))
    if not pkls: raise FileNotFoundError(f"No PKLs under {source}")
    workers = max(1, min(args.workers or int(get(config, "parallel.qc_workers", 32)), len(pkls)))
    compact = {
        "qc": {
            "min_segment_frames": int(get(config, "qc.min_segment_frames", 24)),
            "overlays_per_segment": int(get(config, "qc.overlays_per_segment", 3)),
            "parent_episode_suffix_regex": str(get(config, "qc.parent_episode_suffix_regex", r"_app_aug\d+$")),
        },
        "fields": {
            "pocket_valid": str(get(config, "fields.human_pocket_valid", "humanGraspPocketValid")),
            "global_image": str(get(config, "fields.global_image", "globalCanonicalImage")),
            "local_image": str(get(config, "fields.local_image", "localCanonicalImage")),
            "source_image": str(get(config, "fields.source_image", "rgbImage")),
        },
        "labels": {
            "o6_action_field": str(get(config, "labels.o6_action_field", "fused_o6_command")),
            "wuji_action_field": str(get(config, "labels.wuji_action_field", "fused_wuji_command")),
            "trajectory_field": str(get(config, "labels.trajectory_field", "trajectoryPose")),
            "trajectory_min_dims": int(get(config, "labels.trajectory_min_dims", 6)),
        },
    }
    print(json.dumps({"source": str(source), "output": str(output), "pkls": len(pkls), "workers": workers, **compact}, indent=2))
    if args.dry_run: return 0
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, str(item), str(source), str(output), compact) for item in pkls]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result(); results.append(result)
            if index == 1 or index % 25 == 0 or index == len(futures): print(f"[07] {index}/{len(futures)}", flush=True)
    summary = {"source": str(source), "output": str(output), "pkls": len(pkls), "workers": workers, "results": results}
    reports = output / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(output, "07_qc_and_segment", summary)
    return 0


if __name__ == "__main__": raise SystemExit(main())
