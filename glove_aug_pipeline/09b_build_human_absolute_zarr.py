#!/usr/bin/env python3
"""Create a paired Human-Absolute PKL tree and O6 Zarr dataset.

The source is the already validated single-view Human2Dex segment tree.  Images
are reused through symlinks; only PKLs are rewritten with objectAbsoluteObs.
The original objectPocketObs is retained as a training-only auxiliary target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception as exc:
    raise SystemExit("PyYAML is required") from exc

def install_numpy_pickle_compat() -> None:
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric

def read_pkl(path: Path) -> dict[str, Any]:
    install_numpy_pickle_compat()
    with path.open("rb") as f:
        value = pickle.load(f)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"Invalid DexUMI PKL: {path}")
    return value

def write_pkl(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

def field_prefix(spec: dict[str, Any]) -> str:
    return str(spec.get("field_prefix") or "task_object_" + safe_name(str(spec.get("name"))))

def finite_pair(value: Any) -> tuple[float, float] | None:
    try:
        a = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if a.shape != (2,) or not np.isfinite(a).all():
        return None
    return float(a[0]), float(a[1])

def as_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return x if math.isfinite(x) else default

def load_task_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid YAML config: {path}")
    return value

def resolve_reference_scales(
    config: dict[str, Any],
    reference_meta_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float]]:
    objects = config.get("task_objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("config.task_objects must be a non-empty list")
    meta = json.loads(reference_meta_path.read_text(encoding="utf-8"))
    areas = {str(k): float(v) for k, v in (meta.get("reference_areas_px2") or {}).items()}
    obs_cfg = dict(meta.get("observation") or {})
    config_obs = dict(config.get("object_observation") or {})
    scales: dict[str, float] = {}
    for raw in objects:
        name = str(raw["name"])
        area = areas.get(name)
        if area is None or not math.isfinite(area) or area <= 0:
            raise ValueError(f"missing resolved reference area for {name}")
        configured = raw.get("reference_scale_px", config_obs.get("reference_scale_px", obs_cfg.get("reference_scale_px")))
        if configured is None or str(configured).lower() in {"auto", "auto_sqrt_area", "sqrt_reference_area"}:
            scale = math.sqrt(area)
        else:
            scale = float(configured)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid reference scale for {name}: {scale}")
        scales[name] = scale
    return [dict(x) for x in objects], areas, scales

def build_absolute_obs(
    message: dict[str, Any],
    objects: list[dict[str, Any]],
    scales: dict[str, float],
    source_size: tuple[int, int],
    policy_size: tuple[int, int],
    log_clip: float,
) -> tuple[list[float], bool, str | None]:
    src_w, src_h = source_size
    out_w, out_h = policy_size
    sx = float(out_w) / float(src_w)
    sy = float(out_h) / float(src_h)
    center_u = float(out_w) * 0.5
    center_v = float(out_h) * 0.5
    existing = np.asarray(message.get("objectPocketObs", []), dtype=np.float64).reshape(-1)
    expected = 5 * len(objects)
    blocks: list[float] = []
    all_valid = True
    errors: list[str] = []
    for index, spec in enumerate(objects):
        name = str(spec["name"])
        prefix = field_prefix(spec)
        uv = finite_pair(message.get(f"{prefix}_policy_uv"))
        if uv is None:
            uv = finite_pair(message.get(f"{prefix}_raw_uv"))
        block = existing[5 * index:5 * (index + 1)] if existing.size >= expected else np.zeros(5, dtype=np.float64)
        block = np.asarray(block, dtype=np.float64)
        block_valid = bool(block.shape == (5,) and block[4] > 0.5)
        memory_valid = bool(message.get(f"{prefix}_memory_valid", False))
        usable = uv is not None and block_valid and memory_valid
        if not usable:
            blocks.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            all_valid = False
            errors.append(f"{name}:invalid_memory_or_uv")
            continue
        ref_scale_final_x = float(scales[name]) * sx
        ref_scale_final_y = float(scales[name]) * sy
        u_final = float(uv[0]) * sx
        v_final = float(uv[1]) * sy
        dx = (u_final - center_u) / ref_scale_final_x
        dy = (v_final - center_v) / ref_scale_final_y
        log_area = float(np.clip(as_float(block[2]), -log_clip, log_clip))
        confidence = float(np.clip(as_float(block[3]), 0.0, 1.0))
        blocks.extend([float(dx), float(dy), log_area, confidence, 1.0])
        message[f"{prefix}_absolute_obs"] = [float(dx), float(dy), log_area, confidence, 1.0]
    return blocks, all_valid, None if not errors else ";".join(errors)

def process_one(
    source_text: str,
    output_root_text: str,
    objects: list[dict[str, Any]],
    scales: dict[str, float],
    areas: dict[str, float],
    source_size: tuple[int, int],
    policy_size: tuple[int, int],
    log_clip: float,
) -> dict[str, Any]:
    source = Path(source_text)
    output_root = Path(output_root_text)
    rel_parent = source.parent.name
    destination_dir = output_root / rel_parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    src_images = source.parent / "images"
    dst_images = destination_dir / "images"
    if dst_images.exists() or dst_images.is_symlink():
        if dst_images.is_dir() and not dst_images.is_symlink():
            shutil.rmtree(dst_images)
        else:
            dst_images.unlink()
    if src_images.is_dir():
        os.symlink(src_images, dst_images, target_is_directory=True)
    data = read_pkl(source)
    messages = data["messages"]
    valid_count = 0
    for message in messages:
        obs, valid, error = build_absolute_obs(
            message, objects, scales, source_size, policy_size, log_clip
        )
        message["objectAbsoluteObs"] = obs
        message["objectAbsoluteObsValid"] = bool(valid)
        message["objectAbsoluteObsError"] = error
        if valid:
            valid_count += 1
    metadata = data.setdefault("metadata", {})
    metadata["humanAbsoluteObjectObservationV1"] = {
        "reference_mode": "image_center",
        "policy_image_size": list(policy_size),
        "source_image_size": list(source_size),
        "image_center": [policy_size[0] * 0.5, policy_size[1] * 0.5],
        "objects": [str(x["name"]) for x in objects],
        "reference_areas_px2_source": areas,
        "reference_scales_px_source": scales,
        "strategy_input_field": "objectAbsoluteObs",
        "auxiliary_pocket_field": "objectPocketObs",
        "pocket_in_strategy_input": False,
    }
    destination_pkl = destination_dir / source.name
    write_pkl(destination_pkl, data)
    return {
        "source": str(source),
        "destination": str(destination_pkl),
        "frames": len(messages),
        "absolute_valid_frames": valid_count,
    }

def discover_pkls(root: Path) -> list[Path]:
    """Find segment PKLs without descending into image trees."""
    found: list[Path] = []
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix == ".pkl":
            found.append(entry)
            continue
        if not entry.is_dir() or entry.name in {"images", "canonical_views", "masks"}:
            continue
        direct = entry / f"{entry.name}.pkl"
        if direct.is_file():
            found.append(direct)
            continue
        for dirpath, dirnames, filenames in os.walk(entry):
            dirnames[:] = [
                name for name in dirnames
                if name not in {"images", "canonical_views", "masks"}
                and not name.startswith(".")
            ]
            for filename in filenames:
                if filename.endswith(".pkl"):
                    found.append(Path(dirpath) / filename)
    return sorted(set(found))

def ensure_training_manifest(output_root: Path, zarr_output: Path) -> Path:
    """Ensure UmiDataset's sibling manifest.json exists for the generated Zarr.

    Older converter runs could produce dataset.zarr.zip without the sidecar
    manifest.  The Human-Absolute manifest already contains the complete
    episode order, so convert destination -> pkl entries without touching the
    Zarr data itself.
    """
    manifest_path = zarr_output.parent / "manifest.json"
    if manifest_path.is_file():
        return manifest_path
    source_manifest = output_root / "human_absolute_manifest.json"
    raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    episodes = []
    for item in raw.get("episodes", []):
        pkl_path = item.get("destination") or item.get("pkl")
        if not pkl_path:
            raise ValueError("human_absolute_manifest episode missing destination/pkl")
        episodes.append({
            "pkl": str(pkl_path),
            "frames": int(item.get("frames", 0)),
        })
    manifest = dict(raw)
    manifest.update({
        "format": "DexUMI PKL to Data-Scaling-Laws dataset.zarr (zip)",
        "zarr": str(zarr_output.with_name("dataset.zarr")),
        "zarr_zip": str(zarr_output),
        "source_episode_count": len(episodes),
        "selected_episode_count": len(episodes),
        "converted_episode_count": len(episodes),
        "total_frames": int(sum(item["frames"] for item in episodes)),
        "episodes": episodes,
    })
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-meta", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zarr-output", type=Path, required=True)
    parser.add_argument("--source-size", type=int, nargs=2, default=[480, 480], metavar=("W", "H"))
    parser.add_argument("--policy-size", type=int, nargs=2, default=[224, 224], metavar=("W", "H"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--converter-workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N segment PKLs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_size[0] <= 0 or args.source_size[1] <= 0:
        raise SystemExit("source size must be positive")
    if args.policy_size[0] <= 0 or args.policy_size[1] <= 0:
        raise SystemExit("policy size must be positive")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    zarr_output = args.zarr_output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source root missing: {source_root}")
    if args.overwrite:
        if output_root.exists():
            shutil.rmtree(output_root)
        if zarr_output.exists():
            if zarr_output.is_dir():
                shutil.rmtree(zarr_output)
            else:
                zarr_output.unlink()
    elif output_root.exists() or zarr_output.exists():
        raise SystemExit("output exists; use --overwrite")
    objects, areas, scales = resolve_reference_scales(args.config and load_task_config(args.config), args.reference_meta)
    pkls = discover_pkls(source_root)
    if args.limit is not None:
        pkls = pkls[: max(0, int(args.limit))]
    if not pkls:
        raise SystemExit(f"no PKLs under {source_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    worker_count = max(1, min(int(args.workers), len(pkls)))
    print(json.dumps({
        "source_root": str(source_root),
        "output_root": str(output_root),
        "pkls": len(pkls),
        "objects": [str(x["name"]) for x in objects],
        "reference_areas_px2": areas,
        "reference_scales_px_source": scales,
        "source_size": args.source_size,
        "policy_size": args.policy_size,
        "workers": worker_count,
    }, ensure_ascii=False, indent=2), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_one, str(p), str(output_root), objects, scales, areas,
                tuple(args.source_size), tuple(args.policy_size),
                float((load_task_config(args.config).get("object_observation") or {}).get("log_area_clip", 4.0)),
            )
            for p in pkls
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index == 1 or index % 50 == 0 or index == len(futures):
                print(f"[absolute-pkl] {index}/{len(futures)}", flush=True)
    results.sort(key=lambda x: x["destination"])
    manifest = {
        "format": "Human-Absolute paired PKL tree",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_size": list(args.source_size),
        "policy_size": list(args.policy_size),
        "image_center": [args.policy_size[0] * 0.5, args.policy_size[1] * 0.5],
        "reference_areas_px2_source": areas,
        "reference_scales_px_source": scales,
        "objects": [str(x["name"]) for x in objects],
        "explicit_policy_field": "objectAbsoluteObs",
        "hidden_auxiliary_field": "objectPocketObs",
        "pocket_in_policy_input": False,
        "episodes": results,
    }
    (output_root / "human_absolute_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    converter = Path("/home/zjc/Desktop/human2dex/tools/convert_pkl_to_training_zarr.py")
    command = [
        sys.executable, str(converter),
        "--input", str(output_root),
        "--output", str(zarr_output),
        "--resize", "224,224",
        "--data-scaling-laws-root", "/home/zjc/Desktop/human2dex",
        "--rgb-image-field", "rgbImage",
        "--object-pocket-obs-field", "objectPocketObs",
        "--object-pocket-obs-dim", str(5 * len(objects)),
        "--object-absolute-obs-field", "objectAbsoluteObs",
        "--object-absolute-obs-dim", str(5 * len(objects)),
        "--trajectory-pose-source", "trajectoryPose_tcp",
        "--gripper-source", "fused_o6_command",
        "--output-format", "zip",
        "--workers", str(int(args.converter_workers)),
        "--opencv-threads", "1",
        "--blosc-threads", "1",
        "--image-batch-size", "32",
        "--max-inflight-tasks", "128",
        "--image-compressor", "blosc",
        "--skip-missing-images",
        "--skip-empty-episodes",
        "--overwrite",
        "--force-unlock",
    ]
    print("running converter:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    manifest_path = ensure_training_manifest(output_root=output_root, zarr_output=zarr_output)
    print(f"wrote {zarr_output}", flush=True)
    print(f"ensured training manifest {manifest_path}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
