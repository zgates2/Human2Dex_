#!/usr/bin/env python3
"""08: Convert validated O6 and Wuji segment roots into separate Zarr datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
from pipeline_common import get, load_config, path, python_bin, repo_root, run, write_manifest


def build_command(config: dict, target: str) -> tuple[list[str], Path]:
    root = repo_root(config); segments = path(config, "paths.qc_segments_root") / target
    output = path(config, f"paths.{target}_zarr_root")
    if not segments.is_dir(): raise FileNotFoundError(f"Run 07 first: {segments}")
    if output.exists(): raise FileExistsError(f"Zarr output already exists: {output}")
    source = str(get(config, f"labels.{target}_action_field"))
    command = [python_bin(config), str(root / "tools/convert_pkl_to_training_zarr.py"), "--input", str(segments), "--output", str(output),
               "--data-scaling-laws-root", str(path(config, "runtime.data_scaling_laws_root")), "--rgb-image-field", str(get(config, "fields.global_image", "globalCanonicalImage")),
               "--local-rgb-image-field", str(get(config, "fields.local_image", "localCanonicalImage")), "--trajectory-pose-source", str(get(config, "labels.trajectory_source", "trajectoryPose")),
               "--gripper-source", source, "--output-format", str(get(config, "zarr.output_format", "zip")),
               "--workers", str(get(config, "parallel.zarr_workers", 64)), "--opencv-threads", str(get(config, "parallel.zarr_opencv_threads", 1)),
               "--blosc-threads", str(get(config, "parallel.zarr_blosc_threads", 1)), "--image-batch-size", str(get(config, "parallel.zarr_image_batch_size", 32)),
               "--max-inflight-tasks", str(get(config, "parallel.zarr_max_inflight_tasks", 128)), "--image-compressor", str(get(config, "zarr.image_compressor", "blosc")),
               "--skip-missing-images", "--skip-empty-episodes"]
    return command, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--target", choices=("o6", "wuji", "both"), default="both"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    targets = ("o6", "wuji") if args.target == "both" else (args.target,)
    for target in targets:
        command, output = build_command(config, target)
        run(command, cwd=repo_root(config), dry_run=args.dry_run)
        write_manifest(output, f"08_build_training_zarr_{target}", {"command": command, "target": target}, dry_run=args.dry_run)
    return 0


if __name__ == "__main__": raise SystemExit(main())
