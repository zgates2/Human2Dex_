#!/usr/bin/env python3
"""04: Call the existing multi-GPU SAM3 hand-mask generator on raw RGB."""
from __future__ import annotations

import argparse
from pathlib import Path
from pipeline_common import get, load_config, path, python_bin, repo_root, run, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    root = repo_root(config); raw = path(config, "paths.raw_root"); masks = path(config, "paths.masks_root")
    if not raw.is_dir(): raise FileNotFoundError(raw)
    if masks.exists(): raise FileExistsError(f"Masks output already exists: {masks}")
    command = [python_bin(config), str(root / "glove_aug_pipeline/01_generate_sam3_masks.py"),
               "--config", str(path(config, "upstream.sam3_config")), "--input", str(raw), "--output", str(masks),
               "--num-gpus", str(get(config, "parallel.sam3_num_gpus", 8)), "--batch-size", str(get(config, "parallel.sam3_batch_size", 32)),
               "--prefetch", str(get(config, "parallel.sam3_prefetch", 8))]
    for image_subdir in get(config, "streams.image_subdirs", ["images"]): command += ["--image-subdir", str(image_subdir)]
    run(command, cwd=root, dry_run=args.dry_run)
    write_manifest(masks, "04_generate_sam3_masks", {"command": command, "raw_root": str(raw)}, dry_run=args.dry_run)
    return 0


if __name__ == "__main__": raise SystemExit(main())
