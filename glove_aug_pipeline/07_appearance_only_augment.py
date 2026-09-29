#!/usr/bin/env python3
"""07: Appearance-only augmentation using only the independent hand masks."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import get, load_config, path, python_bin, repo_root, run, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = repo_root(config)
    base = path(config, "paths.base_root")
    hand_masks = path(config, "paths.hand_masks_root")
    output = path(config, "paths.appearance_root")
    qc = path(config, "paths.appearance_qc_root")
    if not base.is_dir():
        raise FileNotFoundError(f"Run stages 01-06 first: {base}")
    if not (hand_masks / "masks").is_dir():
        raise FileNotFoundError(f"Run hand-mask stage 04 first: {hand_masks / 'masks'}")
    if output.exists():
        raise FileExistsError(f"Appearance output already exists: {output}")
    command = [
        python_bin(config),
        str(root / "glove_aug_pipeline/02_augment_dataset.py"),
        "--config", str(path(config, "upstream.appearance_config")),
        "--input", str(base),
        "--output", str(output),
        "--mask-root", str(hand_masks / "masks"),
        "--qc-output", str(qc),
        "--variants", str(get(config, "appearance.variants", 3)),
        "--episode-suffix", str(get(config, "appearance.episode_suffix", "_app_aug")),
        "--seed", str(get(config, "appearance.seed", 20260606)),
        "--quality", str(get(config, "appearance.jpeg_quality", 95)),
        "--qc-frames", str(get(config, "appearance.qc_frames", 8)),
        "--workers", str(get(config, "parallel.appearance_workers", 32)),
        "--io-workers", str(get(config, "parallel.appearance_io_workers", 2)),
        "--mp-start-method", str(get(config, "parallel.appearance_mp_start_method", "fork")),
        "--camera-mount-aug-disabled",
        "--camera-mount-frame-jitter-disabled",
    ]
    if bool(get(config, "skeleton.enabled", True)):
        command += [
            "--render-wrist-skeleton",
            "--skeleton-image-subdir", str(get(config, "skeleton.image_subdir", "images")),
            "--skeleton-line-width", str(get(config, "skeleton.line_width", 2)),
            "--skeleton-point-radius", str(get(config, "skeleton.point_radius", 4)),
        ]
    for image_subdir in get(config, "streams.image_subdirs", ["images"]):
        command += ["--image-subdir", str(image_subdir)]
    run(command, cwd=root, dry_run=args.dry_run)
    write_manifest(
        output,
        "07_appearance_only_augment",
        {
            "command": command,
            "base_root": str(base),
            "hand_masks_root": str(hand_masks),
            "task_object_masks_used_for_augmentation": False,
        },
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
