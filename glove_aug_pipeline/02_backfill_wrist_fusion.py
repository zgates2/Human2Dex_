#!/usr/bin/env python3
"""02: Invoke the existing wrist+PICO fusion writer on base_enriched_v1."""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import checkpoint_manifest, comma_args, get, load_config, path, python_bin, repo_root, run, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = repo_root(config); base = path(config, "paths.base_root")
    if not base.is_dir(): raise FileNotFoundError(f"Run 01 first: {base}")
    command = [python_bin(config), str(root / "tools/add_wrist_fusion_predictions_to_pkl.py"), "--data-root", str(base),
               "--checkpoint", str(path(config, "models.stage2_checkpoint")),
               "--devices", str(get(config, "parallel.wrist_devices", "auto")),
               "--batch-size", str(get(config, "parallel.wrist_batch_size", 128)),
               "--num-workers", str(get(config, "parallel.wrist_loader_workers", 8)),
               "--fusion-workers", str(get(config, "parallel.fusion_workers", 32)),
               "--log-every", str(get(config, "parallel.wrist_log_every", 5)),
               "--precision", str(get(config, "parallel.wrist_precision", "auto")),
               "--image-field", str(get(config, "fields.source_image", "rgbImage"))]
    projection = path(config, "models.wrist_projection_head_checkpoint", required=False)
    if projection is not None:
        command += ["--projection-head-checkpoint", str(projection), "--write-wrist-projection"]
    # The fusion writer always materializes both O6 and Wuji labels. The
    # labels.skip_wuji switch is consumed later when selecting Zarr outputs.
    run(command, cwd=root, dry_run=args.dry_run)
    write_manifest(base, "02_backfill_wrist_fusion", {"command": command, "checkpoints": checkpoint_manifest(config)}, dry_run=args.dry_run)
    return 0


if __name__ == "__main__": raise SystemExit(main())
