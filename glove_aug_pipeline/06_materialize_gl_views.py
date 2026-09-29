#!/usr/bin/env python3
"""06: Render G/L from augmented RGB and the copied direct pocket field."""
from __future__ import annotations

import argparse
from pathlib import Path
from pipeline_common import comma_args, get, load_config, path, python_bin, repo_root, run, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    root = repo_root(config); source = path(config, "paths.appearance_root"); output = path(config, "paths.gl_root")
    if not source.is_dir(): raise FileNotFoundError(f"Run 05 first: {source}")
    if output.exists(): raise FileExistsError(f"G/L output already exists: {output}")
    views = get(config, "views", {})
    command = [python_bin(config), str(root / "tools/materialize_human_direct_pocket_views.py"), "materialize", "--input", str(source),
               "--intrinsics", str(path(config, "views.intrinsics")), "--source-image-field", str(get(config, "fields.source_image", "rgbImage")),
               "--pocket-uv-field", str(get(config, "fields.pocket_uv", "wrist_grasp_pocket_uv")), "--pocket-valid-field", str(get(config, "fields.pocket_valid", "wrist_grasp_pocket_valid")),
               "--pocket-confidence-field", str(get(config, "fields.pocket_confidence", "wrist_grasp_pocket_confidence")),
               "--min-confidence", str(get(config, "pocket.min_confidence", 0.10)), "--global-size", str(get(config, "views.global_size", "224x224")),
               "--local-size", str(get(config, "views.local_size", "224x224")), "--anchor-uv-ratio", str(get(config, "views.global_anchor", "0.5,0.35")),
               "--local-anchor-uv-ratio", str(get(config, "views.local_anchor", "0.5,0.5")), "--local-mode", str(get(config, "views.local_mode", "anchor_wide")),
               "--local-fov-deg", str(get(config, "views.local_fov_deg", 78.0)),
               "--global-border-mode", str(get(config, "views.global_border_mode", "constant")), "--local-border-mode", str(get(config, "views.local_border_mode", "reflect101")),
               "--output", str(output), "--global-image-field", str(get(config, "fields.global_image", "globalCanonicalImage")),
               "--local-image-field", str(get(config, "fields.local_image", "localCanonicalImage")), "--jpeg-quality", str(get(config, "views.jpeg_quality", 95)),
               "--renderer", str(get(config, "parallel.gl_renderer", "cuda")), "--devices", comma_args(get(config, "parallel.gl_devices", list(range(8)))),
               "--gpu-batch-size", str(get(config, "parallel.gl_gpu_batch_size", 64)), "--opencv-threads", str(get(config, "parallel.gl_opencv_threads", 4))]
    run(command, cwd=root, dry_run=args.dry_run)
    write_manifest(output, "06_materialize_gl_views", {"command": command, "source_root": str(source)}, dry_run=args.dry_run)
    return 0


if __name__ == "__main__": raise SystemExit(main())
