#!/usr/bin/env python3
"""01: Create a writable base dataset without touching raw data.

PKLs are copied because later stages rewrite fields atomically.  RGB and other
large immutable files are hard-linked when possible, then copied as a safe
fallback.  This is a storage optimization, not permission enforcement: later
stages must only write PKLs in this derived directory.
"""
from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pipeline_common import checkpoint_manifest, load_config, path, require_new_directory, write_manifest


def hardlink_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def copy_file(source: str, raw_root: str, output_root: str, resume: bool) -> tuple[str, str]:
    source_path = Path(source)
    target = Path(output_root) / source_path.relative_to(Path(raw_root))
    if target.exists():
        if resume:
            return "skipped", str(target)
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix.lower() == ".pkl":
        shutil.copy2(source_path, target)
        return "pkl_copy", str(target)
    if source_path.is_symlink():
        target.symlink_to(os.readlink(source_path))
        return "symlink", str(target)
    return hardlink_or_copy(source_path, target), str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_root = path(config, "paths.raw_root")
    output_root = path(config, "paths.base_root")
    if raw_root == output_root:
        raise ValueError("paths.raw_root and paths.base_root must differ")
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    require_new_directory(output_root, resume=args.resume)
    files = [item for item in raw_root.rglob("*") if item.is_file() or item.is_symlink()]
    workers = args.workers or int(config.get("parallel", {}).get("prepare_workers", 64))
    workers = max(1, min(int(workers), len(files) or 1))
    print({"raw_root": str(raw_root), "base_root": str(output_root), "files": len(files), "workers": workers, "resume": args.resume})
    if args.dry_run:
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(copy_file, str(item), str(raw_root), str(output_root), args.resume) for item in files]
        for index, future in enumerate(as_completed(futures), start=1):
            kind, _ = future.result()
            counters[kind] = counters.get(kind, 0) + 1
            if index == 1 or index % 5000 == 0 or index == len(futures):
                print(f"[01] {index}/{len(futures)} {counters}", flush=True)
    write_manifest(output_root, "01_prepare_base_enriched", {
        "raw_root": str(raw_root), "base_root": str(output_root), "files": len(files), "workers": workers,
        "copy_stats": counters, "checkpoints": checkpoint_manifest(config),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
