#!/usr/bin/env python3
"""03: Multi-GPU partitioned direct-RGB grasp-pocket backfill.

This is an orchestration wrapper.  Every worker invokes the existing pocket
writer on a disjoint PKL list, so fields and model semantics remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline_common import checkpoint_manifest, get, load_config, path, python_bin, repo_root, write_manifest


def chunks(items: list[Path], n: int) -> list[list[Path]]:
    result = [[] for _ in range(n)]
    for index, item in enumerate(items): result[index % n].append(item)
    return [part for part in result if part]


def execute(command: list[str], cwd: Path, log: Path, dry_run: bool) -> dict:
    print("$ " + subprocess.list2cmdline(command), flush=True)
    if dry_run: return {"ok": True, "command": command, "log": str(log)}
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    return {"ok": completed.returncode == 0, "returncode": completed.returncode, "command": command, "log": str(log)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); config = load_config(args.config)
    root = repo_root(config); base = path(config, "paths.base_root")
    if not base.is_dir(): raise FileNotFoundError(f"Run 01 first: {base}")
    pkls = sorted(base.rglob("*.pkl"))
    if not pkls: raise FileNotFoundError(f"No PKLs under {base}")
    devices = [str(item).replace("cuda:", "") for item in get(config, "parallel.pocket_devices", [0])]
    if not devices: raise ValueError("parallel.pocket_devices must contain at least one GPU")
    shard_dir = base / ".human2dex_pipeline" / "03_pocket_shards"
    log_dir = base / ".human2dex_pipeline" / "logs" / "03_pocket"
    if not args.dry_run:
        shard_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[list[str], Path]] = []
    for index, part in enumerate(chunks(pkls, len(devices))):
        shard = shard_dir / f"gpu{devices[index]}_of{len(devices)}.txt"
        if not args.dry_run:
            shard.write_text("\n".join(str(item.relative_to(base)) for item in part) + "\n", encoding="utf-8")
        command = [python_bin(config), str(root / "tools/add_grasp_pocket_predictions_to_pkl.py"), "--data-root", str(base), "--in-place",
                   "--stage1-checkpoint", str(path(config, "models.stage2_checkpoint")),
                   "--pocket-head-checkpoint", str(path(config, "models.pocket_head_checkpoint")),
                   "--pkl-list", str(shard), "--device", f"cuda:{devices[index]}",
                   "--batch-size", str(get(config, "parallel.pocket_batch_size", 128)),
                   "--num-workers", str(get(config, "parallel.pocket_loader_workers", 4)),
                   "--precision", str(get(config, "parallel.pocket_precision", "auto")),
                   "--min-confidence", str(get(config, "pocket.min_confidence", 0.10))]
        if bool(get(config, "pocket.overwrite", False)):
            command.append("--overwrite")
        commands.append((command, log_dir / f"gpu{devices[index]}.log"))
    print(json.dumps({"pkls": len(pkls), "devices": devices, "shards": [len(part) for part in chunks(pkls, len(devices))]}, indent=2))
    results = []
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [executor.submit(execute, command, root, log, args.dry_run) for command, log in commands]
        for future in as_completed(futures): results.append(future.result())
    failed = [result for result in results if not result["ok"]]
    write_manifest(base, "03_backfill_grasp_pocket", {"results": results, "checkpoints": checkpoint_manifest(config)}, dry_run=args.dry_run)
    if failed:
        raise RuntimeError(f"{len(failed)} pocket shards failed; inspect {log_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
