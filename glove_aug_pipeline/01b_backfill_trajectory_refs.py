#!/usr/bin/env python3
"""01b: Add palm/TCP trajectory reference fields inside the writable base."""
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
    if not base.is_dir():
        raise FileNotFoundError(f"Run 01 first: {base}")

    tcp_offset = get(config, "trajectory.tcp_offset_mm", [-30.0, 40.0, 0.0])
    if not isinstance(tcp_offset, list) or len(tcp_offset) != 3:
        raise ValueError("trajectory.tcp_offset_mm must be a 3-element list")
    invalid_policy = str(get(config, "trajectory.invalid_policy", "none"))

    command = [
        python_bin(config),
        str(root / "tools/add_trajectory_reference_poses.py"),
        "--input", str(base),
        "--in-place",
        "--invalid-policy", invalid_policy,
        "--tcp-offset-mm",
        str(float(tcp_offset[0])),
        str(float(tcp_offset[1])),
        str(float(tcp_offset[2])),
    ]
    if bool(get(config, "trajectory.overwrite", False)):
        command.append("--overwrite-existing-fields")

    run(command, cwd=root, dry_run=args.dry_run)
    write_manifest(
        base,
        "01b_backfill_trajectory_refs",
        {
            "command": command,
            "base_root": str(base),
            "tcp_offset_mm": [float(item) for item in tcp_offset],
            "invalid_policy": invalid_policy,
        },
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
