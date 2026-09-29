#!/usr/bin/env python3
"""Trim tail messages from every DexUMI PKL episode into a new dataset.

This stage is intentionally non-destructive: it writes a new PKL tree and
hard-links the retained image files required by the trimmed messages.  The
source dataset is never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from pipeline_common import get, load_config, path, write_manifest


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to unpickle arrays written by NumPy 2.x."""
    try:
        import numpy as np
    except Exception:
        return
    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def read_pkl(pkl_path: Path) -> dict[str, Any]:
    install_numpy_pickle_compat()
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict/messages: {pkl_path}")
    return data


def find_pkl_paths(root: Path) -> list[Path]:
    patterns = ("demo_*/*.pkl", "episode_*/*.pkl", "episodes_*/*.pkl")
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        if paths:
            return paths
    return sorted(root.rglob("*.pkl"))


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def maybe_link_message_images(
    *,
    src_episode_dir: Path,
    dst_episode_dir: Path,
    messages: list[Any],
    image_fields: list[str],
) -> dict[str, int]:
    stats = {"linked": 0, "copied": 0, "exists": 0, "missing": 0, "absolute": 0, "unsafe": 0}
    for message in messages:
        if not isinstance(message, dict):
            continue
        for field in image_fields:
            value = message.get(field)
            if not isinstance(value, str) or not value:
                continue
            rel_path = Path(value)
            if rel_path.is_absolute():
                stats["absolute"] += 1
                continue
            if ".." in rel_path.parts:
                stats["unsafe"] += 1
                continue
            src = src_episode_dir / rel_path
            dst = dst_episode_dir / rel_path
            if not src.is_file():
                stats["missing"] += 1
                continue
            status = link_or_copy(src, dst)
            stats[status] += 1
    return stats


def write_trimmed_pkl(src_pkl: Path, dst_pkl: Path, data: dict[str, Any], trimmed_messages: list[Any]) -> None:
    output_data = dict(data)
    output_data["messages"] = trimmed_messages
    metadata = dict(output_data.get("metadata") or {})
    metadata["tailTrim"] = {
        "source_pkl": str(src_pkl),
        "created_unix_s": time.time(),
    }
    output_data["metadata"] = metadata
    dst_pkl.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_pkl.with_suffix(dst_pkl.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst_pkl)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Pipeline YAML; uses paths.appearance_root -> paths.trimmed_appearance_root")
    parser.add_argument("--input", type=Path, default=None, help="Source PKL dataset root")
    parser.add_argument("--output", type=Path, default=None, help="Trimmed PKL dataset root")
    parser.add_argument("--tail-frames", type=int, default=None, help="Frames to remove from the end of every episode")
    parser.add_argument("--tail-fraction-numerator", type=int, default=None, help="Tail fraction numerator to remove, e.g. 1 for 1/4")
    parser.add_argument("--tail-fraction-denominator", type=int, default=None, help="Tail fraction denominator to remove, e.g. 4 for 1/4")
    parser.add_argument("--tail-fraction-rounding", choices=("floor", "ceil"), default=None, help="Rounding used for fractional tail frame count")
    parser.add_argument("--extra-tail-frames", type=int, default=None, help="Additional tail frames to remove for long episodes")
    parser.add_argument(
        "--extra-tail-frames-if-source-gt",
        type=int,
        default=None,
        help="Apply --extra-tail-frames only when source episode length is greater than this threshold",
    )
    parser.add_argument("--image-field", action="append", default=None, help="PKL image field to hard-link; repeatable")
    parser.add_argument("--drop-short", action="store_true", help="Drop episodes with <= tail-frames instead of failing")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.config is not None:
        config = load_config(args.config)

    input_root = args.input or path(config, "paths.appearance_root")
    output_root = args.output or path(config, "paths.trimmed_appearance_root")
    tail_frames = args.tail_frames
    if tail_frames is None:
        tail_frames = int(get(config, "trim_tail.frames", 150))
    tail_fraction_numerator = args.tail_fraction_numerator
    if tail_fraction_numerator is None:
        value = get(config, "trim_tail.fraction_numerator", None)
        tail_fraction_numerator = None if value is None else int(value)
    tail_fraction_denominator = args.tail_fraction_denominator
    if tail_fraction_denominator is None:
        value = get(config, "trim_tail.fraction_denominator", None)
        tail_fraction_denominator = None if value is None else int(value)
    tail_fraction_rounding = args.tail_fraction_rounding
    if tail_fraction_rounding is None:
        tail_fraction_rounding = str(get(config, "trim_tail.fraction_rounding", "floor"))
    extra_tail_frames = args.extra_tail_frames
    if extra_tail_frames is None:
        extra_tail_frames = int(get(config, "trim_tail.extra_frames", 0))
    extra_tail_frames_if_source_gt = args.extra_tail_frames_if_source_gt
    if extra_tail_frames_if_source_gt is None:
        value = get(config, "trim_tail.extra_frames_if_source_gt", None)
        extra_tail_frames_if_source_gt = None if value is None else int(value)
    image_fields = args.image_field or [str(get(config, "fields.source_image", "rgbImage"))]

    if input_root is None or output_root is None:
        raise ValueError("Both --input/--output are required unless config provides paths.appearance_root/paths.trimmed_appearance_root")
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if tail_frames < 0:
        raise ValueError("--tail-frames must be non-negative")
    if (tail_fraction_numerator is None) != (tail_fraction_denominator is None):
        raise ValueError("--tail-fraction-numerator and --tail-fraction-denominator must be provided together")
    if tail_fraction_numerator is not None and tail_fraction_denominator is not None:
        if tail_fraction_numerator < 0:
            raise ValueError("--tail-fraction-numerator must be non-negative")
        if tail_fraction_denominator <= 0:
            raise ValueError("--tail-fraction-denominator must be positive")
        if tail_fraction_numerator >= tail_fraction_denominator:
            raise ValueError("--tail-fraction-numerator must be smaller than denominator")
    if tail_fraction_rounding not in {"floor", "ceil"}:
        raise ValueError("--tail-fraction-rounding must be floor or ceil")
    if extra_tail_frames < 0:
        raise ValueError("--extra-tail-frames must be non-negative")
    if extra_tail_frames and extra_tail_frames_if_source_gt is None:
        raise ValueError("--extra-tail-frames requires --extra-tail-frames-if-source-gt")
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    if output_root.exists() and not args.dry_run:
        raise FileExistsError(f"Trimmed output already exists: {output_root}")
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError(f"Refusing to write output inside input: input={input_root}, output={output_root}")

    pkl_paths = find_pkl_paths(input_root)
    if not pkl_paths:
        raise FileNotFoundError(f"No PKL files found under {input_root}")

    summary: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "tail_frames": tail_frames,
        "tail_fraction_numerator": tail_fraction_numerator,
        "tail_fraction_denominator": tail_fraction_denominator,
        "tail_fraction_rounding": tail_fraction_rounding,
        "extra_tail_frames": extra_tail_frames,
        "extra_tail_frames_if_source_gt": extra_tail_frames_if_source_gt,
        "image_fields": image_fields,
        "source_pkls": len(pkl_paths),
        "written_pkls": 0,
        "dropped_short_pkls": 0,
        "source_frames": 0,
        "trimmed_frames": 0,
        "retained_frames": 0,
        "image_link_stats": {"linked": 0, "copied": 0, "exists": 0, "missing": 0, "absolute": 0, "unsafe": 0},
        "episodes": [],
    }

    for src_pkl in pkl_paths:
        data = read_pkl(src_pkl)
        messages = data["messages"]
        source_len = len(messages)
        applied_tail_frames = tail_frames
        if tail_fraction_numerator is not None and tail_fraction_denominator is not None:
            raw_fraction_frames = source_len * tail_fraction_numerator / tail_fraction_denominator
            if tail_fraction_rounding == "ceil":
                applied_tail_frames += math.ceil(raw_fraction_frames)
            else:
                applied_tail_frames += math.floor(raw_fraction_frames)
        if (
            extra_tail_frames_if_source_gt is not None
            and source_len > extra_tail_frames_if_source_gt
        ):
            applied_tail_frames += extra_tail_frames
        if source_len <= applied_tail_frames:
            if not args.drop_short:
                raise ValueError(
                    f"Episode is too short to trim {applied_tail_frames} tail frames without becoming empty: "
                    f"{src_pkl} frames={source_len}. Use --drop-short if this is intended."
                )
            summary["dropped_short_pkls"] += 1
            summary["source_frames"] += source_len
            summary["trimmed_frames"] += source_len
            summary["episodes"].append(
                {
                    "pkl": str(src_pkl),
                    "source_frames": source_len,
                    "applied_tail_frames": applied_tail_frames,
                    "retained_frames": 0,
                    "dropped": True,
                }
            )
            continue

        retained = messages[: source_len - applied_tail_frames] if applied_tail_frames else list(messages)
        rel_pkl = src_pkl.relative_to(input_root)
        dst_pkl = output_root / rel_pkl
        dst_episode_dir = dst_pkl.parent
        src_episode_dir = src_pkl.parent
        link_stats = {"linked": 0, "copied": 0, "exists": 0, "missing": 0, "absolute": 0, "unsafe": 0}
        if not args.dry_run:
            link_stats = maybe_link_message_images(
                src_episode_dir=src_episode_dir,
                dst_episode_dir=dst_episode_dir,
                messages=retained,
                image_fields=image_fields,
            )
            write_trimmed_pkl(src_pkl, dst_pkl, data, retained)

        summary["written_pkls"] += 1
        summary["source_frames"] += source_len
        summary["trimmed_frames"] += source_len - len(retained)
        summary["retained_frames"] += len(retained)
        for key, value in link_stats.items():
            summary["image_link_stats"][key] += int(value)
        summary["episodes"].append(
            {
                "source_pkl": str(src_pkl),
                "output_pkl": str(dst_pkl),
                "source_frames": source_len,
                "retained_frames": len(retained),
                "trimmed_tail_frames": source_len - len(retained),
                "applied_tail_frames": applied_tail_frames,
                "image_link_stats": link_stats,
            }
        )

    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        if extra_tail_frames and extra_tail_frames_if_source_gt is not None:
            report_name = f"trim_tail{tail_frames}_gt{extra_tail_frames_if_source_gt}_extra{extra_tail_frames}_report.json"
        elif tail_fraction_numerator is not None and tail_fraction_denominator is not None:
            report_name = f"trim_tail_fraction_{tail_fraction_numerator}_of_{tail_fraction_denominator}_report.json"
        else:
            report_name = f"trim_tail{tail_frames}_report.json"
        write_manifest(
            output_root,
            "10_trim_episode_tail",
            {
                **summary,
                "episodes_report": str(output_root / report_name),
            },
            dry_run=False,
        )
        (output_root / report_name).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
