#!/usr/bin/env python3
"""Extract one camera stream from a DexUMI pkl image dataset.

The script keeps episode names and selected stream-relative paths unchanged,
while writing into a new dataset root. It also rewrites each episode pkl so
metadata and per-frame messages only contain the selected camera stream.

  python /home/zjc/Desktop/human2dex/tools/extract_camera_stream_dataset.py \
    --stream wrist \
    --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw \
    --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw_wrist_only


"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_INPUT = Path(
    "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw"
)

STREAM_ALIASES = {
    "wrist": "wrist",
    "l515/ego": "l515/ego",
    "l515/ego/rgb": "l515/ego",
    "ego": "l515/ego",
    "l515/external": "l515/external",
    "l515/external/rgb": "l515/external",
    "external": "l515/external",
}

TOP_LEVEL_WRIST_RGB_PREFIXES = ("rgb",)


def natural_key(path: Path):
    import re

    text = path.name
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def normalize_stream(value: str) -> str:
    key = str(value).strip().replace("\\", "/").strip("/").lower()
    if key not in STREAM_ALIASES:
        allowed = ", ".join(sorted(set(STREAM_ALIASES.values())))
        raise argparse.ArgumentTypeError(
            f"unsupported stream '{value}'. Expected one of: {allowed}"
        )
    return STREAM_ALIASES[key]


def default_output_root(input_root: Path, stream: str) -> Path:
    suffix = stream.replace("/", "_")
    return input_root.parent / f"{input_root.name}_{suffix}_only"


def iter_episode_dirs(root: Path) -> Iterable[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=natural_key)


def selected_stream_dir(episode_dir: Path, stream: str) -> Path:
    return episode_dir / stream


def selected_l515_camera(stream: str) -> Optional[str]:
    if stream == "l515/ego":
        return "ego"
    if stream == "l515/external":
        return "external"
    return None


def is_top_level_wrist_rgb_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lower = key.lower()
    return any(lower.startswith(prefix) for prefix in TOP_LEVEL_WRIST_RGB_PREFIXES)


def prune_message(message: Any, stream: str) -> Any:
    if not isinstance(message, dict):
        return message

    camera = selected_l515_camera(stream)
    if camera is None:
        message.pop("l515", None)
        return message

    for key in list(message.keys()):
        if is_top_level_wrist_rgb_key(key):
            message.pop(key, None)

    l515 = message.get("l515")
    if isinstance(l515, dict):
        if camera in l515:
            message["l515"] = {camera: l515[camera]}
        else:
            message["l515"] = {}
    return message


def prune_metadata(metadata: Any, stream: str) -> Any:
    if not isinstance(metadata, dict):
        return metadata

    camera = selected_l515_camera(stream)
    if camera is None:
        metadata.pop("l515", None)
        return metadata

    metadata.pop("rgb", None)
    l515 = metadata.get("l515")
    if isinstance(l515, dict):
        cameras = l515.get("cameras")
        if isinstance(cameras, dict):
            l515["cameras"] = {camera: cameras[camera]} if camera in cameras else {}
    return metadata


def prune_pkl_object(obj: Any, stream: str) -> Any:
    if not isinstance(obj, dict):
        return obj

    prune_metadata(obj.get("metadata"), stream)
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in messages:
            prune_message(message, stream)
    return obj


def load_filter_save_pkl(src: Path, dst: Path, stream: str) -> None:
    with src.open("rb") as f:
        obj = pickle.load(f)
    obj = prune_pkl_object(obj, stream)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def filter_l515_intrinsics(data: Any, camera: str) -> Any:
    if not isinstance(data, dict):
        return data
    cameras = data.get("cameras")
    if isinstance(cameras, dict):
        data["cameras"] = {camera: cameras[camera]} if camera in cameras else {}
    return data


def copy_l515_intrinsics(src: Path, dst: Path, camera: str) -> None:
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = filter_l515_intrinsics(data, camera)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def copy_file(src: Path, dst: Path, overwrite: bool, hardlink: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, overwrite: bool, hardlink: bool) -> int:
    if not src.is_dir():
        return 0
    copied = 0
    for src_path in sorted(src.rglob("*")):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(src)
        copy_file(src_path, dst / rel, overwrite=overwrite, hardlink=hardlink)
        copied += 1
    return copied


def copy_support_files(
    episode_dir: Path,
    out_episode: Path,
    stream: str,
    overwrite: bool,
    hardlink: bool,
) -> int:
    copied = 0
    camera = selected_l515_camera(stream)
    for src_path in sorted(episode_dir.iterdir(), key=natural_key):
        if src_path.is_dir() or src_path.suffix == ".pkl":
            continue
        if src_path.name == "l515_intrinsics.json":
            if camera is None:
                continue
            copy_l515_intrinsics(src_path, out_episode / src_path.name, camera)
            copied += 1
            continue
        copy_file(src_path, out_episode / src_path.name, overwrite=overwrite, hardlink=hardlink)
        copied += 1
    return copied


def process_episode(
    episode_dir: Path,
    output_root: Path,
    stream: str,
    overwrite: bool,
    hardlink: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    out_episode = output_root / episode_dir.name
    src_stream = selected_stream_dir(episode_dir, stream)
    pkl_files = sorted(episode_dir.glob("*.pkl"), key=natural_key)

    result = {
        "episode": episode_dir.name,
        "stream": stream,
        "skipped": False,
        "reason": "",
        "stream_files": 0,
        "support_files": 0,
        "pkl_files": len(pkl_files),
    }

    if not src_stream.is_dir():
        result.update(skipped=True, reason=f"missing stream directory: {stream}")
        return result
    if not pkl_files:
        result.update(skipped=True, reason="missing pkl")
        return result
    if out_episode.exists() and not overwrite:
        result.update(skipped=True, reason="output episode exists")
        return result

    if dry_run:
        result["stream_files"] = sum(1 for p in src_stream.rglob("*") if p.is_file())
        return result

    if out_episode.exists() and overwrite:
        shutil.rmtree(out_episode)
    out_episode.mkdir(parents=True, exist_ok=True)

    result["stream_files"] = copy_tree(
        src_stream,
        out_episode / stream,
        overwrite=overwrite,
        hardlink=hardlink,
    )
    result["support_files"] = copy_support_files(
        episode_dir,
        out_episode,
        stream,
        overwrite=overwrite,
        hardlink=hardlink,
    )

    for pkl_path in pkl_files:
        load_filter_save_pkl(pkl_path, out_episode / pkl_path.name, stream)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract wrist, l515/ego, or l515/external from a DexUMI pkl dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input dataset root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dataset root. Default: <input>_<stream>_only.",
    )
    parser.add_argument(
        "--stream",
        required=True,
        type=normalize_stream,
        help="Stream to keep: wrist, l515/ego, or l515/external.",
    )
    parser.add_argument("--episode", action="append", default=[], help="Only process this episode name. Repeatable.")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Process at most N episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output episodes.")
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Hardlink stream/support files when possible, falling back to copy2.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input.expanduser().resolve()
    output_root = args.output
    if output_root is None:
        output_root = default_output_root(input_root, args.stream)
    output_root = output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise SystemExit(f"input dataset root not found: {input_root}")
    if input_root == output_root:
        raise SystemExit("output must be different from input")

    episodes = list(iter_episode_dirs(input_root))
    if args.episode:
        wanted = set(args.episode)
        episodes = [p for p in episodes if p.name in wanted]
        missing = sorted(wanted - {p.name for p in episodes})
        if missing:
            raise SystemExit(f"episodes not found: {', '.join(missing)}")
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    print(f"input: {input_root}")
    print(f"output: {output_root}")
    print(f"stream: {args.stream}")
    print("output episode structure:")
    if args.stream == "wrist":
        print("  <output>/<episode>/wrist/<original images>")
        print("  <output>/<episode>/<original pkl>")
    else:
        print(f"  <output>/<episode>/{args.stream}/rgb/<original rgb images>")
        print(f"  <output>/<episode>/{args.stream}/depth/<original depth images>")
        print("  <output>/<episode>/l515_intrinsics.json")
        print("  <output>/<episode>/<original pkl>")

    processed = 0
    skipped = 0
    stream_files = 0
    for episode_dir in episodes:
        stats = process_episode(
            episode_dir,
            output_root,
            args.stream,
            overwrite=args.overwrite,
            hardlink=args.hardlink,
            dry_run=args.dry_run,
        )
        if stats["skipped"]:
            skipped += 1
            print(f"SKIP {stats['episode']}: {stats['reason']}")
            continue
        processed += 1
        stream_files += int(stats["stream_files"])
        print(
            f"OK {stats['episode']}: stream_files={stats['stream_files']} "
            f"support_files={stats['support_files']} pkl_files={stats['pkl_files']}"
        )

    print(
        f"done: processed={processed} skipped={skipped} "
        f"stream_files={stream_files} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
