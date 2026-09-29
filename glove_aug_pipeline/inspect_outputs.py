#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

from PIL import Image, ImageDraw

from common import evenly_spaced_indices, list_episode_dirs, list_images, write_json


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Inspect a mirrored augmented dataset against the source dataset."
    )
    parser.add_argument("--input", required=True, type=Path, help="Source dataset root.")
    parser.add_argument("--output", required=True, type=Path, help="Augmented dataset root.")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report JSON path. Default: <output>_inspect/report.json.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Directory for source/augmented side-by-side previews.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes to preview.")
    parser.add_argument("--frames", type=int, default=4, help="Frames per preview episode.")
    return parser


def pkl_message_count(episode_dir: Path):
    pkl_files = list(episode_dir.glob("*.pkl"))
    if not pkl_files:
        return None
    try:
        with pkl_files[0].open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            return len(obj["messages"])
    except Exception:
        return None
    return None


def make_pair_preview(src_path: Path, aug_path: Path, out_path: Path):
    src = Image.open(src_path).convert("RGB")
    aug = Image.open(aug_path).convert("RGB")
    if aug.size != src.size:
        aug = aug.resize(src.size, Image.Resampling.LANCZOS)
    gap = 8
    canvas = Image.new("RGB", (src.width * 2 + gap, src.height), (20, 20, 20))
    canvas.paste(src, (0, 0))
    canvas.paste(aug, (src.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "source", fill=(255, 255, 255))
    draw.text((src.width + gap + 8, 8), "augmented", fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main():
    args = build_arg_parser().parse_args()
    report_path = args.report or Path(str(args.output) + "_inspect") / "report.json"
    preview_dir = args.preview_dir or report_path.parent / "previews"
    src_eps = {ep.name: ep for ep in list_episode_dirs(args.input)}
    out_eps = {ep.name: ep for ep in list_episode_dirs(args.output)}

    missing = sorted(set(src_eps) - set(out_eps))
    extra = sorted(set(out_eps) - set(src_eps))
    common = sorted(set(src_eps) & set(out_eps))
    mismatches = []

    for ep_name in common:
        src_images = list_images(src_eps[ep_name])
        out_images = list_images(out_eps[ep_name])
        src_pkl_count = pkl_message_count(src_eps[ep_name])
        out_pkl_count = pkl_message_count(out_eps[ep_name])
        item = {
            "episode": ep_name,
            "source_images": len(src_images),
            "output_images": len(out_images),
            "source_pkl_messages": src_pkl_count,
            "output_pkl_messages": out_pkl_count,
            "ok": len(src_images) == len(out_images)
            and src_pkl_count == out_pkl_count
            and (out_pkl_count is None or out_pkl_count == len(out_images)),
        }
        if not item["ok"]:
            mismatches.append(item)

    for ep_name in common[: args.episodes]:
        src_images = list_images(src_eps[ep_name])
        out_images = list_images(out_eps[ep_name])
        by_name = {p.name: p for p in out_images}
        for idx in evenly_spaced_indices(len(src_images), args.frames):
            src_img = src_images[idx]
            aug_img = by_name.get(src_img.name)
            if aug_img is None:
                continue
            make_pair_preview(
                src_img,
                aug_img,
                preview_dir / ep_name / f"{idx:06d}_{src_img.stem}_pair.jpg",
            )

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "source_episode_count": len(src_eps),
        "output_episode_count": len(out_eps),
        "missing_episodes": missing,
        "extra_episodes": extra,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:200],
        "preview_dir": str(preview_dir),
    }
    write_json(report_path, report)
    print(f"report: {report_path}")
    print(f"previews: {preview_dir}")
    if missing or extra or mismatches:
        print(
            f"issues: missing={len(missing)} extra={len(extra)} mismatches={len(mismatches)}"
        )
    else:
        print("structure check passed")


if __name__ == "__main__":
    main()
