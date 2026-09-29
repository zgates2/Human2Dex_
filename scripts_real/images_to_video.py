#!/usr/bin/env python3
"""Convert a directory of still images into an MP4 video."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_images(image_dir: Path, glob_pattern: str | None) -> list[Path]:
    if not image_dir.is_dir():
        raise NotADirectoryError(f"image directory does not exist: {image_dir}")

    if glob_pattern:
        candidates = [path for path in image_dir.glob(glob_pattern) if path.is_file()]
    else:
        candidates = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    images = sorted(candidates, key=natural_key)
    if not images:
        suffixes = ", ".join(sorted(IMAGE_EXTENSIONS))
        hint = f" matching {glob_pattern!r}" if glob_pattern else f" with extensions {suffixes}"
        raise FileNotFoundError(f"no images found in {image_dir}{hint}")
    return [path.resolve() for path in images]


def default_output_path(image_dir: Path) -> Path:
    parent = image_dir.parent
    return parent / f"{parent.name}_{image_dir.name}.mp4"


def create_numbered_sequence(images: list[Path], sequence_dir: Path) -> Path:
    suffix = images[0].suffix.lower() or ".img"
    for index, image in enumerate(images):
        target = sequence_dir / f"frame_{index:06d}{suffix}"
        try:
            target.symlink_to(image)
        except OSError:
            shutil.copy2(image, target)
    return sequence_dir / f"frame_%06d{suffix}"


def encode_video(
    images: list[Path],
    output: Path,
    *,
    fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="images_to_video_") as tmp_dir:
        input_pattern = create_numbered_sequence(images, Path(tmp_dir))
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-start_number",
            "0",
            "-i",
            str(input_pattern),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output video frame rate. Default: 10.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output MP4 path. Default: <image-dir-parent>/<episode>_images.mp4.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Optional filename glob relative to --image-dir, e.g. 'episode_0007_rgb_*.jpg'.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 quality, lower is better/larger. Default: 18.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 preset. Default: medium.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output if it already exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(image_dir).resolve()
    )
    images = find_images(image_dir, args.glob)
    encode_video(
        images,
        output,
        fps=float(args.fps),
        crf=int(args.crf),
        preset=str(args.preset),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({
        "image_dir": str(image_dir),
        "frames": len(images),
        "fps": float(args.fps),
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
