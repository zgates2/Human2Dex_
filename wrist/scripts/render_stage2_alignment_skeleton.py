#!/usr/bin/env python3
"""Render Stage2 + 4DoF-aligned hand skeleton overlays for wrist RGB images.

This is an offline data-generation/QC tool. It does not train a diffusion
policy and does not modify source images. It reads images, predicts Stage2
MANO joints, applies a trained 4DoF alignment head, then writes skeleton-RGB
images plus a manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
ALIGN_SCRIPT = WRIST_ROOT / "scripts" / "train_stage2_4dof_alignment.py"
DEFAULT_ALIGNMENT_CHECKPOINT = Path(
    "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/wrist_2D/"
    "stage2_4dof_alignment_signscan/none_xy_1_1/checkpoints/best.pt"
)

if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.constants import HAND_BONES  # noqa: E402
from wrist_pose.stage2_infer import load_stage2_model  # noqa: E402
from wrist_pose.transforms import WristImageTransform, read_rgb  # noqa: E402


POINT_COLORS_BGR = {
    "palm_center": (80, 80, 255),
    "thumb_tip": (80, 180, 255),
    "index_tip": (80, 255, 120),
    "middle_tip": (255, 210, 80),
    "ring_tip": (255, 120, 180),
    "pinky_tip": (180, 120, 255),
}
JOINT_COLORS_BGR = [
    (245, 245, 245),
    *((80, 180, 255) for _ in range(4)),
    *((80, 255, 120) for _ in range(4)),
    *((255, 210, 80) for _ in range(4)),
    *((255, 120, 180) for _ in range(4)),
    *((180, 120, 255) for _ in range(4)),
]


@dataclass(frozen=True)
class ImageRecord:
    index: int
    image: Path
    image_rel: str
    output_rel: str


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def torch_load_trusted(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_alignment_module():
    spec = importlib.util.spec_from_file_location("stage2_4dof_alignment_module", ALIGN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import alignment script: {ALIGN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_image_list(path: Path) -> list[Path]:
    images: list[Path] = []
    base = path.parent
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        img = Path(line)
        if not img.is_absolute():
            img = base / img
        images.append(img.expanduser().resolve())
    return images


def list_images_in_dir(path: Path) -> list[Path]:
    return sorted(
        [
            item.expanduser().resolve()
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )


def discover_images(input_path: Path, episodes: Iterable[str]) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        return read_image_list(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if (input_path / "images").is_dir():
        return list_images_in_dir(input_path / "images")
    direct = list_images_in_dir(input_path)
    if direct:
        return direct

    requested = {item for item in episodes if item}
    images: list[Path] = []
    for child in sorted(input_path.iterdir(), key=natural_key):
        if not child.is_dir():
            continue
        if requested and child.name not in requested:
            continue
        if (child / "images").is_dir():
            images.extend(list_images_in_dir(child / "images"))
    if requested:
        found = {path.parent.parent.name for path in images if path.parent.name == "images"}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"episodes not found or contain no images: {', '.join(missing)}")
    return images


def safe_output_rel(index: int, image_path: Path, input_root: Path | None, preserve_tree: bool) -> str:
    suffix = image_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"
    if preserve_tree and input_root is not None:
        try:
            rel = image_path.relative_to(input_root)
            return str(rel.with_suffix(".jpg"))
        except ValueError:
            pass
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)
    return f"{index:06d}_{digest}_{stem}.jpg"


def make_records(
    image_paths: list[Path],
    *,
    input_path: Path,
    start: int,
    stride: int,
    limit: int | None,
    preserve_tree: bool,
) -> list[ImageRecord]:
    stride = max(1, int(stride))
    start = max(0, int(start))
    sampled = image_paths[start::stride]
    if limit is not None:
        sampled = sampled[: int(limit)]
    input_root = input_path.expanduser().resolve() if input_path.expanduser().is_dir() else None
    records: list[ImageRecord] = []
    for idx, image in enumerate(sampled):
        if not image.is_file():
            raise FileNotFoundError(f"image does not exist: {image}")
        rel = str(image.relative_to(input_root)) if input_root is not None and image.is_relative_to(input_root) else image.name
        records.append(
            ImageRecord(
                index=idx,
                image=image,
                image_rel=rel,
                output_rel=safe_output_rel(idx, image, input_root, preserve_tree),
            )
        )
    return records


class RenderDataset(Dataset):
    def __init__(self, records: list[ImageRecord], image_size: int) -> None:
        self.records = list(records)
        self.transform = WristImageTransform(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        rgb = read_rgb(str(record.image))
        result = self.transform(rgb)
        return {
            "image": result.image,
            "scale": torch.tensor(float(result.scale), dtype=torch.float32),
            "offset_xy": torch.tensor(result.offset_xy, dtype=torch.float32),
            "original_hw": torch.tensor(result.original_hw, dtype=torch.float32),
            "record_index": torch.tensor(idx, dtype=torch.long),
        }


def invert_uv_to_original(uv: np.ndarray, scale: float, offset_xy: np.ndarray) -> np.ndarray:
    out = uv.copy()
    out[:, 0] = (out[:, 0] - float(offset_xy[0])) / float(scale)
    out[:, 1] = (out[:, 1] - float(offset_xy[1])) / float(scale)
    return out


def valid_uv_mask(uv: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    return (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h)
    )


def draw_full21(image_bgr: np.ndarray, uv21: np.ndarray, line_width: int, point_radius: int) -> np.ndarray:
    out = image_bgr.copy()
    valid = valid_uv_mask(uv21, out)
    for a, b in HAND_BONES:
        if not (valid[a] and valid[b]):
            continue
        pa = tuple(np.round(uv21[a]).astype(int))
        pb = tuple(np.round(uv21[b]).astype(int))
        cv2.line(out, pa, pb, JOINT_COLORS_BGR[b], int(line_width), cv2.LINE_AA)
    for j, uv in enumerate(uv21):
        if not valid[j]:
            continue
        cv2.circle(out, tuple(np.round(uv).astype(int)), int(point_radius), JOINT_COLORS_BGR[j], -1, cv2.LINE_AA)
    return out


def draw_anchors(
    image_bgr: np.ndarray,
    anchor_uv: np.ndarray,
    labels: list[str],
    line_width: int,
    point_radius: int,
) -> np.ndarray:
    out = image_bgr.copy()
    valid = valid_uv_mask(anchor_uv, out)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    palm_idx = label_to_idx.get("palm_center")
    if palm_idx is not None and bool(valid[palm_idx]):
        palm = tuple(np.round(anchor_uv[palm_idx]).astype(int))
        for label in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"):
            idx = label_to_idx.get(label)
            if idx is None or not bool(valid[idx]):
                continue
            tip = tuple(np.round(anchor_uv[idx]).astype(int))
            cv2.line(out, palm, tip, POINT_COLORS_BGR[label], int(line_width), cv2.LINE_AA)
    for idx, label in enumerate(labels):
        if not bool(valid[idx]):
            continue
        p = tuple(np.round(anchor_uv[idx]).astype(int))
        cv2.circle(out, p, int(point_radius), POINT_COLORS_BGR.get(label, (255, 255, 255)), -1, cv2.LINE_AA)
    return out


def write_image(path: Path, image_bgr: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    if not cv2.imwrite(str(path), image_bgr, params):
        raise RuntimeError(f"failed to write image: {path}")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def choose_precision(requested: str, device: torch.device) -> str:
    if requested == "fp32" or device.type != "cuda":
        return "fp32"
    if requested == "bf16":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp16"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image dir, episode dir, dataset root, single image, or txt file of image paths.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for skeleton images and manifest.")
    parser.add_argument("--alignment-checkpoint", type=Path, default=DEFAULT_ALIGNMENT_CHECKPOINT)
    parser.add_argument("--stage2-checkpoint", type=Path, default=None, help="Override Stage2 checkpoint. By default uses alignment checkpoint source.")
    parser.add_argument("--episode", action="append", default=[], help="Episode name to include when input is a dataset root. Repeatable.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--mode", choices=("full21", "anchors", "both"), default="both")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preserve-tree", action="store_true", help="Preserve input relative paths under output mode dirs when possible.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only list discovered images and checkpoint metadata.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image_paths = discover_images(args.input, args.episode)
    records = make_records(
        image_paths,
        input_path=args.input,
        start=int(args.start),
        stride=int(args.stride),
        limit=args.limit,
        preserve_tree=bool(args.preserve_tree),
    )
    if not records:
        raise ValueError(f"no images selected from input: {args.input}")

    align_ckpt_path = args.alignment_checkpoint.expanduser().resolve()
    align_ckpt = torch_load_trusted(align_ckpt_path)
    align_cfg = align_ckpt.get("config", {})
    anchor_def = align_ckpt.get("anchor_definition", {})
    labels = list(align_ckpt.get("labels") or align_cfg.get("labels") or [])
    if not labels:
        labels = ["palm_center", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    source_axes = tuple(int(v) for v in anchor_def.get("source_axes", align_cfg.get("source_axes", [0, 1])))
    source_signs = tuple(float(v) for v in anchor_def.get("source_signs", align_cfg.get("source_signs", [1.0, 1.0])))
    stage2_checkpoint = args.stage2_checkpoint or Path(str(align_ckpt.get("checkpoint_source") or align_cfg.get("checkpoint")))

    print(
        json.dumps(
            {
                "input": str(args.input),
                "selected_images": len(records),
                "output_dir": str(args.output_dir),
                "mode": args.mode,
                "alignment_checkpoint": str(align_ckpt_path),
                "stage2_checkpoint": str(stage2_checkpoint),
                "labels": labels,
                "source_axes": list(source_axes),
                "source_signs": list(source_signs),
                "sample_images": [str(record.image) for record in records[:10]],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return 0

    alignment_module = load_alignment_module()
    device = resolve_device(str(args.device))
    precision = choose_precision(str(args.precision), device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"loading Stage2 checkpoint: {stage2_checkpoint}")
    stage2, _transform, cfg = load_stage2_model(
        checkpoint_path=stage2_checkpoint,
        repo_root=REPO_ROOT,
        device=device,
        config_path=None,
    )
    if "stage2_model" in align_ckpt:
        stage2.load_state_dict(align_ckpt["stage2_model"], strict=True)
        stage2.to(device).eval()

    image_size = int(cfg["data"]["image_size"])
    feature_dim = int(cfg["model"].get("feature_dim", 384))
    head = alignment_module.Alignment4DoFHead(
        feature_dim=feature_dim,
        hidden_dim=256,
        dropout=0.1,
        init_scale=float(align_cfg.get("init_scale", 1800.0)),
    ).to(device)
    head.load_state_dict(align_ckpt["alignment_head"], strict=True)
    head.eval()
    stage2.eval()

    dataset = RenderDataset(records, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(int(args.num_workers) > 0),
    )
    output_dir = args.output_dir.expanduser().resolve()
    outputs: list[dict[str, Any]] = []
    written = 0
    skipped = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            pred = alignment_module.forward_alignment(
                stage2,
                head,
                image,
                labels,
                source_axes,
                source_signs,
                image_size,
                float(align_cfg.get("max_theta_rad", np.pi)),
                float(align_cfg.get("min_scale", 100.0)),
                float(align_cfg.get("max_scale", 8000.0)),
                train_stage2=False,
                precision=precision,
                device=device,
            )
            pred_full = pred["pred_full_uv"].detach().cpu().float().numpy()
            pred_anchor = pred["pred_anchor_uv"].detach().cpu().float().numpy()
            theta = pred["theta"].detach().cpu().float().numpy()
            scale_pred = pred["scale"].detach().cpu().float().numpy()
            translation = pred["translation"].detach().cpu().float().numpy()
            transform_scales = batch["scale"].numpy()
            offsets = batch["offset_xy"].numpy()
            record_indices = batch["record_index"].numpy()
            for local_idx, record_idx in enumerate(record_indices):
                record = records[int(record_idx)]
                image_bgr = cv2.imread(str(record.image), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise FileNotFoundError(f"failed to read image: {record.image}")
                uv21 = invert_uv_to_original(pred_full[local_idx], float(transform_scales[local_idx]), offsets[local_idx])
                anchor_uv = invert_uv_to_original(pred_anchor[local_idx], float(transform_scales[local_idx]), offsets[local_idx])
                item: dict[str, Any] = {
                    "index": int(record.index),
                    "source_image": record.image,
                    "image_rel": record.image_rel,
                    "output_rel": record.output_rel,
                    "uv21": uv21,
                    "anchor_uv": {label: anchor_uv[i] for i, label in enumerate(labels)},
                    "theta": float(theta[local_idx]),
                    "scale": float(scale_pred[local_idx]),
                    "translation": translation[local_idx],
                }
                if args.mode in {"full21", "both"}:
                    out_path = output_dir / "full21" / record.output_rel
                    item["full21_image"] = out_path
                    if args.skip_existing and out_path.exists():
                        skipped += 1
                    else:
                        write_image(out_path, draw_full21(image_bgr, uv21, args.line_width, args.point_radius), int(args.jpeg_quality))
                        written += 1
                if args.mode in {"anchors", "both"}:
                    out_path = output_dir / "anchors" / record.output_rel
                    item["anchors_image"] = out_path
                    if args.skip_existing and out_path.exists():
                        skipped += 1
                    else:
                        write_image(out_path, draw_anchors(image_bgr, anchor_uv, labels, args.line_width, args.point_radius), int(args.jpeg_quality))
                        written += 1
                outputs.append(item)

    manifest = {
        "version": 1,
        "task": "render_stage2_alignment_skeleton",
        "input": args.input,
        "output_dir": output_dir,
        "mode": args.mode,
        "alignment_checkpoint": align_ckpt_path,
        "stage2_checkpoint": stage2_checkpoint,
        "labels": labels,
        "source_axes": list(source_axes),
        "source_signs": list(source_signs),
        "image_size": int(image_size),
        "num_images": len(records),
        "written_images": int(written),
        "skipped_images": int(skipped),
        "records": outputs,
    }
    atomic_write_json(output_dir / "manifest.json", to_jsonable(manifest))
    print(f"done: {output_dir}")
    print(f"manifest: {output_dir / 'manifest.json'}")
    print(f"written_images={written} skipped_images={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
