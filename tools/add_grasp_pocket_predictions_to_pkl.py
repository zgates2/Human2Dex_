#!/usr/bin/env python3
"""Write RGB-only direct ``grasp_pocket`` predictions into copied or source PKLs.

This tool has no retargeting and never uses PICO/MANO points as an input to the
pocket head.  It is deliberately separate from the wrist-action writer so the
visual G/L anchor has an auditable field contract.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
WRIST_ROOT = REPO_ROOT / "wrist"
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.stage2_infer import load_stage2_model  # noqa: E402
from wrist_pose.stage2_projection import forward_projection_from_stage1_outputs, load_projection_head, transform_model_uv_to_original, uv_valid_mask  # noqa: E402
from wrist_pose.transforms import WristImageTransform, read_rgb  # noqa: E402


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == "numpy._core" or module.startswith("numpy._core."):
                return super().find_class("numpy.core" + module[len("numpy._core"):], name)
            raise


def read_pkl(path: Path) -> dict[str, Any]:
    try:
        importlib.import_module("numpy._core")
    except ImportError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.numeric", core.numeric)
        sys.modules.setdefault("numpy._core.multiarray", core.multiarray)
    with path.open("rb") as handle:
        value = NumpyCompatUnpickler(handle).load()
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
        raise ValueError(f"PKL has no messages list: {path}")
    return value


def atomic_write_pkl(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def load_pkl_list(path: Path, source_root: Path) -> list[Path]:
    """Load a newline-delimited PKL shard, rejecting paths outside the data root.

    This is intentionally only a work-partitioning option.  It does not alter
    model inference, field names, confidence handling, or PKL rewrite logic.
    ``03_backfill_grasp_pocket.py`` uses it to give each GPU a disjoint set of
    episodes while retaining this script's original single-device behavior.
    """
    if not path.is_file():
        raise FileNotFoundError(f"--pkl-list does not exist: {path}")
    selected: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        candidate = Path(text).expanduser()
        candidate = candidate if candidate.is_absolute() else source_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"--pkl-list line {line_number} is outside --data-root: {candidate}"
            ) from exc
        if not candidate.is_file() or candidate.suffix.lower() != ".pkl":
            raise FileNotFoundError(f"--pkl-list line {line_number} is not a PKL file: {candidate}")
        if candidate not in seen:
            selected.append(candidate)
            seen.add(candidate)
    if not selected:
        raise ValueError(f"--pkl-list has no usable PKL paths: {path}")
    return sorted(selected)


def copy_selected_pkl_parents(source_root: Path, output_root: Path, source_pkls: list[Path]) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=False)
    copied_dirs: set[Path] = set()
    output_pkls: list[Path] = []
    for source_pkl in source_pkls:
        relative_parent = source_pkl.parent.relative_to(source_root)
        source_parent = source_root / relative_parent
        output_parent = output_root / relative_parent
        if relative_parent not in copied_dirs:
            output_parent.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_parent, output_parent, copy_function=hardlink_or_copy)
            copied_dirs.add(relative_parent)
        output_pkls.append(output_parent / source_pkl.name)
    return output_pkls


class FrameDataset(Dataset):
    def __init__(self, items: list[tuple[int, Path]], transform: WristImageTransform) -> None:
        self.items, self.transform = items, transform
    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int) -> dict[str, Any]:
        frame, path = self.items[index]
        try:
            result = self.transform(read_rgb(str(path)))
            return {"ok": True, "frame": frame, "image": result.image, "scale": float(result.scale), "offset": tuple(float(v) for v in result.offset_xy), "hw": tuple(int(v) for v in result.original_hw)}
        except Exception as exc:
            return {"ok": False, "frame": frame, "error": f"{type(exc).__name__}: {exc}"}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    good = [item for item in batch if item["ok"]]
    return {"images": None if not good else torch.stack([item["image"] for item in good]), "frames": [item["frame"] for item in good], "scales": [item["scale"] for item in good], "offsets": [item["offset"] for item in good], "hws": [item["hw"] for item in good], "errors": [(item["frame"], item["error"]) for item in batch if not item["ok"]]}


def resolve_image(pkl: Path, message: dict[str, Any], field: str) -> Path | None:
    value = message.get(field)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (pkl.parent / path).resolve()


def choose_precision(value: str, device: torch.device) -> str:
    if device.type != "cuda" or value == "fp32": return "fp32"
    if value == "auto": return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return value


def autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32": return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def set_error(message: dict[str, Any], error: str) -> None:
    message.update({"wrist_grasp_pocket_uv": None, "wrist_grasp_pocket_valid": False, "wrist_grasp_pocket_confidence": 0.0, "wrist_grasp_pocket_error": error, "wrist_grasp_pocket_source": "direct_rgb_pocket_head_v1"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None, help="Preferred: create a new copied dataset root before writing fields.")
    parser.add_argument("--in-place", action="store_true", help="Explicitly allow modifying --data-root. Mutually exclusive with --output-root.")
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--pocket-head-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--image-field", default="rgbImage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="auto")
    parser.add_argument("--min-confidence", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-pkls", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument(
        "--pkl-list",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited PKL paths relative to --data-root. "
            "Used only to partition independent work across devices."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source_root = args.data_root.expanduser().resolve()
    if bool(args.output_root) == bool(args.in_place):
        raise ValueError("Specify exactly one of --output-root or --in-place")
    source_pkls = (
        load_pkl_list(args.pkl_list.expanduser().resolve(), source_root)
        if args.pkl_list is not None
        else sorted(source_root.rglob("*.pkl"))
    )
    if args.limit_pkls is not None: source_pkls = source_pkls[:args.limit_pkls]
    if not source_pkls: raise FileNotFoundError(f"No PKLs under {source_root}")
    root = source_root
    pkls = source_pkls
    if args.output_root is not None:
        root = args.output_root.expanduser().resolve()
        if root.exists():
            raise FileExistsError(f"--output-root must not already exist: {root}")
        pkls = [root / p.relative_to(source_root) for p in source_pkls]
        if not args.dry_run:
            pkls = copy_selected_pkl_parents(source_root, root, source_pkls)
    if args.limit_pkls is not None: pkls = pkls[:args.limit_pkls]
    if not pkls: raise FileNotFoundError(f"No PKLs under {root}")
    print({"source_root": str(source_root), "output_root": str(root), "pkls": len(pkls), "stage1": str(args.stage1_checkpoint), "pocket_head": str(args.pocket_head_checkpoint), "overwrite": args.overwrite})
    if args.dry_run: return 0
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    stage1, transform, cfg = load_stage2_model(checkpoint_path=args.stage1_checkpoint, repo_root=REPO_ROOT, device=device, config_path=args.config)
    feature_dim = int(cfg["model"].get("feature_dim", 384))
    head, info = load_projection_head(args.pocket_head_checkpoint, feature_dim, device)
    if info.projection_head_type != "direct_pocket" or info.labels != ["grasp_pocket"]:
        raise ValueError(f"Expected direct_pocket checkpoint with labels=[grasp_pocket], got type={info.projection_head_type}, labels={info.labels}")
    precision = choose_precision(args.precision, device)
    stats: Counter[str] = Counter()
    for pkl in pkls:
        data = read_pkl(pkl); messages = data["messages"]
        items: list[tuple[int, Path]] = []
        max_frames = len(messages) if args.limit_frames is None else min(len(messages), args.limit_frames)
        for frame in range(max_frames):
            message = messages[frame]
            if not isinstance(message, dict): stats["non_dict"] += 1; continue
            if not args.overwrite and message.get("wrist_grasp_pocket_error") is None and bool(message.get("wrist_grasp_pocket_valid")):
                stats["already_done"] += 1; continue
            image = resolve_image(pkl, message, args.image_field)
            if image is None or not image.is_file(): set_error(message, f"missing image: {args.image_field}"); stats["missing_image"] += 1; continue
            items.append((frame, image))
        if not items:
            atomic_write_pkl(data, pkl)
            print(f"{pkl.parent.name}: {dict(stats)}", flush=True)
            continue
        loader = DataLoader(FrameDataset(items, transform), batch_size=max(1, args.batch_size), shuffle=False, num_workers=max(0, args.num_workers), pin_memory=device.type == "cuda", collate_fn=collate, persistent_workers=False)
        for batch in loader:
            for frame, error in batch["errors"]: set_error(messages[frame], error); stats["read_error"] += 1
            if batch["images"] is None: continue
            images = batch["images"].to(device, non_blocking=True)
            try:
                with torch.no_grad():
                    with autocast(device, precision): outputs = stage1(images, return_tokens=True)
                    result = forward_projection_from_stage1_outputs(outputs, head, info.labels, int(transform.image_size), info.min_scale, info.max_scale)
                uv_batch = result["grasp_pocket_uv"].detach().cpu().float().numpy()
                conf_batch = result["grasp_pocket_confidence"].detach().cpu().float().numpy()
            except Exception as exc:
                for frame in batch["frames"]: set_error(messages[frame], f"model error: {type(exc).__name__}: {exc}"); stats["model_error"] += 1
                continue
            for idx, frame in enumerate(batch["frames"]):
                uv = transform_model_uv_to_original(uv_batch[idx][None], batch["scales"][idx], batch["offsets"][idx])[0]
                confidence = float(conf_batch[idx]); valid = bool(uv_valid_mask(uv[None], batch["hws"][idx])[0] and confidence >= args.min_confidence)
                messages[frame].update({"wrist_grasp_pocket_uv": np.asarray(uv, dtype=np.float32), "wrist_grasp_pocket_valid": valid, "wrist_grasp_pocket_confidence": confidence, "wrist_grasp_pocket_error": None if valid else f"low confidence ({confidence:.3f}) or out of image", "wrist_grasp_pocket_source": "direct_rgb_pocket_head_v1", "wrist_grasp_pocket_checkpoint": str(Path(args.pocket_head_checkpoint).resolve())})
                stats["valid" if valid else "low_confidence"] += 1
        atomic_write_pkl(data, pkl)
        del loader
        gc.collect()
        print(f"{pkl.parent.name}: {dict(stats)}", flush=True)
    print(dict(stats)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
