#!/usr/bin/env python3
"""Create human G/L fisheye views centered on a direct RGB grasp-pocket field.

``wrist_grasp_pocket_uv`` is the only geometric anchor.  This tool never reads
PICO, MANO, palm width, or a camera-to-hand fit.  G is a wide fisheye ray
rotation; L is a fixed-angular ray view.  Source PKLs are never modified.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import pickle
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from fisheye_canonical_views import FisheyeCanonicalViewRenderer, fisheye_pixels_to_unit_rays  # noqa: E402


CAMERA_ACROSS = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def load_structured(path: Path) -> dict[str, Any]:
    text = path.expanduser().read_text(encoding="utf-8")
    if path.suffix.lower() == ".json": return json.loads(text)
    import yaml
    value = yaml.safe_load(text)
    if not isinstance(value, dict): raise ValueError(f"Expected mapping in {path}")
    return value


def read_pkl(path: Path) -> dict[str, Any]:
    try: importlib.import_module("numpy._core")
    except ImportError:
        core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", core); sys.modules.setdefault("numpy._core.numeric", core.numeric); sys.modules.setdefault("numpy._core.multiarray", core.multiarray)
    with path.open("rb") as handle: value = pickle.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("messages"), list): raise ValueError(f"Not a DexUMI PKL: {path}")
    return value


def atomic_pickle(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2: raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def parse_pair(value: str) -> tuple[float, float]:
    values = [float(v) for v in value.split(",")]
    if len(values) != 2: raise argparse.ArgumentTypeError("expected X,Y")
    return values[0], values[1]


def natural(path: Path) -> list[Any]: return [int(v) if v.isdigit() else v for v in re.split(r"(\d+)", path.as_posix())]


def selected_pkls(root: Path, episodes: str | None, limit: int | None, regex: str | None) -> list[Path]:
    paths = sorted(root.rglob("*.pkl"), key=natural)
    if regex:
        expression = re.compile(regex); paths = [path for path in paths if expression.fullmatch(path.parent.name)]
    if episodes:
        wanted = {token if token.startswith("episode_") else f"episode_{int(token):04d}" for token in episodes.split(",") if token.strip()}
        paths = [path for path in paths if path.parent.name in wanted]
    if limit is not None: paths = paths[:limit]
    if not paths: raise FileNotFoundError("No selected PKLs")
    return paths


def hardlink_or_copy(source: str, destination: str) -> str:
    try: os.link(source, destination)
    except OSError: shutil.copy2(source, destination)
    return destination


def load_camera(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    value = load_structured(path)
    K = np.asarray(value.get("camera_matrix", value.get("K")), dtype=np.float64)
    D = np.asarray(value.get("dist_coeffs", value.get("D")), dtype=np.float64).reshape(-1)
    width = int(value.get("image_width", value.get("image_size", [0, 0])[0])); height = int(value.get("image_height", value.get("image_size", [0, 0])[1]))
    if K.shape != (3, 3) or D.shape != (4,) or width < 2 or height < 2: raise ValueError("intrinsics need 3x3 K, four D and image size")
    return K, D.reshape(4, 1), (width, height)


def direct_geometry(message: dict[str, Any], image_hw: tuple[int, int], K: np.ndarray, D: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    uv = np.asarray(message.get(args.pocket_uv_field), dtype=np.float64).reshape(-1)
    confidence = float(message.get(args.pocket_confidence_field, 0.0))
    height, width = image_hw
    if uv.shape != (2,) or not np.all(np.isfinite(uv)) or not (0 <= uv[0] < width and 0 <= uv[1] < height): raise ValueError("invalid direct pocket uv")
    if not bool(message.get(args.pocket_valid_field, False)) or confidence < args.min_confidence: raise ValueError(f"invalid/low confidence direct pocket ({confidence:.3f})")
    return fisheye_pixels_to_unit_rays(uv[None], K, D)[0], confidence


def render(image: np.ndarray, ray: np.ndarray, wide: FisheyeCanonicalViewRenderer, local: FisheyeCanonicalViewRenderer, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    global_view, _ = wide.render_wide(image, ray, CAMERA_ACROSS)
    if args.local_mode == "anchor_wide":
        local_view, _ = local.render_wide(image, ray, CAMERA_ACROSS)
    elif args.local_mode == "angular":
        local_view = local.render_local_angular(image, ray, CAMERA_ACROSS, horizontal_fov_deg=args.local_fov_deg)
    else:
        raise ValueError(f"Unknown local mode: {args.local_mode}")
    return global_view, local_view


def build_renderers(
        K: np.ndarray,
        D: np.ndarray,
        source_size: tuple[int, int],
        args: argparse.Namespace,
        global_border: str,
        local_border: str,
) -> tuple[FisheyeCanonicalViewRenderer, FisheyeCanonicalViewRenderer]:
    wide = FisheyeCanonicalViewRenderer(
        K, D, output_size=args.global_size, source_size=source_size,
        anchor_uv_ratio=args.anchor_uv_ratio, border=global_border,
    )
    local = FisheyeCanonicalViewRenderer(
        K, D, output_size=args.local_size, source_size=args.local_size,
        anchor_uv_ratio=args.local_anchor_uv_ratio, border=local_border,
    )
    return wide, local


def materialize_episode_worker(
        source_pkl_text: str,
        source_root_text: str,
        output_root_text: str,
        args: argparse.Namespace,
        K: np.ndarray,
        D: np.ndarray,
        source_size: tuple[int, int],
        global_border: str,
        local_border: str,
        metadata_settings: dict[str, Any],
        opencv_threads: int,
) -> tuple[str, dict[str, int]]:
    """Copy and materialize one episode in an isolated process.

    Episodes do not share output files, so process-level parallelism preserves
    PKL/image provenance while allowing independent JPEG encode and remap work.
    """
    cv2.setNumThreads(max(1, int(opencv_threads)))
    cv2.setUseOptimized(True)
    source_pkl = Path(source_pkl_text)
    source_root = Path(source_root_text)
    output_root = Path(output_root_text)
    source_dir = source_pkl.parent
    target_dir = output_root / source_dir.relative_to(source_root)
    shutil.copytree(source_dir, target_dir, copy_function=hardlink_or_copy)
    target_pkl = target_dir / source_pkl.name
    data = read_pkl(target_pkl)
    wide, local = build_renderers(K, D, source_size, args, global_border, local_border)
    global_dir = target_dir / "canonical_views" / "global"
    local_dir = target_dir / "canonical_views" / "local"
    global_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    for frame, message in enumerate(data["messages"]):
        if not isinstance(message, dict):
            counts["non_dict"] += 1
            continue
        image = cv2.imread(str(target_dir / str(message.get(args.source_image_field, ""))), cv2.IMREAD_COLOR)
        try:
            if image is None:
                raise FileNotFoundError(message.get(args.source_image_field))
            ray, conf = direct_geometry(message, image.shape[:2], K, D, args)
            global_view, local_view = render(image, ray, wide, local, args)
            global_rel = Path("canonical_views") / "global" / f"frame_{frame:06d}.jpg"
            local_rel = Path("canonical_views") / "local" / f"frame_{frame:06d}.jpg"
            cv2.imwrite(
                str(target_dir / global_rel), global_view,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            cv2.imwrite(
                str(target_dir / local_rel), local_view,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            update_materialized_message(message, frame, conf, args, counts)
        except Exception as exc:
            message.update({
                args.global_image_field: None,
                args.local_image_field: None,
                "humanGraspPocketValid": False,
                "humanCanonicalViewError": repr(exc),
            })
            counts["error"] += 1
    metadata = data.setdefault("metadata", {})
    metadata["humanCanonicalViews"] = {
        **metadata_settings,
        "episodeCounts": dict(counts),
    }
    atomic_pickle(data, target_pkl)
    return source_dir.name, dict(counts)


def resolve_num_workers(requested: int) -> int:
    value = int(requested)
    if value < 0:
        raise ValueError("--num-workers must be >= 0")
    if value > 0:
        return value
    # 64 independent episodes gives the 256-core server enough work while
    # avoiding hundreds of simultaneous image streams on the shared filesystem.
    return min(64, max(1, int(os.cpu_count() or 1)))


def parse_devices(value: str) -> list[str]:
    devices = [token.strip() for token in str(value).split(",") if token.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one CUDA device, e.g. 0,1")
    return [token if token.startswith("cuda:") else f"cuda:{int(token)}" for token in devices]


class TorchBatchFisheyeRenderer:
    """Numerically checked CUDA implementation of the existing wide ray remap.

    It intentionally supports only the active ``anchor_wide`` local mode.  The
    input rays, camera model and target bases are identical to the OpenCV path;
    only batched projection and sampling are moved to CUDA.
    """

    def __init__(
            self,
            wide: FisheyeCanonicalViewRenderer,
            local: FisheyeCanonicalViewRenderer,
            device: str):
        try:
            import torch
            import torch.nn.functional as torch_functional
        except ImportError as exc:
            raise RuntimeError("CUDA renderer requires PyTorch in the selected environment") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA renderer requested, but torch.cuda.is_available() is false")
        self.torch = torch
        self.functional = torch_functional
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.K = torch.as_tensor(wide.K, dtype=torch.float32, device=self.device)
        self.D = torch.as_tensor(wide.D.reshape(-1), dtype=torch.float32, device=self.device)
        self.wide = self._make_view(wide, "zeros")
        self.local = self._make_view(local, "reflection")

    def _make_view(self, renderer: FisheyeCanonicalViewRenderer, padding_mode: str) -> dict[str, Any]:
        return {
            "output_rays": self.torch.as_tensor(renderer._output_rays, dtype=self.torch.float32, device=self.device),
            "target_basis": self.torch.as_tensor(renderer._target_basis, dtype=self.torch.float32, device=self.device),
            "output_size": renderer.output_size,
            "padding_mode": padding_mode,
        }

    def _normalize(self, value):
        return value / self.torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-9)

    def _source_basis(self, pocket):
        torch = self.torch
        z_axis = self._normalize(pocket)
        across = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand_as(z_axis)
        x_axis = across - (across * z_axis).sum(dim=-1, keepdim=True) * z_axis
        x_norm = torch.linalg.vector_norm(x_axis, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand_as(z_axis)
        fallback = fallback - (fallback * z_axis).sum(dim=-1, keepdim=True) * z_axis
        x_axis = torch.where(x_norm > 1e-7, x_axis, fallback)
        x_axis = self._normalize(x_axis)
        y_axis = self._normalize(torch.linalg.cross(z_axis, x_axis, dim=-1))
        x_axis = self._normalize(torch.linalg.cross(y_axis, z_axis, dim=-1))
        return torch.stack((x_axis, y_axis, z_axis), dim=-1)

    def _project_fisheye(self, rays):
        torch = self.torch
        values = self._normalize(rays)
        z = values[..., 2]
        z = torch.where(z.abs() > 1e-8, z, torch.where(z >= 0, torch.full_like(z, 1e-8), torch.full_like(z, -1e-8)))
        a = values[..., 0] / z
        b = values[..., 1] / z
        radius = torch.sqrt(a * a + b * b)
        theta = torch.atan(radius)
        theta2 = theta * theta
        theta_distorted = theta * (
            1.0 + self.D[0] * theta2 + self.D[1] * theta2.square()
            + self.D[2] * theta2.pow(3) + self.D[3] * theta2.pow(4)
        )
        scale = torch.where(radius > 1e-8, theta_distorted / radius, torch.ones_like(radius))
        return torch.stack((
            self.K[0, 0] * a * scale + self.K[0, 2],
            self.K[1, 1] * b * scale + self.K[1, 2],
        ), dim=-1)

    def _sample_view(self, images, pockets, view: dict[str, Any]):
        torch = self.torch
        source_basis = self._source_basis(pockets)
        rotation = view["target_basis"].unsqueeze(0) @ source_basis.transpose(-1, -2)
        source_rays = torch.einsum("nc,bcd->bnd", view["output_rays"], rotation)
        source_uv = self._project_fisheye(source_rays)
        height, width = images.shape[-2:]
        grid = torch.empty_like(source_uv)
        grid[..., 0] = 2.0 * source_uv[..., 0] / float(width - 1) - 1.0
        grid[..., 1] = 2.0 * source_uv[..., 1] / float(height - 1) - 1.0
        out_width, out_height = view["output_size"]
        return self.functional.grid_sample(
            images,
            grid.reshape(images.shape[0], out_height, out_width, 2),
            mode="bilinear",
            padding_mode=view["padding_mode"],
            align_corners=True,
        )

    def render_pair(self, images: list[np.ndarray], rays: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not images:
            return [], []
        shape = images[0].shape
        if any(image.shape != shape for image in images):
            raise ValueError("CUDA batch requires equal image shapes")
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Expected HxWx3 images, got {shape}")
        torch = self.torch
        with torch.inference_mode():
            source = np.ascontiguousarray(np.stack(images, axis=0))
            tensor = torch.from_numpy(source).to(self.device, dtype=torch.float32, non_blocking=False)
            tensor = tensor.permute(0, 3, 1, 2).div_(255.0)
            pocket = torch.as_tensor(rays, dtype=torch.float32, device=self.device)
            global_view = self._sample_view(tensor, pocket, self.wide)
            local_view = self._sample_view(tensor, pocket, self.local)
            global_uint8 = global_view.permute(0, 2, 3, 1).mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            local_uint8 = local_view.permute(0, 2, 3, 1).mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return list(global_uint8), list(local_uint8)


def update_materialized_message(
        message: dict[str, Any],
        frame: int,
        confidence: float,
        args: argparse.Namespace,
        counts: Counter[str],
) -> tuple[Path, Path]:
    global_rel = Path("canonical_views") / "global" / f"frame_{frame:06d}.jpg"
    local_rel = Path("canonical_views") / "local" / f"frame_{frame:06d}.jpg"
    message.update({
        args.global_image_field: global_rel.as_posix(),
        args.local_image_field: local_rel.as_posix(),
        "humanGraspPocketUV": np.asarray(message[args.pocket_uv_field], dtype=np.float32),
        "humanGraspPocketConfidence": confidence,
        "humanGraspPocketValid": True,
        "humanCanonicalViewSource": "direct_rgb_pocket_head_v1",
        "canonicalSourceFrameIndex": frame,
    })
    counts["valid"] += 1
    return global_rel, local_rel


def materialize_episode_cuda_worker(
        source_pkl_text: str,
        source_root_text: str,
        output_root_text: str,
        args: argparse.Namespace,
        K: np.ndarray,
        D: np.ndarray,
        source_size: tuple[int, int],
        global_border: str,
        local_border: str,
        metadata_settings: dict[str, Any],
        opencv_threads: int,
        device: str,
        batch_size: int,
) -> tuple[str, dict[str, int]]:
    """CUDA batch version of one isolated output episode."""
    if args.local_mode != "anchor_wide":
        raise ValueError("CUDA renderer currently supports only --local-mode anchor_wide")
    if int(batch_size) < 1:
        raise ValueError("--gpu-batch-size must be >= 1")
    cv2.setNumThreads(max(1, int(opencv_threads)))
    cv2.setUseOptimized(True)
    source_pkl = Path(source_pkl_text)
    source_root = Path(source_root_text)
    output_root = Path(output_root_text)
    source_dir = source_pkl.parent
    target_dir = output_root / source_dir.relative_to(source_root)
    shutil.copytree(source_dir, target_dir, copy_function=hardlink_or_copy)
    target_pkl = target_dir / source_pkl.name
    data = read_pkl(target_pkl)
    wide, local = build_renderers(K, D, source_size, args, global_border, local_border)
    renderer = TorchBatchFisheyeRenderer(wide, local, device)
    global_dir = target_dir / "canonical_views" / "global"
    local_dir = target_dir / "canonical_views" / "local"
    global_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    pending: list[tuple[int, dict[str, Any], np.ndarray, np.ndarray, float]] = []

    def flush_pending() -> None:
        if not pending:
            return
        frames = [record[2] for record in pending]
        rays = np.asarray([record[3] for record in pending], dtype=np.float32)
        globals_out, locals_out = renderer.render_pair(frames, rays)
        for (frame, message, _image, _ray, confidence), global_view, local_view in zip(pending, globals_out, locals_out):
            global_rel, local_rel = update_materialized_message(message, frame, confidence, args, counts)
            cv2.imwrite(
                str(target_dir / global_rel), global_view,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            cv2.imwrite(
                str(target_dir / local_rel), local_view,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
        pending.clear()

    for frame, message in enumerate(data["messages"]):
        if not isinstance(message, dict):
            counts["non_dict"] += 1
            continue
        image = cv2.imread(str(target_dir / str(message.get(args.source_image_field, ""))), cv2.IMREAD_COLOR)
        try:
            if image is None:
                raise FileNotFoundError(message.get(args.source_image_field))
            ray, confidence = direct_geometry(message, image.shape[:2], K, D, args)
            if pending and (image.shape != pending[0][2].shape or len(pending) >= int(batch_size)):
                flush_pending()
            pending.append((frame, message, image, ray, confidence))
        except Exception as exc:
            message.update({
                args.global_image_field: None,
                args.local_image_field: None,
                "humanGraspPocketValid": False,
                "humanCanonicalViewError": repr(exc),
            })
            counts["error"] += 1
    flush_pending()
    metadata = data.setdefault("metadata", {})
    metadata["humanCanonicalViews"] = {
        **metadata_settings,
        "episodeCounts": dict(counts),
    }
    atomic_pickle(data, target_pkl)
    return source_dir.name, dict(counts)


def materialize_cuda_shard_worker(
        source_pkl_texts: list[str],
        common_args: tuple[Any, ...],
        device: str,
        batch_size: int,
) -> list[tuple[str, dict[str, int]]]:
    return [
        materialize_episode_cuda_worker(
            source_pkl_text, *common_args, device, batch_size
        )
        for source_pkl_text in source_pkl_texts
    ]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--episodes", default=None); parser.add_argument("--limit-episodes", type=int, default=None); parser.add_argument("--episode-name-regex", default=None)
    parser.add_argument("--source-image-field", default="rgbImage"); parser.add_argument("--pocket-uv-field", default="wrist_grasp_pocket_uv"); parser.add_argument("--pocket-valid-field", default="wrist_grasp_pocket_valid"); parser.add_argument("--pocket-confidence-field", default="wrist_grasp_pocket_confidence")
    parser.add_argument("--min-confidence", type=float, default=0.10); parser.add_argument("--global-size", type=parse_size, default=(224, 224)); parser.add_argument("--local-size", type=parse_size, default=(224, 224)); parser.add_argument("--anchor-uv-ratio", type=parse_pair, default=(0.5, 0.35), help="G anchor; keep the pocket above center to retain the forward workspace."); parser.add_argument("--local-anchor-uv-ratio", type=parse_pair, default=(0.5, 0.5), help="L anchor; keep the pocket at image center, matching deployment local view."); parser.add_argument("--local-mode", choices=("anchor_wide", "angular"), default="anchor_wide"); parser.add_argument("--local-fov-deg", type=float, default=78.0); parser.add_argument("--border-mode", default=None, help="Backward-compatible fallback for both views."); parser.add_argument("--global-border-mode", default=None); parser.add_argument("--local-border-mode", default=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest="command", required=True)
    preview = subs.add_parser("preview"); add_arguments(preview); preview.add_argument("--output", type=Path, required=True); preview.add_argument("--stride", type=int, default=4); preview.add_argument("--max-frames-per-episode", type=int, default=40)
    materialize = subs.add_parser("materialize"); add_arguments(materialize); materialize.add_argument("--output", type=Path, required=True); materialize.add_argument("--global-image-field", default="globalCanonicalImage"); materialize.add_argument("--local-image-field", default="localCanonicalImage"); materialize.add_argument("--jpeg-quality", type=int, default=95); materialize.add_argument("--renderer", choices=("cpu", "cuda"), default="cpu", help="CPU OpenCV multiprocess renderer or numerically checked CUDA batch renderer."); materialize.add_argument("--num-workers", type=int, default=0, help="CPU episode workers; 0 selects up to 64 workers. Ignored by CUDA mode."); materialize.add_argument("--opencv-threads", type=int, default=1, help="OpenCV threads per worker; keep at 1 when using multiple workers."); materialize.add_argument("--devices", default=None, help="CUDA devices for --renderer cuda, e.g. 0,1,2,3."); materialize.add_argument("--gpu-batch-size", type=int, default=64, help="Frames per CUDA remap batch on each GPU.")
    args = parser.parse_args(); source_root = args.input.expanduser().resolve(); pkls = selected_pkls(source_root, args.episodes, args.limit_episodes, args.episode_name_regex)
    K, D, size = load_camera(args.intrinsics)
    global_border = args.global_border_mode or args.border_mode or "constant"
    local_border = args.local_border_mode or args.border_mode or "reflect101"
    wide, local = build_renderers(K, D, size, args, global_border, local_border)
    if args.command == "preview":
        output = args.output.expanduser().resolve(); output.mkdir(parents=True, exist_ok=True); report: dict[str, Any] = {"format": "human_direct_pocket_preview_v1", "counts": Counter(), "frames": []}
        for pkl in pkls:
            data = read_pkl(pkl); saved = 0
            for frame, message in enumerate(data["messages"]):
                if saved >= args.max_frames_per_episode or frame % args.stride or not isinstance(message, dict): continue
                source = pkl.parent / str(message.get(args.source_image_field, "")); image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                try:
                    if image is None: raise FileNotFoundError(source)
                    ray, conf = direct_geometry(message, image.shape[:2], K, D, args); g, l = render(image, ray, wide, local, args)
                    raw = image.copy(); uv = np.asarray(message[args.pocket_uv_field], dtype=np.float64); cv2.circle(raw, tuple(np.rint(uv).astype(int)), 7, (0, 220, 255), -1, cv2.LINE_AA); cv2.putText(raw, f"direct pocket conf={conf:.2f}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245,245,245), 1, cv2.LINE_AA)
                    visual = np.concatenate([cv2.resize(raw, args.global_size), g, l], axis=1); target = output / pkl.parent.name / f"frame_{frame:06d}.jpg"; target.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(target), visual); report["frames"].append({"pkl": str(pkl), "frame": frame, "output": str(target), "confidence": conf}); report["counts"]["valid"] += 1; saved += 1
                except Exception as exc: report["counts"]["error"] += 1; report["frames"].append({"pkl": str(pkl), "frame": frame, "error": repr(exc)})
        report["counts"] = dict(report["counts"]); (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report["counts"], ensure_ascii=False)); return 0
    output_root = args.output.expanduser().resolve()
    if output_root.exists(): raise FileExistsError(f"output must not exist: {output_root}")
    output_root.mkdir(parents=True)
    if int(args.opencv_threads) < 1:
        raise ValueError("--opencv-threads must be >= 1")
    if args.renderer == "cpu":
        num_workers = resolve_num_workers(args.num_workers)
        devices: list[str] = []
    else:
        if not args.devices:
            raise ValueError("--renderer cuda requires explicit --devices, e.g. --devices 0,1,2,3")
        num_workers = 0
        devices = parse_devices(args.devices)
    settings = {
        "localMode": args.local_mode,
        "localFovDeg": args.local_fov_deg,
        "globalAnchorUvRatio": args.anchor_uv_ratio,
        "localAnchorUvRatio": args.local_anchor_uv_ratio,
        "globalBorderMode": global_border,
        "localBorderMode": local_border,
        "source": "direct_rgb_pocket_head",
        "rendererBackend": args.renderer,
        "numWorkers": num_workers,
        "opencvThreadsPerWorker": int(args.opencv_threads),
        "cudaDevices": devices,
        "gpuBatchSize": int(args.gpu_batch_size) if args.renderer == "cuda" else None,
    }
    report: dict[str, Any] = {
        "format": "human_direct_pocket_materialize_v1",
        "sourceRoot": str(source_root),
        "counts": Counter(),
        "settings": settings,
    }
    worker_args = (
        str(source_root), str(output_root), args, K, D, size,
        global_border, local_border, settings, int(args.opencv_threads),
    )
    if args.renderer == "cpu":
        if num_workers == 1:
            results = (
                materialize_episode_worker(str(source_pkl), *worker_args)
                for source_pkl in pkls
            )
            for episode_name, counts in results:
                report["counts"].update(counts)
                print(f"{episode_name}: {counts}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                pending = {
                    pool.submit(materialize_episode_worker, str(source_pkl), *worker_args): source_pkl
                    for source_pkl in pkls
                }
                for future in as_completed(pending):
                    episode_name, counts = future.result()
                    report["counts"].update(counts)
                    print(f"{episode_name}: {counts}", flush=True)
    else:
        if args.local_mode != "anchor_wide":
            raise ValueError("--renderer cuda currently supports only --local-mode anchor_wide")
        shards = [[] for _ in devices]
        for index, source_pkl in enumerate(pkls):
            shards[index % len(devices)].append(str(source_pkl))
        common_args = worker_args
        with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp.get_context("spawn")) as pool:
            pending = {
                pool.submit(materialize_cuda_shard_worker, shard, common_args, device, int(args.gpu_batch_size)): device
                for shard, device in zip(shards, devices)
                if shard
            }
            for future in as_completed(pending):
                for episode_name, counts in future.result():
                    report["counts"].update(counts)
                    print(f"{episode_name}: {counts}", flush=True)
    report["counts"] = dict(report["counts"]); (output_root / "human_direct_pocket_views_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report["counts"], ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
