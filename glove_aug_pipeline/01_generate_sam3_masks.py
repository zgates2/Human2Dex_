#!/usr/bin/env python3
"""Generate human hand masks with SAM3 text prompts.

用途：
    使用 SAM3 根据文本 prompt 自动识别人手，并为每张图片生成二值 mask。
    输出的 mask 可直接传给 `02_augment_dataset.py --mask-root` 做人手外观增强。

推荐运行方式：
    现在参数已经集中到 YAML，优先通过 `--config` 运行。命令行里只保留
    配置文件路径；如临时传入同名 CLI 参数，会覆盖 YAML 中的值。

配置文件：
    /home/zjc/Desktop/human2dex/glove_aug_pipeline/01_generate_sam3_masks.yaml

    
----------------------程序运行------------------------
    conda activate /share/project/liyuanyuan/anaconda3/envs/sam3

    cd /home/zjc/Desktop/human2dex

#使用默认 YAML 配置处理 pick_3_raw 数据集的 wrist 视角
python glove_aug_pipeline/01_generate_sam3_masks.py \
    --config glove_aug_pipeline/01_generate_sam3_masks.yaml \
    --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw \
    --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw_wrist_sam3_masks \
    --image-subdir wrist \
    --overwrite

    python /home/zjc/Desktop/human2dex/glove_aug_pipeline/01_generate_sam3_masks.py \
      --config /home/zjc/Desktop/human2dex/glove_aug_pipeline/01_generate_sam3_masks.yaml


    python /home/zjc/Desktop/human2dex/glove_aug_pipeline/01_generate_sam3_masks.py --config /home/zjc/Desktop/human2dex/glove_aug_pipeline/01_generate_sam3_masks.yaml


    跑完先看：
        <output>/overlays/      质检 SAM3 mask 是否覆盖人手
        <output>/stats/         每帧 score、面积、bbox 等统计
        <output>/summary.json   本次批处理摘要


临时覆盖 YAML：
    如果只想临时改输出或帧数，不必编辑 YAML，可以在命令行追加同名参数：
        python /home/zjc/Desktop/human2dex/glove_aug_pipeline/01_generate_sam3_masks.py --config /home/zjc/Desktop/human2dex/glove_aug_pipeline/01_generate_sam3_masks.yaml --output /tmp/pick_sponge_sam3_masks_test --limit-frames 20


常用 YAML 配置项：
    prompt                         文本提示列表，可以补充多个 prompt
    candidate_confidence_threshold SAM3 内部候选召回阈值，低一些能找回不完整人手
    confidence_threshold           最终置信度参考阈值
    force_single_instance          强制每帧选 1 个最像人手的实例
    glove_prior_weight             兼容旧参数名，实际是人手形状/外观先验
    track_overlap_weight           偏向和上一帧重叠的候选，提高时序稳定性
    dilate_iterations              mask 略微膨胀，补一点不完整边缘
    reuse_previous_on_miss         某帧完全没有候选时复用上一帧 mask
    min_mask_area                  过滤很小的噪声 mask
    max_area_ratio                 过滤覆盖画面过大的误检
    qc_frames                      每个 episode 保存多少张 overlay 质检图
    overwrite                      是否覆盖已有 mask；false 时可断点续跑

输出结构：
    <output>/
      masks/<episode>/<image_stem>_mask.png      二值 mask，白色为人手区域
      overlays/<episode>/*_overlay.jpg           原图和 mask overlay，供人工质检
      stats/<episode>.json                       每帧 mask 面积、bbox、score 等统计
      summary.json                               本次批处理摘要
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from tqdm import tqdm

from common import (
    bbox_from_mask,
    evenly_spaced_indices,
    load_rgb,
    natural_key,
)


DEFAULT_INPUT = Path("/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge")
DEFAULT_OUTPUT = Path("/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_sam3_masks")
DEFAULT_CHECKPOINT = Path("/home/zjc/Desktop/human2dex/sam3/sam3_pt/sam3.pt")
DEFAULT_SAM3_CODE = Path("/home/zjc/Desktop/human2dex/sam3/sam3-code")
DEFAULT_IMAGE_SUBDIRS = ("images", "l515/ego/rgb", "l515/external/rgb")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _normalize_image_subdir(value) -> str:
    text = str(value).strip().replace("\\", "/").strip("/")
    return text or "images"


def _normalize_image_subdirs(value) -> List[str]:
    if value is None:
        items = DEFAULT_IMAGE_SUBDIRS
    elif isinstance(value, (str, Path)):
        items = [value]
    else:
        items = value

    result = []
    seen = set()
    for item in items:
        if item is None:
            continue
        parts = str(item).split(",") if isinstance(item, str) else [item]
        for part in parts:
            subdir = _normalize_image_subdir(part)
            if subdir not in seen:
                result.append(subdir)
                seen.add(subdir)
    return result or ["images"]


def _image_dir_for_stream(episode_dir: Path, image_subdir: str) -> Path:
    return Path(episode_dir) / _normalize_image_subdir(image_subdir)


def _list_stream_images(episode_dir: Path, image_subdir: str) -> List[Path]:
    image_dir = _image_dir_for_stream(episode_dir, image_subdir)
    if not image_dir.is_dir():
        return []
    return sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def _stream_output_dir(root: Path, episode_name: str, image_subdir: str) -> Path:
    base = Path(root) / episode_name
    subdir = _normalize_image_subdir(image_subdir)
    if subdir == "images":
        return base
    return base / subdir


def _stream_stats_path(output_root: Path, episode_name: str, image_subdir: str) -> Path:
    stats_root = Path(output_root) / "stats"
    subdir = _normalize_image_subdir(image_subdir)
    if subdir == "images":
        return stats_root / f"{episode_name}.json"
    return stats_root / episode_name / Path(subdir).with_suffix(".json")


def _relative_image_path(episode_dir: Path, image_path: Path) -> str:
    try:
        return str(Path(image_path).relative_to(episode_dir))
    except ValueError:
        return str(image_path)


def _stream_frame_count(episode_dir: Path, image_subdir: str, args) -> int:
    count = len(_list_stream_images(episode_dir, image_subdir))
    if args.limit_frames is not None:
        return min(count, args.limit_frames)
    return count


def _increment_frame_counter(frame_counter, frame_lock=None, amount: int = 1) -> bool:
    if frame_counter is None or amount <= 0:
        return False
    if frame_lock is not None:
        with frame_lock:
            frame_counter.value += amount
    else:
        get_lock = getattr(frame_counter, "get_lock", None)
        if get_lock is None:
            frame_counter.value += amount
        else:
            with get_lock():
                frame_counter.value += amount
    return True


class _ImagePrefetcher:
    """Prefetch images from disk using a thread pool."""

    def __init__(self, paths: Sequence[Path], lookahead: int = 8):
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._futures: deque[Future] = deque()
        self._paths = paths
        self._idx = 0
        for _ in range(min(lookahead, len(paths))):
            self._submit_next()

    def _load(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _submit_next(self):
        if self._idx < len(self._paths):
            self._futures.append(self._pool.submit(self._load, self._paths[self._idx]))
            self._idx += 1

    def get(self) -> Image.Image:
        fut = self._futures.popleft()
        self._submit_next()
        return fut.result()

    def shutdown(self):
        self._pool.shutdown(wait=False)


class _AsyncSaver:
    """Async mask/overlay writer backed by a thread pool."""

    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: List[Future] = []

    @staticmethod
    def _write_mask(data: np.ndarray, path: Path):
        Image.fromarray(data, mode="L").save(path)

    @staticmethod
    def _write_overlay(overlay: Image.Image, path: Path, quality: int):
        overlay.save(path, quality=quality)

    def save_mask(self, mask: np.ndarray, path: Path):
        data = (mask.astype(np.uint8) * 255)
        self._futures.append(self._pool.submit(self._write_mask, data, path))

    def save_overlay(self, overlay: Image.Image, path: Path, quality: int = 92):
        self._futures.append(self._pool.submit(self._write_overlay, overlay, path, quality))

    def flush(self):
        wait(self._futures)
        self._futures.clear()

    def shutdown(self):
        self.flush()
        self._pool.shutdown(wait=False)


def _slice_backbone_out(backbone_out: Dict, idx: int) -> Dict:
    """Slice per-image features from a batched backbone output."""
    sliced = {}
    sliced["vision_features"] = backbone_out["vision_features"][idx:idx+1].contiguous()
    sliced["vision_pos_enc"] = [p[idx:idx+1].contiguous() for p in backbone_out["vision_pos_enc"]]
    sliced["backbone_fpn"] = [f[idx:idx+1].contiguous() for f in backbone_out["backbone_fpn"]]
    if backbone_out.get("sam2_backbone_out") is not None:
        sam2 = backbone_out["sam2_backbone_out"]
        sliced["sam2_backbone_out"] = {
            "vision_features": sam2["vision_features"][idx:idx+1].contiguous(),
            "vision_pos_enc": [p[idx:idx+1].contiguous() for p in sam2["vision_pos_enc"]],
            "backbone_fpn": [f[idx:idx+1].contiguous() for f in sam2["backbone_fpn"]],
        }
    else:
        sliced["sam2_backbone_out"] = None
    return sliced


def _tensor_to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask & False
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    return labels == int(np.argmax(areas))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union <= 0:
        return 0.0
    return float(inter / union)


def _dedupe_instances(instances: List[Dict], iou_threshold: float) -> List[Dict]:
    selected: List[Dict] = []
    for item in sorted(instances, key=lambda x: x["score"], reverse=True):
        if any(_mask_iou(item["mask"], old["mask"]) >= iou_threshold for old in selected):
            continue
        selected.append(item)
    return selected


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _hand_prior_score(image: np.ndarray, mask: np.ndarray, args) -> float:
    """Score whether a candidate has plausible hand geometry and appearance."""
    if not mask.any():
        return 0.0

    pixels = image[mask].astype(np.float32)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    sat = (maxc - minc) / np.maximum(maxc, 1.0)

    visible_score = _clamp01((float(luma.mean()) - 20.0) / 150.0)
    not_gray_score = _clamp01(float(sat.mean()) / 0.45)
    skin_like = (
        (r > b * 0.85)
        & (g > b * 0.65)
        & (r > 28.0)
        & (g > 20.0)
        & (luma > 22.0)
    )
    skin_score = float(skin_like.mean()) if skin_like.size else 0.0

    area_ratio = float(mask.sum() / max(mask.size, 1))
    if area_ratio < args.prior_min_area_ratio:
        area_score = _clamp01(area_ratio / max(args.prior_min_area_ratio, 1e-6))
    elif area_ratio > args.prior_max_area_ratio:
        area_score = _clamp01(
            1.0
            - (area_ratio - args.prior_max_area_ratio)
            / max(args.max_area_ratio - args.prior_max_area_ratio, 1e-6)
        )
    else:
        area_score = 1.0

    box = bbox_from_mask(mask, padding=0)
    border_score = 1.0
    shape_score = 0.5
    if box is not None:
        h, w = mask.shape
        x0, y0, x1, y1 = box
        touches = int(x0 <= 1) + int(y0 <= 1) + int(x1 >= w - 1) + int(y1 >= h - 1)
        border_score = _clamp01(1.0 - 0.12 * touches)
        box_area = max((x1 - x0) * (y1 - y0), 1)
        fill_ratio = float(mask.sum() / box_area)
        if fill_ratio < 0.10:
            shape_score = _clamp01(fill_ratio / 0.10)
        elif fill_ratio > 0.85:
            shape_score = _clamp01(1.0 - (fill_ratio - 0.85) / 0.15)
        else:
            shape_score = 1.0

    return (
        0.30 * area_score
        + 0.20 * border_score
        + 0.18 * shape_score
        + 0.15 * visible_score
        + 0.10 * skin_score
        + 0.07 * not_gray_score
    )


def _draw_overlay(
    image: np.ndarray,
    merged_mask: np.ndarray,
    instances: Sequence[Dict],
    prompt_text: str,
) -> Image.Image:
    base = Image.fromarray(image.astype(np.uint8), mode="RGB")
    overlay = base.copy()
    tint = Image.new("RGB", overlay.size, (255, 40, 80))
    alpha = Image.fromarray((merged_mask.astype(np.uint8) * 125), mode="L")
    overlay = Image.composite(tint, overlay, alpha)

    draw = ImageDraw.Draw(overlay)
    colors = [(0, 255, 255), (255, 210, 40), (80, 255, 120), (190, 120, 255)]
    for idx, item in enumerate(instances):
        box = item.get("box")
        if box is None:
            box = bbox_from_mask(item["mask"], padding=0)
        if box is None:
            continue
        color = colors[idx % len(colors)]
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        if item.get("reused_previous"):
            label = "prev"
        else:
            label = f"{item['score']:.2f}"
            if "rank_score" in item and abs(item["rank_score"] - item["score"]) > 1e-4:
                label += f"/{item['rank_score']:.2f}"
            if item.get("below_confidence"):
                label += "*"
        draw.text((x0 + 3, max(0, y0 - 14)), label, fill=color)

    draw.text((6, 6), prompt_text, fill=(255, 255, 255))
    gap = 8
    canvas = Image.new("RGB", (base.width * 2 + gap, base.height), (20, 20, 20))
    canvas.paste(base, (0, 0))
    canvas.paste(overlay, (base.width + gap, 0))
    return canvas


def _choose_episodes(args) -> List[Path]:
    root = Path(args.input)
    episodes = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir()
            and any(_list_stream_images(p, subdir) for subdir in args.image_subdirs)
        ],
        key=natural_key,
    )
    if args.episode:
        by_name = {p.name: p for p in episodes}
        missing = [name for name in args.episode if name not in by_name]
        if missing:
            raise FileNotFoundError(
                f"episodes not found under {args.input}: {', '.join(missing)}"
            )
        episodes = [by_name[name] for name in args.episode]
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]
    return sorted(episodes, key=natural_key)


def _make_stream_tasks(episodes: Sequence[Path], args) -> List[Tuple[Path, str]]:
    tasks: List[Tuple[Path, str]] = []
    for episode_dir in episodes:
        for image_subdir in args.image_subdirs:
            if _stream_frame_count(episode_dir, image_subdir, args) > 0:
                tasks.append((episode_dir, image_subdir))
    return tasks


def _load_sam3(args):
    if args.sam3_code and args.sam3_code.exists():
        sys.path.insert(0, str(args.sam3_code))

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. SAM3 image inference "
            "requires a CUDA-capable PyTorch runtime for the official code path."
        )
    if device == "cpu":
        raise RuntimeError(
            "CUDA is not available. The official SAM3 image model code path used "
            "here requires CUDA and cannot run on CPU in this environment."
        )

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=device,
        compile=args.compile,
    )
    # Some SAM3 submodules, especially the text encoder, may remain on CPU after
    # construction. Move the full model explicitly so prompt tokens and weights
    # are on the same GPU in multi-process inference.
    model = model.to(device).eval()
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=args.candidate_confidence_threshold,
    )
    return processor, device


def _amp_context(args, device: str):
    if not str(device).startswith("cuda") or args.amp_dtype == "float32":
        return nullcontext()

    import torch

    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return torch.autocast(
        device_type="cuda",
        dtype=dtype_by_name[args.amp_dtype],
    )


def _instances_from_output(output: Dict, prompt: str, args) -> List[Dict]:
    masks = _tensor_to_numpy(output.get("masks"))
    boxes = _tensor_to_numpy(output.get("boxes"))
    scores = _tensor_to_numpy(output.get("scores"))
    if masks is None or scores is None or masks.size == 0:
        return []

    masks = masks.astype(bool)
    if masks.ndim == 4:
        masks = masks[:, 0]
    elif masks.ndim != 3:
        raise ValueError(f"unexpected SAM3 mask shape: {masks.shape}")

    instances = []
    h, w = masks.shape[-2:]
    for idx, raw_mask in enumerate(masks):
        mask = raw_mask.astype(bool)
        if args.keep_largest_component:
            mask = _largest_component(mask)
        if args.close_iterations > 0:
            mask = ndimage.binary_closing(mask, iterations=args.close_iterations)
            mask = ndimage.binary_fill_holes(mask)
        if args.dilate_iterations > 0:
            mask = ndimage.binary_dilation(mask, iterations=args.dilate_iterations)
            mask = ndimage.binary_fill_holes(mask)

        area = int(mask.sum())
        area_ratio = float(area / max(h * w, 1))
        if area < args.min_mask_area or area_ratio > args.max_area_ratio:
            continue
        score = float(scores[idx])
        below_confidence = score < args.confidence_threshold
        if below_confidence and not args.force_single_instance:
            continue

        box = None
        if boxes is not None and idx < len(boxes):
            box = [float(v) for v in boxes[idx].tolist()]
        instances.append(
            {
                "mask": mask,
                "box": box,
                "score": score,
                "prompt": prompt,
                "area": area,
                "area_ratio": area_ratio,
                "below_confidence": below_confidence,
            }
        )
    return instances


def _rank_instances(
    instances: List[Dict],
    image: np.ndarray,
    prev_mask: Optional[np.ndarray],
    args,
) -> List[Dict]:
    ranked = []
    for item in instances:
        item = dict(item)
        if prev_mask is not None and prev_mask.any() and args.track_overlap_weight > 0:
            temporal = _mask_iou(item["mask"], prev_mask)
        else:
            temporal = 0.0
        if args.glove_prior_weight > 0:
            prior = _hand_prior_score(image, item["mask"], args)
        else:
            prior = 0.0
        item["temporal_iou"] = temporal
        item["hand_prior_score"] = prior
        item["glove_prior_score"] = prior
        item["rank_score"] = (
            item["score"]
            + args.track_overlap_weight * temporal
            + args.glove_prior_weight * prior
        )
        ranked.append(item)
    return sorted(ranked, key=lambda x: x["rank_score"], reverse=True)


def process_image(
    processor,
    image_path: Path,
    prompts: Sequence[str],
    args,
    prev_mask,
    device: str,
    *,
    pil_image: Optional[Image.Image] = None,
    text_cache: Optional[Dict[str, Dict]] = None,
    batch_state: Optional[Dict] = None,
):
    image = pil_image if pil_image is not None else Image.open(image_path).convert("RGB")
    image_array = np.asarray(image)
    with _amp_context(args, device):
        if batch_state is not None:
            state = batch_state
        else:
            state = processor.set_image(image)

        all_instances: List[Dict] = []
        for prompt in prompts:
            if text_cache is not None:
                state["backbone_out"].update(text_cache[prompt])
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = processor.model._get_dummy_prompt()
                output = processor._forward_grounding(state)
            else:
                output = processor.set_text_prompt(state=state, prompt=prompt)
            all_instances.extend(_instances_from_output(output, prompt, args))

    all_instances = _dedupe_instances(all_instances, args.dedupe_iou)
    all_instances = _rank_instances(all_instances, image_array, prev_mask, args)
    selected_limit = 1 if args.force_single_instance else args.max_instances
    selected = all_instances[:selected_limit]

    if (
        not selected
        and args.reuse_previous_on_miss
        and prev_mask is not None
        and prev_mask.any()
    ):
        area = int(prev_mask.sum())
        selected = [
            {
                "mask": prev_mask,
                "box": bbox_from_mask(prev_mask, padding=0),
                "score": 0.0,
                "rank_score": 0.0,
                "temporal_iou": 1.0,
                "hand_prior_score": 0.0,
                "glove_prior_score": 0.0,
                "prompt": "previous-mask",
                "area": area,
                "area_ratio": float(area / max(prev_mask.size, 1)),
                "below_confidence": True,
                "reused_previous": True,
            }
        ]

    if selected:
        merged = np.zeros_like(selected[0]["mask"], dtype=bool)
        for item in selected:
            merged |= item["mask"]
    else:
        h, w = image.height, image.width
        merged = np.zeros((h, w), dtype=bool)

    return image_array, merged, selected, all_instances


def process_episode(
    processor,
    episode_dir: Path,
    image_subdir: str,
    args,
    prompts: Sequence[str],
    device: str,
    text_cache: Optional[Dict[str, Dict]] = None,
    frame_counter=None,
    frame_lock=None,
    pbar: Optional[tqdm] = None,
) -> Dict:
    import torch

    image_subdir = _normalize_image_subdir(image_subdir)
    images = _list_stream_images(episode_dir, image_subdir)
    if args.limit_frames is not None:
        images = images[: args.limit_frames]
    if not images:
        return {
            "episode": episode_dir.name,
            "image_subdir": image_subdir,
            "skipped": True,
            "reason": "no_images",
        }

    mask_dir = _stream_output_dir(args.output / "masks", episode_dir.name, image_subdir)
    overlay_dir = _stream_output_dir(args.output / "overlays", episode_dir.name, image_subdir)
    stats_path = _stream_stats_path(args.output, episode_dir.name, image_subdir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    preview_indices = set(evenly_spaced_indices(len(images), args.qc_frames))
    prev_mask = None
    prev_rgb = None
    records = []
    written = 0
    skipped_existing = 0
    skipped_similar = 0
    failures = []

    prompt_text = " | ".join(prompts)
    batch_size = args.batch_size
    prefetcher = _ImagePrefetcher(images, lookahead=args.prefetch)
    saver = _AsyncSaver(max_workers=2)

    try:
        idx = 0
        while idx < len(images):
            batch_paths = []
            batch_pil = []
            batch_indices = []

            for _ in range(batch_size):
                if idx >= len(images):
                    break
                image_path = images[idx]
                mask_path = mask_dir / f"{image_path.stem}_mask.png"
                pil_img = prefetcher.get()

                if mask_path.exists() and not args.overwrite:
                    skipped_existing += 1
                    idx += 1
                    if not _increment_frame_counter(frame_counter, frame_lock):
                        if pbar is not None:
                            pbar.update(1)
                    continue

                if args.skip_threshold > 0 and prev_rgb is not None and prev_mask is not None:
                    curr_arr = np.asarray(pil_img)
                    diff = np.abs(curr_arr.astype(np.float32) - prev_rgb.astype(np.float32)).mean() / 255.0
                    if diff < args.skip_threshold:
                        mask = prev_mask
                        saver.save_mask(mask, mask_path)
                        written += 1
                        skipped_similar += 1
                        mask_bbox = bbox_from_mask(mask, padding=0)
                        if idx in preview_indices:
                            overlay = _draw_overlay(curr_arr, mask, [], prompt_text)
                            saver.save_overlay(overlay, overlay_dir / f"{idx:06d}_{image_path.stem}_overlay.jpg")
                        records.append({
                            "index": idx,
                            "image": str(image_path),
                            "image_relative": _relative_image_path(episode_dir, image_path),
                            "image_subdir": image_subdir,
                            "mask": str(mask_path),
                            "mask_area": int(mask.sum()),
                            "mask_area_ratio": round(float(mask.sum() / max(mask.size, 1)), 6),
                            "mask_bbox": [int(v) for v in mask_bbox] if mask_bbox is not None else None,
                            "selected_count": 0,
                            "candidate_count": 0,
                            "instances": [],
                            "skipped_similar": True,
                        })
                        prev_rgb = curr_arr
                        idx += 1
                        if not _increment_frame_counter(frame_counter, frame_lock):
                            if pbar is not None:
                                pbar.update(1)
                        continue

                batch_paths.append((idx, image_path, mask_path))
                batch_pil.append(pil_img)
                batch_indices.append(idx)
                idx += 1

            if not batch_pil:
                continue

            with _amp_context(args, device):
                if len(batch_pil) > 1:
                    try:
                        batch_state = processor.set_image_batch(batch_pil)
                    except (RuntimeError, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                        batch_state = None
                else:
                    batch_state = None

            for bi, (frame_idx, image_path, mask_path) in enumerate(batch_paths):
                if batch_state is not None:
                    sliced = _slice_backbone_out(batch_state["backbone_out"], bi)
                    heights = batch_state["original_heights"]
                    widths = batch_state["original_widths"]
                    single_state = {
                        "original_height": heights[bi],
                        "original_width": widths[bi],
                        "backbone_out": sliced,
                    }
                else:
                    single_state = None

                rgb, mask, selected, candidates = process_image(
                    processor, image_path, prompts, args, prev_mask, device,
                    pil_image=batch_pil[bi],
                    text_cache=text_cache,
                    batch_state=single_state,
                )
                if mask.any():
                    prev_mask = mask
                    prev_rgb = rgb
                else:
                    failures.append({
                        "index": frame_idx,
                        "image": image_path.name,
                        "image_subdir": image_subdir,
                        "reason": "no_mask",
                    })
                    prev_rgb = rgb

                saver.save_mask(mask, mask_path)
                written += 1

                mask_bbox = bbox_from_mask(mask, padding=0)
                if frame_idx in preview_indices:
                    overlay = _draw_overlay(rgb, mask, selected, prompt_text)
                    saver.save_overlay(overlay, overlay_dir / f"{frame_idx:06d}_{image_path.stem}_overlay.jpg")

                records.append(
                    {
                        "index": frame_idx,
                        "image": str(image_path),
                        "image_relative": _relative_image_path(episode_dir, image_path),
                        "image_subdir": image_subdir,
                        "mask": str(mask_path),
                        "mask_area": int(mask.sum()),
                        "mask_area_ratio": round(float(mask.sum() / max(mask.size, 1)), 6),
                        "mask_bbox": [int(v) for v in mask_bbox] if mask_bbox is not None else None,
                        "selected_count": len(selected),
                        "candidate_count": len(candidates),
                        "instances": [
                            {
                                "prompt": item["prompt"],
                                "score": round(float(item["score"]), 6),
                                "rank_score": round(float(item.get("rank_score", item["score"])), 6),
                                "temporal_iou": round(float(item.get("temporal_iou", 0.0)), 6),
                                "hand_prior_score": round(float(item.get("hand_prior_score", item.get("glove_prior_score", 0.0))), 6),
                                "glove_prior_score": round(float(item.get("glove_prior_score", 0.0)), 6),
                                "below_confidence": bool(item.get("below_confidence", False)),
                                "reused_previous": bool(item.get("reused_previous", False)),
                                "area": int(item["area"]),
                                "area_ratio": round(float(item["area_ratio"]), 6),
                                "box": [round(float(v), 3) for v in item["box"]]
                                if item.get("box") is not None
                                else None,
                            }
                            for item in selected
                        ],
                    }
                )
                if not _increment_frame_counter(frame_counter, frame_lock):
                    if pbar is not None:
                        pbar.update(1)

        saver.flush()
    finally:
        prefetcher.shutdown()
        saver.shutdown()

    result = {
        "episode": episode_dir.name,
        "image_subdir": image_subdir,
        "input_images": len(images),
        "written": written,
        "skipped_existing": skipped_existing,
        "skipped_similar": skipped_similar,
        "failures": failures,
        "records": records,
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def _distribute_stream_tasks(
    tasks: List[Tuple[Path, str]],
    num_workers: int,
    args,
) -> List[List[Tuple[Path, str]]]:
    """Distribute episode/camera-stream tasks by frame count."""
    task_frames = [
        (task, _stream_frame_count(task[0], task[1], args))
        for task in tasks
    ]
    task_frames.sort(key=lambda x: x[1], reverse=True)

    worker_loads = [[] for _ in range(num_workers)]
    worker_frame_counts = [0] * num_workers

    for task, frame_count in task_frames:
        min_idx = worker_frame_counts.index(min(worker_frame_counts))
        worker_loads[min_idx].append(task)
        worker_frame_counts[min_idx] += frame_count

    return worker_loads


def _gpu_worker(rank: int, args, task_assignments: List[Tuple[Path, str]], prompts: List[str], frame_counter, frame_lock):
    """Worker process: load model on GPU rank, process assigned episodes."""
    import torch

    torch.cuda.set_device(rank)
    args_copy = argparse.Namespace(**vars(args))
    args_copy.device = f"cuda:{rank}"

    processor, device = _load_sam3(args_copy)

    text_cache: Dict[str, Dict] = {}
    with torch.inference_mode():
        for prompt in prompts:
            text_cache[prompt] = processor.model.backbone.forward_text([prompt], device=device)

    worker_stats = []
    for episode_dir, image_subdir in task_assignments:
        stats = process_episode(
            processor, episode_dir, image_subdir, args_copy, prompts, device, text_cache,
            frame_counter=frame_counter, frame_lock=frame_lock, pbar=None
        )
        worker_stats.append(stats)

    return worker_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file. CLI arguments override values loaded from this file.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam3-code", type=Path, default=DEFAULT_SAM3_CODE)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Text prompt. Repeat to merge candidates from multiple prompts.",
    )
    parser.add_argument(
        "--image-subdir",
        dest="image_subdirs",
        action="append",
        default=None,
        help=(
            "Image directory relative to each episode. Repeat to process multiple "
            "camera streams. Default: images, l515/ego/rgb, l515/external/rgb."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="CUDA autocast dtype for SAM3 inference. float32 disables autocast.",
    )
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument(
        "--candidate-confidence-threshold",
        type=float,
        default=None,
        help="Low internal SAM3 threshold used to recall candidates before local ranking.",
    )
    parser.add_argument("--min-mask-area", type=int, default=180)
    parser.add_argument("--max-area-ratio", type=float, default=0.45)
    parser.add_argument("--max-instances", type=int, default=1)
    parser.add_argument("--dedupe-iou", type=float, default=0.85)
    parser.add_argument("--track-overlap-weight", type=float, default=0.25)
    parser.add_argument(
        "--force-single-instance",
        action="store_true",
        help="Assume exactly one hand exists in each frame and keep the top-ranked candidate.",
    )
    parser.add_argument(
        "--glove-prior-weight",
        type=float,
        default=0.0,
        help="Weight for hand geometry/appearance prior during ranking.",
    )
    parser.add_argument(
        "--prior-min-area-ratio",
        type=float,
        default=0.001,
        help="Expected lower mask area ratio used only by the hand prior scorer.",
    )
    parser.add_argument(
        "--prior-max-area-ratio",
        type=float,
        default=0.22,
        help="Expected upper mask area ratio used only by the hand prior scorer.",
    )
    parser.add_argument("--close-iterations", type=int, default=1)
    parser.add_argument(
        "--dilate-iterations",
        type=int,
        default=0,
        help="Dilate masks after SAM3 output. Use 1 for incomplete hands.",
    )
    parser.add_argument(
        "--reuse-previous-on-miss",
        action="store_true",
        help="If SAM3 returns no candidate in a frame, reuse the previous frame mask.",
    )
    parser.add_argument("--keep-largest-component", action="store_true", default=True)
    parser.add_argument("--no-keep-largest-component", dest="keep_largest_component", action="store_false")
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--qc-frames", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4, help="Backbone batch encoding size. Reduce if OOM.")
    parser.add_argument("--prefetch", type=int, default=8, help="Number of frames to prefetch from disk.")
    parser.add_argument("--skip-threshold", type=float, default=0, help="Frame similarity skip threshold (0=disabled). Try 0.02.")
    parser.add_argument("--num-gpus", type=str, default="auto", help="Number of GPUs to use (auto=all available, or specify int).")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return parser


def _config_path_from_argv(argv: Optional[Sequence[str]] = None) -> Optional[Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _parse_simple_yaml_value(value: str):
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"[]", "[ ]"}:
        return []
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if any(c in value for c in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> Dict:
    data: Dict = {}
    current_key = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_key is None:
                raise ValueError(f"list item without a key in {path}: {raw_line}")
            data.setdefault(current_key, []).append(_parse_simple_yaml_value(stripped[2:]))
            continue
        current_key = None
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line in {path}: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            current_key = key
        else:
            data[key] = _parse_simple_yaml_value(value)
    return data


def _load_config_defaults(config_path: Path, parser: argparse.ArgumentParser) -> Dict:
    path = Path(config_path)
    try:
        import yaml
    except ImportError:
        data = _load_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {config_path}")

    valid_dests = {action.dest for action in parser._actions if action.dest != "help"}
    path_fields = {"config", "input", "output", "checkpoint", "sam3_code"}
    list_fields = {"prompt", "episode", "image_subdirs"}
    normalized = {}

    for raw_key, value in data.items():
        key = str(raw_key).replace("-", "_")
        if key == "prompts":
            key = "prompt"
        if key == "image_subdir":
            key = "image_subdirs"
        if key not in valid_dests:
            allowed = ", ".join(sorted(valid_dests | {"prompts", "image_subdir"}))
            raise ValueError(
                f"unknown config key '{raw_key}' in {config_path}. Allowed keys: {allowed}"
            )
        if value is None:
            normalized[key] = None
        elif key in path_fields:
            normalized[key] = Path(value)
        elif key in list_fields:
            if key == "image_subdirs" and isinstance(value, str):
                normalized[key] = [value]
            elif not isinstance(value, list):
                raise ValueError(f"config key '{raw_key}' must be a list")
            else:
                normalized[key] = value
        else:
            normalized[key] = value

    return normalized


def main() -> None:
    argv = sys.argv[1:]
    parser = build_parser()
    config_path = _config_path_from_argv(argv)
    config_defaults = {}
    if config_path is not None:
        config_defaults = _load_config_defaults(config_path, parser)
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    if config_defaults.get("prompt") and any(a == "--prompt" for a in argv):
        args.prompt = args.prompt[len(config_defaults["prompt"]) :]
    if config_defaults.get("episode") and any(a == "--episode" for a in argv):
        args.episode = args.episode[len(config_defaults["episode"]) :]
    if config_defaults.get("image_subdirs") and any(a == "--image-subdir" for a in argv):
        args.image_subdirs = args.image_subdirs[len(config_defaults["image_subdirs"]) :]
    args.image_subdirs = _normalize_image_subdirs(args.image_subdirs)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.candidate_confidence_threshold is None:
        args.candidate_confidence_threshold = args.confidence_threshold
    if args.force_single_instance:
        args.max_instances = 1
    prompts = args.prompt or ["human hand", "bare hand", "fingers", "palm"]

    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    episodes = _choose_episodes(args)
    stream_tasks = _make_stream_tasks(episodes, args)
    if not stream_tasks:
        raise SystemExit(
            f"No images found under {args.input} for image_subdirs={args.image_subdirs}"
        )

    # Determine number of GPUs to use
    if args.num_gpus == "auto":
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    else:
        num_gpus = int(args.num_gpus)
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 1))

    # Count total frames for progress bar. Respect --limit-frames during tests.
    total_frames = sum(
        _stream_frame_count(episode_dir, image_subdir, args)
        for episode_dir, image_subdir in stream_tasks
    )

    if num_gpus > 1:
        # Multi-GPU mode
        print(
            f"使用 {num_gpus} 张 GPU 并行处理 {len(episodes)} 个 episodes、"
            f"{len(stream_tasks)} 个 image streams，共 {total_frames} 帧"
        )

        task_assignments = _distribute_stream_tasks(stream_tasks, num_gpus, args)

        mp.set_start_method("spawn", force=True)
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        frame_counter = manager.Value("i", 0)
        frame_lock = manager.Lock()

        pool = ctx.Pool(processes=num_gpus)

        async_results = []
        for rank in range(num_gpus):
            if task_assignments[rank]:
                result = pool.apply_async(
                    _gpu_worker,
                    args=(rank, args, task_assignments[rank], prompts, frame_counter, frame_lock)
                )
                async_results.append(result)

        pool.close()

        # Progress bar monitoring
        if not args.no_progress:
            with tqdm(total=total_frames, desc="总进度", unit="frame", ncols=100) as pbar:
                last_count = 0
                while any(not r.ready() for r in async_results):
                    time.sleep(0.5)
                    current = frame_counter.value
                    if current > last_count:
                        pbar.update(current - last_count)
                        last_count = current
                pbar.update(frame_counter.value - last_count)

        pool.join()
        manager.shutdown()

        # Collect results
        all_stats = []
        for result in async_results:
            worker_stats = result.get()
            all_stats.extend(worker_stats)

        device = "multi-gpu"
    else:
        # Single-GPU mode (with progress bar)
        processor, device = _load_sam3(args)

        text_cache: Dict[str, Dict] = {}
        with torch.inference_mode():
            for prompt in prompts:
                text_cache[prompt] = processor.model.backbone.forward_text(
                    [prompt], device=device
                )

        all_stats = []
        task_iterator = tqdm(stream_tasks, desc="Image streams", disable=args.no_progress, ncols=100)

        for episode_dir, image_subdir in task_iterator:
            task_iterator.set_postfix_str(f"{episode_dir.name}:{image_subdir}")

            stream_frames = _stream_frame_count(episode_dir, image_subdir, args)

            if not args.no_progress:
                pbar = tqdm(
                    total=stream_frames,
                    desc=f"  {episode_dir.name} {image_subdir}",
                    leave=False,
                    ncols=100,
                    unit="frame",
                )
            else:
                pbar = None

            stats = process_episode(
                processor,
                episode_dir,
                image_subdir,
                args,
                prompts,
                device,
                text_cache,
                pbar=pbar,
            )
            all_stats.append(stats)

            if pbar is not None:
                pbar.close()

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "config": str(args.config) if args.config else None,
        "checkpoint": str(args.checkpoint),
        "device": device,
        "image_subdirs": args.image_subdirs,
        "amp_dtype": args.amp_dtype,
        "prompts": prompts,
        "confidence_threshold": args.confidence_threshold,
        "candidate_confidence_threshold": args.candidate_confidence_threshold,
        "force_single_instance": args.force_single_instance,
        "hand_prior_weight": args.glove_prior_weight,
        "glove_prior_weight": args.glove_prior_weight,
        "track_overlap_weight": args.track_overlap_weight,
        "dilate_iterations": args.dilate_iterations,
        "reuse_previous_on_miss": args.reuse_previous_on_miss,
        "max_instances": args.max_instances,
        "episodes": [
            {
                "episode": s.get("episode"),
                "image_subdir": s.get("image_subdir"),
                "skipped": s.get("skipped", False),
                "reason": s.get("reason"),
                "written": s.get("written", 0),
                "skipped_existing": s.get("skipped_existing", 0),
                "failure_count": len(s.get("failures", [])),
            }
            for s in all_stats
        ],
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"\n完成！summary: {args.output / 'summary.json'}")
    print(f"mask root for augmentation: {args.output / 'masks'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
