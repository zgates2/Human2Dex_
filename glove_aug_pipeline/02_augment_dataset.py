#!/usr/bin/env python3
"""Use hand masks to generate an augmented copy of a DexUMI image dataset.

推荐用法：先用 `01_generate_sam3_masks.py` 生成人手 mask，再通过 YAML 运行本脚本。
输出 episode 默认会追加配置里的后缀，方便和原始 episode 放在同一训练数据根目录下而不重名。
旧的命令行参数仍然可用；如果同时使用 YAML 和命令行，命令行参数会覆盖 YAML。

YAML 小批量测试示例：
    conda activate /share/project/liyuanyuan/anaconda3/envs/sam3

    cd /home/zjc/Desktop/human2dex

  python glove_aug_pipeline/02_augment_dataset.py \
    --config glove_aug_pipeline/02_augment_dataset.yaml \
    --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw \
    --output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw_wrist_hand_aug \
    --mask-root /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw_wrist_sam3_masks/masks \
    --qc-output /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_raw_wrist_hand_aug_qc \
    --image-subdir wrist \
    --overwrite



常用参数：

    --config PATH
        YAML 配置文件。推荐把路径、增强强度、并行参数放在 YAML 中管理。
        命令行中显式写出的参数会覆盖 YAML。
    --input PATH
        原始数据集根目录。每个 episode 下面应有 images/ 和 pkl 元数据。
    --output PATH
        增强后数据集输出目录。episode 结构、图片名和 pkl 会尽量保持不变。
    --mask-root PATH
        预生成 mask 根目录，推荐使用 SAM3 输出的 `.../masks`。
        本脚本会读取：<mask-root>/<episode>/<image_stem>_mask.png。
        使用该参数时不需要 --roi。
    --roi PATH
        人工 ROI 标注 json，仅在不用 --mask-root 时需要。
    --qc-output PATH
        质检图和 stats.json 输出目录；默认是 <output>_qc。
        质检图格式是：原图 | mask overlay | 增强图。
    --variants N
        每张原图生成几个增强版本。
        N=1 时输出 episode 名默认追加 --episode-suffix。
        N>1 时输出 episode 名默认追加 --episode-suffix + 两位编号。
    --episode-suffix TEXT
        增强 episode 名后缀，默认是 _glove_aug。
        例如 demo_ep0001 -> demo_ep0001_glove_aug。
        若 variants=2，则输出 demo_ep0001_glove_aug00 和 demo_ep0001_glove_aug01。
    --limit-episodes N / --limit-frames N
        小样本测试用，只处理前 N 个 episode 或每个 episode 前 N 帧。
    --episode NAME
        只处理指定 episode。
    --overwrite
        覆盖已经存在的输出图片；不加时会跳过已有图片，方便断点续跑。
    --save-masks
        把本次增强实际使用的二值 mask 也保存到 qc-output/masks/。
    --workers N
        按 episode/variant 进行多进程 CPU 并行。默认 1，保持旧的串行行为。
        数据集较大时建议先试 8；如果磁盘压力不大，可再提高到 16。
    --io-workers N
        每个进程内部异步保存图片/mask/overlay 的线程数。默认 2；设 0 可恢复同步保存。
    --mp-start-method fork|spawn|forkserver
        多进程启动方式。Linux 远端默认 fork 更快；如遇到第三方库 fork 问题再试 spawn。
    --no-progress
        关闭 tqdm 进度条，适合把日志重定向到文件时使用。
    --min-mask-area N
        mask 面积小于 N 时认为失败，避免用空 mask 或噪声 mask 增强。
    --bbox-padding N
        仅 ROI fallback 使用，颜色分割时扩展/跟踪 bbox 的像素数。
    --feather-radius FLOAT
        mask 边缘羽化半径，让人手增强和原图自然融合。
    --glove-variation TEXT
        Hand appearance change strength: off, light, normal, strong, extreme. Default: strong.
        中文说明：日常先用 strong；变化太小用 extreme；太假用 normal。
    --glove-material TEXT
        Hand texture style: mixed, natural, skin, fabric, rubber, dots, stripe. Default: mixed.
        中文说明：真实人手优先用 natural 或 skin；mixed 保留为兼容预设。
    --background-variation TEXT
        Background change strength: off, light, normal, strong. Default: light.
        中文说明：只想增强人手、不想动桌面/背景时，用 off。

高级参数：
    下面这些数字范围一般不用管。只有预设不满足时，再用它们覆盖某一项。

    --glove-mix-range A,B
        人手肤色扰动混合强度范围。
    --glove-texture-strength-range A,B
        人手细纹理强度范围。
    --glove-noise-strength-range A,B
        人手区域噪声/纹理扰动强度范围。
    --glove-rib-strength-range A,B
        人手细线/纹理扰动强度范围。
    --glove-speckle-strength-range A,B
        人手斑点、颗粒噪声强度范围。
    --glove-sheen-strength-range A,B
        人手高光/油光强度范围。
    --glove-shade-contrast-range A,B
        人手内部随原图明暗变化的塑形强度范围。
    --bg-brightness-range A,B
        人手外背景亮暗变化范围；1.0 表示不变。
    --bg-contrast-range A,B
        人手外背景对比度变化范围；1.0 表示不变。
    --bg-hue-shift-range A,B
        人手外背景色调偏移范围，单位是 hue 圆环比例。
        例如 -0.03,0.03 表示正负约 3% 色相偏移。
    --bg-saturation-range A,B
        人手外背景饱和度变化范围；1.0 表示不变。
    --bg-mix-range A,B
        背景变换和原图的混合强度范围；设为 0,0 可关闭背景增强。
"""

import argparse
import os
import multiprocessing as mp
import pickle
import queue
import shutil
import sys
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bar is optional.
    tqdm = None

from camera_mount_augmentation import (
    apply_camera_mount_to_points,
    apply_camera_mount_to_mask,
    apply_camera_mount_to_rgb,
    camera_mount_config_from_args,
    sample_camera_mount_profile,
    summarize_camera_mount_config,
    summarize_camera_mount_profile,
    validate_camera_mount_config,
)
from wrist_skeleton_overlay import draw_wrist_skeleton_rgb

from common import (
    MaskStats,
    annotation_for_episode,
    apply_glove_augmentation,
    bbox_from_mask,
    clip_bbox,
    copy_episode_metadata,
    episode_seed,
    evenly_spaced_indices,
    list_episode_dirs,
    list_images,
    load_annotations,
    load_rgb,
    make_overlay,
    make_style,
    natural_key,
    save_rgb,
    segment_human_hand,
    summarize_style,
    write_json,
)


DEFAULT_IMAGE_SUBDIRS = ("images", "l515/ego/rgb", "l515/external/rgb")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FUSED_SYNC_FIELDS = {
    "fused_pts21_mano",
    "fusionPicoWeights",
    "fusionVisionWeights",
    "fusionMode",
    "fusionDiagnostics",
    "fused_o6_joint_radians",
    "fused_o6_command",
    "fused_wuji_command",
    "fused_o6_error",
    "fused_wuji_error",
    "fused_error",
}
UV_SYNC_FIELDS = ("wrist_uv21_rgb", "wrist_anchor_uv_rgb")
UV_VALID_FIELDS = {
    "wrist_uv21_rgb": "wrist_uv21_valid",
    "wrist_anchor_uv_rgb": "wrist_anchor_valid",
}


def load_binary_mask(path: Path, shape):
    if not path.exists():
        return None
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
    return mask.astype(bool)


def read_episode_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain dict/messages: {path}")
    return data


def atomic_write_pkl(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _first_episode_pkl(episode_dir: Path) -> Path | None:
    paths = sorted(Path(episode_dir).glob("*.pkl"), key=natural_key)
    return paths[0] if paths else None


def _load_source_messages(source_episode_dir: Path) -> list[dict[str, Any]]:
    pkl_path = _first_episode_pkl(source_episode_dir)
    if pkl_path is None:
        return []
    try:
        data = read_episode_pkl(pkl_path)
    except Exception:
        return []
    return data.get("messages", []) if isinstance(data.get("messages"), list) else []


def _as_uv(value: Any, expected_points: int | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
        return None
    if expected_points is not None and arr.shape[0] != int(expected_points):
        return None
    return arr


def _as_valid(value: Any, expected_points: int) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=bool).reshape(-1)
    except Exception:
        return np.ones(expected_points, dtype=bool)
    if arr.shape != (expected_points,):
        return np.ones(expected_points, dtype=bool)
    return arr.astype(bool, copy=False)


def _uv_valid_in_bounds(uv: np.ndarray, source_valid: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    valid = np.asarray(source_valid, dtype=bool).reshape(-1).copy()
    arr = np.asarray(uv, dtype=np.float32)
    valid &= np.isfinite(arr).all(axis=-1)
    valid &= arr[:, 0] >= 0
    valid &= arr[:, 0] < w
    valid &= arr[:, 1] >= 0
    valid &= arr[:, 1] < h
    return valid.astype(bool, copy=False)


def _transform_message_uv_fields(
    msg: dict[str, Any],
    camera_mount_config,
    camera_mount_profile,
    frame_index: int,
    image_shape: Tuple[int, int] | None,
) -> None:
    if image_shape is None:
        return
    for field in UV_SYNC_FIELDS:
        expected = 21 if field == "wrist_uv21_rgb" else None
        uv = _as_uv(msg.get(field), expected)
        if uv is None:
            continue
        valid_field = UV_VALID_FIELDS[field]
        source_valid = _as_valid(msg.get(valid_field), int(uv.shape[0]))
        transformed = apply_camera_mount_to_points(
            uv,
            camera_mount_config,
            camera_mount_profile,
            frame_index,
            image_shape,
        )
        msg[field] = transformed.astype(np.float32, copy=False)
        msg[valid_field] = _uv_valid_in_bounds(transformed, source_valid, image_shape)


def _copy_generated_prediction_fields(src_msg: dict[str, Any], dst_msg: dict[str, Any]) -> None:
    for key, value in src_msg.items():
        if key.startswith("wrist_") or key in FUSED_SYNC_FIELDS:
            dst_msg[key] = value


def _rewrite_image_reference(value: Any, source_episode: Path, output_episode: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        return value
    try:
        rel = path.relative_to(source_episode)
    except ValueError:
        return value
    return str((output_episode / rel).resolve())


def _message_image_shape(source_episode: Path, msg: dict[str, Any], image_field: str) -> Tuple[int, int] | None:
    value = msg.get(image_field)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = source_episode / path
    try:
        with Image.open(path) as image:
            width, height = image.size
        return int(height), int(width)
    except Exception:
        return None


def _sync_prediction_pkls(
    *,
    source_episode: Path,
    output_episode: Path,
    camera_mount_config,
    camera_mount_profile,
    frame_shapes: dict[str, Tuple[int, int]],
    frame_indices: dict[str, int],
    image_field: str,
    limit_frames: int | None,
) -> dict[str, Any]:
    pkl_paths = sorted(Path(source_episode).glob("*.pkl"), key=natural_key)
    if not pkl_paths:
        return {"enabled": True, "pkl_files": 0, "messages_synced": 0, "warning": "no_source_pkl"}
    synced_messages = 0
    written = 0
    missing_image_messages = 0
    path_aligned_messages = 0
    fallback_index_messages = 0
    for src_pkl in pkl_paths:
        data = read_episode_pkl(src_pkl)
        messages = data.get("messages", [])
        max_count = len(messages) if limit_frames is None else min(len(messages), int(limit_frames))
        for frame_idx, msg in enumerate(messages[:max_count]):
            if not isinstance(msg, dict):
                continue
            original_image_key = _normalize_image_key(msg.get(image_field), source_episode)
            msg[image_field] = _rewrite_image_reference(msg.get(image_field), source_episode, output_episode)
            _copy_generated_prediction_fields(msg, msg)
            if original_image_key is not None and original_image_key in frame_indices:
                transform_frame_idx = int(frame_indices[original_image_key])
                image_shape = frame_shapes.get(original_image_key)
                path_aligned_messages += 1
            elif not frame_indices:
                transform_frame_idx = frame_idx
                image_shape = _message_image_shape(source_episode, msg, image_field)
                fallback_index_messages += 1
            else:
                transform_frame_idx = frame_idx
                image_shape = None
                missing_image_messages += 1
            _transform_message_uv_fields(
                msg,
                camera_mount_config,
                camera_mount_profile,
                transform_frame_idx,
                image_shape,
            )
            synced_messages += 1
        metadata = data.setdefault("metadata", {})
        metadata["augmentation_prediction_sync"] = {
            "source_episode": str(source_episode),
            "output_episode": str(output_episode),
            "image_field": image_field,
            "fields_copied": ["wrist_*", *sorted(FUSED_SYNC_FIELDS)],
            "uv_fields_transformed": list(UV_SYNC_FIELDS),
            "limit_frames": limit_frames,
            "alignment": "rgbImage_path_first",
            "path_aligned_messages": path_aligned_messages,
            "fallback_index_messages": fallback_index_messages,
            "missing_augmented_image_messages": missing_image_messages,
        }
        atomic_write_pkl(data, output_episode / src_pkl.name)
        written += 1
    return {"enabled": True, "pkl_files": written, "messages_synced": synced_messages}


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


def _available_image_streams(episode_dir: Path, image_subdirs: Sequence[str]) -> List[Tuple[str, List[Path]]]:
    streams: List[Tuple[str, List[Path]]] = []
    for image_subdir in image_subdirs:
        normalized = _normalize_image_subdir(image_subdir)
        images = _list_stream_images(episode_dir, normalized)
        if images:
            streams.append((normalized, images))
    return streams


def _stream_output_dir(root: Path, image_subdir: str) -> Path:
    subdir = _normalize_image_subdir(image_subdir)
    if subdir == "images":
        return Path(root) / "images"
    return Path(root) / subdir


def _stream_qc_dir(root: Path, image_subdir: str) -> Path:
    subdir = _normalize_image_subdir(image_subdir)
    if subdir == "images":
        return Path(root)
    return Path(root) / subdir


def _stream_mask_dir(root: Path, image_subdir: str) -> Path:
    subdir = _normalize_image_subdir(image_subdir)
    if subdir == "images":
        return Path(root) / "masks"
    return Path(root) / "masks" / subdir


def _relative_image_path(episode_dir: Path, image_path: Path) -> str:
    try:
        return str(Path(image_path).relative_to(episode_dir)).replace("\\", "/")
    except ValueError:
        return str(image_path).replace("\\", "/")


def _normalize_image_key(value: Any, episode_dir: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(episode_dir)
        except ValueError:
            return None
    text = str(path).replace("\\", "/").lstrip("./")
    return text or None


def _build_message_by_image(
    messages: Sequence[Any],
    episode_dir: Path,
    image_field: str,
) -> dict[str, dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        key = _normalize_image_key(msg.get(image_field), episode_dir)
        if key is not None and key not in by_image:
            by_image[key] = msg
    return by_image


def _message_for_image(
    *,
    messages_by_image: dict[str, dict[str, Any]],
    messages: Sequence[Any],
    episode_dir: Path,
    image_field: str,
    image_key: str,
    fallback_index: int,
) -> tuple[dict[str, Any] | None, str]:
    msg = messages_by_image.get(image_key)
    if msg is not None:
        return msg, "path"
    if 0 <= fallback_index < len(messages) and isinstance(messages[fallback_index], dict):
        fallback = messages[fallback_index]
        fallback_key = _normalize_image_key(fallback.get(image_field), episode_dir)
        if fallback_key == image_key or fallback_key is None:
            return fallback, "index"
    return None, "missing"


def _path_is_under_subdir(path: Path, image_subdir: str) -> bool:
    target = Path(_normalize_image_subdir(image_subdir))
    try:
        Path(path).relative_to(target)
    except ValueError:
        return False
    return True


def _copy_episode_support_files(
    src_episode: Path,
    dst_episode: Path,
    target_subdirs: Sequence[str],
    overwrite: bool,
):
    dst_episode.mkdir(parents=True, exist_ok=True)
    normalized_targets = [_normalize_image_subdir(p) for p in target_subdirs]
    for src_path in Path(src_episode).rglob("*"):
        rel = src_path.relative_to(src_episode)
        dst_path = dst_episode / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        if any(_path_is_under_subdir(rel, target) for target in normalized_targets):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            continue
        if overwrite or not dst_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)


def _choose_episodes(args) -> List[Path]:
    root = Path(args.input)
    episodes = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_dir()
            and _available_image_streams(p, args.image_subdirs)
        ],
        key=natural_key,
    )
    if args.episode:
        episodes = [ep for ep in episodes if ep.name == args.episode]
        if not episodes:
            raise SystemExit(f"Episode not found: {args.episode}")
    if args.limit_episodes is not None:
        episodes = episodes[: args.limit_episodes]
    return episodes


def mask_path_for_image(mask_root: Path, episode_name: str, image_subdir: str, image_path: Path) -> Path:
    subdir = _normalize_image_subdir(image_subdir)
    if subdir == "images":
        return Path(mask_root) / episode_name / f"{image_path.stem}_mask.png"
    return Path(mask_root) / episode_name / subdir / f"{image_path.stem}_mask.png"


def parse_float_range(value: str):
    parts = [float(p.strip()) for p in value.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0], parts[0]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must be VALUE or MIN,MAX")
    lo, hi = parts
    if hi < lo:
        raise argparse.ArgumentTypeError("range max must be >= min")
    return lo, hi


def progress_enabled(args) -> bool:
    return bool(tqdm is not None and not args.no_progress)


def make_progress(iterable, args, **kwargs):
    if not progress_enabled(args):
        return iterable
    return tqdm(iterable, dynamic_ncols=True, mininterval=0.5, **kwargs)


def progress_write(args, message: str):
    if progress_enabled(args):
        tqdm.write(message, file=sys.stderr)
    else:
        print(message)


def _save_rgb_array(array: np.ndarray, path: Path, quality: int):
    save_rgb(array, path, quality=quality)


def _save_mask_array(mask: np.ndarray, path: Path):
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(path)


def _save_overlay_image(overlay: Image.Image, path: Path, quality: int):
    overlay.save(path, quality=quality)


class AsyncSaver:
    def __init__(self, max_workers: int = 2):
        self.max_workers = max(0, int(max_workers))
        self._pool = (
            ThreadPoolExecutor(max_workers=self.max_workers)
            if self.max_workers > 0
            else None
        )
        self._futures = []
        self._max_pending = max(1, self.max_workers * 8)

    def _submit(self, fn, *args):
        if self._pool is None:
            fn(*args)
            return
        self._futures.append(self._pool.submit(fn, *args))
        if len(self._futures) >= self._max_pending:
            self._drain(block=False)

    def _drain(self, block: bool):
        if not self._futures:
            return
        if block:
            done = self._futures
            self._futures = []
        else:
            done, pending = wait(
                self._futures,
                timeout=0,
                return_when=FIRST_COMPLETED,
            )
            self._futures = list(pending)
        for future in done:
            future.result()

    def save_rgb(self, array: np.ndarray, path: Path, quality: int):
        self._submit(_save_rgb_array, array, path, quality)

    def save_mask(self, mask: np.ndarray, path: Path):
        self._submit(_save_mask_array, mask, path)

    def save_overlay(self, overlay: Image.Image, path: Path, quality: int = 92):
        self._submit(_save_overlay_image, overlay, path, quality)

    def flush(self):
        self._drain(block=True)

    def shutdown(self):
        try:
            self.flush()
        finally:
            if self._pool is not None:
                self._pool.shutdown(wait=True)


def _notify_progress(progress_queue, increment: int = 1):
    if progress_queue is not None and increment > 0:
        progress_queue.put(increment)


# User-facing preset values are plain English. Chinese names are accepted only as aliases
# in PRESET_ALIASES below, so old local commands still work.
# 中文说明：
# - glove_variation 控制人手增强强度，比如肤色、明暗、细纹理扰动幅度。
# - glove_material 作为兼容旧配置的手部纹理风格预设。
# - background_variation 控制手部以外区域要不要一起变化。
# - strong/extreme 现在是明显增强档，适合主动制造更大的颜色、纹理和高光差异。
GLOVE_VARIATION_PRESETS = {
    "off": {
        "glove_mix_range": (0.0, 0.0),
        "glove_brightness_range": (1.0, 1.0),
        "glove_contrast_range": (1.0, 1.0),
        "glove_texture_strength_range": (0.0, 0.0),
        "glove_noise_strength_range": (0.0, 0.0),
        "glove_shade_contrast_range": (1.0, 1.0),
    },
    "light": {
        "glove_mix_range": (0.24, 0.40),
        "glove_brightness_range": (0.92, 1.10),
        "glove_contrast_range": (0.92, 1.12),
        "glove_texture_strength_range": (1.0, 4.0),
        "glove_noise_strength_range": (0.8, 2.5),
        "glove_shade_contrast_range": (0.86, 1.16),
    },
    "normal": {
        "glove_mix_range": (0.38, 0.60),
        "glove_brightness_range": (0.86, 1.18),
        "glove_contrast_range": (0.84, 1.28),
        "glove_texture_strength_range": (3.0, 9.0),
        "glove_noise_strength_range": (2.0, 5.0),
        "glove_shade_contrast_range": (0.74, 1.38),
    },
    "strong": {
        "glove_mix_range": (0.62, 0.88),
        "glove_brightness_range": (0.74, 1.30),
        "glove_contrast_range": (0.70, 1.48),
        "glove_texture_strength_range": (8.0, 18.0),
        "glove_noise_strength_range": (4.0, 10.0),
        "glove_shade_contrast_range": (0.58, 1.70),
    },
    "extreme": {
        "glove_mix_range": (0.82, 1.00),
        "glove_brightness_range": (0.60, 1.45),
        "glove_contrast_range": (0.55, 1.85),
        "glove_texture_strength_range": (16.0, 34.0),
        "glove_noise_strength_range": (8.0, 18.0),
        "glove_shade_contrast_range": (0.42, 2.10),
    },
}

GLOVE_MATERIAL_PRESETS = {
    "mixed": {
        "glove_rib_strength_range": (3.0, 12.0),
        "glove_speckle_strength_range": (4.0, 14.0),
        "glove_sheen_strength_range": (4.0, 18.0),
    },
    "natural": {
        "glove_rib_strength_range": (2.0, 8.0),
        "glove_speckle_strength_range": (2.0, 8.0),
        "glove_sheen_strength_range": (2.0, 12.0),
    },
    "skin": {
        "glove_rib_strength_range": (2.0, 9.0),
        "glove_speckle_strength_range": (3.0, 10.0),
        "glove_sheen_strength_range": (2.0, 13.0),
    },
    "fabric": {
        "glove_rib_strength_range": (6.0, 16.0),
        "glove_speckle_strength_range": (2.0, 8.0),
        "glove_sheen_strength_range": (0.0, 8.0),
    },
    "rubber": {
        "glove_rib_strength_range": (1.0, 6.0),
        "glove_speckle_strength_range": (2.0, 8.0),
        "glove_sheen_strength_range": (12.0, 26.0),
    },
    "dots": {
        "glove_rib_strength_range": (1.0, 8.0),
        "glove_speckle_strength_range": (10.0, 24.0),
        "glove_sheen_strength_range": (0.0, 10.0),
    },
    "stripe": {
        "glove_rib_strength_range": (12.0, 28.0),
        "glove_speckle_strength_range": (1.0, 8.0),
        "glove_sheen_strength_range": (0.0, 12.0),
    },
}

BACKGROUND_VARIATION_PRESETS = {
    "off": {
        "bg_brightness_range": (1.0, 1.0),
        "bg_contrast_range": (1.0, 1.0),
        "bg_hue_shift_range": (0.0, 0.0),
        "bg_saturation_range": (1.0, 1.0),
        "bg_mix_range": (0.0, 0.0),
    },
    "light": {
        "bg_brightness_range": (0.94, 1.06),
        "bg_contrast_range": (0.95, 1.06),
        "bg_hue_shift_range": (-0.02, 0.02),
        "bg_saturation_range": (0.95, 1.05),
        "bg_mix_range": (0.75, 1.0),
    },
    "normal": {
        "bg_brightness_range": (0.88, 1.12),
        "bg_contrast_range": (0.90, 1.16),
        "bg_hue_shift_range": (-0.04, 0.04),
        "bg_saturation_range": (0.90, 1.12),
        "bg_mix_range": (0.85, 1.0),
    },
    "strong": {
        "bg_brightness_range": (0.78, 1.22),
        "bg_contrast_range": (0.82, 1.28),
        "bg_hue_shift_range": (-0.07, 0.07),
        "bg_saturation_range": (0.82, 1.25),
        "bg_mix_range": (1.0, 1.0),
    },
}

PRESET_ALIASES = {
    "关闭": "off",
    "关": "off",
    "none": "off",
    "轻微": "light",
    "mild": "light",
    "标准": "normal",
    "medium": "normal",
    "强": "strong",
    "high": "strong",
    "极强": "extreme",
    "max": "extreme",
    "混合": "mixed",
    "mix": "mixed",
    "自然": "natural",
    "真人手": "natural",
    "人手": "natural",
    "皮肤": "skin",
    "布料": "fabric",
    "cloth": "fabric",
    "橡胶": "rubber",
    "颗粒": "dots",
    "speckle": "dots",
    "grain": "dots",
    "dot": "dots",
    "条纹": "stripe",
    "rib": "stripe",
}


def normalize_preset(value: str, preset_map: Dict[str, Dict], option_name: str) -> str:
    key = str(value).strip()
    normalized = PRESET_ALIASES.get(key.lower(), key.lower())
    if normalized not in preset_map:
        allowed = "、".join(preset_map.keys())
        raise SystemExit(f"{option_name} must be one of: {allowed}")
    return normalized


def apply_preset_defaults(args):
    args.glove_variation = normalize_preset(
        args.glove_variation,
        GLOVE_VARIATION_PRESETS,
        "--glove-variation",
    )
    args.glove_material = normalize_preset(
        args.glove_material,
        GLOVE_MATERIAL_PRESETS,
        "--glove-material",
    )
    args.background_variation = normalize_preset(
        args.background_variation,
        BACKGROUND_VARIATION_PRESETS,
        "--background-variation",
    )

    for preset in (
        GLOVE_VARIATION_PRESETS[args.glove_variation],
        GLOVE_MATERIAL_PRESETS[args.glove_material],
        BACKGROUND_VARIATION_PRESETS[args.background_variation],
    ):
        for attr, value in preset.items():
            if getattr(args, attr) is None:
                setattr(args, attr, value)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate a mirrored dataset with augmented human hand appearance."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file. CLI arguments override values loaded from this file.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input dataset root.")
    parser.add_argument("--output", type=Path, default=None, help="Output dataset root.")
    parser.add_argument(
        "--roi",
        type=Path,
        default=None,
        help="roi_annotations.json. Required only when --mask-root is not used.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Precomputed masks root, e.g. pick_sponge_sam3_masks/masks.",
    )
    parser.add_argument(
        "--qc-output",
        type=Path,
        default=None,
        help="Directory for overlay previews and stats. Default: <output>_qc.",
    )
    parser.add_argument(
        "--image-subdir",
        dest="image_subdirs",
        action="append",
        default=None,
        help=(
            "RGB image directory relative to each episode. Repeat to process multiple "
            "camera streams. Default: images, l515/ego/rgb, l515/external/rgb."
        ),
    )
    parser.add_argument(
        "--sync-predictions-from",
        type=Path,
        default=None,
        help="Original dataset root whose PKLs already contain wrist_* / fused_* fields to copy into augmented PKLs.",
    )
    parser.add_argument(
        "--sync-image-field",
        default="rgbImage",
        help="PKL message image field whose relative path should point to the augmented RGB. Default: rgbImage.",
    )
    parser.add_argument(
        "--render-wrist-skeleton",
        action="store_true",
        help="Draw wrist_uv21_rgb skeleton on the final augmented RGB after mask and camera-mount augmentation.",
    )
    parser.add_argument(
        "--skeleton-image-subdir",
        default="images",
        help="Only render skeleton on this image stream. Default: images.",
    )
    parser.add_argument("--skeleton-line-width", type=int, default=2)
    parser.add_argument("--skeleton-point-radius", type=int, default=4)
    parser.add_argument("--variants", type=int, default=1, help="Variants per source image.")
    parser.add_argument(
        "--episode-suffix",
        default="_glove_aug",
        help="Suffix appended to output episode names. Use empty string to keep old names.",
    )
    parser.add_argument("--seed", type=int, default=20260606, help="Base random seed.")
    parser.add_argument("--quality", type=int, default=95, help="JPG quality.")
    parser.add_argument(
        "--qc-frames", type=int, default=8, help="Overlay previews per episode."
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=8,
        help="Pixels to expand ROI/tracked bbox during segmentation.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=180,
        help="Minimum valid hand mask area in pixels.",
    )
    parser.add_argument(
        "--feather-radius",
        type=float,
        default=4.0,
        help="Mask edge feather radius in pixels.",
    )
    parser.add_argument(
        "--glove-variation",
        default="strong",
        help="Easy hand augmentation strength preset: off/light/normal/strong/extreme.",
    )
    parser.add_argument(
        "--glove-material",
        default="mixed",
        help="Hand texture preset: mixed/natural/skin/fabric/rubber/dots/stripe.",
    )
    parser.add_argument(
        "--background-variation",
        default="light",
        help="Easy background preset: off/light/normal/strong.",
    )
    parser.add_argument(
        "--glove-mix-range",
        type=parse_float_range,
        default=None,
        help="Hand skin-tone mix range, VALUE or MIN,MAX.",
    )
    parser.add_argument(
        "--glove-brightness-range",
        type=parse_float_range,
        default=None,
        help="Hand-only brightness range.",
    )
    parser.add_argument(
        "--glove-contrast-range",
        type=parse_float_range,
        default=None,
        help="Hand-only contrast range.",
    )
    parser.add_argument(
        "--glove-texture-strength-range",
        type=parse_float_range,
        default=None,
        help="Hand fine texture strength range.",
    )
    parser.add_argument(
        "--glove-noise-strength-range",
        type=parse_float_range,
        default=None,
        help="Hand noise/texture perturbation strength range.",
    )
    parser.add_argument(
        "--glove-rib-strength-range",
        type=parse_float_range,
        default=None,
        help="Hand fine-line texture perturbation strength range.",
    )
    parser.add_argument(
        "--glove-speckle-strength-range",
        type=parse_float_range,
        default=None,
        help="Hand speckle/noise strength range.",
    )
    parser.add_argument(
        "--glove-sheen-strength-range",
        type=parse_float_range,
        default=None,
        help="Hand highlight/sheen strength range.",
    )
    parser.add_argument(
        "--glove-shade-contrast-range",
        type=parse_float_range,
        default=None,
        help="Hand internal shading contrast range.",
    )
    parser.add_argument(
        "--bg-brightness-range",
        type=parse_float_range,
        default=None,
        help="Background brightness range outside the hand mask.",
    )
    parser.add_argument(
        "--bg-contrast-range",
        type=parse_float_range,
        default=None,
        help="Background contrast range outside the hand mask.",
    )
    parser.add_argument(
        "--bg-hue-shift-range",
        type=parse_float_range,
        default=None,
        help="Background hue shift range as a fraction of the hue wheel.",
    )
    parser.add_argument(
        "--bg-saturation-range",
        type=parse_float_range,
        default=None,
        help="Background saturation range outside the hand mask.",
    )
    parser.add_argument(
        "--bg-mix-range",
        type=parse_float_range,
        default=None,
        help="Blend strength for background augmentation. Use 0,0 to disable.",
    )
    parser.add_argument(
        "--camera-mount-aug-enabled",
        action="store_true",
        default=False,
        help="Enable episode-stable full-image wrist camera mount augmentation.",
    )
    parser.add_argument(
        "--camera-mount-aug-disabled",
        dest="camera_mount_aug_enabled",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camera-mount-aug-p",
        type=float,
        default=0.8,
        help="Probability of applying camera mount augmentation to each episode variant.",
    )
    parser.add_argument(
        "--camera-mount-rotation-deg",
        type=parse_float_range,
        default=(-10.0, 10.0),
        help="Episode-stable rotation range in degrees, VALUE or MIN,MAX.",
    )
    parser.add_argument(
        "--camera-mount-translate-ratio-x",
        type=parse_float_range,
        default=(-0.10, 0.10),
        help="Episode-stable horizontal translation as image-width ratio.",
    )
    parser.add_argument(
        "--camera-mount-translate-ratio-y",
        type=parse_float_range,
        default=(-0.10, 0.10),
        help="Episode-stable vertical translation as image-height ratio.",
    )
    parser.add_argument(
        "--camera-mount-scale",
        type=parse_float_range,
        default=(0.88, 1.15),
        help="Episode-stable zoom scale range.",
    )
    parser.add_argument(
        "--camera-mount-shear-deg",
        type=parse_float_range,
        default=(-4.0, 4.0),
        help="Episode-stable x/y shear range in degrees.",
    )
    parser.add_argument(
        "--camera-mount-frame-jitter-enabled",
        action="store_true",
        default=True,
        help="Enable small frame-level camera jitter on top of the episode profile.",
    )
    parser.add_argument(
        "--camera-mount-frame-jitter-disabled",
        dest="camera_mount_frame_jitter_enabled",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camera-mount-frame-jitter-rotation-deg",
        type=parse_float_range,
        default=(-1.0, 1.0),
        help="Small per-frame rotation jitter in degrees.",
    )
    parser.add_argument(
        "--camera-mount-frame-jitter-translate-px",
        type=parse_float_range,
        default=(-3.0, 3.0),
        help="Small per-frame translation jitter in pixels.",
    )
    parser.add_argument(
        "--camera-mount-frame-jitter-scale",
        type=parse_float_range,
        default=(0.99, 1.01),
        help="Small per-frame scale jitter.",
    )
    parser.add_argument(
        "--camera-mount-interpolation",
        choices=("nearest", "bilinear", "bicubic"),
        default="bilinear",
        help="RGB interpolation used by camera mount augmentation.",
    )
    parser.add_argument(
        "--camera-mount-padding-mode",
        choices=("edge", "reflect", "constant"),
        default="edge",
        help="Padding mode used by camera mount augmentation; edge avoids black borders.",
    )
    parser.add_argument(
        "--camera-mount-fill-value",
        type=int,
        default=0,
        help="Fill value only used when --camera-mount-padding-mode constant.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Process at most N episodes for testing.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Process at most N frames per episode for testing.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output images."
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save binary masks under qc-output/masks/<episode>/.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU worker processes over episode/variant tasks. 1 keeps old serial behavior.",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=2,
        help="Async image-save threads inside each worker. Use 0 to save synchronously.",
    )
    parser.add_argument(
        "--mp-start-method",
        choices=("fork", "spawn", "forkserver"),
        default="fork",
        help="Multiprocessing start method for --workers > 1.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars and use plain per-episode logs.",
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="Only process one episode name, e.g. demo_20260521_171105_ep0006.",
    )
    return parser


def _config_path_from_argv(argv: Optional[Sequence[str]] = None) -> Optional[Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=None)
    args, _ = parser.parse_known_args(argv)
    return args.config


def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for raw_key, value in data.items():
        key = str(raw_key).replace("-", "_")
        if isinstance(value, dict):
            if key == "camera_mount_aug":
                flattened.update(_flatten_camera_mount_config(value))
            else:
                flattened.update(_flatten_config(value))
        else:
            flattened[key] = value
    return flattened


def _flatten_camera_mount_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map readable nested YAML camera_mount_aug keys to argparse dest names."""
    key_map = {
        "enabled": "camera_mount_aug_enabled",
        "p": "camera_mount_aug_p",
        "rotation_deg": "camera_mount_rotation_deg",
        "translate_ratio_x": "camera_mount_translate_ratio_x",
        "translate_ratio_y": "camera_mount_translate_ratio_y",
        "scale": "camera_mount_scale",
        "shear_deg": "camera_mount_shear_deg",
        "interpolation": "camera_mount_interpolation",
        "padding_mode": "camera_mount_padding_mode",
        "fill_value": "camera_mount_fill_value",
    }
    frame_jitter_key_map = {
        "enabled": "camera_mount_frame_jitter_enabled",
        "rotation_deg": "camera_mount_frame_jitter_rotation_deg",
        "translate_px": "camera_mount_frame_jitter_translate_px",
        "scale": "camera_mount_frame_jitter_scale",
    }
    flattened: Dict[str, Any] = {}
    for raw_key, value in data.items():
        key = str(raw_key).replace("-", "_")
        if key == "frame_jitter":
            if not isinstance(value, dict):
                raise ValueError("camera_mount_aug.frame_jitter must be a mapping")
            for raw_child_key, child_value in value.items():
                child_key = str(raw_child_key).replace("-", "_")
                if child_key not in frame_jitter_key_map:
                    allowed = ", ".join(sorted(frame_jitter_key_map))
                    raise ValueError(
                        f"unknown camera_mount_aug.frame_jitter key '{raw_child_key}'. "
                        f"Allowed keys: {allowed}"
                    )
                flattened[frame_jitter_key_map[child_key]] = child_value
            continue
        if key not in key_map:
            allowed = ", ".join(sorted(list(key_map) + ["frame_jitter"]))
            raise ValueError(
                f"unknown camera_mount_aug key '{raw_key}'. Allowed keys: {allowed}"
            )
        flattened[key_map[key]] = value
    return flattened


def _coerce_range_config(value: Any, key: str):
    if value is None:
        return None
    if isinstance(value, str):
        return parse_float_range(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return float(value[0]), float(value[0])
        if len(value) != 2:
            raise ValueError(f"config key '{key}' must be VALUE or [MIN, MAX]")
        lo, hi = float(value[0]), float(value[1])
        if hi < lo:
            raise ValueError(f"config key '{key}' max must be >= min")
        return lo, hi
    raise ValueError(f"config key '{key}' must be VALUE or [MIN, MAX]")


def _load_config_defaults(config_path: Path, parser: argparse.ArgumentParser) -> Dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                "Reading augment_dataset YAML configs requires PyYAML in this environment."
            ) from exc
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {config_path}")

    valid_dests = {action.dest for action in parser._actions if action.dest != "help"}
    path_fields = {"config", "input", "output", "roi", "mask_root", "qc_output", "sync_predictions_from"}
    range_fields = {dest for dest in valid_dests if dest.endswith("_range")}
    range_fields.update(
        {
            "camera_mount_rotation_deg",
            "camera_mount_translate_ratio_x",
            "camera_mount_translate_ratio_y",
            "camera_mount_scale",
            "camera_mount_shear_deg",
            "camera_mount_frame_jitter_rotation_deg",
            "camera_mount_frame_jitter_translate_px",
            "camera_mount_frame_jitter_scale",
        }
    )
    list_fields = {"image_subdirs"}
    normalized = {}

    for raw_key, value in _flatten_config(data).items():
        key = str(raw_key).replace("-", "_")
        if key == "image_subdir":
            key = "image_subdirs"
        if key not in valid_dests:
            allowed = ", ".join(sorted(valid_dests | {"image_subdir"}))
            raise ValueError(
                f"unknown config key '{raw_key}' in {config_path}. Allowed keys: {allowed}"
            )
        if key in path_fields and value is not None:
            normalized[key] = Path(value)
        elif key in range_fields:
            normalized[key] = _coerce_range_config(value, key)
        elif key in list_fields:
            if isinstance(value, str):
                normalized[key] = [value]
            elif isinstance(value, (list, tuple)):
                normalized[key] = list(value)
            else:
                raise ValueError(f"config key '{raw_key}' must be a list")
        else:
            normalized[key] = value

    return normalized


def parse_args(argv: Optional[Sequence[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    config_path = _config_path_from_argv(argv)
    if config_path is not None:
        parser.set_defaults(**_load_config_defaults(config_path, parser))
    args = parser.parse_args(argv)

    missing = []
    if args.input is None:
        missing.append("--input or input in YAML")
    if args.output is None:
        missing.append("--output or output in YAML")
    if missing:
        parser.error("missing required config: " + ", ".join(missing))

    camera_mount_config = camera_mount_config_from_args(args)
    try:
        validate_camera_mount_config(camera_mount_config)
    except ValueError as exc:
        parser.error(str(exc))

    return args


def validate_episode_metadata(episode_dir: Path, image_count: int):
    pkl_files = list(episode_dir.glob("*.pkl"))
    if not pkl_files:
        return {"pkl": None, "messages": None, "matches_images": None, "copy_only": True}
    info = {
        "pkl": pkl_files[0].name,
        "messages": None,
        "matches_images": None,
        "copy_only": True,
    }
    try:
        with pkl_files[0].open("rb") as f:
            obj = pickle.load(f)
        messages = obj.get("messages") if isinstance(obj, dict) else None
        if messages is not None:
            info["messages"] = len(messages)
            info["matches_images"] = len(messages) == image_count
    except Exception as exc:
        info["validation_warning"] = repr(exc)
    return info


def variant_output_episode(
    output_root: Path,
    episode_name: str,
    variant: int,
    variants: int,
    episode_suffix: str,
):
    if variants == 1:
        return output_root / f"{episode_name}{episode_suffix}"
    return output_root / f"{episode_name}{episode_suffix}{variant:02d}"


def process_episode(
    args,
    ep: Path,
    annotation_data,
    variant: int,
    total_variants: int,
    progress_queue=None,
):
    streams = []
    for image_subdir, images in _available_image_streams(ep, args.image_subdirs):
        if args.limit_frames is not None:
            images = images[: args.limit_frames]
        if images:
            streams.append((image_subdir, images))
    total_input_images = sum(len(images) for _, images in streams)
    if not streams:
        return {"episode": ep.name, "skipped": True, "reason": "no_images"}

    bbox, exclude_bboxes = annotation_for_episode(annotation_data, ep.name)
    if bbox is None and args.mask_root is None:
        _notify_progress(progress_queue, total_input_images)
        return {"episode": ep.name, "skipped": True, "reason": "missing_roi"}

    out_ep = variant_output_episode(
        args.output,
        ep.name,
        variant,
        total_variants,
        args.episode_suffix,
    )
    _copy_episode_support_files(
        ep,
        out_ep,
        target_subdirs=args.image_subdirs,
        overwrite=args.overwrite,
    )

    qc_root = args.qc_output or Path(str(args.output) + "_qc")
    qc_ep = variant_output_episode(
        qc_root,
        ep.name,
        variant,
        total_variants,
        args.episode_suffix,
    )
    qc_ep.mkdir(parents=True, exist_ok=True)
    prediction_source_root = args.sync_predictions_from or args.input
    prediction_source_episode = Path(prediction_source_root) / ep.name
    source_messages = (
        _load_source_messages(prediction_source_episode)
        if (args.render_wrist_skeleton or args.sync_predictions_from is not None)
        else []
    )
    source_messages_by_image = _build_message_by_image(
        source_messages,
        prediction_source_episode,
        str(args.sync_image_field),
    )
    skeleton_alignment_counts = {"path": 0, "index": 0, "missing": 0}

    style = make_style(
        episode_seed(args.seed, ep.name),
        variant=variant,
        glove_mix_range=args.glove_mix_range,
        glove_brightness_range=args.glove_brightness_range,
        glove_contrast_range=args.glove_contrast_range,
        glove_texture_strength_range=args.glove_texture_strength_range,
        glove_noise_strength_range=args.glove_noise_strength_range,
        glove_rib_strength_range=args.glove_rib_strength_range,
        glove_speckle_strength_range=args.glove_speckle_strength_range,
        glove_sheen_strength_range=args.glove_sheen_strength_range,
        glove_shade_contrast_range=args.glove_shade_contrast_range,
        background_brightness_range=args.bg_brightness_range,
        background_contrast_range=args.bg_contrast_range,
        background_hue_shift_range=args.bg_hue_shift_range,
        background_saturation_range=args.bg_saturation_range,
        background_mix_range=args.bg_mix_range,
    )
    camera_mount_config = camera_mount_config_from_args(args)
    camera_mount_profiles = {}
    camera_mount_profile_objects = {}
    pkl_frame_shapes: dict[str, Tuple[int, int]] = {}
    pkl_frame_indices: dict[str, int] = {}

    frame_stats = []
    abnormal_frames = []
    stream_stats = []
    written = 0
    skipped_existing = 0

    saver = AsyncSaver(args.io_workers)
    try:
        for image_subdir, input_images in streams:
            out_image_dir = _stream_output_dir(out_ep, image_subdir)
            out_image_dir.mkdir(parents=True, exist_ok=True)
            qc_stream_dir = _stream_qc_dir(qc_ep, image_subdir)
            qc_stream_dir.mkdir(parents=True, exist_ok=True)
            mask_dir = _stream_mask_dir(qc_ep, image_subdir)
            if args.save_masks:
                mask_dir.mkdir(parents=True, exist_ok=True)

            preview_indices = set(evenly_spaced_indices(len(input_images), args.qc_frames))
            camera_mount_profile = sample_camera_mount_profile(
                camera_mount_config,
                args.seed,
                ep.name,
                variant,
                image_subdir=image_subdir,
            )
            camera_mount_profile_objects[image_subdir] = camera_mount_profile
            camera_mount_profiles[image_subdir] = summarize_camera_mount_profile(
                camera_mount_profile
            )
            prev_mask = None
            current_bbox = bbox
            stream_written = 0
            stream_skipped_existing = 0
            stream_abnormal_start = len(abnormal_frames)

            frame_iter = (
                input_images
                if progress_queue is not None
                else make_progress(
                    input_images,
                    args,
                    total=len(input_images),
                    desc=f"{ep.name} {image_subdir} v{variant + 1}/{total_variants}",
                    unit="frame",
                    leave=False,
                    position=1,
                )
            )

            for idx, image_path in enumerate(frame_iter):
                out_path = out_image_dir / image_path.name
                if out_path.exists() and not args.overwrite:
                    skipped_existing += 1
                    stream_skipped_existing += 1
                    _notify_progress(progress_queue)
                    if hasattr(frame_iter, "set_postfix"):
                        frame_iter.set_postfix(
                            written=stream_written,
                            skipped=stream_skipped_existing,
                            abnormal=len(abnormal_frames) - stream_abnormal_start,
                            refresh=False,
                        )
                    continue

                image = load_rgb(image_path)
                image_key = _relative_image_path(ep, image_path)
                if image_subdir == _normalize_image_subdir(args.skeleton_image_subdir):
                    pkl_frame_shapes[image_key] = tuple(int(v) for v in image.shape[:2])
                    pkl_frame_indices[image_key] = int(idx)
                if args.mask_root is not None:
                    mask_path = mask_path_for_image(
                        args.mask_root,
                        ep.name,
                        image_subdir,
                        image_path,
                    )
                    mask = load_binary_mask(mask_path, image.shape[:2])
                    if mask is None:
                        stats = MaskStats(
                            area=0,
                            area_ratio=0.0,
                            component_count=0,
                            threshold=0.0,
                            failed=True,
                            reason="missing_precomputed_mask",
                        )
                    else:
                        area = int(mask.sum())
                        failed = area < args.min_mask_area
                        stats = MaskStats(
                            area=area,
                            area_ratio=float(area / max(mask.size, 1)),
                            component_count=1 if area > 0 else 0,
                            threshold=0.0,
                            failed=failed,
                            reason="too_small" if failed else "",
                        )
                else:
                    mask, stats = segment_human_hand(
                        image,
                        current_bbox,
                        exclude_bboxes=exclude_bboxes,
                        prev_mask=prev_mask,
                        min_area=args.min_mask_area,
                        bbox_padding=args.bbox_padding,
                    )

                    if stats.failed and prev_mask is not None:
                        mask, stats = segment_human_hand(
                            image,
                            bbox,
                            exclude_bboxes=exclude_bboxes,
                            prev_mask=None,
                            min_area=args.min_mask_area,
                            bbox_padding=args.bbox_padding,
                        )

                if not stats.failed:
                    prev_mask = mask
                    tracked_bbox = bbox_from_mask(mask, padding=args.bbox_padding)
                    if tracked_bbox is not None:
                        h, w = mask.shape
                        current_bbox = clip_bbox(tracked_bbox, w, h)
                else:
                    abnormal_frames.append(
                        {
                            "index": idx,
                            "image": image_path.name,
                            "image_relative": _relative_image_path(ep, image_path),
                            "image_subdir": image_subdir,
                            "reason": stats.reason,
                        }
                    )
                    mask = np.zeros(image.shape[:2], dtype=bool)

                augmented = apply_glove_augmentation(
                    image,
                    mask,
                    style,
                    idx,
                    feather_radius=args.feather_radius,
                )
                augmented = apply_camera_mount_to_rgb(
                    augmented,
                    camera_mount_config,
                    camera_mount_profile,
                    idx,
                )
                if (
                    args.render_wrist_skeleton
                    and image_subdir == _normalize_image_subdir(args.skeleton_image_subdir)
                ):
                    source_msg, alignment = _message_for_image(
                        messages_by_image=source_messages_by_image,
                        messages=source_messages,
                        episode_dir=prediction_source_episode,
                        image_field=str(args.sync_image_field),
                        image_key=image_key,
                        fallback_index=idx,
                    )
                    skeleton_alignment_counts[alignment] = skeleton_alignment_counts.get(alignment, 0) + 1
                    uv21 = _as_uv(source_msg.get("wrist_uv21_rgb"), 21) if source_msg is not None else None
                    if uv21 is not None:
                        valid21 = _as_valid(source_msg.get("wrist_uv21_valid"), 21)
                        uv21_aug = apply_camera_mount_to_points(
                            uv21,
                            camera_mount_config,
                            camera_mount_profile,
                            idx,
                            image.shape[:2],
                        )
                        valid21_aug = _uv_valid_in_bounds(uv21_aug, valid21, image.shape[:2])
                        augmented = draw_wrist_skeleton_rgb(
                            augmented,
                            uv21_aug,
                            valid21_aug,
                            line_width=int(args.skeleton_line_width),
                            point_radius=int(args.skeleton_point_radius),
                        )
                saver.save_rgb(augmented, out_path, quality=args.quality)
                written += 1
                stream_written += 1

                if args.save_masks:
                    qc_mask = apply_camera_mount_to_mask(
                        mask,
                        camera_mount_config,
                        camera_mount_profile,
                        idx,
                    )
                    saver.save_mask(qc_mask, mask_dir / f"{image_path.stem}_mask.png")

                if idx in preview_indices:
                    overlay_bbox = (
                        current_bbox
                        if current_bbox is not None
                        else bbox_from_mask(mask, padding=0)
                    )
                    overlay = make_overlay(
                        image,
                        augmented,
                        mask,
                        bbox=overlay_bbox,
                        exclude_bboxes=exclude_bboxes,
                    )
                    saver.save_overlay(
                        overlay,
                        qc_stream_dir / f"{idx:06d}_{image_path.stem}_overlay.jpg",
                        quality=92,
                    )

                frame_stats.append(
                    {
                        "index": idx,
                        "image": image_path.name,
                        "image_relative": _relative_image_path(ep, image_path),
                        "image_subdir": image_subdir,
                        "mask_area": stats.area,
                        "mask_area_ratio": round(stats.area_ratio, 6),
                        "component_count": stats.component_count,
                        "threshold": round(stats.threshold, 4),
                        "failed": stats.failed,
                        "reason": stats.reason,
                    }
                )
                _notify_progress(progress_queue)
                if hasattr(frame_iter, "set_postfix"):
                    frame_iter.set_postfix(
                        written=stream_written,
                        skipped=stream_skipped_existing,
                        abnormal=len(abnormal_frames) - stream_abnormal_start,
                        refresh=False,
                    )

            stream_stats.append(
                {
                    "image_subdir": image_subdir,
                    "input_images": len(input_images),
                    "written": stream_written,
                    "skipped_existing": stream_skipped_existing,
                    "abnormal_count": len(abnormal_frames) - stream_abnormal_start,
                }
            )
    finally:
        saver.shutdown()

    metadata_frame_count = max((len(images) for _, images in streams), default=0)
    metadata_info = validate_episode_metadata(ep, metadata_frame_count)
    sync_stats = {"enabled": False}
    if args.sync_predictions_from is not None:
        pkl_profile = camera_mount_profile_objects.get(
            _normalize_image_subdir(args.skeleton_image_subdir)
        )
        if pkl_profile is None:
            pkl_profile = sample_camera_mount_profile(
                camera_mount_config,
                args.seed,
                ep.name,
                variant,
                image_subdir=_normalize_image_subdir(args.skeleton_image_subdir),
            )
        sync_stats = _sync_prediction_pkls(
            source_episode=prediction_source_episode,
            output_episode=out_ep,
            camera_mount_config=camera_mount_config,
            camera_mount_profile=pkl_profile,
            frame_shapes=pkl_frame_shapes,
            frame_indices=pkl_frame_indices,
            image_field=str(args.sync_image_field),
            limit_frames=args.limit_frames,
        )
    episode_stats = {
        "episode": ep.name,
        "output_episode": out_ep.name,
        "variant": variant,
        "image_subdirs": [image_subdir for image_subdir, _ in streams],
        "input_images": total_input_images,
        "written": written,
        "skipped_existing": skipped_existing,
        "mask_source": "precomputed" if args.mask_root is not None else "roi_color",
        "mask_root": str(args.mask_root) if args.mask_root is not None else None,
        "episode_suffix": args.episode_suffix,
        "feather_radius": args.feather_radius,
        "glove_variation": args.glove_variation,
        "glove_material": args.glove_material,
        "background_variation": args.background_variation,
        "camera_mount_aug": summarize_camera_mount_config(camera_mount_config),
        "camera_mount_profiles": camera_mount_profiles,
        "prediction_source_episode": str(prediction_source_episode),
        "render_wrist_skeleton": bool(args.render_wrist_skeleton),
        "prediction_sync": sync_stats,
        "skeleton_alignment": skeleton_alignment_counts,
        "roi_bbox": list(bbox) if bbox is not None else None,
        "exclude_bboxes": [list(b) for b in exclude_bboxes],
        "style": summarize_style(style),
        "metadata": metadata_info,
        "streams": stream_stats,
        "abnormal_frames": abnormal_frames,
        "frames": frame_stats,
    }
    write_json(qc_ep / "stats.json", episode_stats)
    return episode_stats


def task_frame_count(args, ep: Path) -> int:
    total = 0
    for _, images in _available_image_streams(ep, args.image_subdirs):
        total += min(len(images), args.limit_frames) if args.limit_frames is not None else len(images)
    return total


def _process_episode_worker(task):
    index, args, ep, annotation_data, variant, total_variants, progress_queue = task
    worker_args = argparse.Namespace(**vars(args))
    worker_args.no_progress = True
    return (
        index,
        process_episode(
            worker_args,
            ep,
            annotation_data,
            variant,
            total_variants,
            progress_queue=progress_queue,
        ),
    )


def _drain_progress_queue(progress_queue, pbar):
    if progress_queue is None or pbar is None:
        return
    pending = 0
    while True:
        try:
            pending += int(progress_queue.get_nowait())
        except queue.Empty:
            break
    if pending:
        pbar.update(pending)


def run_parallel_tasks(args, tasks, annotation_data, total_variants: int):
    max_workers = min(args.workers, len(tasks))
    total_frames = sum(task_frame_count(args, ep) for ep, _ in tasks)
    ctx = mp.get_context(args.mp_start_method)
    manager = None
    progress_queue = None
    pbar = None
    results = [None] * len(tasks)

    if progress_enabled(args):
        manager = ctx.Manager()
        progress_queue = manager.Queue()
        pbar = tqdm(
            total=total_frames,
            desc=f"frames ({max_workers} workers)",
            unit="frame",
            dynamic_ncols=True,
            mininterval=0.5,
        )

    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _process_episode_worker,
                    (
                        index,
                        args,
                        ep,
                        annotation_data,
                        variant,
                        total_variants,
                        progress_queue,
                    ),
                )
                for index, (ep, variant) in enumerate(tasks)
            }
            completed = 0
            while futures:
                done, futures = wait(
                    futures,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )
                _drain_progress_queue(progress_queue, pbar)
                for future in done:
                    index, stats = future.result()
                    results[index] = stats
                    completed += 1
                    if pbar is not None:
                        pbar.set_postfix(
                            tasks=f"{completed}/{len(tasks)}",
                            refresh=False,
                        )
                    if stats.get("skipped"):
                        progress_write(
                            args,
                            f"  skipped: {stats.get('episode')} {stats.get('reason')}",
                        )
                    else:
                        progress_write(
                            args,
                            f"  done: {stats['episode']} variant={stats['variant']} "
                            f"written={stats['written']} "
                            f"skipped_existing={stats['skipped_existing']} "
                            f"abnormal={len(stats['abnormal_frames'])}",
                        )
            _drain_progress_queue(progress_queue, pbar)
    finally:
        if pbar is not None:
            pbar.close()
        if manager is not None:
            manager.shutdown()

    return results


def main():
    args = parse_args()
    args.image_subdirs = _normalize_image_subdirs(args.image_subdirs)
    apply_preset_defaults(args)
    if args.mask_root is None and args.roi is None:
        raise SystemExit("Either --mask-root or --roi is required.")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.io_workers < 0:
        raise SystemExit("--io-workers must be >= 0")
    annotation_data = load_annotations(args.roi) if args.roi is not None else {"episodes": {}}
    episodes = _choose_episodes(args)

    args.output.mkdir(parents=True, exist_ok=True)
    qc_root = args.qc_output or Path(str(args.output) + "_qc")
    qc_root.mkdir(parents=True, exist_ok=True)

    all_stats = []
    total = len(episodes) * args.variants
    tasks = [(ep, variant) for ep in episodes for variant in range(args.variants)]
    if args.workers > 1 and len(tasks) > 1:
        progress_write(
            args,
            f"parallel: {args.workers} CPU workers, {args.io_workers} async IO workers/process",
        )
        all_stats = run_parallel_tasks(args, tasks, annotation_data, args.variants)
    else:
        task_iter = make_progress(
            tasks,
            args,
            total=total,
            desc="episodes",
            unit="episode",
            leave=True,
            position=0,
        )
        for count, (ep, variant) in enumerate(task_iter, start=1):
            if hasattr(task_iter, "set_description_str"):
                task_iter.set_description_str(f"episode {count}/{total} {ep.name}")
                task_iter.set_postfix(
                    variant=f"{variant + 1}/{args.variants}",
                    refresh=False,
                )
            else:
                print(f"[{count}/{total}] {ep.name} variant={variant}")

            stats = process_episode(args, ep, annotation_data, variant, args.variants)
            all_stats.append(stats)
            if stats.get("skipped"):
                progress_write(args, f"  skipped: {stats.get('reason')}")
            else:
                progress_write(
                    args,
                    f"  done: {ep.name} variant={variant} "
                    f"written={stats['written']} "
                    f"skipped_existing={stats['skipped_existing']} "
                    f"abnormal={len(stats['abnormal_frames'])}",
                )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "roi": str(args.roi) if args.roi is not None else None,
        "mask_root": str(args.mask_root) if args.mask_root is not None else None,
        "sync_predictions_from": str(args.sync_predictions_from) if args.sync_predictions_from is not None else None,
        "sync_image_field": str(args.sync_image_field),
        "render_wrist_skeleton": bool(args.render_wrist_skeleton),
        "skeleton_image_subdir": str(args.skeleton_image_subdir),
        "image_subdirs": args.image_subdirs,
        "variants": args.variants,
        "workers": args.workers,
        "io_workers": args.io_workers,
        "episode_suffix": args.episode_suffix,
        "glove_variation": args.glove_variation,
        "glove_material": args.glove_material,
        "background_variation": args.background_variation,
        "camera_mount_aug": summarize_camera_mount_config(
            camera_mount_config_from_args(args)
        ),
        "augmentation_ranges": {
            "feather_radius": args.feather_radius,
            "glove_mix_range": list(args.glove_mix_range),
            "glove_brightness_range": list(args.glove_brightness_range),
            "glove_contrast_range": list(args.glove_contrast_range),
            "glove_texture_strength_range": list(args.glove_texture_strength_range),
            "glove_noise_strength_range": list(args.glove_noise_strength_range),
            "glove_rib_strength_range": list(args.glove_rib_strength_range),
            "glove_speckle_strength_range": list(args.glove_speckle_strength_range),
            "glove_sheen_strength_range": list(args.glove_sheen_strength_range),
            "glove_shade_contrast_range": list(args.glove_shade_contrast_range),
            "bg_brightness_range": list(args.bg_brightness_range),
            "bg_contrast_range": list(args.bg_contrast_range),
            "bg_hue_shift_range": list(args.bg_hue_shift_range),
            "bg_saturation_range": list(args.bg_saturation_range),
            "bg_mix_range": list(args.bg_mix_range),
        },
        "episodes_requested": len(episodes),
        "episodes": [
            {
                "episode": s.get("episode"),
                "output_episode": s.get("output_episode"),
                "variant": s.get("variant"),
                "image_subdirs": s.get("image_subdirs"),
                "skipped": s.get("skipped", False),
                "reason": s.get("reason"),
                "written": s.get("written", 0),
                "skipped_existing": s.get("skipped_existing", 0),
                "abnormal_count": len(s.get("abnormal_frames", [])),
                "prediction_sync": s.get("prediction_sync"),
            }
            for s in all_stats
        ],
    }
    write_json(qc_root / "summary.json", summary)
    progress_write(args, f"summary: {qc_root / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
