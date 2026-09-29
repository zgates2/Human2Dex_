#!/usr/bin/env python3
"""将 DexUMI 采集的 PKL episode 转成 Data-Scaling-Laws 可训练的 replay-buffer zarr。


  cd /share/project/liyuanyuan/code/dexglove/

  conda activate /share/project/liyuanyuan/anaconda3/envs/sam3

  # hand_command 版本：robot0_gripper_width 读取 o6_command/handCommand。
  python /home/zjc/Desktop/human2dex/tools/convert_pkl_to_training_zarr.py \
    --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_aug/pick_sponge_aug_few_data \
    --output /share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6/pick_sponge_aug_few_data \
    --overwrite \
    --gripper-source hand_command \
    --data-scaling-laws-root /home/zjc/Desktop/human2dex \
    --workers 32 \
    --opencv-threads 1 \
    --image-compressor blosc

    # wuji_command 版本：robot0_gripper_width 读取 wuji_command 并展平。
python /home/zjc/Desktop/human2dex/tools/convert_pkl_to_training_zarr.py \
    --input /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_3_aug/pick_3_crop \
    --output /share/project/liyuanyuan/data/dexglove_data/dex_data/wuji/pick_3_aug \
    --overwrite \
    --gripper-source wuji_command \
    --data-scaling-laws-root /home/zjc/Desktop/human2dex \
    --workers 32 \
    --opencv-threads 1 \
    --image-compressor blosc


输出文件:
  <output>/dataset.zarr.zip
  <output>/count.txt
  <output>/manifest.json

zarr 数据格式:
  data/camera0_rgb                uint8   [N, 224, 224, 3]
  data/camera0_local_rgb          uint8   [N, 224, 224, 3]  # only with --local-rgb-image-field
  data/robot0_eef_pos             float32 [N, 3]  # 由 --trajectory-pose-source[:3] 决定
  data/robot0_eef_rot_axis_angle  float32 [N, 3]  # 由 --trajectory-pose-source[3:] 决定
  data/robot0_gripper_width       float32 [N, D]   # 由 --gripper-source 决定
  data/action                     float32 [N, 6+D] # 选定的 pose 6维 + robot0_gripper_width
  meta/episode_ends               int64   [num_episodes]

可选 --trajectory-pose-source:
  trajectoryPose       原始 trajectoryPose（默认）
  trajectoryPose_tcp   TCP 参考位姿
  trajectoryPose_palm  手掌参考位姿

可选 --gripper-source:
  hand_command       o6_command/handCommand -> D=6, action=12
  wuji_command       wuji_command[5,4] -> D=20, action=26
  pts21_mano         pts21_mano[21,3] -> D=63, action=69
  wrist_pts21_mano   wrist_pts21_mano[21,3] -> D=63, action=69
  wrist_o6_command   wrist_o6_command -> D=6, action=12
  wrist_wuji_command wrist_wuji_command[5,4] -> D=20, action=26
  fused_o6_command   fused_o6_command -> D=6, action=12
  fused_wuji_command fused_wuji_command[5,4] -> D=20, action=26
  fused_pts21_mano   fused_pts21_mano[21,3] -> D=63, action=69

上传文件夹
ks3 sync -u --jobs=32 /home/zjc/Desktop/human2dex/converted_data/pick_sponge_wuji ks3://baai-vision-tmp/luoshaqi/zjc_data/dex_data/

ks3://baai-vision-tmp/luoshaqi/zjc_data/dex_data/pick_sponge_wuji

ks3 cp /home/zjc/Desktop/human2dex/train_scripts/data/outputs/2026.06.23/22.23.20/checkpoints/latest.ckpt ks3://baai-vision-tmp/luoshaqi/zjc_data/dex_data/ckpt/wuji/pick_3_raw/latest.ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import socket
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OUTPUT = Path("converted_data/data_scaling_laws")
DEFAULT_DSL_ROOT = Path("/home/zjc/Desktop/Data-Scaling-Laws")
DEFAULT_RESIZE_HW = (224, 224)
EPISODE_DIR_RE = re.compile(r"^(episodes?_\d+)(?:$|[_-].*)")
GRIPPER_SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "hand_command": {
        "fields": ("o6_command", "handCommand"),
        "shape": (6,),
        "description": "o6_command/handCommand flattened to 6D",
    },
    "wuji_command": {
        "fields": ("wuji_command",),
        "shape": (5, 4),
        "description": "wuji_command[5,4] flattened to 20D",
    },
    "pts21_mano": {
        "fields": ("pts21_mano",),
        "shape": (21, 3),
        "description": "pts21_mano[21,3] flattened to 63D",
    },
    "wrist_pts21_mano": {
        "fields": ("wrist_pts21_mano",),
        "shape": (21, 3),
        "description": "wrist_pts21_mano[21,3] flattened to 63D",
    },
    "wrist_o6_command": {
        "fields": ("wrist_o6_command",),
        "shape": (6,),
        "description": "wrist_o6_command flattened to 6D",
    },
    "wrist_wuji_command": {
        "fields": ("wrist_wuji_command",),
        "shape": (5, 4),
        "description": "wrist_wuji_command[5,4] flattened to 20D",
    },
    "fused_o6_command": {
        "fields": ("fused_o6_command",),
        "shape": (6,),
        "description": "fused_o6_command flattened to 6D",
    },
    "fused_wuji_command": {
        "fields": ("fused_wuji_command",),
        "shape": (5, 4),
        "description": "fused_wuji_command[5,4] flattened to 20D",
    },
    "fused_pts21_mano": {
        "fields": ("fused_pts21_mano",),
        "shape": (21, 3),
        "description": "fused_pts21_mano[21,3] flattened to 63D",
    },
}
GRIPPER_SOURCES = tuple(GRIPPER_SOURCE_SPECS.keys())
TRAJECTORY_POSE_SOURCES = (
    "trajectoryPose",
    "trajectoryPose_tcp",
    "trajectoryPose_palm",
)
IMAGE_COMPRESSORS = ("auto", "jpegxl", "blosc", "none")
OUTPUT_FORMATS = ("zip", "dir", "both")
LOCK_FILE_NAME = ".convert_pkl_to_training_zarr.lock"
LOWDIM_KEYS = (
    "action",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
)
DEFAULT_IMAGE_BATCH_SIZE = 16


class EmptyEpisodeError(ValueError):
    """Raised when a PKL contains no frames usable for zarr conversion."""


def _acquire_output_lock(output_dir: Path, force_unlock: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_FILE_NAME
    if force_unlock and lock_path.exists():
        lock_path.unlink()
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": str(output_dir),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags, 0o644)
    except FileExistsError as exc:
        try:
            existing = lock_path.read_text(encoding="utf-8")
        except OSError:
            existing = "<cannot read lock file>"
        raise SystemExit(
            f"输出目录正在被另一个转换进程使用，或上次异常退出后残留锁文件: {lock_path}\n"
            f"lock content: {existing}\n"
            "确认没有其他 convert_pkl_to_training_zarr.py 写同一个 output 后，"
            "可删除该锁文件，或加 --force-unlock 重跑。"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return lock_path


def _release_output_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _default_workers() -> int:
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    return min(32, max(1, available_cpus))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return parsed


def _jpegxl_level(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 100:
        raise argparse.ArgumentTypeError("JPEG XL level 需要在 1..100")
    return parsed


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to unpickle arrays written by NumPy 2.x."""
    import sys

    if "numpy._core" not in sys.modules and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
    if "numpy._core.multiarray" not in sys.modules and hasattr(np.core, "multiarray"):
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
    if "numpy._core.numeric" not in sys.modules and hasattr(np.core, "numeric"):
        sys.modules["numpy._core.numeric"] = np.core.numeric


def _configure_cv2_threads(cv2: Any, threads: int) -> None:
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(threads)


def _configure_blosc_threads(numcodecs: Any, threads: int) -> None:
    blosc_module = getattr(numcodecs, "blosc", None)
    if blosc_module is not None and hasattr(blosc_module, "set_nthreads"):
        blosc_module.set_nthreads(threads)


def _import_optional_runtime(
    dsl_root: Path | None,
    image_compressor_mode: str,
    jpegxl_level: int,
    jpegxl_threads: int,
) -> tuple[Any, Any, Any, Any | None, str]:
    try:
        import cv2
        import numcodecs
        import zarr
    except ImportError as exc:
        raise SystemExit(
            "缺少转换依赖。建议用已有环境运行:\n"
            "  /home/zjc/miniconda3/envs/umi204/bin/python "
            "tools/convert_pkl_to_training_zarr.py\n"
            f"原始错误: {exc}"
        ) from exc

    if image_compressor_mode == "none":
        return cv2, numcodecs, zarr, None, "none"
    if image_compressor_mode == "blosc":
        return cv2, numcodecs, zarr, None, "blosc"

    if dsl_root is None or not dsl_root.exists():
        if image_compressor_mode == "jpegxl":
            raise SystemExit(f"找不到 Data-Scaling-Laws 路径，无法注册 JPEG XL: {dsl_root}")
        return cv2, numcodecs, zarr, None, "blosc"

    if image_compressor_mode in {"auto", "jpegxl"}:
        sys.path.insert(0, str(dsl_root))
        try:
            from diffusion_policy.codecs.imagecodecs_numcodecs import JpegXl, register_codecs

            register_codecs(verbose=False)
            image_compressor = JpegXl(level=jpegxl_level, numthreads=jpegxl_threads)
            compressor_name = f"jpegxl(level={jpegxl_level},threads={jpegxl_threads})"
            return cv2, numcodecs, zarr, image_compressor, compressor_name
        except ImportError as exc:
            if image_compressor_mode == "jpegxl":
                raise SystemExit(f"无法导入 JPEG XL codec: {exc}") from exc
            print(f"warning: JPEG XL codec 不可用，camera0_rgb 回退到 blosc: {exc}", file=sys.stderr)

    return cv2, numcodecs, zarr, None, "blosc"


def _episode_group_name(path: Path) -> str:
    parent = path.parent.name
    episode_match = EPISODE_DIR_RE.match(parent)
    if episode_match is not None:
        return episode_match.group(1)
    if "_ep" in parent:
        return parent.rsplit("_ep", 1)[0]
    return parent


def _episode_sort_key(path: Path) -> tuple[str, int, int, str, str]:
    parent = path.parent.name
    group = _episode_group_name(path)
    episode_idx = -1
    episode_match = re.search(r"_(\d+)$", group)
    if episode_match is not None:
        episode_idx = int(episode_match.group(1))
    elif "_ep" in parent:
        _, ep_text = parent.rsplit("_ep", 1)
        try:
            episode_idx = int(ep_text)
        except ValueError:
            episode_idx = -1
    variant_rank = 0 if parent == group else 1
    return group, episode_idx, variant_rank, parent, path.name


def _find_pkl_paths(input_dir: Path) -> list[Path]:
    for pattern in ("demo_*/*.pkl", "episode_*/*.pkl", "episodes_*/*.pkl"):
        paths = sorted(input_dir.glob(pattern), key=_episode_sort_key)
        if paths:
            return paths
    return sorted(input_dir.rglob("*.pkl"), key=_episode_sort_key)


def _read_pkl(path: Path) -> dict[str, Any]:
    _install_numpy_pickle_compat()
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL 缺少 dict/messages: {path}")
    return data


def _valid_array(value: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape != shape:
        return None
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        return None
    return arr.astype(dtype, copy=False)


def _read_rgb(path: Path, cv2: Any, resize_hw: tuple[int, int] | None) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    if resize_hw is not None:
        height, width = resize_hw
        img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    if value.lower() in {"none", "native", "original"}:
        return None
    normalized = value.lower().replace("x", ",")
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--resize 需要格式 H,W 或 HxW")
    height, width = (int(parts[0]), int(parts[1]))
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("--resize 的 H/W 必须为正数")
    return height, width


def _count_episode_groups(pkl_paths: list[Path]) -> OrderedDict[str, int]:
    groups = OrderedDict()
    for path in pkl_paths:
        group = _episode_group_name(path)
        groups[group] = groups.get(group, 0) + 1
    return groups


def _extract_gripper_command(msg: dict[str, Any], source: str) -> np.ndarray | None:
    spec = GRIPPER_SOURCE_SPECS.get(source)
    if spec is None:
        raise ValueError(f"unsupported gripper source: {source}")

    expected_shape = tuple(spec["shape"])
    for field in spec["fields"]:
        arr = _valid_array(msg.get(field), expected_shape, np.float32)
        if arr is not None:
            return arr.reshape(-1)
    return None


def _gripper_source_help() -> str:
    return "；".join(
        f"{name}: {spec['description']}"
        for name, spec in GRIPPER_SOURCE_SPECS.items()
    )



def _message_field_value(msg: dict[str, Any], field: str) -> Any:
    """Read a PKL message field, allowing nested paths such as l515.ego.rgbImage.

    Direct top-level fields retain their original behavior. Missing intermediate
    dictionaries return None so the frame is handled by the existing validation.
    """
    value: Any = msg
    for part in str(field).split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def _scan_episode(
    pkl_path: Path,
    resize_hw: tuple[int, int] | None,
    skip_missing_images: bool,
    trajectory_pose_source: str,
    gripper_source: str,
    image_fields: dict[str, str],
    extra_lowdim_fields: dict[str, tuple[str, int]],
    cv2: Any | None,
) -> dict[str, Any]:
    """Read only metadata/low-dimensional fields; never return decoded RGB arrays."""
    data = _read_pkl(pkl_path)
    lowdim_keys = LOWDIM_KEYS + tuple(extra_lowdim_fields)
    rows: dict[str, list[np.ndarray]] = {key: [] for key in lowdim_keys}
    image_paths: dict[str, list[Path]] = {key: [] for key in image_fields}
    stats: Counter[str] = Counter()

    for msg in data["messages"]:
        if not isinstance(msg, dict):
            stats["message_not_dict"] += 1
            continue
        pose = _valid_array(msg.get(trajectory_pose_source), (6,), np.float32)
        gripper_command = _extract_gripper_command(msg=msg, source=gripper_source)
        image_rels = {zarr_key: _message_field_value(msg, message_field) for zarr_key, message_field in image_fields.items()}
        extra_values = {
            zarr_key: _valid_array(msg.get(message_field), (dimension,), np.float32)
            for zarr_key, (message_field, dimension) in extra_lowdim_fields.items()
        }

        if pose is None:
            stats[f"invalid_{trajectory_pose_source}"] += 1
        if gripper_command is None:
            stats[f"invalid_{gripper_source}"] += 1
        for zarr_key, image_rel in image_rels.items():
            if not image_rel:
                stats[f"missing_{image_fields[zarr_key]}"] += 1
        for zarr_key, value in extra_values.items():
            if value is None:
                stats[f"invalid_{extra_lowdim_fields[zarr_key][0]}"] += 1
        if (
            pose is None
            or gripper_command is None
            or not all(image_rels.values())
            or not all(value is not None for value in extra_values.values())
        ):
            continue

        frame_paths = {zarr_key: pkl_path.parent / str(image_rel) for zarr_key, image_rel in image_rels.items()}
        if skip_missing_images:
            if cv2 is None:
                raise RuntimeError("skip_missing_images validation requires cv2")
            try:
                for image_path in frame_paths.values():
                    _read_rgb(image_path, cv2=cv2, resize_hw=resize_hw)
            except FileNotFoundError:
                stats["missing_image_file"] += 1
                continue

        gripper_command = gripper_command.astype(np.float32, copy=False)
        action = np.concatenate([pose, gripper_command], axis=0)
        rows["action"].append(action)
        rows["robot0_eef_pos"].append(pose[:3])
        rows["robot0_eef_rot_axis_angle"].append(pose[3:])
        rows["robot0_gripper_width"].append(gripper_command)
        for zarr_key, value in extra_values.items():
            assert value is not None
            rows[zarr_key].append(value)
        for zarr_key, image_path in frame_paths.items():
            image_paths[zarr_key].append(image_path)
        stats["valid_frames"] += 1

    if not rows["action"]:
        raise EmptyEpisodeError(
            "没有有效帧: "
            f"{pkl_path}; messages={len(data['messages'])}; "
            f"trajectory_pose_source={trajectory_pose_source}; "
            f"gripper_source={gripper_source}; stats={dict(stats)}"
        )

    episode: dict[str, Any] = {
        key: np.stack(values, axis=0).astype(np.float32, copy=False)
        for key, values in rows.items()
    }
    episode["image_paths"] = {key: tuple(paths) for key, paths in image_paths.items()}

    expected_action_dim = 6 + episode["robot0_gripper_width"].shape[1]
    if episode["action"].shape[1] != expected_action_dim:
        raise ValueError(
            f"action dim mismatch: got {episode['action'].shape[1]}, "
            f"expected {expected_action_dim}"
        )
    return episode


def _scan_episode_worker(
    task: tuple[int, Path, tuple[int, int] | None, bool, str, str, dict[str, str], dict[str, tuple[str, int]], int, bool],
) -> tuple[int, Path, dict[str, Any] | None, float, str | None]:
    (
        idx,
        pkl_path,
        resize_hw,
        skip_missing_images,
        trajectory_pose_source,
        gripper_source,
        image_fields,
        extra_lowdim_fields,
        opencv_threads,
        skip_empty_episodes,
    ) = task
    cv2 = None
    if skip_missing_images:
        import cv2 as cv2_module

        cv2 = cv2_module
        _configure_cv2_threads(cv2, opencv_threads)

    start = time.perf_counter()
    try:
        episode = _scan_episode(
            pkl_path=pkl_path,
            resize_hw=resize_hw,
            skip_missing_images=skip_missing_images,
            trajectory_pose_source=trajectory_pose_source,
            gripper_source=gripper_source,
            image_fields=image_fields,
            extra_lowdim_fields=extra_lowdim_fields,
            cv2=cv2,
        )
    except EmptyEpisodeError as exc:
        if skip_empty_episodes:
            return idx, pkl_path, None, time.perf_counter() - start, str(exc)
        raise
    return idx, pkl_path, episode, time.perf_counter() - start, None


def _bounded_map_unordered(
    executor: Any,
    fn: Any,
    tasks: Any,
    max_inflight_tasks: int,
) -> Any:
    """Yield completed task results while keeping submitted work bounded."""
    task_iter = iter(tasks)
    pending: set[Any] = set()

    def submit_next() -> bool:
        try:
            task = next(task_iter)
        except StopIteration:
            return False
        pending.add(executor.submit(fn, task))
        return True

    for _ in range(max_inflight_tasks):
        if not submit_next():
            break

    try:
        while pending:
            completed, still_pending = wait(pending, return_when=FIRST_COMPLETED)
            pending = set(still_pending)
            for future in completed:
                yield future.result()
            for _ in completed:
                submit_next()
    except BaseException:
        for future in pending:
            future.cancel()
        raise


def _iter_scanned_episodes(
    pkl_paths: list[Path],
    resize_hw: tuple[int, int] | None,
    skip_missing_images: bool,
    trajectory_pose_source: str,
    gripper_source: str,
    image_fields: dict[str, str],
    extra_lowdim_fields: dict[str, tuple[str, int]],
    workers: int,
    opencv_threads: int,
    skip_empty_episodes: bool,
    max_inflight_tasks: int,
) -> Any:
    tasks = (
        (
            idx,
            pkl_path,
            resize_hw,
            skip_missing_images,
            trajectory_pose_source,
            gripper_source,
            image_fields,
            extra_lowdim_fields,
            opencv_threads,
            skip_empty_episodes,
        )
        for idx, pkl_path in enumerate(pkl_paths, start=1)
    )
    if workers <= 1:
        for task in tasks:
            yield _scan_episode_worker(task)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from _bounded_map_unordered(
            executor=executor,
            fn=_scan_episode_worker,
            tasks=tasks,
            max_inflight_tasks=max_inflight_tasks,
        )


def _iter_image_batches(episodes: list[dict[str, Any]], batch_size: int, image_keys: tuple[str, ...]) -> Any:
    global_idx = 0
    batch_start = 0
    batch_paths: dict[str, list[Path]] = {key: [] for key in image_keys}
    for episode in episodes:
        frame_count = len(episode["image_paths"][image_keys[0]])
        for frame_index in range(frame_count):
            if not batch_paths[image_keys[0]]:
                batch_start = global_idx
            for key in image_keys:
                batch_paths[key].append(episode["image_paths"][key][frame_index])
            global_idx += 1
            if len(batch_paths[image_keys[0]]) >= batch_size:
                yield batch_start, {key: tuple(paths) for key, paths in batch_paths.items()}
                batch_paths = {key: [] for key in image_keys}
    if batch_paths[image_keys[0]]:
        yield batch_start, {key: tuple(paths) for key, paths in batch_paths.items()}


def _write_image_batch(
    task: tuple[int, dict[str, tuple[Path, ...]]],
    *,
    camera_arrays: dict[str, Any],
    cv2: Any,
    resize_hw: tuple[int, int] | None,
    image_hw: tuple[int, int],
) -> tuple[int, int, float, float]:
    start_idx, image_paths = task
    decode_start = time.perf_counter()
    height, width = image_hw
    image_count = len(next(iter(image_paths.values())))
    decoded: dict[str, np.ndarray] = {}
    for key, paths in image_paths.items():
        images = np.empty((image_count, height, width, 3), dtype=np.uint8)
        for local_idx, rgb_path in enumerate(paths):
            rgb = _read_rgb(rgb_path, cv2=cv2, resize_hw=resize_hw)
            if rgb.shape != (height, width, 3):
                raise ValueError(
                    f"图片尺寸不一致: {rgb_path}; got={rgb.shape}, "
                    f"expected={(height, width, 3)}"
                )
            images[local_idx] = rgb
        decoded[key] = images
    decode_seconds = time.perf_counter() - decode_start

    write_start = time.perf_counter()
    stop_idx = start_idx + image_count
    for key, images in decoded.items():
        camera_arrays[key][start_idx:stop_idx] = images
    write_seconds = time.perf_counter() - write_start
    return start_idx, image_count, decode_seconds, write_seconds



def _resolve_image_hw(
    episodes: list[dict[str, Any]],
    cv2: Any,
    resize_hw: tuple[int, int] | None,
) -> tuple[int, int]:
    if resize_hw is not None:
        return resize_hw
    first_rgb_path = episodes[0]["image_paths"]["camera0_rgb"][0]
    first_rgb = _read_rgb(first_rgb_path, cv2=cv2, resize_hw=None)
    return int(first_rgb.shape[0]), int(first_rgb.shape[1])


def _init_preallocated_replay_buffer(
    first_episode: dict[str, Any],
    episode_ends: np.ndarray,
    total_frames: int,
    image_hw: tuple[int, int],
    zarr_dir: Path,
    zarr: Any,
    numcodecs: Any,
    image_compressor: Any | None,
    image_compressor_name: str,
    image_keys: tuple[str, ...],
    lowdim_keys: tuple[str, ...],
) -> Any:
    store = zarr.DirectoryStore(str(zarr_dir))
    root = zarr.group(store=store, overwrite=True)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    lowdim_compressor = numcodecs.Blosc(
        cname="zstd",
        clevel=5,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )
    if image_compressor is None and image_compressor_name != "none":
        image_compressor = lowdim_compressor

    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends,
        shape=episode_ends.shape,
        chunks=(1024,),
        dtype=np.int64,
        compressor=None,
        overwrite=True,
    )

    lowdim_chunks = min(max(len(first_episode["action"]), 1), 1024)
    for key in lowdim_keys:
        arr = first_episode[key]
        data_group.create_dataset(
            key,
            shape=(total_frames,) + arr.shape[1:],
            chunks=(lowdim_chunks,) + arr.shape[1:],
            dtype=arr.dtype,
            compressor=lowdim_compressor,
            overwrite=True,
        )

    height, width = image_hw
    for key in image_keys:
        data_group.create_dataset(
            key,
            shape=(total_frames, height, width, 3),
            chunks=(1, height, width, 3),
            dtype=np.uint8,
            compressor=image_compressor,
            overwrite=True,
        )
    return root


def _write_preallocated_lowdim(
    root: Any,
    episodes: list[dict[str, Any]],
    lowdim_keys: tuple[str, ...],
) -> None:
    data_group = root["data"]
    offset = 0
    for episode in episodes:
        episode_len = len(episode["action"])
        stop = offset + episode_len
        for key in lowdim_keys:
            value = episode[key]
            arr = data_group[key]
            if arr.shape[1:] != value.shape[1:]:
                raise ValueError(
                    f"{key} shape mismatch: zarr {arr.shape[1:]} "
                    f"vs episode {value.shape[1:]}"
                )
            arr[offset:stop] = value
        offset = stop

    expected_total = int(root["meta"]["episode_ends"][-1])
    if offset != expected_total:
        raise RuntimeError(
            f"preallocated write length mismatch: wrote={offset}, expected={expected_total}"
        )



def _atomic_write_text(path: Path, text_value: str) -> None:
    temp_path = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temp_path.open("x", encoding="utf-8") as f:
            f.write(text_value)
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _write_count_txt(pkl_paths: list[Path], output_dir: Path) -> None:
    groups = _count_episode_groups(pkl_paths)
    text_value = "".join(f"{count}\n" for count in groups.values())
    _atomic_write_text(output_dir / "count.txt", text_value)


def _write_skipped_episodes(
    skipped_empty: list[tuple[Path, str]],
    output_dir: Path,
) -> Path | None:
    skipped_path = output_dir / "skipped_empty_episodes.txt"
    if not skipped_empty:
        try:
            skipped_path.unlink()
        except FileNotFoundError:
            pass
        return None

    text_value = "".join(
        f"{path}\t{reason}\n" for path, reason in skipped_empty
    )
    _atomic_write_text(skipped_path, text_value)
    return skipped_path


def _write_manifest(
    pkl_paths: list[Path],
    episode_lengths: list[int],
    selected_episode_count: int,
    skipped_episode_count: int,
    output_dir: Path,
    dataset_path: Path,
    zarr_dir: Path,
    zarr_dir_retained: bool,
    zip_path: Path,
    resize_hw: tuple[int, int] | None,
    image_hw: tuple[int, int],
    trajectory_pose_source: str,
    gripper_source: str,
    gripper_dim: int,
    action_dim: int,
    output_format: str,
    workers: int,
    requested_workers: int,
    scan_workers: int,
    image_workers: int,
    image_batch_size: int,
    requested_max_inflight_tasks: int | None,
    scan_max_inflight_tasks: int,
    image_max_inflight_tasks: int,
    blosc_threads: int,
    timings: dict[str, float],
    opencv_threads: int,
    image_compressor_name: str,
    image_fields: dict[str, str],
    extra_lowdim_fields: dict[str, tuple[str, int]],
) -> None:
    gripper_spec = GRIPPER_SOURCE_SPECS[gripper_source]
    manifest = {
        "format": f"DexUMI PKL to Data-Scaling-Laws dataset.zarr ({output_format})",
        "zarr": str(dataset_path),
        "zarr_dir": str(zarr_dir) if zarr_dir_retained else None,
        "zarr_dir_retained": zarr_dir_retained,
        "zarr_zip": str(zip_path) if output_format in {"zip", "both"} else None,
        "source_episode_count": selected_episode_count,
        "selected_episode_count": selected_episode_count,
        "converted_episode_count": len(pkl_paths),
        "skipped_episode_count": skipped_episode_count,
        "total_frames": int(sum(episode_lengths)),
        "resize_hw": resize_hw,
        "image_hw": image_hw,
        "trajectory_pose_source": trajectory_pose_source,
        "gripper_source": gripper_source,
        "gripper_dim": gripper_dim,
        "raw_action_dim": action_dim,
        "workers": workers,
        "requested_workers": requested_workers,
        "pipeline": "preallocated_process_scan_threaded_image_write",
        "scan_workers": scan_workers,
        "image_workers": image_workers,
        "image_batch_size": image_batch_size,
        "max_inflight_tasks": scan_max_inflight_tasks,
        "requested_max_inflight_tasks": requested_max_inflight_tasks,
        "scan_max_inflight_tasks": scan_max_inflight_tasks,
        "image_max_inflight_tasks": image_max_inflight_tasks,
        "blosc_threads": blosc_threads,
        "timings_seconds": {
            key: float(value) for key, value in timings.items()
        },
        "opencv_threads": opencv_threads,
        "image_compressor": image_compressor_name,
        "image_fields": dict(image_fields),
        "extra_lowdim_fields": {
            key: {"message_field": field, "dimension": dimension}
            for key, (field, dimension) in extra_lowdim_fields.items()
        },
        "gripper_source_fields": list(gripper_spec["fields"]),
        "gripper_source_shape": list(gripper_spec["shape"]),
        "gripper_source_description": gripper_spec["description"],
        "fields": {
            "camera0_rgb": f"RGB image loaded from message {image_fields['camera0_rgb']} path, uint8 HWC",
            "robot0_eef_pos": f"{trajectory_pose_source}[:3], float32 shape (3,)",
            "robot0_eef_rot_axis_angle": (
                f"{trajectory_pose_source}[3:], float32 shape (3,)"
            ),
            "robot0_gripper_width": (
                f"{gripper_source} copied as float32 and flattened from "
                f"{tuple(gripper_spec['shape'])} to shape ({gripper_dim},). "
                "Despite the key name this may be a multi-DoF hand command or MANO points."
            ),
            "action": (
                f"concat {trajectory_pose_source} and robot0_gripper_width, "
                f"float32 shape ({action_dim},)"
            ),
        },
        "episodes": [
            {"pkl": str(path), "frames": int(length)}
            for path, length in zip(pkl_paths, episode_lengths)
        ],
    }
    for key, message_field in image_fields.items():
        if key != "camera0_rgb":
            manifest["fields"][key] = f"RGB image loaded from message {message_field} path, uint8 HWC"
    for key, (message_field, dimension) in extra_lowdim_fields.items():
        manifest["fields"][key] = (
            f"PKL message {message_field}, float32 shape ({dimension},)"
        )
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(output_dir / "manifest.json", manifest_text)

def _zip_zarr_dir(zarr_dir: Path, zip_path: Path, zarr: Any) -> None:
    src_store = zarr.DirectoryStore(str(zarr_dir))
    with zarr.ZipStore(str(zip_path), mode="w") as dst_store:
        zarr.copy_store(src_store, dst_store, if_exists="replace")


def _resolve_output_paths(output: Path) -> tuple[Path, Path, Path]:
    if output.name.endswith(".zarr.zip"):
        output_dir = output.parent
        zip_path = output
        zarr_dir = output.with_suffix("")
    else:
        output_dir = output
        zip_path = output_dir / "dataset.zarr.zip"
        zarr_dir = output_dir / "dataset.zarr"
    return output_dir, zarr_dir, zip_path


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _publish_directory(staging_dir: Path, final_dir: Path) -> None:
    backup_dir: Path | None = None
    if final_dir.exists():
        backup_dir = final_dir.with_name(
            f".{final_dir.name}.backup.{os.getpid()}.{time.time_ns()}"
        )
        os.replace(final_dir, backup_dir)
    try:
        os.replace(staging_dir, final_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not final_dir.exists():
            os.replace(backup_dir, final_dir)
        raise
    if backup_dir is not None:
        _remove_path(backup_dir)


def _validate_dataset_path(
    dataset_path: Path,
    zarr: Any,
    expected_episode_ends: np.ndarray,
    expected_image_hw: tuple[int, int],
    image_keys: tuple[str, ...],
    lowdim_keys: tuple[str, ...],
) -> None:
    store = None
    try:
        if dataset_path.is_file():
            store = zarr.ZipStore(str(dataset_path), mode="r")
        else:
            store = zarr.DirectoryStore(str(dataset_path))
        root = zarr.group(store=store)
        actual_episode_ends = np.asarray(root["meta"]["episode_ends"][:])
        if not np.array_equal(actual_episode_ends, expected_episode_ends):
            raise RuntimeError(
                "staging dataset episode_ends mismatch: "
                f"got={actual_episode_ends}, expected={expected_episode_ends}"
            )

        expected_total = int(expected_episode_ends[-1])
        required_keys = set(lowdim_keys) | set(image_keys)
        actual_keys = set(root["data"].keys())
        if actual_keys != required_keys:
            raise RuntimeError(
                f"staging dataset keys mismatch: got={actual_keys}, "
                f"expected={required_keys}"
            )
        for key in required_keys:
            if int(root["data"][key].shape[0]) != expected_total:
                raise RuntimeError(
                    f"staging dataset length mismatch for {key}: "
                    f"{root['data'][key].shape[0]} vs {expected_total}"
                )

        height, width = expected_image_hw
        expected_chunks = (1, height, width, 3)
        for key in image_keys:
            camera_array = root["data"][key]
            if camera_array.chunks != expected_chunks:
                raise RuntimeError(
                    f"{key} chunk invariant violated: {camera_array.chunks} "
                    f"vs {expected_chunks}"
                )
            _ = camera_array[0]
            _ = camera_array[-1]
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()


def _print_root_summary(output_dir: Path, dataset_path: Path, root: Any) -> None:
    print(f"输出目录: {output_dir}")
    print(f"训练 zarr: {dataset_path}")
    print(f"episodes: {len(root['meta']['episode_ends'])}")
    print(f"total_frames: {int(root['meta']['episode_ends'][-1])}")
    print("data fields:")
    for key in sorted(root["data"].keys()):
        arr = root["data"][key]
        print(f"  {key}: shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks}")


def _print_summary(output_dir: Path, dataset_path: Path, zarr: Any) -> None:
    if dataset_path.name.endswith(".zip"):
        with zarr.ZipStore(str(dataset_path), mode="r") as store:
            root = zarr.group(store=store)
            _print_root_summary(output_dir=output_dir, dataset_path=dataset_path, root=root)
    else:
        store = zarr.DirectoryStore(str(dataset_path))
        root = zarr.group(store=store)
        _print_root_summary(output_dir=output_dir, dataset_path=dataset_path, root=root)
    count_lines = (output_dir / "count.txt").read_text(encoding="utf-8").strip().splitlines()
    print(f"count.txt: {count_lines}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data"), help="DexUMI 采集 data 目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出训练数据目录，或 dataset.zarr.zip 路径")
    parser.add_argument("--resize", type=_parse_resize, default=DEFAULT_RESIZE_HW, help="图片 resize，格式 H,W / HxW；none 表示原图")
    parser.add_argument("--data-scaling-laws-root", type=Path, default=DEFAULT_DSL_ROOT, help="Data-Scaling-Laws 仓库路径，用于注册 JPEG XL 图像压缩")
    parser.add_argument("--rgb-image-field", default="rgbImage", help="写入 data/camera0_rgb 的 PKL 图片字段；支持点号嵌套路径，如 rgbImage 或 l515.ego.rgbImage")
    parser.add_argument("--local-rgb-image-field", default=None, help="可选：写入 data/camera0_local_rgb 的 PKL 图片字段，例如 localCanonicalImage")
    parser.add_argument(
        "--object-pocket-obs-field",
        default=None,
        help="可选：写入 data/objectPocketObs 的 PKL message 字段",
    )
    parser.add_argument(
        "--object-pocket-obs-dim",
        type=_positive_int,
        default=10,
        help="objectPocketObs 维数，默认两个物体各 5 维，共 10 维",
    )
    parser.add_argument(
        "--object-absolute-obs-field",
        default=None,
        help="optional PKL field written to data/objectAbsoluteObs",
    )
    parser.add_argument(
        "--object-absolute-obs-dim",
        type=_positive_int,
        default=10,
        help="objectAbsoluteObs dimension, default 10",
    )
    parser.add_argument(
        "--trajectory-pose-source",
        choices=TRAJECTORY_POSE_SOURCES,
        default="trajectoryPose",
        help=(
            "写入 robot0_eef_pos、robot0_eef_rot_axis_angle 和 action 前 6 维的 "
            "PKL message 字段，默认 trajectoryPose"
        ),
    )
    parser.add_argument(
        "--gripper-source",
        choices=GRIPPER_SOURCES,
        default="hand_command",
        help="写入 robot0_gripper_width/action 的来源；" + _gripper_source_help(),
    )
    parser.add_argument("--limit", type=_positive_int, default=None, help="只转换前 N 个 episode，调试用")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出目录")
    parser.add_argument("--keep-zarr-dir", action="store_true", help="保留中间 dataset.zarr 目录")
    parser.add_argument("--output-format", choices=OUTPUT_FORMATS, default="zip", help="输出 zip、目录 zarr，或两者都保留")
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=_default_workers(),
        help="PKL 扫描进程数和图像解码/压缩/写入线程数；默认最多 32",
    )
    parser.add_argument(
        "--opencv-threads",
        type=_nonnegative_int,
        default=1,
        help="每个 OpenCV 操作的内部线程数，默认 1，避免与外层并发超卖",
    )
    parser.add_argument(
        "--blosc-threads",
        type=_positive_int,
        default=1,
        help="每次 Blosc 压缩的内部线程数，默认 1，外层线程池负责并行图片",
    )
    parser.add_argument(
        "--image-batch-size",
        type=_positive_int,
        default=DEFAULT_IMAGE_BATCH_SIZE,
        help="每个图像任务连续处理的帧数，默认 16",
    )
    parser.add_argument(
        "--max-inflight-tasks",
        type=_positive_int,
        default=None,
        help="最多同时提交的扫描/图像任务数；默认 2*workers，限制内存峰值",
    )
    parser.add_argument(
        "--image-compressor",
        choices=IMAGE_COMPRESSORS,
        default="auto",
        help="camera0_rgb 压缩器；auto 优先 JPEG XL，不可用则 blosc；none 最快但体积最大",
    )
    parser.add_argument("--jpegxl-level", type=_jpegxl_level, default=99, help="JPEG XL 压缩 level，默认保持原脚本的 99")
    parser.add_argument("--jpegxl-threads", type=_positive_int, default=1, help="JPEG XL 单张图片压缩线程数")
    parser.add_argument("--dry-run", action="store_true", help="只扫描并打印 episode 分组，不写文件")
    parser.add_argument("--skip-missing-images", action="store_true", help="图片缺失或损坏时跳过该帧")
    parser.add_argument(
        "--skip-empty-episodes",
        action="store_true",
        help="跳过没有任何有效帧的 PKL，并在日志中打印原因；默认遇到这种数据会报错停止。",
    )
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="删除 output 目录中的残留转换锁文件。只在确认没有其他转换进程写同一 output 时使用。",
    )
    args = parser.parse_args()
    image_fields: dict[str, str] = {"camera0_rgb": str(args.rgb_image_field)}
    if args.local_rgb_image_field:
        image_fields["camera0_local_rgb"] = str(args.local_rgb_image_field)
    image_keys = tuple(image_fields)
    extra_lowdim_fields: dict[str, tuple[str, int]] = {}
    if args.object_pocket_obs_field:
        extra_lowdim_fields["objectPocketObs"] = (
            str(args.object_pocket_obs_field),
            int(args.object_pocket_obs_dim),
        )
    if args.object_absolute_obs_field:
        if "objectAbsoluteObs" in extra_lowdim_fields:
            raise ValueError("objectAbsoluteObs field configured more than once")
        extra_lowdim_fields["objectAbsoluteObs"] = (
            str(args.object_absolute_obs_field),
            int(args.object_absolute_obs_dim),
        )
    lowdim_keys = LOWDIM_KEYS + tuple(extra_lowdim_fields)

    pkl_paths = _find_pkl_paths(args.input)
    if args.limit is not None:
        pkl_paths = pkl_paths[: args.limit]
    if not pkl_paths:
        raise SystemExit(f"未找到 pkl: {args.input}")

    group_counts = _count_episode_groups(pkl_paths)
    print(f"found pkl episodes: {len(pkl_paths)}")
    print("count.txt groups:", dict(group_counts))
    if args.dry_run:
        return

    cv2, numcodecs, zarr, image_compressor, image_compressor_name = _import_optional_runtime(
        dsl_root=args.data_scaling_laws_root,
        image_compressor_mode=args.image_compressor,
        jpegxl_level=args.jpegxl_level,
        jpegxl_threads=args.jpegxl_threads,
    )
    _configure_cv2_threads(cv2, args.opencv_threads)
    _configure_blosc_threads(numcodecs, args.blosc_threads)

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    resolved_max_inflight_tasks = (
        args.max_inflight_tasks or max(1, args.workers * 2)
    )
    scan_inflight_tasks = min(
        resolved_max_inflight_tasks,
        len(pkl_paths),
    )
    scan_worker_count = min(
        args.workers,
        len(pkl_paths),
        scan_inflight_tasks,
    )
    print(
        "convert config: "
        f"requested_workers={args.workers}, scan_processes={scan_worker_count}, "
        f"opencv_threads={args.opencv_threads}, blosc_threads={args.blosc_threads}, "
        f"image_batch_size={args.image_batch_size}, "
        f"max_inflight_tasks={resolved_max_inflight_tasks}, "
        f"trajectory_pose_source={args.trajectory_pose_source}, "
        f"image_fields={image_fields}, "
        f"image_compressor={image_compressor_name}, output_format={args.output_format}",
        flush=True,
    )
    if args.skip_missing_images:
        print(
            "warning: --skip-missing-images 会在扫描阶段预解码图片以保证 offset 正确，"
            "随后写入阶段会再次解码；容错模式会比正常模式慢。",
            file=sys.stderr,
            flush=True,
        )
    if args.workers > available_cpus:
        print(
            f"warning: workers={args.workers} 超过可用 CPU={available_cpus}。",
            file=sys.stderr,
            flush=True,
        )
    if args.workers > 1 and args.opencv_threads > 1:
        print(
            "warning: workers 和 opencv_threads 同时大于 1，可能造成嵌套线程超卖。",
            file=sys.stderr,
            flush=True,
        )
    codec_threads = 1
    if image_compressor_name.startswith("jpegxl"):
        codec_threads = args.jpegxl_threads
    elif image_compressor_name == "blosc":
        codec_threads = args.blosc_threads
    if args.workers * codec_threads > available_cpus:
        print(
            "warning: 外层 workers * 图像 codec 内部线程数超过可用 CPU，"
            "可能因线程超卖而变慢。",
            file=sys.stderr,
            flush=True,
        )

    output_dir, zarr_dir, zip_path = _resolve_output_paths(args.output)
    lock_path = _acquire_output_lock(output_dir=output_dir, force_unlock=args.force_unlock)
    staging_token = f"{os.getpid()}.{time.time_ns()}"
    staging_zarr_dir = zarr_dir.with_name(
        f".{zarr_dir.name}.partial.{staging_token}"
    )
    staging_zip_path = zip_path.with_name(
        f".{zip_path.name}.partial.{staging_token}"
    )
    needs_zip = args.output_format in {"zip", "both"}
    retain_zarr_dir = (
        args.output_format in {"dir", "both"} or args.keep_zarr_dir
    )
    pipeline_start = time.perf_counter()

    try:
        if zarr_dir.exists() or zip_path.exists():
            if not args.overwrite:
                raise SystemExit(f"输出已存在，请加 --overwrite: {zarr_dir} / {zip_path}")

        scan_results: list[
            tuple[Path, dict[str, Any] | None, float, str | None] | None
        ] = [None] * len(pkl_paths)
        scan_start = time.perf_counter()
        for idx, pkl_path, episode, scan_item_seconds, skip_reason in _iter_scanned_episodes(
            pkl_paths=pkl_paths,
            resize_hw=args.resize,
            skip_missing_images=args.skip_missing_images,
            trajectory_pose_source=args.trajectory_pose_source,
            gripper_source=args.gripper_source,
            image_fields=image_fields,
            extra_lowdim_fields=extra_lowdim_fields,
            workers=scan_worker_count,
            opencv_threads=args.opencv_threads,
            skip_empty_episodes=args.skip_empty_episodes,
            max_inflight_tasks=scan_inflight_tasks,
        ):
            scan_results[idx - 1] = (
                pkl_path,
                episode,
                scan_item_seconds,
                skip_reason,
            )
            if episode is None:
                print(
                    f"[scan {idx:04d}/{len(pkl_paths):04d}] SKIP {pkl_path} "
                    f"time={scan_item_seconds:.2f}s reason={skip_reason}",
                    flush=True,
                )
            else:
                print(
                    f"[scan {idx:04d}/{len(pkl_paths):04d}] {pkl_path} "
                    f"frames={len(episode['action'])} time={scan_item_seconds:.2f}s",
                    flush=True,
                )
        scan_seconds = time.perf_counter() - scan_start

        episodes: list[dict[str, Any]] = []
        used_paths: list[Path] = []
        episode_lengths: list[int] = []
        skipped_empty: list[tuple[Path, str]] = []
        for result in scan_results:
            if result is None:
                raise RuntimeError("episode scan result missing")
            pkl_path, episode, _, skip_reason = result
            if episode is None:
                skipped_empty.append((pkl_path, skip_reason or "empty episode"))
                continue
            episodes.append(episode)
            used_paths.append(pkl_path)
            episode_lengths.append(len(episode["action"]))
        del scan_results

        if not episodes:
            raise SystemExit("没有可写入的 episode")

        episode_ends = np.cumsum(
            np.asarray(episode_lengths, dtype=np.int64),
            dtype=np.int64,
        )
        total_frames = int(episode_ends[-1])
        image_hw = _resolve_image_hw(
            episodes=episodes,
            cv2=cv2,
            resize_hw=args.resize,
        )
        root = _init_preallocated_replay_buffer(
            first_episode=episodes[0],
            episode_ends=episode_ends,
            total_frames=total_frames,
            image_hw=image_hw,
            zarr_dir=staging_zarr_dir,
            zarr=zarr,
            numcodecs=numcodecs,
            image_compressor=image_compressor,
            image_compressor_name=image_compressor_name,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
        )

        lowdim_start = time.perf_counter()
        _write_preallocated_lowdim(
            root=root,
            episodes=episodes,
            lowdim_keys=lowdim_keys,
        )
        lowdim_seconds = time.perf_counter() - lowdim_start
        image_episodes = [{"image_paths": episode["image_paths"]} for episode in episodes]
        del episodes

        num_image_batches = (
            total_frames + args.image_batch_size - 1
        ) // args.image_batch_size
        image_inflight_tasks = min(
            resolved_max_inflight_tasks,
            max(1, num_image_batches),
        )
        image_worker_count = min(
            args.workers,
            max(1, num_image_batches),
            image_inflight_tasks,
        )
        height, width = image_hw
        raw_batch_memory_bytes = (
            image_worker_count
            * args.image_batch_size
            * height
            * width
            * 3
        )
        raw_batch_memory_mib = raw_batch_memory_bytes / (1024 ** 2)
        print(
            f"image stage: frames={total_frames}, hw={image_hw}, "
            f"threads={image_worker_count}, batches={num_image_batches}, "
            f"max_inflight={image_inflight_tasks}, "
            f"raw_batch_memory~={raw_batch_memory_mib:.1f}MiB",
            flush=True,
        )
        if raw_batch_memory_bytes > 4 * 1024 ** 3:
            print(
                "warning: 当前 workers*batch*分辨率估算的原始 RGB batch 超过 4GiB；"
                "建议减小 --workers 或 --image-batch-size。",
                file=sys.stderr,
                flush=True,
            )

        expected_camera_chunks = (1, height, width, 3)
        camera_arrays = {key: root["data"][key] for key in image_keys}
        for key, camera_array in camera_arrays.items():
            if camera_array.chunks != expected_camera_chunks:
                raise RuntimeError(
                    f"{key} chunk invariant violated: {camera_array.chunks} "
                    f"vs {expected_camera_chunks}"
                )
        image_writer = partial(
            _write_image_batch,
            camera_arrays=camera_arrays,
            cv2=cv2,
            resize_hw=args.resize,
            image_hw=image_hw,
        )
        image_tasks = _iter_image_batches(
            episodes=image_episodes,
            batch_size=args.image_batch_size,
            image_keys=image_keys,
        )
        completed_frames = 0
        decode_work_seconds = 0.0
        image_write_work_seconds = 0.0
        report_every = max(args.image_batch_size, total_frames // 20, 1)
        next_report = report_every
        image_start = time.perf_counter()

        def record_image_result(result: tuple[int, int, float, float]) -> None:
            nonlocal completed_frames
            nonlocal decode_work_seconds
            nonlocal image_write_work_seconds
            nonlocal next_report
            _, frame_count, decode_seconds, write_seconds = result
            completed_frames += frame_count
            decode_work_seconds += decode_seconds
            image_write_work_seconds += write_seconds
            if completed_frames >= next_report or completed_frames == total_frames:
                elapsed = max(time.perf_counter() - image_start, 1e-9)
                print(
                    f"images {completed_frames}/{total_frames} "
                    f"({completed_frames / elapsed:.1f} frames/s)",
                    flush=True,
                )
                while next_report <= completed_frames:
                    next_report += report_every

        if image_worker_count <= 1:
            for image_task in image_tasks:
                record_image_result(image_writer(image_task))
        else:
            with ThreadPoolExecutor(max_workers=image_worker_count) as executor:
                for result in _bounded_map_unordered(
                    executor=executor,
                    fn=image_writer,
                    tasks=image_tasks,
                    max_inflight_tasks=image_inflight_tasks,
                ):
                    record_image_result(result)
        image_seconds = time.perf_counter() - image_start
        del image_episodes
        if completed_frames != total_frames:
            raise RuntimeError(
                f"image write length mismatch: wrote={completed_frames}, "
                f"expected={total_frames}"
            )

        gripper_dim = int(root["data"]["robot0_gripper_width"].shape[1])
        action_dim = int(root["data"]["action"].shape[1])

        zip_seconds = 0.0
        if needs_zip:
            zip_start = time.perf_counter()
            print(
                f"zipping {staging_zarr_dir} -> {staging_zip_path}",
                flush=True,
            )
            _zip_zarr_dir(
                zarr_dir=staging_zarr_dir,
                zip_path=staging_zip_path,
                zarr=zarr,
            )
            zip_seconds = time.perf_counter() - zip_start
            print(f"zip done in {zip_seconds:.2f}s", flush=True)

        validation_start = time.perf_counter()
        validation_path = staging_zip_path if needs_zip else staging_zarr_dir
        _validate_dataset_path(
            dataset_path=validation_path,
            zarr=zarr,
            expected_episode_ends=episode_ends,
            expected_image_hw=image_hw,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
        )
        validation_seconds = time.perf_counter() - validation_start

        publish_start = time.perf_counter()
        if needs_zip:
            os.replace(staging_zip_path, zip_path)
        if retain_zarr_dir:
            _publish_directory(staging_zarr_dir, zarr_dir)
        else:
            _remove_path(staging_zarr_dir)
            if zarr_dir.exists():
                _remove_path(zarr_dir)
        if not needs_zip and zip_path.exists():
            _remove_path(zip_path)
        publish_seconds = time.perf_counter() - publish_start

        dataset_path = zip_path if needs_zip else zarr_dir
        timings = {
            "scan": scan_seconds,
            "lowdim_write": lowdim_seconds,
            "image_stage": image_seconds,
            "image_decode_worker_sum": decode_work_seconds,
            "image_zarr_write_worker_sum": image_write_work_seconds,
            "zip": zip_seconds,
            "validation": validation_seconds,
            "publish": publish_seconds,
            "total": time.perf_counter() - pipeline_start,
        }
        _write_count_txt(pkl_paths=used_paths, output_dir=output_dir)
        _write_manifest(
            pkl_paths=used_paths,
            episode_lengths=episode_lengths,
            selected_episode_count=len(pkl_paths),
            skipped_episode_count=len(skipped_empty),
            output_dir=output_dir,
            dataset_path=dataset_path,
            zarr_dir=zarr_dir,
            zarr_dir_retained=retain_zarr_dir,
            zip_path=zip_path,
            resize_hw=args.resize,
            image_hw=image_hw,
            trajectory_pose_source=args.trajectory_pose_source,
            gripper_source=args.gripper_source,
            gripper_dim=gripper_dim,
            action_dim=action_dim,
            output_format=args.output_format,
            workers=scan_worker_count,
            requested_workers=args.workers,
            scan_workers=scan_worker_count,
            image_workers=image_worker_count,
            image_batch_size=args.image_batch_size,
            requested_max_inflight_tasks=args.max_inflight_tasks,
            scan_max_inflight_tasks=scan_inflight_tasks,
            image_max_inflight_tasks=image_inflight_tasks,
            blosc_threads=args.blosc_threads,
            opencv_threads=args.opencv_threads,
            image_compressor_name=image_compressor_name,
            image_fields=image_fields,
            extra_lowdim_fields=extra_lowdim_fields,
            timings=timings,
        )
        skipped_path = _write_skipped_episodes(
            skipped_empty=skipped_empty,
            output_dir=output_dir,
        )
        if skipped_path is not None:
            print(
                f"skipped empty episodes: {len(skipped_empty)} -> {skipped_path}",
                flush=True,
            )

        print(
            "timings: "
            f"scan={scan_seconds:.2f}s lowdim={lowdim_seconds:.2f}s "
            f"images={image_seconds:.2f}s zip={zip_seconds:.2f}s "
            f"validate={validation_seconds:.2f}s "
            f"total={timings['total']:.2f}s",
            flush=True,
        )
        _print_summary(output_dir=output_dir, dataset_path=dataset_path, zarr=zarr)
    finally:
        _remove_path(staging_zarr_dir)
        _remove_path(staging_zip_path)
        _release_output_lock(lock_path)


if __name__ == "__main__":
    main()
