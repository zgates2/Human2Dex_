#!/usr/bin/env python3
"""
Batch quality check for DexUMI PKL episodes.

Metrics:
  1. RGB image duplicate rate per episode
  2. trajectoryPose delta per episode
  3. o6_command anomaly statistics

Examples:
    python3 test/check_pkl_dataset_quality.py /home/zjc/Desktop/human2dex/data

    #递归检查目录下所有子目录中的 PKL 文件
    python3 test/check_pkl_dataset_quality.py /home/zjc/Desktop/human2dex/data --recursive 
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_pkl_trajectory_delta import (  # noqa: E402
    _max_pair,
    _pose_deltas,
    load_episode,
)


@dataclass
class EpisodeQuality:
    pkl_path: Path
    bundle_dir: Path
    num_messages: int
    rgb_messages: int
    rgb_existing: int
    rgb_missing: int
    rgb_unique_ratio: float
    rgb_repeat_ratio: float
    rgb_consecutive_dup_ratio: float
    rgb_dup_time_ratio: float
    rgb_max_repeat_run: int
    traj_frames: int
    max_trans_delta_m: float
    max_trans_pair: tuple[int, int] | None
    max_rot_delta_deg: float
    max_rot_pair: tuple[int, int] | None
    trans_spike_count: int
    rot_spike_count: int
    cmd_frames: int
    cmd_min: int
    cmd_max: int
    cmd_over250_count: int
    cmd_over250_ratio: float
    cmd_step_max: int
    cmd_step_pair: tuple[int, int] | None
    cmd_step_channel: int | None
    cmd_step_mean: float
    cmd_step_p95: float
    cmd_spike_count: int


def _msg_get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _msg_time_sec(msg: Any) -> float | None:
    for key, scale in (
        ("sampleClockNs", 1e-9),
        ("mainClockMonotonicNs", 1e-9),
        ("timestamp", 1.0),
    ):
        value = _msg_get(msg, key)
        if value is None:
            continue
        try:
            return float(value) * scale
        except (TypeError, ValueError):
            continue
    return None


def _iter_pkl_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*.pkl" if recursive else "*.pkl"
    return sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and "_quality_checks" not in p.parts
    )


def _hash_file(path: Path) -> bytes:
    return hashlib.sha1(path.read_bytes()).digest()


def _image_quality_stats(episode) -> dict[str, Any]:
    rgb_hashes: list[bytes] = []
    rgb_times: list[float] = []
    rgb_existing = 0
    rgb_missing = 0

    for msg in episode.messages:
        rel = _msg_get(msg, "rgbImage")
        if rel in (None, "", False):
            continue
        rgb_existing += 1
        img_path = episode.pkl_path.parent / str(rel)
        if not img_path.is_file():
            rgb_missing += 1
            continue
        rgb_hashes.append(_hash_file(img_path))
        t = _msg_time_sec(msg)
        if t is not None:
            rgb_times.append(t)

    if not rgb_hashes:
        return {
            "rgb_messages": rgb_existing,
            "rgb_existing": 0,
            "rgb_missing": rgb_missing,
            "rgb_unique_ratio": 0.0,
            "rgb_repeat_ratio": 0.0,
            "rgb_consecutive_dup_ratio": 0.0,
            "rgb_dup_time_ratio": 0.0,
            "rgb_max_repeat_run": 0,
        }

    unique_ratio = len(set(rgb_hashes)) / max(len(rgb_hashes), 1)
    repeat_ratio = 1.0 - unique_ratio
    dup_consecutive = 0
    max_repeat_run = 0
    current_run = 0
    dup_time = 0.0

    for idx in range(1, len(rgb_hashes)):
        if rgb_hashes[idx] == rgb_hashes[idx - 1]:
            dup_consecutive += 1
            current_run += 1
            if idx - 1 < len(rgb_times) and idx < len(rgb_times):
                dt = rgb_times[idx] - rgb_times[idx - 1]
                if dt > 0:
                    dup_time += dt
        else:
            current_run = 0
        max_repeat_run = max(max_repeat_run, current_run)

    total_time = 0.0
    if len(rgb_times) >= 2:
        total_time = max(0.0, rgb_times[-1] - rgb_times[0])

    return {
        "rgb_messages": rgb_existing,
        "rgb_existing": len(rgb_hashes),
        "rgb_missing": rgb_missing,
        "rgb_unique_ratio": unique_ratio,
        "rgb_repeat_ratio": repeat_ratio,
        "rgb_consecutive_dup_ratio": dup_consecutive / max(len(rgb_hashes) - 1, 1),
        "rgb_dup_time_ratio": dup_time / total_time if total_time > 0 else 0.0,
        "rgb_max_repeat_run": max_repeat_run,
    }


def _trajectory_quality_stats(episode, trans_threshold: float, rot_threshold_deg: float) -> dict[str, Any]:
    trans_delta, rot_delta, _ = _pose_deltas(episode.pose_mats)
    if trans_delta is None or rot_delta is None:
        return {
            "traj_frames": 0,
            "max_trans_delta_m": 0.0,
            "max_trans_pair": None,
            "max_rot_delta_deg": 0.0,
            "max_rot_pair": None,
            "trans_spike_count": 0,
            "rot_spike_count": 0,
        }

    max_trans, trans_pair, _ = _max_pair(trans_delta)
    max_rot_rad, rot_pair, _ = _max_pair(rot_delta)
    trans_spike_count = int(np.count_nonzero(trans_delta.values > trans_threshold))
    rot_spike_count = int(np.count_nonzero(np.rad2deg(rot_delta.values) > rot_threshold_deg))
    return {
        "traj_frames": int(len(episode.pose_mats.values)),
        "max_trans_delta_m": float(max_trans),
        "max_trans_pair": trans_pair,
        "max_rot_delta_deg": float(np.rad2deg(max_rot_rad)),
        "max_rot_pair": rot_pair,
        "trans_spike_count": trans_spike_count,
        "rot_spike_count": rot_spike_count,
    }


def _hand_command_quality_stats(episode, cmd_step_threshold: int) -> dict[str, Any]:
    if episode.hand_cmds is None or len(episode.hand_cmds.values) == 0:
        return {
            "cmd_frames": 0,
            "cmd_min": 0,
            "cmd_max": 0,
            "cmd_over250_count": 0,
            "cmd_over250_ratio": 0.0,
            "cmd_step_max": 0,
            "cmd_step_pair": None,
            "cmd_step_channel": None,
            "cmd_step_mean": 0.0,
            "cmd_step_p95": 0.0,
            "cmd_spike_count": 0,
        }

    cmd = np.asarray(episode.hand_cmds.values, dtype=np.int16)
    cmd_min = int(cmd.min())
    cmd_max = int(cmd.max())
    over250_count = int(np.count_nonzero(cmd > 250))
    over250_ratio = over250_count / max(cmd.size, 1)

    if len(cmd) > 1:
        step = np.abs(np.diff(cmd, axis=0))
        step_peak = np.max(step, axis=1)
        max_idx = int(np.argmax(step_peak))
        max_step = int(step_peak[max_idx])
        max_channel = int(np.argmax(step[max_idx]))
        step_pair = (
            int(episode.hand_cmds.frame_indices[max_idx]),
            int(episode.hand_cmds.frame_indices[max_idx + 1]),
        )
        step_mean = float(step.mean())
        step_p95 = float(np.percentile(step, 95))
        spike_count = int(np.count_nonzero(step > cmd_step_threshold))
    else:
        max_step = 0
        max_channel = None
        step_pair = None
        step_mean = 0.0
        step_p95 = 0.0
        spike_count = 0

    return {
        "cmd_frames": int(len(cmd)),
        "cmd_min": cmd_min,
        "cmd_max": cmd_max,
        "cmd_over250_count": over250_count,
        "cmd_over250_ratio": over250_ratio,
        "cmd_step_max": max_step,
        "cmd_step_pair": step_pair,
        "cmd_step_channel": max_channel,
        "cmd_step_mean": step_mean,
        "cmd_step_p95": step_p95,
        "cmd_spike_count": spike_count,
    }


def analyze_episode(
    pkl_path: Path,
    trans_threshold: float,
    rot_threshold_deg: float,
    cmd_step_threshold: int,
) -> EpisodeQuality:
    episode = load_episode(pkl_path)
    rgb_stats = _image_quality_stats(episode)
    traj_stats = _trajectory_quality_stats(episode, trans_threshold, rot_threshold_deg)
    cmd_stats = _hand_command_quality_stats(episode, cmd_step_threshold)

    return EpisodeQuality(
        pkl_path=pkl_path,
        bundle_dir=pkl_path.parent,
        num_messages=len(episode.messages),
        **rgb_stats,
        **traj_stats,
        **cmd_stats,
    )


def _fmt_pair(pair: tuple[int, int] | None) -> str:
    return "" if pair is None else f"{pair[0]}->{pair[1]}"


def _write_csv(results: list[EpisodeQuality], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(results[0].__dataclass_fields__.keys()) if results else [])
        for r in results:
            writer.writerow([
                r.pkl_path,
                r.bundle_dir,
                r.num_messages,
                r.rgb_messages,
                r.rgb_existing,
                r.rgb_missing,
                f"{r.rgb_unique_ratio:.6f}",
                f"{r.rgb_repeat_ratio:.6f}",
                f"{r.rgb_consecutive_dup_ratio:.6f}",
                f"{r.rgb_dup_time_ratio:.6f}",
                r.rgb_max_repeat_run,
                r.traj_frames,
                f"{r.max_trans_delta_m:.6f}",
                _fmt_pair(r.max_trans_pair),
                f"{r.max_rot_delta_deg:.6f}",
                _fmt_pair(r.max_rot_pair),
                r.trans_spike_count,
                r.rot_spike_count,
                r.cmd_frames,
                r.cmd_min,
                r.cmd_max,
                r.cmd_over250_count,
                f"{r.cmd_over250_ratio:.6f}",
                r.cmd_step_max,
                _fmt_pair(r.cmd_step_pair),
                "" if r.cmd_step_channel is None else r.cmd_step_channel,
                f"{r.cmd_step_mean:.6f}",
                f"{r.cmd_step_p95:.6f}",
                r.cmd_spike_count,
            ])


def _top(rows: list[EpisodeQuality], key_fn, k: int = 10) -> list[EpisodeQuality]:
    return sorted(rows, key=key_fn, reverse=True)[:k]


def _write_markdown(results: list[EpisodeQuality], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = len(results)
    mean_dup = float(np.mean([r.rgb_consecutive_dup_ratio for r in results])) if results else 0.0
    mean_trans = float(np.mean([r.max_trans_delta_m for r in results])) if results else 0.0
    mean_rot = float(np.mean([r.max_rot_delta_deg for r in results])) if results else 0.0
    mean_cmd_over = float(np.mean([r.cmd_over250_ratio for r in results])) if results else 0.0
    any_cmd_over = sum(1 for r in results if r.cmd_over250_count > 0)

    lines: list[str] = []
    lines.append("# PKL Quality Summary")
    lines.append("")
    lines.append(f"- episodes: {episodes}")
    lines.append(f"- mean image consecutive duplicate ratio: {mean_dup:.4f}")
    lines.append(f"- mean trajectory max translation delta: {mean_trans:.6f} m")
    lines.append(f"- mean trajectory max rotation delta: {mean_rot:.3f} deg")
    lines.append(f"- mean o6_command >250 ratio: {mean_cmd_over:.6f}")
    lines.append(f"- episodes with o6_command >250: {any_cmd_over}")
    lines.append("")

    def add_table(title: str, rows: list[EpisodeQuality], metric_label: str, value_fn) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| episode | value | detail |")
        lines.append("|---|---:|---|")
        for r in rows:
            lines.append(f"| `{r.pkl_path.parent.name}` | {value_fn(r)} | {metric_label(r)} |")
        lines.append("")

    add_table(
        "Top image repeat episodes",
        _top(results, lambda r: r.rgb_consecutive_dup_ratio),
        lambda r: f"{r.rgb_consecutive_dup_ratio:.4f}",
        lambda r: f"repeat={r.rgb_repeat_ratio:.4f}, time={r.rgb_dup_time_ratio:.4f}, max_run={r.rgb_max_repeat_run}",
    )
    add_table(
        "Top trajectory delta episodes",
        _top(results, lambda r: max(r.max_trans_delta_m, np.deg2rad(r.max_rot_delta_deg))),
        lambda r: f"{r.max_trans_delta_m:.6f} m / {r.max_rot_delta_deg:.2f} deg",
        lambda r: f"trans_pair={_fmt_pair(r.max_trans_pair)}, rot_pair={_fmt_pair(r.max_rot_pair)}, spikes={r.trans_spike_count}/{r.rot_spike_count}",
    )
    add_table(
        "Top o6_command anomaly episodes",
        _top(results, lambda r: (r.cmd_over250_count, r.cmd_step_max)),
        lambda r: f"{r.cmd_over250_count} over250, step={r.cmd_step_max}",
        lambda r: f"step_pair={_fmt_pair(r.cmd_step_pair)}, ch={r.cmd_step_channel}, p95={r.cmd_step_p95:.2f}",
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch quality check for DexUMI PKL episodes."
    )
    parser.add_argument("input", type=Path, help="PKL file or directory.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search PKL files recursively when input is a directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSV/Markdown. Default: <input-parent>/data_quality_checks",
    )
    parser.add_argument(
        "--trans-threshold",
        type=float,
        default=0.05,
        help="Trajectory translation spike threshold in meters.",
    )
    parser.add_argument(
        "--rot-threshold-deg",
        type=float,
        default=5.0,
        help="Trajectory rotation spike threshold in degrees.",
    )
    parser.add_argument(
        "--cmd-step-threshold",
        type=int,
        default=80,
        help="HandCommand step spike threshold.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.out_dir is None:
        base_dir = input_path.parent if input_path.is_file() else input_path.parent
        out_dir = base_dir / "data_quality_checks"
    else:
        out_dir = args.out_dir.expanduser().resolve()

    pkl_files = _iter_pkl_files(input_path, args.recursive)
    if not pkl_files:
        raise FileNotFoundError(f"No PKL files found under: {input_path}")

    print(f"[Info] Found {len(pkl_files)} PKL files.")
    results: list[EpisodeQuality] = []
    for idx, pkl_path in enumerate(pkl_files, 1):
        try:
            result = analyze_episode(
                pkl_path,
                trans_threshold=args.trans_threshold,
                rot_threshold_deg=args.rot_threshold_deg,
                cmd_step_threshold=args.cmd_step_threshold,
            )
            results.append(result)
            print(
                f"[{idx:03d}/{len(pkl_files):03d}] {pkl_path.parent.name}: "
                f"img_dup={result.rgb_consecutive_dup_ratio:.4f}, "
                f"traj={result.max_trans_delta_m:.4f}m/{result.max_rot_delta_deg:.2f}deg, "
                f"cmd>250={result.cmd_over250_count}, cmd_step={result.cmd_step_max}"
            )
        except Exception as exc:
            print(f"[Skip] {pkl_path}: {exc}")

    if not results:
        raise RuntimeError("No PKL file was processed successfully.")

    csv_path = out_dir / "pkl_quality_summary.csv"
    md_path = out_dir / "pkl_quality_summary.md"
    _write_csv(results, csv_path)
    _write_markdown(results, md_path)

    print(f"[Done] CSV: {csv_path}")
    print(f"[Done] Markdown: {md_path}")


if __name__ == "__main__":
    main()
