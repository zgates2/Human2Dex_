#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime visualization for raw PICO hand tracking data.

The script reads PicoHandReader.read_raw(), which returns the untouched
PICO 26x7 array:

    [x, y, z, qx, qy, qz, qw]

Each point is rendered in PICO World coordinates, with its local quaternion
axes drawn as:

    X: red, Y: green, Z: blue

Run:
    python3 test/visualize_pico_raw26.py
    python3 test/visualize_pico_raw26.py --hand left --center wrist
    python3 test/visualize_pico_raw26.py --axis-labels
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TELEOP_DIR = _REPO_ROOT / "teleop"
for _p in (_REPO_ROOT, _TELEOP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pico_hand import PicoHandReader  # noqa: E402


PICO26_NAMES = [
    "palm",
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index_tip",
    "middle_metacarpal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle_tip",
    "ring_metacarpal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "ring_tip",
    "pinky_metacarpal",
    "pinky_proximal",
    "pinky_intermediate",
    "pinky_distal",
    "pinky_tip",
]


PICO26_EDGES = [
    (0, 1),
    (0, 2), (2, 3), (3, 4), (4, 5),
    (0, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (0, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert [qx, qy, qz, qw] to a 3x3 rotation matrix."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(q)):
        return np.eye(3, dtype=np.float64)

    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def batch_rotmats(raw26x7: np.ndarray) -> np.ndarray:
    return np.stack([quat_xyzw_to_rotmat(q) for q in raw26x7[:, 3:7]], axis=0)


def segments_from_edges(points: np.ndarray, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    return np.asarray([[points[i], points[j]] for i, j in edges], dtype=np.float64)


def axis_segments(points: np.ndarray, rotmats: np.ndarray, axis: int, length: float) -> np.ndarray:
    return np.stack([points, points + rotmats[:, :, axis] * length], axis=1)


def set_equal_limits(ax, points: np.ndarray, radius: float) -> None:
    center = points.mean(axis=0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def display_origin(raw_points: np.ndarray, center: str) -> np.ndarray:
    if center == "palm":
        return raw_points[0].copy()
    if center == "wrist":
        return raw_points[1].copy()
    if center == "mean":
        return raw_points.mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime 3D visualization for raw PICO 26x7 hand tracking data."
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="PICO hand to read. Default: right.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Visualization update rate. Default: 30.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.025,
        help="Length of each local xyz axis in meters. Default: 0.025.",
    )
    parser.add_argument(
        "--view-radius",
        type=float,
        default=0.18,
        help="Half-width of the 3D view in meters. Default: 0.18.",
    )
    parser.add_argument(
        "--center",
        choices=["palm", "wrist", "mean", "none"],
        default="palm",
        help="Display-only centering. Raw PICO data is still unmodified before rendering. Default: palm.",
    )
    parser.add_argument(
        "--no-point-labels",
        action="store_true",
        help="Hide index/name labels for the 26 points.",
    )
    parser.add_argument(
        "--axis-labels",
        action="store_true",
        help="Draw small x/y/z letters at every local axis endpoint. This is verbose.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print one status line every N frames. Set 0 to disable. Default: 30.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    period = 1.0 / max(float(args.hz), 1e-3)

    plt.ion()
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("PICO raw 26-point viewer")

    ax.set_xlabel("PICO X (m)")
    ax.set_ylabel("PICO Y (m)")
    ax.set_zlabel("PICO Z (m)")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    scatter = ax.scatter([], [], [], s=35, c="black", depthshade=True)
    skeleton = Line3DCollection([], colors="0.35", linewidths=1.5, alpha=0.75)
    x_axes = Line3DCollection([], colors="red", linewidths=1.0, alpha=0.95)
    y_axes = Line3DCollection([], colors="green", linewidths=1.0, alpha=0.95)
    z_axes = Line3DCollection([], colors="blue", linewidths=1.0, alpha=0.95)

    for collection in (skeleton, x_axes, y_axes, z_axes):
        # Newer matplotlib versions try to autoscale immediately when a 3D
        # collection is added. Empty collections do not provide X/Y/Z arrays
        # yet, so disable that first autoscale and set limits per frame below.
        ax.add_collection3d(collection, autolim=False)

    point_labels = []
    if not args.no_point_labels:
        point_labels = [
            ax.text(0.0, 0.0, 0.0, f"{i}:{name}", fontsize=7, color="black")
            for i, name in enumerate(PICO26_NAMES)
        ]

    axis_labels = []
    if args.axis_labels:
        for _ in range(26):
            axis_labels.append(
                (
                    ax.text(0.0, 0.0, 0.0, "x", fontsize=6, color="red"),
                    ax.text(0.0, 0.0, 0.0, "y", fontsize=6, color="green"),
                    ax.text(0.0, 0.0, 0.0, "z", fontsize=6, color="blue"),
                )
            )

    ax.legend(
        handles=[
            Line2D([0], [0], color="0.35", lw=1.5, label="skeleton"),
            Line2D([0], [0], color="red", lw=1.5, label="local X"),
            Line2D([0], [0], color="green", lw=1.5, label="local Y"),
            Line2D([0], [0], color="blue", lw=1.5, label="local Z"),
        ],
        loc="upper right",
    )

    print("[Init] Connecting to PICO...")
    frame_idx = 0
    last_valid = None

    with PicoHandReader(hand=args.hand) as reader:
        print("[Run] Close the figure or press Ctrl+C to stop.")
        try:
            while plt.fignum_exists(fig.number):
                t0 = time.perf_counter()
                raw26x7, active = reader.read_raw()
                ts_ns = reader.get_timestamp_ns()

                valid = (
                    active == 1
                    and raw26x7 is not None
                    and raw26x7.ndim == 2
                    and raw26x7.shape[0] == 26
                    and raw26x7.shape[1] >= 7
                    and np.all(np.isfinite(raw26x7[:, :7]))
                )

                if valid:
                    last_valid = raw26x7[:, :7].astype(np.float64, copy=True)
                elif last_valid is None:
                    ax.set_title(f"PICO raw 26 points | active={active} | waiting for valid data")
                    plt.pause(max(0.001, period))
                    continue

                raw = last_valid
                raw_points = raw[:, :3]
                origin = display_origin(raw_points, args.center)
                points = raw_points - origin
                rotmats = batch_rotmats(raw)

                scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
                skeleton.set_segments(segments_from_edges(points, PICO26_EDGES))
                x_axes.set_segments(axis_segments(points, rotmats, 0, args.axis_length))
                y_axes.set_segments(axis_segments(points, rotmats, 1, args.axis_length))
                z_axes.set_segments(axis_segments(points, rotmats, 2, args.axis_length))

                for label, p in zip(point_labels, points):
                    label.set_position((p[0], p[1]))
                    label.set_3d_properties(p[2])

                if axis_labels:
                    for labels, p, rot in zip(axis_labels, points, rotmats):
                        ends = [p + rot[:, axis] * args.axis_length for axis in range(3)]
                        for label, end in zip(labels, ends):
                            label.set_position((end[0], end[1]))
                            label.set_3d_properties(end[2])

                set_equal_limits(ax, points, args.view_radius)
                center_text = "raw" if args.center == "none" else f"center={args.center}"
                palm_xyz = raw_points[0]
                wrist_xyz = raw_points[1]
                ax.set_title(
                    "PICO raw 26 points | "
                    f"hand={args.hand} active={active} ts={ts_ns} | {center_text}\n"
                    f"palm xyz={np.round(palm_xyz, 4).tolist()}  "
                    f"wrist xyz={np.round(wrist_xyz, 4).tolist()}"
                )

                fig.canvas.draw_idle()
                elapsed = time.perf_counter() - t0
                plt.pause(max(0.001, period - elapsed))

                if args.print_every > 0 and frame_idx % args.print_every == 0:
                    print(
                        f"[Frame {frame_idx}] active={active} ts={ts_ns} "
                        f"palm={np.round(palm_xyz, 4).tolist()} "
                        f"wrist={np.round(wrist_xyz, 4).tolist()}"
                    )
                frame_idx += 1
        except KeyboardInterrupt:
            print("\n[Exit] Interrupted by user.")


if __name__ == "__main__":
    main()
