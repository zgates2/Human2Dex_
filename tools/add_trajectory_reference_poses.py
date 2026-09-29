#!/usr/bin/env python3
"""
Append palm- and TCP-referenced trajectory poses to DexUMI PKL episodes.

For every message with a valid ``raw26x7`` field, this script adds:

``trajectoryPose_palm``
    A float32 pose ``[x, y, z, rx, ry, rz]`` whose reference position and
    orientation come from the PICO palm pose ``raw26x7[0]``.

``trajectoryPose_tcp``
    The same orientation as ``trajectoryPose_palm``. Its position is offset
    from the palm in the *raw PICO palm local frame* by the configured offset,
    which defaults to ``[-30, +40, 0]`` millimetres.

The output orientation uses the same local-axis remapping as DexUMI's current
``trajectoryPose`` implementation:

    R_trajectory = R_raw_palm @ PICO_WRIST_TO_TARGET_TCP_ROT

By default, source PKLs are not modified. A sibling output directory named
``<input>_trajectory_refs`` is created. Use ``--in-place`` explicitly to
atomically replace source PKLs.
""" 

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:  # pragma: no cover - scipy is optional
    _Rotation = None


DEFAULT_INPUT_ROOT = Path(
    "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_bread"
)
DEFAULT_TCP_OFFSET_MM = (-30.0, 40.0, 0.0)

# PICO raw local axes: x=right, y=up, z=backward.
# DexUMI trajectory local axes: x=up, y=right, z=forward.
PICO_WRIST_TO_TARGET_TCP_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

NUMPY2_CORE_PREFIX = "numpy._core"
NUMPY1_CORE_PREFIX = "numpy.core"


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 2-created PKLs in a NumPy 1 environment when necessary."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == NUMPY2_CORE_PREFIX or module.startswith(
                f"{NUMPY2_CORE_PREFIX}."
            ):
                compat_module = NUMPY1_CORE_PREFIX + module[len(NUMPY2_CORE_PREFIX) :]
                return super().find_class(compat_module, name)
            raise


def install_numpy_pickle_compat() -> list[str]:
    try:
        core = importlib.import_module("numpy._core")
    except ImportError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            core = importlib.import_module("numpy.core")

    aliases = {
        "numpy._core": core,
        "numpy._core.multiarray": getattr(core, "multiarray", None),
        "numpy._core.numeric": getattr(core, "numeric", None),
    }
    added: list[str] = []
    for name, module in aliases.items():
        if module is not None and name not in sys.modules:
            sys.modules[name] = module
            added.append(name)
    return added


def read_pkl(path: Path) -> dict[str, Any]:
    added = install_numpy_pickle_compat()
    try:
        with path.open("rb") as file:
            data = NumpyCompatUnpickler(file).load()
    finally:
        for name in reversed(added):
            sys.modules.pop(name, None)

    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        raise ValueError(f"PKL must contain a dict with list field 'messages': {path}")
    return data


def atomic_write_pkl(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def quat_xyzw_to_rotmat(quaternion: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion contains NaN or Inf")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")

    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_rotvec_fallback(rotmat: Any) -> np.ndarray:
    """Convert a rotation matrix to an axis-angle vector without SciPy."""
    rotation = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
    cos_angle = (float(np.trace(rotation)) - 1.0) * 0.5
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    sin_angle = float(np.sin(angle))
    if abs(sin_angle) > 1e-6:
        axis /= 2.0 * sin_angle
    else:
        index = int(np.argmax(np.diag(rotation)))
        axis = np.zeros(3, dtype=np.float64)
        axis[index] = np.sqrt(max(rotation[index, index] + 1.0, 0.0) * 0.5)
        denominator = 4.0 * axis[index] + 1e-12
        if index == 0:
            axis[1] = (rotation[0, 1] + rotation[1, 0]) / denominator
            axis[2] = (rotation[0, 2] + rotation[2, 0]) / denominator
        elif index == 1:
            axis[0] = (rotation[0, 1] + rotation[1, 0]) / denominator
            axis[2] = (rotation[1, 2] + rotation[2, 1]) / denominator
        else:
            axis[0] = (rotation[0, 2] + rotation[2, 0]) / denominator
            axis[1] = (rotation[1, 2] + rotation[2, 1]) / denominator
        axis /= np.linalg.norm(axis) + 1e-12
    return axis * angle


def rotmat_to_rotvec(rotmat: Any) -> np.ndarray:
    rotation = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
    if _Rotation is not None:
        return _Rotation.from_matrix(rotation).as_rotvec()
    return rotmat_to_rotvec_fallback(rotation)


def validate_raw26x7(value: Any) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 26 or raw.shape[1] < 7:
        raise ValueError(f"expected raw26x7 shape (26, >=7), got {raw.shape}")
    if not np.all(np.isfinite(raw[0, :7])):
        raise ValueError("raw26x7[0, :7] contains NaN or Inf")
    return raw


def build_palm_and_tcp_poses(
    raw26x7: Any,
    tcp_offset_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build palm and TCP 6D poses from the PICO palm pose at raw index 0."""
    raw = validate_raw26x7(raw26x7)
    palm_pose = raw[0, :7]
    palm_position = palm_pose[:3]
    raw_palm_rotation = quat_xyzw_to_rotmat(palm_pose[3:7])

    # The offset is defined in raw26x7[0]'s local coordinate system.
    tcp_position = palm_position + raw_palm_rotation @ tcp_offset_m

    # Match the orientation convention of the existing trajectoryPose field.
    trajectory_rotation = raw_palm_rotation @ PICO_WRIST_TO_TARGET_TCP_ROT
    trajectory_rotvec = rotmat_to_rotvec(trajectory_rotation)

    trajectory_palm = np.concatenate([palm_position, trajectory_rotvec]).astype(
        np.float32
    )
    trajectory_tcp = np.concatenate([tcp_position, trajectory_rotvec]).astype(
        np.float32
    )
    return trajectory_palm, trajectory_tcp


def valid_pose6(value: Any) -> bool:
    try:
        pose = np.asarray(value, dtype=np.float32)
    except Exception:
        return False
    return pose.shape == (6,) and bool(np.all(np.isfinite(pose)))


@dataclass
class FileStats:
    messages: int = 0
    added: int = 0
    already_complete: int = 0
    invalid: int = 0

    def __iadd__(self, other: "FileStats") -> "FileStats":
        self.messages += other.messages
        self.added += other.added
        self.already_complete += other.already_complete
        self.invalid += other.invalid
        return self


def augment_episode(
    data: dict[str, Any],
    tcp_offset_m: np.ndarray,
    overwrite_existing_fields: bool,
    invalid_policy: str,
) -> tuple[dict[str, Any], FileStats]:
    stats = FileStats(messages=len(data["messages"]))

    for frame_index, message in enumerate(data["messages"]):
        if not isinstance(message, dict):
            raise ValueError(f"message {frame_index} is not a dict")

        palm_present = "trajectoryPose_palm" in message
        tcp_present = "trajectoryPose_tcp" in message
        fields_complete = valid_pose6(message.get("trajectoryPose_palm")) and valid_pose6(
            message.get("trajectoryPose_tcp")
        )

        if fields_complete and not overwrite_existing_fields:
            stats.already_complete += 1
            continue
        if (palm_present or tcp_present) and not overwrite_existing_fields:
            raise ValueError(
                f"message {frame_index} has incomplete/invalid existing trajectory fields; "
                "use --overwrite-existing-fields to replace them"
            )

        try:
            palm_pose, tcp_pose = build_palm_and_tcp_poses(
                message.get("raw26x7"), tcp_offset_m
            )
        except (TypeError, ValueError) as exc:
            stats.invalid += 1
            if invalid_policy == "error":
                raise ValueError(f"message {frame_index}: {exc}") from exc
            message["trajectoryPose_palm"] = None
            message["trajectoryPose_tcp"] = None
            continue

        message["trajectoryPose_palm"] = palm_pose
        message["trajectoryPose_tcp"] = tcp_pose
        stats.added += 1

    return data, stats


def iter_pkl_paths(input_root: Path) -> list[Path]:
    if input_root.is_file():
        if input_root.suffix.lower() != ".pkl":
            raise ValueError(f"input file is not a .pkl: {input_root}")
        return [input_root]
    if not input_root.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_root}")
    return sorted(path for path in input_root.rglob("*.pkl") if path.is_file())


def output_path_for(source: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / source.name
    return output_root / source.relative_to(input_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append trajectoryPose_palm and trajectoryPose_tcp using raw26x7[0]. "
            "The default TCP offset is -30 mm on local x and +40 mm on local y."
        )
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Optional positional PKL file or dataset root. "
            "Kept for backward compatibility."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_option",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"PKL file or dataset root (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory. Default: sibling <input>_trajectory_refs.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace source PKLs instead of writing a new dataset directory.",
    )
    parser.add_argument(
        "--tcp-offset-mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=DEFAULT_TCP_OFFSET_MM,
        help="TCP offset in raw palm local coordinates (default: -30 40 0).",
    )
    parser.add_argument(
        "--overwrite-existing-fields",
        action="store_true",
        help="Replace existing trajectoryPose_palm/trajectoryPose_tcp fields.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacing PKLs that already exist under --output-root.",
    )
    parser.add_argument(
        "--invalid-policy",
        choices=("error", "none"),
        default="error",
        help="On invalid raw26x7: stop (error) or write both fields as None (none).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only the first N PKLs; useful for validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and compute all selected PKLs without writing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_root is not None and args.input_option is not None:
        raise SystemExit("use either positional input_root or --input, not both")
    selected_input = args.input_option or args.input_root or DEFAULT_INPUT_ROOT
    input_root = selected_input.expanduser().resolve()

    if args.in_place and args.output_root is not None:
        raise SystemExit("--in-place and --output-root cannot be used together")
    if args.max_files is not None and args.max_files <= 0:
        raise SystemExit("--max-files must be positive")

    if args.in_place:
        output_root = input_root if input_root.is_dir() else input_root.parent
    elif args.output_root is not None:
        output_root = args.output_root.expanduser().resolve()
    elif input_root.is_file():
        output_root = input_root.parent / f"{input_root.stem}_trajectory_refs"
    else:
        output_root = input_root.parent / f"{input_root.name}_trajectory_refs"

    tcp_offset_m = np.asarray(args.tcp_offset_mm, dtype=np.float64) / 1000.0
    paths = iter_pkl_paths(input_root)
    if args.max_files is not None:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit(f"no PKL files found under {input_root}")

    print(f"input: {input_root}")
    print(f"output: {'in-place' if args.in_place else output_root}")
    print(f"files: {len(paths)}")
    print(f"tcp_offset_local_mm: {np.asarray(args.tcp_offset_mm, dtype=float).tolist()}")

    total = FileStats()
    written_files = 0
    for file_index, source_path in enumerate(paths, start=1):
        destination_path = (
            source_path
            if args.in_place
            else output_path_for(source_path, input_root, output_root)
        )
        if (
            not args.in_place
            and destination_path.exists()
            and not args.overwrite_output
            and not args.dry_run
        ):
            raise FileExistsError(
                f"output already exists: {destination_path}; use --overwrite-output"
            )

        data = read_pkl(source_path)
        data, stats = augment_episode(
            data=data,
            tcp_offset_m=tcp_offset_m,
            overwrite_existing_fields=args.overwrite_existing_fields,
            invalid_policy=args.invalid_policy,
        )
        total += stats

        if not args.dry_run:
            atomic_write_pkl(data, destination_path)
            written_files += 1

        print(
            f"[{file_index:04d}/{len(paths):04d}] {source_path.name}: "
            f"messages={stats.messages} added={stats.added} "
            f"already={stats.already_complete} invalid={stats.invalid}"
        )

    print(
        "summary: "
        f"files={len(paths)} written_files={written_files} messages={total.messages} "
        f"added={total.added} already={total.already_complete} invalid={total.invalid} "
        f"dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
