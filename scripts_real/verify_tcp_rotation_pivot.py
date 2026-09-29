"""Visually verify the configured Franka policy TCP rotation center.

The script reads ``eval_franka_o6_config.yaml`` and uses the same
``FrankaInterpolationController``, robot config, speed limits, and
``tx_flange_tip`` path as ``eval_real_franka_o6.py``.  It keeps the configured
policy TCP translation fixed while applying small positive and negative local
rotations:

    center -> +angle -> center -> -angle -> center

If ``tx_flange_tip`` is correct, the physical point selected as the policy TCP
(for example, the dexterous-hand wrist center) should remain stationary while
the flange and the rest of the hand move around it.  Mark that physical point
or place a fixed visual reference next to it before running the test.

The script is print-only by default.  Real motion requires ``--execute`` and,
unless ``--yes`` is also supplied, confirmation before every tested axis.

Examples
--------

Print configuration only::

    python scripts_real/verify_tcp_rotation_pivot.py

Safest first real test, local policy z axis only, +/- 5 degrees::

    python scripts_real/verify_tcp_rotation_pivot.py \
        --execute --axes y --angle-deg 20 --duration 3

Test all local policy axes, +/- 8 degrees::

    python scripts_real/verify_tcp_rotation_pivot.py \
        --execute --axes xyz --angle-deg 8 --duration 3
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.common.pose_util import mat_to_pose, pose_to_mat


DEFAULT_INFERENCE_CONFIG_REL = "eval_franka_o6_config.yaml"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def load_yaml_mapping(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Expected a YAML mapping in {config_path}")
    return cfg


def load_franka_config(config_path: Path, robot_idx: int) -> dict:
    cfg = load_yaml_mapping(config_path)
    robots = cfg.get("robots", [])
    if robot_idx < 0 or robot_idx >= len(robots):
        raise RuntimeError(f"robot_idx={robot_idx} out of range for {config_path}")
    rc = robots[robot_idx]
    if not str(rc.get("robot_type", "")).startswith("franka"):
        raise RuntimeError(
            f"robots[{robot_idx}] is not franka: {rc.get('robot_type')!r}"
        )
    return rc


def make_controller(
    shm_manager,
    rc: dict,
    init_joints: bool,
    max_pos_speed: float,
    max_rot_speed: float,
):
    from umi.real_world.franka_interpolation_controller import (
        FrankaInterpolationController,
    )

    joints_init = rc.get("joints_init") if init_joints else None
    return FrankaInterpolationController(
        shm_manager=shm_manager,
        robot_ip=rc["robot_ip"],
        robot_port=rc.get("robot_port", 4242),
        frequency=rc.get("frequency", 200),
        Kx_scale=rc.get("Kx_scale", 1.0),
        Kxd_scale=rc.get(
            "Kxd_scale", np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0])
        ),
        launch_timeout=rc.get("launch_timeout", 20),
        pose_api=rc.get("pose_api", "rpy"),
        rpc_timeout=rc.get("rpc_timeout", 5.0),
        read_only=False,
        start_impedance_on_start=rc.get("start_impedance_on_start", False),
        impedance_start_delay=rc.get("impedance_start_delay", 0.5),
        joints_init=joints_init,
        joints_init_duration=4,
        verbose=rc.get("verbose", False),
        max_pos_speed=max_pos_speed,
        max_rot_speed=max_rot_speed,
        max_commands_per_cycle=rc.get("max_commands_per_cycle", 1),
        tx_flange_tip=rc.get("tx_flange_tip"),
        receive_latency=rc.get("robot_obs_latency", 0.0),
    )


def latest_tcp_pose(controller) -> np.ndarray:
    state = controller.get_state()
    return np.asarray(state["ActualTCPPose"], dtype=np.float64).copy()


def format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.6f}" for v in vec) + "]"


def build_local_rotation_target(
    anchor_tcp_pose: np.ndarray,
    axis: str,
    angle_rad: float,
) -> np.ndarray:
    """Rotate the policy TCP frame locally while keeping its xyz fixed."""
    anchor_mat = pose_to_mat(anchor_tcp_pose)
    axis_vec = np.zeros(3, dtype=np.float64)
    axis_vec[AXIS_INDEX[axis]] = float(angle_rad)
    delta_mat = np.eye(4, dtype=np.float64)
    delta_mat[:3, :3] = Rotation.from_rotvec(axis_vec).as_matrix()
    target_mat = anchor_mat @ delta_mat
    target_mat[:3, 3] = anchor_mat[:3, 3]
    return mat_to_pose(target_mat)


def tcp_to_link8_pose(tcp_pose: np.ndarray, tx_tip_flange: np.ndarray) -> np.ndarray:
    return mat_to_pose(pose_to_mat(tcp_pose) @ tx_tip_flange)


def schedule_and_wait(
    controller,
    target_pose: np.ndarray,
    duration_s: float,
    settle_s: float,
) -> None:
    lead_s = 0.15
    controller.schedule_waypoint(
        np.asarray(target_pose, dtype=np.float64),
        target_time=time.time() + lead_s + float(duration_s),
    )
    time.sleep(lead_s + float(duration_s) + float(settle_s))


def confirm_axis(axis: str, angle_deg: float, assume_yes: bool) -> bool:
    message = (
        f"Test local policy TCP {axis}-axis at +/-{angle_deg:.1f} deg. "
        "Ensure at least 15 cm clearance around the hand and flange."
    )
    if assume_yes:
        print(message)
        return True
    reply = input(message + "\nPress Enter to execute, or type q then Enter to skip: ")
    return reply.strip().lower() not in {"q", "quit", "skip", "n", "no"}


def confirm_connection(assume_yes: bool) -> bool:
    message = (
        "The inference program must be stopped. Keep only the Franka/ZeroRPC "
        "interface service running, clear the workspace, and keep the emergency "
        "stop within reach."
    )
    if assume_yes:
        print(message)
        return True
    reply = input(
        message
        + "\nPress Enter to connect and start impedance control, or type q then Enter to abort: "
    )
    return reply.strip().lower() not in {"q", "quit", "abort", "n", "no"}


def print_target(
    label: str,
    tcp_pose: np.ndarray,
    tx_tip_flange: np.ndarray,
    anchor_xyz: np.ndarray,
) -> None:
    link8_pose = tcp_to_link8_pose(tcp_pose, tx_tip_flange)
    tcp_delta_mm = (tcp_pose[:3] - anchor_xyz) * 1000.0
    print(f"  [{label}]")
    print(f"    policy_tcp_xyz       {format_vec(tcp_pose[:3])}")
    print(f"    policy_tcp_dxyz_mm   {format_vec(tcp_delta_mm)}")
    print(f"    policy_tcp_rotvec    {format_vec(tcp_pose[3:])}")
    print(f"    panda_link8_xyz      {format_vec(link8_pose[:3])}")
    print(f"    panda_link8_rotvec   {format_vec(link8_pose[3:])}")


def execute_target(
    controller,
    label: str,
    target_pose: np.ndarray,
    tx_tip_flange: np.ndarray,
    anchor_xyz: np.ndarray,
    duration_s: float,
    settle_s: float,
) -> None:
    print_target(label, target_pose, tx_tip_flange, anchor_xyz)
    schedule_and_wait(controller, target_pose, duration_s, settle_s)
    reached_tcp = latest_tcp_pose(controller)
    reached_link8 = tcp_to_link8_pose(reached_tcp, tx_tip_flange)
    drift_mm = (reached_tcp[:3] - anchor_xyz) * 1000.0
    print(f"    reached_tcp_xyz      {format_vec(reached_tcp[:3])}")
    print(f"    controller_drift_mm  {format_vec(drift_mm)}")
    print(f"    reached_link8_xyz    {format_vec(reached_link8[:3])}")


def parse_axes(value: str) -> list[str]:
    axes: list[str] = []
    for axis in value.lower():
        if axis not in AXIS_INDEX:
            raise ValueError(
                f"Unsupported axis {axis!r}; use a combination of x, y, z"
            )
        if axis not in axes:
            axes.append(axis)
    if not axes:
        raise ValueError("At least one axis is required")
    return axes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate a Franka around the configured policy TCP point."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(ROOT_DIR / DEFAULT_INFERENCE_CONFIG_REL),
        help="Path to eval_franka_o6_config.yaml.",
    )
    parser.add_argument(
        "-rc",
        "--robot-config",
        default=None,
        help="Optional robot YAML override; defaults to config.robot_config.",
    )
    parser.add_argument("--robot-idx", type=int, default=0)
    parser.add_argument(
        "--axes",
        default="xyz",
        help="Local policy TCP axes to test, e.g. z, xy, or xyz.",
    )
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=8.0,
        help="Positive and negative rotation magnitude in degrees (max 90).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Seconds for each center/rotation leg.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="Seconds to wait after each leg.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually connect and move. Without this flag the script is print-only.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the confirmation prompt before each axis.",
    )
    parser.add_argument(
        "--init-joints",
        "-j",
        action="store_true",
        help="Move to robots[*].joints_init before the pivot test.",
    )
    args = parser.parse_args()

    axes = parse_axes(args.axes)
    angle_deg = abs(float(args.angle_deg))
    if angle_deg <= 0.0 or angle_deg > 90.0:
        raise ValueError("--angle-deg must be in (0, 20]")
    if args.duration < 0.5:
        raise ValueError("--duration must be at least 0.5 seconds")
    if args.settle < 0.0:
        raise ValueError("--settle must be non-negative")

    inference_config_path = Path(args.config).expanduser().resolve()
    inference_cfg = load_yaml_mapping(inference_config_path)
    robot_config_value = args.robot_config or inference_cfg.get("robot_config")
    if not robot_config_value:
        raise RuntimeError(
            f"No robot_config in {inference_config_path}; pass --robot-config"
        )
    robot_config_path = Path(robot_config_value).expanduser()
    if not robot_config_path.is_absolute():
        robot_config_path = inference_config_path.parent / robot_config_path
    robot_config_path = robot_config_path.resolve()
    rc = load_franka_config(robot_config_path, args.robot_idx)
    from umi.real_world.franka_interpolation_controller import _parse_tx_flange_tip

    tx_flange_tip = _parse_tx_flange_tip(rc.get("tx_flange_tip"))
    tx_tip_flange = np.linalg.inv(tx_flange_tip)
    max_pos_speed = float(
        inference_cfg.get("max_pos_speed", rc.get("max_pos_speed", np.inf))
    )
    max_rot_speed = float(
        inference_cfg.get("max_rot_speed", rc.get("max_rot_speed", np.inf))
    )

    print(f"[inference_config] {inference_config_path}")
    print(f"[robot_config] {robot_config_path}")
    print(
        f"[robot] idx={args.robot_idx} ip={rc['robot_ip']} "
        f"port={rc.get('robot_port', 4242)}"
    )
    print(f"[pose_api] {rc.get('pose_api', 'rpy')!r}")
    print(f"[tx_flange_tip] {rc.get('tx_flange_tip')}")
    print(f"[axes] {axes}")
    print(f"[angle] +/-{angle_deg:.1f} deg")
    print(f"[duration] {args.duration:.2f}s per leg")
    print(f"[speed_limits] position={max_pos_speed} m/s rotation={max_rot_speed} rad/s")

    if rc.get("pose_api", "rpy") != "rotvec":
        raise RuntimeError(
            "This test requires pose_api='rotvec'; otherwise tx_flange_tip is bypassed."
        )

    if not args.execute:
        print("[print-only] No robot connection and no motion.")
        print("Add --execute after checking the configuration and clearing the workspace.")
        return
    if not confirm_connection(assume_yes=args.yes):
        print("[abort] No robot connection and no motion.")
        return

    angle_rad = np.deg2rad(angle_deg)
    with SharedMemoryManager() as shm_manager:
        with make_controller(
            shm_manager,
            rc,
            init_joints=bool(args.init_joints),
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
        ) as controller:
            time.sleep(0.5)
            anchor_pose = latest_tcp_pose(controller)

            # Start/warm the same impedance path without changing the target pose.
            schedule_and_wait(controller, anchor_pose, duration_s=0.3, settle_s=0.5)
            anchor_pose = latest_tcp_pose(controller)
            anchor_xyz = anchor_pose[:3].copy()

            print("\n[anchor]")
            print_target("center", anchor_pose, tx_tip_flange, anchor_xyz)
            print(
                "Observe the real dexterous-hand wrist center, not only the "
                "controller-reported TCP. A wrong offset makes that real point draw an arc."
            )

            try:
                for axis in axes:
                    if not confirm_axis(axis, angle_deg, assume_yes=args.yes):
                        print(f"[skip] axis={axis}")
                        continue

                    positive = build_local_rotation_target(
                        anchor_pose, axis, +angle_rad
                    )
                    negative = build_local_rotation_target(
                        anchor_pose, axis, -angle_rad
                    )

                    print(f"\n[axis={axis}] sequence: +angle -> center -> -angle -> center")
                    execute_target(
                        controller,
                        f"{axis} +{angle_deg:.1f}deg",
                        positive,
                        tx_tip_flange,
                        anchor_xyz,
                        args.duration,
                        args.settle,
                    )
                    execute_target(
                        controller,
                        "center",
                        anchor_pose,
                        tx_tip_flange,
                        anchor_xyz,
                        args.duration,
                        args.settle,
                    )
                    execute_target(
                        controller,
                        f"{axis} -{angle_deg:.1f}deg",
                        negative,
                        tx_tip_flange,
                        anchor_xyz,
                        args.duration,
                        args.settle,
                    )
                    execute_target(
                        controller,
                        "center",
                        anchor_pose,
                        tx_tip_flange,
                        anchor_xyz,
                        args.duration,
                        args.settle,
                    )
            except KeyboardInterrupt:
                print(
                    "\n[interrupt] No new waypoint will be scheduled. "
                    "The controller context will stop; verify the robot state before continuing."
                )
                raise

    print("[done] Pivot test completed and returned to the anchor TCP pose.")


if __name__ == "__main__":
    main()
