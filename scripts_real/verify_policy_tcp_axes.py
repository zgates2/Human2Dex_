"""Verify policy/TCP or pre-rotation flange +x/+y/+z directions on a Franka.

This script reads the Franka connection and ``tx_flange_tip`` settings from
``example/eval_robots_config.yaml`` by default, starts the same
``FrankaInterpolationController`` path used by real inference, and commands
small translations along either the current policy/TCP frame axes or the
pre-``tx_flange_tip`` Franka flange frame axes.

Run on the robot machine, with the arm in a clear workspace:

    python scripts_real/verify_policy_tcp_axes.py --dry-run
    python scripts_real/verify_policy_tcp_axes.py --axes xyz --distance 0.01
    python scripts_real/verify_policy_tcp_axes.py --frame flange --axes xyz --distance 0.005

For each selected axis, the script prints the current TCP pose, the local axis
vector expressed in robot-base coordinates, then waits for Enter before moving.
By default it returns to the starting pose after each axis so the three tests
are easy to compare visually.
"""

import argparse
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.common.pose_util import pose_to_mat


DEFAULT_CONFIG_REL = "example/eval_robots_config.yaml"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
MOVE_FRAMES = ("policy", "flange")


def load_franka_config(config_path: Path, robot_idx: int) -> dict:
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    robots = cfg.get("robots", [])
    if robot_idx >= len(robots):
        raise RuntimeError(f"robot_idx={robot_idx} out of range for {config_path}")
    rc = robots[robot_idx]
    if not str(rc.get("robot_type", "")).startswith("franka"):
        raise RuntimeError(
            f"robots[{robot_idx}] is not franka: {rc.get('robot_type')!r}"
        )
    return rc


def make_controller(shm_manager, rc: dict, init_joints: bool):
    from umi.real_world.franka_interpolation_controller import FrankaInterpolationController

    joints_init = rc.get("joints_init") if init_joints else None
    return FrankaInterpolationController(
        shm_manager=shm_manager,
        robot_ip=rc["robot_ip"],
        robot_port=rc.get("robot_port", 4242),
        frequency=rc.get("frequency", 200),
        Kx_scale=rc.get("Kx_scale", 1.0),
        Kxd_scale=rc.get("Kxd_scale", np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0])),
        launch_timeout=rc.get("launch_timeout", 20),
        pose_api=rc.get("pose_api", "rpy"),
        rpc_timeout=rc.get("rpc_timeout", 5.0),
        read_only=False,
        start_impedance_on_start=rc.get("start_impedance_on_start", False),
        impedance_start_delay=rc.get("impedance_start_delay", 0.5),
        joints_init=joints_init,
        joints_init_duration=4,
        verbose=rc.get("verbose", False),
        max_pos_speed=rc.get("max_pos_speed", np.inf),
        max_rot_speed=rc.get("max_rot_speed", np.inf),
        max_commands_per_cycle=rc.get("max_commands_per_cycle", 1),
        tx_flange_tip=rc.get("tx_flange_tip", None),
        receive_latency=rc.get("robot_obs_latency", 0.0),
    )


def latest_pose(controller) -> np.ndarray:
    return np.asarray(controller.get_state()["ActualTCPPose"], dtype=np.float64).copy()


def wait_for_enter(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(prompt)
        return True
    reply = input(f"{prompt} Press Enter to continue, or type q then Enter to skip: ")
    return reply.strip().lower() not in {"q", "quit", "skip", "n", "no"}


def schedule_and_wait(controller, pose: np.ndarray, duration_s: float, settle_s: float):
    target_time = time.time() + 0.1
    controller.schedule_waypoint(pose, target_time + duration_s)
    time.sleep(0.1 + duration_s + settle_s)


def format_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.5f}" for v in vec) + "]"


def frame_mat_from_tip_pose(
        tip_pose: np.ndarray,
        tx_flange_tip: np.ndarray,
        move_frame: str) -> np.ndarray:
    """Return the selected local frame pose expressed in robot-base coordinates."""
    tip_mat = pose_to_mat(tip_pose)
    if move_frame == "policy":
        return tip_mat
    if move_frame == "flange":
        return tip_mat @ np.linalg.inv(tx_flange_tip)
    raise ValueError(f"Unsupported move frame: {move_frame!r}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Move along positive xyz axes of either the policy/TCP frame or "
            "the pre-rotation Franka flange frame."
        )
    )
    parser.add_argument(
        "-rc",
        "--robot-config",
        default=str(ROOT_DIR / DEFAULT_CONFIG_REL),
        help="Path to eval_robots_config.yaml.",
    )
    parser.add_argument("--robot-idx", type=int, default=0, help="robots[] index.")
    parser.add_argument(
        "--axes",
        default="xyz",
        help="Axes to test, e.g. x, yz, or xyz. Only positive directions are used.",
    )
    parser.add_argument(
        "--frame",
        "--move-frame",
        dest="move_frame",
        choices=MOVE_FRAMES,
        default="policy",
        help=(
            "Local frame used for movement: 'policy' is the rotated TCP frame "
            "used by inference; 'flange' is the frame before tx_flange_tip rotation."
        ),
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.01,
        help="Translation distance in meters for each local +axis movement.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Move duration in seconds for each waypoint.",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.5,
        help="Extra wait time after each waypoint.",
    )
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="Do not return to the starting pose after each axis.",
    )
    parser.add_argument(
        "-j",
        "--init-joints",
        action="store_true",
        help="Move to robots[*].joints_init before testing.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Do not ask for Enter before each movement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned poses without starting the controller or moving.",
    )
    args = parser.parse_args()

    axes = []
    for axis in args.axes.lower():
        if axis not in AXIS_INDEX:
            raise ValueError(f"Unsupported axis {axis!r}; use some combination of x, y, z")
        if axis not in axes:
            axes.append(axis)

    if args.distance <= 0:
        raise ValueError("--distance must be positive")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    config_path = Path(args.robot_config).expanduser()
    rc = load_franka_config(config_path, args.robot_idx)
    from umi.real_world.franka_interpolation_controller import _parse_tx_flange_tip

    tx_flange_tip = _parse_tx_flange_tip(rc.get("tx_flange_tip", None))
    print(f"[config] {config_path}")
    print(f"[robot] idx={args.robot_idx} ip={rc['robot_ip']} port={rc.get('robot_port', 4242)}")
    print(f"[pose_api] {rc.get('pose_api', 'rpy')!r}")
    print(f"[tx_flange_tip] {rc.get('tx_flange_tip', None)}")
    print(f"[move_frame] {args.move_frame!r}")
    if rc.get("pose_api", "rpy") != "rotvec":
        print("[warn] pose_api is not 'rotvec'; tx_flange_tip is bypassed in this code path.")
        if args.move_frame == "flange":
            raise ValueError("--frame flange requires pose_api='rotvec'")

    if args.dry_run:
        print("[dry-run] Not connecting to robot. Run without --dry-run to execute motions.")
        return

    with SharedMemoryManager() as shm_manager:
        with make_controller(shm_manager, rc, init_joints=args.init_joints) as controller:
            time.sleep(0.5)
            current_pose = latest_pose(controller)

            # Warm up lazy impedance without changing the target pose.
            schedule_and_wait(controller, current_pose, duration_s=0.2, settle_s=0.5)
            current_pose = latest_pose(controller)

            for axis in axes:
                start_pose = current_pose.copy()
                move_frame_mat = frame_mat_from_tip_pose(
                    tip_pose=start_pose,
                    tx_flange_tip=tx_flange_tip,
                    move_frame=args.move_frame,
                )
                axis_idx = AXIS_INDEX[axis]
                axis_in_base = move_frame_mat[:3, :3][:, axis_idx]
                target_pose = start_pose.copy()
                target_pose[:3] = start_pose[:3] + axis_in_base * args.distance

                print("")
                print(f"[frame={args.move_frame} axis=+{axis}]")
                print(f"  start_tip_xyz       {format_vec(start_pose[:3])}")
                if args.move_frame == "flange":
                    print(f"  start_flange_xyz    {format_vec(move_frame_mat[:3, 3])}")
                print(f"  {args.move_frame} +{axis} in base {format_vec(axis_in_base)}")
                print(f"  target_tip_xyz      {format_vec(target_pose[:3])}")
                if not wait_for_enter(
                    f"Move {args.distance:.4f} m along {args.move_frame} +{axis}.",
                    assume_yes=args.yes,
                ):
                    print(f"  skipped +{axis}")
                    continue

                schedule_and_wait(controller, target_pose, args.duration, args.settle)
                reached_pose = latest_pose(controller)
                print(f"  reached_xyz     {format_vec(reached_pose[:3])}")

                if not args.no_return:
                    if wait_for_enter(
                        "Return to the axis start pose.",
                        assume_yes=args.yes,
                    ):
                        schedule_and_wait(controller, start_pose, args.duration, args.settle)
                        current_pose = latest_pose(controller)
                        print(f"  returned_xyz    {format_vec(current_pose[:3])}")
                    else:
                        current_pose = reached_pose
                else:
                    current_pose = reached_pose

    print("[done]")


if __name__ == "__main__":
    main()
