"""Teleop interface - bring up Franka arm + LkMotor gripper without ROS.

Reads hardware config from ``example/eval_robots_config.yaml`` (or
``$UMI_ROBOT_CONFIG``) and starts both controllers so a separate process
(eval_real / umi_policy_client) can talk to the same shared memory.

Run with zero args:
    python teleop_interface.py
"""
import os
import signal
import sys
import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import psutil
import yaml

os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")
os.environ.setdefault("OMP_NUM_THREADS", "12")

import cv2
cv2.setNumThreads(12)

# Best-effort CPU affinity for the controllers (Linux only).
total_cores = psutil.cpu_count() or 1
_n_bind = min(8, total_cores)
_bind_set = set(range(total_cores - _n_bind, total_cores))
try:
    os.sched_setaffinity(0, _bind_set)
except Exception as e:
    print(f"[teleop] could not set CPU affinity: {e}")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.lk_gripper_proxy import LkGripperProxy


DEFAULT_CONFIG_REL = "example/eval_robots_config.yaml"


def _load_config():
    path = os.environ.get("UMI_ROBOT_CONFIG")
    if path is None:
        path = str(ROOT_DIR / DEFAULT_CONFIG_REL)
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg, path


class TeleopInterface:
    def __init__(self, robot_cfg: dict, gripper_cfg: dict, verbose: bool = True):
        self.robot_cfg = robot_cfg
        self.gripper_cfg = gripper_cfg
        self.verbose = verbose

        self.shm_manager = None
        self.robot = None
        self.gripper = None
        self._running = False

    def start(self):
        print("=" * 50)
        print("Starting Teleop Interface")
        print("=" * 50)

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        print(f"[teleop] starting Franka @ {self.robot_cfg['robot_ip']}:{self.robot_cfg.get('robot_port', 4242)}")
        self.robot = FrankaInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip=self.robot_cfg['robot_ip'],
            robot_port=self.robot_cfg.get('robot_port', 4242),
            frequency=self.robot_cfg.get('frequency', 200),
            Kx_scale=self.robot_cfg.get('Kx_scale', 1.0),
            Kxd_scale=self.robot_cfg.get('Kxd_scale', 2.0),
            verbose=self.verbose,
            receive_latency=self.robot_cfg.get('robot_obs_latency', 0.0),
        )
        self.robot.start(wait=True)

        proxy_cfg = {
            'mode': self.gripper_cfg.get('mode', 'single'),
            'port1': self.gripper_cfg['port1'],
            'motor1_id': self.gripper_cfg['motor1_id'],
        }
        for key in ['close_offset_rad', 'close_offset_active_below_rad']:
            if key in self.gripper_cfg:
                proxy_cfg[key] = self.gripper_cfg[key]
        if proxy_cfg['mode'] == 'dual':
            proxy_cfg['port2'] = self.gripper_cfg['port2']
            proxy_cfg['motor2_id'] = self.gripper_cfg['motor2_id']

        print(f"[teleop] starting LkGripperProxy ({proxy_cfg['mode']}) on {proxy_cfg['port1']}")
        self.gripper = LkGripperProxy(
            config=proxy_cfg,
            receive_latency=self.gripper_cfg.get('gripper_obs_latency', 0.0),
        )
        self.gripper.start(wait=True)

        self._running = True
        print("=" * 50)
        print("Teleop Interface ready")
        print("=" * 50)

    def stop(self):
        print("\n[teleop] stopping")
        self._running = False
        for thing, name in [(self.gripper, 'gripper'), (self.robot, 'robot')]:
            if thing is None:
                continue
            try:
                thing.stop(wait=True)
                print(f"[teleop] {name} stopped")
            except Exception as e:
                print(f"[teleop] error stopping {name}: {e}")
        if self.shm_manager is not None:
            try:
                self.shm_manager.shutdown()
            except Exception as e:
                print(f"[teleop] error shutting down shm_manager: {e}")

    def is_ready(self) -> bool:
        return (
            self.robot is not None and self.robot.is_ready
            and self.gripper is not None and self.gripper.is_ready
        )

    def get_robot_state(self):
        return self.robot.get_state() if self.robot is not None else None

    def get_gripper_state(self):
        return self.gripper.get_state() if self.gripper is not None else None

    def run(self):
        self.start()

        def _handler(sig, frame):
            print("\n[teleop] interrupt received")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

        try:
            print("\nTeleop Interface running. Press Ctrl+C to stop.")
            while self._running:
                if self.is_ready():
                    rs = self.get_robot_state()
                    if rs is not None:
                        pose = rs.get('ActualTCPPose')
                        if pose is not None:
                            print(
                                f"\r[robot] TCP xyz=({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f})",
                                end="",
                                flush=True,
                            )
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    cfg, cfg_path = _load_config()
    print(f"[teleop] config loaded from {cfg_path}")

    robots = cfg.get('robots') or []
    grippers = cfg.get('grippers') or []
    if not robots or not grippers:
        raise RuntimeError(f"{cfg_path}: missing 'robots' or 'grippers'")

    iface = TeleopInterface(
        robot_cfg=robots[0],
        gripper_cfg=grippers[0],
        verbose=True,
    )
    iface.run()


if __name__ == "__main__":
    main()
