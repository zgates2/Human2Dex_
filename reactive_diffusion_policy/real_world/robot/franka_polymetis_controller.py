import os
import pickle
import time
import threading
import queue as _queue
import numpy as np
import zerorpc
import scipy.spatial.transform as st
from typing import List, Dict, Optional
from loguru import logger

import sys
sys.path.append("/home/ps/reactive_diffusion_policy")

from reactive_diffusion_policy.real_world.robot.gripper_process_worker import GripperProcessProxy
from reactive_diffusion_policy.real_world.robot.later_force_publisher import ForceSensorAsync


class FrankaPolymetisController:
    """Franka controller via Polymetis zerorpc server on nuc2."""

    def __init__(
        self,
        server_ip: str = "172.16.13.170",
        server_port: int = 4242,
        gripper_config: Dict = {
            "mode": "single",
            "port1": "/dev/ttyUSB2",
            "motor1_id": 1,
        },
        force_sensor_port: str = "/dev/ttyUSB0",
    ):
        self.server_addr = f"tcp://{server_ip}:{server_port}"

        # zerorpc uses gevent internally — must keep all rpc calls
        # in a single dedicated thread that owns the gevent Hub.
        self._rpc_queue = _queue.Queue()
        self._rpc_ready = threading.Event()
        self._rpc_running = True
        self._rpc_thread = threading.Thread(target=self._rpc_worker, daemon=True)
        self._rpc_thread.start()
        self._rpc_ready.wait()
        logger.info(f"Connected to Polymetis server at {self.server_addr}")

        # default impedance gains
        self.Kx = np.array([600.0, 600.0, 600.0, 150.0, 150.0, 150.0]) * 1.0
        self.Kxd = np.sqrt(self.Kx) * 1.0

        self._desired_pose_rpy: Optional[np.ndarray] = None
        self._impedance_active = False

        # ---- gravity compensation calibration ---- #
        self.cali_info = None
        self.cali_bias = None
        self.T_force_ee = None
        self.load_params()

        # ---- gripper ---- #
        self.gripper_controller = GripperProcessProxy(config=gripper_config)
        self.gripper_controller.start()

        # ---- force sensor ---- #
        self.force_sensor = ForceSensorAsync(
            port=force_sensor_port,
            baud_rate=1000000,
            sensor_name="force_sensor",
            allow_repeat_frames=True,
            debug=False,
        )
        self.force_sensor.connect()

    # ------------------------------------------------------------------ #
    #  zerorpc thread-safe dispatcher
    # ------------------------------------------------------------------ #
    def _rpc_worker(self):
        """Dedicated thread that owns the zerorpc client + gevent Hub."""
        import gevent
        self._client = zerorpc.Client()
        self._client.connect(self.server_addr)
        self._rpc_ready.set()

        while self._rpc_running:
            try:
                method_name, args, kwargs, result_holder, done_event = \
                    self._rpc_queue.get_nowait()
                try:
                    result_holder['result'] = getattr(self._client, method_name)(*args, **kwargs)
                except Exception as e:
                    result_holder['error'] = e
                done_event.set()
            except _queue.Empty:
                gevent.sleep(0.001)

        self._client.close()

    def _rpc_call(self, method_name, *args, **kwargs):
        """Thread-safe wrapper: dispatches any zerorpc call to the worker."""
        result_holder = {}
        done_event = threading.Event()
        self._rpc_queue.put((method_name, args, kwargs, result_holder, done_event))
        done_event.wait()
        if 'error' in result_holder:
            raise result_holder['error']
        return result_holder['result']

    # ------------------------------------------------------------------ #
    #  reset_to_home
    # ------------------------------------------------------------------ #
    def reset_to_home(self, joint_angles: List[float], time_to_go: float = 4.0) -> None:
        if self._impedance_active:
            self._stop_impedance()
        self._rpc_call('move_to_joint_positions', joint_angles, time_to_go)
        self._desired_pose_rpy = None
        logger.info("Robot reset to home")

    # ------------------------------------------------------------------ #
    #  get_current_robot_states
    # ------------------------------------------------------------------ #
    def get_current_robot_states(self) -> Dict[str, List[float]]:
        tcp_pose_rpy = np.array(self._rpc_call('get_ee_pose_rpy'))  # [x,y,z,r,p,y]
        tcp_vel = self._rpc_call('get_ee_vel')                       # [vx,vy,vz,wx,wy,wz]
        joint_pos = self._rpc_call('get_joint_positions')            # [q1..q7]

        # convert to 7D quaternion [x,y,z,qw,qx,qy,qz]
        xyz = tcp_pose_rpy[:3]
        R = st.Rotation.from_euler("xyz", tcp_pose_rpy[3:])
        quat_xyzw = R.as_quat()  # scipy returns [qx,qy,qz,qw]
        tcp_pose_7d = np.concatenate([xyz, [quat_xyzw[3]], quat_xyzw[:3]]).tolist()

        # target pose
        if self._desired_pose_rpy is not None:
            tgt_xyz = self._desired_pose_rpy[:3]
            R_tgt = st.Rotation.from_euler("xyz", self._desired_pose_rpy[3:])
            tgt_quat_xyzw = R_tgt.as_quat()
            tcp_target_7d = np.concatenate([tgt_xyz, [tgt_quat_xyzw[3]], tgt_quat_xyzw[:3]]).tolist()
        else:
            tcp_target_7d = tcp_pose_7d

        # force sensor + gravity compensation
        rot_matrix = R.as_matrix()
        original_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            raw = self.force_sensor.get_wrench()
            if raw is not None and len(raw) == 6:
                original_wrench = raw
        except Exception:
            pass

        ee_offset = self.T_force_ee if self.T_force_ee is not None else np.array([[0], [0], [0.1865]])
        f_corrected, t_ee = self.gravity_compensation(np.array(original_wrench), rot_matrix, ee_offset)
        if f_corrected is not None:
            tcp_wrench = f_corrected.flatten().tolist() + t_ee.flatten().tolist()
        else:
            tcp_wrench = original_wrench

        # gripper
        gripper_state = self.get_current_gripper_states()

        return {
            "leftRobotTCP": tcp_pose_7d,
            "leftRobotTCPVel": tcp_vel,
            "leftRobotTCPWrench": tcp_wrench,
            "leftGripperState": gripper_state,
            "leftRobotTCPTarget": tcp_target_7d,
            "leftJoinStates": joint_pos,
        }

    # ------------------------------------------------------------------ #
    #  get_control_error_xyzrpy
    # ------------------------------------------------------------------ #
    def get_control_error_xyzrpy(self) -> Dict[str, List[float]]:
        current = np.array(self._rpc_call('get_ee_pose_rpy'))
        cur_xyz_mm = (current[:3] * 1000.0).tolist()
        cur_rpy_deg = np.degrees(current[3:]).tolist()

        if self._desired_pose_rpy is None:
            return {
                "error_xyz_mm": [0.0, 0.0, 0.0],
                "error_rpy_deg": [0.0, 0.0, 0.0],
                "current_xyz_mm": cur_xyz_mm,
                "current_rpy_deg": cur_rpy_deg,
                "target_xyz_mm": [0.0, 0.0, 0.0],
                "target_rpy_deg": [0.0, 0.0, 0.0],
            }

        target = self._desired_pose_rpy
        err_xyz_mm = ((target[:3] - current[:3]) * 1000.0).tolist()

        R_cur = st.Rotation.from_euler("xyz", current[3:])
        R_tgt = st.Rotation.from_euler("xyz", target[3:])
        R_err = R_cur.inv() * R_tgt
        err_rpy_deg = np.degrees(R_err.as_euler("xyz")).tolist()

        tgt_xyz_mm = (target[:3] * 1000.0).tolist()
        tgt_rpy_deg = np.degrees(target[3:]).tolist()

        return {
            "error_xyz_mm": err_xyz_mm,
            "error_rpy_deg": err_rpy_deg,
            "current_xyz_mm": cur_xyz_mm,
            "current_rpy_deg": cur_rpy_deg,
            "target_xyz_mm": tgt_xyz_mm,
            "target_rpy_deg": tgt_rpy_deg,
        }

    # ------------------------------------------------------------------ #
    #  tcp_move
    # ------------------------------------------------------------------ #
    def tcp_move(self, goal_pose: List[float]) -> None:
        """Move end-effector to goal_pose [x, y, z, r, p, y] (meters, radians)."""
        if not self._impedance_active:
            self._start_impedance()
        self._desired_pose_rpy = np.array(goal_pose)
        self._rpc_call('update_desired_ee_pose_rpy', goal_pose)

    # ------------------------------------------------------------------ #
    #  gripper
    # ------------------------------------------------------------------ #
    def get_current_gripper_states(self):
        return self.gripper_controller.get_current_gripper_states()

    def get_current_gripper_force(self):
        return self.gripper_controller.get_current_gripper_force()

    def get_current_gripper_width(self):
        return self.gripper_controller.get_current_gripper_width()

    # ------------------------------------------------------------------ #
    #  gravity compensation
    # ------------------------------------------------------------------ #
    def gravity_compensation(self, raw_wrench_array, T_map_force, T_force_ee):
        if self.cali_info is None or self.cali_bias is None:
            logger.debug("Calibration params not loaded, skipping gravity compensation")
            return None, None

        f_raw = raw_wrench_array[:3].reshape(3, 1)
        t_raw = raw_wrench_array[3:].reshape(3, 1)

        f_bias = self.cali_bias[3:6].reshape(3, 1)
        t_bias = self.cali_info[3:6].reshape(3, 1)

        r_cg = self.cali_info[0:3].reshape(3, 1)

        g_vec = np.array([[0], [0], [-9.81]])
        m = np.linalg.norm(self.cali_bias[0:3]) / 9.81

        g_sensor = T_map_force[:3, :3].T @ g_vec

        f_gravity = m * g_sensor
        t_gravity = m * np.cross(r_cg.flatten(), g_sensor.flatten()).reshape(3, 1)

        f_corrected = f_raw - f_bias - f_gravity
        t_corrected = t_raw - t_bias - t_gravity

        t_ee = t_corrected + np.cross(T_force_ee.flatten(), f_corrected.flatten()).reshape(3, 1)

        return f_corrected, t_ee

    def load_params(self, filename="cali_params.pkl"):
        try:
            base_path = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join('.', filename),
                os.path.join(base_path, filename),
                os.path.join(base_path, '..', filename),
            ]

            param_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    param_path = p
                    break

            if param_path is None:
                logger.info(f"Calibration file '{filename}' not found.")
                return

            with open(param_path, 'rb') as f:
                params = pickle.load(f)
            self.cali_info = params['cali_info']
            self.cali_bias = params['cali_bias']
            self.T_force_ee = np.array([[0], [0], [0.1865]])
            logger.info(f"Loaded calibration params from '{param_path}'")
        except Exception as e:
            logger.info(f"Failed to load calibration params: {e}")

    # ------------------------------------------------------------------ #
    #  impedance helpers
    # ------------------------------------------------------------------ #
    def _start_impedance(self) -> None:
        self._rpc_call('start_cartesian_impedance', self.Kx.tolist(), self.Kxd.tolist())
        time.sleep(0.5)  # wait for policy to register on server
        self._impedance_active = True
        logger.info("Cartesian impedance started")

    def _stop_impedance(self) -> None:
        if self._impedance_active:
            try:
                self._rpc_call('terminate_current_policy')
            except Exception as e:
                logger.warning(f"terminate_current_policy failed: {e}")
            self._impedance_active = False
            logger.info("Cartesian impedance stopped")

    # ------------------------------------------------------------------ #
    #  cleanup
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._stop_impedance()
        if self.gripper_controller:
            self.gripper_controller.stop()
        self._rpc_running = False
        self._rpc_thread.join(timeout=2)
        logger.info("Connection closed")


if __name__ == "__main__":
    ctrl = FrankaPolymetisController()

    home_joints = [0.24886237143896478, -0.10823570083692469, -0.2110291881155267,
                   -2.6155513998496334, -0.17867161943735246, 3.024342456234826, 0.3109925351142883]
    print("Resetting to home...")
    ctrl.reset_to_home(home_joints)
    print("Reset done.")

    dt = 1.0 / 10.0
    try:
        while True:
            t0 = time.time()
            states = ctrl.get_current_robot_states()
            print("=" * 60)
            for k, v in states.items():
                print(f"  {k}: {v}")
            elapsed = time.time() - t0
            print(f"  query time: {elapsed*1000:.1f}ms")
            time.sleep(max(0, dt - elapsed))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ctrl.close()
