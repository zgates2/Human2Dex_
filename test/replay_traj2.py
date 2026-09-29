import sys
sys.path.append("/home/ps/reactive_diffusion_policy")
import time
import numpy as np
import pickle
from pathlib import Path

from spatialmath import SE3
from spatialmath.base import *

from reactive_diffusion_policy.real_world.robot.single_flexiv_controller import FlexivController
from reactive_diffusion_policy.common.space_utils import pose_6d_to_pose_7d, matrix4x4_to_pose_6d, pose_6d_to_4x4matrix

def load_and_extract_poses_from_pkl(pkl_path: Path) -> np.ndarray:
    """从 .pkl 文件加载数据并提取 TCP 位姿轨迹。"""
    print(f"正在从 '{pkl_path}' 加载轨迹数据...")
    if not pkl_path.exists():
        raise FileNotFoundError(f"错误: PKL 文件不存在: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    
    try:
        poses = np.array([frame.leftRobotTCP for frame in all_data.sensorMessages])
        print(f"成功加载 {len(poses)} 个位姿点。")
        return poses
    except AttributeError:
        raise ValueError("错误: .pkl 文件的结构不符合预期。无法找到 'sensorMessages' 或 'leftRobotTCP'。")


def replay_relative_from_pkl(controller: FlexivController, pkl_file_path: str):
    """
    从 .pkl 计算每一步的相对运动(delta)，并将其应用到机器人上进行回放。
    """
    pkl_poses_6d = load_and_extract_poses_from_pkl(Path(pkl_file_path))
    if len(pkl_poses_6d) < 2:
        print("轨迹点不足2个，无法计算delta。")
        return
    
    # trajectory_se3 = [SE3.Trans(p[:3]) * SE3.RPY(p[3:], order='xyz') for p in pkl_poses_6d]
    trajectory_se3 = [pose_6d_to_4x4matrix(p) for p in pkl_poses_6d]


    robot_start_pose_mat = np.array(controller.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
    current_target_pose = SE3(robot_start_pose_mat, check=False)
    
    print(f"机器人起始位置 (xyz): {current_target_pose.t}")
    print("将从此位置开始应用相对轨迹。")
    
    print(f"准备回放 {len(trajectory_se3) - 1} 个相对运动...")
    
    last_time = time.time()
    for i in range(1, len(trajectory_se3)):
        prev_pkl_pose = trajectory_se3[i-1]
        curr_pkl_pose = trajectory_se3[i]
        # delta_pkl = curr_pkl_pose * prev_pkl_pose.inv()
        # delta_pkl = prev_pkl_pose.inv() * curr_pkl_pose
        delta_pkl = np.linalg.inv(prev_pkl_pose) * curr_pkl_pose
        delta = SE3(delta_pkl, check=False)

        current_target_pose = current_target_pose * delta
        
        # current_target_pose = SE3(current_target_pose, check=False)
        theta, v = tr2angvec(current_target_pose.R)
        axisangle = v * theta
        target_pose_7d = current_target_pose.t.tolist() + axisangle.flatten().tolist() + [-1.0]
        
        controller.set_target_pose(target_pose_7d)
        
        time.sleep(1.0 / controller.control_frequency)
        
        current_time = time.time()
        print(f"发送 Delta {i}/{len(trajectory_se3)-1} | 频率: {1/(current_time - last_time):.2f} Hz")
        last_time = current_time

    print("-------------------------------- 相对轨迹回放完成 --------------------------------")
    time.sleep(5.0)
    

if __name__ == "__main__":
    FORCE_SENSOR_PORT = '/dev/ttyACM0'
    PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0926_less/episode0030.pkl"

    controller = FlexivController(
        force_sensor_port=FORCE_SENSOR_PORT
    )

    try:
        print("重置到初始位置...")
        controller.reset_to_home()
        print("到达初始位置，准备回放。")

        replay_relative_from_pkl(controller, PKL_FILE_TO_REPLAY)

    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        print("回放结束或出现错误，正在停止机器人...")
        if hasattr(controller, 'control_thread') and controller.control_thread is not None and controller.control_thread.is_alive():
             print("正在停止控制线程...")
             controller._stop_control_thread()
        print("程序已安全退出。")