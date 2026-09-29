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

import transformations as tft
Rz_45deg = tft.rotation_matrix(np.pi/4, (0, 0, 1))
def _6dpose_Rz_45deg(pose: np.ndarray) -> np.ndarray:
    tmat = pose_6d_to_4x4matrix(pose)
    tmat_Rz_45deg = tmat @ Rz_45deg
    _6dpose_Rz_45deg = matrix4x4_to_pose_6d(tmat_Rz_45deg)
    return _6dpose_Rz_45deg

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
    trajectory_se3 = [pose_6d_to_4x4matrix(p) for p in pkl_poses_6d] # 没有变化
    trajectory_se3 = [pose_6d_to_4x4matrix(p)@ Rz_45deg for p in pkl_poses_6d] # Rz 45deg
    relative_traj = []
    delta_6dpose = np.zeros((len(trajectory_se3), 6))
    for i in range(len(trajectory_se3)):
        print(('i: ', i))
        relative_traj.append(np.linalg.inv(trajectory_se3[0]) @ trajectory_se3[i])
        print(f"relative_traj: {matrix4x4_to_pose_6d(relative_traj[i])}")
        delta_6dpose[i,:] = matrix4x4_to_pose_6d(relative_traj[i])
        # import pdb; pdb.set_trace()
        
    # plot the relative trajectory
    import matplotlib.pyplot as plt
    
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Delta 6D pose trajectory', fontsize=16)
    
    # 维度标签
    labels = ['X (m)', 'Y (m)', 'Z (m)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    
    # 绘制每个维度
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        ax.plot(delta_6dpose[:, i], color=colors[i], linewidth=2, label=labels[i])
        ax.set_title(f'{labels[i]}', fontsize=12)
        ax.set_xlabel('采样点', fontsize=10)
        ax.set_ylabel('数值', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    print("正在等待机器人状态稳定 (固定延时)...")
    time.sleep(2.0)
    
    robot_start_pose_mat = np.array(controller.robot_interface._state_buffer[-1].O_T_EE).reshape(4, 4).transpose()
    # current_target_pose = SE3(robot_start_pose_mat, check=False)
    
    # print(f"机器人起始位置 (xyz): {current_target_pose.t}")
    print("将从此位置开始应用相对轨迹。")
    
    print(f"准备回放 {len(trajectory_se3) - 1} 个相对运动...")
    
    last_time = time.time()
    for i in range(3, len(trajectory_se3)):
        relative_traj = np.linalg.inv(trajectory_se3[3]) @ trajectory_se3[i]
        print(f"relative_traj: {matrix4x4_to_pose_6d(relative_traj)}")
        # import pdb; pdb.set_trace()
        target_pose = robot_start_pose_mat @ relative_traj
        current_target_pose = SE3(target_pose, check=False)

        # current_target_pose = target_pose
        
        theta, v = tr2angvec(current_target_pose.R)
        axisangle = v * theta
        target_pose_7d = current_target_pose.t.tolist() + axisangle.flatten().tolist() + [-1.0]
        
        controller.set_target_pose(target_pose_7d)
        
        # time.sleep(1.0 / controller.control_frequency)
        time.sleep(1/30.0)
        
        current_time = time.time()
        print(f"发送 Delta {i}/{len(trajectory_se3)-1} | 频率: {1/(current_time - last_time):.2f} Hz")
        last_time = current_time

    print("-------------------------------- 相对轨迹回放完成 --------------------------------")
    time.sleep(5.0)
    

if __name__ == "__main__":
    FORCE_SENSOR_PORT = '/dev/ttyUSB0'
    # PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0926_less/episode0030.pkl"
    # PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0926/episode0025.pkl"
    # PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0925/episode0029.pkl"
    # PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0922/episode0098.pkl"
    # PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0926_less/episode0006.pkl"
    # PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0928/episode0013.pkl"
    # PKL_FILE_TO_REPLAY = "/mnt/wd/wipe/data/dataset_full/pickplace0928/episode0015.pkl"
    PKL_FILE_TO_REPLAY = "/mnt/storage/1201pickplace_mvs/episode0002.pkl"

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