"""
Temporal Ensemble.
"""

import torch
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R, Slerp
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix


class EnsembleBuffer:
    """
    Temporal ensemble buffer.
    """
    def __init__(self,
                 ensemble_mode = "new",
                 k: float = 0.01,
                 tau: float = 0.9,
                 **kwargs):
        assert ensemble_mode in ["new", "old", "avg", "act", "hato"], f"Ensemble mode {ensemble_mode} not supported now."
        self.mode = ensemble_mode
        self.timestep = 0
        self.actions_start_timestep = 0
        self.actions = deque([])
        self.actions_timestep = deque([])
        self.action_shape = None
        if ensemble_mode == "act":
            self.k = kwargs.get("k", k)
        if ensemble_mode == "hato":
            self.tau = kwargs.get("tau", tau)

    def clear(self):
        """
        Clear the ensemble buffer.
        """
        self.timestep = 0
        self.actions_start_timestep = 0
        self.actions = deque([])
        self.actions_timestep = deque([])
    
    def get_remaining_actions_count(self):
        """
        获取剩余动作数量
        
        Returns:
            int: 剩余动作数量
        """
        remaining_timesteps = len(self.actions) - (self.timestep - self.actions_start_timestep)
        return max(0, remaining_timesteps)
    
    def get_buffer_status(self):
        """
        获取buffer状态信息
        
        Returns:
            dict: 包含buffer状态的字典
        """
        return {
            "total_actions": len(self.actions),
            "timestep": self.timestep,
            "actions_start_timestep": self.actions_start_timestep,
            "remaining_actions": self.get_remaining_actions_count(),
            "consumed_actions": self.timestep - self.actions_start_timestep
        }

    def add_action(self, action, timestep):
        """
        Add action to the ensemble buffer:

        Parameters:
        - action: horizon x action_dim (...);
        - timestep: action[0]'s timestep.
        """
        action = np.array(action)
        if self.action_shape == None:
            self.action_shape = action.shape[1:]
            assert len(self.action_shape) == 1, "Only support action with 1D shape."
        else:
            assert self.action_shape == action.shape[1:], "Incompatible action shape."
        idx = timestep - self.actions_start_timestep
        horizon = action.shape[0]
        while idx + horizon - 1 >= len(self.actions):
            self.actions.append([])
            self.actions_timestep.append([])
        for i in range(idx, idx + horizon):
            self.actions[i].append(action[i - idx, ...]) # 啥意思？  # 把局部的aciton推到全局action中i-idx第一次为0,self.action是全局，action是外部00
            self.actions_timestep[i].append(timestep)   # 时间步
    
    def get_action(self):
        """
        Get ensembled action from buffer.
        """
        if self.timestep - self.actions_start_timestep >= len(self.actions):
            return None      # no data
        while self.actions_start_timestep < self.timestep:
            self.actions.popleft()
            self.actions_timestep.popleft()
            self.actions_start_timestep += 1
        actions = self.actions[0]
        actions_timestep = self.actions_timestep[0]
        if actions == []:
            return None      # no data
        sorted_actions = sorted(zip(actions_timestep, actions))
        all_actions = np.array([x for _, x in sorted_actions])
        all_timesteps = np.array([t for t, _ in sorted_actions])
        if self.mode == "new":
            action = all_actions[-1]
        elif self.mode == "old":
            action = all_actions[0]
        elif self.mode == "avg":
            weights = np.ones_like(all_timesteps)
            weights = weights / weights.sum()
            action = self._weighted_average_action(all_actions, weights)
        elif self.mode == "act":
            weights = np.exp(-self.k * (self.timestep - all_timesteps))
            weights = weights / weights.sum()
            action = self._weighted_average_action(all_actions, weights)
        elif self.mode == "hato":
            weights = self.tau ** (self.timestep - all_timesteps)
            weights = weights / weights.sum()
            action = self._weighted_average_action(all_actions, weights)
        else:
            raise AttributeError("Ensemble mode {} not supported.".format(self.mode))
        self.timestep += 1
        return action

    def _weighted_average_action(self, actions, weights):
        """
        Weighted average action.
        """
        D = actions.shape[-1]

        if D == 1: # gripper
            cartesian_dim_list = [np.arange(1)]
            rotation_dim_list = None
        elif D == 2: # gripper
            cartesian_dim_list = [np.arange(2)]
            rotation_dim_list = None
        elif D == 3:
            cartesian_dim_list = [np.arange(3)]
            rotation_dim_list = None
        elif D == 6:
            cartesian_dim_list = [np.arange(3), np.arange(3, 6)]
            rotation_dim_list = None
        elif D == 9:
            cartesian_dim_list = [np.arange(3)]
            rotation_dim_list = [np.arange(3, 9)]
        elif D == 18:
            cartesian_dim_list = [np.arange(3), np.arange(9, 12)]
            rotation_dim_list = [np.arange(3, 9), np.arange(12, 18)]
        else:
            raise NotImplementedError

        avg_action = np.zeros_like(actions[0])
        # Cartesian
        for cartesian_dim in cartesian_dim_list:
            avg_action[cartesian_dim] = (actions[:, cartesian_dim] * weights.reshape(-1, 1)).sum(axis=0)

        # Rotation
        if rotation_dim_list is not None:
            for rotation_dim in rotation_dim_list:
                assert len(rotation_dim) == 6, "Only support 6D rotation now"
                avg_rotation = R.from_matrix(ortho6d_to_rotation_matrix(actions[0:, rotation_dim])[0]).as_quat()
                cumulative_weight = weights[0]
                for i in range(1, len(weights)):
                    start_rotation = avg_rotation
                    end_rotation = R.from_matrix(ortho6d_to_rotation_matrix(actions[i:, rotation_dim])[0]).as_quat()
                    slerp = Slerp([0, 1], R.from_quat([start_rotation, end_rotation]))
                    avg_rotation = slerp(cumulative_weight / (cumulative_weight + weights[i])).as_quat()
                    cumulative_weight += weights[i]
                avg_action[rotation_dim] = R.from_quat(avg_rotation).as_matrix()[:3, :2].T.flatten()

        return avg_action