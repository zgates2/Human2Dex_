# PYTHONPATH=/share/project/lsq/rdp/reactive_diffusion_policy:$PYTHONPATH /share/project/lsq/miniconda3/envs/rdp/bin/python /share/project/lsq/rdp/reactive_diffusion_policy/reactive_diffusion_policy/dataset/real_image_tactile_dataset.py
from typing import Dict
import torch
import numpy as np
import os
from threadpoolctl import threadpool_limits
import copy
import tqdm
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from reactive_diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_action_normalizer
)
from reactive_diffusion_policy.common.action_utils import absolute_actions_to_relative_actions, get_inter_gripper_actions
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms

class RealImageTactileDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 obs_temporal_downsample_ratio=1, # for latent diffusion
                 n_latency_steps=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 delta_action=False,
                 relative_action=False,
                 relative_tcp_obs_for_relative_action=True,
                 transform_params=None,
                 ):
        # import pdb;pdb.set_trace()
        assert os.path.isdir(dataset_path)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        action_meta = shape_meta['action']
        self.action_dim = action_meta['shape'][0]
        self.action_extra_meta = shape_meta.get('action_extra', dict())
        self.extra_action_keys = list(self.action_extra_meta.keys())
        self.action_extra_dim = sum(
            attr.get('shape', [0])[0] for attr in self.action_extra_meta.values()
        )

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key) # ['left_gripper1_marker_offset_emb']

        zarr_path = os.path.join(dataset_path, 'replay_buffer.zarr')
        zarr_load_keys = set(rgb_keys + lowdim_keys + extended_rgb_keys + extended_lowdim_keys + self.extra_action_keys + ['action'])
        zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=zarr_load_keys)
        
        # import pdb; pdb.set_trace()
        # replay_buffer.left_robot_tcp_wrench = np.zeros_like(replay_buffer['left_robot_tcp_wrench'])
        # self.replay_buffer = replay_buffer

        if delta_action: # lsq: False in yaml file
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        self.base_action_dim = replay_buffer['action'].shape[1]
        if self.base_action_dim + self.action_extra_dim != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: base {self.base_action_dim} + extra {self.action_extra_dim} != declared {self.action_dim}"
            )

        self.relative_action = relative_action # lsq: True in the yaml file
        self.relative_tcp_obs_for_relative_action = relative_tcp_obs_for_relative_action # lsq: True in the initialization of this class
        self.transforms = RealWorldTransforms(option=transform_params)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                if key not in extended_rgb_keys + extended_lowdim_keys:
                    key_first_k[key] = n_obs_steps * obs_temporal_downsample_ratio
        self.key_first_k = key_first_k

        # import pdb;pdb.set_trace()

        self.seed = seed
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # import pdb;pdb.set_trace()
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.obs_downsample_ratio = obs_temporal_downsample_ratio
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # calculate relative action / obs
        if "left_robot_wrt_right_robot_tcp_pose" in self.lowdim_keys or "right_robot_wrt_left_robot_tcp_pose" in self.lowdim_keys: # lsq: 目前不执行这段代码
            inter_gripper_data_dict = {key: list() for key in self.lowdim_keys if 'robot_tcp_pose' in key and 'wrt' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating inter-gripper relative obs for normalizer'):
                for key in inter_gripper_data_dict.keys():
                    inter_gripper_data_dict[key].append(data['obs'][key])
            inter_gripper_data_dict = dict_apply(inter_gripper_data_dict, np.stack)

        # import pdb; pdb.set_trace()
        if self.relative_action:
            relative_data_dict = {key: list() for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key}
            
            for data in tqdm.tqdm(self, leave=False, desc='Calculating relative action/obs for normalizer'):
                for key in relative_data_dict.keys():
                    if key == 'action':
                        relative_data_dict[key].append(data[key])
                    else:
                        relative_data_dict[key].append(data['obs'][key])
            relative_data_dict = dict_apply(relative_data_dict, np.stack)

        # action
        if self.relative_action:
            action_all = relative_data_dict['action']
            print('[Dataset] relative action_all shape for normalizer:', action_all.shape)
            
        else:
            action_all = self.replay_buffer['action'][:, :self.base_action_dim]
            
            if self.extra_action_keys:
                extra_arrays = [
                    self.replay_buffer[key][:, :self.action_extra_meta[key]['shape'][0]]
                    for key in self.extra_action_keys
                ]
                
                if len(extra_arrays) > 0:
                    action_all = np.concatenate([action_all] + extra_arrays, axis=-1)
            action_all = action_all.astype(np.float32)

        normalizer['action'] = get_action_normalizer(action_all)

        
        
        # obs
        for key in list(set(self.lowdim_keys)):
            if self.relative_action and key in relative_data_dict:
                normalizer[key] = get_action_normalizer(relative_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' in key:
                normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' not in key:
                normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                    self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])

        for key in list(set(self.extended_lowdim_keys)):
            if key in self.lowdim_keys:
                assert self.shape_meta['extended_obs'][key]['shape'][0] == self.shape_meta['obs'][key]['shape'][0], \
                    f"Extended obs {key} has different shape from obs {key}"
            else:
                if self.relative_action and key in relative_data_dict:
                    normalizer[key] = get_action_normalizer(relative_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' in key:
                    normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' not in key: # not used now
                    normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                else:
                    normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])

        # image
        for key in list(set(self.rgb_keys + self.extended_rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        # import pdb;pdb.set_trace()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = self.replay_buffer['action'][:, :self.base_action_dim]
        
        if self.extra_action_keys:
            extra_arrays = [
                self.replay_buffer[key][:, :self.action_extra_meta[key]['shape'][0]]
                for key in self.extra_action_keys
            ]
            if len(extra_arrays) > 0:
                actions = np.concatenate([actions] + extra_arrays, axis=-1)
        return torch.from_numpy(actions.astype(np.float32))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        # import pdb;pdb.set_trace()
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        obs_downsample_ratio = self.obs_downsample_ratio

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice][::-obs_downsample_ratio][::-1],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            if key not in self.rgb_keys:
                del data[key]
        for key in self.lowdim_keys:
            if 'wrt' not in key:
                obs_dict[key] = data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice][::-obs_downsample_ratio][::-1].astype(np.float32)
                # save ram
                if key not in self.extended_lowdim_keys:
                    del data[key]

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in ['left_robot_wrt_right_robot_tcp_pose', 'right_robot_wrt_left_robot_tcp_pose']:
            if key in obs_dict:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
        
        extended_obs_dict = dict()
        for key in self.extended_rgb_keys:
            extended_obs_dict[key] = np.moveaxis(data[key],-1,1
                ).astype(np.float32) / 255.
            del data[key]
        for key in self.extended_lowdim_keys:
            if 'wrt' not in key:
                extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                del data[key]

        action = data['action'][:, :self.base_action_dim].astype(np.float32)
        extra_action_list = list()
        for key in self.extra_action_keys:
            key_dim = self.action_extra_meta[key]['shape'][0]
            extra_values = data[key][:, :key_dim].astype(np.float32)
            extra_action_list.append(extra_values)
            if key not in self.lowdim_keys and key not in self.extended_lowdim_keys:
                del data[key]
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
            if extra_action_list:
                extra_action_list = [extra[self.n_latency_steps:] for extra in extra_action_list]
        # self.relative_action = False
        if self.relative_action:
            # import pdb; pdb.set_trace()
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs_dict else np.array([])
            ], axis=-1)
            # action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action) # lsq: base_absolute_action.shape = (2, *), action.shape = (T, *), 这里是 32，4
            if isinstance(base_absolute_action, np.ndarray) and base_absolute_action.size == 0:
                action = absolute_actions_to_relative_actions(action, base_absolute_action=None)
                # action = absolute_actions_to_relative_actions(action, base_absolute_action=action[0].copy())# action[0] = obs_dict['left_robot_tcp_pose'][1]
            else:
                action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)
            # import pdb; pdb.set_trace()
            if self.relative_tcp_obs_for_relative_action: # lsq: True in the initialization of this class
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key: # 由于没有 tcp pose，所以不执行下面操作
                        obs_dict[key]  = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        if extra_action_list:
            extra_action = np.concatenate(extra_action_list, axis=-1)
            action = np.concatenate([action, extra_action], axis=-1)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'extended_obs': dict_apply(extended_obs_dict, torch.from_numpy)
        }
        return torch_data


def test_init():
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with initialize('../config'):
        # cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
        #                     overrides=['task=real_peel_image_gelsight_emb_ldp_24fps'])
        cfg = hydra.compose('train_latent_diffusion_unet_real_image_workspace',
                            overrides=['task=real_peel_image_gelsight_emb_ldp_24fps',
                            # 'task.dataset_path=/share/project/lsq/rdp/reactive_diffusion_policy/data/dataset_mini/dataset_mini/pickplace_downsample1_zarr'])# 
                            'task.dataset_path=/share/project/lsq/rdp/data_backup/wipeblackboard1009_downsample1_zarr'])# 
                            # 'task.dataset_path=/share/project/lsq/rdp/reactive_diffusion_policy/data/dataset_mini/dataset_mini/wipe_blackboard_task_downsample1_zarr'])# 
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        print(len(dataset))
    return dataset, cfg

            
def print_dataset_keys(dataset, indent=0): 
    # lsq: 打印数据集的结构和内容信息 print_dataset_keys(dataset[0]), print_dataset_keys(dataset.replay_buffer)
    """
    递归打印数据集的结构和内容信息
    
    Args:
        dataset: 要打印的数据集
        indent: 缩进级别，用于格式化输出
    """
    indent_str = "   " * indent
    
    for key in dataset.keys():
        value = dataset[key]
        
        if isinstance(value, dict):
            print(f"{indent_str}{key}: (dict)")
            print_dataset_keys(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(f"{indent_str}{key}: (numpy.ndarray)")
            print(f"{indent_str}  shape: {value.shape}")
            print(f"{indent_str}  dtype: {value.dtype}")
            print(f"{indent_str}  min: {value.min():.4f}, max: {value.max():.4f}")
            print(f"{indent_str}  mean: {value.mean():.4f}")
            print(f"{indent_str}{'-' * 50}")
        elif isinstance(value, list):
            print(f"{indent_str}{key}: (list)")
            print(f"{indent_str}  length: {len(value)}")
            if len(value) > 0:
                print(f"{indent_str}  first element type: {type(value[0])}")
                if isinstance(value[0], (np.ndarray, list)):
                    print(f"{indent_str}  first element shape/length: {value[0].shape if hasattr(value[0], 'shape') else len(value[0])}")
            print(f"{indent_str}{'-' * 50}")
        elif isinstance(value, torch.Tensor):
            print(f"{indent_str}{key}: (torch.Tensor)")
            print(f"{indent_str}  shape: {value.shape}")
            print(f"{indent_str}  dtype: {value.dtype}")
            print(f"{indent_str}  device: {value.device}")
            if value.numel() > 0:
                print(f"{indent_str}  min: {value.min().item():.4f}, max: {value.max().item():.4f}")
                print(f"{indent_str}  mean: {value.mean().item():.4f}")
            print(f"{indent_str}{'-' * 50}")
        else:
            print(f"{indent_str}{key}: ({type(value).__name__})")
            if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                print(f"{indent_str}  length: {len(value)}")
            print(f"{indent_str}  value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            print(f"{indent_str}{'-' * 50}")

def test_plot_action():
    dataset, cfg = test_init()
    # import pdb; pdb.set_trace()
    # 绘制原始zarr的数据
    print(dataset.replay_buffer['action'].shape)
    print(dataset.replay_buffer['action'][:,:3])
    print(dataset.replay_buffer['action'][:,:3].max())
    print(dataset.replay_buffer['action'][:,:3].min())
    print(dataset.replay_buffer['action'][:,:3].mean())
    print(dataset.replay_buffer['action'][:,:3].std())
    print(dataset.replay_buffer['action'][:,:3].var())
    print(dataset.replay_buffer['action'][:,3].max())
    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(10, 5))
    plt.plot(dataset.replay_buffer['action'][:100,0], label='x')
    plt.plot(dataset.replay_buffer['action'][:100,1], label='y')
    plt.plot(dataset.replay_buffer['action'][:100,2], label='z')
    plt.legend()
    plt.show()
    plt.figure(2, figsize=(10, 5))
    plt.plot(dataset.replay_buffer['action'][:,3], label='girpper width')
    plt.legend()
    plt.show()
    
    
    ###
    for key in dataset[0].keys():
        
        print(key)
        
        print(dataset[0][key].shape)
        print('-'*100)
    
    # 绘制dataset中的第i个action的xyz
    i=1
    for i in range(1):
        plt.figure()
        plt.plot(dataset[i]['action'][:,0], label='x')
        plt.plot(dataset[i]['action'][:,1], label='y')
        plt.plot(dataset[i]['action'][:,2], label='z')
        plt.legend()
        plt.show()
    
    
    # rgb_keys = list()
    # lowdim_keys = list()
    # obs_shape_meta = cfg.task.dataset.shape_meta['obs']
    # for key, attr in obs_shape_meta.items():
    #     type = attr.get('type', 'low_dim')
    #     if type == 'rgb':
    #         rgb_keys.append(key)
    #     elif type == 'low_dim':
    #         lowdim_keys.append(key)

    # extended_rgb_keys = list()
    # extended_lowdim_keys = list()
    # extended_obs_shape_meta = cfg.task.dataset.shape_meta.get('extended_obs', dict())
    # for key, attr in extended_obs_shape_meta.items():
    #     type = attr.get('type', 'low_dim')
    #     if type == 'rgb':
    #         extended_rgb_keys.append(key)
    #     elif type == 'low_dim':
    #         extended_lowdim_keys.append(key) # ['left_gripper1_marker_offset_emb']

    # zarr_path = os.path.join(cfg.task.dataset.dataset_path, 'replay_buffer.zarr')
    # zarr_load_keys = set(rgb_keys + lowdim_keys + extended_rgb_keys + extended_lowdim_keys + ['action'])
    # zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
    # replay_buffer = ReplayBuffer.copy_from_path(
    #     zarr_path, keys=zarr_load_keys)

    # print(replay_buffer['action'].shape)
    # print(replay_buffer['action'][:,:3])
    # print(replay_buffer['action'][:,:3].max())
    # print(replay_buffer['action'][:,:3].min())
    # print(replay_buffer['action'][:,:3].mean())
    # print(replay_buffer['action'][:,:3].std())
    # print(replay_buffer['action'][:,:3].var())

def test():
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with initialize('../config'):
        # cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
        #                     overrides=['task=real_peel_image_gelsight_emb_ldp_24fps'])
        # cfg = hydra.compose('train_latent_diffusion_unet_real_image_workspace',
        #                     overrides=['task=real_peel_image_gelsight_emb_ldp_24fps',
        #                     # 'task.dataset_path=/share/project/lsq/rdp/data_backup/wipeblackboard1009_downsample1_zarr']) # 48385
        #                     'task.dataset_path=/share/project/lsq/rdp/data_backup/wipeblackboard1009_downsample1_zarr_less/wipeblackboard1009_downsample1_zarr']) # 29965
        #                     # 'task.dataset_path=/share/project/lsq/rdp/reactive_diffusion_policy/data/dataset_mini/dataset_mini/pickplace_downsample1_zarr'])# 
        cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
                            overrides=['task=real_peel_image_gelsight_img_dp_absolute_12fps',
                            # 'task.dataset_path=/share/project/lsq/rdp/data_backup/wipeblackboard1009_downsample1_zarr']) # 48385
                            # 'task.dataset_path=/share/project/lsq/rdp/data_backup/wipeblackboard1009_downsample1_zarr_less/wipeblackboard1009_downsample1_zarr']) # 29965
                            'task.dataset_path=/home/ps/reactive_diffusion_policy/data/dataset_mini/dataset_mini/pickplace_downsample1_zarr'])
                            # 'task.dataset_path=/share/project/lsq/rdp/reactive_diffusion_policy/data/dataset_mini/dataset_mini/pickplace_downsample1_zarr'])# 
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        print(len(dataset))
        
    for key in cfg.keys():
        print(key)
        print(cfg[key])
        print('-'*100)

    from matplotlib import pyplot as plt
    # import pdb; pdb.set_trace()
    normalizer = dataset.get_normalizer()
    # print('dataset[200]["action"]: ',dataset[200]['action'])
    print('dataset[200]["action"]: ',dataset[200]['action'])
    # 
    action_max = normalizer['action'].params_dict['input_stats']['max']
    action_min = normalizer['action'].params_dict['input_stats']['min']
    left_robot_tcp_pose = normalizer['left_robot_tcp_pose'].params_dict['input_stats']['max']
    latent_action = normalizer['latent_action'].params_dict['input_stats']['max']
    print(action_max, action_min)
    
    xyz = dataset[200]['action'][:,:3]
    # plot xyz
    plt.figure()
    plt.plot(xyz[:,0], label='x')
    plt.plot(xyz[:,1], label='y')
    plt.plot(xyz[:,2], label='z')
    plt.legend()
    plt.show()
    
    for i in range(1000):
        print('len of traj:', len(dataset[i]['action']))

    for i in range(len(dataset)):
        # data = dataset[i]
        xyz = dataset[i]['action'][:,:3]
        if xyz[:,0] > 3 or xyz[:,1] > 3 or xyz[:,2] > 3:
            print(i)
            # break

if __name__ == '__main__':
    test()
    # test_plot_action()
