if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
# import wandb
try:
    import swanlab as wandb
except ImportError:
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard against missing loggers
        raise ImportError(
            "Neither 'swanlab' nor 'wandb' is installed. Please install one of them or adjust the logging settings."
        ) from exc
import tqdm
import numpy as np
import shutil
import pickle
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from reactive_diffusion_policy.common.json_logger import JsonLogger
from reactive_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from reactive_diffusion_policy.model.diffusion.ema_model import EMAModel
from reactive_diffusion_policy.model.common.lr_scheduler import get_scheduler
from reactive_diffusion_policy.model.common.lr_decay import param_groups_lrd
from reactive_diffusion_policy.common.space_utils import delta_between2matrices, pose_3d_9d_to_homo_matrix_batch, matrices_to_pose_6d_batch
from accelerate import Accelerator
DEG_TO_RAD = 180.0 / np.pi
MM_PER_METER = 1000.0
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        print(f"self.model: {self.model}")
        # import pdb; pdb.set_trace()
        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        action_shape = cfg.shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        action_extra_meta = cfg.shape_meta.get('action_extra', dict())
        self.action_extra_keys = list(action_extra_meta.keys())
        self.action_extra_dim = sum(
            attr.get('shape', [0])[0] for attr in action_extra_meta.values()
        )
        self.base_action_dim = self.action_dim - self.action_extra_dim
        self.action_gripper_index = 9 #hardcoded for 10D action space
        
        # configure training state

        if 'timm' in cfg.policy.obs_encoder._target_:
            if cfg.training.layer_decay < 1.0:
                assert not cfg.policy.obs_encoder.use_lora
                assert not cfg.policy.obs_encoder.share_rgb_model
                obs_encorder_param_groups = param_groups_lrd(self.model.obs_encoder,
                                                             shape_meta=cfg.shape_meta,
                                                             weight_decay=cfg.optimizer.encoder_weight_decay,
                                                             no_weight_decay_list=self.model.obs_encoder.no_weight_decay(),
                                                             layer_decay=cfg.training.layer_decay)
                count = 0
                for group in obs_encorder_param_groups:
                    count += len(group['params'])
                if cfg.policy.obs_encoder.feature_aggregation == 'map':
                    obs_encorder_param_groups.extend([{'params': self.model.obs_encoder.attn_pool.parameters()}])
                    for _ in self.model.obs_encoder.attn_pool.parameters():
                        count += 1
                print(f'obs_encorder params: {count}')
                param_groups = [{'params': self.model.model.parameters()}]
                param_groups.extend(obs_encorder_param_groups)
            else:
                obs_encorder_lr = cfg.optimizer.lr
                if cfg.policy.obs_encoder.pretrained and not cfg.policy.obs_encoder.use_lora:
                    obs_encorder_lr *= cfg.training.encoder_lr_coefficient
                    print('==> reduce pretrained obs_encorder\'s lr')
                obs_encorder_params = list()
                for param in self.model.obs_encoder.parameters():
                    if param.requires_grad:
                        obs_encorder_params.append(param)
                print(f'obs_encorder params: {len(obs_encorder_params)}')
                param_groups = [
                    {'params': self.model.model.parameters()},
                    {'params': obs_encorder_params, 'lr': obs_encorder_lr}
                ]
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('_target_')
            if 'encoder_weight_decay' in optimizer_cfg.keys():
                optimizer_cfg.pop('encoder_weight_decay')
            self.optimizer = torch.optim.AdamW(
                params=param_groups,
                **optimizer_cfg
            )
        else:
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('encoder_weight_decay')
            # hack: use larger learning rate for multiple gpus
            accelerator = Accelerator()
            cuda_count = accelerator.num_processes
            if accelerator.is_main_process:
                print("=======================================================")
                print(f"Number of available CUDA devices: {cuda_count}.")
                print(f"Original learning rate: {optimizer_cfg['lr']}")
            # optimizer_cfg['lr'] = optimizer_cfg['lr'] * cuda_count
            # print(f"Updated learning rate: {optimizer_cfg['lr']}")
            # print("###########################################")
            self.optimizer = hydra.utils.instantiate(
                optimizer_cfg, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        #accelerator = Accelerator(log_with='swanlab')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
           # init_kwargs={"swanlab": wandb_cfg}
        )


        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        # dataset summary if needed
        if accelerator.is_main_process:

            replay_buffer = getattr(dataset, 'replay_buffer', None)
            if replay_buffer is not None and hasattr(replay_buffer, 'root'):
                data_group = replay_buffer.root.get('data', None)
                meta_group = replay_buffer.root.get('meta', None)

                def _iter_arrays(group):
                    if group is None:
                        return []
                    if hasattr(group, 'array_keys'):
                        keys = group.array_keys()
                    else:
                        keys = group.keys()
                    return [key for key in keys]

                if data_group is not None:
                    total_entries = data_group['action'].shape[0] if 'action' in data_group else 'unknown'
                    print(f"[Dataset] total timesteps: {total_entries}")
                    for key in sorted(_iter_arrays(data_group)):
                        arr = data_group[key]
                        if not hasattr(arr, 'shape'):
                            continue
                        print(f"[Dataset] data/{key}: shape={tuple(arr.shape)}, dtype={arr.dtype}")
                if meta_group is not None:
                    for key in sorted(_iter_arrays(meta_group)):
                        arr = meta_group[key]
                        if not hasattr(arr, 'shape'):
                            continue
                        print(f"[Dataset] meta/{key}: shape={tuple(arr.shape)}, dtype={arr.dtype}")
            else:
                print("  replay buffer not available for summary")
                
            sample = dataset[0]
            action_tensor = sample['action']
            print(
                f"[Dataset] input action_all tensor shape: {tuple(action_tensor.shape)} "
                f"(base={self.base_action_dim}, extra={self.action_extra_dim})"
            )

            if self.action_extra_dim:
                extra_slice = action_tensor[:, -self.action_extra_dim:]
                print(f"[Dataset] input action extra keys: {self.action_extra_keys}")
                print(f"[Dataset] check input wrench values: {extra_slice[0].tolist()}")
            else:
                print("[Dataset] no action extra dimensions detected")
        # import pdb;pdb.set_trace()
        
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        
        # normalizer = dataset.get_normalizer()
        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            with open(normalizer_path, 'wb') as f:
                pickle.dump(normalizer, f)

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True
            )

        # device transfer
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None
        val_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            train_batches_per_epoch = len(train_dataloader)
            if cfg.training.max_train_steps is not None:
                train_batches_per_epoch = min(train_batches_per_epoch, cfg.training.max_train_steps)
            total_train_updates = cfg.training.num_epochs * train_batches_per_epoch
            bar_format = (
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            )

            train_progress = tqdm.tqdm(
                total=total_train_updates,
                desc=f"Training epoch {self.epoch}",
                leave=True,
                mininterval=cfg.training.tqdm_interval_sec,
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
                bar_format=bar_format,
            )

            val_progress = None
            show_val_progress = bool(cfg.training.get('show_val_tqdm', False))
            if (
                show_val_progress
                and cfg.task.dataset.val_ratio > 0
                and accelerator.is_main_process
                and cfg.training.val_every > 0
            ):
                val_batches_per_eval = len(val_dataloader)
                if cfg.training.max_val_steps is not None:
                    val_batches_per_eval = min(val_batches_per_eval, cfg.training.max_val_steps)
                num_val_epochs = len(range(0, cfg.training.num_epochs, cfg.training.val_every))
                total_val_updates = num_val_epochs * val_batches_per_eval
                if total_val_updates > 0:
                    val_progress = tqdm.tqdm(
                        total=total_val_updates,
                        desc="Validation",
                        leave=True,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=False,
                        position=1,
                        dynamic_ncols=True,
                        bar_format=bar_format,
                    )

            try:
                for local_epoch_idx in range(cfg.training.num_epochs):
                    step_log = dict()
                    # ========= train for this epoch ==========
                    if cfg.training.freeze_encoder:
                        self.model.obs_encoder.eval()
                        self.model.obs_encoder.requires_grad_(False)

                    train_losses = list()
                    current_epoch_1based = self.epoch + 1
                    train_progress.set_description(
                        f"Training epoch {current_epoch_1based}/{cfg.training.num_epochs}"
                    )
                    for batch_idx, batch in enumerate(train_dataloader):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        train_progress.update(1)
                        train_progress.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                    # at the end of each epoch
                    # replace train_loss with epoch average
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                    # ========= eval for this epoch ==========
                    policy = accelerator.unwrap_model(self.model)
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    # run validation
                    
                    if cfg.task.dataset.val_ratio > 0 and (self.epoch % cfg.training.val_every) == 0 and accelerator.is_main_process:
                        with torch.no_grad():
                            val_losses = list()
                            if val_progress is not None:
                                val_progress.set_description(
                                    f"Validation epoch {current_epoch_1based}/{cfg.training.num_epochs}"
                                )
                            for batch_idx, batch in enumerate(val_dataloader):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                if val_sampling_batch is None: # lsq
                                    val_sampling_batch = batch
                                    # print(f"val_sampling_batch: {val_sampling_batch}")
                                    # import pdb; pdb.set_trace()
                                loss = self.model(batch)
                                loss_cpu = loss.item()
                                val_losses.append(loss_cpu)
                                if val_progress is not None:
                                    val_progress.update(1)
                                    val_progress.set_postfix(loss=loss_cpu, refresh=False)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                            if len(val_losses) > 0:
                                val_loss = float(np.mean(val_losses))
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss

                    # run diffusion sampling on a validation batch
                    if cfg.task.dataset.val_ratio > 0 and (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                        with torch.no_grad():
                            # sample trajectory from validation set, and evaluate difference
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            extended_obs_dict = batch['extended_obs']
                            gt_action = batch['action']

                            if 'latent' in cfg.name:
                                dataset_obs_temporal_downsample_ratio = cfg.task.dataset.obs_temporal_downsample_ratio
                                result = policy.predict_action(obs_dict,
                                                               extended_obs_dict=extended_obs_dict,
                                                               dataset_obs_temporal_downsample_ratio=dataset_obs_temporal_downsample_ratio)
                            else:
                                result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']

                            # since only main process runs this, no need for gather_for_metrics
                            pred_xyz = pred_action[..., 0:3]
                            gt_xyz = gt_action[..., 0:3]
                            xyz_mse = torch.nn.functional.mse_loss(pred_xyz, gt_xyz)
                            val_metrics = {'val_xyz_tll_mse': xyz_mse.item()}
                            
                            ## lsq: compute delta between predicted and ground truth pose
                            # pred_action[..., 0:9] shape: (batch_size, T, 9) , should be reshape to (batch_size*T, 9)
                            # NOTE: space_utils uses numpy; convert tensors to cpu numpy first
                            device = pred_action.device
                            dtype = pred_action.dtype
                            pred_action_reshape = pred_action[..., 0:9].reshape(-1, 9).detach().cpu().numpy()
                            gt_action_reshape = gt_action[..., 0:9].reshape(-1, 9).detach().cpu().numpy()
                            pred_pose_mat = pose_3d_9d_to_homo_matrix_batch(pred_action_reshape)
                            gt_pose_mat = pose_3d_9d_to_homo_matrix_batch(gt_action_reshape)
                            delta_pose_mat = delta_between2matrices(pred_pose_mat, gt_pose_mat)
                            delta_pose_6d_np = matrices_to_pose_6d_batch(delta_pose_mat)  # (N, 6) numpy
                            delta_pose_6d = torch.from_numpy(delta_pose_6d_np).to(device=device, dtype=dtype)

                            # delta_pose_6d_xyz_mse = torch.nn.functional.mse_loss(
                            #     delta_pose_6d[..., 0:3], torch.zeros_like(delta_pose_6d[..., 0:3])
                            # )
                            # val_metrics['val_xyz_mse'] = delta_pose_6d_xyz_mse.item()
                            # delta_pose_6d_rpy_mse = torch.nn.functional.mse_loss(
                            #     delta_pose_6d[..., 3:6], torch.zeros_like(delta_pose_6d[..., 3:6])
                            # )
                            # val_metrics['val_rpy_mse'] = delta_pose_6d_rpy_mse.item()
                            # # component-wise MSE
                            # val_metrics['val_x_mse'] = (delta_pose_6d[..., 0] ** 2).mean().item()
                            # val_metrics['val_y_mse'] = (delta_pose_6d[..., 1] ** 2).mean().item()
                            # val_metrics['val_z_mse'] = (delta_pose_6d[..., 2] ** 2).mean().item()
                            # val_metrics['val_roll_mse'] = (delta_pose_6d[..., 3] ** 2).mean().item()
                            # val_metrics['val_pitch_mse'] = (delta_pose_6d[..., 4] ** 2).mean().item()
                            # val_metrics['val_yaw_mse'] = (delta_pose_6d[..., 5] ** 2).mean().item()
                            delta_pose_6d_xyz_mae = torch.nn.functional.l1_loss(
                                delta_pose_6d[..., 0:3], torch.zeros_like(delta_pose_6d[..., 0:3])
                            )
                            val_metrics['val_xyz_mae'] = delta_pose_6d_xyz_mae.item() * MM_PER_METER
                            delta_pose_6d_rpy_mae = torch.nn.functional.l1_loss(
                                delta_pose_6d[..., 3:6], torch.zeros_like(delta_pose_6d[..., 3:6])
                            )
                            val_metrics['val_rpy_mae'] = delta_pose_6d_rpy_mae.item() * DEG_TO_RAD
                            # component-wise MSE
                            val_metrics['val_x_mae'] = (delta_pose_6d[..., 0]).abs().mean().item() * MM_PER_METER
                            val_metrics['val_y_mae'] = (delta_pose_6d[..., 1]).abs().mean().item() * MM_PER_METER
                            val_metrics['val_z_mae'] = (delta_pose_6d[..., 2]).abs().mean().item() * MM_PER_METER
                            val_metrics['val_roll_mae'] = (delta_pose_6d[..., 3]).abs().mean().item() * DEG_TO_RAD
                            val_metrics['val_pitch_mae'] = (delta_pose_6d[..., 4]).abs().mean().item() * DEG_TO_RAD
                            val_metrics['val_yaw_mae'] = (delta_pose_6d[..., 5]).abs().mean().item() * DEG_TO_RAD
                            
                            

                            pred_gripper = None
                            gt_gripper = None
                            if self.action_dim > 9:
                                pred_gripper = pred_action[..., 9:10]
                                gt_gripper = gt_action[..., 9:10]
                                # gripper_mse = torch.nn.functional.mse_loss(pred_gripper, gt_gripper)
                                gripper_mse = torch.nn.functional.l1_loss(pred_gripper, gt_gripper)
                                val_metrics['val_gripper_mae'] = gripper_mse.item()

                            if self.action_dim >= 13:
                                pred_wrench = pred_action[..., 10:13]
                                gt_wrench = gt_action[..., 10:13]
                                # wrench_mse = torch.nn.functional.mse_loss(pred_wrench, gt_wrench)
                                wrench_mse = torch.nn.functional.l1_loss(pred_wrench, gt_wrench)
                                val_metrics['val_wrench_mae'] = wrench_mse.item()
                                val_metrics['val_fx_mae'] = (pred_wrench[..., 0] - gt_wrench[..., 0]).abs().mean().item()
                                val_metrics['val_fy_mae'] = (pred_wrench[..., 1] - gt_wrench[..., 1]).abs().mean().item()
                                val_metrics['val_fz_mae'] = (pred_wrench[..., 2] - gt_wrench[..., 2]).abs().mean().item()
                                # val_metrics['val_mx_mae'] = (pred_wrench[..., 3] - gt_wrench[..., 3]).abs().mean().item()
                                # val_metrics['val_my_mae'] = (pred_wrench[..., 4] - gt_wrench[..., 4]).abs().mean().item()
                                # val_metrics['val_mz_mae'] = (pred_wrench[..., 5] - gt_wrench[..., 5]).abs().mean().item()
                                

                            # combined action metric (xyz + gripper when available)
                            core_pred_parts = [pred_xyz]
                            core_gt_parts = [gt_xyz]
                            if pred_gripper is not None and gt_gripper is not None:
                                core_pred_parts.append(pred_gripper)
                                core_gt_parts.append(gt_gripper)
                            pred_action_core = torch.cat(core_pred_parts, dim=-1) if len(core_pred_parts) > 1 else core_pred_parts[0]
                            gt_action_core = torch.cat(core_gt_parts, dim=-1) if len(core_gt_parts) > 1 else core_gt_parts[0]
                            action_mse = torch.nn.functional.mse_loss(pred_action_core, gt_action_core)
                            val_metrics['val_action_mse'] = action_mse.item()

                            step_log.update(val_metrics)
                            metrics_with_epoch = {**val_metrics, 'epoch': self.epoch}

                            accelerator.log(metrics_with_epoch, step=self.global_step)
                            json_logger.log({**metrics_with_epoch, 'global_step': self.global_step})
                            
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                    accelerator.wait_for_everyone()
                    
                    # checkpoint
                    if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                        # unwrap the model to save ckpt
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)

                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                        # recover the DDP model
                        self.model = model_ddp
                        
                    # ========= eval end for this epoch ==========
                    policy.train()

                    if accelerator.is_main_process:
                        summary_items = {
                            key: value for key, value in step_log.items()
                            if key == 'train_loss' or key.startswith('val_')
                        }
                        if summary_items:
                            summary_line = (
                                f"[Epoch {current_epoch_1based}] "
                                + ", ".join(f"{k}={v:.6f}" for k, v in summary_items.items())
                            )
                            train_progress.write(summary_line)

                    # end of epoch
                    # log of last step is combined with validation and rollout
                    accelerator.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1
                    self.epoch += 1
            finally:
                train_progress.close()
                if val_progress is not None:
                    val_progress.close()

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
