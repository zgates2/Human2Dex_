from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.model.common.lowdim_obs_augmentation import (
    apply_lowdim_obs_augmentation,
)
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            lowdim_obs_augmentation=None,
            action_loss_weighting=None,
            object_pocket_aux_loss=None,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.lowdim_obs_augmentation = lowdim_obs_augmentation
        self.action_loss_weighting = action_loss_weighting
        self.object_pocket_aux_loss = object_pocket_aux_loss
        self.object_pocket_aux_head = None
        if bool(self._cfg_get(object_pocket_aux_loss, 'enabled', False)):
            obs_key = str(self._cfg_get(object_pocket_aux_loss, 'obs_key', 'objectPocketObs'))
            if obs_key not in shape_meta['obs']:
                raise KeyError(f'object pocket auxiliary loss requires obs key {obs_key!r}')
            obs_shape = tuple(shape_meta['obs'][obs_key]['shape'])
            if obs_shape != (5,):
                raise ValueError(
                    f'object pocket auxiliary loss expects {obs_key} shape (5,), got {obs_shape}'
                )
            rgb_feature_dim = int(np.prod(obs_encoder.rgb_output_shape()))
            hidden_dim = int(self._cfg_get(object_pocket_aux_loss, 'hidden_dim', 256))
            if hidden_dim <= 0:
                raise ValueError('object pocket auxiliary hidden_dim must be positive')
            self.object_pocket_aux_head = nn.Sequential(
                nn.LayerNorm(rgb_feature_dim),
                nn.Linear(rgb_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),
            )
        self.last_loss_components = {}
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @staticmethod
    def _cfg_get(config, key, default=None):
        if config is None:
            return default
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _action_dim_groups(self, action_dim: int):
        """Return hand slices for arm+hand action layouts used by O6/Wuji.

        Current layouts:
          Linker O6: [arm pose10 without scalar gripper -> 9, hand -> 6] = 15
          Wuji:      [arm pose10 without scalar gripper -> 9, hand -> 20] = 29

        If a future hand-only checkpoint has no arm prefix, this falls back to
        weighting all dimensions as hand action.
        """
        if action_dim in (15, 29):
            return [(9, action_dim)]
        if action_dim in (6, 20):
            return [(0, action_dim)]
        if action_dim > 9:
            return [(9, action_dim)]
        return []

    @staticmethod
    def _dilate_time_mask(event_mask: torch.Tensor, pre_steps: int, post_steps: int):
        """Dilate a [B,T] bool mask without adding extra dependencies."""
        if event_mask.numel() == 0:
            return event_mask
        result = event_mask.clone()
        bsz, horizon = event_mask.shape
        for offset in range(-int(pre_steps), int(post_steps) + 1):
            if offset == 0:
                continue
            shifted = torch.zeros_like(event_mask)
            if offset < 0:
                shifted[:, :horizon + offset] = event_mask[:, -offset:]
            else:
                shifted[:, offset:] = event_mask[:, :horizon - offset]
            result = result | shifted
        return result

    def _weighted_action_loss(self, element_loss: torch.Tensor, nactions: torch.Tensor):
        cfg = self.action_loss_weighting
        enabled = bool(self._cfg_get(cfg, "enabled", False))
        if not enabled:
            return element_loss.mean()

        if element_loss.ndim != 3:
            raise RuntimeError(f"expected action loss shape [B,T,D], got {element_loss.shape}")

        bsz, horizon, action_dim = element_loss.shape
        device = element_loss.device
        dtype = element_loss.dtype

        hand_slices = self._action_dim_groups(action_dim)
        dim_weight = torch.ones(action_dim, device=device, dtype=dtype)
        hand_weight = float(self._cfg_get(cfg, "hand_weight", 1.0))
        for start, end in hand_slices:
            dim_weight[start:end] = hand_weight

        time_weight = torch.ones((bsz, horizon), device=device, dtype=dtype)
        contact_window_weight = float(self._cfg_get(cfg, "contact_window_weight", 1.0))
        threshold = float(self._cfg_get(cfg, "hand_delta_threshold", 0.0))
        pre_steps = int(self._cfg_get(cfg, "contact_pre_steps", 0))
        post_steps = int(self._cfg_get(cfg, "contact_post_steps", 0))

        contact_mask = torch.zeros((bsz, horizon), device=device, dtype=torch.bool)
        if hand_slices and horizon > 1 and contact_window_weight != 1.0 and threshold > 0.0:
            hand_parts = [nactions[..., start:end] for start, end in hand_slices]
            hand_action = torch.cat(hand_parts, dim=-1)
            hand_delta = torch.linalg.norm(hand_action[:, 1:] - hand_action[:, :-1], dim=-1)
            contact_mask[:, 1:] = hand_delta > threshold
            contact_mask = self._dilate_time_mask(contact_mask, pre_steps, post_steps)
            time_weight = torch.where(
                contact_mask,
                torch.full_like(time_weight, contact_window_weight),
                time_weight,
            )

        weights = time_weight[:, :, None] * dim_weight[None, None, :]
        weighted_loss = element_loss * weights
        denom = weights.expand_as(element_loss).sum().clamp_min(1.0)
        loss = weighted_loss.sum() / denom

        with torch.no_grad():
            hand_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if hand_slices:
                hand_values = [element_loss[..., start:end] for start, end in hand_slices]
                hand_loss = torch.cat(hand_values, dim=-1).mean()
            arm_mask = torch.ones(action_dim, device=device, dtype=torch.bool)
            for start, end in hand_slices:
                arm_mask[start:end] = False
            arm_loss = (
                element_loss[..., arm_mask].mean()
                if bool(arm_mask.any().item())
                else torch.tensor(0.0, device=device, dtype=dtype)
            )
            self.last_loss_components = {
                "loss/unweighted": element_loss.mean().detach(),
                "loss/weighted": loss.detach(),
                "loss/arm": arm_loss.detach(),
                "loss/hand": hand_loss.detach(),
                "loss/contact_window_ratio": contact_mask.float().mean().detach(),
            }

        return loss

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        if self.training:
            nobs = apply_lowdim_obs_augmentation(
                nobs,
                self.lowdim_obs_augmentation,
            )
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        aux_enabled = self.object_pocket_aux_head is not None
        if aux_enabled:
            global_cond, rgb_feature = self.obs_encoder(
                nobs, return_rgb_feature=True
            )
        else:
            global_cond = self.obs_encoder(nobs)
            rgb_feature = None

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        dp_loss = self._weighted_action_loss(loss, nactions)
        total_loss = dp_loss

        if aux_enabled:
            cfg = self.object_pocket_aux_loss
            obs_key = str(self._cfg_get(cfg, 'obs_key', 'objectPocketObs'))
            target = batch['obs'][obs_key][:, -1, :].to(dtype=rgb_feature.dtype)
            prediction = self.object_pocket_aux_head(rgb_feature)

            confidence = target[:, 3].clamp(0.0, 1.0)
            valid = target[:, 4] > 0.5
            min_confidence = float(self._cfg_get(cfg, 'min_confidence', 0.3))
            finite = torch.isfinite(target[:, :5]).all(dim=-1)
            valid_mask = valid & (confidence >= min_confidence) & finite
            sample_weight = torch.where(
                valid_mask,
                confidence.square(),
                torch.zeros_like(confidence),
            )
            denom = sample_weight.sum().clamp_min(1.0)

            element_loss = F.smooth_l1_loss(
                prediction,
                target[:, :3],
                reduction='none',
            )
            position_loss = (
                element_loss[:, :2].mean(dim=-1) * sample_weight
            ).sum() / denom
            area_loss = (
                element_loss[:, 2] * sample_weight
            ).sum() / denom
            position_weight = float(self._cfg_get(cfg, 'position_weight', 1.0))
            area_weight = float(self._cfg_get(cfg, 'area_weight', 0.25))
            aux_loss = position_weight * position_loss + area_weight * area_loss
            aux_weight = float(self._cfg_get(cfg, 'weight', 0.05))
            total_loss = dp_loss + aux_weight * aux_loss

            self.last_loss_components.update({
                'loss/dp': dp_loss.detach(),
                'loss/object_aux': aux_loss.detach(),
                'loss/object_position_aux': position_loss.detach(),
                'loss/object_area_aux': area_loss.detach(),
                'loss/object_aux_valid_ratio': valid_mask.float().mean().detach(),
                'loss/total': total_loss.detach(),
            })

        return total_loss

    def forward(self, batch):
        return self.compute_loss(batch)
