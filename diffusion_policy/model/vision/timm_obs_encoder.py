import copy

import timm
import peft
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from functools import partial

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.choice_randomizer import RandomChoice
from timm.layers.attention_pool import AttentionPoolLatent

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            three_augment: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            use_lora: bool = False,
            lora_rank: int = 8,
            drop_path_rate: float = 0.0,
            fused_model_name: str = '',
            # Optional Human2Dex G/L fusion. ``concat`` preserves the
            # repository's original multi-RGB behavior exactly.
            dual_view_fusion: str = 'concat',
            dual_view_mode: str = 'gl',
            dual_view_global_key: str = 'camera0_rgb',
            dual_view_local_key: str = 'camera0_local_rgb',
            dual_view_dim: int = 256,
            dual_view_heads: int = 4,
            dual_view_output_dim: int = 1024,
            record_dual_view_attention: bool = False,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_fused_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        key_eval_transform_map = nn.ModuleDict()

        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        assert global_pool == ''
        if 'resnet' in model_name:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,    # '' means no pooling
                num_classes=0,              # remove classification layer
            )
        else:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,    # '' means no pooling
                num_classes=0,              # remove classification layer
                img_size=image_shape[0],    # 224
                drop_path_rate=drop_path_rate,  # stochastic depth
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        if use_lora:
            assert pretrained and not frozen
            lora_config = peft.LoraConfig(
                r=lora_rank,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=["qkv"],
            )
            model = peft.get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        fused_model = None
        if fused_model_name != '':
            assert feature_aggregation == 'map'
            fused_model = timm.create_model(
                model_name=fused_model_name,
                pretrained=True,
                global_pool=global_pool,
                num_classes=0,
                img_size=image_shape[0],
                drop_path_rate=0.0,
            )
            for param in fused_model.parameters():
                param.requires_grad = False

        feature_dim = None
        num_heads = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('vit'):
            feature_dim = model.num_features
            num_heads = model.blocks[0].attn.num_heads
            if fused_model_name != '':
                feature_dim = feature_dim + fused_model.num_features

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        # ``transforms: null`` is used by the G/L policy to prohibit online
        # geometry changes while retaining deterministic DINO normalization.
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
            if imagenet_norm:
                transforms = transforms + [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        normalization = (
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            if imagenet_norm
            else nn.Identity()
        )
        transform = normalization if transforms is None else torch.nn.Sequential(*transforms)

        eval_transforms = None
        if transforms is not None:
            eval_transforms = [torchvision.transforms.Resize(size=image_shape[0], antialias=True)]
            if imagenet_norm:
                eval_transforms = eval_transforms + [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        eval_transform = normalization if transforms is None else torch.nn.Sequential(*eval_transforms)

        if three_augment:
            # Following DeiT III: https://arxiv.org/abs/2204.07118
            primary_tfl = [
                torchvision.transforms.RandomCrop(image_shape[0], padding=4, padding_mode='reflect'),
            ]
            secondary_tfl = [
                RandomChoice([torchvision.transforms.Grayscale(num_output_channels=3),
                              torchvision.transforms.RandomSolarize(threshold=0.5, p=1.0),
                              torchvision.transforms.GaussianBlur(kernel_size=5)]),
                torchvision.transforms.ColorJitter(0.3, 0.3, 0.3)
            ]
            final_tfl = [
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            transform = torch.nn.Sequential(*primary_tfl, *secondary_tfl, *final_tfl)
            assert eval_transform is not None and eval_transform != nn.Identity()
            
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                key_fused_model_map[key] = fused_model

                this_transform = transform
                key_transform_map[key] = this_transform
                key_eval_transform_map[key] = eval_transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_fused_model_map = key_fused_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.key_eval_transform_map = key_eval_transform_map
        self.feature_aggregation = feature_aggregation
        self.fused_model_name = fused_model_name
        self.dual_view_fusion = str(dual_view_fusion)
        self.dual_view_mode = str(dual_view_mode)
        self.dual_view_global_key = str(dual_view_global_key)
        self.dual_view_local_key = str(dual_view_local_key)
        self.record_dual_view_attention = bool(record_dual_view_attention)
        self._last_dual_view_attention = None
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == 'cls_token':
                pass
            elif self.feature_aggregation == 'map':
                # Multihead Attention Pooling, following https://arxiv.org/abs/1810.00825
                self.attn_pool = AttentionPoolLatent(
                    in_features=feature_dim,
                    num_heads=num_heads,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                )
            else:
                raise NotImplementedError(f"Unsupported feature_aggregation: {self.feature_aggregation}")

        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )

        if self.dual_view_fusion not in ('concat', 'g_local_patch_attention'):
            raise ValueError(
                "dual_view_fusion must be 'concat' or 'g_local_patch_attention', "
                f"got {self.dual_view_fusion!r}"
            )
        if self.dual_view_fusion == 'g_local_patch_attention':
            if not model_name.startswith('vit'):
                raise ValueError('g_local_patch_attention currently requires a ViT/DINO backbone')
            if self.feature_aggregation != 'cls_token':
                raise ValueError('g_local_patch_attention requires feature_aggregation=cls_token')
            if not share_rgb_model:
                raise ValueError('g_local_patch_attention requires share_rgb_model=true')
            if fused_model_name != '':
                raise ValueError('g_local_patch_attention does not support fused_model_name')
            if self.dual_view_global_key not in rgb_keys or self.dual_view_local_key not in rgb_keys:
                raise KeyError(
                    'dual-view RGB keys must both be declared in shape_meta.obs: '
                    f'{self.dual_view_global_key!r}, {self.dual_view_local_key!r}'
                )
            if self.dual_view_mode not in ('gl', 'g_only'):
                raise ValueError("dual_view_mode must be 'gl' or 'g_only'")
            if int(dual_view_dim) <= 0 or int(dual_view_output_dim) <= 0:
                raise ValueError('dual_view_dim and dual_view_output_dim must be positive')
            if int(dual_view_dim) % int(dual_view_heads) != 0:
                raise ValueError('dual_view_dim must divide evenly by dual_view_heads')

            # DINO tokens already contain the backbone's absolute patch
            # position.  This additional coordinate projection explicitly
            # preserves the L-view's fixed pocket-relative image layout.
            self.dual_view_prefix_tokens = int(getattr(model, 'num_prefix_tokens', 1))
            self.dual_global_norm = nn.LayerNorm(feature_dim)
            self.dual_local_cls_norm = nn.LayerNorm(feature_dim)
            self.dual_local_patch_norm = nn.LayerNorm(feature_dim)
            self.dual_global_proj = nn.Linear(feature_dim, int(dual_view_dim))
            self.dual_local_cls_proj = nn.Linear(feature_dim, int(dual_view_dim))
            self.dual_local_patch_proj = nn.Linear(feature_dim, int(dual_view_dim))
            self.dual_local_position = nn.Sequential(
                nn.Linear(2, int(dual_view_dim)),
                nn.GELU(),
                nn.Linear(int(dual_view_dim), int(dual_view_dim)),
            )
            self.dual_cross_attention = nn.MultiheadAttention(
                embed_dim=int(dual_view_dim),
                num_heads=int(dual_view_heads),
                batch_first=True,
            )
            self.dual_context_norm = nn.LayerNorm(int(dual_view_dim))
            # These learned placeholders are only read by the G-only branch.
            # Keep them in both modes for checkpoint compatibility, but exclude
            # them from DDP/optimization in GL mode where they are unused.
            empty_local_requires_grad = (self.dual_view_mode == 'g_only')
            self.dual_empty_local_cls = nn.Parameter(
                torch.zeros(1, 1, int(dual_view_dim)),
                requires_grad=empty_local_requires_grad,
            )
            self.dual_empty_local_context = nn.Parameter(
                torch.zeros(1, 1, int(dual_view_dim)),
                requires_grad=empty_local_requires_grad,
            )
            self.dual_fusion = nn.Sequential(
                nn.LayerNorm(4 * int(dual_view_dim)),
                nn.Linear(4 * int(dual_view_dim), int(dual_view_output_dim)),
                nn.GELU(),
                nn.Linear(int(dual_view_output_dim), int(dual_view_output_dim)),
                nn.LayerNorm(int(dual_view_output_dim)),
            )
            self.dual_view_output_dim = int(dual_view_output_dim)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature, fused_feature=None):
        if self.model_name.startswith('vit'):
            if self.feature_aggregation == 'cls_token':
                return feature[:, 0, :]
            elif self.feature_aggregation == 'map':
                feature = feature[:, 1:, :]
                if fused_feature is not None:
                    num_tokens = feature.shape[1]
                    fused_feature = fused_feature[:, -num_tokens:, :]
                    feature = torch.cat([feature, fused_feature], dim=2)
                feature = self.attn_pool(feature)
                return feature

        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature

    @staticmethod
    def _local_coordinate_grid(num_tokens: int, device, dtype):
        """Return normalized x/y coordinates for a square local patch grid."""
        side = int(round(math.sqrt(num_tokens)))
        if side * side != num_tokens:
            raise RuntimeError(
                f'Expected a square local patch grid, got {num_tokens} patch tokens'
            )
        axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        return torch.stack((xx, yy), dim=-1).reshape(1, num_tokens, 2)

    def _apply_rgb_transform(self, key, image):
        return self.key_transform_map[key](image) if self.training else self.key_eval_transform_map[key](image)

    def _forward_dual_view(self, obs_dict, batch_size):
        global_image = obs_dict[self.dual_view_global_key]
        B, T = global_image.shape[:2]
        if B != batch_size:
            raise RuntimeError('dual-view global batch size mismatch')
        if global_image.shape[2:] != self.key_shape_map[self.dual_view_global_key]:
            raise RuntimeError('dual-view global RGB shape mismatch')
        global_image = global_image.reshape(B * T, *global_image.shape[2:])
        global_image = self._apply_rgb_transform(self.dual_view_global_key, global_image)

        # G-only is a true visual ablation: the same fusion parameters remain,
        # but no local image token can influence the condition.
        if self.dual_view_mode == 'g_only':
            raw_global = self.key_model_map[self.dual_view_global_key](global_image)
            raw_local = None
        else:
            local_image = obs_dict[self.dual_view_local_key]
            if local_image.shape[:2] != (B, T):
                raise RuntimeError('dual-view G/L horizons must match')
            if local_image.shape[2:] != self.key_shape_map[self.dual_view_local_key]:
                raise RuntimeError('dual-view local RGB shape mismatch')
            local_image = local_image.reshape(B * T, *local_image.shape[2:])
            local_image = self._apply_rgb_transform(self.dual_view_local_key, local_image)
            # One batched shared-DINO call lowers launch overhead at deployment.
            raw_global, raw_local = self.key_model_map[self.dual_view_global_key](
                torch.cat((global_image, local_image), dim=0)
            ).chunk(2, dim=0)

        if raw_global.ndim != 3:
            raise RuntimeError(
                'g_local_patch_attention needs unpooled ViT tokens [B,token,D]; '
                f'got {tuple(raw_global.shape)}'
            )
        global_cls = raw_global[:, 0, :]
        global_context = self.dual_global_proj(self.dual_global_norm(global_cls))

        if raw_local is None:
            local_cls = self.dual_empty_local_cls.expand(B * T, -1, -1).squeeze(1)
            local_context = self.dual_empty_local_context.expand(B * T, -1, -1).squeeze(1)
            self._last_dual_view_attention = None
        else:
            if raw_local.ndim != 3 or raw_local.shape[1] <= self.dual_view_prefix_tokens:
                raise RuntimeError('local ViT output does not contain patch tokens')
            local_cls = self.dual_local_cls_proj(self.dual_local_cls_norm(raw_local[:, 0, :]))
            local_patch = raw_local[:, self.dual_view_prefix_tokens:, :]
            local_patch = self.dual_local_patch_proj(self.dual_local_patch_norm(local_patch))
            coordinate = self._local_coordinate_grid(
                local_patch.shape[1], local_patch.device, local_patch.dtype
            )
            local_patch = local_patch + self.dual_local_position(coordinate)
            query = global_context.unsqueeze(1)
            need_weights = self.record_dual_view_attention and not self.training
            local_context, attention = self.dual_cross_attention(
                query=query,
                key=local_patch,
                value=local_patch,
                need_weights=need_weights,
                average_attn_weights=False,
            )
            local_context = self.dual_context_norm(local_context.squeeze(1))
            self._last_dual_view_attention = attention.detach() if attention is not None else None

        fusion_input = torch.cat(
            (global_context, local_cls, local_context, global_context * local_context), dim=-1
        )
        feature = self.dual_fusion(fusion_input)
        return feature.reshape(B, T * self.dual_view_output_dim)

    @torch.no_grad()
    def get_last_dual_view_attention(self):
        """Return [B*T, heads, 1, local_patches] after an eval forward, if enabled."""
        return self._last_dual_view_attention
        
    def forward(self, obs_dict, return_rgb_feature=False):
        features = list()
        rgb_features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        dual_keys = set()
        if self.dual_view_fusion == 'g_local_patch_attention':
            dual_feature = self._forward_dual_view(obs_dict, batch_size)
            features.append(dual_feature)
            rgb_features.append(dual_feature)
            dual_keys = {self.dual_view_global_key, self.dual_view_local_key}
        
        # process rgb input
        for key in self.rgb_keys:
            if key in dual_keys:
                continue
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            img = self._apply_rgb_transform(key, img)
            raw_feature = self.key_model_map[key](img)
            fused_feature = None
            if self.fused_model_name != '':
                fused_feature = self.key_fused_model_map[key](img)

            feature = self.aggregate_feature(raw_feature, fused_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            feature = feature.reshape(B, -1)
            features.append(feature)
            rgb_features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

        if return_rgb_feature:
            if not rgb_features:
                raise RuntimeError('return_rgb_feature=True requires at least one RGB input')
            return result, torch.cat(rgb_features, dim=-1)
        return result
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape

    @torch.no_grad()
    def rgb_output_shape(self):
        """Return the pure-RGB feature shape before low-dimensional concatenation."""
        example_obs_dict = dict()
        for key, attr in self.shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            example_obs_dict[key] = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device,
            )
        _, rgb_feature = self.forward(example_obs_dict, return_rgb_feature=True)
        assert rgb_feature.ndim == 2 and rgb_feature.shape[0] == 1
        return rgb_feature.shape

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
