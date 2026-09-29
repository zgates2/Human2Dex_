"""Training-only corruption for normalized low-dimensional observations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch


def _value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    getter = getattr(config, "get", None)
    if getter is None:
        return default
    return getter(key, default)


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_std(name: str, value: float) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def apply_lowdim_obs_augmentation(
    obs_dict: Dict[str, torch.Tensor],
    config: Optional[Mapping[str, Any]],
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Corrupt selected normalized observations without mutating the input.

    Dropout is applied per batch sample and replaces the complete selected
    observation tensor with ``dropout_fill``.  Because the input has already
    been normalized, a fill value of zero represents the normalizer midpoint
    instead of an extreme raw hand command.

    ``per_dimension_bias_std`` samples one offset per sample and hand dimension
    and keeps it constant across the observation horizon.  It approximates a
    persistent command-to-measured-state bias.  ``element_noise_std`` adds
    independent noise to every observation element.
    """

    if config is None or not bool(_value(config, "enabled", False)):
        return obs_dict

    keys = _value(config, "keys", ("robot0_gripper_width",))
    if isinstance(keys, str):
        keys = (keys,)
    keys = tuple(str(key) for key in keys)

    dropout_prob = _validate_probability(
        "dropout_prob", _value(config, "dropout_prob", 0.0)
    )
    dropout_fill = float(_value(config, "dropout_fill", 0.0))
    bias_std = _validate_std(
        "per_dimension_bias_std",
        _value(config, "per_dimension_bias_std", 0.0),
    )
    noise_std = _validate_std(
        "element_noise_std", _value(config, "element_noise_std", 0.0)
    )
    clip_abs_raw = _value(config, "clip_abs", None)
    clip_abs = None if clip_abs_raw is None else float(clip_abs_raw)
    if clip_abs is not None and clip_abs <= 0.0:
        raise ValueError(f"clip_abs must be positive or null, got {clip_abs}")

    result = dict(obs_dict)
    for key in keys:
        if key not in obs_dict:
            continue
        value = obs_dict[key]
        if not torch.is_floating_point(value):
            raise TypeError(f"{key} must be floating point, got {value.dtype}")
        if value.ndim < 2:
            raise ValueError(
                f"{key} must have batch and feature dimensions, got {value.shape}"
            )

        augmented = value.clone()
        batch_size = augmented.shape[0]

        if bias_std > 0.0:
            bias_shape = [batch_size]
            if augmented.ndim > 2:
                bias_shape.extend([1] * (augmented.ndim - 2))
            bias_shape.append(augmented.shape[-1])
            bias = torch.randn(
                bias_shape,
                device=augmented.device,
                dtype=augmented.dtype,
                generator=generator,
            )
            augmented = augmented + bias * bias_std

        if noise_std > 0.0:
            noise = torch.randn(
                augmented.shape,
                device=augmented.device,
                dtype=augmented.dtype,
                generator=generator,
            )
            augmented = augmented + noise * noise_std

        if dropout_prob > 0.0:
            mask_shape = [batch_size] + [1] * (augmented.ndim - 1)
            dropout_mask = torch.rand(
                mask_shape,
                device=augmented.device,
                generator=generator,
            ) < dropout_prob
            augmented = torch.where(
                dropout_mask,
                torch.full_like(augmented, dropout_fill),
                augmented,
            )

        if clip_abs is not None:
            augmented = augmented.clamp(min=-clip_abs, max=clip_abs)

        result[key] = augmented

    return result
