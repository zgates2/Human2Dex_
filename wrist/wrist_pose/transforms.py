"""Image preprocessing and lightweight photometric augmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch


@dataclass
class TransformResult:
    image: torch.Tensor
    scale: float
    offset_xy: tuple[int, int]
    original_hw: tuple[int, int]


class WristImageTransform:
    def __init__(
        self,
        image_size: int = 448,
        train: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        color_jitter: float = 0.25,
        gamma_range: tuple[float, float] = (0.85, 1.15),
        gaussian_noise_std: float = 0.015,
        motion_blur_prob: float = 0.15,
        random_erasing_prob: float = 0.25,
        mild_scale_range: tuple[float, float] = (0.92, 1.08),
    ) -> None:
        self.image_size = int(image_size)
        self.train = bool(train)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.color_jitter = float(color_jitter)
        self.gamma_range = gamma_range
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.motion_blur_prob = float(motion_blur_prob)
        self.random_erasing_prob = float(random_erasing_prob)
        self.mild_scale_range = mild_scale_range

    def __call__(self, rgb: np.ndarray) -> TransformResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected RGB HWC image, got shape {rgb.shape}")
        rgb = np.asarray(rgb, dtype=np.uint8)
        original_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
        rgb, scale, offset_xy = self._resize_pad(rgb)

        if self.train:
            rgb = self._mild_scale_crop(rgb)
            rgb = self._color_jitter(rgb)
            rgb = self._gamma(rgb)
            rgb = self._motion_blur(rgb)
            rgb = self._gaussian_noise(rgb)
            rgb = self._random_erasing(rgb)

        image = rgb.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return TransformResult(
            image=image,
            scale=scale,
            offset_xy=offset_xy,
            original_hw=original_hw,
        )

    def _resize_pad(self, rgb: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = rgb.shape[:2]
        size = self.image_size
        scale = min(size / float(h), size / float(w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        x0 = (size - new_w) // 2
        y0 = (size - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return canvas, float(scale), (int(x0), int(y0))

    def _mild_scale_crop(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            return rgb
        size = self.image_size
        factor = random.uniform(*self.mild_scale_range)
        new_size = max(1, int(round(size * factor)))
        resized = cv2.resize(rgb, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        if new_size >= size:
            max_x = new_size - size
            max_y = new_size - size
            x0 = random.randint(0, max_x) if max_x > 0 else 0
            y0 = random.randint(0, max_y) if max_y > 0 else 0
            return resized[y0 : y0 + size, x0 : x0 + size]
        canvas = np.zeros_like(rgb)
        x0 = (size - new_size) // 2
        y0 = (size - new_size) // 2
        canvas[y0 : y0 + new_size, x0 : x0 + new_size] = resized
        return canvas

    def _color_jitter(self, rgb: np.ndarray) -> np.ndarray:
        if self.color_jitter <= 0 or random.random() > 0.8:
            return rgb
        image = rgb.astype(np.float32) / 255.0
        brightness = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        contrast = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
        image = image * brightness
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast + mean
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _gamma(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            return rgb
        gamma = random.uniform(*self.gamma_range)
        image = np.clip(rgb.astype(np.float32) / 255.0, 0, 1)
        image = np.power(image, gamma)
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _gaussian_noise(self, rgb: np.ndarray) -> np.ndarray:
        if self.gaussian_noise_std <= 0 or random.random() > 0.35:
            return rgb
        image = rgb.astype(np.float32) / 255.0
        noise = np.random.normal(0.0, self.gaussian_noise_std, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _motion_blur(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() > self.motion_blur_prob:
            return rgb
        k = random.choice([3, 5])
        kernel = np.zeros((k, k), dtype=np.float32)
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k
        return cv2.filter2D(rgb, -1, kernel)

    def _random_erasing(self, rgb: np.ndarray) -> np.ndarray:
        if random.random() > self.random_erasing_prob:
            return rgb
        out = rgb.copy()
        size = self.image_size
        erase_h = random.randint(max(8, size // 24), max(12, size // 8))
        erase_w = random.randint(max(8, size // 24), max(12, size // 8))
        x0 = random.randint(0, max(0, size - erase_w))
        y0 = random.randint(0, max(0, size - erase_h))
        fill = np.asarray(out.mean(axis=(0, 1)), dtype=np.uint8)
        out[y0 : y0 + erase_h, x0 : x0 + erase_w] = fill
        return out


def read_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def denormalize_image(tensor: torch.Tensor, mean: Any = None, std: Any = None) -> np.ndarray:
    if mean is None:
        mean = (0.485, 0.456, 0.406)
    if std is None:
        std = (0.229, 0.224, 0.225)
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    arr = np.clip(arr * std_arr + mean_arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)
