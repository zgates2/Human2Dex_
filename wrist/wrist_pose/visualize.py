"""Visualization helpers for RGB and 3D skeleton diagnostics."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .constants import HAND_BONES
from .model import expand20_to21
from .transforms import denormalize_image


def save_pose_figure(
    image_tensor: torch.Tensor,
    pred20: torch.Tensor,
    gt20: torch.Tensor,
    output_path: str | Path,
    attention: torch.Tensor | None = None,
    image_size: int = 448,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = denormalize_image(image_tensor)
    pred21 = expand20_to21(pred20.detach().cpu().float().unsqueeze(0))[0].numpy()
    gt21 = expand20_to21(gt20.detach().cpu().float().unsqueeze(0))[0].numpy()

    cols = 4 if attention is None else 5
    fig = plt.figure(figsize=(4.2 * cols, 4.2))
    ax_img = fig.add_subplot(1, cols, 1)
    ax_img.imshow(image)
    ax_img.set_title("RGB")
    ax_img.axis("off")

    ax_gt = fig.add_subplot(1, cols, 2, projection="3d")
    plot_hand_3d(ax_gt, gt21, "GT 3D", color="tab:blue")

    ax_pred = fig.add_subplot(1, cols, 3, projection="3d")
    plot_hand_3d(ax_pred, pred21, "Pred 3D", color="tab:orange")

    ax_both = fig.add_subplot(1, cols, 4, projection="3d")
    plot_hand_3d(ax_both, gt21, "Pred vs GT", color="tab:blue", label="GT")
    plot_hand_3d(ax_both, pred21, "Pred vs GT", color="tab:orange", label="Pred")
    ax_both.legend(loc="upper right")

    if attention is not None:
        ax_attn = fig.add_subplot(1, cols, 5)
        ax_attn.imshow(make_attention_overlay(image, attention, image_size=image_size))
        ax_attn.set_title("Attention")
        ax_attn.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_hand_3d(ax, joints21: np.ndarray, title: str, color: str, label: str | None = None) -> None:
    joints21 = np.asarray(joints21)
    for src, dst in HAND_BONES:
        xs = [joints21[src, 0], joints21[dst, 0]]
        ys = [joints21[src, 1], joints21[dst, 1]]
        zs = [joints21[src, 2], joints21[dst, 2]]
        ax.plot(xs, ys, zs, color=color, linewidth=2, label=label)
        label = None
    ax.scatter(joints21[:, 0], joints21[:, 1], joints21[:, 2], s=12, color=color)
    ax.set_title(title)
    set_equal_3d_axes(ax, joints21)
    ax.set_xlabel("x m")
    ax.set_ylabel("y m")
    ax.set_zlabel("z m")


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    points = np.asarray(points)
    center = points.mean(axis=0)
    radius = max(float(np.max(np.abs(points - center))), 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def make_attention_overlay(image: np.ndarray, attention: torch.Tensor, image_size: int = 448) -> np.ndarray:
    weights = attention.detach().cpu().float().numpy()
    patch_count = weights.shape[0]
    side = int(round(np.sqrt(patch_count)))
    if side * side != patch_count:
        return image
    heat = weights.reshape(side, side)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    heat = cv2.resize(heat, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return np.clip(0.55 * image + 0.45 * color, 0, 255).astype(np.uint8)


def save_batch_visuals(
    batch: dict,
    pred20: torch.Tensor,
    output_dir: str | Path,
    prefix: str,
    attention: torch.Tensor | None = None,
    max_items: int = 8,
    image_size: int = 448,
) -> None:
    output_dir = Path(output_dir)
    count = min(max_items, int(pred20.shape[0]))
    for i in range(count):
        attn_i = None if attention is None else attention[i]
        save_pose_figure(
            batch["image"][i],
            pred20[i],
            batch["joints20"][i],
            output_dir / f"{prefix}_{i:03d}.png",
            attention=attn_i,
            image_size=image_size,
        )


def projection_skipped_message(has_extrinsic: bool) -> str | None:
    if has_extrinsic:
        return None
    return "projection skipped: missing camera_T_hand extrinsic"
