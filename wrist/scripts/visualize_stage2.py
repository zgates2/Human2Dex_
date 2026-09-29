#!/usr/bin/env python3
"""生成 Stage2 MANO pose 预测可视化。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
WRIST_ROOT = REPO_ROOT / "wrist"
if str(WRIST_ROOT) not in sys.path:
    sys.path.insert(0, str(WRIST_ROOT))

from wrist_pose.stage2_dataset import Stage2ManoDataset  # noqa: E402
from wrist_pose.stage2_infer import load_stage2_model  # noqa: E402
from wrist_pose.transforms import denormalize_image  # noqa: E402
from wrist_pose.visualize import plot_hand_3d  # noqa: E402


def choose_precision(requested: str) -> str:
    if requested == "fp32" or not torch.cuda.is_available():
        return "fp32"
    if requested == "bf16":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp16"


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def save_stage2_figure(image, gt21, pred21, fit21, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 4))
    ax_img = fig.add_subplot(1, 4, 1)
    ax_img.imshow(denormalize_image(image))
    ax_img.set_title("RGB")
    ax_img.axis("off")
    ax_gt = fig.add_subplot(1, 4, 2, projection="3d")
    plot_hand_3d(ax_gt, gt21, "GT pts21", color="tab:blue")
    ax_pred = fig.add_subplot(1, 4, 3, projection="3d")
    plot_hand_3d(ax_pred, pred21, "Pred MANO joints", color="tab:orange")
    ax_both = fig.add_subplot(1, 4, 4, projection="3d")
    plot_hand_3d(ax_both, gt21, "Pred/Fit/GT", color="tab:blue", label="GT")
    plot_hand_3d(ax_both, fit21, "Pred/Fit/GT", color="tab:green", label="Fit")
    plot_hand_3d(ax_both, pred21, "Pred/Fit/GT", color="tab:orange", label="Pred")
    ax_both.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, cfg = load_stage2_model(args.checkpoint, REPO_ROOT, device, config_path=args.config)
    precision = choose_precision(str(cfg["train"].get("precision", "auto")))
    index_path = args.index_path or (args.checkpoint.parent.parent / "splits" / "val.jsonl")
    dataset = Stage2ManoDataset(index_path=index_path, image_size=int(cfg["data"]["image_size"]), train=False)
    dataset = Subset(dataset, list(range(min(int(args.num_samples), len(dataset)))))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    output_dir = args.output_dir or (args.checkpoint.parent.parent / "visualizations" / "stage2")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            image = batch["image"].to(device)
            beta = batch["mano_beta"].to(device)
            with autocast_context(device, precision):
                outputs = model(image, mano_beta=beta)
            pred21 = outputs["joints21"].detach().cpu()
            for i in range(pred21.shape[0]):
                save_stage2_figure(
                    batch["image"][i],
                    batch["joints21"][i].numpy(),
                    pred21[i].numpy(),
                    batch["mano_joints21_fit"][i].numpy(),
                    output_dir / f"stage2_{batch_idx:03d}_{i:03d}.png",
                )
                saved += 1
    metadata = {"checkpoint": str(args.checkpoint), "index_path": str(index_path), "saved": saved}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
