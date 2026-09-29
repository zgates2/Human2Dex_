"""Utility helpers for config, JSONL, checkpoints, and reproducibility."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _remap_state_dict_prefix(state_dict: dict, src: str, dst: str) -> dict:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(src):
            remapped[dst + key[len(src) :]] = value
        else:
            remapped[key] = value
    return remapped


def load_model_state_compatible(model, state_dict: dict, strict: bool = True):
    """
    Load checkpoint weights across official/fallback DINO module wrappers.

    Some environments load DINOv3 through a wrapper whose keys are prefixed as
    `backbone.model.*`, while the local fallback exposes `backbone.*` directly.
    The tensor names and shapes are otherwise identical, so remapping the prefix
    is safe and keeps checkpoints portable across training and runtime envs.
    """
    try:
        return model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as first_error:
        attempts = (
            ("backbone.model.", "backbone."),
            ("backbone.", "backbone.model."),
        )
        errors = []
        for src, dst in attempts:
            if not any(key.startswith(src) for key in state_dict):
                continue
            remapped = _remap_state_dict_prefix(state_dict, src=src, dst=dst)
            try:
                result = model.load_state_dict(remapped, strict=strict)
                print(f"[Checkpoint] remapped state_dict prefix {src!r} -> {dst!r}")
                return result
            except RuntimeError as exc:
                errors.append(f"{src}->{dst}: {exc}")
        if errors:
            raise RuntimeError(
                f"{first_error}\n\nCompatibility remap attempts also failed:\n"
                + "\n".join(errors)
            ) from first_error
        raise


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header:
            fieldnames = header
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base).expanduser() / path
    return path.resolve()


def resolve_config_paths(cfg: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Resolve repo-relative config paths in-place."""
    path_keys = (
        ("data", "data_root"),
        ("data", "index_dir"),
        ("data", "stage2_index"),
        ("data", "stage2_fits"),
        ("model", "dino_dir"),
        ("mano", "model_dir"),
        ("projection", "intrinsics_yaml"),
        ("output", "run_dir"),
    )
    for section, key in path_keys:
        values = cfg.get(section)
        if not isinstance(values, dict):
            continue
        value = values.get(key)
        if value in (None, ""):
            continue
        values[key] = str(resolve_path(value, base=repo_root))
    return cfg


def now_run_name(prefix: str = "dinov3_vits16") -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class WarmupCosineScheduler:
    """Small per-step warmup + cosine scheduler without extra dependencies."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.step_idx = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self) -> None:
        self.step_idx += 1
        if self.warmup_steps > 0 and self.step_idx <= self.warmup_steps:
            scale = self.step_idx / float(self.warmup_steps)
        else:
            denom = max(1, self.total_steps - self.warmup_steps)
            progress = (self.step_idx - self.warmup_steps) / float(denom)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        for lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr * scale


def worker_init_fn(worker_id: int) -> None:
    seed = (os.getpid() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
