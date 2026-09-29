"""manotorch MANO wrapper for Stage 2."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ManoDependencyError(RuntimeError):
    """Raised when manotorch or MANO assets are unavailable."""


def resolve_mano_assets_root(model_dir: str | Path, side: str = "right") -> Path:
    raw = Path(model_dir).expanduser()
    side_name = f"MANO_{side.upper()}.pkl"
    candidates = [raw, raw / "models", raw.parent if raw.name == "models" else raw]
    checked = []
    seen = set()
    for candidate in candidates:
        models_dir = candidate if candidate.name == "models" else candidate / "models"
        if models_dir in seen:
            continue
        seen.add(models_dir)
        checked.append(models_dir / side_name)
        if (models_dir / side_name).is_file():
            return models_dir.parent.resolve()
    checked_text = "\n  ".join(str(path) for path in checked)
    raise ManoDependencyError(
        "MANO model files not found. Expected MANO_RIGHT.pkl and/or "
        f"MANO_LEFT.pkl under a models/ directory. Checked:\n  {checked_text}\n"
        "Download MANO model files from https://mano.is.tue.mpg.de according "
        "to the MANO license, then set mano.model_dir in the Stage2 config."
    )


def import_manotorch_layer():
    # chumpy is an old MANO dependency and still references APIs removed from
    # recent Python / NumPy releases. Patch only the missing aliases before the
    # import so runtime code can stay on the current pico environment.
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    numpy_aliases = {
        "bool": np.bool_,
        "int": np.int_,
        "float": np.float64,
        "complex": np.complex128,
        "object": np.object_,
        "str": np.str_,
        "unicode": np.str_,
    }
    for name, value in numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    try:
        from manotorch.manolayer import ManoLayer
    except ImportError as exc:
        raise ManoDependencyError(
            "Stage2 MANO backend requires manotorch, but it is not installed. "
            "Install manotorch before running MANO fitting or Stage2 training."
        ) from exc
    return ManoLayer


class ManoTorchLayer(nn.Module):
    """
    Thin wrapper around manotorch.manolayer.ManoLayer.

    Input hand_pose is MANO axis-angle without global orient, shape [B, 45].
    Output joints and vertices are wrist-centered in meters.
    """

    def __init__(
        self,
        model_dir: str | Path,
        side: str = "right",
        use_pca: bool = False,
        flat_hand_mean: bool = False,
        output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if side not in ("right", "left"):
            raise ValueError(f"MANO side must be 'right' or 'left', got {side!r}")
        if use_pca:
            raise ValueError("Stage2 expects full 45D MANO hand pose; set use_pca=false.")
        ManoLayer = import_manotorch_layer()
        assets_root = resolve_mano_assets_root(model_dir, side=side)
        try:
            self.layer = ManoLayer(
                rot_mode="axisang",
                side=side,
                center_idx=0,
                mano_assets_root=str(assets_root),
                use_pca=False,
                flat_hand_mean=flat_hand_mean,
            )
        except (AssertionError, TypeError, FileNotFoundError) as exc:
            raise ManoDependencyError(
                "Failed to initialize manotorch ManoLayer. Check that your "
                "manotorch version supports ManoLayer(rot_mode, side, "
                "mano_assets_root, use_pca, flat_hand_mean) and that MANO pkl "
                f"files exist under {assets_root / 'models'}. Original error: {exc}"
            ) from exc
        self.side = side
        self.model_dir = str(model_dir)
        self.assets_root = str(assets_root)
        self.output_scale = float(output_scale)

    def forward(
        self,
        hand_pose: torch.Tensor,
        betas: torch.Tensor | None = None,
        global_orient: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hand_pose.ndim != 2:
            raise ValueError(f"hand_pose must be [B, 45] or full pose [B, 48], got {tuple(hand_pose.shape)}")
        batch = hand_pose.shape[0]
        if hand_pose.shape[1] == 48:
            if global_orient is not None:
                raise ValueError("global_orient must not be passed when hand_pose already contains full [B, 48] pose")
            full_pose = hand_pose
            global_orient = full_pose[:, :3]
            hand_pose = full_pose[:, 3:]
        elif hand_pose.shape[1] == 45:
            if global_orient is None:
                global_orient = torch.zeros(batch, 3, dtype=hand_pose.dtype, device=hand_pose.device)
            else:
                global_orient = global_orient.to(dtype=hand_pose.dtype, device=hand_pose.device)
                if global_orient.ndim != 2 or global_orient.shape != (batch, 3):
                    raise ValueError(f"global_orient must be [B, 3], got {tuple(global_orient.shape)}")
            full_pose = torch.cat([global_orient, hand_pose], dim=1)
        else:
            raise ValueError(f"hand_pose must be [B, 45] or full pose [B, 48], got {tuple(hand_pose.shape)}")
        if betas is not None:
            betas = betas.to(dtype=hand_pose.dtype, device=hand_pose.device)
            if betas.ndim != 2 or betas.shape[1] != 10:
                raise ValueError(f"betas must be [B, 10], got {tuple(betas.shape)}")
        output = self.layer(full_pose, betas)
        # manotorch returns 16 MANO joints plus 5 fingertip vertices reordered as:
        # wrist, thumb(4), index(4), middle(4), ring(4), pinky(4).
        joints21 = output.joints * self.output_scale
        if joints21.ndim != 3 or joints21.shape[1:] != (21, 3):
            raise ManoDependencyError(
                "manotorch ManoLayer did not return MediaPipe-style 21 joints. "
                f"Got output.joints shape {tuple(joints21.shape)}."
            )
        vertices = output.verts * self.output_scale
        # center_idx=0 should already make the output wrist-centered. This extra
        # subtraction makes the contract explicit across manotorch versions.
        wrist = joints21[:, :1, :]
        joints21 = joints21 - wrist
        vertices = vertices - wrist
        return {
            "vertices": vertices,
            "mano_joints": joints21,
            "joints21": joints21,
            "global_orient": global_orient,
            "hand_pose": hand_pose,
            "full_pose": getattr(output, "full_poses", full_pose),
            "mano_output_betas": getattr(output, "betas", betas),
        }


def check_mano_backend(model_dir: str | Path, side: str = "right") -> dict[str, str]:
    import_manotorch_layer()
    assets_root = resolve_mano_assets_root(model_dir, side=side)
    return {"backend": "manotorch", "assets_root": str(assets_root), "side": side}
