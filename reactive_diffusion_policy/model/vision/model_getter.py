import os
from pathlib import Path

import torch
import torchvision


def main_log(message: str) -> None:
    """Print only from the main process to avoid multi-GPU spam."""
    if int(os.environ.get("RANK", "0")) == 0:
        print(message)

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """Load an R3M backbone, preferring a local checkpoint when available."""

    checkpoint_path = kwargs.pop('checkpoint_path', None)
    resolved_path = _resolve_r3m_checkpoint(checkpoint_path)

    if resolved_path is not None and resolved_path.is_file():
      #  main_log(f"[R3M] Loading local checkpoint for {name} from {resolved_path}")
        return _load_local_r3m(name=name, checkpoint_path=resolved_path, **kwargs)

    try:
        import r3m  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Local R3M checkpoint not found and the 'r3m' package is unavailable. "
            "Please install r3m or place the checkpoint at "
            f"{resolved_path if resolved_path is not None else 'r3m-18/pytorch_model.bin'}."
        ) from exc

    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def _load_local_r3m(name: str, checkpoint_path: str | os.PathLike | None = None, **kwargs):
    """Load an R3M backbone from a local checkpoint when the r3m package is unavailable."""

    func = getattr(torchvision.models, name)
    resnet = func(weights=None, **kwargs)
    resnet.fc = torch.nn.Identity()

    if checkpoint_path is None:
        checkpoint_path = _resolve_r3m_checkpoint()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"R3M checkpoint not found at {checkpoint_path}. Please download it first."
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('net', checkpoint)

    prefix = 'convnet.'
    processed_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    missing, unexpected = resnet.load_state_dict(processed_state_dict, strict=False)
    if missing or unexpected:
        main_log(f"[get_r3m] Loaded local R3M checkpoint with missing keys {missing} and unexpected keys {unexpected}.")

    main_log(f"[R3M] Finished loading {name} weights from {checkpoint_path}")
    return resnet

def _resolve_r3m_checkpoint(checkpoint_path: str | os.PathLike | None = None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / 'r3m-18' / 'pytorch_model.bin'


