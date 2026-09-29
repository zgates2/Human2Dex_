"""Shared, deliberately small helpers for the numbered Human2Dex pipeline.

The stage scripts are independently executable.  This module only centralizes
configuration loading, safe subprocess execution, and stage provenance; it
does not implement or change any model/augmentation behaviour.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline config does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Pipeline config must be a mapping: {path}")
    return expand_environment(value)


def expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {str(key): expand_environment(item) for key, item in value.items()}
    return value


def get(config: dict[str, Any], dotted: str, default: Any = None, *, required: bool = False) -> Any:
    current: Any = config
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            if required:
                raise KeyError(f"Missing required config key: {dotted}")
            return default
        current = current[key]
    if current is None and required:
        raise ValueError(f"Config key must not be null: {dotted}")
    return current


def path(config: dict[str, Any], dotted: str, *, required: bool = True) -> Path | None:
    value = get(config, dotted, required=required)
    return None if value is None else Path(str(value)).expanduser().resolve()


def python_bin(config: dict[str, Any]) -> str:
    return str(get(config, "runtime.python", sys.executable))


def repo_root(config: dict[str, Any]) -> Path:
    value = path(config, "runtime.repo_root", required=False)
    return value if value is not None else REPO_ROOT


def run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    printable = subprocess.list2cmdline([str(item) for item in command])
    print(f"$ {printable}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(cwd), check=True)


def require_new_directory(target: Path, *, resume: bool = False) -> None:
    if target.exists() and not resume:
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {target}. "
            "Use a new versioned path, or pass --resume only after verifying it is the same run."
        )


def sha256_file(path_value: str | Path | None) -> str | None:
    if path_value is None:
        return None
    file_path = Path(path_value)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(
    root: Path,
    stage: str,
    payload: dict[str, Any],
    *,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return
    directory = root / ".human2dex_pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "stage": stage,
        "created_unix_s": time.time(),
        "repo_root": str(REPO_ROOT),
        **payload,
    }
    destination = directory / f"{stage}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def comma_args(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in values)


def checkpoint_manifest(config: dict[str, Any]) -> dict[str, str | None]:
    return {
        "stage2": sha256_file(get(config, "models.stage2_checkpoint")),
        "wrist_projection": sha256_file(get(config, "models.wrist_projection_head_checkpoint")),
        "pocket": sha256_file(get(config, "models.pocket_head_checkpoint")),
        "intrinsics": sha256_file(get(config, "views.intrinsics")),
    }
