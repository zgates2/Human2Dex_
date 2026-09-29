#!/usr/bin/env python3
"""Append an operator outcome annotation beside a recorded inference episode.

The original PKL and images are never modified.  Repeated annotations are
preserved in ``episode_outcome.json`` so corrections remain auditable.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import pickle
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTCOMES = (
    "success_lift",
    "arm_misaligned",
    "hand_closed_early",
    "hand_closed_late",
    "insufficient_closure",
    "object_slipped",
    "collision_or_safety_stop",
    "operator_stop_before_outcome",
)
TRISTATE = ("yes", "no", "unknown")


def find_episode_pkl(episode_dir: Path) -> Path:
    expected = episode_dir / f"{episode_dir.name}.pkl"
    if expected.is_file():
        return expected
    candidates = sorted(episode_dir.glob("*.pkl"))
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected exactly one episode PKL in {episode_dir}, found {len(candidates)}"
        )
    return candidates[0]


def read_episode_summary(pkl_path: Path) -> dict[str, Any]:
    with pkl_path.open("rb") as stream:
        payload = pickle.load(stream)
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return {
        "pkl": str(pkl_path),
        "formatVersion": payload.get("formatVersion") if isinstance(payload, dict) else None,
        "frameCount": len(messages),
        "checkpoint": metadata.get("checkpoint"),
        "finishReason": metadata.get("recording", {}).get("finishReason"),
        "provenance": metadata.get("provenance"),
    }


def load_sidecar(path: Path, episode_dir: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "formatVersion": 1,
            "episode": episode_dir.name,
            "annotations": [],
        }
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("episode") != episode_dir.name:
        raise SystemExit(
            f"Outcome sidecar episode mismatch: {data.get('episode')} vs {episode_dir.name}"
        )
    data.setdefault("annotations", [])
    return data


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--outcome", choices=OUTCOMES, required=True)
    parser.add_argument("--sponge-moved", choices=TRISTATE, default="unknown")
    parser.add_argument("--pinched", choices=TRISTATE, default="unknown")
    parser.add_argument("--lifted", choices=TRISTATE, default="unknown")
    parser.add_argument("--retained-two-seconds", choices=TRISTATE, default="unknown")
    parser.add_argument("--operator", default=getpass.getuser())
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir = args.episode_dir.expanduser().resolve()
    if not episode_dir.is_dir():
        raise SystemExit(f"Episode directory not found: {episode_dir}")
    pkl_path = find_episode_pkl(episode_dir)
    episode_summary = read_episode_summary(pkl_path)
    sidecar_path = episode_dir / "episode_outcome.json"
    sidecar = load_sidecar(sidecar_path, episode_dir)
    annotation = {
        "timestampUtc": datetime.now(timezone.utc).isoformat(),
        "operator": args.operator,
        "annotationHost": socket.gethostname(),
        "outcome": args.outcome,
        "observations": {
            "spongeMoved": args.sponge_moved,
            "pinched": args.pinched,
            "lifted": args.lifted,
            "retainedTwoSeconds": args.retained_two_seconds,
        },
        "notes": args.notes,
        "episodeSummary": episode_summary,
    }
    sidecar["annotations"].append(annotation)
    sidecar["latest"] = annotation
    write_json_atomic(sidecar_path, sidecar)
    print(json.dumps(sidecar, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
