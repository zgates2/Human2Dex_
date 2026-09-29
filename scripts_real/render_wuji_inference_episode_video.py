#!/usr/bin/env python3
"""Convenience entry point for Wuji inference-episode QC videos."""

from __future__ import annotations

from render_o6_inference_episode_video import (
    DEFAULT_OBJECT_PROMPTS,
    build_parser,
    run,
)


def main() -> int:
    parser = build_parser()
    parser.set_defaults(hand="wuji", episode_dir=None)
    args = parser.parse_args()
    if args.episode_dir is None:
        parser.error("the Wuji wrapper requires --episode-dir")
    if args.object_prompt is None:
        args.object_prompt = list(DEFAULT_OBJECT_PROMPTS)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
