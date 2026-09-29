#!/usr/bin/env python3
"""Franka inference with a 21x3 MANO-keypoint hand action head."""

from __future__ import annotations

from eval_real_franka_o6 import (
    config_from_args,
    run_inference,
    run_timing_self_test,
)
from eval_real_franka_o6 import build_arg_parser as _build_arg_parser


def build_arg_parser(description: str = "Franka + 21x3 mono关键点灵巧手推理脚本"):
    """Keep the established CLI surface while selecting the pts21 config."""
    parser = _build_arg_parser(description=description)
    parser.set_defaults(config="eval_franka_pts21_config.yaml")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.timing_self_test:
        run_timing_self_test()
        return
    cfg = config_from_args(args)
    run_inference(cfg)


if __name__ == "__main__":
    main()
