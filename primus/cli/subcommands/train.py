###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import sys
from os import getenv
from typing import List


def _resolve_pretrain_runtime(args) -> str:
    """
    Resolve the runtime entry for pretrain.

    Priority:
      1) Explicit env override via PRIMUS_TRAIN_RUNTIME
      2) Auto-detect by backend framework (TorchTitan -> core, others -> legacy)
    """
    runtime_entry = getenv("PRIMUS_TRAIN_RUNTIME", "").strip().lower()
    if runtime_entry in ("legacy", "core"):
        return runtime_entry
    if runtime_entry:
        print(
            f"[Primus:Train] Ignoring invalid PRIMUS_TRAIN_RUNTIME='{runtime_entry}'.",
            file=sys.stderr,
        )

    try:
        from primus.core.utils import yaml_utils

        exp_cfg = yaml_utils.parse_yaml_to_namespace(args.config)
        modules_cfg = getattr(exp_cfg, "modules", None)
        pre_trainer_cfg = getattr(modules_cfg, "pre_trainer", None) if modules_cfg is not None else None
        framework = getattr(pre_trainer_cfg, "framework", None) if pre_trainer_cfg is not None else None
    except Exception:
        framework = None

    return "core" if framework == "torchtitan" else "legacy"


def run(args, overrides: List[str]):
    """
    Entry point for the 'train' subcommand.
    """
    if args.suite == "pretrain":
        runtime_entry = _resolve_pretrain_runtime(args)

        if runtime_entry == "core":
            # New core runtime path: mirror `train_launcher.launch_train`.
            from primus.core.runtime.train_runtime import PrimusRuntime

            runtime = PrimusRuntime(args=args)
            runtime.run_train_module(module_name="pre_trainer", overrides=overrides or [])
        else:
            # Legacy pretrain flow.
            from primus.pretrain import launch_pretrain_from_cli

            launch_pretrain_from_cli(args, overrides)
    elif args.suite == "posttrain":
        # Post-training (SFT/alignment) currently runs via the new core runtime.
        # It expects a training module named "sft_trainer" in the experiment config.
        from primus.core.runtime.train_runtime import PrimusRuntime

        # from primus.core.utils.constant_vars import SFT_TRAINER

        runtime = PrimusRuntime(args=args)
        runtime.run_train_module(module_name="post_trainer", overrides=overrides or [])
    else:
        raise NotImplementedError(
            f"Unsupported train suite: {args.suite}. " "Expected one of: pretrain, posttrain."
        )


def register_subcommand(subparsers):
    """
    Register the 'train' subcommand to the main CLI parser.

    Supported suites (training workflows):
        - pretrain: Pre-training workflow (Megatron, TorchTitan, etc.)

    Future extensions:
        - posttrain: Post-training workflow (alignment, preference tuning, etc.)

    Example:
        primus train pretrain --config exp.yaml --backend-path /path/to/megatron

    Args:
        subparsers: argparse subparsers object from main.py

    Returns:
        parser: The parser for this subcommand
    """

    parser = subparsers.add_parser(
        "train",
        help="Launch Primus pretrain with Megatron or TorchTitan",
        description="Primus training entry. Supports pretrain and posttrain (SFT); finetune/evaluate reserved for future use.",
    )
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- pretrain ----------
    pretrain = suite_parsers.add_parser("pretrain", help="Pre-training workflow.")
    from primus.core.launcher.parser import add_pretrain_parser

    add_pretrain_parser(pretrain)

    # ---------- posttrain ----------
    posttrain = suite_parsers.add_parser("posttrain", help="Post-training workflow (SFT/alignment).")
    from primus.core.launcher.parser import add_posttrain_parser

    add_posttrain_parser(posttrain)

    parser.set_defaults(func=run)

    return parser
