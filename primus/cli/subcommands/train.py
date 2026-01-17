###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def run(args, overrides):
    """
    Entry point for the 'train' subcommand.
    """
    if args.suite == "pretrain":
        # Select which training entry to use.
        #
        # To avoid changing the existing CLI surface, the choice between the
        # legacy and core runtime is controlled via an environment variable:
        #
        #   PRIMUS_TRAIN_RUNTIME = "legacy" | "core"
        #
        # Priority:
        #   1) Explicit env override via PRIMUS_TRAIN_RUNTIME
        #   2) If not set, auto-select default by backend framework:
        #        - TorchTitan  -> "core" (new runtime)
        #        - others      -> "legacy" (keep existing behavior)
        from os import getenv

        # 1) Explicit env override (highest priority)
        runtime_entry = getenv("PRIMUS_TRAIN_RUNTIME", "").strip().lower()

        if runtime_entry not in ("legacy", "core"):
            # 2) Auto-detect framework from exp config to choose a sensible default
            #    without requiring users to set PRIMUS_TRAIN_RUNTIME explicitly.
            try:
                from primus.core.utils import yaml_utils

                exp_cfg = yaml_utils.parse_yaml_to_namespace(args.config)
                modules_cfg = getattr(exp_cfg, "modules", None)
                pre_trainer_cfg = (
                    getattr(modules_cfg, "pre_trainer", None) if modules_cfg is not None else None
                )
                framework = (
                    getattr(pre_trainer_cfg, "framework", None) if pre_trainer_cfg is not None else None
                )
            except Exception:
                framework = None

            if framework == "torchtitan":
                runtime_entry = "core"
            else:
                runtime_entry = "legacy"

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
        from primus.core.utils.constant_vars import SFT_TRAINER

        runtime = PrimusRuntime(args=args)
        runtime.run_train_module(module_name=SFT_TRAINER, overrides=overrides or [])
    else:
        raise NotImplementedError(f"Unsupported train suite: {args.suite}")


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
        description="Primus training entry. Supports pretrain now; posttrain/finetune/evaluate reserved for future use.",
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
