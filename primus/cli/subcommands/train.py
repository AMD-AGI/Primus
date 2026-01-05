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
        # Default is "legacy" to keep current training workflows unchanged.
        from os import getenv

        runtime_entry = getenv("PRIMUS_TRAIN_RUNTIME", "legacy").strip().lower()

        if runtime_entry == "core":
            # New core runtime path: mirror `train_launcher.launch_train`.
            from primus.core.runtime.train_runtime import PrimusRuntime

            runtime = PrimusRuntime(args=args)
            runtime.run_train_module(module_name="pre_trainer", overrides=overrides or [])
        else:
            # Legacy pretrain flow.
            from primus.pretrain import launch_pretrain_from_cli

            launch_pretrain_from_cli(args, overrides)
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

    parser.set_defaults(func=run)

    return parser
