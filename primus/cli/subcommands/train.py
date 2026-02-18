###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def run(args, overrides):
    """
    Entry point for the 'train' subcommand.
    """
    print(f"[PRIMUS-TRAIN] run() entered: suite={args.suite}", flush=True)
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
        print(f"[PRIMUS-TRAIN] PRIMUS_TRAIN_RUNTIME env={runtime_entry!r}", flush=True)

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

            print(f"[PRIMUS-TRAIN] auto-detected framework={framework}", flush=True)
            if framework == "torchtitan":
                runtime_entry = "core"
            else:
                runtime_entry = "legacy"

        print(f"[PRIMUS-TRAIN] selected runtime_entry={runtime_entry!r}", flush=True)

        if runtime_entry == "core":
            # New core runtime path: mirror `train_launcher.launch_train`.
            print("[PRIMUS-TRAIN] importing PrimusRuntime (core runtime)...", flush=True)
            from primus.core.runtime.train_runtime import PrimusRuntime

            runtime = PrimusRuntime(args=args)
            print("[PRIMUS-TRAIN] PrimusRuntime created, calling run_train_module()...", flush=True)
            runtime.run_train_module(module_name="pre_trainer", overrides=overrides or [])
            print("[PRIMUS-TRAIN] PrimusRuntime.run_train_module() completed", flush=True)
        else:
            # Legacy pretrain flow.
            print("[PRIMUS-TRAIN] importing launch_pretrain_from_cli (legacy flow)...", flush=True)
            from primus.pretrain import launch_pretrain_from_cli

            print("[PRIMUS-TRAIN] calling launch_pretrain_from_cli()...", flush=True)
            launch_pretrain_from_cli(args, overrides)
            print("[PRIMUS-TRAIN] launch_pretrain_from_cli() completed", flush=True)
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
