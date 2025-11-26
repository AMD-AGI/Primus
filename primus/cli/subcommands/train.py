###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.launcher.train_launcher import add_train_parser, launch_train


def run(args, overrides):
    """
    Entry point for the 'train' subcommand.
    """
    from primus.tools.utils import finalize_distributed, init_distributed

    init_distributed()
    launch_train(args, overrides, module=args.suite)
    finalize_distributed()


def register_subcommand(subparsers):
    """
    Register the 'train' subcommand to the main CLI parser.

    Supported training workflows:
        - pretrain: Pre-training from scratch (language modeling)

    Future extensions:
        - sft: Supervised fine-tuning (instruction tuning, task adaptation)
        - posttrain: General post-training workflows

    Examples:
        # Pre-training
        primus train pretrain --config pretrain.yaml --backend-path /path/to/megatron

    Args:
        subparsers: argparse subparsers object from main.py

    Returns:
        parser: The parser for this subcommand
    """

    parser = subparsers.add_parser(
        "train",
        help="Launch Primus training workflows (pretrain, sft, etc.)",
        description=(
            "Unified training entry for Primus. "
            "Supports multiple training workflows with various backends "
            "(Megatron, TorchTitan, MaxText, etc.)."
        ),
    )

    # Create subparsers for different training workflows
    suite_parsers = parser.add_subparsers(dest="suite", required=True, help="Training workflow type")

    # ---------- pretrain ----------
    pretrain = suite_parsers.add_parser(
        "pretrain",
        help="Pre-training workflow (language modeling from scratch)",
        description=(
            "Pre-train large language models from scratch using "
            "next-token prediction objective. Supports distributed training "
            "with various parallelism strategies (DP, TP, PP, CP)."
        ),
    )
    add_train_parser(pretrain)

    # ---------- sft ----------
    # sft = suite_parsers.add_parser(
    #     "sft",
    #     help="Supervised fine-tuning workflow (instruction tuning)",
    #     description=(
    #         "Fine-tune pre-trained models on supervised datasets for "
    #         "instruction following, task adaptation, or domain specialization. "
    #         "Supports full fine-tuning and parameter-efficient methods (LoRA, etc.)."
    #     ),
    # )
    # add_train_parser(sft)

    parser.set_defaults(func=run)

    return parser
