###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Arguments Patches

Patches Megatron arguments for Primus-specific configurations:
- Checkpoint and logging paths
- TensorBoard and W&B integration
- Data path preprocessing
- Logging level configuration
"""

import os

from primus.core.patches import PatchContext, register_patch


@register_patch(
    "megatron.args.profile_tensorboard",
    backend="megatron",
    phase="build_args",
    description="Enable TensorBoard when profiling is enabled",
)
def patch_profile_tensorboard(ctx: PatchContext):
    """
    Ensure TensorBoard is enabled when profiling is active.

    Profiling requires TensorBoard for visualization, so we force
    disable_tensorboard=False when profile=True.
    """
    args = ctx.extra.get("args")
    if args and getattr(args, "profile", False):
        args.disable_tensorboard = False
        print("[Patch] Enabled TensorBoard for profiling (profile=True)")


@register_patch(
    "megatron.args.checkpoint_path",
    backend="megatron",
    phase="build_args",
    description="Set checkpoint save path based on experiment root",
)
def patch_checkpoint_path(ctx: PatchContext):
    """
    Configure checkpoint save path.

    Sets args.save to <exp_root>/checkpoints and warns if user
    provided a different path.
    """
    args = ctx.extra.get("args")
    config = ctx.extra.get("config", {})

    # Get exp_root_path from config (injected by PrimusConfig)
    exp_root = config.get("primus_exp_root_path")

    if args and exp_root:
        ckpt_path = os.path.abspath(os.path.join(exp_root, "checkpoints"))

        if hasattr(args, "save") and args.save is not None and args.save != ckpt_path:
            print(f"[Patch] WARNING: args.save is deprecated, using: {ckpt_path}")

        args.save = ckpt_path
        print(f"[Patch] Checkpoint path: {ckpt_path}")


@register_patch(
    "megatron.args.tensorboard_path",
    backend="megatron",
    phase="build_args",
    description="Set TensorBoard directory based on experiment root",
)
def patch_tensorboard_path(ctx: PatchContext):
    """
    Configure TensorBoard directory.

    Sets args.tensorboard_dir to <exp_root>/tensorboard if TensorBoard
    is enabled, otherwise sets it to None.
    """
    args = ctx.extra.get("args")
    config = ctx.extra.get("config", {})

    # Get exp_root_path from config (injected by PrimusConfig)
    exp_root = config.get("primus_exp_root_path")

    if args and exp_root:
        if not getattr(args, "disable_tensorboard", False):
            tb_path = os.path.abspath(os.path.join(exp_root, "tensorboard"))

            if (
                hasattr(args, "tensorboard_dir")
                and args.tensorboard_dir is not None
                and args.tensorboard_dir != tb_path
            ):
                print(f"[Patch] WARNING: args.tensorboard_dir is deprecated, using: {tb_path}")

            args.tensorboard_dir = tb_path
            print(f"[Patch] TensorBoard directory: {tb_path}")
        else:
            args.tensorboard_dir = None
            print("[Patch] TensorBoard disabled")


@register_patch(
    "megatron.args.wandb_config",
    backend="megatron",
    phase="build_args",
    description="Configure W&B project and experiment names",
)
def patch_wandb_config(ctx: PatchContext):
    """
    Configure Weights & Biases (W&B) integration.

    Sets up W&B project name, experiment name, and save directory
    based on Primus experiment metadata.
    """
    args = ctx.extra.get("args")
    config = ctx.extra.get("config", {})

    # Get Primus metadata from config (injected by PrimusConfig)
    exp_root = config.get("primus_exp_root_path")
    work_group = config.get("primus_work_group", "default")
    user_name = config.get("primus_user_name", "user")
    exp_name = config.get("primus_exp_name", "experiment")

    if not args or not exp_root:
        return

    # Check if W&B is enabled
    if getattr(args, "disable_wandb", False):
        # Ensure wandb_project is None if W&B is disabled
        if hasattr(args, "wandb_project") and args.wandb_project is not None:
            args.wandb_project = None
            print("[Patch] Disabled W&B project (disable_wandb=True)")
        return

    # Set W&B save directory
    wandb_path = exp_root
    if hasattr(args, "wandb_save_dir") and args.wandb_save_dir is not None:
        if args.wandb_save_dir != wandb_path:
            print(f"[Patch] WARNING: args.wandb_save_dir is deprecated, using: {wandb_path}/wandb")
    args.wandb_save_dir = wandb_path

    # Set W&B project name
    if not hasattr(args, "wandb_project") or args.wandb_project is None:
        args.wandb_project = f"{work_group}_{user_name}"
        print(f"[Patch] Created W&B project name: {args.wandb_project}")

    # Set W&B experiment name
    if not hasattr(args, "wandb_exp_name") or args.wandb_exp_name is None:
        args.wandb_exp_name = exp_name
        print(f"[Patch] Created W&B experiment name: {args.wandb_exp_name}")

    # Check for W&B API key
    if "WANDB_API_KEY" not in os.environ:
        print(
            "[Patch] WARNING: WANDB_API_KEY not set. "
            "Set it before training or enable 'disable_wandb' in config."
        )

    print(f"[Patch] W&B configuration:")
    print(f"  - project: {args.wandb_project}")
    print(f"  - exp_name: {args.wandb_exp_name}")
    print(f"  - save_dir: {args.wandb_save_dir}")
    if hasattr(args, "wandb_entity"):
        print(f"  - entity: {args.wandb_entity}")


@register_patch(
    "megatron.args.logging_level",
    backend="megatron",
    phase="build_args",
    description="Set logging level based on stderr_sink_level",
)
def patch_logging_level(ctx: PatchContext):
    """
    Configure logging level.

    Maps stderr_sink_level (DEBUG/INFO/WARNING/ERROR) to numeric
    logging level and sets args.logging_level accordingly.
    """
    args = ctx.extra.get("args")
    if not args:
        return

    level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    stderr_level = getattr(args, "stderr_sink_level", "INFO")
    if stderr_level not in level_map:
        print(f"[Patch] WARNING: Invalid stderr_sink_level '{stderr_level}', using INFO")
        stderr_level = "INFO"

    logging_level = level_map[stderr_level]

    if hasattr(args, "logging_level") and args.logging_level is not None:
        if args.logging_level != logging_level:
            print(
                f"[Patch] WARNING: args.logging_level is deprecated, "
                f"setting to {logging_level} (from stderr_sink_level={stderr_level})"
            )

    args.logging_level = logging_level
    print(f"[Patch] Logging level: {logging_level} ({stderr_level})")


@register_patch(
    "megatron.args.data_path_split",
    backend="megatron",
    phase="build_args",
    description="Split space-separated data paths into lists",
)
def patch_data_path_split(ctx: PatchContext):
    """
    Preprocess data paths.

    Converts space-separated data path strings into lists:
    "data1 data2 data3" -> ["data1", "data2", "data3"]

    Applies to:
    - data_path
    - train_data_path
    - valid_data_path
    - test_data_path
    """
    args = ctx.extra.get("args")
    if not args:
        return

    # Helper function to split path
    def split_path(path_attr):
        if hasattr(args, path_attr):
            path = getattr(args, path_attr)
            if path is not None and isinstance(path, str):
                path_list = path.split()
                setattr(args, path_attr, path_list)
                print(f"[Patch] Split {path_attr}: {path_list}")

    # Split all data paths
    split_path("data_path")
    split_path("train_data_path")
    split_path("valid_data_path")
    split_path("test_data_path")


@register_patch(
    "megatron.args.mock_data",
    backend="megatron",
    phase="build_args",
    description="Disable data paths when using mock data",
)
def patch_mock_data(ctx: PatchContext):
    """
    Handle mock data mode.

    When mock_data=True, sets all data paths to None to prevent
    Megatron from trying to load actual data files.
    """
    args = ctx.extra.get("args")
    if not args:
        return

    if getattr(args, "mock_data", False):
        args.data_path = None
        args.train_data_path = None
        args.valid_data_path = None
        args.test_data_path = None
        print("[Patch] Mock data enabled, disabled all data paths")
