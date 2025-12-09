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
from primus.modules.module_utils import log_rank_0


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
        log_rank_0("[Patch:megatron.args.profile_tensorboard] Enabled TensorBoard (profile=True)")


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
            log_rank_0(
                f"[Patch:megatron.args.checkpoint_path][WARN] args.save is deprecated; overriding to: {ckpt_path}"
            )

        args.save = ckpt_path
        log_rank_0(f"[Patch:megatron.args.checkpoint_path] save → {ckpt_path}")


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
                log_rank_0(
                    f"[Patch:megatron.args.tensorboard_path][WARN] args.tensorboard_dir is deprecated; "
                    f"overriding to: {tb_path}"
                )

            args.tensorboard_dir = tb_path
            log_rank_0(f"[Patch:megatron.args.tensorboard_path] tensorboard_dir → {tb_path}")
        else:
            args.tensorboard_dir = None
            log_rank_0("[Patch:megatron.args.tensorboard_path] TensorBoard disabled")


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
        log_rank_0("[Patch:megatron.args.wandb_config] W&B disabled (disable_wandb=True)")
        return

    # Set W&B save directory (dedicated 'wandb' subdirectory under experiment root)
    wandb_path = os.path.join(exp_root, "wandb")
    if hasattr(args, "wandb_save_dir") and args.wandb_save_dir is not None:
        if args.wandb_save_dir != wandb_path:
            log_rank_0(
                f"[Patch:megatron.args.wandb_config][WARN] args.wandb_save_dir is deprecated; "
                f"overriding to: {wandb_path}"
            )
    args.wandb_save_dir = wandb_path

    # Set W&B project name
    if not hasattr(args, "wandb_project") or args.wandb_project is None:
        args.wandb_project = f"{work_group}_{user_name}"

    # Set W&B experiment name
    if not hasattr(args, "wandb_exp_name") or args.wandb_exp_name is None:
        args.wandb_exp_name = exp_name

    # Check for W&B API key
    if "WANDB_API_KEY" not in os.environ:
        log_rank_0(
            "[Patch:megatron.args.wandb_config][WARN] WANDB_API_KEY not set; "
            "set it before training or enable 'disable_wandb' in config."
        )

    entity = getattr(args, "wandb_entity", None)
    log_rank_0(
        "[Patch:megatron.args.wandb_config] "
        f"project={args.wandb_project!r}, exp_name={args.wandb_exp_name!r}, "
        f"save_dir={args.wandb_save_dir!r}, entity={entity!r}"
    )


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
        log_rank_0(
            f"[Patch:megatron.args.logging_level][WARN] Invalid stderr_sink_level '{stderr_level}', using INFO"
        )
        stderr_level = "INFO"

    logging_level = level_map[stderr_level]

    if hasattr(args, "logging_level") and args.logging_level is not None:
        if args.logging_level != logging_level:
            log_rank_0(
                "[Patch:megatron.args.logging_level][WARN] args.logging_level is deprecated; "
                f"setting to {logging_level} (from stderr_sink_level={stderr_level})"
            )

    args.logging_level = logging_level
    log_rank_0(f"[Patch:megatron.args.logging_level] logging_level={logging_level} ({stderr_level})")


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
            if path is None:
                return
            if isinstance(path, str):
                path_list = path.split()
                setattr(args, path_attr, path_list)
                log_rank_0(f"[Patch:megatron.args.data_path_split] {path_attr} → {path_list}")
            elif isinstance(path, list):
                # Already in list form; log for visibility but do not modify.
                log_rank_0(
                    f"[Patch:megatron.args.data_path_split] {path_attr} already list; keeping value: {path}"
                )
            else:
                # Unexpected type; log a warning to aid debugging.
                log_rank_0(
                    f"[Patch:megatron.args.data_path_split][WARN] {path_attr} has unsupported type "
                    f"{type(path).__name__}; value left unchanged: {path}"
                )

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
        log_rank_0("[Patch:megatron.args.mock_data] Mock data enabled; all data paths set to None")
