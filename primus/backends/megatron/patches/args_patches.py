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
from primus.modules.module_utils import log_kv_rank_0, log_rank_0, warning_rank_0


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
    args = ctx.extra.get("backend_args", {})
    primus_config = ctx.extra.get("primus_config", {})

    if args and primus_config.exp_root_path:
        ckpt_path = os.path.abspath(os.path.join(primus_config.exp_root_path, "checkpoints"))

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

    args = ctx.extra.get("backend_args", {})
    primus_config = ctx.extra.get("primus_config", {})
    module_config = ctx.extra.get("module_config", {})

    disable_tensorboard = getattr(module_config.params, "disable_tensorboard", False)

    if args and getattr(args, "profile", False):
        disable_tensorboard = False
        log_rank_0("[Patch:megatron.args.profile_tensorboard] Enabled TensorBoard (profile=True)")

    exp_root_path = primus_config.exp_root_path
    if args and exp_root_path:
        if not disable_tensorboard:
            tb_path = os.path.abspath(os.path.join(exp_root_path, "tensorboard"))
            args.tensorboard_dir = tb_path
            log_rank_0(f"[Patch:megatron.args.tensorboard_path] tensorboard_dir → {tb_path}")
        else:
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
    args = ctx.extra.get("backend_args", {})
    module_config = ctx.extra.get("module_config", {})
    primus_config = ctx.extra.get("primus_config", {})

    # Get Primus metadata from config (injected by PrimusConfig)
    exp_root_path = primus_config.exp_root_path
    work_group = primus_config.exp_meta_info["work_group"]
    user_name = primus_config.exp_meta_info["user_name"]
    exp_name = primus_config.exp_meta_info["exp_name"]

    if not args or not exp_root_path:
        return

    # Check if W&B is enabled
    disable_wandb = getattr(module_config.params, "disable_wandb", False)
    if not disable_wandb:
        # Set W&B save directory (dedicated 'wandb' subdirectory under experiment root)
        wandb_path = os.path.join(exp_root_path, "wandb")
        if hasattr(args, "wandb_save_dir") and args.wandb_save_dir is not None:
            warning_rank_0(
                f"[Patch:megatron.args.wandb_config] args.wandb_save_dir is deprecated; overriding to: {wandb_path}"
            )

        log_rank_0(f"[Patch:megatron.args.wandb_config] wandb_save_dir → {wandb_path}")
        args.wandb_save_dir = wandb_path

        # Set W&B project name
        if not hasattr(args, "wandb_project") or args.wandb_project is None:
            args.wandb_project = f"{work_group}_{user_name}"
            log_rank_0(f"[Patch:megatron.args.wandb_config] wandb_project → {args.wandb_project}")

        # Set W&B experiment name
        if not hasattr(args, "wandb_exp_name") or args.wandb_exp_name is None:
            args.wandb_exp_name = exp_name
            log_rank_0(f"[Patch:megatron.args.wandb_config] wandb_exp_name → {args.wandb_exp_name}")
    else:
        if hasattr(args, "wandb_project") and args.wandb_project is not None:
            args.wandb_project = None

    if not disable_wandb and "WANDB_API_KEY" not in os.environ:
        warning_rank_0(
            "The environment variable WANDB_API_KEY is not set. "
            "Please set it before proceeding or enable 'disable_wandb' in yaml config"
        )
    log_kv_rank_0(f"[Patch:megatron.args.wandb_config] -disable_wandb", f"{disable_wandb}")
    log_kv_rank_0(f"[Patch:megatron.args.wandb_config]   -wandb_project", f"{args.wandb_project}")
    log_kv_rank_0(f"[Patch:megatron.args.wandb_config]   -wandb_exp_name", f"{args.wandb_exp_name}")
    log_kv_rank_0(f"[Patch:megatron.args.wandb_config]   -wandb_save_dir", f"{args.wandb_save_dir}")
    log_kv_rank_0(f"[Patch:megatron.args.wandb_config]   -wandb_entity", f"{args.wandb_entity}")


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
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    data_path = getattr(args, "data_path", None)
    train_data_path = getattr(args, "train_data_path", None)
    valid_data_path = getattr(args, "valid_data_path", None)
    test_data_path = getattr(args, "test_data_path", None)

    if data_path is not None:
        args.data_path = data_path.split(" ")
        log_kv_rank_0(f"[Patch:megatron.args.data_path_split]   -data_path", f"{args.data_path}")
    if train_data_path is not None:
        args.train_data_path = train_data_path.split(" ")
        log_kv_rank_0(f"[Patch:megatron.args.data_path_split]   -train_data_path", f"{args.train_data_path}")
    if valid_data_path is not None:
        args.valid_data_path = valid_data_path.split(" ")
        log_kv_rank_0(f"[Patch:megatron.args.data_path_split]   -valid_data_path", f"{args.valid_data_path}")
    if test_data_path is not None:
        args.test_data_path = test_data_path.split(" ")
        log_kv_rank_0(f"[Patch:megatron.args.data_path_split]   -test_data_path", f"{args.test_data_path}")


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
    args = ctx.extra.get("backend_args", {})
    module_config = ctx.extra.get("module_config", {})

    if not args or not module_config:
        return

    mock_data = getattr(module_config.params, "mock_data", False)
    if mock_data:
        args.data_path = None
        args.train_data_path = None
        args.valid_data_path = None
        args.test_data_path = None
        log_rank_0(f"[Patch:megatron.args.mock_data] Mock data enabled; all data paths set to None")

    log_kv_rank_0(f"[Patch:megatron.args.mock_data] -mock_data", f"{mock_data}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -data_path", f"{args.data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -train_data_path", f"{args.train_data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -valid_data_path", f"{args.valid_data_path}")
    log_kv_rank_0(f"[Patch:megatron.args.mock_data]   -test_data_path", f"{args.test_data_path}")


@register_patch(
    "megatron.args.sequence_parallel_tp1",
    backend="megatron",
    phase="build_args",
    description="Disable sequence_parallel when tensor_model_parallel_size == 1",
)
def patch_sequence_parallel_tp1(ctx: PatchContext):
    """
    Align sequence_parallel behavior with trainer defaults:

        if args.tensor_model_parallel_size == 1:
            args.sequence_parallel = False
    """
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    tp_size = getattr(args, "tensor_model_parallel_size", None)
    if tp_size == 1:
        # Only log when we actually change the flag.
        if getattr(args, "sequence_parallel", None):
            log_rank_0(
                "[Patch:megatron.args.sequence_parallel_tp1] "
                "sequence_parallel=True is incompatible with tp_size=1; forcing to False."
            )
        args.sequence_parallel = False
        log_kv_rank_0(
            "[Patch:megatron.args.sequence_parallel_tp1] -sequence_parallel", f"{args.sequence_parallel}"
        )


@register_patch(
    "megatron.args.iterations_to_skip_default",
    backend="megatron",
    phase="build_args",
    description="Ensure iterations_to_skip has a list default instead of None",
)
def patch_iterations_to_skip_default(ctx: PatchContext):
    """
    Align iterations_to_skip behavior with trainer defaults:

        if args.iterations_to_skip is None:
            args.iterations_to_skip = []
    """
    args = ctx.extra.get("backend_args", {})
    if not args:
        return

    if getattr(args, "iterations_to_skip", None) is None:
        args.iterations_to_skip = []
        log_kv_rank_0(
            "[Patch:megatron.args.iterations_to_skip_default] -iterations_to_skip",
            f"{args.iterations_to_skip}",
        )
