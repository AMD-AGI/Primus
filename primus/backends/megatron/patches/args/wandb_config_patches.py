###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_kv_rank_0, log_rank_0, warning_rank_0


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
                f"[Patch:megatron.args.wandb_config] args.wandb_save_dir is deprecated; "
                f"overriding to: {wandb_path}"
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
