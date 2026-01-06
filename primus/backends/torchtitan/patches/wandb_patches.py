###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Weights & Biases (wandb) Patch

This patch mirrors ``TorchTitanPretrainTrainer.patch_torchtitan_wandb`` using
the generic Primus patch system so that wandb-related environment variables are
initialized in a backend-agnostic way.

Behavior:
    - Only applies when ``metrics.enable_wandb`` is True in the TorchTitan
      job/module config.
    - If ``WANDB_PROJECT`` is not set, it is derived from Primus exp_meta_info:
          Primus-Titan-Pretrain-{work_group}_{user_name}
    - If ``WANDB_RUN_NAME`` is not set, it is set to the experiment name.
    - Logs the resolved wandb configuration via Primus logger.
"""

import os

from primus.core.patches import PatchContext, get_param, register_patch


@register_patch(
    "torchtitan.metrics.wandb_env",
    backend="torchtitan",
    phase="before_train",
    description="Initialize WANDB_* env vars from Primus/TorchTitan config",
    condition=lambda ctx: get_param(ctx, "metrics.enable_wandb", False),
)
def patch_torchtitan_wandb_env(ctx: PatchContext) -> None:
    """
    Initialize WANDB_* environment variables for TorchTitan runs.
    """
    from primus.core.utils.logger import _logger as primus_logger

    # Module/TorchTitan-side config (metrics/job section)
    metrics_cfg = get_param(ctx, "metrics", None)
    job_cfg = get_param(ctx, "job", None)

    # Primus-side experiment metadata
    primus_cfg = ctx.extra.get("primus_config")
    exp_meta = getattr(primus_cfg, "exp_meta_info", {}) if primus_cfg is not None else {}

    work_group = exp_meta.get("work_group")
    user_name = exp_meta.get("user_name")
    exp_name = exp_meta.get("exp_name")

    primus_logger.warning("Monkey patch torchtitan wandb...")

    # Derive WANDB_PROJECT if not explicitly set
    if os.getenv("WANDB_PROJECT") is None and work_group and user_name:
        os.environ["WANDB_PROJECT"] = f"Primus-Titan-Pretrain-{work_group}_{user_name}"

    # Derive WANDB_RUN_NAME if not explicitly set
    if os.getenv("WANDB_RUN_NAME") is None and exp_name:
        os.environ["WANDB_RUN_NAME"] = exp_name

    # Compute wandb save directory if possible (for logging only)
    wandb_save_dir = None
    if job_cfg is not None and metrics_cfg is not None:
        dump_folder = getattr(job_cfg, "dump_folder", None)
        save_tb_folder = getattr(metrics_cfg, "save_tb_folder", None)
        if dump_folder is not None and save_tb_folder is not None:
            wandb_save_dir = os.path.join(dump_folder, save_tb_folder)

    primus_logger.info(f"torchtitan wandb_project: {os.getenv('WANDB_PROJECT')}")
    primus_logger.info(f"torchtitan wandb_exp_name: {os.getenv('WANDB_RUN_NAME')}")
    primus_logger.info(f"torchtitan wandb_entity: {os.getenv('WANDB_TEAM')}")
    if wandb_save_dir is not None:
        primus_logger.info(f"torchtitan wandb_save_dir under: {wandb_save_dir}")
