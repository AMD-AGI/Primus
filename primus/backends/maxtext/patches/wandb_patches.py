###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Weights & Biases (wandb) Patch

Behavior:
    - Derives ``WANDB_PROJECT`` from Primus ``exp_meta_info`` if not already set.
    - Replaces ``MetricLogger`` in MaxText and Primus train modules with
      ``PrimusMetricLogger`` which adds wandb integration.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.patches.context import get_param
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    patch_id="maxtext.wandb",
    backend="maxtext",
    phase="setup",
    description="Initialize WANDB_PROJECT and replace MetricLogger with PrimusMetricLogger",
    condition=lambda ctx: get_param(ctx, "enable_wandb", False),
)
def patch_wandb(ctx: PatchContext) -> None:
    """
    Set ``WANDB_PROJECT`` from Primus experiment metadata and replace
    ``MetricLogger`` references with ``PrimusMetricLogger``.
    """
    log_rank_0("[Patch:maxtext.wandb] Patching wandb / MetricLogger...")

    # --- Set WANDB_PROJECT from Primus experiment metadata ---
    primus_config = ctx.extra.get("primus_config")
    if primus_config is not None:
        exp_meta_info = getattr(primus_config, "exp_meta_info", None)
        if exp_meta_info:
            work_group = exp_meta_info.get("work_group", "")
            user_name = exp_meta_info.get("user_name", "")
            if work_group and user_name and os.getenv("WANDB_PROJECT") is None:
                os.environ["WANDB_PROJECT"] = f"Primus-MaxText-Pretrain-{work_group}_{user_name}"
                log_rank_0(f"[Patch:maxtext.wandb] WANDB_PROJECT set to: {os.environ['WANDB_PROJECT']}")

    # --- Replace MetricLogger with PrimusMetricLogger ---
    import MaxText.metric_logger as orig_metric_logger
    import MaxText.train as orig_train

    from primus.backends.maxtext.metric_logger import PrimusMetricLogger

    orig_metric_logger.MetricLogger = PrimusMetricLogger
    orig_train.MetricLogger = PrimusMetricLogger

    warning_rank_0("[Patch:maxtext.wandb] wandb / MetricLogger patched successfully.")
