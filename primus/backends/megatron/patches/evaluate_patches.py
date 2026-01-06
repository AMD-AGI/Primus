###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Evaluate Patches

This module contains patches that modify Megatron's evaluation routine
to integrate Primus-specific behavior (e.g., monkey-patching the
training.evaluate function to use Primus-provided evaluation).
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.training.evaluate",
    backend="megatron",
    phase="before_train",
    description=("Monkey patch evaluate to support Primus-provided evaluate."),
    condition=lambda ctx: (
        getattr(get_args(ctx), "eval_iters", -1) > 0
        or getattr(get_args(ctx), "full_validation", False) is True
    )
    and getattr(get_args(ctx), "eval_interval", -1) > 0,
)
def patch_evaluate(ctx: PatchContext):
    """
    Patch Megatron's Training's evaluate to support Primus-provided evaluate.

    Behavior (moved from MegatronTrainer.patch_evaluate):
        - If module_config.eval_iters or module_config.full_validation is provided,
          replace megatron.training.training.evaluate with primus_evaluate.
    """

    log_rank_0("MegatronPatches: monkey patch evaluate...")

    import megatron.training.training as orig_training

    from primus.backends.megatron.training.evaluator import primus_evaluate

    orig_training.evaluate = primus_evaluate
