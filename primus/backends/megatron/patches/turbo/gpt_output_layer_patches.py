###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo GPT Output Layer Patches

Patches for replacing GPT output layer with PrimusTurbo implementation.
"""

import importlib.util

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_turbo_parallel_linear_enabled(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo parallel linear is enabled.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
      - use_turbo_parallel_linear == True
    """
    # Check if primus_turbo package is available
    if importlib.util.find_spec("primus_turbo") is None:
        return False

    args = get_args(ctx)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))
    use_turbo_parallel_linear = bool(getattr(args, "use_turbo_parallel_linear", False))

    return tp_size == 1 and enable_primus_turbo and use_turbo_parallel_linear


@register_patch(
    "megatron.turbo.gpt_output_layer",
    backend="megatron",
    phase="before_train",
    description="Replace GPT ColumnParallelLinear with PrimusTurbo implementation",
    condition=_is_turbo_parallel_linear_enabled,
)
def patch_gpt_output_layer(ctx: PatchContext):
    """
    Patch GPT output layer to use PrimusTurbo ColumnParallelLinear.

    This replaces the standard tensor_parallel.ColumnParallelLinear with
    PrimusTurboColumnParallelLinearTorch in gpt_model.
    """
    from megatron.core.models.gpt import gpt_model

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboColumnParallelLinearTorch,
    )

    log_rank_0("[Patch:megatron.turbo.gpt_output_layer] Patching GPT output layer...")

    gpt_model.tensor_parallel.ColumnParallelLinear = PrimusTurboColumnParallelLinearTorch
    log_rank_0(
        "[Patch:megatron.turbo.gpt_output_layer]   Patched "
        f"megatron.core.models.gpt.gpt_model.tensor_parallel.ColumnParallelLinear "
        f"-> {PrimusTurboColumnParallelLinearTorch.__name__}"
    )
