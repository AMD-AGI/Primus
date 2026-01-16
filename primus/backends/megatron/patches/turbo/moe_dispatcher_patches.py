###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo MoE Dispatcher Patches

Patches for replacing MoE token dispatcher with PrimusTurbo DeepEP implementation.
"""

import importlib.util

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_turbo_deepep_enabled(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo DeepEP MoE dispatcher is enabled.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
      - use_turbo_deepep == True
    """
    # Check if primus_turbo package is available
    if importlib.util.find_spec("primus_turbo") is None:
        return False

    args = get_args(ctx)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))
    use_turbo_deepep = bool(getattr(args, "use_turbo_deepep", False))

    return tp_size == 1 and enable_primus_turbo and use_turbo_deepep


@register_patch(
    "megatron.turbo.moe_dispatcher",
    backend="megatron",
    phase="before_train",
    description="Replace MoE token dispatcher with PrimusTurbo DeepEP implementation",
    condition=_is_turbo_deepep_enabled,
)
def patch_moe_dispatcher(ctx: PatchContext):
    """
    Patch MoE token dispatcher to use PrimusTurbo DeepEP implementation.

    This replaces MoEFlexTokenDispatcher with PrimusTurboDeepEPTokenDispatcher
    and automatically enables moe_enable_deepep and sets moe_token_dispatcher_type to 'flex'.
    """
    from megatron.core.transformer.moe import moe_layer, token_dispatcher

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboDeepEPTokenDispatcher,
    )

    log_rank_0("[Patch:megatron.turbo.moe_dispatcher] Patching MoE token dispatcher...")

    args = get_args(ctx)

    # Auto-enable DeepEP and set dispatcher type
    args.moe_enable_deepep = True
    args.moe_token_dispatcher_type = "flex"
    log_rank_0(
        "[Patch:megatron.turbo.moe_dispatcher]   Set moe_enable_deepep=True, moe_token_dispatcher_type='flex'"
    )

    token_dispatcher.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher
    log_rank_0(
        "[Patch:megatron.turbo.moe_dispatcher]   Patched "
        f"megatron.core.transformer.moe.token_dispatcher.MoEFlexTokenDispatcher "
        f"-> {PrimusTurboDeepEPTokenDispatcher.__name__}"
    )

    moe_layer.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher
    log_rank_0(
        "[Patch:megatron.turbo.moe_dispatcher]   Patched "
        f"megatron.core.transformer.moe.moe_layer.MoEFlexTokenDispatcher "
        f"-> {PrimusTurboDeepEPTokenDispatcher.__name__}"
    )
