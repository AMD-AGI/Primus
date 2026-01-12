###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo TESpecProvider Patches

Patches for replacing Transformer Engine TESpecProvider with PrimusTurboSpecProvider.
"""

import importlib.util

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_primus_turbo_enabled(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo is enabled and can be used.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
    """
    # Check if primus_turbo package is available
    if importlib.util.find_spec("primus_turbo") is None:
        log_rank_0("[Patch:megatron.turbo.te_spec_provider] primus_turbo not found, use TE backend...")
        return False

    args = get_args(ctx)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))

    if tp_size != 1:
        if enable_primus_turbo:
            log_rank_0(
                "[Patch:megatron.turbo.te_spec_provider] "
                "Primus Turbo does not support TP; using TE backend instead..."
            )
        else:
            log_rank_0("[Patch:megatron.turbo.te_spec_provider] TP > 1; using TE backend...")
        return False

    if not enable_primus_turbo:
        log_rank_0("[Patch:megatron.turbo.te_spec_provider] enable_primus_turbo=False; using TE backend...")
        return False

    return True


@register_patch(
    "megatron.turbo.te_spec_provider",
    backend="megatron",
    phase="before_train",
    description="Replace TESpecProvider with PrimusTurboSpecProvider when PrimusTurbo is enabled",
    condition=_is_primus_turbo_enabled,
)
def patch_te_spec_provider(ctx: PatchContext):
    """
    Patch Transformer Engine integration to use PrimusTurboSpecProvider.

    This replaces TESpecProvider in all relevant Megatron modules with
    PrimusTurboSpecProvider to enable PrimusTurbo backend.
    """
    import megatron.core.extensions as meg_ext
    from megatron.core.extensions import (
        transformer_engine as transformer_engine_spec_provider,
    )
    from megatron.core.models.gpt import gpt_layer_specs, moe_module_specs
    from megatron.core.transformer import multi_token_prediction

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    log_rank_0(
        "[Patch:megatron.turbo.te_spec_provider] "
        "Patch TESpecProvider to PrimusTurboSpecProvider; PrimusTurbo backend enabled"
    )

    assert (
        meg_ext.transformer_engine.HAVE_TE
    ), "PrimusTurboSpecProvider patch failed, can't find transformer_engine"

    # Replace TESpecProvider in all relevant locations
    transformer_engine_spec_provider.TESpecProvider = PrimusTurboSpecProvider
    log_rank_0(
        "[Patch:megatron.turbo.te_spec_provider]   Patched "
        f"megatron.core.extensions.transformer_engine.TESpecProvider -> {PrimusTurboSpecProvider.__name__}"
    )

    gpt_layer_specs.TESpecProvider = PrimusTurboSpecProvider
    log_rank_0(
        "[Patch:megatron.turbo.te_spec_provider]   Patched "
        f"megatron.core.models.gpt.gpt_layer_specs.TESpecProvider -> {PrimusTurboSpecProvider.__name__}"
    )

    moe_module_specs.TESpecProvider = PrimusTurboSpecProvider
    log_rank_0(
        "[Patch:megatron.turbo.te_spec_provider]   Patched "
        f"megatron.core.models.gpt.moe_module_specs.TESpecProvider -> {PrimusTurboSpecProvider.__name__}"
    )

    multi_token_prediction.TESpecProvider = PrimusTurboSpecProvider
    log_rank_0(
        "[Patch:megatron.turbo.te_spec_provider]   Patched "
        f"megatron.core.transformer.multi_token_prediction.TESpecProvider -> {PrimusTurboSpecProvider.__name__}"
    )

    log_rank_0("[Patch:megatron.turbo.te_spec_provider] Using PrimusTurbo backend (PT)")
