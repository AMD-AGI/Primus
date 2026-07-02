###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo TESpecProvider Patches

Patches for replacing Transformer Engine TESpecProvider with PrimusTurboSpecProvider.
"""

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _use_legacy_grouped_gemm(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return bool(
        getattr(args, "moe_grouped_gemm", False)
        and getattr(args, "moe_use_legacy_grouped_gemm", False)
        and not getattr(args, "use_turbo_grouped_gemm", False)
    )


def _needs_primus_spec_provider(ctx: PatchContext) -> bool:
    if _use_legacy_grouped_gemm(ctx):
        log_rank_0(
            "[Patch:megatron.turbo.te_spec_provider] "
            "legacy grouped GEMM enabled; using Primus spec provider..."
        )
        return True
    return is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.turbo.te_spec_provider",
    backend="megatron",
    phase="before_train",
    description="Replace TESpecProvider with Primus provider for PrimusTurbo or legacy grouped GEMM",
    condition=_needs_primus_spec_provider,
    backend_versions=["<0.17"],
    priority=41,
)
def patch_te_spec_provider(ctx: PatchContext):
    """
    Patch Transformer Engine integration to use PrimusTurboSpecProvider.

    This replaces TESpecProvider in all relevant Megatron modules. It is used
    for PrimusTurbo features and for legacy grouped-GEMM expert selection.
    """
    import megatron.core.extensions as meg_ext
    from megatron.core.extensions import transformer_engine_spec_provider
    from megatron.core.models.gpt import gpt_layer_specs, moe_module_specs
    from megatron.core.transformer import multi_token_prediction

    from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
        PrimusTurboSpecProvider,
    )

    args = get_args(ctx)
    if _use_legacy_grouped_gemm(ctx):
        reason = "legacy grouped GEMM enabled"
    elif bool(getattr(args, "enable_primus_turbo", False)):
        reason = "PrimusTurbo backend enabled"
    else:
        reason = "Primus spec provider requested"

    log_rank_0(
        "[Patch:megatron.turbo.te_spec_provider] "
        f"Patch TESpecProvider to PrimusTurboSpecProvider; {reason}"
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

    log_rank_0(f"[Patch:megatron.turbo.te_spec_provider] Using Primus spec provider ({reason})")
