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


def _primus_turbo_deep_importable() -> bool:
    """Deep-probe the primus_turbo import chain that the patch body actually uses.

    A shallow ``importlib.util.find_spec("primus_turbo")`` returns a valid spec
    even when the package's transitive dependencies (e.g. a broken ``aiter`` /
    ``csrc`` install in the runtime image) cause the real import to crash. In
    that case the patch would be selected for application and then fail at
    module-load time, leaving the model in an inconsistent half-patched state
    that can produce silent NaNs during FP8 training.

    By probing the exact symbols the patch body imports, we either confirm the
    full chain works or skip the patch cleanly so callers fall back to the
    stock TE provider.
    """
    try:
        from primus.backends.megatron.core.extensions.primus_turbo import (  # noqa: F401  pylint: disable=W0611
            PrimusTurboAttention,
            PrimusTurboColumnParallelLinear,
        )

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _is_primus_turbo_enabled(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo is enabled and can be used.

    Requires:
      - primus_turbo package is installed AND its deep import chain works
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
      - If use_turbo_grouped_mlp is enabled with moe_grouped_gemm,
        must use legacy grouped gemm (moe_use_legacy_grouped_gemm=True)
    """
    # Check if primus_turbo package is available *and* its deep import chain works.
    # A shallow find_spec is not enough: if a transitive dep (e.g. aiter/csrc) is
    # broken in the runtime image, the patch body will crash at import time and
    # leave the model half-patched, which can produce silent NaNs in FP8 training.
    if importlib.util.find_spec("primus_turbo") is None or not _primus_turbo_deep_importable():
        log_rank_0("[Patch:megatron.turbo.te_spec_provider] primus_turbo not importable, use TE backend...")
        return False

    args = get_args(ctx)
    return bool(
        getattr(args, "moe_grouped_gemm", False)
        and getattr(args, "moe_use_legacy_grouped_gemm", False)
        and not (
            getattr(args, "use_turbo_grouped_gemm", False) or getattr(args, "use_turbo_grouped_mlp", False)
        )
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
