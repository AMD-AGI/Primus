###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo RMSNorm Patches

Patches for replacing RMSNorm with PrimusTurbo implementation.
"""

import importlib.util

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_turbo_rms_norm_enabled(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo RMSNorm is enabled.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
      - use_turbo_rms_norm == True
    """
    # Check if primus_turbo package is available
    if importlib.util.find_spec("primus_turbo") is None:
        return False

    # Primus yaml env-var substitution ( ${VAR:default} ) returns raw strings,
    # and ``bool("false")`` is True. Coerce common string forms to bool so
    # that the turbo patch is correctly gated when the flag is toggled via an
    # environment variable.
    def _coerce_bool(v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    args = get_args(ctx)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = _coerce_bool(getattr(args, "enable_primus_turbo", False))
    use_turbo_rms_norm = _coerce_bool(getattr(args, "use_turbo_rms_norm", False))

    return tp_size == 1 and enable_primus_turbo and use_turbo_rms_norm


@register_patch(
    "megatron.turbo.rms_norm",
    backend="megatron",
    phase="before_train",
    description="Replace Transformer Engine RMSNorm with PrimusTurbo implementation",
    condition=_is_turbo_rms_norm_enabled,
)
def patch_rms_norm(ctx: PatchContext):
    """
    Patch Transformer Engine RMSNorm to use PrimusTurbo implementation.

    Two sites are patched:

    1. ``te.pytorch.RMSNorm`` -> ``PrimusTurboRMSNorm``
       Covers everything that goes through ``TENorm`` (q/k_norm,
       pre_mlp_layernorm, final_layernorm, ...).

    2. ``TELayerNormColumnParallelLinear`` (referenced by the TE spec
       provider as the ``column_parallel_layer_norm_linear`` module) ->
       ``PrimusTurboLayerNormColumnParallelLinear``.
       Covers the *fused* norm baked into ``linear_qkv`` and (for dense
       MLP) ``mlp.linear_fc1``. Without (2) those sites still call TE's
       internal ``rmsnorm_fwd_general_kernel`` / ``rmsnorm_bwd_general_kernel``
       / ``rmsnorm_bwd_finalize_general_kernel``, which trace shows is
       ~13 ms / 3 steps of GPU time and ~5 ms / step of wall time on
       GPT-OSS-20B.
    """
    import transformer_engine as te
    from megatron.core.extensions import transformer_engine_spec_provider

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboLayerNormColumnParallelLinear,
        PrimusTurboRMSNorm,
    )

    log_rank_0("[Patch:megatron.turbo.rms_norm] Patching RMSNorm...")

    te.pytorch.RMSNorm = PrimusTurboRMSNorm
    log_rank_0(
        "[Patch:megatron.turbo.rms_norm]   Patched "
        f"transformer_engine.pytorch.RMSNorm -> {PrimusTurboRMSNorm.__name__}"
    )

    transformer_engine_spec_provider.TELayerNormColumnParallelLinear = (
        PrimusTurboLayerNormColumnParallelLinear
    )
    log_rank_0(
        "[Patch:megatron.turbo.rms_norm]   Patched "
        "transformer_engine_spec_provider.TELayerNormColumnParallelLinear -> "
        f"{PrimusTurboLayerNormColumnParallelLinear.__name__}"
    )
