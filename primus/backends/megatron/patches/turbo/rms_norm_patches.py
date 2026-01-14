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

    args = get_args(ctx)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))
    use_turbo_rms_norm = bool(getattr(args, "use_turbo_rms_norm", False))

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

    This replaces transformer_engine.pytorch.RMSNorm with PrimusTurboRMSNorm
    for improved performance on ROCm platforms.
    """
    import transformer_engine as te

    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    log_rank_0("[Patch:megatron.turbo.rms_norm] Patching RMSNorm...")

    te.pytorch.RMSNorm = PrimusTurboRMSNorm
    log_rank_0(
        "[Patch:megatron.turbo.rms_norm]   Patched "
        f"transformer_engine.pytorch.RMSNorm -> {PrimusTurboRMSNorm.__name__}"
    )
