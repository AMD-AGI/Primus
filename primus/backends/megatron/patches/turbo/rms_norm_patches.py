###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo RMSNorm Patches

Patches for replacing RMSNorm with PrimusTurbo implementation.
"""


from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_turbo_rms_norm_can_patch(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo RMSNorm is enabled.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
      - use_turbo_rms_norm == True
    """
    args = get_args(ctx)
    use_turbo_rms_norm = bool(getattr(args, "use_turbo_rms_norm", False))

    return use_turbo_rms_norm and is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.turbo.rms_norm",
    backend="megatron",
    phase="before_train",
    description="Replace Transformer Engine RMSNorm with PrimusTurbo implementation",
    condition=_is_turbo_rms_norm_can_patch,
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
