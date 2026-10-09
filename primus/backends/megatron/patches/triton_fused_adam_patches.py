###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Triton Fused AdamW Patch
========================

Set ``use_triton_fused_adam: true`` to build Megatron's Adam optimizer from
``TritonFusedAdam`` instead of TE ``FusedAdam``. TE builds that use the generic
``multi_tensor_apply`` launcher cap each Adam launch at 320 workgroups, which
leaves MI455X HBM far from saturated; the Triton kernel sizes its grid to the
device's CU count. ROCm TE builds with the custom Adam kernel are not capped.

Mechanism:
    ``_get_megatron_optimizer_based_on_param_groups`` reads the module global
    ``Adam`` at call time, so rebinding ``megatron.core.optimizer.Adam`` is
    enough. ``TritonFusedAdam`` subclasses TE ``FusedAdam``, which keeps the
    ``isinstance`` check in ``distrib_optimizer`` and TE-specific state/step
    handling valid.
"""

import os

from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.optimizer.triton_fused_adam"


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Use TritonFusedAdam (device-sized grid) instead of TE FusedAdam for the Adam optimizer.",
    priority=40,
    condition=lambda ctx: getattr(get_args(ctx), "use_triton_fused_adam", False),
)
def patch_triton_fused_adam(ctx: PatchContext) -> None:
    import megatron.core.optimizer as optimizer_module

    if is_patched(optimizer_module, _PATCH_KEY):
        return

    # fused_adam_clip_patches defers gradient clipping into TE FusedAdam.step,
    # which TritonFusedAdam.step overrides; combining them would skip clipping.
    if os.environ.get("PRIMUS_FUSED_ADAM_CLIP", "0") == "1":
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] PRIMUS_FUSED_ADAM_CLIP=1 owns the TE FusedAdam step; "
            "keeping TE FusedAdam."
        )
        return

    if optimizer_module.USING_PYTORCH_OPTIMIZER:
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] Transformer Engine / Apex optimizers are unavailable; "
            "keeping the torch optimizer."
        )
        return

    from transformer_engine.pytorch.optimizers import FusedAdam

    if optimizer_module.Adam is not FusedAdam:
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] megatron.core.optimizer.Adam is {optimizer_module.Adam}, "
            "not TE FusedAdam; skipping."
        )
        return

    from primus.backends.megatron.core.optimizer.triton_fused_adam import (
        TritonFusedAdam,
    )

    optimizer_module.Adam = TritonFusedAdam

    mark_patched(optimizer_module, _PATCH_KEY)
    log_rank_0(f"[Patch:{_PATCH_KEY}] megatron.core.optimizer.Adam -> TritonFusedAdam.")
