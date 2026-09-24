###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Triton Fused AdamW Patch
========================

Set ``use_triton_fused_adam: true`` to build Megatron's Adam optimizer from
``TritonFusedAdam`` instead of TE ``FusedAdam``. TE's ``multi_tensor_apply``
launches at most 320 workgroups per kernel, which leaves MI455X HBM far from
saturated; the Triton kernel sizes its grid to the device's CU count.

Mechanism:
    ``_get_megatron_optimizer_based_on_param_groups`` reads the module global
    ``Adam`` at call time, so rebinding ``megatron.core.optimizer.Adam`` is
    enough. ``TritonFusedAdam`` subclasses TE ``FusedAdam``, which keeps the
    ``isinstance`` check in ``distrib_optimizer`` and TE-specific state/step
    handling valid.
"""

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
