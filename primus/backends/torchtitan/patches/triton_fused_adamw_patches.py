###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Triton Fused AdamW Patch
===================================

Set ``optimizer.use_triton_fused_adam: true`` to build TorchTitan's AdamW from
``TritonFusedAdamW`` instead of ``torch.optim.AdamW(fused=True)``, whose ATen
``multi_tensor_apply`` launches are capped at 320 workgroups.

``OptimizersContainer.__init__`` is wrapped rather than ``build_optimizers``,
because train specs capture ``build_optimizers`` at registration time. The FT
container goes through the same ``__init__``; the optimizer-in-backward
container builds one optimizer per parameter and is left unchanged.
"""

import functools

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_ID = "torchtitan.optimizer.triton_fused_adamw"


def _enabled(ctx: PatchContext) -> bool:
    optimizer = getattr(get_args(ctx), "optimizer", None)
    return bool(getattr(optimizer, "use_triton_fused_adam", False))


@register_patch(
    patch_id=_PATCH_ID,
    backend="torchtitan",
    phase="setup",  # before Trainer builds the optimizers
    description="Build AdamW from TritonFusedAdamW (device-sized grid) instead of ATen fused AdamW",
    condition=_enabled,
)
def patch_torchtitan_triton_fused_adamw(ctx: PatchContext) -> None:
    import torch
    from torchtitan.components.optimizer import OptimizersContainer

    from primus.backends.torchtitan.components.optimizer.triton_adamw import (
        TritonFusedAdamW,
    )

    if getattr(OptimizersContainer.__init__, "_primus_triton_adamw", False):
        return

    original_init = OptimizersContainer.__init__

    @functools.wraps(original_init)
    def __init__(self, model_parts, optimizer_cls, optimizer_kwargs, *args, **kwargs):
        if optimizer_cls is torch.optim.AdamW:
            optimizer_cls = TritonFusedAdamW
        original_init(self, model_parts, optimizer_cls, optimizer_kwargs, *args, **kwargs)

    __init__._primus_triton_adamw = True
    OptimizersContainer.__init__ = __init__
    log_rank_0(f"[Patch:{_PATCH_ID}] torch.optim.AdamW -> TritonFusedAdamW in OptimizersContainer.")
