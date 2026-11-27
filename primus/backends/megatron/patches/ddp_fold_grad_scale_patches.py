###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fold Megatron's DDP gradient scaling into the gradient-norm and fused-Adam kernels.

Without ``average_in_collective`` Megatron's ``start_grad_sync`` multiplies every gradient bucket by its
``gradient_scaling_factor`` (1/dp for dense, expert_dp/dp for expert buckets) before the reduction: one full
read-modify-write of every gradient buffer per step (DeepSeek-V4 4-layer EP4 DP4 proxy: 1.6G dense + 6.4G expert
elements, ~3.1 ms of BF16 elementwise kernels at the step end). Every reader of the reduced gradients under
``PRIMUS_FUSED_ADAM_CLIP`` is the gradient norm and the Primus fused Adam kernels, so the factor is applied
there instead: the norm is multiplied by ``f`` and Adam multiplies the gradient by ``f * clip_coeff``.

With a power-of-two ``f`` every step is exact (the RCCL BF16 sum, the FP32 sum of squares, sqrt and the Adam
gradient multiply all commute with the scaling), so the update is bit-identical to Megatron's order (barring
subnormal gradients). The patch stays off for non-power-of-two factors, mixed factors, ``average_in_collective``,
Megatron-FSDP and without the fused Adam clip path.

Gate: ``PRIMUS_DDP_FOLD_GRAD_SCALE=1`` (default off).
"""

import functools
import math
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_STATE = {"factor": None}


def grad_prescale() -> float:
    """Factor the gradient consumers must apply (1.0 when the buckets were scaled by Megatron)."""
    f = _STATE["factor"]
    return 1.0 if f is None else f


def _enabled(ctx: PatchContext) -> bool:
    return (
        os.environ.get("PRIMUS_DDP_FOLD_GRAD_SCALE", "0") == "1"
        and os.environ.get("PRIMUS_FUSED_ADAM_CLIP", "0") == "1"
        and os.environ.get("PRIMUS_DDP_PRESCALE_LOSS", "0") != "1"
    )


def _pow2(f: float) -> bool:
    return f > 0 and math.frexp(f)[0] == 0.5


@register_patch(
    "megatron.ddp.fold_grad_scale",
    backend="megatron",
    phase="before_train",
    description="Apply the DDP gradient scaling inside the grad-norm and fused-Adam kernels",
    condition=_enabled,
)
def patch_ddp_fold_grad_scale(ctx: PatchContext):
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )

    original_init = DistributedDataParallel.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        buffers = list(getattr(self, "buffers", [])) + list(getattr(self, "expert_parallel_buffers", []))
        buckets = [b for buf in buffers for b in buf.buckets]
        factors = {float(b.gradient_scaling_factor) for b in buckets}
        if not factors or factors == {1.0}:
            return
        ok = (
            not getattr(self.ddp_config, "average_in_collective", False)
            and not getattr(self.ddp_config, "use_megatron_fsdp", False)
            and len(factors) == 1
            and _pow2(next(iter(factors)))
            and _STATE["factor"] in (None, next(iter(factors)))
        )
        if not ok:
            if _STATE["factor"] is not None:
                raise RuntimeError(
                    "PRIMUS_DDP_FOLD_GRAD_SCALE: a later DDP instance has gradient scaling factors "
                    f"{sorted(factors)} incompatible with the folded factor {_STATE['factor']}"
                )
            return
        (f,) = factors
        for buf in buffers:
            buf.gradient_scaling_factor = 1.0
        for b in buckets:
            b.gradient_scaling_factor = 1.0
        _STATE["factor"] = f
        log_rank_0(
            f"[Patch:megatron.ddp.fold_grad_scale] {len(buckets)} gradient buckets left unscaled; "
            f"grad norm and fused Adam apply x {f}"
        )

    DistributedDataParallel.__init__ = __init__
    log_rank_0("[Patch:megatron.ddp.fold_grad_scale] DDP gradient scaling folded into grad norm / fused Adam (pow2)")
