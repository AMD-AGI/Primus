###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Apply Megatron's data-parallel gradient average to the loss gradient instead of the gradient buffers.

Without ``average_in_collective`` Megatron's ``start_grad_sync`` multiplies every gradient bucket by
``1/dp_size`` before the reduction (``bucket.grad_data *= gradient_scaling_factor``): one full read-modify-write of
every gradient buffer per step (DeepSeek-V4 4-layer EP4 proxy: the 6.4G-element expert buffer alone, ~2.5 ms).
The backward is linear in the loss gradient, so the same factor applied through ``config.grad_scale_func`` (which
also scales the MoE auxiliary and MTP losses) yields the same gradients. When the factor is a power of two the
multiply is exact in bf16 / fp32 and the FP8 UE8M0 block scales absorb it, so the gradients are bit-identical
(barring subnormal underflow); otherwise this patch stays off.

Gate: ``PRIMUS_DDP_PRESCALE_LOSS=1`` (default off). Measured on the DSv4 EP4 proxy: not bit-identical (per-parameter
grad norms differ at the 1e-4 level: some backward path is not exactly power-of-two scale-equivariant), -3.5 ms/step.
"""

import functools
import math
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_STATE = {"factor": None}


def _enabled(ctx: PatchContext) -> bool:
    return os.environ.get("PRIMUS_DDP_PRESCALE_LOSS", "0") == "1"


def _pow2(f: float) -> bool:
    return f > 0 and math.frexp(f)[0] == 0.5


@register_patch(
    "megatron.ddp.prescale_loss",
    backend="megatron",
    phase="before_train",
    description="Apply the DDP 1/dp gradient scaling to the loss gradient instead of every gradient bucket",
    condition=_enabled,
)
def patch_ddp_prescale_loss(ctx: PatchContext):
    import megatron.training.training as megatron_training
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )

    original_init = DistributedDataParallel.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self.ddp_config, "average_in_collective", False) or getattr(
            self.ddp_config, "use_megatron_fsdp", False
        ):
            return
        buffers = list(getattr(self, "buffers", [])) + list(getattr(self, "expert_parallel_buffers", []))
        buckets = [b for buf in buffers for b in buf.buckets]
        factors = {float(b.gradient_scaling_factor) for b in buckets}
        if len(factors) != 1:
            return
        (f,) = factors
        if f == 1.0 or not _pow2(f) or _STATE["factor"] not in (None, f):
            return
        for buf in buffers:
            buf.gradient_scaling_factor = 1.0
        for b in buckets:
            b.gradient_scaling_factor = 1.0
        _STATE["factor"] = f

    DistributedDataParallel.__init__ = __init__

    original_train_step = megatron_training.train_step

    @functools.wraps(original_train_step)
    def train_step(*args, **kwargs):
        f = _STATE["factor"]
        if f is not None:
            from megatron.core.utils import get_model_config

            model = kwargs.get("model", args[2] if len(args) > 2 else None)
            config = get_model_config(model[0])
            if not getattr(config, "_primus_prescaled", False):
                inner = config.grad_scale_func

                def grad_scale_func(t, _inner=inner, _f=f):
                    return (t if _inner is None else _inner(t)) * _f

                config.grad_scale_func = grad_scale_func
                config._primus_prescaled = True
                try:  # loss scales the fused loss tail cached before this point (e.g. in pp_warmup) lack the factor
                    from primus.backends.megatron.patches import fuse2_loss_patches

                    logged = fuse2_loss_patches._SCALES.get("_logged")
                    fuse2_loss_patches._SCALES.clear()
                    if logged:
                        fuse2_loss_patches._SCALES["_logged"] = logged
                except ImportError:
                    pass
                log_rank_0(f"[Patch:megatron.ddp.prescale_loss] gradient buckets unscaled; loss gradient x {f}")
        return original_train_step(*args, **kwargs)

    megatron_training.train_step = train_step
    log_rank_0("[Patch:megatron.ddp.prescale_loss] DDP 1/dp gradient scaling moved to the loss gradient (pow2 only)")
