###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pin the Primus-Turbo RMSNorm backward launch config instead of autotuning it.

``PRIMUS_TURBO_RMSNORM_BWD_CONFIG=<num_warps>,<num_stages>`` (unset: Turbo's autotune).

Turbo's persistent-grid RMSNorm backward kernels (``rmsnorm_bwd_kernel_grid_stride`` and its
fused-residual twin) are ``@triton.autotune``d over num_warps 4/8 × num_stages 1/2. num_warps
changes the shape of the per-row ``tl.sum`` that forms ``mean(dy·g·x̂)``, so the configs give dx
results that differ in the last bit (dgamma is a per-column accumulation and is not affected).
The autotuner picks by timing at the first call, so the gradient bits of a run depend on how
the GPU happened to time four near-equal configs in step 1. Pinning the config makes the
numerics independent of timing, and it also skips the step-1 benchmarking.
"""

from __future__ import annotations

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_ENV = "PRIMUS_TURBO_RMSNORM_BWD_CONFIG"


@register_patch(
    "primus_turbo.rmsnorm_bwd_pinned_config",
    backend="megatron",
    phase="before_train",
    description="Pin the Turbo RMSNorm backward Triton config (PRIMUS_TURBO_RMSNORM_BWD_CONFIG=warps,stages).",
    condition=lambda ctx: bool(os.environ.get(_ENV)),
)
def patch_turbo_rmsnorm_bwd_config(ctx: PatchContext) -> None:
    warps, stages = (int(v) for v in os.environ[_ENV].split(","))
    try:
        from primus_turbo.triton.normalization import rmsnorm_kernel as rk
    except Exception as e:  # Turbo not installed: nothing to pin
        log_rank_0(f"[Patch:primus_turbo.rmsnorm_bwd_pinned_config] skipped ({e!r})")
        return
    pinned = []
    for name in ("rmsnorm_bwd_kernel_grid_stride", "rmsnorm_bwd_residual_kernel_grid_stride"):
        tuner = getattr(rk, name, None)
        configs = getattr(tuner, "configs", None)
        if not configs:
            continue
        match = [c for c in configs if c.num_warps == warps and c.num_stages == stages]
        if not match:
            raise ValueError(f"{_ENV}={warps},{stages} is not one of {name}'s configs")
        tuner.configs = match[:1]
        tuner.cache.clear()
        pinned.append(name)
    log_rank_0(f"[Patch:primus_turbo.rmsnorm_bwd_pinned_config] num_warps={warps} num_stages={stages}: {pinned}")
