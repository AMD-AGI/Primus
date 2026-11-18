###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pin the Primus-Turbo RMSNorm backward launch config instead of autotuning it.

``turbo_rmsnorm_bwd_config: [num_warps, num_stages]`` (null: Turbo's autotune).

Turbo's persistent-grid RMSNorm backward kernels (``rmsnorm_bwd_kernel_grid_stride`` and its
fused-residual twin) are ``@triton.autotune``d over num_warps 4/8 × num_stages 1/2. num_warps
changes the shape of the per-row ``tl.sum`` that forms ``mean(dy·g·x̂)``, so the configs give dx
results that differ in the last bit (dgamma is a per-column accumulation and is not affected).
The autotuner picks by timing at the first call, so the gradient bits of a run depend on how
the GPU happened to time four near-equal configs in step 1. Pinning the config makes the
numerics independent of timing, and it also skips the step-1 benchmarking.
"""

from __future__ import annotations

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCH_KEY = "primus_turbo.rmsnorm_bwd_pinned_config"


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Pin the Turbo RMSNorm backward Triton config (turbo_rmsnorm_bwd_config: [warps, stages])",
    condition=lambda ctx: bool(getattr(get_args(ctx), "turbo_rmsnorm_bwd_config", None)),
)
def patch_turbo_rmsnorm_bwd_config(ctx: PatchContext) -> None:
    config = get_args(ctx).turbo_rmsnorm_bwd_config
    if len(config) != 2:
        raise ValueError(f"turbo_rmsnorm_bwd_config must be [num_warps, num_stages], got {config!r}")
    warps, stages = (int(v) for v in config)
    try:
        from primus_turbo.triton.normalization import rmsnorm_kernel as rk
    except ImportError as e:
        if getattr(get_args(ctx), "use_turbo_rms_norm", False):
            raise RuntimeError(
                "turbo_rmsnorm_bwd_config is set and use_turbo_rms_norm is on, but the Primus-Turbo "
                f"RMSNorm kernel module does not import ({e!r}); this Primus-Turbo version needs a new pin"
            ) from e
        log_rank_0(f"[Patch:{_PATCH_KEY}] skipped: Turbo RMSNorm not in use ({e!r})")
        return
    pinned = []
    for name in ("rmsnorm_bwd_kernel_grid_stride", "rmsnorm_bwd_residual_kernel_grid_stride"):
        tuner = getattr(rk, name, None)
        configs = getattr(tuner, "configs", None)
        if not configs:
            continue
        match = [c for c in configs if c.num_warps == warps and c.num_stages == stages]
        if not match:
            raise ValueError(f"turbo_rmsnorm_bwd_config {[warps, stages]} is not one of {name}'s configs")
        tuner.configs = match[:1]
        tuner.cache.clear()
        pinned.append(name)
    if not pinned:
        raise RuntimeError(
            f"turbo_rmsnorm_bwd_config is set but {rk.__name__} has no autotuned grid-stride "
            "RMSNorm backward kernel to pin; this Primus-Turbo version needs a new pin"
        )
    log_rank_0(f"[Patch:{_PATCH_KEY}] num_warps={warps} num_stages={stages}: {pinned}")
