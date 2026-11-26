###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Let Primus-Turbo's FlyDSL expert weight-gradient GEMM take more than 64 experts on gfx1250.

``GroupedGEMMVariableKFlyDSLBackend`` declines more than ``MAX_G`` (64) groups. Its own
comment gives 64 as a measured limit on gfx950 only and says the gfx1250 kernel was
measured correct up to 160 groups. A backend the user selected that then declines is an
error, not a fallback, so with ``turbo_grouped_gemm_backend`` on FlyDSL a model with more
local experts than the cap stops at its first expert weight gradient (MiniMax-M3 on one
GPU has 128).

On gfx1250 this raises the cap to the local expert count, up to 160. Nothing changes on
other architectures, with at most 64 local experts, or above 160.
"""

from typing import Optional

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

GFX1250_MEASURED_MAX_GROUPS = 160


def local_experts(args) -> int:
    num_experts = getattr(args, "num_experts", None) or 0
    return num_experts // (getattr(args, "expert_model_parallel_size", None) or 1)


def raised_max_groups(cap: int, num_local_experts: int, gfx1250: bool) -> Optional[int]:
    """The cap to set, or None to leave ``cap`` alone."""
    if not gfx1250 or num_local_experts <= cap or num_local_experts > GFX1250_MEASURED_MAX_GROUPS:
        return None
    return num_local_experts


def _is_flydsl_grouped_gemm(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    backend = str(getattr(args, "turbo_grouped_gemm_backend", "default")).lower()
    return (
        bool(getattr(args, "use_turbo_grouped_gemm", False))
        and "flydsl" in backend
        and local_experts(args) > 0
        and is_primus_turbo_can_patch(ctx)
    )


@register_patch(
    "megatron.turbo.flydsl_grouped_gemm_max_groups",
    backend="megatron",
    phase="before_train",
    description="Raise Primus-Turbo's FlyDSL wgrad grouped GEMM expert cap on gfx1250",
    condition=_is_flydsl_grouped_gemm,
)
def patch_flydsl_grouped_gemm_max_groups(ctx: PatchContext):
    from primus_turbo.pytorch.core.utils import is_gfx1250
    from primus_turbo.pytorch.kernels.grouped_gemm import grouped_gemm_impl

    tag = "[Patch:megatron.turbo.flydsl_grouped_gemm_max_groups]"
    backend = getattr(grouped_gemm_impl, "GroupedGEMMVariableKFlyDSLBackend", None)
    cap = getattr(backend, "MAX_G", None)
    if cap is None:
        log_rank_0(f"{tag} this Primus-Turbo has no FlyDSL variable-K expert cap, skipped")
        return
    n = local_experts(get_args(ctx))
    new_cap = raised_max_groups(cap, n, is_gfx1250())
    if new_cap is None:
        if n > cap:
            warning_rank_0(f"{tag} {n} local experts are over the FlyDSL cap of {cap}; left as is")
        return
    backend.MAX_G = new_cap
    log_rank_0(f"{tag} GroupedGEMMVariableKFlyDSLBackend.MAX_G {cap} -> {new_cap}")
