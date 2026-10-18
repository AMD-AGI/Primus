###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in fused SwiGLU (``use_primus_fused_swiglu``).

Rebinds the (prob-weighted) SwiGLU entry points in ``_TARGETS``, and every module that imported them by name,
to the drop-ins of ``primus.backends.megatron.core.fusions.triton_swiglu``. Calls outside the fused contract
(bias, clamp, fp8 input store, CPU offload) keep the original function. The Turbo path
(``use_turbo_fused_act_with_probs``) does not use these names.
"""

import importlib

from primus.backends.megatron.core.fusions._notices import warn_if_not_gfx1250
from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.backends.megatron.patches._rebind import rebind_everywhere
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.activation.fused_swiglu"
_ARG = "use_primus_fused_swiglu"
_MEGATRON = "megatron.core.fusions.fused_bias_swiglu"
# (module, function, drop-in factory in triton_swiglu)
_TARGETS = (
    (
        "primus.backends.megatron.core.fusions.fused_bias_swiglu",
        "weighted_bias_swiglu_impl",
        "make_primus_weighted",
    ),
    (_MEGATRON, "weighted_bias_swiglu_impl", "make_megatron_weighted"),
    (_MEGATRON, "bias_swiglu_impl", "make_megatron_plain"),
)


def _patch_sites() -> list:
    """Rebind every target and its by-name importers (idempotent). Returns the rebound ``module.function``."""
    from primus.backends.megatron.core.fusions import triton_swiglu

    rebound = []
    for modname, name, factory in _TARGETS:
        mod = importlib.import_module(modname)
        key = f"{_PATCH_KEY}:{name}"
        if is_patched(mod, key):
            continue
        new = getattr(triton_swiglu, factory)(getattr(mod, name), _PATCH_KEY)
        rebound += [f"{m}.{name}" for m in rebind_everywhere(mod, name, new)]
        mark_patched(mod, key)
    return rebound


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="Fused (prob-weighted) SwiGLU fwd/bwd with one pinned Triton config",
    condition=lambda ctx: bool(getattr(get_args(ctx), _ARG, False)),
    tags=["activation", "moe", "triton"],
)
def patch_fused_swiglu(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_PATCH_KEY, _ARG)
    args = get_args(ctx)
    # turbo_sync_free_moe_stage 3 turns use_turbo_fused_act_with_probs on later, in validate_args.
    if (
        getattr(args, "use_turbo_fused_act_with_probs", False)
        or (getattr(args, "turbo_sync_free_moe_stage", 0) or 0) >= 3
    ):
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] use_turbo_fused_act_with_probs (or turbo_sync_free_moe_stage 3) is set: "
            "routed experts take the Turbo path and do not reach the patched names."
        )
    log_rank_0(f"[Patch:{_PATCH_KEY}] patched {_patch_sites()}")
