###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in Triton MoE router gating GEMM (``use_primus_triton_router_gemm``).

Replaces ``router_gating_linear`` in ``megatron.core.transformer.moe.router`` (looked up at call time by
``TopKRouter.gating`` and so by ``PrimusTopKRouter``) with the drop-in of
``primus.backends.megatron.core.fusions.triton_router_gemm``: fp32 logits and fp32-accurate gradients without
fp32 copies of the input and the weight. Covers ``moe_router_dtype: fp32`` without router bias at the expert
count and shapes listed there; other calls keep the original function (including the ``moe_router_bwd_bf16``
backward when that is also set).
"""

import importlib

from primus.backends.megatron.core.fusions._notices import warn_if_not_gfx1250
from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.moe.triton_router_gemm"
_ARG = "use_primus_triton_router_gemm"
_TARGET = "megatron.core.transformer.moe.router"


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="MoE router gating linear (fp32 logits) on pinned Triton kernels",
    condition=lambda ctx: bool(getattr(get_args(ctx), _ARG, False)),
    tags=["moe", "router", "triton"],
)
def patch_triton_router_gemm(ctx: PatchContext) -> None:
    warn_if_not_gfx1250(_PATCH_KEY, _ARG)
    from primus.backends.megatron.core.fusions.triton_router_gemm import (
        E_SUPPORTED,
        make_router_gating_linear,
    )

    args = get_args(ctx)
    uncovered = []
    if getattr(args, "moe_router_dtype", None) != "fp32":
        uncovered.append(f"moe_router_dtype={getattr(args, 'moe_router_dtype', None)!r} (needs 'fp32')")
    if getattr(args, "add_bias_linear", False):
        uncovered.append("add_bias_linear (router bias)")
    if getattr(args, "num_experts", None) not in E_SUPPORTED:
        uncovered.append(f"num_experts={getattr(args, 'num_experts', None)} (supported: {E_SUPPORTED})")
    if uncovered:
        warning_rank_0(
            f"[Patch:{_PATCH_KEY}] {', '.join(uncovered)}: the Triton router GEMM does not cover this "
            "setup; those calls keep the original path."
        )
    if getattr(args, "moe_router_bwd_bf16", False):
        log_rank_0(
            f"[Patch:{_PATCH_KEY}] moe_router_bwd_bf16 is also set: it applies only to the calls the "
            "Triton router GEMM does not cover."
        )
    mod = importlib.import_module(_TARGET)
    if is_patched(mod, _PATCH_KEY):
        return
    mod.router_gating_linear = make_router_gating_linear(mod.router_gating_linear, _PATCH_KEY)
    mark_patched(mod, _PATCH_KEY)
    log_rank_0(f"[Patch:{_PATCH_KEY}] patched {_TARGET}.router_gating_linear (pinned Triton fwd/dgrad/wgrad)")
