###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MoE router gating GEMM (pinned Triton)
======================================

With ``use_primus_triton_router_gemm: true`` the router gating linear
(``router_gating_linear`` as bound in ``megatron.core.transformer.moe.router``,
which ``TopKRouter.gating`` and therefore ``PrimusTopKRouter`` look up at call
time) runs on three Triton kernels with one pinned config each
(``primus.backends.megatron.core.fusions.triton_router_gemm``; num_warps 4,
no autotune):

  fwd    bf16 x bf16 -> fp32 logits, fp32 accumulation
  dgrad  fp32 logits grad split exactly into three bf16 terms, fp32 accumulation
  wgrad  same split, fixed-order split-M reduction (deterministic, no atomics)

The backward keeps the precision of Megatron's fp32 GEMMs (fp32 accumulation is
the only rounding) without the fp32 copies of the input and the weight. At 32
experts these skinny GEMMs otherwise land on large generic hipBLASLt tiles.

Scope: ``moe_router_dtype: fp32``, no router bias, bf16 contiguous input with
tokens % 1024 == 0 and hidden % 128 == 0, 32 experts. Other calls run the
original function, including the ``moe_router_bwd_bf16`` backward when that is
also set (the two coexist: eligible calls take the Triton path). The routing
math itself (top-k, aux-loss scores, aux loss) is fused independently by
Megatron's ``moe_router_fusion`` (Transformer Engine's router kernels).
"""

from __future__ import annotations

import importlib

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PATCH_KEY = "megatron.moe.triton_router_gemm"
_ARG = "use_primus_triton_router_gemm"
_TARGET = "megatron.core.transformer.moe.router"


def patch_router_module() -> bool:
    """Replace ``router_gating_linear`` in Megatron's router module (idempotent). Returns True if newly patched."""
    from primus.backends.megatron.core.fusions import triton_router_gemm as ops

    mod = importlib.import_module(_TARGET)
    cur = mod.router_gating_linear
    if getattr(cur, ops.ORIGINAL_ATTR, None) is not None:
        return False
    mod.router_gating_linear = ops.make_router_gating_linear(cur)
    return True


@register_patch(
    _PATCH_KEY,
    backend="megatron",
    phase="before_train",
    description="MoE router gating linear (fp32 logits) on pinned Triton kernels",
    condition=lambda ctx: bool(getattr(get_args(ctx), _ARG, False)),
    tags=["moe", "router", "triton"],
)
def patch_triton_router_gemm(ctx: PatchContext) -> None:
    from primus.backends.megatron.core.fusions.triton_router_gemm import E_SUPPORTED

    args = get_args(ctx)
    uncovered = [
        what
        for what, hit in (
            (
                f"moe_router_dtype={getattr(args, 'moe_router_dtype', None)!r} (needs 'fp32')",
                getattr(args, "moe_router_dtype", None) != "fp32",
            ),
            ("add_bias_linear (router bias)", getattr(args, "add_bias_linear", False)),
            (
                f"num_experts={getattr(args, 'num_experts', None)} (supported: {E_SUPPORTED})",
                getattr(args, "num_experts", None) not in E_SUPPORTED,
            ),
        )
        if hit
    ]
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
    newly = patch_router_module()
    log_rank_0(
        f"[Patch:{_PATCH_KEY}] {'patched' if newly else 'already patched'} "
        f"{_TARGET}.router_gating_linear (pinned Triton fwd/dgrad/wgrad)"
    )
