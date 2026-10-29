###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Report fused attention FLOPs to ``torch.utils.flop_counter.FlopCounterMode``.

Kernels such as AITER flash attention are opaque custom ops with no FLOP formula,
so ``FlopCounterMode`` counts them as zero. That is also what AMDiffusionBenchmark
reports. The marker below can add the FLOPs of the equivalent
``scaled_dot_product_attention`` call, but only when
``PRIMUS_COUNT_ATTENTION_FLOPS=1``. It stays off by default so benchmark TFLOPS
match AMDiffusionBenchmark.
"""

from __future__ import annotations

import os

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from torch.utils.flop_counter import (
    register_flop_formula,
    sdpa_backward_flop_count,
    sdpa_flop_count,
)


@torch.library.custom_op("primus_diffusion::attention_flops", mutates_args=())
def _attention_flops(anchor: torch.Tensor, flops: int) -> None:
    return None


@_attention_flops.register_fake
def _(anchor, flops):
    return None


@register_flop_formula(torch.ops.primus_diffusion.attention_flops)
def _attention_flops_formula(anchor_shape, flops, *args, out_shape=None, **kwargs) -> int:
    return int(flops)


def flop_counter_active() -> bool:
    return any(type(mode).__name__ == "_FlopCounterMode" for mode in _get_current_dispatch_mode_stack())


def count_attention_flops() -> bool:
    return os.environ.get("PRIMUS_COUNT_ATTENTION_FLOPS", "0") == "1"


class _BackwardFlops(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, flops):
        ctx.flops = flops
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        torch.ops.primus_diffusion.attention_flops(grad, ctx.flops)
        return grad, None


def mark_attention_flops(
    out: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    qk_pairs: int,
    v_head_dim: int | None = None,
) -> torch.Tensor:
    """Attribute SDPA-equivalent FLOPs for ``qk_pairs`` = sum over sequences of ``len_q * len_k``."""
    if not flop_counter_active() or not count_attention_flops():
        return out
    dv = head_dim if v_head_dim is None else v_head_dim
    q_shape, k_shape, v_shape = (1, heads, qk_pairs, head_dim), (1, heads, 1, head_dim), (1, heads, 1, dv)
    torch.ops.primus_diffusion.attention_flops(out, sdpa_flop_count(q_shape, k_shape, v_shape))
    if not (torch.is_grad_enabled() and out.requires_grad):
        return out
    bwd = sdpa_backward_flop_count((1, heads, qk_pairs, dv), q_shape, k_shape, v_shape)
    return _BackwardFlops.apply(out, bwd)


def varlen_qk_pairs(cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor) -> int:
    q_lens = cu_seqlens_q.diff().tolist()
    k_lens = cu_seqlens_k.diff().tolist()
    return int(sum(lq * lk for lq, lk in zip(q_lens, k_lens, strict=True)))
