###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autograd wrapper around the flydsl MSA forward and backward kernels."""

import torch

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
    msa_token_bwd,
)
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
    msa_token_fwd,
)


class _MSAAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_table, softmax_scale, block_size, return_slot_lse):
        outs = msa_token_fwd(q, k, v, block_table, softmax_scale, block_size, return_slot_lse=return_slot_lse)
        o, lse = outs[0], outs[1]
        ctx.save_for_backward(q, k, v, o, lse, block_table)
        ctx.softmax_scale = softmax_scale
        ctx.block_size = block_size
        ctx.mark_non_differentiable(*outs[1:])
        return outs

    @staticmethod
    def backward(ctx, dout, *_):
        q, k, v, o, lse, block_table = ctx.saved_tensors
        dq, dk, dv = msa_token_bwd(dout, q, k, v, o, lse, block_table, ctx.softmax_scale, ctx.block_size)
        return dq, dk, dv, None, None, None, None


def msa_attention(q, k, v, block_table, softmax_scale, block_size=128, return_slot_lse=False):
    """Differentiable MSA over ``[S, B, H, 128]`` q/k/v.

    ``block_table`` is the indexer's ``[B, Hkv, S, topk]`` selection. Returns
    ``(o, lse)``, or ``(o, lse, slot_lse)``; only ``o`` carries a gradient.
    """
    q, k, v = (t.contiguous() for t in (q, k, v))
    block_table = block_table.to(torch.int32).contiguous()
    return _MSAAttention.apply(q, k, v, block_table, softmax_scale, block_size, return_slot_lse)
