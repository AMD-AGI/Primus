###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Autograd wrapper around the flydsl MSA forward and backward kernels."""

import torch

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
    build_block_plan,
    msa_token_bwd,
)
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
    msa_token_fwd,
)


class _MSAAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        block_table,
        softmax_scale,
        block_size,
        return_slot_lse,
        plan,
        shared_tiles=False,
        step_mask_skip=False,
    ):
        # lse / slot_lse never carry a gradient: don't materialise zeros for them
        ctx.set_materialize_grads(False)
        outs = msa_token_fwd(
            q,
            k,
            v,
            block_table,
            softmax_scale,
            block_size,
            return_slot_lse=return_slot_lse,
            shared_tiles=shared_tiles,
            step_mask_skip=step_mask_skip,
        )
        o, lse = outs[0], outs[1]
        ctx.save_for_backward(q, k, v, o, lse, block_table)
        ctx.softmax_scale = softmax_scale
        ctx.block_size = block_size
        ctx.plan = plan
        ctx.shared_tiles = shared_tiles
        ctx.step_mask_skip = step_mask_skip
        ctx.mark_non_differentiable(*outs[1:])
        return outs

    @staticmethod
    def backward(ctx, dout, *_):
        q, k, v, o, lse, block_table = ctx.saved_tensors
        dq, dk, dv = msa_token_bwd(
            dout,
            q,
            k,
            v,
            o,
            lse,
            block_table,
            ctx.softmax_scale,
            ctx.block_size,
            plan=ctx.plan,
            shared_tiles=ctx.shared_tiles,
            step_mask_skip=ctx.step_mask_skip,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def msa_attention(
    q,
    k,
    v,
    block_table,
    softmax_scale,
    block_size=128,
    return_slot_lse=False,
    plan=None,
    shared_tiles=False,
    walk_table=None,
    step_mask_skip=False,
):
    """Differentiable MSA over ``[S, B, H, 128]`` q/k/v.

    ``block_table`` is the indexer's ``[B, Hkv, S, topk]`` selection and
    ``plan`` its :class:`BlockPlan`, when the indexer built one; the backward
    builds it otherwise. Returns ``(o, lse)``, or ``(o, lse, slot_lse)``; only
    ``o`` carries a gradient. ``shared_tiles`` is ``msa_shared_tile_staging`` and ``step_mask_skip``
    ``msa_lean_masks`` (gfx1250 kernels, both bit-identical).

    ``walk_table`` (``msa_aligned_block_walk``, see ``block_walk.py``) is ``block_table`` with each
    row permuted, int32: the forward and dQ kernels walk a token's blocks in that order, and
    ``slot_lse`` then comes back in that order too. ``plan`` stays the plan of ``block_table``
    (dK/dV walks blocks, not rows).
    """
    q, k, v = (t.contiguous() for t in (q, k, v))
    table = getattr(plan, "table", None) if plan is not None else None
    if walk_table is not None:
        if walk_table.shape != block_table.shape or walk_table.dtype != torch.int32:
            raise ValueError("walk_table must be an int32 row permutation of block_table")
        if plan is None:
            # the plan belongs to the indexer's table: its entries are that table's slots
            plan = build_block_plan(block_table.to(torch.int32).contiguous(), block_size)
        block_table = walk_table.contiguous()
    elif table is not None and table.shape == block_table.shape:
        block_table = table  # select_plan's int32 copy of the same table
    else:
        block_table = block_table.to(torch.int32).contiguous()
    return _MSAAttention.apply(
        q,
        k,
        v,
        block_table,
        softmax_scale,
        block_size,
        return_slot_lse,
        plan,
        bool(shared_tiles),
        bool(step_mask_skip),
    )
