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
        dq_from_ds=False,
        walk_slots=None,
        owner_carry=False,
        ds_chunk_store=False,
        dq_early_issue=False,
        two_tile_slots=False,
        dkdv_two_token_steps=False,
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
            owner_carry=owner_carry,
            two_tile_slots=two_tile_slots,
        )
        o, lse = outs[0], outs[1]
        ctx.save_for_backward(q, k, v, o, lse, block_table)
        ctx.softmax_scale = softmax_scale
        ctx.block_size = block_size
        ctx.plan = plan
        ctx.shared_tiles = shared_tiles
        ctx.step_mask_skip = step_mask_skip
        ctx.dq_from_ds = dq_from_ds
        ctx.walk_slots = walk_slots
        ctx.ds_chunk_store = ds_chunk_store
        ctx.dq_early_issue = dq_early_issue
        ctx.dkdv_two_token_steps = dkdv_two_token_steps
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
            dq_from_ds=ctx.dq_from_ds,
            walk_slots=ctx.walk_slots,
            ds_chunk_store=ctx.ds_chunk_store,
            dq_early_issue=ctx.dq_early_issue,
            dkdv_two_token_steps=ctx.dkdv_two_token_steps,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


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
    dq_from_ds=False,
    walk_slots=None,
    owner_carry=False,
    ds_chunk_store=False,
    dq_early_issue=False,
    two_tile_slots=False,
    dkdv_two_token_steps=False,
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
    (dK/dV walks blocks, not rows); without one it is built here (with the selection's chunk
    rule, ``msa_bounded_plan_chunks``), and only when a backward can follow (grad mode on and q,
    k or v requiring grad).

    ``dq_from_ds`` (``msa_dq_from_ds``, ``msa_bwd_store_ds``; ``msa_token_bwd``): the backward takes dQ
    from the dK/dV kernel's dS. With a ``walk_table`` it needs ``walk_slots``, the int32 slot of
    each walk position in ``block_table``; without them the backward keeps the reference dQ kernel.

    ``owner_carry`` (``msa_owner_carry``) and ``ds_chunk_store`` (``msa_ds_chunk_store``, with
    ``dq_from_ds``): the forward / the dS stores with fewer scalar table loads and buffer
    descriptors per step; same bits as without them (``msa_token_fwd_wave32``, ``msa_token_bwd``).

    ``dq_early_issue`` (``msa_dq_early_issue``, with ``dq_from_ds``): the slim dQ kernel issues a
    step's copies one wait earlier; same bits (``msa_token_bwd``).

    ``dkdv_two_token_steps`` (``msa_dkdv_two_token_steps``): the backward's dK/dV kernel takes two
    tokens per step instead of four; same bits, checked on the first call for a shape.

    ``two_tile_slots`` (``msa_two_tile_slots``, with ``owner_carry``): the forward with two LDS
    tile slots per wave instead of four; same bits (``msa_token_fwd_wave32``).
    """
    q, k, v = (t.contiguous() for t in (q, k, v))
    table = getattr(plan, "table", None) if plan is not None else None
    if walk_table is not None:
        if walk_table.shape != block_table.shape or walk_table.dtype != torch.int32:
            raise ValueError("walk_table must be an int32 row permutation of block_table")
        if plan is None and torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
            # the plan belongs to the indexer's table (its entries are that table's slots), with the
            # indexer's chunk rule; only a backward reads it
            from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_select import (
                bounded_plan_chunks_enabled,
            )

            plan = build_block_plan(
                block_table.to(torch.int32).contiguous(), block_size, bounded_plan_chunks_enabled()
            )
        block_table = walk_table.contiguous()
        if dq_from_ds and walk_slots is None:
            dq_from_ds = False
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
        bool(dq_from_ds),
        walk_slots if dq_from_ds else None,
        bool(owner_carry),
        bool(ds_chunk_store) and bool(dq_from_ds),
        bool(dq_early_issue) and bool(dq_from_ds),
        bool(two_tile_slots) and bool(owner_carry),
        bool(dkdv_two_token_steps),
    )
