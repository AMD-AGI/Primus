###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MiniMax-M3 indexer's block selection, fused: O(S * n_blocks) memory.

Forward: ``index_block_max`` scores and max-pools in one kernel, then the
indexer's own boost + top-k runs on the ``[b, n, S, n_blocks]`` block scores.
When a backward will follow, the selection's :class:`BlockPlan` -- the block
table inverted into per-block entry lists -- is built here once and handed to
the caller, whose attention backward walks the same lists.

Backward: a block score is the max over its keys, so its gradient lands on the
one winning key (the kernel's argmax) -- ``d q_t += g * k_j*`` and
``d k_j* += g * q_t``. Only the selected, non-forced slots can carry a
gradient: the sparse loss reads nothing else, and a forced block's +inf was
written over its score. So the backward gathers at those <= topk slots per
row instead of touching the whole block axis. dQ sums each token's slots
straight from the key rows (``index_dq``). dK walks the plan block by block
(``index_dk``): a block's keys only receive gradient from the slots that
picked it, which is exactly a plan row -- deterministic, no atomics, and no
sort of its own.
"""

import torch

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
    index_block_max,
    index_dk,
    index_dq,
)
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
    build_block_plan,
)


class _FusedSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, select_from_block_scores, block_size, plan_slot):
        raw, argmax = index_block_max(q, k, block_size)
        block_indices, block_scores = select_from_block_scores(raw, q.shape[0])

        slots = block_indices.clamp_min(0)
        keys = argmax.gather(-1, slots)
        forced = torch.isposinf(block_scores.gather(-1, slots))
        live = (block_indices >= 0) & ~forced & (keys >= 0)
        ctx.save_for_backward(q, k, block_indices, keys, live)
        ctx.block_size = block_size
        # filled by fused_select_blocks once the selection exists
        ctx.plan_slot = plan_slot
        ctx.mark_non_differentiable(block_indices)
        return block_indices, block_scores

    @staticmethod
    def backward(ctx, _grad_indices, grad_scores):
        q, k, block_indices, keys, live = ctx.saved_tensors
        S, B, H, D = q.shape
        if grad_scores is None:
            return torch.zeros_like(q), torch.zeros_like(k), None, None, None

        g = (
            grad_scores.gather(-1, block_indices.clamp_min(0)).masked_fill(~live, 0.0).float()
        )  # [B, H, S, topk]
        dq = index_dq(g, keys.clamp_min(0), k.reshape(S, B, D), H)
        plan = ctx.plan_slot[0] if ctx.plan_slot else build_block_plan(block_indices, ctx.block_size)
        dk = index_dk(plan, g, keys, q, ctx.block_size)
        return dq, dk.reshape(k.shape), None, None, None


def fused_select_blocks(q, k, select_from_block_scores, block_size):
    """``(block_indices, block_scores, plan)``: the first two exactly as
    ``MinimaxM3Indexer.select_blocks``; ``plan`` is the selection's
    :class:`BlockPlan` when grad is enabled (a backward will need it), else None.

    Args:
        q: ``[S, B, n_index, D]`` bf16 index queries (normed, roped).
        k: ``[S, B, 1, D]`` bf16 index keys.
        select_from_block_scores: the indexer's boost + top-k.
    """
    plan_slot = []
    block_indices, block_scores = _FusedSelect.apply(q, k, select_from_block_scores, block_size, plan_slot)
    plan = None
    if torch.is_grad_enabled():
        plan = build_block_plan(block_indices, block_size)
        plan_slot.append(plan)
    return block_indices, block_scores, plan
