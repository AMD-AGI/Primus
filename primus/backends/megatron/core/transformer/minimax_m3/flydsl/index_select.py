###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The MiniMax-M3 indexer's block selection, fused: O(S * n_blocks) memory.

Forward: ``index_block_max`` scores and max-pools in one kernel, then the
indexer's own boost + top-k runs on the ``[b, n, S, n_blocks]`` block scores.

Backward: a block score is the max over its keys, so its gradient lands on the
one winning key (the kernel's argmax) -- ``d q_t += g * k_j*`` and
``d k_j* += g * q_t``. Only the selected, non-forced slots can carry a
gradient: the sparse loss reads nothing else, and a forced block's +inf was
written over its score. So the backward gathers at those <= topk slots per
row instead of touching the whole block axis. dQ sums each token's slots
straight from the key rows (``index_dq``); dK groups the slots by winning key
with one stable sort and sums the sorted runs in fixed-size chunks
(``index_dk``) -- deterministic, without atomics, and balanced when a few
popular keys win most of the blocks, which real activations do.
"""

import torch

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.index_block_max import (
    index_block_max,
    index_dk,
    index_dq,
)


class _FusedSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, select_from_block_scores, block_size):
        raw, argmax = index_block_max(q, k, block_size)
        block_indices, block_scores = select_from_block_scores(raw, q.shape[0])

        slots = block_indices.clamp_min(0)
        keys = argmax.gather(-1, slots)
        forced = torch.isposinf(block_scores.gather(-1, slots))
        live = (block_indices >= 0) & ~forced & (keys >= 0)
        ctx.save_for_backward(q, k, slots, keys, live)
        ctx.mark_non_differentiable(block_indices)
        return block_indices, block_scores

    @staticmethod
    def backward(ctx, _grad_indices, grad_scores):
        q, k, slots, keys, live = ctx.saved_tensors
        S, B, H, D = q.shape
        if grad_scores is None:
            return torch.zeros_like(q), torch.zeros_like(k), None, None

        topk = slots.shape[-1]
        g = grad_scores.gather(-1, slots).masked_fill(~live, 0.0).float().contiguous()  # [B, H, S, topk]
        keys = keys.clamp_min(0).to(torch.int32)

        dq = index_dq(g, keys, k.reshape(S, B, D), H)

        # Dead slots get the sentinel key B * S: they sort last and dK never reads them.
        batch = torch.arange(B, device=q.device, dtype=torch.int32).view(B, 1, 1, 1)
        key_of_entry = torch.where(live, batch * S + keys, B * S).reshape(-1)
        sorted_keys, order = torch.sort(key_of_entry, stable=True)
        bounds = torch.searchsorted(
            sorted_keys, torch.arange(B * S + 1, device=q.device, dtype=torch.int32), out_int32=True
        )
        dk = index_dk(order.to(torch.int32), sorted_keys, bounds, g, q, topk)
        return dq, dk.reshape(k.shape), None, None


def fused_select_blocks(q, k, select_from_block_scores, block_size):
    """``(block_indices, block_scores)`` exactly as ``MinimaxM3Indexer.select_blocks``.

    Args:
        q: ``[S, B, n_index, D]`` bf16 index queries (normed, roped).
        k: ``[S, B, 1, D]`` bf16 index keys.
        select_from_block_scores: the indexer's boost + top-k.
    """
    return _FusedSelect.apply(q, k, select_from_block_scores, block_size)
