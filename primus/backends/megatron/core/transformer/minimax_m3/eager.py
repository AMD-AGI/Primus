###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eager block-sparse attention for MiniMax-M3.

The reference backend: plain PyTorch, no Transformer Engine, no fused kernel.
It materialises the full ``[b, h, sq, sk]`` score matrix, so it is quadratic in
sequence length and slower than the dense TE path it replaces. That is the
point -- it exists to be obviously correct and to be the thing a fast backend
(flydsl) is aligned against numerically.

Mirrors ``build_block_mask`` and ``eager_attention_forward`` in transformers'
``models/minimax_m3_vl/modeling_minimax_m3_vl.py``.
"""

from typing import Optional

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[b, n_kv, s, d]`` -> ``[b, n_kv * n_rep, s, d]``, as in the reference."""
    if n_rep == 1:
        return hidden_states
    b, n_kv, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)


def build_block_keep(block_indices: torch.Tensor, key_length: int, block_size: int) -> torch.Tensor:
    """Expand selected block ids into a per-key boolean keep mask.

    Args:
        block_indices: ``[b, n_index, sq, topk]``, ``-1`` marking unused slots.
        key_length: number of keys, ``sk``.
        block_size: keys per block.

    Returns:
        ``[b, n_index, sq, sk]`` bool, True where the query may read the key.
        One row per GQA group; expanding to query heads is the caller's job so
        the wide tensor is only built once.
    """
    b, n_index, sq, _ = block_indices.shape
    n_blocks = -(-key_length // block_size)

    # Park the -1 slots in a throwaway column, then drop it.
    safe = block_indices.masked_fill(block_indices < 0, n_blocks)
    keep_blocks = block_indices.new_zeros((b, n_index, sq, n_blocks + 1), dtype=torch.bool)
    keep_blocks.scatter_(-1, safe, True)
    keep_blocks = keep_blocks[..., :n_blocks]

    return keep_blocks.repeat_interleave(block_size, dim=-1)[..., :key_length]


def eager_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_keep: torch.Tensor,
    softmax_scale: float,
    attention_mask: Optional[torch.Tensor] = None,
    return_probs: bool = False,
):
    """Causal attention restricted to the selected blocks.

    Args:
        query: ``[b, n_q, sq, d]``.
        key, value: ``[b, n_kv, sk, d]``.
        block_keep: ``[b, n_kv, sq, sk]`` bool from :func:`build_block_keep`.
        softmax_scale: the usual ``1/sqrt(head_dim)``.
        attention_mask: optional Megatron mask broadcastable to
            ``[b, 1, sq, sk]``, ``True`` where the key must NOT be attended.
            Composed on top of causality, so padding is honoured.
        return_probs: also return the attention probabilities.

    Returns:
        ``(output [b, n_q, sq, d], dense_scores [b, n_q, sq, sk])``, plus
        ``probs [b, n_q, sq, sk]`` fp32 with ``return_probs``. The scores are the
        causal-but-not-block-masked logits, which the dense indexer loss distils
        from; the probabilities are the block-sparse attention itself, which the
        sparse indexer loss distils from.
    """
    b, n_q, sq, _ = query.shape
    n_kv = key.shape[1]
    n_rep = n_q // n_kv

    key_states = repeat_kv(key, n_rep)
    value_states = repeat_kv(value, n_rep)

    scores = torch.matmul(query.float(), key_states.float().transpose(2, 3)) * softmax_scale

    sk = key_states.shape[2]
    positions = torch.arange(sk, device=query.device)
    future = positions.view(1, 1, 1, sk) > positions[:sq].view(1, 1, sq, 1)
    if attention_mask is not None:
        future = future | attention_mask
    dense_scores = scores.masked_fill(future, float("-inf"))

    # One selection per GQA group -> one per query head, the way the reference
    # expands it (`block_keep.repeat_interleave(num_heads // n_idx_heads, dim=1)`).
    keep = block_keep.repeat_interleave(n_rep, dim=1)
    sparse_scores = dense_scores.masked_fill(~keep, float("-inf"))

    # A query the mask leaves with no attendable key at all -- a padding row --
    # would softmax to NaN. Zero those rows instead, and zero their output: a
    # padded position contributes nothing either way, and a NaN here is silent
    # and fatal. Real queries always keep their own (local) block.
    empty_row = ~torch.isfinite(sparse_scores.max(dim=-1, keepdim=True).values)
    sparse_scores = sparse_scores.masked_fill(empty_row, 0.0)

    probs = torch.softmax(sparse_scores, dim=-1).masked_fill(empty_row, 0.0)
    output = torch.matmul(probs.to(value_states.dtype), value_states)
    if return_probs:
        return output, dense_scores, probs
    return output, dense_scores
