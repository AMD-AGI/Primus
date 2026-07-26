###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Indexer distillation loss for DeepSeek-V4 CSA layers.

The CSA lightning indexer picks which ``index_topk`` compressed KV entries each
query attends to. ``topk`` is not differentiable, so without an auxiliary
objective the indexer never receives a gradient and a from-scratch run selects
essentially at random.

DeepSeek-V3.2 (and Megatron-LM's DSv4 port, ``compute_dsa_indexer_loss`` in
``experimental_attention_variant/dsa.py``) trains it by distillation: the
indexer's score distribution is pulled towards the distribution the *real*
attention places over the same entries, via ``KL(attention || indexer)``.

This module implements the sparse variant -- the loss is evaluated only on the
entries the indexer actually selected, which is what Megatron's Flash recipe
uses (``dsa_indexer_use_sparse_loss: true``). Because those entries are already
gathered per query, the computation stays in the ``[B, S, K]`` top-k space and
never materialises the dense ``[B, H, S, P]`` score tensor.

The loss is attached to the autograd graph with :class:`V4IndexerLossAutoScaler`
(the same trick Megatron uses for MoE aux losses and MTP): it passes a tensor
through untouched in forward and seeds the auxiliary loss with a gradient of
one in backward, so the aux objective backpropagates without having to be
threaded through every forward return signature.
"""

from __future__ import annotations

import torch

__all__ = [
    "V4IndexerLossAutoScaler",
    "compute_indexer_distill_loss",
]

# Guard for log(0) / division by zero; matches Megatron's constant.
_EPS = 1e-10


class V4IndexerLossAutoScaler(torch.autograd.Function):
    """Attach an auxiliary loss to an existing tensor's backward pass.

    ``forward`` returns ``output`` unchanged; ``backward`` seeds ``aux_loss``
    with ``main_loss_backward_scale`` so its subgraph is differentiated as part
    of the main backward. Mirrors Megatron's ``MoEAuxLossAutoScaler`` /
    ``MTPLossAutoScaler``.
    """

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (aux_loss,) = ctx.saved_tensors
        scale = V4IndexerLossAutoScaler.main_loss_backward_scale.to(
            device=aux_loss.device, dtype=aux_loss.dtype
        )
        return grad_output, torch.ones_like(aux_loss) * scale

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Set the gradient seeded into the auxiliary loss.

        Pipeline / gradient-accumulation schedules use this to keep the aux
        loss on the same scale as the main loss.
        """
        V4IndexerLossAutoScaler.main_loss_backward_scale = scale


def compute_indexer_distill_loss(
    *,
    index_topk_scores: torch.Tensor,
    topk_idxs: torch.Tensor,
    query: torch.Tensor,
    pool: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
) -> torch.Tensor:
    """``KL(attention || indexer)`` over the selected compressed entries.

    Args:
        index_topk_scores: indexer scores at the selected slots, ``[B, S, K]``.
            Invalid slots are ``-inf`` (fewer than K legal entries exist for
            early queries).
        topk_idxs: selected pool indices, ``[B, S, K]``; ``-1`` marks invalid.
        query: post-RoPE queries, ``[B, H, S, head_dim]``.
        pool: compressed KV pool, ``[B, P, head_dim]``.
        softmax_scale: the attention softmax temperature (``1/sqrt(head_dim)``).
        loss_coeff: scaling applied to the KL.

    Returns:
        Scalar loss (fp32). Rows with no legal entry contribute nothing, so a
        fully-masked row can never produce NaN.
    """
    B, H, S, _ = query.shape
    valid = topk_idxs >= 0  # [B, S, K]
    row_valid = valid.any(dim=-1)  # [B, S]

    # Gather the selected pool entries: [B, S, K, head_dim]. Clamp the -1
    # sentinels to a legal index first; they are masked out below.
    batch_idx = torch.arange(B, device=pool.device).view(B, 1, 1)
    gathered = pool[batch_idx, topk_idxs.clamp_min(0)]

    # True attention logits over exactly the selected entries.
    attn_logits = torch.einsum("bhsd,bskd->bhsk", query.float(), gathered.float()) * softmax_scale
    attn_logits = attn_logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    idx_logits = index_topk_scores.float()

    # Rows with zero legal entries would softmax over all -inf and yield NaN.
    # Neutralise their logits before the softmax and drop their contribution
    # after it, so no NaN is ever produced (and no gradient flows from them).
    attn_row_mask = row_valid.view(B, 1, S, 1)
    idx_row_mask = row_valid.view(B, S, 1)
    attn_logits = attn_logits.masked_fill(~attn_row_mask, 0.0)
    idx_logits = idx_logits.masked_fill(~idx_row_mask, 0.0)

    attn_probs = torch.softmax(attn_logits, dim=-1, dtype=torch.float32) * attn_row_mask.to(torch.float32)
    idx_probs = torch.softmax(idx_logits, dim=-1, dtype=torch.float32) * idx_row_mask.to(torch.float32)

    # The indexer emits one distribution per query while attention has H heads,
    # so aggregate the heads and renormalise to a distribution (L1 is enough --
    # softmax outputs are already non-negative).
    target = attn_probs.sum(dim=1)  # [B, S, K]
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=_EPS)

    kl_per_row = (target * (torch.log(target + _EPS) - torch.log(idx_probs + _EPS))).sum(dim=-1)
    return kl_per_row.mean() * loss_coeff
