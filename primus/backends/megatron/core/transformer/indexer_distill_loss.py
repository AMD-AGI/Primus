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

DeepSeek-V3.2 (section 2.1) trains it by distillation: the indexer's score
distribution is pulled towards the distribution the *real* attention places over
the same entries, via ``KL(attention || indexer)``.

This module implements the sparse variant -- the loss is evaluated only on the
entries the indexer actually selected. That keeps the objective consistent with
what the forward pass actually consumes, and because those entries are already
gathered per query the computation stays in the ``[B, S, K]`` top-k space and
never materialises the dense ``[B, H, S, P]`` score tensor.

The gradient flows **one way**: into the indexer only. The target side (the
attention queries and the compressed pool) is detached inside
:func:`compute_indexer_distill_loss`, and the caller feeds the indexer from a
detached hidden state, so the KL can neither reshape the attention distribution
it is trying to imitate nor leak into the layers below.

The loss is attached to the autograd graph with :class:`V4IndexerLossAutoScaler`,
the same aux-loss autoscaler pattern this training stack already uses for the
MoE auxiliary loss and the MTP loss: it passes a tensor through untouched in
forward and seeds the auxiliary loss with a gradient of one in backward, so the
aux objective backpropagates without having to be threaded through every forward
return signature.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "INDEXER_DISTILL_LOSS_NAME",
    "V4IndexerLossAutoScaler",
    "compute_indexer_distill_loss",
    "log_indexer_distill_loss",
]

# Guard for log(0) / division by zero.
_EPS = 1e-10

# Key under which the loss is reported. Shares the framework's MoE aux-loss
# tracker, so it lands in the training log / TensorBoard / W&B next to the MoE
# losses with no extra plumbing.
INDEXER_DISTILL_LOSS_NAME = "indexer_distill_loss"


def _moe_aux_loss_scale() -> Optional[torch.Tensor]:
    """The scale the pipeline schedule installed for the MoE auxiliary loss.

    ``forward_step_calc_loss`` sets it once per microbatch to
    ``grad_scale * cp_size / num_microbatches`` (or just ``grad_scale`` under
    ``calculate_per_token_loss``) whenever ``num_moe_experts`` is configured --
    which is always true for V4. The indexer distillation loss needs exactly
    that quantity, so it reads it rather than maintaining a second copy that
    could silently drift. Returns ``None`` when the framework is not importable
    (the torch-only unit tests).
    """
    try:
        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    except Exception:
        return None
    return getattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)


def log_indexer_distill_loss(
    loss: Optional[torch.Tensor],
    *,
    layer_number: Optional[int],
    num_layers: int,
    device: torch.device,
) -> None:
    """Record this layer's indexer loss in the MoE aux-loss tracker.

    Call this from **every** V4 attention layer whenever the loss is enabled,
    passing ``None`` on the layers that do not own an indexer. The tracker is
    reduced across pipeline ranks over whatever keys each rank happens to hold,
    so a key that only appears on the ranks that own a CSA layer would make the
    collective diverge. Non-CSA layers therefore contribute an explicit zero,
    which keeps the key present everywhere and leaves the sum unchanged.

    The reported value is the per-layer sum divided by the layer count (the
    denominator ``track_moe_metrics`` applies to every tracked loss), not
    divided by the number of CSA layers.
    """
    # ``layer_number`` is 1-based and indexes the tracker directly, so 0 (the
    # "unnumbered" default on a standalone attention module) would silently
    # write to the last slot.
    if not layer_number:
        return
    try:
        from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
    except Exception:
        return

    value = (
        loss.detach().to(device=device, dtype=torch.float32)
        if loss is not None
        else torch.zeros((), device=device, dtype=torch.float32)
    )
    save_to_aux_losses_tracker(INDEXER_DISTILL_LOSS_NAME, value, layer_number, num_layers)


class V4IndexerLossAutoScaler(torch.autograd.Function):
    """Attach an auxiliary loss to an existing tensor's backward pass.

    ``forward`` returns ``output`` unchanged; ``backward`` seeds ``aux_loss``
    with the current aux-loss scale so its subgraph is differentiated as part
    of the main backward. Same shape as the autoscalers already used for the MoE
    auxiliary loss and the MTP loss.

    The scale is not maintained here. Seeding a gradient of one would make the
    effective coefficient ``num_microbatches`` times too large under gradient
    accumulation and would ignore the grad scaler entirely, so by default this
    follows :func:`_moe_aux_loss_scale` -- the same per-microbatch quantity the
    schedule installs for the MoE auxiliary loss. :meth:`set_loss_scale`
    overrides it for schedules that need to drive it explicitly.
    """

    # ``None`` means "follow the MoE aux-loss scale"; a tensor is an explicit
    # override installed via ``set_loss_scale``.
    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def current_loss_scale(reference: torch.Tensor) -> torch.Tensor:
        """Resolve the scale to seed, on ``reference``'s device / dtype."""
        scale = V4IndexerLossAutoScaler.main_loss_backward_scale
        if scale is None:
            scale = _moe_aux_loss_scale()
        if scale is None:
            return torch.ones((), device=reference.device, dtype=reference.dtype)
        return scale.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (aux_loss,) = ctx.saved_tensors
        scale = V4IndexerLossAutoScaler.current_loss_scale(aux_loss)
        return grad_output, torch.ones_like(aux_loss) * scale

    @staticmethod
    def set_loss_scale(scale: Optional[torch.Tensor]) -> None:
        """Override the gradient seeded into the auxiliary loss.

        Pass ``None`` to go back to following the MoE auxiliary loss scale.
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
    head_reduce_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> torch.Tensor:
    """``KL(attention || indexer)`` over the selected compressed entries.

    Distillation is **one-directional**: only the indexer learns. ``query`` and
    ``pool`` are detached here so the KL cannot pull the main attention's Q
    projection or the compressor towards whatever the indexer currently
    predicts -- the open-source reference does the same, detaching both the
    query and the key on the way into its indexer loss and feeding the indexer
    itself from a detached hidden state.

    Args:
        index_topk_scores: indexer scores at the selected slots, ``[B, S, K]``.
            The one tensor that stays attached -- it is the learning signal.
            Invalid slots are ``-inf`` (fewer than K legal entries exist for
            early queries).
        topk_idxs: selected pool indices, ``[B, S, K]``; ``-1`` marks invalid.
        query: post-RoPE queries, ``[B, H, S, head_dim]``. Detached internally.
        pool: compressed KV pool, ``[B, P, head_dim]``. Detached internally.
        softmax_scale: the attention softmax temperature (``1/sqrt(head_dim)``).
        loss_coeff: scaling applied to the KL.
        head_reduce_group: process group the attention heads are sharded over,
            or ``None`` when every rank holds all of them. Only needed if the Q
            projection stops gathering its output -- see
            :meth:`DeepseekV4Attention._indexer_loss_head_group`.

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
    gathered = pool.detach()[batch_idx, topk_idxs.clamp_min(0)]

    # True attention logits over exactly the selected entries -- the KL target,
    # hence detached (see the note above).
    attn_logits = torch.einsum("bhsd,bskd->bhsk", query.detach().float(), gathered.float()) * softmax_scale
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
    # softmax outputs are already non-negative). The head sum must span all
    # heads before the renormalisation, hence the optional all-reduce.
    target = attn_probs.sum(dim=1)  # [B, S, K]
    if head_reduce_group is not None and torch.distributed.get_world_size(head_reduce_group) > 1:
        target = target.contiguous()
        torch.distributed.all_reduce(target, group=head_reduce_group)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=_EPS)

    kl_per_row = (target * (torch.log(target + _EPS) - torch.log(idx_probs + _EPS))).sum(dim=-1)
    return kl_per_row.mean() * loss_coeff
