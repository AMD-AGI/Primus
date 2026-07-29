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

import os
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

# Query rows per chunk when building the KL target; see ``_target_chunk_size``.
_TARGET_CHUNK_ENV = "PRIMUS_V4_DISTILL_TARGET_CHUNK"
_TARGET_CHUNK_DEFAULT = 512

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


def _target_chunk_size(seq_len: int) -> int:
    """Query rows per chunk when building the KL target.

    The gathered pool is ``[B, S, K, head_dim]``, which at V4-Flash widths
    (S=4096, K=512, head_dim=512) is 2.1 GB in BF16 for a single microbatch.
    Materialising it whole costs more in HBM traffic than the GEMM that consumes
    it, so slice the query axis. ``0`` disables chunking.
    """
    override = os.environ.get(_TARGET_CHUNK_ENV)
    if override is not None:
        try:
            return max(int(override), 0)
        except ValueError:
            pass
    return _TARGET_CHUNK_DEFAULT if seq_len > _TARGET_CHUNK_DEFAULT else 0


def _target_distribution(
    *,
    query: torch.Tensor,
    pool: torch.Tensor,
    topk_idxs: torch.Tensor,
    valid: torch.Tensor,
    row_valid: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Head-summed attention distribution over the selected entries: ``[B, S, K]``.

    This is the KL target, and it is fully detached, so nothing here needs to be
    kept for backward. Three things follow from that and keep it cheap:

    * The gather and the GEMM run at the **model dtype** instead of being
      promoted to fp32 first. Promoting the gathered pool costs
      ``head_dim`` times more traffic than promoting the result, and the logits
      the main attention actually produces are computed at the model dtype too,
      so staying there tracks the distribution being imitated more closely, not
      less.
    * The head sum happens **inside** the loop, so the ``[B, H, S, K]`` logits
      never exist in full either -- only ``[B, H, chunk, K]``.
    * The query axis is chunked, so neither does the ``[B, S, K, head_dim]``
      gather, which at V4-Flash widths would be 2.1 GB per microbatch.
    """
    B, H, S, _ = query.shape
    K = topk_idxs.shape[-1]
    q = query.detach()
    kv = pool.detach()
    batch_idx = torch.arange(B, device=pool.device).view(B, 1, 1)

    def block(sl: slice) -> torch.Tensor:
        idx = topk_idxs[:, sl]
        # Clamp the -1 sentinels to a legal row; masked back out right after.
        gathered = kv[batch_idx, idx.clamp_min(0)]
        logits = torch.einsum("bhsd,bskd->bhsk", q[:, :, sl], gathered).float()
        logits *= softmax_scale
        logits.masked_fill_(~valid[:, sl].unsqueeze(1), float("-inf"))
        # An all -inf row would softmax to NaN; flatten it and drop it after.
        rows = row_valid[:, sl].view(B, 1, -1, 1)
        logits.masked_fill_(~rows, 0.0)
        probs = torch.softmax(logits, dim=-1)
        probs.mul_(rows)
        return probs.sum(dim=1)  # [B, chunk, K]

    chunk = _target_chunk_size(S)
    if chunk <= 0 or chunk >= S:
        return block(slice(0, S))

    out = torch.empty((B, S, K), device=query.device, dtype=torch.float32)
    for start in range(0, S, chunk):
        stop = min(start + chunk, S)
        out[:, start:stop] = block(slice(start, stop))
    return out


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
    B, _, S, _ = query.shape
    valid = topk_idxs >= 0  # [B, S, K]
    row_valid = valid.any(dim=-1)  # [B, S]

    # The KL target: the head-summed attention distribution over the same
    # entries. Fully detached (see the note above), so it is built under
    # ``no_grad`` and never has to materialise the dense per-head tensors.
    with torch.no_grad():
        target = _target_distribution(
            query=query,
            pool=pool,
            topk_idxs=topk_idxs,
            valid=valid,
            row_valid=row_valid,
            softmax_scale=softmax_scale,
        )
        # The head sum must span all heads before the renormalisation below.
        if head_reduce_group is not None and torch.distributed.get_world_size(head_reduce_group) > 1:
            target = target.contiguous()
            torch.distributed.all_reduce(target, group=head_reduce_group)
        # Renormalise to a distribution; L1 is enough since softmax outputs are
        # already non-negative.
        target /= target.sum(dim=-1, keepdim=True).clamp(min=_EPS)

    # The indexer side is the one that learns, so it stays attached. Rows with
    # zero legal entries are neutralised before the softmax and dropped after,
    # so a fully-masked row yields 0 rather than NaN and no gradient.
    idx_row_mask = row_valid.view(B, S, 1)
    idx_logits = index_topk_scores.float().masked_fill(~idx_row_mask, 0.0)
    idx_probs = torch.softmax(idx_logits, dim=-1, dtype=torch.float32) * idx_row_mask.to(torch.float32)

    kl_per_row = (target * (torch.log(target + _EPS) - torch.log(idx_probs + _EPS))).sum(dim=-1)
    return kl_per_row.mean() * loss_coeff
