###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Sequence-wise load-balancing loss for the DeepSeek-V4 MoE router.

V4 balances experts with the auxiliary-loss-free bias (``expert_bias`` /
"noaux_tc"), "augmented by a slight sequence-wise balance loss that prevents
extreme imbalance within individual sequences" (paper 2.1). The bias reacts over
many steps and across the whole batch, so it cannot stop a single sequence from
collapsing onto a handful of experts; this loss is what covers that.

Following DeepSeek-V3, for each sequence of length ``T`` with ``E`` routed
experts and top-``K`` routing:

.. math::

    s'_{i,t} = s_{i,t} / \\sum_j s_{j,t}

    f_i = \\frac{E}{K T} \\sum_t \\mathbb{1}(\\text{token } t \\to \\text{expert } i)

    P_i = \\frac{1}{T} \\sum_t s'_{i,t}

    L = \\alpha \\sum_i f_i P_i

``f`` sums to ``E`` and ``P`` sums to 1, so a perfectly balanced sequence gives
exactly ``L = alpha`` -- which makes the logged value directly readable.

Why this is not the framework's built-in ``seq_aux_loss``: that path runs
through ``compute_routing_scores_for_aux_loss``, which only knows ``softmax``
and ``sigmoid`` and raises on V4's ``sqrtsoftplus``. The normalisation it
applies to ``sigmoid`` is the same one used here.

Two deliberate choices about which quantities feed the formula:

* ``f`` counts the **actual** routing, i.e. with ``expert_bias`` applied. The
  paper defines it over the tokens the router really dispatched, and the point of
  the loss is real load, not the router's unbiased preference.
* ``P`` uses the **unbiased** normalised affinities, which is what ``s'_{i,t}``
  denotes and the only part with a gradient.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "SEQ_BALANCE_LOSS_NAME",
    "log_seq_balance_loss",
    "normalise_affinities",
    "sequence_balance_loss",
]

# Reported through the framework's MoE aux-loss tracker.
SEQ_BALANCE_LOSS_NAME = "seq_load_balancing_loss"

_EPS = 1e-20


def normalise_affinities(scores: torch.Tensor) -> torch.Tensor:
    """``s'_{i,t} = s_{i,t} / sum_j s_{j,t}`` over the expert axis.

    V4's ``sqrtsoftplus`` score function is pointwise and non-negative but does
    not sum to one, so the affinities need normalising before they can act as
    the ``P_i`` distribution -- the same treatment the framework's aux-loss path
    gives ``sigmoid``.
    """
    return scores / (scores.sum(dim=-1, keepdim=True) + _EPS)


def sequence_balance_loss(
    *,
    scores: torch.Tensor,
    routing_map: torch.Tensor,
    topk: int,
    coeff: float,
    reduce_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> torch.Tensor:
    """``alpha * mean_b sum_i f_i P_i`` over the sequences in the batch.

    Args:
        scores: ``[B, S, E]`` **normalised** affinities (see
            :func:`normalise_affinities`); the differentiable input.
        routing_map: ``[B, S, E]`` bool, the routing actually dispatched.
        topk: experts per token.
        coeff: ``alpha``.
        reduce_group: group the sequence axis is sharded over (tensor +
            context parallel when sequence parallelism is on), or ``None`` when
            each rank holds whole sequences.

    Returns:
        Scalar loss in ``scores.dtype``. Under sharding this is the local
        rank's share: the expert counts are reduced so ``f`` is global, while
        ``P`` stays local so each rank differentiates only its own tokens --
        the same split the framework's aux-loss path uses. The reported value
        therefore needs the same group as its tracker reduction.
    """
    if scores.dim() != 3 or routing_map.shape != scores.shape:
        raise ValueError(
            f"scores and routing_map must both be [B, S, E]; got {tuple(scores.shape)} "
            f"and {tuple(routing_map.shape)}"
        )

    _, seq_len, num_experts = scores.shape

    counts = routing_map.sum(dim=1).to(scores.dtype)  # [B, E]
    score_sums = scores.sum(dim=1)  # [B, E], differentiable

    if reduce_group is not None and torch.distributed.get_world_size(reduce_group) > 1:
        world_size = torch.distributed.get_world_size(reduce_group)
        counts = counts.contiguous()
        # No gradient flows through the counts (they come from a bool mask), so
        # a plain all-reduce is enough.
        torch.distributed.all_reduce(counts, group=reduce_group)
        seq_len = seq_len * world_size

    # f_i * P_i = (E / (K*T)) * C_i * (1/T) * S_i, summed over experts.
    scale = num_experts / (float(topk) * float(seq_len) ** 2)
    per_sequence = (counts * score_sums).sum(dim=-1) * scale  # [B]
    return per_sequence.mean() * coeff


def log_seq_balance_loss(
    loss: torch.Tensor,
    *,
    layer_number: Optional[int],
    num_layers: int,
    reduce_group: Optional["torch.distributed.ProcessGroup"] = None,
) -> None:
    """Report the loss through the framework's MoE aux-loss tracker.

    Uses the same group :func:`sequence_balance_loss` was given, so the logged
    value is the global loss rather than one rank's share.
    """
    if not layer_number:
        return
    try:
        from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
    except Exception:
        return

    save_to_aux_losses_tracker(
        SEQ_BALANCE_LOSS_NAME,
        loss.detach(),
        layer_number,
        num_layers,
        reduce_group=reduce_group,
    )
