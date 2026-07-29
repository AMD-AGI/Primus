###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Learned Top-K MoE router for DeepSeek-V4.

Reference: techblog §4 ("MoE: routing scoring") and the inference
reference at ``DeepSeek-V4-Flash/inference/model.py:Gate.forward``.

For layers with ``layer_idx >= num_hash_layers`` V4 uses a learned
top-K router. The HF-released ``Gate`` module computes a single
``[D -> num_experts]`` linear and supports three score functions:

* ``softmax`` — standard competitive normalization.
* ``sigmoid`` — independent expert scoring (V3 fallback).
* ``sqrtsoftplus`` — V4 default. ``sqrt(softplus(x))`` combines the
  positive-only behavior of softplus with the sub-linear growth of
  sqrt; sits between sigmoid (saturating) and softmax (competition)
  and yields smoother routing gradients in long training runs.

Optionally the router supports an **expert bias** correction
(``moe_router_enable_expert_bias`` / "noaux_tc" — V3-style auxiliary-free
balancing): a learnable per-expert bias is added to the score *only for
top-K selection*, and the returned routing weights are gathered from the
**un-biased** scores. This keeps gradient flow clean (probs flow back to
the gate weight, not the bias) while still letting the bias term steer
load balance.

After top-K selection, the routing weights are renormalized to sum to 1
**only when the score function is non-softmax** (matches HF; with
softmax the sum is already 1 by construction). A final scalar
``topk_scaling_factor`` ("route_scale" in the HF reference) is applied
multiplicatively.

Plan-2 P14 contract:

* :class:`DeepseekV4LearnedRouter` — standalone ``nn.Module`` that
  produces sparse ``(probs, routing_map)`` with the same ``[N, num_experts]``
  shape contract as Megatron's :class:`TopKRouter`. The eager,
  CPU-testable form is the canonical reference for G4 unit tests.
* Parameter layout:
   - ``weight``: ``nn.Parameter`` of shape ``[num_experts, hidden_size]``
     (matches both Megatron's ``TopKRouter.weight`` and HF reference
     ``Gate.weight``).
   - ``expert_bias`` (optional): ``nn.Parameter`` of shape
     ``[num_experts]`` (matches HF reference ``Gate.bias``).
* ``score_function`` ∈ ``{"softmax", "sigmoid", "sqrtsoftplus"}``.

Phase-2 of P14 will subclass Megatron's :class:`TopKRouter` directly so
the router participates in aux-loss / z-loss / dispatcher lifecycle in
production. The standalone form here remains the reference for unit
tests and the state-dict adapter (P17).

Back-compat alias ``V4TopKRouter`` is exposed but deprecated; new
callers should use :class:`DeepseekV4LearnedRouter`.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.moe._triton.v4_router_post import (
    is_triton_path_enabled as _v4_router_triton_enabled,
)
from primus.backends.megatron.core.transformer.moe._triton.v4_router_post import (
    v4_router_post_triton,
)
from primus.backends.megatron.core.transformer.moe.v4_seq_balance_loss import (
    log_seq_balance_loss,
    normalise_affinities,
    sequence_balance_loss,
)

logger = logging.getLogger(__name__)

_VALID_SCORE_FUNCTIONS = {"softmax", "sigmoid", "sqrtsoftplus"}


def v4_score_fn(logits: torch.Tensor, *, score_function: str) -> torch.Tensor:
    """Apply a V4-supported score function to gate logits.

    Args:
        logits: ``[..., num_experts]`` tensor of pre-score linear
            outputs. Must be float (fp32 in the HF reference; we follow).
        score_function: one of ``"softmax"``, ``"sigmoid"``,
            ``"sqrtsoftplus"``.

    Returns:
        Tensor of the same shape, post score-function. ``softmax`` sums
        to 1 along the expert axis; the other two are pointwise.
    """
    if score_function == "softmax":
        return F.softmax(logits, dim=-1)
    if score_function == "sigmoid":
        return torch.sigmoid(logits)
    if score_function == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    raise ValueError(
        f"Unknown score_function: {score_function!r}. " f"Expected one of {sorted(_VALID_SCORE_FUNCTIONS)}."
    )


def _compute_route(
    *,
    logits: torch.Tensor,
    expert_bias: Optional[torch.Tensor],
    score_function: str,
    topk: int,
    topk_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared selection / renormalization core for V4 routers.

    Returns sparse ``(probs[N, E], routing_map[N, E])``. The dense
    ``(weights[N, K], indices[N, K])`` form follows the HF reference;
    we expose only the sparse contract here so downstream Megatron
    dispatchers consume it directly.

    NOTE: this helper assumes ``logits`` is already shaped ``[N,
    num_experts]`` and in fp32. Callers are responsible for the cast.
    """
    # Pre-compute topk indices on the host (heavy GPU compute that
    # benefits from being its own kernel).  When PRIMUS_V4_ROUTER_TRITON
    # is on (and supported), route the rest of the chain through one
    # fused Triton kernel; otherwise fall back to the eager body.
    scores_for_selection = v4_score_fn(logits, score_function=score_function)
    if expert_bias is not None:
        sel_score = scores_for_selection + expert_bias.to(scores_for_selection.dtype)
    else:
        sel_score = scores_for_selection
    indices = sel_score.topk(topk, dim=-1).indices  # [N, K]

    # The V4 router is GPU-only: both the Triton and eager paths run on
    # CUDA/HIP tensors (there is no CPU compute path).
    assert logits.is_cuda, "V4 router requires CUDA / HIP tensors"
    if _v4_router_triton_enabled():
        # The Triton kernel re-applies the score function inside (so
        # it can save the full row for backward).  This keeps the
        # eager path's bit-equivalent behaviour for the gathered
        # weights at the cost of one extra pass over [N, E] of fp32
        # logits -- negligible at V4-Flash widths.
        probs, routing_map = v4_router_post_triton(
            logits,
            indices,
            score_function=score_function,
            topk_scaling_factor=topk_scaling_factor,
            out_dtype=scores_for_selection.dtype,
        )
        return probs, routing_map

    # Eager fallback (verbatim pre-P39 body).
    original_scores = scores_for_selection
    weights = original_scores.gather(1, indices)  # [N, K]

    if score_function != "softmax":
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
        weights = weights / denom

    if topk_scaling_factor != 1.0:
        weights = weights * float(topk_scaling_factor)

    num_experts = logits.shape[-1]
    N = logits.shape[0]
    device = logits.device

    probs = torch.zeros(N, num_experts, dtype=weights.dtype, device=device)
    probs.scatter_(1, indices, weights)

    routing_map = torch.zeros(N, num_experts, dtype=torch.bool, device=device)
    routing_map.scatter_(1, indices, True)

    return probs, routing_map


class DeepseekV4LearnedRouter(nn.Module):
    """Learned top-K router for DeepSeek-V4 MoE layers (l >= num_hash_layers).

    Args:
        hidden_size: model dim ``D``; the gate is a single ``D -> num_experts``
            linear.
        num_experts: total number of routed experts.
        topk: number of experts each token is routed to.
        score_function: one of ``{"softmax", "sigmoid", "sqrtsoftplus"}``.
            V4 default is ``"sqrtsoftplus"``.
        enable_expert_bias: if True, allocate a learnable per-expert bias
            used for selection only ("noaux_tc"). Probabilities are
            re-read from the un-biased score so probs gradient flows
            only into ``weight``, not ``expert_bias``.
        topk_scaling_factor: scalar multiplier applied to the
            renormalized probs (V3-style ``moe_router_topk_scaling_factor``,
            HF reference ``Gate.route_scale``). Defaults to ``1.0``.
        dtype: dtype of the gate weight; defaults to fp32 (matches HF
            reference; the routing math runs in fp32 regardless).
        seq_balance_loss_coeff: coefficient of the sequence-wise balance loss
            (paper 2.1). ``0`` disables it. The caller is responsible for
            checking ``moe_router_load_balancing_type``, so this single number
            fully decides whether the loss runs.
        seq_balance_reduce_group: group the sequence axis is sharded over, or
            ``None`` when each rank holds whole sequences.
        layer_number: 1-based layer index, and ``num_layers`` the total, both
            only used to report the loss. Reporting is skipped without them.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        topk: int,
        score_function: str = "sqrtsoftplus",
        enable_expert_bias: bool = False,
        topk_scaling_factor: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        seq_balance_loss_coeff: float = 0.0,
        seq_balance_reduce_group: Optional["torch.distributed.ProcessGroup"] = None,
        layer_number: Optional[int] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if topk <= 0 or topk > num_experts:
            raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")
        if score_function not in _VALID_SCORE_FUNCTIONS:
            raise ValueError(
                f"Unknown score_function: {score_function!r}. "
                f"Expected one of {sorted(_VALID_SCORE_FUNCTIONS)}."
            )

        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.score_function = str(score_function)
        self.topk_scaling_factor = float(topk_scaling_factor)
        self.seq_balance_loss_coeff = float(seq_balance_loss_coeff or 0.0)
        # Held by reference so the group is not treated as module state.
        self._seq_balance_reduce_group = seq_balance_reduce_group
        self.layer_number = layer_number
        self.num_layers = num_layers
        # Last computed value (detached), for tests and ad-hoc inspection.
        self.last_seq_balance_loss: Optional[torch.Tensor] = None

        weight_dtype = dtype or torch.float32
        # Gate weight: [num_experts, hidden_size] (matches Megatron TopKRouter
        # AND the HF reference Gate.weight). State-dict key: ``weight``.
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, dtype=weight_dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if enable_expert_bias:
            # Per-expert selection bias. State-dict key: ``expert_bias``.
            self.expert_bias = nn.Parameter(torch.zeros(self.num_experts, dtype=weight_dtype))
        else:
            self.register_parameter("expert_bias", None)

    # ------------------------------------------------------------------

    @property
    def seq_balance_loss_enabled(self) -> bool:
        """Whether the sequence-wise balance loss is on for this router."""
        return self.seq_balance_loss_coeff > 0.0

    def _apply_seq_balance_loss(
        self,
        probs: torch.Tensor,
        logits: torch.Tensor,
        routing_map: torch.Tensor,
        *,
        batch_size: int,
        seq_length: int,
    ) -> torch.Tensor:
        """Attach the sequence-wise balance loss to ``probs``.

        ``probs`` is returned unchanged in value; the loss rides along in the
        autograd graph via ``MoEAuxLossAutoScaler``, the same mechanism the
        framework's own routers use, so the schedule's per-microbatch loss scale
        applies automatically.
        """
        num_experts = logits.shape[-1]
        # Unbiased, normalised affinities -- the ``s'_{i,t}`` of the paper.
        affinities = normalise_affinities(v4_score_fn(logits, score_function=self.score_function))

        loss = sequence_balance_loss(
            scores=affinities.view(batch_size, seq_length, num_experts),
            routing_map=routing_map.view(batch_size, seq_length, num_experts),
            topk=self.topk,
            coeff=self.seq_balance_loss_coeff,
            reduce_group=self._seq_balance_reduce_group,
        )
        self.last_seq_balance_loss = loss.detach()

        if self.num_layers:
            log_seq_balance_loss(
                loss,
                layer_number=self.layer_number,
                num_layers=int(self.num_layers),
                reduce_group=self._seq_balance_reduce_group,
            )

        from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

        return MoEAuxLossAutoScaler.apply(probs, loss)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route ``hidden`` to top-K experts.

        Args:
            hidden: ``[B, S, D]`` (or any shape with last dim ``D``).

        Returns:
            probs: ``[N, num_experts]`` float tensor, ``N = numel/D``.
                Non-selected experts have probability 0; selected experts
                hold the (possibly renormalized + scaled) un-biased
                score.
            routing_map: ``[N, num_experts]`` bool tensor; ``True`` at
                ``(n, e)`` iff token ``n`` is routed to expert ``e``.
        """
        flat = hidden.reshape(-1, self.hidden_size)
        # Match HF reference: routing math runs in fp32 regardless of
        # input dtype.
        logits = F.linear(flat.to(torch.float32), self.weight.to(torch.float32))
        probs, routing_map = _compute_route(
            logits=logits,
            expert_bias=self.expert_bias,
            score_function=self.score_function,
            topk=self.topk,
            topk_scaling_factor=self.topk_scaling_factor,
        )

        # The balance loss is per-sequence, so it needs the [B, S] structure
        # that ``flat`` just collapsed. V4 carries activations as [B, S, D]
        # throughout (not the framework's usual [S, B, D]), so the sequence is
        # axis -2.
        if (
            self.seq_balance_loss_enabled
            and self.training
            and torch.is_grad_enabled()
            and hidden.dim() >= 3
        ):
            seq_length = int(hidden.shape[-2])
            num_tokens = flat.shape[0]
            if seq_length > 0 and num_tokens % seq_length == 0:
                probs = self._apply_seq_balance_loss(
                    probs,
                    logits,
                    routing_map,
                    batch_size=num_tokens // seq_length,
                    seq_length=seq_length,
                )
            else:
                # Grouping tokens into the wrong sequences would silently change
                # what the loss measures, so skip rather than guess.
                logger.warning(
                    "[V4-router] skipping the sequence-wise balance loss: %d tokens do not "
                    "divide into sequences of %d (input shape %s)",
                    num_tokens,
                    seq_length,
                    tuple(hidden.shape),
                )

        return probs, routing_map


# Back-compat alias. New callers should use ``DeepseekV4LearnedRouter``.
V4TopKRouter = DeepseekV4LearnedRouter

# Back-compat alias for the standalone score-function helper. The leading
# underscore was dropped because the helper is part of the test surface.
_score_fn = v4_score_fn

__all__ = [
    "DeepseekV4LearnedRouter",
    "V4TopKRouter",
    "v4_score_fn",
    "_score_fn",
]
