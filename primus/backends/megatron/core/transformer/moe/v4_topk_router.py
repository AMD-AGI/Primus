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

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    scores = v4_score_fn(logits, score_function=score_function)
    original_scores = scores

    if expert_bias is not None:
        sel_score = scores + expert_bias.to(scores.dtype)
    else:
        sel_score = scores

    indices = sel_score.topk(topk, dim=-1).indices  # [N, K]
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
        return _compute_route(
            logits=logits,
            expert_bias=self.expert_bias,
            score_function=self.score_function,
            topk=self.topk,
            topk_scaling_factor=self.topk_scaling_factor,
        )


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
