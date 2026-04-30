###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Top-K MoE router for DeepSeek-V4 (sqrtsoftplus / sigmoid / softmax).

Reference: techblog §4 ("MoE: routing scoring") and
``DeepSeek-V4-Flash/inference/model.py:Gate``.

For layers with ``layer_idx >= num_hash_layers`` V4 uses a learned
top-K router. The novelty over V3 is the **scoring function**: V4
introduces ``sqrtsoftplus(x) = sqrt(softplus(x))`` which combines the
positive-only behavior of softplus with the sub-linear growth of sqrt;
it sits between ``sigmoid`` (saturating) and ``softmax`` (competition)
and gives smoother routing gradients in long training runs.

Optionally the router supports an **expert bias** correction
(``moe_router_enable_expert_bias`` / "noaux_tc" — V3-style auxiliary-free
balancing): a learnable per-expert bias is added to the score *only for
top-K selection*, but the returned probabilities use the un-biased score.
This keeps gradient flow clean while still letting the bias term steer
load balance.

Phase 5 contract:
* Standalone module that produces ``(probs, routing_map)`` with the same
  ``[N, num_experts]`` shape contract as :class:`HashRouter`. P6 will
  swap this for a Megatron-integrated version that participates in EP /
  token-dispatcher / TP routing.
* Score functions: ``"sqrtsoftplus"`` (V4 default), ``"sigmoid"`` (V3
  fallback), ``"softmax"`` (vanilla).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _score_fn(logits: torch.Tensor, *, score_function: str) -> torch.Tensor:
    """Apply V4-supported score functions to gate logits."""
    if score_function == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    if score_function == "sigmoid":
        return torch.sigmoid(logits)
    if score_function == "softmax":
        return F.softmax(logits, dim=-1)
    raise ValueError(
        f"Unknown score_function: {score_function!r}. "
        "Expected one of {'sqrtsoftplus', 'sigmoid', 'softmax'}."
    )


class V4TopKRouter(nn.Module):
    """Learned top-K router with V4 scoring options.

    Args:
        hidden_size: model dim ``D``; the gate is a single ``D -> num_experts``
            linear.
        num_experts: total number of routed experts.
        topk: number of experts each token is routed to.
        score_function: one of ``{"sqrtsoftplus", "sigmoid", "softmax"}``.
        enable_expert_bias: if True, allocate a learnable per-expert bias
            used for selection only ("noaux_tc"). Probabilities are
            re-read from the un-biased score.
        renormalize: if True, divide the selected top-K probs by their
            sum so each token's gate weights sum to 1. Matches the
            inference reference for ``sqrtsoftplus`` (recommended on).
        topk_scaling_factor: optional scalar multiplier applied to the
            renormalized probs (V3-style ``moe_router_topk_scaling_factor``).
        gate_dtype: dtype for the gate weights; if None matches input.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        topk: int,
        score_function: str = "sqrtsoftplus",
        enable_expert_bias: bool = False,
        renormalize: bool = True,
        topk_scaling_factor: float = 1.0,
        gate_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if topk <= 0 or topk > num_experts:
            raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")

        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.score_function = score_function
        self.renormalize = bool(renormalize)
        self.topk_scaling_factor = float(topk_scaling_factor)

        kw = {} if gate_dtype is None else {"dtype": gate_dtype}
        # No bias on the gate linear; the optional expert-bias is separate.
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, **kw)

        if enable_expert_bias:
            self.expert_bias = nn.Parameter(torch.zeros(num_experts, **kw))
        else:
            self.register_parameter("expert_bias", None)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route ``hidden`` to top-K experts.

        Args:
            hidden: ``[B, S, D]`` (or any shape with last dim ``D``).

        Returns:
            probs: ``[N, num_experts]`` float tensor, ``N = numel/D``.
                Non-selected experts have probability 0; selected experts
                hold the (optionally renormalized + scaled) score.
            routing_map: ``[N, num_experts]`` bool tensor; ``True`` at
                ``(n, e)`` iff token ``n`` is routed to expert ``e``.
        """
        flat = hidden.reshape(-1, self.hidden_size)
        logits = self.gate(flat)  # [N, E]

        # Score function applied to logits (un-biased).
        scores = _score_fn(logits, score_function=self.score_function)  # [N, E]

        # Top-K selection: with expert bias, add it to the selection score
        # only; return probs from un-biased ``scores``.
        if self.expert_bias is not None:
            sel_score = scores + self.expert_bias
        else:
            sel_score = scores
        topk_idx = sel_score.topk(self.topk, dim=-1).indices  # [N, K]

        # Gather un-biased score values for the selected experts.
        topk_score = scores.gather(1, topk_idx)  # [N, K]

        if self.renormalize:
            denom = topk_score.sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
            topk_score = topk_score / denom

        if self.topk_scaling_factor != 1.0:
            topk_score = topk_score * self.topk_scaling_factor

        N = flat.shape[0]
        device = flat.device
        probs = torch.zeros(N, self.num_experts, dtype=topk_score.dtype, device=device)
        probs.scatter_(1, topk_idx, topk_score)

        routing_map = torch.zeros(N, self.num_experts, dtype=torch.bool, device=device)
        routing_map.scatter_(1, topk_idx, True)

        return probs, routing_map


__all__ = ["V4TopKRouter", "_score_fn"]
