###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 Mixture-of-Experts module.

Reference: techblog §4 ("MoE: hash routing + sqrtsoftplus + shared experts")
and ``DeepSeek-V4-Flash/inference/model.py:MoE``.

V4's MoE block has three pieces:

1. **Router** — either :class:`HashRouter` (first ``num_hash_layers``
   layers) or :class:`V4TopKRouter` (the rest). Both produce the same
   ``(probs, routing_map)`` shape contract: ``[N, num_experts]``.
2. **Routed experts** — ``num_experts`` clamped-SwiGLU MLPs. Each token
   contributes to ``moe_router_topk`` of them, weighted by the router
   probability.
3. **Shared expert(s)** — always-on MLP(s) whose output is added to every
   token's contribution. V4-Flash has 1 shared expert with the same
   ``moe_intermediate_size`` as the routed experts.

This module is a **pure-PyTorch** implementation suitable for CPU /
single-GPU validation. It does not yet integrate with Megatron's MoE
token-dispatcher, EP, or grouped-GEMM — those land in P6 alongside the
TP / PP / EP wiring. The public API
(``forward(hidden, *, token_ids=None) -> [B, S, D]``) is stable across
P5 → P6: the dispatcher swap-in lives strictly inside ``forward``.

Phase 5 contract:
* Layer-aware: caller passes ``layer_idx``; values ``< num_hash_layers``
  use :class:`HashRouter`, otherwise :class:`V4TopKRouter`.
* Token-id aware: hash routing needs ``token_ids`` since the routing
  table is keyed on the token itself, not on the hidden state.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from primus.backends.megatron.core.transformer.clamped_swiglu import ClampedSwiGLUMLP
from primus.backends.megatron.core.transformer.moe.v4_hash_router import HashRouter
from primus.backends.megatron.core.transformer.moe.v4_topk_router import V4TopKRouter


class DeepseekV4MoE(nn.Module):
    """V4 MoE FFN sub-block.

    Args:
        hidden_size: model dim ``D``.
        moe_intermediate_size: per-expert FFN inner dim.
        num_routed_experts: number of routed experts.
        moe_router_topk: top-K experts per token.
        num_shared_experts: number of always-on shared experts (each of
            size ``moe_intermediate_size``). Set to 0 to disable.
        layer_idx: 0-based decoder layer index. Used to pick router type
            against ``num_hash_layers``.
        num_hash_layers: number of layers from the bottom that use the
            static :class:`HashRouter`.
        hash_vocab_size: vocab size for the hash routing table; required
            when ``layer_idx < num_hash_layers``.
        hash_seed: deterministic seed for the hash table.
        score_function: scoring function for the learned top-K router
            (one of ``"sqrtsoftplus" / "sigmoid" / "softmax"``).
        enable_expert_bias: whether the learned router uses a noaux_tc
            per-expert bias for selection.
        clamp_alpha: clamp bound for clamped SwiGLU (V4 default 7.0).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        moe_intermediate_size: int,
        num_routed_experts: int,
        moe_router_topk: int,
        num_shared_experts: int = 1,
        layer_idx: int,
        num_hash_layers: int = 0,
        hash_vocab_size: Optional[int] = None,
        hash_seed: int = 0,
        score_function: str = "sqrtsoftplus",
        enable_expert_bias: bool = True,
        clamp_alpha: float = 7.0,
    ) -> None:
        super().__init__()
        if num_routed_experts <= 0:
            raise ValueError(f"num_routed_experts must be > 0, got {num_routed_experts}")
        if moe_router_topk <= 0 or moe_router_topk > num_routed_experts:
            raise ValueError(
                f"moe_router_topk must be in [1, {num_routed_experts}], " f"got {moe_router_topk}"
            )
        if num_shared_experts < 0:
            raise ValueError(f"num_shared_experts must be >= 0, got {num_shared_experts}")

        self.hidden_size = int(hidden_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.num_routed_experts = int(num_routed_experts)
        self.moe_router_topk = int(moe_router_topk)
        self.num_shared_experts = int(num_shared_experts)
        self.layer_idx = int(layer_idx)
        self.num_hash_layers = int(num_hash_layers)
        self.use_hash_router = self.layer_idx < self.num_hash_layers

        # ---- router ----
        if self.use_hash_router:
            if hash_vocab_size is None or hash_vocab_size <= 0:
                raise ValueError(
                    "hash_vocab_size must be provided (and > 0) when " "layer_idx < num_hash_layers"
                )
            self.router = HashRouter(
                num_experts=num_routed_experts,
                topk=moe_router_topk,
                vocab_size=hash_vocab_size,
                seed=hash_seed,
            )
            self.learned_router = None
        else:
            self.router = None
            self.learned_router = V4TopKRouter(
                hidden_size=hidden_size,
                num_experts=num_routed_experts,
                topk=moe_router_topk,
                score_function=score_function,
                enable_expert_bias=enable_expert_bias,
                renormalize=True,
                topk_scaling_factor=1.0,
            )

        # ---- routed experts ----
        # Each expert is an independent ClampedSwiGLUMLP. P6 will swap
        # this for a grouped-GEMM-friendly layout when EP lands.
        self.experts = nn.ModuleList(
            [
                ClampedSwiGLUMLP(
                    hidden_size=hidden_size,
                    intermediate_size=moe_intermediate_size,
                    alpha=clamp_alpha,
                )
                for _ in range(num_routed_experts)
            ]
        )

        # ---- shared experts ----
        if num_shared_experts > 0:
            self.shared_expert = ClampedSwiGLUMLP(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size * num_shared_experts,
                alpha=clamp_alpha,
            )
        else:
            self.shared_expert = None

    # ------------------------------------------------------------------

    def _route(
        self,
        hidden: torch.Tensor,
        token_ids: Optional[torch.Tensor],
    ):
        """Return ``(probs, routing_map)`` for the current router."""
        if self.use_hash_router:
            assert self.router is not None
            if token_ids is None:
                raise ValueError(
                    f"layer {self.layer_idx} uses HashRouter; " "token_ids is required (shape [B, S])."
                )
            return self.router(token_ids)
        assert self.learned_router is not None
        return self.learned_router(hidden)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run V4 MoE FFN.

        Args:
            hidden: ``[B, S, D]`` input.
            token_ids: ``[B, S]`` integer token ids, required only when
                ``layer_idx < num_hash_layers``.

        Returns:
            ``[B, S, D]`` output. Sum of routed-expert and shared-expert
            contributions.
        """
        B, S, D = hidden.shape
        flat = hidden.reshape(-1, D)  # [N, D]
        flat.shape[0]

        probs, routing_map = self._route(hidden, token_ids)  # [N, E], bool
        # Cast probs to hidden dtype so the convex combo doesn't upcast
        # the rest of the residual stream.
        probs = probs.to(hidden.dtype)

        out = torch.zeros_like(flat)

        # Per-expert dispatch: gather tokens routed to expert i, run FFN,
        # scatter-add weighted output. Slow but correct; P6 swaps for
        # token-dispatcher + grouped-gemm.
        for i, expert in enumerate(self.experts):
            mask_i = routing_map[:, i]  # [N]
            if not mask_i.any():
                continue
            tokens_i = flat[mask_i]  # [n_i, D]
            y_i = expert(tokens_i)  # [n_i, D]
            w_i = probs[mask_i, i].unsqueeze(-1)  # [n_i, 1]
            out.index_add_(0, mask_i.nonzero(as_tuple=True)[0], y_i * w_i)

        if self.shared_expert is not None:
            out = out + self.shared_expert(flat)

        return out.reshape(B, S, D)


__all__ = ["DeepseekV4MoE"]
