###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hash router for DeepSeek-V4's first ``num_hash_layers`` MoE layers.

Reference: techblog §4 ("MoE: hash routing for the first N layers") and
the inference reference at ``DeepSeek-V4-Flash/inference/model.py:Gate``
(the ``self.hash`` branch).

For the first ``num_hash_layers`` of V4 (V4-Flash = 3), expert
**selection** is static: each token id is permanently assigned to a
fixed set of ``moe_router_topk`` experts via a deterministic
``tid2eid`` table. Routing **weights**, however, are *not* uniform —
they come from the same learned linear gate as the non-hash layers; we
just gather the scores at the prescribed expert ids instead of running
a top-K argmax.

Released-checkpoint contract (per HF ``Gate.__init__``):

* ``weight`` : ``nn.Parameter`` of shape ``[num_experts, hidden_size]``
  — the learned gate. State-dict key matches the learned router so the
  V4 state-dict adapter can map ``mlp.gate.weight`` uniformly.
* ``tid2eid`` : ``nn.Parameter`` of shape ``[vocab_size, topk]``, dtype
  ``int32``, ``requires_grad=False``. The released checkpoint stores
  it as a parameter (not a buffer) so it is preserved across
  ``state_dict`` round-trips.

Forward semantics:

    scores         = score_fn(linear(hidden, weight))   # fp32
    indices        = tid2eid[token_ids]                 # static
    weights        = scores.gather(1, indices)          # learned weights
    if score_fn != softmax: weights /= weights.sum(-1)
    weights       *= route_scale

Plan-2 P14 contract:

* :class:`DeepseekV4HashRouter` is a standalone ``nn.Module`` that
  produces sparse ``(probs, routing_map)`` with the same ``[N,
  num_experts]`` shape contract as :class:`DeepseekV4LearnedRouter`.
* ``score_function`` ∈ ``{"softmax", "sigmoid", "sqrtsoftplus"}`` and
  ``topk_scaling_factor`` are honored identically.

Back-compat alias ``HashRouter`` is exposed but deprecated; new
callers should use :class:`DeepseekV4HashRouter`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
    _VALID_SCORE_FUNCTIONS,
    v4_score_fn,
)


def _build_default_tid2eid(*, vocab_size: int, num_experts: int, topk: int, seed: int) -> torch.Tensor:
    """Build a deterministic ``[vocab_size, topk]`` int32 expert table.

    Each token id gets ``topk`` distinct expert ids drawn uniformly
    without replacement from ``[0, num_experts)``. The seed is fixed
    across all ranks so PP / TP / EP shards see identical routing.

    The table layout matches the HF reference: ``int32`` to keep on-disk
    size small, indexed via long-cast at gather time.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    rows = []
    for _ in range(int(vocab_size)):
        perm = torch.randperm(num_experts, generator=gen)[:topk]
        rows.append(perm)
    table = torch.stack(rows, dim=0).to(torch.int32)
    return table


class DeepseekV4HashRouter(nn.Module):
    """Static hash-based MoE router with a *learned* score gate.

    Args:
        hidden_size: model dim ``D``; the gate is a single ``D ->
            num_experts`` linear shared in shape with the learned router.
        num_experts: total number of routed experts.
        topk: number of experts each token is routed to.
        vocab_size: tokenizer vocabulary size; controls the table length.
        seed: deterministic seed for the hash table; same across all
            ranks. Used only when ``tid2eid`` is built locally; if a
            checkpoint provides ``tid2eid`` directly, the seed is
            ignored at load time.
        score_function: one of ``{"softmax", "sigmoid", "sqrtsoftplus"}``;
            applied to the learned scores (matches the learned router).
        topk_scaling_factor: scalar multiplier applied to the
            renormalized routing weights (V3 ``moe_router_topk_scaling_factor``,
            HF ``Gate.route_scale``).
        dtype: dtype of the gate weight; defaults to fp32.

    Parameters:
        weight: ``[num_experts, hidden_size]`` learned gate.
        tid2eid: ``[vocab_size, topk]`` int32 frozen mapping.
            ``requires_grad=False``; this is a parameter (matching the
            HF reference checkpoint) so that ``state_dict`` round-trips
            preserve it.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        topk: int,
        vocab_size: int,
        seed: int = 0,
        score_function: str = "sqrtsoftplus",
        topk_scaling_factor: float = 1.0,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if topk <= 0 or topk > num_experts:
            raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if score_function not in _VALID_SCORE_FUNCTIONS:
            raise ValueError(
                f"Unknown score_function: {score_function!r}. "
                f"Expected one of {sorted(_VALID_SCORE_FUNCTIONS)}."
            )

        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self.score_function = str(score_function)
        self.topk_scaling_factor = float(topk_scaling_factor)

        weight_dtype = dtype or torch.float32
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, dtype=weight_dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        # tid2eid is a non-trainable parameter (matches HF reference layout
        # so checkpoint round-trips include it). int32 to keep memory small.
        tid2eid_init = _build_default_tid2eid(
            vocab_size=self.vocab_size,
            num_experts=self.num_experts,
            topk=self.topk,
            seed=self.seed,
        )
        self.tid2eid = nn.Parameter(tid2eid_init, requires_grad=False)

    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens whose expert ids are prescribed by ``tid2eid``.

        Args:
            hidden: ``[B, S, D]`` (or any shape with last dim ``D``).
                The learned gate is evaluated against ``hidden``; the
                resulting scores supply the *weights*.
            token_ids: ``[B, S]`` (or any compatible shape) integer
                tensor of token ids. Provides the *indices* via
                ``tid2eid``.

        Returns:
            probs: ``[N, num_experts]`` float tensor, ``N = numel/D``.
                Non-selected experts have probability 0; selected
                experts hold the (renormalized + scaled) score.
            routing_map: ``[N, num_experts]`` bool tensor; ``True`` at
                ``(n, e)`` iff token ``n`` is routed to expert ``e``.
        """
        if (token_ids.dtype != torch.long) and (token_ids.dtype != torch.int):
            raise TypeError(f"DeepseekV4HashRouter expects integer token_ids, got {token_ids.dtype}")
        flat_hidden = hidden.reshape(-1, self.hidden_size)
        flat_ids = token_ids.reshape(-1).long()
        if flat_hidden.shape[0] != flat_ids.shape[0]:
            raise ValueError(
                "DeepseekV4HashRouter: hidden and token_ids must flatten to the same length; "
                f"got hidden={flat_hidden.shape[0]} vs token_ids={flat_ids.shape[0]}."
            )
        if flat_ids.numel() == 0:
            n_zero = 0
            device = flat_hidden.device
            probs = torch.zeros(n_zero, self.num_experts, dtype=torch.float32, device=device)
            routing_map = torch.zeros(n_zero, self.num_experts, dtype=torch.bool, device=device)
            return probs, routing_map
        if int(flat_ids.max().item()) >= self.vocab_size:
            raise ValueError(
                f"token_ids has values >= vocab_size ({self.vocab_size}); "
                f"max found = {int(flat_ids.max().item())}"
            )

        # Learned scores (fp32) over the full expert axis.
        logits = F.linear(flat_hidden.to(torch.float32), self.weight.to(torch.float32))
        scores = v4_score_fn(logits, score_function=self.score_function)  # [N, E]

        # Static expert assignment from the table — cast to long for gather.
        indices = self.tid2eid[flat_ids].long()  # [N, K]

        # Routing weights: gather learned scores at the prescribed expert ids.
        weights = scores.gather(1, indices)  # [N, K]

        if self.score_function != "softmax":
            denom = weights.sum(dim=-1, keepdim=True).clamp(min=1.0e-12)
            weights = weights / denom

        if self.topk_scaling_factor != 1.0:
            weights = weights * float(self.topk_scaling_factor)

        N = flat_hidden.shape[0]
        device = flat_hidden.device

        probs = torch.zeros(N, self.num_experts, dtype=weights.dtype, device=device)
        probs.scatter_(1, indices, weights)

        routing_map = torch.zeros(N, self.num_experts, dtype=torch.bool, device=device)
        routing_map.scatter_(1, indices, True)

        return probs, routing_map


# Back-compat alias. New callers should use ``DeepseekV4HashRouter``.
HashRouter = DeepseekV4HashRouter

__all__ = [
    "DeepseekV4HashRouter",
    "HashRouter",
]
