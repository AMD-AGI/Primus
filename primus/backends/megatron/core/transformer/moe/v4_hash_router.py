###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Hash router for DeepSeek-V4's first ``num_hash_layers`` MoE layers.

Reference: techblog §4 ("MoE: hash routing for the first N layers") and
``DeepSeek-V4-Flash/inference/model.py:HashRouter``.

For the first ``num_hash_layers`` of V4 (V4-Flash = 3), routing is
**static**: each token id is permanently assigned to a fixed set of
``moe_router_topk`` experts via a deterministic ``tid2eid`` table. There
are no learned routing weights, no balancing loss, and no top-K softmax.
Gate weights are fixed at ``1 / topk`` so the convex combination is just
an average across the routed experts.

This is identical for both training and inference: the table is built
once, deterministically, from the tokenizer vocab size and a fixed seed,
which keeps the routing consistent across PP / TP / EP ranks.

Phase 5 contract:
* ``HashRouter(num_experts, topk, vocab_size, seed=...)`` builds the
  table and does pure-PyTorch dispatch. Returns
  ``(probs[B*S, num_experts], routing_map[B*S, num_experts])``, the same
  shape Megatron's :class:`TopKRouter` returns, so a downstream
  dispatcher can be wired in identically in P6.
* The ``tid2eid`` table is a ``[vocab_size, topk]`` long tensor stored as
  a non-trainable buffer.

Phase 6 will integrate with Megatron's :class:`MoELayer` /
``token_dispatcher`` (notably making sure the EP ``expert_to_rank`` map
matches the static routing table). For P5 we keep the router self-contained
so it can be unit-tested on CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class HashRouter(nn.Module):
    """Static hash-based MoE router.

    Args:
        num_experts: total number of routed experts.
        topk: number of experts each token is routed to.
        vocab_size: tokenizer vocabulary size; controls the table length.
        seed: deterministic seed for the hash; same across all ranks.
        dtype: dtype of the returned ``probs`` tensor; defaults to
            ``torch.float32``.

    Buffers:
        tid2eid: ``[vocab_size, topk]`` long tensor mapping token id to a
            fixed set of expert ids.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        topk: int,
        vocab_size: int,
        seed: int = 0,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if topk <= 0 or topk > num_experts:
            raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")

        self.num_experts = int(num_experts)
        self.topk = int(topk)
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        self._dtype = dtype or torch.float32

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        # For each token id, pick ``topk`` distinct expert ids deterministically.
        # randperm(num_experts) is a stable, dense permutation; slicing the
        # first ``topk`` rows gives uniform-without-replacement routing.
        rows = []
        for _ in range(vocab_size):
            perm = torch.randperm(num_experts, generator=gen)[:topk]
            rows.append(perm)
        tid2eid = torch.stack(rows, dim=0).long()  # [vocab_size, topk]
        self.register_buffer("tid2eid", tid2eid, persistent=False)

    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map token ids to expert assignments.

        Args:
            token_ids: ``[B, S]`` (or any shape) integer tensor of token
                ids. Will be flattened internally.

        Returns:
            probs: ``[N, num_experts]`` float tensor where ``N`` is
                ``token_ids.numel()``. Every routed expert gets weight
                ``1/topk``; un-routed experts get 0. Returned in
                ``self._dtype``.
            routing_map: ``[N, num_experts]`` bool tensor; ``True`` at
                ``(n, e)`` iff token ``n`` is routed to expert ``e``.
        """
        if (token_ids.dtype != torch.long) and (token_ids.dtype != torch.int):
            raise TypeError(f"HashRouter expects integer token_ids, got {token_ids.dtype}")
        if token_ids.numel() == 0:
            n_zero = 0
            probs = torch.zeros(n_zero, self.num_experts, dtype=self._dtype, device=token_ids.device)
            routing_map = torch.zeros(n_zero, self.num_experts, dtype=torch.bool, device=token_ids.device)
            return probs, routing_map
        if int(token_ids.max().item()) >= self.vocab_size:
            raise ValueError(
                f"token_ids has values >= vocab_size ({self.vocab_size}); "
                f"max found = {int(token_ids.max().item())}"
            )

        flat_ids = token_ids.reshape(-1).long()  # [N]
        eids = self.tid2eid[flat_ids]  # [N, topk]
        N = flat_ids.shape[0]
        device = flat_ids.device

        routing_map = torch.zeros(N, self.num_experts, dtype=torch.bool, device=device)
        routing_map.scatter_(1, eids, True)

        weight = 1.0 / float(self.topk)
        probs = torch.zeros(N, self.num_experts, dtype=self._dtype, device=device)
        probs.scatter_(1, eids, weight)

        return probs, routing_map


__all__ = ["HashRouter"]
