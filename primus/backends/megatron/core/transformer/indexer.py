###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

r"""
DeepSeek-V4 Indexer (sparse position selector for CSA).

Reference: techblog §1.4 ("Indexer: CSA's Sparse Selector").

The Indexer is **only** used by CSA layers (``compress_ratio == 4``). For
each query position ``t`` it picks ``index_topk`` compressed-KV positions
``s`` (out of all compressed positions ``[0, P)`` where ``P = S // ratio``).

Math (from the techblog):

.. math::

    q^Q_t = h_t W^{DQ},\quad q^I_{t,h} = q^Q_t W^{IUQ}_h,\quad
    w^I_{t,h} = h_t W^w_h

.. math::

    I_{t,s} = \\sum_h w^I_{t,h}\\cdot \\mathrm{ReLU}(q^I_{t,h}\\cdot K^{IComp}_s)

.. math::

    \\mathrm{topk\\_idxs}_t = \\mathrm{argTopK}_s\\,I_{t,s}

The Indexer carries its **own** mini-Compressor (``index_head_dim``,
``index_n_heads``); the ``K^{IComp}`` it produces is independent of the
main attention's compressed KV pool. It is only used to **select** top-k
positions; the actual values fetched into main attention come from the
main Compressor in the surrounding CSA layer.

Phase 4 contract:
* Plain ``nn.Linear`` projections (TP integration in P6).
* Causal masking: positions ``s`` whose start raw-token index exceeds the
  query's raw-token index get a value of ``-inf`` so they cannot be
  selected. Out-of-range positions are returned as ``-1`` in the output
  ``topk_idxs`` so the caller can treat them as "no key".
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.core.transformer.compressor import Compressor


class Indexer(nn.Module):
    """Sparse position selector for CSA.

    Args:
        hidden_size: input feature dim ``D`` (same as main attention).
        index_head_dim: head dim used by the mini-Compressor and the
            low-rank query projection.
        index_n_heads: number of indexer "heads".
        index_topk: number of compressed positions to select per query.
        compress_ratio: ratio ``m`` of the mini-Compressor (matches the
            main Compressor of the surrounding CSA layer; usually ``4``).
        dq_rank: rank of the shared low-rank query projection ``W^{DQ}``.
            Defaults to ``index_head_dim`` (the V4 reference doesn't expose
            a separate setting; ``W^{IUQ}_h`` then projects from ``dq_rank``
            to ``index_head_dim``).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        compress_ratio: int = 4,
        dq_rank: int = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.compress_ratio = compress_ratio
        self.dq_rank = dq_rank if dq_rank is not None else index_head_dim

        # W^{DQ}: low-rank query down-projection.
        self.w_dq = nn.Linear(hidden_size, self.dq_rank, bias=False)
        # W^{IUQ}_h: per-head up-projection from dq_rank → index_head_dim.
        self.w_iuq = nn.Linear(self.dq_rank, index_n_heads * index_head_dim, bias=False)
        # W^w_h: per-head scalar weight.
        self.w_w = nn.Linear(hidden_size, index_n_heads, bias=False)

        # Mini-Compressor producing K^{IComp}.
        self.indexer_compressor = Compressor(
            hidden_size=hidden_size,
            head_dim=index_head_dim,
            ratio=compress_ratio,
        )

    # ------------------------------------------------------------------

    def _causal_mask(
        self,
        n_queries: int,
        n_pool: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return ``[n_queries, n_pool]`` mask: 0.0 if pool position ``s``
        is allowed for query ``t``, ``-inf`` otherwise.

        A compressed position ``s`` covers raw tokens ``[s*ratio, (s+1)*ratio)``;
        a query at raw token ``t`` may attend to ``s`` iff its window
        end ``(s+1)*ratio - 1 <= t``.
        """
        t_idx = torch.arange(n_queries, device=device).unsqueeze(1)  # [t, 1]
        s_end = (torch.arange(n_pool, device=device).unsqueeze(0) + 1) * self.compress_ratio - 1  # [1, s]
        allowed = s_end <= t_idx  # [t, s] bool
        mask = torch.where(allowed, 0.0, float("-inf")).to(dtype)
        return mask

    # ------------------------------------------------------------------

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k compressed positions for each query.

        Args:
            hidden: ``[B, S, D]``.

        Returns:
            ``(topk_idxs, topk_scores)`` where:
            * ``topk_idxs`` ``[B, S, K]`` (long): selected pool positions
              in ``[0, P)`` for valid slots, ``-1`` for masked / invalid.
            * ``topk_scores`` ``[B, S, K]``: the selection scores ``I_{t,s}``
              (``-inf`` for masked positions).
        """
        B, S, D = hidden.shape
        assert S % self.compress_ratio == 0, (
            f"Indexer: sequence length {S} not divisible by compress_ratio " f"{self.compress_ratio}"
        )
        K = self.index_topk
        H = self.index_n_heads
        Hd = self.index_head_dim

        # 1) K^{IComp}: pool hidden via the mini-Compressor → [B, P, Hd]
        k_icomp = self.indexer_compressor(hidden)  # [B, P, Hd]
        P = k_icomp.shape[1]
        k_icomp = k_icomp.unsqueeze(2)  # [B, P, 1, Hd]

        # 2) Per-head query and per-head weight.
        q_q = self.w_dq(hidden)  # [B, S, dq_rank]
        q_i = self.w_iuq(q_q).view(B, S, H, Hd)  # [B, S, H, Hd]
        w_i = self.w_w(hidden)  # [B, S, H]

        # 3) Score I_{t,s} = Σ_h w_i[t,h] * ReLU(q_i[t,h] · k_icomp[s])
        #    q_i [B,S,H,Hd] · k_icomp[B,P,Hd] → relu[B,S,H,P]; w_i[B,S,H,1] → sum over H
        relu = F.relu(torch.einsum("bshd,bpd->bshp", q_i, k_icomp.squeeze(2)))
        scores = (relu * w_i.unsqueeze(-1)).sum(dim=2)  # [B, S, P]

        # 4) Causal mask + (effective topk capped at P).
        mask = self._causal_mask(S, P, scores.device, scores.dtype)  # [S, P]
        scores = scores + mask.unsqueeze(0)  # [B, S, P]

        topk_eff = min(K, P)
        topk_scores, topk_idxs = scores.topk(topk_eff, dim=-1)  # [B, S, topk_eff]

        # 5) Replace selections that are still -inf (i.e. fewer than K valid
        #    pool positions for very early queries) with sentinel ``-1`` so
        #    callers can drop them.
        invalid = torch.isneginf(topk_scores)
        topk_idxs = torch.where(invalid, torch.full_like(topk_idxs, -1), topk_idxs)

        # 6) Pad with -1 to exactly K columns if topk_eff < K (S smaller than
        #    K * ratio in unit tests).
        if topk_eff < K:
            pad_idxs = torch.full((B, S, K - topk_eff), -1, dtype=topk_idxs.dtype, device=topk_idxs.device)
            pad_scores = torch.full(
                (B, S, K - topk_eff), float("-inf"), dtype=topk_scores.dtype, device=topk_scores.device
            )
            topk_idxs = torch.cat([topk_idxs, pad_idxs], dim=-1)
            topk_scores = torch.cat([topk_scores, pad_scores], dim=-1)

        return topk_idxs, topk_scores


__all__ = ["Indexer"]
