###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Compressed-Sparse Attention (CSA) for DeepSeek-V4.

Reference: techblog §1 ("Hybrid Attention"), CSA branch (``compress_ratio == 4``).

CSA augments the dense / SWA attention with a **sparse** compressed-KV
selection: an :class:`Indexer` picks ``index_topk`` compressed-pool
positions per query, and only those positions are attended to.

Compared to HCA:
* Compressor uses **overlap** mode (``coff=2``) so adjacent compressed
  positions share window context.
* Indexer's mini-Compressor is *separate* from the main Compressor — it
  only selects positions; the main Compressor produces the values that
  flow into attention.
* The mask makes unselected compressed positions ``-inf`` so they do not
  receive softmax weight.
"""

from __future__ import annotations

from typing import Tuple

import torch

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.compressor import Compressor
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    _LegacyDeepseekV4Attention,
)
from primus.backends.megatron.core.transformer.dual_rope import (
    apply_interleaved_partial_rope,
)
from primus.backends.megatron.core.transformer.indexer import Indexer


class CSAAttention(_LegacyDeepseekV4Attention):
    """CSA attention layer.

    Args:
        compress_ratio: must be ``4`` per V4 release; we leave it
            parametric for testability.
        See :class:`DeepseekV4Attention` for the rest.
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        rope,
        compress_ratio: int = 4,
        submodules=None,
    ) -> None:
        if compress_ratio <= 0:
            raise ValueError(f"CSAAttention requires compress_ratio > 0, got {compress_ratio}")
        index_topk = int(config.index_topk)
        index_head_dim = int(config.index_head_dim)
        index_n_heads = int(config.index_n_heads)
        compressor_overlap = bool(compress_ratio == 4)
        super().__init__(
            config=config,
            rope=rope,
            compress_ratio=compress_ratio,
            submodules=submodules,
        )

        # Main Compressor (its output is the actual K/V of the sparse branch).
        self.compressor = Compressor(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            ratio=compress_ratio,
            overlap=compressor_overlap,
        )

        # Indexer (its own mini-Compressor inside).
        self.index_topk = index_topk
        self.indexer = Indexer(
            hidden_size=self.hidden_size,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            compress_ratio=compress_ratio,
        )

    # ------------------------------------------------------------------

    def _gather_topk_kv(
        self,
        pool: torch.Tensor,  # [B, P, head_dim]
        topk_idxs: torch.Tensor,  # [B, S, K]  (-1 for masked)
    ) -> torch.Tensor:
        """Gather ``[B, P, head_dim]`` along ``P`` per query → ``[B, S, K, head_dim]``.

        Out-of-range / masked indices (``-1``) are clamped to ``0`` for the
        gather, then *zero-masked* afterwards.
        """
        B, S, K = topk_idxs.shape
        P, Hd = pool.shape[1], pool.shape[2]
        valid = topk_idxs >= 0  # [B, S, K]
        safe_idx = topk_idxs.clamp(min=0)
        # Expand idx to gather along P for each (B, S, K, Hd).
        idx_expand = safe_idx.unsqueeze(-1).expand(B, S, K, Hd)
        pool_expand = pool.unsqueeze(1).expand(B, S, P, Hd)  # [B, S, P, Hd]
        gathered = torch.gather(pool_expand, dim=2, index=idx_expand)  # [B, S, K, Hd]
        gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
        return gathered, valid

    # ------------------------------------------------------------------

    def _extra_kv(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the sparse compressed-KV branch via Compressor + Indexer.

        Output shapes (so the parent's cat along the Sk axis works):

        * ``extra_k`` / ``extra_v``: ``[B, K, H, head_dim]`` — but K is the
          *per-query* selection, not a global Sk. To keep the interface
          simple and tensor-shape consistent, we lay out the selected
          KV as a "broadcast-pooled" pseudo-Sk of length ``K`` and emit
          a per-query mask that turns off all but each query's own K.

        Concretely we still produce the full per-query selection
        ``[B, S, K, H, head_dim]``, but the parent's eager attention path
        works with ``[B, H, Sq, Sk]`` logits — so we instead express CSA as
        a *per-query* attention over its top-K, computed inline here, and
        return ``None`` from this hook to skip the parent's cat-attention
        path. The parent's ``forward`` then mixes our per-query result back
        in before the o_proj.

        Implementation note: rather than fight the parent's cat-based path,
        we inject CSA results by overriding ``_compute_attention_output``
        instead. See :meth:`_compute_attention_output` below; this hook
        just stashes the compressed pool / topk idxs on the module so the
        attention computation can read them.
        """
        B, S, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        # 1) Pool hidden via main Compressor.
        pooled = self.compressor(hidden)  # [B, P, head_dim]
        P = pooled.shape[1]

        # 2) Compress-base RoPE on the pool.
        pool_kv = pooled.unsqueeze(2)  # [B, P, 1, head_dim]
        comp_pos = torch.arange(P, device=device)
        cos, sin = self.rope.compress_rope(comp_pos)
        cos = cos[..., : self.rotary_dim // 2]
        sin = sin[..., : self.rotary_dim // 2]
        pool_kv = apply_interleaved_partial_rope(
            pool_kv, cos, sin, rotary_dim=self.rotary_dim
        )  # [B, P, 1, head_dim]
        pool_kv = pool_kv.squeeze(2)  # [B, P, head_dim]

        # 3) Indexer top-K.
        topk_idxs, _ = self.indexer(hidden)  # [B, S, K]

        # 4) Gather selected pool positions per query.
        gathered, valid = self._gather_topk_kv(pool_kv, topk_idxs)  # [B, S, K, head_dim]

        # 5) Stash for ``_compute_attention_output`` to consume.
        gathered.shape[2]
        # Build mask for the compressed branch: ``-inf`` where invalid.
        # This is per-query, shape [S, K]; we keep it on the module as a
        # full [B, S, K] additive mask.
        sparse_mask = torch.where(valid, 0.0, float("-inf")).to(dtype)  # [B, S, K]
        self._csa_state = {
            "gathered": gathered,  # [B, S, K, head_dim]
            "sparse_mask": sparse_mask,  # [B, S, K]
        }

        # Tell the parent: no cat-extension; we handle CSA inside
        # ``_compute_attention_output``.
        return None, None, None

    # ------------------------------------------------------------------

    def _compute_attention_output(
        self,
        q: torch.Tensor,  # [B, H, Sq, head_dim]
        k: torch.Tensor,  # [B, H, Sk_local, head_dim]
        v: torch.Tensor,  # [B, H, Sk_local, head_dim]
        attn_mask: torch.Tensor,  # [Sq, Sk_local]
    ) -> torch.Tensor:
        """Combined SWA + sparse-compressed attention.

        We compute logits over **both** the local sliding-window KV (k, v)
        and the per-query gathered top-K compressed KV (from
        ``self._csa_state``), softmax them jointly, and produce a single
        weighted sum. Optional ``attn_sink`` inserts an extra sink column
        across the *combined* logits.
        """
        state = getattr(self, "_csa_state", None)
        if state is None:
            # Should not happen; if it does, fall back to dense path.
            return super()._compute_attention_output(q, k, v, attn_mask)

        gathered = state["gathered"]  # [B, S, K, head_dim]
        sparse_mask = state["sparse_mask"]  # [B, S, K]
        scale = self._attention_scale()

        # Local logits (same as base): [B, H, Sq, Sk_local]
        local_logits = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_mask

        # Sparse logits: per-query Q · K_sparse(t,k), shape [B, H, S, K].
        # gathered: [B, S, K, head_dim] -> broadcast over H by repeat.
        gathered_h = gathered.unsqueeze(1).expand(-1, self.num_heads, -1, -1, -1)  # [B, H, S, K, head_dim]
        # q: [B, H, S, head_dim] -> einsum with gathered_h
        sparse_logits = torch.einsum("bhsd,bhskd->bhsk", q, gathered_h) * scale
        # Add per-query (B, S, K) mask, broadcasting across H.
        sparse_logits = sparse_logits + sparse_mask.unsqueeze(1)

        # Concatenate along the key axis: [B, H, S, Sk_local + K].
        joint_logits = torch.cat([local_logits, sparse_logits], dim=-1)
        # And the joint values: [B, H, S, Sk_local + K, head_dim] — but for
        # local v we have [B, H, Sk_local, head_dim] (independent of S),
        # while sparse v depends on S. Build a "value tensor" with the
        # same shape on both paths by broadcasting local v:
        v.shape[2]
        v_local_per_q = v.unsqueeze(2).expand(-1, -1, q.shape[2], -1, -1)  # [B, H, S, Sk_local, head_dim]
        v_sparse = gathered_h  # [B, H, S, K, head_dim]
        v_joint = torch.cat([v_local_per_q, v_sparse], dim=-2)  # [B, H, S, Sk_local+K, head_dim]

        if self.attn_sink is not None:
            probs = self.attn_sink.softmax_with_sink(joint_logits, dim=-1)
            if self.attn_dropout > 0.0 and self.training:
                probs = torch.nn.functional.dropout(probs, p=self.attn_dropout)
        else:
            probs = joint_logits.softmax(dim=-1)
            if self.attn_dropout > 0.0 and self.training:
                probs = torch.nn.functional.dropout(probs, p=self.attn_dropout)

        # Per-query weighted sum: [B, H, S, Sk_local+K] * [B, H, S, Sk_local+K, head_dim]
        out = torch.einsum("bhsk,bhskd->bhsd", probs, v_joint)
        return out


__all__ = ["CSAAttention"]
