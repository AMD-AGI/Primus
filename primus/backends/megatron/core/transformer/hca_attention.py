###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Heavily-Compressed Attention (HCA) for DeepSeek-V4.

Reference: techblog §1 ("Hybrid Attention"), HCA branch (``compress_ratio == 128``).

HCA augments the dense / SWA attention with a **fully-visible** compressed
KV pool: each of the ``S/ratio`` compressed positions is concatenated to
the sliding-window KV (no Indexer / no sparse selection). The compressed
positions are **causally** masked so that compressed pool position ``s``
covers raw tokens ``[s*ratio, (s+1)*ratio - 1]`` and may only be attended
to by queries at raw token ``t >= (s+1)*ratio - 1``.

This module reuses :class:`Compressor` (non-overlap mode for ratio=128)
and the shared :class:`DeepseekV4Attention` backbone.
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


class HCAAttention(_LegacyDeepseekV4Attention):
    """HCA attention layer.

    Args:
        compress_ratio: must be a positive integer; ``128`` per V4 release.
        See :class:`DeepseekV4Attention` for the rest.
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        rope,
        compress_ratio: int,
        submodules=None,
    ) -> None:
        if compress_ratio <= 0:
            raise ValueError(f"HCAAttention requires compress_ratio > 0, got {compress_ratio}")
        super().__init__(
            config=config,
            rope=rope,
            compress_ratio=compress_ratio,
            submodules=submodules,
        )
        resolved_compressor_overlap = False

        # HCA's K-projection from compressed pool. We project ``head_dim`` →
        # ``num_kv_heads * head_dim`` so the compressed positions can be
        # broadcast across H-heads same as the local KV.
        # The compressed pool is shared between K and V (V4 reuses the
        # compressed latent for both); use two small Linears to give the
        # model a degree of freedom.
        self.compressor = Compressor(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            ratio=compress_ratio,
            overlap=resolved_compressor_overlap,
        )

    # ------------------------------------------------------------------

    def _compressed_causal_mask(
        self,
        n_queries: int,
        n_pool: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compressed-pool causal mask: pool ``s`` is allowed for query ``t``
        iff ``(s+1)*ratio - 1 <= t``.
        """
        t = torch.arange(n_queries, device=device).unsqueeze(1)
        s_end = (torch.arange(n_pool, device=device).unsqueeze(0) + 1) * self.compress_ratio - 1
        allowed = s_end <= t
        return torch.where(allowed, 0.0, float("-inf")).to(dtype)

    # ------------------------------------------------------------------

    def _extra_kv(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the extra (compressed-pool) KV branch.

        * Compress ``hidden`` to ``[B, P, head_dim]`` via the compressor.
        * Apply compress-base partial RoPE using *compressed* position ids
          ``arange(P)`` (each pool position represents the END of its
          covered raw window — using its index ``s`` as the rotary phase is
          the V4 convention).
        * Broadcast over H heads so the cat with local KV is shape-compatible.
        """
        B, S, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        pooled = self.compressor(hidden)  # [B, P, head_dim]
        P = pooled.shape[1]

        # Treat the compressed pool as if it has a single KV-head shape
        # ``[B, P, 1, head_dim]`` so RoPE / repeat-interleave use the same
        # code path as the local KV.
        pool_kv = pooled.unsqueeze(2)  # [B, P, 1, head_dim]

        # Compress-base RoPE; positions are the compressed indices [0..P).
        comp_pos = torch.arange(P, device=device)
        cos, sin = self.rope.compress_rope(comp_pos)
        cos = cos[..., : self.rotary_dim // 2]
        sin = sin[..., : self.rotary_dim // 2]
        pool_kv = apply_interleaved_partial_rope(pool_kv, cos, sin, rotary_dim=self.rotary_dim)

        # Broadcast over Q-heads.
        extra_kv_h = self._broadcast_kv_heads(pool_kv)  # [B, P, H, head_dim]

        # Causal mask for compressed positions.
        extra_mask = self._compressed_causal_mask(S, P, device=device, dtype=dtype)

        # Same K & V for the compressed branch (no separate v projection;
        # the Compressor already encodes both the value and gating info).
        return extra_kv_h, extra_kv_h, extra_mask


__all__ = ["HCAAttention"]
