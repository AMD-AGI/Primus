###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 attention base class.

Reference: techblog §1 ("Hybrid Attention").

The base class owns:

* QKV projection (with optional ``q_lora_rank`` low-rank decomposition).
  V4 uses ``num_key_value_heads=1, head_dim=512`` (MQA-flavor MLA).
* Partial dual-RoPE on Q and K.
* Sliding-window causal KV (window = ``attn_sliding_window``).
* Attention sink (per-head learnable scalar).
* Output projection (plain ``nn.Linear``; the grouped low-rank
  ``o_groups`` / ``o_lora_rank`` form is deferred to P6 / P8 once perf
  becomes the priority).

Concrete subclasses (``HCAAttention``, ``CSAAttention``) override only
``_extra_kv(hidden, q)`` to inject the compressed / sparse KV.

This module deliberately uses **plain torch ops** (no TransformerEngine /
flash-attn) so we can validate correctness on CPU. The eventual production
attention kernel selection is handled by the surrounding ``ModuleSpec``
during P4.6 and the perf phase (P8).

Forward signature (all subclasses):

.. code-block:: python

    out = attn(
        hidden,                     # [B, S, D]
        position_ids,               # [B, S] or [S]
        attention_mask=None,        # ignored when sliding-window is on
    )
    # out: [B, S, D]

Phase 4 contract:
* hc_mult collapse / expand happens in the *block*, not here. This module
  consumes a single-stream ``hidden`` and returns a single-stream ``out``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from primus.backends.megatron.core.transformer.attn_sink import AttentionSink
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE
from primus.backends.megatron.core.transformer.sliding_window_kv import (
    sliding_window_causal_mask,
)


class DeepseekV4Attention(nn.Module):
    """Dense V4 attention (``compress_ratio == 0`` layers).

    Args:
        hidden_size: ``D``.
        num_heads: number of Q heads ``H``.
        num_kv_heads: number of K/V heads (V4 uses 1; we keep it general
            for unit-test flexibility).
        head_dim: per-head channel dim.
        rotary_dim: partial-RoPE dim (``qk_pos_emb_head_dim``).
        rope: a :class:`DualRoPE` instance shared across the model.
        attn_sliding_window: SWA window length.
        attn_sink_enabled: whether to add the per-head sink scalar.
        q_lora_rank: optional low-rank rank for Q. ``None`` → plain Linear.
        attn_dropout: dropout probability for the softmax probs (training only).
        compress_ratio: ``0`` for the base / dense case. Subclasses override.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        rope: DualRoPE,
        attn_sliding_window: int = 0,
        attn_sink_enabled: bool = False,
        q_lora_rank: Optional[int] = None,
        attn_dropout: float = 0.0,
        compress_ratio: int = 0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.attn_sliding_window = int(attn_sliding_window)
        self.attn_dropout = float(attn_dropout)
        self.compress_ratio = int(compress_ratio)

        # Q projection (optional LoRA).
        q_out = num_heads * head_dim
        if q_lora_rank is None or q_lora_rank <= 0:
            self.q_a = None
            self.q_b = nn.Linear(hidden_size, q_out, bias=False)
        else:
            self.q_a = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_b = nn.Linear(q_lora_rank, q_out, bias=False)

        # K, V projections (MQA-style: ``num_kv_heads * head_dim``).
        kv_out = num_kv_heads * head_dim
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=False)

        # Output projection.
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Shared dual-RoPE (held by reference; not registered as submodule
        # to avoid double-counting parameters when several attention layers
        # share the same instance).
        self._rope = [rope]

        if attn_sink_enabled:
            self.attn_sink = AttentionSink(num_heads=num_heads)
        else:
            self.attn_sink = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @property
    def rope(self) -> DualRoPE:
        return self._rope[0]

    def _project_q(self, hidden: torch.Tensor) -> torch.Tensor:
        """``[B, S, D]`` → ``[B, S, H, head_dim]``."""
        x = self.q_a(hidden) if self.q_a is not None else hidden
        q = self.q_b(x) if self.q_a is not None else self.q_b(hidden)
        B, S, _ = q.shape
        return q.view(B, S, self.num_heads, self.head_dim)

    def _project_kv(self, hidden: torch.Tensor):
        """``[B, S, D]`` → ``(k, v)`` each ``[B, S, num_kv_heads, head_dim]``."""
        B, S, _ = hidden.shape
        k = self.k_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)
        return k, v

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """Apply partial RoPE on Q and K at the layer's ``compress_ratio``."""
        q = self.rope.apply(q, position_ids=position_ids, compress_ratio=self.compress_ratio)
        k = self.rope.apply(k, position_ids=position_ids, compress_ratio=self.compress_ratio)
        return q, k

    def _attention_scale(self) -> float:
        """Softmax scale ``1/sqrt(head_dim)``, optionally multiplied by YaRN
        ``m_scale`` for compressed layers (techblog §6 implementation note 4)."""
        base = self.head_dim**-0.5
        rope_scale = self.rope.attn_scale(compress_ratio=self.compress_ratio)
        return base * rope_scale

    def _broadcast_kv_heads(self, kv: torch.Tensor) -> torch.Tensor:
        """Replicate ``[B, S, num_kv_heads, head_dim]`` →
        ``[B, S, num_heads, head_dim]`` so MQA / GQA can dot-product per Q-head.
        """
        if self.num_kv_heads == self.num_heads:
            return kv
        repeats = self.num_heads // self.num_kv_heads
        return kv.repeat_interleave(repeats, dim=2)

    def _compute_attention_output(
        self,
        q: torch.Tensor,  # [B, H, Sq, head_dim]
        k: torch.Tensor,  # [B, H, Sk, head_dim]
        v: torch.Tensor,  # [B, H, Sk, head_dim]
        attn_mask: torch.Tensor,  # [Sq, Sk] additive mask
    ) -> torch.Tensor:
        """Eager scaled-dot-product attention, optionally with sink."""
        scale = self._attention_scale()
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Sq, Sk]
        logits = logits + attn_mask

        if self.attn_sink is not None:
            return self.attn_sink(logits, v, dropout=self.attn_dropout)

        probs = logits.softmax(dim=-1)
        if self.attn_dropout > 0.0 and self.training:
            probs = torch.nn.functional.dropout(probs, p=self.attn_dropout)
        return torch.matmul(probs, v)

    # ------------------------------------------------------------------
    # extra-KV hook (subclasses override)
    # ------------------------------------------------------------------

    def _extra_kv(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        q: torch.Tensor,  # [B, S, H, head_dim] post-RoPE
    ):
        """Return ``(extra_k, extra_v, extra_mask)`` to be concatenated to
        the sliding-window KV along the key dim. Default: no extra (dense).
        ``extra_mask`` shape ``[Sq, Sk_extra]`` (additive).
        """
        return None, None, None

    # ------------------------------------------------------------------
    # public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, S, D] → [B, S, D]``."""
        B, S, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        # Project Q / K / V.
        q = self._project_q(hidden)  # [B, S, H, head_dim]
        k_local, v_local = self._project_kv(hidden)  # [B, S, num_kv_heads, head_dim]

        # Apply dual-RoPE on Q / K (partial RoPE on last rotary_dim channels).
        q, k_local = self._apply_rope(q, k_local, position_ids)

        # Broadcast K/V over H so MQA / GQA does per-head dot products.
        k_local_h = self._broadcast_kv_heads(k_local)  # [B, S, H, head_dim]
        v_local_h = self._broadcast_kv_heads(v_local)

        # Sliding-window mask.
        window = self.attn_sliding_window
        local_mask = sliding_window_causal_mask(S, window, device=device, dtype=dtype)  # [S, S]

        # Subclass hook: extra K/V (compressed pool, sparse top-K, etc.).
        # Subclass should return tensors already broadcast to [B, S_extra, H, head_dim]
        # so they can be cat'd along the Sk axis.
        extra_k, extra_v, extra_mask = self._extra_kv(hidden, position_ids, q)

        # Concatenate sliding-window KV with extra KV (if any).
        if extra_k is not None:
            k_full = torch.cat([k_local_h, extra_k], dim=1)  # [B, Sk_total, H, head_dim]
            v_full = torch.cat([v_local_h, extra_v], dim=1)
            full_mask = torch.cat([local_mask, extra_mask], dim=-1)  # [Sq, Sk_total]
        else:
            k_full = k_local_h
            v_full = v_local_h
            full_mask = local_mask

        # Move heads dim before sequence: [B, S, H, head_dim] -> [B, H, S, head_dim]
        q_bh = q.transpose(1, 2)
        k_bh = k_full.transpose(1, 2)
        v_bh = v_full.transpose(1, 2)

        out_bh = self._compute_attention_output(q_bh, k_bh, v_bh, full_mask)
        # back to [B, S, H, head_dim]
        out = out_bh.transpose(1, 2).contiguous()

        # Output projection.
        out = out.reshape(B, S, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        return out


__all__ = ["DeepseekV4Attention"]
