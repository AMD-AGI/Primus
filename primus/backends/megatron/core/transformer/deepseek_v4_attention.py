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

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.attn_sink import AttentionSink
from primus.backends.megatron.core.transformer.dual_rope import DualRoPE
from primus.backends.megatron.core.transformer.sliding_window_kv import (
    sliding_window_causal_mask,
)

logger = logging.getLogger(__name__)


@dataclass
class DeepseekV4AttentionSubmodules:
    """Projection submodules for DeepSeek-V4 attention."""

    linear_q_a: Optional[ModuleSpec] = None
    linear_q_b: Optional[ModuleSpec] = None
    linear_k_proj: Optional[ModuleSpec] = None
    linear_v_proj: Optional[ModuleSpec] = None
    linear_o_proj: Optional[ModuleSpec] = None


def _build_projection(
    projection_submodule: Optional[ModuleSpec],
    *,
    in_features: int,
    out_features: int,
) -> nn.Module:
    """Build a projection by submodule spec with local fallback."""
    if projection_submodule is None:
        return nn.Linear(in_features, out_features, bias=False)
    try:
        return build_module(projection_submodule)
    except Exception as exc:
        logger.warning(
            "DeepSeek-V4 attention projection submodule init failed (%s); fallback to nn.Linear.",
            exc,
        )
        return nn.Linear(in_features, out_features, bias=False)


def _projection_forward(proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = proj(x)
    if isinstance(out, tuple):
        return out[0]
    return out


class DeepseekV4Attention(nn.Module):
    """Dense V4 attention (``compress_ratio == 0`` layers).

    Args:
        config: runtime DeepSeek-V4 config. All attention dimensions and
            feature toggles are read directly from config.
        rope: a :class:`DualRoPE` instance shared across the model.
        compress_ratio: ``0`` for the base / dense case. Subclasses override.
    """

    def __init__(
        self,
        config: DeepSeekV4TransformerConfig,
        *,
        rope: DualRoPE,
        compress_ratio: int = 0,
        submodules: Optional[DeepseekV4AttentionSubmodules] = None,
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_query_groups)
        head_dim = int(config.kv_channels)
        rotary_dim = int(config.qk_pos_emb_head_dim)
        attn_sliding_window = int(config.attn_sliding_window)
        attn_sink_enabled = bool(config.attn_sink)
        q_lora_rank = config.q_lora_rank
        attn_dropout = float(config.attention_dropout)

        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.attn_sliding_window = attn_sliding_window
        self.attn_dropout = attn_dropout
        self.compress_ratio = int(compress_ratio)
        submodules = submodules or DeepseekV4AttentionSubmodules()

        # Q projection (optional LoRA).
        q_out = self.num_heads * self.head_dim
        if q_lora_rank is None or q_lora_rank <= 0:
            self.q_a = None
            self.q_b = _build_projection(
                submodules.linear_q_b,
                in_features=self.hidden_size,
                out_features=q_out,
            )
        else:
            self.q_a = _build_projection(
                submodules.linear_q_a,
                in_features=self.hidden_size,
                out_features=q_lora_rank,
            )
            self.q_b = _build_projection(
                submodules.linear_q_b,
                in_features=q_lora_rank,
                out_features=q_out,
            )

        # K, V projections (MQA-style: ``num_kv_heads * head_dim``).
        kv_out = self.num_kv_heads * self.head_dim
        self.k_proj = _build_projection(
            submodules.linear_k_proj,
            in_features=self.hidden_size,
            out_features=kv_out,
        )
        self.v_proj = _build_projection(
            submodules.linear_v_proj,
            in_features=self.hidden_size,
            out_features=kv_out,
        )

        # Output projection.
        self.o_proj = _build_projection(
            submodules.linear_o_proj,
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
        )

        # Shared dual-RoPE (held by reference; not registered as submodule
        # to avoid double-counting parameters when several attention layers
        # share the same instance).
        self._rope = [rope]

        if attn_sink_enabled:
            self.attn_sink = AttentionSink(num_heads=self.num_heads)
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
        x = _projection_forward(self.q_a, hidden) if self.q_a is not None else hidden
        q = (
            _projection_forward(self.q_b, x)
            if self.q_a is not None
            else _projection_forward(self.q_b, hidden)
        )
        B, S, _ = q.shape
        return q.view(B, S, self.num_heads, self.head_dim)

    def _project_kv(self, hidden: torch.Tensor):
        """``[B, S, D]`` → ``(k, v)`` each ``[B, S, num_kv_heads, head_dim]``."""
        B, S, _ = hidden.shape
        k = _projection_forward(self.k_proj, hidden).view(B, S, self.num_kv_heads, self.head_dim)
        v = _projection_forward(self.v_proj, hidden).view(B, S, self.num_kv_heads, self.head_dim)
        return k, v

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor):
        """Apply partial RoPE on Q and K at the layer's ``compress_ratio``."""
        q = self.rope.apply_rope(q, position_ids=position_ids, compress_ratio=self.compress_ratio)
        k = self.rope.apply_rope(k, position_ids=position_ids, compress_ratio=self.compress_ratio)
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
        probs = probs.to(v.dtype)
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
        out = out.to(dtype=dtype)

        # Output projection.
        out = out.reshape(B, S, self.num_heads * self.head_dim)
        out = _projection_forward(self.o_proj, out)
        return out


__all__ = [
    "DeepseekV4AttentionSubmodules",
    "DeepseekV4Attention",
]
