###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus-Turbo sink attention mirror for TorchTitan GPT-OSS.

Upstream GPT-OSS attention (``torchtitan.models.gpt_oss.model.model.Attention``)
uses ``FlexAttentionWrapper`` with:
    * learnable per-head attention sinks (``self.sinks``), applied as a post-hoc
      ``sigmoid(lse - sinks)`` rescaling of the FlexAttention output, and
    * a sliding-window mask applied on even layers (``layer_id % 2 == 0``).

This mirror keeps the exact same parameters (``wq/wk/wv/wo`` incl. biases and the
learnable ``self.sinks``) but routes attention through the Primus-Turbo
*functional* kernel ``primus_turbo.pytorch.ops.flash_attn_func``, which natively
supports both:
    * ``sink=`` (per-head learnable sinks; automatically dispatches to the Triton
      backend), matching the Megatron ``PrimusTurboAttention`` semantics, and
    * ``window_size=(left, 0)`` for the sliding window.

Because ``self.sinks`` is inherited from the upstream module (not re-created), the
learned sink weights are preserved on checkpoint load/save. The mirror is only
installed when ``primus_turbo.enable_primus_turbo`` and
``primus_turbo.use_turbo_attention`` are both set for a ``gpt_oss`` run (see
``patches/turbo/gptoss_sink_attention_patches.py``); the default GPT-OSS path
keeps upstream FlexAttention untouched.
"""

import torch
from torchtitan.models.gpt_oss.model.model import Attention as TTGptOssAttention
from torchtitan.models.gpt_oss.model.model import TransformerBlock as TTGptOssBlock
from torchtitan.models.gpt_oss.model.model import apply_rotary_emb

# Sentinel meaning "no sliding window" for primus_turbo's flash_attn_func.
_FULL_WINDOW = (-1, -1)


class Attention(TTGptOssAttention):
    """GPT-OSS attention backed by Primus-Turbo ``flash_attn_func`` (sink-aware)."""

    def __init__(self, model_args):
        super().__init__(model_args)
        # Per-layer sliding window is injected by the mirror TransformerBlock
        # before each forward; default to full (causal, no window).
        self.sliding_window_size = model_args.sliding_window_size
        self._turbo_window = _FULL_WINDOW

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks=None,  # noqa: ARG002 - flex masks unused on the turbo path
    ):
        bsz, seqlen, _ = x.size()
        hidden_shape = (bsz, seqlen, -1, self.head_dim)

        q = self.wq(x).view(hidden_shape)
        k = self.wk(x).view(hidden_shape)
        v = self.wv(x).view(hidden_shape)

        # RoPE is applied on the (b, s, h, d) layout, exactly like upstream.
        q, k = apply_rotary_emb(q, k, rope_cache)

        # primus_turbo.flash_attn_func consumes the (b, s, h, d) layout directly
        # and handles GQA (n_heads > n_kv_heads) internally, so we do NOT
        # transpose to (b, h, s, d) and do NOT repeat_kv. The learnable sinks are
        # applied inside the kernel (Triton backend), so the upstream post-hoc
        # sigmoid(lse - sinks) rescaling is not needed here.
        import primus_turbo.pytorch as turbo

        output = turbo.ops.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=self._turbo_window,
            sink=self.sinks.to(q.dtype),
        )

        output = output.reshape(bsz, seqlen, -1).contiguous()
        return self.wo(output)


class TransformerBlock(TTGptOssBlock):
    """GPT-OSS block that selects the per-layer sliding window for turbo sink attn."""

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks=None,  # noqa: ARG002 - flex masks unused on the turbo path
    ):
        # gpt-oss pattern: even layers use sliding-window attention.
        if self.use_sliding_attention:
            self.attention._turbo_window = (self.attention.sliding_window_size, 0)
        else:
            self.attention._turbo_window = _FULL_WINDOW

        x = x + self.attention(self.attention_norm(x), rope_cache, None)
        x = x + self.moe(self.ffn_norm(x))
        return x
