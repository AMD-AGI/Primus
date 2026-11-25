###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX attention processor that runs q/k RMSNorm, the stream concat and RoPE as Primus-Turbo's
``dit_qk_norm_rope``; ``qk_norm_rope.py`` says why and installs it.

A subclass of diffusers' ``FluxAttnProcessor``: diffusers' ``__call__`` serves every call the op does
not take, and the blocks' checks for diffusers' processor still hold. Kept apart from
``qk_norm_rope.py`` because subclassing imports diffusers, which the patch condition must not.
"""
from __future__ import annotations

import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

from primus.backends.automodel.models.flux.plain_linear import is_plain_linear
from primus.backends.automodel.models.flux.qk_norm_rope import _fusable


class PrimusFluxAttnProcessor(FluxAttnProcessor):
    """diffusers' FLUX attention with the q/k norms, the stream concat and RoPE in one op."""

    # Primus-Turbo's dit_qk_norm_rope, set by qk_norm_rope.install().
    dit_qk_norm_rope = None

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        eps = _fusable(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        if eps is None or self.dit_qk_norm_rope is None:
            return super().__call__(
                attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )
        cos, sin = image_rotary_emb
        dual = encoder_hidden_states is not None
        projections = [attn.to_qkv] + ([attn.to_added_qkv] if dual else [])
        if all(is_plain_linear(p) for p in projections):
            # The op adds the biases and returns their gradients from its backward, so the GEMMs
            # run without them: a bias in the GEMM as well would be added twice.
            qkv = F.linear(hidden_states, attn.to_qkv.weight)
            txt_qkv = None
            if dual:
                txt_qkv = F.linear(encoder_hidden_states, attn.to_added_qkv.weight)
            biases = (attn.to_qkv.bias, attn.to_added_qkv.bias if dual else None)
        else:
            qkv = attn.to_qkv(hidden_states)
            txt_qkv = attn.to_added_qkv(encoder_hidden_states) if dual else None
            biases = (None, None)
        query, key, value = self.dit_qk_norm_rope(
            qkv,
            attn.norm_q.weight,
            attn.norm_k.weight,
            cos,
            sin,
            eps,
            layout="bshd",
            txt_qkv=txt_qkv,
            w_txt_q=attn.norm_added_q.weight if dual else None,
            w_txt_k=attn.norm_added_k.weight if dual else None,
            bias=biases[0],
            txt_bias=biases[1],
        )

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
