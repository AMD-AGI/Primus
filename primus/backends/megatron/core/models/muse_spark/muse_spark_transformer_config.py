###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark (``muse_spark``) text-decoder transformer config.

The text decoder of ``meta-models/Muse-Spark-Open-816B-A42B`` is a GQA
transformer with:

* a weightless RMSNorm on the token embeddings;
* sandwich norms: centered pre-norms (``1 + w``), a weightless post-attention
  norm and an input-scaled post-MoE norm (both ``post_norm_eps``);
* learned residual gates (``residual_gate_temperature``) in place of the adds;
* gated attention: weightless QK-norm, queries scaled by ``qk_scale_factor``,
  a per-head weightless ``o_norm`` and a sigmoid output gate;
  sliding window (3 of 4 layers) and NoPE on the full-attention layers;
* a latent MoE: sigmoid top-k router on fp32 logits scaled by
  ``moe_router_output_multiplier``, ``pre_expert_proj`` + centered norm,
  SwiGLU experts with a per-expert input-scaled norm before the prob weighting,
  ``post_expert_proj``;
* an untied head scaled by ``output_logit_multiplier`` and soft-capped at
  ``final_logit_softcapping``.

Upstream fields carry the rest: ``softmax_scale`` (= ``qk_scale_factor`` /
sqrt(head_dim)), ``attention_output_gate``, ``window_size`` /
``window_attn_skip_freq``, ``no_rope_freq``, ``moe_latent_size``,
``layernorm_zero_centered_gamma``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = ["MuseSparkTransformerConfig"]


def _layer_pattern(value, num_layers):
    """A per-layer 0/1 pattern (int / str / list) as an int or a list cut to ``num_layers``."""
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if "[" not in s:
            return int(s)
        value = eval(s, {"__builtins__": {}}, {})  # trusted config, same form as moe_layer_freq
    value = [int(v) for v in value]
    if num_layers and len(value) > num_layers:
        value = value[:num_layers]
    return value


@dataclass
class MuseSparkTransformerConfig(TransformerConfig):
    """Config for the Muse Spark text backbone."""

    # eps of the weightless post-attention norm, the input-scaled post-MoE norm
    # and the per-expert post norm.
    post_norm_eps: float = 1e-8
    # residual gate: beta = sigmoid(g / T), alpha = sqrt(clamp(sigmoid(-g / T) * (1 + beta), 1e-3)).
    residual_gate_temperature: float = 0.3
    # queries are multiplied by this after the weightless QK-norm.
    qk_scale_factor: float = 5.4730064863875
    # router logits are computed in fp32 and multiplied by this before the sigmoid.
    moe_router_output_multiplier: float = 0.17677669529663687
    # the lm-head logits: T * tanh(logits * mult / T).
    output_logit_multiplier: float = 0.17677669529663687
    final_logit_softcapping: Optional[float] = 20.0
    # weightless RMSNorm after the embedding lookup (eps = norm_epsilon).
    embedding_norm: bool = True

    window_size_is_hf_sliding_window: bool = True
    """``window_size[0]`` is HF ``sliding_window`` (keys incl. the query); converted once to TE's left size."""

    def __post_init__(self):
        # Yaml / CLI may give the per-layer patterns as Python-list strings, or the
        # full 62-layer pattern for a depth-reduced proxy: parse and cut to num_layers.
        for name in ("window_attn_skip_freq", "no_rope_freq"):
            setattr(self, name, _layer_pattern(getattr(self, name), self.num_layers))
        if isinstance(self.window_size, str):
            self.window_size = tuple(int(x) for x in self.window_size.strip("()[] ").split(","))
        elif isinstance(self.window_size, list):
            self.window_size = tuple(self.window_size)
        if self.window_size and self.window_size_is_hf_sliding_window:
            # HF ``sliding_window`` counts the query token (kv > q - W); TE's left window does not.
            self.window_size = (int(self.window_size[0]) - 1, int(self.window_size[1]))
            self.window_size_is_hf_sliding_window = False
        if self.softmax_scale is None and self.kv_channels:
            # HF: q = qk_norm(q) * qk_scale_factor, scores * head_dim**-0.5.
            self.softmax_scale = self.qk_scale_factor / math.sqrt(self.kv_channels)
        super().__post_init__()
        if self.residual_gate_temperature <= 0:
            raise ValueError("residual_gate_temperature must be > 0")
        if self.num_moe_experts and self.moe_latent_size is None:
            raise ValueError("Muse Spark MoE needs moe_latent_size (HF moe_hidden_size)")
