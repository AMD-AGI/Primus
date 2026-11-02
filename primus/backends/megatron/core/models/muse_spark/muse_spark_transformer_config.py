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

from primus.backends.megatron.core.models.muse_spark.layer_patterns import (
    check_all_moe_layers,
    hf_sliding_window,
    normalize_layer_pattern,
    parse_window_size,
)

__all__ = ["MuseSparkTransformerConfig"]


@dataclass
class MuseSparkTransformerConfig(TransformerConfig):
    """Config for the Muse Spark text backbone."""

    # eps of the weightless post-attention norm, the input-scaled post-MoE norm
    # and the per-expert post norm.
    post_norm_eps: float = 1e-8
    # residual gate: beta = sigmoid(g / T), alpha = sqrt(clamp(sigmoid(-g / T) * (1 + beta), 1e-3)).
    residual_gate_temperature: float = 0.3
    residual_gate_init_value: float = -0.9
    """Initial value of every residual gate ``g`` for a run from random weights (a checkpoint
    load overwrites it). HF initialises ``g ~ N(0, 0.02)`` (beta = 0.5, alpha = 0.87): the
    backward pass then grows about 2x per layer through the post-norms, and at the released
    62 layers the local grad norm overflows to Inf at iteration 1. ``g = -0.9`` (beta =
    sigmoid(-3) = 0.05, alpha = 1.0) starts each layer close to an identity residual and
    keeps the 62-layer grad norm O(1). Set 0.0 for the HF-like (zero-mean) start."""
    # Defaults below are the released HF config values; they must match the checkpoint
    # (the native converter copies them from config.json when present).
    # queries are multiplied by this after the weightless QK-norm (HF ``qk_scale_factor``).
    qk_scale_factor: float = 5.4730064863875
    # router logits are computed in fp32 and multiplied by this before the sigmoid.
    # HF ``output_multiplier`` = 0.1767... = 1/sqrt(32); HF applies the one value to the
    # router logits and to the lm-head logits.
    moe_router_output_multiplier: float = 0.17677669529663687
    # the lm-head logits: T * tanh(logits * mult / T) (HF ``output_multiplier``, 1/sqrt(32)).
    output_logit_multiplier: float = 0.17677669529663687
    final_logit_softcapping: Optional[float] = 20.0
    # weightless RMSNorm after the embedding lookup (eps = layernorm_epsilon).
    embedding_norm: bool = True

    window_size_is_hf_sliding_window: bool = True
    """``window_size[0]`` is HF ``sliding_window`` (keys incl. the query), as in the presets.

    ``__post_init__`` converts it once to TE's left size (``W - 1``) and then sets this
    flag to False. The flag is part of the dataclass state on purpose: ``dataclasses.replace``
    re-runs ``__post_init__`` with the flag False, so the window is not shifted twice
    (``deepcopy`` does not re-run ``__post_init__``; it copies the converted state). Set it
    to False to pass a TE window directly.
    """

    muse_audio_config: Optional[dict] = None
    """The waveform audio encoder settings (``audio_encoder.audio_config_from_hf``); None
    builds no audio tower (text-only training). The native converter sets it from the HF
    ``audio_config``, so a converted checkpoint carries the tower."""
    audio_token_id: int = 200038
    """HF ``audio_token_id``: the placeholder positions that the audio features replace."""

    derived_softmax_scale: Optional[float] = None
    """The ``softmax_scale`` that ``__post_init__`` derived (``qk_scale_factor /
    sqrt(kv_channels)``); internal. When ``softmax_scale`` still equals it, the scale is
    derived again, so ``dataclasses.replace(cfg, qk_scale_factor=..., kv_channels=...)`` gets
    the new scale and never a stale one."""

    def __post_init__(self):
        # Yaml / CLI may give the per-layer patterns as Python-list strings, or the
        # full pattern for a depth-reduced proxy: parse (no eval) and cut to num_layers.
        for name in ("window_attn_skip_freq", "no_rope_freq"):
            setattr(self, name, normalize_layer_pattern(getattr(self, name), self.num_layers, name))
        # parse_window_size checks the syntax, right == 0 and left >= -1 in either convention.
        self.window_size = parse_window_size(self.window_size)
        if self.window_size is not None and self.window_size_is_hf_sliding_window:
            hf = hf_sliding_window(self.window_size)  # also rejects an HF window of 0
            # HF ``sliding_window`` counts the query token (kv > q - W); TE's left window does
            # not. -1 is TE's "infinite" left window: nothing to convert.
            self.window_size = (-1 if hf is None else hf - 1, 0)
            self.window_size_is_hf_sliding_window = False
        super().__post_init__()
        # After the base __post_init__, which fills kv_channels (hidden / heads) when unset.
        # HF: q = qk_norm(q) * qk_scale_factor, scores * head_dim**-0.5.
        derived = self.qk_scale_factor / math.sqrt(self.kv_channels)
        if self.softmax_scale is None or self.softmax_scale == self.derived_softmax_scale:
            self.softmax_scale = derived  # unset, or a value derived before (maybe stale)
        elif not math.isclose(self.softmax_scale, derived, rel_tol=1e-6):
            raise ValueError(
                f"Muse Spark: softmax_scale={self.softmax_scale} does not match qk_scale_factor / "
                f"sqrt(kv_channels) = {self.qk_scale_factor} / sqrt({self.kv_channels}) = {derived}; "
                "set qk_scale_factor instead (leave softmax_scale unset)"
            )
        self.derived_softmax_scale = self.softmax_scale
        if self.residual_gate_temperature <= 0:
            raise ValueError(f"residual_gate_temperature must be > 0, got {self.residual_gate_temperature}")
        if self.num_moe_experts and self.moe_latent_size is None:
            raise ValueError("Muse Spark MoE needs moe_latent_size (HF moe_hidden_size)")
        if (self.expert_tensor_parallel_size or 1) > 1:
            # post_expert_norm is an RMSNorm over the full expert output row; expert TP
            # would give each rank a partial row (and 1/tp of the norm-weight gradient).
            raise ValueError(
                "Muse Spark needs expert_tensor_parallel_size=1 (post_expert_norm needs full rows), "
                f"got {self.expert_tensor_parallel_size}"
            )
        # The soft-cap / logit multiplier live in MuseSparkModel._scale_logits, which the
        # output-layer and fused linear-CE loss paths call. The MTP loss path does not, so
        # MTP is rejected here.
        if self.mtp_num_layers:
            raise ValueError("Muse Spark has no MTP layers; the MTP loss would skip the logit soft-cap")
        check_all_moe_layers(self.moe_layer_freq, self.num_layers)
