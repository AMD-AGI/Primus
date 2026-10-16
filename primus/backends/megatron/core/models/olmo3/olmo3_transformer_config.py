###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 transformer config: the YaRN fields and the sliding-window convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

__all__ = ["Olmo3TransformerConfig"]


@dataclass
class Olmo3TransformerConfig(TransformerConfig):
    """Config for the OLMo-3 dense decoder.

    YaRN: when ``yarn_rotary_scaling_factor`` is set, the model builds Megatron's
    ``YarnRotaryEmbedding`` (``position_embedding_type='yarn'``) from these fields.
    ``yarn_mscale``/``yarn_mscale_all_dim`` left at None give the HF default
    attention factor ``0.1 * ln(factor) + 1`` (1.2079 for factor 8), which
    Megatron multiplies into cos/sin, as HF does.

    Sliding window: ``window_size[0]`` is given in Hugging Face ``sliding_window``
    units (the number of keys a query sees, itself included: HF masks
    ``kv_idx > q_idx - sliding_window``). Transformer Engine's left window does
    not count the query itself, so it is converted to ``(sliding_window - 1, 0)``
    here. ``window_attn_skip_freq: 4`` keeps every fourth layer (1-indexed
    4, 8, ...; HF ``full_attention``) global.
    """

    # 1.0 = no YaRN. Megatron's attention reads this field (if present) for the cos/sin mscale.
    yarn_rotary_scaling_factor: float = 1.0
    yarn_original_max_position_embeddings: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: Optional[float] = None
    yarn_mscale_all_dim: Optional[float] = None
    yarn_correction_range_round_to_int: bool = True

    # GPTModel prefers config.position_embedding_type; __post_init__ sets 'yarn' when YaRN is on
    # (args keep 'rope': Megatron's CLI has no 'yarn' choice for non-MLA models).
    position_embedding_type: str = "rope"

    # window_size[0] is in HF sliding_window units (see the docstring).
    olmo3_window_size_hf_units: bool = True

    def __post_init__(self):
        if self.window_size is not None:
            left, right = (int(x) for x in self.window_size)
            if self.olmo3_window_size_hf_units and left > 0:
                left -= 1
                self.olmo3_window_size_hf_units = False  # converted once (configs get re-built/copied)
            self.window_size = (left, right)
        super().__post_init__()
        if self.use_yarn:
            self.position_embedding_type = "yarn"
            # mscale != 1 is not supported by TE's fused RoPE (Megatron falls back anyway).
            self.apply_rope_fusion = False
        assert not self.multi_latent_attention, "OLMo-3 uses its own attention, not MLA."
        assert not self.qk_layernorm, (
            "OLMo-3 uses a full-width QK norm built by its own attention; keep qk_layernorm false "
            "(per-head q/k_layernorm would be a different model)."
        )

    @property
    def use_yarn(self) -> bool:
        return float(self.yarn_rotary_scaling_factor) > 1.0
