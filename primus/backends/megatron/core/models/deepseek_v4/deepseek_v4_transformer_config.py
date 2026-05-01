###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 specific transformer config.

This config extends Megatron's ``MLATransformerConfig`` with DeepSeek-V4
runtime fields that are referenced by V4 modules but are not part of the
upstream ``TransformerConfig``/``MLATransformerConfig`` schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from megatron.core.transformer.transformer_config import MLATransformerConfig


@dataclass
class DeepSeekV4TransformerConfig(MLATransformerConfig):
    # ---- DeepSeek-V4 hybrid attention / HC ----
    hc_mult: int = 1
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6

    compress_ratios: Optional[Union[str, List[int], Tuple[int, ...]]] = None
    compress_rope_theta: float = 160000.0

    # ---- DeepSeek-V4 attention extras ----
    attn_sliding_window: int = 0
    attn_sink: bool = False
    index_topk: int = 0
    index_head_dim: int = 128
    index_n_heads: int = 64

    # ---- DeepSeek-V4 grouped low-rank output projection ----
    # Mirrors the released checkpoint's `wo_a` / `wo_b` layout.
    # When ``o_lora_rank == 0`` the attention falls back to a flat O proj
    # (Megatron's ``linear_proj``); set it >0 to use the grouped low-rank
    # form (``linear_o_a`` + ``linear_o_b``) with ``o_groups`` groups.
    o_groups: int = 8
    o_lora_rank: int = 0

    # ---- DeepSeek-V4 MoE routing / expert extras ----
    num_hash_layers: int = 0
    hash_routing_seed: int = 0

    moe_intermediate_size: Optional[int] = None
    moe_use_legacy_grouped_gemm: bool = False

    swiglu_limit: float = 0.0
    v4_grouped_experts_support_clamped_swiglu: bool = False

    # ---- Vocab helpers used by hash router ----
    vocab_size: Optional[int] = None
    padded_vocab_size: Optional[int] = None

    # ---- Compat aliases for V4 code paths ----
    norm_epsilon: Optional[float] = None
    position_embedding_type: str = "none"

    # ---- Optional V4 MTP extras ----
    mtp_compress_ratios: Optional[Union[str, List[int], Tuple[int, ...]]] = None
    v4_use_custom_mtp_block: bool = False

    def __post_init__(self) -> None:
        # Keep V4's ``norm_epsilon`` alias consistent with MCore's
        # ``layernorm_epsilon`` before parent validation runs.
        if self.norm_epsilon is None:
            self.norm_epsilon = float(self.layernorm_epsilon)
        self.layernorm_epsilon = float(self.norm_epsilon)

        # DeepSeek naming compatibility for MoE hidden size.
        if self.moe_ffn_hidden_size is None and self.moe_intermediate_size is not None:
            self.moe_ffn_hidden_size = int(self.moe_intermediate_size)

        # Keep DeepSeek clamp name aligned with MCore clamp field.
        clamp_from_activation = self.activation_func_clamp_value
        clamp_from_swiglu = float(self.swiglu_limit)
        if clamp_from_activation is None:
            if clamp_from_swiglu > 0.0:
                self.activation_func_clamp_value = clamp_from_swiglu
        elif clamp_from_swiglu <= 0.0:
            self.swiglu_limit = float(clamp_from_activation)

        # Ensure hash-router vocab lookups always have a concrete size.
        if self.padded_vocab_size is None and self.vocab_size is not None:
            self.padded_vocab_size = int(self.vocab_size)
        if self.vocab_size is None and self.padded_vocab_size is not None:
            self.vocab_size = int(self.padded_vocab_size)

        super().__post_init__()

        if self.moe_intermediate_size is None and self.moe_ffn_hidden_size is not None:
            self.moe_intermediate_size = int(self.moe_ffn_hidden_size)


__all__ = ["DeepSeekV4TransformerConfig"]
