###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA1 config fields that Megatron's ``TransformerConfig`` does not declare.

``core_transformer_config_from_args`` copies every dataclass field it finds on
``args``. YAML keys therefore have to live here; they do not need argparse
registration. ``multi_latent_attention`` must stay false, or that helper
replaces this class with ``MLATransformerConfig`` and drops the fields below.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Zaya1TransformerConfig(TransformerConfig):
    """ZAYA1 architecture knobs, plus the PID step size the papers leave unspecified."""

    zaya_mlp_expansion: int = 256
    zaya_use_mod: bool = True
    zaya_use_eda: bool = True
    zaya_high_prec: bool = True
    scale_residual_merge: bool = True
    clamp_temp: bool = False
    cca_time0: int = 2
    cca_time1: int = 2
    # Half-head RoPE. ``rotary_percent`` is the Megatron name of the same fraction;
    # this field is what ``CCA`` reads.
    partial_rotary_factor: float = 0.5
    # Microbatch AdamW on ``p_e - 1/E``. ``0`` freezes ``balancing_biases``.
    zaya_balance_lr: float = 1.0e-3
    zaya_balance_beta1: float = 0.9
    zaya_balance_beta2: float = 0.999
    zaya_balance_wd: float = 0.0
    # Optional explicit stage list. ``None`` alternates CCA, MoE, CCA, MoE, ...
    zaya_layers: Optional[Sequence] = None
    # Kernel paths; zaya1_modules.py and zaya1_chunked_ce.py describe each one.
    zaya_attn_flash: bool = True
    zaya_fuse_norm: bool = True
    zaya_fuse_cca: bool = True
    zaya_conv_gemm: bool = True
    zaya_fuse_moe: bool = True
    zaya_router_defer_ar: bool = False
    zaya_chunked_ce: bool = True
    zaya_chunked_ce_tokens: int = 4096
    zaya_chunked_ce_compile: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.partial_rotary_factor <= 0 or self.partial_rotary_factor > 1:
            raise ValueError(f"partial_rotary_factor must be in (0, 1], got {self.partial_rotary_factor}.")
        if self.zaya_chunked_ce_tokens < 1:
            raise ValueError(f"zaya_chunked_ce_tokens must be at least 1, got {self.zaya_chunked_ce_tokens}.")
        if self.num_moe_experts is None:
            raise ValueError("ZAYA1 requires num_experts (num_moe_experts) to be set.")
        if self.moe_router_topk != 1:
            raise ValueError(
                f"This ZAYA1 port implements top-1 routing, got moe_router_topk={self.moe_router_topk}."
            )
