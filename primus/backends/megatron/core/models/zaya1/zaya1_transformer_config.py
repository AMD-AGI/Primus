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

from primus.backends.megatron.core.fusions.chunked_linear_cross_entropy import (
    CHUNKED_CE_MODE,
    DEFAULT_CHUNK_TOKENS,
    check_chunk_tokens,
)

# A CLI override such as ``--zaya_fuse_norm off``, or a quoted YAML value,
# arrives as a string, which the modules' bool() would read as on.
_ON_OFF_FIELDS = (
    "zaya_attn_flash",
    "zaya_fuse_norm",
    "zaya_fuse_cca",
    "zaya_conv_gemm",
    "zaya_fuse_moe",
    "zaya_sort_dispatch",
    "zaya_router_defer_ar",
    "fused_ce_compile",
)


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
    # Kernel paths, all off by default; zaya1_modules.py describes each one.
    zaya_attn_flash: bool = False
    zaya_fuse_norm: bool = False
    zaya_fuse_cca: bool = False
    zaya_conv_gemm: bool = False
    zaya_fuse_moe: bool = False
    zaya_sort_dispatch: bool = False
    zaya_router_defer_ar: bool = False
    # The loss, selected as in the Mamba models: fused_ce_mode 3 is the chunked
    # linear cross-entropy (chunked_linear_cross_entropy.py), and unset or 0 is
    # Megatron's cross-entropy.
    fused_ce_mode: Optional[int] = None
    fused_ce_chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    fused_ce_compile: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.partial_rotary_factor <= 0 or self.partial_rotary_factor > 1:
            raise ValueError(f"partial_rotary_factor must be in (0, 1], got {self.partial_rotary_factor}.")
        for name in _ON_OFF_FIELDS:
            value = getattr(self, name)
            if type(value) is int and value in (0, 1):
                setattr(self, name, bool(value))
            elif not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}.")
        mode = self.fused_ce_mode
        if mode is not None and (type(mode) is not int or mode not in (0, 1, 2, CHUNKED_CE_MODE)):
            raise ValueError(f"fused_ce_mode must be 0, 1, 2 or {CHUNKED_CE_MODE}, got {mode!r}.")
        check_chunk_tokens(self.fused_ce_chunk_tokens)
        if self.num_moe_experts is None:
            raise ValueError("ZAYA1 requires num_experts (num_moe_experts) to be set.")
        if self.moe_router_topk != 1:
            raise ValueError(
                f"This ZAYA1 port implements top-1 routing, got moe_router_topk={self.moe_router_topk}."
            )
