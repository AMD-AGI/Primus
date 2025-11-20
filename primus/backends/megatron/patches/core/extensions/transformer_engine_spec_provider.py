###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import warnings
from typing import Optional, Tuple

from megatron.core.extensions.transformer_engine import (
    TEActivationOp,
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import (
    GroupedMLP,
    SequentialMLP,
    TEGroupedMLP,
)
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.training.global_vars import get_args

try:
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboAttention,
        PrimusTurboColumnParallelLinear,
        PrimusTurboGroupedMLP,
        PrimusTurboLayerNormColumnParallelLinear,
        PrimusTurboRowParallelLinear,
    )

    HAVE_PRIMUS_TURBO = True
except ImportError:

    HAVE_PRIMUS_TURBO = False


class PrimusTurboSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def __init__(self):
        if not HAVE_PRIMUS_TURBO:
            raise ImportError(
                "PrimusTurbo extension requires the primus_Turbo package. " "Please install it."
            )

        self.cfg = get_args()

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return (
            PrimusTurboColumnParallelLinear if self.cfg.use_turbo_parallel_linear else TEColumnParallelLinear
        )

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module TE backend uses"""
        return PrimusTurboRowParallelLinear if self.cfg.use_turbo_parallel_linear else TERowParallelLinear

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return (
            PrimusTurboLayerNormColumnParallelLinear
            if self.cfg.use_turbo_parallel_linear
            else TELayerNormColumnParallelLinear
        )

    def layer_norm(self, rms_norm: bool = False, for_qk: bool = False) -> type:
        """Which module to use for layer norm"""
        if for_qk and not is_te_min_version("1.9.0"):
            # TENorm significantly harms convergence when used
            # for QKLayerNorm if TE Version < 1.9;
            # we instead use the Apex implementation.
            return FusedLayerNorm
        return TENorm

    def core_attention(self) -> type:
        """Which module to use for attention"""
        return PrimusTurboAttention if self.cfg.use_turbo_attention else TEDotProductAttention

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: bool
    ) -> Tuple[type, Optional[MLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        if (
            moe_use_grouped_gemm
            and TEColumnParallelGroupedLinear is not None
            and not moe_use_legacy_grouped_gemm
        ):
            assert not self.cfg.use_turbo_grouped_mlp, "PrimusTurbo not support RowParallelGroupedLinear"

            return TEGroupedMLP, MLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
            )
        elif moe_use_grouped_gemm:
            warnings.warn(
                "The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. "
                "Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP."
            )
            return PrimusTurboGroupedMLP if self.cfg.use_turbo_grouped_mlp else GroupedMLP, None
        else:
            if not is_te_min_version("1.7.0.dev0"):
                warnings.warn(
                    "Only transformer-engine>=1.7.0 supports MoE experts, "
                    f"but your version is {get_te_version()}. "
                    "Use local linear implementation instead."
                )
                return SequentialMLP, MLPSubmodules(
                    linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                )
            return SequentialMLP, MLPSubmodules(
                linear_fc1=self.column_parallel_linear(), linear_fc2=self.row_parallel_linear()
            )

    def activation_func(self) -> type:
        """Which module to use for activation function"""
        return TEActivationOp
