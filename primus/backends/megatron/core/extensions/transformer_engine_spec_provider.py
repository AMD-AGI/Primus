###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import warnings
from typing import Optional

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
    SequentialMLP,
    TEGroupedMLP,
    TEGroupedMLPSubmodules,
)
from megatron.core.utils import get_te_version, is_te_min_version

from primus.backends.megatron.core.transformer.experts import PrimusGroupedMLP

try:
    from megatron.core.transformer.moe.experts import GroupedMLP
except ImportError:
    from primus.backends.megatron.core.transformer.moe.deprecated_2caa681a1.experts import (
        DeprecatedGroupedMLP as GroupedMLP,
    )

from primus.backends.megatron.core.extensions.primus_turbo import (
    PrimusTurboAttention,
    PrimusTurboColumnParallelGroupedLinear,
    PrimusTurboColumnParallelLinear,
    PrimusTurboLayerNormColumnParallelLinear,
    PrimusTurboLinear,
    PrimusTurboRowParallelGroupedLinear,
    PrimusTurboRowParallelLinear,
)
from primus.backends.megatron.training.global_vars import get_primus_args


class PrimusTurboSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def __init__(self):
        self.cfg = get_primus_args()

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return PrimusTurboLinear if self.cfg.use_turbo_parallel_linear else TELinear

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
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: Optional[bool] = None
    ) -> tuple[type[TEGroupedMLP], TEGroupedMLPSubmodules] | tuple[type[SequentialMLP], MLPSubmodules]:
        """Which module and submodules to use for grouped mlp"""
        if moe_use_legacy_grouped_gemm is None:
            # Megatron callers only pass ``moe_use_grouped_gemm`` here, so when Primus
            # args do not expose the legacy switch we must match upstream TESpecProvider
            # and prefer TEGroupedMLP by default.
            # let it raise an error if cfg does not have moe_use_legacy_grouped_gemm
            moe_use_legacy_grouped_gemm = self.cfg.moe_use_legacy_grouped_gemm

        assert not (
            moe_use_legacy_grouped_gemm and use_turbo_grouped_gemm
        ), "moe_use_legacy_grouped_gemm and use_turbo_grouped_gemm are not compatible."
        use_turbo_grouped_gemm = self.cfg.use_turbo_grouped_gemm or self.cfg.use_turbo_grouped_mlp
        if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
            # dispatch to turbo grouped gemm or TE grouped gemm
            _GroupedMLP = PrimusGroupedMLP if use_turbo_grouped_gemm else TEGroupedMLP
            GroupedMLPSubmodules = TEGroupedMLPSubmodules(
                linear_fc1=(
                    PrimusTurboColumnParallelGroupedLinear
                    if use_turbo_grouped_gemm
                    else TEColumnParallelGroupedLinear
                ),
                linear_fc2=(
                    PrimusTurboRowParallelGroupedLinear
                    if use_turbo_grouped_gemm
                    else TERowParallelGroupedLinear
                ),
            )
            return _GroupedMLP, GroupedMLPSubmodules
        elif moe_use_grouped_gemm:
            # dispatch to legacy grouped gemm
            warnings.warn(
                "The legacy GroupedMLP was removed from this Megatron version; "
                "Primus is using its local compatibility implementation."
            )
            return GroupedMLP, None
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
