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

from primus.backends.megatron.core.extensions.primus_turbo import (
    PrimusTurboAttention,
    PrimusTurboColumnParallelLinear,
    PrimusTurboGroupedMLP,
    PrimusTurboLayerNormColumnParallelLinear,
    PrimusTurboLinear,
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
        """Which module for sequential layernorm and linear.

        Selection logic (priority high -> low):
          1. ``use_turbo_parallel_linear=True`` -> the *existing* primus
             implementation that subclasses ``te.pytorch.LayerNormLinear``
             and is mostly used as part of the broader Primus parallel-linear
             stack.
          2. ``use_turbo_rms_norm=True`` -> our split implementation
             ``[PrimusTurboRMSNorm (Triton), TEColumnParallelLinear]``. The
             fused TE LayerNormLinear bakes ``rmsnorm_fwd_general_kernel``
             into its C++ kernel and is *not* affected by the
             ``te.pytorch.RMSNorm`` patch, so we have to swap the whole
             module to push the Triton kernel into the linear_qkv site.
             Microbench (16384x2880 -> 10240, FP8 DelayedScaling, MI355X)
             shows split is 3-4% faster end-to-end vs the fused TE path:
                 fused TE (RMS) FP8 fwd+bwd  : 1850 us
                 Triton + TELinear FP8 f+b   : 1784 us
          3. otherwise -> stock TE fused LayerNormLinear.
        """
        if self.cfg.use_turbo_parallel_linear:
            return PrimusTurboLayerNormColumnParallelLinear
        # Primus yaml env-var substitution returns the raw string ``"true"`` /
        # ``"false"`` (both truthy under plain ``bool(...)``), so coerce
        # common string forms explicitly here.
        _flag = getattr(self.cfg, "use_turbo_rms_norm", False)
        if _flag is True or (
            isinstance(_flag, str) and _flag.strip().lower() in ("true", "1", "yes", "on")
        ):
            return PrimusTurboLayerNormColumnParallelLinear
        return TELayerNormColumnParallelLinear

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
