###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import warnings
from types import SimpleNamespace
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
    SequentialMLP,
    TEGroupedMLP,
    TEGroupedMLPSubmodules,
)
from megatron.core.utils import get_te_version, is_te_min_version

try:
    from megatron.core.transformer.moe.experts import GroupedMLP
except ImportError:
    from primus.backends.megatron.core.transformer.moe.deprecated_2caa681a1.experts import (
        DeprecatedGroupedMLP as GroupedMLP,
    )

from primus.backends.megatron.core.extensions.primus_turbo import (
    PrimusTurboAttention,
    PrimusTurboColumnParallelLinear,
    PrimusTurboGroupedMLP,
    PrimusTurboLayerNormColumnParallelLinear,
    PrimusTurboLinear,
    PrimusTurboRowParallelLinear,
)
from primus.backends.megatron.training.global_vars import get_primus_args


def _build_default_primus_args() -> SimpleNamespace:
    """Fallback args for environments without initialized Primus globals."""
    return SimpleNamespace(
        enable_primus_turbo=False,
        use_turbo_parallel_linear=False,
        use_turbo_attention=False,
        use_turbo_grouped_mlp=False,
        moe_use_legacy_grouped_gemm=False,
    )


class PrimusTurboSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def __init__(self):
        try:
            self.cfg = get_primus_args()
        except AssertionError:
            self.cfg = _build_default_primus_args()

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return PrimusTurboLinear if getattr(self.cfg, "use_turbo_parallel_linear", False) else TELinear

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return (
            PrimusTurboColumnParallelLinear
            if getattr(self.cfg, "use_turbo_parallel_linear", False)
            else TEColumnParallelLinear
        )

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module TE backend uses"""
        return (
            PrimusTurboRowParallelLinear
            if getattr(self.cfg, "use_turbo_parallel_linear", False)
            else TERowParallelLinear
        )

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return (
            PrimusTurboLayerNormColumnParallelLinear
            if getattr(self.cfg, "use_turbo_parallel_linear", False)
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
        return (
            PrimusTurboAttention if getattr(self.cfg, "use_turbo_attention", False) else TEDotProductAttention
        )

    def grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: Optional[bool] = None
    ) -> Tuple[type, Optional[MLPSubmodules | TEGroupedMLPSubmodules]]:
        """Which module and submodules to use for grouped mlp"""
        if moe_use_legacy_grouped_gemm is None:
            # Megatron callers only pass ``moe_use_grouped_gemm`` here, so when Primus
            # args do not expose the legacy switch we must match upstream TESpecProvider
            # and prefer TEGroupedMLP by default.
            moe_use_legacy_grouped_gemm = getattr(self.cfg, "moe_use_legacy_grouped_gemm", False)

        if (
            moe_use_grouped_gemm
            and TEColumnParallelGroupedLinear is not None
            and not moe_use_legacy_grouped_gemm
        ):
            _GroupedMLP = (
                PrimusTurboGroupedMLP if getattr(self.cfg, "use_turbo_grouped_mlp", False) else TEGroupedMLP
            )
            # TODO: need to update primus_turbo to support TEColumnParallelGroupedLinear?
            return _GroupedMLP, TEGroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
            )
        elif moe_use_grouped_gemm:
            warnings.warn(
                "The legacy GroupedMLP was removed from this Megatron version; "
                "Primus is using its local compatibility implementation."
            )
            if getattr(self.cfg, "use_turbo_grouped_mlp", False):
                raise NotImplementedError("PrimusTurbo does not support Legacy GroupedMLP")
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


class DeepSeekV4SpecProvider(PrimusTurboSpecProvider):
    """DeepSeek-V4 provider rooted on PrimusTurboSpecProvider."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def v4_norm_module(self):
        """Norm module used by V4 specs (block / layer / final)."""
        return self.layer_norm(rms_norm=True)

    def v4_q_layernorm(self) -> type:
        """Norm module for V4's `q_norm` (RMSNorm on `q_lora_rank`).

        Same as MLA's `q_layernorm`; we use the for_qk path so the TE
        version selection picks Apex / FusedLayerNorm on older TE.
        """
        return self.layer_norm(rms_norm=True, for_qk=True)

    def v4_kv_layernorm(self) -> type:
        """Norm module for V4's `kv_norm` (RMSNorm on `head_dim`).

        Single-latent KV: V4 normalizes the `wkv` output (one shared head)
        BEFORE broadcasting to all query heads.
        """
        return self.layer_norm(rms_norm=True, for_qk=True)

    def v4_attention_sink(self) -> type:
        """Module class for V4's per-head learnable attention sink.

        Returns the local :class:`AttentionSink` wrapper (TE does not
        currently expose a fused softmax-with-sink primitive that matches
        V4's "extra virtual key with zero value" semantics).
        """
        # Lazy import to avoid pulling V4 modules at extension import time.
        from primus.backends.megatron.core.transformer.attn_sink import AttentionSink

        return AttentionSink

    def v4_grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: Optional[bool] = None
    ):
        """Grouped-MLP module selection for V4 MoE expert path."""
        return self.grouped_mlp_modules(
            moe_use_grouped_gemm=moe_use_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )
