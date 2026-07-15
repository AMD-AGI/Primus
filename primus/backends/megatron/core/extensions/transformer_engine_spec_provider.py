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
from primus.backends.megatron.training.global_vars import get_primus_args

try:
    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboAttention,
        PrimusTurboColumnParallelGroupedLinear,
        PrimusTurboColumnParallelLinear,
        PrimusTurboLayerNormColumnParallelLinear,
        PrimusTurboLinear,
        PrimusTurboRowParallelGroupedLinear,
        PrimusTurboRowParallelLinear,
    )
except (ImportError, ModuleNotFoundError):
    PrimusTurboAttention = None
    PrimusTurboColumnParallelGroupedLinear = None
    PrimusTurboColumnParallelLinear = None
    PrimusTurboLayerNormColumnParallelLinear = None
    PrimusTurboLinear = None
    PrimusTurboRowParallelGroupedLinear = None
    PrimusTurboRowParallelLinear = None

_LEGACY_GROUPED_MLP_CLS = None


def _require_primus_turbo(symbol: Optional[type], feature: str) -> type:
    if symbol is None:
        raise RuntimeError(f"PrimusTurbo {feature} was requested, but primus_turbo is not importable.")
    return symbol


def _build_legacy_grouped_mlp_class():
    """Return an adapter class that bridges DeprecatedGroupedMLP into new MoELayer.

    The Megatron upstream (``moe_layer.MoELayer``) calls
    ``build_module(experts_spec, num_local_experts, config, pg_collection=...)``
    with the new ``pg_collection`` keyword. The Primus-bundled
    ``DeprecatedGroupedMLP`` predates that signature and only accepts the
    legacy ``model_comm_pgs`` keyword, so a thin adapter is needed to keep
    the constructor calls compatible without forking the deprecated module.
    """
    global _LEGACY_GROUPED_MLP_CLS
    if _LEGACY_GROUPED_MLP_CLS is not None:
        return _LEGACY_GROUPED_MLP_CLS

    from primus.backends.megatron.core.transformer.moe.deprecated_20251209.experts import (
        DeprecatedGroupedMLP,
    )

    class PrimusLegacyGroupedMLP(DeprecatedGroupedMLP):
        """DeprecatedGroupedMLP shim that accepts the new ``pg_collection`` kwarg."""

        def __init__(
            self,
            num_local_experts: int,
            config,
            pg_collection=None,
            model_comm_pgs=None,
            submodules=None,
        ):
            del submodules  # DeprecatedGroupedMLP holds ``weight1`` / ``weight2`` directly.
            comm_pgs = model_comm_pgs if model_comm_pgs is not None else pg_collection
            super().__init__(
                num_local_experts=num_local_experts,
                config=config,
                model_comm_pgs=comm_pgs,
            )

    _LEGACY_GROUPED_MLP_CLS = PrimusLegacyGroupedMLP
    return PrimusLegacyGroupedMLP


class PrimusTurboSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def __init__(self):
        self.cfg = get_primus_args()

    def linear(self) -> type:
        """Which linear module TE backend uses"""
        return (
            _require_primus_turbo(PrimusTurboLinear, "parallel linear")
            if self.cfg.use_turbo_gemm
            else TELinear
        )

    def column_parallel_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return (
            _require_primus_turbo(PrimusTurboColumnParallelLinear, "column parallel linear")
            if self.cfg.use_turbo_gemm
            else TEColumnParallelLinear
        )

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module TE backend uses"""
        return (
            _require_primus_turbo(PrimusTurboRowParallelLinear, "row parallel linear")
            if self.cfg.use_turbo_gemm
            else TERowParallelLinear
        )

    def fuse_layernorm_and_linear(self) -> bool:
        """TE backend chooses a single module for layernorm and linear"""
        return True

    def column_parallel_layer_norm_linear(self) -> Optional[type]:
        """Which module for sequential layernorm and linear"""
        return (
            _require_primus_turbo(
                PrimusTurboLayerNormColumnParallelLinear, "layernorm column parallel linear"
            )
            if self.cfg.use_turbo_gemm
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
            _require_primus_turbo(PrimusTurboAttention, "attention")
            if self.cfg.use_turbo_attention
            else TEDotProductAttention
        )

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

        use_turbo_grouped_gemm = self.cfg.use_turbo_grouped_gemm
        assert not (
            moe_use_legacy_grouped_gemm and use_turbo_grouped_gemm
        ), "moe_use_legacy_grouped_gemm and use_turbo_grouped_gemm are not compatible."
        if moe_use_grouped_gemm and not moe_use_legacy_grouped_gemm:
            # dispatch to turbo grouped gemm or TE grouped gemm
            _GroupedMLP = PrimusGroupedMLP if use_turbo_grouped_gemm else TEGroupedMLP
            GroupedMLPSubmodules = TEGroupedMLPSubmodules(
                linear_fc1=(
                    _require_primus_turbo(
                        PrimusTurboColumnParallelGroupedLinear, "column parallel grouped linear"
                    )
                    if use_turbo_grouped_gemm
                    else TEColumnParallelGroupedLinear
                ),
                linear_fc2=(
                    _require_primus_turbo(PrimusTurboRowParallelGroupedLinear, "row parallel grouped linear")
                    if use_turbo_grouped_gemm
                    else TERowParallelGroupedLinear
                ),
            )
            return _GroupedMLP, GroupedMLPSubmodules
        elif moe_use_grouped_gemm and moe_use_legacy_grouped_gemm and not use_turbo_grouped_gemm:
            # Legacy grouped-GEMM path without PrimusTurbo: Megatron upstream
            # removed the original ``GroupedMLP`` class, but the Primus
            # pipeline scheduler still relies on its grouped-gemm wgrad-split
            # semantics (see legacy_grouped_mlp_wgrad_patches.py). Route this
            # combination through the bundled DeprecatedGroupedMLP so the
            # zerobubble delayed wgrad path remains intact.
            return _build_legacy_grouped_mlp_class(), None
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
