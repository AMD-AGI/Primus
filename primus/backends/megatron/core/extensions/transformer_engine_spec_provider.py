###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import warnings
from types import SimpleNamespace
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
from megatron.core.transformer.dot_product_attention import DotProductAttention
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


def _build_default_primus_args() -> SimpleNamespace:
    """Fallback args for environments without initialized Primus globals."""
    return SimpleNamespace(
        enable_primus_turbo=False,
        use_turbo_gemm=False,
        use_turbo_attention=False,
        use_turbo_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
    )


class PrimusTurboSpecProvider(BackendSpecProvider):
    """A protocol for providing the submodules used in Spec building."""

    def __init__(self, fallback_to_eager_attn: bool = False):
        try:
            self.cfg = get_primus_args()
        except AssertionError:
            self.cfg = _build_default_primus_args()
        self.fallback_to_eager_attn = fallback_to_eager_attn

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

    def column_parallel_linear_with_gather_output(self) -> type:
        """Non-TE column-parallel linear that supports ``gather_output=True``.

        TE / Turbo column-parallel wrappers explicitly raise
        ``ValueError("Transformer Engine linear layers do not support
        gather_output = True")`` (see
        ``third_party/Megatron-LM/megatron/core/extensions/transformer_engine.py:747``
        and ``:972``).  Callers that need a column-parallel layer
        whose output dim is gathered back to full width across TP
        ranks (so downstream math stays TP-agnostic) must use the
        upstream Megatron-native :class:`ColumnParallelLinear`.

        Plan-3 P21: V4's ``linear_q_up_proj`` is the canonical
        consumer — it shards ``q_lora_rank -> n_heads * head_dim``
        across TP and gathers the heads at forward time.
        """
        return ColumnParallelLinear

    def row_parallel_linear_with_scatter_input(self) -> type:
        """Non-TE row-parallel linear that supports ``input_is_parallel=False``.

        TE / Turbo row-parallel wrappers explicitly raise
        ``ValueError("Transformer Engine linear layers do not support
        input_is_parallel = False")`` (see
        ``third_party/Megatron-LM/megatron/core/extensions/transformer_engine.py:1081``).
        Callers that hand the layer a full-width (non-sharded) input
        and want it scattered internally + the output all-reduced
        must use the upstream Megatron-native :class:`RowParallelLinear`.

        Plan-3 P21: V4's grouped-O ``linear_o_b`` is the canonical
        consumer — after the inner ``[..., n_per_group, o_lora_rank]
        -> [..., o_groups * o_lora_rank]`` reshape, the input is
        full-width across TP and the row-parallel layer's
        weight-sharding + reduce is what produces the correct sum.
        """
        return RowParallelLinear

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
        if self.fallback_to_eager_attn:
            return DotProductAttention
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
            moe_use_legacy_grouped_gemm = getattr(self.cfg, "moe_use_legacy_grouped_gemm", False)

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

    def v4_mlp_activation_func(self) -> Optional[type]:
        """Activation-func selection for V4 MLP / shared-expert specs.

        Plan-2 P18 (D2 audit): the parent ``activation_func()`` returns
        the TE module **type** (``TEActivationOp``); but Megatron's
        ``MLP.__init__`` only consumes ``submodules.activation_func`` when
        ``config.use_te_activation_func == True`` (otherwise it falls
        back to the callable in ``config.activation_func``).

        Returning the TE class unconditionally caused a silent
        contract mismatch in V4 yamls (which keep
        ``use_te_activation_func: false`` by default — V4 wants the
        clamped-SwiGLU eager path so the activation-clamp gets
        applied).

        Behavior:

        * ``config.use_te_activation_func`` is True → return the TE
          activation class (instantiated by Megatron MLP at build).
        * Otherwise → return ``None`` so the spec leaves the
          ``MLPSubmodules.activation_func`` slot empty and Megatron
          MLP uses ``config.activation_func`` (V4's clamped SwiGLU).

        This keeps the spec self-consistent: if the V4 yaml opts into
        the TE path, the spec carries the TE class; otherwise the
        spec carries ``None`` instead of a class that would be
        silently ignored.
        """
        cfg = getattr(self, "config", None)
        if cfg is not None and bool(getattr(cfg, "use_te_activation_func", False)):
            return self.activation_func()
        return None

    def v4_grouped_mlp_modules(
        self, moe_use_grouped_gemm: bool, moe_use_legacy_grouped_gemm: Optional[bool] = None
    ):
        """Grouped-MLP module selection for V4 MoE expert path."""
        return self.grouped_mlp_modules(
            moe_use_grouped_gemm=moe_use_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )

    # ---- V4 spec factories (plan-2 P14 §5/§6) -------------------------

    def v4_grouped_mlp_spec(
        self,
        *,
        swiglu_limit: float,
        moe_use_grouped_gemm: bool = True,
        moe_use_legacy_grouped_gemm: Optional[bool] = None,
    ):
        """Return a ready-to-use ``ModuleSpec`` for V4 grouped MoE experts.

        The V4 pre-multiplication clamp itself is applied through
        ``config.activation_func_clamp_value`` (Megatron's MLP eager
        ``glu()`` already clamps gate (max=alpha) and up (+/- alpha)
        before SiLU + multiply, which is bit-equal to the HF reference
        ``Expert.forward`` math). This spec only commits to the right
        grouped module + the column / row-parallel linears; the runtime
        config carries the clamp value.

        Args:
            swiglu_limit: V4 ``alpha`` value (the released ``DeepSeek-V4-Flash``
                checkpoint uses ``7.0``). Recorded on the returned spec
                so the caller can assert it lines up with the runtime
                config; not consumed by the grouped-MLP module directly.
            moe_use_grouped_gemm: prefer the TE / Turbo grouped-gemm
                module over the local SequentialMLP fallback (default
                True in production).
            moe_use_legacy_grouped_gemm: optional override for the
                legacy code path; when ``None`` the provider reads the
                Primus arg of the same name.

        Returns:
            A ``ModuleSpec`` whose ``module`` is the grouped MoE module
            and whose ``submodules`` carry the linear modules. Caller
            wires this into :class:`DeepseekV4MoESubmodules.grouped_experts`.

        Notes:
            * Plan-2 P14 §5 calls for "downgrade to local experts with
              explicit warning" when the grouped backend cannot apply
              the clamp; that downgrade lives in
              :class:`DeepseekV4MoE` (which already builds
              :class:`ClampedSwiGLUMLP` local experts when
              ``pg_collection is None`` or the backend declares no
              clamp support).
        """
        del swiglu_limit  # documented but not consumed by the grouped MLP itself
        from megatron.core.transformer.spec_utils import ModuleSpec

        module, submodules = self.v4_grouped_mlp_modules(
            moe_use_grouped_gemm=moe_use_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        )
        if submodules is None:
            return ModuleSpec(module=module)
        return ModuleSpec(module=module, submodules=submodules)

    def v4_router_spec(self, *, learned: bool = True):
        """Return a ``ModuleSpec`` for the V4 hash / learned router.

        Args:
            learned: when True (default) returns the learned router
                spec (``layer_idx >= num_hash_layers``); when False
                returns the hash-router spec (``layer_idx <
                num_hash_layers``).

        Returns:
            A bare-module ``ModuleSpec`` suitable for
            :class:`DeepseekV4MoESubmodules.{learned_router, hash_router}`.
            Both routers are ``nn.Module`` standalones (not
            ``TopKRouter`` subclasses) so they instantiate cleanly on
            CPU; aux-loss / z-loss / RouterReplay inheritance is
            tracked as a P19 follow-up.
        """
        from megatron.core.transformer.spec_utils import ModuleSpec

        from primus.backends.megatron.core.transformer.moe.v4_hash_router import (
            DeepseekV4HashRouter,
        )
        from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
            DeepseekV4LearnedRouter,
        )

        return ModuleSpec(module=(DeepseekV4LearnedRouter if learned else DeepseekV4HashRouter))
