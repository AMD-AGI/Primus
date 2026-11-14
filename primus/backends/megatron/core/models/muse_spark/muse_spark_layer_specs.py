###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark layer specs (TE backend).

Each decoder layer is :class:`MuseSparkTransformerLayer` (Megatron's ``TransformerLayer``)
with:

* ``input_layernorm`` / ``pre_mlp_layernorm``: centered RMSNorm (``1 + w``);
  ``linear_qkv`` is a plain column-parallel linear (no fused TE norm, whose
  zero-centered switch would also change the final norm);
* ``self_attention``: :class:`MuseSparkSelfAttention` (output gate packed in
  ``linear_qkv``, weightless QK-norm, per-head ``o_norm``);
* ``self_attn_bda`` / ``mlp_bda``: :class:`MuseSparkGatedResidual`
  (post-norm + learned residual gate instead of bias-dropout-add);
* ``mlp``: :class:`MuseSparkMoELayer` with :class:`MuseSparkTopKRouter` and
  the provider's experts (``PrimusGroupedMLP`` with Turbo grouped GEMM, else
  ``TEGroupedMLP`` / ``SequentialMLP``), each with ``post_expert_norm``.

Only this pipeline stage's layers are built (``get_num_layers_to_build``); the
per-layer choices use the global ``layer_number``, so PP and VPP come from ``TransformerBlock``.
"""

import functools
from typing import Optional

from megatron.core.extensions import (
    transformer_engine_spec_provider as _te_spec_provider,
)
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from primus.backends.megatron.core.transformer.muse_spark.attention import (
    MuseSparkQKNorm,
    MuseSparkSelfAttention,
)
from primus.backends.megatron.core.transformer.muse_spark.moe import (
    MuseSparkMoELayer,
    MuseSparkTopKRouter,
    muse_spark_expert_cls,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkCenteredRMSNorm,
    MuseSparkGatedResidual,
    MuseSparkRMSNorm,
)

__all__ = [
    "MuseSparkTransformerLayer",
    "get_muse_spark_layer_spec",
    "get_muse_spark_decoder_block_spec",
]


class MuseSparkTransformerLayer(TransformerLayer):
    """Megatron's ``TransformerLayer`` for the Muse Spark decoder (its own class, so the
    built-module log line names the Muse Spark layer). The pinned Megatron-core builds every
    mlp, a ``MoELayer`` subclass included, with ``pg_collection`` / ``is_mtp_layer``, so
    nothing is overridden."""


def _spec_provider():
    """The TE spec provider, looked up at call time: the Primus-Turbo patch replaces the
    module attribute ``TESpecProvider`` (with ``PrimusTurboSpecProvider``) after import."""
    return _te_spec_provider.TESpecProvider()


def _expert_module_and_submodules(backend, grouped_gemm: bool):
    """``(module, submodules)`` of the provider's experts on every Megatron version.

    Older Megatron returns the tuple; Megatron-core 0.19+ returns an ``ExpertsBuilder``: a
    ``functools.partial`` (Megatron's TESpecProvider) or Primus's ``build`` closure, whose
    ``(module, submodules)`` the Primus provider also exposes directly."""
    inner = getattr(backend, "_grouped_mlp_module_and_submodules", None)
    if inner is not None:
        return inner(grouped_gemm)
    experts = backend.grouped_mlp_modules(grouped_gemm)
    if isinstance(experts, functools.partial):
        return experts.func, experts.keywords.get("submodules")
    return experts


def get_muse_spark_moe_spec(backend=None, grouped_gemm: bool = True) -> ModuleSpec:
    """Latent MoE spec. The provider picks the experts (Turbo grouped GEMM, TE grouped
    GEMM or sequential); each gets its Muse Spark subclass with ``post_expert_norm``."""
    backend = backend or _spec_provider()
    base_cls, expert_submodule = _expert_module_and_submodules(backend, grouped_gemm)
    if expert_submodule is not None:
        expert_submodule.activation_func = backend.activation_func()
    expert_cls = muse_spark_expert_cls(base_cls)
    return ModuleSpec(
        module=MuseSparkMoELayer,
        submodules=MoESubmodules(
            router=MuseSparkTopKRouter,
            # a ModuleSpec is callable as the 0.19+ ExpertsBuilder too:
            # spec(num_local_experts, config, pg_collection=..., name=...)
            experts=ModuleSpec(module=expert_cls, submodules=expert_submodule),
            shared_experts=None,
        ),
        metainfo={"fuse_pre_mlp_layernorm": False},
    )


def get_muse_spark_layer_spec(config) -> ModuleSpec:
    backend = _spec_provider()
    return ModuleSpec(
        module=MuseSparkTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=MuseSparkCenteredRMSNorm,
            self_attention=ModuleSpec(
                module=MuseSparkSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=MuseSparkQKNorm,
                    k_layernorm=MuseSparkQKNorm,
                ),
            ),
            self_attn_bda=ModuleSpec(
                module=MuseSparkGatedResidual, params={"config": config, "kind": "attn"}
            ),
            pre_mlp_layernorm=MuseSparkCenteredRMSNorm,
            mlp=get_muse_spark_moe_spec(backend, grouped_gemm=bool(config.moe_grouped_gemm)),
            mlp_bda=ModuleSpec(module=MuseSparkGatedResidual, params={"config": config, "kind": "mlp"}),
        ),
    )


def get_muse_spark_decoder_block_spec(
    config, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> TransformerBlockSubmodules:
    layer_spec = get_muse_spark_layer_spec(config)
    # Every layer has the same spec; the per-layer window / NoPE choice comes from the
    # global layer_number inside TransformerBlock, so no offset is needed here.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return TransformerBlockSubmodules(
        layer_specs=[layer_spec] * num_layers_to_build, layer_norm=MuseSparkRMSNorm
    )
