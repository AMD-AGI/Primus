###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark layer specs (TE backend).

Each decoder layer is Megatron's ``TransformerLayer`` with:

* ``input_layernorm`` / ``pre_mlp_layernorm``: centered RMSNorm (``1 + w``);
  ``linear_qkv`` is a plain column-parallel linear (no fused TE norm, whose
  zero-centered switch would also change the final norm);
* ``self_attention``: :class:`MuseSparkSelfAttention` (output gate packed in
  ``linear_qkv``, weightless QK-norm, per-head ``o_norm``);
* ``self_attn_bda`` / ``mlp_bda``: :class:`MuseSparkGatedResidual`
  (post-norm + learned residual gate instead of bias-dropout-add);
* ``mlp``: :class:`MuseSparkMoELayer` with :class:`MuseSparkTopKRouter` and
  :class:`MuseSparkGroupedMLP` experts.

Only this pipeline stage's layers are built (``get_num_layers_to_build`` /
``get_transformer_layer_offset``), so PP and VPP come from ``TransformerBlock``.
"""

from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
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
    get_transformer_layer_offset,
)

from primus.backends.megatron.core.transformer.muse_spark.attention import (
    MuseSparkQKNorm,
    MuseSparkSelfAttention,
)
from primus.backends.megatron.core.transformer.muse_spark.moe import (
    MuseSparkGroupedMLP,
    MuseSparkMoELayer,
    MuseSparkSequentialMLP,
    MuseSparkTopKRouter,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import (
    MuseSparkCenteredRMSNorm,
    MuseSparkGatedResidual,
    MuseSparkRMSNorm,
)

__all__ = ["get_muse_spark_layer_spec", "get_muse_spark_decoder_block_spec"]


def get_muse_spark_moe_spec(backend=None, grouped_gemm: bool = True) -> ModuleSpec:
    """Latent MoE spec; ``grouped_gemm`` picks TEGroupedMLP or SequentialMLP experts."""
    backend = backend or TESpecProvider()
    base_cls, expert_submodule = backend.grouped_mlp_modules(grouped_gemm)
    if expert_submodule is not None:
        expert_submodule.activation_func = backend.activation_func()
    expert_cls = MuseSparkGroupedMLP if issubclass(MuseSparkGroupedMLP, base_cls) else MuseSparkSequentialMLP
    return ModuleSpec(
        module=MuseSparkMoELayer,
        submodules=MoESubmodules(
            router=MuseSparkTopKRouter,
            experts=ModuleSpec(module=expert_cls, submodules=expert_submodule),
            shared_experts=None,
        ),
        metainfo={"fuse_pre_mlp_layernorm": False},
    )


def get_muse_spark_layer_spec(config) -> ModuleSpec:
    backend = TESpecProvider()
    return ModuleSpec(
        module=TransformerLayer,
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
            mlp=get_muse_spark_moe_spec(
                backend, grouped_gemm=bool(getattr(config, "moe_grouped_gemm", True))
            ),
            mlp_bda=ModuleSpec(module=MuseSparkGatedResidual, params={"config": config, "kind": "mlp"}),
        ),
    )


def get_muse_spark_decoder_block_spec(
    config, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> TransformerBlockSubmodules:
    layer_spec = get_muse_spark_layer_spec(config)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    del offset  # every layer has the same spec; per-layer differences come from layer_number
    return TransformerBlockSubmodules(
        layer_specs=[layer_spec] * num_layers_to_build, layer_norm=MuseSparkRMSNorm
    )
