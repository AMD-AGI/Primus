###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark decoder spec: the stock TE GPT block spec with five classes swapped.

``get_gpt_decoder_block_spec`` already provides everything else Muse needs:
GQA with ``qk_l2_norm`` (weightless QK RMSNorm), the attention output gate,
per-layer sliding window and NoPE from the config, latent MoE projections,
and pipeline slicing. The swaps are:

* ``TransformerLayer`` -> :class:`MuseSparkTransformerLayer`
* ``SelfAttention``    -> :class:`MuseSparkSelfAttention`
* query ``L2Norm``     -> :class:`MuseSparkQueryNorm`
* ``MoELayer``         -> :class:`MuseSparkMoELayer`
* router               -> :class:`MuseSparkTopKRouter`
"""

import dataclasses
from typing import Optional

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer

from primus.backends.megatron.core.models.muse_spark.muse_spark_modules import (
    MuseSparkMoELayer,
    MuseSparkMoESubmodules,
    MuseSparkQueryNorm,
    MuseSparkSelfAttention,
    MuseSparkTopKRouter,
    MuseSparkTransformerLayer,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
    MuseSparkTransformerConfig,
)

__all__ = ["get_muse_spark_decoder_block_spec"]


def _expect_module(spec, expected: type, what: str) -> None:
    module = getattr(spec, "module", None)
    if module is not expected:
        raise TypeError(
            f"Muse Spark expects the upstream {what} spec module to be {expected.__name__}, "
            f"got {getattr(module, '__name__', module)!r}. A patch that replaces it "
            "(e.g. Turbo MegaMoE) is not compatible with Muse Spark."
        )


def _muse_spark_layer_spec(layer_spec: ModuleSpec) -> ModuleSpec:
    _expect_module(layer_spec, TransformerLayer, "decoder layer")
    submodules = layer_spec.submodules

    _expect_module(submodules.self_attention, SelfAttention, "self-attention")
    attention = submodules.self_attention.submodules
    if attention.q_layernorm is not L2Norm:
        raise TypeError(f"Muse Spark expects the upstream L2Norm query norm, got {attention.q_layernorm!r}.")
    self_attention = dataclasses.replace(
        submodules.self_attention,
        module=MuseSparkSelfAttention,
        submodules=dataclasses.replace(attention, q_layernorm=MuseSparkQueryNorm),
    )

    mlp = submodules.mlp
    mlp_module = getattr(mlp, "module", None)
    if isinstance(mlp_module, type) and issubclass(mlp_module, MoELayer):
        _expect_module(mlp, MoELayer, "MoE")
        moe = mlp.submodules
        mlp = dataclasses.replace(
            mlp,
            module=MuseSparkMoELayer,
            submodules=MuseSparkMoESubmodules(
                experts=moe.experts,
                shared_experts=moe.shared_experts,
                router=MuseSparkTopKRouter,
                pre_expert_norm=submodules.pre_mlp_layernorm,
            ),
        )

    return dataclasses.replace(
        layer_spec,
        module=MuseSparkTransformerLayer,
        submodules=dataclasses.replace(submodules, self_attention=self_attention, mlp=mlp),
    )


def get_muse_spark_decoder_block_spec(
    config: MuseSparkTransformerConfig,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Return the Muse Spark block spec for the layers built on this pipeline stage."""
    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        normalization=config.normalization,
        qk_l2_norm=True,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )
    return TransformerBlockSubmodules(
        layer_specs=[_muse_spark_layer_spec(spec) for spec in block_spec.layer_specs],
        layer_norm=block_spec.layer_norm,
    )
