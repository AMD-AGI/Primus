###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling decoder spec tree.

One :class:`InklingTransformerLayer` per layer, dispatched on two
independent per-layer patterns:

* ``config.local_layer_ids`` -- sliding-window vs global attention, which
  changes the head geometry and the relative-bias extent;
* ``config.moe_layer_freq`` -- the dense SwiGLU MLP (with ``global_scale``)
  vs the shared-expert-sink MoE.

Norms are separate modules rather than fused into the following GEMM
(``TENorm`` + ``TEColumnParallelLinear``), which keeps the parameter names
one-to-one with the checkpoint; fusing them is a later optimisation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)
from primus.backends.megatron.core.transformer.inkling.attention import (
    InklingCoreAttention,
    InklingSelfAttention,
    InklingSelfAttentionSubmodules,
)
from primus.backends.megatron.core.transformer.inkling.layer import (
    InklingTransformerLayer,
)
from primus.backends.megatron.core.transformer.inkling.mlp import InklingMLP
from primus.backends.megatron.core.transformer.inkling.moe import (
    InklingGroupedMLP,
    InklingMoELayer,
    InklingSharedExperts,
)
from primus.backends.megatron.core.transformer.inkling.router import InklingTopKRouter

logger = logging.getLogger(__name__)

__all__ = ["get_inkling_decoder_block_spec", "get_inkling_layer_spec"]


def _attention_spec(backend: TESpecProvider) -> ModuleSpec:
    norm = backend.layer_norm(rms_norm=True, for_qk=True)
    return ModuleSpec(
        module=InklingSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=InklingSelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(),
            linear_rel=backend.column_parallel_linear(),
            core_attention=InklingCoreAttention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=norm,
            k_layernorm=norm,
        ),
    )


def _dense_mlp_spec(backend: TESpecProvider) -> ModuleSpec:
    return ModuleSpec(
        module=InklingMLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
        ),
    )


def _moe_spec(config: InklingTransformerConfig, backend: TESpecProvider) -> ModuleSpec:
    expert_module, expert_submodules = backend.grouped_mlp_modules(bool(config.moe_grouped_gemm))
    if expert_module is TEGroupedMLP:
        expert_module = InklingGroupedMLP
    elif config.inkling_fp32_activations:
        raise NotImplementedError(
            "inkling_fp32_activations needs TE grouped experts (moe_grouped_gemm: true, "
            f"use_turbo_grouped_gemm: false), got {expert_module.__name__}"
        )
    return ModuleSpec(
        module=InklingMoELayer,
        submodules=MoESubmodules(
            experts=ModuleSpec(module=expert_module, submodules=expert_submodules),
            shared_experts=ModuleSpec(
                module=InklingSharedExperts,
                submodules=MLPSubmodules(
                    linear_fc1=backend.column_parallel_linear(),
                    linear_fc2=backend.row_parallel_linear(),
                ),
            ),
            router=InklingTopKRouter,
        ),
    )


def get_inkling_layer_spec(config: InklingTransformerConfig, *, is_moe: bool) -> ModuleSpec:
    """One decoder layer. Attention geometry is resolved per layer from ``layer_number``."""
    backend = TESpecProvider()
    norm = backend.layer_norm(rms_norm=True)
    return ModuleSpec(
        module=InklingTransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=norm,
            self_attention=_attention_spec(backend),
            pre_mlp_layernorm=norm,
            mlp=_moe_spec(config, backend) if is_moe else _dense_mlp_spec(backend),
        ),
    )


def _stage_layer_range(config, vp_stage: Optional[int], pp_rank: Optional[int]):
    num_layers = int(config.num_layers)
    try:
        count = int(get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank))
        offset = int(get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank))
    except Exception:
        # No initialised parallel_state (spec-only tests): whole stack on one stage.
        count, offset = num_layers, 0
    start = max(0, offset)
    return range(start, min(num_layers, start + max(0, count)))


def get_inkling_decoder_block_spec(
    config: InklingTransformerConfig,
    *,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Layer specs for this pipeline stage plus the final RMSNorm.

    ``TransformerBlock`` indexes ``layer_specs`` by local position and gives
    each layer its global ``layer_number``, so the specs are listed for the
    stage's global range only.
    """
    moe = config.moe_layer_pattern()
    layers = _stage_layer_range(config, vp_stage, pp_rank)
    layer_specs: List[ModuleSpec] = [get_inkling_layer_spec(config, is_moe=bool(moe[i])) for i in layers]
    assert layer_specs, "Inkling requires non-empty stage layer specs"
    logger.info(
        "[Primus:Inkling] layers %d-%d: attention=%s mlp=%s",
        layers.start,
        layers.stop - 1,
        "".join("S" if config.is_sliding_layer(i) else "G" for i in layers),
        "".join("E" if moe[i] else "D" for i in layers),
    )
    return TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=TESpecProvider().layer_norm(rms_norm=True),
    )
