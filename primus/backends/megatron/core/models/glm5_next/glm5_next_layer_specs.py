###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``glm5_next``) runtime decoder spec.

Per layer ``i``:

* attention: KDA when ``config.is_kda_layer(i)`` else DSA (NoPE absorbed MLA
  with the kpool indexer);
* MLP: dense SwiGLU for ``i < first_k_dense_replace`` else the upstream
  ``MoELayer`` (sigmoid ``noaux_tc`` router, one shared expert);
* both wrapped by mHC inside :class:`Glm5NextLayer`.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from primus.backends.megatron.core.models.glm5_next.build_context import (
    resolve_glm5_next_provider,
)
from primus.backends.megatron.core.models.glm5_next.glm5_next_block import (
    Glm5NextLayer,
    Glm5NextLayerSubmodules,
    Glm5NextTransformerBlock,
    Glm5NextTransformerBlockSubmodules,
)
from primus.backends.megatron.core.transformer.glm5_next.dsa_attention import (
    Glm5NextDSAAttention,
    Glm5NextDSASubmodules,
)
from primus.backends.megatron.core.transformer.glm5_next.kda import (
    Glm5NextKimiDeltaAttention,
)
from primus.backends.megatron.core.transformer.kimi_k3.kimi_delta_attention import (
    KimiDeltaAttentionSubmodules,
)

logger = logging.getLogger(__name__)

__all__ = ["get_glm5_next_moe_layer_pattern", "get_glm5_next_runtime_decoder_spec"]


def get_glm5_next_moe_layer_pattern(config) -> List[int]:
    num_layers = int(config.num_layers)
    if not int(config.num_moe_experts or 0):
        return [0] * num_layers
    freq = config.moe_layer_freq
    if isinstance(freq, str):
        if re.search(r"[^,\d\[\]\(\)\+\*\s]", freq):
            raise ValueError(f"moe_layer_freq contains unsupported characters: {freq!r}")
        freq = eval(freq, {"__builtins__": {}}, {})  # noqa: S307
    if isinstance(freq, int):
        return [1 if (i % freq == 0) else 0 for i in range(num_layers)]
    if len(freq) != num_layers:
        raise ValueError(f"moe_layer_freq has length {len(freq)}, expected num_layers={num_layers}")
    return [int(x) for x in freq]


def _kda_spec(provider) -> ModuleSpec:
    column = provider.column_parallel_linear()
    return ModuleSpec(
        module=Glm5NextKimiDeltaAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=KimiDeltaAttentionSubmodules(
            q_proj=column,
            k_proj=column,
            v_proj=column,
            f_a_proj=provider.column_parallel_linear_with_gather_output(),
            f_b_proj=column,
            b_proj=column,
            g_a_proj=provider.column_parallel_linear_with_gather_output(),
            g_b_proj=column,
            o_proj=provider.row_parallel_linear(),
        ),
    )


def _dsa_spec(provider) -> ModuleSpec:
    norm = provider.glm5_norm_module()
    return ModuleSpec(
        module=Glm5NextDSAAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=Glm5NextDSASubmodules(
            linear_q_down_proj=provider.linear(),
            q_layernorm=norm,
            linear_q_up_proj=provider.column_parallel_linear(),
            linear_kv_down_proj=provider.linear(),
            kv_layernorm=norm,
            linear_kv_up_proj=provider.column_parallel_linear(),
            linear_proj=provider.row_parallel_linear(),
            indexer_linear=provider.linear(),
        ),
    )


def _dense_mlp_spec(provider) -> ModuleSpec:
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=provider.column_parallel_linear(),
            linear_fc2=provider.row_parallel_linear(),
        ),
    )


def _moe_spec(config, provider) -> ModuleSpec:
    return get_moe_module_spec_for_backend(
        backend=provider,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=False,
    )


def build_glm5_next_layer_spec(config, *, provider, layer_idx: int, is_moe: bool) -> ModuleSpec:
    is_kda = bool(config.is_kda_layer(layer_idx))
    norm = provider.glm5_norm_module()
    submodules = Glm5NextLayerSubmodules(
        input_layernorm=norm,
        self_attention=_kda_spec(provider) if is_kda else _dsa_spec(provider),
        pre_mlp_layernorm=norm,
        mlp=_moe_spec(config, provider) if is_moe else _dense_mlp_spec(provider),
    )
    return ModuleSpec(
        module=Glm5NextLayer,
        params={"layer_idx": layer_idx, "is_kda_layer": is_kda},
        submodules=submodules,
    )


def get_glm5_next_runtime_decoder_spec(
    config, *, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
) -> ModuleSpec:
    provider = resolve_glm5_next_provider(config)
    num_layers = int(config.num_layers)
    moe_pattern = get_glm5_next_moe_layer_pattern(config)
    try:
        local_count = int(get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank))
        offset = int(get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank))
    except Exception:
        local_count, offset = num_layers, 0
    start = max(0, offset)
    end = min(num_layers, start + max(0, local_count))
    layer_specs = [
        build_glm5_next_layer_spec(config, provider=provider, layer_idx=i, is_moe=bool(moe_pattern[i]))
        for i in range(start, end)
    ]
    logger.info(
        "[Primus:GLM5-Next] provider=%s layers [%d, %d) attention=%s mlp=%s",
        type(provider).__name__,
        start,
        end,
        "".join("K" if s.params["is_kda_layer"] else "D" for s in layer_specs),
        "".join("E" if moe_pattern[i] else "d" for i in range(start, end)),
    )
    return ModuleSpec(
        module=Glm5NextTransformerBlock,
        submodules=Glm5NextTransformerBlockSubmodules(
            layer_specs=layer_specs,
            final_layernorm=provider.glm5_norm_module(),
        ),
    )
