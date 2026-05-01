###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 spec entry points.

This module only defines DeepSeek-native runtime specs.
"""

import logging
from typing import List, Optional

from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
    DeepSeekV4SpecProvider,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_block import (
    DeepseekV4HybridLayer,
    DeepseekV4HybridLayerSubmodules,
    DeepseekV4TransformerBlock,
    DeepseekV4TransformerBlockSubmodules,
    _DenseSwiGLUMLP,
    _normalize_compress_ratios,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)
from primus.backends.megatron.core.transformer.csa_attention import CSAAttention
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
    DeepseekV4AttentionSubmodules,
    _LegacyDeepseekV4AttentionSubmodules,
)
from primus.backends.megatron.core.transformer.hca_attention import HCAAttention
from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperHead,
    HyperMixer,
)
from primus.backends.megatron.core.transformer.moe.v4_hash_router import HashRouter
from primus.backends.megatron.core.transformer.moe.v4_moe import (
    DeepseekV4MoE,
    DeepseekV4MoESubmodules,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import V4TopKRouter

logger = logging.getLogger(__name__)


def _default_init_method(_weight) -> None:
    return None


def _build_linear_projection_spec(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    in_features: int,
    out_features: int,
) -> ModuleSpec:
    return ModuleSpec(
        module=provider.linear(),
        params={
            "input_size": in_features,
            "output_size": out_features,
            "parallel_mode": "duplicated",
            "config": config,
            "init_method": config.init_method or _default_init_method,
            "bias": False,
            "skip_bias_add": False,
            "skip_weight_param_allocation": False,
            "tp_comm_buffer_name": None,
            "is_expert": False,
        },
    )


def _build_legacy_attention_submodules(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
) -> _LegacyDeepseekV4AttentionSubmodules:
    """Plan-1 submodules — used by the compressed branches (CSA / HCA)
    until plan-2 P13 follow-up folds compressor / indexer onto the new
    :class:`DeepseekV4Attention` as spec submodules."""
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_query_groups or num_heads)
    head_dim = int(config.kv_channels or (hidden_size // num_heads))
    q_lora_rank = config.q_lora_rank or None
    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim
    q_a_spec = None
    q_input_dim = hidden_size
    if q_lora_rank is not None and q_lora_rank > 0:
        q_a_spec = _build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=hidden_size,
            out_features=q_lora_rank,
        )
        q_input_dim = q_lora_rank

    return _LegacyDeepseekV4AttentionSubmodules(
        linear_q_a=q_a_spec,
        linear_q_b=_build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=q_input_dim,
            out_features=q_out,
        ),
        linear_k_proj=_build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=hidden_size,
            out_features=kv_out,
        ),
        linear_v_proj=_build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=hidden_size,
            out_features=kv_out,
        ),
        linear_o_proj=_build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=q_out,
            out_features=hidden_size,
        ),
    )


def _build_v4_attention_submodules(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
) -> DeepseekV4AttentionSubmodules:
    """Plan-2 P13 submodules for :class:`DeepseekV4Attention`.

    Field names match the released V4-Flash checkpoint layout (and MLA's
    canonical names where they overlap):

    * ``linear_q_down_proj``  : ``hidden -> q_lora_rank``  (= ``wq_a``)
    * ``q_layernorm``        : RMSNorm(``q_lora_rank``)    (= ``q_norm``)
    * ``linear_q_up_proj``    : ``q_lora_rank -> n_heads * head_dim`` (= ``wq_b``)
    * ``linear_kv``           : ``hidden -> head_dim``     (= ``wkv``,
      single-latent KV)
    * ``kv_layernorm``       : RMSNorm(``head_dim``)       (= ``kv_norm``)
    * ``linear_o_a``         : grouped low-rank O down-proj
    * ``linear_o_b``         : grouped low-rank O up-proj  (-> ``hidden``)
    * ``attn_sink``          : per-head learnable scalar
    """
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.kv_channels or (hidden_size // num_heads))
    q_lora_rank = int(config.q_lora_rank or 0)
    o_groups = max(int(getattr(config, "o_groups", 1) or 1), 1)
    o_lora_rank = int(getattr(config, "o_lora_rank", 0) or 0)

    if q_lora_rank <= 0:
        raise ValueError(
            "DeepSeek-V4 requires q_lora_rank > 0; the released checkpoint "
            "always low-rank-projects Q via wq_a / wq_b."
        )

    q_out = num_heads * head_dim
    submods = DeepseekV4AttentionSubmodules(
        linear_q_down_proj=_build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=hidden_size,
            out_features=q_lora_rank,
        ),
        linear_q_up_proj=_build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=q_lora_rank,
            out_features=q_out,
        ),
        linear_kv=_build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=hidden_size,
            out_features=head_dim,  # single-latent: K = V = wkv(hidden)
        ),
        q_layernorm=ModuleSpec(module=provider.v4_q_layernorm()),
        kv_layernorm=ModuleSpec(module=provider.v4_kv_layernorm()),
        attn_sink=(
            ModuleSpec(module=provider.v4_attention_sink())
            if bool(getattr(config, "attn_sink", False))
            else None
        ),
    )

    if o_lora_rank > 0:
        n_per_group = q_out // o_groups
        submods.linear_o_a = _build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=n_per_group,
            out_features=o_groups * o_lora_rank,
        )
        submods.linear_o_b = _build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=o_groups * o_lora_rank,
            out_features=hidden_size,
        )
    else:
        # Fallback flat O projection (MLA style).
        submods.linear_proj = _build_linear_projection_spec(
            config=config,
            provider=provider,
            in_features=q_out,
            out_features=hidden_size,
        )

    return submods


def _build_norm_spec(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
):
    del config
    norm_module = provider.v4_norm_module()
    assert norm_module is not None, "DeepSeek-V4 norm module must be provided by DeepSeekV4SpecProvider."
    return ModuleSpec(module=norm_module)


def _build_attention_spec(
    *,
    compress_ratio: int,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
) -> ModuleSpec:
    if compress_ratio == 0:
        # Plan-2 P13: the dense path uses the faithful, MLA-rooted
        # DeepseekV4Attention with V4-canonical submodules.
        return ModuleSpec(
            module=DeepseekV4Attention,
            params={"compress_ratio": 0},
            submodules=_build_v4_attention_submodules(
                config=config,
                provider=provider,
            ),
        )

    # Compressed branches still ride on the plan-1 legacy attention until
    # their compressor / indexer logic is folded into DeepseekV4Attention
    # in a P13 follow-up.
    legacy_submodules = _build_legacy_attention_submodules(
        config=config,
        provider=provider,
    )
    if compress_ratio == 4:
        return ModuleSpec(
            module=CSAAttention,
            params={"compress_ratio": compress_ratio},
            submodules=legacy_submodules,
        )
    return ModuleSpec(
        module=HCAAttention,
        params={"compress_ratio": compress_ratio},
        submodules=legacy_submodules,
    )


def _build_ffn_spec(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    layer_idx: int,
) -> ModuleSpec:
    num_routed_experts = int(config.num_moe_experts)
    moe_use_grouped_gemm = bool(config.moe_grouped_gemm)
    moe_use_legacy_grouped_gemm = bool(config.moe_use_legacy_grouped_gemm)
    grouped_mlp_module, grouped_mlp_submodules = provider.v4_grouped_mlp_modules(
        moe_use_grouped_gemm=moe_use_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    dispatcher_type = str(config.moe_token_dispatcher_type).lower()
    if dispatcher_type == "allgather":
        dispatcher_cls = MoEAllGatherTokenDispatcher
    elif dispatcher_type == "flex":
        dispatcher_cls = MoEFlexTokenDispatcher
    else:
        if dispatcher_type != "alltoall":
            logger.warning(
                "[DeepSeek-V4] unsupported moe_token_dispatcher_type=%s; fallback to alltoall.",
                dispatcher_type,
            )
        dispatcher_type = "alltoall"
        dispatcher_cls = MoEAlltoAllTokenDispatcher

    assert (
        grouped_mlp_module is not None
    ), "DeepSeek-V4 grouped MLP module must be provided by DeepSeekV4SpecProvider."

    grouped_experts_spec = ModuleSpec(
        module=grouped_mlp_module,
        submodules=grouped_mlp_submodules,
    )

    shared_expert_submodules = MLPSubmodules(
        linear_fc1=provider.column_parallel_linear(),
        linear_fc2=provider.row_parallel_linear(),
        activation_func=provider.activation_func(),
    )
    shared_expert_spec = ModuleSpec(
        module=SharedExpertMLP,
        submodules=shared_expert_submodules,
    )

    moe_submodules = DeepseekV4MoESubmodules(
        hash_router=ModuleSpec(module=HashRouter),
        learned_router=ModuleSpec(module=V4TopKRouter),
        token_dispatcher=ModuleSpec(module=dispatcher_cls),
        grouped_experts=grouped_experts_spec,
        shared_expert=shared_expert_spec,
    )

    if num_routed_experts > 0:
        return ModuleSpec(
            module=DeepseekV4MoE,
            params={
                "layer_idx": layer_idx,
            },
            submodules=moe_submodules,
        )
    return ModuleSpec(module=_DenseSwiGLUMLP)


def _build_hybrid_layer_spec(
    config: DeepSeekV4TransformerConfig,
    *,
    provider: DeepSeekV4SpecProvider,
    layer_idx: int,
    compress_ratio: int,
) -> ModuleSpec:
    hc_mult = int(config.hc_mult)

    layer_submodules = DeepseekV4HybridLayerSubmodules(
        attn_norm=_build_norm_spec(config=config, provider=provider),
        attention=_build_attention_spec(
            compress_ratio=compress_ratio,
            config=config,
            provider=provider,
        ),
        ffn_norm=_build_norm_spec(config=config, provider=provider),
        ffn=_build_ffn_spec(
            config=config,
            provider=provider,
            layer_idx=layer_idx,
        ),
        attn_hc=ModuleSpec(module=HyperMixer) if hc_mult > 1 else None,
        ffn_hc=ModuleSpec(module=HyperMixer) if hc_mult > 1 else None,
    )

    return ModuleSpec(
        module=DeepseekV4HybridLayer,
        params={
            "layer_idx": layer_idx,
            "compress_ratio": int(compress_ratio),
        },
        submodules=layer_submodules,
    )


def _build_stage_hybrid_layer_specs(
    config: DeepSeekV4TransformerConfig,
    *,
    provider: DeepSeekV4SpecProvider,
    vp_stage: Optional[int],
) -> List[ModuleSpec]:
    """Build the current stage's decoder layer specs.

    DeepSeek-V4 runtime always materializes a concrete stage-local
    `layer_specs` list for `DeepseekV4TransformerBlock`.
    """
    num_layers = int(config.num_layers)
    mtp_num_layers = int(config.mtp_num_layers)
    compress_ratios = _normalize_compress_ratios(
        config.compress_ratios,
        num_layers=num_layers,
        mtp_num_layers=mtp_num_layers,
    )

    try:
        local_layer_count = int(get_num_layers_to_build(config, vp_stage=vp_stage))
        layer_offset = int(get_transformer_layer_offset(config, vp_stage=vp_stage))
    except Exception:
        local_layer_count = num_layers
        layer_offset = 0

    local_start = max(0, layer_offset)
    local_end = min(num_layers, local_start + max(0, local_layer_count))
    local_layer_indices = range(local_start, local_end)

    return [
        _build_hybrid_layer_spec(
            config,
            provider=provider,
            layer_idx=layer_idx,
            compress_ratio=int(compress_ratios[layer_idx]),
        )
        for layer_idx in local_layer_indices
    ]


def get_deepseek_v4_runtime_decoder_spec(
    config: DeepSeekV4TransformerConfig,
    *,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> ModuleSpec:
    """Return the effective V4 runtime decoder spec tree.

    The returned block submodules always include a non-empty `layer_specs`.
    """
    del pp_rank

    provider = DeepSeekV4SpecProvider(config=config)
    logger.info("[DeepSeek-V4] resolve spec provider=%s", type(provider).__name__)

    hc_mult = int(config.hc_mult)
    stage_layer_specs = _build_stage_hybrid_layer_specs(
        config,
        provider=provider,
        vp_stage=vp_stage,
    )
    assert stage_layer_specs, "DeepSeek-V4 requires non-empty stage layer specs."
    block_submodules = DeepseekV4TransformerBlockSubmodules(
        layer_specs=stage_layer_specs,
        hyper_head=ModuleSpec(module=HyperHead) if hc_mult > 1 else None,
        # DeepseekV4TransformerBlock decides whether this stage owns final norm.
        final_layernorm=_build_norm_spec(config=config, provider=provider),
    )
    return ModuleSpec(module=DeepseekV4TransformerBlock, submodules=block_submodules)


__all__ = [
    "get_deepseek_v4_runtime_decoder_spec",
]
