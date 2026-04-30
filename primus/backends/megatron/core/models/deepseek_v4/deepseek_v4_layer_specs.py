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

import torch.nn as nn
from megatron.core.transformer import TransformerConfig
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
    _RMSNorm,
)
from primus.backends.megatron.core.transformer.csa_attention import CSAAttention
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
    DeepseekV4AttentionSubmodules,
)
from primus.backends.megatron.core.transformer.hca_attention import HCAAttention
from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperHead,
    HyperMixer,
)
from primus.backends.megatron.core.transformer.moe.v4_moe import DeepseekV4MoE

logger = logging.getLogger(__name__)


def _default_init_method(_weight) -> None:
    return None


def _build_linear_projection_spec(
    *,
    config: TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    in_features: int,
    out_features: int,
) -> ModuleSpec:
    if not provider.use_provider_modules():
        return ModuleSpec(
            module=nn.Linear,
            params={
                "in_features": in_features,
                "out_features": out_features,
                "bias": False,
            },
        )

    return ModuleSpec(
        module=provider.linear(),
        params={
            "input_size": in_features,
            "output_size": out_features,
            "parallel_mode": "duplicated",
            "config": config,
            "init_method": getattr(config, "init_method", _default_init_method),
            "bias": False,
            "skip_bias_add": False,
            "skip_weight_param_allocation": False,
            "tp_comm_buffer_name": None,
            "is_expert": False,
        },
    )


def _build_attention_submodules(
    *,
    config: TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_lora_rank: Optional[int],
) -> DeepseekV4AttentionSubmodules:
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

    return DeepseekV4AttentionSubmodules(
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


def _build_norm_spec(
    *,
    config: TransformerConfig,
    provider: DeepSeekV4SpecProvider,
):
    del config
    norm_module = provider.v4_norm_module()
    if norm_module is None:
        return _RMSNorm
    return ModuleSpec(module=norm_module)


def _build_attention_spec(
    *,
    compress_ratio: int,
    config: TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    provider_mode: str,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    attn_sliding_window: int,
    attn_sink_enabled: bool,
    q_lora_rank: Optional[int],
    index_topk: int,
    index_head_dim: int,
    index_n_heads: int,
) -> ModuleSpec:
    attention_submodules = _build_attention_submodules(
        config=config,
        provider=provider,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
    )

    common_params = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "attn_sliding_window": attn_sliding_window,
        "attn_sink_enabled": attn_sink_enabled,
        "q_lora_rank": q_lora_rank,
        "provider_mode": provider_mode,
    }

    if compress_ratio == 0:
        return ModuleSpec(
            module=DeepseekV4Attention,
            params={"compress_ratio": 0, **common_params},
            submodules=attention_submodules,
        )
    if compress_ratio == 4:
        return ModuleSpec(
            module=CSAAttention,
            params={
                "compress_ratio": compress_ratio,
                "index_topk": index_topk,
                "index_head_dim": index_head_dim,
                "index_n_heads": index_n_heads,
                "compressor_overlap": True,
                **common_params,
            },
            submodules=attention_submodules,
        )
    return ModuleSpec(
        module=HCAAttention,
        params={
            "compress_ratio": compress_ratio,
            "compressor_overlap": False,
            **common_params,
        },
        submodules=attention_submodules,
    )


def _build_ffn_spec(
    *,
    config: TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    provider_mode: str,
    hidden_size: int,
    ffn_hidden_size: int,
    layer_idx: int,
    num_routed_experts: int,
    moe_router_topk: int,
    moe_intermediate_size: int,
    num_shared_experts: int,
    num_hash_layers: int,
    hash_vocab_size: Optional[int],
    hash_seed: int,
    moe_score_function: str,
    moe_enable_expert_bias: bool,
    clamp_alpha: float,
) -> ModuleSpec:
    moe_use_grouped_gemm = bool(getattr(config, "moe_grouped_gemm", False))
    moe_use_legacy_grouped_gemm = bool(getattr(config, "moe_use_legacy_grouped_gemm", False))
    grouped_mlp_module, grouped_mlp_submodules = provider.v4_grouped_mlp_modules(
        moe_use_grouped_gemm=moe_use_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    if num_routed_experts > 0:
        return ModuleSpec(
            module=DeepseekV4MoE,
            params={
                "hidden_size": hidden_size,
                "moe_intermediate_size": moe_intermediate_size,
                "num_routed_experts": num_routed_experts,
                "moe_router_topk": moe_router_topk,
                "num_shared_experts": num_shared_experts,
                "layer_idx": layer_idx,
                "num_hash_layers": num_hash_layers,
                "hash_vocab_size": hash_vocab_size,
                "hash_seed": hash_seed,
                "score_function": moe_score_function,
                "enable_expert_bias": moe_enable_expert_bias,
                "clamp_alpha": clamp_alpha,
                "provider_mode": provider_mode,
                "moe_use_grouped_gemm": moe_use_grouped_gemm,
                "provider_grouped_mlp_module": grouped_mlp_module,
                "provider_grouped_mlp_submodules": grouped_mlp_submodules,
            },
        )
    return ModuleSpec(
        module=_DenseSwiGLUMLP,
        params={
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "provider_mode": provider_mode,
        },
    )


def _build_hybrid_layer_spec(
    config: TransformerConfig,
    *,
    provider: DeepSeekV4SpecProvider,
    provider_mode: str,
    layer_idx: int,
    compress_ratio: int,
) -> ModuleSpec:
    hidden_size = config.hidden_size
    ffn_hidden_size = getattr(config, "ffn_hidden_size", None) or 4 * hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_query_groups", None) or num_heads
    head_dim = getattr(config, "kv_channels", None) or hidden_size // num_heads
    rotary_dim = getattr(config, "qk_pos_emb_head_dim", 64)
    norm_eps = getattr(config, "norm_epsilon", 1.0e-6)

    hc_mult = int(getattr(config, "hc_mult", 1) or 1)
    hc_eps = float(getattr(config, "hc_eps", 1.0e-6) or 1.0e-6)
    hc_sinkhorn_iters = int(getattr(config, "hc_sinkhorn_iters", 20) or 20)

    attn_sliding_window = int(getattr(config, "attn_sliding_window", 128) or 0)
    attn_sink_enabled = bool(getattr(config, "attn_sink", False))
    q_lora_rank = getattr(config, "q_lora_rank", None) or None
    index_topk = int(getattr(config, "index_topk", 512) or 512)
    index_head_dim = int(getattr(config, "index_head_dim", 128) or 128)
    index_n_heads = int(getattr(config, "index_n_heads", 64) or 64)

    num_routed_experts = int(getattr(config, "num_moe_experts", 0) or 0)
    moe_router_topk = int(getattr(config, "moe_router_topk", 1) or 1)
    moe_intermediate_size = int(
        getattr(config, "moe_ffn_hidden_size", None)
        or getattr(config, "moe_intermediate_size", None)
        or ffn_hidden_size
    )
    num_shared_experts = int(getattr(config, "moe_shared_expert_intermediate_size", 0) > 0) or int(
        getattr(config, "num_shared_experts", 1)
    )
    num_hash_layers = int(getattr(config, "num_hash_layers", 0) or 0)
    hash_vocab_size = getattr(config, "padded_vocab_size", None) or getattr(config, "vocab_size", None)
    hash_seed = int(getattr(config, "hash_routing_seed", 0) or 0)
    moe_score_function = str(getattr(config, "moe_router_score_function", "sqrtsoftplus"))
    moe_enable_expert_bias = bool(getattr(config, "moe_router_enable_expert_bias", True))
    clamp_alpha = float(getattr(config, "swiglu_limit", 7.0) or 7.0)

    layer_submodules = DeepseekV4HybridLayerSubmodules(
        attn_norm=_build_norm_spec(config=config, provider=provider),
        attention=_build_attention_spec(
            compress_ratio=compress_ratio,
            config=config,
            provider=provider,
            provider_mode=provider_mode,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            attn_sliding_window=attn_sliding_window,
            attn_sink_enabled=attn_sink_enabled,
            q_lora_rank=q_lora_rank,
            index_topk=index_topk,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
        ),
        ffn_norm=_build_norm_spec(config=config, provider=provider),
        ffn=_build_ffn_spec(
            config=config,
            provider=provider,
            provider_mode=provider_mode,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            layer_idx=layer_idx,
            num_routed_experts=num_routed_experts,
            moe_router_topk=moe_router_topk,
            moe_intermediate_size=moe_intermediate_size,
            num_shared_experts=num_shared_experts,
            num_hash_layers=num_hash_layers,
            hash_vocab_size=hash_vocab_size,
            hash_seed=hash_seed,
            moe_score_function=moe_score_function,
            moe_enable_expert_bias=moe_enable_expert_bias,
            clamp_alpha=clamp_alpha,
        ),
        attn_hc=ModuleSpec(module=HyperMixer) if hc_mult > 1 else None,
        ffn_hc=ModuleSpec(module=HyperMixer) if hc_mult > 1 else None,
    )

    return ModuleSpec(
        module=DeepseekV4HybridLayer,
        params={
            "layer_idx": layer_idx,
            "compress_ratio": int(compress_ratio),
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
            "attn_sliding_window": attn_sliding_window,
            "attn_sink_enabled": attn_sink_enabled,
            "q_lora_rank": q_lora_rank,
            "index_topk": index_topk,
            "index_head_dim": index_head_dim,
            "index_n_heads": index_n_heads,
            "hc_mult": hc_mult,
            "hc_eps": hc_eps,
            "hc_sinkhorn_iters": hc_sinkhorn_iters,
            "norm_eps": norm_eps,
            "num_routed_experts": num_routed_experts,
            "moe_router_topk": moe_router_topk,
            "moe_intermediate_size": moe_intermediate_size,
            "num_shared_experts": num_shared_experts,
            "num_hash_layers": num_hash_layers,
            "hash_vocab_size": hash_vocab_size,
            "hash_seed": hash_seed,
            "moe_score_function": moe_score_function,
            "moe_enable_expert_bias": moe_enable_expert_bias,
            "clamp_alpha": clamp_alpha,
            "provider_mode": provider_mode,
        },
        submodules=layer_submodules,
    )


def _build_local_hybrid_layer_specs(
    config: TransformerConfig,
    *,
    provider: DeepSeekV4SpecProvider,
    provider_mode: str,
    vp_stage: Optional[int],
) -> List[ModuleSpec]:
    num_layers = int(config.num_layers)
    mtp_num_layers = int(getattr(config, "mtp_num_layers", 0) or 0)
    compress_ratios = _normalize_compress_ratios(
        getattr(config, "compress_ratios", None),
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
            provider_mode=provider_mode,
            layer_idx=layer_idx,
            compress_ratio=int(compress_ratios[layer_idx]),
        )
        for layer_idx in local_layer_indices
    ]


def get_deepseek_v4_runtime_decoder_spec(
    config: TransformerConfig,
    *,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> ModuleSpec:
    """Return the effective V4 runtime decoder spec tree."""
    del pp_rank

    provider = DeepSeekV4SpecProvider(config=config)
    provider_mode = provider.runtime_mode()
    logger.info("[DeepSeek-V4] resolve spec provider mode=%s", provider_mode)

    hc_mult = int(getattr(config, "hc_mult", 1) or 1)
    local_layer_specs = _build_local_hybrid_layer_specs(
        config,
        provider=provider,
        provider_mode=provider_mode,
        vp_stage=vp_stage,
    )
    block_submodules = DeepseekV4TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        hyper_head=ModuleSpec(module=HyperHead) if hc_mult > 1 else None,
        # DeepseekV4TransformerBlock decides whether this stage owns final norm.
        final_layernorm=_build_norm_spec(config=config, provider=provider),
    )
    return ModuleSpec(
        module=DeepseekV4TransformerBlock,
        params={"provider_mode": provider_mode},
        submodules=block_submodules,
    )


__all__ = [
    "get_deepseek_v4_runtime_decoder_spec",
]
