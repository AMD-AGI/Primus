###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 spec entry points.

This module only defines DeepSeek-native runtime specs.
"""

from typing import List, Optional

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

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
)
from primus.backends.megatron.core.transformer.hca_attention import HCAAttention
from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperHead,
    HyperMixer,
)
from primus.backends.megatron.core.transformer.moe.v4_moe import DeepseekV4MoE


def _build_attention_spec(
    *,
    compress_ratio: int,
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
    common_params = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "attn_sliding_window": attn_sliding_window,
        "attn_sink_enabled": attn_sink_enabled,
        "q_lora_rank": q_lora_rank,
    }

    if compress_ratio == 0:
        return ModuleSpec(
            module=DeepseekV4Attention,
            params={"compress_ratio": 0, **common_params},
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
        )
    return ModuleSpec(
        module=HCAAttention,
        params={
            "compress_ratio": compress_ratio,
            "compressor_overlap": False,
            **common_params,
        },
    )


def _build_ffn_spec(
    *,
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
            },
        )
    return ModuleSpec(
        module=_DenseSwiGLUMLP,
        params={
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
        },
    )


def _build_hybrid_layer_spec(
    config: TransformerConfig,
    *,
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
        attn_norm=_RMSNorm,
        attention=_build_attention_spec(
            compress_ratio=compress_ratio,
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
        ffn_norm=_RMSNorm,
        ffn=_build_ffn_spec(
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
        },
        submodules=layer_submodules,
    )


def _build_local_hybrid_layer_specs(
    config: TransformerConfig,
    *,
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

    hc_mult = int(getattr(config, "hc_mult", 1) or 1)
    local_layer_specs = _build_local_hybrid_layer_specs(config, vp_stage=vp_stage)
    block_submodules = DeepseekV4TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        hyper_head=ModuleSpec(module=HyperHead) if hc_mult > 1 else None,
        # DeepseekV4TransformerBlock decides whether this stage owns final norm.
        final_layernorm=_RMSNorm,
    )
    return ModuleSpec(
        module=DeepseekV4TransformerBlock,
        submodules=block_submodules,
    )


__all__ = [
    "get_deepseek_v4_runtime_decoder_spec",
]
