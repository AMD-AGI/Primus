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
from primus.backends.megatron.core.transformer.compressor import Compressor
from primus.backends.megatron.core.transformer.deepseek_v4_attention import (
    DeepseekV4Attention,
    DeepseekV4AttentionSubmodules,
)
from primus.backends.megatron.core.transformer.hyper_connection import (
    HyperHead,
    HyperMixer,
)
from primus.backends.megatron.core.transformer.indexer import Indexer
from primus.backends.megatron.core.transformer.moe.v4_hash_router import (
    DeepseekV4HashRouter,
)
from primus.backends.megatron.core.transformer.moe.v4_moe import (
    DeepseekV4MoE,
    DeepseekV4MoESubmodules,
)
from primus.backends.megatron.core.transformer.moe.v4_topk_router import (
    DeepseekV4LearnedRouter,
)

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
    """Default projection spec — a duplicated TE linear (no TP sharding).

    Used for projections that V4's grouped-low-rank O does not natively
    shard along TP (``linear_q_down_proj``, ``linear_kv``, ``linear_o_a``).
    Keep these duplicated for now; full TP sharding of the grouped O
    projection is tracked in P14.
    """
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


def _build_column_parallel_spec(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    in_features: int,
    out_features: int,
    gather_output: bool = True,
) -> ModuleSpec:
    """Column-parallel projection spec.

    With ``gather_output=True`` the output dim is gathered back to full
    width across TP ranks, so downstream attention math (which assumes
    ``H * head_dim`` per rank) does not need to know about TP at all.
    Memory of the projection's weight matrix is sharded across TP ranks
    even at ``gather_output=True``.

    Plan-2 P13 follow-up: this is used for ``linear_q_up_proj``. The
    gather-then-shard variant for full sharded heads is tracked in P14
    once the grouped-O TP plan lands.
    """
    return ModuleSpec(
        module=provider.column_parallel_linear(),
        params={
            "input_size": in_features,
            "output_size": out_features,
            "config": config,
            "init_method": config.init_method or _default_init_method,
            "gather_output": gather_output,
            "bias": False,
            "skip_bias_add": False,
            "skip_weight_param_allocation": False,
            "tp_comm_buffer_name": None,
            "is_expert": False,
        },
    )


def _build_row_parallel_spec(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    in_features: int,
    out_features: int,
    input_is_parallel: bool = False,
) -> ModuleSpec:
    """Row-parallel projection spec.

    With ``input_is_parallel=False`` the linear scatters the input across
    TP ranks internally and all-reduces the output, so the caller can
    pass a full-width input tensor and get a full-width output tensor.
    Weight memory is sharded across TP ranks. Used for ``linear_o_b``
    and the flat-O fallback ``linear_proj``.
    """
    return ModuleSpec(
        module=provider.row_parallel_linear(),
        params={
            "input_size": in_features,
            "output_size": out_features,
            "config": config,
            "init_method": config.init_method or _default_init_method,
            "input_is_parallel": input_is_parallel,
            "bias": False,
            "skip_bias_add": False,
            "tp_comm_buffer_name": None,
            "is_expert": False,
        },
    )


def _build_v4_attention_submodules(
    *,
    config: DeepSeekV4TransformerConfig,
    provider: DeepSeekV4SpecProvider,
    compress_ratio: int,
) -> DeepseekV4AttentionSubmodules:
    """V4-canonical submodules for :class:`DeepseekV4Attention`.

    Field names match the released V4-Flash checkpoint layout (and MLA's
    canonical names where they overlap):

    * ``linear_q_down_proj``  : ``hidden -> q_lora_rank``  (= ``wq_a``)
    * ``q_layernorm``        : RMSNorm(``q_lora_rank``)    (= ``q_norm``)
    * ``linear_q_up_proj``    : ``q_lora_rank -> n_heads * head_dim`` (= ``wq_b``)
      — built as **column-parallel** so the projection's weight is
      sharded across TP at ``tp > 1``. ``gather_output=True`` keeps
      downstream math TP-agnostic.
    * ``linear_kv``           : ``hidden -> head_dim``     (= ``wkv``,
      single-latent KV)
    * ``kv_layernorm``       : RMSNorm(``head_dim``)       (= ``kv_norm``)
    * ``linear_o_a``         : grouped low-rank O down-proj (duplicated;
      grouped-O TP plan is P14).
    * ``linear_o_b``         : grouped low-rank O up-proj  (-> ``hidden``)
      — built as **row-parallel** so its weight is sharded across TP.
    * ``linear_proj``        : flat-O fallback (``o_lora_rank == 0``,
      e.g. unit tests) — also row-parallel.
    * ``attn_sink``          : per-head learnable scalar
    * ``compressor``         : :class:`Compressor` (compressed branches)
    * ``indexer``            : :class:`Indexer` (CSA branch only)
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
        linear_q_up_proj=_build_column_parallel_spec(
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
        submods.linear_o_b = _build_row_parallel_spec(
            config=config,
            provider=provider,
            in_features=o_groups * o_lora_rank,
            out_features=hidden_size,
        )
    else:
        submods.linear_proj = _build_row_parallel_spec(
            config=config,
            provider=provider,
            in_features=q_out,
            out_features=hidden_size,
        )

    if compress_ratio > 0:
        submods.compressor = ModuleSpec(module=Compressor)
        if compress_ratio == 4:
            submods.indexer = ModuleSpec(module=Indexer)

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
    """Plan-2 P13 attention spec — single :class:`DeepseekV4Attention`
    class for all three V4 layer types (dense / HCA / CSA), dispatched
    inside the class on ``compress_ratio``."""
    return ModuleSpec(
        module=DeepseekV4Attention,
        params={"compress_ratio": int(compress_ratio)},
        submodules=_build_v4_attention_submodules(
            config=config,
            provider=provider,
            compress_ratio=int(compress_ratio),
        ),
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
        hash_router=ModuleSpec(module=DeepseekV4HashRouter),
        learned_router=ModuleSpec(module=DeepseekV4LearnedRouter),
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
        input_layernorm=_build_norm_spec(config=config, provider=provider),
        self_attention=_build_attention_spec(
            compress_ratio=compress_ratio,
            config=config,
            provider=provider,
        ),
        pre_mlp_layernorm=_build_norm_spec(config=config, provider=provider),
        mlp=_build_ffn_spec(
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
