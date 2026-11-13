###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4 layer specs, built for this pipeline stage's layers only.

Every layer is a Megatron ``TransformerLayer``: RMSNorm, ``MistralLarge4Attention`` (an
``MLASelfAttention``), RMSNorm, then a dense SwiGLU ``MLP`` on the first
``first_k_dense_replace`` layers and Megatron's ``MoELayer`` (TopKRouter, routed experts, one
shared expert) on the others. With ``transformer_impl: transformer_engine`` (training) the norms
and linears are TE's and the routed experts are grouped through Primus-Turbo or TE; with
``local`` (CPU tests, the converter's checks) the norms, MLP and experts are Megatron's local
modules and the core attention is ``EagerCoreAttention``.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
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

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    turbo_enabled,
)
from primus.backends.megatron.core.models.mistral_large4.mistral_large4_transformer_config import (
    moe_layer_freq_list,
)
from primus.backends.megatron.core.transformer.mistral_large4.attention import (
    EagerCoreAttention,
    MistralLarge4Attention,
    MistralLarge4Indexer,
)

__all__ = ["get_mistral_large4_decoder_block_spec", "get_mistral_large4_layer_spec", "moe_layer_pattern"]


def moe_layer_pattern(config):
    """1 = MoE, 0 = dense, per layer (``moe_layer_freq_list``)."""
    return moe_layer_freq_list(config.first_k_dense_replace, config.num_layers, config.num_moe_experts)


# the Primus args a PrimusTurboSpecProvider reads to pick a Turbo module over TE's
TURBO_MODULE_SWITCHES = (
    "use_turbo_gemm",
    "use_turbo_norm_te_linear",
    "use_turbo_attention",
    "use_turbo_grouped_gemm",
)


def _backend(config):
    """TE (training): ``PrimusTurboSpecProvider``, which picks each Turbo module by its
    ``use_turbo_*`` switch. Without Primus-Turbo (``enable_primus_turbo`` off, or not installed)
    the provider sees those switches off, so every module is TE's whatever ``use_turbo_*`` says."""
    if config.transformer_impl == "transformer_engine":
        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            PrimusTurboSpecProvider,
        )

        provider = PrimusTurboSpecProvider()
        if not turbo_enabled(provider.cfg):
            off = dict(vars(provider.cfg))
            off.update({k: False for k in TURBO_MODULE_SWITCHES})
            provider.cfg = SimpleNamespace(**off)
        return provider
    return LocalSpecProvider()


def get_mistral_large4_layer_spec(config, is_moe: bool, backend=None) -> ModuleSpec:
    """A decoder layer's spec: MoE or dense MLP by ``is_moe`` (from ``moe_layer_pattern(config)``);
    the per-layer attention variants (NoPE, KV sharing, index sharing) are built from the layer
    number by ``MistralLarge4Attention`` itself."""
    backend = backend or _backend(config)
    norm = backend.layer_norm(rms_norm=True, for_qk=False)
    kv_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    if is_moe:
        mlp = get_mlp_module_spec_for_backend(
            backend=backend, num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
        )
    else:
        # dense SwiGLU with an explicit pre-MLP norm (not fused into fc1), like every other layer
        mlp = partial(
            MLP.as_mlp_submodule,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(), linear_fc2=backend.row_parallel_linear()
            ),
        )
    local = not hasattr(backend, "linear")
    # TE: a duplicated (non-TP) TELinear for MLA's kv_a; local backend: ColumnParallelLinear
    kv_down = backend.column_parallel_linear() if local else backend.linear()
    attention = ModuleSpec(
        module=MistralLarge4Attention,
        params={
            "attn_mask_type": AttnMaskType.causal,
            "kv_layernorm": kv_norm,
            "indexer": ModuleSpec(module=MistralLarge4Indexer),
        },
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_kv_down_proj=kv_down,
            linear_kv_up_proj=backend.column_parallel_linear(),
            core_attention=EagerCoreAttention if local else backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            kv_layernorm=IdentityOp,
        ),
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=norm,
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=norm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_mistral_large4_decoder_block_spec(
    config, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):
    backend = _backend(config)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    count = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    pattern = moe_layer_pattern(config)
    specs = [
        get_mistral_large4_layer_spec(config, is_moe=pattern[i] == 1, backend=backend)
        for i in range(offset, offset + count)
    ]
    return TransformerBlockSubmodules(
        layer_specs=specs, layer_norm=backend.layer_norm(rms_norm=True, for_qk=False)
    )
