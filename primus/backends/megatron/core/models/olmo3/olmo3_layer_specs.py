###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Layer spec for OLMo-3 (Transformer Engine linears / attention / norms)."""

from __future__ import annotations

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from primus.backends.megatron.core.models.olmo3.olmo3_attention import (
    Olmo3SelfAttention,
)
from primus.backends.megatron.core.models.olmo3.olmo3_layer import Olmo3TransformerLayer


def get_olmo3_layer_spec() -> ModuleSpec:
    """Post-norm layer: plain (non-layernorm-fused) TE linears; QK norm lives in Olmo3SelfAttention."""
    return ModuleSpec(
        module=Olmo3TransformerLayer,
        params={"post_norm": TENorm},
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=Olmo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=None,
                    k_layernorm=None,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
