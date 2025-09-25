###############################################################################
# The design of some components (e.g., ModuleSpec style) is inspired by
# NVIDIA's Megatron-LM project (https://github.com/NVIDIA/Megatron-LM).
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dataclasses import dataclass, field
from typing import Union

from primus.core.projection.meta_modules.base_meta_module import BaseMetaModule


@dataclass
class ModuleSpec:
    module: BaseMetaModule
    params: dict = field(default_factory=lambda: {})
    submodules: type = None


@dataclass
class TransformerLayerSubmodules:
    """
    Submodules that compose a full Transformer layer.

    Includes:
      - attention components (input LN, self-attention, attention bias-dropout-add)
      - MLP components (pre-MLP LN, MLP, MLP bias-dropout-add)
      - embedding (only for the first layer)
      - final layernorm (only for the last layer)
      - output layer (only for the last layer)
      - loss computation (only for the last layer)
    """

    # attention
    input_layernorm: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    self_attention: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    self_attn_bda: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    # mlp
    pre_mlp_layernorm: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    mlp: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    mlp_bda: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    # embedding (only for first layer)
    embedding: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    # final layernorm (only for last layer)
    final_layernorm: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    # output layer (only for last layer)
    output_layer: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    # loss (only for last layer)
    calc_loss: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule


@dataclass
class SelfAttentionSubmodules:
    """
    Submodules used inside the self-attention block.

    Includes:
      - QKV projection layers
      - optional layernorms for Q and K
      - rotary embeddings
      - core attention kernel
      - output projection
    """

    linear_qkv: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    q_layernorm: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    k_layernorm: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    rotary_emb: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    core_attention: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    linear_proj: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule


@dataclass
class MLPSubmodules:
    """
    Submodules used inside the MLP block.

    Includes:
      - dense MLP components (two linear layers)
      - sparse MoE MLP components (router, dispatch/combine, grouped GEMM, shared experts)
    """

    # dense MLP
    linear_fc1: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    linear_fc2: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    # sparse MoE MLP
    router: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    dispatcher_dispach: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    dispatcher_combine: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    grouped_gemm: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule
    shared_experts: Union[ModuleSpec, BaseMetaModule] = IdentityMetamodule


# Transformer Layer Data Flow
#
#             +----------------+
#             |     Input      |
#             +----------------+
#                     | ----------------------
#        +-------------------------+         |
#        |     Input LayerNorm     |         |
#        +-------------------------+         |
#                     |                      |
#        +------------------------+          |
#        |     Self-Attention     |          |
#        +------------------------+          |
#                     |                      |
#            +-----------------+             |
#            |     Dropout     |             |
#            +-----------------+             |
#                     |                      |
#                     o ---------------------|
#         +----------------------+
#         |     Residual Add     |
#         +----------------------+
#                     | ----------------------
#        +-------------------------+         |
#        |    Pre-mlp LayerNorm    |         |
#        +-------------------------+         |
#                     |                      |
#              +-------------+               |
#              |     MLP     |               |
#              +-------------+               |
#                     |                      |
#            +-----------------+             |
#            |     Dropout     |             |
#            +-----------------+             |
#                     |                      |
#                     o ---------------------|
#         +----------------------+
#         |     Residual Add     |
#         +----------------------+
#                     |
#             +----------------+
#             |     Output      |
#             +----------------+
