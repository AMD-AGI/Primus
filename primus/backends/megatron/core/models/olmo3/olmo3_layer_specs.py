###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 layer spec (Transformer Engine).

Post-norm layout: no input / pre-MLP norm (``IdentityOp``); the attention and
the MLP each normalise their own output before the residual add.
"""

from megatron.core.extensions import transformer_engine_spec_provider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from primus.backends.megatron.core.transformer.olmo3.olmo3_attention import (
    Olmo3SelfAttention,
)
from primus.backends.megatron.core.transformer.olmo3.olmo3_mlp import Olmo3MLP

__all__ = ["get_olmo3_layer_spec"]


def get_olmo3_layer_spec() -> ModuleSpec:
    """TE layer spec of OLMo-3 (fixed: per-layer window / RoPE choices live in the config and
    are applied by ``Olmo3SelfAttention``).

    The modules come from ``transformer_engine_spec_provider.TESpecProvider``, read at call time,
    as upstream's GPT spec does: the Primus Turbo patch rebinds it to ``PrimusTurboSpecProvider``,
    so ``use_turbo_gemm`` (Turbo linears) applies to OLMo-3's QKV / proj / fc1 / fc2 too. The
    q/k norms are ``FullWidthRMSNorm`` (owned by the attention); the post-norms are the
    provider's ``layer_norm()``.
    """
    backend = transformer_engine_spec_provider.TESpecProvider()
    norm = backend.layer_norm()
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=IdentityOp,
            self_attention=ModuleSpec(
                module=Olmo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal, "post_norm": norm},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=None,
                    k_layernorm=None,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=Olmo3MLP,
                params={"post_norm": norm},
                submodules=MLPSubmodules(
                    linear_fc1=backend.column_parallel_linear(),
                    linear_fc2=backend.row_parallel_linear(),
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )
