# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TELinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from primus.backends.megatron.core.models.hybrid.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
from primus.backends.megatron.core.models.hybrid.gated_delta_net_layer import GatedDeltaNetLayer, GatedDeltaNetLayerSubmodules
from primus.backends.megatron.core.models.hybrid.kimi_delta_attention import KimiDeltaAttention, KimiDeltaAttentionSubmodules
from primus.backends.megatron.core.models.hybrid.kimi_delta_attention_layer import KimiDeltaAttentionLayer, KimiDeltaAttentionLayerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)

# Import HybridStack from relative path
from primus.backends.megatron.core.models.hybrid.hybrid_block import (
    HybridStack,
    HybridStackSubmodules,
)

# Inference layers may not be available in older Megatron versions
# They're only used in hybrid_inference_stack_spec, not the training spec
try:
    from megatron.core.tensor_parallel import (
        InferenceLayerNormColumnParallelLinear,
        InferenceRowParallelLinear,
    )

    HAS_INFERENCE_LAYERS = True
except ImportError:
    # Fallback to regular layers for inference spec
    InferenceLayerNormColumnParallelLinear = TELayerNormColumnParallelLinear
    InferenceRowParallelLinear = TERowParallelLinear
    HAS_INFERENCE_LAYERS = False

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

moe = get_moe_module_spec(
    use_te=True,
    num_experts=8,  # Can be any positive integer (must not be None).
    moe_grouped_gemm=True,
    moe_use_legacy_grouped_gemm=False,
)


hybrid_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    params={
                        "expand": 1,
                        "d_conv": 4,
                    },
                    submodules=MambaMixerSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TELinear,
                        linear_q_up_proj=TELayerNormColumnParallelLinear,
                        linear_kv_down_proj=TELinear,
                        linear_kv_up_proj=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=moe,
    ),
)


gdn_hybrid_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=GatedDeltaNetLayer,
            submodules=GatedDeltaNetLayerSubmodules(
                mixer=ModuleSpec(
                    module=GatedDeltaNet,
                    submodules=GatedDeltaNetSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_norm=TENorm, out_proj=TERowParallelLinear
                    ),
                ),
                gdn_bda=get_bias_dropout_add,
            ),
        ),
        attention_layer = ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TELinear,
                        linear_q_up_proj=TELayerNormColumnParallelLinear,
                        linear_kv_down_proj=TELinear,
                        linear_kv_up_proj=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=moe,
    ),
)

# KDA (Kimi Delta Attention) variant: uses fused Q/K/V in_proj with
# TELayerNormColumnParallelLinear (matching GDN), combined depthwise conv1d,
# and low-rank gate factorization. Requires fla-core (fla.ops.kda).
kda_hybrid_stack_spec = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=KimiDeltaAttentionLayer,
            submodules=KimiDeltaAttentionLayerSubmodules(
                mixer=ModuleSpec(
                    module=KimiDeltaAttention,
                    submodules=KimiDeltaAttentionSubmodules(
                        in_proj=TELayerNormColumnParallelLinear,
                        gate_norm=TENorm,
                        out_norm=TENorm,
                        out_proj=TERowParallelLinear,
                    ),
                ),
                kda_bda=get_bias_dropout_add,
            ),
        ),
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=TENorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=TEColumnParallelLinear,
                        linear_q_down_proj=TELinear,
                        linear_q_up_proj=TELayerNormColumnParallelLinear,
                        linear_kv_down_proj=TELinear,
                        linear_kv_up_proj=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=moe,
    ),
)


# No-TE KDA spec — mirrors gdn_hybrid_stack_spec_no_te. Replaces every TE
# wrapper with plain WrappedTorchNorm / ColumnParallelLinear / RowParallelLinear.
# On ROCm this removes TE's per-call dispatch indirection + dtype recasts that
# account for the bulk of Megatron's per-iter overhead vs FLA's HF-Trainer loop.
#
# Architectural match to fla/models/kda/modeling_kda.py KDABlock:
#   - Single pre-norm at the layer (KimiDeltaAttentionLayer.norm = WrappedTorchNorm)
#   - Mixer in_proj is plain ColumnParallelLinear (no fused norm-and-project)
#   - Mixer gate_norm = IdentityOp (FLA has no re-norm for the gate path; the
#     pre-norm-once-and-reuse pattern saves 1 norm launch per layer)
#   - Mixer out_norm stays as WrappedTorchNorm (becomes FusedRMSNormGated when
#     PRIMUS_FLA_NORM=1 + use_fla_triton_kda=true, see KimiDeltaAttention.__init__)
kda_hybrid_stack_spec_no_te = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=KimiDeltaAttentionLayer,
            submodules=KimiDeltaAttentionLayerSubmodules(
                norm=WrappedTorchNorm,
                mixer=ModuleSpec(
                    module=KimiDeltaAttention,
                    submodules=KimiDeltaAttentionSubmodules(
                        in_proj=ColumnParallelLinear,
                        gate_norm=IdentityOp,
                        out_norm=WrappedTorchNorm,
                        out_proj=RowParallelLinear,
                    ),
                ),
                kda_bda=get_bias_dropout_add,
            ),
        ),
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=WrappedTorchNorm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=ColumnParallelLinear,
                        linear_q_down_proj=ColumnParallelLinear,
                        linear_q_up_proj=ColumnParallelLinear,
                        linear_kv_down_proj=ColumnParallelLinear,
                        linear_kv_up_proj=ColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=RowParallelLinear,
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=WrappedTorchNorm,
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=moe,
    ),
)
