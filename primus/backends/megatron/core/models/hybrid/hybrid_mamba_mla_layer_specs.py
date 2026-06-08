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
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)

from primus.backends.megatron.core.models.hybrid.gated_delta_net import (
    GatedDeltaNet,
    GatedDeltaNetSubmodules,
)
from primus.backends.megatron.core.models.hybrid.gated_delta_net_layer import (
    GatedDeltaNetLayer,
    GatedDeltaNetLayerSubmodules,
)
from primus.backends.megatron.core.models.hybrid.hybrid_block import (
    HybridStack,
    HybridStackSubmodules,
)
from primus.backends.megatron.core.models.hybrid.kimi_delta_attention import (
    KimiDeltaAttention,
    KimiDeltaAttentionSubmodules,
)
from primus.backends.megatron.core.models.hybrid.kimi_delta_attention_layer import (
    KimiDeltaAttentionLayer,
    KimiDeltaAttentionLayerSubmodules,
)
from primus.backends.megatron.core.models.hybrid.mamba_layer_adapter import (
    Mamba2HybridLayer,
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

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from primus.backends.megatron.core.transformer.fla_flash_attention import (
    FLAFlashAttention,
)
from primus.backends.megatron.core.transformer.fla_flash_attention import (
    is_enabled as _fla_mla_attn_enabled,
)

# Route MLA's `core_attention` through a direct `flash_attn_func` call
# instead of TransformerEngine's `TEDotProductAttention` whenever the
# installed flash-attn version is newer than TE supports (>2.8.1) — that's
# the case where TE silently drops to its Composable-Kernel backend and
# loses ~30 ms/MLA-block on MI300X.  Auto-enabled by default; override
# with `fla_mla_attn: "0"` in YAML (or `PRIMUS_FLA_MLA_ATTN=0`) to force the TE path.
_MLA_CORE_ATTENTION = FLAFlashAttention if _fla_mla_attn_enabled() else TEDotProductAttention


# Module-load diagnostic: drop a per-rank marker file so we can verify
# unambiguously which copy of this spec was actually imported by training.
# This sidesteps Megatron's stdout filtering and any later monkey-patching.
def _record_spec_import_marker() -> None:
    import os
    import sys
    import time

    try:
        rank = int(os.environ.get("RANK", "-1"))
        marker = f"/tmp/primus_hybrid_spec_imported.rank{rank}.txt"
        with open(marker, "w") as fh:
            fh.write(f"file        = {__file__}\n")
            fh.write(f"_MLA_CORE_ATTENTION = {_MLA_CORE_ATTENTION!r}\n")
            try:
                from megatron.training import get_args as _ga
                _mla_val = getattr(_ga(), 'fla_mla_attn', '')
            except Exception:
                _mla_val = '(args unavailable)'
            fh.write(f"args.fla_mla_attn   = {_mla_val!r}\n")
            fh.write(f"is_enabled()        = {_fla_mla_attn_enabled()}\n")
            fh.write(f"pid                 = {os.getpid()}\n")
            fh.write(f"ts                  = {time.time()}\n")
            fh.write("sys.path[:6]:\n")
            for p in sys.path[:6]:
                fh.write(f"  {p}\n")
    except Exception:
        pass


_record_spec_import_marker()

moe = get_moe_module_spec(
    use_te=True,
    num_experts=8,  # Can be any positive integer (must not be None).
    moe_grouped_gemm=True,
)

hybrid_stack_spec = ModuleSpec(
    module=MambaStack,
    submodules=MambaStackSubmodules(
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
                        core_attention=_MLA_CORE_ATTENTION,
                        linear_proj=TERowParallelLinear,
                        # FLA's MLA wraps every LoRA projection in
                        # `nn.Sequential(Linear → RMSNorm(fp32) → Linear)`; with
                        # IdentityOp we skip the intermediate norm and the model
                        # plateaus ~0.12 above FLA's loss curve from iter 100
                        # onwards (iter-1 still matches bit-perfect).  TENorm
                        # matches FLA's `RMSNorm(dtype=fp32)` exactly.
                        q_layernorm=TENorm,
                        kv_layernorm=TENorm,
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
                        core_attention=_MLA_CORE_ATTENTION,
                        linear_proj=TERowParallelLinear,
                        # FLA's MLA applies `RMSNorm(dtype=fp32)` between every
                        # LoRA down/up projection (see fla/layers/mla.py).
                        # IdentityOp skips it and the loss plateaus ~0.12 above
                        # FLA from iter 100 onwards.  TENorm = fp32 RMSNorm.
                        q_layernorm=TENorm,
                        kv_layernorm=TENorm,
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


gdn_hybrid_stack_spec_no_te = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        mamba_layer=ModuleSpec(
            module=GatedDeltaNetLayer,
            submodules=GatedDeltaNetLayerSubmodules(
                norm=WrappedTorchNorm,
                mixer=ModuleSpec(
                    module=GatedDeltaNet,
                    submodules=GatedDeltaNetSubmodules(
                        in_proj=ColumnParallelLinear,
                        out_norm=WrappedTorchNorm,
                        out_proj=RowParallelLinear,
                    ),
                ),
                gdn_bda=get_bias_dropout_add,
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
                        core_attention=_MLA_CORE_ATTENTION,
                        linear_proj=RowParallelLinear,
                        # FLA's MLA wraps every LoRA projection in
                        # `nn.Sequential(Linear → RMSNorm(fp32) → Linear)`
                        # (fla/layers/mla.py lines 99-112).  Skipping this norm
                        # (IdentityOp) leaves the loss curve plateaued ~0.12
                        # above FLA from iter 100 onward; iter-1 still matches
                        # bit-perfect because both models start from identical
                        # init and the missing norm only kicks in after the
                        # LoRA weights diverge from their init.  Using
                        # WrappedTorchNorm here makes the per-LoRA norm pick up
                        # FLA's Triton RMSNorm when `PRIMUS_FLA_NORM=1`.
                        q_layernorm=WrappedTorchNorm,
                        kv_layernorm=WrappedTorchNorm,
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
                    submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=moe,
    ),
)


mamba_hybrid_stack_spec_no_te = ModuleSpec(
    module=HybridStack,
    submodules=HybridStackSubmodules(
        # Mamba2 mixer wrapped in our `Mamba2HybridLayer` adapter (subclass of
        # upstream MambaLayer that accepts `residual_in_fp32` so it slots into
        # Primus's `HybridStack` builder — see mamba_layer_adapter.py).
        # No TE column-parallel-LayerNorm fusion here — matches the no-TE GDN
        # variant so the same `_MLA_CORE_ATTENTION` (FLA flash-attn or TE) path
        # is used end-to-end without TE-norm folding.
        mamba_layer=ModuleSpec(
            module=Mamba2HybridLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    params={
                        "expand": 2,
                        "d_conv": 4,
                    },
                    submodules=MambaMixerSubmodules(
                        in_proj=ColumnParallelLinear,
                        out_proj=RowParallelLinear,
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
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
                        core_attention=_MLA_CORE_ATTENTION,
                        linear_proj=RowParallelLinear,
                        # Same MLA LoRA-norm fix as gdn_hybrid_stack_spec_no_te:
                        # WrappedTorchNorm = RMSNorm(fp32) between every LoRA
                        # down/up projection (matches FLA's
                        # `nn.Sequential(Linear → RMSNorm(fp32) → Linear)`).
                        q_layernorm=WrappedTorchNorm,
                        kv_layernorm=WrappedTorchNorm,
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
                    submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=moe,
    ),
)


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
                        core_attention=_MLA_CORE_ATTENTION,
                        linear_proj=TERowParallelLinear,
                        # FLA's MLA applies `RMSNorm(dtype=fp32)` between LoRA
                        # down/up projections (fla/layers/mla.py).  IdentityOp
                        # here breaks training-dynamics parity even when the
                        # init checkpoint matches bit-perfect at iter 1.
                        q_layernorm=TENorm,
                        kv_layernorm=TENorm,
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
                        core_attention=_MLA_CORE_ATTENTION,
                        linear_proj=RowParallelLinear,
                        # FLA wraps every LoRA proj in
                        # `nn.Sequential(Linear → RMSNorm(fp32) → Linear)`
                        # (fla/layers/mla.py).  WrappedTorchNorm gives us the
                        # equivalent and, under PRIMUS_FLA_NORM=1, swaps to
                        # FLA's Triton `RMSNorm` for bit-exact parity.
                        q_layernorm=WrappedTorchNorm,
                        kv_layernorm=WrappedTorchNorm,
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
                    submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
        moe_layer=moe,
    ),
)
