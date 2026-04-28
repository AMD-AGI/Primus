###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus GPT Layer Specs

Primus implementation of Megatron GPT layer specs.
"""

import warnings
from typing import Optional

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.transformer.attention import (
    Lfm2ShortConv,
    Lfm2ShortConvSubmodules,
)


def get_gpt_decoder_layer_specs(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """GPT block spec."""
    assert use_transformer_engine, "use_transformer_engine must be True"

    from megatron.training import get_args

    args = get_args()

    lfm_layer_types = args.lfm_layer_types

    if lfm_layer_types is None:
        lfm_layer_types = ["full_attention"] * args.num_layers
    else:
        assert isinstance(
            lfm_layer_types, list
        ), f"lfm_layer_types must be list[str] or None, got {type(lfm_layer_types)}"
        invalid_type_values = [
            layer_type for layer_type in lfm_layer_types if not isinstance(layer_type, str)
        ]
        assert (
            not invalid_type_values
        ), f"lfm_layer_types must contain only str values, got invalid values: {invalid_type_values}"
        valid_layer_types = {"conv", "full_attention"}
        invalid_layer_types = sorted(
            {layer_type for layer_type in lfm_layer_types if layer_type not in valid_layer_types}
        )
        assert not invalid_layer_types, (
            f"Invalid lfm_layer_types values: {invalid_layer_types}. "
            f"Only {sorted(valid_layer_types)} are allowed."
        )

    assert len(lfm_layer_types) == args.num_layers, (
        f"Invalid lfm_layer_types length: {len(lfm_layer_types)}, "
        f"expected {args.num_layers} (args.num_layers)."
    )

    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    dense_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )
    moe_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )
    dense_lfm_layer_spec = get_lfm_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )
    moe_lfm_layer_spec = get_lfm_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
        use_te_activation_func=config.use_te_activation_func,
        use_kitchen_attention=config.use_kitchen_attention,
        kitchen_attention_backend=config.kitchen_attention_backend,
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}")

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        layer_type = lfm_layer_types[layer_number]
        # print(f"{layer_number=}, {layer_type=}, {moe_layer_pattern[layer_number]=}")
        if moe_layer_pattern[layer_number] == 1:
            if layer_type == "conv":
                layer_specs.append(moe_lfm_layer_spec)
            else:
                layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            if layer_type == "conv":
                layer_specs.append(dense_lfm_layer_spec)
            else:
                layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    return layer_specs


def get_lfm_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_kitchen: bool = False,
    use_te_activation_func: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MLA. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules

    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    assert not use_kitchen, "use_kitchen is not supported with LFM model."
    # Note: patch_gpt_decoder_layer_specs must be applied bafter te_spec_provider_patches
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    backend = TESpecProvider()

    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSubmodules,
    )

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
        use_te_activation_func=use_te_activation_func,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Lfm2ShortConv,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=Lfm2ShortConvSubmodules(
                    in_proj=backend.linear(),
                    out_proj=backend.linear(),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
