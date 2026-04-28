###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 model builder + model_provider entry points.

In the upstream Megatron-LM convention there are two pieces:

* ``model_provider``: a thin wrapper that handles ``args.record_memory_history``,
  ModelOpt etc. and delegates to a ``model_builder`` callable. Defined once
  in ``Megatron-LM/model_provider.py``.
* ``<modeltype>_builders.py``: contains ``<modeltype>_builder(args, ...)`` --
  the actual model-class instantiation logic.

For DeepSeek-V4 we keep both in a single primus-owned module so the dispatch
in ``primus/core/utils/import_utils.py`` doesn't have to chase symbols across
``third_party/Megatron-LM``.

Phase 2 contract (this file):
- ``deepseek_v4_builder`` builds a :class:`DeepseekV4Model` (a thin subclass
  of ``GPTModel``) using vanilla GPT layer specs. This is enough to exercise
  the end-to-end dispatch path.
- Phase 3 will swap the layer-spec helpers for V4-specific variants in
  ``deepseek_v4_layer_specs.py`` without changing this builder's signature.
"""

from typing import Optional

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_model import (
    DeepseekV4Model,
)


def _resolve_layer_spec(args, config, vp_stage):
    """Pick the transformer-layer spec for V4 (Phase 2 = GPT specs)."""
    if args.spec is not None:
        return import_module(args.spec)

    use_te = args.transformer_impl == "transformer_engine"

    if args.num_experts:
        return get_gpt_decoder_block_spec(
            config=config,
            use_transformer_engine=use_te,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            vp_stage=vp_stage,
        )

    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts,
            moe_grouped_gemm=args.moe_grouped_gemm,
            qk_layernorm=args.qk_layernorm,
            multi_latent_attention=getattr(config, "multi_latent_attention", False),
            experimental_attention_variant=getattr(config, "experimental_attention_variant", None),
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
            fallback_to_eager_attn=config.fallback_to_eager_attn,
        )

    return get_gpt_layer_local_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=getattr(config, "multi_latent_attention", False),
        experimental_attention_variant=getattr(config, "experimental_attention_variant", None),
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        normalization=args.normalization,
        qk_l2_norm=args.qk_l2_norm,
        use_kitchen=config.use_kitchen,
    )


def _resolve_mtp_block_spec(args, config, vp_stage):
    if args.mtp_num_layers is None:
        return None

    use_te = args.transformer_impl == "transformer_engine"
    if args.spec is not None:
        mtp_layer_spec = import_module(args.spec)
    else:
        mtp_layer_spec = _resolve_layer_spec(args, config, vp_stage)
    return get_gpt_mtp_block_spec(
        config=config,
        spec=mtp_layer_spec,
        use_transformer_engine=use_te,
        vp_stage=vp_stage,
    )


def deepseek_v4_builder(args, pre_process, post_process, vp_stage=None, config=None):
    """Build a DeepSeek-V4 model.

    Phase 2: behaves like ``gpt_builder`` but instantiates ``DeepseekV4Model``,
    so the dispatch path is exercised end-to-end. Phase 3 swaps the layer-spec
    helpers for V4-specific ones.
    """
    print_rank_0("[Primus:DeepSeek-V4] building DeepseekV4Model (Phase-2 stub)...")

    if config is None:
        config = core_transformer_config_from_args(args)

    assert not args.use_legacy_models, "DeepSeek-V4 requires use_legacy_models=False (Mcore-only)."

    transformer_layer_spec = _resolve_layer_spec(args, config, vp_stage)
    mtp_block_spec = _resolve_mtp_block_spec(args, config, vp_stage)

    model = DeepseekV4Model(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )
    return model


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``.

    ``MegatronPretrainTrainer`` will pass ``deepseek_v4_builder`` as the
    first arg via ``functools.partial`` so the upstream ``pretrain()`` can
    call this with the standard ``(pre_process, post_process, vp_stage)``
    signature.
    """
    if model_builder is None:
        model_builder = deepseek_v4_builder

    args = get_args()
    if args.record_memory_history:
        import torch

        torch.cuda.memory._record_memory_history(
            True,
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,
        )

    return model_builder(args, pre_process, post_process, vp_stage)
