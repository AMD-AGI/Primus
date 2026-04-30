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

Phase 8 contract:
- Resolve V4 runtime decoder spec externally in builder.
- Pass runtime spec as ``transformer_layer_spec`` into ``DeepseekV4Model``.
- DeepseekV4Model is rooted at ``LanguageModule`` and has no GPT placeholder
  spec dependency.
"""

from typing import Optional

from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_layer_specs import (
    get_deepseek_v4_runtime_decoder_spec,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_model import (
    DeepseekV4Model,
)
from primus.backends.megatron.core.models.deepseek_v4.deepseek_v4_transformer_config import (
    DeepSeekV4TransformerConfig,
)


def _resolve_runtime_decoder_spec(
    args,
    config: DeepSeekV4TransformerConfig,
    vp_stage,
):
    """Resolve effective runtime decoder spec for DeepSeek-V4 decoder path."""
    if args.spec is not None:
        return import_module(args.spec)
    return get_deepseek_v4_runtime_decoder_spec(config=config, vp_stage=vp_stage)


def deepseek_v4_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[DeepSeekV4TransformerConfig] = None,
    pg_collection=None,
):
    """Build a DeepSeek-V4 model.

    Phase 8: build from a DeepSeek runtime spec tree only.
    """
    print_rank_0("[Primus:DeepSeek-V4] building DeepseekV4Model...")

    if config is None:
        config = core_transformer_config_from_args(
            args,
            config_class=DeepSeekV4TransformerConfig,
        )

    assert not args.use_legacy_models, "DeepSeek-V4 requires use_legacy_models=False (Mcore-only)."

    runtime_decoder_spec = _resolve_runtime_decoder_spec(args, config, vp_stage)

    model = DeepseekV4Model(
        config=config,
        transformer_layer_spec=runtime_decoder_spec,
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
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )
    return model


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[DeepSeekV4TransformerConfig] = None,
    pg_collection=None,
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

    return model_builder(
        args,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
