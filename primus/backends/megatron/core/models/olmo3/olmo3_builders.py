###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 (``olmo3``) builder + ``model_provider``: an mcore ``GPTModel`` with OLMo-3 layers."""

from typing import Optional

from megatron.core.models.gpt import GPTModel
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.olmo3.olmo3_layer_specs import get_olmo3_layer_spec
from primus.backends.megatron.core.models.olmo3.olmo3_transformer_config import (
    Olmo3TransformerConfig,
)


def olmo3_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[Olmo3TransformerConfig] = None,
    pg_collection=None,
):
    print_rank_0("[Primus:OLMo-3] building GPTModel with OLMo-3 layers...")
    if config is None:
        config = core_transformer_config_from_args(args, config_class=Olmo3TransformerConfig)
    assert isinstance(config, Olmo3TransformerConfig), type(config)
    assert not args.use_legacy_models, "OLMo-3 requires use_legacy_models=False (Mcore-only)."
    # YaRN: args keep `rope` (Megatron's CLI has no `yarn` choice for non-MLA models); the config's
    # position_embedding_type ('yarn' when the yarn_* fields enable it) is what GPTModel uses.
    position_embedding_type = config.position_embedding_type
    return GPTModel(
        config=config,
        transformer_layer_spec=get_olmo3_layer_spec(),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=False,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[Olmo3TransformerConfig] = None,
    pg_collection=None,
):
    if model_builder is None:
        model_builder = olmo3_builder
    args = get_args()
    if args.record_memory_history:
        import torch

        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
        )
    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)
