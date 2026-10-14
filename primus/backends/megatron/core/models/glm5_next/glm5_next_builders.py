###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 (``glm5_next``) builder + ``model_provider`` (DeepSeek-V4 layout)."""

from typing import Optional

from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.glm5_next.glm5_next_layer_specs import (
    get_glm5_next_runtime_decoder_spec,
)
from primus.backends.megatron.core.models.glm5_next.glm5_next_model import Glm5NextModel
from primus.backends.megatron.core.models.glm5_next.glm5_next_transformer_config import (
    Glm5NextTransformerConfig,
)


def glm5_next_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[Glm5NextTransformerConfig] = None,
    pg_collection=None,
):
    print_rank_0("[Primus:GLM5-Next] building Glm5NextModel...")
    if config is None:
        config = core_transformer_config_from_args(args, config_class=Glm5NextTransformerConfig)
    assert not args.use_legacy_models, "GLM-5.3 requires use_legacy_models=False (Mcore-only)."

    if args.spec is not None:
        decoder_spec = import_module(args.spec)
    else:
        decoder_spec = get_glm5_next_runtime_decoder_spec(config=config, vp_stage=vp_stage)

    return Glm5NextModel(
        config=config,
        transformer_layer_spec=decoder_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[Glm5NextTransformerConfig] = None,
    pg_collection=None,
):
    if model_builder is None:
        model_builder = glm5_next_builder
    args = get_args()
    if args.record_memory_history:
        import torch

        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
        )
    return model_builder(
        args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
