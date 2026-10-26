###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark (``muse_spark``) builder + ``model_provider``."""

from typing import Optional

from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.muse_spark.muse_spark_layer_specs import (
    get_muse_spark_decoder_block_spec,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_model import (
    MuseSparkModel,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
    MuseSparkTransformerConfig,
)


def muse_spark_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[MuseSparkTransformerConfig] = None,
    pg_collection=None,
):
    print_rank_0("[Primus:MuseSpark] building MuseSparkModel...")
    if config is None:
        config = core_transformer_config_from_args(args, config_class=MuseSparkTransformerConfig)
    assert not args.use_legacy_models, "Muse Spark requires use_legacy_models=False (Mcore-only)."

    if args.spec is not None:
        decoder_spec = import_module(args.spec)
    else:
        decoder_spec = get_muse_spark_decoder_block_spec(config=config, vp_stage=vp_stage)

    return MuseSparkModel(
        config=config,
        transformer_layer_spec=decoder_spec,
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
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[MuseSparkTransformerConfig] = None,
    pg_collection=None,
):
    if model_builder is None:
        model_builder = muse_spark_builder
    args = get_args()
    return model_builder(
        args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
