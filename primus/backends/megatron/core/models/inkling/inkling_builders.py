###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling model builder + ``model_provider`` (dispatched from
``primus/core/utils/import_utils.get_model_provider`` for ``model_type: inkling``)."""

from typing import Optional

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.inkling.inkling_model import InklingModel
from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)

__all__ = ["inkling_builder", "model_provider"]


def inkling_builder(args, pre_process, post_process, vp_stage=None,
                    config: Optional[InklingTransformerConfig] = None, pg_collection=None):
    """Build an Inkling model."""
    print_rank_0("[Primus:Inkling] building InklingModel...")
    if config is None:
        config = core_transformer_config_from_args(args, config_class=InklingTransformerConfig)
    assert isinstance(config, InklingTransformerConfig), (
        f"Expected an InklingTransformerConfig, got {type(config).__name__}; "
        "multi_latent_attention must stay false for Inkling.")
    assert not args.use_legacy_models, "Inkling requires use_legacy_models=False (Mcore-only)."
    return InklingModel(
        config=config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


def model_provider(model_builder=None, pre_process: bool = True, post_process: bool = True,
                   vp_stage: Optional[int] = None, config: Optional[InklingTransformerConfig] = None,
                   pg_collection=None):
    """``model_provider`` entry point used by Megatron's ``pretrain()``."""
    if model_builder is None:
        model_builder = inkling_builder
    args = get_args()
    return model_builder(args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)
