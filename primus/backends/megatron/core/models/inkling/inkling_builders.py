###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Model provider / builder for Inkling (``model_type: inkling``)."""

import dataclasses
from typing import Optional

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.inkling.inkling_model import InklingModel
from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)


def inkling_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0("[Primus:Inkling] building InklingModel...")
    if config is None:
        config = core_transformer_config_from_args(args, config_class=InklingTransformerConfig)
    if getattr(args, "multi_latent_attention", False):
        raise ValueError("Inkling has its own attention: multi_latent_attention must be false")
    if args.use_legacy_models:
        raise ValueError("Inkling requires use_legacy_models=False (Mcore-only)")
    if getattr(args, "reset_position_ids", False):
        # The global-layer log scaling (tau) uses the index in the row as the position.
        raise ValueError("Inkling does not support reset_position_ids (set reset_position_ids: false)")
    if not isinstance(config, InklingTransformerConfig):
        raise ValueError(f"Inkling needs an InklingTransformerConfig, got {type(config).__name__}")
    if (
        getattr(args, "mock_data", False)
        and getattr(args, "tokenizer_type", None) == "NullTokenizer"
        and config.inkling_ignored_label_id is None
        and config.inkling_unpadded_vocab_size is not None
        and int(args.vocab_size) >= config.inkling_unpadded_vocab_size
    ):
        # Mock data: the NullTokenizer EOD id (= vocab_size) is a label past the HF vocab that
        # eod_mask_loss does not mask (it masks EOD inputs). That one id scores 0; any other
        # label past the HF vocab still raises. A validated copy (dataclasses.replace reruns
        # __post_init__): the caller's config, and any other model built from it, is unchanged.
        config = dataclasses.replace(config, inkling_ignored_label_id=int(args.vocab_size))
        print_rank_0(
            f"[Primus:Inkling] mock data (NullTokenizer): label id {args.vocab_size} (the EOD id, "
            "past the HF vocab) scores 0 (inkling_ignored_label_id)"
        )
    return InklingModel(
        config=config,
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
    config=None,
    pg_collection=None,
):
    """``model_provider`` for Megatron's ``pretrain()`` (``get_model_provider`` binds the builder)."""
    if model_builder is None:
        model_builder = inkling_builder
    args = get_args()
    if args.record_memory_history:
        import torch

        torch.cuda.memory._record_memory_history(
            True, trace_alloc_max_entries=100000, trace_alloc_record_context=True
        )
    return model_builder(
        args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
