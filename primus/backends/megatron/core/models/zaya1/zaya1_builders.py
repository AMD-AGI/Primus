###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""ZAYA1 model builder and the ``model_provider`` Megatron's ``pretrain()`` calls."""

from typing import Optional

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.zaya1.zaya1_model import Zaya1Model
from primus.backends.megatron.core.models.zaya1.zaya1_transformer_config import (
    Zaya1TransformerConfig,
)

__all__ = ["zaya1_builder", "model_provider"]


def _assert_data_parallel_only(args) -> None:
    """CCA head-parallel TP, conv/value-shift CP, and expert-parallel dispatch are not implemented."""
    blocked = []
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_model_parallel_size",
    ):
        size = int(getattr(args, name, 1) or 1)
        if size != 1:
            blocked.append(f"{name}={size}")
    if getattr(args, "sequence_parallel", False):
        blocked.append("sequence_parallel=True")
    if blocked:
        raise NotImplementedError(
            "ZAYA1 pretraining currently runs with data parallel only "
            "(the base paper trained DP + ZeRO-1). Not implemented: " + ", ".join(blocked) + "."
        )


def zaya1_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[Zaya1TransformerConfig] = None,
    pg_collection=None,
):
    """Build a ZAYA1 model."""
    print_rank_0("[Primus:ZAYA1] building Zaya1Model...")
    _assert_data_parallel_only(args)

    if config is None:
        config = core_transformer_config_from_args(args, config_class=Zaya1TransformerConfig)

    # core_transformer_config_from_args replaces the requested class with
    # MLATransformerConfig when multi_latent_attention is true, which drops
    # every ZAYA1 field. Fail instead of training a different model.
    assert isinstance(config, Zaya1TransformerConfig), (
        f"Expected a Zaya1TransformerConfig, got {type(config).__name__}. "
        "Leave multi_latent_attention false."
    )
    assert not args.use_legacy_models, "ZAYA1 requires use_legacy_models=False."
    assert not getattr(args, "multi_latent_attention", False), (
        "ZAYA1 must leave multi_latent_attention false."
    )
    if args.position_embedding_type != "rope":
        raise ValueError(
            "ZAYA1 applies partial RoPE inside CCA. "
            f"position_embedding_type must be 'rope', got {args.position_embedding_type!r}."
        )

    return Zaya1Model(
        config=config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        scatter_embedding_sequence_parallel=False,
        pg_collection=pg_collection,
        vp_stage=vp_stage,
    )


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[Zaya1TransformerConfig] = None,
    pg_collection=None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``.

    ``get_model_provider`` binds :func:`zaya1_builder` as the first argument
    via ``functools.partial``.
    """
    if model_builder is None:
        model_builder = zaya1_builder

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
