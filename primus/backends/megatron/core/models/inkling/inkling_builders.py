###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Inkling model builder + ``model_provider`` entry points.

Same shape as ``kimi_k3_builders.py``: ``model_provider`` is the thin wrapper
Megatron's ``pretrain()`` calls and ``inkling_builder`` does the
instantiation. :func:`build_inkling_model` is the args-free core, shared by
the builder, the converter's dry run and the tests; ``meta=True`` builds the
module tree on the meta device so a full-size model can be inspected
without memory.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Optional

import torch
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.inkling.inkling_layer_specs import (
    get_inkling_decoder_block_spec,
)
from primus.backends.megatron.core.models.inkling.inkling_model import InklingModel
from primus.backends.megatron.core.models.inkling.inkling_transformer_config import (
    InklingTransformerConfig,
)

__all__ = [
    "build_inkling_model",
    "inkling_builder",
    "inkling_config_from_model_yaml",
    "megatron_args_from_model_yaml",
    "model_provider",
]


def megatron_args_from_model_yaml(path: str, params_dtype=torch.bfloat16, **overrides):
    """Megatron's argument defaults overlaid with a merged model preset (plus ``overrides``).

    Lets the converter and the tests build exactly what a training run would,
    without launching. ``validate_args`` is not run, so the preset must spell
    out what it would derive (``kv_channels``, ``ffn_hidden_size``,
    ``num_query_groups``); the one derivation applied here is turning
    sequence parallelism off at TP=1.
    """
    import sys

    from megatron.training.arguments import parse_args

    from primus.core.config.yaml_loader import parse_yaml

    values = parse_yaml(path)
    values.update(overrides)
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        args = parse_args(ignore_unknown_args=True)
    finally:
        sys.argv = argv
    for key, value in values.items():
        setattr(args, key, value)
    args.params_dtype = params_dtype
    args.bf16 = params_dtype == torch.bfloat16
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False
    return args


def inkling_config_from_model_yaml(
    path: str, params_dtype=torch.bfloat16, **overrides
) -> InklingTransformerConfig:
    """The config :func:`inkling_builder` would build from a model preset."""
    args = megatron_args_from_model_yaml(path, params_dtype, **overrides)
    return core_transformer_config_from_args(args, config_class=InklingTransformerConfig)


def build_inkling_model(
    config: InklingTransformerConfig,
    *,
    vocab_size: int,
    max_sequence_length: int,
    pre_process: bool = True,
    post_process: bool = True,
    decoder_spec=None,
    vp_stage: Optional[int] = None,
    pg_collection=None,
    fp16_lm_cross_entropy: bool = False,
    meta: bool = False,
) -> InklingModel:
    if meta:
        config = copy.copy(config)
        config.init_model_with_meta_device = True
        config.use_cpu_initialization = False
        config.perform_initialization = False
    if decoder_spec is None:
        decoder_spec = get_inkling_decoder_block_spec(config, vp_stage=vp_stage)
    ctx = torch.device("meta") if meta else contextlib.nullcontext()
    with ctx:
        return InklingModel(
            config=config,
            decoder_spec=decoder_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=True,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )


def inkling_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[InklingTransformerConfig] = None,
    pg_collection=None,
):
    """Build an Inkling model from Megatron args."""
    print_rank_0("[Primus:Inkling] building InklingModel...")

    if config is None:
        config = core_transformer_config_from_args(args, config_class=InklingTransformerConfig)
    # core_transformer_config_from_args swaps in MLATransformerConfig when
    # args.multi_latent_attention is set, which would drop every Inkling field.
    assert isinstance(config, InklingTransformerConfig), (
        f"Expected an InklingTransformerConfig, got {type(config).__name__}; "
        "Inkling must keep multi_latent_attention: false"
    )
    assert not args.use_legacy_models, "Inkling requires use_legacy_models=False (Mcore-only)."
    assert args.untie_embeddings_and_output_weights, "Inkling's embedding and LM head are untied"
    assert not getattr(args, "reset_attention_mask", False), (
        "Inkling does not support reset_attention_mask: the ShortConvs would mix tokens across "
        "document boundaries that the attention mask hides (per-document masking is item O8)"
    )

    decoder_spec = import_module(args.spec) if args.spec is not None else None
    return build_inkling_model(
        config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        decoder_spec=decoder_spec,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
    )


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[InklingTransformerConfig] = None,
    pg_collection=None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``.

    ``get_model_provider`` binds :func:`inkling_builder` as the first arg via
    ``functools.partial``.
    """
    if model_builder is None:
        model_builder = inkling_builder
    args = get_args()
    return model_builder(
        args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
