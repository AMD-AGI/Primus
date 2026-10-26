###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared builder and ``model_provider`` for the Primus-owned families in ``registry.py``."""

from typing import Optional

from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.registry import MODEL_FAMILIES

__all__ = ["build_model", "model_provider"]


def build_model(
    model_type: str,
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    """Build the model ``MODEL_FAMILIES[model_type]`` declares, with Megatron's builder signature."""
    family = MODEL_FAMILIES[model_type]
    assert (
        family.config_class and family.model_class
    ), f"model_type {model_type!r} is built by {family.builder or 'nothing'}, not by the Primus builder."
    config_class = family.resolve(family.config_class)
    model_class = family.resolve(family.model_class)
    print_rank_0(f"[Primus:{model_type}] building {model_class.__name__}...")

    if family.prepare_args:
        family.resolve(family.prepare_args)(args)

    if config is None:
        config = core_transformer_config_from_args(args, config_class=config_class)
    assert isinstance(config, config_class), (
        f"Expected a {config_class.__name__}, got {type(config).__name__}. "
        "core_transformer_config_from_args swaps the config class when multi_latent_attention "
        f"or heterogeneous_layers_config_path is set; model_type {model_type!r} must leave both unset."
    )
    assert not args.use_legacy_models, f"model_type {model_type!r} requires use_legacy_models=False."

    kwargs = dict(
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
    if family.decoder_spec:
        if args.spec is not None:
            kwargs["transformer_layer_spec"] = import_module(args.spec)
        else:
            decoder_spec = family.resolve(family.decoder_spec)
            kwargs["transformer_layer_spec"] = decoder_spec(config, vp_stage=vp_stage)
    if family.model_kwargs:
        kwargs.update(family.resolve(family.model_kwargs)(args))

    model = model_class(**kwargs)
    if family.check_model:
        family.resolve(family.check_model)(args, config, model)
    return model


def model_provider(
    model_builder,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    """``model_provider`` Megatron's ``pretrain()`` calls; ``get_model_provider`` binds ``model_builder``."""
    args = get_args()
    if args.record_memory_history:
        import torch

        torch.cuda.memory._record_memory_history(
            True,
            trace_alloc_max_entries=100000,
            trace_alloc_record_context=True,
        )

    return model_builder(
        args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
