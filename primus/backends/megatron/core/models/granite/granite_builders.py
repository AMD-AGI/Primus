###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""IBM Granite model builder.

Granite uses upstream ``GPTModel``, upstream ``TransformerLayer`` and upstream
``model_provider``. ``get_model_provider("granite")`` returns
``partial(model_provider, granite_builder)`` from this module (as for the other
Primus-owned models). ``model_provider`` (defined in ``granite_model_provider``, which
imports without Megatron) refuses ModelOpt, then calls upstream
``model_provider`` with this builder. Against
upstream ``gpt_builder`` (``third_party/Megatron-LM/gpt_builders.py``) the builder changes the config class
(``GraniteTransformerConfig``, which adds ``residual_multiplier``) and the layer spec's
bias-dropout-adds (scaled by ``residual_multiplier``, see ``granite_layer_specs``).

It covers only the dense Llama-layout Granite. Upstream options that select another
config source or builder, and the layout args the converter maps weights to (norm, biases,
QK norm, RoPE, attention variants), are refused with an explicit error instead of being
ignored (see ``granite_args.reject_unsupported_args``).

A user layer spec (``--spec``) is accepted as given: its bias-dropout-adds are scaled by
``residual_multiplier`` (``granite_layer_spec_from``), but the modules it names (norm class,
core attention, QK norm, biases) are not checked against the Granite layout. A spec is code
the user chose; the build logs a warning that the Granite layout checks do not cover it.
"""

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.granite.granite_args import (
    reject_unsupported_args,
)
from primus.backends.megatron.core.models.granite.granite_layer_specs import (
    built_model_summary,
    check_scaled_residual_layers,
    get_granite_layer_spec,
    granite_layer_spec_from,
)
from primus.backends.megatron.core.models.granite.granite_model_provider import (
    model_provider,
)
from primus.backends.megatron.core.models.granite.granite_transformer_config import (
    GraniteTransformerConfig,
)
from primus.core.utils.module_utils import log_rank_0

__all__ = ["granite_builder", "model_provider"]


def granite_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    """Build a Granite ``GPTModel``."""
    log_rank_0("[Primus:Granite] building Granite GPTModel ...")
    reject_unsupported_args(args)
    if config is None:
        config = core_transformer_config_from_args(args, config_class=GraniteTransformerConfig)
    if not isinstance(config, GraniteTransformerConfig):
        raise TypeError(
            f"Granite needs a GraniteTransformerConfig, got {type(config).__name__}; "
            "pass config=None or build it with config_class=GraniteTransformerConfig"
        )

    if args.spec is not None:
        # A user spec still gets the scaled bdas, or residual_multiplier is lost. Its modules
        # are not checked against the Granite layout (module docstring).
        log_rank_0(
            f"[Primus:Granite] WARNING: user layer spec {args.spec}: its modules (norm, core "
            "attention, QK norm, biases) are not checked against the Granite layout the "
            "converter maps weights to; only its bias-dropout-adds are scaled"
        )
        transformer_layer_spec = granite_layer_spec_from(import_module(args.spec), config.residual_multiplier)
    else:
        transformer_layer_spec = get_granite_layer_spec(args, config)

    # INFO level, so the line is in the console log at the default stderr_sink_level.
    log_rank_0(
        f"[Primus:Granite] multipliers: residual_multiplier={config.residual_multiplier} "
        f"softmax_scale={config.softmax_scale} mup_embedding_mult={config.mup_embedding_mult} "
        f"mup_output_mult={config.mup_output_mult} mup_width_mult={config.mup_width_mult}"
    )

    model = GPTModel(
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
        rope_scaling=False,  # reject_unsupported_args refuses use_rope_scaling
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
    check_scaled_residual_layers(model, config.residual_multiplier)
    log_rank_0(f"[Primus:Granite] built model: {built_model_summary(model)}")
    return model
