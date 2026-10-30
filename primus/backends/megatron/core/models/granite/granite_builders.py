###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""IBM Granite model builder and ``model_provider``.

Granite uses upstream ``GPTModel``. Against upstream ``gpt_builder``
(``third_party/Megatron-LM/gpt_builders.py``) the builder changes the config class
(``GraniteTransformerConfig``, which carries ``residual_multiplier`` and
``attention_multiplier``) and the layer module (``GraniteTransformerLayer``).

It covers only the dense Llama-layout Granite. Upstream options that select another
config source or layer layout are refused with an explicit error instead of being
ignored: ``yaml_cfg`` (``core_transformer_config_from_yaml`` cannot build the Granite
config class), ``experimental_attention_variant``, ``heterogeneous_layers_config_path``,
``transformer_impl='inference_optimized'``, MoE (``num_experts``), MTP
(``mtp_num_layers``), legacy models and ``apply_query_key_layer_scaling`` (with it the
stored softmax scale is not the applied one, so the read-back check below cannot work).

After the build, the softmax scale held by every built core attention (TE, local or
Primus Turbo) is read back and must equal ``attention_multiplier``.

``--spec`` accepts what upstream accepts (a ``TransformerLayer`` ``ModuleSpec`` or a
``TransformerBlockSubmodules``); each layer module becomes ``GraniteTransformerLayer``,
and a custom ``TransformerLayer`` subclass is refused (see ``granite_layer_spec_from``).
"""

from typing import Optional

from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.granite.granite_args import (
    reject_unsupported_args,
)
from primus.backends.megatron.core.models.granite.granite_checks import (
    check_built_granite_model,
)
from primus.backends.megatron.core.models.granite.granite_layer_specs import (
    get_granite_layer_spec,
    granite_layer_spec_from,
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
        # A user spec must still use the Granite layer, or residual_multiplier is lost.
        transformer_layer_spec = granite_layer_spec_from(import_module(args.spec))
    else:
        transformer_layer_spec = get_granite_layer_spec(args, config)

    # GraniteTransformerConfig.__post_init__ sets softmax_scale = attention_multiplier; the
    # value used by the built attention modules is checked after the build.
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
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
    check_built_granite_model(model, config)
    return model


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    """``model_provider`` used by Megatron ``pretrain()`` (bound to ``granite_builder``)."""
    if model_builder is None:
        model_builder = granite_builder
    args = get_args()
    return model_builder(
        args, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection
    )
