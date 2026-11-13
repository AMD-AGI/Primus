###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark (``muse_spark``) builder + ``model_provider``."""

from typing import Optional

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


def check_builder_args(args):
    """Refuse the GPT builder options Muse Spark does not support, instead of ignoring them.

    * ``spec``: the Muse Spark layer spec is the model (gated residuals, sandwich norms,
      latent MoE with the per-expert norm); a custom spec would build stock layers with a
      Muse Spark config.
    * ``use_rope_scaling`` / ``rotary_seq_len_interpolation_factor``: the released model has
      no RoPE scaling (``rope_type`` default) and the Muse Spark rotary embedding applies
      none, so a set value would train with unscaled RoPE."""
    problems = []
    if getattr(args, "spec", None) is not None:
        problems.append(f"spec={args.spec!r} (the Muse Spark layer spec is always used)")
    if getattr(args, "use_rope_scaling", False):
        problems.append("use_rope_scaling=True (Muse Spark has no RoPE scaling)")
    if getattr(args, "rotary_seq_len_interpolation_factor", None) not in (None, 1, 1.0):
        problems.append(
            f"rotary_seq_len_interpolation_factor={args.rotary_seq_len_interpolation_factor!r} "
            "(Muse Spark has no RoPE scaling)"
        )
    if problems:
        raise ValueError("Muse Spark does not support: " + "; ".join(problems))


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
    # Megatron-LM core_v0.19 dropped the legacy models (and the arg): absent means Mcore
    assert not getattr(
        args, "use_legacy_models", False
    ), "Muse Spark requires use_legacy_models=False (Mcore-only)."

    check_builder_args(args)
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
