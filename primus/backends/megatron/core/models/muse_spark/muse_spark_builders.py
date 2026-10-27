###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark model builder and the ``model_provider`` Megatron's ``pretrain()`` calls."""

import os
from typing import Optional

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from primus.backends.megatron.core.models.muse_spark.muse_spark_layer_specs import (
    get_muse_spark_decoder_block_spec,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_model import (
    MuseSparkGPTModel,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
    MuseSparkTransformerConfig,
)

__all__ = ["muse_spark_builder", "model_provider"]


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


def _assert_supported_args(args, config: MuseSparkTransformerConfig) -> None:
    """Reject options that bypass or replace a piece of the Muse Spark forward."""
    blocked = []
    if args.use_legacy_models:
        blocked.append("use_legacy_models=True")
    if args.transformer_impl != "transformer_engine":
        blocked.append(f"transformer_impl={args.transformer_impl!r} (needs 'transformer_engine')")
    if args.position_embedding_type != "rope":
        blocked.append(f"position_embedding_type={args.position_embedding_type!r} (needs 'rope')")
    if getattr(args, "spec", None) is not None:
        blocked.append("spec (the Muse Spark spec is fixed)")
    if getattr(args, "experimental_attention_variant", None) is not None:
        blocked.append("experimental_attention_variant")
    if args.mtp_num_layers:
        blocked.append("mtp_num_layers (Muse Spark has no MTP head)")
    if getattr(args, "use_turbo_mega_moe", False):
        blocked.append("use_turbo_mega_moe=True (replaces MoELayer)")
    if getattr(args, "use_turbo_fused_qk_rmsnorm_rope", False):
        blocked.append("use_turbo_fused_qk_rmsnorm_rope=True (replaces SelfAttention.forward)")
    if getattr(args, "use_deprecated_20241209_moe_layer", False):
        blocked.append("use_deprecated_20241209_moe_layer=True")
    if getattr(args, "moe_use_legacy_grouped_gemm", False):
        blocked.append("moe_use_legacy_grouped_gemm=True (the legacy GroupedMLP ignores moe_latent_size)")
    if _env_truthy("PRIMUS_FUSED_RESIDUAL_NORM") or _env_truthy("PRIMUS_FUSED_RESIDUAL_NORM_V2"):
        blocked.append("PRIMUS_FUSED_RESIDUAL_NORM(_V2) (skips the gated residual)")
    if getattr(args, "enable_fused_linear_ce", False) or os.environ.get("FUSED_LINEAR_CE", "0") == "1":
        blocked.append("fused linear cross entropy (skips the logit multiplier and softcap)")
    if (
        config.moe_router_enable_expert_bias
        and getattr(args, "enable_primus_turbo", False)
        and getattr(args, "moe_use_fused_router_with_aux_score", False)
    ):
        blocked.append(
            "moe_use_fused_router_with_aux_score=True with expert bias (the fused router "
            "ignores expert_bias; it is also forced on by turbo_sync_free_moe_stage >= 1)"
        )
    if blocked:
        raise ValueError("Muse Spark does not support: " + "; ".join(blocked) + ".")


def muse_spark_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config: Optional[MuseSparkTransformerConfig] = None,
    pg_collection=None,
):
    """Build a Muse Spark text model on top of ``GPTModel``."""
    print_rank_0("[Primus:MuseSpark] building MuseSparkGPTModel...")

    if config is None:
        config = core_transformer_config_from_args(args, config_class=MuseSparkTransformerConfig)

    # core_transformer_config_from_args replaces the requested class with
    # MLATransformerConfig when multi_latent_attention is true, which drops
    # every Muse Spark field.
    assert isinstance(config, MuseSparkTransformerConfig), (
        f"Expected a MuseSparkTransformerConfig, got {type(config).__name__}. "
        "Leave multi_latent_attention false."
    )
    _assert_supported_args(args, config)

    return MuseSparkGPTModel(
        config=config,
        transformer_layer_spec=get_muse_spark_decoder_block_spec(config, vp_stage=vp_stage),
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


def model_provider(
    model_builder=None,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional[MuseSparkTransformerConfig] = None,
    pg_collection=None,
):
    """``model_provider`` entry point used by Megatron's ``pretrain()``.

    ``get_model_provider`` binds :func:`muse_spark_builder` as the first argument
    via ``functools.partial``.
    """
    if model_builder is None:
        model_builder = muse_spark_builder

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
