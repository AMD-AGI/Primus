###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark hooks for the shared builder (``MODEL_FAMILIES["muse_spark"]`` in ``registry.py``)."""

from functools import partial

from primus.backends.megatron.core.models.builder import (
    build_model,
    family_model_provider,
)

__all__ = ["muse_spark_builder", "model_provider", "check_builder_args", "muse_spark_model_kwargs"]


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
    if getattr(args, "rotary_seq_len_interpolation_factor", None) not in (None, 1):
        problems.append(
            f"rotary_seq_len_interpolation_factor={args.rotary_seq_len_interpolation_factor!r} "
            "(Muse Spark has no RoPE scaling)"
        )
    if problems:
        raise ValueError("Muse Spark does not support: " + "; ".join(problems))


def muse_spark_model_kwargs(args) -> dict:
    return {
        "position_embedding_type": args.position_embedding_type,
        "rotary_percent": args.rotary_percent,
        "rotary_base": args.rotary_base,
    }


muse_spark_builder = partial(build_model, "muse_spark")
model_provider = family_model_provider(muse_spark_builder)
