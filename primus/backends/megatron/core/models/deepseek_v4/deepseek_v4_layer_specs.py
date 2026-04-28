###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
DeepSeek-V4 layer specs (Phase 4 wrapper).

Why this file still exists, given that Phase 4's :class:`DeepseekV4TransformerBlock`
is standalone and **bypasses Megatron's ``ModuleSpec`` mechanism**:

* :class:`DeepseekV4Model` inherits from ``megatron.core.models.gpt.GPTModel``.
  ``GPTModel.__init__`` unconditionally constructs a stock ``TransformerBlock``
  from the ``transformer_layer_spec`` argument as ``self.decoder``. The V4
  model class then swaps that out for ``DeepseekV4TransformerBlock`` after
  ``super().__init__`` finishes (see ``deepseek_v4_model.py``).

* That intermediate ``TransformerBlock`` still has to be constructable, so we
  must hand it a *valid* layer spec. Phase 4 keeps the shape of that spec
  identical to a vanilla GPT layer spec — the V4-specific behavior lives
  entirely in :class:`DeepseekV4TransformerBlock`.

Phase 6 (deferred) will refactor :class:`DeepseekV4Model` to skip the
intermediate ``TransformerBlock`` allocation entirely; at that point this
file can collapse to a no-op stub or be deleted.

Why no V4-specific ``ModuleSpec`` here:
* V4's per-layer attention dispatch (CSA / HCA / dense + SWA / sink) is
  driven by ``compress_ratios`` at the *block* level, not the spec level.
* HC multi-stream is also a block-level concern (the block holds the K
  parallel streams).
* The submodule-level customization Megatron's spec system was designed for
  (Linear → ColumnParallelLinear, attention backend selection, etc.) is a
  Phase-6 / perf-phase concern — not Phase 4.
"""

from typing import Optional

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_decoder_layer_specs,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec


def get_deepseek_v4_layer_spec(
    config: TransformerConfig,
    *,
    use_transformer_engine: bool = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    normalization: Optional[str] = None,
    use_kitchen: bool = False,
    fallback_to_eager_attn: bool = False,
) -> ModuleSpec:
    """Return a placeholder layer spec used to satisfy ``GPTModel.__init__``.

    The returned spec is a vanilla GPT layer spec; it is consumed once during
    super-init then immediately replaced by :class:`DeepseekV4TransformerBlock`
    in the V4 model's constructor.
    """
    if use_transformer_engine:
        return get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            qk_layernorm=qk_layernorm,
            multi_latent_attention=getattr(config, "multi_latent_attention", False),
            experimental_attention_variant=getattr(config, "experimental_attention_variant", None),
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=use_kitchen,
            fallback_to_eager_attn=fallback_to_eager_attn,
        )

    return get_gpt_layer_local_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        qk_layernorm=qk_layernorm,
        multi_latent_attention=getattr(config, "multi_latent_attention", False),
        experimental_attention_variant=getattr(config, "experimental_attention_variant", None),
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=use_kitchen,
    )


def get_deepseek_v4_decoder_block_spec(
    config: TransformerConfig,
    *,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    """Placeholder block spec; same caveats as :func:`get_deepseek_v4_layer_spec`."""
    return get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=use_transformer_engine,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )


def get_deepseek_v4_decoder_layer_specs(
    config: TransformerConfig,
    *,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
):
    """Per-layer spec list (used by the placeholder MTP block spec below)."""
    return get_gpt_decoder_layer_specs(
        config=config,
        use_transformer_engine=use_transformer_engine,
        normalization=normalization,
        qk_l2_norm=qk_l2_norm,
        vp_stage=vp_stage,
    )


def get_deepseek_v4_mtp_block_spec(
    config: TransformerConfig,
    spec,
    *,
    use_transformer_engine: bool,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    """Placeholder MTP block spec.

    Phase 5 will replace this with V4's MTP variant (separate ``HyperHead``
    per MTP layer, optional dedicated routing). For now it delegates to
    upstream so super-init succeeds.
    """
    return get_gpt_mtp_block_spec(
        config=config,
        spec=spec,
        use_transformer_engine=use_transformer_engine,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )


__all__ = [
    "get_deepseek_v4_layer_spec",
    "get_deepseek_v4_decoder_block_spec",
    "get_deepseek_v4_decoder_layer_specs",
    "get_deepseek_v4_mtp_block_spec",
]
