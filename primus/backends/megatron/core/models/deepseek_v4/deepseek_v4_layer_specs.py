###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 layer specs (Phase 3 scaffolding).

The four ``get_deepseek_v4_*`` helpers mirror the upstream GPT helpers
(``get_gpt_layer_local_spec`` / ``get_gpt_decoder_block_spec`` / ...). They
delegate to the GPT helpers verbatim in Phase 3, then call two **resolution
hooks** that Phase 4 (attention) and Phase 5 (MoE / SwiGLU) can override
without touching the call sites.

Hook contract:

* :func:`_resolve_attention_module_spec(config, base_spec, ...)`
  Phase 4 will return a V4-specific attention ``ModuleSpec`` (Dense / HCA /
  CSA + SWA + sink + dual-RoPE) here. Phase 3 returns ``None`` -> caller
  keeps the GPT spec.

* :func:`_resolve_mlp_module_spec(config, base_spec, ...)`
  Phase 5 will return a V4-specific MLP ``ModuleSpec`` (sqrtsoftplus router
  + hash routing + clamped SwiGLU) here. Phase 3 returns ``None``.

Phase 4 / 5 implement these hooks by patching this module (or by switching
the ``Spec.submodules`` dataclass fields to V4 module classes); the V4
builder doesn't need to change.
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

# ---------------------------------------------------------------------------
# Resolution hooks (Phase 4 / 5 will fill these in).
# ---------------------------------------------------------------------------


def _resolve_attention_module_spec(
    config: TransformerConfig, base_spec: ModuleSpec, **_kw
) -> Optional[ModuleSpec]:
    """Return a V4-specific attention spec, or ``None`` to keep the GPT spec.

    Phase 3: always returns ``None`` (V4 attention modules don't exist yet).
    """
    return None


def _resolve_mlp_module_spec(config: TransformerConfig, base_spec: ModuleSpec, **_kw) -> Optional[ModuleSpec]:
    """Return a V4-specific MLP spec, or ``None`` to keep the GPT spec.

    Phase 3: always returns ``None``.
    """
    return None


# ---------------------------------------------------------------------------
# Public layer-spec helpers (delegating to GPT for now).
# ---------------------------------------------------------------------------


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
    """Return a V4 transformer-layer spec.

    Phase 3 delegates to the GPT helper; the resolution hooks above let
    Phase 4 / 5 swap in V4 attention / MLP module specs.
    """
    if use_transformer_engine:
        spec = get_gpt_layer_with_transformer_engine_spec(
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
    else:
        spec = get_gpt_layer_local_spec(
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

    # Phase 4 / 5 hooks. ``None`` means keep the GPT spec as is.
    v4_attn = _resolve_attention_module_spec(config, spec)
    if v4_attn is not None:
        spec.submodules.self_attention = v4_attn

    v4_mlp = _resolve_mlp_module_spec(config, spec)
    if v4_mlp is not None:
        spec.submodules.mlp = v4_mlp

    return spec


def get_deepseek_v4_decoder_block_spec(
    config: TransformerConfig,
    *,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    """Per-layer spec list for the decoder block.

    Phase 3: delegates to the GPT helper. Phase 4 will iterate its return
    value and apply :func:`_resolve_attention_module_spec` per layer based
    on ``compress_ratios[layer_id]``.
    """
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
    """Per-layer spec list (used by the MTP block spec below)."""
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
    """V4 MTP block spec.

    Phase 3 delegates to the GPT helper. Phase 5 will replace this with the
    V4 MTP variant (separate :class:`HyperHead` per MTP layer and an
    optional dedicated routing path).
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
