###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Layer Patches

Replaces MaxText layer implementations with Primus-optimised versions:
    - ``NANOOFp8Quantization`` → ``PrimusNANOOFp8Quantization``
    - ``AttentionOp`` → ``PrimusAttentionOp``
    - ``Attention`` → ``PrimusAttention``
    - ``RoutedMoE`` → ``PrimusRoutedMoE``
    - Injects ``query_pre_attn_scalar`` into selected DecoderLayer classes
"""

import functools

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0

# ---------------------------------------------------------------------------
# Helper: wrap a DecoderLayer's __init__ to post-inject query_pre_attn_scalar
# ---------------------------------------------------------------------------


def _patch_decoder_init(cls, attention_attrs):
    """Wrap *cls.__init__* so that after the original initialisation every
    ``Attention`` sub-module listed in *attention_attrs* gets
    ``query_pre_attn_scalar = config.head_dim ** -0.5``.

    This avoids having to duplicate the entire ``__init__`` body (often
    60-200 lines) just to add one keyword argument to the ``Attention(...)``
    constructor call.  The attribute is only **stored** in
    ``Attention.__init__`` and consumed later in ``__call__``, so post-fixing
    it here is safe.
    """
    _orig_init = cls.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        scalar = self.config.head_dim**-0.5
        for attr_name in attention_attrs:
            getattr(self, attr_name).query_pre_attn_scalar = scalar

    cls.__init__ = _patched_init


# ============================================================================
# Quantization
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.quantization",
    backend="maxtext",
    phase="setup",
    description="Replace NANOOFp8Quantization with Primus implementation",
    condition=lambda ctx: True,
)
def patch_quantization(ctx: PatchContext) -> None:
    """
    Replace ``MaxText.layers.quantizations.NANOOFp8Quantization`` with
    ``PrimusNANOOFp8Quantization``.
    """
    log_rank_0("[Patch:maxtext.layers.quantization] Patching NANOOFp8Quantization...")

    import MaxText.layers.quantizations as orig_quantizations

    from primus.backends.maxtext.layers.quantizations import PrimusNANOOFp8Quantization

    orig_quantizations.NANOOFp8Quantization = PrimusNANOOFp8Quantization

    warning_rank_0("[Patch:maxtext.layers.quantization] NANOOFp8Quantization patched successfully.")


# ============================================================================
# Attention
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.attention",
    backend="maxtext",
    phase="setup",
    description="Replace AttentionOp and Attention with Primus implementations",
    condition=lambda ctx: True,
    backend_versions=["0.1.1"],
)
def patch_attention(ctx: PatchContext) -> None:
    """
    Replace MaxText attention op classes with Primus versions:
        - ``AttentionOp`` → ``PrimusAttentionOp``
    """
    log_rank_0("[Patch:maxtext.layers.attention] Patching Attention layers...")

    import MaxText.layers.attention_op as orig_attention_op
    import MaxText.layers.attentions as orig_attentions

    from primus.backends.maxtext.layers.attention_op import PrimusAttentionOp

    orig_attention_op.AttentionOp = PrimusAttentionOp
    orig_attentions.AttentionOp = PrimusAttentionOp

    warning_rank_0("[Patch:maxtext.layers.attention] Attention layers patched successfully.")


# ============================================================================
# Mixture-of-Experts (MoE)
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.moe",
    backend="maxtext",
    phase="setup",
    description="Replace RoutedMoE with Primus implementation (version-agnostic via *args/**kwargs)",
    condition=lambda ctx: True,
)
def patch_moe(ctx: PatchContext) -> None:
    """
    Replace ``MaxText.layers.moe.RoutedMoE`` with ``PrimusRoutedMoE``.

    The unified ``PrimusRoutedMoE`` uses ``*args, **kwargs`` for
    ``dense_matmul`` / ``sparse_matmul``, so a single class works with both
    the 6-param (Aug-version) and 9-param (Dec-version, with bias) signatures.
    """
    log_rank_0("[Patch:maxtext.layers.moe] Patching RoutedMoE...")

    import MaxText.layers.moe as orig_moe

    from primus.backends.maxtext.layers.moe import PrimusRoutedMoE

    orig_moe.RoutedMoE = PrimusRoutedMoE

    warning_rank_0("[Patch:maxtext.layers.moe] RoutedMoE patched successfully.")


# ============================================================================
# Decoder Layer
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.decoder_layer",
    backend="maxtext",
    phase="setup",
    description="Inject query_pre_attn_scalar into targeted DecoderLayer classes",
    condition=lambda ctx: True,
    backend_versions=["0.1.1"],
)
def patch_decoder_layer(ctx: PatchContext) -> None:
    """Inject ``query_pre_attn_scalar = config.head_dim ** -0.5`` into the
    ``Attention`` sub-modules of selected DecoderLayer classes.

    Only the five model families that originally lacked this parameter are
    patched; models that already set it (Gemma3, Llama4, Qwen3, GPT-OSS …)
    are left untouched.
    """
    log_rank_0("[Patch:maxtext.layers.decoder_layer] Patching DecoderLayer (query_pre_attn_scalar)...")

    from MaxText.layers.gemma import GemmaDecoderLayer
    from MaxText.layers.gemma2 import Gemma2DecoderLayer
    from MaxText.layers.llama2 import LlamaDecoderLayer
    from MaxText.layers.mistral import MistralDecoderLayer
    from MaxText.layers.mixtral import MixtralDecoderLayer

    _patch_decoder_init(GemmaDecoderLayer, ["self_attention"])
    _patch_decoder_init(Gemma2DecoderLayer, ["self_attention_local", "self_attention_global"])
    _patch_decoder_init(LlamaDecoderLayer, ["self_attention"])
    _patch_decoder_init(MistralDecoderLayer, ["self_attention"])
    _patch_decoder_init(MixtralDecoderLayer, ["self_attention"])

    warning_rank_0("[Patch:maxtext.layers.decoder_layer] DecoderLayer patched successfully.")


# ============================================================================
# Legacy patches for Aug-version MaxText (2025.*)
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.attention.legacy",
    backend="maxtext",
    phase="setup",
    description="Replace AttentionOp and Attention with Primus legacy implementations (Aug version)",
    condition=lambda ctx: True,
    backend_versions=["2025.*"],
)
def patch_attention_legacy(ctx: PatchContext) -> None:
    """
    Replace MaxText attention classes with legacy Primus versions:
        - ``AttentionOp`` → ``PrimusAttentionOp`` (no nnx_wrappers/lazy_init)
        - ``Attention`` → ``PrimusAttention`` (query_w init override)
    """
    log_rank_0("[Patch:maxtext.layers.attention.legacy] Patching Attention layers (legacy)...")
    import MaxText.layers.attention_mla as orig_attention_mla
    import MaxText.layers.attention_op as orig_attention_op
    import MaxText.layers.attentions as orig_attentions

    from primus.backends.maxtext.legacy.layers.attention_op import PrimusAttentionOp
    from primus.backends.maxtext.legacy.layers.attentions import PrimusAttention

    orig_attention_op.AttentionOp = PrimusAttentionOp
    orig_attentions.AttentionOp = PrimusAttentionOp
    orig_attentions.Attention = PrimusAttention
    orig_attention_mla.Attention = PrimusAttention
    warning_rank_0("[Patch:maxtext.layers.attention.legacy] Attention layers patched successfully.")
