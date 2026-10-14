###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Runtime Megatron-Bridge patches for Gemma 4.

1. ``gemma4.bridge.text_mode_for_moe`` — make the MoE path honor
   ``GEMMA4_CONVERSION_MODE=text``. ``Gemma4VLBridge.provider_bridge`` only
   consults the conversion mode on the dense path::

       if not getattr(text_config, "enable_moe_block", False):
           self._is_dense = True
           if self._conversion_mode() == "text":
               return self._build_dense_provider(text_config)
           return self._build_dense_vl_provider(...)

       self._is_dense = False
       # ... unconditionally builds Gemma4VLModelProvider

   The published MoE checkpoint (``google/gemma-4-26B-A4B``) sets
   ``enable_moe_block=true`` and declares ``Gemma4ForConditionalGeneration``, so
   it always lands on the VL provider and builds vision + audio towers. Training
   then fails in the loss: the VL model returns the LLaVA-style
   ``(loss, new_loss_mask)`` tuple, but ``gpt_step.forward_step`` never passes
   ``loss_mask`` into the model, so ``masked_next_token_loss`` dereferences
   ``None``.

2. ``gemma4.dense.te_core_attention`` — opt-in Transformer Engine attention for
   the dense path, which Bridge otherwise pins to ``LocalSpecProvider``.

The upstream fix for (1) is two lines in ``gemma4_vl_bridge.py`` right after
``self._is_dense = False``; until that lands, patching here keeps
``third_party/Megatron-Bridge`` untouched.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Optional

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

_PATCHED_ATTR = "_primus_gemma4_text_moe_patched"
_TE_PATCHED_ATTR = "_primus_gemma4_dense_te_patched"


def _find_vl_bridge_class() -> Optional[type]:
    """Locate ``Gemma4VLBridge`` without importing ``megatron.bridge.models.gemma_vl``.

    That package is stubbed by the Primus adapter whenever its NV-only deps
    (megatron-energon / qwen-vl-utils) are missing, so importing it directly is
    not safe. The class is reachable as a subclass of the always-real
    ``Gemma4Bridge`` once Bridge's model registry has been imported.
    """
    try:
        from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge
    except Exception:
        return None

    pending = list(Gemma4Bridge.__subclasses__())
    while pending:
        cls = pending.pop()
        if cls.__name__ == "Gemma4VLBridge":
            return cls
        pending.extend(cls.__subclasses__())
    return None


@register_patch(
    "gemma4.bridge.text_mode_for_moe",
    backend="megatron_bridge",
    phase="setup",
    description="Honor GEMMA4_CONVERSION_MODE=text on the Gemma 4 MoE path, not just dense",
)
def patch_gemma4_text_mode_for_moe(ctx: PatchContext) -> None:
    vl_bridge = _find_vl_bridge_class()
    if vl_bridge is None:
        return

    original = vl_bridge.__dict__.get("provider_bridge")
    if original is None or getattr(original, _PATCHED_ATTR, False):
        return

    def provider_bridge(self, hf_pretrained: Any):
        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", None) or hf_config

        wants_text_moe = (
            getattr(text_config, "enable_moe_block", False)
            and self._conversion_mode() == "text"
            and hasattr(self, "_build_moe_provider")
        )
        if not wants_text_moe:
            return original(self, hf_pretrained)

        self._is_dense = False
        provider = self._build_moe_provider(text_config)
        log_rank_0(
            "[Patch:gemma4.bridge.text_mode_for_moe] GEMMA4_CONVERSION_MODE=text: "
            f"built {type(provider).__name__} instead of Gemma4VLModelProvider"
        )
        return provider

    setattr(provider_bridge, _PATCHED_ATTR, True)
    vl_bridge.provider_bridge = provider_bridge
    log_rank_0(f"[Patch:gemma4.bridge.text_mode_for_moe] Patched {vl_bridge.__name__}.provider_bridge")


def _build_dense_te_core_attention() -> Optional[type]:
    """Build a TE core-attention class for the Gemma 4 *dense* layer spec.

    ``get_gemma4_layer_spec`` pins ``LocalSpecProvider``, so dense Gemma 4 runs on
    ``DotProductAttention``: no flash attention, and the full [b, h, s, s] score
    matrix is materialized and kept for backward. On 31B / seq 4096 that is ~94 GB
    of the 161 GB peak, which caps the reachable micro-batch size.

    The MoE path already has ``Gemma4TEDotProductAttention``, but it keys the
    sliding-vs-global decision off ``interleaved_attn_pattern`` (the MoE
    convention). Dense uses ``_is_gemma4_sliding_layer`` instead, so it needs its
    own subclass with the same window bookkeeping.
    """
    try:
        from megatron.bridge.models.gemma.modeling_gemma4 import (
            _is_gemma4_sliding_layer,
        )
        from megatron.core.extensions.transformer_engine import TEDotProductAttention
    except Exception:
        return None

    class Gemma4DenseTEDotProductAttention(TEDotProductAttention):
        """Dense Gemma 4 core attention on Transformer Engine (flash attention)."""

        def __init__(
            self, config, layer_number, attn_mask_type, attention_type, attention_dropout=None, **kwargs
        ):
            config = copy.deepcopy(config)
            # Global layers must see the full context. Gemma4DenseProvider already
            # stores the TE-ready (left, right) span; only the MoE provider
            # carries a bare int.
            if not _is_gemma4_sliding_layer(config, layer_number):
                config.window_size = None
            elif not isinstance(config.window_size, tuple):
                config.window_size = (config.window_size - 1, 0)

            super().__init__(
                config=config,
                layer_number=layer_number,
                attn_mask_type=attn_mask_type,
                attention_type=attention_type,
                attention_dropout=attention_dropout,
                **kwargs,
            )

    return Gemma4DenseTEDotProductAttention


@register_patch(
    "gemma4.dense.te_core_attention",
    backend="megatron_bridge",
    phase="setup",
    description="Opt-in: run dense Gemma 4 core attention on Transformer Engine instead of LocalSpecProvider",
)
def patch_gemma4_dense_te_attention(ctx: PatchContext) -> None:
    if os.environ.get("PRIMUS_GEMMA4_DENSE_ATTENTION_BACKEND", "").lower() != "te":
        return

    try:
        from megatron.bridge.models.gemma import gemma4_provider
    except Exception:
        return

    original = getattr(gemma4_provider, "get_gemma4_layer_spec", None)
    if original is None or getattr(original, _TE_PATCHED_ATTR, False):
        return

    te_core_attention = _build_dense_te_core_attention()
    if te_core_attention is None:
        log_rank_0(
            "[Patch:gemma4.dense.te_core_attention] Transformer Engine unavailable; keeping local attention"
        )
        return

    def get_gemma4_layer_spec(config=None):
        spec = original(config)
        spec.submodules.self_attention.submodules.core_attention = te_core_attention
        return spec

    setattr(get_gemma4_layer_spec, _TE_PATCHED_ATTR, True)
    gemma4_provider.get_gemma4_layer_spec = get_gemma4_layer_spec
    log_rank_0("[Patch:gemma4.dense.te_core_attention] Dense Gemma 4 core attention -> TEDotProductAttention")
