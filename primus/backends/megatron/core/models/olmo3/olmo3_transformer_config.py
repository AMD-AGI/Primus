###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMo-3 transformer config.

OLMo-3 (HF ``model_type: olmo3``) differs from the Llama layout in four ways:

* post-norm: ``post_attention_layernorm`` / ``post_feedforward_layernorm`` are
  applied to each sublayer *output* before the residual add; there is no
  pre-norm on the sublayer input;
* QK-norm over the **full** projection width (``num_heads * head_dim``), not
  per head;
* sliding-window attention on three of every four layers
  (``window_attn_skip_freq``) and full attention on the rest;
* plain RoPE on the sliding layers and YaRN RoPE on the full layers.

Window: ``olmo3_sliding_window`` is the HF ``sliding_window`` W (visible keys
including the current token, 4096 for OLMo-3). ``window_size`` holds the upstream
Megatron / TE ``(left, right)`` convention ``(W - 1, 0)``, so every upstream consumer
(TE / local DotProductAttention, saved args) reads it correctly. Only the HF boundary
(converter, model preset, args patch) uses W.

The YaRN fields carry an ``olmo3_`` prefix on purpose: upstream
``Attention`` multiplies RoPE by a YaRN concentration factor on **every**
layer whenever the config has ``yarn_rotary_scaling_factor``; OLMo-3 needs it on
the full-attention layers only, which the OLMo-3 attention applies itself.
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.models.olmo3.olmo3_validation import (
    YARN_BETA_FAST_DEFAULT,
    YARN_BETA_SLOW_DEFAULT,
    check_olmo3_unsupported,
    olmo3_window_from_settings,
    parse_olmo3_skip_freq,
)

__all__ = [
    "OLMO3_ROPE_LAYOUTS",
    "Olmo3TransformerConfig",
    "yarn_attention_factor",
    "resolve_olmo3_rope_layout",
]

logger = logging.getLogger(__name__)


OLMO3_ROPE_LAYOUTS = ("per_type", "yarn_all")


# transformers' Olmo3 RoPE layout by version. Measured points (logprob parity against the HF model):
# 4.57 builds plain RoPE on sliding layers and YaRN on full ones (per_type); 5.10 builds one YaRN
# table for every layer (yarn_all).
# The bounds 5.0 and 5.13 are unverified (no installed 5.0 or 5.13 was tested). Olmo3 first ships in 4.57; older
# versions are refused, and so is a pre-release at a bound. ``auto`` is a parity-tool convenience
# only; training sets the layout.
_OLMO3_FIRST_TRANSFORMERS = (4, 57)
_YARN_ALL_TRANSFORMERS = ((5, 0), (5, 13))  # [first, end)


def _rope_layout_of_installed_transformers() -> tuple:
    """(layout, version): the layout the installed transformers' ``Olmo3Model`` uses (for ``auto``
    only) and the transformers version it was read from."""
    try:
        import transformers
    except ImportError as e:
        raise ValueError(
            "olmo3_rope_layout: auto needs transformers installed; set per_type or yarn_all"
        ) from e
    text = transformers.__version__
    m = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)*(.*)", text.strip())
    if m is None:
        raise ValueError(
            f"olmo3_rope_layout: auto cannot parse transformers version {text!r}; set per_type or yarn_all"
        )
    ver, suffix = (int(m.group(1)), int(m.group(2))), m.group(3)
    if ver < _OLMO3_FIRST_TRANSFORMERS:
        raise ValueError(
            f"olmo3_rope_layout: auto: transformers {text} has no Olmo3 model (it needs >= 4.57); "
            "set per_type or yarn_all"
        )
    first, end = _YARN_ALL_TRANSFORMERS
    if suffix and ver in (first, end):  # e.g. 5.0.0rc1, 5.13.0.dev0: either side of a change
        raise ValueError(
            f"olmo3_rope_layout: auto: transformers pre-release {text} is at a layout change; "
            "set per_type or yarn_all"
        )
    return ("yarn_all" if first <= ver < end else "per_type"), text


def resolve_olmo3_rope_layout(layout: str = "per_type") -> str:
    """Per-layer-type RoPE layout.

    * ``"per_type"`` (default): plain RoPE on sliding layers, YaRN on full
      layers. This is the trained OLMo-3 model (OLMo-core); transformers 4.57
      measured.
    * ``"yarn_all"``: one YaRN table (and attention factor) on every layer;
      transformers 5.10 measured (the version range is unverified, see the
      version note above ``_OLMO3_FIRST_TRANSFORMERS``).
    * ``"auto"`` (opt-in): follow the transformers installed in this
      environment; the resolved layout is logged. Use it only to match a
      transformers reference, never for training.

    An explicit ``None`` or an unknown value is refused; omitting it gives ``per_type``.
    """
    if layout in OLMO3_ROPE_LAYOUTS:
        return layout
    if layout != "auto":
        raise ValueError(f"olmo3_rope_layout must be per_type, yarn_all or auto, got {layout!r}")
    resolved, version = _rope_layout_of_installed_transformers()
    logger.warning("olmo3_rope_layout: auto resolved to %r (transformers %s)", resolved, version)
    return resolved


def yarn_attention_factor(scaling_factor: float, attention_factor: Optional[float] = None) -> float:
    """YaRN attention (concentration) factor, as HF ``_compute_yarn_parameters``.

    ``scaling_factor == 1`` is YaRN off: 1.0, even if an explicit ``attention_factor`` is given
    (HF applies it for rope_type yarn only, and OLMo-3 builds no YaRN table then). A factor
    below 1 and an ``attention_factor <= 0`` (it would zero the softmax scale) raise
    ``ValueError``.
    """
    scaling_factor = float(scaling_factor)
    if not scaling_factor >= 1.0:
        raise ValueError(f"OLMo-3 YaRN scaling factor must be >= 1 (1 = YaRN off), got {scaling_factor}")
    if attention_factor is not None and not float(attention_factor) > 0.0:
        raise ValueError(f"OLMo-3 YaRN attention_factor must be > 0, got {attention_factor}")
    if scaling_factor == 1.0:
        return 1.0
    if attention_factor is not None:
        return float(attention_factor)
    return 0.1 * math.log(scaling_factor) + 1.0


@dataclass
class Olmo3TransformerConfig(TransformerConfig):
    """TransformerConfig plus the OLMo-3 full-attention-layer YaRN settings."""

    olmo3_yarn_rotary_scaling_factor: float = 1.0
    """YaRN ``factor`` for the full-attention layers. 1.0 (default) disables YaRN; the
    model preset / converter set the checkpoint's value (8.0 for Olmo-3-1025-7B)."""

    olmo3_yarn_original_max_position_embeddings: Optional[int] = None
    """YaRN ``original_max_position_embeddings``; required when YaRN is on."""

    olmo3_yarn_beta_fast: float = YARN_BETA_FAST_DEFAULT
    """YaRN ``beta_fast``."""

    olmo3_yarn_beta_slow: float = YARN_BETA_SLOW_DEFAULT
    """YaRN ``beta_slow``."""

    olmo3_yarn_attention_factor: Optional[float] = None
    """YaRN ``attention_factor``; None derives ``0.1 * ln(factor) + 1``. Ignored (1.0) when YaRN is off."""

    olmo3_rope_layout: str = "per_type"
    """RoPE on sliding layers: ``per_type`` (plain, default) or ``yarn_all`` (YaRN). ``auto`` is
    refused here: resolve it first with :func:`resolve_olmo3_rope_layout` (the converter does)."""

    olmo3_sliding_window: Optional[int] = None
    """HF ``sliding_window`` W: keys a sliding-layer query sees, current token included.
    Required. ``window_size`` must agree with it in the form ``olmo3_window_size_form`` declares
    (``(W - 1, 0)`` for ``megatron`` or unset, ``(W, 0)`` for ``hf``), or be unset (then derived);
    any other ``window_size`` raises (see ``olmo3_validation.olmo3_window_from_settings``)."""

    olmo3_window_size_form: Optional[str] = None
    """Form of the ``window_size`` passed in: ``megatron`` ``(W - 1, 0)`` (what the args patch and the
    converter write) or ``hf`` ``(W, 0)``; unset means ``megatron`` (``olmo3_sliding_window`` is required either way). After
    ``__post_init__`` the config always holds the Megatron form and this field is ``megatron``."""

    def __post_init__(self):
        self._olmo3_check_window()
        # one table for args, config and builder (olmo3_validation.OLMO3_UNSUPPORTED)
        check_olmo3_unsupported(self)
        m = self.olmo3_yarn_mscale
        if self.olmo3_yarn_attention_factor is not None and self.olmo3_yarn_enabled:
            derived = yarn_attention_factor(self.olmo3_yarn_rotary_scaling_factor)
            if not math.isclose(m, derived, rel_tol=1e-9):
                # an HF checkpoint may carry its own attention_factor, so it is used, not refused
                logger.warning(
                    "OLMo-3 olmo3_yarn_attention_factor=%s differs from 0.1*ln(factor)+1=%s for "
                    "olmo3_yarn_rotary_scaling_factor=%s; the explicit value is used (unset it to "
                    "derive it from the factor)",
                    m,
                    derived,
                    self.olmo3_yarn_rotary_scaling_factor,
                )
        if self.olmo3_yarn_enabled and not self.olmo3_yarn_original_max_position_embeddings:
            raise ValueError(
                "OLMo-3 YaRN (olmo3_yarn_rotary_scaling_factor > 1) needs "
                "olmo3_yarn_original_max_position_embeddings"
            )
        super().__post_init__()

    def _olmo3_check_window(self):
        """``window_size = (W - 1, 0)`` (Megatron / TE) + ``olmo3_sliding_window = W``.

        The config needs ``olmo3_sliding_window`` W (HF convention). ``window_size`` is read in the form
        ``olmo3_window_size_form`` declares (``megatron`` or unset: ``(W - 1, 0)``; ``hf``:
        ``(W, 0)``) and must agree with W, or is unset and derived. Either way the config then
        holds the Megatron / TE ``(W - 1, 0)`` and the form ``megatron``. Without a declared form
        an HF ``(W, 0)`` raises, so a value is never guessed.
        """
        if self.olmo3_sliding_window is None:
            raise ValueError(
                "Olmo3TransformerConfig needs olmo3_sliding_window (HF sliding_window W); window_size "
                "then holds the Megatron/TE (W - 1, 0). Build the config from args after the OLMo-3 "
                "args patch, or from the converter's margs."
            )
        ws, w = olmo3_window_from_settings(
            self.window_size, self.olmo3_sliding_window, self.olmo3_window_size_form
        )
        self.window_size, self.olmo3_sliding_window = ws, w
        self.olmo3_window_size_form = "megatron"
        self.window_attn_skip_freq = parse_olmo3_skip_freq(self.window_attn_skip_freq, self.num_layers)

    @property
    def olmo3_yarn_enabled(self) -> bool:
        """YaRN acts only with a scaling factor > 1 (factor 1: the YaRN and plain tables agree)."""
        return float(self.olmo3_yarn_rotary_scaling_factor) > 1.0

    @property
    def olmo3_yarn_mscale(self) -> float:
        """YaRN attention factor m, applied (as m**2 on the logits) on every layer that uses YaRN:
        the full-attention layers, and with ``olmo3_rope_layout: yarn_all`` the sliding ones too."""
        return yarn_attention_factor(self.olmo3_yarn_rotary_scaling_factor, self.olmo3_yarn_attention_factor)
