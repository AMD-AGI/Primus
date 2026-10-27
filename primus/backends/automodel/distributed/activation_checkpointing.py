###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Shared activation-checkpointing application for diffusion parallelizers.

WHY THIS IS SHARED:
  Every diffusion model on this path wants the same three settings -- full,
  selective, off -- applied to whatever attributes hold its transformer blocks.
  Only those attribute names differ between models. Two bugs in this area are
  invisible at runtime, so they are decided once, here:

  - ``"selective"`` is a non-empty string and therefore truthy, so a bare
    ``if activation_checkpointing:`` followed by the full-AC branch silently makes
    ``selective`` and ``full`` the same thing.
  - ``"false"`` is *also* a non-empty string, so the same test enables
    checkpointing on a run configured to have none.

  The wrapping itself is AutoModel's: ``apply_full_layer_checkpointing_to_layers``
  and ``apply_selective_checkpointing_to_layers`` from
  ``nemo_automodel.components.distributed.activation_checkpointing``. This module
  only chooses which blocks to hand them.

WHY THERE IS A STRIDE:
  Off, selective and full are three points, and at long sequence lengths none of
  them is the right size. Peak activation memory is reached at the end of the
  forward pass, so checkpointing k of the N blocks sheds roughly k/N of it and
  costs roughly k/N of the recompute. A stride of n wraps block indices 0, n, 2n
  and so on, counted across all the block lists; 0 or 1 is plain full AC. It
  applies to full AC only: selective AC decides per operation inside every
  block, so a per-block stride would compose two granularities.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence, Tuple

logger = logging.getLogger(__name__)

# Spellings that mean "off" but are truthy in Python. Some CLI and config paths
# forward the flag as a raw string.
AC_OFF_VALUES = frozenset({"false", "0", "off", "no", "none", ""})

# What apply() did, for logging and for tests to assert against.
MODE_OFF = "off"
MODE_FULL = "full"
MODE_SELECTIVE = "selective"
MODE_NO_BLOCKS = "no-blocks"


def upstream_helpers():
    """AutoModel's activation-checkpointing module, which does the wrapping."""
    from nemo_automodel.components.distributed import activation_checkpointing

    return activation_checkpointing


def normalize(value: Any) -> Any:
    """Map false-like strings to False; leave bools and 'full'/'selective' alone.

    Deliberately does not coerce to bool: ``"selective"`` has to survive so the
    caller can still distinguish it from ``"full"``.
    """
    if isinstance(value, str) and value.strip().lower() in AC_OFF_VALUES:
        return False
    return value


def apply(
    helpers: Any,
    model: Any,
    block_attrs: Sequence[str],
    value: Any,
    *,
    enable_compile: bool = False,
    stride: int = 0,
    log_prefix: str,
) -> Tuple[str, int]:
    """Apply activation checkpointing to the model's transformer blocks.

    Args:
        helpers: AutoModel's activation-checkpointing module (see
            ``upstream_helpers``), passed in so tests can record the calls.
        block_attrs: attributes holding block lists, in a stable order. All of
            them are covered, so a model with dual-stream and single-stream
            lists gets both.
        value: the raw ``activation_checkpointing`` setting.
        stride: with full AC, wrap only every nth block. 0 or 1 wraps every
            block. Refused with selective AC.
        log_prefix: the caller's log tag.

    Returns:
        ``(mode, count)``, where ``count`` is the number of blocks wrapped.
    """
    value = normalize(value)
    if not value:
        logger.info("%s activation checkpointing OFF", log_prefix)
        return MODE_OFF, 0

    block_lists = [getattr(model, attr) for attr in block_attrs if getattr(model, attr, None) is not None]
    if not block_lists:
        logger.warning(
            "%s activation_checkpointing requested but the model has none of the block "
            "lists %s; nothing checkpointed.",
            log_prefix,
            ", ".join(block_attrs),
        )
        return MODE_NO_BLOCKS, 0

    if stride and stride < 1:
        raise ValueError(f"the activation-checkpointing stride must be >= 1, got {stride}")

    layers = [block for block_list in block_lists for block in block_list]

    if helpers.is_selective_activation_checkpointing(value):
        if stride > 1:
            raise ValueError(
                "a block stride cannot be combined with selective activation "
                "checkpointing: selective AC decides per operation inside every "
                "block, so the two are different granularities. Use full AC with a "
                "stride, or selective AC on its own."
            )
        # has_kv_sharing=False: these are diffusion transformers, with no KV cache.
        helpers.apply_selective_checkpointing_to_layers(
            model, layers, False, enable_compile=bool(enable_compile)
        )
        logger.info(
            "%s wrapped %d blocks with SELECTIVE (partial) activation checkpointing",
            log_prefix,
            len(layers),
        )
        return MODE_SELECTIVE, len(layers)

    chosen = layers[::stride] if stride > 1 else layers
    helpers.apply_full_layer_checkpointing_to_layers(model, chosen)
    if stride > 1:
        logger.info(
            "%s wrapped %d of %d blocks with FULL activation checkpointing (every %d)",
            log_prefix,
            len(chosen),
            len(layers),
            stride,
        )
    else:
        logger.info("%s wrapped %d blocks with FULL activation checkpointing", log_prefix, len(chosen))
    return MODE_FULL, len(chosen)
