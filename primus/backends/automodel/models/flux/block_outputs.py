###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Return fresh tensors from FLUX's single-stream block instead of two views of one.

WHY:
  ``FluxSingleTransformerBlock.forward`` concatenates the text and image tokens
  and ends with ``hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]``:
  two views of one intermediate. AutoModel compiles each block on its own
  (``module.compile()``), and AOTAutograd hands aliased outputs back by
  returning the intermediate and replaying the slices in eager mode. In the
  backward, each replayed slice then fills a zero tensor of the full
  [batch, sequence, hidden] shape and copies its gradient in, and autograd adds
  the two: a fill, a copy and an add per block boundary, outside the compiled
  graphs.

WHAT:
  Wraps the forward and clones its two outputs, so the compiled graph returns
  fresh tensors. Inductor fuses the clones into the residual add that produces
  them, and the backward's split into the fused kernels that consume the
  gradients. One block's outputs and gradients are bit-identical to the
  unpatched block's; along the block chain they differ by at most one BF16
  rounding step, because the next block receives contiguous inputs instead of
  strided views and Inductor compiles slightly different kernels for it. The
  wrapper calls diffusers' own forward, so no diffusers code is copied.

  Meant for compiled blocks (``fsdp.enable_compile: true``). Without
  compilation the clones are real copies, one [batch, sequence, hidden]
  tensor per single-stream block, and nothing is saved.

  Declines, with a warning, when the block class is not importable or its
  forward does not take ``encoder_hidden_states``: older diffusers return one
  tensor from this block, not the text and image pair.

Settings (module config, ``primus_perf:`` section):
    flux_single_block_fresh_outputs: false      apply this patch
"""
from __future__ import annotations

import functools
import inspect
import logging

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusFluxBlockOutputs]"
_PATCHED_ATTR = "_primus_single_block_fresh_outputs"


def is_enabled() -> bool:
    return options.flag("primus_perf.flux_single_block_fresh_outputs")


def install() -> bool:
    """Wrap ``FluxSingleTransformerBlock.forward`` to clone its two outputs.

    Idempotent: returns True if it is already wrapped. Returns False, and
    changes nothing, when the block is not importable or its forward does not
    return the text and image pair.
    """
    try:
        from diffusers.models.transformers.transformer_flux import (
            FluxSingleTransformerBlock,
        )
    except ImportError as exc:
        logger.warning(
            "%s FluxSingleTransformerBlock is not importable (%s); not installed", _LOG_PREFIX, exc
        )
        return False

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    orig = FluxSingleTransformerBlock.forward
    if "encoder_hidden_states" not in inspect.signature(orig).parameters:
        logger.warning(
            "%s FluxSingleTransformerBlock.forward takes no encoder_hidden_states, so it does not "
            "return the text and image pair; not installed",
            _LOG_PREFIX,
        )
        return False

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        encoder_hidden_states, hidden_states = orig(self, *args, **kwargs)
        return encoder_hidden_states.clone(), hidden_states.clone()

    FluxSingleTransformerBlock.forward = forward
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info("%s single-stream blocks return fresh tensors, not slices of one.", _LOG_PREFIX)
    return True
