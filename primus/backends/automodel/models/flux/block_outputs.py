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
  wrapper calls diffusers' own forward, so no diffusers code is copied. When
  another patch's forward already returns two fresh tensors (it sets
  ``FRESH_OUTPUTS_ATTR`` on the block class, as ``adaln_flydsl.py`` does), the
  wrapper passes them on uncloned: a clone of an opaque op's output would be a
  real copy.

Settings (module config, ``primus_perf:`` section):
    flux_unalias_block_outputs: false      apply this patch
"""
from __future__ import annotations

import functools
import logging

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_primus_unaliased_outputs"
_SECTION = "primus_perf"
# Set on FluxSingleTransformerBlock by a patch whose forward returns two fresh tensors.
FRESH_OUTPUTS_ATTR = "_primus_fresh_outputs"


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_unalias_block_outputs")


def install() -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    orig = FluxSingleTransformerBlock.forward

    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        encoder_hidden_states, hidden_states = orig(self, *args, **kwargs)
        if (
            getattr(type(self), FRESH_OUTPUTS_ATTR, False)
            and encoder_hidden_states._base is None
            and hidden_states._base is None
        ):
            return encoder_hidden_states, hidden_states
        return encoder_hidden_states.clone(), hidden_states.clone()

    FluxSingleTransformerBlock.forward = forward
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info("[PrimusFluxBlockOutputs] single-stream blocks return fresh tensors, not slices of one.")
    return True
