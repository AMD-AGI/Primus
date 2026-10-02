###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Activation policy for the Ideogram-4 var-len attention path.

Separate from the processor for the same reason ``_fp4_common`` is separate from
the linear swap: the patch condition has to be answerable during patch discovery,
which happens before anything has decided a run needs torch, a GPU, or diffusers.
Keeping the gates here -- with no import heavier than the settings module --
means discovery can ask "is this wanted?" without loading the machinery that would
answer "how would it work?".

Settings (module config, ``primus_ideogram4:`` section):

  varlen_attention: false        install the var-len processor. Off by default,
                                 leaving the stock SDPA path untouched.
  precompute_cu_seqlens: true    let the adapter build the packing on the host.
                                 Turning it off restores the mask-derived path;
                                 it exists for A/B comparison and rollback, not
                                 as a routine knob.
  attention_assume_dense: false  skip the mask analysis and run dense flash.
                                 EXACT ONLY for equal-length, unpadded batches.
"""
from __future__ import annotations

from primus.backends.nemo_automodel import options


def is_varlen_attn_enabled() -> bool:
    """Whether to install the var-len processor at all."""
    return options.flag("primus_ideogram4.varlen_attention")


def precompute_cu_seqlens_enabled() -> bool:
    """Whether the adapter is asked to build the packing on the host."""
    return options.flag("primus_ideogram4.precompute_cu_seqlens", default=True)


def precompute_cu_seqlens_active() -> bool:
    """Whether a precomputed packing would actually be READ by anything.

    Both switches have to be on, which is why this is not the same question as the
    flag above. Without the processor installed the stock path is in place and has
    no packing parameter, so precomputing would cost a build and a reserved token
    position every step for something nothing reads. This is the gate the adapter
    consults; the flag is only the flag.
    """
    return is_varlen_attn_enabled() and precompute_cu_seqlens_enabled()


def assume_dense_enabled() -> bool:
    """Whether to skip the mask analysis and go straight to dense flash.

    Exact only when every row has the same length, since dense attention over a
    padded row lets padding attend to real tokens. The adapter refuses a ragged
    batch while this is set, because it holds the lengths on the host and so can
    check for nothing.
    """
    return options.flag("primus_ideogram4.attention_assume_dense")
