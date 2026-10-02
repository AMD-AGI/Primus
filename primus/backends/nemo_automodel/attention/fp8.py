###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-Turbo FP8 flash-attention override.

FP8 *attention*, orthogonal to the FP8/MXFP4 GEMM swaps: those replace
``nn.Linear`` and leave scaled-dot-product attention in bf16. On long-sequence
models attention dominates the step, so the attention kernel is the remaining
lever there. Primus-Turbo's ``flash_attn_fp8_func`` (E4M3, block-scaled q/k/v,
autograd forward and backward) is the same kernel family as the GEMM path. There
is no FP4 attention kernel, so FP8 is the only option, and it pairs with any GEMM
precision.

WHAT:
  Rebinds the FLASH and AITER backend functions to an FP8 wrapper. See
  ``_backend_registry`` for the seam, the fallback conditions and the mutual
  exclusion with the non-deterministic override. Inputs arrive bf16 and the
  wrapper casts internally, so the registered constraint checks still hold.

SEQUENCE LENGTHS:
  The kernel block-scales q/k/v along the sequence with ``block_size=64`` and so
  requires ``seqlen % 64 == 0``; a non-conforming length fails with a reshape
  error. The two sides are handled differently because padding is exact for one
  and not the other:

  * Queries are zero-padded up to the next multiple of 64 and the real rows
    sliced back out. Each query row is independent, so this is exact.
  * Keys are never padded. A zero key adds ``exp(0)`` to every row's softmax
    denominator, which dilutes the output by a large factor when the real key
    count is small (a 77-token text context, say). A call whose key length is not
    a multiple of 64 runs the original kernel instead, logged once.

Settings (module config):
  primus_turbo.fp8_attention: false    enable the override (default off = no-op)

The override replaces the kernels behind ``model.attention_backend: flash`` and
``aiter``; with any other backend it is installed but never called, and says so.
"""
from __future__ import annotations

import logging
from typing import Optional

from primus.backends.nemo_automodel import options
from primus.backends.nemo_automodel.attention import _backend_registry

logger = logging.getLogger(__name__)

OVERRIDE_NAME = "turbo-fp8"
_LOG_PREFIX = "[PrimusTurbo-FP8Attn]"

# The kernel block-scales q/k/v along the sequence with this block size, so the
# sequence length must be a multiple of it.
ATTN_BLOCK = 64


def pad_to_block(t, mult: int = ATTN_BLOCK):
    """Zero-pad a (B, S, H, D) tensor's sequence dim up to a multiple of ``mult``.

    Returns ``(padded, real_seqlen)``. Returns the input untouched when it already
    conforms, so the common case allocates nothing.
    """
    import torch

    s = t.shape[1]
    s_pad = ((s + mult - 1) // mult) * mult
    if s_pad == s:
        return t, s
    # F.pad fills from the last dim backwards: (D_lo, D_hi, H_lo, H_hi, S_lo, S_hi).
    return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, s_pad - s)), s


def unaligned_keys(q, k, v) -> Optional[str]:
    """Why the FP8 kernel cannot serve this call exactly, or None if it can."""
    skv = k.shape[1]
    if skv % ATTN_BLOCK:
        return f"the key length {skv} is not a multiple of {ATTN_BLOCK}"
    return None


def flash_attn_fp8_pad64(q, k, v, softmax_scale=None, causal: bool = False):
    """FP8 flash attention, padding the query sequence up to a multiple of 64.

    q is (B, Sq, H, D) and k/v are (B, Skv, H, D) with ``Skv % 64 == 0`` (see
    ``unaligned_keys``); returns (B, Sq, H, D).
    """
    from primus_turbo.pytorch.ops import flash_attn_fp8_func

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q_pad, sq_real = pad_to_block(q)
    out = flash_attn_fp8_func(q_pad, k, v, softmax_scale=softmax_scale, causal=causal)
    return out[:, :sq_real].contiguous()


def is_enabled() -> bool:
    """Whether the FP8 attention override was requested."""
    return options.flag("primus_turbo.fp8_attention")


def install(configured_backend: Optional[str] = None) -> bool:
    """Rebind the target backends to the FP8 kernel."""

    def probe() -> None:
        # Fail fast rather than silently running bf16 attention.
        from primus_turbo.pytorch.ops import flash_attn_fp8_func  # noqa: F401

    return _backend_registry.install_override(
        kernel=flash_attn_fp8_pad64,
        override_name=OVERRIDE_NAME,
        log_prefix=_LOG_PREFIX,
        description="FP8 flash attention (flash_attn_fp8_func)",
        probe=probe,
        configured_backend=configured_backend,
        unsupported=unaligned_keys,
    )
