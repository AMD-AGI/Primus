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
  requires physically padded sequence dimensions. The wrapper zero-pads q/k/v
  to the next block, passes their unpadded logical lengths to Primus-Turbo, and
  slices the output back to the real query length. Both forward and backward
  bound their tiles by those logical lengths, so padded keys never enter the
  softmax and padded gradients remain zero. Causal alignment also uses the
  logical lengths rather than the physical padding.

Settings (module config):
  primus_turbo.fp8_attention: false    enable the override (default off = no-op)

The override replaces the kernels behind ``model.attention_backend: flash`` and
``aiter``; with any other backend it is installed but never called, and says so.
"""
from __future__ import annotations

import logging
from typing import Optional

from primus.backends.automodel import options
from primus.backends.automodel.attention import _backend_registry

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


def unsupported_lengths(q, k, v, is_causal: bool = False) -> Optional[str]:
    """Why logical tail masking cannot serve this call exactly, or None."""
    if k.shape[1] != v.shape[1]:
        return f"key and value lengths differ ({k.shape[1]} != {v.shape[1]})"
    return None


def flash_attn_fp8_pad64(q, k, v, softmax_scale=None, causal: bool = False):
    """FP8 flash attention with masked q/k/v padding to a multiple of 64.

    q/k/v are BSHD. The Turbo kernel receives both physical padded tensors and
    logical sequence bounds; this function returns the original query length.
    """
    from primus_turbo.pytorch.ops import flash_attn_fp8_func

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q_pad, sq_real = pad_to_block(q)
    k_pad, sk_real = pad_to_block(k)
    v_pad, sv_real = pad_to_block(v)
    if sk_real != sv_real:
        raise ValueError(f"key and value lengths differ ({sk_real} != {sv_real})")
    out = flash_attn_fp8_func(
        q_pad,
        k_pad,
        v_pad,
        softmax_scale=softmax_scale,
        causal=causal,
        logical_seqlen_q=sq_real,
        logical_seqlen_k=sk_real,
    )
    return out[:, :sq_real].contiguous()


def is_enabled() -> bool:
    """Whether the FP8 attention override was requested."""
    return options.flag("primus_turbo.fp8_attention")


def install(configured_backend: Optional[str] = None) -> bool:
    """Rebind the target backends to the FP8 kernel."""

    def probe() -> None:
        # Raise before rebinding anything; see install_override's ``probe``.
        from inspect import signature

        from primus_turbo.pytorch.ops import flash_attn_fp8_func

        params = signature(flash_attn_fp8_func).parameters
        required = {"logical_seqlen_q", "logical_seqlen_k"}
        missing = required.difference(params)
        if missing:
            raise RuntimeError(
                "Primus-Turbo FP8 attention lacks logical tail masking: "
                + ", ".join(sorted(missing))
            )

    return _backend_registry.install_override(
        kernel=flash_attn_fp8_pad64,
        override_name=OVERRIDE_NAME,
        log_prefix=_LOG_PREFIX,
        description="FP8 flash attention (flash_attn_fp8_func)",
        probe=probe,
        configured_backend=configured_backend,
        unsupported=unsupported_lengths,
    )
