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

PAD-TO-64:
  The kernel block-scales q/k/v along the sequence with ``block_size=64`` and so
  requires ``seqlen % 64 == 0``; a non-conforming length fails with a reshape
  error. Sequences are zero-padded up to the next multiple of 64 and the real
  queries sliced back out. Zero-padded keys each add ``exp(0) = 1`` to the softmax
  denominator, so they dilute the result -- but by at most 63 keys against a
  sequence long enough to be worth running in FP8 at all, which is far inside
  FP8's own accuracy floor. No softmax correction is applied.

  The dilution is bounded but not zero, which is why ``pad_to_block`` is written
  to return the real length rather than letting callers recompute it.

Activation (env, no config schema change):
  PRIMUS_TURBO_FP8_ATTN=1    enable the override (default off = no-op)
"""
from __future__ import annotations

import logging

from primus.backends.nemo_automodel._env import env_flag
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


def flash_attn_fp8_pad64(q, k, v, softmax_scale=None, causal: bool = False):
    """FP8 flash attention, padding a non-conforming sequence up to a multiple of 64.

    q is (B, Sq, H, D) and k/v are (B, Skv, H, D); returns (B, Sq, H, D).
    """
    from primus_turbo.pytorch.ops import flash_attn_fp8_func

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q_pad, sq_real = pad_to_block(q)
    k_pad, _ = pad_to_block(k)
    v_pad, _ = pad_to_block(v)

    out = flash_attn_fp8_func(q_pad, k_pad, v_pad, softmax_scale=softmax_scale, causal=causal)
    return out[:, :sq_real].contiguous()


def is_enabled() -> bool:
    """Whether the FP8 attention override was requested."""
    return env_flag("PRIMUS_TURBO_FP8_ATTN")


def install() -> bool:
    """Rebind the target backends to the FP8 kernel."""

    def probe() -> None:
        # Fail fast rather than silently running bf16 attention.
        from primus_turbo.pytorch.ops import flash_attn_fp8_func  # noqa: F401

    return _backend_registry.install_override(
        kernel=flash_attn_fp8_pad64,
        override_name=OVERRIDE_NAME,
        log_prefix=_LOG_PREFIX,
        description="FP8 flash attention (flash_attn_fp8_func, pad-to-64)",
        probe=probe,
    )
