###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Non-deterministic bf16 flash-attention override.

A determinism-for-speed knob, orthogonal to the low-precision GEMM swaps.

WHY:
  The flash-attention kernel exposes a ``deterministic`` flag. The default
  reproducible backward is slower; ``deterministic=False`` uses atomic
  accumulation for the dQ/dK/dV reduction, which is numerically equivalent up to
  floating-point atomic-add ordering. For long-sequence models where the
  attention backward dominates the step, that is a large and essentially free
  saving. The dispatch signature diffusers uses has no ``deterministic``
  argument, so it cannot be threaded through the config -- hence an override.

  Since the only difference from the default backend is that one flag, this
  changes the ordering of a reduction and nothing else. Runs that need
  bit-reproducibility should leave it off, which is the default.

WHAT:
  Rebinds the FLASH and AITER backend functions to call Primus-Turbo's
  ``flash_attn_func`` with ``deterministic=False``. bf16 in, bf16 out; no cast and
  no padding. See ``_backend_registry`` for the seam, the fallback conditions and
  the mutual exclusion with the FP8 attention override.

Activation (env, no config schema change):
  PRIMUS_ATTN_NONDETERMINISTIC=1    enable the override (default off = no-op)
"""
from __future__ import annotations

import logging

from primus.backends.nemo_automodel._env import env_flag
from primus.backends.nemo_automodel.attention import _backend_registry

logger = logging.getLogger(__name__)

OVERRIDE_NAME = "nondeterministic-bf16"
_LOG_PREFIX = "[PrimusAttn-NonDet]"


def is_enabled() -> bool:
    """Whether the non-deterministic attention override was requested."""
    return env_flag("PRIMUS_ATTN_NONDETERMINISTIC")


def flash_attn_bf16_nondet(q, k, v, softmax_scale=None, causal: bool = False):
    """bf16 flash attention with the non-deterministic (atomic) backward.

    q is (B, Sq, H, D) and k/v are (B, Skv, H, D); returns (B, Sq, H, D).

    Turbo's wrapper requests the LSE from the underlying kernel itself (its
    autograd forward needs it saved for the backward) and returns a bare output
    tensor when ``return_lse`` is false, so this needs no unwrapping.
    """
    from primus_turbo.pytorch.ops import flash_attn_func

    return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal, deterministic=False)


def install() -> bool:
    """Rebind the target backends to the non-deterministic bf16 kernel."""

    def probe() -> None:
        # Fail fast rather than silently running the deterministic backward.
        from primus_turbo.pytorch.ops import flash_attn_func  # noqa: F401

    return _backend_registry.install_override(
        kernel=flash_attn_bf16_nondet,
        override_name=OVERRIDE_NAME,
        log_prefix=_LOG_PREFIX,
        description="non-deterministic bf16 flash attention (deterministic=False)",
        probe=probe,
    )
