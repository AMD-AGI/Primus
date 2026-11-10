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
  no padding. Under context parallelism (``fsdp.cp_size`` > 1) the same kernels
  run through Primus-Turbo's ``flash_attn_usp_func`` on diffusers' ulysses and
  ring groups, instead of the backend's own CP path (a Turbo without it keeps
  the backend's CP path, with a warning). See ``_backend_registry``
  for the seam, the fallback conditions and the mutual exclusion with the FP8
  attention override.

Settings (module config):
  primus_turbo.nondeterministic_attention: false   enable the override (default off)
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from primus.backends.automodel import options
from primus.backends.automodel.attention import _backend_registry

logger = logging.getLogger(__name__)

OVERRIDE_NAME = "nondeterministic-bf16"
_LOG_PREFIX = "[PrimusAttn-NonDet]"


def is_enabled() -> bool:
    """Whether the non-deterministic attention override was requested."""
    return options.flag("primus_turbo.nondeterministic_attention")


def flash_attn_bf16_nondet(q, k, v, softmax_scale=None, causal: bool = False):
    """bf16 flash attention with the non-deterministic (atomic) backward.

    q is (B, Sq, H, D) and k/v are (B, Skv, H, D); returns (B, Sq, H, D).

    Turbo's wrapper requests the LSE from the underlying kernel itself (its
    autograd forward needs it saved for the backward) and returns a bare output
    tensor when ``return_lse`` is false, so this needs no unwrapping.
    """
    from primus_turbo.pytorch.ops import flash_attn_func

    return flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal, deterministic=False)


def flash_attn_bf16_nondet_cp(q, k, v, softmax_scale=None, causal: bool = False, parallel_config=None):
    """The same kernels under diffusers' context parallelism.

    q/k/v are this rank's sequence shard, (B, S/cp, H, D). Primus-Turbo's
    sequence-parallel entry exchanges heads inside the ulysses group and runs its
    ring over the ring group, with the forward and backward ``flash_attn_func``
    uses. Both groups are passed even at size 1: Turbo takes the ulysses degree
    from its group's size, and it reads a missing ring group as the default
    (world) group, which also spans the data-parallel ranks.
    """
    from primus_turbo.pytorch.ops import flash_attn_usp_func

    cp = parallel_config.context_parallel_config
    return flash_attn_usp_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=False,
        ulysses_group=cp._ulysses_mesh.get_group(),
        ring_group=cp._ring_mesh.get_group(),
    )


def cp_kernel_if_available() -> Optional[Callable]:
    """The CP kernel, or None on a Primus-Turbo without ``flash_attn_usp_func``."""
    try:
        from primus_turbo.pytorch.ops import flash_attn_usp_func  # noqa: F401
    except ImportError:
        return None
    return flash_attn_bf16_nondet_cp


def install(configured_backend: Optional[str] = None) -> bool:
    """Rebind the target backends to the non-deterministic bf16 kernel."""

    def probe() -> None:
        # Raise before rebinding anything; see install_override's ``probe``.
        from primus_turbo.pytorch.ops import flash_attn_func  # noqa: F401

    cp_kernel = cp_kernel_if_available()
    description = "non-deterministic bf16 flash attention (deterministic=False)"
    installed = _backend_registry.install_override(
        kernel=flash_attn_bf16_nondet,
        override_name=OVERRIDE_NAME,
        log_prefix=_LOG_PREFIX,
        description=description + (", also under context parallelism" if cp_kernel else ""),
        probe=probe,
        configured_backend=configured_backend,
        cp_kernel=cp_kernel,
    )
    if installed and cp_kernel is None:
        logger.warning(
            "%s this Primus-Turbo has no flash_attn_usp_func, so context-parallel calls "
            "keep the original kernel.",
            _LOG_PREFIX,
        )
    return installed
