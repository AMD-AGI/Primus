###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Non-deterministic bf16 flash-attention override for the NeMo AutoModel
diffusion recipe (a determinism/performance trade-off knob, orthogonal to the
FP8/MXFP4 GEMM and FP8-attention hooks).

WHY:
  The aiter flash-attention kernel exposes a ``deterministic`` flag. The default
  (``deterministic=True``) uses a reproducible but slower backward; the
  ``deterministic=False`` path uses atomic accumulation for the dQ/dK/dV
  reduction, which is substantially faster in the backward while being
  numerically equivalent up to floating-point atomic-add ordering. For
  long-sequence models where the attention backward dominates the step, running
  the backward non-deterministically is a large, essentially free throughput
  win. The dispatch signature that diffusers uses has no ``deterministic``
  argument, so it cannot be threaded through -- hence this backend override.

WHAT this does (NO diffusers / Automodel fork):
  diffusers routes every model's attention through ``_AttentionBackendRegistry``:
  the config's ``model.attention_backend`` name selects a registered backend fn,
  and ``dispatch_attention_fn`` looks it up in ``_backends[name]`` at call time.
  This hook rebinds the backend *function* for the two backends the diffusion
  recipes use -- FLASH and AITER -- to a wrapper that, for the plain path, calls
  ``aiter.flash_attn_func(..., deterministic=False)`` directly (bf16 in, bf16
  out; no cast, no padding). The backend NAME, its ``_supported_arg_names`` and
  its constraint list are left untouched, so the config still selects the same
  backend by name; only the backward numerics/ordering change.

  Env-gated by ``PRIMUS_ATTN_NONDETERMINISTIC=1`` (default off = current
  deterministic behavior). Install BEFORE the recipe builds the transformer
  (i.e. before ``set_attention_backend``); the swap is a module-global dict entry
  resolved at forward, so as long as it is in place before the first forward it
  takes effect.

FALLBACK (correctness first): the non-deterministic path serves the plain,
  no-mask, dropout-free, non-context-parallel, single-output call -- exactly what
  the diffusion hero forwards use with ``cp_size: 1``. For anything it cannot
  serve identically (``_parallel_config`` set i.e. context parallelism,
  ``return_lse``, an additive ``attn_mask``, nonzero dropout, or a sliding
  ``window_size``) it transparently calls the ORIGINAL backend and warns ONCE per
  reason. So enabling this override never changes results for a call it does not
  handle.

Activation (env, no config schema change):
    PRIMUS_ATTN_NONDETERMINISTIC=1    enable the non-deterministic bf16 attention
                                      override (default off = no-op)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}


def is_nondeterministic_attn_enabled() -> bool:
    """Whether the non-deterministic bf16 attention override should be installed."""
    return os.getenv("PRIMUS_ATTN_NONDETERMINISTIC", "0") in _TRUTHY


# --------------------------------------------------------------------------- #
# Non-deterministic bf16 flash attention                                      #
# --------------------------------------------------------------------------- #
def _unwrap(out):
    """aiter may return (out, lse, ...) depending on flags; keep only the output."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _flash_attn_bf16_nondet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """bf16 flash attention with the non-deterministic (atomic) backward.

    q: (B, Sq, H, D), k/v: (B, Skv, H, D). Returns (B, Sq, H, D). No cast, no
    padding -- the only difference from the default backend is ``deterministic``.

    ``return_lse=True`` is required by aiter's autograd forward (it asserts the
    LSE is requested so it can be saved for the backward); we always request it
    and drop the LSE, returning only the attention output to the caller (the
    diffusion forwards here never consume the LSE).
    """
    import aiter

    out = aiter.flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=causal,
        deterministic=False, return_lse=True,
    )
    return _unwrap(out)


# --------------------------------------------------------------------------- #
# Backend override                                                            #
# --------------------------------------------------------------------------- #
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


def _make_nondet_attn_backend(orig_fn, backend_value: str, supports_window: bool):
    """Build the non-deterministic backend fn that replaces ``orig_fn``.

    Signature is a superset of both the FLASH and AITER backend signatures so
    that ``dispatch_attention_fn`` (which filters kwargs to each backend's
    ORIGINAL ``_supported_arg_names`` and then calls ``backend_fn(**kwargs)``) can
    drive it unchanged. ``orig_fn`` is captured for the fallback path;
    ``supports_window`` tells us whether ``orig_fn`` accepts ``window_size``
    (FLASH does, AITER does not).
    """

    def _fallback(query, key, value, attn_mask, dropout_p, is_causal, scale,
                  window_size, return_lse, _parallel_config):
        kw = dict(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            return_lse=return_lse,
            _parallel_config=_parallel_config,
        )
        if supports_window:
            kw["window_size"] = window_size
        return orig_fn(**kw)

    def nondet_attn_backend(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        window_size: tuple[int, int] = (-1, -1),
        return_lse: bool = False,
        _parallel_config=None,
    ):
        # Correctness-first fallbacks: use the original kernel for anything the
        # plain non-deterministic path cannot faithfully serve (warn once).
        if _parallel_config is not None:
            _warn_once(
                f"{backend_value}:cp",
                f"[PrimusAttn-NonDet] context parallelism active on backend "
                f"'{backend_value}' -> default attention fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if return_lse:
            _warn_once(
                f"{backend_value}:lse",
                f"[PrimusAttn-NonDet] return_lse requested on backend "
                f"'{backend_value}' -> default attention fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if attn_mask is not None:
            _warn_once(
                f"{backend_value}:mask",
                f"[PrimusAttn-NonDet] additive attn_mask on backend "
                f"'{backend_value}' -> default attention fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if dropout_p:
            _warn_once(
                f"{backend_value}:dropout",
                f"[PrimusAttn-NonDet] dropout_p={dropout_p} on backend "
                f"'{backend_value}' -> default attention fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if window_size not in ((-1, -1), None):
            _warn_once(
                f"{backend_value}:window",
                f"[PrimusAttn-NonDet] sliding window_size={window_size} on backend "
                f"'{backend_value}' -> default attention fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)

        return _flash_attn_bf16_nondet(query, key, value, softmax_scale=scale, causal=is_causal)

    nondet_attn_backend._primus_nondet_attn = True  # idempotence marker
    nondet_attn_backend._primus_orig_fn = orig_fn
    return nondet_attn_backend


# Backends the diffusion recipes use: FLASH (accepts window_size) and AITER
# (no window_size). Override both; only the active one is dispatched.
_TARGET_BACKENDS = (("FLASH", True), ("AITER", False))


def install() -> bool:
    """Install the non-deterministic attention override by rebinding diffusers
    backend functions.

    No-op (returns False) unless ``PRIMUS_ATTN_NONDETERMINISTIC`` is set. Modifies
    NO diffusers/Automodel source; only rebinds
    ``_AttentionBackendRegistry._backends`` entries at runtime. Idempotent. Call
    BEFORE the transformer's first forward (in practice before the recipe build /
    ``set_attention_backend``).
    """
    if not is_nondeterministic_attn_enabled():
        return False

    # Fail fast if aiter's flash-attention is unavailable so the run errors
    # clearly rather than silently running the default backward.
    import aiter  # noqa: F401

    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )

    reg = _AttentionBackendRegistry
    installed = []
    for name_str, supports_window in _TARGET_BACKENDS:
        name = getattr(AttentionBackendName, name_str, None)
        if name is None:
            continue
        orig = reg._backends.get(name)
        if orig is None:
            # Backend not registered in this diffusers build.
            continue
        if getattr(orig, "_primus_nondet_attn", False):
            installed.append(name.value)  # already installed
            continue
        reg._backends[name] = _make_nondet_attn_backend(orig, name.value, supports_window)
        # Leave reg._supported_arg_names[name] and reg._constraints[name] untouched:
        # dispatch filters kwargs to the ORIGINAL supported set (our fn accepts a
        # superset), and the qkv/device/shape checks still hold.
        installed.append(name.value)

    if not installed:
        logger.warning(
            "[PrimusAttn-NonDet] PRIMUS_ATTN_NONDETERMINISTIC set but neither FLASH "
            "nor AITER backend is registered in diffusers; override NOT active."
        )
        return False

    logger.info(
        "[PrimusAttn-NonDet] Installed non-deterministic bf16 flash-attention "
        "(aiter flash_attn_func, deterministic=False) override for backends: %s "
        "(config attention_backend name unchanged)",
        ", ".join(installed),
    )
    return True
