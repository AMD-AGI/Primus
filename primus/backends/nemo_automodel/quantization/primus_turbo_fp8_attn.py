###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-Turbo FP8 flash-attention override for the NeMo AutoModel diffusion
recipe (FP8 *attention*, orthogonal to the FP8/MXFP4 GEMM swap).

WHY:
  The GEMM low-precision hook (primus_turbo_fp8.py) swaps nn.Linear -> FP8/MXFP4
  but leaves scaled-dot-product attention in bf16. Attention is the dominant cost
  on the long-sequence Wan hero (61,200 tokens), so an FP8 attention kernel is the
  biggest remaining lever there. Primus-Turbo ships ``flash_attn_fp8_func`` (E4M3,
  block-scaled q/k/v, autograd fwd+bwd) -- the SAME kernel family as the GEMM path.
  There is NO FP4 attention kernel; FP8 attention is the only option and pairs with
  any GEMM precision (BF16 / FP8 / MXFP4).

WHAT this does (NO diffusers / Automodel fork):
  diffusers routes every model's attention through ``_AttentionBackendRegistry``:
  the config's ``model.attention_backend`` name selects a registered backend fn,
  and ``dispatch_attention_fn`` looks it up in ``_backends[name]`` at call time.
  This hook rebinds the backend *function* for the two backends our heroes use --
  FLASH (FLUX, ``flash_attn_func``) and AITER (Wan, ``aiter_flash_attn_func``) --
  to an FP8 flash-attn wrapper, while leaving the backend NAME, the registered
  ``_supported_arg_names`` and the constraint list untouched. So the config still
  says ``attention_backend: flash`` (FLUX) / ``aiter`` (Wan); only the numerics of
  the kernel change. Mirrors the nn.Linear->Float8Linear GEMM hook's philosophy.

  Env-gated by ``PRIMUS_TURBO_FP8_ATTN=1`` (default off = no-op). Install BEFORE the
  recipe builds the transformer (i.e. before ``set_attention_backend``); the swap is
  a module-global dict entry resolved at forward, so as long as it is in place before
  the first forward it takes effect.

PAD-TO-64:
  The FP8 flash kernel block-scales q/k/v along the sequence with block_size=64, so
  it requires ``seqlen % 64 == 0``. FLUX joint attn (4608) and Wan cross-attn kv
  (256) are already %64, but Wan self-attn seqlen is 61,200 (= 64*956 + 16, NOT %64)
  -> the raw kernel fails with a reshape error. We zero-pad q/k/v's sequence up to
  the next multiple of 64, run FP8 attention, and slice the real queries back.
  Padded-key dilution (padded keys are zeros -> each adds exp(0)=1 to the softmax
  denominator) is negligible because n_pad <= 63 << seqlen, well within FP8's own
  accuracy floor. No softmax correction is applied (an LSE-based one was tried and
  dropped as unnecessary).

FALLBACK (correctness first): the FP8 pad-64 kernel serves the plain, non-causal,
  no-mask, dropout-free, non-context-parallel path -- exactly what the FLUX/Wan
  hero forwards use with ``cp_size: 1``. For anything it cannot serve
  (``_parallel_config`` set i.e. context parallelism, ``return_lse``, an additive
  ``attn_mask``, nonzero dropout, or a sliding ``window_size``) it transparently
  calls the ORIGINAL bf16 backend and warns ONCE per reason. So enabling FP8
  attention never changes results for a call it doesn't handle -- e.g. a Wan run
  bumped to ``cp_size>1`` for memory will run bf16 attention (revisit: a CP-aware
  FP8 attention forward_op is future work).

Activation (env, no config schema change):
    PRIMUS_TURBO_FP8_ATTN=1    enable the FP8 attention override (default off = no-op)

Launch: install() is called by the backend entrypoint when the env is set; the
  config keeps ``attention_backend: flash`` (FLUX) / ``aiter`` (Wan) unchanged.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}

# The FP8 flash kernel block-scales q/k/v along the sequence with this block size,
# so the sequence length must be a multiple of it (see module docstring).
_ATTN_BLOCK = 64


def is_fp8_attn_enabled() -> bool:
    """Whether the Primus-Turbo FP8 attention override should be installed."""
    return os.getenv("PRIMUS_TURBO_FP8_ATTN", "0") in _TRUTHY


# --------------------------------------------------------------------------- #
# Pad-to-64 FP8 flash attention                                               #
# --------------------------------------------------------------------------- #
def _pad_seq(t: torch.Tensor, mult: int = _ATTN_BLOCK) -> tuple[torch.Tensor, int]:
    """Zero-pad a (B, S, H, D) tensor's seq dim (dim=1) up to a multiple of ``mult``.

    Returns (padded_tensor, real_seqlen).
    """
    s = t.shape[1]
    s_pad = ((s + mult - 1) // mult) * mult
    if s_pad == s:
        return t, s
    # F.pad pads from the last dim backwards: (D_lo, D_hi, H_lo, H_hi, S_lo, S_hi).
    return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, s_pad - s)), s


def _flash_attn_fp8_pad64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """FP8 flash attention with automatic pad-to-64 of a non-%64 sequence.

    q: (B, Sq, H, D), k/v: (B, Skv, H, D). Returns (B, Sq, H, D). Zero-pads q/k/v'
    sequence dims up to a multiple of 64 (the kernel's block-scale requirement) and
    slices the real queries back. Padded-key dilution is negligible (module docstring).
    """
    from primus_turbo.pytorch.ops import flash_attn_fp8_func

    b, sq, h, d = q.shape
    if softmax_scale is None:
        softmax_scale = d ** -0.5

    qp, sq_real = _pad_seq(q)
    kp, _ = _pad_seq(k)
    vp, _ = _pad_seq(v)

    out = flash_attn_fp8_func(qp, kp, vp, softmax_scale=softmax_scale, causal=causal)
    return out[:, :sq_real].contiguous()


# --------------------------------------------------------------------------- #
# Backend override                                                            #
# --------------------------------------------------------------------------- #
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


def _make_fp8_attn_backend(orig_fn, backend_value: str, supports_window: bool):
    """Build the FP8-attention backend fn that replaces ``orig_fn`` in the registry.

    Signature is a superset of both the FLASH and AITER backend signatures so that
    ``dispatch_attention_fn`` (which filters kwargs to each backend's ORIGINAL
    ``_supported_arg_names`` and then calls ``backend_fn(**kwargs)``) can drive it
    unchanged. ``orig_fn`` is captured for the fallback path; ``supports_window``
    tells us whether ``orig_fn`` accepts ``window_size`` (FLASH does, AITER doesn't).
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

    def fp8_attn_backend(
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
        # Correctness-first fallbacks: use the original bf16 kernel for anything the
        # FP8 pad-64 path cannot faithfully serve (warn once per reason).
        if _parallel_config is not None:
            _warn_once(
                f"{backend_value}:cp",
                f"[PrimusTurbo-FP8Attn] context parallelism active on backend "
                f"'{backend_value}' (no FP8 CP path yet) -> bf16 attention fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if return_lse:
            _warn_once(
                f"{backend_value}:lse",
                f"[PrimusTurbo-FP8Attn] return_lse requested on backend "
                f"'{backend_value}' (FP8 pad-64 has no LSE) -> bf16 attention fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if attn_mask is not None:
            _warn_once(
                f"{backend_value}:mask",
                f"[PrimusTurbo-FP8Attn] additive attn_mask on backend "
                f"'{backend_value}' unsupported by FP8 flash-attn -> bf16 fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if dropout_p:
            _warn_once(
                f"{backend_value}:dropout",
                f"[PrimusTurbo-FP8Attn] dropout_p={dropout_p} on backend "
                f"'{backend_value}' unsupported by FP8 flash-attn -> bf16 fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)
        if window_size not in ((-1, -1), None):
            _warn_once(
                f"{backend_value}:window",
                f"[PrimusTurbo-FP8Attn] sliding window_size={window_size} on backend "
                f"'{backend_value}' unsupported by FP8 flash-attn -> bf16 fallback.",
            )
            return _fallback(query, key, value, attn_mask, dropout_p, is_causal,
                             scale, window_size, return_lse, _parallel_config)

        return _flash_attn_fp8_pad64(query, key, value, softmax_scale=scale, causal=is_causal)

    fp8_attn_backend._primus_fp8_attn = True  # idempotence marker
    fp8_attn_backend._primus_orig_fn = orig_fn
    return fp8_attn_backend


# Backends our diffusion heroes use: FLUX -> FLASH (accepts window_size),
# Wan -> AITER (no window_size). Override both; only the active one is dispatched.
_TARGET_BACKENDS = (("FLASH", True), ("AITER", False))


def install() -> bool:
    """Install the FP8 attention override by rebinding diffusers backend functions.

    No-op (returns False) unless ``PRIMUS_TURBO_FP8_ATTN`` is set. Modifies NO
    diffusers/Automodel source; only rebinds ``_AttentionBackendRegistry._backends``
    entries at runtime. Idempotent. Call BEFORE the transformer's first forward
    (in practice before the recipe build / ``set_attention_backend``).
    """
    if not is_fp8_attn_enabled():
        return False

    # Fail fast if the FP8 attention op is unavailable so the run errors clearly
    # rather than silently running bf16 attention.
    from primus_turbo.pytorch.ops import flash_attn_fp8_func  # noqa: F401

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
            # Backend not registered in this diffusers build (e.g. aiter absent).
            continue
        if getattr(orig, "_primus_fp8_attn", False):
            installed.append(name.value)  # already installed
            continue
        reg._backends[name] = _make_fp8_attn_backend(orig, name.value, supports_window)
        # Leave reg._supported_arg_names[name] and reg._constraints[name] untouched:
        # dispatch filters kwargs to the ORIGINAL supported set (our fn accepts a
        # superset), and the qkv-bf16/device/shape checks still hold (inputs arrive
        # bf16; the wrapper casts to FP8 internally).
        installed.append(name.value)

    if not installed:
        logger.warning(
            "[PrimusTurbo-FP8Attn] PRIMUS_TURBO_FP8_ATTN set but neither FLASH nor "
            "AITER backend is registered in diffusers; FP8 attention NOT active."
        )
        return False

    logger.info(
        "[PrimusTurbo-FP8Attn] Installed FP8 flash-attention (flash_attn_fp8_func, "
        "pad-to-64) override for backends: %s (config attention_backend name unchanged)",
        ", ".join(installed),
    )
    return True
