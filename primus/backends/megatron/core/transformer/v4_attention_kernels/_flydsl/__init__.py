# SPDX-License-Identifier: Apache-2.0
"""FlyDSL attention-kernel backend for DeepSeek-V4 (gfx950 / MI355X).

A *soft-dependency* alternate backend, structured exactly like the sibling
``_tilelang`` package: the V4 attention layer calls :func:`should_dispatch`
with the ``enabled`` flag from a config knob, and this module lazily imports
the FlyDSL kernel wrappers. If the ``flydsl`` runtime (or a kernel submodule)
is not importable, :func:`should_dispatch` returns ``False`` and the caller
transparently falls back to the in-tree Triton path -- so a container without
FlyDSL never breaks and never changes behaviour.

Kernels live under ``_flydsl/kernels/``. Forward-only for now (inference/eval);
training autograd is a follow-up.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Set

__all__ = [
    "should_dispatch",
    "is_flydsl_available",
    "v4_attention_fwd_flydsl",
    "v4_csa_attention_fwd_flydsl",
]

# Kernel names the layer may ask for, mirroring the _tilelang registry.
_KNOWN_KERNEL_NAMES: Set[str] = {
    "v4_attention_fwd",       # SWA / HCA forward (MQA launcher)
    "v4_csa_attention_fwd",   # CSA forward
    "v4_attention_bwd",       # present in-package; training-autograd hook TBD
    "v4_csa_attention_bwd",   # present in-package; training-autograd hook TBD
}

# Forward kernels that are actually wired (have a working adapter below).
_WIRED_FWD: Set[str] = {"v4_attention_fwd", "v4_csa_attention_fwd"}

_PROBE_DONE: bool = False
_FLYDSL_AVAILABLE: bool = False


def _probe_flydsl() -> bool:
    """Import the ``flydsl`` runtime once (cached); warn once on rank 0 if absent."""
    global _PROBE_DONE, _FLYDSL_AVAILABLE
    if _PROBE_DONE:
        return _FLYDSL_AVAILABLE
    _PROBE_DONE = True
    try:
        # The kernel wrappers add the FlyDSL build dir to sys.path at import;
        # allow an override for non-default install locations.
        src = os.environ.get("PRIMUS_V4_FLYDSL_SRC", "/workspace/FlyDSL-amd")
        import sys

        if src and src not in sys.path and os.path.isdir(src):
            sys.path.insert(0, src)
        import flydsl  # noqa: F401

        _FLYDSL_AVAILABLE = True
    except Exception as exc:  # ImportError, or a runtime/arch probe failure
        if int(os.environ.get("RANK", "0")) == 0:
            warnings.warn(
                f"[v4-flydsl] FlyDSL runtime unavailable ({exc!r}); FlyDSL "
                f"attention backend disabled, falling back to Triton.",
                RuntimeWarning,
                stacklevel=3,
            )
        _FLYDSL_AVAILABLE = False
    return _FLYDSL_AVAILABLE


def is_flydsl_available() -> bool:
    """True iff the FlyDSL runtime imported successfully (cached)."""
    return _probe_flydsl()


# --- lazy adapter cache -----------------------------------------------------
_FWD_MQA = None      # _launch_v4_attention_fwd_flydsl_mqa
_FWD_CSA = None      # _launch_v4_attention_fwd_csa


def _load_fwd_mqa():
    global _FWD_MQA
    if _FWD_MQA is None:
        from .kernels.v4_attention_fwd_flydsl_mqa import (
            _launch_v4_attention_fwd_flydsl_mqa as fn,
        )

        _FWD_MQA = fn
    return _FWD_MQA


def _load_fwd_csa():
    global _FWD_CSA
    if _FWD_CSA is None:
        from .kernels.v4_attention_fwd_flydsl_csa import _launch_v4_attention_fwd_csa as fn

        _FWD_CSA = fn
    return _FWD_CSA


def should_dispatch(kernel_name: str, *, enabled: bool) -> bool:
    """True iff FlyDSL should handle ``kernel_name`` (else Triton fallback).

    Short-circuits when ``enabled`` is False so the off path never imports FlyDSL.
    """
    if not enabled:
        return False
    if kernel_name not in _KNOWN_KERNEL_NAMES:
        raise ValueError(
            f"Unknown FlyDSL kernel {kernel_name!r}; expected one of "
            f"{sorted(_KNOWN_KERNEL_NAMES)}"
        )
    if kernel_name not in _WIRED_FWD:
        # bwd kernels are present in-package but lack a training-autograd hook
        return False
    if not _probe_flydsl():
        return False
    try:
        _load_fwd_csa() if kernel_name == "v4_csa_attention_fwd" else _load_fwd_mqa()
    except Exception as exc:
        if int(os.environ.get("RANK", "0")) == 0:
            warnings.warn(
                f"[v4-flydsl] kernel {kernel_name!r} import failed ({exc!r}); "
                f"falling back to Triton.",
                RuntimeWarning,
                stacklevel=3,
            )
        return False
    return True


# --- forward adapters (signatures match the dispatch sites) -----------------
def v4_attention_fwd_flydsl(
    q,
    k,
    v,
    *,
    sink: Optional[Any] = None,
    swa_window: int = 0,
    additive_mask: Optional[Any] = None,
    scale: float,
    hca_local_seqlen: int = 0,
):
    """SWA/HCA forward via FlyDSL. Returns the attention output tensor."""
    out, _lse = _load_fwd_mqa()(
        q, k, v, sink, int(swa_window), additive_mask, float(scale), int(hca_local_seqlen)
    )
    return out


def v4_csa_attention_fwd_flydsl(
    q,
    k_local,
    v_local,
    gathered,
    *,
    sparse_mask,
    sink: Optional[Any] = None,
    swa_window: int = 0,
    scale: float,
    attn_dropout: float = 0.0,
    training: bool = False,
):
    """CSA forward via FlyDSL (the 2.79x kernel). Returns the output tensor."""
    out, _lse = _load_fwd_csa()(
        q, k_local, v_local, gathered, sink, int(swa_window), sparse_mask, float(scale)
    )
    return out
