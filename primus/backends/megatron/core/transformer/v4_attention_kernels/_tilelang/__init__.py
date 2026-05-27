###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-8 P49 — Tilelang V4 attention dispatcher (infra-only).

This module is the dispatch point for the plan-8 tilelang-backed V4
attention kernels (cr ∈ {0, 4, 128}).  At P49 it ships only the
**infra**:

* The pinned tilelang version probe (one-time module-import warning
  if the installed tilelang ≠ the plan-8 pin).
* The :func:`should_dispatch` predicate that takes the per-call
  ``enabled`` flag (from the ``use_v4_tilelang_attention`` /
  ``use_v4_tilelang_csa_attention`` config flags surfaced by
  ``DeepSeekV4TransformerConfig``).  Plan-8 P57 close-out 2: the
  dispatcher no longer reads ``PRIMUS_V4_TILELANG_ATTN`` — callers
  pass the config flag explicitly so a container without tilelang
  installed can simply leave the flag ``False`` and never trigger
  any tilelang import.
* The :func:`is_tilelang_kernel_available` predicate that lets each
  plan-8 phase (P50..P55) register its kernel name as it lands.
  Empty at P49 — every dispatcher call falls through to the
  plan-4 P25 / P26 Triton path with a one-time rank-0 warning.
* Four stub entry points that raise :class:`NotImplementedError`
  until the corresponding plan-8 phase lands them:

  - :func:`v4_attention_fwd_tilelang`           (P50 — cr=0 / cr=128 FWD)
  - :func:`v4_attention_bwd_tilelang`           (P51 — cr=0 / cr=128 BWD)
  - :func:`v4_csa_attention_fwd_tilelang`       (P54 — cr=4 FWD)
  - :func:`v4_csa_attention_bwd_tilelang`       (P55 — cr=4 BWD)

The dispatcher precedence (enforced inside the V4 attention
functional wrappers ``v4_attention`` / ``v4_csa_attention``) is:

.. code-block:: text

    cr ∈ {0, 128}:
      use_turbo_attention > use_v4_tilelang_attention
                          > use_v4_triton_attention > eager
    cr == 4:
      use_v4_tilelang_csa_attention > use_v4_triton_csa_attention > eager

At P49 (default-off + empty available-kernels set) the dispatcher
emits **no** behaviour change vs the plan-7 P48 anchor.  Plan-4..7
ratchet stays green by construction.

R6.2 — Tilelang is vendored at ``tilelang/`` and NEVER edited.  This
module imports it lazily so a missing install does not raise at
module import time; it only logs a one-time rank-0 warning the
first time a caller actually asks the dispatcher to use tilelang.
"""

from __future__ import annotations

import warnings
from typing import Any, Set

# ---------------------------------------------------------------------------
# Tilelang version pin (set by plan-8 P49; bump when upstream is updated)
# ---------------------------------------------------------------------------

# Read from ``tilelang/VERSION`` at plan-8 P49 land time.  Future plan-8
# phases (P50..P55) MUST re-run the G50..G55 ratchets before bumping
# this pin.
TILELANG_VERSION_PIN: str = "0.1.9+cuda.gitbcb2da33"


# ---------------------------------------------------------------------------
# Dispatch-enable signal (replaces the PRIMUS_V4_TILELANG_ATTN env knob)
# ---------------------------------------------------------------------------
#
# Plan-8 P57 close-out 2 (2026-05-15): the dispatcher no longer reads an
# environment variable.  Callers pass ``enabled`` to :func:`should_dispatch`
# from the V4 attention layer's config flag
# (``use_v4_tilelang_attention`` / ``use_v4_tilelang_csa_attention``),
# which is plumbed through :class:`DeepSeekV4TransformerConfig` from
# the run-time CLI args.  Default-False everywhere — running on a
# container without tilelang installed leaves the flag off and the
# dispatcher never imports tilelang.


# ---------------------------------------------------------------------------
# Per-kernel availability registry
# ---------------------------------------------------------------------------

# Mutable set of kernel names that plan-8 phases register as they
# land.  Empty at P49; populated by P50..P55 as each kernel ships
# via ``register_available_kernel(<name>)``.  Names follow the
# four-stub convention below.
_AVAILABLE_KERNELS: Set[str] = set()

_KNOWN_KERNEL_NAMES: Set[str] = {
    "v4_attention_fwd",  # P50 — dense / HCA FWD
    "v4_attention_bwd",  # P51 — dense / HCA BWD
    "v4_csa_attention_fwd",  # P54 — CSA FWD
    "v4_csa_attention_bwd",  # P55 — CSA BWD
}


def is_tilelang_kernel_available(name: str) -> bool:
    """Return True iff the plan-8 kernel ``name`` has landed.

    Empty at P49; each plan-8 phase (P50..P55) registers its
    kernel name via :func:`register_available_kernel` once the
    parity gate passes.
    """
    if name not in _KNOWN_KERNEL_NAMES:
        raise ValueError(
            f"Unknown tilelang kernel name {name!r}; expected one of " f"{sorted(_KNOWN_KERNEL_NAMES)}"
        )
    return name in _AVAILABLE_KERNELS


def register_available_kernel(name: str) -> None:
    """Mark a plan-8 tilelang kernel as available.

    Called at module-import time by the phase that lands the
    corresponding kernel.  P50: ``register_available_kernel("v4_attention_fwd")``
    inside ``v4_attention_fwd_tilelang.py``.
    """
    if name not in _KNOWN_KERNEL_NAMES:
        raise ValueError(
            f"Unknown tilelang kernel name {name!r}; expected one of " f"{sorted(_KNOWN_KERNEL_NAMES)}"
        )
    _AVAILABLE_KERNELS.add(name)


# Map kernel name -> submodule name to lazy-import.  Each submodule
# registers its kernel name + overrides the stub in this namespace on
# first import.
_KERNEL_SUBMODULES: dict[str, str] = {
    "v4_attention_fwd": "v4_attention_fwd_tilelang",
    "v4_attention_bwd": "v4_attention_bwd_tilelang",
    "v4_csa_attention_fwd": "v4_csa_attention_fwd_tilelang",
    "v4_csa_attention_bwd": "v4_csa_attention_bwd_tilelang",
}

_LAZY_LOADED: Set[str] = set()


def _lazy_load(name: str) -> bool:
    """Import the submodule that implements kernel ``name`` (if landed).

    Returns True iff the import succeeded.  Each plan-8 kernel
    submodule's import-time code:

    1. Calls :func:`register_available_kernel` to flip
       :func:`is_tilelang_kernel_available` to True.
    2. Replaces the module-level stub in this namespace via
       ``setattr(<this module>, name + "_tilelang", real_fn)``.

    Idempotent — subsequent calls no-op.
    """
    if name in _LAZY_LOADED:
        return name in _AVAILABLE_KERNELS
    _LAZY_LOADED.add(name)
    sub = _KERNEL_SUBMODULES.get(name)
    if sub is None:
        return False
    try:
        from importlib import import_module

        sub_mod = import_module(__name__ + "." + sub)
    except ImportError as exc:
        warnings.warn(
            f"[plan-8] tilelang submodule {sub!r} import failed ({exc}); "
            f"kernel {name!r} stays unavailable.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    # Override the stub: the submodule exposes a function with the same
    # name as the submodule (e.g. `v4_attention_fwd_tilelang.py`
    # exposes `v4_attention_fwd_tilelang`).  Importing the submodule
    # would otherwise set `_tilelang.v4_attention_fwd_tilelang` to
    # the SUBMODULE, shadowing the stub.  We restore the function
    # attribute here.
    real_fn = getattr(sub_mod, sub, None)
    if real_fn is None:
        warnings.warn(
            f"[plan-8] tilelang submodule {sub!r} does not expose a "
            f"function named {sub!r}; kernel {name!r} stays unavailable.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    import sys

    sys.modules[__name__].__dict__[sub] = real_fn
    return name in _AVAILABLE_KERNELS


# ---------------------------------------------------------------------------
# Tilelang import probe (lazy)
# ---------------------------------------------------------------------------

_TILELANG_PROBE_DONE: bool = False
_TILELANG_AVAILABLE: bool = False


def _probe_tilelang() -> bool:
    """Lazily import tilelang on first dispatcher call.

    Returns True iff the import succeeds at the pinned version.
    Emits a one-time warning when the installed version differs from
    the pin or when tilelang is not importable.
    """
    global _TILELANG_PROBE_DONE, _TILELANG_AVAILABLE
    if _TILELANG_PROBE_DONE:
        return _TILELANG_AVAILABLE
    _TILELANG_PROBE_DONE = True
    try:
        import tilelang  # noqa: F401
    except ImportError as exc:
        warnings.warn(
            f"[plan-8 P49] tilelang import failed ({exc}); falling back to "
            "the plan-4 P25 / P26 Triton kernels.  Install the pinned "
            f"tilelang {TILELANG_VERSION_PIN} to enable the plan-8 path.",
            RuntimeWarning,
            stacklevel=2,
        )
        _TILELANG_AVAILABLE = False
        return False
    installed = getattr(tilelang, "__version__", "<unknown>")
    if installed != TILELANG_VERSION_PIN:
        warnings.warn(
            f"[plan-8 P49] installed tilelang {installed!r} != pinned "
            f"{TILELANG_VERSION_PIN!r}; the plan-8 G50..G55 parity gates were "
            "run against the pin.  Proceeding anyway; please re-run the "
            "ratchets if the version drifted.",
            RuntimeWarning,
            stacklevel=2,
        )
    _TILELANG_AVAILABLE = True
    return True


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------


def cache_dir() -> str:
    """Tilelang autotune cache dir.

    Defaults to ``output/.tilelang_cache/v4/`` (gitignored under
    ``output/``).  Override via ``PRIMUS_V4_TILELANG_CACHE_DIR``.
    """
    import os as _os

    override = _os.environ.get("PRIMUS_V4_TILELANG_CACHE_DIR")
    if override:
        return override
    return _os.path.join("output", ".tilelang_cache", "v4")


# ---------------------------------------------------------------------------
# Stub entry points — replaced by P50 / P51 / P54 / P55
# ---------------------------------------------------------------------------


def _stub_raise(phase: str, name: str) -> None:
    raise NotImplementedError(
        f"plan-8 {phase} tilelang kernel {name!r} has not landed yet; "
        "the dispatcher should fall back to the plan-4 / plan-5 Triton "
        "path via `is_tilelang_kernel_available({name!r})` returning False."
    )


def v4_attention_fwd_tilelang(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Stub for plan-8 P50 (dense / HCA FWD)."""
    _stub_raise("P50", "v4_attention_fwd")


def v4_attention_bwd_tilelang(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Stub for plan-8 P51 (dense / HCA BWD)."""
    _stub_raise("P51", "v4_attention_bwd")


def v4_csa_attention_fwd_tilelang(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Stub for plan-8 P54 (CSA FWD)."""
    _stub_raise("P54", "v4_csa_attention_fwd")


def v4_csa_attention_bwd_tilelang(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
    """Stub for plan-8 P55 (CSA BWD)."""
    _stub_raise("P55", "v4_csa_attention_bwd")


# ---------------------------------------------------------------------------
# Dispatcher helpers — used by the v4_attention / v4_csa_attention wrappers
# ---------------------------------------------------------------------------

_FALLBACK_WARNED: Set[str] = set()


def _maybe_warn_fallback(kernel_name: str) -> None:
    """Emit a one-time rank-0 warning when a tilelang dispatch falls
    back to Triton.

    Hit conditions:

    * Config flag ``use_v4_tilelang_attention`` /
      ``use_v4_tilelang_csa_attention`` set but the kernel hasn't
      landed yet (``is_tilelang_kernel_available`` returns False).
    * Config flag set but tilelang import failed (e.g. missing
      install / version drift).

    Once per kernel name per process.  Banned-warning ratchet is
    extended in `plan-8/03-test-strategy.md` to allow this string
    (since it's the documented dispatcher fallback signal).
    """
    if kernel_name in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(kernel_name)
    # Best-effort rank-0 detection — the V4 attention wrappers may
    # run in unit tests without parallel-state initialised.
    rank = 0
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        pass
    if rank == 0:
        warnings.warn(
            f"[plan-8 P57] use_v4_tilelang_*=True but kernel "
            f"{kernel_name!r} is not available; falling back to the "
            "plan-4 / plan-5 Triton path.",
            RuntimeWarning,
            stacklevel=3,
        )


def should_dispatch(kernel_name: str, enabled: bool = False) -> bool:
    """Single dispatcher predicate used by ``v4_attention`` /
    ``v4_csa_attention`` wrappers.

    Returns True iff all three conditions hold:

    1. ``enabled`` is True (i.e. the caller passes the relevant
       ``use_v4_tilelang_*`` config flag).
    2. The named plan-8 kernel has been registered (P50..P55 land it).
    3. The tilelang import succeeds at the pinned version (probed
       lazily on the first dispatcher call).

    Otherwise emits a one-time rank-0 warning + returns False so the
    caller falls through to the Triton path.  ``enabled=False`` short-
    circuits before any tilelang import attempt, so containers without
    tilelang installed can leave the config flag off and the dispatcher
    never touches tilelang.

    Plan-8 P57 close-out 2 (2026-05-15): previously gated by the
    ``PRIMUS_V4_TILELANG_ATTN`` env var; now driven by the caller's
    config flag so default-off runs in tilelang-free containers do
    not need to set or unset any env knob.
    """
    if not enabled:
        return False
    if not _probe_tilelang():
        _maybe_warn_fallback(kernel_name)
        return False
    # Lazy-import the submodule that implements ``kernel_name``.
    # The submodule's import-time code flips
    # ``is_tilelang_kernel_available(...)`` to True and overrides
    # the stub function in this namespace.  Cheap on subsequent calls.
    _lazy_load(kernel_name)
    if not is_tilelang_kernel_available(kernel_name):
        _maybe_warn_fallback(kernel_name)
        return False
    return True


__all__ = [
    "TILELANG_VERSION_PIN",
    "cache_dir",
    "is_tilelang_kernel_available",
    "register_available_kernel",
    "should_dispatch",
    "v4_attention_bwd_tilelang",
    "v4_attention_fwd_tilelang",
    "v4_csa_attention_bwd_tilelang",
    "v4_csa_attention_fwd_tilelang",
]
