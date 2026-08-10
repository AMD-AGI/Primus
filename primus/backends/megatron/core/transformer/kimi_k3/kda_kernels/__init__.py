###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kimi Delta Attention kernels — single entry point for every backend.

``KimiDeltaAttention`` resolves its compute kernel through
:func:`resolve_kda_backend`, so this module is the one place that maps a
backend name to its functional entry. The layout mirrors
:mod:`primus.backends.megatron.core.transformer.v4_attention_kernels`.

Backends
--------

``eager``
    Pure-PyTorch chunkwise-parallel reference (:mod:`._eager`):
    :func:`eager_chunk_kda`. The numerical ground truth; always
    importable, no optional dependency, and differentiable.
``eager_recurrent``
    The ``O(T)`` literal recurrence (:mod:`._eager`):
    :func:`eager_recurrent_kda`. Correct by inspection and used as the
    oracle for the chunked form; far too slow for training.
``fla``
    ``flash-linear-attention``'s fused Triton ``chunk_kda``
    (:mod:`._fla`), loaded LAZILY via :func:`load_fla_kda_backend`.
``flydsl``
    Native FlyDSL kernel (:mod:`._flydsl_v1`), gfx950 / CDNA4 only,
    loaded LAZILY via :func:`load_flydsl_kda_backend`. Its intra-chunk
    score matrices come from a ``@flyc.kernel``; the surrounding
    projections and state sweep are batched torch GEMMs. See
    ``kda_kernels/README.md`` for what it does and does not accelerate.

``fla`` and ``flydsl`` are imported on demand only. Each is an optional dependency
(declared for the Megatron pretrain hook, not for the core package), and
importing it eagerly would make *any* ``import ...kda_kernels`` fail on a
build without it — even when the caller selected ``eager``. That is the
same rationale as the gluon/flydsl lazy loaders in
``v4_attention_kernels``.

All backends share one signature so the eager reference and a fused
kernel are interchangeable at the call site::

    o, final_state = backend(
        q, k, v, g, beta,
        scale=None, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, chunk_size=64,
    )

with ``q, k, g: [B, T, H, K]``, ``v: [B, T, H, V]``, ``beta: [B, T, H]``,
``g`` in log space and ``beta`` already sigmoid-activated. See
:mod:`._eager.reference` for the full contract.
"""

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._eager import (
    eager_chunk_kda,
    eager_recurrent_kda,
    kda_gate,
    kda_l2norm,
)

KDA_BACKENDS = ("eager", "eager_recurrent", "fla", "flydsl")


def load_fla_kda_backend():
    """Lazily import the ``fla`` fused-Triton KDA entry.

    ``flash-linear-attention`` is an optional dependency (see
    ``runner/helpers/hooks/train/pretrain/megatron/requirements-megatron.txt``).
    Deferring the import keeps selecting ``eager`` free of it and avoids
    crashing on a build without it. Call it only when a layer actually
    selects ``fla``.

    Returns ``fla_chunk_kda``. Raises :class:`ImportError` with an
    actionable message when ``fla`` (or its KDA ops, which landed after
    the gated-delta-rule ops) is unavailable.

    NOTE: the import is intentionally inline (optional dependency); it
    must not be hoisted to module scope.
    """
    try:
        from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._fla import (
            fla_chunk_kda,
        )
    except ImportError as exc:
        raise ImportError(
            "kda_backend = 'fla' requires flash-linear-attention with the KDA ops "
            f"(`fla.ops.kda.chunk_kda`), which failed to import: {exc}. Select a different "
            "backend (eager | eager_recurrent), or `pip install -U fla-core`."
        ) from exc
    return fla_chunk_kda


def _require_gfx950() -> None:
    """Raise :class:`ImportError` unless a gfx950 (CDNA4) device is visible.

    The FlyDSL KDA kernel is built for CDNA4 and its builder asserts the
    arch, but that assert fires at *build* time, deep inside a compile.
    Checking here makes selecting the backend on the wrong hardware fail
    at model-construction time with a message that names the fallbacks —
    the same reason
    ``v4_attention_kernels/__init__.py:137-161`` puts its hardware note in
    the loader's error rather than in the kernel.

    Raised as ``ImportError`` (not ``RuntimeError``) so it composes with
    the surrounding ``except ImportError`` and so callers have one
    exception type to catch for "this backend is unavailable here".
    """
    import torch

    if not torch.cuda.is_available():
        raise ImportError("no ROCm/CUDA device is visible")
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        raise ImportError(f"device arch is {arch!r}, not gfx950 (CDNA4)")


def load_flydsl_kda_backend():
    """Lazily import the native-FlyDSL KDA entry (:mod:`._flydsl_v1`).

    The backend hard-depends on the installed ``flydsl`` pip package and
    on gfx950 / CDNA4. Deferring the import keeps selecting any other
    backend free of it — and keeps ``import ...kda_kernels`` working on a
    build or GPU without flydsl. Call it only when a layer actually
    selects ``flydsl``.

    Returns ``flydsl_chunk_kda``. Raises :class:`ImportError` with an
    actionable message when flydsl or the hardware is unavailable.

    NOTE: the import is intentionally inline (optional, hardware-specific
    dependency); it must not be hoisted to module scope.
    """
    try:
        _require_gfx950()
        from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
            flydsl_chunk_kda,
        )
    except ImportError as exc:
        raise ImportError(
            "kda_backend = 'flydsl' requires the native FlyDSL KDA kernel (the `flydsl` "
            f"pip package on gfx950 / CDNA4), which is unavailable: {exc}. Select a "
            "different backend (eager | eager_recurrent | fla)."
        ) from exc
    return flydsl_chunk_kda


def resolve_kda_backend(name: str):
    """Return the functional KDA entry for ``name``.

    Args:
        name: one of :data:`KDA_BACKENDS`.

    Raises:
        ValueError: on an unknown name.
        ImportError: when the named backend's optional dependency is
            missing (``fla``) or the backend is not implemented
            (``flydsl``).
    """
    if name == "eager":
        return eager_chunk_kda
    if name == "eager_recurrent":
        return eager_recurrent_kda
    if name == "fla":
        return load_fla_kda_backend()
    if name == "flydsl":
        return load_flydsl_kda_backend()
    raise ValueError(f"Unknown KDA backend {name!r}; expected one of {list(KDA_BACKENDS)}.")


__all__ = [
    # eager reference (always available)
    "eager_chunk_kda",
    "eager_recurrent_kda",
    # shared input transforms the fused kernels fold in
    "kda_gate",
    "kda_l2norm",
    # fla fused Triton chunk kernel — lazily loaded
    "load_fla_kda_backend",
    # native FlyDSL kernel — lazily loaded
    "load_flydsl_kda_backend",
    # dispatch
    "KDA_BACKENDS",
    "resolve_kda_backend",
]
