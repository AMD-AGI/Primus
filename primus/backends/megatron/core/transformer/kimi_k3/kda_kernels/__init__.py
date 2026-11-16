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
    Native FlyDSL kernels, loaded LAZILY via :func:`load_flydsl_kda_backend`,
    one implementation per architecture:

    * gfx950 / CDNA4 (:mod:`._flydsl_v1`): the intra-chunk score matrices come
      from a ``@flyc.kernel``; the surrounding projections and state sweep are
      batched torch GEMMs. See ``kda_kernels/README.md``.
    * gfx1250 (:mod:`._flydsl_gfx1250`): every step of the chunked delta rule
      (fla's ``chunk_kda`` algorithm with ``safe_gate`` and
      ``disable_recompute``) is a FlyDSL kernel, and so are the ops around it
      (:func:`load_flydsl_kda_ops`: q / k L2 norm, bounded gate, gated output
      RMSNorm, short causal conv + SiLU). It does not import ``fla``. It needs
      the bounded gate (``kda_gate_lower_bound`` in ``[-5, 0)``), head dim 128
      and sequence lengths that are multiples of 64; packed sequences go
      through the per-sample path, context parallelism needs ``fla``.

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


def load_fla_fused_ops():
    """Lazily import ``fla``'s fused kernels for the ops around the chunk kernel (``kda_fla_fused_ops``).

    Returns ``(l2norm, gate, rms_norm_gated)``, drop-in replacements for
    :func:`kda_l2norm`, :func:`kda_gate` and ``KimiGatedRMSNorm.forward``
    (``rms_norm_gated(x, gate, weight, eps)``). Same optional-dependency
    rationale and inline import as :func:`load_fla_kda_backend`.
    """
    try:
        from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._fla import (
            fla_kda_gate,
            fla_l2norm,
            fla_rms_norm_gated,
        )
    except ImportError as exc:
        raise ImportError(
            "kda_fla_fused_ops = true requires flash-linear-attention's fused l2norm, KDA gate and "
            f"gated RMSNorm kernels, which failed to import: {exc}. Set kda_fla_fused_ops: false, "
            "or `pip install -U fla-core`."
        ) from exc
    return fla_l2norm, fla_kda_gate, fla_rms_norm_gated


#: The architectures the ``flydsl`` backend has an implementation for.
FLYDSL_KDA_ARCHS = ("gfx950", "gfx1250")


def _flydsl_arch() -> str:
    """The visible device's architecture among :data:`FLYDSL_KDA_ARCHS`; :class:`ImportError` otherwise.

    The FlyDSL KDA kernels are built per architecture and their builders
    assert it, but that assert fires at *build* time, deep inside a compile.
    Checking here makes selecting the backend on the wrong hardware fail
    at model-construction time with a message that names the fallbacks —
    the same reason ``v4_attention_kernels/__init__.py`` puts its
    hardware note in the loader's error rather than in the kernel.

    Raised as ``ImportError`` (not ``RuntimeError``) so it composes with
    the surrounding ``except ImportError`` and so callers have one
    exception type to catch for "this backend is unavailable here".
    """
    import torch

    if not torch.cuda.is_available():
        raise ImportError("no ROCm/CUDA device is visible")
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    for a in FLYDSL_KDA_ARCHS:
        if arch.startswith(a):
            return a
    raise ImportError(f"device arch is {arch!r}, not one of {list(FLYDSL_KDA_ARCHS)}")


def load_flydsl_kda_backend():
    """Lazily import the native-FlyDSL KDA entry (:mod:`._flydsl_v1`).

    The backend hard-depends on the installed ``flydsl`` pip package and
    on gfx950 / CDNA4 (:mod:`._flydsl_v1`) or gfx1250 (:mod:`._flydsl_gfx1250`).
    Deferring the import keeps selecting any other
    backend free of it — and keeps ``import ...kda_kernels`` working on a
    build or GPU without flydsl. Call it only when a layer actually
    selects ``flydsl``.

    Returns ``flydsl_chunk_kda``. Raises :class:`ImportError` with an
    actionable message when flydsl or the hardware is unavailable.

    NOTE: the import is intentionally inline (optional, hardware-specific
    dependency); it must not be hoisted to module scope.
    """
    try:
        if _flydsl_arch() == "gfx1250":
            from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_gfx1250 import (
                flydsl_chunk_kda,
            )
        else:
            from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
                flydsl_chunk_kda,
            )
    except ImportError as exc:
        raise ImportError(
            "kda_backend = 'flydsl' requires the native FlyDSL KDA kernels (the `flydsl` "
            f"pip package on gfx950 / CDNA4 or gfx1250), which are unavailable: {exc}. Select a "
            "different backend (eager | eager_recurrent | fla)."
        ) from exc
    return flydsl_chunk_kda


def load_flydsl_kda_ops():
    """The gfx1250 ``flydsl`` backend's kernels for the ops around the chunk kernel, or ``None`` elsewhere.

    Returns ``(l2norm, gate, rms_norm_gated, causal_conv1d_silu)`` on gfx1250:
    drop-in replacements for :func:`kda_l2norm`, :func:`kda_gate`,
    ``KimiGatedRMSNorm.forward`` (``rms_norm_gated(x, gate, weight, eps)``) and
    the short conv + SiLU (``causal_conv1d_silu(x [B, T, D], weight [D, W])``).
    ``None`` on gfx950, whose ``flydsl`` backend keeps the default ops. Call it
    only when a layer selects ``flydsl``.
    """
    try:
        if _flydsl_arch() != "gfx1250":
            return None
        from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_gfx1250 import (
            flydsl_causal_conv1d_silu,
            ops_kernel,
        )
    except ImportError as exc:
        raise ImportError(
            f"kda_backend = 'flydsl' requires the native FlyDSL KDA kernels, which are unavailable: {exc}. "
            "Select a different backend (eager | eager_recurrent | fla)."
        ) from exc
    return ops_kernel.l2norm, ops_kernel.kda_gate, ops_kernel.rms_norm_gated, flydsl_causal_conv1d_silu


def resolve_kda_backend(name: str):
    """Return the functional KDA entry for ``name``.

    Args:
        name: one of :data:`KDA_BACKENDS`.

    Raises:
        ValueError: on an unknown name.
        ImportError: when the named backend's optional dependency is
            missing (``fla``) or the hardware is not supported
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
    # fla fused Triton chunk kernel and the ops around it — lazily loaded
    "load_fla_kda_backend",
    "load_fla_fused_ops",
    # native FlyDSL kernels — lazily loaded
    "FLYDSL_KDA_ARCHS",
    "load_flydsl_kda_backend",
    "load_flydsl_kda_ops",
    # dispatch
    "KDA_BACKENDS",
    "resolve_kda_backend",
]
