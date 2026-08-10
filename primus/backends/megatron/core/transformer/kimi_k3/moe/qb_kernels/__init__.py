###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Quantile Balancing statistic kernels — one entry point per backend.

:class:`...k3_quantile_balancing.QuantileBalancingMixin` resolves its statistic
through :func:`resolve_qb_backend`, so this module is the one place that maps a
backend name to its functional entry. Layout mirrors :mod:`...kda_kernels` and
:mod:`...attn_res_kernels`, including the lazy loader and the hardware gate.

Backends
--------

``eager``
    Pure-PyTorch (:mod:`._eager`): :func:`compute_margin_histogram`. The
    numerical ground truth, always importable, and the only backend that runs on
    CPU. Nine kernel launches.
``flydsl``
    One fused FlyDSL kernel behind ``torch.topk`` (:mod:`._flydsl_v1`),
    gfx950 / CDNA4 only, loaded LAZILY via :func:`load_flydsl_qb_backend`.

Why this and not the expert GEMMs
---------------------------------
Stable Latent MoE is measured at 26.3 % of a forward + 19.2 % of a backward at
the scaled shape, but almost none of that is Kimi K3's own code. Per MoE layer
the grouped GEMM is 47.3 %, the router 13.6 %, the dispatcher 15.3 % and the
shared experts 8.6 % — all upstream Megatron — while the **one** thing
``StableLatentMoE`` adds, the latent RMSNorm, is **1.9 % of the layer and 0.98 %
of the step**, and it is already a fused Transformer Engine norm that Primus
Turbo additionally fuses into the grouped GEMM (``perf/RESULTS_fusion.tsv``:
``turbo_gg_rms`` is worth +26 % end to end at MBS 4). Writing a FlyDSL kernel
for it would duplicate upstream work and could not win.

Quantile Balancing is the opposite case: it is entirely ours, upstream has no
equivalent, and it is **2.7 % of the step** in nine launches over a 0.5 MB
tensor — i.e. almost pure launch and pass overhead, which is exactly what
fusion removes.

Both backends share one signature::

    hist, clamped = backend(
        scores, expert_bias,
        topk=..., num_bins=..., margin_min=..., margin_max=...,
    )

with ``scores: [num_tokens, num_experts]`` fp32 raw sigmoid scores,
``expert_bias: [num_experts]``, and returns ``[num_experts, num_bins]`` int64
counts plus a ``[2]`` int64 ``(below, above)`` pair. There is no backward: the
statistic runs under ``no_grad`` by construction.
"""

import functools
from typing import Optional

from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._eager import (
    compute_margin_histogram,
    margin_cutoff,
)

QB_BACKENDS = ("eager", "flydsl")


def _require_gfx950() -> None:
    """Raise :class:`ImportError` unless a gfx950 (CDNA4) device is visible."""
    import torch

    if not torch.cuda.is_available():
        raise ImportError("no ROCm/CUDA device is visible")
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        raise ImportError(f"device arch is {arch!r}, not gfx950 (CDNA4)")


def load_flydsl_qb_backend():
    """Lazily import the fused-FlyDSL histogram (:mod:`._flydsl_v1`).

    NOTE: the import is intentionally inline (optional, hardware-specific
    dependency); it must not be hoisted to module scope.
    """
    try:
        _require_gfx950()
        from primus.backends.megatron.core.transformer.kimi_k3.moe.qb_kernels._flydsl_v1 import (
            flydsl_compute_margin_histogram,
        )
    except ImportError as exc:
        raise ImportError(
            "quantile_balancing_backend = 'flydsl' requires the fused FlyDSL histogram "
            f"kernel (the `flydsl` pip package on gfx950 / CDNA4), which is unavailable: "
            f"{exc}. Select a different backend (eager)."
        ) from exc
    return flydsl_compute_margin_histogram


def resolve_qb_backend(name: str, max_tokens: Optional[int] = None):
    """Return the functional histogram entry for ``name``.

    Args:
        name: one of :data:`QB_BACKENDS`.
        max_tokens: shape guard for the ``flydsl`` backend, bound here so the
            returned callable has the same signature as the eager one and the
            caller's hot path carries no extra argument. ``None`` keeps the
            measured default; ``0`` disables the guard. Ignored by ``eager``.

    **The guard is not cosmetic.** The kernel is atomic-contention bound and was
    measured at **0.61x of eager at 32 768 tokens** (2.58x at 4096), so selecting
    it at a large micro-batch would slow training down while looking like an
    optimisation. Above the threshold the flydsl entry runs the eager path and
    warns once.

    Raises:
        ValueError: on an unknown name.
        ImportError: when the named backend's dependency or hardware is missing.
    """
    if name == "eager":
        return compute_margin_histogram
    if name == "flydsl":
        entry = load_flydsl_qb_backend()
        if max_tokens is None:
            return entry
        return functools.partial(entry, max_tokens=int(max_tokens))
    raise ValueError(f"Unknown Quantile Balancing backend {name!r}; expected one of {list(QB_BACKENDS)}.")


__all__ = [
    "compute_margin_histogram",
    "margin_cutoff",
    "load_flydsl_qb_backend",
    "QB_BACKENDS",
    "resolve_qb_backend",
]
