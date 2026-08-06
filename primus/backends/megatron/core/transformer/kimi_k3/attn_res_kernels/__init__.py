###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Attention-residual mixer kernels — single entry point for every backend.

:class:`...attention_residual.AttentionResidualMixer` resolves its compute
kernel through :func:`resolve_attn_res_backend`, so this module is the one place
that maps a backend name to its functional entry. The layout deliberately
mirrors :mod:`...kda_kernels`, down to the lazy loader and the hardware gate.

Backends
--------

``eager``
    Pure-PyTorch (:mod:`._eager`): :func:`eager_attn_res_mix`. The numerical
    ground truth, always importable, differentiable, and the only backend that
    runs on CPU.
``flydsl``
    One fused FlyDSL kernel per direction (:mod:`._flydsl_v1`), gfx950 / CDNA4
    only, loaded LAZILY via :func:`load_flydsl_attn_res_backend`.

Why a kernel at all — the measurement
-------------------------------------
Attention residuals are ~29 K parameters per layer, which is a bad proxy for
time. Measured on the scaled config (24 layers / hidden 2048 / 32 experts,
seq 2048 x mbs 2 = 4096 tokens, 1 GPU, EP=1), the 47 mixers plus the output
head are **16.8 % of a forward+backward step** — 17.6 ms forward and 33.1 ms
backward out of 302.9 ms. The reason is materialisation, not arithmetic: the
eager path writes six full-size ``[tokens, num_blocks+1, hidden]``
intermediates, five of them fp32, so at ``num_blocks = 3`` it moves ~1.07 GB
for a 67 MB bf16 input. Numbers and the per-op breakdown are in
``flydsl_modules/FINDINGS.md``.

All backends share one signature, so the eager reference and the fused kernel
are interchangeable at the call site::

    mixed = backend(prefix_sum, block_residual, norm_weight, proj_weight, eps)

with ``prefix_sum: [*, hidden]``, ``block_residual: [*, num_blocks, hidden]``,
``norm_weight: [hidden]`` and ``proj_weight: [1, hidden]``. The two scorer
factors are passed separately rather than pre-fused because the released
checkpoint stores them separately; every backend folds them itself.
"""

from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._eager import (
    accum_dtype,
    eager_attn_res_mix,
    fused_score_weight,
)

ATTN_RES_BACKENDS = ("eager", "flydsl")


def _require_gfx950() -> None:
    """Raise :class:`ImportError` unless a gfx950 (CDNA4) device is visible.

    Same rationale as ``kda_kernels.__init__._require_gfx950``: the kernel
    builder asserts the arch, but that assert fires at *build* time deep inside
    a compile, so checking here makes selecting the backend on the wrong
    hardware fail at model-construction time with a message that names the
    fallback. Raised as ``ImportError`` so it composes with the surrounding
    ``except ImportError`` and callers have one exception type for "this backend
    is unavailable here".
    """
    import torch

    if not torch.cuda.is_available():
        raise ImportError("no ROCm/CUDA device is visible")
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    if not arch.startswith("gfx950"):
        raise ImportError(f"device arch is {arch!r}, not gfx950 (CDNA4)")


def load_flydsl_attn_res_backend():
    """Lazily import the fused-FlyDSL mixer (:mod:`._flydsl_v1`).

    The backend hard-depends on the installed ``flydsl`` pip package and on
    gfx950 / CDNA4. Deferring the import keeps selecting ``eager`` free of it
    and keeps ``import ...attn_res_kernels`` working on a build or GPU without
    flydsl.

    Returns ``flydsl_attn_res_mix``. Raises :class:`ImportError` with an
    actionable message when flydsl or the hardware is unavailable.

    NOTE: the import is intentionally inline (optional, hardware-specific
    dependency); it must not be hoisted to module scope.
    """
    try:
        _require_gfx950()
        from primus.backends.megatron.core.transformer.kimi_k3.attn_res_kernels._flydsl_v1 import (
            flydsl_attn_res_mix,
        )
    except ImportError as exc:
        raise ImportError(
            "attn_res_backend = 'flydsl' requires the fused FlyDSL attention-residual "
            f"kernel (the `flydsl` pip package on gfx950 / CDNA4), which is unavailable: "
            f"{exc}. Select a different backend (eager)."
        ) from exc
    return flydsl_attn_res_mix


def resolve_attn_res_backend(name: str):
    """Return the functional mixer entry for ``name``.

    Args:
        name: one of :data:`ATTN_RES_BACKENDS`.

    Raises:
        ValueError: on an unknown name.
        ImportError: when the named backend's optional dependency or hardware
            is missing.
    """
    if name == "eager":
        return eager_attn_res_mix
    if name == "flydsl":
        return load_flydsl_attn_res_backend()
    raise ValueError(
        f"Unknown attention-residual backend {name!r}; expected one of {list(ATTN_RES_BACKENDS)}."
    )


__all__ = [
    "eager_attn_res_mix",
    "fused_score_weight",
    "accum_dtype",
    "load_flydsl_attn_res_backend",
    "ATTN_RES_BACKENDS",
    "resolve_attn_res_backend",
]
