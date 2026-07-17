###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused ``torch.stack + transpose(1, 2) + contiguous`` for per-expert
GroupedMLP weights (plan-6 P34).

The eager implementation in :class:`PrimusTurboGroupedMLP._stack_grouped_linear_weight`:

.. code-block:: python

    weights = [getattr(module, f"weight{i}") for i in range(E)]
    return torch.stack(weights, dim=0).transpose(1, 2).contiguous()

does **two** full passes over the per-expert weight data:

* ``torch.stack`` allocates ``[E, K, N]`` and issues ``E`` per-expert
  ``copy_`` calls (one full pass aggregate);
* ``.contiguous()`` after ``.transpose(1, 2)`` allocates a second
  ``[E, N, K]`` buffer and writes the transposed copy (a second full
  pass).

At V4-Flash EP=8 widths (``E=32``, fc1: ``K=4096, N=4096``; fc2:
``K=4096, N=2048``, bf16) the P32 final trace attributes
``hipMemcpyWithStream`` **289.6 ms / 32 calls** to this op chain
(``2 stack ops per layer × 8 layers × 2 (FWD + BWD VJP) = 32``). At ~9 ms
per call writing 512 MiB the effective bandwidth is only ~57 GB/s — far
below the MI355X HBM peak — because each call serializes E small
allocations plus a separate transpose copy.

This module collapses the two passes into one Triton kernel that:

1. Indexes per-expert weight tensors via an ``int64`` pointer tensor
   (``weight_ptrs[e] = weights[e].data_ptr()``) — Triton's
   ``tl.load(...).to(tl.pointer_type(...))`` idiom (same pattern as the
   upstream grouped-GEMM tutorial).
2. Per program processes a ``[BLOCK_K, BLOCK_N]`` tile of one expert,
   doing a tile-level transpose: reads ``weight[e][k, n]`` (row-major
   ``[K, N]``) and writes ``out[e][n, k]`` (row-major ``[E, N, K]``).
3. BWD is the inverse — reads a ``[BLOCK_N, BLOCK_K]`` tile of ``dout
   [e, n, k]`` and writes ``dweight[e][k, n]``.

The two layouts form a bijection so **no atomics are needed in either
direction**.

Nomenclature note (matches plan-6 P34 design doc):

* ``K`` = ``N_out`` (``weight.shape[0]`` for ``nn.Linear`` — output features)
* ``N`` = ``N_in``  (``weight.shape[1]`` for ``nn.Linear`` — input  features)

The eager output is ``[E, N, K]`` (after ``.transpose(1, 2)`` on the
stacked ``[E, K, N]``); the Triton path produces the same layout.

Gating: routed through :class:`PrimusTurboGroupedMLP._stack_grouped_linear_weight`
when ``PRIMUS_STACK_GROUPED_WEIGHT_TRITON != "0"`` (default-on).  Set to
``"0"`` to fall back to the eager ``torch.stack + transpose + contiguous``
chain (kept in tree for A/B testing and as the reference path for the
G37 unit tests).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton dtype mapping
# ---------------------------------------------------------------------------

_TORCH_TO_TL_DTYPE = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _triton_dtype(t: torch.dtype):
    try:
        return _TORCH_TO_TL_DTYPE[t]
    except KeyError as exc:
        raise TypeError(
            f"stack_grouped_weight: unsupported dtype {t}; " f"expected one of {list(_TORCH_TO_TL_DTYPE)}"
        ) from exc


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _stack_grouped_weight_fwd_kernel(
    WEIGHT_PTRS,  # [E] int64 — tl.load() yields each expert's data_ptr()
    OUT,  # [E, N, K] contiguous output, row-major (strides [N*K, K, 1])
    E,
    K,
    N,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Per-expert ``[K, N] -> [N, K]`` tile-transpose, fused across experts.

    Each program writes one ``[BLOCK_K, BLOCK_N]`` tile of one expert's
    output.  Grid: ``(E, ceil(K / BLOCK_K), ceil(N / BLOCK_N))``.

    Read  : ``weight[expert][k, n]`` from per-expert pointer, stride ``[N, 1]``.
    Write : ``out[expert][n, k]``,    stride ``[E*N*K -> N*K, K, 1]``.

    Both load and store carry bounds masks so non-multiple-of-BLOCK shapes
    are supported.
    """
    pid_e = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)

    src_ptr = tl.load(WEIGHT_PTRS + pid_e).to(tl.pointer_type(DTYPE))

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    mask_n = offs_n < N

    src_offsets = offs_k[:, None] * N + offs_n[None, :]
    tile = tl.load(
        src_ptr + src_offsets,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0,
    )

    dst_offsets = pid_e * (N * K) + offs_n[None, :] * K + offs_k[:, None]
    tl.store(
        OUT + dst_offsets,
        tile,
        mask=mask_k[:, None] & mask_n[None, :],
    )


@triton.jit
def _stack_grouped_weight_bwd_kernel(
    DWEIGHT_PTRS,  # [E] int64 — tl.load() yields each expert's grad data_ptr()
    DOUT,  # [E, N, K] grad tensor (contiguous, same layout as FWD OUT)
    E,
    K,
    N,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Inverse of the FWD: reads ``dout[expert][n, k]`` and writes
    ``dweight[expert][k, n]``.

    Grid mirrors the FWD; the kernel is a pure bijection memcpy so no
    atomics needed, and the BLOCK tile is the same shape.
    """
    pid_e = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = tl.program_id(2)

    dst_ptr = tl.load(DWEIGHT_PTRS + pid_e).to(tl.pointer_type(DTYPE))

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_k = offs_k < K
    mask_n = offs_n < N

    src_offsets = pid_e * (N * K) + offs_n[None, :] * K + offs_k[:, None]
    tile = tl.load(
        DOUT + src_offsets,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0,
    )

    dst_offsets = offs_k[:, None] * N + offs_n[None, :]
    tl.store(
        dst_ptr + dst_offsets,
        tile,
        mask=mask_k[:, None] & mask_n[None, :],
    )


# ---------------------------------------------------------------------------
# Block-size autotune
#
# The kernel is bandwidth-bound; only a small handful of block sizes are
# worth scanning.  ``BLOCK_K = BLOCK_N = 64`` is the default — it fits
# comfortably in LDS at bf16 (64 * 64 * 2 = 8 KiB / program) and matches
# the typical tile size MI355X HBM controllers like for transpose copies.
# Larger blocks ((128, 64), (64, 128), (128, 128)) widen the per-program
# tile and reduce launch count when E is small; smaller (32, 32) is
# kept as a safety floor for the fast-tier shapes.
# ---------------------------------------------------------------------------

_BLOCK_CANDIDATES: Tuple[Tuple[int, int], ...] = (
    (32, 32),
    (64, 64),
    (128, 64),
    (64, 128),
)


def _pick_block(K: int, N: int) -> Tuple[int, int]:
    """Pick a ``(BLOCK_K, BLOCK_N)`` tile that divides reasonably into K, N.

    Conservative heuristic — we do not run a full autotune sweep at module
    import to keep cold start cheap.  The selected tile is always one of
    :data:`_BLOCK_CANDIDATES`, picked as the largest block that does not
    leave more than half the program's elements masked off on the small
    dimension (i.e. the last program's tile-fill is at least 50 %).
    """
    best: Tuple[int, int] = (64, 64)
    best_score = -1.0
    for bk, bn in _BLOCK_CANDIDATES:
        # Fraction of useful work in the last program along each axis;
        # exact-multiple tiles get 1.0; partial tiles get the fraction.
        k_full = (K // bk) * bk
        k_tail = K - k_full
        k_fill = 1.0 if k_tail == 0 else max(k_tail / bk, 0.5)
        n_full = (N // bn) * bn
        n_tail = N - n_full
        n_fill = 1.0 if n_tail == 0 else max(n_tail / bn, 0.5)
        # Score = tile area × tile-fill product → prefer bigger tiles when
        # fill is high.
        score = (bk * bn) * k_fill * n_fill
        if score > best_score:
            best_score = score
            best = (bk, bn)
    return best


# ---------------------------------------------------------------------------
# autograd.Function entry point
# ---------------------------------------------------------------------------


class StackGroupedWeightFn(torch.autograd.Function):
    """Fused ``torch.stack(weights).transpose(1, 2).contiguous()`` with
    in-kernel ``[K, N] -> [N, K]`` transpose, fused across all experts.

    Inputs (variadic):
        ``*weights``: ``E`` per-expert tensors, each ``[K, N]``, all same
        ``dtype`` and ``device``, all contiguous (Megatron's parameter
        allocator always returns contiguous; a defensive assertion is
        kept in :meth:`forward` regardless).

    Output:
        ``[E, N, K]`` contiguous tensor — bit-identical to the eager
        ``torch.stack(weights, dim=0).transpose(1, 2).contiguous()`` chain
        (the operation is a pure layout transform so there is no fp
        rounding to worry about).

    BWD returns one ``[K, N]`` grad tensor per input weight; PyTorch's
    autograd then writes each into ``weights[i].grad``.
    """

    @staticmethod
    def forward(ctx, *weights: torch.Tensor) -> torch.Tensor:
        if not weights:
            raise ValueError("StackGroupedWeightFn requires at least one weight tensor")

        first = weights[0]
        if first.ndim != 2:
            raise ValueError(
                f"StackGroupedWeightFn: each weight must be 2D, got " f"weight0.shape={tuple(first.shape)}"
            )
        K, N = int(first.shape[0]), int(first.shape[1])
        dtype = first.dtype
        device = first.device

        for i, w in enumerate(weights):
            if w.shape != first.shape:
                raise ValueError(
                    f"StackGroupedWeightFn: weight{i}.shape={tuple(w.shape)} "
                    f"differs from weight0.shape={tuple(first.shape)}"
                )
            if w.dtype is not dtype:
                raise TypeError(
                    f"StackGroupedWeightFn: weight{i}.dtype={w.dtype} " f"differs from weight0.dtype={dtype}"
                )
            if w.device != device:
                raise RuntimeError(
                    f"StackGroupedWeightFn: weight{i}.device={w.device} "
                    f"differs from weight0.device={device}"
                )
            if not w.is_contiguous():
                raise ValueError(
                    f"StackGroupedWeightFn: weight{i} must be contiguous; "
                    "Megatron's parameter allocator always returns contiguous "
                    "tensors so a non-contiguous input here indicates a bug "
                    "upstream"
                )

        E = len(weights)

        weight_ptrs = torch.tensor([w.data_ptr() for w in weights], dtype=torch.int64, device=device)

        out = torch.empty(E, N, K, dtype=dtype, device=device)

        block_k, block_n = _pick_block(K, N)
        grid = (E, triton.cdiv(K, block_k), triton.cdiv(N, block_n))
        _stack_grouped_weight_fwd_kernel[grid](
            weight_ptrs,
            out,
            E,
            K,
            N,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            DTYPE=_triton_dtype(dtype),
        )

        ctx.E = E
        ctx.K = K
        ctx.N = N
        ctx.dtype = dtype
        ctx.device = device
        ctx.block_k = block_k
        ctx.block_n = block_n
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):  # type: ignore[override]
        E = ctx.E
        K = ctx.K
        N = ctx.N
        dtype = ctx.dtype
        device = ctx.device

        if not dout.is_contiguous():
            dout = dout.contiguous()
        if tuple(dout.shape) != (E, N, K):
            raise ValueError(
                f"StackGroupedWeightFn backward: dout.shape={tuple(dout.shape)} "
                f"!= expected (E={E}, N={N}, K={K})"
            )
        if dout.dtype is not dtype:
            dout = dout.to(dtype)

        dweights: List[torch.Tensor] = [torch.empty(K, N, dtype=dtype, device=device) for _ in range(E)]
        dweight_ptrs = torch.tensor([dw.data_ptr() for dw in dweights], dtype=torch.int64, device=device)

        block_k, block_n = ctx.block_k, ctx.block_n
        grid = (E, triton.cdiv(K, block_k), triton.cdiv(N, block_n))
        _stack_grouped_weight_bwd_kernel[grid](
            dweight_ptrs,
            dout,
            E,
            K,
            N,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            DTYPE=_triton_dtype(dtype),
        )

        return tuple(dweights)


# ---------------------------------------------------------------------------
# Public Python entry points
# ---------------------------------------------------------------------------


_ENV_FLAG = "PRIMUS_STACK_GROUPED_WEIGHT_TRITON"


def is_triton_path_enabled() -> bool:
    """Returns ``True`` when the Triton path is active.

    Default-on; set ``PRIMUS_STACK_GROUPED_WEIGHT_TRITON=0`` to fall back
    to the eager ``torch.stack + transpose + contiguous`` chain.  Treated
    as a soft env (not a model-config flag) so an operator can A/B the
    Triton path on a live training job without re-launching with a
    different YAML.
    """
    return os.environ.get(_ENV_FLAG, "1") != "0"


def eager_stack_grouped_weight(weights: List[torch.Tensor]) -> torch.Tensor:
    """Reference implementation — the exact eager chain the Triton path
    replaces.  Kept exported for unit tests and for the env-flag-off path
    in :class:`PrimusTurboGroupedMLP`.
    """
    return torch.stack(weights, dim=0).transpose(1, 2).contiguous()


def stack_grouped_weight(weights: List[torch.Tensor]) -> torch.Tensor:
    """Dispatch entry point used by :class:`PrimusTurboGroupedMLP`.

    Routes through the Triton path when :func:`is_triton_path_enabled`
    returns True; otherwise calls :func:`eager_stack_grouped_weight`.
    The Triton path uses an :class:`autograd.Function` so BWD scatters
    the gradient back to each ``weights[i].grad`` via the inverse
    transpose kernel; the eager path inherits PyTorch's default VJP
    chain for ``torch.stack + transpose + contiguous``.
    """
    if is_triton_path_enabled():
        return StackGroupedWeightFn.apply(*weights)
    return eager_stack_grouped_weight(weights)


__all__ = [
    "StackGroupedWeightFn",
    "eager_stack_grouped_weight",
    "is_triton_path_enabled",
    "stack_grouped_weight",
]
