###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused Sinkhorn-Knopp FWD/BWD (plan-6 P36).

Replaces the plan-5 P29 ``torch.compile`` Sinkhorn path with a
hand-rolled Triton kernel that runs the entire alternating row/col
normalize trajectory **in registers** per row of the leading axis.
The plan-5 P32 final trace attributes:

* ``Torch-Compiled Region``         ≈ **21 ms / 16 calls** (FWD-side)
* ``CompiledFunctionBackward``      ≈ **41 ms / 16 calls** (BWD-side)

to the cached compiled artefact built in
:func:`primus.backends.megatron.core.transformer.hyper_connection._build_compiled_sinkhorn`.
``torch.compile`` collapses the 1 + 2*(n_iters - 1) fp32 reductions
into one Inductor-fused Triton kernel, but its Dynamo-side
bookkeeping (the ``Torch-Compiled Region`` event itself) and
Inductor's BWD path are still non-trivial per-call overhead.

The Triton path emits **one** FWD kernel and **one** BWD kernel; no
Dynamo bookkeeping per call.  At V4-Flash widths
(``[..., K=4, K=4]``, ``n_iters=20``, fp32 internal compute) the
``K*K = 16`` elements per row fit in fp32 registers; the 39 normalize
steps (1 priming col-normalize + 19 row+col pairs) all run in
registers; the BWD recomputes the FWD trajectory in registers and
walks the analytic VJP backward step-by-step (the doubly-stochastic
projection has a closed-form VJP per step that fits in the same
register budget).

Eager reference (see :func:`primus...hyper_connection.sinkhorn_normalize`):

.. code-block:: python

    in_dtype = logits.dtype
    m = logits.float()
    m = m / (m.sum(dim=-2, keepdim=True) + eps)     # initial col-normalize
    for _ in range(n_iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps) # row-normalize
        m = m / (m.sum(dim=-2, keepdim=True) + eps) # col-normalize
    return m.to(in_dtype)

Per-step VJP (closed form):

.. code-block:: python

    # y = x / s, where s = sum(x, axis=axis) + eps
    # dx = (dy - sum(dy * y, axis=axis, keep_dims=True)) / s

Both axes (row / col) have the same shape of VJP (Triton's ``tl.sum``
handles either with the right ``axis=`` argument), so the BWD kernel
is just the FWD-trajectory recompute followed by 39 of these VJP
steps in reverse order.

Gating: routed through :func:`apply_sinkhorn_normalize_triton` when
``PRIMUS_SINKHORN_TRITON != "0"`` (default-on).  Set to ``"0"`` to
fall back to either the plan-5 P29 compiled path (if
``use_compiled=True`` on the call site) or the eager path.

Supported shape constraints:

* Last two dims must be square and ``K ∈ {1, 2, 4, 8, 16}`` (Triton's
  ``tl.arange`` needs a power-of-2 block); V4-Flash uses ``K=4``.
* ``logits`` must be contiguous (the wrapper calls ``.contiguous()``
  defensively).
* Any leading shape is supported (the wrapper flattens to
  ``[N, K, K]``).
"""

from __future__ import annotations

import os

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
            f"sinkhorn: unsupported dtype {t}; expected one of {list(_TORCH_TO_TL_DTYPE)}"
        ) from exc


# Supported K (last two dim) values for the in-register Triton path.
# Block-size constraints (Triton requires power-of-2 block extents) plus
# register budget (~K*K*BLOCK_LEADING fp32 elements live per program)
# practically limit K to {1, 2, 4, 8, 16}.  V4-Flash uses K=4.
_SUPPORTED_K = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _sinkhorn_fwd_kernel(
    X_PTR,  # [N, K, K] contiguous
    Y_PTR,  # [N, K, K] contiguous
    STATES_PTR,  # [N, 2 * N_ITERS, K, K] contiguous (FWD trajectory cache)
    N,
    K: tl.constexpr,
    N_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
    DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Run the 1 + 2*(N_ITERS - 1) alternating row/col normalize
    trajectory for ``BLOCK_LEADING`` rows of the leading axis.

    The internal ``m`` tensor stays in fp32 throughout (matches the
    eager :func:`sinkhorn_normalize` fp32 contract); the trailing cast
    to ``DTYPE`` happens at the final store.

    Writes the full FWD state trajectory ``m_0, m_1, ..., m_{2*N_ITERS-1}``
    to ``STATES_PTR``; the BWD kernel reads it back so it can walk the
    analytic VJP backward step-by-step without re-running FWD.  At
    ``K=4, N_ITERS=20`` the cache is ``40 * 16 * 4 = 2560`` bytes per
    row (~10 MiB total at V4-Flash ``N=4096``) -- negligible HBM
    overhead vs the 256 KiB FWD output of the same call.
    """

    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)

    KK: tl.constexpr = K * K
    base = offs[:, None, None] * KK + r_offs[None, :, None] * K + c_offs[None, None, :]
    full_mask = mask_leading[:, None, None]

    # State buffer layout: [N, N_STATES, K, K] flattened row-major.
    # state_base[s] = offs * (N_STATES * K * K) + s * (K * K) + r*K + c.
    n_states_total: tl.constexpr = 2 * N_ITERS
    state_row_stride: tl.constexpr = n_states_total * KK
    state_base = offs[:, None, None] * state_row_stride + r_offs[None, :, None] * K + c_offs[None, None, :]

    m = tl.load(X_PTR + base, mask=full_mask, other=0.0).to(COMPUTE_DTYPE)
    # Save m_0 = x.float() (state index 0).
    tl.store(STATES_PTR + state_base, m, mask=full_mask)

    # Priming step: col-normalize (sum over the row axis, dim=-2 in the
    # caller's [...,K,K]; this is axis=1 in our [BLOCK,K,K] layout).
    s = tl.sum(m, axis=1, keep_dims=True) + EPS
    m = m / s
    tl.store(STATES_PTR + state_base + 1 * KK, m, mask=full_mask)  # m_1

    # Alternating loop -- N_ITERS - 1 (row, col) pairs.  After the
    # priming step we are at state index 1; each loop iteration writes
    # two more states (the row half then the col half).
    for it in tl.static_range(N_ITERS - 1):
        s = tl.sum(m, axis=2, keep_dims=True) + EPS  # row-normalize
        m = m / s
        tl.store(STATES_PTR + state_base + (2 + 2 * it) * KK, m, mask=full_mask)
        s = tl.sum(m, axis=1, keep_dims=True) + EPS  # col-normalize
        m = m / s
        tl.store(STATES_PTR + state_base + (3 + 2 * it) * KK, m, mask=full_mask)

    tl.store(Y_PTR + base, m.to(DTYPE), mask=full_mask)


@triton.jit
def _sinkhorn_bwd_kernel(
    STATES_PTR,  # [N, 2 * N_ITERS, K, K] contiguous (FWD trajectory cache)
    DY_PTR,  # [N, K, K] contiguous (upstream grad in caller dtype)
    DX_PTR,  # [N, K, K] contiguous (output)
    N,
    K: tl.constexpr,
    N_ITERS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_LEADING: tl.constexpr,
    DTYPE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Apply the analytic VJP for the FWD trajectory.

    Reads the cached FWD states ``m_0, m_1, ..., m_{2*N_ITERS-1}`` from
    HBM (written by :func:`_sinkhorn_fwd_kernel`) and walks the
    trajectory backward, applying the per-step closed-form VJP:

    .. code-block:: python

        # forward:  y = x / s, where s = sum(x, axis=axis) + eps
        # backward: dx = (dy - sum(dy * y, axis=axis, keep_dims=True)) / s

    The HBM round-trip for the cache is ~10 MiB total at V4-Flash
    widths (``K=4, N=4096``, 40 states / row × 16 fp32 / state); on a
    3 TB/s HBM device that's ~3 microseconds of read traffic --
    negligible vs the BWD's actual arithmetic.  This sidesteps a
    Triton AST-visitor restriction in our toolchain that rejects
    runtime indexing of Python lists holding ``tl.tensor`` bundles,
    which would otherwise let us recompute the trajectory in
    registers.

    Trajectory indexing convention:
    * ``m_0`` (state 0) = ``x.float()`` (input, BEFORE the priming step);
    * step ``s`` transforms ``m_{s-1}`` -> ``m_s`` via
      ``axis = 1`` (col-normalize) when ``s`` is odd, ``axis = 2``
      (row-normalize) when ``s`` is even;
    * total steps ``n_steps = 1 + 2*(N_ITERS - 1) = 2*N_ITERS - 1``.
    """

    pid = tl.program_id(0)
    offs = pid * BLOCK_LEADING + tl.arange(0, BLOCK_LEADING)
    mask_leading = offs < N

    r_offs = tl.arange(0, K)
    c_offs = tl.arange(0, K)

    KK: tl.constexpr = K * K
    n_states_total: tl.constexpr = 2 * N_ITERS
    state_row_stride: tl.constexpr = n_states_total * KK

    base = offs[:, None, None] * KK + r_offs[None, :, None] * K + c_offs[None, None, :]
    full_mask = mask_leading[:, None, None]
    state_base = offs[:, None, None] * state_row_stride + r_offs[None, :, None] * K + c_offs[None, None, :]

    dy = tl.load(DY_PTR + base, mask=full_mask, other=0.0).to(COMPUTE_DTYPE)
    dm = dy

    # Walk the FWD trajectory backward.  Trajectory parity is regular:
    # step 1 (priming) is col-normalize, then alternating row, col, ...
    # ending with step ``2*N_ITERS - 1`` (col).  Walking BACKWARD we
    # always see a (col, row) pair, repeated ``N_ITERS - 1`` times,
    # followed by the priming col-step.
    #
    # We pair the loop iterations explicitly instead of using a
    # ``step % 2`` runtime check so the axis= argument to ``tl.sum``
    # stays a compile-time Python int (otherwise the two branches'
    # ``keep_dims=True`` outputs disagree on shape and Triton refuses
    # to compile).
    for i in range(N_ITERS - 1):
        # Outer (col) step: indices 2*N_ITERS - 1, 2*N_ITERS - 3, ..., 3.
        col_step = 2 * N_ITERS - 1 - 2 * i
        m_before_c = tl.load(
            STATES_PTR + state_base + (col_step - 1) * KK,
            mask=full_mask,
            other=0.0,
        )
        m_after_c = tl.load(
            STATES_PTR + state_base + col_step * KK,
            mask=full_mask,
            other=0.0,
        )
        s_c = tl.sum(m_before_c, axis=1, keep_dims=True) + EPS
        dot_c = tl.sum(dm * m_after_c, axis=1, keep_dims=True)
        dm = (dm - dot_c) / s_c

        # Inner (row) step: indices 2*N_ITERS - 2, 2*N_ITERS - 4, ..., 2.
        row_step = col_step - 1
        m_before_r = tl.load(
            STATES_PTR + state_base + (row_step - 1) * KK,
            mask=full_mask,
            other=0.0,
        )
        m_after_r = tl.load(
            STATES_PTR + state_base + row_step * KK,
            mask=full_mask,
            other=0.0,
        )
        s_r = tl.sum(m_before_r, axis=2, keep_dims=True) + EPS
        dot_r = tl.sum(dm * m_after_r, axis=2, keep_dims=True)
        dm = (dm - dot_r) / s_r

    # Final priming col-normalize step (step 1: m_0 -> m_1).
    m_before_p = tl.load(STATES_PTR + state_base, mask=full_mask, other=0.0)
    m_after_p = tl.load(STATES_PTR + state_base + KK, mask=full_mask, other=0.0)
    s_p = tl.sum(m_before_p, axis=1, keep_dims=True) + EPS
    dot_p = tl.sum(dm * m_after_p, axis=1, keep_dims=True)
    dm = (dm - dot_p) / s_p

    tl.store(DX_PTR + base, dm.to(DTYPE), mask=full_mask)


# ---------------------------------------------------------------------------
# Block-leading heuristic
# ---------------------------------------------------------------------------


def _pick_block_leading(n: int, k: int) -> int:
    """Pick ``BLOCK_LEADING`` based on K and the work-axis size.

    At K=4 the in-register tensor is ``[BLOCK_LEADING, 4, 4] = 16
    fp32 / row``; with ~256 VGPRs per warp on MI355 a BLOCK_LEADING of
    ``128`` keeps live registers + VGPR-spilled state comfortable.

    At K=8 (16x larger per-row footprint) drop to 32.

    At K=16 drop to 8 -- the FWD trajectory keeps ~40 intermediate
    matrices live during BWD recomputation, so per-program LDS / spill
    needs to stay bounded.
    """

    if k <= 4:
        cap = 128
    elif k <= 8:
        cap = 32
    else:
        cap = 8

    if n < cap:
        # Round to next power of 2 >= n (Triton requires power-of-2 block).
        return max(1, triton.next_power_of_2(n))
    return cap


# ---------------------------------------------------------------------------
# torch.autograd.Function wrapper
# ---------------------------------------------------------------------------


class SinkhornNormalizeFn(torch.autograd.Function):
    """Autograd-aware wrapper around the FWD/BWD Triton kernels.

    Saves only the input ``x`` for the backward; ``n_iters`` and
    ``eps`` are static keys (compiled into the kernel binary cache).

    Shape: any ``[..., K, K]`` where ``K`` is a power of 2 in
    ``{1, 2, 4, 8, 16}``.  V4-Flash uses ``K=4, n_iters=20, eps=1e-6``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        n_iters: int,
        eps: float,
    ) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError(
                f"sinkhorn_normalize: input must be at least 2-D, got shape {tuple(logits.shape)}"
            )
        K = logits.shape[-1]
        if logits.shape[-2] != K:
            raise ValueError(
                f"sinkhorn_normalize: input must be square in the last two dims, "
                f"got shape {tuple(logits.shape)}"
            )
        if K not in _SUPPORTED_K:
            raise ValueError(f"sinkhorn Triton path: unsupported K={K}; expected one of {_SUPPORTED_K}")
        if int(n_iters) < 1:
            raise ValueError(f"n_iters must be >= 1, got {n_iters}")

        x = logits.contiguous()
        leading_shape = x.shape[:-2]
        N = 1
        for s in leading_shape:
            N *= s

        out = torch.empty_like(x)
        # FWD trajectory cache: 2 * N_ITERS states per row of K*K fp32
        # elements.  Saved for backward.  At V4-Flash widths (K=4,
        # N_ITERS=20, N=4096) this is 10 MiB per call -- negligible vs
        # the 256 KiB FWD output and the ~170 GiB total HBM footprint
        # of the proxy.
        n_states_total = 2 * int(n_iters)
        states_buf = torch.empty(
            (N, n_states_total, K, K),
            dtype=torch.float32,
            device=x.device,
        )

        block_leading = _pick_block_leading(N, K)
        grid = (triton.cdiv(N, block_leading),)
        _sinkhorn_fwd_kernel[grid](
            x,
            out,
            states_buf,
            N,
            K=K,
            N_ITERS=int(n_iters),
            EPS=float(eps),
            BLOCK_LEADING=block_leading,
            DTYPE=_triton_dtype(x.dtype),
            COMPUTE_DTYPE=tl.float32,
        )

        ctx.save_for_backward(states_buf)
        ctx.n_iters = int(n_iters)
        ctx.eps = float(eps)
        ctx.K = K
        ctx.leading_shape = tuple(leading_shape)
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):  # type: ignore[override]
        (states_buf,) = ctx.saved_tensors
        n_iters: int = ctx.n_iters
        eps: float = ctx.eps
        K: int = ctx.K
        leading_shape = ctx.leading_shape

        dy = dy.contiguous()
        N = 1
        for s in leading_shape:
            N *= s

        dx = torch.empty_like(dy)
        block_leading = _pick_block_leading(N, K)
        grid = (triton.cdiv(N, block_leading),)
        _sinkhorn_bwd_kernel[grid](
            states_buf,
            dy,
            dx,
            N,
            K=K,
            N_ITERS=n_iters,
            EPS=eps,
            BLOCK_LEADING=block_leading,
            DTYPE=_triton_dtype(dy.dtype),
            COMPUTE_DTYPE=tl.float32,
        )
        # n_iters / eps gradients: not differentiable parameters.
        return dx, None, None


# ---------------------------------------------------------------------------
# Public Python entry points
# ---------------------------------------------------------------------------


def is_triton_path_enabled() -> bool:
    """Return True iff the ``PRIMUS_SINKHORN_TRITON`` env knob is not ``"0"``.

    Default-on; A/B toggle via ``PRIMUS_SINKHORN_TRITON=0``.
    """

    return os.environ.get("PRIMUS_SINKHORN_TRITON", "1") != "0"


def is_triton_kernel_supported(logits: torch.Tensor) -> bool:
    """Return True iff the input shape / device is supported by the Triton path.

    Used by the dispatcher in
    :func:`primus.backends.megatron.core.transformer.hyper_connection.sinkhorn_normalize`
    to safely fall back to the compiled / eager path when the kernel
    can't handle the input (non-CUDA, non-square, K out-of-range).
    """

    if not logits.is_cuda:
        return False
    if logits.dim() < 2:
        return False
    K = logits.shape[-1]
    if logits.shape[-2] != K:
        return False
    if K not in _SUPPORTED_K:
        return False
    return True


def eager_sinkhorn_normalize(
    logits: torch.Tensor,
    *,
    n_iters: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference eager implementation matching
    :func:`primus.backends.megatron.core.transformer.hyper_connection.sinkhorn_normalize`
    eager body bit-for-bit (same op order, same fp32 cast contract).

    Kept here so the unit tests / bench can A/B against the canonical
    eager body without depending on the consumer module.
    """

    in_dtype = logits.dtype
    m = logits.float()
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(n_iters - 1, 0)):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m.to(in_dtype)


def sinkhorn_normalize_triton(
    logits: torch.Tensor,
    *,
    n_iters: int = 20,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run the Triton-fused Sinkhorn-Knopp normalize.

    Dispatcher: if the env knob is on AND the shape is supported, call
    the Triton path; else fall back to the eager body in
    :func:`eager_sinkhorn_normalize`.
    """

    if is_triton_path_enabled() and is_triton_kernel_supported(logits):
        return SinkhornNormalizeFn.apply(logits, n_iters, eps)
    return eager_sinkhorn_normalize(logits, n_iters=n_iters, eps=eps)


__all__ = [
    "SinkhornNormalizeFn",
    "sinkhorn_normalize_triton",
    "eager_sinkhorn_normalize",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]
