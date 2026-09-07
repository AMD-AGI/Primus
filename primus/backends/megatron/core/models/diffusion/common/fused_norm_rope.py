###############################################################################
# Fused QK-RMSNorm + RoPE for Flux.
#
# Flux applies an RMS norm to Q and K and then rotates them. Left to Inductor
# these are one kernel forward and three backward, so the rope backward writes a
# full-size intermediate the norm backward reads straight back, per layer, which
# is traffic nothing needs. This does each direction in one pass.
#
# The rotation layout matters more than anything else here. Megatron's
# ``_rotate_half(x, rotary_interleaved)`` either pairs index i with i + D/2 or
# pairs 2i with 2i+1, and ``FluxConfig.rotary_interleaved`` defaults to True, so
# Flux runs the interleaved branch. Inductor generates that branch's
# ``x[..., 0::2]`` / ``torch.stack`` / ``view`` as literal strided access and
# reaches only 53% of achievable bandwidth on the forward and 35% on the
# backward, against 87% and 45% for the same code in the half-split layout.
#
# The whole trick below is to keep the pairing out of the addresses: load and
# store rows contiguously and interleave in registers with tl.split / tl.join.
# Measured on gfx950 at the production shape [512,64,24,128] bf16 with cold caches,
# both directions move from a modest fraction of the bandwidth roof to most of it.
#
# Numerics against an fp64 reference are no worse than the path this replaces, and
# better on dx, because the baseline rounds to bf16 before applying the norm weight
# while this carries fp32 through to a single final rounding.
#
# Registered with @triton_op rather than @torch.library.custom_op so Inductor can
# still reason across it. Removing part of an Inductor-fused region and making it
# opaque is what made the TransformerEngine path measurably slower, and the
# surrounding split and dtype casts here are exactly that kind of region.
###############################################################################

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import custom_op, triton_op, wrap_triton


@triton.jit
def _pair(P, rows, mask, SM, D: tl.constexpr, BLOCK_M: tl.constexpr, INTERLEAVED: tl.constexpr):
    """Load [BLOCK_M, D] rows of P and return the two members of each rotation pair.

    ``SM`` is the distance between consecutive rows. It is D for a packed tensor, and
    larger for Q and K, which are slices of ``mixed_qkv`` and so carry a gap to the
    next head. Taking it as an argument is what lets those be read in place.
    """
    half: tl.constexpr = D // 2
    if INTERLEAVED:
        off = rows[:, None] * SM + tl.arange(0, D)[None, :]
        v = tl.load(P + off, mask=mask, other=0.0).to(tl.float32)
        return tl.split(tl.reshape(v, (BLOCK_M, half, 2)))
    lo = tl.arange(0, half)
    off = rows[:, None] * SM + lo[None, :]
    a = tl.load(P + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(P + off + half, mask=mask, other=0.0).to(tl.float32)
    return a, b


@triton.jit
def _unpair(
    P,
    rows,
    mask,
    v_lo,
    v_hi,
    SM,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    """Inverse of _pair: interleave or concatenate, then store at row stride SM."""
    half: tl.constexpr = D // 2
    if INTERLEAVED:
        v = tl.reshape(tl.join(v_lo, v_hi), (BLOCK_M, D))
        off = rows[:, None] * SM + tl.arange(0, D)[None, :]
        tl.store(P + off, v.to(P.dtype.element_ty), mask=mask)
    else:
        lo = tl.arange(0, half)
        off = rows[:, None] * SM + lo[None, :]
        tl.store(P + off, v_lo.to(P.dtype.element_ty), mask=mask)
        tl.store(P + off + half, v_hi.to(P.dtype.element_ty), mask=mask)


@triton.jit
def _pair_w(W, D: tl.constexpr, INTERLEAVED: tl.constexpr):
    """The norm weight is D-wide and shared by every row; split it the same way."""
    half: tl.constexpr = D // 2
    if INTERLEAVED:
        v = tl.load(W + tl.arange(0, D)).to(tl.float32)
        a, b = tl.split(tl.reshape(v, (half, 2)))
        return a[None, :], b[None, :]
    lo = tl.arange(0, half)
    return (
        tl.load(W + lo).to(tl.float32)[None, :],
        tl.load(W + lo + half).to(tl.float32)[None, :],
    )


def _fwd_configs():
    return [
        triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
        for bm in (1, 2, 4, 8, 16)
        for nw in (1, 2, 4, 8)
        for ns in (1, 2)
    ]


@triton.autotune(configs=_fwd_configs(), key=["M", "D"])
@triton.jit
def _fwd_kernel(
    X,
    W,
    COS,
    SIN,
    OUT,
    RSTD,
    M,
    D: tl.constexpr,
    BH,
    EPS,
    SX,
    SO,
    SC,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < M
    mask = mask_m[:, None]

    x_lo, x_hi = _pair(X, rows, mask, SX, D, BLOCK_M, INTERLEAVED)
    # The mean is over the whole row, so it does not care how the row was split.
    rstd = tl.rsqrt((tl.sum(x_lo * x_lo, axis=1) + tl.sum(x_hi * x_hi, axis=1)) / D + EPS)[:, None]

    w_lo, w_hi = _pair_w(W, D, INTERLEAVED)
    n_lo, n_hi = x_lo * rstd * w_lo, x_hi * rstd * w_hi

    # cos/sin are [S,D] or [S*B,D]: rows sharing a position share a row of the table.
    s = rows // BH
    c_lo, c_hi = _pair(COS, s, mask, SC, D, BLOCK_M, INTERLEAVED)
    s_lo, s_hi = _pair(SIN, s, mask, SC, D, BLOCK_M, INTERLEAVED)

    _unpair(
        OUT,
        rows,
        mask,
        n_lo * c_lo - n_hi * s_lo,
        n_hi * c_hi + n_lo * s_hi,
        SO,
        D,
        BLOCK_M,
        INTERLEAVED,
    )
    tl.store(RSTD + rows, tl.reshape(rstd, (BLOCK_M,)), mask=mask_m)


# Fixed rather than autotuned. The partial buffer has to be sized before launch, so
# tuning this meant allocating for the largest candidate and reducing all of it --
# 16 MB zeroed and summed per call, against a kernel far too short to absorb that.
# Kernel bandwidth is flat across the useful range anyway.
NPROG = 4096


def _bwd_configs():
    return [
        triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
        for bm in (1, 2, 4, 8, 16, 32, 64)
        for nw in (1, 2, 4, 8)
        for ns in (1, 2, 3)
    ]


# dw is a reduction over every token onto a D-wide vector. Doing that with atomic_add
# from one program per row puts ~200k programs on the same 128 addresses, which measured
# 10x slower than the three-kernel baseline it was meant to beat. Instead the grid is
# persistent: a fixed number of programs each stride over many rows, keep a private
# accumulator in registers, and write one partial row for the host to sum. No atomics.
@triton.autotune(configs=_bwd_configs(), key=["M", "D"])
@triton.jit
def _bwd_kernel(
    G,
    X,
    W,
    COS,
    SIN,
    RSTD,
    DX,
    DWP,
    M,
    D: tl.constexpr,
    BH,
    SG,
    SX,
    SDX,
    SC,
    BLOCK_M: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    # Passed explicitly rather than defaulted to the module global: under
    # torch.compile, Inductor re-emits this kernel's source into a fresh module and
    # a default that references a global resolves to NameError there.
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    half: tl.constexpr = D // 2
    w_lo, w_hi = _pair_w(W, D, INTERLEAVED)
    dw_lo = tl.zeros((half,), dtype=tl.float32)
    dw_hi = tl.zeros((half,), dtype=tl.float32)

    for base in tl.range(pid * BLOCK_M, M, NPROG * BLOCK_M):
        rows = base + tl.arange(0, BLOCK_M)
        mask = (rows < M)[:, None]

        g_lo, g_hi = _pair(G, rows, mask, SG, D, BLOCK_M, INTERLEAVED)
        x_lo, x_hi = _pair(X, rows, mask, SX, D, BLOCK_M, INTERLEAVED)
        rstd = tl.load(RSTD + rows, mask=rows < M, other=0.0)[:, None]

        s = rows // BH
        c_lo, c_hi = _pair(COS, s, mask, SC, D, BLOCK_M, INTERLEAVED)
        s_lo, s_hi = _pair(SIN, s, mask, SC, D, BLOCK_M, INTERLEAVED)

        # Inverse of out_lo = n_lo*cos_lo - n_hi*sin_lo, out_hi = n_hi*cos_hi + n_lo*sin_hi.
        dn_lo = g_lo * c_lo + g_hi * s_hi
        dn_hi = g_hi * c_hi - g_lo * s_lo

        u_lo, u_hi = x_lo * rstd, x_hi * rstd
        du_lo, du_hi = dn_lo * w_lo, dn_hi * w_hi
        m = (tl.sum(du_lo * u_lo, axis=1) + tl.sum(du_hi * u_hi, axis=1))[:, None] / D

        _unpair(
            DX,
            rows,
            mask,
            rstd * (du_lo - u_lo * m),
            rstd * (du_hi - u_hi * m),
            SDX,
            D,
            BLOCK_M,
            INTERLEAVED,
        )
        dw_lo += tl.sum(tl.where(mask, dn_lo * u_lo, 0.0), axis=0)
        dw_hi += tl.sum(tl.where(mask, dn_hi * u_hi, 0.0), axis=0)

    # The accumulators are in pair order, so they need unpairing before the host sums
    # them onto the layout the weight actually has.
    if INTERLEAVED:
        tl.store(DWP + pid * D + tl.arange(0, D), tl.reshape(tl.join(dw_lo, dw_hi), (D,)))
    else:
        lo = tl.arange(0, half)
        tl.store(DWP + pid * D + lo, dw_lo)
        tl.store(DWP + pid * D + lo + half, dw_hi)


def _cs_div(M: int, cos: torch.Tensor) -> int:
    """How many [S,B,H] rows share one row of cos/sin.

    Flux's RoPE frequencies are per image, so ``freqs`` is [S, B, 1, D] and cos/sin
    flatten to [S*B, D] -- one row per position *and* batch element, shared only
    across heads. A model whose positions are shared across the batch gives [S, D]
    instead. Both are just a divisor on the row index, so the kernel needs no branch:
    M // rows is B*H in the first case and H in the second.
    """
    rows = cos.shape[0]
    assert M % rows == 0, f"cos/sin rows {rows} do not divide {M} tensor rows"
    return M // rows


def _row_stride(t: torch.Tensor):
    """The row stride of a [S,B,H,D] tensor, or None if the kernel cannot address it.

    The kernel treats its operands as M rows of D contiguous elements, evenly spaced.
    Q and K fit that description without being contiguous: they are ``[..., 0:D]`` and
    ``[..., D:2D]`` slices of ``mixed_qkv``, so each row is D wide with a gap to the
    next head. Reading them at their natural stride rather than copying them first
    removes the copy kernels entirely, and it keeps the tensor saved for backward a
    view of ``mixed_qkv`` -- which the QKV GEMM holds anyway -- instead of a fresh
    allocation of activations.
    """
    if t.dim() != 4:
        return None
    _, B, H, D = t.shape
    s0, s1, s2, s3 = t.stride()
    # Rows must be contiguous, must not overlap, and must tile the outer dims evenly.
    if s3 != 1 or s2 < D or s1 != H * s2 or s0 != B * s1:
        return None
    return s2


def _row_strided(t: torch.Tensor):
    """Return t addressable at a uniform row stride, copying only if it is not."""
    s = _row_stride(t)
    if s is None:
        t = t.contiguous()
        s = t.shape[-1]
    return t, s


def _launchable(kernel, traceable):
    """``kernel``, wrapped for tracing only when the caller can actually be traced.

    ``wrap_triton`` exists so Inductor can see through a ``@triton_op`` into the kernel it
    launches. Inside a ``@custom_op`` there is nothing to see through -- the op is opaque by
    construction -- and the wrapper is not free: it routes every launch through the
    higher-order-op dispatch path, which measures **0.359 ms of host time against 0.031 ms
    for the same launch made directly**, a 12.7x overhead on a kernel whose GPU work is a
    fraction of that.

    Repeated across every QKV call a step makes, two launches each, that wrapper alone was
    enough to make the whole-QKV backward host-bound rather than GPU-bound: the enqueue loop
    measured 0.8555 ms/call against a batched wall time of 0.8593, i.e. the GPU was idle
    waiting for Python. It is also why the cost barely moved
    with sequence length, which is the symptom that gave it away -- seq 256 measured 0.845 ms
    against seq 512's 0.864, when the work halves.

    So the per-tensor ``@triton_op`` entry points pass ``traceable=True`` and keep the
    wrapper they need; the whole-QKV ``@custom_op`` ones pass ``False``.
    """
    return wrap_triton(kernel) if traceable else kernel


def _launch_fwd(x, w, cos, sin, eps, interleaved, traceable=True):
    S, B, H, D = x.shape
    M = S * B * H
    x, sx = _row_strided(x)
    # The output is packed even when the input is not: it feeds FMHA, which wants it
    # that way, and writing it packed costs nothing the strided read has not saved.
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _launchable(_fwd_kernel, traceable)[grid](
        x,
        w,
        cos,
        sin,
        out,
        rstd,
        M,
        D,
        _cs_div(M, cos),
        eps,
        sx,
        D,
        cos.stride(0),
        INTERLEAVED=interleaved,
    )
    return out, rstd


def _launch_bwd(g, x, w, cos, sin, rstd, dx, interleaved, traceable=True):
    """Backward into a caller-supplied dx, which need only be evenly strided.

    dx is a separate argument rather than an allocation because the QKV entry point
    below points it at a column of d(mixed_qkv); see the note there.

    ``traceable`` selects whether the launch goes through ``wrap_triton``; see
    ``_launchable``, which is where the cost of getting that wrong is recorded.
    """
    S, B, H, D = x.shape
    M = S * B * H
    g, sg = _row_strided(g)
    x, sx = _row_strided(x)
    sdx = _row_stride(dx)
    assert sdx is not None, "dx must be addressable at a uniform row stride"
    # Every row is written by exactly one program, so empty() is safe.
    dwp = torch.empty(NPROG, D, device=x.device, dtype=torch.float32)
    _launchable(_bwd_kernel, traceable)[(NPROG,)](
        g,
        x,
        w,
        cos,
        sin,
        rstd,
        dx,
        dwp,
        M,
        D,
        _cs_div(M, cos),
        sg,
        sx,
        sdx,
        cos.stride(0),
        INTERLEAVED=interleaved,
        NPROG=NPROG,
    )
    return dwp


@triton_op("primus::fused_qk_norm_rope", mutates_args=())
def _fused_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _launch_fwd(x, w, cos, sin, eps, interleaved)


@_fused_fwd.register_fake
def _fused_fwd_fake(x, w, cos, sin, eps, interleaved):
    S, B, H, D = x.shape
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty(S * B * H, device=x.device, dtype=torch.float32),
    )


@triton_op("primus::fused_qk_norm_rope_backward", mutates_args=())
def _fused_bwd(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rstd: torch.Tensor,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(_row_strided(x)[0], memory_format=torch.contiguous_format)
    return dx, _launch_bwd(g, x, w, cos, sin, rstd, dx, interleaved)


@_fused_bwd.register_fake
def _fused_bwd_fake(g, x, w, cos, sin, rstd, interleaved):
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty(NPROG, x.shape[-1], device=x.device, dtype=torch.float32),
    )


def _setup_context(ctx, inputs, output):
    x, w, cos, sin, eps, interleaved = inputs
    ctx.save_for_backward(x, w, cos, sin, output[1])
    ctx.interleaved = interleaved


def _backward(ctx, grad_out, _grad_rstd):
    x, w, cos, sin, rstd = ctx.saved_tensors
    dx, dwp = _fused_bwd(grad_out, x, w, cos, sin, rstd, ctx.interleaved)
    return dx, dwp.sum(0).to(w.dtype), None, None, None, None


_fused_fwd.register_autograd(_backward, setup_context=_setup_context)


# ---------------------------------------------------------------------------
# Whole-QKV entry point.
#
# The per-tensor op above leaves behind one kernel the unfused path never paid for:
# the backward of the QKV split. Inductor used to fold that concatenation into the
# epilogue of its own norm-backward and write d(mixed_qkv) directly; against an opaque
# op it cannot, so it emits a standalone gather of dq, dk and dv.
#
# Taking mixed_qkv as the differentiable input instead of its three slices removes the
# split, and with it the thing that had to be undone. The backward kernels write dq and
# dk straight into the q and k columns of d(mixed_qkv) at stride 3D, which costs them
# nothing: the destination row stride has been a runtime argument ever since Q and K
# became strided reads. Only dv still has to be copied in, since it is produced by FMHA.
# ---------------------------------------------------------------------------


# Opaque, unlike the per-tensor op above. That op is a @triton_op so Inductor can fuse
# its neighbours into it; here there are none -- the input is a GEMM output and the
# outputs go to FMHA, both opaque. Transparency is not merely unnecessary at this level,
# it is wrong: tracing in exposes two triton kernels writing disjoint columns of one
# freshly allocated buffer, and functionalizing that pattern produced a kernel that read
# 786304 elements out of a 262144-element clone. Nothing here needs to be seen.
@custom_op("primus::fused_qkv_norm_rope", mutates_args=())
def _qkv_fwd(
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    D = qkv.shape[-1] // 3
    q, q_rstd = _launch_fwd(qkv[..., :D], wq, cos, sin, eps, interleaved, traceable=False)
    k, k_rstd = _launch_fwd(qkv[..., D : 2 * D], wk, cos, sin, eps, interleaved, traceable=False)
    # V is not merely forwarded: a custom op may not return an alias of its input, and
    # FMHA wants it packed regardless, so this replaces the copy the caller was making.
    v = qkv[..., 2 * D :].contiguous()
    return q, k, v, q_rstd, k_rstd


@_qkv_fwd.register_fake
def _qkv_fwd_fake(qkv, wq, wk, cos, sin, eps, interleaved):
    S, B, H, T = qkv.shape
    D = T // 3
    head = torch.empty((S, B, H, D), device=qkv.device, dtype=qkv.dtype)
    rstd = torch.empty(S * B * H, device=qkv.device, dtype=torch.float32)
    return (head, torch.empty_like(head), torch.empty_like(head), rstd, torch.empty_like(rstd))


@custom_op("primus::fused_qkv_norm_rope_backward", mutates_args=())
def _qkv_bwd(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    qkv: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D = qkv.shape[-1] // 3
    d_qkv = torch.empty_like(qkv, memory_format=torch.contiguous_format)
    dwqp = _launch_bwd(dq, qkv[..., :D], wq, cos, sin, q_rstd, d_qkv[..., :D], interleaved,
                       traceable=False)
    dwkp = _launch_bwd(dk, qkv[..., D : 2 * D], wk, cos, sin, k_rstd, d_qkv[..., D : 2 * D],
                       interleaved, traceable=False)
    d_qkv[..., 2 * D :].copy_(dv)
    return d_qkv, dwqp, dwkp


@_qkv_bwd.register_fake
def _qkv_bwd_fake(dq, dk, dv, qkv, wq, wk, cos, sin, q_rstd, k_rstd, interleaved):
    D = qkv.shape[-1] // 3
    p = torch.empty(NPROG, D, device=qkv.device, dtype=torch.float32)
    return (torch.empty_like(qkv, memory_format=torch.contiguous_format), p, torch.empty_like(p))


def _qkv_setup_context(ctx, inputs, output):
    qkv, wq, wk, cos, sin, eps, interleaved = inputs
    # qkv rather than its q and k slices: one saved tensor instead of two, and it is the
    # QKV GEMM's output, which the graph is holding for that GEMM's own backward anyway.
    ctx.save_for_backward(qkv, wq, wk, cos, sin, output[3], output[4])
    ctx.interleaved = interleaved


def _qkv_backward(ctx, dq, dk, dv, _dqr, _dkr):
    qkv, wq, wk, cos, sin, q_rstd, k_rstd = ctx.saved_tensors
    d_qkv, dwqp, dwkp = _qkv_bwd(dq, dk, dv, qkv, wq, wk, cos, sin, q_rstd, k_rstd, ctx.interleaved)
    return d_qkv, dwqp.sum(0).to(wq.dtype), dwkp.sum(0).to(wk.dtype), None, None, None, None


_qkv_fwd.register_autograd(_qkv_backward, setup_context=_qkv_setup_context)


def fused_qkv_norm_rope(mixed_qkv, wq, wk, cos, sin, eps=1e-6, interleaved=True):
    """Split [S, B, H, 3D] into Q, K, V with QK norm and RoPE already applied.

    Prefer this over calling ``fused_qk_norm_rope`` on Q and K separately: it is the
    same kernels, but because the split happens inside the op there is no concatenation
    to pay for on the way back. Returns packed (Q, K, V).
    """
    q, k, v, _, _ = _qkv_fwd(mixed_qkv, wq, wk, cos, sin, eps, interleaved)
    return q, k, v


def fused_qk_norm_rope(x, weight, cos, sin, eps=1e-6, interleaved=True):
    """RMS-norm x over its last dim, then apply RoPE, in one kernel each direction.

    ``x`` is [S, B, H, D] and need not be contiguous -- a slice of ``mixed_qkv`` is
    read in place, see ``_row_stride``. ``cos``/``sin`` are [rows, D] and broadcast
    over whatever the rows do not index. ``interleaved`` selects Megatron's rotation
    layout and defaults to Flux's.
    """
    return _fused_fwd(x, weight, cos, sin, eps, interleaved)[0]
