# pyre-unsafe
"""Grouped LayerNorm(512) forward tail of ContextualPreprocessor (gin item (30)).

One launch computes

    O = bf16(bf16(LN_x(h_x)) + bf16(bf16(LN_a(h_a)) + bf16(LN_c(h_c))))

(round-before-add, the F2 residual chain) plus fp32 mean/rstd per LN. It is
BITWISE equal to the shipped three-launch chain: triton_weighted_layer_norm_fwd
for content, then _norm_fwd with a residual for additional and action, run with
the pinned autotune configs (BLOCK_N 8 / 4, num_warps 1). It reads h_c, h_a and
h_x once and writes O once (4120 B/row against 8216 B/row for the chain).
Caller: ops/preproc_grouped_tail.py.

Launch config: BLOCK_N=2, num_warps=1, num_stages=1. That is 95 VGPRs when each
input is <= 2 GB (buffer-op specialization) and 122 when inputs are larger, so
4-5 waves/SIMD. Per-row arithmetic is the shipped kernel's with three changes:

  * D is a constexpr. `sum / D` with D a power of two compiles to an exact
    multiply by 2^-D_log2. Under denormal mode "ieee" the correctly rounded
    quotient by a power of two IS that product, so the bits do not change. The
    shipped kernel runs two IEEE division sequences per row per LN.
  * The output store carries cache_modifier=".cs" (streaming).
  * The three row tiles are loaded before any statistics store.

UNCHANGED, and the bitwise contract depends on all of it:

  * block pointers with the runtime row stride and boundary_check on rows.
    Triton 3.9 removed block pointers (ROCm 10.2 port, T7): _tail_ld /
    _tail_st write out the pointer + mask IR Triton 3.6 lowered them to, with
    the helpers the shipped kernel now uses (triton_layer_norm._bp_ptrs): an
    int64 row mask of its own, not the where() row mask below;
  * the shipped where(row_mask & col_mask, x_mean, 0). Its select is what makes
    LLVM emit the shipped mul/fmac/pk_mul sequence for the sum of squares.
    Dropping it moves rstd by 1-2 ulp on ~4% of rows. The row mask reads a
    second runtime copy of N (Nm) and the column mask a runtime copy of D
    (Drt), as in the verified source;
  * Triton's integer specialization of N. For inputs over 2 GB the shipped
    kernel rounds differently for N%16==0 and N%16!=0; neither an
    unspecialized N nor scalar-base addressing tracks that;
  * num_warps=1, which fixes the tl.sum reduction tree.

Verified bitwise for out, mean_k, rstd_k and determinism at N in {1990000,
2333000, 2776000, 2097168, 4099, 7} x NUM_IN {3, 2}: Arbor actckpt iter 42 tune
probe (variant bp_bn2_cs) and iter 43 re-verification of this file.
ROCm 10.2 port: re-verified at the same N x NUM_IN on MI355X with Triton 3.9.0
(TRITON_DEFAULT_FP_FUSION=1, and without it) against the chain at every
(BLOCK_N0, BLOCK_N1) in {2, 4, 8}^2, num_warps 1; as on Triton 3.6, a chain with
a BLOCK_N=1 launch rounds differently above 2 GB. 3.9 needs 96 / 125-132 VGPRs
(<= / > 2 GB inputs).
Microbench against the chain on MI350X: -1.13 / -1.18 / -1.58 ms at
N = 1.99M / 2.33M / 2.78M (1.808 / 2.121 / 2.531 ms, about 4.5 TB/s).
"""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

# The SAME rsqrt binding the shipped kernel uses: imported, not re-resolved.
# _bp_*: the shipped kernel's block-pointer lowering (ROCm 10.2 port, T7).
from generative_recommenders.ops.triton.triton_layer_norm import (
    _bp_i64,
    _bp_ptrs,
    _bp_row_mask,
    libdevice_rsqrt,
)

# Fixed launch config. num_warps MUST stay 1: it fixes the reduction tree.
_TAIL_BLOCK_N: int = 2
_TAIL_NUM_WARPS: int = 1


@triton.jit
def _tail_ld(X, N, stride, start_row, D: tl.constexpr, BLOCK_N: tl.constexpr):
    # == _weighted_layer_norm_fwd's X load (before the fp32 cast). Was a block
    # pointer (shape (N, D), strides (stride, 1), offsets (start_row, 0)) with
    # boundary_check=(0,), padding_option="zero"; Triton 3.9 removed it (ROCm
    # 10.2 port, T7): the same pointer + int64 row mask, see _bp_ptrs.
    bp_n = _bp_i64(N)
    bp_s = _bp_i64(stride)
    bp_row0 = _bp_i64(start_row)
    p, offs = _bp_ptrs(X, bp_row0, bp_s, BLOCK_N, D)
    return tl.load(p, mask=_bp_row_mask(offs, bp_n, BLOCK_N, D), other=0.0)


@triton.jit
def _tail_ln(
    x_raw,
    w,
    b,
    Mean,
    Rstd,
    N,
    Drt,
    Nm,
    eps,
    start_row,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """fp32 LN(x) * w + b over a [BLOCK_N, D] tile; stores mean and rstd.

    Statement for statement the COMPUTE_MEAN_AND_RSTD, TRAINING, non-swish body
    of _weighted_layer_norm_fwd, with D a constexpr. Do not "simplify" any line:
    the bitwise contract depends on the fp32 op sequence, and therefore on the
    FMA contractions LLVM finds, being the shipped one.
    """
    x_block = x_raw.to(tl.float32)
    rows = start_row + tl.arange(0, BLOCK_N)
    mean = tl.sum(x_block, axis=1) / D
    tl.store(Mean + rows, mean, rows < N)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    cols = tl.arange(0, D)
    col_mask = cols < Drt
    row_mask = rows < Nm
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    rstd = libdevice_rsqrt(var + eps)
    tl.store(Rstd + rows, rstd, rows < N)
    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]
    return y


@triton.jit
def _tail_st(Y, acc, N, stride, start_row, D: tl.constexpr, BLOCK_N: tl.constexpr):
    # Was a block pointer store with boundary_check=(0,) (ROCm 10.2 port, T7:
    # the same pointer + int64 row mask, see _bp_ptrs); the streaming hint stays.
    bp_n = _bp_i64(N)
    bp_s = _bp_i64(stride)
    bp_row0 = _bp_i64(start_row)
    p, offs = _bp_ptrs(Y, bp_row0, bp_s, BLOCK_N, D)
    tl.store(
        p, acc, mask=_bp_row_mask(offs, bp_n, BLOCK_N, D), cache_modifier=".cs"
    )


@triton.jit
def _grouped_ln_residual_fwd_kernel(
    H0,
    H1,
    H2,
    Y,
    W0,
    B0,
    W1,
    B1,
    W2,
    B2,
    Mean0,
    Rstd0,
    Mean1,
    Rstd1,
    Mean2,
    Rstd2,
    N,
    Drt,
    Nm,
    eps,
    sh0,
    sh1,
    sh2,
    sy,
    D: tl.constexpr,
    NUM_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.arange(0, D)
    w0 = tl.load(W0 + cols).to(tl.float32)
    b0 = tl.load(B0 + cols).to(tl.float32)
    w1 = tl.load(W1 + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)
    if NUM_IN == 3:
        w2 = tl.load(W2 + cols).to(tl.float32)
        b2 = tl.load(B2 + cols).to(tl.float32)
    else:
        w2 = w1
        b2 = b1
    start_row = tl.program_id(0) * BLOCK_N
    # All row tiles are loaded before the first statistics store. Pointer
    # arguments are not noalias, so a load placed after a tl.store could not
    # be hoisted above it. Only the issue point moves; the values do not.
    x0 = _tail_ld(H0, N, sh0, start_row, D, BLOCK_N)
    x1 = _tail_ld(H1, N, sh1, start_row, D, BLOCK_N)
    if NUM_IN == 3:
        x2 = _tail_ld(H2, N, sh2, start_row, D, BLOCK_N)
    else:
        x2 = x1
    # LN_0 has no residual. The shipped kernel stores y.to(Y dtype), and that
    # bf16 tensor is what the next launch reads back as its residual.
    y0 = _tail_ln(x0, w0, b0, Mean0, Rstd0, N, Drt, Nm, eps, start_row, D, BLOCK_N)
    acc = y0.to(Y.dtype.element_ty)
    # LN_k with residual = acc: ADD_RESIDUAL's round-before-add, then the
    # store-side cast, exactly as _weighted_layer_norm_fwd writes them.
    y1 = _tail_ln(x1, w1, b1, Mean1, Rstd1, N, Drt, Nm, eps, start_row, D, BLOCK_N)
    y1 = y1.to(Y.dtype.element_ty).to(tl.float32) + acc.to(tl.float32)
    acc = y1.to(Y.dtype.element_ty)
    if NUM_IN == 3:
        y2 = _tail_ln(
            x2, w2, b2, Mean2, Rstd2, N, Drt, Nm, eps, start_row, D, BLOCK_N
        )
        y2 = y2.to(Y.dtype.element_ty).to(tl.float32) + acc.to(tl.float32)
        acc = y2.to(Y.dtype.element_ty)
    _tail_st(Y, acc, N, sy, start_row, D, BLOCK_N)


def _check_inputs(
    hs: List[torch.Tensor], ws: List[torch.Tensor], bs: List[torch.Tensor]
) -> Tuple[int, int, int]:
    num_in = len(hs)
    if num_in not in (2, 3):
        raise ValueError(f"grouped_ln_residual_fwd: need 2 or 3 inputs, got {num_in}")
    if len(ws) != num_in or len(bs) != num_in:
        raise ValueError("grouped_ln_residual_fwd: hs/ws/bs length mismatch")
    h0 = hs[0]
    if h0.dim() != 2:
        raise ValueError(
            f"grouped_ln_residual_fwd: expected 2-D inputs, got {h0.dim()}-D"
        )
    N, D = h0.shape
    for k, h in enumerate(hs):
        if h.shape != h0.shape or h.dtype != h0.dtype or h.device != h0.device:
            raise ValueError(
                f"grouped_ln_residual_fwd: input {k} is {tuple(h.shape)}/{h.dtype}/"
                f"{h.device}, input 0 is {tuple(h0.shape)}/{h0.dtype}/{h0.device}"
            )
        if not h.is_contiguous():
            raise ValueError(f"grouped_ln_residual_fwd: input {k} is not contiguous")
        if (
            ws[k].dim() != 1
            or ws[k].numel() != D
            or bs[k].dim() != 1
            or bs[k].numel() != D
        ):
            raise ValueError(f"grouped_ln_residual_fwd: norm params {k} are not [{D}]")
    # The kernel loads the full row as one [BLOCK_N, D] tile: D must be a power
    # of two (no column mask on the loads) and within the shipped 64 KB cap.
    if D <= 0 or D & (D - 1) or D * h0.element_size() > 65536:
        raise ValueError(
            f"grouped_ln_residual_fwd: D={D} must be a power of two of at most "
            "64 KB per row"
        )
    return num_in, N, D


def grouped_ln_residual_fwd(
    hs: List[torch.Tensor],
    ws: List[torch.Tensor],
    bs: List[torch.Tensor],
    eps: float,
    BLOCK_N: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Grouped LN-residual forward tail.

    out = LN_{K-1}(h_{K-1}) (+) ... (+) LN_1(h_1) (+) LN_0(h_0), evaluated left
    to right exactly as the shipped F2 chain:
        acc = bf16(LN_0(h_0))
        acc = bf16(bf16(LN_k(h_k)).float() + acc.float())   for k >= 1
    Returns (out, [mean_k], [rstd_k]): out is [N, D] in h's dtype, and the
    stats are fp32 [N] per LN. hs: 2 or 3 contiguous [N, D] tensors of one float
    dtype. ws, bs: [D] norm weight and bias per input. BLOCK_N overrides the
    fixed config for probe sweeps only. num_warps must be 1.
    """
    num_in, N, D = _check_inputs(hs, ws, bs)
    nw = _TAIL_NUM_WARPS if num_warps is None else int(num_warps)
    if nw != 1:
        raise ValueError(
            "grouped_ln_residual_fwd: num_warps must be 1 (the reduction tree of "
            "the shipped pinned LayerNorm launches)"
        )
    block_n = _TAIL_BLOCK_N if BLOCK_N is None else int(BLOCK_N)
    h0 = hs[0]
    out = torch.empty_like(h0)
    means = [torch.empty((N,), dtype=torch.float32, device=h0.device) for _ in hs]
    rstds = [torch.empty((N,), dtype=torch.float32, device=h0.device) for _ in hs]
    if N == 0:
        return out, means, rstds
    h2 = hs[2] if num_in == 3 else hs[1]
    w2 = ws[2] if num_in == 3 else ws[1]
    b2 = bs[2] if num_in == 3 else bs[1]
    m2 = means[2] if num_in == 3 else means[1]
    r2 = rstds[2] if num_in == 3 else rstds[1]
    grid = (triton.cdiv(N, block_n),)
    _grouped_ln_residual_fwd_kernel[grid](
        hs[0],
        hs[1],
        h2,
        out,
        ws[0],
        bs[0],
        ws[1],
        bs[1],
        w2,
        b2,
        means[0],
        rstds[0],
        means[1],
        rstds[1],
        m2,
        r2,
        N,
        D,
        N,
        eps,
        hs[0].stride(0),
        hs[1].stride(0),
        h2.stride(0),
        out.stride(0),
        D=D,
        NUM_IN=num_in,
        BLOCK_N=block_n,
        num_warps=1,
        num_stages=1,
        # AMD-only launch option (the verified ROCm launch passes it);
        # CUDA Triton rejects unknown keyword arguments.
        **({"waves_per_eu": 0} if torch.version.hip else {}),
    )
    return out, means, rstds


def grouped_ln_residual_fwd_config() -> Tuple[int, int]:
    """(BLOCK_N, num_warps) of the fixed launch config."""
    return _TAIL_BLOCK_N, _TAIL_NUM_WARPS
