###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Fused parameter-less RMSNorm + skinny fp32 projection for the V4 HyperConnection.

``HyperMixer._packed_logits`` / ``HyperHead.forward`` compute

    logits = F.linear(rms_norm(x_flat).float(), W)      x_flat [T, Kd], W [N, Kd] fp32, N = 24 / 4

as ``fused_rms_norm(..., out_dtype=fp32)`` followed by an fp32 GEMM. That writes and re-reads an
fp32 ``[T, Kd]`` tensor (256 MB at T=4096, Kd=16384), runs an N=24 fp32 GEMM that hipBLASLt handles
poorly, and the backward materializes another fp32 ``[T, Kd]`` for ``d_normed``.

RMSNorm only scales each row, so ``logits = (x @ W^T) * rstd``. Here:

* forward reads ``x`` once and accumulates the N dot products and ``sum(x^2)`` together (fp32);
* backward, with ``g = dlogits * rstd`` and ``c = sum_n dlogits * raw`` (``raw = x @ W^T``):
  ``dx = g @ W - x * rstd^3 * c / Kd`` and ``dW = g^T @ x``, one pass over ``x``, no atomics.

Precision (``PREC``): the fp32 products can run as plain fp32 FMA (``"ieee"``), or on the bf16
matrix cores with the fp32 operand split into bf16 parts: ``"split3"`` (hi + mid + lo, the full
24-bit mantissa) or ``"split2"`` (hi + lo, 16 bits). ``x`` is bf16 already, so only the fp32 side
is split. Accumulation is fp32 in every mode; ``dx`` is written in the dtype of ``x``.
"""

import os

import torch
import triton
import triton.language as tl

# Launch config; defaults tuned on gfx1250 (MI455X). T_SPLIT > 1 splits the backward over rows
# and reduces the dW partials afterwards (deterministic).
HC_CFG = dict(
    fwd_bt=32,
    fwd_bk=64,
    fwd_chunk=1024,
    fwd_warps=4,
    fwd_stages=2,
    fwd_wt=False,  # pass W pre-transposed ([Kd, N] contiguous) to the fwd kernel
    fwd_prec="ieee",  # bf16 x: "wmma3" is no faster on gfx1250 (ledger exp 5); "ieee" / "split2" / "split3" = fp32-FMA-side kernels
    bwd_bt=64,
    bwd_bk=64,
    bwd_tsplit=4,
    bwd_warps=8,
    bwd_stages=2,
    bwd_prec="ieee",
    # wmma3 kernels
    w3_fwd_bt=64,
    w3_fwd_bk=128,
    w3_fwd_chunk=2048,
    w3_fwd_warps=4,
    w3_fwd_stages=2,
    w3_bwd_bt=64,
    w3_bwd_bk=128,
    w3_bwd_tsplit=8,
    w3_bwd_warps=4,
    w3_bwd_stages=2,
)


# hc track (PRIMUS_HC_RMSLIN_TUNED=1): bit-identical retune of the fp32-ieee kernels used by HyperHead (N=4, Kd=16384,
# T=4096; bench_rmslin.py, 112 configs, out/dx/dW byte-equal): fwd bt32 -> 64 (41.2 -> 32.0 us wall),
# bwd 8 warps / 2 stages -> 4 warps / 1 stage (124.1 -> 76.5 us wall incl. prep and the dW sum).
_HC_CFG_TUNED = dict(fwd_bt=64, bwd_warps=4, bwd_stages=1)


def _cfg():
    if os.environ.get("PRIMUS_HC_RMSLIN_TUNED", "0") == "1":
        return {**HC_CFG, **_HC_CFG_TUNED}
    return HC_CFG


def hc_rms_linear_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_RMS_LINEAR", "1") != "0"


@triton.jit
def _split_bf16(v, N_PARTS: tl.constexpr):
    """Split fp32 ``v`` into bf16 parts whose sum reproduces ``v`` to 8*N_PARTS mantissa bits."""
    hi = v.to(tl.bfloat16)
    r = v - hi.to(tl.float32)
    mid = r.to(tl.bfloat16)
    if N_PARTS == 3:
        lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
    else:
        lo = mid
    return hi, mid, lo


@triton.jit
def _dot_bf16_x_f32(a_bf16, b_f32, PREC: tl.constexpr):
    """a (any dtype; bf16 in the split modes) @ b (fp32), fp32 result."""
    if PREC == "ieee":
        return tl.dot(a_bf16.to(tl.float32), b_f32, input_precision="ieee")
    elif PREC == "tf32":
        return tl.dot(a_bf16.to(tl.float32), b_f32, input_precision="tf32")
    elif PREC == "split3":
        b_hi, b_mid, b_lo = _split_bf16(b_f32, 3)
        acc = tl.dot(a_bf16, b_hi)
        acc += tl.dot(a_bf16, b_mid)
        acc += tl.dot(a_bf16, b_lo)
        return acc
    else:  # split2
        b_hi, b_lo, _unused = _split_bf16(b_f32, 2)
        acc = tl.dot(a_bf16, b_hi)
        acc += tl.dot(a_bf16, b_lo)
        return acc


@triton.jit
def _dot_f32_x_f32(a_f32, b_f32, PREC: tl.constexpr):
    """a (fp32) @ b (fp32), fp32 result."""
    if PREC == "ieee":
        return tl.dot(a_f32, b_f32, input_precision="ieee")
    elif PREC == "tf32":
        return tl.dot(a_f32, b_f32, input_precision="tf32")
    elif PREC == "split3":
        a_hi, a_mid, a_lo = _split_bf16(a_f32, 3)
        b_hi, b_mid, b_lo = _split_bf16(b_f32, 3)
        # terms down to 2^-24 relative: hi*hi, hi*mid, mid*hi, hi*lo, mid*mid, lo*hi
        acc = tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_mid)
        acc += tl.dot(a_mid, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_mid, b_mid)
        acc += tl.dot(a_lo, b_hi)
        return acc
    else:  # split2
        a_hi, a_lo, _ua = _split_bf16(a_f32, 2)
        b_hi, b_lo, _ub = _split_bf16(b_f32, 2)
        acc = tl.dot(a_hi, b_hi)
        acc += tl.dot(a_hi, b_lo)
        acc += tl.dot(a_lo, b_hi)
        return acc


@triton.jit
def _hc_rms_linear_fwd_kernel(
    X,
    W,
    P_RAW,
    P_SS,
    T,
    KD,
    N,
    stride_x,
    CHUNK,
    stride_wk,
    stride_wn,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    ncol = tl.arange(0, BLOCK_N)
    rmask = rows < T
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_T,), dtype=tl.float32)
    k_start = pid_s * CHUNK
    for kk in range(0, CHUNK, BLOCK_K):
        ks = k_start + kk + tl.arange(0, BLOCK_K)
        kmask = ks < KD
        x = tl.load(X + rows[:, None] * stride_x + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(W + ncol[None, :] * stride_wn + ks[:, None] * stride_wk, mask=(ncol[None, :] < N) & kmask[:, None], other=0.0)
        acc += _dot_bf16_x_f32(x, w, PREC)
        xf = x.to(tl.float32)
        ss += tl.sum(xf * xf, axis=1)
    out_rows = pid_s * T + rows
    tl.store(P_RAW + out_rows[:, None] * BLOCK_N + ncol[None, :], acc, mask=rmask[:, None])
    tl.store(P_SS + out_rows, ss, mask=rmask)


@triton.jit
def _hc_rms_linear_bwd_kernel(
    X,
    W,
    G,
    COEF,
    DX,
    DW,
    T,
    KD,
    N,
    stride_x,
    stride_dx,
    T_PER_SPLIT,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_s = tl.program_id(1)
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = ks < KD
    ncol = tl.arange(0, BLOCK_N)
    nmask = ncol < N
    w = tl.load(W + ncol[:, None] * KD + ks[None, :], mask=nmask[:, None] & kmask[None, :], other=0.0)  # [BN, BK]
    dw = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    t_lo = pid_s * T_PER_SPLIT
    for t0 in range(t_lo, t_lo + T_PER_SPLIT, BLOCK_T):
        rows = t0 + tl.arange(0, BLOCK_T)
        rmask = rows < T
        x = tl.load(X + rows[:, None] * stride_x + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        g = tl.load(G + rows[:, None] * BLOCK_N + ncol[None, :], mask=rmask[:, None], other=0.0)  # [BT, BN]
        coef = tl.load(COEF + rows, mask=rmask, other=0.0)
        dx = _dot_f32_x_f32(g, w, PREC) - x.to(tl.float32) * coef[:, None]
        tl.store(
            DX + rows[:, None] * stride_dx + ks[None, :],
            dx.to(DX.dtype.element_ty),
            mask=rmask[:, None] & kmask[None, :],
        )
        # dW^T-block = g^T @ x : x exact bf16 -> split only g
        dw += tl.trans(_dot_bf16_x_f32(tl.trans(x), g, PREC))
    tl.store(DW + pid_s * BLOCK_N * KD + ncol[:, None] * KD + ks[None, :], dw, mask=nmask[:, None] & kmask[None, :])


@triton.jit
def _hc_fwd_finalize_kernel(
    P_RAW,
    P_SS,
    RAW,
    RSTD,
    OUT,
    T,
    N,
    KD,
    eps,
    N_SPLIT,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum the K-split partials; rstd = (sum(x^2)/Kd + eps)^-1/2; out = raw[:, :N] * rstd."""
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    ncol = tl.arange(0, BLOCK_N)
    rmask = rows < T
    raw = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for s in range(0, N_SPLIT):
        raw += tl.load(P_RAW + (s * T + rows)[:, None] * BLOCK_N + ncol[None, :], mask=rmask[:, None], other=0.0)
        ss += tl.load(P_SS + s * T + rows, mask=rmask, other=0.0)
    rstd = 1.0 / tl.sqrt(ss / KD + eps)
    tl.store(RAW + rows[:, None] * BLOCK_N + ncol[None, :], raw, mask=rmask[:, None])
    tl.store(RSTD + rows, rstd, mask=rmask)
    tl.store(OUT + rows[:, None] * N + ncol[None, :], raw * rstd[:, None], mask=rmask[:, None] & (ncol[None, :] < N))


@triton.jit
def _hc_bwd_prep_kernel(
    DL,
    stride_dl,
    RAW,
    RSTD,
    G,
    COEF,
    T,
    N,
    KD,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """g = pad(dlogits) * rstd ([T, BLOCK_N], zero padded); coef = rstd^3 * sum_n(dlogits * raw) / Kd."""
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    ncol = tl.arange(0, BLOCK_N)
    rmask = rows < T
    m2 = rmask[:, None] & (ncol[None, :] < N)
    dl = tl.load(DL + rows[:, None] * stride_dl + ncol[None, :], mask=m2, other=0.0).to(tl.float32)
    raw = tl.load(RAW + rows[:, None] * BLOCK_N + ncol[None, :], mask=rmask[:, None], other=0.0)
    rstd = tl.load(RSTD + rows, mask=rmask, other=0.0)
    c = tl.sum(dl * raw, axis=1)
    tl.store(G + rows[:, None] * BLOCK_N + ncol[None, :], dl * rstd[:, None], mask=rmask[:, None])
    tl.store(COEF + rows, rstd * rstd * rstd * c / KD, mask=rmask)


# ---------------------------------------------------------------------------------------------------
# "wmma3" path (default for bf16 x): the fp32 operands (W, and g in the backward) are split ONCE into
# three bf16 parts (hi + mid + lo = the full 24-bit mantissa) outside the main loops, so the hot loops
# are pure bf16 WMMA dots with fp32 accumulation. x is exact bf16, so this keeps fp32-level accuracy
# while the FLOPs move off the fp32 FMA units (which made the "ieee" kernels compute-bound).
# ---------------------------------------------------------------------------------------------------


@triton.jit
def _split3_rows_kernel(SRC, DST, R, R_PAD, C, stride_src, scale_ptr, HAS_SCALE: tl.constexpr, BLOCK_C: tl.constexpr):
    """DST[p, r, :] (bf16, p = 0..2) = 3-way bf16 split of SRC[r, :] * scale[r]; rows >= R are zero."""
    r = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < C
    v = tl.load(SRC + r * stride_src + cols, mask=cmask & (r < R), other=0.0).to(tl.float32)
    if HAS_SCALE:
        v = v * tl.load(scale_ptr + r, mask=r < R, other=0.0)
    hi = v.to(tl.bfloat16)
    rem = v - hi.to(tl.float32)
    mid = rem.to(tl.bfloat16)
    lo = (rem - mid.to(tl.float32)).to(tl.bfloat16)
    base = DST + r * C + cols
    tl.store(base, hi, mask=cmask)
    tl.store(base + R_PAD * C, mid, mask=cmask)
    tl.store(base + 2 * R_PAD * C, lo, mask=cmask)


def _split3_rows(src: torch.Tensor, r_pad: int, scale: torch.Tensor | None = None) -> torch.Tensor:
    R, C = src.shape
    dst = torch.empty((3, r_pad, C), dtype=torch.bfloat16, device=src.device)
    BLOCK_C = 1024
    _split3_rows_kernel[(r_pad, triton.cdiv(C, BLOCK_C))](
        src, dst, R, r_pad, C, src.stride(0), scale if scale is not None else src,
        HAS_SCALE=scale is not None, BLOCK_C=BLOCK_C, num_warps=4,
    )
    return dst


@triton.jit
def _hc_fwd_wmma3_kernel(
    X, W3, P_RAW, P_SS, T, KD, stride_x, CHUNK,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    ncol = tl.arange(0, BLOCK_N)
    rmask = rows < T
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_T,), dtype=tl.float32)
    PART = BLOCK_N * KD
    k_start = pid_s * CHUNK
    for kk in range(0, CHUNK, BLOCK_K):
        ks = k_start + kk + tl.arange(0, BLOCK_K)
        kmask = ks < KD
        x = tl.load(X + rows[:, None] * stride_x + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        wp = W3 + ncol[None, :] * KD + ks[:, None]  # [BK, BN] view of part 0
        w0 = tl.load(wp, mask=kmask[:, None], other=0.0)
        w1 = tl.load(wp + PART, mask=kmask[:, None], other=0.0)
        w2 = tl.load(wp + 2 * PART, mask=kmask[:, None], other=0.0)
        acc += tl.dot(x, w0)
        acc += tl.dot(x, w1)
        acc += tl.dot(x, w2)
        xf = x.to(tl.float32)
        ss += tl.sum(xf * xf, axis=1)
    out_rows = pid_s * T + rows
    tl.store(P_RAW + out_rows[:, None] * BLOCK_N + ncol[None, :], acc, mask=rmask[:, None])
    tl.store(P_SS + out_rows, ss, mask=rmask)


@triton.jit
def _hc_bwd_wmma3_kernel(
    X, W3, G3, COEF, DX, DW, T, KD, stride_x, stride_dx, T_PER_SPLIT,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_s = tl.program_id(1)
    ks = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = ks < KD
    ncol = tl.arange(0, BLOCK_N)
    WPART = BLOCK_N * KD
    GPART = T * BLOCK_N
    wp = W3 + ncol[:, None] * KD + ks[None, :]  # [BN, BK]
    w0 = tl.load(wp, mask=kmask[None, :], other=0.0)
    w1 = tl.load(wp + WPART, mask=kmask[None, :], other=0.0)
    w2 = tl.load(wp + 2 * WPART, mask=kmask[None, :], other=0.0)
    dw = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    t_lo = pid_s * T_PER_SPLIT
    for t0 in range(t_lo, t_lo + T_PER_SPLIT, BLOCK_T):
        rows = t0 + tl.arange(0, BLOCK_T)
        rmask = rows < T
        x = tl.load(X + rows[:, None] * stride_x + ks[None, :], mask=rmask[:, None] & kmask[None, :], other=0.0)
        gp = G3 + rows[:, None] * BLOCK_N + ncol[None, :]  # [BT, BN]
        g0 = tl.load(gp, mask=rmask[:, None], other=0.0)
        g1 = tl.load(gp + GPART, mask=rmask[:, None], other=0.0)
        g2 = tl.load(gp + 2 * GPART, mask=rmask[:, None], other=0.0)
        coef = tl.load(COEF + rows, mask=rmask, other=0.0)
        # g @ W to 2^-24: hi*hi + hi*mid + mid*hi + hi*lo + mid*mid + lo*hi
        acc = tl.dot(g0, w0)
        acc += tl.dot(g0, w1)
        acc += tl.dot(g1, w0)
        acc += tl.dot(g0, w2)
        acc += tl.dot(g1, w1)
        acc += tl.dot(g2, w0)
        dx = acc - x.to(tl.float32) * coef[:, None]
        tl.store(
            DX + rows[:, None] * stride_dx + ks[None, :],
            dx.to(DX.dtype.element_ty),
            mask=rmask[:, None] & kmask[None, :],
        )
        # dW block = g^T @ x (x exact bf16)
        dw += tl.dot(tl.trans(g0), x)
        dw += tl.dot(tl.trans(g1), x)
        dw += tl.dot(tl.trans(g2), x)
    tl.store(DW + pid_s * BLOCK_N * KD + ncol[:, None] * KD + ks[None, :], dw, mask=kmask[None, :])


def _block_n(n: int) -> int:
    return max(16, triton.next_power_of_2(n))


class HcRmsLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d: torch.Tensor, weight: torch.Tensor, eps: float):  # type: ignore[override]
        assert x2d.dim() == 2 and weight.dim() == 2 and weight.shape[1] == x2d.shape[1]
        x2d = x2d.contiguous()
        w = weight.float().contiguous()
        T, KD = x2d.shape
        N = w.shape[0]
        BN = _block_n(N)
        c = _cfg()
        # bf16 x feeds the bf16 matrix cores as-is; other dtypes fall back to the fp32-FMA kernels
        prec = c["fwd_prec"] if x2d.dtype in (torch.bfloat16,) else "ieee"
        wmma3 = prec == "wmma3"
        if wmma3:
            BLOCK_T, BLOCK_K, chunk_cfg = c["w3_fwd_bt"], c["w3_fwd_bk"], c["w3_fwd_chunk"]
        else:
            BLOCK_T, BLOCK_K, chunk_cfg = c["fwd_bt"], c["fwd_bk"], c["fwd_chunk"]
        n_split = max(1, triton.cdiv(KD, chunk_cfg))
        chunk = triton.cdiv(triton.cdiv(KD, n_split), BLOCK_K) * BLOCK_K
        n_split = triton.cdiv(KD, chunk)
        p_raw = torch.empty((n_split, T, BN), dtype=torch.float32, device=x2d.device)
        p_ss = torch.empty((n_split, T), dtype=torch.float32, device=x2d.device)
        grid = (triton.cdiv(T, BLOCK_T), n_split)
        w3 = None
        if wmma3:
            w3 = _split3_rows(w, BN)
            _hc_fwd_wmma3_kernel[grid](
                x2d, w3, p_raw, p_ss, T, KD, x2d.stride(0), chunk,
                BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K, BLOCK_N=BN,
                num_warps=c["w3_fwd_warps"], num_stages=c["w3_fwd_stages"],
            )
        else:
            w_src = w.t().contiguous() if c.get("fwd_wt", False) else w
            swk, swn = (w_src.stride(0), w_src.stride(1)) if c.get("fwd_wt", False) else (1, KD)
            _hc_rms_linear_fwd_kernel[grid](
            x2d,
            w_src,
            p_raw,
            p_ss,
            T,
            KD,
            N,
            x2d.stride(0),
            chunk,
            swk,
            swn,
            BLOCK_T=BLOCK_T,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BN,
            PREC=prec,
            num_warps=c["fwd_warps"],
            num_stages=c["fwd_stages"],
        )
        raw = torch.empty((T, BN), dtype=torch.float32, device=x2d.device)  # padded columns are 0
        rstd = torch.empty((T,), dtype=torch.float32, device=x2d.device)
        logits = torch.empty((T, N), dtype=torch.float32, device=x2d.device)
        _hc_fwd_finalize_kernel[(triton.cdiv(T, 64),)](
            p_raw, p_ss, raw, rstd, logits, T, N, KD, float(eps), n_split, BLOCK_T=64, BLOCK_N=BN, num_warps=4
        )
        ctx.save_for_backward(x2d, w, raw, rstd, w3 if wmma3 else w)
        ctx.wmma3 = wmma3
        ctx.n = N
        ctx.weight_dtype = weight.dtype
        return logits

    @staticmethod
    def backward(ctx, dlogits: torch.Tensor):  # type: ignore[override]
        x2d, w, raw, rstd, w3 = ctx.saved_tensors
        N = ctx.n
        T, KD = x2d.shape
        BN = raw.shape[1]
        if dlogits.stride(-1) != 1:
            dlogits = dlogits.contiguous()
        if ctx.wmma3:
            return HcRmsLinearFn._backward_wmma3(ctx, dlogits, x2d, w3, raw, rstd, N, T, KD, BN)
        g = torch.empty((T, BN), dtype=torch.float32, device=x2d.device)
        coef = torch.empty((T,), dtype=torch.float32, device=x2d.device)
        _hc_bwd_prep_kernel[(triton.cdiv(T, 64),)](
            dlogits, dlogits.stride(0), raw, rstd, g, coef, T, N, KD, BLOCK_T=64, BLOCK_N=BN, num_warps=4
        )
        dx = torch.empty_like(x2d)
        c = _cfg()
        prec = c["bwd_prec"] if x2d.dtype in (torch.bfloat16,) else "ieee"
        BLOCK_T, BLOCK_K, tsplit = c["bwd_bt"], c["bwd_bk"], c["bwd_tsplit"]
        t_per = triton.cdiv(triton.cdiv(T, tsplit), BLOCK_T) * BLOCK_T
        tsplit = triton.cdiv(T, t_per)
        dw_p = torch.empty((tsplit, BN, KD), dtype=torch.float32, device=x2d.device)
        _hc_rms_linear_bwd_kernel[(triton.cdiv(KD, BLOCK_K), tsplit)](
            x2d,
            w,
            g,
            coef,
            dx,
            dw_p,
            T,
            KD,
            N,
            x2d.stride(0),
            dx.stride(0),
            t_per,
            BLOCK_T=BLOCK_T,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BN,
            PREC=prec,
            num_warps=c["bwd_warps"],
            num_stages=c["bwd_stages"],
        )
        dw = dw_p[0, :N] if tsplit == 1 else dw_p.sum(0)[:N]
        return dx, dw.to(ctx.weight_dtype), None

    @staticmethod
    def _backward_wmma3(ctx, dlogits, x2d, w3, raw, rstd, N, T, KD, BN):
        # g = pad(dlogits) * rstd and coef, then g split into 3 bf16 parts [3, T, BN]
        g = torch.empty((T, BN), dtype=torch.float32, device=x2d.device)
        coef = torch.empty((T,), dtype=torch.float32, device=x2d.device)
        _hc_bwd_prep_kernel[(triton.cdiv(T, 64),)](
            dlogits, dlogits.stride(0), raw, rstd, g, coef, T, N, KD, BLOCK_T=64, BLOCK_N=BN, num_warps=4
        )
        g3 = _split3_rows(g, T)  # [3, T, BN] bf16
        dx = torch.empty_like(x2d)
        c = _cfg()
        BLOCK_T, BLOCK_K, tsplit = c["w3_bwd_bt"], c["w3_bwd_bk"], c["w3_bwd_tsplit"]
        t_per = triton.cdiv(triton.cdiv(T, tsplit), BLOCK_T) * BLOCK_T
        tsplit = triton.cdiv(T, t_per)
        dw_p = torch.empty((tsplit, BN, KD), dtype=torch.float32, device=x2d.device)
        _hc_bwd_wmma3_kernel[(triton.cdiv(KD, BLOCK_K), tsplit)](
            x2d, w3, g3, coef, dx, dw_p, T, KD, x2d.stride(0), dx.stride(0), t_per,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K, BLOCK_N=BN,
            num_warps=c["w3_bwd_warps"], num_stages=c["w3_bwd_stages"],
        )
        dw = dw_p[0, :N] if tsplit == 1 else dw_p.sum(0)[:N]
        return dx, dw.to(ctx.weight_dtype), None


def hc_rms_linear(x_flat: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``F.linear(rms_norm(x_flat), weight)`` in fp32: ``x_flat`` ``[..., Kd]`` -> ``[..., N]`` fp32."""
    lead = x_flat.shape[:-1]
    out = HcRmsLinearFn.apply(x_flat.reshape(-1, x_flat.shape[-1]), weight, eps)
    return out.reshape(*lead, weight.shape[0])
