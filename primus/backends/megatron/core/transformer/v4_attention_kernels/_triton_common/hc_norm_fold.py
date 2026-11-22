###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""RMSNorm folded into the HC collapse (F2) and dpre (B2) kernels, plus q-norm fused with the wq_b input quant
(opt3-mem, 2026-11-20). Opt-in: PRIMUS_HC_FOLD_NORM=1 (HC folds) - CPU-verified, not yet timed on the GPU.

Per V4 layer the two big norms (input_layernorm on the attention HC collapse, pre_mlp_layernorm on the FFN HC
collapse, [4096, 4096] bf16) read the collapsed hidden right after the HC wrote it, and in backward write a dx that
the HC dpre kernel immediately reads back:

  unfused fwd : collapse (read x 128 MB, write h 32 MB) -> norm (read h 32, write y 32)              4 launches/HC*
  fused fwd   : collapse+norm (read x 128 MB, write y 32 MB [+ FP8 y 16 MB + scales])              1 launch
  unfused bwd : norm bwd (read h, dy; write dh) + dg reduce + dpre2d (read x 128, dh 32)          3 launches
  fused bwd   : dpre+norm bwd (read x 128, dy 32; write dh 32 = HC dcoll) + dg reduce              2 launches
  (*rmsnorm_g90 fwd 1 + bwd 2 launches + HC collapse/dpre)

Saved per HC application ~96 MB of traffic and 2 launches (32 per step at 16 HC applications with a norm);
estimated 0.9 ms/step with the q-norm + quant fusion below.

Numerics are those of the unfused path: the collapsed hidden is rounded to bf16 before the norm (the unfused path
stores it in bf16), rstd / y / dh use the rmsnorm_g90 formulas, dh is rounded to bf16 before the dpre dot (as the
unfused dcoll), dpre partials are per t2_bc column tile (the layout _hc_pre_bwd_kernel reads).

FP8 out (HCF_CFG["fold_fp8"]): the 1x128 row quant of y (Turbo BLOCKWISE layout: fp8 [M, D], scale_inv fp32
[M, D/128], UE8M0 when PRIMUS_TURBO_FP8_BLOCK_UE8M0=1) is written by the same kernel and attached to y as
``y._fp8_row = (q, scale_inv)``. Consuming it needs a pre-quantized-input entry in the Turbo blockwise linear (not
present yet); the col (128 x 1) quant for wgrad still needs all 128 rows of a block, so it stays in the dual quant.

q-norm + wq_b input quant (_qnorm_dual_quant_kernel): RMSNorm of q_a [4096, 1024] fused with the blockwise dual
(row 1x128 + col 128x1) FP8 quant that the wq_b linear runs on its input; grid (M/128, N/128); each program
recomputes the rstd of its 128 rows over the full row (L2 hits) and quantizes its 128x128 tile both ways.
Outputs match quant_fp8_blockwise_dual_kernel(rmsnorm(x)) on the bf16-rounded y.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def hc_fold_norm_enabled() -> bool:
    return os.environ.get("PRIMUS_HC_FOLD_NORM", "0") == "1"


@triton.jit
def _pow2_floor(s):
    return (s.to(tl.int32, bitcast=True) & 0x7F800000).to(tl.float32, bitcast=True)


@triton.jit
def _ld_rows4(X, t64, cols, m2, D):
    base = X + t64[:, None] * (4 * D) + cols[None, :]
    x0 = tl.load(base, mask=m2, other=0.0)
    x1 = tl.load(base + D, mask=m2, other=0.0)
    x2 = tl.load(base + 2 * D, mask=m2, other=0.0)
    x3 = tl.load(base + 3 * D, mask=m2, other=0.0)
    return x0, x1, x2, x3


@triton.jit
def _ld_pre4(P, t64, rm):
    p0 = tl.load(P + t64 * 4, mask=rm, other=0.0).to(tl.float32)
    p1 = tl.load(P + t64 * 4 + 1, mask=rm, other=0.0).to(tl.float32)
    p2 = tl.load(P + t64 * 4 + 2, mask=rm, other=0.0).to(tl.float32)
    p3 = tl.load(P + t64 * 4 + 3, mask=rm, other=0.0).to(tl.float32)
    return p0, p1, p2, p3


@triton.jit
def _collapse_bf16(x0, x1, x2, x3, p0, p1, p2, p3):
    h = p0[:, None] * x0.to(tl.float32) + p1[:, None] * x1.to(tl.float32)
    h += p2[:, None] * x2.to(tl.float32) + p3[:, None] * x3.to(tl.float32)
    return h.to(tl.bfloat16).to(tl.float32)  # the unfused path stores the collapsed hidden in bf16


# ------------------------------------------------------------------------------------------------
# F2': y = rmsnorm(sum_k pre_k x_k) * gamma  (+ optional 1x128 FP8 row quant of y)
# ------------------------------------------------------------------------------------------------
@triton.jit
def _hc_collapse_norm_fwd_kernel(X, PRE, GAMMA, Y, RSTD, YQ, YS, M, D, eps,
                                 BD: tl.constexpr, RB: tl.constexpr, FP8: tl.constexpr, FP8_MAX: tl.constexpr,
                                 POW2: tl.constexpr, QB: tl.constexpr):
    rows = tl.program_id(0) * RB + tl.arange(0, RB)
    rm = rows < M
    cols = tl.arange(0, BD)
    cm = cols < D
    t64 = rows.to(tl.int64)
    m2 = rm[:, None] & cm[None, :]
    x0, x1, x2, x3 = _ld_rows4(X, t64, cols, m2, D)
    p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    h = _collapse_bf16(x0, x1, x2, x3, p0, p1, p2, p3)
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=1) / D + eps)
    y = (h * rstd[:, None] * g[None, :]).to(Y.dtype.element_ty)
    off = t64[:, None] * D + cols[None, :]
    tl.store(Y + off, y, mask=m2)
    tl.store(RSTD + rows, rstd, mask=rm)
    if FP8:
        NG: tl.constexpr = BD // QB
        y3 = tl.reshape(y.to(tl.float32), (RB, NG, QB))
        amax = tl.maximum(tl.max(tl.abs(y3), axis=2), 1e-4)
        sc = FP8_MAX / amax
        if POW2:
            sc = _pow2_floor(sc)
        q = tl.clamp(y3 * sc[:, :, None], -FP8_MAX, FP8_MAX)
        tl.store(YQ + off, tl.reshape(q, (RB, BD)).to(YQ.dtype.element_ty), mask=m2)
        gi = tl.arange(0, NG)
        tl.store(YS + t64[:, None] * (D // QB) + gi[None, :], 1.0 / sc, mask=rm[:, None] & (gi[None, :] < D // QB))


# ------------------------------------------------------------------------------------------------
# B2': dh = rmsnorm_bwd(dy) (written as the HC dcoll), dpre partials per BC column tile, dgamma partials.
# Persistent row programs; the next row block's loads are issued before the current one is consumed.
# ------------------------------------------------------------------------------------------------
@triton.jit
def _dpre_store(DPREP, xk, dhf, c_t, t64, rm, M, k: tl.constexpr, RB: tl.constexpr, NCT: tl.constexpr,
                BC: tl.constexpr):
    pk = tl.sum(tl.reshape(xk.to(tl.float32) * dhf, (RB, NCT, BC)), axis=2)  # [RB, NCT]
    tl.store(DPREP + c_t[None, :].to(tl.int64) * M * 4 + t64[:, None] * 4 + k, pk, mask=rm[:, None])


@triton.jit
def _hc_norm_dpre_bwd_kernel(X, PRE, GAMMA, RSTD, DY, DH, DPREP, DGP, M, D,
                             BD: tl.constexpr, RB: tl.constexpr, BC: tl.constexpr, NPROG: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BD)
    cm = cols < D
    g = tl.load(GAMMA + cols, mask=cm, other=0.0).to(tl.float32)
    NCT: tl.constexpr = BD // BC
    c_t = tl.arange(0, NCT)
    dg = tl.zeros((RB, BD), dtype=tl.float32)
    STEP: tl.constexpr = NPROG * RB
    rows = pid * RB + tl.arange(0, RB)
    rm = rows < M
    t64 = rows.to(tl.int64)
    m2 = rm[:, None] & cm[None, :]
    a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
    p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
    dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
    rs = tl.load(RSTD + rows, mask=rm, other=0.0)
    for r0 in range(pid * RB, M, STEP):
        x0, x1, x2, x3 = a0, a1, a2, a3
        q0, q1, q2, q3 = p0, p1, p2, p3
        dy = dyb.to(tl.float32)
        rstd = rs
        tc, rmc, m2c = t64, rm, m2
        # prefetch the next row block
        rows = r0 + STEP + tl.arange(0, RB)
        rm = rows < M
        t64 = rows.to(tl.int64)
        m2 = rm[:, None] & cm[None, :]
        a0, a1, a2, a3 = _ld_rows4(X, t64, cols, m2, D)
        p0, p1, p2, p3 = _ld_pre4(PRE, t64, rm)
        dyb = tl.load(DY + t64[:, None] * D + cols[None, :], mask=m2, other=0.0)
        rs = tl.load(RSTD + rows, mask=rm, other=0.0)

        h = _collapse_bf16(x0, x1, x2, x3, q0, q1, q2, q3)
        xh = h * rstd[:, None]
        dxh = dy * g[None, :]
        mm = tl.sum(dxh * xh, axis=1) / D
        dhb = ((dxh - xh * mm[:, None]) * rstd[:, None]).to(DH.dtype.element_ty)
        tl.store(DH + tc[:, None] * D + cols[None, :], dhb, mask=m2c)
        dhf = dhb.to(tl.float32)
        _dpre_store(DPREP, x0, dhf, c_t, tc, rmc, M, 0, RB, NCT, BC)
        _dpre_store(DPREP, x1, dhf, c_t, tc, rmc, M, 1, RB, NCT, BC)
        _dpre_store(DPREP, x2, dhf, c_t, tc, rmc, M, 2, RB, NCT, BC)
        _dpre_store(DPREP, x3, dhf, c_t, tc, rmc, M, 3, RB, NCT, BC)
        dg += dy * xh
    tl.store(DGP + pid * D + cols, tl.sum(dg, axis=0), mask=cm)


# ------------------------------------------------------------------------------------------------
# q-norm + wq_b input dual quant: y = rmsnorm(x) * gamma (bf16), row 1x128 and col 128x1 FP8 of y.
# ------------------------------------------------------------------------------------------------
@triton.jit
def _qnorm_dual_quant_kernel(X, GAMMA, Y, RSTD, QR, SR, QC, SC, M, N, eps,
                             BLOCK: tl.constexpr, FP8_MAX: tl.constexpr, POW2: tl.constexpr, WRITE_Y: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = (pid_m * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    rm = rows < M
    cc = tl.arange(0, BLOCK)
    # rstd of the 128 rows over the full row (other column blocks come from L2)
    ss = tl.zeros((BLOCK,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK):
        cn = n0 + cc
        xs = tl.load(X + rows[:, None] * N + cn[None, :], mask=rm[:, None] & (cn[None, :] < N), other=0.0)
        xs = xs.to(tl.float32)
        ss += tl.sum(xs * xs, axis=1)
    rstd = 1.0 / tl.sqrt(ss / N + eps)
    cols = pid_n * BLOCK + cc
    cmask = cols < N
    m2 = rm[:, None] & cmask[None, :]
    off = rows[:, None] * N + cols[None, :]
    x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
    g = tl.load(GAMMA + cols, mask=cmask, other=0.0).to(tl.float32)
    yb = (x * rstd[:, None] * g[None, :]).to(tl.bfloat16)
    if WRITE_Y:
        tl.store(Y + off, yb, mask=m2)
    if pid_n == 0:
        tl.store(RSTD + rows, rstd, mask=rm)
    y = yb.to(tl.float32)
    ya = tl.abs(y)
    # row (1 x 128) quant
    sr = FP8_MAX / tl.maximum(tl.max(ya, axis=1), 1e-4)
    if POW2:
        sr = _pow2_floor(sr)
    tl.store(QR + off, tl.clamp(y * sr[:, None], -FP8_MAX, FP8_MAX).to(QR.dtype.element_ty), mask=m2)
    tl.store(SR + rows * tl.cdiv(N, BLOCK) + pid_n, 1.0 / sr, mask=rm)
    # col (128 x 1) quant
    sc = FP8_MAX / tl.maximum(tl.max(ya, axis=0), 1e-4)
    if POW2:
        sc = _pow2_floor(sc)
    tl.store(QC + off, tl.clamp(y * sc[None, :], -FP8_MAX, FP8_MAX).to(QC.dtype.element_ty), mask=m2)
    tl.store(SC + pid_m * N + cols, 1.0 / sc, mask=cmask)


# ------------------------------------------------------------------------------------------------
# host wrappers
# ------------------------------------------------------------------------------------------------
FOLD_CFG = dict(fwd_rb=1, fwd_warps=8, bwd_rb=1, bwd_warps=8, bwd_ppc=4, qb=128, qn_warps=16)


def _fold_cfg():
    """FOLD_CFG with the hcfuse override PRIMUS_HCFUSE_FOLD_CFG="bwd_warps=16,bwd_ppc=2,..." (warps / rows / programs
    per CU of the folded-norm kernels; changes reduction trees and the dgamma partial grouping, so off = int)."""
    v = os.environ.get("PRIMUS_HCFUSE_FOLD_CFG", "").strip()
    if not v:
        return FOLD_CFG
    c = dict(FOLD_CFG)
    for kv in v.split(","):
        k, x = kv.split("=")
        c[k.strip()] = int(x)
    return c


def _ue8m0() -> bool:
    return os.environ.get("PRIMUS_TURBO_FP8_BLOCK_UE8M0", "0").strip().lower() in ("1", "true")


def _fp8_dtype():
    return torch.float8_e4m3fn


def _n_cu(dev) -> int:
    try:
        return torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:
        return 64


def collapse_norm_fwd(x3, pre, gamma, eps, fp8=False):
    """x3 [M, 4, D] bf16, pre [M, 4] -> y [M, D], rstd [M] (+ (q, scale_inv) when fp8)."""
    M, K, D = x3.shape
    assert K == 4
    BD = triton.next_power_of_2(D)
    f = _fold_cfg()
    y = torch.empty((M, D), dtype=x3.dtype, device=x3.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x3.device)
    fp8 = bool(fp8) and BD == D and D % f["qb"] == 0
    if fp8:
        q = torch.empty((M, D), dtype=_fp8_dtype(), device=x3.device)
        s = torch.empty((M, D // f["qb"]), dtype=torch.float32, device=x3.device)
    else:
        q = s = rstd
    _hc_collapse_norm_fwd_kernel[(triton.cdiv(M, f["fwd_rb"]),)](
        x3, pre, gamma, y, rstd, q, s, M, D, float(eps),
        BD=BD, RB=f["fwd_rb"], FP8=fp8, FP8_MAX=448.0, POW2=_ue8m0(), QB=f["qb"], num_warps=f["fwd_warps"],
    )
    return y, rstd, ((q, s) if fp8 else None)


def norm_dpre_bwd(x3, pre, gamma, rstd, dy, bc, reduce=True):
    """-> dh [M, D] (bf16, = HC dcoll), dprep [D/bc, M, 4] fp32 partials, dgamma (gamma dtype); with
    ``reduce=False`` the [P, D] fp32 dgamma partials instead of dgamma (fuse-hc reduces them in its tail launch)."""
    from .rmsnorm_g90 import _rms_dg_reduce_kernel

    M, K, D = x3.shape
    BD = triton.next_power_of_2(D)
    assert BD == D and D % bc == 0
    f = _fold_cfg()
    dev = x3.device
    dh = torch.empty((M, D), dtype=x3.dtype, device=dev)
    nc = D // bc
    dprep = torch.empty((nc, M, K), dtype=torch.float32, device=dev)
    rb = f["bwd_rb"]
    nprog = min(triton.cdiv(M, rb), f["bwd_ppc"] * _n_cu(dev))
    dgp = torch.empty((nprog, D), dtype=torch.float32, device=dev)
    _hc_norm_dpre_bwd_kernel[(nprog,)](
        x3, pre, gamma, rstd, dy, dh, dprep, dgp, M, D, BD=BD, RB=rb, BC=bc, NPROG=nprog, num_warps=f["bwd_warps"],
    )
    if not reduce:
        return dh, dprep, dgp
    dg = torch.empty_like(gamma)
    _rms_dg_reduce_kernel[(triton.cdiv(D, 64),)](dgp, dg, nprog, D, BH=64, BP=64, num_warps=4)
    return dh, dprep, dg


def qnorm_dual_quant(x, gamma, eps, write_y=True):
    """x [M, N] bf16 -> y (bf16 or None), rstd, (q_row, s_row [M, N/128]), (q_col, s_col [M/128, N])."""
    M, N = x.shape
    B = 128
    dev = x.device
    y = torch.empty_like(x) if write_y else x
    rstd = torch.empty((M,), dtype=torch.float32, device=dev)
    qr = torch.empty((M, N), dtype=_fp8_dtype(), device=dev)
    qc = torch.empty((M, N), dtype=_fp8_dtype(), device=dev)
    sr = torch.empty((M, triton.cdiv(N, B)), dtype=torch.float32, device=dev)
    sc = torch.empty((triton.cdiv(M, B), N), dtype=torch.float32, device=dev)
    _qnorm_dual_quant_kernel[(triton.cdiv(M, B), triton.cdiv(N, B))](
        x, gamma, y, rstd, qr, sr, qc, sc, M, N, float(eps), BLOCK=B, FP8_MAX=448.0, POW2=_ue8m0(), WRITE_Y=write_y,
        num_warps=FOLD_CFG["qn_warps"],
    )
    return (y if write_y else None), rstd, (qr, sr), (qc, sc)


# ------------------------------------------------------------------------------------------------
# CPU self-test (TRITON_INTERPRET=1) and compile jobs
# ------------------------------------------------------------------------------------------------
def _ref_quant(y, axis, pow2):
    yf = y.float()
    if axis == 1:
        g = yf.view(yf.shape[0], -1, 128)
        amax = g.abs().amax(-1).clamp_min(1e-4)
        s = 448.0 / amax
        if pow2:
            s = torch.exp2(torch.floor(torch.log2(s)))
        q = (g * s[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).view_as(yf)
        return q, 1.0 / s
    g = yf.view(-1, 128, yf.shape[1])
    amax = g.abs().amax(1).clamp_min(1e-4)
    s = 448.0 / amax
    if pow2:
        s = torch.exp2(torch.floor(torch.log2(s)))
    q = (g * s[:, None, :]).clamp(-448, 448).to(torch.float8_e4m3fn).view_as(yf)
    return q, 1.0 / s


def _ulp_ok(a, b, frac=0.0):
    """a, b bf16-valued: |a-b| <= 1 bf16 ulp of |b| everywhere; returns (ok, max_abs, fraction differing)."""
    a, b = a.float(), b.float()
    d = (a - b).abs()
    ulp = torch.where(b.abs() > 0, torch.exp2(torch.floor(torch.log2(b.abs())) - 7), torch.full_like(b, 2.0 ** -133))
    return bool((d <= ulp * 1.0001).all()), d.max().item(), (d > 0).float().mean().item()


def _fp8_ok(q, r):
    """1 fp8 (e4m3) ulp tolerance: the Triton interpreter's fp32->fp8 rounding differs from torch's RNE; on the
    GPU both this kernel and Turbo's quant run the same Triton cast."""
    a, b = q.float(), r.float()
    d = (a - b).abs()
    ulp = torch.exp2(torch.floor(torch.log2(b.abs().clamp_min(2.0 ** -6))) - 3)
    return bool((d <= ulp * 1.0001).all()), (d > 0).float().mean().item()


def _turbo_quant_mod():
    """Turbo's own blockwise quant kernels (opt3-mem Turbo worktree), so the FP8 outputs are compared against the
    exact kernel the blockwise linear would run, in the same (interpreter) arithmetic."""
    import importlib.util

    p = "/dsv41/work/opt3/mem/Primus-Turbo/primus_turbo/triton/quantization/quant_blockwise.py"
    spec = importlib.util.spec_from_file_location("turbo_quant_blockwise_ref", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bf16_like_kernel(t):
    """fp32 -> bf16 the way the executing Triton backend rounds: RNE on the GPU, truncation in the interpreter
    (verified: dbg_round.py, 49% of values differ from torch's RNE in the interpreter)."""
    if os.environ.get("TRITON_INTERPRET") == "1":
        return (t.float().contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)
    return t.to(torch.bfloat16)


def cpu_selftest(report):
    from .rmsnorm_g90 import RMSNormG90Fn

    tq = _turbo_quant_mod()

    torch.manual_seed(3)
    for M, D, bc in [(9, 512, 128), (16, 1024, 256)]:
        x3 = torch.randn(M, 4, D).to(torch.bfloat16)
        pre = torch.rand(M, 4).to(torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(D)).to(torch.bfloat16)
        eps = 1e-6
        os.environ["PRIMUS_TURBO_FP8_BLOCK_UE8M0"] = "1"
        y, rstd, fq = collapse_norm_fwd(x3, pre, gamma, eps, fp8=True)
        # unfused path = HC collapse (bf16 out) + rmsnorm_g90 (same kernels as the e2e path, run by the interpreter)
        h = _bf16_like_kernel((pre.float()[:, :, None] * x3.float()).sum(1)).requires_grad_(True)
        g_ref = gamma.clone().requires_grad_(True)
        y_ref = RMSNormG90Fn.apply(h, g_ref, eps)
        ok, mx, fr = _ulp_ok(y, y_ref)
        report(f"fold fwd y vs collapse+rmsnorm_g90 [M{M} D{D}]", ok, f"<=1 bf16 ulp, max {mx:.1e}, {fr:.2%} differ")
        rq = torch.empty_like(fq[0])
        rs = torch.empty_like(fq[1])
        tq.quant_fp8_blockwise_kernel[(triton.cdiv(M, 128), triton.cdiv(D, 128))](
            y, rq, rs, M, D, BLOCK_SIZE=128, FP8_MAX=448.0, AXIS=1, POW2_SCALE=True)
        okq = torch.equal(fq[0].view(torch.uint8), rq.view(torch.uint8)) and torch.equal(fq[1], rs)
        report(f"fold fwd FP8 row quant (UE8M0) vs Turbo quant_fp8_blockwise_kernel(y) [M{M} D{D}]", okq,
               "bit-exact q and scale_inv" if okq else "MISMATCH")
        dy = torch.randn(M, D).to(torch.bfloat16)
        dh, dprep, dg = norm_dpre_bwd(x3, pre, gamma, rstd, dy, bc)
        y_ref.backward(dy)
        ok1, mx1, fr1 = _ulp_ok(dh, h.grad)
        report(f"fold bwd dh vs rmsnorm_g90 bwd [M{M} D{D}]", ok1, f"max {mx1:.1e}, {fr1:.2%} differ")
        e2 = ((dg.double() - g_ref.grad.double()).norm() / g_ref.grad.double().norm()).item()
        ok2, _, fr2 = _ulp_ok(dg, g_ref.grad)
        report(f"fold bwd dgamma vs rmsnorm_g90 bwd [M{M} D{D}]", ok2, f"relF {e2:.1e}, {fr2:.2%} differ (<=1 ulp)")
        dp_ref = (x3.float() * dh.float()[:, None, :]).view(M, 4, D // bc, bc).sum(-1).permute(2, 0, 1)
        e3 = (dprep - dp_ref).abs().max().item() / dp_ref.abs().max().item()
        report(f"fold bwd dpre partials vs (x . dh) [M{M} D{D}]", e3 < 1e-5, f"rel max {e3:.1e}")
    for M, N in [(256, 1024), (128, 384)]:
        x = torch.randn(M, N).to(torch.bfloat16)
        gamma = (1 + 0.1 * torch.randn(N)).to(torch.bfloat16)
        y, rstd, (qr, sr), (qc, sc) = qnorm_dual_quant(x, gamma, 1e-6)
        y_ref = RMSNormG90Fn.apply(x, gamma, 1e-6)
        ok, mx, fr = _ulp_ok(y, y_ref)
        rq, cq = torch.empty_like(qr), torch.empty_like(qc)
        rs, cs = torch.empty_like(sr), torch.empty_like(sc)
        tq.quant_fp8_blockwise_dual_kernel[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
            y, rq, rs, cq, cs, M, N, BLOCK_SIZE=128, FP8_MAX=448.0, POW2_SCALE=True)
        okr = torch.equal(qr.view(torch.uint8), rq.view(torch.uint8))
        okc = torch.equal(qc.view(torch.uint8), cq.view(torch.uint8))
        oks = torch.equal(sr, rs) and torch.equal(sc, cs)
        report(f"qnorm+dual quant [M{M} N{N}]", ok and okr and okc and oks,
               f"y vs rmsnorm_g90 <=1 ulp ({fr:.2%} differ); vs Turbo dual quant kernel: row/col q bit-exact={okr and okc}, scales exact={oks}")


def compile_jobs():
    P, F, Q = "*bf16", "*fp32", "*fp8e4nv"
    j = []
    for rb, nw in [(1, 8), (2, 8), (1, 4)]:
        j.append((f"fold_fwd_rb{rb}_w{nw}_fp8", _hc_collapse_norm_fwd_kernel,
                  dict(X=P, PRE=P, GAMMA=P, Y=P, RSTD=F, YQ=Q, YS=F, M="i32", D="i32", eps="fp32"),
                  dict(BD=4096, RB=rb, FP8=True, FP8_MAX=448.0, POW2=True, QB=128), nw))
        j.append((f"fold_fwd_rb{rb}_w{nw}_bf16", _hc_collapse_norm_fwd_kernel,
                  dict(X=P, PRE=P, GAMMA=P, Y=P, RSTD=F, YQ=F, YS=F, M="i32", D="i32", eps="fp32"),
                  dict(BD=4096, RB=rb, FP8=False, FP8_MAX=448.0, POW2=True, QB=128), nw))
        j.append((f"fold_bwd_rb{rb}_w{nw}", _hc_norm_dpre_bwd_kernel,
                  dict(X=P, PRE=P, GAMMA=P, RSTD=F, DY=P, DH=P, DPREP=F, DGP=F, M="i32", D="i32"),
                  dict(BD=4096, RB=rb, BC=256, NPROG=1024), nw))
    for nw in (8, 16):
        j.append((f"qnorm_dual_quant_w{nw}", _qnorm_dual_quant_kernel,
                  dict(X=P, GAMMA=P, Y=P, RSTD=F, QR=Q, SR=F, QC=Q, SC=F, M="i32", N="i32", eps="fp32"),
                  dict(BLOCK=128, FP8_MAX=448.0, POW2=True, WRITE_Y=True), nw))
    return j


__all__ = ["hc_fold_norm_enabled", "collapse_norm_fwd", "norm_dpre_bwd", "qnorm_dual_quant", "FOLD_CFG"]
