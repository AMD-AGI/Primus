# pyre-strict
"""E-M3a producers: FP8-emitting preprocessor LayerNorm kernels (DEFAULT OFF).

Only ops/preproc_fp8_l2.py calls these, and only with DLRM_HSTU_FP8_PREPROC_L2=1.
Nothing in the shipped path imports this module's kernels.

P2  swishln256_fwd_emit: the preprocessor SwishLayerNorm(256) forward,
        a1 = h1 * sigmoid(LN(h1) * w + b)
    that writes e4m3 a1 instead of bf16 a1, in two layouts: row-major [M, 256]
    (the L2 forward GEMM's A operand) and token-major [256, Mp] (Mp = M rounded
    up to 128, zero tail; the L2 wgrad's NN B operand). It also writes mean/rstd
    exactly as the shipped kernel does (the SwishLN backward recomputes from h1
    with them) and per-program amax partials of |bf16 a1| for the delayed scale.
    Borrowed from Arbor r16 iteration 29 (em3a/swishln_emit.py, gated bitwise
    there against _weighted_layer_norm_fwd_v3 -- v2's arithmetic with D
    constexpr -- by 1818 checks plus an independent 244-check review). The
    arithmetic up to the bf16 rounding of a1 is the shipped v2 kernel's, op for
    op, using the same helpers and the same EMU mode the shipped launcher would
    pick; the e4m3 bytes are then e4m3(clamp(fp32(bf16 a1) * inv_scale)).
    Row-major only (tm=False, eval): one 16-row tile per program, programs
    grouped per XCD (r16's sched 2).
    With the token-major copy (tm=True; Arbor actckpt iteration 51), a
    program owns 64 consecutive tokens as four row-interleaved 16-token
    sub-tiles (sub-tile i: tokens base + 4j + i), so the e4m3 bytes of tokens
    4j..4j+3 at one column sit in one thread and pack into a uint32 word with
    shifts; the token-major tile is then stored as [256, 16] words (64-byte
    segments) through a dword LDS transpose, with nothing re-quantized in the
    transposed layout, and each sub-tile's bf16 rows are loaded one sub-tile
    ahead of its compute. The per-token arithmetic is the same op for op, and
    every output (a1_rm, a1_tm with its zero tail, mean, rstd and the amax
    partials buffer, same length, same index per 16-token block) is bit-equal
    to the one-tile kernel's (iteration-51 lab: 98 cases x 7 M from 1 to
    2.78M, EMU 0/1/2, saturating scales, NaN/inf rows, sigmoid fallback; and
    the armed E-M3a chain hashes are unchanged). Lab, MI350X, sustained 6 s
    soaks at the 1000 W cap, M = 2.34-2.78M: -0.19..-0.25 ms/call against the
    one-tile kernel with the token-major copy (expandable-segment buffers, as
    in production; -0.16..-0.19 with plain hipMalloc buffers, where the
    one-tile kernel pays less); v3's bf16 kernel stays 0.13..0.17 ms/call
    faster. tm=False keeps the one-tile kernel (its TM branch is unused).

P3  ln512_bwd_dxsum_emit: the LayerNorm(512) dx-sum backward. Its per-element
    source is the body of _weighted_layer_norm_bwd_dx_v2 for non-swish, no
    residual, SUM_DX; the dx epilogue writes e5m2 dx = e5m2(clamp(fp32 dx *
    inv_scale)) instead of bf16 dx, plus amax partials of |fp32 dx|. NOT bitwise
    v2: the grid is sms*8 (v2: sms*4), so the [tile, 512] dw/db/dxsum partials
    split the rows differently and the finalize (the shipped
    _layer_norm_bwd_dwdbdx) sums them in another fp32 order, and the pipelined
    loop's peeled iterations contract a few rows' FMAs differently (ulp-level;
    iter-48 lab: 17 of 1.2e9 bf16-rounded dx differ, <= 4 ulp). Config from the
    Arbor actckpt iteration-48 producer lab (variant rm_b4m8_v8a0: BLOCK_N 4,
    1 warp, grid sms*8, 2 stages; sustained -0.22..-0.46 ms per call vs v2's
    bf16 kernel).

Both casts clamp with NaN propagation (torch.clamp semantics): a finite value
beyond the format's range saturates, a NaN stays NaN in the FP8 operand (the
shipped bwd_fp8 cast maps NaN to -max instead). Their amax partials ignore NaN.

Launch diet (Arbor actckpt iteration 50). Two single-launch helpers replace
chains of tiny torch ops on E-M3a's armed path, value for value:
  site_steady  one delayed-scale site's per-call bookkeeping, reading P2's / P3's
               amax partials directly: the partials' NaN-propagating amax (was
               partials.amax()), the clipped-window count, the window amax
               (nan_to_num + maximum) and the decode-scale snapshot (was 7-8
               launches per site call).
  quantize_w2_dual  W2's e4m3 cast at unit margin (bwd_fp8.quantize_weight:
               abs/amax/float/clamp_min/div/reciprocal + the cast, 9 launches),
               also writing the transposed bytes the dgrad reads (was
               w8.t().contiguous() in the backward) and, from an fp32 master
               weight under autocast, b2's bf16 rounding (torch's .to(bfloat16));
               W2's bf16 rounding is applied in registers before the cast and is
               not stored (together they were 2 more launches).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl
from generative_recommenders.ops.triton import triton_layer_norm as TLN
from generative_recommenders.ops.triton.triton_layer_norm import (
    _layer_norm_bwd_dwdbdx,
    _wln_row_moments,
    _wln_sigmoid_ieee_fast,
    libdevice_rsqrt,
)

E4M3: torch.dtype = torch.float8_e4m3fn
E5M2: torch.dtype = torch.float8_e5m2
E4M3_MAX: float = 448.0
E5M2_MAX: float = 57344.0
D256: int = 256
D512: int = 512
# Token-major operands are padded to a multiple of this many tokens (zero tail).
TM_PAD: int = 128

# P2: (BLOCK_N, num_warps); one tile per program, XCD-grouped (r16 iter-29 default).
_P2_BLOCK_N: int = 16
_P2_NUM_WARPS: int = 4
# P2 with the token-major copy (iteration 51): rows per sub-tile (4 sub-tiles = 64
# tokens per program), num_warps, programs per XCD group.
_P2_TM_S: int = 16
_P2_TM_NUM_WARPS: int = 4
_P2_TM_XCD_GROUP: int = 2
# P3: (BLOCK_N, num_warps, grid multiplier of the CU count, NUM_STAGES).
_P3_CFG: Tuple[int, int, int, int] = (4, 1, 8, 2)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def tm_cols(m: int) -> int:
    """Token-major width for m tokens: m rounded up to TM_PAD."""
    return _cdiv(m, TM_PAD) * TM_PAD


@triton.jit
def _xcd_remap(v, XCD_GROUP: tl.constexpr):
    # Programs k, k+8, ..., k+8*(XCD_GROUP-1) (one XCD, one L2) take XCD_GROUP
    # consecutive row blocks.
    xcd = v % 8
    r = v // 8
    return ((r // XCD_GROUP) * 8 + xcd) * XCD_GROUP + (r % XCD_GROUP)


@triton.jit
def _p2_emit_tile(
    X,
    Y8,
    YT8,
    Mean,
    Rstd,
    w,
    b,
    inv,
    N,
    Mp,
    eps,
    stride_x,
    stride_yt,
    blk,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EMU: tl.constexpr,
    LIMIT: tl.constexpr,
    TM: tl.constexpr,
):
    cols = tl.arange(0, D)
    # ---- _weighted_layer_norm_fwd_v2 (IS_SWISH) loop body up to the bf16 rounding ----
    rows = blk * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    mask_2d = row_mask[:, None]
    r64 = rows.to(tl.int64)
    x_block = tl.load(
        X + r64[:, None] * stride_x + cols[None, :], mask=mask_2d, other=0.0
    ).to(tl.float32)
    mean, x_mean, var = _wln_row_moments(x_block, mask_2d, D, BLOCK_N, D, EMU)
    tl.store(Mean + rows, mean, row_mask)
    rstd = libdevice_rsqrt(var + eps)
    tl.store(Rstd + rows, rstd, row_mask)
    y = x_mean * tl.expand_dims(rstd, 1)
    y = tl.fma(y, w[None, :], b[None, :])
    s, d_sig = _wln_sigmoid_ieee_fast(y)
    if tl.max(tl.max(d_sig.to(tl.uint32, bitcast=True), axis=1), axis=0) > 0x7E800000:
        s = tl.sigmoid(y)
    y = s * x_block
    # ---- emit: bf16 rounding first (the bits the shipped kernel stores), then e4m3 ----
    a1f = y.to(tl.bfloat16).to(tl.float32)
    # max |a1| in fp32 before rounding, rounded to bf16 once per tile: RNE is
    # monotonic and sign-symmetric, so this equals max |bf16 a1| (finite y).
    # Rows >= N load x = 0, so a1 = sigmoid(b) * 0 = +0 there.
    m = tl.max(tl.max(tl.abs(y), axis=1), axis=0)
    amax_bits = m.to(tl.bfloat16).to(tl.float32).to(tl.int32, bitcast=True)
    # NaN-propagating clamp (torch.clamp semantics): a NaN a1 stays NaN in the
    # GEMM operand, so the step's loss goes NaN as on the shipped path; finite
    # values beyond +-LIMIT saturate. The amax above ignores NaN, so a NaN does
    # not poison the delayed scale.
    v = a1f * inv
    v = tl.maximum(v, -LIMIT, propagate_nan=tl.PropagateNan.ALL)
    v = tl.minimum(v, LIMIT, propagate_nan=tl.PropagateNan.ALL)
    q = v.to(tl.float8e4nv)
    tl.store(Y8 + r64[:, None] * D + cols[None, :], q, mask=mask_2d)
    if TM:
        qb = q.to(tl.uint8, bitcast=True)
        # Rows in [N, Mp) are written too: q = 0x00 there (a1 = +0), the zero tail.
        tl.store(
            YT8 + cols[:, None].to(tl.int64) * stride_yt + r64[None, :],
            tl.trans(qb),
            mask=(rows < Mp)[None, :],
        )
    return amax_bits


@triton.jit
def _p2_swishln256_fwd_emit_kernel(
    X,
    Y8,
    YT8,
    W,
    B,
    Mean,
    Rstd,
    AMAXP,
    INV,
    N,
    Mp,
    n_blocks,
    eps,
    stride_x,
    stride_yt,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EMU: tl.constexpr,
    XCD_GROUP: tl.constexpr,
    LIMIT: tl.constexpr,
    TM: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, D)
    w = tl.load(W + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    inv = tl.load(INV)
    acc = pid * 0
    blk = _xcd_remap(pid, XCD_GROUP)
    if blk < n_blocks:
        acc = _p2_emit_tile(
            X, Y8, YT8, Mean, Rstd, w, b, inv, N, Mp, eps, stride_x, stride_yt,
            blk, D, BLOCK_N, EMU, LIMIT, TM,
        )
    tl.store(AMAXP + pid, acc.to(tl.float32, bitcast=True))


@triton.jit
def _p2_inv_remap16(blk16, G: tl.constexpr):
    # The one-tile kernel's program for 16-token block blk16 (inverse of _xcd_remap).
    q = blk16 // G
    r = (q // 8) * G + blk16 % G
    return r * 8 + q % 8


@triton.jit
def _p2_load_rows(X, toks, N, stride_x, D: tl.constexpr):
    cols = tl.arange(0, D)
    return tl.load(
        X + toks.to(tl.int64)[:, None] * stride_x + cols[None, :],
        mask=(toks < N)[:, None],
        other=0.0,
    )


@triton.jit
def _p2_emit_rows(
    xb,
    Y8,
    w,
    b,
    inv,
    toks,
    N,
    eps,
    D: tl.constexpr,
    S: tl.constexpr,
    EMU: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """_p2_emit_tile's arithmetic for the S tokens `toks` (any set of rows), from
    their bf16 rows xb (rows >= N loaded as 0). Stores the e4m3 row-major bytes;
    returns (the bytes as uint32 [S, D], per-token max |y| [S], mean, rstd)."""
    cols = tl.arange(0, D)
    row_mask = toks < N
    mask_2d = row_mask[:, None]
    x_block = xb.to(tl.float32)
    mean, x_mean, var = _wln_row_moments(x_block, mask_2d, D, S, D, EMU)
    rstd = libdevice_rsqrt(var + eps)
    y = x_mean * tl.expand_dims(rstd, 1)
    y = tl.fma(y, w[None, :], b[None, :])
    s, d_sig = _wln_sigmoid_ieee_fast(y)
    # Tile-uniform fallback as in _p2_emit_tile; the fast sigmoid is bit-equal to
    # tl.sigmoid wherever it is valid, so the tile granularity changes no bits.
    if tl.max(tl.max(d_sig.to(tl.uint32, bitcast=True), axis=1), axis=0) > 0x7E800000:
        s = tl.sigmoid(y)
    y = s * x_block
    a1f = y.to(tl.bfloat16).to(tl.float32)
    rmax = tl.max(tl.abs(y), axis=1)
    v = a1f * inv
    v = tl.maximum(v, -LIMIT, propagate_nan=tl.PropagateNan.ALL)
    v = tl.minimum(v, LIMIT, propagate_nan=tl.PropagateNan.ALL)
    q = v.to(tl.float8e4nv)
    tl.store(Y8 + toks.to(tl.int64)[:, None] * D + cols[None, :], q, mask=mask_2d)
    return q.to(tl.uint8, bitcast=True).to(tl.uint32), rmax, mean, rstd


@triton.jit
def _p2_join4(a0, a1, a2, a3, S: tl.constexpr):
    # [S] x 4 -> [S, 4] with [j, i] = a_i.
    return tl.reshape(tl.join(tl.join(a0, a2), tl.join(a1, a3)), (S, 4))


# N and n_valid are not specialized: one compiled variant (per EMU) for every M; the
# one-tile kernel compiles up to four as M % 16 and Mp % 256 vary (no cost: lab, iter 51).
@triton.jit(do_not_specialize=["N", "n_valid"])
def _p2_swishln256_fwd_emit_tm_kernel(
    X,
    Y8,
    YT32,
    W,
    B,
    Mean,
    Rstd,
    AMAXP,
    INV,
    N,
    n_valid,
    n_part,
    eps,
    stride_x,
    stride_yt32,
    D: tl.constexpr,
    S: tl.constexpr,
    EMU: tl.constexpr,
    XCD_GROUP: tl.constexpr,
    G16: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """P2 with the token-major copy: 4*S tokens [base, base + 4S) per program as four
    row-interleaved sub-tiles (sub-tile i: tokens base + 4j + i, j < S). Word j of
    token-major row c packs tokens base + 4j .. base + 4j + 3 (little-endian), so a
    program writes one 4S-byte segment per token-major row. Sub-tile i+1's rows are
    loaded before sub-tile i is computed (the loads sit ahead of the stores in
    program order). Amax partials: one per 16-token block at the one-tile kernel's
    index, rounded to bf16 after the max as there; zero for the padding blocks."""
    pid = tl.program_id(0)
    P: tl.constexpr = 4 * S
    NB16: tl.constexpr = P // 16
    blk = _xcd_remap(pid, XCD_GROUP)
    pidx = _p2_inv_remap16(blk * NB16 + tl.arange(0, NB16), G16)
    if blk < n_valid:
        jj = tl.arange(0, S)
        base = blk * P
        t0 = base + jj * 4 + 0
        t1 = base + jj * 4 + 1
        t2 = base + jj * 4 + 2
        t3 = base + jj * 4 + 3
        cols = tl.arange(0, D)
        w = tl.load(W + cols).to(tl.float32)
        b = tl.load(B + cols).to(tl.float32)
        inv = tl.load(INV)
        x0 = _p2_load_rows(X, t0, N, stride_x, D)
        x1 = _p2_load_rows(X, t1, N, stride_x, D)
        q0, r0, m0, s0 = _p2_emit_rows(x0, Y8, w, b, inv, t0, N, eps, D, S, EMU, LIMIT)
        word = q0
        acc = r0
        x2 = _p2_load_rows(X, t2, N, stride_x, D)
        q1, r1, m1, s1 = _p2_emit_rows(x1, Y8, w, b, inv, t1, N, eps, D, S, EMU, LIMIT)
        word = word | (q1 << 8)
        acc = tl.maximum(acc, r1)
        x3 = _p2_load_rows(X, t3, N, stride_x, D)
        q2, r2, m2, s2 = _p2_emit_rows(x2, Y8, w, b, inv, t2, N, eps, D, S, EMU, LIMIT)
        word = word | (q2 << 16)
        acc = tl.maximum(acc, r2)
        q3, r3, m3, s3 = _p2_emit_rows(x3, Y8, w, b, inv, t3, N, eps, D, S, EMU, LIMIT)
        word = word | (q3 << 24)
        acc = tl.maximum(acc, r3)
        t4 = base + jj[:, None] * 4 + tl.arange(0, 4)[None, :]
        tl.store(Mean + t4, _p2_join4(m0, m1, m2, m3, S), t4 < N)
        tl.store(Rstd + t4, _p2_join4(s0, s1, s2, s3, S), t4 < N)
        # Block k (tokens base + 16k .. + 15) is j in [4k, 4k + 4): max is exact, so this
        # equals the one-tile kernel's tile max; NaN is ignored as there.
        bm = tl.max(tl.reshape(acc, (NB16, 4)), axis=1)
        amax_bits = bm.to(tl.bfloat16).to(tl.float32).to(tl.int32, bitcast=True)
        tl.store(AMAXP + pidx, amax_bits.to(tl.float32, bitcast=True))
        # Tokens in [N, Mp) are written too: x = 0 there, so the bytes are 0x00 (zero tail).
        tl.store(
            YT32 + cols[:, None].to(tl.int64) * stride_yt32 + (base // 4 + jj)[None, :],
            tl.trans(word),
        )
    else:
        tl.store(AMAXP + pidx, tl.zeros([NB16], dtype=tl.float32), mask=pidx < n_part)


def swishln256_emu(h1: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> int:
    """The EMU mode the shipped v2 launcher (maybe_weighted_layer_norm_swish_fwd_v2)
    uses for these inputs, or -1 when the shipped path runs the old kernel (then
    the caller must stay on the shipped path for this call)."""
    key = TLN._wln_fwd_v2_key(h1, w, b)
    return -1 if key is None else TLN._WLN_FWD_V2_MODE.get(key, -1)


def swishln256_fwd_emit(
    h1: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    inv_scale: torch.Tensor,
    emu: int,
    tm: bool = True,
    return_partials: bool = False,
) -> Tuple[
    torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """-> (a1_rm [M, 256] e4m3, a1_tm [256, Mp] e4m3 (None unless tm), mean [M] f32,
    rstd [M] f32, amax [] f32 = max |bf16 a1| (NaN ignored)).  inv_scale: fp32
    device scalar (1 / dequant scale). tm=False skips the token-major copy, which
    only the L2 wgrad reads (eval forwards). return_partials=True returns the
    per-program amax partials (fp32 [grid]) instead of their amax, for
    site_steady (no reduction launch here)."""
    assert h1.dim() == 2 and h1.shape[1] == D256 and h1.stride(1) == 1, h1.shape
    assert h1.dtype == torch.bfloat16
    assert w.is_contiguous() and b.is_contiguous()
    assert w.numel() == D256 and b.numel() == D256
    assert inv_scale.dtype == torch.float32 and inv_scale.numel() == 1
    assert emu >= 0
    M = h1.shape[0]
    dev = h1.device
    Mp = tm_cols(M)
    a1_rm = torch.empty((M, D256), dtype=E4M3, device=dev)
    a1_tm = torch.empty((D256, Mp), dtype=E4M3, device=dev) if tm else None
    mean = torch.empty((M,), dtype=torch.float32, device=dev)
    rstd = torch.empty((M,), dtype=torch.float32, device=dev)
    if M == 0:
        if a1_tm is not None:
            a1_tm.view(torch.uint8).zero_()
        return a1_rm, a1_tm, mean, rstd, torch.zeros(
            (1,) if return_partials else (), dtype=torch.float32, device=dev
        )
    block_n = _P2_BLOCK_N
    n_blocks = _cdiv(Mp if tm else M, block_n)
    group = max(1, D256 // block_n)
    grid = _cdiv(n_blocks, 8 * group) * 8 * group
    partials = torch.empty((grid,), dtype=torch.float32, device=dev)
    if a1_tm is not None:
        # One partial per 16-token block, as below (grid entries, same indices).
        tok = 4 * _P2_TM_S
        xg = _P2_TM_XCD_GROUP
        a1_tm32 = a1_tm.view(torch.int32)
        _p2_swishln256_fwd_emit_tm_kernel[(_cdiv(grid * block_n // tok, 8 * xg) * 8 * xg,)](
            h1,
            a1_rm,
            a1_tm32,
            w,
            b,
            mean,
            rstd,
            partials,
            inv_scale,
            M,
            Mp // tok,
            grid,
            eps,
            h1.stride(0),
            a1_tm32.stride(0),
            D=D256,
            S=_P2_TM_S,
            EMU=emu,
            XCD_GROUP=xg,
            G16=group,
            LIMIT=E4M3_MAX,
            num_warps=_P2_TM_NUM_WARPS,
        )
        return a1_rm, a1_tm, mean, rstd, partials if return_partials else partials.amax()
    _p2_swishln256_fwd_emit_kernel[(grid,)](
        h1,
        a1_rm,
        a1_tm.view(torch.uint8) if a1_tm is not None else a1_rm.view(torch.uint8),
        w,
        b,
        mean,
        rstd,
        partials,
        inv_scale,
        M,
        Mp,
        n_blocks,
        eps,
        h1.stride(0),
        a1_tm.stride(0) if a1_tm is not None else 0,
        D=D256,
        BLOCK_N=block_n,
        EMU=emu,
        XCD_GROUP=group,
        LIMIT=E4M3_MAX,
        TM=bool(tm),
        num_warps=_P2_NUM_WARPS,
    )
    return a1_rm, a1_tm, mean, rstd, partials if return_partials else partials.amax()


@triton.jit
def _p3_ln512_bwd_dxsum_emit_kernel(
    DX8,
    DY,
    X,
    W,
    Mean,
    Rstd,
    DW,
    DB,
    DXSUM,
    PARTIAL,
    INV_SCALE,
    stride_dy,
    stride_x,
    D,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    LIMIT: tl.constexpr,
):
    """_weighted_layer_norm_bwd_dx_v2's per-element source (non-swish, no
    residual, SUM_DX, EVEN_D, strided walk) with an e5m2 dx epilogue; its own
    grid, so not v2's bits (see the module docstring)."""
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    extra = num_blocks % tile_num
    if pid < extra:
        blocks_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    inv = tl.load(INV_SCALE)

    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_dxsum = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_amax = tl.zeros([BLOCK_D], dtype=tl.float32)
    cst = tl.max_contiguous(tl.arange(0, BLOCK_D), 8)

    for idx in tl.range(0, blocks_per_tile, num_stages=NUM_STAGES):
        current_block = pid + idx * tile_num
        rows = current_block * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        rows64 = rows.to(tl.int64)
        m2 = row_mask[:, None] & (cols < BLOCK_D)[None, :]
        x_block = tl.load(
            X + rows64[:, None] * stride_x + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        dy_block = tl.load(
            DY + rows64[:, None] * stride_dy + cols[None, :], mask=m2, other=0.0
        ).to(tl.float32)
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)
        rstd = tl.expand_dims(rstd, 1)
        xhat = (x_block - mean) * rstd
        wdy = w[None, :] * dy_block
        c1 = tl.sum(xhat * wdy, axis=1) / D
        c2 = tl.sum(wdy, axis=1) / D
        c1 = tl.expand_dims(c1, 1)
        c2 = tl.expand_dims(c2, 1)
        dx = (wdy - (xhat * c1 + c2)) * rstd
        partial_dw = tl.sum(dy_block * xhat, axis=0)
        partial_db = tl.sum(dy_block, axis=0)
        acc_amax = tl.maximum(acc_amax, tl.max(tl.abs(dx), axis=0))
        s = dx * inv
        s = tl.maximum(s, -LIMIT, propagate_nan=tl.PropagateNan.ALL)
        s = tl.minimum(s, LIMIT, propagate_nan=tl.PropagateNan.ALL)
        tl.store(
            DX8 + rows64[:, None] * BLOCK_D + cst[None, :],
            s.to(DX8.dtype.element_ty),
            mask=m2,
        )
        acc_dw += partial_dw
        acc_db += partial_db
        acc_dxsum += tl.sum(dx, axis=0)

    tl.store(DW + pid.to(tl.int64) * D + cols, acc_dw, mask=col_mask)
    tl.store(DB + pid.to(tl.int64) * D + cols, acc_db, mask=col_mask)
    tl.store(DXSUM + pid.to(tl.int64) * D + cols, acc_dxsum, mask=col_mask)
    tl.store(PARTIAL + pid, tl.max(acc_amax, axis=0))


def ln512_bwd_dxsum_emit(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    inv_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    return_partials: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (dx e5m2 [N, 512], dweight [512] (weight dtype), dbias [512] (weight
    dtype), dxsum [512] f32, amax [] f32 = max |fp32 dx|). dy/x bf16 [N, 512]
    row-major; inv_scale: fp32 device scalar. return_partials=True returns the
    per-program amax partials (fp32 [tile_num]) instead of their amax, for
    site_steady (no reduction launch here)."""
    N, D = x.shape
    assert D == D512 and dy.shape == x.shape
    assert x.stride(1) == 1 and dy.stride(1) == 1
    assert inv_scale.dtype == torch.float32 and inv_scale.numel() == 1
    dev = x.device
    dx8 = (
        torch.empty((N, D), dtype=E5M2, device=dev)
        if out is None
        else out
    )
    assert dx8.shape == (N, D) and dx8.is_contiguous() and dx8.dtype == E5M2
    dweight = torch.empty((D,), dtype=weight.dtype, device=dev)
    dbias = torch.empty((D,), dtype=weight.dtype, device=dev)
    dxsum = torch.empty((D,), dtype=torch.float32, device=dev)
    if N == 0:
        dweight.zero_()
        dbias.zero_()
        dxsum.zero_()
        return dx8, dweight, dbias, dxsum, torch.zeros(
            (1,) if return_partials else (), dtype=torch.float32, device=dev
        )
    block_n, num_warps, mult, num_stages = _P3_CFG
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    tile_num = max(1, min(sms * mult, triton.cdiv(N, 4 * block_n)))
    pdw = torch.empty((tile_num, D), dtype=torch.float32, device=dev)
    pdb = torch.empty((tile_num, D), dtype=torch.float32, device=dev)
    pdxs = torch.empty((tile_num, D), dtype=torch.float32, device=dev)
    part = torch.empty((tile_num,), dtype=torch.float32, device=dev)
    _p3_ln512_bwd_dxsum_emit_kernel[(tile_num,)](
        dx8,
        dy,
        x,
        weight,
        mean,
        rstd,
        pdw,
        pdb,
        pdxs,
        part,
        inv_scale,
        dy.stride(0),
        x.stride(0),
        D,
        N,
        BLOCK_D=D512,
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        LIMIT=E5M2_MAX,
        num_warps=num_warps,
    )

    # The shipped finalize, launched exactly as triton_weighted_layer_norm_bwd_dxsum does.
    def grid(META):  # pyre-ignore[2,3]
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    final_block_d = triton.next_power_of_2(triton.cdiv(D, blocks))
    final_block_d = min(max(final_block_d, 4), 128)
    _layer_norm_bwd_dwdbdx[grid](
        pdw,
        pdb,
        pdxs,
        dweight,
        dbias,
        dxsum,
        tile_num,
        D,
        BLOCK_D=final_block_d,
    )
    return dx8, dweight, dbias, dxsum, part if return_partials else part.amax()


# --------------------------------------------------------------------------- launch diet (iter 50)
# One site call of E-M3a's delayed scale (preproc_fp8_l2._Site.steady), in one
# launch. The torch sequence it replaces, value for value:
#     amax = partials.amax()                       # NaN-propagating
#     clipped += (amax > scale * LIMIT).to(int64)  # a NaN amax compares False
#     pending = maximum(pending, nan_to_num(amax, nan=0, posinf=0, neginf=0))
#     decode = scale.clone()                       # when the caller returns a copy
# max is exact and order-free, so reading the partials directly gives the same
# amax; NaN is tracked apart because tl.maximum ignores it while torch's amax
# propagates it (and a NaN amax only ever reaches the compare and nan_to_num).
_SITE_BLOCKS: Tuple[Tuple[int, int, int], ...] = (
    # (max partials, BLOCK, num_warps): P3 has <= sms*8 partials, P2 one per 16-row tile.
    # Single program: ~2 us at P3's 2048 partials; at P2's ~156k partials ~12 us standalone
    # (best of the iter-50 tune: 11.9-22.6 us over BLOCK 2k-16k, 4-16 warps, 1-4 loop
    # stages; this config is near the best) and ~18-19 us per call in a 2.5M-row profile.
    (4096, 4096, 4),
    (1 << 62, 16384, 16),
)


@triton.jit
def _site_steady_kernel(
    PART,
    n_part,
    SCALE,
    PENDING,
    CLIPPED,
    DECODE,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mx = tl.full([BLOCK], float("-inf"), tl.float32)
    nan = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(0, n_part, BLOCK):
        idx = start + offs
        p = tl.load(PART + idx, mask=idx < n_part, other=float("-inf"))
        nan = tl.maximum(nan, (p != p).to(tl.int32))
        mx = tl.maximum(mx, p)
    m = tl.max(mx, axis=0)
    finite_amax = tl.max(nan, axis=0) == 0
    scale = tl.load(SCALE)
    over = (m > scale * LIMIT) & finite_amax
    tl.store(CLIPPED, tl.load(CLIPPED) + over.to(tl.int64))
    keep = finite_amax & (m != float("inf")) & (m != float("-inf"))
    fin = tl.where(keep, m, 0.0)
    pending = tl.load(PENDING)
    # torch.maximum: a NaN operand wins (pending never holds one; mirrored anyway).
    tl.store(PENDING, tl.where(pending != pending, pending, tl.maximum(pending, fin)))
    # Always written (4 bytes): a snap flag would be a constexpr variant first
    # compiled at the first publishing call, ~200 steps into training.
    tl.store(DECODE, scale)


def site_steady(
    partials: torch.Tensor,
    scale: torch.Tensor,
    pending: torch.Tensor,
    clipped: torch.Tensor,
    limit: float,
    snap: bool,
) -> Optional[torch.Tensor]:
    """One launch: book this call's amax partials into (clipped, pending) and,
    if snap, return a fresh copy of `scale` taken before anything publishes
    (== scale.clone()). partials: fp32 [n] (n >= 1); scale fp32 [1]; pending
    fp32 []; clipped int64 []."""
    assert partials.dtype == torch.float32 and partials.dim() == 1 and partials.is_contiguous()
    assert scale.dtype == torch.float32 and scale.numel() == 1
    assert pending.dtype == torch.float32 and pending.numel() == 1
    assert clipped.dtype == torch.int64 and clipped.numel() == 1
    n = partials.numel()
    assert n >= 1
    decode = torch.empty_like(scale)
    for cap, block, warps in _SITE_BLOCKS:
        if n <= cap:
            break
    _site_steady_kernel[(1,)](
        partials,
        n,
        scale,
        pending,
        clipped,
        decode,
        LIMIT=float(limit),
        BLOCK=block,
        num_warps=warps,
    )
    return decode if snap else None


# W2's e4m3 cast (bwd_fp8.quantize_weight) in one launch. The torch chain:
#     amax  = w.abs().amax().float().clamp_min(1e-30)
#     scale = (amax / 448.0).reshape(1)   # CPU-scalar divisor: amax * fl32(1/448)
#     inv   = 1.0 / scale                 # reciprocal(): correctly rounded fp32
#     w8    = clamp(fp32(w) * inv, -448, 448) -> e4m3   (_quantize_amax_kernel)
# Measured on this torch (2.12 ROCm, gfx950; iter-50 probe_sem; re-run unit50.py
# after a torch upgrade): x / 448.0 equals
# x * fl32(1/448) bit for bit and differs from the correctly rounded quotient in
# half the cases; 1.0 / s is the correctly rounded reciprocal, reproduced here
# as fp64 1/s rounded once to fp32 (innocuous double rounding: 53 >= 2*24+2).
# NaN: bf16 abs() clears the sign bit only and amax() returns the NaN quieted,
# so a NaN weight gives scale = fp32((bits & 0x7FFF) | 0x40) * fl32(1/448) and
# every byte -448 (NaN inv; maxnum clamps). With several distinct NaN payloads
# torch's pick depends on its reduction order; here the largest payload wins.
# Every program computes the whole-tensor amax (identical), then casts its
# rows; the transposed bytes are what w8.t().contiguous() would hold.
_W_INV_MAX = tl.constexpr(float(np.float32(1.0) / np.float32(E4M3_MAX)))
_W_CLAMP_LO = tl.constexpr(float(np.float32(1.0e-30)))
# ~21 us per call. The iter-50 tune tried 16/32/64-row tiles, 4-16 warps and 1-4 loop
# stages: 19.3-68.9 us, with this config near the best (the per-program whole-tensor
# amax pass is latency-bound).
_W2_BLOCK_R: int = 32
_W2_AMAX_R: int = 64
_W2_NUM_WARPS: int = 8


@triton.jit
def _bf16_bits_of_f32(x):
    """torch's fp32 -> bf16 (ROCm 7.2 build) as 16 bits in a uint32: round to
    nearest even; NaN keeps its sign and top payload bits, quieted."""
    u = x.to(tl.uint32, bitcast=True)
    rne = (u + (((u >> 16) & 1) + 0x7FFF)) >> 16
    qnan = (u >> 16) | 0x40
    return tl.where(x != x, qnan, rne)


@triton.jit
def _w2_load(W, rows, cols, C: tl.constexpr, CAST_BF16: tl.constexpr):
    """-> (fp32 value of the bf16 weight element, its 16 bf16 bits as int32)."""
    off = rows[:, None].to(tl.int64) * C + cols[None, :]
    if CAST_BF16:
        b = _bf16_bits_of_f32(tl.load(W + off))
        bits = b.to(tl.int32)
        return (b << 16).to(tl.float32, bitcast=True), bits
    else:
        x = tl.load(W + off)
        bits = x.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
        return x.to(tl.float32), bits


@triton.jit
def _w2_scale(m, nan_bits, has_nan):
    """(scale, inv) from the NaN-free amax m (or the quieted NaN bits)."""
    a = tl.where(has_nan, (nan_bits << 16).to(tl.float32, bitcast=True), m)
    a = tl.where(a != a, a, tl.maximum(a, _W_CLAMP_LO))
    scale = a * _W_INV_MAX
    inv = (1.0 / scale.to(tl.float64)).to(tl.float32)
    return scale, inv


@triton.jit
def _w2_quant_dual_kernel(
    W,
    W8,
    W8T,
    SCALE,
    B,
    B_OUT,
    n_b,
    R: tl.constexpr,
    C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    AMAX_R: tl.constexpr,
    LIMIT: tl.constexpr,
    CAST_BF16: tl.constexpr,
    CAST_B: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, C)
    mx = tl.full([AMAX_R, C], float("-inf"), tl.float32)
    nb = tl.zeros([AMAX_R, C], dtype=tl.int32)
    for r0 in range(0, R, AMAX_R):
        x, bits = _w2_load(W, r0 + tl.arange(0, AMAX_R), cols, C, CAST_BF16)
        nb = tl.maximum(nb, tl.where(x != x, (bits & 0x7FFF) | 0x40, 0))
        mx = tl.maximum(mx, tl.abs(x))
    m = tl.max(tl.max(mx, axis=1), axis=0)
    nan_bits = tl.max(tl.max(nb, axis=1), axis=0)
    scale, inv = _w2_scale(m, nan_bits, nan_bits != 0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    x, _ = _w2_load(W, rows, cols, C, CAST_BF16)
    # == _quantize_amax_kernel's cast
    scaled = x * inv
    scaled = tl.minimum(tl.maximum(scaled, -LIMIT), LIMIT)
    q = scaled.to(W8.dtype.element_ty)
    r64 = rows.to(tl.int64)
    tl.store(W8 + r64[:, None] * C + cols[None, :], q)
    qb = q.to(tl.uint8, bitcast=True)
    tl.store(W8T + cols[:, None].to(tl.int64) * R + r64[None, :], tl.trans(qb))
    if pid == 0:
        tl.store(SCALE, scale)
        if CAST_B:
            bo = tl.arange(0, BLOCK_B)
            bm = bo < n_b
            bb = _bf16_bits_of_f32(tl.load(B + bo, mask=bm, other=0.0))
            tl.store(B_OUT + bo, bb.to(tl.int16), mask=bm)


def quantize_w2_dual(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cast_bf16: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """-> (w8 [R, C] e4m3, w8_t [C, R] e4m3 = w8.t().contiguous(), scale [1] f32,
    bias_bf16 or None), bitwise == bwd_fp8.quantize_weight(w) (+ the transpose)
    with w = weight (bf16), or w = weight.to(bf16) when cast_bf16 (weight fp32;
    then bias, fp32, is also returned as bias.to(bf16))."""
    assert weight.dim() == 2 and weight.is_contiguous()
    R, C = weight.shape
    assert R % _W2_BLOCK_R == 0 and R % _W2_AMAX_R == 0 and C in (256, 512), weight.shape
    if cast_bf16:
        assert weight.dtype == torch.float32
    else:
        assert weight.dtype == torch.bfloat16 and bias is None
    dev = weight.device
    w8 = torch.empty((R, C), dtype=E4M3, device=dev)
    w8_t = torch.empty((C, R), dtype=E4M3, device=dev)
    scale = torch.empty((1,), dtype=torch.float32, device=dev)
    b_out = None
    if bias is not None:
        assert bias.dtype == torch.float32 and bias.dim() == 1 and bias.is_contiguous()
        b_out = torch.empty(bias.shape, dtype=torch.bfloat16, device=dev)
    n_b = 0 if bias is None else bias.numel()
    _w2_quant_dual_kernel[(R // _W2_BLOCK_R,)](
        weight,
        w8,
        w8_t.view(torch.uint8),
        scale,
        bias if bias is not None else scale,
        b_out.view(torch.int16) if b_out is not None else scale,
        n_b,
        R=R,
        C=C,
        BLOCK_R=_W2_BLOCK_R,
        AMAX_R=_W2_AMAX_R,
        LIMIT=E4M3_MAX,
        CAST_BF16=bool(cast_bf16),
        CAST_B=bias is not None,
        BLOCK_B=max(16, triton.next_power_of_2(max(n_b, 1))),
        num_warps=_W2_NUM_WARPS,
    )
    return w8, w8_t, scale, b_out
