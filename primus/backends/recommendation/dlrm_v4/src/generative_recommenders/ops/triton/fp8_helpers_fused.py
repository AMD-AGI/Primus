# pyre-unsafe
"""Fused replacements for the tiny FP8 helper kernels (Arbor r14 iter-34, M1c).

DEFAULT OFF. Two knobs, both read at import by the module that calls in here:

  DLRM_HSTU_FP8_SHADOW_FUSED=1       triton_fp8._get_weight_shadow: the forward
                                     E4M3 weight-shadow refresh (abs, amax,
                                     clamp_min, mul, div, transpose copy, div,
                                     clamp, cast; plus the pack copies of a
                                     multi-range layout) becomes TWO kernels:
                                     per-tile |w| maxima, then a kernel whose
                                     every program reduces those maxima, forms
                                     the scale and writes its tile of the
                                     column-major E4M3 shadow (program 0 also
                                     writes the scale).
  DLRM_HSTU_FP8_BWD_HELPERS_FUSED=1  bwd_fp8: the per-site scale bookkeeping
                                     (partial-amax reduce, note_clipping,
                                     note_t_clipping, accumulate, decode-scale
                                     snapshot and its 2^shift product) runs as
                                     one extra program of the weight-amax
                                     kernel, and quantize_weight (abs, amax,
                                     clamp_min, div, reciprocal, mul, cast,
                                     discarded partial amax) becomes that
                                     kernel plus one cast kernel.

Bitwise contract (both knobs): every byte the replaced torch/Triton sequence
produced is reproduced exactly, for all finite inputs:
  * max reductions are order-free (fp32 max is exact), NaN-propagating like
    torch.amax / torch.maximum;
  * a tensor divided by a Python float is `a * fp32(1 / b)` in PyTorch's CUDA
    kernel (div_true_kernel_cuda, CPU-scalar branch); a tensor divided by a
    device tensor is IEEE division (tl.math.div_rn here); `1.0 / t` is
    `t.reciprocal() * 1.0` (IEEE 1/t);
  * clamp/clamp_min keep NaN (torch semantics) where torch did them; the
    backward cast keeps the shipped Triton min/max clamp it replaces;
  * the forward shadow's E4M3 conversion is a bit-exact port of c10's
    fp8e4m3fn_from_fp32_value (round-to-nearest-even incl. subnormals),
    the backward weight cast keeps the shipped Triton `.to(e4m3)`.
The colsum sums (fp32 adds) are NOT touched.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

import triton
import triton.language as tl

_E4M3_MAX: float = 448.0


def _fp32_exact(value: float) -> float:
    """The fp32 value nearest `value`, as an (exactly representable) double."""
    return float(torch.tensor(value, dtype=torch.float32).item())


def _fp32_recip(value: float) -> float:
    """fp32(1) / fp32(value), IEEE: PyTorch's `opmath_t(1.0) / scalar` on host."""
    one = torch.tensor([1.0], dtype=torch.float32)
    den = torch.tensor([value], dtype=torch.float32)
    return float(torch.div(one, den).item())


# Constants of the replaced expressions, as exact fp32 values.
_RECIP_E4M3_MAX: float = _fp32_recip(_E4M3_MAX)  # `amax / 448.0` on device
_CLAMP_FWD: float = _fp32_exact(1.0e-12)  # calibrated_scale clamp_min
_CLAMP_BWD: float = _fp32_exact(1.0e-30)  # quantize_weight clamp_min

_MAX_SEGMENTS: int = 4


@triton.jit
def _nanmax_vec(x):
    """max over a vector with torch.amax NaN semantics (any NaN -> NaN)."""
    m = tl.max(x, axis=0)
    has_nan = tl.max((x != x).to(tl.int32), axis=0)
    return tl.where(has_nan > 0, float("nan"), m)


@triton.jit
def _packed_to_src_col(
    j, c1, c2, c3, d0, d1, d2, d3, NSEG: tl.constexpr
):
    src = j + d0
    if NSEG > 1:
        src = tl.where(j >= c1, j + d1, src)
    if NSEG > 2:
        src = tl.where(j >= c2, j + d2, src)
    if NSEG > 3:
        src = tl.where(j >= c3, j + d3, src)
    return src


@triton.jit
def _fp8h_amax_tiles_kernel(
    W,
    PARTIAL,
    R,
    NP,
    stride_r,
    stride_c,
    c1, c2, c3, d0, d1, d2, d3,
    n_tiles_c,
    n_tiles,
    # optional site-bookkeeping program (pid 0 when HAS_SITE)
    SITE_PARTIAL,
    n_site_partial,
    SCALE,
    CLIPPED,
    T_CLIPPED,
    PENDING,
    SNAP,
    SNAP_T,
    grad_max,
    t_factor,
    t_mul,
    NSEG: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_SITE: tl.constexpr,
    HAS_T: tl.constexpr,
    HAS_SNAP: tl.constexpr,
    HAS_SNAP_T: tl.constexpr,
    SITE_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = pid
    if HAS_SITE:
        # The site program is pid 0 (dispatched first, overlaps the tiles).
        tile = pid - 1
        if pid == 0:
            # ---- _SiteScale bookkeeping of one backward call ------------
            # amax = PARTIAL.amax() (torch.amax: exact max, NaN-propagating)
            offs = tl.arange(0, SITE_BLOCK)
            run_max = tl.full([SITE_BLOCK], 0.0, tl.float32)
            run_nan = tl.zeros([SITE_BLOCK], dtype=tl.int32)
            for start in range(0, n_site_partial, SITE_BLOCK):
                p = tl.load(
                    SITE_PARTIAL + start + offs,
                    mask=(start + offs) < n_site_partial,
                    other=0.0,
                )
                run_nan = run_nan | (p != p).to(tl.int32)
                run_max = tl.maximum(run_max, p)
            amax = tl.max(run_max, axis=0)
            amax = tl.where(tl.max(run_nan, axis=0) > 0, float("nan"), amax)
            s = tl.load(SCALE)
            # note_clipping: clipped_windows += (amax > scale*GRAD_MAX).to(int64)
            clip = (amax > s * grad_max).to(tl.int64)
            tl.store(CLIPPED, tl.load(CLIPPED) + clip)
            if HAS_T:
                # note_t_clipping: amax > scale * (t_limit * 2^t_shift)
                tclip = (amax > s * t_factor).to(tl.int64)
                tl.store(T_CLIPPED, tl.load(T_CLIPPED) + tclip)
            # accumulate: torch.maximum(pending, amax, out=pending)
            pend = tl.load(PENDING)
            newp = tl.where(
                pend != pend,
                pend,
                tl.where(amax != amax, amax, tl.maximum(pend, amax)),
            )
            tl.store(PENDING, newp)
            if HAS_SNAP:
                tl.store(SNAP, s)  # state.scale.clone()
            if HAS_SNAP_T:
                tl.store(SNAP_T, s * t_mul)  # decode_scale * 2^t_shift (exact)
    if tile >= 0:
        # ---- per-tile max |w| of the (packed) [R, NP] weight -------------
        tr = tile // n_tiles_c
        tc = tile % n_tiles_c
        rows = tr * BLOCK_R + tl.arange(0, BLOCK_R)
        cols = tc * BLOCK_C + tl.arange(0, BLOCK_C)
        src_cols = _packed_to_src_col(cols, c1, c2, c3, d0, d1, d2, d3, NSEG)
        mask = (rows[:, None] < R) & (cols[None, :] < NP)
        x = tl.load(
            W
            + rows[:, None].to(tl.int64) * stride_r
            + src_cols[None, :].to(tl.int64) * stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        a = tl.abs(x)
        m = tl.max(tl.max(a, axis=1), axis=0)
        has_nan = tl.max(tl.max((a != a).to(tl.int32), axis=1), axis=0)
        tl.store(PARTIAL + tile, tl.where(has_nan > 0, float("nan"), m))


@triton.jit
def _reduce_partials(PARTIAL, n_partial, BLOCK_P: tl.constexpr):
    offs = tl.arange(0, BLOCK_P)
    p = tl.load(PARTIAL + offs, mask=offs < n_partial, other=0.0)
    return _nanmax_vec(p)


@triton.jit
def _clamp_min_nan(x, lo):
    # torch clamp_min(Scalar): NaN passes through, else max(x, lo)
    return tl.where(x != x, x, tl.maximum(x, lo))


@triton.jit
def _e4m3fn_bits_from_fp32(v):
    """Bit-exact port of c10::detail::fp8e4m3fn_from_fp32_value (RNE)."""
    f_bits = v.to(tl.uint32, bitcast=True)
    sign = f_bits & 0x80000000
    f_bits = f_bits ^ sign
    # >= 464 (incl. inf/NaN) -> NaN 0x7f
    is_nan = f_bits >= (1087 << 20)
    small = f_bits < (121 << 23)
    # subnormal/small path: (f + 2^14) rounds to a multiple of 2^-9 (RNE)
    fs = f_bits.to(tl.float32, bitcast=True) + 16384.0
    small_res = fs.to(tl.uint32, bitcast=True) - (141 << 23)
    # normal path
    mant_odd = (f_bits >> 20) & 1
    nb = f_bits - (120 << 23) + 0x7FFFF + mant_odd
    normal_res = nb >> 20
    res = tl.where(small, small_res, normal_res) & 0xFF
    res = tl.where(is_nan, 0x7F, res)
    res = res | (sign >> 24)
    return res.to(tl.uint8)


@triton.jit
def _fp8h_shadow_quant_t_kernel(
    W,
    OUT_T,
    SCALE_OUT,
    PARTIAL,
    n_partial,
    R,
    NP,
    stride_r,
    stride_c,
    c1, c2, c3, d0, d1, d2, d3,
    n_tiles_c,
    clamp_lo,
    recip_max,
    NSEG: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HW_CVT: tl.constexpr,
):
    # calibrated_scale(packed, margin=1.0):
    #   amax = |w|.amax().float().clamp_min(1e-12); scale = amax * 1.0 / 448.0
    # (`/ 448.0` is `* fp32(1/448)` in PyTorch's CUDA kernel; `* 1.0` exact)
    amax = _reduce_partials(PARTIAL, n_partial, BLOCK_P)
    scale = _clamp_min_nan(amax, clamp_lo) * recip_max
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(SCALE_OUT, scale)
    tr = pid // n_tiles_c
    tc = pid % n_tiles_c
    rows = tr * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tc * BLOCK_C + tl.arange(0, BLOCK_C)
    src_cols = _packed_to_src_col(cols, c1, c2, c3, d0, d1, d2, d3, NSEG)
    mask = (rows[:, None] < R) & (cols[None, :] < NP)
    x = tl.load(
        W
        + rows[:, None].to(tl.int64) * stride_r
        + src_cols[None, :].to(tl.int64) * stride_c,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    # quantize_with_scale(column_major=True): (w^T / scale).clamp(+-448).to(e4m3)
    v = tl.math.div_rn(x, tl.zeros([BLOCK_R, BLOCK_C], dtype=tl.float32) + scale)
    v = tl.where(v != v, v, tl.minimum(tl.maximum(v, -448.0), 448.0))
    if HW_CVT:
        q = v.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        q = _e4m3fn_bits_from_fp32(v)
    # OUT_T is the contiguous [NP, R] buffer whose .t() is the shadow
    tl.store(
        OUT_T + cols[None, :].to(tl.int64) * R + rows[:, None],
        q,
        mask=mask,
    )


@triton.jit
def _fp8h_weight_quant_kernel(
    SRC,
    DST,
    SCALE_OUT,
    PARTIAL,
    n_partial,
    n_elements,
    clamp_lo,
    recip_max,
    BLOCK: tl.constexpr,
    BLOCK_P: tl.constexpr,
    LIMIT: tl.constexpr,
):
    # quantize_weight: amax = |w|.amax().float().clamp_min(1e-30)
    #   scale = amax / 448.0  (= amax * fp32(1/448));  inv = 1.0 / scale
    #   (= scale.reciprocal() * 1.0, IEEE 1/scale)
    amax = _reduce_partials(PARTIAL, n_partial, BLOCK_P)
    scale = _clamp_min_nan(amax, clamp_lo) * recip_max
    inv = tl.math.div_rn(1.0, scale)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(SCALE_OUT, scale)
    # Same arithmetic as bwd_fp8._quantize_amax_kernel (the cast it replaces).
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    x = tl.load(SRC + offs, mask=mask, other=0.0)
    scaled = x.to(tl.float32) * inv
    scaled = tl.minimum(tl.maximum(scaled, -LIMIT), LIMIT)
    tl.store(DST + offs, scaled.to(DST.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------

_AMAX_BLOCK_R: int = 32
_AMAX_BLOCK_C: int = 256
_SHADOW_BLOCK_R: int = 64
_SHADOW_BLOCK_C: int = 64
_WQ_BLOCK: int = 8192
_SITE_BLOCK: int = 16384
_dummy: dict = {}


def _dummy_buf(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    t = _dummy.get(key)
    if t is None:
        t = torch.zeros(1, dtype=dtype, device=device)
        _dummy[key] = t
    return t


def _segments(
    layout: Optional[Sequence[Tuple[int, int]]], ncols: int
) -> Tuple[int, Tuple[int, int, int], Tuple[int, int, int, int]]:
    """(NP, (c1, c2, c3), (d0..d3)): packed column j < c_{k+1} reads source
    column j + d_k (segments in layout order, like _pack_weight_layout)."""
    if layout is None:
        layout = ((0, ncols),)
    assert 1 <= len(layout) <= _MAX_SEGMENTS
    cs = []
    ds = []
    dest = 0
    for start, end in layout:
        cs.append(dest)
        ds.append(start - dest)
        dest += end - start
    big = 1 << 30
    while len(cs) < _MAX_SEGMENTS:
        cs.append(big)
        ds.append(0)
    return dest, (cs[1], cs[2], cs[3]), (ds[0], ds[1], ds[2], ds[3])


def _amax_tiles(
    weight: torch.Tensor,
    layout,
    site: Optional[dict] = None,
) -> torch.Tensor:
    """Launch the per-tile |w| max kernel (optionally + the site program)."""
    r, ncols = weight.shape
    np_, (c1, c2, c3), (d0, d1, d2, d3) = _segments(layout, ncols)
    nseg = 1 if layout is None else len(layout)
    n_tiles_r = triton.cdiv(r, _AMAX_BLOCK_R)
    n_tiles_c = triton.cdiv(np_, _AMAX_BLOCK_C)
    n_tiles = n_tiles_r * n_tiles_c
    partial = torch.empty(n_tiles, dtype=torch.float32, device=weight.device)
    dev = weight.device
    f32 = _dummy_buf(dev, torch.float32)
    i64 = _dummy_buf(dev, torch.int64)
    if site is None:
        grid = (n_tiles,)
        args = dict(
            SITE_PARTIAL=f32, n_site_partial=1, SCALE=f32, CLIPPED=i64,
            T_CLIPPED=i64, PENDING=f32, SNAP=f32, SNAP_T=f32,
            grad_max=1.0, t_factor=1.0, t_mul=1.0,
            HAS_SITE=False, HAS_T=False, HAS_SNAP=False, HAS_SNAP_T=False,
        )
    else:
        grid = (n_tiles + 1,)
        args = dict(
            SITE_PARTIAL=site["partial"],
            n_site_partial=site["partial"].numel(),
            SCALE=site["scale"],
            CLIPPED=site["clipped"],
            T_CLIPPED=site["t_clipped"] if site["t_clipped"] is not None else i64,
            PENDING=site["pending"],
            SNAP=site["snap"] if site["snap"] is not None else f32,
            SNAP_T=site["snap_t"] if site["snap_t"] is not None else f32,
            grad_max=float(site["grad_max"]),
            t_factor=float(site["t_factor"]),
            t_mul=float(site["t_mul"]),
            HAS_SITE=True,
            HAS_T=site["t_clipped"] is not None,
            HAS_SNAP=site["snap"] is not None,
            HAS_SNAP_T=site["snap_t"] is not None,
        )
    _fp8h_amax_tiles_kernel[grid](
        weight,
        partial,
        r,
        np_,
        weight.stride(0),
        weight.stride(1),
        c1, c2, c3, d0, d1, d2, d3,
        n_tiles_c,
        n_tiles,
        NSEG=nseg,
        BLOCK_R=_AMAX_BLOCK_R,
        BLOCK_C=_AMAX_BLOCK_C,
        SITE_BLOCK=_SITE_BLOCK,
        num_warps=8,
        **args,
    )
    return partial


def shadow_refresh(
    weight: torch.Tensor,
    layout: Optional[Sequence[Tuple[int, int]]],
    hw_cvt: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused `_pack_weight_layout` + `calibrated_scale(margin=1.0)` +
    `quantize_with_scale(column_major=True)`: (shadow [K, NP] column-major
    E4M3 view, scale [1, 1] fp32). Two kernels."""
    assert weight.dim() == 2 and weight.dtype in (
        torch.float32, torch.bfloat16, torch.float16
    )
    r, ncols = weight.shape
    np_, (c1, c2, c3), (d0, d1, d2, d3) = _segments(layout, ncols)
    nseg = 1 if layout is None else len(layout)
    partial = _amax_tiles(weight, layout)
    out_t = torch.empty((np_, r), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((1, 1), dtype=torch.float32, device=weight.device)
    n_tiles_c = triton.cdiv(np_, _SHADOW_BLOCK_C)
    grid = (triton.cdiv(r, _SHADOW_BLOCK_R) * n_tiles_c,)
    _fp8h_shadow_quant_t_kernel[grid](
        weight,
        out_t.view(torch.uint8),
        scale,
        partial,
        partial.numel(),
        r,
        np_,
        weight.stride(0),
        weight.stride(1),
        c1, c2, c3, d0, d1, d2, d3,
        n_tiles_c,
        _CLAMP_FWD,
        _RECIP_E4M3_MAX,
        NSEG=nseg,
        BLOCK_R=_SHADOW_BLOCK_R,
        BLOCK_C=_SHADOW_BLOCK_C,
        BLOCK_P=max(16, triton.next_power_of_2(partial.numel())),
        HW_CVT=hw_cvt,
        num_warps=4,
    )
    return out_t.t(), scale


def weight_amax_with_site(
    weight: torch.Tensor, site: Optional[dict]
) -> torch.Tensor:
    """Per-tile |w| maxima of a contiguous 2-D weight; with `site`, the same
    launch also runs the site bookkeeping program. Returns the partials."""
    w2 = weight.detach()
    if w2.dim() != 2:
        w2 = w2.reshape(w2.shape[0], -1)
    return _amax_tiles(w2, None, site)


def weight_quant_from_partials(
    weight: torch.Tensor, partial: torch.Tensor, fp8_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """quantize_weight's (quantized [same shape], scale [1]) from the partials
    of `weight_amax_with_site`. One kernel."""
    source = weight.detach()
    assert source.is_contiguous()
    flat = source.reshape(-1)
    n = flat.numel()
    out = torch.empty(n, dtype=fp8_dtype, device=source.device)
    scale = torch.empty(1, dtype=torch.float32, device=source.device)
    grid = (triton.cdiv(n, _WQ_BLOCK),)
    _fp8h_weight_quant_kernel[grid](
        flat,
        out,
        scale,
        partial,
        partial.numel(),
        n,
        _CLAMP_BWD,
        _RECIP_E4M3_MAX,
        BLOCK=_WQ_BLOCK,
        BLOCK_P=max(16, triton.next_power_of_2(partial.numel())),
        LIMIT=_E4M3_MAX,
        num_warps=8,
    )
    return out.view_as(source), scale
