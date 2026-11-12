# pyre-unsafe
"""Fused FP8 weight-shadow refresh (DLRM_HSTU_FP8_WSHADOW_FUSED, default off; gin turns it on).

What it replaces. ``triton_fp8._get_weight_shadow`` re-quantizes every FP8
GEMM weight after each optimizer step (the cache is keyed on the weight's
``_version``) with an aten chain of about ten launches per (weight, layout):

    packed = _pack_weight_layout(weight, layout)          # view, or empty + copy_ per range
    amax   = packed.abs().amax().float().clamp_min(1e-12) # abs, memset + reduce, clamp_min
    scale  = (amax * 1.0 / 448.0).reshape(1, 1)           # mul, div by a CPU scalar
    q      = (packed.t().contiguous() / scale)            # transposing copy_, div
             .clamp(-448, 448).to(float8_e4m3fn).t()      # clamp, cast

Seven (weight, layout) shadows per step in the 3-layer model: 72 launches,
0.534 ms per rank-step on the main stream (ship10 1n census, call sites under
``## hstu_fp8_weight_shadow_refresh ##``).

What it does instead. Two launches per refresh batch, any number of shadows:

* ``_wshadow_amax_kernel``: one program per [BLOCK_K, BLOCK_N] tile of every
  shadow's packed region, read in place from the weight (no packing copy),
  stores the tile's max |w| as int32 bits.
* ``_wshadow_quant_kernel``: same tiling. Every program re-reduces its
  shadow's tile partials (<= a few hundred int32), derives the scale, writes
  it once per shadow, and stores its tile transposed as E4M3 bits.

Bitwise contract (fp32 or bf16 weights), each step mirroring the CUDA kernel
torch runs today (sources: torch 7661cd9 BinaryDivTrueKernel.cu,
TensorCompare.cu, Copy.cu, headeronly/util/Float8_e4m3fn.h):

* amax: max is exact and order-free; |w| for IEEE non-negative floats orders
  like its int32 bits, and NaN bits sort above +inf, so an int max of
  ``bits & 0x7fffffff`` is the float amax (NaN if any NaN).
* ``clamp_min(1e-12)``: ``isnan(v) ? v : max(v, (float)1e-12)``.
* ``* 1.0``: exact. ``/ 448.0`` with a CPU scalar is NOT a division in aten:
  ``div_true_kernel_cuda`` multiplies by ``(float)(1.0 / 448.0)`` (double
  reciprocal rounded to float). Replicated: ``(amax * margin) * inv_448``.
* ``t / scale`` with a device [1, 1] scale is the IEEE fp32 division
  (DivFunctor) -> ``tl.math.div_rn``.
* ``clamp(-448, 448)``: NaN passes through, else ``min(max(v, lo), hi)``.
* ``.to(float8_e4m3fn)``: on ROCm ``ConvertToFloat8E4M3fnOp`` is the software
  ``fp8e4m3fn_from_fp32_value`` (round-to-nearest-even, 0x7f for |v| >= 480
  or NaN, sign or-ed in). Replicated bit for bit in ``_e4m3fn_bits``; the
  hardware cvt is not used, so the result does not depend on how Triton
  lowers fp32 -> fp8.

NaN or Inf weights still give NaN shadows in the same positions; NaN payload
and the sign of an fp8 NaN are not promised (the fp32 NaN that reaches the
cast may carry a different payload). Every finite weight, including +-0 and
fp32 subnormals, is covered by the contract and by validate_wshadow.py.

Layout of the result: identical to the aten chain -- per shadow a [K, N] view
with strides (1, K) of an [N, K] row-major E4M3 block, and a contiguous [1, 1]
fp32 scale. The blocks and scales of one batch live in two fresh buffers
(allocations only, no launches) at 4 KiB / 512 B aligned offsets, so pointer
alignment seen by hipBLASLt is at least what the caching allocator gave the
separate tensors before.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

import triton
import triton.language as tl

FP8_MAX: float = 448.0
# Exactly what aten computes for `x / 448.0` on a CUDA tensor and a CPU scalar:
# static_cast<float>(double(1.0) / double(448.0)). The float32 value is
# re-expanded to a Python float so the launcher's double -> float conversion
# is exact.
INV_FP8_MAX_F32: float = float(np.float32(1.0 / 448.0))
# Scalar(1e-12).to<float>() in launch_clamp_scalar.
AMAX_FLOOR_F32: float = float(np.float32(1.0e-12))

MAX_SOURCES: int = 8  # distinct weight tensors per launch (pointer args S0..S7)
_META_W: int = 12
# META columns (int64)
_M_K, _M_NS, _M_STRIDE, _M_TILE0, _M_TILES_N, _M_GTILE0, _M_GTILES = 0, 1, 2, 3, 4, 5, 6
_M_OUT_OFF, _M_DST_COL0, _M_GROUP, _M_SRC, _M_SRC_OFF = 7, 8, 9, 10, 11

_OUT_ALIGN: int = 4096  # bytes, per shadow block
# tl.math.div_rn (precise, round-to-nearest) is the only lowering launched: it
# is the one validate_wshadow.py proved bitwise on gfx950 / Triton 3.6.0
# (precise_div_in_use=true). If this Triton cannot lower it, the launch raises
# and triton_fp8 disables the fused path for the process (aten chain); the
# PRECISE_DIV=False branch of the kernel is kept only for experiments.
_PRECISE_DIV: bool = True
_SCALE_STRIDE: int = 128  # floats (512 B) between per-shadow scales


@triton.jit
def _e4m3fn_bits(f):
    """c10::detail::fp8e4m3fn_from_fp32_value on fp32 lanes -> int32 in [0, 255].

    All intermediate values are non-negative and < 2^31, so int32 arithmetic
    reproduces the uint32 code exactly.
    """
    bits = f.to(tl.int32, bitcast=True)
    neg = bits < 0
    fb = bits & 0x7FFFFFFF
    # |f| < 2^-6: add the magic 2^14 (bits 141 << 23) in fp32 (RNE onto the
    # e4m3 subnormal grid), then subtract its bits.
    den = (fb.to(tl.float32, bitcast=True) + 16384.0).to(tl.int32, bitcast=True) - (
        141 << 23
    )
    # normal range: exponent rebias + round-to-nearest-even on bit 20.
    odd = (fb >> 20) & 1
    nrm = (fb - (120 << 23) + 0x7FFFF + odd) >> 20
    r = tl.where(fb >= (1087 << 20), 0x7F, tl.where(fb < (121 << 23), den, nrm))
    r = r & 0xFF
    return tl.where(neg, r | 0x80, r)


@triton.jit
def _load_src_tile(
    S0, S1, S2, S3, S4, S5, S6, S7, src, ptr_off, mask
):
    # Scalar branch per source pointer: no pointer select (the ROCm pointer
    # canonicalization rejects selects between pointer tensors).
    if src == 0:
        w = tl.load(S0 + ptr_off, mask=mask, other=0.0)
    elif src == 1:
        w = tl.load(S1 + ptr_off, mask=mask, other=0.0)
    elif src == 2:
        w = tl.load(S2 + ptr_off, mask=mask, other=0.0)
    elif src == 3:
        w = tl.load(S3 + ptr_off, mask=mask, other=0.0)
    elif src == 4:
        w = tl.load(S4 + ptr_off, mask=mask, other=0.0)
    elif src == 5:
        w = tl.load(S5 + ptr_off, mask=mask, other=0.0)
    elif src == 6:
        w = tl.load(S6 + ptr_off, mask=mask, other=0.0)
    else:
        w = tl.load(S7 + ptr_off, mask=mask, other=0.0)
    return w.to(tl.float32)


@triton.jit
def _tile_coords(META, TILE_SEG, pid, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    seg = tl.load(TILE_SEG + pid).to(tl.int64)
    m = META + seg * 12
    K = tl.load(m + 0)
    NS = tl.load(m + 1)
    stride = tl.load(m + 2)
    tile0 = tl.load(m + 3)
    tiles_n = tl.load(m + 4)
    src = tl.load(m + 10)
    src_off = tl.load(m + 11)
    local = pid - tile0
    tk = local // tiles_n
    tn = local - tk * tiles_n
    offs_k = tk * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tn * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_k[:, None] < K) & (offs_n[None, :] < NS)
    ptr_off = src_off + offs_k[:, None].to(tl.int64) * stride + offs_n[None, :]
    return m, K, NS, src, offs_k, offs_n, mask, ptr_off


@triton.jit
def _wshadow_amax_kernel(
    S0, S1, S2, S3, S4, S5, S6, S7,
    META,
    TILE_SEG,
    PARTIAL,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    m, K, NS, src, offs_k, offs_n, mask, ptr_off = _tile_coords(
        META, TILE_SEG, pid, BLOCK_K, BLOCK_N
    )
    x = _load_src_tile(S0, S1, S2, S3, S4, S5, S6, S7, src, ptr_off, mask)
    abits = x.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    tl.store(PARTIAL + pid, tl.max(tl.max(abits, axis=1), axis=0))


@triton.jit
def _wshadow_quant_kernel(
    S0, S1, S2, S3, S4, S5, S6, S7,
    META,
    TILE_SEG,
    PARTIAL,
    OUT,
    SCALES,
    margin,
    inv_fp8_max,
    amax_floor,
    fp8_max,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    PRECISE_DIV: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    m, K, NS, src, offs_k, offs_n, mask, ptr_off = _tile_coords(
        META, TILE_SEG, pid, BLOCK_K, BLOCK_N
    )
    g_tile0 = tl.load(m + 5)
    g_tiles = tl.load(m + 6)
    out_off = tl.load(m + 7)
    dst_col0 = tl.load(m + 8)
    gid = tl.load(m + 9)
    # Shadow amax from the tile partials (exact: max of int32 bits).
    offs_p = tl.arange(0, BLOCK_P)
    pbits = tl.load(PARTIAL + g_tile0 + offs_p, mask=offs_p < g_tiles, other=0)
    amax = tl.max(pbits, axis=0).to(tl.float32, bitcast=True)
    # .clamp_min(1e-12): NaN passes through.
    amax = tl.where(amax != amax, amax, tl.maximum(amax, amax_floor))
    # (amax * margin) / 448.0 == (amax * margin) * (float)(1.0 / 448.0) in aten.
    scale = (amax * margin) * inv_fp8_max
    if pid == g_tile0:
        tl.store(SCALES + gid * SCALE_STRIDE, scale)
    x = _load_src_tile(S0, S1, S2, S3, S4, S5, S6, S7, src, ptr_off, mask)
    if PRECISE_DIV:
        q = tl.math.div_rn(x, scale)
    else:  # plain fdiv; also IEEE on gfx950 unless fast-math flags are set
        q = x / scale
    q = tl.where(q != q, q, tl.minimum(tl.maximum(q, -fp8_max), fp8_max))
    b = _e4m3fn_bits(q).to(tl.uint8)
    # Element (k, n) of the [K, N] shadow lives at n * K + k of its [N, K]
    # row-major block: store the tile transposed so k is the contiguous dim.
    n_glob = dst_col0 + offs_n
    optr = OUT + out_off + n_glob[:, None].to(tl.int64) * K + offs_k[None, :]
    mask_t = (offs_n[:, None] < NS) & (offs_k[None, :] < K)
    tl.store(optr, tl.trans(b), mask=mask_t)


# --------------------------------------------------------------------------
# Host side
# --------------------------------------------------------------------------

_Layout = Tuple[Tuple[int, int], ...]


class _Plan:
    """Static per-batch geometry: device tables + host offsets. No data_ptrs."""

    def __init__(
        self,
        shapes: Sequence[Tuple[int, int]],
        strides: Sequence[int],
        layouts: Sequence[_Layout],
        src_index: Sequence[int],
        device: torch.device,
        block_k: int,
        block_n: int,
    ) -> None:
        meta: List[List[int]] = []
        tile_seg: List[int] = []
        self.group_out_off: List[int] = []
        self.group_kn: List[Tuple[int, int]] = []
        out_off = 0
        tile = 0
        max_gtiles = 1
        for g, ((k, _cols), stride, layout, src) in enumerate(
            zip(shapes, strides, layouts, src_index)
        ):
            n_total = sum(end - start for start, end in layout)
            g_tile0 = tile
            g_segments = []
            dst = 0
            for start, end in layout:
                ns = end - start
                tiles_k = (k + block_k - 1) // block_k
                tiles_n = (ns + block_n - 1) // block_n
                g_segments.append([k, ns, stride, tile, tiles_n, 0, 0, out_off, dst, g, src, start])
                tile_seg.extend([len(meta) + len(g_segments) - 1] * (tiles_k * tiles_n))
                tile += tiles_k * tiles_n
                dst += ns
            g_tiles = tile - g_tile0
            max_gtiles = max(max_gtiles, g_tiles)
            for row in g_segments:
                row[_M_GTILE0] = g_tile0
                row[_M_GTILES] = g_tiles
            meta.extend(g_segments)
            self.group_out_off.append(out_off)
            self.group_kn.append((k, n_total))
            out_off += ((n_total * k + _OUT_ALIGN - 1) // _OUT_ALIGN) * _OUT_ALIGN
        self.total_tiles = tile
        self.out_bytes = max(out_off, _OUT_ALIGN)
        self.num_groups = len(shapes)
        self.block_p = max(16, triton.next_power_of_2(max_gtiles))
        self.block_k = block_k
        self.block_n = block_n
        # Pinned + non_blocking: a pageable H2D would block the host on the
        # stream. Built once per geometry; the pinned sources stay referenced.
        meta_h = torch.tensor(meta, dtype=torch.int64).pin_memory() if device.type == "cuda" else torch.tensor(meta, dtype=torch.int64)
        seg_h = torch.tensor(tile_seg, dtype=torch.int32).pin_memory() if device.type == "cuda" else torch.tensor(tile_seg, dtype=torch.int32)
        self._pinned = (meta_h, seg_h)
        self.meta = meta_h.to(device, non_blocking=True)
        self.tile_seg = seg_h.to(device, non_blocking=True)


_plans: Dict[Tuple[object, ...], _Plan] = {}
_PLAN_CACHE_MAX: int = 16


def _launch_quant(grid, ptrs, plan, partial, out, scales, margin) -> None:
    # No silent downgrade to `/`: a lowering failure propagates to the caller
    # (triton_fp8 then falls back to the aten chain for the whole process).
    _wshadow_quant_kernel[grid](
        *ptrs,
        plan.meta,
        plan.tile_seg,
        partial,
        out,
        scales,
        margin,
        INV_FP8_MAX_F32,
        AMAX_FLOOR_F32,
        FP8_MAX,
        BLOCK_K=plan.block_k,
        BLOCK_N=plan.block_n,
        BLOCK_P=plan.block_p,
        SCALE_STRIDE=_SCALE_STRIDE,
        PRECISE_DIV=_PRECISE_DIV,
        num_warps=4,
    )


def precise_div_in_use() -> bool:
    return _PRECISE_DIV


def fused_shadow_supported(weight: torch.Tensor) -> bool:
    return (
        weight.dim() == 2
        and weight.dtype in (torch.float32, torch.bfloat16)
        and weight.stride(1) == 1
        and weight.stride(0) >= weight.shape[1]
        and weight.device.type in ("cuda", "cpu")
    )


def refresh_shadows(
    weights: Sequence[torch.Tensor],
    layouts: Sequence[_Layout],
    *,
    margin: float = 1.0,
    block_k: int = 64,
    block_n: int = 64,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Quantize every (weight, layout) pair; returns [(q_col_major [K,N] e4m3, scale [1,1] fp32)].

    All weights must share device and dtype and pass fused_shadow_supported.
    Bitwise equal to calibrated_scale(packed, margin) + quantize_with_scale(
    packed, scale, column_major=True) for finite weights.
    """
    if not weights:
        return []
    device = weights[0].device
    dtype = weights[0].dtype
    e4m3 = torch.float8_e4m3fn
    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    # Chunk by distinct source tensors (<= MAX_SOURCES pointer args per launch).
    start = 0
    while start < len(weights):
        srcs: List[torch.Tensor] = []
        src_index: List[int] = []
        end = start
        while end < len(weights):
            w = weights[end]
            idx = next((i for i, s in enumerate(srcs) if s is w), -1)
            if idx < 0:
                if len(srcs) == MAX_SOURCES:
                    break
                srcs.append(w)
                idx = len(srcs) - 1
            src_index.append(idx)
            end += 1
        chunk_w = weights[start:end]
        chunk_l = layouts[start:end]
        key = (
            device.type,
            device.index,
            dtype,
            tuple((tuple(w.shape), w.stride(0)) for w in chunk_w),
            tuple(chunk_l),
            tuple(src_index),
            block_k,
            block_n,
        )
        plan = _plans.get(key)
        if plan is None:
            if len(_plans) >= _PLAN_CACHE_MAX:
                _plans.pop(next(iter(_plans)))
            plan = _Plan(
                [tuple(w.shape) for w in chunk_w],
                [w.stride(0) for w in chunk_w],
                chunk_l,
                src_index,
                device,
                block_k,
                block_n,
            )
            _plans[key] = plan
        out = torch.empty(plan.out_bytes, dtype=torch.uint8, device=device)
        scales = torch.empty(
            plan.num_groups * _SCALE_STRIDE, dtype=torch.float32, device=device
        )
        partial = torch.empty(plan.total_tiles, dtype=torch.int32, device=device)
        ptrs = [s.detach() for s in srcs] + [srcs[0].detach()] * (MAX_SOURCES - len(srcs))
        grid = (plan.total_tiles,)
        _wshadow_amax_kernel[grid](
            *ptrs,
            plan.meta,
            plan.tile_seg,
            partial,
            BLOCK_K=plan.block_k,
            BLOCK_N=plan.block_n,
            num_warps=4,
        )
        _launch_quant(
            grid, ptrs, plan, partial, out, scales, float(np.float32(margin))
        )
        q_all = out.view(e4m3)
        for g in range(plan.num_groups):
            k, n = plan.group_kn[g]
            off = plan.group_out_off[g]
            q = q_all[off : off + n * k].view(n, k).t()
            s = scales[g * _SCALE_STRIDE : g * _SCALE_STRIDE + 1].view(1, 1)
            results.append((q, s))
        start = end
    return results


def reference_shadow(
    weight: torch.Tensor, layout: _Layout, margin: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The incumbent aten chain, verbatim (triton_fp8 _pack_weight_layout +
    calibrated_scale + quantize_with_scale), for tests and the validator."""
    if layout == ((0, weight.shape[1]),):
        packed = weight
    elif len(layout) == 1:
        packed = weight[:, layout[0][0] : layout[0][1]]
    else:
        packed = torch.empty(
            (weight.shape[0], sum(e - s for s, e in layout)),
            dtype=weight.dtype,
            device=weight.device,
        )
        d = 0
        for s, e in layout:
            packed[:, d : d + e - s].copy_(weight[:, s:e])
            d += e - s
    amax = packed.detach().abs().amax().float().clamp_min(1.0e-12)
    scale = (amax * margin / FP8_MAX).reshape(1, 1)
    target = torch.float8_e4m3fn
    max_value = torch.finfo(target).max
    transposed = packed.detach().t().contiguous()
    q = (transposed / scale).clamp(-max_value, max_value).to(target).t()
    return q, scale
