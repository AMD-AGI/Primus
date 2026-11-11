###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX dual-stream block's gated residual + second adaLN as one FlyDSL op each way.

WHY:
  After its attention, each stream of a dual-stream block computes ``x1 = x + gate_msa * a`` (``a``
  the attention's output projection) and the modulated LayerNorm ``y = LN(x1) * (1 + scale_mlp) +
  shift_mlp`` that feeds its MLP. Inductor's backward for that is a row pass that writes the
  normalized rows as an FP32 buffer and reads it back, then two more passes that re-read the
  gradients for the column sums of the scale/shift and gate gradients. This op computes the column
  sums inside the row pass, from the normalized rows it recomputes in registers.

WHAT:
  ``primus_flux::dual_res_norm(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt,
  eps) -> (x1_img, y_img, x1_txt, y_txt, stats)`` for both streams in one launch: ``x1 = x + g * a``,
  ``y = LN(x1) * (1 + s) + h`` and each row's FP32 mean and rstd for the backward. ``g``, ``s`` and
  ``h`` are the [B, C] chunks of the stream's modulation GEMM output. The block's final residual
  reads ``x1``, so the residual gradient comes back as the gradient of ``x1``, which the backward adds;
  the first adaLN's backward (Inductor's) adds the returned gradient of ``x`` to its own.

  Three waves share each token row, 1,024 columns each (two vectors of 8 BF16 per lane), and add their
  row sums through LDS, double-buffered by row parity so one barrier per row suffices. With one wave
  per row, as in ``adaln_flydsl.py``, the three column accumulators of 48 values per lane spill out
  of the backward's registers. The backward accumulates the shift, scale and gate gradients per lane
  over the workgroup's 32 rows (all in one sample of one stream), and a small Triton kernel adds the
  per-workgroup partials.

NUMERICS:
  Inductor's arithmetic: ``x1`` in FP32 (``x + g * a``), ``1 + s`` in FP32, one rounding per output;
  each wave's row mean and squared deviations, combined exactly (Chan et al.), instead of Welford's.
  The LayerNorm output and all gradients are as close to FP64 as Inductor's or closer. ``x1`` is
  stored in BF16 for the final residual, which Inductor computed from the FP32 value: the block
  output carries one more BF16 rounding step.

REQUIRES:
  diffusers 0.40.0's ``FluxTransformerBlock.forward`` (source hash), BF16 activations and the
  diffusers attention processor (two outputs). Other calls keep diffusers' forward.

Settings (module config, ``primus_perf:`` section):
    flux_fused_adaln_dual_flydsl: false     apply this patch
"""
# No ``from __future__ import annotations``: FlyDSL reads the jit function's annotations as objects.
import functools
import hashlib
import inspect
import logging

import torch
import triton
import triton.language as tl

from primus.backends.automodel import options

logger = logging.getLogger(__name__)

_SECTION = "primus_perf"
_PATCHED_ATTR = "_primus_fused_adaln_dual_flydsl"
# sha256 of diffusers 0.40.0's FluxTransformerBlock.forward.
_DUAL_FORWARD_SHA256 = "0b184cbe172708b6f072a779e13866a44ef5a8ee706fcd3e6887923b11c00785"

VEC, LANES = 8, 64
# Three waves share each row, 1,024 columns each (two vectors of 8 per lane); they combine their
# row sums through LDS, double-buffered by row parity so one barrier per row suffices.
WAVES = 3
FWD_ROWS, BWD_ROWS = 8, 32  # rows per workgroup, in sequence
N_SUMS = 3  # shift, scale and gate gradients
_K = {}


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_adaln_dual_flydsl")


def supported(x_img: torch.Tensor, x_txt: torch.Tensor, vectors) -> bool:
    if not (x_img.dim() == 3 and x_txt.dim() == 3):
        return False
    batch, s_img, c = x_img.shape
    s_txt = x_txt.shape[1]
    return (
        all(t.is_cuda and t.dtype == torch.bfloat16 for t in (x_img, x_txt, *vectors))
        and x_txt.shape[0] == batch
        and x_txt.shape[2] == c
        and c % (WAVES * LANES * VEC) == 0
        and all(
            v.dim() == 2
            and v.shape == (batch, c)
            and v.stride(-1) == 1
            and v.stride(0) == vectors[0].stride(0)
            for v in vectors
        )
        and s_img % BWD_ROWS == 0
        and s_txt % BWD_ROWS == 0
        and s_img == s_txt
    )


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import flydsl.expr.math as fmath
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.buffer_ops import (
        _unwrap_value,
        buffer_load,
        buffer_store,
        create_buffer_resource,
    )
    from flydsl.expr.typing import Vector as Vec

    # LDS exchange: [parity][wave][lane][2] floats; every lane writes its own pair, readers take lane 0's.
    lds_floats = 2 * WAVES * LANES * 2

    def wave_sum(vs):
        # Sum of the vectors' elements over the 64 lanes; every lane gets the scalar.
        s = ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.ADD, vs[0]))
        for v in vs[1:]:
            s = s.addf(ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.ADD, v)))
        for o in (1, 2, 4, 8, 16, 32):
            s = s.addf(ArithValue(s.shuffle_xor(o, LANES)))
        return s

    def exchange(red, it, wave, lane, a, b):
        # Every wave's (a, b), after one barrier; slots alternate with the row's parity.
        base = (ArithValue(it) % fx.Int32(2)) * fx.Int32(WAVES * LANES * 2)
        mine = fx.add_offset(red.ptr, fx.make_int_tuple(base + (wave * fx.Int32(LANES) + lane) * fx.Int32(2)))
        fx.make_view(mine, fx.make_layout(2, 1)).store(Vec.from_elements([a, b], fx.Float32))
        fx.barrier()
        out = []
        for w in range(WAVES):
            p = fx.add_offset(red.ptr, fx.make_int_tuple(base + fx.Int32(w * LANES * 2)))
            v = Vec(fx.make_view(p, fx.make_layout(2, 1)).load())
            out.append((v[0], v[1]))
        return out

    def stream_select(blk_row0, n_img_rows, s_img, s_txt):
        # A workgroup's rows lie in one stream: image rows first, then text rows.
        is_img = blk_row0 < fx.Int32(n_img_rows)
        cond = _unwrap_value(is_img.ir_value() if hasattr(is_img, "ir_value") else is_img)
        off = ArithValue(fx.arith.select(cond, fx.Int32(0).ir_value(), fx.Int32(n_img_rows).ir_value()))
        s_seg = ArithValue(fx.arith.select(cond, fx.Int32(s_img).ir_value(), fx.Int32(s_txt).ir_value()))

        def pick(i, t):
            return _llvm.SelectOp(
                cond, create_buffer_resource(i, max_size=True), create_buffer_resource(t, max_size=True)
            ).result

        return pick, off, s_seg

    def lane_cols(c, wave, lane):
        per_wave = c // WAVES
        return [
            wave * fx.Int32(per_wave) + (lane + fx.Int32(k * LANES)) * fx.Int32(VEC)
            for k in range(per_wave // (LANES * VEC))
        ]

    def make_fwd(c, n_img_rows, s_img, s_txt, mod_stride, eps):
        @flyc.kernel(known_block_size=[LANES * WAVES, 1, 1])
        def fwd(
            XI: fx.Tensor,
            AI: fx.Tensor,
            GI: fx.Tensor,
            SI: fx.Tensor,
            HI: fx.Tensor,
            X1I: fx.Tensor,
            YI: fx.Tensor,
            XT: fx.Tensor,
            AT: fx.Tensor,
            GT: fx.Tensor,
            ST: fx.Tensor,
            HT: fx.Tensor,
            X1T: fx.Tensor,
            YT: fx.Tensor,
            STATS: fx.Tensor,
        ):
            @fx.struct
            class Smem:
                red: fx.Array[fx.Float32, lds_floats, 16]

            red = fx.SharedAllocator().allocate(Smem).peek().red
            f32v = fx.T.VectorType.get([VEC], fx.T.f32())
            bf16v = fx.T.VectorType.get([VEC], fx.T.bf16())
            bf16, f32 = fx.T.bf16(), fx.T.f32()
            t = fx.thread_idx.x
            bx, _, _ = fx.block_idx
            lane, wave = t % fx.Int32(LANES), t // fx.Int32(LANES)
            blk_row0 = bx * fx.Int32(FWD_ROWS)
            pick, off, s_seg = stream_select(blk_row0, n_img_rows, s_img, s_txt)
            x_r, a_r, g_r, s_r, h_r = pick(XI, XT), pick(AI, AT), pick(GI, GT), pick(SI, ST), pick(HI, HT)
            x1_r, y_r = pick(X1I, X1T), pick(YI, YT)
            st_r = create_buffer_resource(STATS, max_size=True)
            seg_row0 = blk_row0 - off
            mbase = (seg_row0 // s_seg) * fx.Int32(mod_stride)
            cols = lane_cols(c, wave, lane)
            one = fx.arith.constant_vector(1.0, f32v)
            n_w = float(c // WAVES)
            inv_nw = fx.arith.constant(1.0 / n_w, type=f32)
            inv_c = fx.arith.constant(1.0 / c, type=f32)
            eps_s = fx.arith.constant(eps, type=f32)
            # The sample's gate, scale and shift, kept packed in BF16 for the workgroup's rows.
            gv, sv, hv = (
                [buffer_load(r, mbase + col, vec_width=VEC, dtype=bf16) for col in cols]
                for r in (g_r, s_r, h_r)
            )

            def body(it):
                seg = (seg_row0 + ArithValue(it)) * fx.Int32(c)
                row = blk_row0 + ArithValue(it)
                xs = [buffer_load(x_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
                as_ = [buffer_load(a_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
                x1 = [
                    fx.arith.addf(
                        fx.arith.extf(f32v, x), fx.arith.mulf(fx.arith.extf(f32v, g), fx.arith.extf(f32v, a))
                    )
                    for x, a, g in zip(xs, as_, gv)
                ]
                for v, col in zip(x1, cols):
                    buffer_store(fx.arith.trunc_f(bf16v, v), x1_r, seg + col)
                # Each wave's mean and sum of squared deviations over its columns, combined as in
                # Chan et al. (equal counts): exact two-pass statistics of the whole row.
                mean_w = fx.arith.mulf(wave_sum(x1), inv_nw)
                d = [fx.arith.subf(v, _vector.broadcast(f32v, mean_w)) for v in x1]
                m2_w = wave_sum([fx.arith.mulf(v, v) for v in d])
                parts = exchange(red, it, wave, lane, mean_w, m2_w)
                mean_s = (parts[0][0] + parts[1][0] + parts[2][0]) * fx.Float32(1.0 / WAVES)
                m2 = parts[0][1] + parts[1][1] + parts[2][1]
                for mw, _ in parts:
                    dm = mw - mean_s
                    m2 = m2 + dm * dm * fx.Float32(n_w)
                mean_v = mean_s.ir_value() if hasattr(mean_s, "ir_value") else mean_s
                m2_v = m2.ir_value() if hasattr(m2, "ir_value") else m2
                rstd_s = fmath.rsqrt(fx.arith.addf(fx.arith.mulf(m2_v, inv_c), eps_s))
                rstd = _vector.broadcast(f32v, rstd_s)
                mean = _vector.broadcast(f32v, mean_v)
                for v, s, h, col in zip(x1, sv, hv, cols):
                    y = fx.arith.addf(
                        fx.arith.mulf(
                            fx.arith.mulf(fx.arith.subf(v, mean), rstd),
                            fx.arith.addf(fx.arith.extf(f32v, s), one),
                        ),
                        fx.arith.extf(f32v, h),
                    )
                    buffer_store(fx.arith.trunc_f(bf16v, y), y_r, seg + col)
                # Every lane stores the same two words: no branch needed.
                buffer_store(mean_v, st_r, row * fx.Int32(2))
                buffer_store(rstd_s, st_r, row * fx.Int32(2) + fx.Int32(1))

            for it in range(fx.Int32(FWD_ROWS)):
                body(it)

        return fwd

    def make_bwd(c, n_img_rows, s_img, s_txt, mod_stride):
        @flyc.kernel(known_block_size=[LANES * WAVES, 1, 1])
        def bwd(
            GYI: fx.Tensor,
            GXI: fx.Tensor,
            XI: fx.Tensor,
            AI: fx.Tensor,
            GI: fx.Tensor,
            SI: fx.Tensor,
            DXI: fx.Tensor,
            DAI: fx.Tensor,
            GYT: fx.Tensor,
            GXT: fx.Tensor,
            XT: fx.Tensor,
            AT: fx.Tensor,
            GT: fx.Tensor,
            ST: fx.Tensor,
            DXT: fx.Tensor,
            DAT: fx.Tensor,
            STATS: fx.Tensor,
            PART: fx.Tensor,
        ):
            @fx.struct
            class Smem:
                red: fx.Array[fx.Float32, lds_floats, 16]

            red = fx.SharedAllocator().allocate(Smem).peek().red
            f32v = fx.T.VectorType.get([VEC], fx.T.f32())
            f32v4 = fx.T.VectorType.get([4], fx.T.f32())
            bf16v = fx.T.VectorType.get([VEC], fx.T.bf16())
            bf16, f32 = fx.T.bf16(), fx.T.f32()
            t = fx.thread_idx.x
            bx, _, _ = fx.block_idx
            lane, wave = t % fx.Int32(LANES), t // fx.Int32(LANES)
            blk_row0 = bx * fx.Int32(BWD_ROWS)
            pick, off, s_seg = stream_select(blk_row0, n_img_rows, s_img, s_txt)
            gy_r, gx_r, x_r, a_r = pick(GYI, GYT), pick(GXI, GXT), pick(XI, XT), pick(AI, AT)
            g_r, s_r, dx_r, da_r = pick(GI, GT), pick(SI, ST), pick(DXI, DXT), pick(DAI, DAT)
            st_r, part_r = create_buffer_resource(STATS, max_size=True), create_buffer_resource(
                PART, max_size=True
            )
            seg_row0 = blk_row0 - off
            mbase = (seg_row0 // s_seg) * fx.Int32(mod_stride)
            cols = lane_cols(c, wave, lane)
            nv = len(cols)
            one = fx.arith.constant_vector(1.0, f32v)
            zero = fx.arith.constant_vector(0.0, f32v)
            inv_c = fx.arith.constant_vector(1.0 / c, f32v)
            c_v = fx.arith.constant_vector(float(c), f32v)
            gv = [buffer_load(g_r, mbase + col, vec_width=VEC, dtype=bf16) for col in cols]
            sv = [buffer_load(s_r, mbase + col, vec_width=VEC, dtype=bf16) for col in cols]
            accs = [
                [fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(nv)]
                for _ in range(N_SUMS)
            ]
            for acc in accs:
                for a in acc:
                    fx.memref_store_vec(zero, a)

            def body(it):
                row = blk_row0 + ArithValue(it)
                seg = (seg_row0 + ArithValue(it)) * fx.Int32(c)
                mean = _vector.broadcast(f32v, buffer_load(st_r, row * fx.Int32(2), vec_width=1, dtype=f32))
                rstd = _vector.broadcast(
                    f32v, buffer_load(st_r, row * fx.Int32(2) + fx.Int32(1), vec_width=1, dtype=f32)
                )
                gys = [buffer_load(gy_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
                xs = [buffer_load(x_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
                as_ = [buffer_load(a_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
                gxs = [buffer_load(gx_r, seg + col, vec_width=VEC, dtype=bf16) for col in cols]
                gyf = [fx.arith.extf(f32v, g) for g in gys]
                af = [fx.arith.extf(f32v, a) for a in as_]
                xhat = [
                    fx.arith.mulf(
                        fx.arith.subf(
                            fx.arith.addf(fx.arith.extf(f32v, x), fx.arith.mulf(fx.arith.extf(f32v, g), a)),
                            mean,
                        ),
                        rstd,
                    )
                    for x, a, g in zip(xs, af, gv)
                ]
                dxh = [
                    fx.arith.mulf(gy, fx.arith.addf(fx.arith.extf(f32v, s), one)) for gy, s in zip(gyf, sv)
                ]
                parts = exchange(
                    red,
                    it,
                    wave,
                    lane,
                    wave_sum(dxh),
                    wave_sum([fx.arith.mulf(a, h) for a, h in zip(dxh, xhat)]),
                )
                s1_s = parts[0][0] + parts[1][0] + parts[2][0]
                s2_s = parts[0][1] + parts[1][1] + parts[2][1]
                s1 = _vector.broadcast(f32v, s1_s.ir_value() if hasattr(s1_s, "ir_value") else s1_s)
                s2 = _vector.broadcast(f32v, s2_s.ir_value() if hasattr(s2_s, "ir_value") else s2_s)
                # Inductor's native_layer_norm_backward: rstd / C * (C * dxh - s1 - xhat * s2), + residual.
                scale_r = fx.arith.mulf(rstd, inv_c)
                for k, col in enumerate(cols):
                    dx = fx.arith.mulf(
                        scale_r,
                        fx.arith.subf(
                            fx.arith.subf(fx.arith.mulf(dxh[k], c_v), s1), fx.arith.mulf(xhat[k], s2)
                        ),
                    )
                    dx = fx.arith.addf(fx.arith.extf(f32v, gxs[k]), dx)
                    buffer_store(fx.arith.trunc_f(bf16v, dx), dx_r, seg + col)
                    buffer_store(
                        fx.arith.trunc_f(bf16v, fx.arith.mulf(dx, fx.arith.extf(f32v, gv[k]))),
                        da_r,
                        seg + col,
                    )
                    for acc, v in (
                        (accs[0][k], gyf[k]),
                        (accs[1][k], fx.arith.mulf(gyf[k], xhat[k])),
                        (accs[2][k], fx.arith.mulf(af[k], dx)),
                    ):
                        fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(acc), v), acc)

            for it in range(fx.Int32(BWD_ROWS)):
                body(it)

            # PART [workgroups, 3, C]: the shift-, scale- and gate-gradient column sums over the
            # workgroup's rows; each wave writes its own columns.
            pbase = bx * fx.Int32(N_SUMS * c)
            for k, col in enumerate(cols):
                for j in range(N_SUMS):
                    v = fx.memref_load_vec(accs[j][k])
                    for half in (0, 1):
                        buffer_store(
                            _vector.extract_strided_slice(f32v4, v, [half * 4], [4], [1]),
                            part_r,
                            pbase + fx.Int32(j * c) + col + fx.Int32(half * 4),
                        )

        return bwd

    @flyc.jit
    def launch_fwd(
        XI,
        AI,
        GI,
        SI,
        HI,
        X1I,
        YI,
        XT,
        AT,
        GT,
        ST,
        HT,
        X1T,
        YT,
        STATS,
        c: fx.Constexpr[int],
        n_rows: fx.Constexpr[int],
        n_img_rows: fx.Constexpr[int],
        s_img: fx.Constexpr[int],
        s_txt: fx.Constexpr[int],
        mod_stride: fx.Constexpr[int],
        eps: fx.Constexpr[float],
        stream: fx.Stream,
    ):
        make_fwd(c, n_img_rows, s_img, s_txt, mod_stride, eps)(
            XI, AI, GI, SI, HI, X1I, YI, XT, AT, GT, ST, HT, X1T, YT, STATS
        ).launch(grid=(n_rows // FWD_ROWS, 1, 1), block=(LANES * WAVES, 1, 1), stream=stream)

    @flyc.jit
    def launch_bwd(
        GYI,
        GXI,
        XI,
        AI,
        GI,
        SI,
        DXI,
        DAI,
        GYT,
        GXT,
        XT,
        AT,
        GT,
        ST,
        DXT,
        DAT,
        STATS,
        PART,
        c: fx.Constexpr[int],
        n_rows: fx.Constexpr[int],
        n_img_rows: fx.Constexpr[int],
        s_img: fx.Constexpr[int],
        s_txt: fx.Constexpr[int],
        mod_stride: fx.Constexpr[int],
        stream: fx.Stream,
    ):
        make_bwd(c, n_img_rows, s_img, s_txt, mod_stride)(
            GYI, GXI, XI, AI, GI, SI, DXI, DAI, GYT, GXT, XT, AT, GT, ST, DXT, DAT, STATS, PART
        ).launch(grid=(n_rows // BWD_ROWS, 1, 1), block=(LANES * WAVES, 1, 1), stream=stream)

    return launch_fwd, launch_bwd


def _kernels():
    if not _K:
        _K["fwd"], _K["bwd"] = _build()
    return _K["fwd"], _K["bwd"]


@triton.jit
def _sample_colsum_kernel(src_ptr, out_ptr, n_parts, n_cols, BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    # out[b, c] = sum_p src[b, p, c] for the FP32 per-wave partials.
    b = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < n_cols
    acc = tl.zeros([BLOCK_P, BLOCK_C], dtype=tl.float32)
    for p0 in range(0, n_parts, BLOCK_P):
        p = p0 + tl.arange(0, BLOCK_P)
        acc += tl.load(
            src_ptr + (b * n_parts + p)[:, None] * n_cols + cols[None, :],
            mask=(p < n_parts)[:, None] & col_mask[None, :],
            other=0.0,
        )
    tl.store(out_ptr + b * n_cols + cols, tl.sum(acc, axis=0).to(out_ptr.dtype.element_ty), mask=col_mask)


def forward(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps):
    launch_fwd = _kernels()[0]
    x_img, a_img, x_txt, a_txt = (t.contiguous() for t in (x_img, a_img, x_txt, a_txt))
    batch, n_img, c = x_img.shape
    n_txt = x_txt.shape[1]
    x1_img, y_img, x1_txt, y_txt = (torch.empty_like(t) for t in (x_img, x_img, x_txt, x_txt))
    n_rows = batch * (n_img + n_txt)
    stats = torch.empty((n_rows, 2), device=x_img.device, dtype=torch.float32)
    launch_fwd(
        x_img,
        a_img,
        g_img,
        s_img,
        h_img,
        x1_img,
        y_img,
        x_txt,
        a_txt,
        g_txt,
        s_txt,
        h_txt,
        x1_txt,
        y_txt,
        stats,
        c=c,
        n_rows=n_rows,
        n_img_rows=batch * n_img,
        s_img=n_img,
        s_txt=n_txt,
        mod_stride=g_img.stride(0),
        eps=float(eps),
        stream=torch.cuda.current_stream(),
    )
    return x1_img, y_img, x1_txt, y_txt, stats


def backward(gy_img, gx_img, x_img, a_img, g_img, s_img, gy_txt, gx_txt, x_txt, a_txt, g_txt, s_txt, stats):
    launch_bwd = _kernels()[1]
    gy_img, gx_img, gy_txt, gx_txt = (t.contiguous() for t in (gy_img, gx_img, gy_txt, gx_txt))
    batch, n_img, c = x_img.shape
    n_txt = x_txt.shape[1]
    if n_txt != n_img:
        raise ValueError("dual_res_norm backward: the per-sample partial sum needs equal stream lengths")
    dx_img, da_img, dx_txt, da_txt = (torch.empty_like(t) for t in (x_img, x_img, x_txt, x_txt))
    n_rows = batch * (n_img + n_txt)
    part = torch.empty((n_rows // BWD_ROWS, N_SUMS * c), device=x_img.device, dtype=torch.float32)
    launch_bwd(
        gy_img,
        gx_img,
        x_img,
        a_img,
        g_img,
        s_img,
        dx_img,
        da_img,
        gy_txt,
        gx_txt,
        x_txt,
        a_txt,
        g_txt,
        s_txt,
        dx_txt,
        da_txt,
        stats,
        part,
        c=c,
        n_rows=n_rows,
        n_img_rows=batch * n_img,
        s_img=n_img,
        s_txt=n_txt,
        mod_stride=g_img.stride(0),
        stream=torch.cuda.current_stream(),
    )
    # The partials of a sample's rows are consecutive within its stream, image samples first; both
    # streams have the same number of rows per sample.
    parts_per_sample = n_img // BWD_ROWS
    vec = torch.empty((2 * batch, N_SUMS, c), device=x_img.device, dtype=g_img.dtype)
    _sample_colsum_kernel[(2 * batch, triton.cdiv(N_SUMS * c, 256))](
        part, vec, parts_per_sample, N_SUMS * c, BLOCK_P=8, BLOCK_C=256, num_warps=4
    )
    return dx_img, da_img, dx_txt, da_txt, vec


# --- the custom op -----------------------------------------------------------------------------


@torch.library.custom_op("primus_flux::dual_res_norm", mutates_args=(), device_types="cuda")
def dual_res_norm(
    x_img: torch.Tensor,
    a_img: torch.Tensor,
    g_img: torch.Tensor,
    s_img: torch.Tensor,
    h_img: torch.Tensor,
    x_txt: torch.Tensor,
    a_txt: torch.Tensor,
    g_txt: torch.Tensor,
    s_txt: torch.Tensor,
    h_txt: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per stream ``x1 = x + g[:, None] * a`` and ``layer_norm(x1) * (1 + s[:, None]) + h[:, None]``, as
    (x1_img, y_img, x1_txt, y_txt, FP32 [rows, 2] row mean and rstd); the statistics are for the backward."""
    return forward(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps)


@dual_res_norm.register_fake
def _(x_img, a_img, g_img, s_img, h_img, x_txt, a_txt, g_txt, s_txt, h_txt, eps):
    n_rows = x_img.shape[0] * (x_img.shape[1] + x_txt.shape[1])
    return (
        torch.empty_like(x_img),
        torch.empty_like(x_img),
        torch.empty_like(x_txt),
        torch.empty_like(x_txt),
        x_img.new_empty((n_rows, 2), dtype=torch.float32),
    )


@torch.library.custom_op("primus_flux::dual_res_norm_backward", mutates_args=(), device_types="cuda")
def dual_res_norm_backward(
    gy_img: torch.Tensor,
    gx_img: torch.Tensor,
    x_img: torch.Tensor,
    a_img: torch.Tensor,
    g_img: torch.Tensor,
    s_img: torch.Tensor,
    gy_txt: torch.Tensor,
    gx_txt: torch.Tensor,
    x_txt: torch.Tensor,
    a_txt: torch.Tensor,
    g_txt: torch.Tensor,
    s_txt: torch.Tensor,
    stats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(gradients of x_img, a_img, x_txt, a_txt, and [2B, 3, C] shift, scale and gate gradients, image
    samples first) for the gradients ``gy`` of the LayerNorm outputs and ``gx`` of ``x1``."""
    return backward(
        gy_img, gx_img, x_img, a_img, g_img, s_img, gy_txt, gx_txt, x_txt, a_txt, g_txt, s_txt, stats
    )


@dual_res_norm_backward.register_fake
def _(gy_img, gx_img, x_img, a_img, g_img, s_img, gy_txt, gx_txt, x_txt, a_txt, g_txt, s_txt, stats):
    return (
        torch.empty_like(x_img),
        torch.empty_like(x_img),
        torch.empty_like(x_txt),
        torch.empty_like(x_txt),
        g_img.new_empty((2 * x_img.shape[0], N_SUMS, x_img.shape[2])),
    )


def _setup_context(ctx, inputs, output):
    x_img, a_img, g_img, s_img, _, x_txt, a_txt, g_txt, s_txt, _, _ = inputs
    ctx.mark_non_differentiable(output[4])
    ctx.save_for_backward(x_img, a_img, g_img, s_img, x_txt, a_txt, g_txt, s_txt, output[4])


def _backward(ctx, gx_img, gy_img, gx_txt, gy_txt, _g_stats):
    x_img, a_img, g_img, s_img, x_txt, a_txt, g_txt, s_txt, stats = ctx.saved_tensors
    gx_img = torch.zeros_like(x_img) if gx_img is None else gx_img
    gy_img = torch.zeros_like(x_img) if gy_img is None else gy_img
    gx_txt = torch.zeros_like(x_txt) if gx_txt is None else gx_txt
    gy_txt = torch.zeros_like(x_txt) if gy_txt is None else gy_txt
    dx_img, da_img, dx_txt, da_txt, vec = dual_res_norm_backward(
        gy_img, gx_img, x_img, a_img, g_img, s_img, gy_txt, gx_txt, x_txt, a_txt, g_txt, s_txt, stats
    )
    batch = x_img.shape[0]
    vi, vt = vec[:batch], vec[batch:]
    return (dx_img, da_img, vi[:, 2], vi[:, 1], vi[:, 0], dx_txt, da_txt, vt[:, 2], vt[:, 1], vt[:, 0], None)


dual_res_norm.register_autograd(_backward, setup_context=_setup_context)


# --- the dual-stream block ---------------------------------------------------------------------


def _plain_norm(n) -> bool:
    return (
        isinstance(n, torch.nn.LayerNorm)
        and n.weight is None
        and n.bias is None
        and len(n.normalized_shape) == 1
    )


def make_forward(orig):
    """diffusers' dual-stream forward with each stream's gated residual + second LayerNorm and modulation
    as ``dual_res_norm``; ``orig`` serves the calls it does not support."""
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        if (
            joint_attention_kwargs
            or not isinstance(self.attn.processor, FluxAttnProcessor)
            or not (_plain_norm(self.norm2) and _plain_norm(self.norm2_context))
            or self.norm2.eps != self.norm2_context.eps
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        vectors = (gate_msa, scale_mlp, shift_mlp, c_gate_msa, c_scale_mlp, c_shift_mlp)
        if not supported(hidden_states, encoder_hidden_states, vectors):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        x1, norm_hidden_states, c_x1, norm_encoder_hidden_states, _ = dual_res_norm(
            hidden_states,
            attn_output,
            gate_msa,
            scale_mlp,
            shift_mlp,
            encoder_hidden_states,
            context_attn_output,
            c_gate_msa,
            c_scale_mlp,
            c_shift_mlp,
            self.norm2.eps,
        )
        hidden_states = x1 + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)
        encoder_hidden_states = c_x1 + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_encoder_hidden_states)
        return encoder_hidden_states, hidden_states

    return forward


def install() -> bool:
    from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

    if getattr(FluxTransformerBlock, _PATCHED_ATTR, False):
        return True
    orig = FluxTransformerBlock.forward
    digest = hashlib.sha256(inspect.getsource(inspect.unwrap(orig)).encode()).hexdigest()
    if digest != _DUAL_FORWARD_SHA256:
        logger.warning(
            "[PrimusFluxAdaLNDual] FluxTransformerBlock.forward is not diffusers 0.40.0's; not applied."
        )
        return False
    FluxTransformerBlock.forward = make_forward(orig)
    setattr(FluxTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxAdaLNDual] dual-stream blocks run the gated residual + second adaLN as one FlyDSL op "
        "each way."
    )
    return True
