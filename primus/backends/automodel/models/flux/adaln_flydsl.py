###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The FLUX single-stream block's adaLN (cat + LayerNorm + modulation) as one FlyDSL op each way.

WHY:
  Inductor compiles the adaLN of a single-stream block into three kernels: a forward that writes
  the concatenated residual and the modulated LayerNorm output (a looped two-pass reduction, the
  rows being 3,072 wide), a LayerNorm backward that also adds the residual gradient, and a
  separate kernel for the shift and scale gradients, which re-reads the output gradient and the
  input: a whole extra pass per call. It also returns the two input gradients as slices of one
  buffer, which AOTAutograd copies to contiguous tensors at every block boundary of the backward.
  This op computes the shift and scale gradients inside the LayerNorm-backward pass and writes the
  input gradients as separate tensors.

WHAT:
  ``primus_flux::adaln_single(x_txt, x_img, scale, shift, eps) -> (r_txt, r_img, y, stats)``:
  ``r_txt``/``r_img`` are copies of the inputs for the residual add, ``y`` the modulated
  LayerNorm of their concat, ``stats`` each row's FP32 mean and rstd for the backward. The
  residual leaves as two tensors so that its gradient comes back as the two block-output
  gradients, which the backward reads directly: with a single concatenated residual, Inductor
  materializes their concatenation around the op and gives the fusion's gain back.

  One wavefront per token row: 64 lanes x 6 vectors of 8 BF16 cover the 3,072 columns, so the row
  sums are 64-lane butterflies. The backward accumulates the shift and scale gradients per lane
  over its wave's 32 rows (all in one sample), and a small Triton kernel adds the per-wave
  partials. A backward block's 128 rows lie in one segment (text or image), so it selects its
  input and gradient tensors once. Written for FlyDSL 0.2.4, the version Primus-Turbo pins.

NUMERICS:
  Inductor's arithmetic: 1 + scale rounded to BF16, FP32 inside, one rounding per output; only
  the order of the sums differs. A small fraction of outputs, input gradients and shift/scale
  gradients differ from Inductor's by one BF16 step; all are as close to an FP64 reference as
  Inductor's.

REQUIRES:
  ``mlp_gelu.py``'s single-stream forward, on diffusers 0.40.0's (source hash), which this forward
  re-implements with the adaLN replaced. Installs after it (priority 75) and before
  ``block_outputs.py`` and ``last_block.py``, which wrap it; the image-only last block keeps
  Inductor's adaLN.

Settings (module config, ``primus_perf:`` section):
    flux_fused_adaln_flydsl: false     apply this patch
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
_PATCHED_ATTR = "_primus_fused_adaln_flydsl"

VEC, LANES = 8, 64
FWD_ROWS, FWD_WAVES = 8, 4
BWD_ROWS, BWD_WAVES = 32, 4
_K = {}


def is_enabled() -> bool:
    return options.flag(f"{_SECTION}.flux_fused_adaln_flydsl")


def supported(x_txt: torch.Tensor, x_img: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    if not (x_txt.dim() == 3 and x_img.dim() == 3 and scale.dim() == 2 and shift.dim() == 2):
        return False
    batch, s_txt, c = x_txt.shape
    s_img = x_img.shape[1]
    bwd_block = BWD_ROWS * BWD_WAVES
    return (
        all(t.is_cuda and t.dtype == torch.bfloat16 for t in (x_txt, x_img, scale, shift))
        and x_img.shape[0] == batch
        and x_img.shape[2] == c
        and c % (LANES * VEC) == 0
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and scale.stride(0) == shift.stride(0)
        and all(s % FWD_ROWS == 0 and (batch * s) % (FWD_ROWS * FWD_WAVES) == 0 for s in (s_txt, s_img))
        and s_txt % bwd_block == 0
        and s_img % bwd_block == 0
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

    def wave_sum(vs):
        # Sum of the vectors' elements over the 64 lanes; every lane gets the scalar.
        s = ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.ADD, vs[0]))
        for v in vs[1:]:
            s = s.addf(ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.ADD, v)))
        for o in (1, 2, 4, 8, 16, 32):
            s = s.addf(ArithValue(s.shuffle_xor(o, LANES)))
        return s

    def make_fwd(c, s_seg, s_out, seq_off, mod_stride, eps):
        nv = c // (LANES * VEC)
        threads = LANES * FWD_WAVES

        @flyc.kernel(known_block_size=[threads, 1, 1])
        def fwd(
            SRC: fx.Tensor, SCALE: fx.Tensor, SHIFT: fx.Tensor, RES: fx.Tensor, Y: fx.Tensor, STATS: fx.Tensor
        ):
            f32v = fx.T.VectorType.get([VEC], fx.T.f32())
            bf16v = fx.T.VectorType.get([VEC], fx.T.bf16())
            bf16, f32 = fx.T.bf16(), fx.T.f32()
            t = fx.thread_idx.x
            bx, _, _ = fx.block_idx
            lane = t % fx.Int32(LANES)
            row0 = (bx * fx.Int32(FWD_WAVES) + t // fx.Int32(LANES)) * fx.Int32(FWD_ROWS)
            b = row0 // fx.Int32(s_seg)
            out0 = b * fx.Int32(s_out) + fx.Int32(seq_off) + row0 % fx.Int32(s_seg)
            cols = [(lane + fx.Int32(k * LANES)) * fx.Int32(VEC) for k in range(nv)]
            src_r, res_r, y_r = (create_buffer_resource(x, max_size=True) for x in (SRC, RES, Y))
            sc_r, sh_r, st_r = (create_buffer_resource(x, max_size=True) for x in (SCALE, SHIFT, STATS))
            one = fx.arith.constant_vector(1.0, f32v)
            inv_c = fx.arith.constant(1.0 / c, type=f32)
            eps_s = fx.arith.constant(eps, type=f32)
            mbase = b * fx.Int32(mod_stride)
            # 1 + scale rounded to BF16, as Inductor stores it; both stay in registers for the wave's rows.
            onep = [
                fx.arith.extf(
                    f32v,
                    fx.arith.trunc_f(
                        bf16v,
                        fx.arith.addf(
                            fx.arith.extf(f32v, buffer_load(sc_r, mbase + col, vec_width=VEC, dtype=bf16)),
                            one,
                        ),
                    ),
                )
                for col in cols
            ]
            shift = [
                fx.arith.extf(f32v, buffer_load(sh_r, mbase + col, vec_width=VEC, dtype=bf16)) for col in cols
            ]

            def body(it):
                src_off = (row0 + ArithValue(it)) * fx.Int32(c)
                out_row = out0 + ArithValue(it)
                out_off = out_row * fx.Int32(c)
                xs = [buffer_load(src_r, src_off + col, vec_width=VEC, dtype=bf16) for col in cols]
                for x, col in zip(xs, cols):
                    buffer_store(x, res_r, src_off + col)
                xf = [fx.arith.extf(f32v, x) for x in xs]
                mean_s = fx.arith.mulf(wave_sum(xf), inv_c)
                d = [fx.arith.subf(x, _vector.broadcast(f32v, mean_s)) for x in xf]
                rstd_s = fmath.rsqrt(
                    fx.arith.addf(fx.arith.mulf(wave_sum([fx.arith.mulf(v, v) for v in d]), inv_c), eps_s)
                )
                rstd = _vector.broadcast(f32v, rstd_s)
                for v, op, sh, col in zip(d, onep, shift, cols):
                    y = fx.arith.addf(fx.arith.mulf(fx.arith.mulf(v, rstd), op), sh)
                    buffer_store(fx.arith.trunc_f(bf16v, y), y_r, out_off + col)
                # Every lane stores the same two words: no branch needed.
                buffer_store(mean_s, st_r, out_row * fx.Int32(2))
                buffer_store(rstd_s, st_r, out_row * fx.Int32(2) + fx.Int32(1))

            for it in range(fx.Int32(FWD_ROWS)):
                body(it)

        return fwd

    def make_bwd(c, s_txt, s_img, mod_stride):
        nv = c // (LANES * VEC)
        threads = LANES * BWD_WAVES
        s_tok = s_txt + s_img

        @flyc.kernel(known_block_size=[threads, 1, 1])
        def bwd(
            G: fx.Tensor,
            XT: fx.Tensor,
            XI: fx.Tensor,
            GRT: fx.Tensor,
            GRI: fx.Tensor,
            SCALE: fx.Tensor,
            STATS: fx.Tensor,
            DXT: fx.Tensor,
            DXI: fx.Tensor,
            PART: fx.Tensor,
        ):
            f32v = fx.T.VectorType.get([VEC], fx.T.f32())
            f32v4 = fx.T.VectorType.get([4], fx.T.f32())
            bf16v = fx.T.VectorType.get([VEC], fx.T.bf16())
            bf16, f32 = fx.T.bf16(), fx.T.f32()
            t = fx.thread_idx.x
            bx, _, _ = fx.block_idx
            lane = t % fx.Int32(LANES)
            # The block's rows are in one sample and one segment: choose that segment's tensors once.
            blk_row0 = bx * fx.Int32(BWD_ROWS * BWD_WAVES)
            b = blk_row0 // fx.Int32(s_tok)
            s_blk = blk_row0 % fx.Int32(s_tok)
            is_txt = s_blk < fx.Int32(s_txt)
            cond = _unwrap_value(is_txt.ir_value() if hasattr(is_txt, "ir_value") else is_txt)

            def pick(a, i):
                return _llvm.SelectOp(
                    cond, create_buffer_resource(a, max_size=True), create_buffer_resource(i, max_size=True)
                ).result

            x_r, res_r, dx_r = pick(XT, XI), pick(GRT, GRI), pick(DXT, DXI)
            g_r, sc_r, st_r, part_r = (
                create_buffer_resource(x, max_size=True) for x in (G, SCALE, STATS, PART)
            )
            seg_len = fx.arith.select(cond, fx.Int32(s_txt).ir_value(), fx.Int32(s_img).ir_value())
            seg_off = fx.arith.select(cond, fx.Int32(0).ir_value(), fx.Int32(s_txt).ir_value())
            wave = bx * fx.Int32(BWD_WAVES) + t // fx.Int32(LANES)
            row0 = wave * fx.Int32(BWD_ROWS)
            seg_row0 = b * ArithValue(seg_len) + row0 % fx.Int32(s_tok) - ArithValue(seg_off)
            cols = [(lane + fx.Int32(k * LANES)) * fx.Int32(VEC) for k in range(nv)]
            one = fx.arith.constant_vector(1.0, f32v)
            zero = fx.arith.constant_vector(0.0, f32v)
            inv_c = fx.arith.constant_vector(1.0 / c, f32v)
            c_v = fx.arith.constant_vector(float(c), f32v)
            mbase = b * fx.Int32(mod_stride)
            # 1 + scale, rounded to BF16 as Inductor stores it, kept packed: 24 registers instead of 48.
            onep = [
                fx.arith.trunc_f(
                    bf16v,
                    fx.arith.addf(
                        fx.arith.extf(f32v, buffer_load(sc_r, mbase + col, vec_width=VEC, dtype=bf16)), one
                    ),
                )
                for col in cols
            ]
            acc_sh = [fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(nv)]
            acc_sc = [fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(nv)]
            for a in acc_sh + acc_sc:
                fx.memref_store_vec(zero, a)

            def load_row(rsrc, off, policy):
                return [
                    buffer_load(rsrc, off + col, vec_width=VEC, dtype=bf16, cache_modifier=policy)
                    for col in cols
                ]

            def terms(gs, xs, mean, rstd):
                gf = [fx.arith.extf(f32v, g) for g in gs]
                xhat = [fx.arith.mulf(fx.arith.subf(fx.arith.extf(f32v, x), mean), rstd) for x in xs]
                dxh = [fx.arith.mulf(g, fx.arith.extf(f32v, op)) for g, op in zip(gf, onep)]
                return gf, xhat, dxh

            def body(it):
                row = row0 + ArithValue(it)
                off = row * fx.Int32(c)
                seg = (seg_row0 + ArithValue(it)) * fx.Int32(c)
                mean = _vector.broadcast(f32v, buffer_load(st_r, row * fx.Int32(2), vec_width=1, dtype=f32))
                rstd = _vector.broadcast(
                    f32v, buffer_load(st_r, row * fx.Int32(2) + fx.Int32(1), vec_width=1, dtype=f32)
                )
                # Pass 1: the row sums. Its temporaries die here; pass 2 reloads the row (from L2) with
                # another cache policy, so LLVM cannot merge the loads and keep 144 FP32 values live.
                _, xhat, dxh = terms(load_row(g_r, off, 0), load_row(x_r, seg, 0), mean, rstd)
                s1 = _vector.broadcast(f32v, wave_sum(dxh))
                s2 = _vector.broadcast(f32v, wave_sum([fx.arith.mulf(a, h) for a, h in zip(dxh, xhat)]))
                gf, xhat, dxh = terms(load_row(g_r, off, 1), load_row(x_r, seg, 1), mean, rstd)
                rs = load_row(res_r, seg, 0)
                # Inductor's native_layer_norm_backward: rstd / C * (C * dxh - s1 - xhat * s2), + residual.
                scale_r = fx.arith.mulf(rstd, inv_c)
                for k, col in enumerate(cols):
                    dx = fx.arith.mulf(
                        scale_r,
                        fx.arith.subf(
                            fx.arith.subf(fx.arith.mulf(dxh[k], c_v), s1), fx.arith.mulf(xhat[k], s2)
                        ),
                    )
                    dx = fx.arith.addf(fx.arith.extf(f32v, rs[k]), dx)
                    buffer_store(fx.arith.trunc_f(bf16v, dx), dx_r, seg + col)
                    fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(acc_sh[k]), gf[k]), acc_sh[k])
                    fx.memref_store_vec(
                        fx.arith.addf(fx.memref_load_vec(acc_sc[k]), fx.arith.mulf(gf[k], xhat[k])), acc_sc[k]
                    )

            for it in range(fx.Int32(BWD_ROWS)):
                body(it)

            # PART [waves, 2, C]: the wave's shift- and scale-gradient column sums over its rows.
            pbase = wave * fx.Int32(2 * c)
            for k, col in enumerate(cols):
                for j, acc in ((0, acc_sh[k]), (1, acc_sc[k])):
                    v = fx.memref_load_vec(acc)
                    for half in (0, 1):
                        buffer_store(
                            _vector.extract_strided_slice(f32v4, v, [half * 4], [4], [1]),
                            part_r,
                            pbase + fx.Int32(j * c) + col + fx.Int32(half * 4),
                        )

        return bwd

    @flyc.jit
    def launch_fwd(
        SRC,
        SCALE,
        SHIFT,
        RES,
        Y,
        STATS,
        c: fx.Constexpr[int],
        n_rows: fx.Constexpr[int],
        s_seg: fx.Constexpr[int],
        s_out: fx.Constexpr[int],
        seq_off: fx.Constexpr[int],
        mod_stride: fx.Constexpr[int],
        eps: fx.Constexpr[float],
        stream: fx.Stream,
    ):
        make_fwd(c, s_seg, s_out, seq_off, mod_stride, eps)(SRC, SCALE, SHIFT, RES, Y, STATS).launch(
            grid=(n_rows // (FWD_ROWS * FWD_WAVES), 1, 1), block=(LANES * FWD_WAVES, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_bwd(
        G,
        XT,
        XI,
        GRT,
        GRI,
        SCALE,
        STATS,
        DXT,
        DXI,
        PART,
        c: fx.Constexpr[int],
        n_rows: fx.Constexpr[int],
        s_txt: fx.Constexpr[int],
        s_img: fx.Constexpr[int],
        mod_stride: fx.Constexpr[int],
        stream: fx.Stream,
    ):
        make_bwd(c, s_txt, s_img, mod_stride)(G, XT, XI, GRT, GRI, SCALE, STATS, DXT, DXI, PART).launch(
            grid=(n_rows // (BWD_ROWS * BWD_WAVES), 1, 1), block=(LANES * BWD_WAVES, 1, 1), stream=stream
        )

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


def forward(x_txt, x_img, scale, shift, eps):
    launch_fwd = _kernels()[0]
    x_txt, x_img = x_txt.contiguous(), x_img.contiguous()
    batch, s_txt, c = x_txt.shape
    s_tok = s_txt + x_img.shape[1]
    r_txt, r_img = torch.empty_like(x_txt), torch.empty_like(x_img)
    y = torch.empty((batch, s_tok, c), device=x_txt.device, dtype=x_txt.dtype)
    stats = torch.empty((batch, s_tok, 2), device=x_txt.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    for src, res, off in ((x_txt, r_txt, 0), (x_img, r_img, s_txt)):
        launch_fwd(
            src,
            scale,
            shift,
            res,
            y,
            stats,
            c=c,
            n_rows=batch * src.shape[1],
            s_seg=src.shape[1],
            s_out=s_tok,
            seq_off=off,
            mod_stride=scale.stride(0),
            eps=float(eps),
            stream=stream,
        )
    return r_txt, r_img, y, stats


def backward(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats):
    launch_bwd = _kernels()[1]
    g_y, g_r_txt, g_r_img = g_y.contiguous(), g_r_txt.contiguous(), g_r_img.contiguous()
    batch, s_txt, c = x_txt.shape
    s_img = x_img.shape[1]
    s_tok = s_txt + s_img
    dx_txt, dx_img = torch.empty_like(x_txt), torch.empty_like(x_img)
    parts_per_sample = s_tok // BWD_ROWS
    part = torch.empty((batch, parts_per_sample, 2 * c), device=x_txt.device, dtype=torch.float32)
    launch_bwd(
        g_y,
        x_txt,
        x_img,
        g_r_txt,
        g_r_img,
        scale,
        stats,
        dx_txt,
        dx_img,
        part,
        c=c,
        n_rows=batch * s_tok,
        s_txt=s_txt,
        s_img=s_img,
        mod_stride=scale.stride(0),
        stream=torch.cuda.current_stream(),
    )
    dss = torch.empty((batch, 2, c), device=x_txt.device, dtype=scale.dtype)
    _sample_colsum_kernel[(batch, triton.cdiv(2 * c, 256))](
        part, dss, parts_per_sample, 2 * c, BLOCK_P=16, BLOCK_C=256, num_warps=4
    )
    return dx_txt, dx_img, dss


# --- the custom op -----------------------------------------------------------------------------


@torch.library.custom_op("primus_flux::adaln_single", mutates_args=(), device_types="cuda")
def adaln_single(
    x_txt: torch.Tensor, x_img: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(copy of x_txt, copy of x_img, layer_norm(cat([x_txt, x_img], 1)) * (1 + scale[:, None]) + shift[:, None],
    FP32 [B, S, 2] row mean and rstd) for BF16 [B, S, C]; the statistics are returned only for the backward.
    """
    return forward(x_txt, x_img, scale, shift, eps)


@adaln_single.register_fake
def _(x_txt, x_img, scale, shift, eps):
    shape = (x_txt.shape[0], x_txt.shape[1] + x_img.shape[1], x_txt.shape[2])
    return (
        torch.empty_like(x_txt),
        torch.empty_like(x_img),
        x_txt.new_empty(shape),
        x_txt.new_empty(shape[:2] + (2,), dtype=torch.float32),
    )


@torch.library.custom_op("primus_flux::adaln_single_backward", mutates_args=(), device_types="cuda")
def adaln_single_backward(
    g_y: torch.Tensor,
    g_r_txt: torch.Tensor,
    g_r_img: torch.Tensor,
    x_txt: torch.Tensor,
    x_img: torch.Tensor,
    scale: torch.Tensor,
    stats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(gradient of x_txt, gradient of x_img, [B, 2, C] gradients of shift and scale)."""
    return backward(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats)


@adaln_single_backward.register_fake
def _(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats):
    return (
        torch.empty_like(x_txt),
        torch.empty_like(x_img),
        scale.new_empty((x_txt.shape[0], 2, x_txt.shape[2])),
    )


def _setup_context(ctx, inputs, output):
    x_txt, x_img, scale, _, _ = inputs
    ctx.mark_non_differentiable(output[3])
    ctx.save_for_backward(x_txt, x_img, scale, output[3])


def _backward(ctx, g_r_txt, g_r_img, g_y, _g_stats):
    x_txt, x_img, scale, stats = ctx.saved_tensors
    g_r_txt = torch.zeros_like(x_txt) if g_r_txt is None else g_r_txt
    g_r_img = torch.zeros_like(x_img) if g_r_img is None else g_r_img
    dx_txt, dx_img, dss = adaln_single_backward(g_y, g_r_txt, g_r_img, x_txt, x_img, scale, stats)
    return dx_txt, dx_img, dss[:, 1], dss[:, 0], None


adaln_single.register_autograd(_backward, setup_context=_setup_context)


# --- the single-stream block -------------------------------------------------------------------


def make_forward(orig):
    """``mlp_gelu.py``'s single-stream forward with cat + LayerNorm + modulation as ``adaln_single``;
    ``orig`` serves the calls it does not support."""
    from primus.backends.automodel.models.flux import mlp_gelu

    @functools.wraps(orig)
    def forward(
        self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None
    ):
        act = self.act_mlp
        norm = self.norm
        if not mlp_gelu.fusable(
            hidden_states, self.proj_mlp, isinstance(act, torch.nn.GELU) and act.approximate == "tanh"
        ):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        emb = norm.linear(norm.silu(temb))
        shift, scale, gate = emb.chunk(3, dim=1)
        if not supported(encoder_hidden_states, hidden_states, scale, shift):
            return orig(
                self, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs
            )
        text_seq_len = encoder_hidden_states.shape[1]
        r_txt, r_img, norm_hidden_states, _ = adaln_single(
            encoder_hidden_states, hidden_states, scale, shift, norm.norm.eps
        )
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **(joint_attention_kwargs or {}),
        )
        batch, seq, dim = norm_hidden_states.shape
        concat, _ = mlp_gelu.gelu_linear(
            norm_hidden_states.reshape(batch * seq, dim),
            self.proj_mlp.weight,
            self.proj_mlp.bias,
            attn_output.reshape(batch * seq, -1),
        )
        gated = gate.unsqueeze(1) * self.proj_out(concat.view(batch, seq, -1))
        # diffusers' residual + gated, split into the text and image tokens, on the two residual halves.
        return r_txt + gated[:, :text_seq_len], r_img + gated[:, text_seq_len:]

    return forward


def install() -> bool:
    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
    )

    from primus.backends.automodel.models.flux import block_outputs, mlp_gelu

    if getattr(FluxSingleTransformerBlock, _PATCHED_ATTR, False):
        return True
    orig = FluxSingleTransformerBlock.forward
    diffusers_forward = inspect.unwrap(orig)
    digest = hashlib.sha256(inspect.getsource(diffusers_forward).encode()).hexdigest()
    if not mlp_gelu.installed() or orig is diffusers_forward or digest != mlp_gelu.SINGLE_FORWARD_SHA256:
        logger.warning(
            "[PrimusFluxAdaLN] needs mlp_gelu.py's single-stream forward (primus_perf.flux_fused_gelu) "
            "on diffusers 0.40.0's; not applied."
        )
        return False
    FluxSingleTransformerBlock.forward = make_forward(orig)
    # Read by block_outputs.py: this forward returns two fresh tensors, nothing to clone.
    setattr(FluxSingleTransformerBlock, block_outputs.FRESH_OUTPUTS_ATTR, True)
    setattr(FluxSingleTransformerBlock, _PATCHED_ATTR, True)
    logger.info(
        "[PrimusFluxAdaLN] single-stream blocks run cat + LayerNorm + modulation as one FlyDSL op each way."
    )
    return True
