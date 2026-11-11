###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FlyDSL backward for ``primus_flux::qk_norm_rope`` (see ``qk_norm_rope.py``).

WHY:
  The Triton backward kernel stays below the GPU's copy bandwidth at the FLUX training shapes:
  each program walks its tokens one at a time, so too little memory traffic is in flight. This
  kernel moves the same bytes with every load of a token in flight together.

WHAT:
  One block of ``heads * 16`` threads per token covers all heads; thread t owns 8 columns of
  one head (16 lanes x 8 per 128-wide head), so RoPE's (2i, 2i+1) pairs sit inside one lane's
  vector and only the two row sums cross lanes, as a 16-lane butterfly. 256 blocks stride over
  the tokens. Per token a thread loads its q, k (pre-norm, from the QKV output), the q, k, v
  gradients and its cos/sin row, and stores the QKV output's gradient for q, k (RMSNorm + RoPE
  backward) and v (copied). Norm-weight and bias gradients accumulate per thread in registers
  and are written once per block, then reduced like the Triton op's partial sums.

  Written for FlyDSL 0.2.4, the version Primus-Turbo pins, so it runs in the training image as
  is. Arithmetic and roundings follow the Triton kernel; the RMS and dot-product sums add in a
  different order, so a few QKV-gradient elements differ by one BF16 step.

Settings (module config, ``primus_perf:`` section):
    flux_fused_qk_norm_rope_flydsl_bwd: false      use this backward (needs flux_fused_qk_norm_rope)
"""
# No ``from __future__ import annotations``: FlyDSL reads the jit function's annotations
# (fx.Constexpr, fx.Stream) as objects, and ``fx`` is local to _build().
import torch

VEC, LANES, D = 8, 16, 128
GRID_X = 256
_FLYDSL = None


def supported(qkv, txt_qkv, cos, heads) -> bool:
    """Whether this backward serves the call; the Triton kernel takes the rest."""
    if cos.dtype != torch.bfloat16 or heads * LANES > 1024 or qkv.shape[-1] != 3 * heads * D:
        return False
    segments = [qkv] if txt_qkv is None else [txt_qkv, qkv]
    return all((s.shape[0] * s.shape[1]) % GRID_X == 0 for s in segments)


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import flydsl.expr.math as fmath
    from flydsl._mlir.dialects import vector as _vector
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource

    def make(H, n_tok, s_src, s_out, seq_off, eps, bias, src_stride):
        threads = H * LANES
        hd, row_w = H * D, 3 * H * D
        n_iter = n_tok // GRID_X

        @flyc.kernel(known_block_size=[threads, 1, 1])
        def bwd(
            SRC: fx.Tensor,
            DSRC: fx.Tensor,
            DQ: fx.Tensor,
            DK: fx.Tensor,
            DV: fx.Tensor,
            WQ: fx.Tensor,
            WK: fx.Tensor,
            COS: fx.Tensor,
            SIN: fx.Tensor,
            PART: fx.Tensor,
            BPART: fx.Tensor,
        ):
            f32v = fx.T.VectorType.get([VEC], fx.T.f32())
            f32v4 = fx.T.VectorType.get([4], fx.T.f32())
            bf16v = fx.T.VectorType.get([VEC], fx.T.bf16())
            bf16 = fx.T.bf16()
            t = fx.thread_idx.x
            bx, _, _ = fx.block_idx
            lane8 = (t % fx.Int32(LANES)) * fx.Int32(VEC)
            col = t * fx.Int32(VEC)
            src_r, dsrc_r = (create_buffer_resource(x, max_size=True) for x in (SRC, DSRC))
            dq_r, dk_r, dv_r = (create_buffer_resource(x, max_size=True) for x in (DQ, DK, DV))
            cos_r, sin_r = (create_buffer_resource(x, max_size=True) for x in (COS, SIN))
            part_r, bpart_r = (create_buffer_resource(x, max_size=True) for x in (PART, BPART))
            inv_d = fx.arith.constant_vector(1.0 / D, f32v)
            eps_v = fx.arith.constant_vector(eps, f32v)
            zero = fx.arith.constant_vector(0.0, f32v)
            neg1 = fx.arith.constant_vector(-1.0, f32v4)
            wq, wk = (
                fx.arith.extf(
                    f32v,
                    buffer_load(create_buffer_resource(w, max_size=True), lane8, vec_width=VEC, dtype=bf16),
                )
                for w in (WQ, WK)
            )
            accs = [fx.make_rmem_tensor(fx.make_layout(VEC, 1), fx.Float32) for _ in range(5 if bias else 2)]
            for a in accs:
                fx.memref_store_vec(zero, a)

            def row_sum(v):
                s = ArithValue(_vector.reduction(fx.T.f32(), _vector.CombiningKind.ADD, v))
                for o in (1, 2, 4, 8):
                    s = s.addf(ArithValue(s.shuffle_xor(o, 64)))
                return _vector.broadcast(f32v, s)

            def norm_rope_bwd(x_bf, go_bf, w, c, s_signed):
                # c: the cos row; s_signed: (sin_e, -sin_o) interleaved. The tables repeat each value
                # across a (2i, 2i+1) pair. Using every loaded element keeps LLVM from narrowing the
                # load to 7 BF16, which the AMDGPU backend cannot lower.
                x = fx.arith.extf(f32v, x_bf)
                go = fx.arith.extf(f32v, go_bf)
                rstd = fmath.rsqrt(fx.arith.addf(fx.arith.mulf(row_sum(fx.arith.mulf(x, x)), inv_d), eps_v))
                xhat = fx.arith.mulf(x, rstd)
                ge, g_o = _vector.deinterleave(go)
                # Transpose of RoPE: even = g_e c + g_o s, odd = g_o c - g_e s.
                rot = fx.arith.addf(
                    fx.arith.mulf(go, c), fx.arith.mulf(_vector.interleave(g_o, ge), s_signed)
                )
                # Gradient of the BF16 norm output, rounded as autograd rounds it.
                dy = fx.arith.extf(f32v, fx.arith.trunc_f(bf16v, rot))
                g = fx.arith.mulf(dy, w)
                dot = fx.arith.mulf(row_sum(fx.arith.mulf(g, xhat)), inv_d)
                dx = fx.arith.trunc_f(bf16v, fx.arith.mulf(rstd, fx.arith.subf(g, fx.arith.mulf(xhat, dot))))
                return dx, fx.arith.mulf(dy, xhat)

            def accumulate(acc, v):
                fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(acc), v), acc)

            def body(it):
                tok = bx + ArithValue(it) * GRID_X
                b = tok // fx.Int32(s_src)
                pos = tok % fx.Int32(s_src) + fx.Int32(seq_off)
                src_off = tok * fx.Int32(src_stride) + col
                go_off = (b * fx.Int32(s_out) + pos) * fx.Int32(hd) + col
                cs_off = pos * fx.Int32(D) + lane8
                xq = buffer_load(src_r, src_off, vec_width=VEC, dtype=bf16)
                xk = buffer_load(src_r, src_off + fx.Int32(hd), vec_width=VEC, dtype=bf16)
                gq = buffer_load(dq_r, go_off, vec_width=VEC, dtype=bf16)
                gk = buffer_load(dk_r, go_off, vec_width=VEC, dtype=bf16)
                dv = buffer_load(dv_r, go_off, vec_width=VEC, dtype=bf16)
                c = fx.arith.extf(f32v, buffer_load(cos_r, cs_off, vec_width=VEC, dtype=bf16))
                s_e, s_o = _vector.deinterleave(
                    fx.arith.extf(f32v, buffer_load(sin_r, cs_off, vec_width=VEC, dtype=bf16))
                )
                s_signed = _vector.interleave(s_e, fx.arith.mulf(s_o, neg1))
                dxq, dwq = norm_rope_bwd(xq, gq, wq, c, s_signed)
                dxk, dwk = norm_rope_bwd(xk, gk, wk, c, s_signed)
                buffer_store(dxq, dsrc_r, src_off)
                buffer_store(dxk, dsrc_r, src_off + fx.Int32(hd))
                buffer_store(dv, dsrc_r, src_off + fx.Int32(2 * hd))
                accumulate(accs[0], dwq)
                accumulate(accs[1], dwk)
                if bias:
                    accumulate(accs[2], fx.arith.extf(f32v, dxq))
                    accumulate(accs[3], fx.arith.extf(f32v, dxk))
                    accumulate(accs[4], fx.arith.extf(f32v, dv))

            for it in range(fx.Int32(n_iter)):
                body(it)

            def store8(v, rsrc, off):
                for half in (0, 1):
                    buffer_store(
                        _vector.extract_strided_slice(f32v4, v, [half * 4], [4], [1]),
                        rsrc,
                        off + fx.Int32(half * 4),
                    )

            # PART [GRID_X, H, 2, D], so the reduction afterwards is one column sum over GRID_X * H rows.
            pbase = (bx * fx.Int32(H) + t // fx.Int32(LANES)) * fx.Int32(2 * D) + lane8
            store8(fx.memref_load_vec(accs[0]), part_r, pbase)
            store8(fx.memref_load_vec(accs[1]), part_r, pbase + fx.Int32(D))
            if bias:
                # BPART [GRID_X, 3 * H * D]: this block's column sums of the QKV output's gradient.
                bbase = bx * fx.Int32(row_w) + col
                for i, a in enumerate(accs[2:]):
                    store8(fx.memref_load_vec(a), bpart_r, bbase + fx.Int32(i * hd))

        return bwd

    @flyc.jit
    def launch(
        SRC,
        DSRC,
        DQ,
        DK,
        DV,
        WQ,
        WK,
        COS,
        SIN,
        PART,
        BPART,
        H: fx.Constexpr[int],
        n_tok: fx.Constexpr[int],
        s_src: fx.Constexpr[int],
        s_out: fx.Constexpr[int],
        seq_off: fx.Constexpr[int],
        eps: fx.Constexpr[float],
        bias: fx.Constexpr[int],
        src_stride: fx.Constexpr[int],
        stream: fx.Stream,
    ):
        make(H, n_tok, s_src, s_out, seq_off, eps, bool(bias), src_stride)(
            SRC, DSRC, DQ, DK, DV, WQ, WK, COS, SIN, PART, BPART
        ).launch(grid=(GRID_X, 1, 1), block=(H * LANES, 1, 1), stream=stream)

    return launch


def backward_segment(dq, dk, dv, src, dsrc, wq, wk, cos, sin, seq_off, eps, heads, bias, colsum):
    """One stream's backward: writes the gradient of ``src`` (that stream's QKV GEMM output, [B, S, 3 H D])
    into ``dsrc`` and returns (dw_q, dw_k, bias gradient or None). ``src`` and ``dsrc`` have unit column
    stride and the same token stride, which may exceed 3 H D (column slices of a wider GEMM output)."""
    global _FLYDSL
    if _FLYDSL is None:
        _FLYDSL = _build()
    part = torch.empty((GRID_X * heads, 2, D), device=src.device, dtype=torch.float32)
    bpart = torch.empty((GRID_X, src.shape[-1]) if bias else (1,), device=src.device, dtype=torch.float32)
    _FLYDSL(
        src,
        dsrc,
        dq,
        dk,
        dv,
        wq.contiguous(),
        wk.contiguous(),
        cos,
        sin,
        part,
        bpart,
        H=heads,
        n_tok=src.shape[0] * src.shape[1],
        s_src=src.shape[1],
        s_out=dq.shape[1],
        seq_off=seq_off,
        eps=float(eps),
        bias=int(bias),
        src_stride=src.stride(1),
        stream=torch.cuda.current_stream(),
    )
    dw = part.sum(0)
    db = None
    if bias:
        n_cols = src.shape[-1]
        db = torch.empty(n_cols, device=src.device, dtype=src.dtype)
        colsum[(-(-n_cols // 64),)](bpart, db, GRID_X, n_cols, BLOCK_R=32, BLOCK_C=64, num_warps=4)
    return dw[0].to(wq.dtype), dw[1].to(wk.dtype), db


def backward(dq, dk, dv, qkv, txt_qkv, w_q, w_k, w_txt_q, w_txt_k, cos, sin, eps, heads, bias, colsum):
    """Same outputs as ``primus_flux::qk_norm_rope_backward``; ``colsum`` is the Triton op's column-sum
    kernel, which reduces the bias partials."""
    dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    if txt_qkv is None:
        segments, weights = [(qkv.contiguous(), 0)], [(w_q, w_k)]
    else:
        segments = [(txt_qkv.contiguous(), 0), (qkv.contiguous(), txt_qkv.shape[1])]
        weights = [(w_txt_q, w_txt_k), (w_q, w_k)]
    grads, wgrads, bgrads = [], [], []
    for (src, off), (wq, wk) in zip(segments, weights):
        dsrc = torch.empty_like(src)
        dwq, dwk, db = backward_segment(
            dq, dk, dv, src, dsrc, wq, wk, cos, sin, off, eps, heads, bias, colsum
        )
        grads.append(dsrc)
        wgrads.extend([dwq, dwk])
        if bias:
            bgrads.append(db)
    if txt_qkv is None:
        return [grads[0], wgrads[0], wgrads[1]] + bgrads
    return [grads[1], grads[0], wgrads[2], wgrads[3], wgrads[0], wgrads[1]] + bgrads[::-1]
