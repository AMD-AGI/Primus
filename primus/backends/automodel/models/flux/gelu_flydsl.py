###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""FlyDSL kernel for ``mlp_gelu.py``'s op: the MLP's GELU after the up-projection GEMM.

WHY:
  ``primus_flux::gelu_linear`` runs the up-projection as ``torch.addmm`` (TunableOp's pick). Unfused,
  Inductor's GELU kernel would then read the pre-activation back, and the single-stream block would
  copy the attention output and the GELU output into one buffer for ``proj_out``. This kernel does
  both in one pass: it writes ``cat([prefix, gelu_tanh(h)])``, the attention half copied and the MLP
  half through the GELU, straight into the buffer ``proj_out`` reads.

  It evaluates Inductor's ``gelu`` expression operation for operation. One block of 384 threads
  covers 3,072 columns, 8 per thread. Written for FlyDSL 0.2.4, the version Primus-Turbo pins.
"""
# No ``from __future__ import annotations``: FlyDSL reads the jit function's annotations as objects.
import torch

VEC, THREADS = 8, 384
TILE = VEC * THREADS  # 3072 columns per block step
FWD_GRID, FWD_ROWS = 1024, 2
_K = {}


def supported(m: int, prefix_width: int, n: int) -> bool:
    # Each of the FWD_GRID blocks walks its rows FWD_ROWS at a time.
    return n % TILE == 0 and prefix_width % TILE == 0 and m % (FWD_GRID * FWD_ROWS) == 0


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import flydsl.expr.math as fmath
    from flydsl.expr.arith import ArithValue
    from flydsl.expr.buffer_ops import buffer_load, buffer_store, create_buffer_resource

    def consts(f32v, *vals):
        return [fx.arith.constant_vector(v, f32v) for v in vals]

    def tanh(v, f32v, one, two, two_log2e):
        # fmath.tanh lowers to llvm.tanh, which the AMDGPU backend cannot compile without the device
        # math library. 1 - 2 / (1 + e^(2v)) on the hardware exp2 and reciprocal (v_exp_f32, v_rcp_f32,
        # ~1 ulp each) saturates cleanly at +-1, and its FP32 error is far below the BF16 rounding that
        # follows.
        e = fmath.exp2(fx.arith.mulf(v, two_log2e))
        r = fx.arith.divf(one, fx.arith.addf(one, e), fastmath=fx.FastMathFlags.afn)
        return fx.arith.subf(one, fx.arith.mulf(two, r))

    def make_fwd(m, p, n, aux_stride):
        width, rows_per_block = p + n, m // FWD_GRID
        copy_steps, gelu_steps = p // TILE, n // TILE

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def fwd(AUX: fx.Tensor, PRE: fx.Tensor, OUT: fx.Tensor):
            f32v = fx.T.VectorType.get([VEC], fx.T.f32())
            bf16v = fx.T.VectorType.get([VEC], fx.T.bf16())
            bf16 = fx.T.bf16()
            col = fx.thread_idx.x * fx.Int32(VEC)
            bx, _, _ = fx.block_idx
            aux_r, out_r = (create_buffer_resource(x, max_size=True) for x in (AUX, OUT))
            pre_r = create_buffer_resource(PRE, max_size=True) if p else None
            half, c3, k, one, two, two_log2e = consts(
                f32v, 0.5, 0.044715, 0.7978845608028654, 1.0, 2.0, 2.8853900817779268
            )

            def rows(it):
                # FWD_ROWS rows per iteration, every load issued before any math.
                rs = [bx + (ArithValue(it) * FWD_ROWS + j) * FWD_GRID for j in range(FWD_ROWS)]
                pres = [
                    [
                        buffer_load(
                            pre_r, r * fx.Int32(p) + fx.Int32(s * TILE) + col, vec_width=VEC, dtype=bf16
                        )
                        for s in range(copy_steps)
                    ]
                    for r in rs
                ]
                hs = [
                    [
                        buffer_load(
                            aux_r,
                            r * fx.Int32(aux_stride) + fx.Int32(s * TILE) + col,
                            vec_width=VEC,
                            dtype=bf16,
                        )
                        for s in range(gelu_steps)
                    ]
                    for r in rs
                ]
                for r, pr, hr in zip(rs, pres, hs):
                    for s, v in enumerate(pr):
                        buffer_store(v, out_r, r * fx.Int32(width) + fx.Int32(s * TILE) + col)
                    for s, hb in enumerate(hr):
                        h = fx.arith.extf(f32v, hb)
                        # Inductor's gelu(approximate="tanh"): (h * 0.5) * (tanh((h + h*h*h*0.044715) * k) + 1).
                        inner = fx.arith.mulf(
                            fx.arith.addf(h, fx.arith.mulf(fx.arith.mulf(fx.arith.mulf(h, h), h), c3)), k
                        )
                        y = fx.arith.mulf(
                            fx.arith.mulf(h, half), fx.arith.addf(tanh(inner, f32v, one, two, two_log2e), one)
                        )
                        buffer_store(
                            fx.arith.trunc_f(bf16v, y),
                            out_r,
                            r * fx.Int32(width) + fx.Int32(p + s * TILE) + col,
                        )

            for it in range(fx.Int32(rows_per_block // FWD_ROWS)):
                rows(it)

        return fwd

    @flyc.jit
    def launch_fwd(
        AUX,
        PRE,
        OUT,
        m: fx.Constexpr[int],
        p: fx.Constexpr[int],
        n: fx.Constexpr[int],
        aux_stride: fx.Constexpr[int],
        stream: fx.Stream,
    ):
        make_fwd(m, p, n, aux_stride)(AUX, PRE, OUT).launch(
            grid=(FWD_GRID, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return launch_fwd


def _kernels():
    if not _K:
        _K["fwd"] = _build()
    return _K["fwd"]


def gelu_into_cat(aux: torch.Tensor, prefix) -> torch.Tensor:
    """cat([prefix, gelu_tanh(aux)], dim=1) for 2-D BF16 aux with unit column stride (its rows may be a
    column slice of a wider buffer) and contiguous prefix."""
    m, n = aux.shape
    if aux.stride(1) != 1:
        aux = aux.contiguous()
    p = 0 if prefix is None else prefix.shape[1]
    out = torch.empty((m, p + n), device=aux.device, dtype=aux.dtype)
    _kernels()(
        aux,
        aux if prefix is None else prefix.contiguous(),
        out,
        m=m,
        p=p,
        n=n,
        aux_stride=aux.stride(0),
        stream=torch.cuda.current_stream(),
    )
    return out
