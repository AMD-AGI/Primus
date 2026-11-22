###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Matrix-core loop: the compute clock read from the shader-cycle counter.

A FlyDSL kernel in which every wave loads one A and one B fragment, then
issues back-to-back bf16 WMMAs on them into independent accumulators -- not a
GEMM, just matrix-core work with no memory traffic in the loop.  Lane 0 of
each wave reads the shader-cycle counter and the constant-rate real-time
counter around the loop.  Two clocks follow from one run:

* **cycle counter**: shader cycles per real-time tick, times the real-time
  counter's rate, itself measured against host time over the kernel's span.
  Needs no hardware constant.
* **matrix-core rate**: the achieved FLOP rate over ``CUs x FLOPs/CU/clock``
  (see :func:`~primus.core.projection.hardware_calibration.compute_clock.mfma_bound_clock_mhz`),
  i.e. the cycle-counter clock times the loop's matrix-core utilisation.

Runs only where FlyDSL is installed and the part has its 16x16x32 bf16 WMMA.
No ``from __future__ import annotations`` here: FlyDSL reads the kernel's
``Constexpr`` annotations at runtime.
"""

import contextlib
import functools
import importlib.util
import statistics
import time
from typing import Any, Dict, List, Optional

from primus.core.projection.hardware_calibration.device_info import SETTLE_TIMEOUT_S

_WMMA_LOOP_ARCHS = ("gfx1250",)

WAVE_SIZE = 32
WAVES_PER_CU = 16
_N_ACC = 8  # independent accumulators per wave, enough to cover the WMMA latency
_WMMA_M, _WMMA_N, _WMMA_K = 16, 16, 32
FLOPS_PER_WMMA = 2 * _WMMA_M * _WMMA_N * _WMMA_K

_TARGET_KERNEL_MS = 100.0
_PROBE_ITERS = 2000


def matrix_loop_unavailable_reason(gcn_arch: Optional[str]) -> Optional[str]:
    """Why the loop cannot run on *gcn_arch*, or *None* if it can."""
    if gcn_arch not in _WMMA_LOOP_ARCHS:
        return f"no matrix-core loop for {gcn_arch}"
    if importlib.util.find_spec("flydsl") is None:
        return "flydsl is not installed"
    return None


@functools.lru_cache(maxsize=None)
def _build_launcher():
    """The FlyDSL launcher, built on first use so importing this module never needs FlyDSL."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import range_constexpr
    from flydsl.expr.arith import _to_raw
    from flydsl.expr.typing import T
    from flydsl.expr.typing import Vector as Vec

    def rsrc(arg):
        return fx.rocdl.get_buffer_rsrc(fx.rocdl.make_buffer_ptr(fx.recast_iter(fx.Int8, fx.get_iter(arg))))

    def aux():
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

    def load_b128(res, byte_off):
        v4i32 = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
        return fx.rocdl.raw_ptr_buffer_load(
            v4i32, res, _to_raw(fx.Int32(byte_off)), _to_raw(fx.Int32(0)), aux=aux()
        )

    def store(value, res, byte_off):
        fx.rocdl.raw_ptr_buffer_store(
            _to_raw(value), res, _to_raw(fx.Int32(byte_off)), _to_raw(fx.Int32(0)), aux=aux()
        )

    def counter(intrinsic):
        return fx.Int64(llvm.call_intrinsic(T.i64, intrinsic, [], [], []))

    def fragment(res, byte_off):
        lo = Vec(load_b128(res, byte_off))
        frag = fx.make_rmem_tensor(8, fx.Int32)
        frag.store(lo.shuffle(Vec(load_b128(res, byte_off + fx.Int32(16))), list(range(8))))
        return frag

    @flyc.jit
    def launch(
        arg_src: fx.Tensor,
        arg_out: fx.Tensor,
        arg_times: fx.Tensor,
        n_iter: fx.Int32,
        grid: fx.Constexpr[int],
        block: fx.Constexpr[int],
        stream: fx.Stream,
    ):
        @flyc.kernel(name="primus_calib_matrix_core_loop_bf16", known_block_size=[block, 1, 1])
        def kernel(arg_src: fx.Tensor, arg_out: fx.Tensor, arg_times: fx.Tensor, n_iter: fx.Int32):
            tid = fx.Int32(fx.thread_idx.x)
            gtid = fx.Int32(fx.block_idx.x) * fx.Int32(block) + tid
            lane = tid % fx.Int32(WAVE_SIZE)
            src, out, times = rsrc(arg_src), rsrc(arg_out), rsrc(arg_times)
            a = fragment(src, lane * fx.Int32(32))
            b = fragment(src, lane * fx.Int32(32) + fx.Int32(WAVE_SIZE * 32))
            atom = fx.make_mma_atom(fx.rocdl.WMMA(_WMMA_M, _WMMA_N, _WMMA_K, fx.BFloat16, fx.Float32))
            acc = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(_N_ACC)]
            for c in acc:
                c.store(Vec.filled(8, 0.0, fx.Float32))
            cyc0 = counter("llvm.readcyclecounter")
            rt0 = counter("llvm.readsteadycounter")
            for _ in range(n_iter):
                for c in acc:
                    fx.gemm(atom, c, a, b, c)
            cyc1 = counter("llvm.readcyclecounter")
            rt1 = counter("llvm.readsteadycounter")
            # The accumulators are stored so the loop is not dead code.
            total = acc[0].load()
            for c in acc[1:]:
                total = total + c.load()
            bits = total.bitcast(fx.Int32)
            store(bits.shuffle(bits, [0, 1, 2, 3]), out, gtid * fx.Int32(16))
            if lane == fx.Int32(0):
                row = (gtid // fx.Int32(WAVE_SIZE)) * fx.Int32(32)
                store(cyc0, times, row)
                store(cyc1, times, row + fx.Int32(8))
                store(rt0, times, row + fx.Int32(16))
                store(rt1, times, row + fx.Int32(24))

        kernel(arg_src, arg_out, arg_times, n_iter).launch(
            grid=(grid, 1, 1), block=(block, 1, 1), stream=stream
        )

    return launch


def _clock_from_counters(times, kernel_ms: float) -> Dict[str, float]:
    """Cycle-counter clock from one launch's per-wave ``[cyc0, cyc1, rt0, rt1]`` rows."""
    t = times.view(-1, 4).double()
    cycles, ticks = t[:, 1] - t[:, 0], t[:, 3] - t[:, 2]
    # One workgroup per CU: every wave is resident from launch, so the
    # earliest start to the latest end spans the kernel.
    span_ticks = (t[:, 3].max() - t[:, 2].min()).item()
    realtime_hz = span_ticks / (kernel_ms / 1e3)
    per_wave_mhz = cycles / ticks * realtime_hz / 1e6
    return {
        "clock_mhz": per_wave_mhz.median().item(),
        "min_wave_mhz": per_wave_mhz.min().item(),
        "max_wave_mhz": per_wave_mhz.max().item(),
        "realtime_counter_mhz": realtime_hz / 1e6,
    }


def _settle_on_counter(launch_clock_mhz, min_s: float, max_s: float, window: int = 3, rel_tol: float = 0.01):
    """Relaunch until the last *window* launches' cycle-counter clocks agree within *rel_tol*.

    The counter is read in the kernel, so unlike the driver's clock metrics
    it does not lag the clock it measures.  Same result keys as
    :func:`run_until_settled`.
    """
    start = time.monotonic()
    clocks: List[float] = []
    while True:
        clocks.append(launch_clock_mhz())
        elapsed = time.monotonic() - start
        recent = clocks[-window:]
        steady = len(recent) == window and max(recent) - min(recent) <= rel_tol * statistics.mean(recent)
        if (elapsed >= min_s and steady) or elapsed >= max_s:
            return {
                "signal": "cycle-counter",
                "settled": elapsed >= min_s and steady,
                "settle_s": elapsed,
                "settled_mhz": recent[-1],
            }


def run_matrix_core_loop(
    device,
    num_cus: int,
    sampler=None,
    min_settle_s: float = 1.0,
    trials: int = 5,
    target_kernel_ms: float = _TARGET_KERNEL_MS,
) -> Dict[str, Any]:
    """Run the loop on *device* until its clock settles, then time *trials* launches.

    Returns the median trial: kernel time, achieved TFLOPS, the bf16 FLOPs
    issued (for the matrix-core-rate clock), and the cycle-counter clock with
    its per-wave spread.  When a :class:`ClockSampler` is given, the timed
    launches run inside its window.
    """
    import torch

    launch = _build_launcher()
    block = WAVES_PER_CU * WAVE_SIZE
    grid = num_cus
    waves = grid * WAVES_PER_CU
    src = torch.randn(2 * WAVE_SIZE * 16, dtype=torch.bfloat16, device=device)
    out = torch.empty(grid * block * 4, dtype=torch.float32, device=device)
    times = torch.zeros(waves * 4, dtype=torch.int64, device=device)
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(n_iter: int) -> float:
        start.record(stream)
        launch(src, out, times, n_iter, grid, block, stream=stream)
        end.record(stream)
        end.synchronize()
        return start.elapsed_time(end)

    timed(_PROBE_ITERS)  # compiles
    probe_ms = timed(_PROBE_ITERS)
    n_iter = max(int(_PROBE_ITERS * target_kernel_ms / max(probe_ms, 1e-3)), _PROBE_ITERS)
    settle = _settle_on_counter(
        lambda: _clock_from_counters(times, timed(n_iter))["clock_mhz"], min_settle_s, SETTLE_TIMEOUT_S
    )

    rows = []
    with sampler.window() if sampler is not None else contextlib.nullcontext():
        for _ in range(trials):
            ms = timed(n_iter)
            rows.append({"time_ms": ms, **_clock_from_counters(times, ms)})
    row = sorted(rows, key=lambda r: r["clock_mhz"])[len(rows) // 2]
    flops = float(waves) * n_iter * _N_ACC * FLOPS_PER_WMMA
    return {
        "dtype": "bf16",
        "waves_per_cu": WAVES_PER_CU,
        "iterations": n_iter,
        "flops": flops,
        "achieved_tflops": flops / (row["time_ms"] / 1e3) / 1e12,
        "settle": settle,
        "trials": trials,
        **row,
    }
