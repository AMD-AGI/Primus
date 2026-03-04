###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import time
from datetime import datetime

import torch
import torch.distributed as dist

from primus.tools.report import write_table_simple
from primus.tools.utils import (
    derive_path,
    gather_hostnames,
    gather_times,
    get_hostname,
    get_rank_world,
    pick_dtype,
)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _timeit(fn, warmup: int, iters: int) -> list[float]:
    dist_ready = dist.is_available() and dist.is_initialized()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist_ready:
        dist.barrier()
    times_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)
    if dist_ready:
        dist.barrier()
    return times_ms


def _bench_router(args) -> list[float]:
    dtype = pick_dtype(args.dtype)
    logits = torch.randn((args.tokens, args.num_experts), device="cuda", dtype=dtype)

    def _run():
        _vals, _idx = torch.topk(logits, k=args.topk, dim=1)
        return _vals, _idx

    return _timeit(_run, args.warmup, args.iters)


def _bench_dispatch(args) -> list[float]:
    _rank, world = get_rank_world()
    dtype = pick_dtype(args.dtype)
    numel = max((args.tokens * args.hidden_size) // max(world, 1), max(world, 1))
    # all_to_all_single requires equal splits by default.
    if world > 1:
        numel = (numel // world) * world
    if numel <= 0:
        numel = max(world, 1)
    inp = torch.randn((numel,), device="cuda", dtype=dtype)
    out = torch.empty_like(inp)
    split = numel // max(world, 1)
    dist_ready = dist.is_available() and dist.is_initialized() and world > 1

    def _run():
        if dist_ready:
            dist.all_to_all_single(out, inp, [split] * world, [split] * world)
        else:
            out.copy_(inp)

    return _timeit(_run, args.warmup, args.iters)


def _bench_grouped_gemm(args) -> list[float]:
    dtype = pick_dtype(args.dtype)
    m = args.tokens
    n = args.hidden_size
    k = args.hidden_size
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)

    def _run():
        _ = a @ b

    return _timeit(_run, args.warmup, args.iters)


def run_moe_benchmark(args) -> None:
    rank, world = get_rank_world()
    if not torch.cuda.is_available():
        raise RuntimeError("moe benchmark requires CUDA devices")

    benches = {
        "router": _bench_router,
        "dispatch": _bench_dispatch,
        "grouped-gemm": _bench_grouped_gemm,
    }
    if args.smoke:
        args.warmup = min(args.warmup, 2)
        args.iters = min(args.iters, 5)
        args.repeat = 1
        args.tokens = min(args.tokens, 2048)

    trace_ops = (
        set([o.strip().lower() for o in args.trace_ops.split(",") if o.strip()])
        if args.per_iter_trace
        else set()
    )
    host = get_hostname()
    hostnames_rank0 = gather_hostnames() if rank == 0 else (gather_hostnames() or [])

    rows = []
    per_rank_rows = []
    trace_rows = []

    for op in args.op:
        for rep in range(max(1, int(args.repeat))):
            times_local = benches[op](args)
            gathered = gather_times(times_local)
            if rank != 0:
                continue
            if not gathered:
                continue

            rankmax = [max(xs[i] for xs in gathered) for i in range(min(len(x) for x in gathered))]
            global_p50 = _percentile(rankmax, 50.0)
            global_p95 = _percentile(rankmax, 95.0)
            rows.append(
                [
                    host,
                    world,
                    op,
                    rep + 1,
                    f"{global_p50:.3f}",
                    f"{global_p95:.3f}",
                    args.tokens,
                    args.hidden_size,
                ]
            )

            if args.per_rank:
                for rk, vals in enumerate(gathered):
                    rk_p50 = _percentile(vals, 50.0)
                    rk_p95 = _percentile(vals, 95.0)
                    rk_host = hostnames_rank0[rk] if rk < len(hostnames_rank0) else ""
                    rel = (rk_p95 / global_p50) if global_p50 > 0 else float("nan")
                    per_rank_rows.append(
                        [rk_host, world, op, rep + 1, rk, f"{rk_p50:.3f}", f"{rk_p95:.3f}", f"{rel:.3f}"]
                    )

            if args.per_iter_trace and (not trace_ops or op in trace_ops):
                max_iters = min(len(x) for x in gathered)
                limit = args.trace_limit if args.trace_limit > 0 else max_iters
                limit = min(limit, max_iters)
                for rk, vals in enumerate(gathered):
                    rk_host = hostnames_rank0[rk] if rk < len(hostnames_rank0) else ""
                    for i in range(limit):
                        trace_rows.append([rk_host, world, op, rep + 1, rk, i, f"{vals[i]:.3f}"])

    if rank == 0:
        preamble = "\n".join(
            [
                "# MoE Microbenchmark Report",
                "",
                f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Ops: {', '.join(args.op)}",
                f"- tokens: {args.tokens}",
                f"- hidden_size: {args.hidden_size}",
                f"- num_experts: {args.num_experts}",
                f"- topk: {args.topk}",
                f"- dtype: {args.dtype}",
                f"- warmup/iters/repeat: {args.warmup}/{args.iters}/{args.repeat}",
                f"- smoke: {'on' if args.smoke else 'off'}",
            ]
        )
        write_table_simple(
            output_file=args.output_file,
            rows=rows,
            header=["host", "world", "op", "repeat", "p50_ms", "p95_ms", "tokens", "hidden_size"],
            preamble=preamble if not args.append else None,
            append=args.append,
        )
        if args.per_rank and per_rank_rows:
            per_rank_path = args.per_rank_file or derive_path(args.output_file, "_rank", default_ext=".csv")
            write_table_simple(
                output_file=per_rank_path,
                rows=per_rank_rows,
                header=["host", "world", "op", "repeat", "rank", "p50_ms", "p95_ms", "rel_p95_to_global_p50"],
                append=args.append,
            )
        if args.per_iter_trace and trace_rows:
            trace_path = args.trace_file or derive_path(args.output_file, "_trace", default_ext=".jsonl")
            write_table_simple(
                output_file=trace_path,
                rows=trace_rows,
                header=["host", "world", "op", "repeat", "rank", "iter", "latency_ms"],
                append=args.append,
            )
