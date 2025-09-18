###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


# from datetime import time
import math
import os
import time
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from git import List

from primus.tools.report import write_table_simple
from primus.tools.utils import (
    derive_path,
    gather_hostnames,
    gather_times,
    get_hostname,
    is_rank_0,
    parse_bytes,
    pick_dtype,
    round_up_div,
)

SUPPORTED_COLLECTIVES = {
    "all_reduce": dist.all_reduce,
    "broadcast": dist.broadcast,
    "reduce_scatter": dist.reduce_scatter,
    "all_gather": dist.all_gather,
    "alltoall": dist.all_to_all,
}


def add_rccl_parser(parser):
    """
    Register RCCL benchmark arguments to the CLI parser.
    """
    parser.add_argument(
        "--op",
        nargs="+",
        default=["allreduce"],
        choices=SUPPORTED_COLLECTIVES.keys(),
        help="Collectives to run",
    )
    # sizes
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Explicit list of sizes, e.g. '1K,2K,4K,8K,1M'. Overrides min/max/num.",
    )
    parser.add_argument("--min-bytes", type=str, default="1K", help="Minimum message size (e.g. 1K / 1M)")
    parser.add_argument("--max-bytes", type=str, default="128M", help="Maximum message size")
    parser.add_argument("--num-sizes", type=int, default=12, help="Number of sizes if generated")
    parser.add_argument(
        "--scale",
        type=str,
        choices=["log2", "linear"],
        default="log2",
        help="Sweep scale when generating sizes",
    )

    # run
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Enable lightweight correctness checks")
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Path to save results (.md/.csv/.tsv/.jsonl[.gz]). If not set or '-', print to stdout (Markdown).",
    )
    parser.add_argument("--append", action="store_true", help="Append to file instead of overwrite.")

    # Optional per-rank summary (one row per {op,size,rank})
    parser.add_argument(
        "--per-rank", action="store_true", help="Emit per-rank summary stats (one line per {op,size,rank})."
    )
    parser.add_argument(
        "--per-rank-file",
        type=str,
        default="",
        help="Output path for per-rank stats. If empty, will derive from --output-file with suffix '_rank'.",
    )

    # Optional per-iteration trace (heavy; one row per {op,size,rank,iter})
    parser.add_argument(
        "--per-iter-trace",
        action="store_true",
        help="Emit per-iteration trace (large). Use filters to limit.",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default="",
        help="Output path for per-iteration trace (JSONL/CSV/TSV). If empty, derive from --output-file with suffix '_trace'.",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=0,
        help="Max iters to record per {op,size}. 0 = all (careful: large).",
    )
    parser.add_argument(
        "--trace-ops",
        type=str,
        default="",
        help="Comma-separated ops to trace (e.g., 'alltoall,allreduce'). Empty = all.",
    )
    parser.add_argument(
        "--trace-sizes",
        type=str,
        default="",
        help="Comma-separated sizes to trace (e.g., '1M,16M'). Empty = all.",
    )

    parser.set_defaults(func=run_rccl_benchmark)

    return parser


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/1024:.1f}{unit}"
        n /= 1024


def algo_bytes_factor(op: str, world_size: int) -> float:
    """
    Per-rank effective bytes factor (commonly used in NCCL/RCCL benchmarks).
    """
    p = world_size
    if op == "allreduce":
        return 2.0 * (p - 1) / p
    elif op in ("allgather", "reduce_scatter", "broadcast", "alltoall"):
        return (p - 1) / p
    else:
        raise ValueError(f"Unknown op: {op}")


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1


# ----------------------- Tensor Builders -----------------------


def _ensure_numel_aligned(numel: int, world_size: int) -> int:
    """
    Some collectives require sizes divisible by world_size (e.g., reduce_scatter / all_to_all).
    """
    return round_up_div(numel, world_size) * world_size


def build_tensor(num_bytes: int, dtype: torch.dtype, world_size: int, op: str) -> Dict[str, torch.Tensor]:
    """
    Allocate tensors needed by a given collective.
    Returns a dict with keys: 'in', 'out' (as needed).
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()

    if op in ("allreduce", "broadcast"):
        numel = max(1, num_bytes // elem_size)
        t = torch.ones(numel, dtype=dtype, device="cuda")
        return {"in": t}

    elif op == "allgather":
        # out needs world_size * in.numel
        in_numel = max(1, num_bytes // elem_size)
        _in = torch.ones(in_numel, dtype=dtype, device="cuda")
        out = torch.empty(in_numel * world_size, dtype=dtype, device="cuda")
        return {"in": _in, "out": out}

    elif op == "reduce_scatter":
        # in needs world_size * out.numel
        out_numel = max(1, num_bytes // elem_size)
        out_numel = round_up_div(out_numel, 1)  # ensure >=1
        in_numel = out_numel * world_size
        _in = torch.ones(in_numel, dtype=dtype, device="cuda")
        out = torch.empty(out_numel, dtype=dtype, device="cuda")
        return {"in": _in, "out": out}

    elif op == "alltoall":
        # Equal split: in/out must be divisible by world_size
        in_numel = max(1, num_bytes // elem_size)
        in_numel = _ensure_numel_aligned(in_numel, world_size)
        out_numel = in_numel  # same total count
        _in = torch.ones(in_numel, dtype=dtype, device="cuda")
        out = torch.empty(out_numel, dtype=dtype, device="cuda")
        return {"in": _in, "out": out}

    else:
        raise ValueError(f"Unsupported op: {op}")


# ----------------------- Timed Ops -----------------------


def do_collective(op: str, tensors: Dict[str, torch.Tensor], world_size: int):
    if op == "allreduce":
        dist.all_reduce(tensors["in"], op=dist.ReduceOp.SUM)
    elif op == "broadcast":
        dist.broadcast(tensors["in"], src=0)
    elif op == "allgather":
        # Prefer into-tensor variant (contiguous & faster)
        dist.all_gather_into_tensor(tensors["out"], tensors["in"])
    elif op == "reduce_scatter":
        dist.reduce_scatter_tensor(tensors["out"], tensors["in"], op=dist.ReduceOp.SUM)
    elif op == "alltoall":
        in_t = tensors["in"]
        out_t = tensors["out"]
        # equal splits
        split = in_t.numel() // world_size
        dist.all_to_all_single(
            out_t, in_t, out_split_sizes=[split] * world_size, in_split_sizes=[split] * world_size
        )
    else:
        raise ValueError(f"Unknown op: {op}")


def check_correctness(op: str, tensors: Dict[str, torch.Tensor], world_size: int) -> bool:
    """
    Light-weight correctness check on small slices to avoid OOM.
    """
    with torch.no_grad():
        if op == "allreduce":
            # Ones reduced by SUM
            ref = torch.full_like(tensors["in"], fill_value=float(world_size))
            return torch.allclose(tensors["in"], ref, rtol=1e-2, atol=1e-2)
        elif op == "broadcast":
            # After bcast, all ranks equal to src's tensor (initialized as ones)
            ref = torch.ones_like(tensors["in"])
            return torch.allclose(tensors["in"], ref, rtol=1e-4, atol=1e-4)
        elif op == "allgather":
            # out is concatenation of 'ones' from every rank
            # We only check a few elements
            return torch.allclose(
                tensors["out"][:16], torch.ones_like(tensors["out"][:16]), rtol=1e-4, atol=1e-4
            )
        elif op == "reduce_scatter":
            # SUM of ones, then split => each element equals world_size
            ref = torch.full_like(tensors["out"], fill_value=float(world_size))
            return torch.allclose(tensors["out"], ref, rtol=1e-2, atol=1e-2)
        elif op == "alltoall":
            # ones -> ones (just permuted), so still ones
            return torch.allclose(
                tensors["out"][:16], torch.ones_like(tensors["out"][:16]), rtol=1e-4, atol=1e-4
            )
        else:
            return True


def timed_run(
    op: str, num_bytes: int, dtype: torch.dtype, warmup: int, iters: int, world_size: int
) -> Tuple[List[float], Dict[str, torch.Tensor]]:
    """
    Returns per-iteration latency(ms) for *this rank* and the last tensors for optional correctness check.
    """
    tensors = build_tensor(num_bytes, dtype, world_size, op)
    torch.cuda.synchronize()
    dist.barrier()

    # Warmup
    for _ in range(warmup):
        do_collective(op, tensors, world_size)
    torch.cuda.synchronize()
    dist.barrier()

    times_ms: List[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        do_collective(op, tensors, world_size)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times_ms.append((t1 - t0) / 1e6)
    dist.barrier()
    return times_ms, tensors


def make_size_list(args) -> List[int]:
    sizes: List[int] = []
    if args.sizes:
        # explicit list like: 1K,2K,4K,8K,... or integers
        for tok in args.sizes.split(","):
            sizes.append(parse_bytes(tok))
        return sizes

    # or generate range [min_bytes, max_bytes] with num_sizes in log2 or linear scale
    min_b = parse_bytes(args.min_bytes)
    max_b = parse_bytes(args.max_bytes)
    if args.scale == "log2":
        # near powers of two
        v = min_b
        while v <= max_b:
            sizes.append(v)
            v = max(v + 1, v * 2)
    else:
        step = max(1, (max_b - min_b) // max(1, args.num_sizes - 1))
        sizes = [min_b + i * step for i in range(args.num_sizes)]
    return sizes


def compute_rankmax_per_iter(all_ranks_times: List[List[float]]) -> List[float]:
    """
    Derive per-iteration latency as max across ranks (critical path).
    """
    if not all_ranks_times:
        return []
    iters = len(all_ranks_times[0])
    # safety: align lengths
    for r in all_ranks_times:
        if len(r) != iters:
            iters = min(iters, len(r))
    rankmax = []
    for i in range(iters):
        rankmax.append(max(r[i] for r in all_ranks_times))
    return rankmax


def per_rank_stats(gathered: List[List[float]]) -> List[Tuple[int, float, float, float, float]]:
    """
    Compute per-rank stats: (rank, p50, p95, min, max) from per-iter latencies.
    """
    out = []
    for r, arr in enumerate(gathered):
        if not arr:
            continue
        p50 = percentile(arr, 50.0)
        p95 = percentile(arr, 95.0)
        out.append((r, p50, p95, min(arr), max(arr)))
    return out


def pick_topk_slowest_by_p95(
    rank_stats: List[Tuple[int, float, float, float, float]],
    k: int,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Sort ranks by p95 latency (desc) and return top-k slowest.
    Each item: (rank, p50, p95, min, max)
    """
    if k <= 0 or not rank_stats:
        return []
    return sorted(rank_stats, key=lambda x: x[2], reverse=True)[:k]


def run_rccl_benchmark(args):
    """
    RCCL benchmark main entry.
    Produces:
      - Summary (default): one row per {op,size} with critical-path stats
      - Optional Per-Rank summary: one row per {op,size,rank}
      - Optional Per-Iter trace: one row per {op,size,rank,iter}
    Output controlled by --output-file (stdout if empty or "-") and --append.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    int(os.environ.get("RANK", "0"))
    hostname = get_hostname()
    dtype = pick_dtype(args.dtype)
    ops = [op.strip().lower() for op in args.op]
    sizes = make_size_list(args)

    if is_rank_0():
        print(
            f"[RCCL-BENCH] host={hostname} world_size={world} dtype={dtype} "
            f"ops={ops} sizes={[human_bytes(s) for s in sizes]}"
        )

        # print a few relevant envs (for reproducibility)
        tracked_envs = [
            "HIP_VISIBLE_DEVICES",
            "NCCL_DEBUG",
            "RCCL_P2P_ENABLE",
            "RCCL_ENABLE_SMI_SUPPORT",
            "RCCL_IB_AR_THRESHOLD",
            "RCCL_NET_GDR_LEVEL",
            "RCCL_NCCL_BUFFER_MEM",
            "RCCL_TOPO_FILE",
            "RCCL_MSCCL_ENABLE",
            "RCCL_MSCCLPP_ENABLE",
            "RCCL_MSCCLPP_THRESHOLD",
        ]
        for k in tracked_envs:
            v = os.environ.get(k)
            if v is not None:
                print(f"[RCCL-BENCH][env] {k}={v}")

    # Derive trace filters
    def _parse_sizes_list(s: str) -> set[int]:
        if not s:
            return set()
        return {parse_bytes(tok.strip()) for tok in s.split(",") if tok.strip()}

    trace_ops = (
        set([o.strip().lower() for o in args.trace_ops.split(",") if o.strip()])
        if getattr(args, "per_iter_trace", False)
        else set()
    )
    trace_sizes = _parse_sizes_list(args.trace_sizes) if getattr(args, "per_iter_trace", False) else set()

    # Prepare headers
    summary_header = [
        "host",
        "world",
        "suite",
        "op",
        "bytes",
        "dtype",
        "p50_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "eff_gbps",
        "slowest_rank",
        "rank_p95_spread_ms",
    ]
    per_rank_header = [
        "host",
        "world",
        "suite",
        "op",
        "bytes",
        "dtype",
        "rank",
        "p50_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "rel_p95_to_global",
    ]
    trace_header = ["host", "world", "suite", "op", "bytes", "dtype", "rank", "iter", "latency_ms"]

    # Collectors on rank0
    all_rows = []  # summary rows
    per_rank_rows = []  # per-rank rows (optional)
    trace_rows = []  # per-iter trace rows (optional)

    # Gather rank->hostname map once (all ranks participate; rank0 collects)
    hostnames_rank0 = []
    if is_rank_0():
        hostnames_rank0 = gather_hostnames()
    else:
        # still participate to avoid hang
        gather_hostnames()

    # Main benchmark loops
    for op in ops:
        for num_bytes in sizes:
            # Run timed collective on this rank
            times_local_ms, tensors = timed_run(
                op=op,
                num_bytes=num_bytes,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                world_size=world,
            )

            # Optional correctness check
            if getattr(args, "check", False):
                ok = check_correctness(op, tensors, world)
                if not ok and is_rank_0():
                    print(f"[WARN] Correctness check failed: op={op} size={human_bytes(num_bytes)}")

            # Gather per-rank per-iter latencies to rank0
            gathered = gather_times(times_local_ms)

            # Rank0 computes and records stats
            if is_rank_0() and gathered:
                # Critical-path per-iter latency = max across ranks per iteration
                rankmax = compute_rankmax_per_iter(gathered)
                p50 = percentile(rankmax, 50.0)
                p95 = percentile(rankmax, 95.0)
                tmin = min(rankmax) if rankmax else float("nan")
                tmax = max(rankmax) if rankmax else float("nan")

                # Effective bandwidth (per-rank normalized)
                factor = algo_bytes_factor(op, world)
                eff_gbps = (num_bytes * factor) / (p50 / 1000.0) / 1e9  # /s to GB/s

                # Per-rank stats to derive slowest rank and p95 spread (even if --per-rank is off)
                pr_stats = per_rank_stats(gathered)  # [(rank, p50, p95, min, max), ...]
                if pr_stats:
                    p95s = [p95 for (_, _, p95, _, _) in pr_stats]
                    # spread = max(p95) - median(p95)
                    spread_ms = (max(p95s) - percentile(p95s, 50.0)) if p95s else float("nan")

                    # top-1 slowest by p95 (or use args.topk if you prefer)
                    k = getattr(args, "topk", 1) or 1
                    topk = pick_topk_slowest_by_p95(pr_stats, k)
                    if topk:
                        slowest_rk = topk[0][0]
                        slowest_host = (
                            hostnames_rank0[slowest_rk]
                            if hostnames_rank0 and slowest_rk < len(hostnames_rank0)
                            else ""
                        )
                        slowest_label = f"r{slowest_rk}@{slowest_host}"
                    else:
                        slowest_label, spread_ms = "", float("nan")
                else:
                    slowest_label, spread_ms = "", float("nan")

                # Append summary row
                all_rows.append(
                    [
                        hostname,
                        world,
                        "rccl",
                        op,
                        num_bytes,
                        str(dtype).replace("torch.", ""),
                        f"{p50:.3f}",
                        f"{p95:.3f}",
                        f"{tmin:.3f}",
                        f"{tmax:.3f}",
                        f"{eff_gbps:.2f}",
                        slowest_label,
                        f"{spread_ms:.3f}",
                    ]
                )

                # Pretty print one-liner for quick view
                print(
                    f"[RCCL][{op:<13}] size={human_bytes(num_bytes):>8} "
                    f"p50={p50:6.3f}ms  p95={p95:6.3f}ms  "
                    f"min={tmin:6.3f}ms  max={tmax:6.3f}ms  "
                    f"eff={eff_gbps:6.2f} GB/s"
                )

                # Optional: per-rank summary rows
                if getattr(args, "per_rank", False):
                    pr_stats = per_rank_stats(gathered)  # [(rank, p50, p95, min, max), ...]
                    global_p50 = p50 if p50 and p50 > 0 else float("nan")
                    for rk, rk_p50, rk_p95, rk_min, rk_max in pr_stats:
                        host_r = hostnames_rank0[rk] if hostnames_rank0 and rk < len(hostnames_rank0) else ""
                        rel = (
                            (rk_p95 / global_p50)
                            if (isinstance(global_p50, float) and global_p50 > 0)
                            else float("nan")
                        )
                        per_rank_rows.append(
                            [
                                host_r,
                                world,
                                "rccl",
                                op,
                                num_bytes,
                                str(dtype).replace("torch.", ""),
                                rk,
                                f"{rk_p50:.3f}",
                                f"{rk_p95:.3f}",
                                f"{rk_min:.3f}",
                                f"{rk_max:.3f}",
                                f"{rel:.3f}",
                            ]
                        )

                # Optional: per-iteration trace rows (heavy)
                do_trace = getattr(args, "per_iter_trace", False)
                if (
                    do_trace
                    and (not trace_ops or op in trace_ops)
                    and (not trace_sizes or num_bytes in trace_sizes)
                ):
                    max_iters = len(gathered[0])
                    limit = args.trace_limit if getattr(args, "trace_limit", 0) > 0 else max_iters
                    limit = min(limit, max_iters)
                    for rk, arr in enumerate(gathered):
                        host_r = hostnames_rank0[rk] if hostnames_rank0 and rk < len(hostnames_rank0) else ""
                        for i in range(limit):
                            trace_rows.append(
                                [
                                    host_r,
                                    world,
                                    "rccl",
                                    op,
                                    num_bytes,
                                    str(dtype).replace("torch.", ""),
                                    rk,
                                    i,
                                    f"{arr[i]:.3f}",
                                ]
                            )

    if is_rank_0():
        # Summary
        write_table_simple(args.output_file, all_rows, summary_header, append=getattr(args, "append", False))

        # Per-rank summary
        if getattr(args, "per_rank", False) and per_rank_rows:
            per_rank_path = args.per_rank_file or derive_path(
                args.output_file or "-", "_rank", default_ext=".csv"
            )
            write_table_simple(
                per_rank_path, per_rank_rows, per_rank_header, append=getattr(args, "append", False)
            )

        # Per-iter trace
        if getattr(args, "per_iter_trace", False) and trace_rows:
            # Prefer JSONL by default due to size
            default_trace_ext = ".jsonl"
            # If user gave output-file with known ext, derive sibling; otherwise use default .jsonl
            trace_path = args.trace_file or derive_path(
                args.output_file or "-", "_trace", default_ext=default_trace_ext
            )
            write_table_simple(trace_path, trace_rows, trace_header, append=getattr(args, "append", False))
