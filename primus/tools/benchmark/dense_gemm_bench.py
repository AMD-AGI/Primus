###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import math
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from primus.tools.report import write_table_simple
from primus.tools.utils import (
    get_current_device,
    get_hostname,
    get_rank_world,
    is_rank_0,
)

CACHE_ROTATING_BUFFER_BYTES = 2 * 1024 * 1024 * 1024  # 2GB rotating buffer


def add_gemm_parser(parser: argparse.ArgumentParser):
    """
    Register DENSE-GEMM benchmark arguments to the CLI parser.
    """
    parser.add_argument("--M", type=int, default=4096, help="GEMM M dimension (default: 4096)")
    parser.add_argument("--N", type=int, default=4096, help="GEMM N dimension (default: 4096)")
    parser.add_argument("--K", type=int, default=4096, help="GEMM K dimension (default: 4096)")
    parser.add_argument("--trans_a", action="store_true", help="Transpose A matrix")
    parser.add_argument("--trans_b", action="store_true", help="Transpose B matrix")
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Data type for GEMM computation."
    )
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds.")
    parser.add_argument(
        "--output_file",
        default="",
        help="Path to save results (.md/.csv/.tsv/.jsonl[.gz]). If not set or '-', print to stdout (Markdown).",
    )

    return parser


def maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


def profile_gemm(m, n, k, dtype, trans_a, trans_b):
    assert dtype in [torch.float16, torch.bfloat16], f"Unsupported dtype: {dtype}"

    device = get_current_device()

    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = math.ceil(CACHE_ROTATING_BUFFER_BYTES / mem_size_bytes) + 1
    num_run = 100

    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    a_list = [torch.randn(a_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    b_list = [torch.randn(b_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
    c_list = [torch.randn((m, n), device=device, dtype=dtype) for _ in range(num_rotations)]

    # Warm-up
    for i in range(num_rotations):
        a = maybe_transpose(a_list[i], trans_a)
        b = maybe_transpose(b_list[i], trans_b)
        c_list[i] = torch.matmul(a, b)
    torch.cuda.synchronize()

    # Run
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_run):
        for i in range(num_rotations):
            a = maybe_transpose(a_list[i], trans_a)
            b = maybe_transpose(b_list[i], trans_b)
            c_list[i] = torch.matmul(a, b)
    end_event.record()
    torch.cuda.synchronize()

    # result
    avg_time_ms = start_event.elapsed_time(end_event) / (num_rotations * num_run)
    tflop = 2 * m * n * k / 1e12
    tflops = tflop / avg_time_ms * 1000  # Convert to TFlops
    bandwidth = mem_size_bytes / 1e9 / avg_time_ms * 1000  # Convert to GB/s
    return {
        "m": m,
        "n": n,
        "k": k,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "dtype": str(dtype),
        "avg_time_ms": avg_time_ms,
        "tflop": tflop,
        "tflops": tflops,
        "bandwidth_gbps": bandwidth,
    }


def gather_records(record: Dict[str, Any], dst: int = 0) -> Optional[List[Dict[str, Any]]]:
    """
    Gather per-rank dict records to dst (root) rank only.
    - Automatically injects 'host', 'rank', and 'world'.
    - Returns:
        * On root (rank==dst): List[Dict[str, Any]] of all records.
        * On non-root: None  (change to [] if you prefer old behavior).
    - Works even if torch.distributed is not initialized (single-process).
    """
    rank, world = get_rank_world()

    # Inject host/rank/world into a shallow copy to avoid mutating caller's dict
    local = dict(record)
    local["host"] = get_hostname()
    local["rank"] = rank
    local["world"] = world

    if world == 1 or not (dist.is_available() and dist.is_initialized()):
        # Single process: just return a singleton list on "root"
        return [local]

    if rank == dst:
        # Root gathers a list of objects from all ranks
        gathered: List[Optional[Dict[str, Any]]] = [None for _ in range(world)]
        dist.gather_object(local, object_gather_list=gathered, dst=dst)
        # Type narrowing: all entries must be dicts here
        return [g for g in gathered if g is not None]
    else:
        # Non-root sends and returns None to avoid extra work/printing
        dist.gather_object(local, dst=dst)
        return None


def run_gemm_benchmark(args):
    if args.M <= 0 or args.N <= 0 or args.K <= 0:
        raise ValueError("M, N, K must be positive integers.")

    m, n, k = args.M, args.N, args.K
    trans_a, trans_b = args.trans_a, args.trans_b

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    res = profile_gemm(m, n, k, dtype, trans_a, trans_b)

    # Build record with GEMM-specific metrics
    record = {
        "m": res["m"],
        "n": res["n"],
        "k": res["k"],
        "trans_a": int(res["trans_a"]),
        "trans_b": int(res["trans_b"]),
        "dtype": res["dtype"],  # "bf16"/"fp16"/"fp32"
        "avg_time_ms": float(f"{res['avg_time_ms']:.6f}"),
        "tflop": float(f"{res['tflop']:.2f}"),
        "tflops": float(f"{res['tflops']:.2f}"),
        "bandwidth_gbps": float(f"{res['bandwidth_gbps']:.2f}"),
    }

    # Gather results
    gathered = gather_records(record)

    if is_rank_0():

        gemm_summary_header = [
            "host",
            "world",
            "rank",
            "m",
            "n",
            "k",
            "trans_a",
            "trans_b",
            "dtype",
            "avg_time_ms",
            "tflop",
            "tflops",
            "bandwidth_gbps",
        ]

        # Convert list[dict] -> list[list] in header order
        float6 = {"avg_time_ms"}
        float2 = {"tflop", "tflops", "bandwidth_gbps"}

        rows_ll = []
        for rec in gathered:
            row = []
            for col in gemm_summary_header:
                v = rec.get(col, "")
                if v is None:
                    v = ""
                elif col in float6:
                    v = f"{float(v):.6f}"
                elif col in float2:
                    v = f"{float(v):.2f}"
                row.append(v)
            rows_ll.append(row)

        write_table_simple(
            header=gemm_summary_header,
            rows=rows_ll,  # <-- pass list-of-lists, not list-of-dicts
            output_file=getattr(args, "output_file", None),
            append=getattr(args, "append", False),
        )


def build_gemm_parser() -> argparse.ArgumentParser:
    """
    Build a standalone parser for local execution.
    """
    parser = argparse.ArgumentParser(description="GEMM benchmark")
    add_gemm_parser(parser)
    return parser


if __name__ == "__main__":
    parser = build_gemm_parser()
    args = parser.parse_args()
    run_gemm_benchmark(args)
