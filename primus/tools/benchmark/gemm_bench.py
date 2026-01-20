###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import math
from datetime import datetime

import torch

from primus.tools.report import write_table_simple
from primus.tools.utils import gather_records, get_current_device, is_rank_0

CACHE_ROTATING_BUFFER_BYTES = 2 * 1024 * 1024 * 1024  # 2GB rotating buffer


def maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


@torch.inference_mode()
def profile_gemm(m, n, k, dtype, trans_a, trans_b, duration_s=10.0):
    # Supported dtypes: FP32, FP16, BF16, and FP8 (if available)
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    # Add FP8 types if available (PyTorch >= 2.1.0)
    if hasattr(torch, "float8_e4m3fn"):
        supported_dtypes.append(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        supported_dtypes.append(torch.float8_e5m2)

    assert dtype in supported_dtypes, f"Unsupported dtype: {dtype}. Supported: {supported_dtypes}"

    device = get_current_device()
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = max(2, math.ceil(CACHE_ROTATING_BUFFER_BYTES / max(1, mem_size_bytes)) + 1)
    # num_run = 100

    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    # Check if dtype is FP8 (requires conversion from BF16/FP32)
    is_fp8 = hasattr(torch, "float8_e4m3fn") and dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    if is_fp8:
        # FP8 requires conversion from higher precision types
        init_dtype = torch.bfloat16
        a_list = [
            torch.randn(a_shape, device=device, dtype=init_dtype).to(dtype) for _ in range(num_rotations)
        ]
        b_list = [
            torch.randn(b_shape, device=device, dtype=init_dtype).to(dtype) for _ in range(num_rotations)
        ]
        c_list = [torch.empty((m, n), device=device, dtype=init_dtype) for _ in range(num_rotations)]
    else:
        a_list = [torch.randn(a_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
        b_list = [torch.randn(b_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
        c_list = [torch.empty((m, n), device=device, dtype=dtype) for _ in range(num_rotations)]

    # Warm-up
    for i in range(num_rotations):
        a = maybe_transpose(a_list[i], trans_a)
        b = maybe_transpose(b_list[i], trans_b)
        torch.matmul(a, b, out=c_list[i])
    torch.cuda.synchronize()

    # Timed run (duration-based)
    target_ms = max(100.0, duration_s * 1000.0)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_calls = 0
    start.record()

    while True:
        for _ in range(num_rotations):
            a = maybe_transpose(a_list[i], trans_a)
            b = maybe_transpose(b_list[i], trans_b)
            torch.matmul(a, b, out=c_list[i])
        end.record()
        torch.cuda.synchronize()

        total_calls += num_rotations

        elapsed = start.elapsed_time(end)  # ms
        if elapsed >= target_ms:
            avg_time_ms = elapsed / total_calls
            break

    tflop = 2.0 * m * n * k / 1e12
    tflops = tflop / (avg_time_ms / 1000.0)
    bandwidth = mem_size_bytes / 1e9 / (avg_time_ms / 1000.0)
    arith_intensity = (2.0 * m * n * k) / mem_size_bytes

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
        "arith_intensity": arith_intensity,
    }


def build_gemm_base_preamble(args) -> str:
    lines = [
        "# Base GEMM Benchmark Report",
        "",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Cluster: amd-aig-poolside",
        f"- Benchmark Duration: {args.duration} sec",
        "",
        "## GEMM Configuration",
        f"- M: {args.M}",
        f"- N: {args.N}",
        f"- K: {args.K}",
        f"- Transpose A: {args.trans_a}",
        f"- Transpose B: {args.trans_b}",
        f"- Dtype: {args.dtype}",
        "",
        "## GEMM Shape",
        f"- A: ({args.M}, {args.K})" if not args.trans_a else f"- Aᵗ: ({args.K}, {args.M})",
        f"- B: ({args.K}, {args.N})" if not args.trans_b else f"- Bᵗ: ({args.N}, {args.K})",
        f"- C: ({args.M}, {args.N})",
        "",
        "## Metrics",
        "- `avg_time_ms`: average time per matmul (ms)",
        "- `tflops`: total TFLOPS (1e12 ops/sec)",
        "- `bandwidth_gbps`: estimated memory bandwidth usage (GB/s)",
        "- `arith_intensity`: arithmetic intensity (FLOPs per byte)",
        "",
    ]
    return "\n".join(lines)


def run_gemm_benchmark(args):
    if args.M <= 0 or args.N <= 0 or args.K <= 0:
        raise ValueError("M, N, K must be positive integers.")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    res = profile_gemm(args.M, args.N, args.K, dtype, args.trans_a, args.trans_b, args.duration)

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
        "arith_intensity": float(f"{res['arith_intensity']:.2f}"),
    }

    # Gather results
    gathered = gather_records(record)

    if is_rank_0():
        header = [
            "host",
            "world",
            "rank",
            "avg_time_ms",
            "tflop",
            "tflops",
            "bandwidth_gbps",
            "arith_intensity",
        ]

        # Convert list[dict] -> list[list] in header order
        float6 = {"avg_time_ms"}
        float2 = {"tflop", "tflops", "bandwidth_gbps", "arith_intensity"}

        rows_ll = []
        for rec in gathered:
            row = []
            for col in header:
                v = rec.get(col, "")
                if v is None:
                    v = ""
                elif col in float6:
                    v = f"{float(v):.6f}"
                elif col in float2:
                    v = f"{float(v):.2f}"
                row.append(v)
            rows_ll.append(row)

        preamble = build_gemm_base_preamble(args)
        write_table_simple(
            header=header,
            rows=rows_ll,
            output_file=getattr(args, "output_file", None),
            append=getattr(args, "append", False),
            preamble=preamble if not getattr(args, "append", False) else None,
        )

        print(f"[✔] GEMM benchmark finished. Results saved to {args.output_file}")


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
