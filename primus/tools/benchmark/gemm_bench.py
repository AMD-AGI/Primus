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

# Try to import torchao for FP8 support
try:
    from torchao.float8 import Float8LinearConfig
    from torchao.float8.float8_linear import matmul_with_hp_or_float8_args
    from torchao.float8.float8_training_tensor import LinearMMConfig, ScaledMMConfig

    TORCHAO_AVAILABLE = True

    # Default config for FP8 GEMM (use_fast_accum=False for fp32 accumulation)
    DEFAULT_MM_CONFIG = LinearMMConfig(
        ScaledMMConfig(emulate=False, use_fast_accum=False, fp8_output=False, pad_inner_dim=False),
        ScaledMMConfig(emulate=False, use_fast_accum=False, fp8_output=False, pad_inner_dim=False),
        ScaledMMConfig(emulate=False, use_fast_accum=False, fp8_output=False, pad_inner_dim=False),
    )
    DEFAULT_CONFIG = Float8LinearConfig()
except ImportError:
    TORCHAO_AVAILABLE = False


def check_fp8_matmul_support(dtype):
    """
    Check if FP8 matmul is actually supported in the current PyTorch/backend.

    Checks two paths:
    1. torchao FP8 support (recommended, more robust)
    2. Native PyTorch FP8 matmul (fallback)

    Returns:
        tuple: (supported: bool, method: str) where method is 'torchao', 'native', or 'none'
    """
    if not hasattr(torch, "float8_e4m3fn"):
        return False, "none"

    if dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return True, "native"  # Not FP8, always supported

    # Try torchao first (recommended)
    if TORCHAO_AVAILABLE:
        try:
            device = get_current_device()
            # Test using the same pattern as turbo: compile and run
            torch._dynamo.reset()
            torch._functorch.config.donated_buffer = False
            test_fn = torch.compile(
                lambda a, b: matmul_with_hp_or_float8_args.apply(a, b.t(), DEFAULT_MM_CONFIG, DEFAULT_CONFIG)
            )
            a = torch.randn(2, 2, device=device, dtype=torch.bfloat16)
            b = torch.randn(2, 2, device=device, dtype=torch.bfloat16)
            _ = test_fn(a, b)
            return True, "torchao"
        except (NotImplementedError, RuntimeError, Exception):
            pass  # Fall through to native check

    # Try native PyTorch FP8 matmul (fallback)
    try:
        device = get_current_device()
        a = torch.randn(2, 2, device=device, dtype=torch.bfloat16).to(dtype)
        b = torch.randn(2, 2, device=device, dtype=torch.bfloat16).to(dtype)
        c = torch.empty(2, 2, device=device, dtype=torch.bfloat16).to(dtype)
        torch.matmul(a, b, out=c)
        return True, "native"
    except (NotImplementedError, RuntimeError):
        return False, "none"


def _gemm_fp8_impl(a, b, trans_b=True):
    """FP8 GEMM implementation using torchao (raw function for compilation)."""
    if trans_b:
        return matmul_with_hp_or_float8_args.apply(a, b.t(), DEFAULT_MM_CONFIG, DEFAULT_CONFIG)
    else:
        return matmul_with_hp_or_float8_args.apply(a, b, DEFAULT_MM_CONFIG, DEFAULT_CONFIG)


# Compiled FP8 GEMM function (initialized on first use)
_compiled_gemm_fp8 = None


def get_compiled_gemm_fp8():
    """Get a compiled FP8 GEMM function (following turbo's pattern)."""
    global _compiled_gemm_fp8
    if _compiled_gemm_fp8 is None:
        torch._dynamo.reset()
        torch._functorch.config.donated_buffer = False
        _compiled_gemm_fp8 = torch.compile(_gemm_fp8_impl)
    return _compiled_gemm_fp8


def maybe_transpose(tensor, transpose):
    return tensor.t() if transpose else tensor


def profile_gemm(m, n, k, dtype, trans_a, trans_b, duration_s=10.0):
    # Supported dtypes: FP32, FP16, BF16, and FP8 (if available)
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    # Add FP8 types if available (PyTorch >= 2.1.0)
    if hasattr(torch, "float8_e4m3fn"):
        supported_dtypes.append(torch.float8_e4m3fn)
    if hasattr(torch, "float8_e5m2"):
        supported_dtypes.append(torch.float8_e5m2)

    assert dtype in supported_dtypes, f"Unsupported dtype: {dtype}. Supported: {supported_dtypes}"

    # Check if FP8 matmul is actually supported (not just the type exists)
    fp8_supported, fp8_method = check_fp8_matmul_support(dtype)
    is_fp8 = hasattr(torch, "float8_e4m3fn") and dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    if is_fp8 and not fp8_supported:
        error_msg = (
            f"FP8 dtype {dtype} is defined in PyTorch but matmul operations are not implemented.\n"
            f"This usually means:\n"
            f"  1. Your PyTorch version supports FP8 types but not FP8 kernels\n"
            f"  2. Your GPU/driver doesn't support FP8 operations\n"
            f"  3. You need a newer PyTorch build with FP8 kernel support\n"
            f"\n"
            f"Recommendations:\n"
            f"  • Install torchao: pip install torchao (recommended for FP8)\n"
            f"  • Or use --dtype bf16 / --dtype fp16 instead\n"
            f"\n"
            f"torchao available: {TORCHAO_AVAILABLE}"
        )
        raise RuntimeError(error_msg)

    # Log which FP8 method is being used
    if is_fp8 and fp8_supported:
        method_name = {"torchao": "torchao (recommended)", "native": "native PyTorch"}.get(
            fp8_method, fp8_method
        )
        print(f"[INFO] Using FP8 method: {method_name}")

    device = get_current_device()
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    mem_size_bytes = (m * k + k * n + m * n) * dtype_size
    num_rotations = max(2, math.ceil(CACHE_ROTATING_BUFFER_BYTES / max(1, mem_size_bytes)) + 1)
    # num_run = 100

    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (n, k) if trans_b else (k, n)

    # Create tensors based on FP8 method
    use_torchao = is_fp8 and fp8_method == "torchao"

    if use_torchao:
        # torchao: use BF16 tensors, conversion handled internally
        # Get compiled FP8 function
        compiled_fp8_fn = get_compiled_gemm_fp8()
        init_dtype = torch.bfloat16
        a_list = [torch.randn(a_shape, device=device, dtype=init_dtype) for _ in range(num_rotations)]
        b_list = [torch.randn(b_shape, device=device, dtype=init_dtype) for _ in range(num_rotations)]
        c_list = None  # torchao allocates output
    elif is_fp8:
        # Native FP8: requires explicit conversion from BF16
        init_dtype = torch.bfloat16
        a_list = [
            torch.randn(a_shape, device=device, dtype=init_dtype).to(dtype) for _ in range(num_rotations)
        ]
        b_list = [
            torch.randn(b_shape, device=device, dtype=init_dtype).to(dtype) for _ in range(num_rotations)
        ]
        # Output tensor must also be FP8 when using out= parameter with FP8 inputs
        c_list = [
            torch.empty((m, n), device=device, dtype=init_dtype).to(dtype) for _ in range(num_rotations)
        ]
    else:
        # Standard dtypes (FP16/BF16/FP32)
        a_list = [torch.randn(a_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
        b_list = [torch.randn(b_shape, device=device, dtype=dtype) for _ in range(num_rotations)]
        c_list = [torch.empty((m, n), device=device, dtype=dtype) for _ in range(num_rotations)]

    # Warm-up
    if use_torchao:
        # torchao path: use compiled function
        for i in range(num_rotations):
            a = maybe_transpose(a_list[i], trans_a)
            b = maybe_transpose(b_list[i], trans_b)
            _ = compiled_fp8_fn(a, b, trans_b=trans_b)
        torch.cuda.synchronize()
    else:
        # Native path: standard matmul with inference_mode for performance
        with torch.inference_mode():
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

    if use_torchao:
        # torchao path
        while True:
            for i in range(num_rotations):
                a = maybe_transpose(a_list[i], trans_a)
                b = maybe_transpose(b_list[i], trans_b)
                _ = compiled_fp8_fn(a, b, trans_b=trans_b)
            end.record()
            torch.cuda.synchronize()

            total_calls += num_rotations

            elapsed = start.elapsed_time(end)  # ms
            if elapsed >= target_ms:
                avg_time_ms = elapsed / total_calls
                break
    else:
        # Native path
        with torch.inference_mode():
            while True:
                for i in range(num_rotations):
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

    # Add FP8 type if available (PyTorch >= 2.1.0)
    # Default to E4M3 format (better for training, widely used)
    if hasattr(torch, "float8_e4m3fn"):
        dtype_map["fp8"] = torch.float8_e4m3fn

    if args.dtype not in dtype_map:
        available = ", ".join(dtype_map.keys())
        raise ValueError(
            f"dtype '{args.dtype}' not available in current PyTorch version. " f"Available types: {available}"
        )

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
