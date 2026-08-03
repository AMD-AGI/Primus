#!/usr/bin/env python3
"""Screen raw FP8 GEMM backends on the FLUX forward/dgrad/wgrad shapes."""

import argparse
import csv
from pathlib import Path

import torch
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import GEMMFP8KernelDispatcher
from primus_turbo.pytorch.ops.quantization import quantize_fp8

# FLUX profiler records these as (tokens, input_features, output_features).
FLUX_SHAPES = (
    (16384, 3072, 9216),
    (16384, 3072, 3072),
    (16384, 3072, 12288),
    (16384, 12288, 3072),
    (32768, 3072, 21504),
    (32768, 15360, 3072),
)
BACKENDS = ("HIPBLASLT", "CK", "TRITON", "FLYDSL")


def synchronize_time(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def operands(pass_name, m, n, k):
    if pass_name == "forward":
        return (m, k), (n, k), float8_e4m3, float8_e4m3, False, True
    if pass_name == "dgrad":
        return (m, n), (n, k), float8_e5m2, float8_e4m3, False, False
    return (m, n), (m, k), float8_e5m2, float8_e4m3, True, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--shape", type=int, action="append", choices=range(len(FLUX_SHAPES)))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is required")
    selected_shapes = args.shape if args.shape is not None else range(len(FLUX_SHAPES))
    available = [
        name
        for name in BACKENDS
        if hasattr(BackendType, name) and BackendType[name] in GEMMFP8KernelDispatcher._backends
    ]
    rows = []

    for shape_id in selected_shapes:
        m, k, n = FLUX_SHAPES[shape_id]
        for pass_name in ("forward", "dgrad", "wgrad"):
            a_shape, b_shape, a_dtype, b_dtype, trans_a, trans_b = operands(pass_name, m, n, k)
            torch.manual_seed(shape_id)
            a = torch.randn(a_shape, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(b_shape, device="cuda", dtype=torch.bfloat16)
            a_fp8, a_scale_inv = quantize_fp8(a, a_dtype, ScalingGranularity.TENSORWISE)
            b_fp8, b_scale_inv = quantize_fp8(b, b_dtype, ScalingGranularity.TENSORWISE)
            # TorchAO normalizes every scaled_mm to row-major A and column-major B.
            logical_a = (a_fp8.t() if trans_a else a_fp8).contiguous()
            logical_b = b_fp8.t() if trans_b else b_fp8
            if logical_b.stride()[0] > logical_b.stride()[1]:
                logical_b = logical_b.t().contiguous().t()

            def torchao_control():
                return torch._scaled_mm(
                    logical_a,
                    logical_b,
                    scale_a=a_scale_inv,
                    scale_b=b_scale_inv,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=False,
                )

            reference = torchao_control()
            control_ms = synchronize_time(torchao_control, args.warmup, args.iterations)
            candidates = [("TORCH_SCALED_MM", None), *((name, BackendType[name]) for name in available)]
            for backend_name, backend in candidates:
                try:
                    if backend is None:
                        output, latency_ms = reference, control_ms
                    else:
                        implementation = GEMMFP8KernelDispatcher._backends[backend].impl
                        backend_args = {
                            "a": a_fp8,
                            "a_scale_inv": a_scale_inv,
                            "trans_a": trans_a,
                            "b": b_fp8,
                            "b_scale_inv": b_scale_inv,
                            "trans_b": trans_b,
                            "out_dtype": torch.bfloat16,
                            "trans_c": False,
                            "granularity": ScalingGranularity.TENSORWISE,
                        }
                        if not implementation.can_handle(**backend_args):
                            raise RuntimeError("backend does not support this dtype/layout/shape")

                        def run_backend():
                            return implementation.execute(**backend_args)

                        output = run_backend()
                        latency_ms = synchronize_time(run_backend, args.warmup, args.iterations)
                    error = (output.float() - reference.float()).norm() / reference.float().norm()
                    row = {
                        "shape_id": shape_id,
                        "pass": pass_name,
                        "m": output.shape[0],
                        "n": output.shape[1],
                        "k": logical_a.shape[1],
                        "layout": f"{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}",
                        "backend": backend_name,
                        "latency_ms": latency_ms,
                        "tflops": 2 * output.shape[0] * output.shape[1] * logical_a.shape[1] / latency_ms / 1e9,
                        "relative_l2": error.item(),
                        "status": "ok",
                    }
                except Exception as exc:
                    row = {
                        "shape_id": shape_id,
                        "pass": pass_name,
                        "m": "",
                        "n": "",
                        "k": "",
                        "layout": f"{'T' if trans_a else 'N'}{'T' if trans_b else 'N'}",
                        "backend": backend_name,
                        "latency_ms": "",
                        "tflops": "",
                        "relative_l2": "",
                        "status": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
                print(row, flush=True)
            del a, b, a_fp8, b_fp8, reference
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
