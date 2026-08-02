#!/usr/bin/env python3
"""Gate 0 MXFP8 eager/compile probe for one MI355X GPU."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import torch


def version_info(provider: str) -> dict[str, str]:
    import triton

    info = {
        "provider": provider,
        "host": platform.node(),
        "torch": torch.__version__,
        "hip": str(torch.version.hip),
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": str(torch.cuda.get_device_capability()),
    }
    try:
        import torchao

        info["torchao"] = getattr(torchao, "__version__", "unknown")
        info["torchao_path"] = torchao.__file__
    except Exception as exc:
        info["torchao_error"] = repr(exc)
    if provider == "alto":
        info["alto_path"] = os.environ.get("ALTO_PATH", "unknown")
    return info


def make_op(
    provider: str,
    torchao_dim0: str,
    torchao_dim1: str,
    torchao_scale: str,
    torchao_kernel: str,
):
    if provider == "bf16":
        return torch.nn.functional.linear

    if provider == "torchao":
        from torchao.prototype.moe_training.mxfp8_linear import mx_mm
        from torchao.prototype.mx_formats.config import (
            MXFP8Dim0CastKernelChoice,
            MXFP8Dim1CastKernelChoice,
            ScaleCalculationMode,
        )
        from torchao.quantization.quantize_.common.kernel_preference import (
            KernelPreference,
        )

        dim0_choice = {
            "triton": MXFP8Dim0CastKernelChoice.TRITON,
            "torch": MXFP8Dim0CastKernelChoice.TORCH,
        }[torchao_dim0]
        dim1_choice = {
            "cuda": MXFP8Dim1CastKernelChoice.CUDA,
            "flydsl": MXFP8Dim1CastKernelChoice.FLYDSL,
        }[torchao_dim1]
        scale_mode = {
            "floor": ScaleCalculationMode.FLOOR,
            "rceil": ScaleCalculationMode.RCEIL,
        }[torchao_scale]
        kernel_preference = {
            "auto": KernelPreference.AUTO,
            "emulated": KernelPreference.EMULATED,
        }[torchao_kernel]

        def op(x, weight, bias):
            out = mx_mm.apply(
                x,
                weight,
                torch.float8_e4m3fn,
                torch.float8_e4m3fn,
                torch.float8_e4m3fn,
                32,
                kernel_preference,
                dim0_choice,
                dim1_choice,
                scale_mode,
                False,
            )
            return out + bias

        return op

    from alto.kernels.mxfp8.mxfp8_linear import MXFP8LinearFunction
    from alto.kernels.mxfp8.mxfp8_quantization import is_cdna4

    if not is_cdna4():
        raise RuntimeError("ALTO did not detect the active Triton target as HIP gfx950")

    def op(x, weight, bias):
        return MXFP8LinearFunction.apply(x, weight, "e4m3", False, False, True) + bias

    return op


def errors(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    reference_float = reference.float()
    delta = (actual_float - reference_float).abs()
    denominator = reference_float.abs().clamp_min(1e-6)
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "max_rel": (delta / denominator).max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_float)
        ).item(),
    }


def run_case(
    op, shape: tuple[int, int, int], compile_op: bool, iterations: int
) -> dict:
    m, k, n = shape
    torch.manual_seed(1234)
    x0 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w0 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    b0 = torch.randn(n, device="cuda", dtype=torch.bfloat16)

    x_ref, w_ref, b_ref = [
        value.detach().clone().requires_grad_() for value in (x0, w0, b0)
    ]
    y_ref = torch.nn.functional.linear(x_ref, w_ref, b_ref)
    y_ref.float().sum().backward()

    x, weight, bias = [
        value.detach().clone().requires_grad_() for value in (x0, w0, b0)
    ]
    call = torch.compile(op, fullgraph=True) if compile_op else op

    def step():
        x.grad = weight.grad = bias.grad = None
        output = call(x, weight, bias)
        output.float().sum().backward()
        return output

    for _ in range(2):
        output = step()
    torch.cuda.synchronize()

    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = step()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000)

    tensors = (output, x.grad, weight.grad, bias.grad)
    if not all(torch.isfinite(value).all().item() for value in tensors):
        raise RuntimeError(f"non-finite output or gradient for shape={shape}")

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as prof:
        step()
        torch.cuda.synchronize()
    kernel_names = sorted(
        {
            item.key
            for item in prof.key_averages()
            if any(token in item.key.lower() for token in ("mxfp", "scaled", "triton"))
        }
    )

    return {
        "shape_mkn": shape,
        "compiled": compile_op,
        "latency_ms": {
            "min": min(elapsed),
            "mean": sum(elapsed) / len(elapsed),
            "max": max(elapsed),
        },
        "output_error": errors(output, y_ref),
        "input_grad_error": errors(x.grad, x_ref.grad),
        "weight_grad_error": errors(weight.grad, w_ref.grad),
        "bias_grad_error": errors(bias.grad, b_ref.grad),
        "profile_ops": kernel_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=("bf16", "torchao", "alto"), required=True
    )
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--torchao-dim0", choices=("triton", "torch"), default="triton")
    parser.add_argument("--torchao-dim1", choices=("cuda", "flydsl"), default="cuda")
    parser.add_argument("--torchao-scale", choices=("floor", "rceil"), default="rceil")
    parser.add_argument("--torchao-kernel", choices=("auto", "emulated"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    shapes = args.shape or ["256,3072,3072", "256,3072,9216", "256,3072,12288"]
    parsed_shapes = [tuple(map(int, value.split(","))) for value in shapes]
    if any(len(shape) != 3 for shape in parsed_shapes):
        raise ValueError("each --shape must be M,K,N")

    report = {
        "environment": version_info(args.provider),
        "config": {
            "torchao_dim0": args.torchao_dim0,
            "torchao_dim1": args.torchao_dim1,
            "torchao_scale": args.torchao_scale,
            "torchao_kernel": args.torchao_kernel,
        },
        "results": [],
    }
    try:
        op = make_op(
            args.provider,
            args.torchao_dim0,
            args.torchao_dim1,
            args.torchao_scale,
            args.torchao_kernel,
        )
        for shape in parsed_shapes:
            for compile_op in (False, True):
                try:
                    result = run_case(op, shape, compile_op, args.iterations)
                except Exception as exc:
                    result = {
                        "shape_mkn": shape,
                        "compiled": compile_op,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                report["results"].append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    except Exception as exc:
        report["setup_error"] = f"{type(exc).__name__}: {exc}"

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
