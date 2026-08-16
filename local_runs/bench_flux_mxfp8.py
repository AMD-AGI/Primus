#!/usr/bin/env python3
"""Benchmark complete dynamic MXFP8 Linear FWD+BWD on real FLUX shapes."""

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch


SHAPES = {
    "double_qkv": (16384, 3072, 9216),
    "double_proj": (16384, 3072, 3072),
    "double_mlp_up": (16384, 3072, 12288),
    "double_mlp_down": (16384, 12288, 3072),
    "single_up": (32768, 3072, 21504),
    "single_down": (32768, 15360, 3072),
}


def make_linear(provider: str, k: int, n: int, turbo_format: str) -> torch.nn.Module:
    kwargs = {"bias": True, "device": "cuda", "dtype": torch.bfloat16}
    if provider == "bf16":
        return torch.nn.Linear(k, n, **kwargs)
    if provider == "torchao":
        from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear

        return MXFP8Linear(k, n, wgrad_with_hp=False, **kwargs)
    if provider == "primus_turbo":
        from primus_turbo.pytorch.core.low_precision import (
            Float8QuantConfig,
            Format,
            ScaleDtype,
            ScalingGranularity,
        )
        from primus_turbo.pytorch.modules import Float8Linear

        config = Float8QuantConfig(
            format={"e4m3": Format.E4M3, "hybrid": Format.HYBRID}[turbo_format],
            granularity=ScalingGranularity.MX_BLOCKWISE,
            scale_dtype=ScaleDtype.E8M0,
            block_size=32,
        )
        return Float8Linear(k, n, config=config, **kwargs)
    raise ValueError(provider)


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    delta = actual.float() - reference.float()
    return (
        torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.float())
    ).item()


def timed_step(model, x, grad_out):
    model.zero_grad(set_to_none=True)
    x.grad = None
    output = model(x)
    output.backward(grad_out)
    return output


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_case(
    provider: str,
    name: str,
    compile_model: bool,
    warmup: int,
    iterations: int,
    turbo_format: str,
) -> dict:
    m, k, n = SHAPES[name]
    torch.manual_seed(1234)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    x_data = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) / n**0.5

    reference = torch.nn.Linear(k, n, bias=True, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(weight)
        reference.bias.copy_(bias)
    x_ref = x_data.clone().requires_grad_()
    out_ref = timed_step(reference, x_ref, grad_out)

    model = make_linear(provider, k, n, turbo_format)
    with torch.no_grad():
        model.weight.copy_(weight)
        model.bias.copy_(bias)
    x = x_data.clone().requires_grad_()
    if compile_model:
        model = torch.compile(model, fullgraph=True, mode="max-autotune-no-cudagraphs")

    for _ in range(warmup):
        output = timed_step(model, x, grad_out)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    elapsed = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = timed_step(model, x, grad_out)
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))

    tensors = (output, x.grad, model.weight.grad, model.bias.grad)
    if not all(torch.isfinite(value).all().item() for value in tensors):
        raise RuntimeError("non-finite output or gradient")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        timed_step(model, x, grad_out)
        torch.cuda.synchronize()
    profile_ops = sorted(
        item.key
        for item in prof.key_averages()
        if any(
            token in item.key.lower() for token in ("mxfp", "scaled", "flydsl", "quant")
        )
    )

    return {
        "provider": provider,
        "shape": name,
        "mkn": [m, k, n],
        "compiled": compile_model,
        "turbo_format": turbo_format if provider == "primus_turbo" else None,
        "latency_ms": {
            "median": statistics.median(elapsed),
            "p10": percentile(elapsed, 0.1),
            "p90": percentile(elapsed, 0.9),
            "mean": statistics.mean(elapsed),
        },
        "relative_l2": {
            "output": relative_l2(output, out_ref),
            "input_grad": relative_l2(x.grad, x_ref.grad),
            "weight_grad": relative_l2(model.weight.grad, reference.weight.grad),
            "bias_grad": relative_l2(model.bias.grad, reference.bias.grad),
        },
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "profile_ops": profile_ops,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=("bf16", "torchao", "primus_turbo"), required=True
    )
    parser.add_argument("--shape", action="append", choices=tuple(SHAPES), default=[])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--turbo-format", choices=("e4m3", "hybrid"), default="e4m3")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {
        "environment": {
            "host": platform.node(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        },
        "results": [],
    }
    for name in args.shape or list(SHAPES):
        try:
            result = run_case(
                args.provider,
                name,
                args.compile,
                args.warmup,
                args.iterations,
                args.turbo_format,
            )
        except Exception as exc:
            result = {
                "provider": args.provider,
                "shape": name,
                "compiled": args.compile,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["results"].append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
