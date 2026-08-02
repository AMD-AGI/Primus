#!/usr/bin/env python3
"""Compare gfx950 torch._scaled_mm scale layouts against MX emulation."""

import json

import torch

from torchao.prototype.mx_formats.config import (
    MXFP8Dim0CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_blocked
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(actual.float() - reference.float())
        / torch.linalg.vector_norm(reference.float())
    ).item()


def main() -> None:
    m, k, n = 128, 3072, 3072
    torch.manual_seed(1234)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5

    kwargs = {
        "elem_dtype": torch.float8_e4m3fn,
        "block_size": 32,
        "scaling_mode": ScaleCalculationMode.RCEIL,
        "kernel_preference": KernelPreference.AUTO,
        "mxfp8_dim0_cast_kernel_choice": MXFP8Dim0CastKernelChoice.TORCH,
    }
    a_mx = MXTensor.to_mx(a, **kwargs)
    weight_mx = MXTensor.to_mx(weight, **kwargs)
    b_mx = weight_mx.t()
    reference = a_mx.dequantize(torch.bfloat16) @ b_mx.dequantize(torch.bfloat16)

    a_scale = a_mx.scale.view(m, k // 32)
    b_scale = b_mx.scale.t().view(n, k // 32)
    layouts = {
        "blocked": (to_blocked(a_scale), to_blocked(b_scale)),
        "row_major": (a_scale, b_scale),
        "row_major_transposed_b": (a_scale, b_scale.t().contiguous()),
    }
    report = {}
    for name, (scale_a, scale_b) in layouts.items():
        try:
            output = torch._scaled_mm(
                a_mx.qdata,
                b_mx.qdata,
                scale_a.view(torch.float8_e8m0fnu),
                scale_b.view(torch.float8_e8m0fnu),
                out_dtype=torch.bfloat16,
            )
            report[name] = {
                "scale_a_shape": list(scale_a.shape),
                "scale_b_shape": list(scale_b.shape),
                "relative_l2": relative_l2(output, reference),
            }
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
