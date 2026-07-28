# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Validate the @triton_op cast_transpose_fp8_triton kernel.

Tests:
  1. Correctness: FP8 cast + transpose + amax match reference
  2. GEMM interaction: output feeds into torch._scaled_mm without regression
  3. torch.compile: triton_op is traceable (no graph breaks)
  4. Autograd: gradients flow through a minimal FP8 linear autograd.Function

Run:
    python -m pytest tests/unit_tests/backends/megatron/diffusion/test_delayed_fp8_triton_op.py -v
or standalone:
    python tests/unit_tests/backends/megatron/diffusion/test_delayed_fp8_triton_op.py
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)

from primus.backends.megatron.core.extensions.fp8_cast_kernels_triton import (
    cast_transpose_fp8_triton,
)
from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
    DelayedFP8LinearTensorwiseFunction,
)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
FP8_DTYPE = float8_e4m3
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_BWD_DTYPE = float8_e5m2
FP8_BWD_MAX = torch.finfo(FP8_BWD_DTYPE).max


# -----------------------------------------------------------------------
# Test 1: Correctness
# -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (4096, 3072),
        (16384, 3072),
        (8192, 8192),
    ],
)
def test_correctness(shape):
    M, N = shape
    x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(FP8_MAX / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)
    amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

    cast_out, trans_out, scale_inv = cast_transpose_fp8_triton(x, FP8_DTYPE, scale, amax_buf)

    assert cast_out.shape == (M, N), f"Cast shape mismatch: {cast_out.shape}"
    assert cast_out.dtype == FP8_DTYPE
    assert trans_out.shape == (N, M), f"Trans shape mismatch: {trans_out.shape}"
    assert trans_out.dtype == FP8_DTYPE
    assert trans_out.is_contiguous(), "Transpose output must be contiguous"
    assert scale_inv.shape == (), f"Scale inv shape mismatch: {scale_inv.shape}"

    ref_scaled = (x.float() * scale.item()).clamp(-FP8_MAX, FP8_MAX)
    ref_fp8 = ref_scaled.to(FP8_DTYPE)
    assert torch.equal(cast_out, ref_fp8), f"Cast output mismatch for {shape}"

    ref_trans = ref_fp8.t().contiguous()
    assert torch.equal(trans_out, ref_trans), f"Transpose output mismatch for {shape}"

    expected_scale_inv = 1.0 / scale.item()
    assert (
        abs(scale_inv.item() - expected_scale_inv) < 1e-5
    ), f"scale_inv {scale_inv.item()} != {expected_scale_inv}"

    expected_amax = x.float().abs().amax().item()
    assert (
        abs(amax_buf.item() - expected_amax) / max(expected_amax, 1e-8) < 1e-3
    ), f"amax {amax_buf.item()} != {expected_amax}"


# -----------------------------------------------------------------------
# Test 2: GEMM interaction (the critical regression test)
# -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        (16384, 3072),
        (4096, 12288),
    ],
)
def test_gemm_interaction(shape):
    M, N = shape
    x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(FP8_MAX / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)
    amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

    cast_out, _, scale_inv = cast_transpose_fp8_triton(x, FP8_DTYPE, scale, amax_buf)

    w = torch.randn(N, N, dtype=DTYPE, device=DEVICE).to(FP8_DTYPE).t()
    w_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    result = torch._scaled_mm(cast_out, w, out_dtype=DTYPE, scale_a=scale_inv, scale_b=w_scale)
    assert result.shape == (M, N)
    assert result.dtype == DTYPE
    assert not result.isnan().any(), "GEMM produced NaN"
    assert not result.isinf().any(), "GEMM produced Inf"


# -----------------------------------------------------------------------
# Test 3: torch.compile traceability
# -----------------------------------------------------------------------
def test_torch_compile():
    M, N = 4096, 3072
    x = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
    scale = torch.tensor(FP8_MAX / x.abs().amax().item(), dtype=torch.float32, device=DEVICE)
    amax_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)

    @torch.compile(fullgraph=True)
    def fn(x, scale, amax_buf):
        return cast_transpose_fp8_triton(x, FP8_DTYPE, scale, amax_buf)

    cast_out, trans_out, scale_inv = fn(x, scale, amax_buf)

    assert cast_out.shape == (M, N)
    assert trans_out.shape == (N, M)
    assert cast_out.dtype == FP8_DTYPE

    # Verify match with eager
    amax_eager = torch.zeros((), dtype=torch.float32, device=DEVICE)
    cast_eager, trans_eager, si_eager = cast_transpose_fp8_triton(x, FP8_DTYPE, scale, amax_eager)
    assert torch.equal(cast_out, cast_eager), "Compiled output != eager output"
    assert torch.equal(trans_out, trans_eager), "Compiled transpose != eager transpose"


# -----------------------------------------------------------------------
# Test 4: Direct test of the *production* delayed-tensorwise FP8 autograd
# Function (DelayedFP8LinearTensorwiseFunction), exercising both the native
# (force_nt=False -> dgrad=NN/wgrad=TN) and forced-NT (force_nt=True) arms.
#
# This replaces the previous in-file ``_MinimalFP8Linear`` replica: it calls the
# shipped Function directly so a regression in its quantization, GEMM layout, or
# fused amax capture is actually caught.
# -----------------------------------------------------------------------
def _rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.float()
    return ((actual.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("force_nt", [True, False])
def test_delayed_fp8_linear_function(force_nt):
    M, K, N = 256, 512, 256

    torch.manual_seed(0)
    inp = torch.randn(M, K, dtype=DTYPE, device=DEVICE, requires_grad=True)
    weight = torch.randn(N, K, dtype=DTYPE, device=DEVICE, requires_grad=True)
    grad_output = torch.randn(M, N, dtype=DTYPE, device=DEVICE)

    # Delayed tensorwise scales (one scalar per track), chosen to fill the FP8
    # range the way the production warmup staging would.
    scale_input = torch.tensor(FP8_MAX / inp.detach().abs().amax().item(), dtype=torch.float32, device=DEVICE)
    scale_weight = torch.tensor(
        FP8_MAX / weight.detach().abs().amax().item(), dtype=torch.float32, device=DEVICE
    )
    scale_grad = torch.tensor(
        FP8_BWD_MAX / grad_output.abs().amax().item(), dtype=torch.float32, device=DEVICE
    )

    staged_input_amax = torch.zeros((), dtype=torch.float32, device=DEVICE)
    staged_weight_amax = torch.zeros((), dtype=torch.float32, device=DEVICE)
    staged_grad_amax = torch.zeros((), dtype=torch.float32, device=DEVICE)

    gran_value = ScalingGranularity.TENSORWISE.value
    backend_value = BackendType.HIPBLASLT.value

    result = DelayedFP8LinearTensorwiseFunction.apply(
        inp,
        weight,
        scale_input,
        scale_weight,
        scale_grad,
        staged_input_amax,
        staged_weight_amax,
        staged_grad_amax,
        FP8_DTYPE,
        FP8_BWD_DTYPE,
        gran_value,
        backend_value,
        force_nt,
    )
    output = result[0]

    # Forward: output ~= input @ weight.T (standard nn.Linear), FP8-tensorwise.
    ref_out = inp.detach().float() @ weight.detach().float().t()
    assert output.shape == (M, N)
    assert torch.isfinite(output).all(), "FP8 forward produced non-finite values"
    out_rel = _rel_err(output, ref_out)
    assert out_rel < 0.1, f"FP8 forward too far from bf16 reference: rel_err={out_rel:.4f}"

    # Fused amax capture (the delayed-scaling contract): the staged buffers must
    # be written with the *current* tensor amaxes during forward.
    assert staged_input_amax.item() > 0, "staged_input_amax not captured in forward"
    assert staged_weight_amax.item() > 0, "staged_weight_amax not captured in forward"
    in_amax_rel = abs(staged_input_amax.item() - inp.detach().float().abs().amax().item()) / max(
        inp.detach().float().abs().amax().item(), 1e-8
    )
    w_amax_rel = abs(staged_weight_amax.item() - weight.detach().float().abs().amax().item()) / max(
        weight.detach().float().abs().amax().item(), 1e-8
    )
    assert in_amax_rel < 1e-2, f"staged input amax mismatch: rel={in_amax_rel:.4f}"
    assert w_amax_rel < 1e-2, f"staged weight amax mismatch: rel={w_amax_rel:.4f}"

    # Backward through the production Function.
    output.backward(grad_output)
    assert inp.grad is not None and weight.grad is not None
    assert torch.isfinite(inp.grad).all(), "grad_input non-finite"
    assert torch.isfinite(weight.grad).all(), "grad_weight non-finite"

    ref_grad_input = grad_output.float() @ weight.detach().float()
    ref_grad_weight = grad_output.float().t() @ inp.detach().float()
    # e5m2 backward has only 2 mantissa bits, so use a looser bound than forward.
    gi_rel = _rel_err(inp.grad, ref_grad_input)
    gw_rel = _rel_err(weight.grad, ref_grad_weight)
    assert gi_rel < 0.2, f"grad_input too far from reference: rel_err={gi_rel:.4f}"
    assert gw_rel < 0.2, f"grad_weight too far from reference: rel_err={gw_rel:.4f}"

    assert staged_grad_amax.item() > 0, "staged_grad_amax not captured in backward"
