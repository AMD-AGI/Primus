###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Parameterized Triton clamped-SwiGLU autograd tests (PRPUNDIT-23).

``test_clamped_swiglu.py`` covers the eager ``clamped_swiglu.py`` module.
This file covers the production Triton wrappers in ``fused_bias_swiglu.py``:
unweighted ``ClampedSwiGLUFunction`` (shared-expert MLP) and weighted
``ClampedWeightedSwiGLUFunction`` (grouped MLP). Linxwang consolidated both
into one ticket; the weighted path is dormant in shipped DeepSeek-V4 config
but stays in this shared suite.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="clamped SwiGLU Triton kernels need CUDA/HIP")


def _fp8_cast_works() -> bool:
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    # This suite only ever runs its parametrized cases on CUDA/HIP (the
    # module is skipped entirely otherwise via the `cuda` mark below), so
    # probe the cast on the device the kernels actually execute on. A
    # CPU-only probe can incorrectly skip fp8_input_store=True on builds
    # where FP8 casting is CUDA-only.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.zeros(4, device=device).to(torch.float8_e4m3fn).to(torch.float32)
        return True
    except (RuntimeError, TypeError):
        return False


# fp8_input_store=True needs a working float8_e4m3fn cast; gate just that
# parameter (mirrors test_v4_fp8_indexer.py) so fp8_input_store=False still
# runs on GPUs/torch builds without FP8 support instead of hard-failing.
_fp8_input_store_cases = [
    False,
    pytest.param(
        True,
        marks=pytest.mark.skipif(
            not _fp8_cast_works(), reason="torch.float8_e4m3fn cast unsupported on this build/device"
        ),
    ),
]


def _eager_clamped_swiglu(y: torch.Tensor, alpha: float) -> torch.Tensor:
    half = y.shape[-1] // 2
    gate, up = y[..., :half], y[..., half:]
    return F.silu(torch.clamp(gate, max=alpha)) * torch.clamp(up, min=-alpha, max=alpha)


def _eager_clamped_weighted_swiglu(y: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    return _eager_clamped_swiglu(y, alpha) * weights.unsqueeze(-1)


@cuda
class TestClampedSwiGLUFunction:
    def setup_method(self):
        pytest.importorskip("triton")
        from primus.backends.megatron.core.fusions.fused_bias_swiglu import (
            ClampedSwiGLUFunction,
            swiglu_impl,
        )

        self.Fn = ClampedSwiGLUFunction
        self.swiglu_impl = swiglu_impl

    @pytest.mark.parametrize("alpha", [7.0, 1.0, 0.5])
    @pytest.mark.parametrize("fp8_input_store", _fp8_input_store_cases)
    def test_forward_matches_eager_reference(self, alpha: float, fp8_input_store: bool):
        torch.manual_seed(2024)
        y = (torch.randn(17, 64, device="cuda", dtype=torch.float32) * 5.0).requires_grad_(True)
        out = self.Fn.apply(y, fp8_input_store, alpha)
        torch.testing.assert_close(out, _eager_clamped_swiglu(y, alpha), atol=1e-5, rtol=1e-5)
        (saved_input,) = out.grad_fn.saved_tensors
        assert saved_input.dtype == (torch.float8_e4m3fn if fp8_input_store else y.dtype)

    @pytest.mark.parametrize("fp8_input_store", _fp8_input_store_cases)
    def test_backward_matches_eager_autograd(self, fp8_input_store: bool):
        torch.manual_seed(6)
        alpha = 7.0
        y_fn = (torch.randn(5, 20, device="cuda", dtype=torch.float32) * 5.0).requires_grad_(True)
        y_eager = y_fn.detach().clone()
        if fp8_input_store:
            # Backward recomputes through the FP8-quantized value forward saved
            # for the input, not the full-precision input; mirror that lossy
            # round trip here so the reference exercises the same path instead
            # of silently comparing against full precision.
            y_eager = y_eager.to(torch.float8_e4m3fn).to(torch.float32)
        y_eager = y_eager.requires_grad_(True)
        grad_out = torch.randn(5, 10, device="cuda", dtype=torch.float32)

        self.Fn.apply(y_fn, fp8_input_store, alpha).backward(grad_out)
        _eager_clamped_swiglu(y_eager, alpha).backward(grad_out)
        torch.testing.assert_close(y_fn.grad, y_eager.grad, atol=1e-5, rtol=1e-5)

    def test_swiglu_impl_routes_to_clamped_function(self):
        torch.manual_seed(9)
        x = torch.randn(3, 5, 16, device="cuda", dtype=torch.float32)
        out = self.swiglu_impl(x, None, fp8_input_store=False, clamp_value=7.0)
        ref = _eager_clamped_swiglu(x.view(-1, 16), 7.0).view(3, 5, 8)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@cuda
class TestClampedWeightedSwiGLUFunction:
    def setup_method(self):
        pytest.importorskip("triton")
        from primus.backends.megatron.core.fusions.fused_bias_swiglu import (
            ClampedWeightedSwiGLUFunction,
            weighted_bias_swiglu_impl,
        )

        self.Fn = ClampedWeightedSwiGLUFunction
        self.impl = weighted_bias_swiglu_impl

    @pytest.mark.parametrize("alpha", [7.0, 1.0])
    @pytest.mark.parametrize("fp8_input_store", _fp8_input_store_cases)
    def test_forward_matches_eager_reference(self, alpha: float, fp8_input_store: bool):
        torch.manual_seed(2024)
        M, half = 5, 16
        y = (torch.randn(M, 2 * half, dtype=torch.float32, device="cuda") * 5.0).requires_grad_()
        weights = torch.rand(M, dtype=torch.float32, device="cuda") + 0.1
        out = self.Fn.apply(y, weights, fp8_input_store, alpha)
        torch.testing.assert_close(
            out, _eager_clamped_weighted_swiglu(y, weights, alpha), atol=1e-5, rtol=1e-5
        )
        saved_input, saved_weights = out.grad_fn.saved_tensors
        assert saved_input.dtype == (torch.float8_e4m3fn if fp8_input_store else y.dtype)
        assert saved_weights.dtype == weights.dtype

    @pytest.mark.parametrize("fp8_input_store", _fp8_input_store_cases)
    def test_backward_matches_eager_autograd(self, fp8_input_store: bool):
        torch.manual_seed(8)
        alpha = 7.0
        y_fn = (torch.randn(6, 16, device="cuda", dtype=torch.float32) * 4.0).requires_grad_(True)
        w_fn = (torch.rand(6, device="cuda", dtype=torch.float32) + 0.1).requires_grad_(True)
        y_e = y_fn.detach().clone()
        if fp8_input_store:
            # Same fp8 round trip as the unweighted backward test above: mirror
            # what ClampedWeightedSwiGLUFunction actually recomputes through.
            y_e = y_e.to(torch.float8_e4m3fn).to(torch.float32)
        y_e = y_e.requires_grad_(True)
        w_e = w_fn.detach().clone().requires_grad_(True)
        grad_out = torch.randn(6, 8, device="cuda", dtype=torch.float32)

        self.Fn.apply(y_fn, w_fn, fp8_input_store, alpha).backward(grad_out)
        _eager_clamped_weighted_swiglu(y_e, w_e, alpha).backward(grad_out)
        torch.testing.assert_close(y_fn.grad, y_e.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(w_fn.grad, w_e.grad, atol=1e-5, rtol=1e-5)

    def test_impl_routes_to_clamped_weighted_function(self):
        torch.manual_seed(3)
        x = torch.randn(4, 16, device="cuda", dtype=torch.float32)
        weights = torch.rand(4, device="cuda", dtype=torch.float32) + 0.1
        out = self.impl(x, None, weights, fp8_input_store=False, clamp_value=7.0)
        torch.testing.assert_close(out, _eager_clamped_weighted_swiglu(x, weights, 7.0), atol=1e-5, rtol=1e-5)
