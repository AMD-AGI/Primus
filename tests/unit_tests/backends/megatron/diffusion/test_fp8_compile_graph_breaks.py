# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
torch.compile graph-break coverage for the FP8 linear / attention path.

Motivation: the @allow_in_graph approach used by older FP8 ops created graph
break boundaries between AdaLN Triton kernels and FP8 quantize/GEMM ops,
costing ~144 ms (10.5%) GPU idle per iteration. Eliminating those breaks
(setup_context Functions + fused amax capture) is what this file guards.

This file has two kinds of tests:

1. Production-guarding tests -- these ``.apply`` the shipped Functions
   (``OpaqueFP8LinearTensorwiseFunction``, ``DualFP8LinearTensorwiseFunction``,
   and ``DelayedFP8LinearTensorwiseFunction``) and assert they trace under
   torch.compile with zero graph breaks and stay numerically equivalent to
   eager. A regression in the production op fails these directly.
2. Characterization / feasibility tests -- these use small in-file
   autograd.Function replicas (``_Level1FP8Linear``, the ``_AllowInGraph*``
   baselines, ``_DelayedFP8*``) to pin the torch.compile *capabilities* the
   production design relies on (buffer ``copy_`` inside a compiled forward,
   tensor hooks firing, ``allow_in_graph`` vs setup_context parity, buffer
   mutation visibility between compiled calls). They do not guard a production
   symbol; they document why the production approach is sound and would surface
   a torch/Inductor behavior change.

Numerical forward/backward correctness of the production Functions lives in
``test_delayed_fp8_triton_op.py`` and the turbo float8 tests; this file is
specifically about compile behavior.
"""

import math

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus_turbo.pytorch as pt
import torch.nn as nn
from primus_turbo.pytorch.core.backend import BackendType, GlobalBackendManager
from primus_turbo.pytorch.core.low_precision import (
    Float8QuantConfig,
    Format,
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)
from primus_turbo.pytorch.kernels.gemm.gemm_fp8_impl import gemm_fp8_impl
from primus_turbo.pytorch.ops.quantization import quantize_fp8

from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
    FP8UnshardedWeightTensor,
)
from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
    DelayedFP8LinearTensorwiseFunction,
    DualFP8LinearTensorwiseFunction,
    OpaqueFP8LinearTensorwiseFunction,
    _extract_fp8_weight,
)
from tests.utils import PrimusUT

_has_cuda = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _has_cuda, reason="CUDA required")

_DIM = 64
_OUT_DIM = 128
_BATCH = 4
_GRAN_VALUE = ScalingGranularity.TENSORWISE.value
_BACKEND_VALUE = BackendType.HIPBLASLT.value


# ---------------------------------------------------------------------------
# Level 1: Bare autograd.Function -- C++ custom ops, primitive args only
# ---------------------------------------------------------------------------


class _Level1FP8Linear(torch.autograd.Function):
    """FP8 linear without @allow_in_graph, using setup_context + primitives.

    Calls the C++ quantize_fp8_tensorwise and gemm_fp8_impl custom ops
    directly. No Float8QuantConfig, no FP8UnshardedWeightTensor.
    """

    @staticmethod
    def forward(input, weight, fp8_dtype, gran_value, backend_value):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp8, a_scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
            input_2d,
            fp8_dtype,
            None,
        )
        b_fp8, b_scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
            weight,
            fp8_dtype,
            None,
        )

        output = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=gran_value,
            default_backend=backend_value,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        return output, a_fp8, a_scale_inv, b_fp8, b_scale_inv

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, fp8_dtype, gran_value, backend_value = inputs
        output_val, a_fp8, a_scale_inv, b_fp8, b_scale_inv = output

        ctx.save_for_backward(a_fp8, a_scale_inv, b_fp8, b_scale_inv)
        ctx.mark_non_differentiable(a_fp8, a_scale_inv, b_fp8, b_scale_inv)
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        ctx.fp8_dtype = fp8_dtype
        ctx.gran_value = gran_value
        ctx.backend_value = backend_value

    @staticmethod
    def backward(ctx, grad_output, *_):
        a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        if not grad_2d.is_contiguous():
            grad_2d = grad_2d.contiguous()

        grad_fp8, grad_scale_inv = torch.ops.primus_turbo_cpp_extension.quantize_fp8_tensorwise(
            grad_2d,
            ctx.fp8_dtype,
            None,
        )
        grad_input = gemm_fp8_impl(
            grad_fp8,
            grad_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            False,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )
        grad_input = grad_input.reshape(ctx.orig_shape)

        grad_weight = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            True,
            grad_fp8,
            grad_scale_inv,
            False,
            ctx.out_dtype,
            True,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )
        return grad_input, grad_weight, None, None, None


class _Level1Module(nn.Module):
    """Minimal module: LayerNorm + Level 1 FP8 linear."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.norm(x)
        out = _Level1FP8Linear.apply(
            x,
            self.weight,
            float8_e4m3,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        return out[0]


# ---------------------------------------------------------------------------
# Bonus: LayerNorm + FP8 linear single compiled graph
# ---------------------------------------------------------------------------


@requires_cuda
class TestFullGraphIntegration(PrimusUT):
    """Verify LayerNorm + FP8 linear compiles into a single graph frame."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_single_compiled_frame(self):
        """LayerNorm → FP8 linear should produce exactly 1 compiled frame."""
        model = _Level1Module(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")

        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        explanation = torch._dynamo.explain(model)(x)

        assert explanation.graph_break_count == 0, (
            f"Expected single graph (0 breaks), got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )
        assert explanation.graph_count == 1, f"Expected 1 graph, got {explanation.graph_count}."


# ---------------------------------------------------------------------------
# Compiled vs eager numerical equivalence (regression for view_as aliasing fix)
# ---------------------------------------------------------------------------


class _OpaqueModule(nn.Module):
    """Module using OpaqueFP8LinearTensorwiseFunction with pre-extracted weights."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.norm(x)
        w_fp8, w_scale = quantize_fp8(self.weight, float8_e4m3, ScalingGranularity.TENSORWISE)
        result = OpaqueFP8LinearTensorwiseFunction.apply(
            x,
            self.weight,
            w_fp8,
            w_scale,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        return result[0]


@requires_cuda
class TestCompiledVsEagerEquivalence(PrimusUT):
    """Verify compiled and eager execution produce identical forward and backward results.

    Regression test for the view_as input-output aliasing bug where AOTAutograd's
    view-replay mechanism could corrupt saved FP8 weight data.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_forward_equivalence(self):
        """Compiled forward matches eager forward exactly."""
        model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        eager_out = model(x)

        torch._dynamo.reset()
        compiled = torch.compile(model)
        compiled_out = compiled(x)

        torch.testing.assert_close(compiled_out, eager_out, atol=0, rtol=0)

    def test_backward_equivalence(self):
        """Compiled backward matches eager backward exactly."""
        model = _OpaqueModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")

        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        eager_out = model(x)
        eager_out.sum().backward()
        eager_x_grad = x.grad.clone()
        eager_w_grad = model.weight.grad.clone()

        x.grad = None
        model.weight.grad = None
        torch._dynamo.reset()
        compiled = torch.compile(model)
        compiled_out = compiled(x)
        compiled_out.sum().backward()

        torch.testing.assert_close(x.grad, eager_x_grad, atol=0, rtol=0)
        torch.testing.assert_close(model.weight.grad, eager_w_grad, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Multi-step convergence: compiled (setup_context) vs compiled (@allow_in_graph)
# vs eager, plus BF16 baseline
# ---------------------------------------------------------------------------

import copy


@torch._dynamo.allow_in_graph
class _AllowInGraphFP8Linear(torch.autograd.Function):
    """Old-style @allow_in_graph FP8 linear for baseline comparison.

    Quantizes both input and weight inside forward, saves FP8 tensors on ctx.
    AOTAutograd never traces through this -- it's fully opaque.
    """

    @staticmethod
    def forward(ctx, input, weight):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp8, a_scale_inv = quantize_fp8(input_2d, float8_e4m3, ScalingGranularity.TENSORWISE)
        b_fp8, b_scale_inv = quantize_fp8(weight, float8_e4m3, ScalingGranularity.TENSORWISE)

        output = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )

        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        ctx.save_for_backward(a_fp8, a_scale_inv, b_fp8, b_scale_inv)
        ctx.out_dtype = out_dtype
        ctx.orig_shape = orig_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_fp8, grad_scale_inv = quantize_fp8(grad_2d, float8_e5m2, ScalingGranularity.TENSORWISE)

        grad_input = gemm_fp8_impl(
            grad_fp8,
            grad_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            False,
            ctx.out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )
        grad_input = grad_input.reshape(ctx.orig_shape)

        grad_weight = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            True,
            grad_fp8,
            grad_scale_inv,
            False,
            ctx.out_dtype,
            True,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )
        return grad_input, grad_weight


class _AllowInGraphModule(nn.Module):
    """Module using old @allow_in_graph FP8 linear."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.norm(x)
        return _AllowInGraphFP8Linear.apply(x, self.weight)


class _BF16LinearModule(nn.Module):
    """Pure BF16 linear for baseline comparison (no FP8 quantization)."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.linear(x)


_CONV_DIM = 256
_CONV_OUT = 512
_CONV_BATCH = 16
_N_STEPS = 200


def _make_inputs(n, dim, batch, seed=42):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(batch, dim, dtype=torch.bfloat16, generator=g).cuda() for _ in range(n)]


def _run_training_loop(model, inputs, n_steps, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    has_prepare = hasattr(model, "prepare_fp8_weight")
    if not has_prepare and hasattr(model, "_orig_mod"):
        has_prepare = hasattr(model._orig_mod, "prepare_fp8_weight")
    losses = []
    grad_norms = []
    for step in range(n_steps):
        if has_prepare:
            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            m.prepare_fp8_weight()
        optimizer.zero_grad()
        out = model(inputs[step % len(inputs)])
        loss = out.sum()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
        grad_norms.append(gn.item())
        optimizer.step()
        losses.append(loss.item())
    return losses, grad_norms


def _clone_state(model):
    return copy.deepcopy(model.state_dict())


def _print_comparison(label_a, losses_a, label_b, losses_b, milestones=None):
    """Print side-by-side loss comparison at milestones for diagnosis."""
    if milestones is None:
        milestones = [0, 1, 5, 10, 20, 50, 100, 150, 199]
    milestones = [m for m in milestones if m < len(losses_a) and m < len(losses_b)]
    print(f"\n{'Step':>6} | {label_a:>20} | {label_b:>20} | {'Rel Diff':>10}")
    print("-" * 65)
    for m in milestones:
        a, b = losses_a[m], losses_b[m]
        rel = abs(a - b) / max(abs(a), 1e-12)
        print(f"{m:>6} | {a:>20.6f} | {b:>20.6f} | {rel:>10.6f}")


@requires_cuda
class TestMultiStepConvergence(PrimusUT):
    """Multi-step convergence comparison across compilation modes.

    Runs 200 optimizer steps on a small model and compares loss/grad_norm
    trajectories to isolate whether AOTAutograd backward compilation
    causes numerical divergence vs the old @allow_in_graph approach.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def _make_fp8_setup_ctx_model(self, state_dict):
        model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        model.load_state_dict(state_dict)
        return model

    def _make_fp8_allow_in_graph_model(self, state_dict):
        model = _AllowInGraphModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        model.load_state_dict(state_dict)
        return model

    def _make_bf16_model(self, state_dict):
        model = _BF16LinearModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        # Map weight -> linear.weight for BF16 module
        bf16_state = {
            "norm.weight": state_dict["norm.weight"],
            "norm.bias": state_dict["norm.bias"],
            "linear.weight": state_dict["weight"],
        }
        model.load_state_dict(bf16_state)
        return model

    def test_compiled_setup_ctx_vs_eager(self):
        """setup_context compiled should track FP8 eager closely over 200 steps."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        eager_losses, eager_gn = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = self._make_fp8_setup_ctx_model(init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, compiled_gn = _run_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison("FP8 Eager", eager_losses, "FP8 Compiled(setup_ctx)", compiled_losses)

        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"FP8 compiled (setup_context) diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_allow_in_graph_vs_eager(self):
        """@allow_in_graph compiled should track FP8 eager closely over 200 steps."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        eager_losses, eager_gn = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        aig_model = self._make_fp8_allow_in_graph_model(init_state)
        aig_model_c = torch.compile(aig_model)
        aig_losses, aig_gn = _run_training_loop(aig_model_c, inputs, _N_STEPS)

        _print_comparison("FP8 Eager", eager_losses, "FP8 @allow_in_graph", aig_losses)

        final_rel = abs(eager_losses[-1] - aig_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"FP8 @allow_in_graph diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, aig={aig_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_bf16_compiled_vs_eager(self):
        """BF16 compiled should match BF16 eager closely (no FP8 noise)."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        eager_model = self._make_bf16_model(init_state)
        eager_losses, _ = _run_training_loop(eager_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = self._make_bf16_model(init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, _ = _run_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison("BF16 Eager", eager_losses, "BF16 Compiled", compiled_losses)

        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.01, (
            f"BF16 compiled diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )


# ---------------------------------------------------------------------------
# FSDP2 FP8UnshardedWeightTensor subclass variants
# ---------------------------------------------------------------------------


class _FP8SubclassModule(nn.Module):
    """Module simulating FSDP2 FP8 all-gather: weight goes through
    FP8UnshardedWeightTensor subclass before _extract_fp8_weight.

    In production, FSDP2 creates the FP8UnshardedWeightTensor outside the
    compiled graph (in fsdp_post_all_gather). The compiled forward only sees
    the subclass as an input and calls _extract_fp8_weight on it. We simulate
    this by pre-creating the subclass in prepare_fp8_weight() which must be
    called before each forward step.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self._fp8_weight = None

    @torch.no_grad()
    def prepare_fp8_weight(self):
        """Simulate FSDP2 fsdp_post_all_gather: create FP8UnshardedWeightTensor
        outside the compiled graph, just like FSDP2 does before forward."""
        w_fp8, w_scale = quantize_fp8(self.weight, float8_e4m3, ScalingGranularity.TENSORWISE)
        fp8_config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
        self._fp8_weight = FP8UnshardedWeightTensor(w_fp8, w_scale, torch.bfloat16, fp8_config)

    def forward(self, x):
        x = self.norm(x)
        weight_fp8, weight_scale_inv = _extract_fp8_weight(self._fp8_weight, float8_e4m3)
        result = OpaqueFP8LinearTensorwiseFunction.apply(
            x,
            self.weight,
            weight_fp8,
            weight_scale_inv,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        return result[0]


@torch._dynamo.allow_in_graph
class _AllowInGraphFP8LinearWithSubclass(torch.autograd.Function):
    """Old-style @allow_in_graph FP8 linear that receives an
    FP8UnshardedWeightTensor and extracts data inside the opaque boundary."""

    @staticmethod
    def forward(ctx, input, weight):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        a_fp8, a_scale_inv = quantize_fp8(input_2d, float8_e4m3, ScalingGranularity.TENSORWISE)

        if isinstance(weight, FP8UnshardedWeightTensor):
            b_fp8, b_scale_inv = weight.get_fp8_data_and_scale_inv()
        else:
            b_fp8, b_scale_inv = quantize_fp8(weight, float8_e4m3, ScalingGranularity.TENSORWISE)

        output = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            True,
            out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )

        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        ctx.save_for_backward(a_fp8, a_scale_inv, b_fp8, b_scale_inv)
        ctx.out_dtype = out_dtype
        ctx.orig_shape = orig_shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        a_fp8, a_scale_inv, b_fp8, b_scale_inv = ctx.saved_tensors

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_fp8, grad_scale_inv = quantize_fp8(grad_2d, float8_e5m2, ScalingGranularity.TENSORWISE)

        grad_input = gemm_fp8_impl(
            grad_fp8,
            grad_scale_inv,
            False,
            b_fp8,
            b_scale_inv,
            False,
            ctx.out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )
        grad_input = grad_input.reshape(ctx.orig_shape)

        grad_weight = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            True,
            grad_fp8,
            grad_scale_inv,
            False,
            ctx.out_dtype,
            True,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )
        return grad_input, grad_weight


class _AllowInGraphSubclassModule(nn.Module):
    """Module using old @allow_in_graph FP8 linear with FP8UnshardedWeightTensor
    passed directly into the opaque boundary (matching old production code).

    Like _FP8SubclassModule, the subclass is created outside compile."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self._fp8_weight = None

    @torch.no_grad()
    def prepare_fp8_weight(self):
        w_fp8, w_scale = quantize_fp8(self.weight, float8_e4m3, ScalingGranularity.TENSORWISE)
        fp8_config = Float8QuantConfig(format=Format.E4M3, granularity=ScalingGranularity.TENSORWISE)
        self._fp8_weight = FP8UnshardedWeightTensor(w_fp8, w_scale, torch.bfloat16, fp8_config)

    def forward(self, x):
        x = self.norm(x)
        return _AllowInGraphFP8LinearWithSubclass.apply(x, self._fp8_weight)


@requires_cuda
class TestFP8SubclassConvergence(PrimusUT):
    """Multi-step convergence with FP8UnshardedWeightTensor subclass path.

    Isolates whether the FSDP2 tensor subclass interaction under torch.compile
    causes numerical divergence vs eager or the old @allow_in_graph path.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_fp8_subclass_compiled_vs_eager(self):
        """setup_context + FP8UnshardedWeightTensor: compiled vs eager."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _FP8SubclassModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        eager_losses, _ = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = _FP8SubclassModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        compiled_model.load_state_dict(init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, _ = _run_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison("FP8 Subclass Eager", eager_losses, "FP8 Subclass Compiled", compiled_losses)

        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"FP8 subclass compiled diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_fp8_subclass_allow_in_graph_vs_eager(self):
        """@allow_in_graph + FP8UnshardedWeightTensor: compiled vs eager."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _AllowInGraphSubclassModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        eager_losses, _ = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = _AllowInGraphSubclassModule(_CONV_DIM, _CONV_OUT).to(
            dtype=torch.bfloat16, device="cuda"
        )
        compiled_model.load_state_dict(init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, _ = _run_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison(
            "FP8 AIG+Subclass Eager", eager_losses, "FP8 AIG+Subclass Compiled", compiled_losses
        )

        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"FP8 @allow_in_graph+subclass compiled diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_subclass_vs_plain_weight_eager(self):
        """FP8UnshardedWeightTensor path vs plain quantize path (both eager).
        Verifies the subclass wrapper doesn't change numerical results."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        plain_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(plain_model)

        plain_losses, _ = _run_training_loop(plain_model, inputs, _N_STEPS)

        subclass_model = _FP8SubclassModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        subclass_model.load_state_dict(init_state)
        subclass_losses, _ = _run_training_loop(subclass_model, inputs, _N_STEPS)

        _print_comparison("FP8 Plain Eager", plain_losses, "FP8 Subclass Eager", subclass_losses)

        final_rel = abs(plain_losses[-1] - subclass_losses[-1]) / max(abs(plain_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.01, (
            f"FP8 subclass path diverged from plain path in eager mode: "
            f"plain={plain_losses[-1]:.6f}, subclass={subclass_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )


# ---------------------------------------------------------------------------
# Dual FP8 linear convergence tests
# ---------------------------------------------------------------------------


class _DualFP8Module(nn.Module):
    """Module using DualFP8LinearTensorwiseFunction (two GEMMs in one node).

    Simulates the JointSelfAttention dual output projection pattern:
    input goes through LayerNorm, then two independent FP8 linear projections
    are computed in a single autograd node. Outputs are summed for loss.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight_a = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self.weight_b = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.norm(x)
        w_fp8_a, w_scale_a = quantize_fp8(self.weight_a, float8_e4m3, ScalingGranularity.TENSORWISE)
        w_fp8_b, w_scale_b = quantize_fp8(self.weight_b, float8_e4m3, ScalingGranularity.TENSORWISE)
        result = DualFP8LinearTensorwiseFunction.apply(
            x,
            self.weight_a,
            w_fp8_a,
            w_scale_a,
            x,
            self.weight_b,
            w_fp8_b,
            w_scale_b,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        return result[0] + result[1]


@torch._dynamo.allow_in_graph
class _AllowInGraphDualFP8Linear(torch.autograd.Function):
    """Old-style @allow_in_graph dual FP8 linear for baseline comparison.

    Quantizes both inputs and weights inside forward, saves FP8 tensors on ctx.
    Matches the old DualFP8LinearTensorwiseFunction behavior before setup_context.
    """

    @staticmethod
    def forward(ctx, input_a, weight_a, input_b, weight_b):
        out_dtype = input_a.dtype

        orig_shape_a = input_a.shape
        orig_shape_b = input_b.shape
        input_a_2d = input_a.reshape(-1, input_a.shape[-1])
        input_b_2d = input_b.reshape(-1, input_b.shape[-1])

        a_fp8_a, a_scale_a = quantize_fp8(input_a_2d, float8_e4m3, ScalingGranularity.TENSORWISE)
        b_fp8_a, b_scale_a = quantize_fp8(weight_a, float8_e4m3, ScalingGranularity.TENSORWISE)
        a_fp8_b, a_scale_b = quantize_fp8(input_b_2d, float8_e4m3, ScalingGranularity.TENSORWISE)
        b_fp8_b, b_scale_b = quantize_fp8(weight_b, float8_e4m3, ScalingGranularity.TENSORWISE)

        output_a = gemm_fp8_impl(
            a_fp8_a,
            a_scale_a,
            False,
            b_fp8_a,
            b_scale_a,
            True,
            out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )
        output_b = gemm_fp8_impl(
            a_fp8_b,
            a_scale_b,
            False,
            b_fp8_b,
            b_scale_b,
            True,
            out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )

        output_a = output_a.reshape(*orig_shape_a[:-1], output_a.shape[-1])
        output_b = output_b.reshape(*orig_shape_b[:-1], output_b.shape[-1])

        ctx.save_for_backward(
            a_fp8_a,
            a_scale_a,
            b_fp8_a,
            b_scale_a,
            a_fp8_b,
            a_scale_b,
            b_fp8_b,
            b_scale_b,
        )
        ctx.out_dtype = out_dtype
        ctx.orig_shape_a = orig_shape_a
        ctx.orig_shape_b = orig_shape_b
        return output_a, output_b

    @staticmethod
    def backward(ctx, grad_output_a, grad_output_b):
        if not grad_output_a.is_contiguous():
            grad_output_a = grad_output_a.contiguous()
        if not grad_output_b.is_contiguous():
            grad_output_b = grad_output_b.contiguous()

        (a_fp8_a, a_scale_a, b_fp8_a, b_scale_a, a_fp8_b, a_scale_b, b_fp8_b, b_scale_b) = ctx.saved_tensors

        grad_a_2d = grad_output_a.reshape(-1, grad_output_a.shape[-1])
        grad_fp8_a, grad_scale_a = quantize_fp8(grad_a_2d, float8_e5m2, ScalingGranularity.TENSORWISE)

        grad_input_a = gemm_fp8_impl(
            grad_fp8_a,
            grad_scale_a,
            False,
            b_fp8_a,
            b_scale_a,
            False,
            ctx.out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )
        grad_input_a = grad_input_a.reshape(ctx.orig_shape_a)

        grad_weight_a = gemm_fp8_impl(
            a_fp8_a,
            a_scale_a,
            True,
            grad_fp8_a,
            grad_scale_a,
            False,
            ctx.out_dtype,
            True,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )

        grad_b_2d = grad_output_b.reshape(-1, grad_output_b.shape[-1])
        grad_fp8_b, grad_scale_b = quantize_fp8(grad_b_2d, float8_e5m2, ScalingGranularity.TENSORWISE)

        grad_input_b = gemm_fp8_impl(
            grad_fp8_b,
            grad_scale_b,
            False,
            b_fp8_b,
            b_scale_b,
            False,
            ctx.out_dtype,
            False,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )
        grad_input_b = grad_input_b.reshape(ctx.orig_shape_b)

        grad_weight_b = gemm_fp8_impl(
            a_fp8_b,
            a_scale_b,
            True,
            grad_fp8_b,
            grad_scale_b,
            False,
            ctx.out_dtype,
            True,
            granularity=_GRAN_VALUE,
            default_backend=_BACKEND_VALUE,
        )

        return grad_input_a, grad_weight_a, grad_input_b, grad_weight_b


class _AllowInGraphDualModule(nn.Module):
    """Module using old @allow_in_graph dual FP8 linear."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight_a = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self.weight_b = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.norm(x)
        out_a, out_b = _AllowInGraphDualFP8Linear.apply(x, self.weight_a, x, self.weight_b)
        return out_a + out_b


class _TwoSinglesModule(nn.Module):
    """Module using two separate OpaqueFP8LinearTensorwiseFunction calls.

    Same computation as _DualFP8Module but using two independent autograd
    nodes instead of one combined node. Used to verify the dual node
    doesn't introduce numerical differences.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight_a = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self.weight_b = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        x = self.norm(x)
        w_fp8_a, w_scale_a = quantize_fp8(self.weight_a, float8_e4m3, ScalingGranularity.TENSORWISE)
        w_fp8_b, w_scale_b = quantize_fp8(self.weight_b, float8_e4m3, ScalingGranularity.TENSORWISE)
        result_a = OpaqueFP8LinearTensorwiseFunction.apply(
            x,
            self.weight_a,
            w_fp8_a,
            w_scale_a,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        result_b = OpaqueFP8LinearTensorwiseFunction.apply(
            x,
            self.weight_b,
            w_fp8_b,
            w_scale_b,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        return result_a[0] + result_b[0]


@requires_cuda
class TestDualFP8Convergence(PrimusUT):
    """Multi-step convergence for DualFP8LinearTensorwiseFunction.

    The dual function bundles two FP8 GEMMs into a single autograd node
    (used by JointSelfAttention). This tests whether the combined node
    under torch.compile causes divergence vs eager or the old @allow_in_graph.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def _make_dual_init_state(self):
        model = _DualFP8Module(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        return model, _clone_state(model)

    def _load_dual_model(self, cls, state_dict):
        model = cls(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        model.load_state_dict(state_dict)
        return model

    def test_dual_compiled_setup_ctx_vs_eager(self):
        """DualFP8 setup_context compiled should track eager over 200 steps."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model, init_state = self._make_dual_init_state()
        eager_losses, _ = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = self._load_dual_model(_DualFP8Module, init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, _ = _run_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison("Dual FP8 Eager", eager_losses, "Dual FP8 Compiled", compiled_losses)

        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"Dual FP8 compiled (setup_context) diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_dual_allow_in_graph_vs_eager(self):
        """DualFP8 @allow_in_graph compiled should track eager over 200 steps."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model, init_state = self._make_dual_init_state()
        eager_losses, _ = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        aig_model = self._load_dual_model(_AllowInGraphDualModule, init_state)
        aig_model_c = torch.compile(aig_model)
        aig_losses, _ = _run_training_loop(aig_model_c, inputs, _N_STEPS)

        _print_comparison("Dual FP8 Eager", eager_losses, "Dual FP8 @allow_in_graph", aig_losses)

        final_rel = abs(eager_losses[-1] - aig_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"Dual FP8 @allow_in_graph diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, aig={aig_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_dual_vs_two_singles_eager(self):
        """Dual node should match two single OpaqueFP8 calls (both eager).

        Verifies that bundling two GEMMs into one autograd node doesn't
        change numerical results compared to two separate nodes.
        """
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        dual_model, init_state = self._make_dual_init_state()
        dual_losses, _ = _run_training_loop(dual_model, inputs, _N_STEPS)

        singles_model = self._load_dual_model(_TwoSinglesModule, init_state)
        singles_losses, _ = _run_training_loop(singles_model, inputs, _N_STEPS)

        _print_comparison("Dual FP8 Eager", dual_losses, "Two Singles Eager", singles_losses)

        final_rel = abs(dual_losses[-1] - singles_losses[-1]) / max(abs(dual_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.01, (
            f"Dual FP8 node diverged from two single nodes: "
            f"dual={dual_losses[-1]:.6f}, singles={singles_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )


# ---------------------------------------------------------------------------
# FP8 Attention Module: QKV projection + flash attention + output projection
# ---------------------------------------------------------------------------

_ATTN_DIM = 256
_ATTN_HEADS = 4
_ATTN_HEAD_DIM = _ATTN_DIM // _ATTN_HEADS  # 64
_ATTN_SEQ = 32
_ATTN_BATCH = 8
_ATTN_N_STEPS = 200


def _hermetic_compile_state():
    """Reset global compile / kernel-dispatch state for hermetic isolation.

    These convergence tests are sensitive to global state that leaks across
    tests in the same process:
      * primus-turbo's kernel-dispatch and origami autotune caches
        (``GlobalBackendManager.reset()`` clears both), and
      * torch dynamo's compiled-graph cache (``torch._dynamo.reset()``).
    Without isolation, kernel/codegen choices made by an earlier test bleed
    in and shift the compiled-vs-eager numerics, which then compounds over the
    200-step loop into order-dependent (flaky) failures. We also pin
    auto-tune off so dispatch deterministically uses the default backend.
    """
    GlobalBackendManager.reset()
    GlobalBackendManager.set_auto_tune(False)
    torch._dynamo.reset()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


class _FP8AttentionModule(nn.Module):
    """Minimal FP8 attention: QKV projection + flash attention + output projection.

    Mirrors the real Flux attention path:
    - FP8 linear for QKV (OpaqueFP8LinearTensorwiseFunction)
    - pt.ops.flash_attn_func for attention
    - FP8 linear for output projection
    """

    def __init__(self, dim, num_heads, deterministic=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.deterministic = deterministic

        self.norm = nn.LayerNorm(dim)
        self.qkv_weight = nn.Parameter(torch.randn(3 * dim, dim, dtype=torch.bfloat16))
        self.proj_weight = nn.Parameter(torch.randn(dim, dim, dtype=torch.bfloat16))

    def forward(self, x):
        B, S, D = x.shape
        x = self.norm(x)

        qkv_w_fp8, qkv_w_scale = quantize_fp8(self.qkv_weight, float8_e4m3, ScalingGranularity.TENSORWISE)
        qkv = OpaqueFP8LinearTensorwiseFunction.apply(
            x,
            self.qkv_weight,
            qkv_w_fp8,
            qkv_w_scale,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )[0]

        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        attn_out = pt.ops.flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=False,
            window_size=(-1, -1),
            deterministic=self.deterministic,
            return_lse=False,
        )

        attn_out = attn_out.reshape(B, S, D)

        proj_w_fp8, proj_w_scale = quantize_fp8(self.proj_weight, float8_e4m3, ScalingGranularity.TENSORWISE)
        out = OpaqueFP8LinearTensorwiseFunction.apply(
            attn_out,
            self.proj_weight,
            proj_w_fp8,
            proj_w_scale,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )[0]

        return out


def _make_seq_inputs(n, seq, dim, batch, seed=42):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(batch, seq, dim, dtype=torch.bfloat16, generator=g).cuda() for _ in range(n)]


@requires_cuda
class TestFP8AttentionConvergence(PrimusUT):
    """FP8 attention convergence: compiled vs eager with flash attention.

    Tests whether torch.compile interacts with FP8 QKV projection +
    flash attention + FP8 output projection to cause divergence.

    Flash attention is non-deterministic by default, so the non-deterministic
    test measures whether compile-induced variance stays within the natural
    variance of flash attention itself.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @pytest.fixture(autouse=True)
    def hermetic_compile_state(self):
        _hermetic_compile_state()
        try:
            yield
        finally:
            _hermetic_compile_state()

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_fp8_attention_nondeterministic_compiled_vs_eager(self):
        """With non-deterministic flash attention, compile variance should be
        within a small multiple of the natural flash attention variance."""
        torch._dynamo.reset()
        inputs = _make_seq_inputs(20, _ATTN_SEQ, _ATTN_DIM, _ATTN_BATCH)

        ref_model = _FP8AttentionModule(_ATTN_DIM, _ATTN_HEADS, deterministic=False).to(
            dtype=torch.bfloat16, device="cuda"
        )
        init_state = _clone_state(ref_model)

        eager1 = _FP8AttentionModule(_ATTN_DIM, _ATTN_HEADS, deterministic=False).to(
            dtype=torch.bfloat16, device="cuda"
        )
        eager1.load_state_dict(init_state)
        eager1_losses, _ = _run_training_loop(eager1, inputs, _ATTN_N_STEPS)

        eager2 = _FP8AttentionModule(_ATTN_DIM, _ATTN_HEADS, deterministic=False).to(
            dtype=torch.bfloat16, device="cuda"
        )
        eager2.load_state_dict(init_state)
        eager2_losses, _ = _run_training_loop(eager2, inputs, _ATTN_N_STEPS)

        torch._dynamo.reset()
        compiled_model = _FP8AttentionModule(_ATTN_DIM, _ATTN_HEADS, deterministic=False).to(
            dtype=torch.bfloat16, device="cuda"
        )
        compiled_model.load_state_dict(init_state)
        compiled_model = torch.compile(compiled_model)
        compiled_losses, _ = _run_training_loop(compiled_model, inputs, _ATTN_N_STEPS)

        _print_comparison(
            "FP8 Attn Eager1",
            eager1_losses,
            "FP8 Attn Eager2",
            eager2_losses,
        )
        _print_comparison(
            "FP8 Attn Eager1",
            eager1_losses,
            "FP8 Attn Compiled",
            compiled_losses,
        )

        natural_var = abs(eager1_losses[-1] - eager2_losses[-1])
        compile_var = abs(eager1_losses[-1] - compiled_losses[-1])
        ref_loss = max(abs(eager1_losses[-1]), 1e-12)

        natural_rel = natural_var / ref_loss
        compile_rel = compile_var / ref_loss

        print(f"\nNatural variance (eager vs eager): {natural_var:.6f} (rel: {natural_rel:.6f})")
        print(f"Compile variance (eager vs compiled): {compile_var:.6f} (rel: {compile_rel:.6f})")

        if natural_var < 1e-6:
            assert compile_rel < 0.05, (
                f"Flash attn showed no natural variance but compile diverged: "
                f"compile_rel={compile_rel:.6f}"
            )
        else:
            ratio = compile_var / natural_var
            print(f"Compile/natural variance ratio: {ratio:.2f}x")
            assert ratio < 5.0, (
                f"Compile variance {ratio:.1f}x larger than natural flash attn variance: "
                f"natural={natural_var:.6f}, compile={compile_var:.6f}"
            )


# ---------------------------------------------------------------------------
# Mock Megatron DDP: param/grad buffer remapping + backward hooks
# ---------------------------------------------------------------------------


class _MockMegatronDDP(nn.Module):
    """Simulates Megatron DDP's param/grad buffer remapping and backward hooks.

    Reproduces the essential behavior from distributed_data_parallel.py
    (lines 419-444) and param_and_grad_buffer.py (lines 760-785) without
    any Megatron imports:

    1. Remaps param.data into a contiguous param_buffer
       (like use_distributed_optimizer=True)
    2. Assigns param.main_grad as views into a contiguous grad_buffer
    3. Registers gradient accumulator hooks that do:
       param.main_grad.add_(param.grad.data); param.grad = None
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

        params = [p for p in module.parameters() if p.requires_grad]

        total_numel = sum(p.numel() for p in params)
        self.param_buffer = torch.zeros(total_numel, dtype=params[0].dtype, device=params[0].device)
        self.grad_buffer = torch.zeros(total_numel, dtype=params[0].dtype, device=params[0].device)

        offset = 0
        self.grad_accs = []
        for param in params:
            numel = param.numel()
            new_data = self.param_buffer[offset : offset + numel].view(param.shape)
            new_data.copy_(param.data)
            param.data = new_data
            param.main_grad = self.grad_buffer[offset : offset + numel].view(param.shape)
            param.grad_added_to_main_grad = False
            offset += numel

            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_hook(param))
            self.grad_accs.append(grad_acc)

    @staticmethod
    def _make_hook(param):
        def hook(*unused):
            if param.grad is not None and not param.grad_added_to_main_grad:
                param.main_grad.add_(param.grad.data)
            param.grad = None

        return hook

    def zero_grad_buffer(self):
        self.grad_buffer.zero_()
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad_added_to_main_grad = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class _MainGradAdamW(torch.optim.AdamW):
    """AdamW that reads from param.main_grad instead of param.grad.

    Simulates how Megatron's distributed optimizer operates on main_grad
    buffers rather than the standard param.grad tensors.
    """

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    p.grad = p.main_grad.clone()
        result = super().step(closure=closure)
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None
        return result


def _run_mock_ddp_training(module, inputs, n_steps, compile_target=None, lr=1e-3):
    """Run training with mock Megatron DDP.

    Args:
        module: The model to wrap with mock DDP.
        inputs: List of input tensors.
        n_steps: Number of training steps.
        compile_target: One of None, "whole_model", "inner_only".
            None = eager, "whole_model" = compile ddp.forward,
            "inner_only" = compile ddp.module.forward.
        lr: Learning rate.
    """
    ddp = _MockMegatronDDP(module)
    optimizer = _MainGradAdamW(ddp.module.parameters(), lr=lr)

    if compile_target == "whole_model":
        ddp.forward = torch.compile(ddp.forward)
    elif compile_target == "inner_only":
        ddp.module.forward = torch.compile(ddp.module.forward)

    losses = []
    grad_norms = []
    for step in range(n_steps):
        ddp.zero_grad_buffer()
        out = ddp(inputs[step % len(inputs)])
        loss = out.sum()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in ddp.module.parameters() if p.main_grad is not None],
            max_norm=float("inf"),
        )
        grad_norms.append(gn.item() if hasattr(gn, "item") else gn)
        optimizer.step()
        losses.append(loss.item())
    return losses, grad_norms


@requires_cuda
class TestMockDDPConvergence(PrimusUT):
    """Mock Megatron DDP convergence: whole_model compile vs inner-only vs eager.

    Tests whether torch.compile scope interacts with Megatron DDP's backward
    hooks (param.grad -> main_grad transfer + param.grad = None) to cause
    numerical divergence.

    The local spec DDP config uses whole_model compile, while the converging
    TE spec reference uses stack (inner-only) compile. Phase 1 proved that
    FP8 compiled vs eager is bit-identical without DDP hooks.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_mock_ddp_whole_model_compiled_vs_eager(self):
        """whole_model compile + mock DDP should track eager + mock DDP."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        eager_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        eager_model.load_state_dict(init_state)
        eager_losses, _ = _run_mock_ddp_training(eager_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        compiled_model.load_state_dict(init_state)
        compiled_losses, _ = _run_mock_ddp_training(
            compiled_model, inputs, _N_STEPS, compile_target="whole_model"
        )

        _print_comparison(
            "MockDDP Eager",
            eager_losses,
            "MockDDP WholeModel",
            compiled_losses,
        )

        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"Mock DDP whole_model compile diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_mock_ddp_inner_only_compiled_vs_eager(self):
        """inner-only compile + mock DDP should track eager + mock DDP."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        eager_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        eager_model.load_state_dict(init_state)
        eager_losses, _ = _run_mock_ddp_training(eager_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        compiled_model.load_state_dict(init_state)
        compiled_losses, _ = _run_mock_ddp_training(
            compiled_model, inputs, _N_STEPS, compile_target="inner_only"
        )

        _print_comparison(
            "MockDDP Eager",
            eager_losses,
            "MockDDP InnerOnly",
            compiled_losses,
        )

        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"Mock DDP inner-only compile diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_mock_ddp_vs_vanilla_eager(self):
        """Mock DDP eager should produce same results as vanilla eager."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)

        vanilla_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        vanilla_model.load_state_dict(init_state)
        vanilla_losses, _ = _run_training_loop(vanilla_model, inputs, _N_STEPS)

        ddp_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        ddp_model.load_state_dict(init_state)
        ddp_losses, _ = _run_mock_ddp_training(ddp_model, inputs, _N_STEPS)

        _print_comparison(
            "Vanilla Eager",
            vanilla_losses,
            "MockDDP Eager",
            ddp_losses,
        )

        final_rel = abs(vanilla_losses[-1] - ddp_losses[-1]) / max(abs(vanilla_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.01, (
            f"Mock DDP eager diverged from vanilla eager: "
            f"vanilla={vanilla_losses[-1]:.6f}, ddp={ddp_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )


# ---------------------------------------------------------------------------
# Phase 0: Delayed FP8 Scaling Feasibility Tests
#
# These tests resolve compile-interaction unknowns required before
# implementing delayed scaling. Each test answers a specific question
# about torch.compile behavior with buffer mutations, tensor hooks,
# and autograd Function side-outputs.
# ---------------------------------------------------------------------------

_FP8_FWD_MAX = torch.finfo(float8_e4m3).max
_FP8_BWD_MAX = torch.finfo(float8_e5m2).max


# -- Test 0a helpers --------------------------------------------------------


class _BufferCopyModule(nn.Module):
    """Module that copies a computed scalar into a registered buffer.

    Mimics the real delayed scaling pattern where input_amax is computed
    from the BF16 input (not the FP8 output) and stored via buffer copy_().
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self.register_buffer("staged_amax", torch.tensor(0.0))

    def forward(self, x):
        x = self.norm(x)
        input_amax = x.detach().abs().amax().float()
        result = _Level1FP8Linear.apply(
            x,
            self.weight,
            float8_e4m3,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        self.staged_amax.copy_(input_amax)
        return result[0]


# -- Test 0c helpers --------------------------------------------------------


def _grad_amax_hook(grad, buf):
    """Tensor hook that records gradient amax into a buffer. Returns None
    to leave the gradient unchanged."""
    buf.copy_(grad.detach().abs().amax().float())
    return None


class _TensorHookModule(nn.Module):
    """Module that registers a backward tensor hook to capture grad amax."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self.register_buffer("grad_amax", torch.tensor(0.0))

    def forward(self, x):
        x = self.norm(x)
        result = _Level1FP8Linear.apply(
            x,
            self.weight,
            float8_e4m3,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        output = result[0]
        buf = self.grad_amax
        output.register_hook(lambda g, b=buf: _grad_amax_hook(g, b))
        return output


# -- Test 0d helpers --------------------------------------------------------


class _DelayedFP8Linear(torch.autograd.Function):
    """Minimal delayed-scaling FP8 linear for feasibility testing.

    Takes 3 pre-computed scales (input, weight, gradient).
    Returns (output, a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv,
             input_amax, weight_amax) with the last 6 mark_non_differentiable.
    """

    @staticmethod
    def forward(
        input,
        weight,
        scale_input,
        scale_weight,
        scale_grad,
        fp8_fwd_dtype,
        fp8_bwd_dtype,
        fp8_fwd_max,
        fp8_bwd_max,
        gran_value,
        backend_value,
    ):
        out_dtype = input.dtype
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])

        scale_in_narrow = scale_input.clamp(max=torch.finfo(out_dtype).max).to(out_dtype)
        a_fp8 = (input_2d * scale_in_narrow).clamp(-fp8_fwd_max, fp8_fwd_max).to(fp8_fwd_dtype)
        a_scale_inv = (1.0 / scale_input).float()

        scale_w_narrow = scale_weight.clamp(max=torch.finfo(out_dtype).max).to(out_dtype)
        w_fp8 = (weight * scale_w_narrow).clamp(-fp8_fwd_max, fp8_fwd_max).to(fp8_fwd_dtype)
        w_scale_inv = (1.0 / scale_weight).float()

        output = gemm_fp8_impl(
            a_fp8,
            a_scale_inv,
            False,
            w_fp8,
            w_scale_inv,
            True,
            out_dtype,
            False,
            granularity=gran_value,
            default_backend=backend_value,
        )
        output = output.reshape(*orig_shape[:-1], output.shape[-1])

        input_amax = input_2d.detach().abs().amax().float()
        weight_amax = weight.detach().abs().amax().float()

        a_t_fp8 = a_fp8.t().contiguous()
        w_t_fp8 = w_fp8.t().contiguous()

        return (output, a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv, input_amax, weight_amax)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            input,
            weight,
            scale_input,
            scale_weight,
            scale_grad,
            fp8_fwd_dtype,
            fp8_bwd_dtype,
            fp8_fwd_max,
            fp8_bwd_max,
            gran_value,
            backend_value,
        ) = inputs
        (output_val, a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv, input_amax, weight_amax) = output

        ctx.save_for_backward(a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv, scale_grad)
        ctx.mark_non_differentiable(a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv, input_amax, weight_amax)
        ctx.out_dtype = input.dtype
        ctx.orig_shape = input.shape
        ctx.fp8_bwd_dtype = fp8_bwd_dtype
        ctx.fp8_bwd_max = fp8_bwd_max
        ctx.gran_value = gran_value
        ctx.backend_value = backend_value

    @staticmethod
    def backward(ctx, grad_output, *_):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        a_t_fp8, a_scale_inv, w_t_fp8, w_scale_inv, scale_grad = ctx.saved_tensors

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        sg_narrow = scale_grad.clamp(max=torch.finfo(ctx.out_dtype).max).to(ctx.out_dtype)
        grad_fp8 = (grad_2d * sg_narrow).clamp(-ctx.fp8_bwd_max, ctx.fp8_bwd_max).to(ctx.fp8_bwd_dtype)
        grad_scale_inv = (1.0 / scale_grad).float()

        grad_input = gemm_fp8_impl(
            grad_fp8,
            grad_scale_inv,
            False,
            w_t_fp8,
            w_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )
        grad_input = grad_input.reshape(ctx.orig_shape)

        grad_t_fp8 = grad_fp8.t().contiguous()
        grad_weight = gemm_fp8_impl(
            grad_t_fp8,
            grad_scale_inv,
            False,
            a_t_fp8,
            a_scale_inv,
            True,
            ctx.out_dtype,
            False,
            granularity=ctx.gran_value,
            default_backend=ctx.backend_value,
        )

        return (grad_input, grad_weight, None, None, None, None, None, None, None, None, None)


class _ProdDelayedModule(nn.Module):
    """Drives the *production* ``DelayedFP8LinearTensorwiseFunction``.

    Unlike the earlier ``_DelayedFnModule`` (which wrapped the in-file
    ``_DelayedFP8Linear`` replica), this calls the shipped Function so a
    regression that introduces a graph break or breaks the fused amax capture
    in production is actually caught. The function writes the current amaxes
    in-place into the ``staged_*`` buffers during forward (fused capture), so
    no side-output ``copy_`` is needed here.

    The scale/amax buffers stay fp32 even though the module runs in bf16: the
    fused ``atomic_max`` amax kernel only supports fp32, mirroring how the
    production layers keep these buffers fp32.
    """

    def __init__(self, dim, out_dim, force_nt):
        super().__init__()
        self._force_nt = force_nt
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        for name in ("scale_input", "scale_weight", "scale_grad"):
            self.register_buffer(name, torch.tensor(1.0, dtype=torch.float32))
        for name in ("staged_input_amax", "staged_weight_amax", "staged_grad_amax"):
            self.register_buffer(name, torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = self.norm(x)
        result = DelayedFP8LinearTensorwiseFunction.apply(
            x,
            self.weight,
            self.scale_input,
            self.scale_weight,
            self.scale_grad,
            self.staged_input_amax,
            self.staged_weight_amax,
            self.staged_grad_amax,
            float8_e4m3,
            float8_e5m2,
            _GRAN_VALUE,
            _BACKEND_VALUE,
            self._force_nt,
        )
        return result[0]


def _build_prod_delayed_module(dim, out_dim, force_nt):
    """Construct a ``_ProdDelayedModule`` on CUDA/bf16 with fp32 scale/amax
    buffers (the blanket ``.to(bfloat16)`` would otherwise downcast them and
    trip the fp32-only fused amax kernel)."""
    model = _ProdDelayedModule(dim, out_dim, force_nt).to(dtype=torch.bfloat16, device="cuda")
    for name in (
        "scale_input",
        "scale_weight",
        "scale_grad",
        "staged_input_amax",
        "staged_weight_amax",
        "staged_grad_amax",
    ):
        model._buffers[name] = model._buffers[name].float()
    return model


class _DelayedFP8E2EModule(nn.Module):
    """Full delayed scaling module with scale update loop for e2e tests."""

    def __init__(self, dim, out_dim, use_grad_hook=True, history_len=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self._use_grad_hook = use_grad_hook
        self._fp8_fwd_max = _FP8_FWD_MAX
        self._fp8_bwd_max = _FP8_BWD_MAX

        self.register_buffer("scale_input", torch.tensor(1.0))
        self.register_buffer("scale_weight", torch.tensor(1.0))
        self.register_buffer("scale_grad", torch.tensor(1.0))

        self.register_buffer("amax_history_input", torch.zeros(history_len))
        self.register_buffer("amax_history_weight", torch.zeros(history_len))
        self.register_buffer("amax_history_grad", torch.zeros(history_len))

        self.register_buffer("staged_input_amax", torch.tensor(0.0))
        self.register_buffer("staged_weight_amax", torch.tensor(0.0))
        self.register_buffer("staged_grad_amax", torch.tensor(0.0))

        self._history_idx = 0

    def update_scales(self):
        idx = self._history_idx
        self.amax_history_input[idx] = self.staged_input_amax
        self.amax_history_weight[idx] = self.staged_weight_amax
        self.amax_history_grad[idx] = self.staged_grad_amax
        self._history_idx = (idx + 1) % self.amax_history_input.shape[0]

        self._compute_scale(self.scale_input, self.amax_history_input, self._fp8_fwd_max)
        self._compute_scale(self.scale_weight, self.amax_history_weight, self._fp8_fwd_max)
        self._compute_scale(self.scale_grad, self.amax_history_grad, self._fp8_bwd_max)

    def _compute_scale(self, scale_buf, history, fp8_max):
        amax = history.max()
        sf = fp8_max / amax.clamp(min=1e-12)
        sf = torch.where(amax > 0.0, sf, scale_buf)
        sf = torch.where(torch.isfinite(amax), sf, scale_buf)
        sf = sf.clamp(max=torch.finfo(torch.float32).max)
        scale_buf.fill_(sf)

    def update_grad_amax_from_weight_grad(self):
        if self.weight.grad is not None:
            self.staged_grad_amax.fill_(self.weight.grad.detach().abs().amax().float())

    def forward(self, x):
        x = self.norm(x)
        result = _DelayedFP8Linear.apply(
            x,
            self.weight,
            self.scale_input,
            self.scale_weight,
            self.scale_grad,
            float8_e4m3,
            float8_e5m2,
            self._fp8_fwd_max,
            self._fp8_bwd_max,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        output = result[0]
        self.staged_input_amax.copy_(result[5])
        self.staged_weight_amax.copy_(result[6])

        if self._use_grad_hook:
            buf = self.staged_grad_amax
            output.register_hook(lambda g, b=buf: _grad_amax_hook(g, b))
        return output


def _run_delayed_training_loop(model, inputs, n_steps, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mod = model._orig_mod if hasattr(model, "_orig_mod") else model
    losses, scale_log = [], []
    for step in range(n_steps):
        mod.update_scales()
        optimizer.zero_grad()
        out = model(inputs[step % len(inputs)])
        loss = out.sum()
        loss.backward()
        if not mod._use_grad_hook:
            mod.update_grad_amax_from_weight_grad()
        optimizer.step()
        losses.append(loss.item())
        scale_log.append(
            (
                mod.scale_input.item(),
                mod.scale_weight.item(),
                mod.scale_grad.item(),
            )
        )
    return losses, scale_log


# -- Test 0e helpers --------------------------------------------------------


class _BufferMutationModule(nn.Module):
    """Module that uses a buffer in the forward path to test mutation visibility."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.weight = nn.Parameter(torch.randn(out_dim, dim, dtype=torch.bfloat16))
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        x = self.norm(x)
        x_scaled = x * self.scale
        result = _Level1FP8Linear.apply(
            x_scaled,
            self.weight,
            float8_e4m3,
            _GRAN_VALUE,
            _BACKEND_VALUE,
        )
        return result[0]


# ---------------------------------------------------------------------------
# Test 0a: Buffer copy_() of autograd Function side-output
# ---------------------------------------------------------------------------


@requires_cuda
class TestDelayedPhase0a_BufferCopy(PrimusUT):
    """Can a compiled forward mutate a registered buffer via copy_()?

    This is the critical gate test for delayed scaling. If buffer copy_()
    works inside compiled code, amax can be routed from the autograd
    Function to the module without graph breaks.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_no_graph_break(self):
        """torch._dynamo.explain reports zero graph breaks with buffer copy_()."""
        model = _BufferCopyModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        explanation = torch._dynamo.explain(model)(x)

        assert explanation.graph_break_count == 0, (
            f"Test 0a: Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )

    def test_buffer_updated(self):
        """After compiled forward, staged_amax buffer should be non-zero."""
        model = _BufferCopyModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        compiled = torch.compile(model)
        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        assert model.staged_amax.item() == 0.0
        _ = compiled(x)
        assert model.staged_amax.item() > 0.0, "Buffer copy_() had no effect: staged_amax is still 0.0"

    def test_multistep_compiled_vs_eager(self):
        """Multi-step: compiled with buffer copy_() should track eager."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _BufferCopyModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)
        eager_losses, _ = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = _BufferCopyModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        compiled_model.load_state_dict(init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, _ = _run_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison("BufferCopy Eager", eager_losses, "BufferCopy Compiled", compiled_losses)
        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"Buffer copy_() compiled diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )


# Test 0c: Tensor hook with buffer copy_() in backward
# ---------------------------------------------------------------------------


@requires_cuda
class TestDelayedPhase0c_TensorHook(PrimusUT):
    """Does register_hook on an intermediate tensor work under compile,
    and can the hook callback mutate a registered buffer?
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_no_graph_break(self):
        """torch._dynamo.explain reports zero graph breaks with tensor hook."""
        model = _TensorHookModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        explanation = torch._dynamo.explain(model)(x)

        print(f"\nTest 0c: graph_break_count = {explanation.graph_break_count}")
        if explanation.graph_break_count > 0:
            print(f"  Break reasons: {explanation.break_reasons}")
        assert explanation.graph_break_count == 0, (
            f"Test 0c: Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )

    def test_hook_fires_and_updates_buffer(self):
        """After forward + backward, grad_amax buffer should be non-zero."""
        model = _TensorHookModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        compiled = torch.compile(model)
        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        assert model.grad_amax.item() == 0.0
        out = compiled(x)
        out.sum().backward()
        assert model.grad_amax.item() > 0.0, "Tensor hook did not fire: grad_amax is still 0.0"

    def test_multistep_compiled_vs_eager(self):
        """Multi-step: compiled with tensor hook should track eager."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _TensorHookModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(ref_model)
        eager_losses, _ = _run_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = _TensorHookModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        compiled_model.load_state_dict(init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, _ = _run_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison("TensorHook Eager", eager_losses, "TensorHook Compiled", compiled_losses)
        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"Tensor hook compiled diverged from eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 0d-fn: _DelayedFP8Linear autograd Function (isolation)
# ---------------------------------------------------------------------------


@requires_cuda
class TestDelayedProductionFunctionCompile(PrimusUT):
    """Graph-break + fused-amax-capture contract of the *production*
    ``DelayedFP8LinearTensorwiseFunction`` under torch.compile.

    This guards the shipped Function directly (both the native and forced-NT
    backward layouts), so a regression that re-introduces a graph break or
    breaks the in-place staged-amax capture fails here. Numerical
    forward/backward correctness of the same Function is covered separately in
    ``test_delayed_fp8_triton_op.py``.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_function_no_graph_break(self):
        """torch._dynamo.explain on the production module -> 0 breaks (both arms)."""
        for force_nt in (True, False):
            torch._dynamo.reset()
            model = _build_prod_delayed_module(_DIM, _OUT_DIM, force_nt)
            x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

            explanation = torch._dynamo.explain(model)(x)

            assert explanation.graph_break_count == 0, (
                f"force_nt={force_nt}: expected 0 graph breaks, got "
                f"{explanation.graph_break_count}. Reasons: {explanation.break_reasons}"
            )

    def test_amax_side_outputs_populated(self):
        """Compiled forward must capture the current amaxes into the staged
        buffers in-place (both arms)."""
        for force_nt in (True, False):
            torch._dynamo.reset()
            model = _build_prod_delayed_module(_DIM, _OUT_DIM, force_nt)
            compiled = torch.compile(model)
            x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

            assert model.staged_input_amax.item() == 0.0
            assert model.staged_weight_amax.item() == 0.0
            _ = compiled(x)
            assert (
                model.staged_input_amax.item() > 0.0
            ), f"force_nt={force_nt}: staged_input_amax not populated"
            assert (
                model.staged_weight_amax.item() > 0.0
            ), f"force_nt={force_nt}: staged_weight_amax not populated"
            # The captured amax must match the actual input/weight amax (fused
            # capture is the delayed-scaling contract, not an arbitrary write).
            xn = model.norm(x)
            expected_in = xn.detach().float().abs().amax().item()
            expected_w = model.weight.detach().float().abs().amax().item()
            assert abs(model.staged_input_amax.item() - expected_in) / max(expected_in, 1e-8) < 1e-2
            assert abs(model.staged_weight_amax.item() - expected_w) / max(expected_w, 1e-8) < 1e-2


# ---------------------------------------------------------------------------
# Test 0d-e2e: Full delayed module with scale update loop
# ---------------------------------------------------------------------------


@requires_cuda
class TestDelayedPhase0d_E2E(PrimusUT):
    """End-to-end delayed scaling: multi-step training with scale updates.

    Tests the complete interaction between eager scale updates and
    compiled forward/backward, including scale convergence and
    loss tracking against dynamic scaling.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_scale_convergence_eager(self):
        """Scales must change from initial 1.0 within 3 steps (eager).

        Uses history_len=1 (most_recent scaling) so scales respond to
        the very first recorded amax rather than being dominated by the
        16-slot history initialization.
        """
        model = _DelayedFP8E2EModule(_CONV_DIM, _CONV_OUT, history_len=1).to(
            dtype=torch.bfloat16, device="cuda"
        )
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        _, scale_log = _run_delayed_training_loop(model, inputs, 10)

        for i, (si, sw, sg) in enumerate(scale_log):
            print(f"  step {i}: scale_input={si:.4f}, " f"scale_weight={sw:.4f}, scale_grad={sg:.4f}")

        s_in_3, s_w_3, s_g_3 = scale_log[3]
        assert s_in_3 != 1.0, f"scale_input stuck at 1.0 after 3 steps"
        assert s_w_3 != 1.0, f"scale_weight stuck at 1.0 after 3 steps"
        assert s_g_3 != 1.0, f"scale_grad stuck at 1.0 after 3 steps"

        for i, (si, sw, sg) in enumerate(scale_log):
            assert math.isfinite(si) and si > 0, f"scale_input non-finite/negative at step {i}: {si}"
            assert math.isfinite(sw) and sw > 0, f"scale_weight non-finite/negative at step {i}: {sw}"
            assert math.isfinite(sg) and sg > 0, f"scale_grad non-finite/negative at step {i}: {sg}"

    def test_compiled_delayed_vs_eager_delayed(self):
        """Compiled delayed should track eager delayed over 200 steps."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        ref_model = _DelayedFP8E2EModule(_CONV_DIM, _CONV_OUT, history_len=1).to(
            dtype=torch.bfloat16, device="cuda"
        )
        init_state = _clone_state(ref_model)
        eager_losses, _ = _run_delayed_training_loop(ref_model, inputs, _N_STEPS)

        torch._dynamo.reset()
        compiled_model = _DelayedFP8E2EModule(_CONV_DIM, _CONV_OUT, history_len=1).to(
            dtype=torch.bfloat16, device="cuda"
        )
        compiled_model.load_state_dict(init_state)
        compiled_model_c = torch.compile(compiled_model)
        compiled_losses, _ = _run_delayed_training_loop(compiled_model_c, inputs, _N_STEPS)

        _print_comparison("Delayed Eager", eager_losses, "Delayed Compiled", compiled_losses)
        final_rel = abs(eager_losses[-1] - compiled_losses[-1]) / max(abs(eager_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.05, (
            f"Delayed compiled diverged from delayed eager: "
            f"eager={eager_losses[-1]:.6f}, compiled={compiled_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )

    def test_compiled_delayed_vs_compiled_dynamic(self):
        """Compiled delayed should be within 10% of compiled dynamic."""
        torch._dynamo.reset()
        inputs = _make_inputs(20, _CONV_DIM, _CONV_BATCH)

        dynamic_model = _OpaqueModule(_CONV_DIM, _CONV_OUT).to(dtype=torch.bfloat16, device="cuda")
        init_state = _clone_state(dynamic_model)
        dynamic_model_c = torch.compile(dynamic_model)
        dynamic_losses, _ = _run_training_loop(dynamic_model_c, inputs, _N_STEPS)

        torch._dynamo.reset()
        delayed_model = _DelayedFP8E2EModule(_CONV_DIM, _CONV_OUT, history_len=1).to(
            dtype=torch.bfloat16, device="cuda"
        )
        delayed_model.norm.load_state_dict(
            {"weight": init_state["norm.weight"], "bias": init_state["norm.bias"]}
        )
        delayed_model.weight.data.copy_(init_state["weight"])
        delayed_model_c = torch.compile(delayed_model)
        delayed_losses, _ = _run_delayed_training_loop(delayed_model_c, inputs, _N_STEPS)

        _print_comparison("Dynamic Compiled", dynamic_losses, "Delayed Compiled", delayed_losses)
        final_rel = abs(dynamic_losses[-1] - delayed_losses[-1]) / max(abs(dynamic_losses[-1]), 1e-12)
        print(f"\nFinal loss relative diff: {final_rel:.6f}")
        assert final_rel < 0.10, (
            f"Delayed diverged too far from dynamic: "
            f"dynamic={dynamic_losses[-1]:.6f}, "
            f"delayed={delayed_losses[-1]:.6f}, "
            f"rel_diff={final_rel:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 0e: Buffer mutation between compiled invocations
# ---------------------------------------------------------------------------


@requires_cuda
class TestDelayedPhase0e_BufferMutation(PrimusUT):
    """Verifies that buffer mutations between compiled calls are visible.

    When an eager hook mutates a registered buffer between calls to a
    compiled forward, the compiled forward must read the new value.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def setup_method(self, method):
        torch._dynamo.reset()

    def test_buffer_mutation_between_compiled_calls(self):
        """Buffer fill_() between compiled calls must change the output."""
        model = _BufferMutationModule(_DIM, _OUT_DIM).to(dtype=torch.bfloat16, device="cuda")
        compiled = torch.compile(model)
        x = torch.randn(_BATCH, _DIM, dtype=torch.bfloat16, device="cuda")

        out1 = compiled(x)
        model.scale.fill_(2.0)
        out2 = compiled(x)

        assert not torch.allclose(
            out1, out2
        ), "Buffer mutation not reflected: compiled forward cached old value"
