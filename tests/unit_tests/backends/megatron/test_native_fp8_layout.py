# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""GPU-only tests for the native (NN/TN) FP8 backward layout.

FP8 GEMMs require an AMD GPU (gfx950); the fused no-transpose cast+amax kernel
is vendored in Primus (fp8_cast_kernels_triton), so every test here is skipped
when CUDA is unavailable. Run on an AMD GPU (gfx950) with the FP8 toolchain available.

Coverage:
  - native (``fp8_force_nt_layout=False``) vs legacy forced-NT numerical
    equivalence for both forward and backward,
  - finiteness of the weight gradient (the wgrad-NaN regression that motivated
    the native layout),
  - backward gradient flow / return-count validation for both
    ``OpaqueFP8LinearTensorwiseFunction`` (9 inputs) and the full delayed-scaling
    layer (``DelayedFP8LinearTensorwiseFunction``, 13 inputs); a count mismatch
    would make ``autograd`` raise, so a successful backward validates the arity.
"""

import functools
import os
from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
    Float8ColumnParallelLinear,
    OpaqueFP8LinearTensorwiseFunction,
)
from tests.utils import PrimusUT

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Native FP8 layout exercises FP8 GEMMs, which require an AMD GPU",
)


def _init_method():
    return functools.partial(torch.nn.init.xavier_uniform_)


def _make_config(force_nt, recipe="tensorwise"):
    """TransformerConfig for an FP8 linear, with the diffusion-only
    fp8_force_nt_layout attribute attached (the layer reads it via getattr)."""
    config = TransformerConfig(
        hidden_size=64,
        num_attention_heads=8,
        num_layers=1,
        params_dtype=torch.bfloat16,
        fp8="e4m3",
        fp8_recipe=recipe,
    )
    config.fp8_force_nt_layout = force_nt
    return config


def _build_linear(force_nt, recipe="tensorwise"):
    return Float8ColumnParallelLinear(
        input_size=64,
        output_size=128,
        config=_make_config(force_nt, recipe=recipe),
        init_method=_init_method(),
        bias=False,
        gather_output=False,
        skip_bias_add=False,
        is_expert=False,
    )


def _forward_only(layer, x):
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


class _GpuLinearBase(PrimusUT):
    """Shared setup: parallel state + a minimal global-args stub for the layers."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        dummy_args = SimpleNamespace(
            rank=0,
            world_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            offload=False,
            offload_ops=[],
            patch_primus_pipeline=False,
            pp_algorithm=None,
            patch_zero_bubble=False,
            enable_zero_bubble=False,
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )
        import megatron.training.global_vars as gvars

        monkeypatch.setattr(gvars, "_GLOBAL_ARGS", dummy_args)

        # Some deploy images bake PRIMUS_TURBO_GEMM_BACKEND="" which makes
        # GlobalBackendManager raise KeyError('') on the native path. Unset it
        # here, exactly as the docs require for native-layout production runs.
        if os.environ.get("PRIMUS_TURBO_GEMM_BACKEND", None) == "":
            monkeypatch.delenv("PRIMUS_TURBO_GEMM_BACKEND", raising=False)


class TestNativeVsForcedNtLayer(_GpuLinearBase):
    """End-to-end equivalence of the native and forced-NT layouts via the layer."""

    @requires_gpu
    def test_forward_backward_match_and_finite(self):
        # NOTE: looped instead of @pytest.mark.parametrize because PrimusUT is a
        # unittest.TestCase, where pytest does not inject parametrized args.
        for recipe in ("tensorwise", "delayed"):
            with self.subTest(recipe=recipe):
                torch.manual_seed(0)

                native = _build_linear(force_nt=False, recipe=recipe).cuda()
                forced = _build_linear(force_nt=True, recipe=recipe).cuda()
                # Identical weights so the only difference is the backward GEMM layout.
                forced.load_state_dict(native.state_dict())

                assert native._force_nt is False
                assert forced._force_nt is True

                x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
                x_native = x.clone().requires_grad_(True)
                x_forced = x.clone().requires_grad_(True)

                out_native = _forward_only(native, x_native)
                out_forced = _forward_only(forced, x_forced)

                assert torch.isfinite(out_native).all(), "native forward produced non-finite output"
                torch.testing.assert_close(out_native, out_forced, atol=2e-2, rtol=2e-2)

                out_native.sum().backward()
                out_forced.sum().backward()

                # wgrad NaN regression guard + native/forced equivalence.
                assert torch.isfinite(native.weight.grad).all(), "native wgrad is non-finite (NaN regression)"
                assert torch.isfinite(x_native.grad).all(), "native dgrad is non-finite"
                torch.testing.assert_close(native.weight.grad, forced.weight.grad, atol=3e-2, rtol=3e-2)
                torch.testing.assert_close(x_native.grad, x_forced.grad, atol=3e-2, rtol=3e-2)


class TestOpaqueFunctionBackward(_GpuLinearBase):
    """Direct autograd-Function checks for OpaqueFP8LinearTensorwiseFunction."""

    @requires_gpu
    def test_native_backward_returns_grads_for_input_and_weight(self):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            float8_e4m3,
            float8_e5m2,
        )
        from primus_turbo.pytorch.ops.quantization import quantize_fp8

        gran = ScalingGranularity.TENSORWISE.value
        backend = BackendType.HIPBLASLT.value

        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w_fp8, w_scale = quantize_fp8(w, float8_e4m3, ScalingGranularity.TENSORWISE)

        # force_nt=False exercises the native arm; backward must return exactly
        # 9 grads (one per forward input) or autograd raises here.
        out = OpaqueFP8LinearTensorwiseFunction.apply(
            x, w, w_fp8, w_scale, float8_e4m3, float8_e5m2, gran, backend, False
        )
        output = out[0] if isinstance(out, tuple) else out
        output.sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert w.grad is not None and torch.isfinite(w.grad).all()
