# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for compile-friendly FP8 linear layers (primus_turbo_float8_local).

Tests _build_fp8_config mapping, Float8ColumnParallelLinear/Float8RowParallelLinear
construction guards, init-time dispatch to the correct autograd Function,
decomposed tensorwise quantize numerical equivalence, and torch.compile
graph-break validation.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from megatron.core.enums import Fp8Recipe
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from primus_turbo.pytorch.core.low_precision import (
    ScalingGranularity,
    float8_e4m3,
    float8_e5m2,
)

from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
    DecomposedFP8LinearTensorwiseFunction,
    Float8ColumnParallelLinear,
    Float8RowParallelLinear,
    FP8LinearBlockwiseFunction,
    OpaqueFP8LinearTensorwiseFunction,
    _build_fp8_config,
    _Float8LinearMixin,
    _quantize_fp8_tensorwise,
)
from tests.utils import PrimusUT


class TestFloat8MixinStructure:
    """Structural guards for the shared _Float8LinearMixin (no construction).

    These do not build modules, so they exercise the class layout rather than
    FP8 numerics. They catch the load-bearing footgun where the mixin is not
    listed first in the bases: in that case the Megatron base's ``_apply`` /
    ``_forward_impl`` would win and FP8 would be silently disabled. The
    init-time tests below would NOT catch that, because ``__init__`` calls
    ``self._init_fp8_state()`` explicitly regardless of MRO order.

    Note: this module imports primus_turbo (CUDA-at-import), so it is still
    gated by the module-level skip_if_no_cuda() and runs in the GPU lane.
    """

    @pytest.mark.parametrize(
        "cls, base",
        [
            (Float8ColumnParallelLinear, ColumnParallelLinear),
            (Float8RowParallelLinear, RowParallelLinear),
        ],
    )
    def test_mixin_precedes_base_in_mro(self, cls, base):
        mro = cls.__mro__
        assert _Float8LinearMixin in mro
        assert base in mro
        assert mro.index(_Float8LinearMixin) < mro.index(base), (
            f"{cls.__name__}: _Float8LinearMixin must precede {base.__name__} in the MRO "
            "or FP8 _apply/_forward_impl would be silently overridden by the base."
        )

    @pytest.mark.parametrize("cls", [Float8ColumnParallelLinear, Float8RowParallelLinear])
    def test_fp8_overrides_resolve_to_mixin(self, cls):
        assert cls._forward_impl is _Float8LinearMixin._forward_impl
        assert cls._apply is _Float8LinearMixin._apply


class TestBuildFP8Config:
    """Tests for _build_fp8_config() — no GPU, no parallel state."""

    def _make_config(self, fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise):
        return SimpleNamespace(fp8=fp8, fp8_recipe=fp8_recipe)

    def test_invalid_recipe_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            _build_fp8_config(self._make_config(fp8_recipe=Fp8Recipe.custom))


def _init_method():
    return functools.partial(torch.nn.init.xavier_uniform_)


def _make_fp8_transformer_config(**overrides):
    """Build a TransformerConfig suitable for Float8 linear layers."""
    defaults = dict(
        hidden_size=64,
        num_attention_heads=8,
        num_layers=1,
        params_dtype=torch.bfloat16,
        fp8="e4m3",
        fp8_recipe="tensorwise",
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


class TestFloat8LinearGuards(PrimusUT):
    """Tests that Float8 linear layers reject invalid configurations at init."""

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_rejects_tp_gt_1(self):
        config = _make_fp8_transformer_config(tensor_model_parallel_size=2)
        with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
            Float8ColumnParallelLinear(
                input_size=64,
                output_size=128,
                config=config,
                init_method=_init_method(),
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                is_expert=False,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_rejects_gaf(self):
        config = _make_fp8_transformer_config(gradient_accumulation_fusion=True)
        with pytest.raises(ValueError, match="gradient_accumulation_fusion"):
            Float8ColumnParallelLinear(
                input_size=64,
                output_size=128,
                config=config,
                init_method=_init_method(),
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                is_expert=False,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_rejects_sequence_parallel(self):
        # SP requires TP>1 at TransformerConfig level, so we set TP=2 to
        # pass config validation and hit our Float8 guard instead.
        config = _make_fp8_transformer_config(
            sequence_parallel=True,
            tensor_model_parallel_size=2,
        )
        with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
            Float8ColumnParallelLinear(
                input_size=64,
                output_size=128,
                config=config,
                init_method=_init_method(),
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                is_expert=False,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_rejects_fp8_none(self):
        config = _make_fp8_transformer_config(fp8=None)
        with pytest.raises(ValueError, match="config.fp8"):
            Float8ColumnParallelLinear(
                input_size=64,
                output_size=128,
                config=config,
                init_method=_init_method(),
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                is_expert=False,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_row_parallel_rejects_fp8_none(self):
        config = _make_fp8_transformer_config(fp8=None)
        with pytest.raises(ValueError, match="config.fp8"):
            Float8RowParallelLinear(
                input_size=64,
                output_size=128,
                config=config,
                init_method=_init_method(),
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                is_expert=False,
            )


class TestFloat8LinearInit(PrimusUT):
    """Tests successful construction and init-time dispatch."""

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_dispatch_tensorwise_uses_opaque(self):
        config = _make_fp8_transformer_config(fp8="e4m3", fp8_recipe="tensorwise")
        layer = Float8ColumnParallelLinear(
            input_size=64,
            output_size=128,
            config=config,
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        assert layer._fp8_fn is OpaqueFP8LinearTensorwiseFunction

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_dispatch_blockwise(self):
        config = _make_fp8_transformer_config(fp8="e4m3", fp8_recipe="blockwise")
        layer = Float8ColumnParallelLinear(
            input_size=64,
            output_size=128,
            config=config,
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        assert layer._fp8_fn is FP8LinearBlockwiseFunction


class TestQuantizeFP8Tensorwise(PrimusUT):
    """Verify _quantize_fp8_tensorwise correctness and properties."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    def _reference_quantize(self, x, fp8_dtype, fp8_max):
        """Pure-PyTorch reference (known correct): compute in FP32."""
        x_f32 = x.float()
        amax = x_f32.abs().amax()
        scale = fp8_max / amax.clamp(min=1e-12)
        x_fp8 = (x_f32 * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        scale_inv = 1.0 / scale
        return x_fp8, scale_inv

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_e4m3_correctness(self):
        """BF16-scale quantize should match FP32 reference within a few percent
        of elements (boundary rounding from BF16 scale precision)."""
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        fp8_dtype = float8_e4m3
        fp8_max = torch.finfo(fp8_dtype).max

        decomp_fp8, decomp_sinv = _quantize_fp8_tensorwise(x, fp8_dtype, fp8_max)
        ref_fp8, ref_sinv = self._reference_quantize(x, fp8_dtype, fp8_max)

        mismatch = (decomp_fp8.to(torch.float32) != ref_fp8.to(torch.float32)).float().mean()
        assert mismatch < 0.05, f"More than 5% of elements differ: {mismatch:.4f}"
        assert decomp_sinv.dtype == torch.float32
        torch.testing.assert_close(decomp_sinv, ref_sinv, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_e5m2_correctness(self):
        """BF16-scale quantize should match FP32 reference within 1 FP8 ULP."""
        x = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda")
        fp8_dtype = float8_e5m2
        fp8_max = torch.finfo(fp8_dtype).max

        decomp_fp8, decomp_sinv = _quantize_fp8_tensorwise(x, fp8_dtype, fp8_max)
        ref_fp8, ref_sinv = self._reference_quantize(x, fp8_dtype, fp8_max)

        mismatch = (decomp_fp8.to(torch.float32) != ref_fp8.to(torch.float32)).float().mean()
        assert mismatch < 0.02, f"More than 2% of elements differ: {mismatch:.4f}"
        assert decomp_sinv.dtype == torch.float32
        torch.testing.assert_close(decomp_sinv, ref_sinv, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_dequant_roundtrip(self):
        """Verify quantize -> dequantize preserves values within FP8 precision."""
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        fp8_dtype = float8_e4m3
        fp8_max = torch.finfo(fp8_dtype).max

        x_fp8, scale_inv = _quantize_fp8_tensorwise(x, fp8_dtype, fp8_max)
        x_recon = x_fp8.float() * scale_inv

        rel_err = ((x_recon - x.float()).abs() / x.float().abs().clamp(min=1e-12)).mean()
        assert rel_err < 0.1, f"Mean relative error {rel_err:.4f} too high for e4m3"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_zero_tensor_no_nan(self):
        x = torch.zeros(32, 64, dtype=torch.bfloat16, device="cuda")
        fp8_dtype = float8_e4m3
        fp8_max = torch.finfo(fp8_dtype).max

        fp8_out, sinv = _quantize_fp8_tensorwise(x, fp8_dtype, fp8_max)
        assert not torch.isnan(sinv).any(), "scale_inv should not be NaN for zero input"
        assert not torch.isinf(sinv).any(), "scale_inv should not be Inf for zero input"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_near_zero_no_nan(self):
        """Tensors with very small amax should not produce NaN or Inf in output."""
        x = torch.randn(32, 64, dtype=torch.bfloat16, device="cuda") * 1e-6
        fp8_dtype = float8_e4m3
        fp8_max = torch.finfo(fp8_dtype).max

        x_fp8, scale_inv = _quantize_fp8_tensorwise(x, fp8_dtype, fp8_max)
        assert not torch.isnan(x_fp8.float()).any(), "FP8 output has NaN for near-zero input"
        assert not torch.isinf(x_fp8.float()).any(), "FP8 output has Inf for near-zero input"
        assert not torch.isnan(scale_inv).any(), "scale_inv has NaN for near-zero input"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_scale_bf16_overflow_guard(self):
        """When amax is tiny, FP32 scale would overflow BF16.  Clamp must prevent it."""
        x = torch.full((4, 4), 1e-8, dtype=torch.bfloat16, device="cuda")
        fp8_dtype = float8_e4m3
        fp8_max = torch.finfo(fp8_dtype).max

        x_fp8, scale_inv = _quantize_fp8_tensorwise(x, fp8_dtype, fp8_max)
        assert not torch.isnan(x_fp8.float()).any(), "BF16 overflow guard failed — NaN in output"
        assert torch.isfinite(scale_inv), "scale_inv should be finite"


class TestDecomposedTensorwiseCompile(PrimusUT):
    """Verify DecomposedFP8LinearTensorwiseFunction works with torch.compile."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_no_graph_break(self):
        """torch._dynamo.explain should report zero graph breaks for the
        decomposed quantize + gemm_fp8_impl forward."""
        from primus_turbo.pytorch.core.backend import BackendType

        fp8_fwd_dtype = float8_e4m3
        fp8_bwd_dtype = float8_e5m2
        fp8_fwd_max = torch.finfo(fp8_fwd_dtype).max
        fp8_bwd_max = torch.finfo(fp8_bwd_dtype).max
        gran_value = ScalingGranularity.TENSORWISE.value
        backend_value = BackendType.HIPBLASLT.value

        x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")

        explanation = torch._dynamo.explain(
            DecomposedFP8LinearTensorwiseFunction.apply,
        )(x, w, fp8_fwd_dtype, fp8_bwd_dtype, fp8_fwd_max, fp8_bwd_max, gran_value, backend_value)
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )


class TestOpaqueTensorwiseCompile(PrimusUT):
    """Verify the refactored OpaqueFP8LinearTensorwiseFunction has zero graph breaks."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_no_graph_break(self):
        """torch._dynamo.explain should report zero graph breaks for the
        setup_context-based OpaqueFP8LinearTensorwiseFunction forward."""
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.ops.quantization import quantize_fp8

        torch._dynamo.reset()

        fp8_fwd_dtype = float8_e4m3
        fp8_bwd_dtype = float8_e5m2
        gran_value = ScalingGranularity.TENSORWISE.value
        backend_value = BackendType.HIPBLASLT.value

        x = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
        w_fp8, w_scale = quantize_fp8(w, fp8_fwd_dtype, ScalingGranularity.TENSORWISE)

        explanation = torch._dynamo.explain(
            OpaqueFP8LinearTensorwiseFunction.apply,
        )(x, w, w_fp8, w_scale, fp8_fwd_dtype, fp8_bwd_dtype, gran_value, backend_value)
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )
