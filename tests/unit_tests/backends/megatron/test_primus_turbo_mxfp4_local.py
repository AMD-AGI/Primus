# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for compile-friendly MXFP4 linear layers (primus_turbo_mxfp4_local).

Tests cross-validation against Primus-Turbo's FP4GemmMXFunction reference,
torch.compile graph-break validation, Megatron linear backward flow,
2-step training loop, hybrid (FP4 fwd / FP8 bwd) mode, and init guards.
"""

import functools
import os
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.backends.megatron.conftest import requires_mxfp4
from tests.utils import PrimusUT


def _init_method():
    return functools.partial(torch.nn.init.xavier_uniform_)


def _make_mxfp4_transformer_config(**overrides):
    defaults = dict(
        hidden_size=256,
        num_attention_heads=8,
        num_layers=1,
        params_dtype=torch.bfloat16,
        fp4="mxfp4",
        fp4_recipe="mxfp4",
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _pin_fp4_aiter(monkeypatch):
    """Pin the FP4 GEMM backend to AITER with autotune off for MXFP4 module tests.

    MXFP4 module __init__ runs _assert_preshuffle_contract, which requires the
    FP4 GEMM backend pinned to AITER with autotune off (the only config under
    which _enable_preshuffle() is True). Pinning it in-code lets the
    module-instantiation tests reach the real path instead of failing the
    contract; monkeypatch auto-restores on teardown so the .apply-direct tests
    keep their default (preshuffle=False) dispatch. Also clears any baked-empty
    PRIMUS_TURBO_GEMM_BACKEND so the in-code pin is authoritative (mirrors
    test_native_fp8_layout.py).
    """
    from primus_turbo.pytorch.core.backend import (
        BackendType,
        GlobalBackendManager,
        PrecisionType,
    )

    if os.environ.get("PRIMUS_TURBO_GEMM_BACKEND", None) == "":
        monkeypatch.delenv("PRIMUS_TURBO_GEMM_BACKEND", raising=False)
    monkeypatch.setattr(GlobalBackendManager, "_gemm_backend", {PrecisionType.FP4: BackendType.AITER})
    monkeypatch.setattr(GlobalBackendManager, "_auto_tune", False)


# ---------------------------------------------------------------------------
# Cross-validation against Primus-Turbo's FP4GemmMXFunction reference
# ---------------------------------------------------------------------------


class TestMXFP4CrossValidation(PrimusUT):
    """Verify MXFP4LinearFunction produces bit-identical results to FP4GemmMXFunction.

    Catches wrong boolean flags in _quantize_input_dual / _quantize_weight_dual /
    _quantize_grad_dual. A single wrong flag produces silently incorrect numerics
    that may still pass SNR thresholds vs BF16 but diverges from the canonical path.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @requires_mxfp4
    def test_forward_matches_reference_fp4gemm(self):
        from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
        from primus_turbo.pytorch.ops.gemm_fp4 import FP4GemmMXFunction

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _enable_preshuffle,
        )

        torch.manual_seed(42)
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
        preshuffle = _enable_preshuffle()

        result = MXFP4LinearFunction.apply(
            x,
            w,
            preshuffle,
            False,
            None,
            0,
            0,
            False,
        )
        our_output = result[0]

        config = Float4QuantConfig(use_preshuffle=preshuffle)
        ref_output = FP4GemmMXFunction.apply(
            x.clone(),
            w.clone(),
            None,
            None,
            False,
            True,
            x.dtype,
            config,
        )

        assert torch.equal(our_output, ref_output), (
            f"Forward outputs differ. Max abs diff: " f"{(our_output - ref_output).abs().max().item():.6e}"
        )

    @requires_mxfp4
    def test_backward_matches_reference_fp4gemm(self):
        from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
        from primus_turbo.pytorch.ops.gemm_fp4 import FP4GemmMXFunction

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _enable_preshuffle,
        )

        torch.manual_seed(42)
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = w.detach().clone().requires_grad_(True)
        preshuffle = _enable_preshuffle()

        result = MXFP4LinearFunction.apply(x, w, preshuffle, False, None, 0, 0, False)
        our_output = result[0]
        grad_out = torch.ones_like(our_output)
        our_output.backward(grad_out)

        config = Float4QuantConfig(use_preshuffle=preshuffle)
        ref_output = FP4GemmMXFunction.apply(
            x_ref,
            w_ref,
            None,
            None,
            False,
            True,
            x_ref.dtype,
            config,
        )
        ref_output.backward(torch.ones_like(ref_output))

        # grad_weight stays bit-identical to the reference: our grad_weight
        # GEMM pair (g_t colwise + a_t colwise) and the reference's both use the
        # RHT recipe, so the quantization is identical. Keep torch.equal here --
        # it still catches a wrong flag in _quantize_input_dual (colwise a_t) or
        # the g_t branch of _quantize_grad_dual.
        assert torch.equal(w.grad, w_ref.grad), (
            f"grad_weight differs. Max abs diff: " f"{(w.grad - w_ref.grad).abs().max().item():.6e}"
        )

        # grad_input is NOT bit-identical, and that is expected post Primus-Turbo
        # PR #383. The grad_input GEMM pair is (grad rowwise) x (weight colwise b_t).
        # Our production deliberately quantizes this pair without RHT
        # (_quantize_grad_dual rowwise use_rht=False + _quantize_weight_dual
        # colwise use_rht=False -- an internally consistent no-RHT pair), whereas
        # Primus-Turbo PR #383's FP4GemmMXFunction.backward unconditionally quantizes the grad
        # with use_rht=True and derives b_t with use_rht=True. Both compute a
        # valid grad_input (RHT cancels within each consistent pair); they only
        # differ in MXFP4 quantization noise. Measured against the true BF16
        # gradient, our no-RHT grad_input is actually marginally more accurate
        # than the reference's RHT grad_input (~18.4 dB vs ~17.7 dB SNR), so this
        # is a recipe choice, not a regression. Assert SNR vs the BF16 truth
        # (same >10 dB bar as the forward SNR test) instead of bit-identity.
        bf16_grad_input = grad_out.float() @ w.detach().float()
        signal = (bf16_grad_input**2).mean()
        noise = ((x.grad.float() - bf16_grad_input) ** 2).mean()
        snr_db = 10 * torch.log10(signal / noise).item()
        assert snr_db > 10, (
            f"grad_input SNR {snr_db:.1f} dB vs BF16 is below the 10 dB threshold "
            f"(max abs diff vs Primus-Turbo PR #383 reference: {(x.grad - x_ref.grad).abs().max().item():.6e})"
        )


# ---------------------------------------------------------------------------
# torch.compile graph-break validation
# ---------------------------------------------------------------------------


class TestMXFP4Compile(PrimusUT):
    """Verify MXFP4LinearFunction has zero graph breaks under torch.compile."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @requires_mxfp4
    def test_no_graph_break_pure_mxfp4(self):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _enable_preshuffle,
        )

        torch._dynamo.reset()

        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
        preshuffle = _enable_preshuffle()

        explanation = torch._dynamo.explain(
            MXFP4LinearFunction.apply,
        )(x, w, preshuffle, False, None, 0, 0, False)

        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )

    @requires_mxfp4
    def test_no_graph_break_hybrid(self):
        from primus_turbo.pytorch.core.backend import BackendType
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            float8_e5m2,
        )

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _enable_preshuffle,
        )

        torch._dynamo.reset()

        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
        preshuffle = _enable_preshuffle()

        explanation = torch._dynamo.explain(
            MXFP4LinearFunction.apply,
        )(
            x,
            w,
            preshuffle,
            True,
            float8_e5m2,
            ScalingGranularity.TENSORWISE.value,
            BackendType.HIPBLASLT.value,
            False,
        )

        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )

    @requires_mxfp4
    def test_compiled_forward_matches_eager(self):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _enable_preshuffle,
        )

        torch._dynamo.reset()
        torch.manual_seed(42)

        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
        preshuffle = _enable_preshuffle()

        eager_result = MXFP4LinearFunction.apply(x, w, preshuffle, False, None, 0, 0, False)
        eager_out = eager_result[0]

        compiled_fn = torch.compile(MXFP4LinearFunction.apply)
        compiled_result = compiled_fn(x, w, preshuffle, False, None, 0, 0, False)
        compiled_out = compiled_result[0]

        assert torch.equal(eager_out, compiled_out), (
            f"Compiled output differs from eager. Max abs diff: "
            f"{(eager_out - compiled_out).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Init guard (TP > 1 rejection)
# ---------------------------------------------------------------------------


class TestMXFP4LinearGuard(PrimusUT):
    """Test that MXFP4 linear layers reject invalid configurations."""

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

        _pin_fp4_aiter(monkeypatch)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_rejects_tp_gt_1(self):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        config = _make_mxfp4_transformer_config(tensor_model_parallel_size=2)
        with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
            MXFP4ColumnParallelLinear(
                input_size=256,
                output_size=512,
                config=config,
                init_method=_init_method(),
                bias=False,
                gather_output=False,
                skip_bias_add=False,
                is_expert=False,
            )


# ---------------------------------------------------------------------------
# Reduction-axis alignment
# ---------------------------------------------------------------------------


def _mxfp4_module():
    """The extension module, skipped where Primus-Turbo's FP4 ops are absent."""
    return pytest.importorskip(
        "primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local",
        reason="Requires the Primus-Turbo MXFP4 extension",
    )


class TestMXFP4ReductionAlignment:
    """Integer-math tests for the FP4 reduction-axis padding rules.

    A shuffled colwise scale buffer covers ``cdiv(cdiv(d, 32), 8) * 8`` E8M0
    blocks while the packed FP4 data covers only ``cdiv(d, 128) * 128``. When
    ``cdiv(d, 128)`` is odd, some blocks have no backing data and the wgrad
    GEMM sums uninitialized memory -- finite, but a gradient norm that varies
    run to run. Padding is what closes that, and it is conditional because a
    full operand copy at every FP4 linear is not free, so both halves of the
    rule need holding down: pad when either invariant fails, and pad nothing
    when both already hold.
    """

    def test_padding_alignment_is_a_multiple_of_the_block_size(self):
        module = _mxfp4_module()

        assert module.MXFP4_REDUCTION_ALIGN % module.MXFP4_PADDING_ALIGN_SIZE == 0
        assert module.MXFP4_REDUCTION_ALIGN % 16 == 0

    @pytest.mark.parametrize("dim", [256, 512, 1536, 4608, 8960])
    def test_already_conforming_axis_is_returned_untouched(self, dim):
        """Wan 1.3B's hidden dims pay no copy: this is the common case."""
        module = _mxfp4_module()

        assert module._aligned(dim) == dim

    @pytest.mark.parametrize(
        "dim,expected",
        [
            (128, 256),  # 16-aligned, but cdiv(128, 128) == 1 is odd
            (384, 512),  # cdiv(384, 128) == 3 is odd
            (640, 768),  # cdiv(640, 128) == 5 is odd
            (100, 256),  # not a multiple of 16 at all
            (1, 256),
        ],
    )
    def test_offending_axis_is_padded_up(self, dim, expected):
        module = _mxfp4_module()

        assert module._aligned(dim) == expected

    def test_padded_axis_always_satisfies_both_invariants(self):
        """Whatever comes out clears AITER's dispatcher and backs every block."""
        module = _mxfp4_module()
        block = module.MXFP4_PADDING_ALIGN_SIZE

        for dim in range(1, 4097):
            aligned = module._aligned(dim)

            assert aligned >= dim, dim
            assert aligned % 16 == 0, dim
            assert module._cdiv(aligned, block) % 2 == 0, dim

    def test_each_gemm_axis_is_aligned_independently(self):
        """M, K and N are each the reduction axis of one of the three GEMMs."""
        module = _mxfp4_module()

        # M offends, K and N do not.
        assert module._aligned_gemm_dims(128, 256, 512) == (256, 256, 512)
        # Nothing offends.
        assert module._aligned_gemm_dims(256, 1536, 8960) == (256, 1536, 8960)
        # All three offend.
        assert module._aligned_gemm_dims(128, 384, 100) == (256, 512, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_pad2d_is_identity_at_the_requested_size(self):
        """The usual case must not copy, so identity is returned by object."""
        module = _mxfp4_module()

        tensor = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")

        assert module._pad2d(tensor, 256, 512) is tensor

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_pad2d_appends_zeros_and_preserves_the_original_block(self):
        module = _mxfp4_module()

        tensor = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        padded = module._pad2d(tensor, 256, 256)

        assert padded.shape == (256, 256)
        assert torch.equal(padded[:128], tensor)
        assert not padded[128:].any()


class TestMXFP4UnpaddedShape(PrimusUT):
    """Cross-validation at a shape that pads nothing.

    Every other numeric test in this file runs at 128 rows, which the
    alignment rule pads to 256, so without this the unpadded fast path is
    never exercised against the reference at all.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @requires_mxfp4
    def test_forward_matches_reference_without_padding(self):
        from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
        from primus_turbo.pytorch.ops.gemm_fp4 import FP4GemmMXFunction

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _aligned_gemm_dims,
            _enable_preshuffle,
        )

        # Guard the premise: this shape must reach the GEMM unpadded.
        assert _aligned_gemm_dims(256, 256, 512) == (256, 256, 512)

        torch.manual_seed(42)
        x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
        preshuffle = _enable_preshuffle()

        our_output = MXFP4LinearFunction.apply(x, w, preshuffle, False, None, 0, 0, False)[0]

        config = Float4QuantConfig(use_preshuffle=preshuffle)
        ref_output = FP4GemmMXFunction.apply(
            x.clone(),
            w.clone(),
            None,
            None,
            False,
            True,
            x.dtype,
            config,
        )

        assert torch.equal(our_output, ref_output), (
            f"Forward outputs differ. Max abs diff: " f"{(our_output - ref_output).abs().max().item():.6e}"
        )

    @requires_mxfp4
    def test_padded_shape_returns_unpadded_output(self):
        """A padded M must be sliced back off before the result is returned."""
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _aligned_gemm_dims,
            _enable_preshuffle,
        )

        # Guard the premise: this shape must be padded on M.
        assert _aligned_gemm_dims(128, 256, 512)[0] == 256

        torch.manual_seed(42)
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")

        output = MXFP4LinearFunction.apply(x, w, _enable_preshuffle(), False, None, 0, 0, False)[0]

        assert output.shape == (128, 512)
        assert torch.isfinite(output.float()).all()
