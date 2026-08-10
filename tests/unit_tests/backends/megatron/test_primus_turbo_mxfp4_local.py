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

        result = MXFP4LinearFunction.apply(x, w, preshuffle, False, None, 0, 0, False, False)
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
        # With mxfp4_full_pipeline_hadamard off (the default asserted here) we
        # quantize this pair without RHT -- an internally consistent no-RHT pair
        # -- whereas Primus-Turbo PR #383's FP4GemmMXFunction.backward
        # unconditionally rotates both sides. Both compute a valid grad_input
        # (RHT cancels within each consistent pair); they differ only in
        # quantization noise.
        #
        # An earlier version of this comment cited ~18.4 dB vs ~17.7 dB as
        # evidence that the no-RHT pair is the better recipe. Do not read it
        # that way: those numbers came from torch.randn operands, which have no
        # outlier channels for a rotation to spread, and a torch.ones gradient,
        # which a Hadamard maps to a maximum spike. TestMXFP4HadamardCoverage
        # re-measures on outlier-bearing tensors and finds the two recipes
        # within 0.02 dB when outliers are absent and the rotated pair ahead by
        # over 1 dB once they are present. This assertion is therefore a
        # not-broken guard, not a verdict on the recipe.
        bf16_grad_input = grad_out.float() @ w.detach().float()
        signal = (bf16_grad_input**2).mean()
        noise = ((x.grad.float() - bf16_grad_input) ** 2).mean()
        snr_db = 10 * torch.log10(signal / noise).item()
        assert snr_db > 10, (
            f"grad_input SNR {snr_db:.1f} dB vs BF16 is below the 10 dB threshold "
            f"(max abs diff vs Primus-Turbo PR #383 reference: {(x.grad - x_ref.grad).abs().max().item():.6e})"
        )


# ---------------------------------------------------------------------------
# Deterministic Hadamard (H16) rotation coverage
# ---------------------------------------------------------------------------


def _outlier_tensor(rows, cols, n_outlier=4, scale=32.0, axis="contract", seed=0):
    """Outlier-bearing tensor in contraction-last layout.

    ``axis="contract"`` scales whole columns, so the outliers lie along the
    contraction dimension and a block of 32 straddles outlier and normal
    coordinates -- the case a rotation exists to fix. ``axis="perp"`` scales
    whole rows instead, so every block is uniformly large and quantizes fine.

    Which one a GEMM sees is decided by the axis it contracts over, and the
    three GEMMs do not agree:

    ==========  ========================  ===========================
    GEMM        contracts over            damaged by
    ==========  ========================  ===========================
    Fprop       input features            channel outliers
    Dgrad       output features           channel outliers
    Wgrad       tokens                    token outliers
    ==========  ========================  ===========================

    An outlier channel is constant across tokens, so in the Wgrad operands it
    is a whole row and harmless. Scoring the Wgrad pair with channel outliers
    measures approximately nothing.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, dtype=torch.bfloat16, device="cuda")
    if scale == 1.0:
        return x
    if axis == "contract":
        x[:, torch.randperm(cols, generator=g, device="cuda")[:n_outlier]] *= scale
    else:
        x[torch.randperm(rows, generator=g, device="cuda")[:n_outlier], :] *= scale
    return x


def _pair_snr(a, b, rht_a, rht_b):
    """SNR in dB of one quantized GEMM pair against the unquantized reference.

    Scores ``a @ b.T`` -- both operands quantized rowwise, so the MX blocks run
    along the shared contraction dimension, which is the layout all three
    production GEMMs use. The FP32 product is the arbiter: rotated and
    unrotated both compute the same mathematical quantity (the rotation
    cancels), so comparing them to each other would only show that they differ.
    """
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity
    from primus_turbo.pytorch.kernels.gemm.gemm_fp4_impl import gemm_fp4_impl

    from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
        _FP4_DTYPE,
        MXFP4_PADDING_ALIGN_SIZE,
        _quantize_mxfp4_dual_op,
    )

    def quant(x, use_rht):
        rowwise, scale, _, _ = _quantize_mxfp4_dual_op(
            x,
            _FP4_DTYPE,
            MXFP4_PADDING_ALIGN_SIZE,
            False,
            False,
            use_rht,
            False,
            False,
            use_rht,
            False,
            False,
            False,
            False,
        )
        return rowwise, scale

    a_fp4, a_scale = quant(a, rht_a)
    b_fp4, b_scale = quant(b, rht_b)

    out = gemm_fp4_impl(
        a_fp4,
        a_scale,
        False,
        b_fp4,
        b_scale,
        True,
        torch.bfloat16,
        False,
        granularity=ScalingGranularity.MX_BLOCKWISE.value,
        default_backend=BackendType.HIPBLASLT.value,
        preshuffled=False,
    )

    ref = a.float() @ b.float().T
    signal = (ref**2).mean()
    noise = ((out.float() - ref) ** 2).mean()
    return (10 * torch.log10(signal / noise)).item()


def _snr_db(actual, reference):
    signal = (reference**2).mean()
    noise = ((actual.float() - reference) ** 2).mean()
    return (10 * torch.log10(signal / noise)).item()


# The two classes below are plain classes rather than PrimusUT subclasses so
# that pytest.mark.parametrize binds -- it does not on unittest.TestCase. The
# only thing PrimusUT adds is logger setup, which conftest already does
# session-wide.


class TestMXFP4HadamardCoverage:
    """Score the H16 rotation on outlier-bearing GEMM pairs.

    Calls the quantization op directly with chosen RHT flags, so none of this
    depends on the production recipe and it stays valid whichever way
    mxfp4_full_pipeline_hadamard defaults.

    Thresholds are deliberately loose. The measured benefit peaks near 1.4 dB
    (about a quarter of a bit) at moderate outlier severity and then falls back
    toward zero as severity grows further, because once the outlier channels
    carry nearly all the signal energy a global SNR stops seeing the small
    coordinates they crush. Do not add a high-severity rung expecting a larger
    number; the useful signal is that the rotation makes quality insensitive to
    outliers, not that it grows without bound.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @requires_mxfp4
    def test_rht_is_neutral_without_outliers(self):
        a = _outlier_tensor(256, 512, scale=1.0, seed=0)
        b = _outlier_tensor(512, 512, scale=1.0, seed=1)

        delta = _pair_snr(a, b, True, True) - _pair_snr(a, b, False, False)

        assert abs(delta) < 0.5, (
            f"On outlier-free operands the rotation should neither help nor "
            f"hurt, but it moved SNR by {delta:+.2f} dB."
        )

    @requires_mxfp4
    @pytest.mark.parametrize(
        "a_shape,b_shape",
        [
            pytest.param((256, 512), (512, 512), id="fprop"),
            pytest.param((256, 1024), (512, 1024), id="dgrad"),
        ],
    )
    def test_rht_recovers_snr_lost_to_channel_outliers(self, a_shape, b_shape):
        a = _outlier_tensor(*a_shape, scale=32.0, axis="contract", seed=0)
        b = _outlier_tensor(*b_shape, scale=1.0, seed=1)

        off = _pair_snr(a, b, False, False)
        on = _pair_snr(a, b, True, True)

        assert on - off > 0.5, (
            f"Rotation recovered only {on - off:+.2f} dB on channel-outlier "
            f"operands ({off:.2f} -> {on:.2f} dB)."
        )

    @requires_mxfp4
    def test_channel_outliers_do_not_damage_the_wgrad_pair(self):
        """The negative control that keeps the fixture honest.

        Wgrad contracts over tokens, so an outlier channel is a whole row of
        its operands and every block is uniformly scaled. If this ever starts
        showing a benefit, the fixture has stopped modelling the geometry and
        the positive results above are suspect.
        """
        a = _outlier_tensor(512, 256, scale=32.0, axis="perp", seed=0)
        b = _outlier_tensor(256, 256, scale=1.0, seed=1)

        delta = _pair_snr(a, b, True, True) - _pair_snr(a, b, False, False)

        assert abs(delta) < 0.5, (
            f"Channel outliers are supposed to be invisible to the Wgrad pair, "
            f"but the rotation moved SNR by {delta:+.2f} dB."
        )

    @requires_mxfp4
    def test_token_outliers_do_damage_the_wgrad_pair(self):
        a = _outlier_tensor(512, 256, scale=32.0, axis="contract", seed=0)
        b = _outlier_tensor(256, 256, scale=1.0, seed=1)

        off = _pair_snr(a, b, False, False)
        on = _pair_snr(a, b, True, True)

        assert on - off > 0.5, (
            f"Rotation recovered only {on - off:+.2f} dB on token-outlier "
            f"operands ({off:.2f} -> {on:.2f} dB)."
        )

    @requires_mxfp4
    @pytest.mark.parametrize("rht_a,rht_b", [(True, False), (False, True)])
    def test_mismatched_rht_collapses_rather_than_degrades(self, rht_a, rht_b):
        """Half-rotated pairs must fail loudly, not quietly get worse.

        The rotation is free only because H H^T = I cancels inside the dot
        product, which needs both operands to carry it. Rotate one side and the
        GEMM computes a different quantity -- in training that surfaces as a
        merely disappointing loss curve, so pin the qualitative gap here. One
        test per orientation covers every single-flag mistake across the six
        operands.
        """
        a = _outlier_tensor(256, 512, scale=32.0, axis="contract", seed=0)
        b = _outlier_tensor(512, 512, scale=1.0, seed=1)

        matched = _pair_snr(a, b, True, True)
        mismatched = _pair_snr(a, b, rht_a, rht_b)

        assert matched > 10.0, f"Matched pair should be healthy, got {matched:.2f} dB."
        assert mismatched < 0.0, (
            f"Mismatched pair (rht_a={rht_a}, rht_b={rht_b}) scored "
            f"{mismatched:.2f} dB. Below zero is expected -- an uncancelled "
            f"rotation makes the GEMM compute the wrong quantity, so anything "
            f"near the matched {matched:.2f} dB means the flag never reached "
            f"the kernel."
        )


class TestMXFP4FullPipelineHadamardGate:
    """Verify mxfp4_full_pipeline_hadamard is real in both directions."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @staticmethod
    def _run(x, w, full_pipeline_rht):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _enable_preshuffle,
        )

        out = MXFP4LinearFunction.apply(
            x, w, _enable_preshuffle(), False, None, 0, 0, False, full_pipeline_rht
        )[0]
        out.backward(torch.ones_like(out))
        return out

    @requires_mxfp4
    def test_gate_moves_exactly_the_fprop_and_dgrad_pairs(self):
        """Pin what the gate does and does not reach.

        Fprop and Dgrad must change: that is the silent-no-op check, since a
        flag that never reaches the kernel leaves the numerics untouched while
        the config reads True.

        Wgrad must NOT change, and that is not an oversight. Its pair is grad
        colwise x input colwise, both rotated unconditionally long before this
        gate existed, so the gate has nothing left to switch on there. If this
        equality ever breaks, the gate has started touching a fourth operand
        pair and the coverage claim in _quantize_input_dual is wrong.

        The equality holds only because _run seeds the backward with
        torch.ones_like, which discards the changed forward output. In a real
        network grad_weight would move too, by inheriting a different incoming
        gradient rather than a different Wgrad recipe.
        """
        torch.manual_seed(42)
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")

        results = {}
        for gate in (False, True):
            xg = x.detach().clone().requires_grad_(True)
            wg = w.detach().clone().requires_grad_(True)
            results[gate] = (self._run(xg, wg, gate), xg.grad, wg.grad)

        off_out, off_gx, off_gw = results[False]
        on_out, on_gx, on_gw = results[True]

        assert not torch.equal(off_out, on_out), "Gate did not change the Fprop output."
        assert not torch.equal(off_gx, on_gx), "Gate did not change grad_input (Dgrad)."
        assert torch.equal(off_gw, on_gw), (
            "Gate changed grad_weight, but the Wgrad operand pair is rotated "
            "unconditionally and should be untouched by it."
        )

    @requires_mxfp4
    def test_gate_on_keeps_every_gemm_pair_consistent(self):
        """The strongest check that the six flags move together.

        Runs the real autograd function with the gate on and scores all three
        outputs against unquantized references. Rotating any single operand
        without its partner leaves an uncancelled H, which drags the affected
        output far below this bar rather than nudging it.
        """
        torch.manual_seed(42)
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        out = self._run(x, w, True)

        grad_out = torch.ones_like(out).float()
        assert _snr_db(out, x.float() @ w.float().T) > 10.0, "Fprop pair is inconsistent."
        assert _snr_db(x.grad, grad_out @ w.detach().float()) > 10.0, "Dgrad pair is inconsistent."
        assert _snr_db(w.grad, grad_out.T @ x.detach().float()) > 10.0, "Wgrad pair is inconsistent."

    @requires_mxfp4
    def test_gate_on_stays_compile_clean(self):
        """The gate is a new traced argument inside the per_block compiled
        region, so it could plausibly cost a graph break or shift fusion."""
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4LinearFunction,
            _enable_preshuffle,
        )

        torch._dynamo.reset()
        torch.manual_seed(42)

        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(512, 256, dtype=torch.bfloat16, device="cuda")
        args = (x, w, _enable_preshuffle(), False, None, 0, 0, False, True)

        explanation = torch._dynamo.explain(MXFP4LinearFunction.apply)(*args)
        assert explanation.graph_break_count == 0, (
            f"Gate-on path has {explanation.graph_break_count} graph breaks. "
            f"Reasons: {explanation.break_reasons}"
        )

        eager = MXFP4LinearFunction.apply(*args)[0]
        compiled = torch.compile(MXFP4LinearFunction.apply)(*args)[0]
        assert torch.equal(eager, compiled), (
            f"Compiled gate-on output differs from eager. Max abs diff: "
            f"{(eager - compiled).abs().max().item():.6e}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_gate_rejects_the_hybrid_fp8_backward(self, monkeypatch):
        """Under the FP8 backward, two of the three rotated pairs do not exist."""
        import megatron.training.global_vars as gvars

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        monkeypatch.setattr(gvars, "_GLOBAL_ARGS", SimpleNamespace(rank=0, world_size=1))
        _pin_fp4_aiter(monkeypatch)

        config = _make_mxfp4_transformer_config()
        config.mxfp4_backward_precision = "fp8"
        config.mxfp4_full_pipeline_hadamard = True

        with pytest.raises(ValueError, match="mxfp4_backward_precision"):
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
        )(x, w, preshuffle, False, None, 0, 0, False, False)

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

        eager_result = MXFP4LinearFunction.apply(x, w, preshuffle, False, None, 0, 0, False, False)
        eager_out = eager_result[0]

        compiled_fn = torch.compile(MXFP4LinearFunction.apply)
        compiled_result = compiled_fn(x, w, preshuffle, False, None, 0, 0, False, False)
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
