# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for compile-friendly MXFP6 linear layers (primus_turbo_mxfp6_local).

Covers cross-validation against Primus-Turbo's FP6GemmMXFunction reference,
gradient accuracy against BF16 truth, torch.compile graph breaks, the hybrid
MXFP6-forward / FP8-backward mode, the 256-alignment contract, and init guards.

Every shape here keeps M, N and K multiples of 256. That is not test tidiness:
MXFP6's backward GEMMs use K as an output dimension, so a 128-aligned K that a
single forward would accept fails in backward. See the module docstring of
``primus_turbo_mxfp6_local``.
"""

import functools
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.unit_tests.backends.megatron.conftest import requires_mxfp6
from tests.utils import PrimusUT

# M, N, K all multiples of 256, the MXFP6 training contract.
M, N, K = 256, 512, 256


def _init_method():
    return functools.partial(torch.nn.init.xavier_uniform_)


def _make_mxfp6_config(**overrides):
    """A BaseDiffusionConfig for the linear layers.

    BaseDiffusionConfig rather than a plain TransformerConfig because ``fp6`` and
    ``mxfp6_backward_precision`` are Primus-owned fields that only exist on the
    diffusion config, and it is a TransformerConfig subclass so the Megatron linears
    accept it unchanged.
    """
    from primus.backends.megatron.core.models.diffusion.common.config import (
        BaseDiffusionConfig,
    )

    defaults = dict(
        hidden_size=K,
        num_attention_heads=8,
        num_layers=1,
        params_dtype=torch.bfloat16,
        fp6="mxfp6",
    )
    defaults.update(overrides)
    return BaseDiffusionConfig(**defaults)


def _snr_db(got, want):
    signal = (want**2).mean()
    noise = ((got.float() - want) ** 2).mean()
    return (10 * torch.log10(signal / noise)).item()


@pytest.fixture
def megatron_global_args(monkeypatch):
    """The subset of Megatron's global args the parallel linears read at init.

    ``ColumnParallelLinear``/``RowParallelLinear`` reach for ``get_args()`` on
    construction, which the unit-test harness never populates.
    """
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


def _pure_args(fuse_wgrad_accum=False, bias=None):
    """Trailing MXFP6LinearFunction args for the pure-MXFP6 path, bias first."""
    return (bias, False, None, 0, 0, fuse_wgrad_accum)


def _hybrid_args(bias=None):
    """Trailing MXFP6LinearFunction args for the MXFP6-fwd / FP8-bwd path, bias first."""
    from primus_turbo.pytorch.core.backend import BackendType
    from primus_turbo.pytorch.core.low_precision import ScalingGranularity, float8_e5m2

    return (
        bias,
        True,
        float8_e5m2,
        ScalingGranularity.TENSORWISE.value,
        BackendType.HIPBLASLT.value,
        False,
    )


# ---------------------------------------------------------------------------
# Cross-validation against Primus-Turbo's FP6GemmMXFunction reference
# ---------------------------------------------------------------------------


class TestMXFP6CrossValidation(PrimusUT):
    """Verify MXFP6LinearFunction matches Primus-Turbo's canonical FP6 autograd op.

    Both paths quantize with the same dual packer and call the same GEMM, so the
    results should be bit-identical. This is the test that would catch a swapped
    operand or a row/column blob mix-up, which SNR-vs-BF16 assertions can miss --
    a wrong-but-consistent pairing still produces a plausible-looking number.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @requires_mxfp6
    def test_forward_matches_reference_fp6gemm(self):
        from primus_turbo.pytorch.core.low_precision import Float6QuantConfig
        from primus_turbo.pytorch.ops.gemm_fp6 import FP6GemmMXFunction

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        our_output = MXFP6LinearFunction.apply(x, w, *_pure_args())[0]
        ref_output = FP6GemmMXFunction.apply(x.clone(), w.clone(), x.dtype, Float6QuantConfig())

        assert torch.equal(our_output, ref_output), (
            "Forward output differs from the Primus-Turbo reference. Max abs diff: "
            f"{(our_output - ref_output).abs().max().item():.6e}"
        )

    @requires_mxfp6
    def test_backward_matches_reference_fp6gemm(self):
        from primus_turbo.pytorch.core.low_precision import Float6QuantConfig
        from primus_turbo.pytorch.ops.gemm_fp6 import FP6GemmMXFunction

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = w.detach().clone().requires_grad_(True)

        our_output = MXFP6LinearFunction.apply(x, w, *_pure_args())[0]
        grad_out = torch.randn_like(our_output)
        our_output.backward(grad_out)

        ref_output = FP6GemmMXFunction.apply(x_ref, w_ref, x_ref.dtype, Float6QuantConfig())
        ref_output.backward(grad_out.clone())

        # Unlike MXFP4, both gradients are bit-identical: MXFP6 has no recipe knobs, so
        # there is no way for the two paths to pick different quantizations.
        assert torch.equal(x.grad, x_ref.grad), (
            "grad_input differs from reference. Max abs diff: "
            f"{(x.grad - x_ref.grad).abs().max().item():.6e}"
        )
        assert torch.equal(w.grad, w_ref.grad), (
            "grad_weight differs from reference. Max abs diff: "
            f"{(w.grad - w_ref.grad).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Accuracy against BF16 truth
# ---------------------------------------------------------------------------


class TestMXFP6Accuracy(PrimusUT):
    """Check all three GEMM directions against the unquantized result.

    The 24 dB floor is set well under the ~28 dB measured on these shapes, but well
    above the ~18 dB an MXFP4 path scores, so it fails if MXFP6 silently degrades to
    4-bit-grade error.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    @requires_mxfp6
    def test_forward_and_backward_snr(self):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        output = MXFP6LinearFunction.apply(x, w, *_pure_args())[0]
        grad_out = torch.randn_like(output)
        output.backward(grad_out)

        xf, wf, gf = x.detach().float(), w.detach().float(), grad_out.detach().float()
        for name, got, want in (
            ("forward", output, xf @ wf.T),
            ("grad_input", x.grad, gf @ wf),
            ("grad_weight", w.grad, gf.T @ xf),
        ):
            snr = _snr_db(got, want)
            assert snr > 24, f"{name} SNR {snr:.1f} dB vs BF16 is below the 24 dB floor"

    @requires_mxfp6
    def test_accepts_3d_input(self):
        """The transformer passes [batch, seq, hidden]; the flatten must round-trip."""
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(42)
        x = torch.randn(2, M // 2, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        output = MXFP6LinearFunction.apply(x, w, *_pure_args())[0]
        assert output.shape == (2, M // 2, N)

        output.backward(torch.randn_like(output))
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape

    @requires_mxfp6
    def test_hybrid_fp8_backward_runs(self):
        """Hybrid mode keeps the MXFP6 forward bit-exact and produces usable grads.

        The FP8 backward is a different numerical path, so it only gets a loose floor;
        the point of the test is that the saved-tensor bookkeeping differs between the
        two modes (BF16 activations vs packed blobs) and that switch must not break.
        """
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        x_pure = x.detach().clone().requires_grad_(True)
        w_pure = w.detach().clone().requires_grad_(True)

        out_hybrid = MXFP6LinearFunction.apply(x, w, *_hybrid_args())[0]
        out_pure = MXFP6LinearFunction.apply(x_pure, w_pure, *_pure_args())[0]

        assert torch.equal(out_hybrid, out_pure), "hybrid mode changed the MXFP6 forward"

        grad_out = torch.randn_like(out_hybrid)
        out_hybrid.backward(grad_out)

        xf, wf, gf = x.detach().float(), w.detach().float(), grad_out.detach().float()
        assert x.grad.shape == x.shape
        assert w.grad.shape == w.shape
        for name, got, want in (
            ("grad_input", x.grad, gf @ wf),
            ("grad_weight", w.grad, gf.T @ xf),
        ):
            snr = _snr_db(got, want)
            assert snr > 15, f"hybrid {name} SNR {snr:.1f} dB vs BF16 is below the 15 dB floor"


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


class TestMXFP6Compile(PrimusUT):
    """The whole reason this module exists is to trace cleanly, so guard that."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        pass

    # PrimusUT is a unittest.TestCase, which silently ignores pytest.mark.parametrize,
    # so the two modes are spelled out rather than parametrized.
    def _assert_no_graph_break(self, args):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch._dynamo.reset()

        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        explanation = torch._dynamo.explain(MXFP6LinearFunction.apply)(x, w, *args)

        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )

    @requires_mxfp6
    def test_no_graph_break_pure(self):
        self._assert_no_graph_break(_pure_args())

    @requires_mxfp6
    def test_no_graph_break_hybrid(self):
        self._assert_no_graph_break(_hybrid_args())


    @requires_mxfp6
    def test_compiled_forward_matches_eager(self):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch._dynamo.reset()
        torch.manual_seed(42)

        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

        eager_out = MXFP6LinearFunction.apply(x, w, *_pure_args())[0]
        compiled_out = torch.compile(MXFP6LinearFunction.apply)(x, w, *_pure_args())[0]

        assert torch.equal(eager_out, compiled_out), (
            "Compiled output differs from eager. Max abs diff: "
            f"{(eager_out - compiled_out).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Module instantiation and init guards
# ---------------------------------------------------------------------------


class TestMXFP6LinearModules(PrimusUT):
    """Instantiate the real Megatron parallel linears and take a training step."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, megatron_global_args):
        pass

    def _column_linear(self, **config_overrides):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6ColumnParallelLinear,
        )

        return MXFP6ColumnParallelLinear(
            input_size=K,
            output_size=N,
            config=_make_mxfp6_config(**config_overrides),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )

    def _row_linear(self, **config_overrides):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6RowParallelLinear,
        )

        return MXFP6RowParallelLinear(
            input_size=N,
            output_size=K,
            config=_make_mxfp6_config(**config_overrides),
            init_method=_init_method(),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )

    @requires_mxfp6
    def test_column_parallel_training_step(self):
        layer = self._column_linear().cuda()
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        output, _ = layer(x)
        assert output.shape == (M, N)

        output.sum().backward()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert torch.isfinite(x.grad).all()

    @requires_mxfp6
    def test_row_parallel_training_step(self):
        layer = self._row_linear().cuda()
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        output, _ = layer(x)
        assert output.shape == (M, K)

        output.sum().backward()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    @requires_mxfp6
    def test_training_loop_decreases_loss(self):
        """The gradients must actually point downhill over a run of steps.

        The step is applied to an FP32 master copy of the weight, which is what
        Megatron's optimizer does and what makes this test meaningful. Stepping the
        BF16 parameter in place instead is vacuous: with this loss the update is ~7e-6
        relative to the weight while BF16 resolves ~8e-3, so every step rounds away and
        the loss stays bit-identical regardless of whether the gradients are correct.
        """
        torch.manual_seed(42)
        layer = self._column_linear().cuda()
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        target = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

        master = layer.weight.detach().float().clone().requires_grad_(True)
        opt = torch.optim.Adam([master], lr=3e-3)

        losses = []
        for _ in range(8):
            layer.weight.grad = None
            output, _ = layer(x)
            loss = ((output.float() - target.float()) ** 2).mean()
            loss.backward()
            master.grad = layer.weight.grad.float()
            opt.step()
            with torch.no_grad():
                layer.weight.copy_(master)
            losses.append(loss.item())

        assert all(
            b < a for a, b in zip(losses, losses[1:])
        ), f"loss did not decrease monotonically: {[round(v, 5) for v in losses]}"

    @requires_mxfp6
    def test_hybrid_backward_selected_from_config(self):
        layer = self._column_linear(mxfp6_backward_precision="fp8").cuda()
        assert layer._backward_is_fp8 is True

        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        output, _ = layer(x)
        output.sum().backward()
        assert torch.isfinite(layer.weight.grad).all()

    @requires_mxfp6
    def test_pure_backward_is_the_default(self):
        assert self._column_linear()._backward_is_fp8 is False

    @requires_mxfp6
    def test_bias_is_applied_in_the_gemm_epilogue(self):
        """Bias now rides in the A6W6 store epilogue rather than a separate pass; check it lands.

        The epilogue reads the bias out of the kernarg slot the fp6 kernel used to ignore, so
        a bias that fails to reach the kernel shows up as no shift at all rather than as an
        error. That is what this pins down.
        """
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6ColumnParallelLinear,
        )

        torch.manual_seed(42)
        layer = MXFP6ColumnParallelLinear(
            input_size=K,
            output_size=N,
            config=_make_mxfp6_config(),
            init_method=_init_method(),
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        with torch.no_grad():
            layer.bias.fill_(1.0)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        with_bias, _ = layer(x)
        with torch.no_grad():
            layer.bias.zero_()
        without_bias, _ = layer(x)

        # Both outputs are BF16, whose spacing around these magnitudes (|out| up to ~5)
        # is about 0.03, so the difference of the two roundings cannot recover 1.0
        # exactly. The tolerance is that spacing, not an accuracy claim about MXFP6. It is
        # if anything looser than needed now: the epilogue rounds once where the separate
        # add rounded twice.
        delta = (with_bias.float() - without_bias.float()).abs()
        assert torch.allclose(
            delta, torch.ones_like(delta), atol=0.05
        ), f"bias shift deviates from 1.0 by up to {(delta - 1).abs().max().item():.4f}"

    @requires_mxfp6
    def test_epilogue_bias_matches_adding_it_afterwards(self):
        """The epilogue must agree with the separate add it replaced, to within one rounding.

        These cannot be compared bitwise, and the reason is worth stating because a naive
        equality check here fails loudly on ~2% of elements. The separate add rounded the GEMM
        result to bf16 and then added, rounding twice; the epilogue adds to the fp32 accumulator
        and rounds once. Where bias nearly cancels the accumulator, the amount the old path
        discarded is many ulps *of the result*, so the bound has to be one ulp of the pre-add
        magnitude rather than of the answer. The epilogue is the more accurate of the two.
        """
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(7)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(N, dtype=torch.bfloat16, device="cuda")

        fused = MXFP6LinearFunction.apply(x, w, *_pure_args(bias=b))[0]
        unbiased = MXFP6LinearFunction.apply(x, w, *_pure_args())[0]
        separate = unbiased + b

        # The bias has to actually arrive: a dropped pointer looks like no shift, not an error.
        assert not torch.equal(fused, unbiased), "the epilogue bias never reached the kernel"

        def ulp(t):
            return torch.exp2(torch.floor(torch.log2(t.float().abs().clamp_min(1e-30))) - 7.0)

        diff = (fused.float() - separate.float()).abs()
        bound = ulp(unbiased) + ulp(separate)
        over = int((diff > bound).sum())
        assert over == 0, (
            f"{over} of {diff.numel()} elements exceed the double-rounding bound, "
            f"worst {diff.max().item():.3e}"
        )

    @requires_mxfp6
    def test_bias_gradient_comes_from_the_packer_column_sums(self):
        """The bias gradient rides along with the backward's existing quantization.

        The reference is the arrangement this replaced: the caller added the bias after
        ``.apply()`` returned, so autograd owned the reduction over ``grad_output``. Both
        reduce the same bf16 tensor, but the packer accumulates across tiles in fp32, so
        this is checked as a reduction and not for bit equality -- in practice the two
        agree exactly at these shapes, and at worst by a bf16 ulp.
        """
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(N, dtype=torch.bfloat16, device="cuda")
        upstream = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

        xg, wg, bg = (t.clone().requires_grad_() for t in (x, w, b))
        MXFP6LinearFunction.apply(xg, wg, *_pure_args(bias=bg))[0].backward(upstream)

        xr, wr, br = (t.clone().requires_grad_() for t in (x, w, b))
        out_ref = MXFP6LinearFunction.apply(xr, wr, *_pure_args())[0] + br
        out_ref.backward(upstream.clone())

        assert bg.grad.shape == br.grad.shape == (N,)
        snr = _snr_db(bg.grad, br.grad.float())
        assert snr > 35, f"bias grad diverges from the autograd reduction: {snr:.1f} dB"

        # The other two gradients must be untouched by where the bias was added.
        assert torch.equal(xg.grad, xr.grad), "grad_input changed"
        assert torch.equal(wg.grad, wr.grad), "grad_weight changed"

    @requires_mxfp6
    def test_bias_gradient_is_skipped_when_bias_needs_no_grad(self):
        """``want_col_sum=False`` when the bias is frozen, so the packer does no extra work."""
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        frozen = torch.randn(N, dtype=torch.bfloat16, device="cuda")

        MXFP6LinearFunction.apply(x, w, *_pure_args(bias=frozen))[0].sum().backward()

        assert frozen.grad is None
        assert x.grad is not None and w.grad is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_column_parallel_rejects_tp_gt_1(self):
        with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
            self._column_linear(tensor_model_parallel_size=2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_row_parallel_rejects_tp_gt_1(self):
        with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
            self._row_linear(tensor_model_parallel_size=2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_rejects_gradient_accumulation_fusion(self):
        """Megatron's knob stays rejected: it would also move the plain linears."""
        with pytest.raises(ValueError, match="mxfp6_fused_wgrad_accum=True instead"):
            self._column_linear(gradient_accumulation_fusion=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fused_wgrad_accum_is_opt_in(self):
        assert self._column_linear()._fuse_wgrad_accum is False
        assert self._column_linear(mxfp6_fused_wgrad_accum=True)._fuse_wgrad_accum is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_fused_wgrad_accum_rejects_fp8_backward(self):
        """The FP8 backward forms its wgrad with a GEMM that has no out-variant."""
        with pytest.raises(ValueError, match="mxfp6_backward_precision"):
            self._column_linear(
                mxfp6_fused_wgrad_accum=True, mxfp6_backward_precision="fp8"
            )

    @requires_mxfp6
    def test_fused_wgrad_lands_in_main_grad(self):
        """The fused wgrad must equal the unfused one and bypass the DDP hook's add.

        Megatron's hook keys off ``grad_added_to_main_grad`` to skip its ``add_``, so a
        wgrad that landed in ``main_grad`` without setting the flag would be counted
        twice, and one that set the flag without landing would be dropped.
        """
        torch.manual_seed(42)
        layer = self._column_linear(mxfp6_fused_wgrad_accum=True).cuda()
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # Stand in for what Megatron DDP attaches: a bf16 buffer pre-filled with garbage
        # a beta=0 store has to obliterate.
        layer.weight.main_grad = torch.full_like(layer.weight, 7.0)
        layer(x)[0].sum().backward()

        reference = self._column_linear().cuda()
        with torch.no_grad():
            reference.weight.copy_(layer.weight)
        reference(x)[0].sum().backward()

        assert layer.weight.grad_added_to_main_grad is True
        assert torch.equal(layer.weight.main_grad, reference.weight.grad)

    @requires_mxfp6
    def test_fused_wgrad_does_not_break_the_compiled_region(self):
        """The fusion must not cost the compiled region, which is what actually pays here.

        Traced through the module with a real ``main_grad`` rather than through
        ``MXFP6LinearFunction.apply`` on a bare tensor: dynamo decides whether to inline an
        autograd Function by tracing its backward too, and the backward's ``main_grad``
        lookup is only there to trace when the attribute exists. Getting this wrong is not
        a small loss. Breaking the region splits one compiled block into hundreds of
        fragments, and on the MBS=32 Flux 12B arm that cost 42.7 ms of eager elementwise
        work per 512 images against the ~16 ms of ``add_`` the fusion removes.
        """
        torch._dynamo.reset()
        layer = self._column_linear(mxfp6_fused_wgrad_accum=True).cuda()
        layer.weight.main_grad = torch.zeros_like(layer.weight)
        # requires_grad is load-bearing: with no autograd graph to build, dynamo never
        # traces the Function's backward and reports zero breaks whatever is in there.
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)

        explanation = torch._dynamo.explain(lambda inp: layer(inp)[0])(x)

        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}. "
            f"Reasons: {explanation.break_reasons}"
        )

    @requires_mxfp6
    def test_fused_wgrad_needs_a_bf16_main_grad(self):
        layer = self._column_linear(mxfp6_fused_wgrad_accum=True).cuda()
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

        # Matched on message rather than type: the autograd engine is free to rewrap an
        # exception raised inside a Function's backward.
        with pytest.raises((RuntimeError, TypeError), match="weight.main_grad"):
            layer(x)[0].sum().backward()

        layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
        with pytest.raises((RuntimeError, TypeError), match="main_grads_dtype"):
            layer(x)[0].sum().backward()

    @requires_mxfp6
    def test_rejects_unaligned_k(self):
        """K only 128-aligned is accepted by a lone forward but fails in backward.

        Enforced in Primus-Turbo's gemm_fp6 entry point, so the assertion fires at the
        call rather than part-way through the backward pass. Asserted here because it is
        the constraint most likely to bite someone configuring a model.
        """
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6LinearFunction,
        )

        x = torch.randn(M, 128, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, 128, dtype=torch.bfloat16, device="cuda")

        with pytest.raises((AssertionError, RuntimeError, ValueError)):
            MXFP6LinearFunction.apply(x, w, *_pure_args())[0].sum().backward()


# ---------------------------------------------------------------------------
# BaseDiffusionConfig validation
# ---------------------------------------------------------------------------


class TestMXFP6ConfigValidation:
    """The fp6 field is Primus-owned, so these cross-checks are the only guard.

    Megatron validates fp4-vs-fp8 itself but has never heard of fp6, so without these
    a config asking for both would just silently get one of them.
    """

    @staticmethod
    def _config(**overrides):
        from primus.backends.megatron.core.models.diffusion.common.config import (
            BaseDiffusionConfig,
        )

        defaults = dict(hidden_size=K, num_attention_heads=8, num_layers=1)
        defaults.update(overrides)
        return BaseDiffusionConfig(**defaults)

    def test_defaults_leave_fp6_off(self):
        config = self._config()
        assert config.fp6 is None
        assert config.mxfp6_backward_precision == "mxfp6"

    def test_accepts_mxfp6(self):
        assert self._config(fp6="mxfp6").fp6 == "mxfp6"

    def test_accepts_hybrid_backward(self):
        config = self._config(fp6="mxfp6", mxfp6_backward_precision="fp8")
        assert config.mxfp6_backward_precision == "fp8"

    def test_rejects_unknown_fp6_format(self):
        with pytest.raises(ValueError, match="Unknown fp6"):
            self._config(fp6="mxfp6_e3m2")

    def test_rejects_fp6_with_fp4(self):
        with pytest.raises(ValueError, match="cannot both be set"):
            self._config(fp6="mxfp6", fp4="mxfp4")

    def test_rejects_fp6_with_fp8(self):
        with pytest.raises(ValueError, match="cannot both be set"):
            self._config(fp6="mxfp6", fp8="e4m3")

    def test_rejects_unknown_backward_precision(self):
        with pytest.raises(ValueError, match="Unknown mxfp6_backward_precision"):
            self._config(fp6="mxfp6", mxfp6_backward_precision="mxfp4")

    def test_rejects_backward_precision_without_fp6(self):
        with pytest.raises(ValueError, match="requires fp6"):
            self._config(mxfp6_backward_precision="fp8")

    def test_rejects_fp6_with_mxfp4_to_fp8_switch(self):
        """The switch would build a zero-layer plan and silently never fire.

        The MXFP4 -> FP8 switch is a separate change. Until it lands the field does not
        exist and there is no combination to reject, so this asserts nothing rather than
        failing on the constructor.
        """
        import dataclasses

        from primus.backends.megatron.core.models.diffusion.common.config import (
            BaseDiffusionConfig,
        )

        if not any(f.name == "mxfp4_to_fp8_switch_iter" for f in dataclasses.fields(BaseDiffusionConfig)):
            pytest.skip("mxfp4_to_fp8_switch_iter not present; nothing to cross-check")

        with pytest.raises(ValueError, match="mxfp4_to_fp8_switch_iter"):
            self._config(fp6="mxfp6", mxfp4_to_fp8_switch_iter=100)


class TestMXFP6RecipeConfig:
    """The shipped Flux 12B MXFP6 recipe must actually carry the fp6 fields through.

    ``fp6`` is not a Megatron argument, so if the YAML key were dropped anywhere in the
    resolution path the run would silently train in BF16 rather than fail.
    """

    RECIPE = (
        "examples/megatron/configs/MI355X/diffusion/"
        "flux_12b_ddp_energon_schnell_resample_local_spec_mxfp6.yaml"
    )

    @staticmethod
    def _load(rel_path):
        import pathlib

        from primus.core.config.primus_config import load_primus_config
        from primus.core.utils import file_utils

        root = pathlib.Path(__file__).resolve().parents[4]
        with pytest.MonkeyPatch.context() as mp:
            # PrimusConfig.__init__ mkdir's the workspace; keep the load side-effect
            # free, matching tests/unit_tests/configs/test_example_configs.py.
            mp.setattr(file_utils, "create_path_if_not_exists", lambda *a, **k: None)
            return load_primus_config(root / rel_path, None)

    def test_recipe_carries_fp6_fields(self):
        cfg = self._load(self.RECIPE)
        pre_trainer = next(m for m in cfg.modules if m.name == "pre_trainer")
        params = pre_trainer.params

        assert getattr(params, "fp6", None) == "mxfp6"
        assert getattr(params, "mxfp6_backward_precision", None) == "mxfp6"
        # fp6 is mutually exclusive with both of these.
        assert getattr(params, "fp4", None) is None
        assert getattr(params, "fp8", None) is None
        # Megatron's knob would also move the plain linears; the MXFP6 linears reject it.
        assert getattr(params, "gradient_accumulation_fusion", True) is False
        # The MXFP6-only fused wgrad store has no beta=1 epilogue, so it is correct at one
        # microbatch per step only. This is the general recipe, whose global batch a user
        # is free to raise, so it stays off here.
        assert getattr(params, "mxfp6_fused_wgrad_accum", False) is False
        assert getattr(params, "transformer_impl", None) == "local"


# ---------------------------------------------------------------------------
# Spec provider wiring
# ---------------------------------------------------------------------------


class TestMXFP6SpecProvider:
    """Verify the provider hands back MXFP6 linears and that fp6 selects it."""

    @requires_mxfp6
    def test_provider_returns_mxfp6_linears(self):
        from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
            PrimusTurboMXFP6LocalSpecProvider,
        )
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6ColumnParallelLinear,
            MXFP6RowParallelLinear,
        )

        provider = PrimusTurboMXFP6LocalSpecProvider()
        assert provider.column_parallel_linear() is MXFP6ColumnParallelLinear
        assert provider.row_parallel_linear() is MXFP6RowParallelLinear

    @requires_mxfp6
    def test_flux_layer_spec_selects_mxfp6_backend(self):
        """fp6 on the config must route get_flux_layer_spec to the MXFP6 linears."""
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6ColumnParallelLinear,
        )
        from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
            get_flux_layer_spec,
        )

        config = SimpleNamespace(
            transformer_impl="local",
            fp4=None,
            fp6="mxfp6",
            fp8=None,
            num_joint_layers=1,
            num_single_layers=1,
            sensitive_layers_enabled=False,
            sensitive_layer_precision="bf16",
        )

        spec = get_flux_layer_spec(config)
        submodules = spec.layer_specs[0].submodules
        rendered = str(submodules)
        assert (
            MXFP6ColumnParallelLinear.__name__ in rendered
        ), "get_flux_layer_spec did not select the MXFP6 linears for fp6='mxfp6'"

    @requires_mxfp6
    def test_provider_returns_fused_mlp(self):
        from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
            PrimusTurboMXFP6LocalSpecProvider,
        )
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
            MXFP6FusedMLP,
        )

        assert PrimusTurboMXFP6LocalSpecProvider().mlp_module() is MXFP6FusedMLP

    @requires_mxfp6
    def test_flux_layer_spec_uses_fused_mlp_for_fp6(self):
        """The MXFP6 spec must actually reach the MLP, not just the linears.

        The two Flux block factories build their ``mlp`` ModuleSpec independently, so this
        is the check that neither was missed -- a spec still naming Megatron's ``MLP`` would
        train correctly and simply never fuse anything.
        """
        from megatron.core.transformer.mlp import MLP

        from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
            get_flux_layer_spec,
        )

        config = SimpleNamespace(
            transformer_impl="local",
            fp4=None,
            fp6="mxfp6",
            fp8=None,
            num_joint_layers=1,
            num_single_layers=1,
            sensitive_layers_enabled=False,
            sensitive_layer_precision="bf16",
        )

        spec = get_flux_layer_spec(config)
        # One joint block and one single block, both of which own an MLP.
        assert len(spec.layer_specs) == 2
        for layer in spec.layer_specs:
            mlp_module = layer.submodules.mlp.module
            assert mlp_module is not MLP, "layer spec still uses the unfused MLP"
            assert mlp_module.__name__ == "MXFP6FusedMLP"


# ---------------------------------------------------------------------------
# Fused MLP epilogue
# ---------------------------------------------------------------------------


def _make_fused_mlp(**config_overrides):
    """Build an MXFP6FusedMLP with the Flux MLP configuration."""
    from megatron.core.transformer.mlp import MLPSubmodules

    from primus.backends.megatron.core.extensions.primus_turbo_mxfp6_local import (
        MXFP6ColumnParallelLinear,
        MXFP6FusedMLP,
        MXFP6RowParallelLinear,
    )

    defaults = dict(
        ffn_hidden_size=512,
        add_bias_linear=True,
        gated_linear_unit=False,
        bias_activation_fusion=False,
        activation_func=functools.partial(torch.nn.functional.gelu, approximate="tanh"),
    )
    defaults.update(config_overrides)
    config = _make_mxfp6_config(**defaults)
    submodules = MLPSubmodules(linear_fc1=MXFP6ColumnParallelLinear, linear_fc2=MXFP6RowParallelLinear)
    return MXFP6FusedMLP(config, submodules).to("cuda:0")


class TestMXFP6FusedMLP(PrimusUT):
    """The fused MLP must be numerically the same module as the one it replaces.

    The fusion removes traffic, not arithmetic: the packed operands the GEMMs consume are the
    same ones either way, down to a rounding of the activation's tanh, so forward and both
    weight gradients should agree to well within MXFP6's own quantization error rather than
    merely correlate. The bias gradient is looser and is checked separately, because it comes
    from a reduction the fusion had to reorder.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, megatron_global_args):
        pass

    @pytest.fixture(autouse=True)
    def default_fused_mlp_mode(self):
        """Pin the mode to the default instead of inheriting it from the shell.

        The submission container exports PRIMUS_MXFP6_FUSED_MLP=on, which is what makes a
        fallback an error there -- so the two fallback tests below would fail inside the
        very container the fused MLP is meant to run in.
        """
        with mock.patch.dict(os.environ):
            os.environ.pop("PRIMUS_MXFP6_FUSED_MLP", None)
            yield

    @requires_mxfp6
    def test_forward_and_grads_match_stock_mlp(self):
        torch.manual_seed(0)
        fused = _make_fused_mlp()
        assert fused._fused_epilogue, "fused path unexpectedly disabled"

        # Same module, same weights, epilogue not fused: the reference is MLP.forward.
        x = torch.randn((M, 1, K), dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()

        out, out_bias = fused(x)
        # MLP.forward via the base class is exactly the path being replaced.
        from megatron.core.transformer.mlp import MLP

        ref_out, ref_bias = MLP.forward(fused, x_ref)

        assert out.shape == ref_out.shape
        assert out_bias is ref_bias  # both hand back linear_fc2.bias unadded

        snr = _snr_db(out, ref_out.float())
        assert snr > 40, f"forward diverges from the unfused MLP: {snr:.1f} dB"

        grad = torch.randn_like(out)
        out.backward(grad)
        ref_out.backward(grad.clone())

        for name, got, want in (
            ("input", x.grad, x_ref.grad),
            ("fc1.weight", fused.linear_fc1.weight.grad, None),
        ):
            if want is None:
                continue
            snr = _snr_db(got, want.float())
            assert snr > 40, f"{name} grad diverges: {snr:.1f} dB"

    @requires_mxfp6
    def test_fused_wgrad_lands_in_both_main_grads(self):
        """The fused MLP owns two weights, so both have to reach the right main_grad.

        Worth its own test rather than trusting the linear's: the two wgrads are formed
        from different operands in the same backward, so swapping them produces gradients
        that are individually plausible and jointly wrong.
        """
        torch.manual_seed(0)
        fused = _make_fused_mlp(mxfp6_fused_wgrad_accum=True)
        assert fused._fused_epilogue

        reference = _make_fused_mlp()
        with torch.no_grad():
            for name, param in fused.named_parameters():
                reference.get_parameter(name).copy_(param)

        # Stand in for what Megatron DDP attaches, pre-filled with garbage a beta=0
        # store has to obliterate.
        for linear in (fused.linear_fc1, fused.linear_fc2):
            linear.weight.main_grad = torch.full_like(linear.weight, 7.0)

        x = torch.randn((M, 1, K), dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()

        out, _ = fused(x)
        grad = torch.randn_like(out)
        out.backward(grad)

        ref_out, _ = reference(x_ref)
        ref_out.backward(grad.clone())

        for name in ("linear_fc1", "linear_fc2"):
            got = getattr(fused, name).weight
            want = getattr(reference, name).weight
            assert got.grad_added_to_main_grad is True, f"{name} did not claim main_grad"
            assert torch.equal(got.main_grad, want.grad), f"{name} wgrad differs"

    @requires_mxfp6
    def test_bias_gradient_matches_eager_reduction(self):
        """fc1's bias gradient comes from the packer's side output, not a separate sum.

        This is the one quantity the fusion does not reproduce bit-for-bit: the tensor it
        would be reduced from no longer reaches HBM, so the sum is taken over LDS tiles in
        fp32 and finished across tiles. That is a different -- and more accurate --
        summation order than a single pass over a bf16 tensor, so it is checked as a
        reduction rather than for equality.
        """
        torch.manual_seed(0)
        fused = _make_fused_mlp()
        assert fused._fused_epilogue

        from megatron.core.transformer.mlp import MLP

        x = torch.randn((M, 1, K), dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()

        out, _ = fused(x)
        grad = torch.randn_like(out)
        out.backward(grad)
        got = fused.linear_fc1.bias.grad.detach().clone()

        fused.zero_grad(set_to_none=True)
        ref_out, _ = MLP.forward(fused, x_ref)
        ref_out.backward(grad.clone())
        want = fused.linear_fc1.bias.grad

        assert got.shape == want.shape
        snr = _snr_db(got, want.float())
        assert snr > 35, f"fc1 bias grad diverges: {snr:.1f} dB"

    @requires_mxfp6
    def test_falls_back_when_activation_is_not_tanh_gelu(self):
        """An activation the prologue does not implement must disable the fusion.

        FluxConfig's *default* activation is a hand-written tanh GELU with a different
        association than ATen's, so this is the realistic misconfiguration, not a synthetic
        one. Silently fusing it would change numerics with nothing to flag it.
        """
        with pytest.warns(UserWarning, match="fused MLP epilogue disabled"):
            fused = _make_fused_mlp(activation_func=torch.nn.functional.silu)
        assert not fused._fused_epilogue

    @requires_mxfp6
    def test_falls_back_when_backward_is_fp8(self):
        """FP8 backward keeps the activation live, which the fusion removes."""
        with pytest.warns(UserWarning, match="mxfp6_backward_precision"):
            fused = _make_fused_mlp(mxfp6_backward_precision="fp8")
        assert not fused._fused_epilogue

    @requires_mxfp6
    def test_env_kill_switch_disables_and_requires(self):
        with mock.patch.dict(os.environ, {"PRIMUS_MXFP6_FUSED_MLP": "off"}):
            assert not _make_fused_mlp()._fused_epilogue

        with mock.patch.dict(os.environ, {"PRIMUS_MXFP6_FUSED_MLP": "on"}):
            assert _make_fused_mlp()._fused_epilogue
            # "on" turns an unusable configuration into an error rather than a fallback.
            with pytest.raises(RuntimeError, match="fused MLP is unusable"):
                _make_fused_mlp(activation_func=torch.nn.functional.silu)
