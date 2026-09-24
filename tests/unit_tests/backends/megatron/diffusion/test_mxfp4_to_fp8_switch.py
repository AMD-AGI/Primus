# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the MXFP4 -> FP8 runtime precision switch.

Two tiers:

- Schedule and plan mechanics, which need neither a GPU nor Primus-Turbo. These
  cover the invariant the whole design rests on: the converted set is a pure
  function of the iteration, so ``mlperf_warmup``'s re-entry is harmless and every
  rank flips identically (a memory- or loss-driven decision would desynchronize
  ranks and *hang* the next collective rather than fail).
- Numerics and tracing, gated on MXFP4 hardware. The equivalence test is the
  load-bearing one: the runtime pre-warm check is structural only (it proves a
  distinct graph was traced, not that it computes the right thing), so proving the
  FP8 arm lands exactly on the production Float8 path is this file's job.
"""

import functools

import pytest
import torch

from tests.unit_tests.backends.megatron.conftest import requires_mxfp4


def _init_method():
    return functools.partial(torch.nn.init.xavier_uniform_)


class _FakeLinear(torch.nn.Module):
    """Stand-in for an MXFP4 linear: the schedule only touches ``_fp8_mode``."""

    def __init__(self):
        super().__init__()
        self._fp8_mode = False


def _fake_plan(num_layers, per_layer=2):
    return [(idx, [_FakeLinear() for _ in range(per_layer)]) for idx in range(num_layers)]


def _modes(plan):
    return [all(lin._fp8_mode for lin in linears) for _, linears in plan]


# ---------------------------------------------------------------------------
# Schedule: a pure function of the iteration
# ---------------------------------------------------------------------------


class TestTargetLayerCount:
    """The switch schedule, which must depend on nothing but the iteration."""

    def test_no_conversion_before_switch_iter(self):
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            target_layer_count,
        )

        for iteration in (0, 1, 599):
            assert target_layer_count(iteration, 600, 0, 57) == 0

    def test_single_boundary_converts_everything_at_once(self):
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            target_layer_count,
        )

        assert target_layer_count(600, 600, 0, 57) == 57
        assert target_layer_count(999, 600, 0, 57) == 57

    def test_missing_iteration_converts_nothing(self):
        """A None iteration must not be read as 0 and trip the switch."""
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            target_layer_count,
        )

        assert target_layer_count(None, 0, 0, 57) == 0

    def test_ramp_grows_by_rate_and_clamps(self):
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            target_layer_count,
        )

        assert target_layer_count(600, 600, 4, 57) == 4
        assert target_layer_count(601, 600, 4, 57) == 8
        assert target_layer_count(613, 600, 4, 57) == 56
        # Clamped, not run past the end.
        assert target_layer_count(614, 600, 4, 57) == 57
        assert target_layer_count(9999, 600, 4, 57) == 57

    def test_rate_at_or_above_plan_length_degenerates_to_single_boundary(self):
        """One code path has to cover both schedules."""
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            target_layer_count,
        )

        assert target_layer_count(600, 600, 57, 57) == 57
        assert target_layer_count(600, 600, 999, 57) == 57

    def test_idempotent_across_repeated_calls(self):
        """mlperf_warmup re-enters the inner chain with the *same* iteration.

        A call-counter-driven schedule would advance the ramp during warmup; this
        is the regression guard for that.
        """
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            target_layer_count,
        )

        counts = {target_layer_count(600, 600, 4, 57) for _ in range(10)}
        assert counts == {4}

    def test_order_independent(self):
        """Evaluating iterations out of order must not change the answer."""
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            target_layer_count,
        )

        forward = [target_layer_count(i, 600, 4, 57) for i in range(598, 615)]
        backward = [target_layer_count(i, 600, 4, 57) for i in reversed(range(598, 615))]
        assert forward == list(reversed(backward))


class TestSetFp8Mode:
    """Absolute assignment, so repeated application is a genuine no-op."""

    def test_converts_prefix_of_plan(self):
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            set_fp8_mode,
        )

        plan = _fake_plan(5)
        set_fp8_mode(plan, [], 2)
        assert _modes(plan) == [True, True, False, False, False]

    def test_repeated_application_is_a_no_op(self):
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            set_fp8_mode,
        )

        plan = _fake_plan(5)
        for _ in range(3):
            set_fp8_mode(plan, [], 3)
        assert _modes(plan) == [True, True, True, False, False]

    def test_reset_to_zero_restores_mxfp4(self):
        """What the pre-warm relies on to leave no trace."""
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            set_fp8_mode,
        )

        plan = _fake_plan(4)
        extras = [_FakeLinear()]
        set_fp8_mode(plan, extras, 4)
        set_fp8_mode(plan, extras, 0)
        assert _modes(plan) == [False] * 4
        assert extras[0]._fp8_mode is False

    def test_extras_convert_with_the_first_flip(self):
        """Linears outside the layer stack must never be left behind."""
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            set_fp8_mode,
        )

        plan = _fake_plan(4)
        extras = [_FakeLinear(), _FakeLinear()]
        set_fp8_mode(plan, extras, 0)
        assert all(not lin._fp8_mode for lin in extras)
        set_fp8_mode(plan, extras, 1)
        assert all(lin._fp8_mode for lin in extras)


class TestEmptyPlanDetection:
    """A model with no MXFP4 linears has to be caught, not planned around.

    This is not hypothetical: the Flux spec selection swallows ImportError when
    resolving the MXFP4 provider and silently falls back to BF16 linears, which is
    what happens whenever the installed Primus-Turbo is out of step with the
    attention API this repo expects. Without a check the run trains to completion
    in the wrong precision while reporting a successful switch.
    """

    def test_plan_is_empty_for_a_model_without_mxfp4_linears(self):
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            build_layer_plan,
        )

        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        plan, extras = build_layer_plan([model])
        assert plan == [] and extras == []

    def test_linear_class_names_reports_what_was_found(self):
        from primus.backends.megatron.patches.mxfp4_to_fp8_switch_patches import (
            _linear_class_names,
        )

        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())
        assert _linear_class_names([model]) == {"Linear"}


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _diffusion_config(**overrides):
    from primus.backends.megatron.core.models.diffusion.common.config import (
        BaseDiffusionConfig,
    )

    defaults = dict(hidden_size=256, num_attention_heads=8, num_layers=2)
    defaults.update(overrides)
    return BaseDiffusionConfig(**defaults)


class TestConfigValidation:
    """Validated before super().__post_init__() so it fails on its own terms."""

    def test_default_is_disabled(self):
        config = _diffusion_config()
        assert config.mxfp4_to_fp8_switch_iter == 0
        assert config.mxfp4_to_fp8_prewarm is True
        assert config.mxfp4_to_fp8_layers_per_iter == 0
        assert config.mxfp4_to_fp8_order == "deep_to_shallow"

    def test_switch_without_fp4_is_rejected(self):
        """Nothing to switch without MXFP4 linears, so say so rather than no-op."""
        with pytest.raises(ValueError, match="requires fp4"):
            _diffusion_config(mxfp4_to_fp8_switch_iter=600)

    def test_switch_with_fp4_is_accepted(self):
        config = _diffusion_config(fp4="mxfp4", mxfp4_to_fp8_switch_iter=600)
        assert config.mxfp4_to_fp8_switch_iter == 600
        # The switch must never set config.fp8: Megatron rejects fp4 and fp8
        # together, and the FP8 dtypes are set straight onto the module instead.
        assert not config.fp8

    def test_negative_switch_iter_is_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            _diffusion_config(mxfp4_to_fp8_switch_iter=-1)

    def test_negative_rate_is_rejected(self):
        with pytest.raises(ValueError, match="layers_per_iter must be >= 0"):
            _diffusion_config(fp4="mxfp4", mxfp4_to_fp8_layers_per_iter=-1)

    def test_unknown_order_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown mxfp4_to_fp8_order"):
            _diffusion_config(fp4="mxfp4", mxfp4_to_fp8_order="sideways")


# ---------------------------------------------------------------------------
# Numerics and tracing (MXFP4 hardware)
# ---------------------------------------------------------------------------

# Token count of the test inputs. Not arbitrary: the weight-gradient GEMM reduces
# over tokens, so this is its K. FlyDSL tensorwise FP8 requires K > 128 (its
# software pipeline needs at least two K tiles) and MX-blockwise requires
# K % 128 == 0 and K >= 256, so anything smaller makes the backend refuse the
# inputs rather than exercise the path. Production token counts are far larger,
# so this only keeps the test off a cliff production never approaches.
_TOKENS = 256


def _mxfp4_config(**overrides):
    """Build a TransformerConfig for MXFP4 linears.

    The mxfp4_* knobs are Flux-side fields that the linears read off the config
    with ``getattr``, so they are not TransformerConfig constructor arguments.
    Anything not a declared field is attached afterwards, which is how the real
    FluxConfig presents them.
    """
    import dataclasses

    from megatron.core.transformer.transformer_config import TransformerConfig

    defaults = dict(
        hidden_size=256,
        num_attention_heads=8,
        num_layers=1,
        params_dtype=torch.bfloat16,
        fp4="mxfp4",
        fp4_recipe="mxfp4",
    )
    declared = {f.name for f in dataclasses.fields(TransformerConfig)}
    extras = {k: v for k, v in overrides.items() if k not in declared}
    defaults.update({k: v for k, v in overrides.items() if k in declared})

    config = TransformerConfig(**defaults)
    for key, value in extras.items():
        setattr(config, key, value)
    return config


def _fp8_config(**overrides):
    from megatron.core.transformer.transformer_config import TransformerConfig

    defaults = dict(
        hidden_size=256,
        num_attention_heads=8,
        num_layers=1,
        params_dtype=torch.bfloat16,
        fp8="hybrid",
        fp8_recipe="tensorwise",
    )
    defaults.update(overrides)
    config = TransformerConfig(**defaults)
    config.fp8_scaling_strategy = "dynamic"
    config.fp8_force_nt_layout = False
    return config


def _pin_gemm_backends(monkeypatch):
    """Pin FP4 to AITER and FP8 to FlyDSL, autotune off -- the production recipe.

    FP4 must be AITER with autotune off or MXFP4 ``__init__`` fails its preshuffle
    contract. FP8 is pinned too so the switch's arm runs on the backend production
    pins rather than whatever the dispatcher would pick.

    Goes through the public setter instead of writing ``_gemm_backend`` directly,
    because the entries are version-dependent: Primus-Turbo used to store a bare
    ``BackendType`` per precision and now stores a ``BackendChoice``. Building the
    table by hand pins one shape and breaks against the other. ``monkeypatch``
    still records the original attribute, so the setter's writes are undone on
    teardown.
    """
    import os

    from primus_turbo.pytorch.core.backend import (
        BackendType,
        GlobalBackendManager,
        PrecisionType,
    )

    if os.environ.get("PRIMUS_TURBO_GEMM_BACKEND", None) == "":
        monkeypatch.delenv("PRIMUS_TURBO_GEMM_BACKEND", raising=False)

    monkeypatch.setattr(GlobalBackendManager, "_gemm_backend", None)
    monkeypatch.setattr(GlobalBackendManager, "_auto_tune", False, raising=False)
    GlobalBackendManager.set_gemm_backend(BackendType.AITER, PrecisionType.FP4)
    GlobalBackendManager.set_gemm_backend(BackendType.FLYDSL, PrecisionType.FP8)


class TestSwitchState:
    """The dormant FP8 state installed on every MXFP4 linear at construction.

    Plain pytest class rather than ``PrimusUT``: the latter is a
    ``unittest.TestCase``, into whose test methods pytest cannot inject
    function-scoped fixtures or ``parametrize`` arguments. The logger these tests
    need comes from the session-scoped autouse fixture in the megatron conftest.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        _pin_gemm_backends(monkeypatch)

    @requires_mxfp4
    def test_switch_state_defaults(self):
        from primus_turbo.pytorch.core.low_precision import (
            ScalingGranularity,
            float8_e4m3,
            float8_e5m2,
        )

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        linear = MXFP4ColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=_mxfp4_config(),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )

        assert linear._fp8_mode is False, "the switch must start dormant"
        assert linear._switch_fp8_fwd_dtype == float8_e4m3
        assert linear._switch_fp8_bwd_dtype == float8_e5m2
        assert linear._switch_fp8_gran_value == ScalingGranularity.TENSORWISE.value
        # FlyDSL handles NT/NN/TN natively, so normalizing to NT would only buy a
        # pre-transposed copy of both operands.
        assert linear._switch_force_nt is False

    @requires_mxfp4
    def test_flydsl_is_the_default_fp8_backend(self):
        from primus_turbo.pytorch.core.backend import BackendType

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        linear = MXFP4ColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=_mxfp4_config(),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        assert linear._switch_fp8_backend_value == BackendType.FLYDSL.value

    @requires_mxfp4
    def test_switch_attributes_do_not_disturb_the_mxfp4_triple(self):
        """The pure-MXFP4 path must stay bit-identical, which means its traced
        constants must not move. Reusing the backward-precision triple would."""
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        linear = MXFP4ColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=_mxfp4_config(mxfp4_backward_precision="mxfp4"),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        assert linear._fp8_bwd_dtype is None
        assert linear._fp8_gran_value == 0
        assert linear._fp8_backend_value == 0


class TestFp8ArmEquivalence:
    """The load-bearing test: the FP8 arm must be the production Float8 path.

    The runtime pre-warm check only proves a distinct graph was traced; it cannot
    catch an FP8 arm that traces its own graph and computes the wrong thing. This
    is what covers that.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        _pin_gemm_backends(monkeypatch)

    @staticmethod
    def _build_pair(mxfp4_cls, fp8_cls, input_size=256, output_size=512):
        kwargs = dict(
            input_size=input_size,
            output_size=output_size,
            init_method=_init_method(),
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        if mxfp4_cls.__name__.startswith("MXFP4Column"):
            extra_mx, extra_fp8 = {"gather_output": False}, {"gather_output": False}
        else:
            extra_mx = extra_fp8 = {"input_is_parallel": False}

        mxfp4_linear = mxfp4_cls(config=_mxfp4_config(), **kwargs, **extra_mx)
        fp8_linear = fp8_cls(config=_fp8_config(), **kwargs, **extra_fp8)

        fp8_linear.weight.data.copy_(mxfp4_linear.weight.data)
        mxfp4_linear._fp8_mode = True
        # Compare the dispatch path, not two GEMM backends: different backends can
        # legitimately differ in accumulation order, which would make a bitwise
        # claim meaningless. FlyDSL-as-default is asserted separately above.
        mxfp4_linear._switch_fp8_backend_value = fp8_linear._fp8_backend_value
        return mxfp4_linear, fp8_linear

    @requires_mxfp4
    @pytest.mark.parametrize("variant", ["column", "row"])
    def test_forward_and_backward_match_float8_linear(self, variant):
        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            Float8ColumnParallelLinear,
            Float8RowParallelLinear,
        )
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
            MXFP4RowParallelLinear,
        )

        if variant == "column":
            mxfp4_cls, fp8_cls = MXFP4ColumnParallelLinear, Float8ColumnParallelLinear
        else:
            mxfp4_cls, fp8_cls = MXFP4RowParallelLinear, Float8RowParallelLinear

        mxfp4_linear, fp8_linear = self._build_pair(mxfp4_cls, fp8_cls)

        x = torch.randn(_TOKENS, 256, dtype=torch.bfloat16, device="cuda")
        x_mx = x.clone().requires_grad_(True)
        x_fp8 = x.clone().requires_grad_(True)

        out_mx = mxfp4_linear(x_mx)[0]
        out_fp8 = fp8_linear(x_fp8)[0]
        torch.testing.assert_close(out_mx, out_fp8, rtol=0, atol=0)

        grad = torch.randn_like(out_mx)
        out_mx.backward(grad)
        out_fp8.backward(grad.clone())
        torch.testing.assert_close(x_mx.grad, x_fp8.grad, rtol=0, atol=0)
        torch.testing.assert_close(mxfp4_linear.weight.grad, fp8_linear.weight.grad, rtol=0, atol=0)

    @requires_mxfp4
    def test_flipping_the_flag_changes_the_result(self):
        """The unit-level version of the pre-warm no-op check."""
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        linear = MXFP4ColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=_mxfp4_config(),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        x = torch.randn(_TOKENS, 256, dtype=torch.bfloat16, device="cuda")

        mxfp4_out = linear(x)[0]
        linear._fp8_mode = True
        fp8_out = linear(x)[0]

        assert not torch.equal(
            mxfp4_out, fp8_out
        ), "MXFP4 and FP8 produced identical bits, which means the flag did not change the dispatch"


class TestNoGraphBreaks:
    """Both arms must trace cleanly; the switch happens under per_block compile."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        _pin_gemm_backends(monkeypatch)

    @requires_mxfp4
    @pytest.mark.parametrize("fp8_mode", [False, True])
    def test_zero_graph_breaks(self, fp8_mode):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        linear = MXFP4ColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=_mxfp4_config(),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        linear._fp8_mode = fp8_mode

        x = torch.randn(_TOKENS, 256, dtype=torch.bfloat16, device="cuda")
        explanation = torch._dynamo.explain(lambda inp: linear(inp)[0])(x)

        assert explanation.graph_break_count == 0, (
            f"_fp8_mode={fp8_mode}: expected 0 graph breaks, got "
            f"{explanation.graph_break_count}. Reasons: {explanation.break_reasons}"
        )

    @requires_mxfp4
    def test_flipping_the_flag_traces_a_new_graph(self):
        """The mechanism the pre-warm assertion depends on.

        If a bool attribute read in an ``if`` is not guarded, the switch is a
        silent no-op and pre-warm is the only thing standing between that and a
        production run that logs success while training in MXFP4.
        """
        from torch._dynamo.utils import counters

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        linear = MXFP4ColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=_mxfp4_config(),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )

        torch._dynamo.reset()
        counters.clear()
        compiled = torch.compile(lambda inp: linear(inp)[0])
        x = torch.randn(_TOKENS, 256, dtype=torch.bfloat16, device="cuda")

        compiled(x)
        after_mxfp4 = counters["stats"].get("unique_graphs", 0)

        linear._fp8_mode = True
        compiled(x)
        after_fp8 = counters["stats"].get("unique_graphs", 0)

        assert after_fp8 > after_mxfp4, (
            "flipping _fp8_mode did not fail a Dynamo guard "
            f"(unique_graphs stayed at {after_fp8}); the switch would be a no-op "
            "under torch.compile"
        )


class _TwoLinearBlock(torch.nn.Module):
    """Stand-in for a transformer block: several instances, one code object."""

    def __init__(self, config, hidden=256, ffn=512):
        super().__init__()
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
            MXFP4RowParallelLinear,
        )

        self.up = MXFP4ColumnParallelLinear(
            input_size=hidden,
            output_size=ffn,
            config=config,
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        self.down = MXFP4RowParallelLinear(
            input_size=ffn,
            output_size=hidden,
            config=config,
            init_method=_init_method(),
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            input_is_parallel=False,
        )

    def forward(self, x):
        return self.down(self.up(x)[0])[0]


class TestGraphSharing:
    """Whether the compiled-graph cost of the switch scales with layer count.

    This is what decides the architecture. If every block instance holds its own
    cache entry, flipping all of them at one iteration pays N compiles inside a
    single step, and the switch has to be spread over many iterations
    (``mxfp4_to_fp8_layers_per_iter``). If instances share entries keyed on the
    code object, the whole model can flip at one boundary for the price of one
    compile, which is the design the default config assumes.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        _pin_gemm_backends(monkeypatch)

    @requires_mxfp4
    def test_instances_share_graphs_across_the_switch(self):
        from torch._dynamo.utils import counters

        n_blocks = 4
        config = _mxfp4_config()
        blocks = [_TwoLinearBlock(config) for _ in range(n_blocks)]
        # One torch.compile per block, as torch_compile_scope=per_block does. Each
        # call returns its own wrapper, so any sharing comes from the Dynamo cache
        # being keyed on the shared forward code object, not from reusing a wrapper.
        compiled = [torch.compile(b) for b in blocks]

        torch._dynamo.reset()
        counters.clear()
        x = torch.randn(_TOKENS, 256, dtype=torch.bfloat16, device="cuda")

        for c in compiled:
            c(x)
        mxfp4_graphs = counters["stats"].get("unique_graphs", 0)

        for b in blocks:
            b.up._fp8_mode = True
            b.down._fp8_mode = True
        for c in compiled:
            c(x)
        total_graphs = counters["stats"].get("unique_graphs", 0)
        fp8_graphs = total_graphs - mxfp4_graphs

        # Two code objects per block (the block forward plus the inlined linears
        # collapse into one graph each), so the bar is "constant in n_blocks",
        # not an exact count.
        assert mxfp4_graphs < n_blocks, (
            f"{n_blocks} blocks traced {mxfp4_graphs} MXFP4 graphs; instances are "
            "not sharing cache entries, so a single-boundary switch would pay one "
            "compile per block. Use mxfp4_to_fp8_layers_per_iter to spread it."
        )
        assert fp8_graphs < n_blocks, (
            f"the flip traced {fp8_graphs} new graphs for {n_blocks} blocks; the "
            "FP8 arm is being compiled per instance."
        )
        assert fp8_graphs > 0, "the flip traced no new graph; the switch is a no-op"


class TestSavedActivationMemory:
    """The memory cost of the switch, which is the reason it is a decision at all.

    MXFP4 stores its saved operands packed two-to-a-byte plus E8M0 block scales;
    tensorwise FP8 stores a byte per element and a scalar scale. The switch
    therefore raises the recurring steady-state activation peak, and a run that
    fits in MXFP4 can fail after the flip. This pins the direction and bounds the
    size of that increase.
    """

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        _pin_gemm_backends(monkeypatch)

    @staticmethod
    def _saved_bytes_per_element(linear, x):
        """Bytes saved for backward, per element of the two GEMM operands.

        Measured as what stays allocated while the autograd graph is alive, minus
        the output, so the result isolates the saved operand encoding instead of
        being diluted by the bf16 output tensor.
        """
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        out = linear(x)[0]
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before
        saved = held - out.numel() * out.element_size()
        del out
        return saved / (x.numel() + linear.weight.numel())

    @requires_mxfp4
    def test_saved_operand_encoding_matches_the_memory_model(self):
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        linear = MXFP4ColumnParallelLinear(
            input_size=256,
            output_size=512,
            config=_mxfp4_config(),
            init_method=_init_method(),
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        )
        x = torch.randn(_TOKENS, 256, dtype=torch.bfloat16, device="cuda")

        linear._fp8_mode = False
        mxfp4_per_elem = self._saved_bytes_per_element(linear, x)
        linear._fp8_mode = True
        fp8_per_elem = self._saved_bytes_per_element(linear, x)

        # MXFP4: half a byte of packed E2M1 plus one E8M0 byte per 32-element
        # block, so 0.5 + 1/32 = 0.53125. FP8 tensorwise: one byte plus two scalar
        # scales, so ~1.0. Bounds are tight on purpose -- a saved bf16 copy
        # appearing on either arm would move these and silently invalidate the
        # activation budget the switch is planned against.
        assert 0.50 <= mxfp4_per_elem <= 0.60, (
            f"MXFP4 saved {mxfp4_per_elem:.3f} B/element, expected ~0.53 "
            "(packed FP4 plus E8M0 block scales)"
        )
        assert 0.95 <= fp8_per_elem <= 1.15, (
            f"FP8 saved {fp8_per_elem:.3f} B/element, expected ~1.0 "
            "(one byte per element plus scalar scales)"
        )
