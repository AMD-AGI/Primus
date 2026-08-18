# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the in-place MXFP4 -> BF16 switch.

What is actually at risk here is the trigger arithmetic, not the flag flip. The
switch exists so that "MXFP4 for N iterations, then BF16" can be one contiguous
run instead of a checkpoint-and-resume pair, and the whole point is for the two
to be comparable: the fastest measured route (N=12,288, gate reached near
iteration 16,000, 4.75 h) was measured as a resume. Off-by-one in the trigger
would silently shift the MXFP4 phase by a step and make the inline run a
measurement of something slightly different, which no loss curve would flag.

Megatron's loop is `while iteration < train_iters`, and it passes `iteration`
into train_step before incrementing, so `iteration` is the count of *completed*
steps. Resuming from a checkpoint written at step N starts with N completed
steps, so the inline switch must fire when iteration == N, leaving iterations
1..N in MXFP4. That equivalence is what these tests pin down.
"""

import types
import unittest
from unittest import mock

import pytest
import torch

from primus.backends.megatron.core.extensions.mxfp4_inline_switch import (
    apply_switch,
    apply_switch_if_due,
    is_switched,
    reset_switch_state,
)


def _fake_module(quantizing=True):
    m = types.SimpleNamespace()
    if quantizing:
        m._mxfp4_enabled = True
    return m


class _FakeChunk:
    """Stands in for a model chunk: `modules()` is all the switch traverses."""

    def __init__(self, modules):
        self._modules_list = modules

    def modules(self):
        return iter(self._modules_list)


def _model(n_mxfp4=4, n_other=2):
    mods = [_fake_module() for _ in range(n_mxfp4)] + [_fake_module(quantizing=False) for _ in range(n_other)]
    return [_FakeChunk(mods)], mods


class TestInlineSwitchTrigger(unittest.TestCase):
    def setUp(self):
        reset_switch_state()

    def tearDown(self):
        reset_switch_state()

    def test_the_step_before_the_trigger_is_still_mxfp4(self):
        model, mods = _model()
        cfg = types.SimpleNamespace(mxfp4_switch_iter=12288)
        self.assertEqual(apply_switch_if_due(model, cfg, 12287), 0)
        self.assertTrue(all(m._mxfp4_enabled for m in mods[:4]))
        self.assertFalse(is_switched())

    def test_fires_once_the_configured_number_of_steps_is_complete(self):
        model, mods = _model()
        cfg = types.SimpleNamespace(mxfp4_switch_iter=12288)
        self.assertEqual(apply_switch_if_due(model, cfg, 12288), 4)
        self.assertTrue(all(m._mxfp4_enabled is False for m in mods[:4]))
        self.assertTrue(is_switched())

    def test_a_resumed_run_past_the_trigger_still_switches(self):
        # `>=` rather than `==`: a run re-entering the loop at an iteration beyond
        # the trigger would otherwise spend its entire BF16 phase in MXFP4.
        model, _ = _model()
        cfg = types.SimpleNamespace(mxfp4_switch_iter=12288)
        self.assertEqual(apply_switch_if_due(model, cfg, 13000), 4)

    def test_zero_means_never(self):
        model, mods = _model()
        cfg = types.SimpleNamespace(mxfp4_switch_iter=0)
        self.assertEqual(apply_switch_if_due(model, cfg, 999999), 0)
        self.assertTrue(all(m._mxfp4_enabled for m in mods[:4]))
        self.assertFalse(is_switched())

    def test_second_call_is_a_no_op(self):
        model, _ = _model()
        cfg = types.SimpleNamespace(mxfp4_switch_iter=100)
        self.assertEqual(apply_switch_if_due(model, cfg, 100), 4)
        self.assertEqual(apply_switch_if_due(model, cfg, 101), 0)

    def test_layers_without_the_flag_are_left_alone(self):
        # Sensitive-layer overrides build BF16 layers that never had the attribute;
        # the switch must not invent one for them.
        model, mods = _model(n_mxfp4=1, n_other=3)
        cfg = types.SimpleNamespace(mxfp4_switch_iter=1)
        self.assertEqual(apply_switch_if_due(model, cfg, 1), 1)
        for m in mods[1:]:
            self.assertFalse(hasattr(m, "_mxfp4_enabled"))

    def test_a_model_with_no_mxfp4_linears_reports_rather_than_silently_passing(self):
        model, _ = _model(n_mxfp4=0, n_other=3)
        cfg = types.SimpleNamespace(mxfp4_switch_iter=5)
        with mock.patch(
            "primus.backends.megatron.core.extensions.mxfp4_inline_switch.log_rank_0"
        ) as logged:
            self.assertEqual(apply_switch_if_due(model, cfg, 5), 0)
        self.assertTrue(
            any("no MXFP4 linear" in str(c) for c in logged.call_args_list),
            "a switch that found nothing to switch must say so",
        )

    def test_apply_switch_accepts_a_bare_model(self):
        mods = [_fake_module(), _fake_module()]
        self.assertEqual(apply_switch(_FakeChunk(mods)), 2)


# The linear classes live in a module that imports primus_turbo, which touches
# torch.cuda at import time, so the fallback test cannot run on a CPU-only host.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="importing primus_turbo requires a GPU")
class TestMXFP4LinearFallback(unittest.TestCase):
    def _instance(self, cls):
        # Bypass __init__: it enforces MXFP4 device support and builds real
        # Megatron parallel state, none of which this test needs. Zero-arg super()
        # inside _forward_impl still resolves because the object is of the class.
        obj = object.__new__(cls)
        obj._mxfp4_enabled = False
        return obj

    def _assert_delegates_to_bf16_parent(self, cls, parent):
        from primus.backends.megatron.core.extensions import (
            primus_turbo_mxfp4_local as mx,
        )

        obj = self._instance(cls)
        sentinel = object()
        with mock.patch.object(parent, "_forward_impl", return_value=sentinel) as bf16, mock.patch.object(
            mx.MXFP4LinearFunction, "apply"
        ) as quantized:
            out = obj._forward_impl("in", "w", bias=None)

        self.assertIs(out, sentinel, "switched module did not return the parent's output")
        quantized.assert_not_called()
        self.assertEqual(bf16.call_count, 1)

    def test_column_parallel_falls_back_to_the_parent(self):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        self._assert_delegates_to_bf16_parent(MXFP4ColumnParallelLinear, ColumnParallelLinear)

    def test_row_parallel_falls_back_to_the_parent(self):
        from megatron.core.tensor_parallel.layers import RowParallelLinear

        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4RowParallelLinear,
        )

        self._assert_delegates_to_bf16_parent(MXFP4RowParallelLinear, RowParallelLinear)


class TestSwitchConfigValidation(unittest.TestCase):
    """The knobs are only useful if a typo fails loudly at config time.

    A switch step of 0 disables the feature, so a mistyped or negative value must
    not silently degrade into "never switch" -- that failure mode looks exactly
    like a normal MXFP4 run and would cost a multi-hour leg to notice.
    """

    def _config(self, **kwargs):
        from primus.backends.megatron.core.models.diffusion.common.config import (
            BaseDiffusionConfig,
        )

        return BaseDiffusionConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kwargs)

    def test_disabled_by_default(self):
        cfg = self._config()
        self.assertEqual(cfg.mxfp4_switch_iter, 0)
        self.assertEqual(cfg.mxfp4_switch_precision, "bf16")

    def test_accepts_the_measured_champion_step(self):
        self.assertEqual(self._config(mxfp4_switch_iter=12288).mxfp4_switch_iter, 12288)

    def test_rejects_a_negative_step(self):
        with self.assertRaises(ValueError):
            self._config(mxfp4_switch_iter=-1)

    def test_rejects_a_landing_precision_the_switch_cannot_deliver(self):
        # The fallback is the BF16 parent path; an FP8 landing would need its own
        # quantizer state, so accepting the string here would switch to BF16 while
        # the config and the logs claimed FP8.
        with self.assertRaises(ValueError):
            self._config(mxfp4_switch_precision="fp8")


if __name__ == "__main__":
    unittest.main()
