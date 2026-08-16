# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for MXFP4 forward-path stochastic rounding.

The MXFP4 quantizer has taken a stochastic-rounding flag from the start, but only
the gradient path ever set it (mxfp4_gradient_stochastic_rounding, measured null on
Flux 12B in Issue 220 Task 3). The forward path hardcoded it off, so the one
untested knob sat behind two positional booleans nobody could reach from a config.

These tests pin the wiring rather than the numerics: that each flag reaches the
right tensor's quantizer, on both rowwise and colwise, and that the default is
still bit-for-bit the old behaviour. Whether SR on the forward path actually buys
convergence is a training question, screened separately with a debt probe.
"""

import unittest
from unittest import mock

import pytest
import torch

# primus_turbo touches torch.cuda at import time, so this module cannot even be
# imported on a CPU-only host. The tests below need no GPU work of their own.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="importing primus_turbo requires a GPU"
)

# Positions of the stochastic-rounding booleans in the _quantize_mxfp4_dual_op
# signature: (tensor, dtype, align, rowwise_2d_block, rowwise_sr, rowwise_rht,
#             colwise_2d_block, colwise_sr, colwise_rht, ...)
ROWWISE_SR = 4
COLWISE_SR = 7


class TestForwardStochasticRoundingWiring(unittest.TestCase):
    def _capture(self, fn, *args, **kwargs):
        from primus.backends.megatron.core.extensions import (
            primus_turbo_mxfp4_local as mx,
        )

        with mock.patch.object(mx, "_quantize_mxfp4_dual_op", return_value=(1, 2, 3, 4)) as op:
            fn(*args, **kwargs)
        return op.call_args.args

    @staticmethod
    def _fns():
        from primus.backends.megatron.core.extensions import (
            primus_turbo_mxfp4_local as mx,
        )

        return mx._quantize_input_dual, mx._quantize_weight_dual

    def test_input_default_is_deterministic_rounding(self):
        quantize_input, _ = self._fns()
        called = self._capture(quantize_input, object(), False)
        self.assertFalse(called[ROWWISE_SR])
        self.assertFalse(called[COLWISE_SR])

    def test_weight_default_is_deterministic_rounding(self):
        _, quantize_weight = self._fns()
        called = self._capture(quantize_weight, object(), False)
        self.assertFalse(called[ROWWISE_SR])
        self.assertFalse(called[COLWISE_SR])

    def test_input_sr_reaches_both_orientations(self):
        quantize_input, _ = self._fns()
        called = self._capture(quantize_input, object(), False, use_sr=True)
        self.assertTrue(called[ROWWISE_SR])
        self.assertTrue(called[COLWISE_SR])

    def test_weight_sr_reaches_both_orientations(self):
        _, quantize_weight = self._fns()
        called = self._capture(quantize_weight, object(), False, use_sr=True)
        self.assertTrue(called[ROWWISE_SR])
        self.assertTrue(called[COLWISE_SR])

    def test_sr_does_not_disturb_the_other_flags(self):
        # The 2d-block and Hadamard flags differ between the activation and weight
        # recipes and a cross-validation test guards their values; this only checks
        # that adding SR left them where they were.
        quantize_input, _ = self._fns()
        off = self._capture(quantize_input, object(), True)
        on = self._capture(quantize_input, object(), True, use_sr=True)
        for i, (a, b) in enumerate(zip(off, on)):
            if i in (ROWWISE_SR, COLWISE_SR):
                continue
            self.assertEqual(a, b, f"argument {i} changed when SR was enabled")


if __name__ == "__main__":
    unittest.main()
