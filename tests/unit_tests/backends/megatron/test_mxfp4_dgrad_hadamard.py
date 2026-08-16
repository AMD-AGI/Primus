# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for the MXFP4 dgrad random Hadamard transform.

The local MXFP4 spec applies the RHT to the wgrad pair (colwise activation
against colwise gradient) but not to the dgrad pair (rowwise gradient against
colwise weight), where Primus-Turbo's own FP4GemmMXFunction applies it to both.
`mxfp4_dgrad_hadamard` turns it on for the dgrad pair.

The transform only cancels if both operands of a GEMM carry it, so the flag has
to reach exactly two of the six rht arguments in the two quantizer calls and
leave the wgrad pair alone. Reaching only one of them is not a rounding
difference, it is a wrong matrix product that no loss curve would explain, so
the pairing is asserted directly here rather than left to the GPU parity test in
test_primus_turbo_mxfp4_local.py.
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

# Positions in the _quantize_mxfp4_dual_op signature: (tensor, dtype, align,
# rowwise_2d_block, rowwise_sr, rowwise_rht, colwise_2d_block, colwise_sr,
# colwise_rht, ...)
ROWWISE_RHT = 5
COLWISE_RHT = 8


class TestDgradHadamardWiring(unittest.TestCase):
    def _capture(self, name, *args, **kwargs):
        from primus.backends.megatron.core.extensions import (
            primus_turbo_mxfp4_local as mx,
        )

        with mock.patch.object(mx, "_quantize_mxfp4_dual_op", return_value=(1, 2, 3, 4)) as op:
            getattr(mx, name)(*args, **kwargs)
        return op.call_args.args

    def test_default_leaves_the_dgrad_pair_untransformed(self):
        grad = self._capture("_quantize_grad_dual", object(), False)
        weight = self._capture("_quantize_weight_dual", object(), False)
        self.assertFalse(grad[ROWWISE_RHT])
        self.assertFalse(weight[COLWISE_RHT])

    def test_flag_transforms_both_dgrad_operands(self):
        grad = self._capture("_quantize_grad_dual", object(), False, use_dgrad_rht=True)
        weight = self._capture("_quantize_weight_dual", object(), False, use_dgrad_rht=True)
        self.assertTrue(grad[ROWWISE_RHT], "dgrad's gradient operand was left untransformed")
        self.assertTrue(weight[COLWISE_RHT], "dgrad's weight operand was left untransformed")
        self.assertEqual(
            grad[ROWWISE_RHT],
            weight[COLWISE_RHT],
            "the dgrad operands disagree on the Hadamard transform, so it cannot cancel",
        )

    def test_flag_does_not_touch_the_wgrad_pair(self):
        # The wgrad GEMM multiplies the colwise gradient by the colwise
        # activation and both are always transformed; the flag must not move them.
        for use_flag in (False, True):
            grad = self._capture("_quantize_grad_dual", object(), False, use_dgrad_rht=use_flag)
            act = self._capture("_quantize_input_dual", object(), False)
            self.assertTrue(grad[COLWISE_RHT])
            self.assertTrue(act[COLWISE_RHT])

    def test_flag_does_not_touch_the_forward_pair(self):
        # Forward multiplies the rowwise activation by the rowwise weight and
        # neither is transformed, so the flag must leave both alone.
        weight = self._capture("_quantize_weight_dual", object(), False, use_dgrad_rht=True)
        act = self._capture("_quantize_input_dual", object(), False)
        self.assertFalse(weight[ROWWISE_RHT])
        self.assertFalse(act[ROWWISE_RHT])

    def test_flag_changes_nothing_else(self):
        off = self._capture("_quantize_weight_dual", object(), True)
        on = self._capture("_quantize_weight_dual", object(), True, use_dgrad_rht=True)
        for i, (a, b) in enumerate(zip(off, on)):
            if i == COLWISE_RHT:
                continue
            self.assertEqual(a, b, f"argument {i} changed when the dgrad transform was enabled")


if __name__ == "__main__":
    unittest.main()
