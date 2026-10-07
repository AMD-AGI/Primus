###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU regressions for selectable FLUX paths; GPU kernel validation is separate."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from primus.backends.diffusion.models.flux import layers
from primus.backends.diffusion.models.quantization import mxfp4_linear


class FluxPortTest(unittest.TestCase):
    def test_mlp_fallback_outputs_and_gradients(self):
        torch.manual_seed(13)
        mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(approximate="tanh"), nn.Linear(64, 32))
        x = torch.randn(2, 3, 32, requires_grad=True)
        gate = torch.randn(2, 1, 32, requires_grad=True)
        expected = gate * mlp(x)
        expected_grads = torch.autograd.grad(expected.sum(), (x, gate, *mlp.parameters()))
        # Stock Linear and non-MXFP4 layers must keep working with the recipe's
        # fusion flags enabled, including during the H16 stock ablation.
        with patch.object(layers, "FUSED_DGELU_MLP", True), patch.object(layers, "FWD_GELU_PACK", True):
            actual = layers._mlp_gelu_fused(mlp, x, gate)
        actual_grads = torch.autograd.grad(actual.sum(), (x, gate, *mlp.parameters()))
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_gate_opt_in(self):
        class Linear(nn.Linear):
            def forward_gated(self, x, gate):
                self.used_fusion = True
                return gate * self(x)

        linear = Linear(32, 32)
        x, gate = torch.randn(2, 32), torch.randn(2, 32)
        for enabled in (False, True):
            linear.used_fusion = False
            with patch.object(layers, "GATE_DGATE", enabled):
                actual = layers._gated_linear(linear, x, gate)
            self.assertEqual(linear.used_fusion, enabled)
            torch.testing.assert_close(actual, gate * linear(x))

    def test_bf16_eval_guard_preserves_training_dispatch(self):
        implementation = mxfp4_linear._implementation
        bf16 = mxfp4_linear.MXFP4LinearConfig(fprop="bf16", dgrad="bf16", wgrad="bf16")
        linear = mxfp4_linear.MXFP4Linear(32, 32, config=bf16)
        # Avoid initializing GPU backends: dispatch is mocked, while eval's
        # real Linear computation and gradients are checked on CPU.
        linear.config = mxfp4_linear.MXFP4LinearConfig()
        x = torch.randn(2, 32)
        with patch.object(implementation, "_BF16_EVAL", True), patch.object(
            implementation._MXFP4LinearFunction, "apply", return_value=torch.zeros(2, 32)
        ) as dispatch:
            linear.eval()
            torch.testing.assert_close(linear(x), nn.functional.linear(x, linear.weight, linear.bias))
            dispatch.assert_not_called()
            linear.train()
            linear(x)
            dispatch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
