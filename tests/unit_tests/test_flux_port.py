###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU regressions for selectable FLUX paths; GPU kernel validation is separate."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from primus.backends.diffusion.models.flux import layers
from primus.backends.diffusion.models.quantization import (
    flux_fprop_switch,
    mxfp4_linear,
    mxfp4_linear_dual,
)


class FluxPortTest(unittest.TestCase):
    def test_mlp_fallback_outputs_and_gradients(self):
        torch.manual_seed(13)
        mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(approximate="tanh"), nn.Linear(64, 32))
        norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-6)
        x = torch.randn(2, 3, 32, requires_grad=True)
        shift = torch.randn(2, 1, 32, requires_grad=True)
        scale = torch.randn(2, 1, 32, requires_grad=True)
        gate = torch.randn(2, 1, 32, requires_grad=True)
        mod = layers.ModulationOut(shift, scale, gate)
        x_mod = (1 + scale) * norm(x) + shift
        expected = gate * mlp(x_mod)
        inputs = (x, shift, scale, gate, *mlp.parameters())
        expected_grads = torch.autograd.grad(expected.sum(), inputs)
        # Stock Linear and non-MXFP4 layers must keep working with the recipe's
        # fusion flags enabled, including during the H16 stock ablation.
        with patch.object(layers, "FUSED_DGELU_MLP", True), patch.object(layers, "FWD_GELU_PACK", True):
            actual, residual = layers._mlp_gelu_fused(mlp, norm, x, mod)
        actual_grads = torch.autograd.grad(actual.sum(), inputs)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(residual, x)
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

    def test_fprop_switch_bf16_gather_roundtrip(self):
        model = nn.Linear(32, 32)
        shard = mxfp4_linear._A6W4AllGatherWeight(model.weight.detach())
        weight = SimpleNamespace(_local_tensor=shard)
        fp = SimpleNamespace(sharded_param=weight)
        with (
            patch.dict(
                os.environ, FLUX_FPROP_SWITCH_STEP="7325", FLUX_FP4_ALL_GATHER="1", FLUX_FP8_ALL_GATHER="0"
            ),
            patch.object(flux_fprop_switch, "_PRECOMPILE", True),
            patch.object(flux_fprop_switch, "_MODS", [(model, None, None, True)]),
            patch.object(flux_fprop_switch, "_A6W4_LOCAL", {}),
            patch.object(flux_fprop_switch, "_fsdp_params", return_value={id(model): fp}),
            patch.object(flux_fprop_switch, "_reset_gather_state"),
        ):
            self.assertEqual(flux_fprop_switch.switch_step(), 7325)
            self.assertEqual(flux_fprop_switch._swap_gather(model, True), (1, 1))
            self.assertIs(weight._local_tensor, shard._tensor)
            self.assertEqual(flux_fprop_switch._swap_gather(model, False), (1, 0))
            self.assertIs(weight._local_tensor, shard)

    def test_fp8_scale_refresh_deferred_until_switch(self):
        model = nn.Linear(32, 32)
        with (
            patch.dict(os.environ, FLUX_FPROP_SWITCH_STEP="50", FLUX_FP8_ALL_GATHER="1"),
            patch.object(flux_fprop_switch, "FP8_ACTIVE", False),
            patch.object(flux_fprop_switch, "_PRECOMPILE", True),
            patch.object(flux_fprop_switch, "_READY", True),
            patch.object(flux_fprop_switch, "_set_mode", return_value=(1, 0)),
            patch("torchao.float8.precompute_float8_dynamic_scale_for_fsdp") as torchao_refresh,
            patch.object(mxfp4_linear_dual, "precompute_fp8_all_gather_amax") as refresh,
        ):
            mxfp4_linear_dual.precompute_float8_dynamic_scale_for_fsdp(model)
            torchao_refresh.assert_called_once_with(model)
            refresh.assert_not_called()
            flux_fprop_switch.switch_fprop(model, 50)
            refresh.assert_called_once_with(model)
            refresh.reset_mock()
            mxfp4_linear_dual.precompute_float8_dynamic_scale_for_fsdp(model)
            refresh.assert_called_once_with(model)

    def test_fp8_scale_refresh_without_switch(self):
        model = nn.Linear(32, 32)
        with (
            patch.dict(os.environ, FLUX_FPROP_SWITCH_STEP=""),
            patch.object(flux_fprop_switch, "FP8_ACTIVE", False),
            patch("torchao.float8.precompute_float8_dynamic_scale_for_fsdp"),
            patch.object(mxfp4_linear_dual, "precompute_fp8_all_gather_amax") as refresh,
        ):
            mxfp4_linear_dual.precompute_float8_dynamic_scale_for_fsdp(model)
            refresh.assert_called_once_with(model)

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
