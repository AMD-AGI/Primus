###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU regressions for selectable FLUX paths; GPU kernel validation is separate."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
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
        norm = nn.LayerNorm(32, elementwise_affine=False, eps=1e-6)
        x = torch.randn(2, 3, 32, requires_grad=True)
        shift = torch.randn(2, 1, 32, requires_grad=True)
        scale = torch.randn(2, 1, 32, requires_grad=True)
        gate = torch.randn(2, 1, 32, requires_grad=True)
        mod = layers.ModulationOut(shift=shift, scale=scale, gate=gate)
        grad_inputs = (x, shift, scale, gate, *mlp.parameters())
        expected = gate * mlp((1 + scale) * norm(x) + shift)
        expected_grads = torch.autograd.grad(expected.sum(), grad_inputs)
        # Stock Linear and non-MXFP4 layers must keep working with the recipe's
        # fusion flags enabled, including during the H16 stock ablation.
        with patch.object(layers, "FUSED_DGELU_MLP", True), patch.object(layers, "FWD_GELU_PACK", True):
            actual = layers._mlp_gelu_fused(mlp, norm, x, mod)
        actual_grads = torch.autograd.grad(actual.sum(), grad_inputs)
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

    def test_dispatch_resolution_rejects_invalid_and_honors_legacy_flags(self):
        resolve = mxfp4_linear._resolve_fp4_dispatch
        with patch.dict(
            os.environ,
            {
                "FLUX_FP4_DISPATCH": "invalid",
                "FLUX_FP4_MXFP4_MM": "0",
                "FLUX_FP4_HOST_DISPATCH": "0",
            },
        ):
            with self.assertRaisesRegex(ValueError, "FLUX_FP4_DISPATCH"):
                resolve()
        with patch.dict(
            os.environ,
            {
                "FLUX_FP4_DISPATCH": "",
                "FLUX_FP4_MXFP4_MM": "0",
                "FLUX_FP4_HOST_DISPATCH": "1",
            },
        ):
            self.assertEqual(resolve(), "host_dispatch")
        with patch.dict(
            os.environ,
            {
                "FLUX_FP4_MXFP4_MM": "0",
                "FLUX_FP4_HOST_DISPATCH": "1",
            },
        ):
            os.environ.pop("FLUX_FP4_DISPATCH", None)
            self.assertEqual(resolve(), "host_dispatch")
        with patch.dict(
            os.environ,
            {
                "FLUX_FP4_DISPATCH": "",
                "FLUX_FP4_MXFP4_MM": "0",
                "FLUX_FP4_HOST_DISPATCH": "0",
            },
        ):
            self.assertEqual(resolve(), "fusion")

        # Dual and stock register overlapping custom ops, so validate each
        # resolver in its own interpreter rather than importing both together.
        root = Path(__file__).resolve().parents[2]
        code = r"""
import importlib
import os
import sys
module = importlib.import_module(
    'primus.backends.diffusion.models.quantization.' + sys.argv[1]
)
resolve = module._resolve_fp4_dispatch
os.environ['FLUX_FP4_DISPATCH'] = 'invalid'
try:
    resolve()
except ValueError:
    pass
else:
    raise AssertionError('invalid explicit dispatch did not fail')
os.environ['FLUX_FP4_DISPATCH'] = ''
os.environ['FLUX_FP4_MXFP4_MM'] = '0'
os.environ['FLUX_FP4_HOST_DISPATCH'] = '1'
assert resolve() == 'host_dispatch'
os.environ.pop('FLUX_FP4_DISPATCH')
assert resolve() == 'host_dispatch'
os.environ['FLUX_FP4_HOST_DISPATCH'] = '0'
assert resolve() == 'fusion'
"""
        env = os.environ.copy()
        env["FLUX_FP4_DISPATCH"] = "fusion"
        for module_name in ("mxfp4_linear_dual", "mxfp4_linear_stock"):
            subprocess.run(
                [sys.executable, "-c", code, module_name],
                cwd=root,
                env=env,
                check=True,
            )

    def test_disabled_paths_do_not_import_fused_or_mxfp4_modules(self):
        root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        env.update(
            FLUX_FUSED_NORM_ROPE="0",
            FLUX_ATTN_REGION="0",
            FLUX_FP4_PASSES="off",
            FLUX_GATE_DGATE="0",
            FLUX_FWD_GELU_PACK="0",
            FLUX_FUSED_DGELU_LEGPACK="0",
            FLUX_FUSED_DGELU_MLP="0",
        )
        code = """
import sys
from primus.backends.diffusion.models.flux import layers, model
from primus.backends.diffusion.models.registrations import flux
assert 'primus.backends.diffusion.models.flux.fused_norm_rope' not in sys.modules
assert 'primus.backends.diffusion.models.quantization.mxfp4_linear' not in sys.modules
assert 'primus.backends.diffusion.models.quantization.mxfp4_linear_dual' not in sys.modules
assert 'primus.backends.diffusion.models.quantization.mxfp4_linear_stock' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True)

    def test_fprop_switch_activates_immediately_for_resumed_step(self):
        from primus.backends.diffusion.models.quantization import flux_fprop_switch

        trainer = SimpleNamespace(global_step=7325, model=object())
        with patch.object(flux_fprop_switch, "switch_fprop") as switch, patch.object(
            flux_fprop_switch, "guard_eval_scales"
        ) as guard:
            self.assertIsNone(flux_fprop_switch.switch_if_due(trainer, 7325))
            switch.assert_called_once_with(trainer.model, trainer.global_step)
            guard.assert_called_once_with(trainer)

            switch.reset_mock()
            guard.reset_mock()
            trainer.global_step = 7324
            self.assertEqual(flux_fprop_switch.switch_if_due(trainer, 7325), 7325)
            switch.assert_not_called()
            guard.assert_not_called()

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
