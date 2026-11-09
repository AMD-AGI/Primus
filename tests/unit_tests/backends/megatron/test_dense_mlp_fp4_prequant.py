###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import sys
from types import ModuleType, SimpleNamespace

import torch

from primus.backends.megatron.patches.turbo import dense_mlp_fp4_patches as mod


def _fake_turbo(monkeypatch):
    low_precision = ModuleType("primus_turbo.pytorch.core.low_precision")
    low_precision.MXFP4_BLOCK_SIZE = 32
    low_precision.ScalingGranularity = SimpleNamespace(MX_BLOCKWISE="mx")
    low_precision.ScalingRecipe = lambda use_rht=False: SimpleNamespace(use_rht=use_rht)
    low_precision.float4_e2m1fn_x2 = "fp4"
    quantized_tensor = ModuleType("primus_turbo.pytorch.core.quantized_tensor")
    quantized_tensor.QuantizedTensor = lambda data, scale, **kw: SimpleNamespace(data=data, scale=scale, **kw)
    for name, module in (
        ("primus_turbo", ModuleType("primus_turbo")),
        ("primus_turbo.pytorch", ModuleType("primus_turbo.pytorch")),
        ("primus_turbo.pytorch.core", ModuleType("primus_turbo.pytorch.core")),
        ("primus_turbo.pytorch.core.low_precision", low_precision),
        ("primus_turbo.pytorch.core.quantized_tensor", quantized_tensor),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_wrap_prequant_keeps_the_activation_shape_on_both_operands(monkeypatch):
    _fake_turbo(monkeypatch)
    x = torch.zeros(6, 64, dtype=torch.bfloat16)
    prequant = (torch.zeros(6, 32), torch.zeros(6, 2), torch.zeros(64, 3), torch.zeros(2, 64))
    row, col = mod._wrap_prequant(x, prequant, SimpleNamespace(scale_rounding_mode=0))

    assert row.shape == col.shape == x.shape
    assert (row.quantized_axis, col.quantized_axis) == (-1, -2)
    assert (row.scaling_recipe.use_rht, col.scaling_recipe.use_rht) == (False, True)
    assert row.data is prequant[0] and col.data is prequant[2]
    assert row.orig_dtype == col.orig_dtype == torch.bfloat16
    assert row.scale_rounding_mode == col.scale_rounding_mode == 0


def test_prequant_ste_passes_the_row_gradient_to_x(monkeypatch):
    monkeypatch.setattr(mod, "_wrap_prequant", lambda x, pre, cfg: (x * 1.0, x.detach() * 2.0))
    x = torch.randn(6, 64, requires_grad=True)
    row, col = mod._PrequantSTE.apply(x, None, None)

    assert not col.requires_grad
    grad = torch.randn_like(row)
    row.backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)
