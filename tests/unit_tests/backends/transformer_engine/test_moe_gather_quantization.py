# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Paired Primus/Turbo GPU coverage for forward-only and bidirectional fusion."""

import gc

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Requires gfx950 and paired Primus Turbo gather support", allow_module_level=True)

moe_gather = pytest.importorskip(
    "primus_turbo.pytorch.ops.moe_gather", reason="Install the companion Primus Turbo gather revision"
)

__import__("transformer_engine.pytorch")  # Register the TE binary extension first.
from primus_turbo.flydsl.quantization import mxfp4_quant_kernel
from primus_turbo.pytorch.core.low_precision import Float4QuantConfig
from primus_turbo.pytorch.ops.grouped_mlp_fp4 import grouped_mlp_fp4

from primus.backends.transformer_engine.pytorch.permutation import (
    moe_permute_with_probs,
    moe_unpermute,
)


class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return Identity.apply(gradient)


@pytest.mark.parametrize("tokens,skew", [(257, False), (1024, True), (32768, False)])
@pytest.mark.parametrize("use_sr", [False, True])
def test_gather_fusions_preserve_outputs_and_all_gradients(monkeypatch, tokens, skew, use_sr):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (props.major, props.minor) != (9, 5):
        pytest.skip("Validated grouped MXFP4 path requires gfx950")
    monkeypatch.setenv("GPTOSS_BACKWARD_GATHER_AUDIT", "1")
    torch.manual_seed(9555)
    selected = torch.rand(tokens, 8 if skew else 32, device="cuda").topk(4, dim=1).indices
    routing = torch.zeros(tokens, 32, dtype=torch.bool, device="cuda").scatter_(1, selected, True)
    data = [
        torch.randn(tokens, 2880, dtype=torch.bfloat16, device="cuda") * 0.1,
        torch.rand(tokens, 32, device="cuda") * 0.25,
        torch.randn(32, 5760, 2880, dtype=torch.bfloat16, device="cuda") * 0.02,
        torch.randn(32, 2880, 2880, dtype=torch.bfloat16, device="cuda") * 0.02,
    ]
    data[1][::7] = 0
    data[1][1::11] = -0.0
    upstream = torch.randn(tokens, 2880, dtype=torch.bfloat16, device="cuda") * 0.1
    reference = None
    saved_counter = mxfp4_quant_kernel._SR_COUNTER[0]
    try:
        for forward, backward in [(0, 0), (1, 0), (1, 1)]:
            mxfp4_quant_kernel._SR_COUNTER[0] = 9555
            moe_gather._BACKWARD_GATHER_AUDIT.clear()
            x, probs, w1, w2 = [value.detach().clone().requires_grad_() for value in data]
            packed, pp, rowmap = moe_permute_with_probs(
                x,
                probs,
                routing,
                tokens * 4,
                fuse_permute_quant=bool(forward),
                fuse_backward_permute_quant=bool(backward),
            )
            if forward:
                assert packed.stride(0) == 0
            out = grouped_mlp_fp4(
                Identity.apply(packed),
                w1,
                w2,
                routing.sum(0, dtype=torch.int64),
                pp,
                trans_w1=True,
                trans_w2=True,
                out_dtype=torch.bfloat16,
                config=Float4QuantConfig(use_gradient_sr=use_sr, scale_rounding_mode=2),
                activation="silu",
                clamp_limit=7,
            )
            restored = moe_unpermute(Identity.apply(out), rowmap, restore_shape=x.shape)
            restored.backward(upstream)
            torch.cuda.synchronize()
            actual = [restored.detach().clone()] + [
                value.grad.detach().clone() for value in (x, probs, w1, w2)
            ]
            counter = mxfp4_quant_kernel._SR_COUNTER[0]
            if reference is None:
                reference = (actual, counter)
            else:
                assert counter == reference[1]
                for value, expected in zip(actual, reference[0]):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
            assert not moe_gather._PERMUTED_ACTIVATION_SEAM_TABLE
            assert not moe_gather._BACKWARD_GATHER_OUTPUTS
            if backward:
                assert moe_gather._BACKWARD_GATHER_AUDIT.get("consumed") == 1
                assert moe_gather._BACKWARD_GATHER_AUDIT.get("materialized", 0) == 0
            del x, probs, w1, w2, packed, pp, rowmap, out, restored, actual
    finally:
        mxfp4_quant_kernel._SR_COUNTER[0] = saved_counter
        gc.collect()
        torch.cuda.empty_cache()


def test_empty_input_keeps_eager_contract():
    x = torch.empty(0, 2880, dtype=torch.bfloat16, device="cuda")
    probs = torch.empty(0, 32, dtype=torch.float32, device="cuda")
    routing = torch.empty(0, 32, dtype=torch.bool, device="cuda")
    expected = moe_permute_with_probs(x, probs, routing, 4)
    actual = moe_permute_with_probs(x, probs, routing, 4, fuse_permute_quant=True)
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert not moe_gather._PERMUTED_ACTIVATION_SEAM_TABLE
