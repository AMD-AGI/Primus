###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the MoE router/expert input-gradient join."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.extensions import router_grad_merge
from primus.core.patches import PatchContext

_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/moe_patches/router_bwd_bf16_patches.py"
)
_spec = importlib.util.spec_from_file_location("router_grad_merge_patch_under_test", _PATCH_PATH)
router_bwd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(router_bwd)


def _patch_context(**kwargs):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**kwargs))},
    )


def _cpu_gemm(a, b, out_dtype, *, layout, grad):
    assert layout == "NN" and grad
    return ((b.float() @ a.float()).to(out_dtype),)


def _record(shape, grad=None, weight=None):
    record = object.__new__(router_grad_merge._MergeRecord)
    record.shape = shape
    record.grad = grad
    record.weight = weight
    record.gemm = _cpu_gemm
    return record


def test_grad_merge_requires_opt_in_and_bf16_router_backward():
    assert not router_bwd._grad_merge_enabled(_patch_context())
    assert router_bwd._grad_merge_enabled(_patch_context(moe_router_grad_merge=True))
    with pytest.raises(ValueError, match="requires moe_router_bwd_bf16"):
        router_bwd.patch_router_grad_merge(_patch_context(moe_router_grad_merge=True))


def test_join_passes_expert_gradient_through_without_router_gradient():
    x = torch.randn(4, 2, 8, requires_grad=True)
    record = _record(tuple(x.shape))
    y = router_grad_merge._RouterGradientJoin.apply(x, record)
    grad = torch.randn_like(y)
    y.backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)


def test_reference_sum_rounds_router_gradient_before_adding_expert_gradient():
    generator = torch.Generator().manual_seed(7)
    grad = torch.randn(8, 4, generator=generator).bfloat16()
    weight = torch.randn(4, 6, generator=generator).bfloat16()
    expert = torch.randn(2, 4, 6, generator=generator).bfloat16()
    result = router_grad_merge._reference_sum(_record((2, 4, 6), grad, weight), expert)
    router = (grad.float() @ weight.float()).bfloat16().view(2, 4, 6)
    torch.testing.assert_close(result, router + expert, rtol=0, atol=0)


def test_scope_stays_closed_until_installed():
    assert not router_grad_merge.stats()["installed"]
    assert router_grad_merge.begin_fused_scope(None, None, None, None, None) is None
    router_grad_merge.end_fused_scope(None)


@pytest.mark.skipif(
    not torch.cuda.is_available() or router_grad_merge.triton is None, reason="requires a GPU and Triton"
)
@pytest.mark.parametrize("rows,num_experts,hidden", [(32768, 32, 2880), (1000, 16, 2048), (77, 128, 640)])
def test_kernel_matches_rounded_gemm_plus_expert_gradient(rows, num_experts, hidden):
    generator = torch.Generator(device="cuda").manual_seed(0)
    grad = torch.randn(rows, num_experts, device="cuda", generator=generator).bfloat16()
    weight = torch.randn(num_experts, hidden, device="cuda", generator=generator).bfloat16()
    expert = torch.randn(rows, hidden, device="cuda", generator=generator).bfloat16()
    result = router_grad_merge.merge_kernel(grad, weight, expert).float()
    router = (grad.double() @ weight.double()).bfloat16().float()
    expected = (router + expert.float()).bfloat16().float()
    # FP32 accumulation order may move the router term by one BF16 rounding.
    bound = torch.finfo(torch.bfloat16).eps * (router.abs() + expected.abs())
    assert ((result - expected).abs() <= bound).all()
