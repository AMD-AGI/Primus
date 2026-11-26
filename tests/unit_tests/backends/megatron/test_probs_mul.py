###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""probs_mul against the eager ``(act * probs).to(act.dtype)`` of PrimusGroupedMLP, bit for bit."""

import pytest
import torch

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")


def _eager(act, probs):
    return (act * probs).to(act.dtype)


@pytest.mark.parametrize("tokens, ffn", [(1, 3072), (777, 3072), (4096, 1536), (300, 1000)])
@pytest.mark.parametrize("probs_grad", [True, False])
def test_probs_mul_bitwise(tokens, ffn, probs_grad):
    # the kernel itself: probs_mul() would run eager here if its first-call check failed
    from primus.backends.megatron.core.fusions.probs_mul import _ProbsMul

    gen = torch.Generator(device="cuda").manual_seed(tokens + ffn)
    act = torch.randn(tokens, ffn, device="cuda", generator=gen).bfloat16()
    probs = torch.rand(tokens, 1, device="cuda", generator=gen)
    grad = torch.randn(tokens, ffn, device="cuda", generator=gen).bfloat16()

    def run(fn):
        a = act.clone().requires_grad_(True)
        p = probs.clone().requires_grad_(probs_grad)
        out = fn(a, p)
        out.backward(grad)
        return out, a.grad, p.grad

    out, da, dp = run(_ProbsMul.apply)
    ref_out, ref_da, ref_dp = run(_eager)
    assert out.dtype == ref_out.dtype and torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
    assert torch.equal(da.view(torch.int16), ref_da.view(torch.int16))
    if probs_grad:
        assert torch.equal(dp.view(torch.int32), ref_dp.view(torch.int32))
    else:
        assert dp is None and ref_dp is None


def test_probs_mul_falls_back_on_other_layouts():
    from primus.backends.megatron.core.fusions.probs_mul import probs_mul

    probs = torch.rand(8, 1, device="cuda")
    act_t = torch.randn(16, 8, device="cuda").bfloat16().t()  # not contiguous
    assert torch.equal(probs_mul(act_t, probs), _eager(act_t, probs))
    act_f32 = torch.randn(8, 16, device="cuda")
    assert torch.equal(probs_mul(act_f32, probs), _eager(act_f32, probs))


def test_probs_mul_zero_tokens():
    from primus.backends.megatron.core.fusions.probs_mul import probs_mul

    act = torch.empty(0, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probs = torch.empty(0, 1, device="cuda", requires_grad=True)
    out = probs_mul(act, probs)
    out.backward(torch.empty_like(out))
    assert out.shape == (0, 3072) and act.grad.shape == act.shape and probs.grad.shape == probs.shape


def test_probs_mul_keeps_eager_when_the_self_check_fails(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.fusions import probs_mul as mod

    def off_by_one_ulp(act, probs):
        return (_eager(act, probs).view(torch.int16) ^ 1).view(torch.bfloat16)

    monkeypatch.setattr(bitwise_check, "_results", {})
    monkeypatch.setattr(mod._ProbsMul, "apply", off_by_one_ulp)
    act = torch.randn(8, 64, device="cuda").bfloat16()
    probs = torch.rand(8, 1, device="cuda")
    assert torch.equal(mod.probs_mul(act, probs), _eager(act, probs))
    assert bitwise_check._results == {("fused_probs_mul", 64): False}


def test_probs_mul_checks_each_width_once(monkeypatch):
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.core.fusions import probs_mul as mod

    monkeypatch.setattr(bitwise_check, "_results", {})
    real, calls = mod._ProbsMul.apply, []
    monkeypatch.setattr(mod._ProbsMul, "apply", lambda *a: calls.append(1) or real(*a))
    for tokens, ffn in [(8, 64), (300, 64), (8, 96)]:
        act = torch.randn(tokens, ffn, device="cuda").bfloat16()
        mod.probs_mul(act, torch.rand(tokens, 1, device="cuda"))
    assert bitwise_check._results == {("fused_probs_mul", 64): True, ("fused_probs_mul", 96): True}
    # a check and a use for each width, a use alone for the decided one
    assert len(calls) == 5


def test_probs_mul_non_contiguous_grad():
    from primus.backends.megatron.core.fusions.probs_mul import probs_mul

    gen = torch.Generator(device="cuda").manual_seed(7)
    act = torch.randn(64, 256, device="cuda", generator=gen).bfloat16()
    probs = torch.rand(64, 1, device="cuda", generator=gen)
    grad = torch.randn(256, 64, device="cuda", generator=gen).bfloat16().t()

    def run(fn):
        a = act.clone().requires_grad_(True)
        p = probs.clone().requires_grad_(True)
        fn(a, p).backward(grad)
        return a.grad, p.grad

    da, dp = run(probs_mul)
    ref_da, ref_dp = run(_eager)
    assert torch.equal(da.view(torch.int16), ref_da.view(torch.int16))
    assert torch.equal(dp.view(torch.int32), ref_dp.view(torch.int32))
