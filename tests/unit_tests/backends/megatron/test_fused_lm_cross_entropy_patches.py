###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused LM cross entropy patch: plumbing on CPU (is_cuda stand-ins), numerics on a GPU."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

import primus.backends.megatron.patches.fused_lm_cross_entropy_patches as CEP  # noqa: E402
from primus.backends.megatron.core.fusions import (  # noqa: E402
    triton_lm_cross_entropy as ce,
)
from tests.unit_tests.backends.megatron.conftest import (  # noqa: E402
    cuda_like,
    patch_ctx,
    requires_gpu,
)


def _module(tp):
    return SimpleNamespace(pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: tp)))


def test_wrapper_falls_back_for_cpu_tensors(monkeypatch):
    warned = []
    monkeypatch.setattr(ce, "warn_fallback_once", lambda key, why: warned.append((key, why)))
    calls = []

    def orig(self, labels, logits):
        calls.append(1)
        return "orig"

    wrapped = ce.make_compute_language_model_loss(orig, "key")
    out = wrapped(SimpleNamespace(), torch.zeros(2, 3, dtype=torch.long), torch.zeros(3, 2, 7))
    assert out == "orig" and calls == [1] and wrapped.__wrapped__ is orig
    assert warned == [("key", "logits not a CUDA tensor")]


def test_wrapper_falls_back_for_tensor_parallel_and_non_contiguous_logits(monkeypatch):
    monkeypatch.setattr(ce, "fused_lm_cross_entropy", lambda logits, labels: "fused")
    wrapped = ce.make_compute_language_model_loss(lambda self, labels, logits: "orig", "key")
    lb = cuda_like(torch.zeros(2, 3, dtype=torch.long))
    lg = cuda_like(torch.randn(3, 2, 7, dtype=torch.bfloat16))
    assert wrapped(_module(1), lb, lg) == "fused"
    assert wrapped(_module(2), lb, lg) == "orig"
    transposed = cuda_like(torch.randn(2, 3, 7, dtype=torch.bfloat16)).transpose(0, 1)
    assert wrapped(_module(1), lb, transposed) == "orig"


def test_ineligible_reasons():
    reason = ce.ineligible_reason
    lg = cuda_like(torch.randn(3, 2, 7, dtype=torch.bfloat16))  # [s, b, V]
    lb = cuda_like(torch.zeros(2, 3, dtype=torch.long))  # [b, s]
    assert reason(lg, lb) is None
    assert reason(lg, cuda_like(torch.zeros(2, 3, dtype=torch.int32))) is None
    assert "CUDA" in reason(torch.zeros(3, 2, 7), torch.zeros(2, 3, dtype=torch.long))
    transposed = cuda_like(torch.randn(2, 3, 7, dtype=torch.bfloat16)).transpose(0, 1)
    assert reason(transposed, lb) == "non-contiguous logits"
    assert "!= (b, s)" in reason(lg, cuda_like(torch.zeros(3, 2, dtype=torch.long)))  # [s, b] labels
    assert "labels dtype" in reason(lg, cuda_like(torch.zeros(2, 3)))
    assert "labels dtype" in reason(lg, cuda_like(torch.zeros(2, 3, dtype=torch.int16)))
    assert "shape" in reason(lg, None)
    assert "shape" in reason(cuda_like(torch.randn(6, 7, dtype=torch.bfloat16)), lb)
    assert "dtype" in reason(cuda_like(torch.zeros(3, 2, 7, dtype=torch.int32)), lb)


def test_patch_installs_once(restore_patched):
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    restore_patched(LanguageModule, "compute_language_model_loss")
    before = LanguageModule.compute_language_model_loss
    CEP.patch_fused_lm_cross_entropy(patch_ctx(use_primus_fused_lm_cross_entropy=True))
    first = LanguageModule.compute_language_model_loss
    assert first is not before and first.__wrapped__ is before
    CEP.patch_fused_lm_cross_entropy(patch_ctx(use_primus_fused_lm_cross_entropy=True))
    assert LanguageModule.compute_language_model_loss is first  # idempotent


def _ce_reference(logits, labels):
    x = logits.float()
    lse = torch.logsumexp(x, dim=-1)  # [s, b]
    lab = labels.t()
    valid = (lab >= 0) & (lab < x.shape[-1])
    tgt = torch.gather(x, -1, lab.clamp(0, x.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
    # Out-of-range label: Megatron's vocab-parallel CE gives loss lse - max and gradient softmax (its custom
    # backward treats the max as a constant), so the max must not carry a gradient here either.
    tgt = torch.where(valid, tgt, x.max(-1).values.detach())
    return (lse - tgt).t()


@requires_gpu
@pytest.mark.parametrize("S,B,V", [(5, 3, 1000), (4, 2, 8192 + 17)])
def test_fused_matches_reference(S, B, V):
    torch.manual_seed(0)
    logits = (torch.randn(S, B, V, device="cuda") * 2).to(torch.bfloat16)
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[0, 0] = V + 5  # out of range: no target term
    ref_in = logits.detach().clone().float().requires_grad_(True)
    ref = _ce_reference(ref_in, labels)
    g = torch.rand(B, S, device="cuda")
    ref.backward(g)

    x = logits.detach().clone().requires_grad_(True)
    loss = ce.fused_lm_cross_entropy(x, labels)
    assert loss.shape == (B, S) and loss.dtype == torch.float32
    # fp32 loss from bf16 logits (other summation order); the gradient is rounded once to bf16.
    torch.testing.assert_close(loss, ref.detach(), atol=2e-3, rtol=1e-3)
    loss.backward(g)
    torch.testing.assert_close(x.grad.float(), ref_in.grad, atol=2e-3, rtol=2e-2)


@requires_gpu
def test_grad_through_a_linear_producer():
    # A non-leaf input: the gradient written into the logits buffer must flow into the producer.
    torch.manual_seed(1)
    S, B, H, V = 16, 2, 64, 1000
    h = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16)
    lin = torch.nn.Linear(H, V, bias=False, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, V, (B, S), device="cuda")
    ce.fused_lm_cross_entropy(lin(h), labels).sum().backward()
    w_fused = lin.weight.grad.float()
    lin.weight.grad = None
    _ce_reference(lin(h).float(), labels).sum().backward()
    # Both weight grads go through a bf16 GEMM; the fused one also through a bf16 logits grad.
    torch.testing.assert_close(w_fused, lin.weight.grad.float(), atol=2e-2, rtol=2e-2)


@requires_gpu
def test_inplace_grad_is_detected_by_autograd():
    # tanh saves its output (the logits): the in-place grad write must raise, not corrupt tanh's backward.
    x = torch.randn(8, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 512, (2, 8), device="cuda")
    loss = ce.fused_lm_cross_entropy(torch.tanh(x), labels).sum()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()
