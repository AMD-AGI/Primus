###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the pinned-Triton LM cross entropy and SwiGLU patches.

CPU part (always): the patches install on the real Megatron classes / module globals and are idempotent; the
CE eligibility checks (device, layout, contiguity, labels shape / dtype) and the tensor-parallel fallback; the
SwiGLU drop-ins take the fused path inside their contract and send every call outside it (CPU tensors, bias,
clamp, fp8 store, CPU offload, dtype, odd last dim, weights shape) to the original function with the full
signature. CPU tensors that report ``is_cuda`` stand in for device tensors, with the fused entry points stubbed.
GPU part (skipped without a device): fused vs an fp32 PyTorch reference on small ragged shapes, including an
out-of-range label (Megatron's target-mask semantics), the probs gradient of the weighted SwiGLU and the 3-D
input path through the drop-ins.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("triton")

import primus.backends.megatron.patches.triton_lm_cross_entropy_patches as CEP  # noqa: E402
import primus.backends.megatron.patches.triton_swiglu_patches as SWP  # noqa: E402
from primus.backends.megatron.core.fusions import (  # noqa: E402
    triton_swiglu as swiglu_ops,
)
from primus.core.patches.context import PatchContext  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


class _CudaLike(torch.Tensor):
    """CPU tensor that reports is_cuda (exercises the non-device part of the contract on CPU)."""

    @property
    def is_cuda(self):
        return True


def _cl(t):
    return t.as_subclass(_CudaLike)


# ---------------------------------------------------------------- cross entropy


def test_ce_wrapper_falls_back_for_cpu_tensors():
    calls = []

    def orig(self, labels, logits):
        calls.append(1)
        return "orig"

    wrapped = CEP.make_compute_language_model_loss(orig)
    out = wrapped(SimpleNamespace(), torch.zeros(2, 3, dtype=torch.long), torch.zeros(3, 2, 7))
    assert out == "orig" and calls == [1]
    assert wrapped.__wrapped__ is orig


def test_ce_wrapper_falls_back_for_tensor_parallel(monkeypatch):
    from primus.backends.megatron.core.fusions import triton_lm_cross_entropy as ce

    monkeypatch.setattr(ce, "ineligible_reason", lambda logits, labels: None)
    monkeypatch.setattr(ce, "fused_lm_cross_entropy", lambda logits, labels: "fused")
    wrapped = CEP.make_compute_language_model_loss(lambda self, labels, logits: "orig")

    def module(tp):
        return SimpleNamespace(pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: tp)))

    assert wrapped(module(1), None, None) == "fused"
    assert wrapped(module(2), None, None) == "orig"


def test_ce_ineligible_reasons():
    from primus.backends.megatron.core.fusions.triton_lm_cross_entropy import (
        ineligible_reason,
    )

    lg = _cl(torch.randn(3, 2, 7, dtype=torch.bfloat16))  # [s, b, V]
    lb = _cl(torch.zeros(2, 3, dtype=torch.long))  # [b, s]
    assert ineligible_reason(lg, lb) is None
    assert ineligible_reason(lg, _cl(torch.zeros(2, 3, dtype=torch.int32))) is None
    assert "CUDA" in ineligible_reason(torch.zeros(3, 2, 7), torch.zeros(2, 3, dtype=torch.long))
    transposed = _cl(torch.randn(2, 3, 7, dtype=torch.bfloat16)).transpose(0, 1)
    assert ineligible_reason(transposed, lb) == "non-contiguous logits"
    assert "!= (b, s)" in ineligible_reason(lg, _cl(torch.zeros(3, 2, dtype=torch.long)))  # [s, b] labels
    assert "labels dtype" in ineligible_reason(lg, _cl(torch.zeros(2, 3)))
    assert "labels dtype" in ineligible_reason(lg, _cl(torch.zeros(2, 3, dtype=torch.int16)))
    assert "shape" in ineligible_reason(lg, None)
    assert "shape" in ineligible_reason(_cl(torch.randn(6, 7, dtype=torch.bfloat16)), lb)
    assert "dtype" in ineligible_reason(_cl(torch.zeros(3, 2, 7, dtype=torch.int32)), lb)


def test_ce_wrapper_sends_non_contiguous_logits_to_megatron(monkeypatch):
    from primus.backends.megatron.core.fusions import triton_lm_cross_entropy as ce

    monkeypatch.setattr(ce, "fused_lm_cross_entropy", lambda logits, labels: "fused")
    wrapped = CEP.make_compute_language_model_loss(lambda self, labels, logits: "orig")
    module = SimpleNamespace(pg_collection=SimpleNamespace(tp=SimpleNamespace(size=lambda: 1)))
    lb = _cl(torch.zeros(2, 3, dtype=torch.long))
    assert wrapped(module, lb, _cl(torch.randn(3, 2, 7, dtype=torch.bfloat16))) == "fused"
    transposed = _cl(torch.randn(2, 3, 7, dtype=torch.bfloat16)).transpose(0, 1)
    assert wrapped(module, lb, transposed) == "orig"


def _ctx(**args):
    return PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(**args))},
    )


def test_ce_patch_installs_once():
    from megatron.core.models.common.language_module import language_module

    cls = language_module.LanguageModule
    before = cls.compute_language_model_loss
    try:
        CEP.patch_fused_lm_cross_entropy(_ctx(use_primus_fused_lm_cross_entropy=True))
        first = cls.compute_language_model_loss
        assert getattr(first, "_primus_fused_lm_ce", False) and first.__wrapped__ is before
        CEP.patch_fused_lm_cross_entropy(_ctx(use_primus_fused_lm_cross_entropy=True))
        assert cls.compute_language_model_loss is first  # idempotent
    finally:
        cls.compute_language_model_loss = before


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


@cuda
@pytest.mark.parametrize("S,B,V", [(5, 3, 1000), (4, 2, 8192 + 17)])
def test_ce_fused_matches_reference(S, B, V):
    from primus.backends.megatron.core.fusions.triton_lm_cross_entropy import (
        fused_lm_cross_entropy,
    )

    torch.manual_seed(0)
    logits = (torch.randn(S, B, V, device="cuda") * 2).to(torch.bfloat16)
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[0, 0] = V + 5  # out of range: no target term
    ref_in = logits.detach().clone().float().requires_grad_(True)
    ref = _ce_reference(ref_in, labels)
    g = torch.rand(B, S, device="cuda")
    ref.backward(g)

    x = logits.detach().clone().requires_grad_(True)
    loss = fused_lm_cross_entropy(x, labels)
    assert loss.shape == (B, S) and loss.dtype == torch.float32
    torch.testing.assert_close(loss, ref.detach(), atol=2e-3, rtol=1e-3)
    loss.backward(g)
    torch.testing.assert_close(x.grad.float(), ref_in.grad, atol=2e-3, rtol=2e-2)


# ---------------------------------------------------------------------- SwiGLU


@pytest.fixture
def restore_swiglu_sites():
    import importlib

    saved = []
    for modname, attrs in SWP.SITES.items():
        try:
            mod = importlib.import_module(modname)
        except Exception:  # noqa: BLE001 - optional site
            continue
        saved.extend((mod, attr, getattr(mod, attr, None)) for attr, _ in attrs)
    yield
    for mod, attr, value in saved:
        if value is not None:
            setattr(mod, attr, value)


def test_swiglu_patch_sites_and_cpu_fallback(restore_swiglu_sites):
    result = SWP.patch_sites()
    import megatron.core.fusions.fused_bias_swiglu as mfus
    import megatron.core.transformer.mlp as mlp

    assert result["megatron.core.transformer.mlp.bias_swiglu_impl"] is True
    assert getattr(mlp.bias_swiglu_impl, swiglu_ops.ORIGINAL_ATTR, None) is not None
    again = SWP.patch_sites()  # idempotent: no double wrapping
    assert again == result
    assert (
        getattr(getattr(mlp.bias_swiglu_impl, swiglu_ops.ORIGINAL_ATTR), swiglu_ops.ORIGINAL_ATTR, None)
        is None
    )

    x = torch.randn(6, 8)
    out = mfus.bias_swiglu_impl(x, None)  # CPU fp32 -> original Megatron function
    ref = torch.nn.functional.silu(x[:, :4]) * x[:, 4:]
    torch.testing.assert_close(out, ref)


def test_swiglu_drop_ins_fall_back_outside_the_contract(monkeypatch):
    monkeypatch.setattr(swiglu_ops, "fused_weighted_swiglu", lambda x, w: "fused")
    monkeypatch.setattr(swiglu_ops, "fused_swiglu", lambda x: "fused")
    calls = []

    def orig(*a):
        calls.append(len(a))
        return "orig"

    cpu = torch.randn(4, 8, dtype=torch.bfloat16)
    x = _cl(cpu)
    w = _cl(torch.rand(4, 1))
    b = x[:, 0]
    plain = swiglu_ops.make_megatron_plain(orig)
    weighted = swiglu_ops.make_megatron_weighted(orig)
    primus_weighted = swiglu_ops.make_primus_weighted(orig)
    assert getattr(plain, swiglu_ops.ORIGINAL_ATTR) is orig
    # control: in-contract calls take the fused path
    assert plain(x, None) == weighted(x, None, w) == primus_weighted(x, None, w) == "fused"
    assert plain(_cl(torch.randn(3, 2, 8, dtype=torch.bfloat16)), None) == "fused"  # 3-D [s, b, 2H]
    assert weighted(x, None, _cl(torch.rand(4))) == "fused"  # 1-D weights
    assert calls == []
    # one contract violation per call
    assert plain(cpu, None) == "orig"  # CPU input
    assert plain(x, b) == "orig"  # bias
    assert plain(x, None, True) == "orig"  # fp8 store
    assert plain(x, None, False, True) == "orig"  # cpu offload
    assert plain(_cl(torch.randn(4, 8)), None) == "orig"  # fp32 input
    assert plain(_cl(torch.randn(4, 7, dtype=torch.bfloat16)), None) == "orig"  # odd last dim
    assert plain(_cl(torch.randn(2, 2, 2, 8, dtype=torch.bfloat16)), None) == "orig"  # 4-D
    assert weighted(x, b, w) == "orig"  # bias
    assert weighted(x, None, w, True) == "orig"  # fp8 store
    assert weighted(x, None, _cl(torch.rand(3, 1))) == "orig"  # weights row count
    assert weighted(x, None, _cl(torch.rand(4, 2))) == "orig"  # weights numel
    assert weighted(x, None, _cl(torch.rand(2, 2))) == "orig"  # weights shape
    assert weighted(x, None, None) == "orig"  # no weights
    assert primus_weighted(x, b, w) == "orig"  # bias
    assert primus_weighted(x, None, w, True) == "orig"  # fp8 store
    assert primus_weighted(x, None, w, False, 7.0) == "orig"  # clamp
    # the full original signature is forwarded
    assert calls == [4] * 7 + [4] * 6 + [5] * 3


def _swiglu_reference(y, w=None):
    y = y.float()
    h = y.shape[-1] // 2
    out = torch.nn.functional.silu(y[..., :h]) * y[..., h:]
    return out if w is None else out * w.float().reshape(*out.shape[:-1], 1)


@cuda
@pytest.mark.parametrize("M,H", [(7, 64), (33, 2048 + 64)])
@pytest.mark.parametrize("weighted", [False, True])
def test_swiglu_fused_matches_reference(M, H, weighted):
    torch.manual_seed(0)
    y = torch.randn(M, 2 * H, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(M, 1, device="cuda", dtype=torch.float32) if weighted else None
    g = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)

    y_ref = y.detach().float().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True) if weighted else None
    ref = _swiglu_reference(y_ref, w_ref)
    ref.backward(g.float())

    y_f = y.detach().clone().requires_grad_(True)
    w_f = w.detach().clone().requires_grad_(True) if weighted else None
    out = swiglu_ops.fused_weighted_swiglu(y_f, w_f) if weighted else swiglu_ops.fused_swiglu(y_f)
    out.backward(g)
    torch.testing.assert_close(out.float(), ref.detach(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(y_f.grad.float(), y_ref.grad, atol=3e-2, rtol=2e-2)
    if weighted:
        assert w_f.grad.dtype == torch.float32 and w_f.grad.shape == w.shape
        torch.testing.assert_close(w_f.grad, w_ref.grad, atol=1e-2, rtol=1e-3)


@cuda
@pytest.mark.parametrize("weighted", [False, True])
def test_swiglu_drop_ins_3d_match_reference(weighted):
    def orig(*a):
        raise AssertionError("original SwiGLU taken")

    torch.manual_seed(0)
    S, B, H = 5, 3, 96
    y = torch.randn(S, B, 2 * H, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(S * B, 1, device="cuda", dtype=torch.float32) if weighted else None
    g = torch.randn(S, B, H, device="cuda", dtype=torch.bfloat16)
    y_ref = y.detach().float().requires_grad_(True)
    ref = _swiglu_reference(y_ref, w)
    ref.backward(g.float())

    y_f = y.detach().clone().requires_grad_(True)
    if weighted:
        out = swiglu_ops.make_megatron_weighted(orig)(y_f, None, w)
    else:
        out = swiglu_ops.make_megatron_plain(orig)(y_f, None)
    assert out.shape == (S, B, H)
    out.backward(g)
    torch.testing.assert_close(out.float(), ref.detach(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(y_f.grad.float(), y_ref.grad, atol=3e-2, rtol=2e-2)
