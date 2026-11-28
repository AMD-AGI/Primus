###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: the Triton logit soft-cap (alone and inside the deferred cross entropy) against the eager ops,
its one-time bitwise checks, and how MuseSparkModel picks a path."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from megatron.core.models.common.language_module import language_module  # noqa: E402
from megatron.core.models.gpt.gpt_model import GPTModel  # noqa: E402

from primus.backends.megatron.core.fusions import bitwise_check  # noqa: E402
from primus.backends.megatron.core.models.muse_spark import (  # noqa: E402
    muse_spark_model as msm,
)
from primus.backends.megatron.core.models.muse_spark import (  # noqa: E402
    softcap_kernels as sk,
)
from primus.backends.megatron.patches.te_patches import (  # noqa: E402
    cross_entropy_deferred_patches as dce,
)

CAP, MULT = 20.0, 0.17677669529663687
DEV = torch.device("cuda")


@pytest.fixture(autouse=True)
def _fresh_checks(monkeypatch):
    monkeypatch.setattr(bitwise_check, "_results", {})


def _logits(rows, vocab, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(rows, vocab, device=DEV) * 300).bfloat16()


def _softcap_grad_formula(x, g):
    """The eager soft-cap's gradient, each autograd step rounded to bf16 (aten.tanh_backward rounds 1 - t^2)."""
    r = lambda t: t.bfloat16().float()  # noqa: E731
    t3 = r(torch.tanh(r(r(x.float() * MULT) / CAP)))
    g2 = r(r(g.float() * CAP) * r(1 - t3 * t3))
    return r(r(g2 / CAP) * MULT).bfloat16()


def _deferred_ce(round_once):
    """The patched te_parallel_cross_entropy, with the deferred cross entropy standing in for TE."""
    return dce.make_deferred_cross_entropy(
        lambda x, y, tp, cg: dce.DeferredCrossEntropy.apply(x, y, round_once), round_once
    )


@pytest.mark.parametrize("vocab", [4096, 5000])
def test_fused_softcap_is_bitwise_the_eager_ops(vocab):
    x = _logits(64, vocab)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    za, zb = sk.fused_softcap(a, MULT, CAP), msm.softcap_logits(b, MULT, CAP)
    assert torch.equal(za, zb)
    g = torch.randn_like(za.float()).bfloat16()
    za.backward(g)
    assert torch.equal(a.grad, _softcap_grad_formula(x, g))
    if x.numel() % sk.CHECK_BLOCK == 0:
        # eager autograd only where its tanh_backward is right (see sk.CHECK_BLOCK)
        zb.backward(g)
        assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("round_once", [False, True])
@pytest.mark.parametrize("vocab", [5000, 50257])
def test_softcap_cross_entropy_is_bitwise_softcap_then_deferred_ce(vocab, round_once):
    S, B = 48, 2
    x = _logits(S * B, vocab).view(S, B, vocab)
    target = torch.randint(0, vocab, (S, B), device=DEV)
    target[::7] = -100
    upstream = torch.rand(S, B, device=DEV)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    la = sk.softcap_cross_entropy(
        a * 1, target, MULT, CAP, round_once
    )  # the backward writes dx over its input
    lb = dce.DeferredCrossEntropy.apply(sk.fused_softcap(b, MULT, CAP), target, round_once)
    assert torch.equal(la, lb)
    (la * upstream).sum().backward()
    (lb * upstream).sum().backward()
    assert torch.equal(a.grad, b.grad)


def test_checks_accept_the_kernels_and_reject_a_mismatch():
    x = _logits(32 * 2, 3000).view(32, 2, 3000)
    target = torch.randint(0, 3000, (32, 2), device=DEV)
    assert sk.softcap_matches_eager(x, MULT, CAP, msm._apply_softcap)
    assert sk.softcap_cross_entropy_matches(x, target, MULT, CAP, True, _deferred_ce(True), None)
    assert sk.softcap_cross_entropy_matches(x, target, MULT, CAP, False, _deferred_ce(False), None)
    # another cap is another case; a wrong reference or the other rounding is refused
    assert not sk.softcap_matches_eager(x, MULT, 30.0, lambda t, m, c: msm._apply_softcap(t, m, c) * 2)
    assert not sk.softcap_cross_entropy_matches(x, target, MULT, 30.0, True, _deferred_ce(False), None)


def _model(**cfg):
    m = object.__new__(msm.MuseSparkModel)
    torch.nn.Module.__init__(m)
    m.config = SimpleNamespace(
        use_mup=False, mup_output_mult=1.0, output_logit_multiplier=MULT, final_logit_softcapping=CAP, **cfg
    )
    m.pg_collection = SimpleNamespace(tp=None)
    m.triton_softcap = True
    m._muse_defer_softcap = False
    return m


@pytest.mark.parametrize("round_once", [False, True])
def test_model_defers_the_softcap_into_the_cross_entropy_once(monkeypatch, round_once):
    monkeypatch.setattr(language_module, "te_parallel_cross_entropy", _deferred_ce(round_once))
    m = _model()
    x = _logits(32 * 2, 3000).view(32, 2, 3000)
    labels = torch.randint(0, 3000, (2, 32), device=DEV)
    # not deferred: the Triton soft-cap, bitwise the eager one
    assert torch.equal(m._scale_logits(x), msm.softcap_logits(x, MULT, CAP))
    m._muse_defer_softcap = True
    tagged = m._scale_logits(x.clone())
    assert not m._muse_defer_softcap and tagged.__dict__["_muse_pending_softcap"] == (MULT, CAP)
    assert torch.equal(m._scale_logits(x), sk.fused_softcap(x, MULT, CAP))  # later calls are not deferred
    loss = m.compute_language_model_loss(labels, tagged)
    assert "_muse_pending_softcap" not in tagged.__dict__
    want = sk.softcap_cross_entropy(x.clone(), labels.t().contiguous(), MULT, CAP, round_once)
    assert loss.shape == (2, 32) and torch.equal(loss, want.t().contiguous())


def _record_parent_loss(monkeypatch):
    seen = []
    monkeypatch.setattr(
        GPTModel, "compute_language_model_loss", lambda self, labels, z: seen.append(z) or z.float().sum()
    )
    return seen


def test_model_cross_entropy_falls_back_for_ineligible_logits(monkeypatch):
    monkeypatch.setattr(language_module, "te_parallel_cross_entropy", _deferred_ce(True))
    seen = _record_parent_loss(monkeypatch)
    m = _model()
    # not contiguous: the deferred cross entropy does not take it
    x = _logits(16 * 2, 1000).view(16, 2, 1000)[:, :, :999]
    x.__dict__["_muse_pending_softcap"] = (MULT, CAP)
    m.compute_language_model_loss(torch.zeros(2, 16, dtype=torch.long, device=DEV), x)
    assert len(seen) == 1 and torch.equal(seen[0], msm.softcap_logits(x, MULT, CAP))


def test_model_keeps_the_eager_softcap_when_the_check_fails(monkeypatch):
    monkeypatch.setattr(language_module, "te_parallel_cross_entropy", _deferred_ce(True))
    monkeypatch.setattr(sk, "softcap_matches_eager", lambda *a: False)
    fused = []
    monkeypatch.setattr(sk, "fused_softcap", lambda *a: fused.append(1))
    monkeypatch.setattr(sk, "softcap_cross_entropy", lambda *a: fused.append(1))
    seen = _record_parent_loss(monkeypatch)
    m = _model()
    x = _logits(16 * 2, 1000).view(16, 2, 1000)
    assert torch.equal(m._scale_logits(x), msm.softcap_logits(x, MULT, CAP))
    x.__dict__["_muse_pending_softcap"] = (MULT, CAP)
    m.compute_language_model_loss(torch.zeros(2, 16, dtype=torch.long, device=DEV), x)
    assert not fused and torch.equal(seen[0], msm.softcap_logits(x, MULT, CAP))
