###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

import primus.backends.megatron.patches.te_patches.cross_entropy_deferred_patches as mod


def _ctx(**args):
    defaults = dict(
        deferred_cross_entropy=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
    )
    defaults.update(args)
    return SimpleNamespace(extra={"args": SimpleNamespace(**defaults)})


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, True),
        ({"deferred_cross_entropy": False}, False),
        ({"cross_entropy_loss_fusion": False}, False),
        ({"cross_entropy_fusion_impl": "native"}, False),
    ],
)
def test_enabled_only_for_te_fused_cross_entropy(monkeypatch, args, expected):
    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.extra["args"])
    assert mod._enabled(_ctx(**args)) is expected


def _eligible_inputs():
    logits = torch.zeros(4, 2, 8, dtype=torch.bfloat16, device="cuda")
    labels = torch.zeros(4, 2, dtype=torch.long, device="cuda")
    return logits, labels, False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "change",
    [
        lambda lo, la, cg: (lo.float(), la, cg),
        lambda lo, la, cg: (lo.transpose(0, 1).contiguous().transpose(0, 1), la, cg),
        lambda lo, la, cg: (lo, la, True),
        lambda lo, la, cg: (lo.cpu(), la.cpu(), cg),
        lambda lo, la, cg: (lo, la.int(), cg),
        lambda lo, la, cg: (lo, la[:, :1], cg),
    ],
    ids=["fp32", "non_contiguous", "cg_capturable", "cpu", "int32_labels", "label_shape"],
)
def test_falls_back_to_original(monkeypatch, change):
    monkeypatch.setattr(mod.DeferredCrossEntropy, "apply", lambda *a: "deferred")
    wrapped = mod.make_deferred_cross_entropy(lambda *a: "original")
    assert wrapped._primus_deferred_ce
    assert wrapped(*_eligible_inputs()[:2], None, False) == "deferred"
    logits, labels, cg = change(*_eligible_inputs())
    assert wrapped(logits, labels, None, cg) == "original"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("vocab", [1009, 128256])
def test_matches_te_numerics(monkeypatch, vocab):
    te_ce = pytest.importorskip("transformer_engine.pytorch.cross_entropy")
    from primus.backends.megatron.patches.te_patches.cross_entropy_overwrite_patches import (
        supports_overwrite_input,
    )

    deferred_calls = []
    apply = mod.DeferredCrossEntropy.apply
    monkeypatch.setattr(mod.DeferredCrossEntropy, "apply", lambda *a: deferred_calls.append(1) or apply(*a))

    torch.manual_seed(0)
    base = (4 * torch.randn(64, 2, vocab, device="cuda")).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (64, 2), device="cuda")
    labels[3, 1] = -100
    upstream = torch.rand(64, 2, device="cuda")

    def original(logits, labels, tp_group, is_cg_capturable=False):
        return te_ce.parallel_cross_entropy(logits, labels, 0.0, False, tp_group, -100, is_cg_capturable)

    def loss_and_grad(fn):
        logits = base.clone().requires_grad_()
        loss = fn(logits, labels, None)
        (loss * upstream).sum().backward()
        return loss.detach(), logits.grad.clone()

    ref_loss, ref_grad = loss_and_grad(original)
    loss, grad = loss_and_grad(mod.make_deferred_cross_entropy(original))
    assert deferred_calls == [1]

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    if supports_overwrite_input(te_ce.parallel_cross_entropy):
        # This TE already defers and rounds once; the forward-gradient rounding is within an ulp.
        torch.testing.assert_close(grad, ref_grad, rtol=2**-7, atol=0)
    else:
        torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
    assert not grad[3, 1].any()
