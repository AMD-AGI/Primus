###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect
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


def test_rejects_turbo_cross_entropy(monkeypatch):
    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.extra["args"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        mod.patch_te_cross_entropy_deferred(_ctx(use_turbo_cross_entropy=True))


@pytest.mark.parametrize(
    "make",
    [
        lambda: (torch.zeros(4, 2, 8, dtype=torch.float32), False),
        lambda: (torch.zeros(2, 4, 8, dtype=torch.bfloat16).transpose(0, 1), False),
        lambda: (torch.zeros(4, 2, 8, dtype=torch.bfloat16), True),
    ],
    ids=["fp32", "non_contiguous", "cg_capturable"],
)
def test_falls_back_to_original(make):
    logits, cg = make()
    wrapped = mod.make_deferred_cross_entropy(lambda *a: "original")
    assert wrapped(logits, torch.zeros(4, 2, dtype=torch.long), None, cg) == "original"
    assert wrapped._primus_deferred_ce


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("vocab", [1009, 128256])
def test_matches_te_numerics(vocab):
    te_ce = pytest.importorskip("transformer_engine.pytorch.cross_entropy")

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

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    if "overwrite_input" in inspect.signature(te_ce.parallel_cross_entropy).parameters:
        # This TE already defers and rounds once; the forward-gradient rounding is within an ulp.
        torch.testing.assert_close(grad, ref_grad, rtol=2**-7, atol=0)
    else:
        torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
    assert not grad[3, 1].any()
