###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.patches.te_patches.cross_entropy_overwrite_patches import (
    _enabled,
    make_overwrite_input_cross_entropy,
    supports_overwrite_input,
)


def _ctx(**args):
    defaults = dict(
        te_ce_overwrite_input=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="te",
    )
    defaults.update(args)
    return SimpleNamespace(extra={"args": SimpleNamespace(**defaults)})


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({}, True),
        ({"te_ce_overwrite_input": False}, False),
        ({"cross_entropy_loss_fusion": False}, False),
        ({"cross_entropy_fusion_impl": "native"}, False),
    ],
)
def test_enabled_only_for_te_fused_cross_entropy(monkeypatch, args, expected):
    import primus.backends.megatron.patches.te_patches.cross_entropy_overwrite_patches as mod

    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.extra["args"])
    assert _enabled(_ctx(**args)) is expected


def test_supports_overwrite_input():
    def old(_input, target, label_smoothing=0.0, reduce_loss=False, dist_process_group=None):
        pass

    def new(_input, target, *args, overwrite_input=False):
        pass

    assert not supports_overwrite_input(old)
    assert supports_overwrite_input(new)


def test_wrapper_passes_overwrite_input_for_contiguous_logits():
    calls = []

    def fake_pce(*args, **kwargs):
        calls.append((args, kwargs))
        return "te"

    wrapped = make_overwrite_input_cross_entropy(lambda *a: "original", fake_pce)
    logits = torch.zeros(4, 2, 8)
    labels = torch.zeros(4, 2, dtype=torch.long)
    group = object()

    assert wrapped(logits, labels, group, True) == "te"
    ((args, kwargs),) = calls
    assert args == (logits, labels, 0.0, False, group, -100, True)
    assert kwargs == {"overwrite_input": True}
    assert wrapped._primus_overwrite_input


def test_wrapper_falls_back_for_non_contiguous_logits():
    def fake_pce(*args, **kwargs):
        raise AssertionError("TE path must not see non-contiguous logits")

    wrapped = make_overwrite_input_cross_entropy(lambda *a: "original", fake_pce)
    logits = torch.zeros(2, 4, 8).transpose(0, 1)
    assert not logits.is_contiguous()
    assert wrapped(logits, torch.zeros(4, 2, dtype=torch.long), None) == "original"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_overwrite_input_matches_default_numerics():
    te_ce = pytest.importorskip("transformer_engine.pytorch.cross_entropy")
    if not supports_overwrite_input(te_ce.parallel_cross_entropy):
        pytest.skip("installed TE has no overwrite_input")

    torch.manual_seed(0)
    base = torch.randn(64, 2, 1024, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(0, 1024, (64, 2), device="cuda")

    def loss_and_grad(fn):
        logits = base.clone().requires_grad_()
        loss = fn(logits, labels, None)
        loss.sum().backward()
        return loss.detach(), logits.grad.clone()

    def original(logits, labels, tp_group, is_cg_capturable=False):
        return te_ce.parallel_cross_entropy(logits, labels, 0.0, False, tp_group, -100, is_cg_capturable)

    wrapped = make_overwrite_input_cross_entropy(original, te_ce.parallel_cross_entropy)
    ref_loss, ref_grad = loss_and_grad(original)
    loss, grad = loss_and_grad(wrapped)

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)
