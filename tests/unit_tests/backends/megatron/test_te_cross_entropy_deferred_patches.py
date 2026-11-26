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


def test_skips_when_te_kernel_cannot_be_inspected(monkeypatch):
    te_ext = pytest.importorskip("megatron.core.extensions.transformer_engine")
    from megatron.core.models.common.language_module import language_module

    def original(*a):
        return "original"

    monkeypatch.setattr(mod, "get_args", lambda ctx: ctx.extra["args"])
    monkeypatch.setattr(mod, "te_rounds_once", lambda: None)
    monkeypatch.setattr(te_ext, "te_parallel_cross_entropy", original)
    monkeypatch.setattr(language_module, "te_parallel_cross_entropy", original)
    mod.patch_te_cross_entropy_deferred(_ctx())
    assert te_ext.te_parallel_cross_entropy is original
    assert language_module.te_parallel_cross_entropy is original
    with pytest.raises(RuntimeError, match="rounds"):
        mod.make_deferred_cross_entropy(original)


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
    monkeypatch.setattr(mod, "_matches_original", lambda *a: True)
    wrapped = mod.make_deferred_cross_entropy(lambda *a: "original", round_once=False)
    assert wrapped._primus_deferred_ce
    assert wrapped(*_eligible_inputs()[:2], None, False) == "deferred"
    logits, labels, cg = change(*_eligible_inputs())
    assert wrapped(logits, labels, None, cg) == "original"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("vocab", [1009, 128256])
@pytest.mark.parametrize("round_once", [None, True, False], ids=["installed_te", "once", "twice"])
def test_bitwise_with_the_te_path_it_replaces(monkeypatch, vocab, round_once, require_bitwise):
    te_ce = pytest.importorskip("transformer_engine.pytorch.cross_entropy")
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.backends.megatron.patches.te_patches.cross_entropy_overwrite_patches import (
        make_overwrite_input_cross_entropy,
        supports_overwrite_input,
    )

    monkeypatch.setattr(bitwise_check, "_results", {})
    torch.manual_seed(0)
    base = (4 * torch.randn(64, 2, vocab, device="cuda")).to(torch.bfloat16)
    labels = torch.randint(0, vocab, (64, 2), device="cuda")
    labels[3, 1] = -100
    upstream = torch.rand(64, 2, device="cuda")

    def te(logits, labels, tp_group, is_cg_capturable=False):
        return te_ce.parallel_cross_entropy(logits, labels, 0.0, False, tp_group, -100, is_cg_capturable)

    # what the patch wraps under the default te_ce_overwrite_input: true
    overwrite = supports_overwrite_input(te_ce.parallel_cross_entropy)
    original = make_overwrite_input_cross_entropy(te, te_ce.parallel_cross_entropy) if overwrite else te

    def loss_and_grad(fn):
        logits = base.clone().requires_grad_()
        loss = fn(logits, labels, None)
        (loss * upstream).sum().backward()
        return loss.detach(), logits.grad.clone()

    ref_loss, ref_grad = loss_and_grad(original)
    loss, grad = loss_and_grad(mod.make_deferred_cross_entropy(original, round_once))

    assert torch.equal(loss.view(torch.int32), ref_loss.view(torch.int32))
    assert torch.equal(grad.view(torch.int16), ref_grad.view(torch.int16))
    assert not grad[3, 1].any()
    used = bitwise_check._results[("deferred_cross_entropy", vocab)]
    if round_once is not None and round_once != mod.te_rounds_once():
        assert not used, "the other rounding must fail the first-call check and keep TE"
    else:
        require_bitwise(used, "deferred_cross_entropy")


def test_first_call_check_probes_wide_logits_and_an_ignored_label(monkeypatch):
    """Early in training the real logits are near uniform; the check adds peaked ones and an
    ignored label."""
    from primus.backends.megatron.core.fusions import bitwise_check

    seen = []
    monkeypatch.setattr(bitwise_check, "_results", {})
    monkeypatch.setattr(
        bitwise_check, "outputs_and_grads", lambda fn, *inputs: seen.append(inputs) or [inputs[0]]
    )
    logits = torch.zeros(64, 2, 1009, dtype=torch.bfloat16)
    labels = torch.ones(64, 2, dtype=torch.long)
    assert mod._matches_original(None, logits, labels, None, True)

    (x, y), (x_ref, y_ref) = seen
    assert torch.equal(x, x_ref) and torch.equal(y, y_ref)
    rows = mod.CHECK_ROWS
    assert x.shape == (2 * rows, 2, 1009) and x.dtype == torch.bfloat16
    assert not x[:rows].any() and x[rows:].abs().max() > 4
    assert y[:rows].eq(1).all() and (y[rows:] == mod.IGNORE_INDEX).sum() == 1
    assert bitwise_check._results == {("deferred_cross_entropy", 1009): True}
