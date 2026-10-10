# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.

"""GPU integration contract for Core's batch/sequence layout and Turbo CE."""

import importlib.util
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@pytest.fixture
def installed_patch(monkeypatch):
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    from primus.core.patches.patch_registry import PatchRegistry

    # Loading the module registers its patch; keep the suite's global registry intact.
    monkeypatch.setattr(PatchRegistry, "_all_patches", [])
    monkeypatch.setattr(
        PatchRegistry,
        "_patches_by_backend_phase",
        defaultdict(lambda: defaultdict(list)),
    )
    calls = []

    def original(self, labels, logits):
        calls.append((labels, logits))
        return "fallback"

    monkeypatch.setattr(LanguageModule, "compute_language_model_loss", original)
    path = (
        Path(__file__).parents[2]
        / "primus/backends/megatron/patches/turbo/cross_entropy_patches.py"
    )
    spec = importlib.util.spec_from_file_location("ce_patch_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, LanguageModule, calls


@pytest.mark.parametrize("overwrite", [False, True])
def test_lm_head_backward_and_layout(installed_patch, monkeypatch, overwrite):
    mod, cls, calls = installed_patch
    monkeypatch.setenv("PRIMUS_TURBO_CE_OVERWRITE_INPUT", str(int(overwrite)))
    mod.patch_cross_entropy(None)
    # Exercise the real linear->loss->linear backward ownership contract.
    torch.manual_seed(312)
    h = torch.randn(7, 3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(1009, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    logits = F.linear(h, w)
    ref_logits = logits.detach().float().requires_grad_()
    labels = torch.randint(1009, (3, 7), device="cuda")
    labels[0, 0] = -100
    weights = torch.rand(3, 7, device="cuda")
    fake = SimpleNamespace(
        config=SimpleNamespace(
            tensor_model_parallel_size=1, cross_entropy_loss_fusion=True
        )
    )
    actual = cls.compute_language_model_loss(fake, labels, logits)
    ref = (
        F.cross_entropy(
            ref_logits.flatten(0, 1), labels.T.contiguous().flatten(), reduction="none"
        )
        .reshape(7, 3)
        .T.contiguous()
    )
    torch.testing.assert_close(actual, ref, rtol=2e-5, atol=2e-5)
    (actual * weights).sum().backward()
    (dlogits,) = torch.autograd.grad(ref, ref_logits, weights)
    expected_h = dlogits.to(h.dtype) @ w.detach()
    expected_w = dlogits.to(h.dtype).flatten(0, 1).T @ h.detach().flatten(0, 1)
    torch.testing.assert_close(h.grad, expected_h, rtol=0.008, atol=0.02)
    torch.testing.assert_close(w.grad, expected_w, rtol=0.008, atol=0.02)
    assert not calls


@pytest.mark.parametrize(
    "tp,fused,dtype",
    [(2, True, torch.bfloat16), (1, False, torch.bfloat16), (1, True, torch.float16)],
)
def test_fallback(installed_patch, tp, fused, dtype):
    mod, cls, calls = installed_patch
    mod.patch_cross_entropy(None)
    first = cls.compute_language_model_loss
    mod.patch_cross_entropy(None)
    assert cls.compute_language_model_loss is first
    fake = SimpleNamespace(
        config=SimpleNamespace(
            tensor_model_parallel_size=tp, cross_entropy_loss_fusion=fused
        )
    )
    logits = torch.empty(7, 3, 1009, device="cuda", dtype=dtype)
    labels = torch.zeros(3, 7, device="cuda", dtype=torch.long)
    assert cls.compute_language_model_loss(fake, labels, logits) == "fallback"
    assert len(calls) == 1
