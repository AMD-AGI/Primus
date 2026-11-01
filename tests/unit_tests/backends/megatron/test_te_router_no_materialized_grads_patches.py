###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: TE's fused router gets no materialized zero gradients for its routing maps."""

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches import PatchContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
te_router = pytest.importorskip("transformer_engine.pytorch.router")

# Load just this patch so the test does not import unrelated GPU patch modules.
_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/moe_patches/te_router_no_materialized_grads_patches.py"
)
_spec = importlib.util.spec_from_file_location("te_router_no_materialized_grads_under_test", _PATCH_PATH)
patch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_mod)

FUNCTIONS = {name: getattr(te_router, name) for name in patch_mod._FUNCTIONS}
# Index of the routing map among each Function's outputs
MAP_OUTPUT = {"FusedTopkScoreFunction": 1, "FusedComputeScoresForMoEAuxLoss": 0}
TOKENS, EXPERTS, TOPK = 1024, 128, 8


@contextlib.contextmanager
def _patched():
    saved = {name: (fn.__dict__["forward"], fn.__dict__["backward"]) for name, fn in FUNCTIONS.items()}
    patch_mod.patch_te_router_no_materialized_grads(PatchContext(backend="megatron", phase="before_train"))
    try:
        yield
    finally:
        for name, fn in FUNCTIONS.items():
            fn.forward, fn.backward = saved[name]
            del fn._primus_no_materialized_grads


@pytest.fixture
def patched():
    with _patched():
        yield


def _router_grad(seed=0):
    """d(logits) through both Functions, DeepSeek-V3's routing (sigmoid, grouped top-k, expert bias)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(TOKENS, EXPERTS, device="cuda", generator=g).requires_grad_()
    bias = torch.randn(EXPERTS, device="cuda", generator=g) * 0.1
    w_probs = torch.randn(TOKENS, EXPERTS, device="cuda", generator=g)
    w_scores = torch.randn(TOKENS, EXPERTS, device="cuda", generator=g)
    probs, _ = te_router.fused_topk_with_score_function(logits, TOPK, False, 8, 4, 2.5, "sigmoid", bias)
    _, scores = te_router.fused_compute_score_for_moe_aux_loss(logits, TOPK, "sigmoid")
    ((probs * w_probs).sum() + (scores * w_scores).sum()).backward()
    return logits.grad


def test_gradients_unchanged():
    reference = _router_grad()
    with _patched():
        assert torch.equal(_router_grad(), reference)


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_routing_map_gradient_not_materialized(patched, monkeypatch, name):
    fn = FUNCTIONS[name]
    patched_backward, seen = fn.backward, []

    def spy(ctx, *grads):
        seen.append(grads)
        return patched_backward(ctx, *grads)

    monkeypatch.setattr(fn, "backward", staticmethod(spy))
    _router_grad()
    assert len(seen) == 1
    assert seen[0][MAP_OUTPUT[name]] is None
    assert seen[0][1 - MAP_OUTPUT[name]] is not None


@pytest.mark.parametrize("gfx1250", [True, False])
@pytest.mark.parametrize("moe_router_fusion", [True, False])
def test_enabled_on_gfx1250_with_moe_router_fusion(monkeypatch, gfx1250, moe_router_fusion):
    monkeypatch.setattr(patch_mod, "is_gfx1250", lambda: gfx1250)
    ctx = PatchContext(
        backend="megatron",
        phase="before_train",
        extra={"module_config": SimpleNamespace(params=SimpleNamespace(moe_router_fusion=moe_router_fusion))},
    )
    assert patch_mod._enabled(ctx) == (gfx1250 and moe_router_fusion)
