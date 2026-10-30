###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU tests: TE's fused router gets no materialized zero gradients for its routing maps."""

import importlib.util
from pathlib import Path

import pytest
import torch

from primus.core.patches import PatchContext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
te_router = pytest.importorskip("transformer_engine.pytorch.router")

# Load just this patch so the test does not import unrelated GPU patch modules.
_PATCH_PATH = (
    Path(__file__).resolve().parents[4]
    / "primus/backends/megatron/patches/moe_patches/te_router_materialize_grads_patches.py"
)
_spec = importlib.util.spec_from_file_location("te_router_materialize_grads_under_test", _PATCH_PATH)
patch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_mod)

FUNCTIONS = {name: getattr(te_router, name) for name in patch_mod._FUNCTIONS}
TOKENS, EXPERTS, TOPK = 1024, 128, 8


@pytest.fixture
def patched():
    saved = {name: (fn.__dict__["forward"], fn.__dict__["backward"]) for name, fn in FUNCTIONS.items()}
    patch_mod.patch_te_router_no_materialized_grads(PatchContext(backend="megatron", phase="before_train"))
    yield
    for name, fn in FUNCTIONS.items():
        fn.forward, fn.backward = saved[name]
        del fn._primus_no_materialize_grads


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
    saved = {name: (fn.__dict__["forward"], fn.__dict__["backward"]) for name, fn in FUNCTIONS.items()}
    patch_mod.patch_te_router_no_materialized_grads(PatchContext(backend="megatron", phase="before_train"))
    try:
        assert torch.equal(_router_grad(), reference)
    finally:
        for name, fn in FUNCTIONS.items():
            fn.forward, fn.backward = saved[name]
            del fn._primus_no_materialize_grads


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_routing_map_gradient_not_materialized(patched, monkeypatch, name):
    fn = FUNCTIONS[name]
    patched_backward, seen = fn.backward, []

    def spy(ctx, *grads):
        seen.append(grads)
        return patched_backward(ctx, *grads)

    monkeypatch.setattr(fn, "backward", staticmethod(spy))
    _router_grad()
    map_index = 1 - patch_mod._FUNCTIONS[name]  # the other output is the routing map
    assert len(seen) == 1
    assert seen[0][map_index] is None
    assert seen[0][patch_mod._FUNCTIONS[name]] is not None
