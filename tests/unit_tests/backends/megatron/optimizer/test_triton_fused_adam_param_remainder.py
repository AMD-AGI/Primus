###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TritonFusedAdam's param-remainder path (the precision-aware optimizer: bf16 params, int16 remainders)
against TE FusedAdam, bit for bit, and the clip coefficient it takes from deferred_clip."""

import pytest
import torch

from primus.core.utils.gpu_arch import is_gfx1250

if not torch.cuda.is_available():
    pytest.skip("TritonFusedAdam requires a GPU", allow_module_level=True)
if not is_gfx1250():
    pytest.skip("the Triton param-remainder step runs on gfx1250 only", allow_module_level=True)

from transformer_engine.pytorch.optimizers import FusedAdam  # noqa: E402

from primus.backends.megatron.core.optimizer import deferred_clip  # noqa: E402
from primus.backends.megatron.core.optimizer.triton_fused_adam import (  # noqa: E402
    TritonFusedAdam,
)

SHAPES = [(4096,), (1000, 37), (3, 5), (2048, 4096), (1,), (32, 7168)]
HPARAMS = dict(lr=7.2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


@pytest.fixture(autouse=True)
def _folding():
    enabled = deferred_clip.ENABLED
    deferred_clip.ENABLED = True
    yield
    deferred_clip.ENABLED = enabled


def _kwargs(moments, decoupled):
    return dict(
        master_weights=True,
        store_param_remainders=True,
        exp_avg_dtype=moments,
        exp_avg_sq_dtype=moments,
        use_decoupled_grad=decoupled,
    )


def _pair(moments, decoupled, groups=False):
    gen = torch.Generator(device="cuda").manual_seed(0)
    ref = [
        torch.nn.Parameter((torch.randn(s, device="cuda", generator=gen) * 0.02).to(torch.bfloat16))
        for s in SHAPES
    ]
    tri = [torch.nn.Parameter(p.detach().clone()) for p in ref]
    if groups:  # Megatron's two groups: with and without weight decay
        ref_g = [{"params": ref[:3]}, {"params": ref[3:], "weight_decay": 0.0}]
        tri_g = [{"params": tri[:3]}, {"params": tri[3:], "weight_decay": 0.0}]
    else:
        ref_g, tri_g = ref, tri
    return (
        FusedAdam(ref_g, **HPARAMS, **_kwargs(moments, decoupled)),
        ref,
        TritonFusedAdam(tri_g, **HPARAMS, **_kwargs(moments, decoupled)),
        tri,
    )


def _set_grads(ref, tri, step, decoupled, dtype=torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(100 + step)
    for a, b in zip(ref, tri):
        g = (torch.randn(a.shape, device="cuda", generator=gen) * 1e-3).to(dtype)
        if decoupled:
            a.decoupled_grad, b.decoupled_grad = g.clone(), g.clone()
        else:
            a.grad, b.grad = g.clone(), g.clone()


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _assert_same(ref_opt, ref, tri_opt, tri):
    for a, b in zip(ref, tri):
        assert torch.equal(_bits(a.data), _bits(b.data))
        for name in ("exp_avg", "exp_avg_sq", "master_param"):
            assert torch.equal(_bits(ref_opt.state[a][name]), _bits(tri_opt.state[b][name])), name
    for ga, gb in zip(ref_opt.param_groups, tri_opt.param_groups):
        assert ga.get("step") == gb.get("step")


def _spy_te(monkeypatch):
    calls = []
    orig = FusedAdam.step

    def spy(self, *a, **k):
        calls.append(type(self).__name__)
        return orig(self, *a, **k)

    monkeypatch.setattr(FusedAdam, "step", spy)
    return calls


@pytest.mark.parametrize("moments", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("decoupled", [True, False])
@pytest.mark.parametrize("groups", [False, True])
def test_matches_te(moments, decoupled, groups, monkeypatch):
    ref_opt, ref, tri_opt, tri = _pair(moments, decoupled, groups)
    calls = _spy_te(monkeypatch)
    for step in range(4):
        _set_grads(ref, tri, step, decoupled)
        ref_opt.step()
        tri_opt.step()
    assert calls == ["FusedAdam"] * 4  # the Triton optimizer never fell back
    _assert_same(ref_opt, ref, tri_opt, tri)


def test_a_group_without_parameters_keeps_its_step_like_te(monkeypatch):
    """Megatron's distributed optimizer keeps a group whose parameters a rank holds none of."""
    ref_opt, ref, tri_opt, tri = _pair(torch.bfloat16, True, groups=True)
    for opt in (ref_opt, tri_opt):
        opt.param_groups[0]["params"] = []
        opt.load_state_dict(opt.state_dict())
    ref, tri = ref[3:], tri[3:]
    calls = _spy_te(monkeypatch)
    for step in range(2):
        _set_grads(ref, tri, step, True)
        ref_opt.step()
        tri_opt.step()
    assert calls == ["FusedAdam"] * 2  # the Triton optimizer never fell back
    _assert_same(ref_opt, ref, tri_opt, tri)


@pytest.mark.parametrize("coeff", [0.3141, 1.0])
def test_deferred_clip_matches_scaling_first(coeff, monkeypatch):
    ref_opt, ref, tri_opt, tri = _pair(torch.bfloat16, True, groups=True)
    calls = _spy_te(monkeypatch)
    c = torch.tensor([coeff], dtype=torch.float32, device="cuda")
    for step in range(3):
        _set_grads(ref, tri, step, True)
        for p in ref:
            p.decoupled_grad.copy_((p.decoupled_grad.float() * c).to(torch.bfloat16))
        assert deferred_clip.defer(tri, c)
        ref_opt.step()
        tri_opt.step()
        assert all(deferred_clip.take(p) is None for p in tri)
    assert calls == ["FusedAdam"] * 3
    _assert_same(ref_opt, ref, tri_opt, tri)


def test_fallback_applies_the_deferred_clip_first(monkeypatch):
    ref_opt, ref, tri_opt, tri = _pair(torch.bfloat16, True)
    calls = _spy_te(monkeypatch)
    c = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    _set_grads(ref, tri, 0, True)
    for p in ref:
        p.decoupled_grad.copy_((p.decoupled_grad.float() * c).to(torch.bfloat16))
    assert deferred_clip.defer(tri, c)
    ref_opt.step()
    tri_opt.step(closure=lambda: None)  # a closure sends the step to TE
    assert calls == ["FusedAdam", "TritonFusedAdam"]
    _assert_same(ref_opt, ref, tri_opt, tri)


def test_defer_needs_every_param_folding_and_enabled():
    folding = [torch.nn.Parameter(torch.zeros(4, device="cuda"))]
    deferred_clip.mark_folding(folding)
    other = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    c = torch.ones(1, device="cuda")
    assert not deferred_clip.defer(folding + [other], c)
    assert deferred_clip.take(folding[0]) is None
    deferred_clip.ENABLED = False
    assert not deferred_clip.defer(folding, c)
    deferred_clip.ENABLED = True
    assert deferred_clip.defer(folding, c)
    assert deferred_clip.take(folding[0]) is c


def test_te_without_remainders_is_not_marked():
    p = torch.nn.Parameter(torch.zeros(8, device="cuda", dtype=torch.bfloat16))
    TritonFusedAdam([p], **HPARAMS, master_weights=True)  # no store_param_remainders: TE path
    assert not deferred_clip.defer([p], torch.ones(1, device="cuda"))


def test_params_swapped_in_after_construction_are_marked():
    """Megatron's distributed optimizer replaces param_groups with its shard params and reloads the state."""
    _, _, tri_opt, tri = _pair(torch.bfloat16, True)
    shards = [p.detach().view(-1)[: max(1, p.numel() // 2)] for p in tri]
    tri_opt.param_groups[0]["params"] = shards
    tri_opt.load_state_dict(tri_opt.state_dict())
    assert deferred_clip.defer(shards, torch.ones(1, device="cuda"))
