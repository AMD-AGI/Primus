###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for ``ddp_prescaled_backward_patches``: the bucket factors fold into the backward seed only
when that is exact, and a 1/DP seed gives the gradients the bucket scaling would have given."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.core.patches import PatchContext

_PATCH_PATH = (
    Path(__file__).resolve().parents[4] / "primus/backends/megatron/patches/ddp_prescaled_backward_patches.py"
)
_spec = importlib.util.spec_from_file_location("ddp_prescaled_backward_under_test", _PATCH_PATH)
patch_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_mod)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.setattr(patch_mod, "_PRESCALE", None)
    monkeypatch.setattr(patch_mod, "_KEPT", False)


def _ddp(dense, expert):
    def buffer(factor, buckets=2):
        return SimpleNamespace(
            gradient_scaling_factor=factor,
            buckets=[SimpleNamespace(gradient_scaling_factor=factor) for _ in range(buckets)],
        )

    return SimpleNamespace(
        buffers=[buffer(f) for f in dense], expert_parallel_buffers=[buffer(f) for f in expert]
    )


def _factors(ddp):
    return list(patch_mod._bucket_factors(ddp))


@pytest.mark.parametrize(
    "x,want", [(0.25, True), (0.5, True), (2.0**-10, True), (1.0 / 3.0, False), (0.0, False), (-0.25, False)]
)
def test_power_of_two(x, want):
    assert patch_mod._power_of_two(x) is want


def test_folds_a_common_power_of_two():
    ddp = _ddp([0.25, 0.25], [0.25])
    assert patch_mod.fold_scaling_into_seed(ddp)
    assert set(_factors(ddp)) == {1.0}
    assert patch_mod._PRESCALE == 0.25
    # a second model chunk with the same factor folds too
    ddp2 = _ddp([0.25], [0.25])
    assert patch_mod.fold_scaling_into_seed(ddp2)
    assert set(_factors(ddp2)) == {1.0}


@pytest.mark.parametrize(
    "dense,expert",
    [
        ([0.25], [0.5]),  # expert and dense factors differ
        ([1.0 / 3.0], [1.0 / 3.0]),  # not exact as a seed scale
        ([1.0], [1.0]),  # nothing to fold
    ],
)
def test_declines(dense, expert):
    ddp = _ddp(dense, expert)
    before = _factors(ddp)
    assert not patch_mod.fold_scaling_into_seed(ddp)
    assert _factors(ddp) == before
    assert patch_mod._PRESCALE is None


def test_declines_when_not_foldable():
    ddp = _ddp([0.25], [0.25])
    assert not patch_mod.fold_scaling_into_seed(ddp, foldable=False)
    assert set(_factors(ddp)) == {0.25}
    assert patch_mod._PRESCALE is None


@pytest.mark.parametrize(
    "first, second",
    [
        (_ddp([0.25], [0.25]), _ddp([0.125], [0.125])),  # another 1/DP
        (_ddp([0.25], [0.25]), _ddp([1.0 / 3.0], [1.0 / 3.0])),  # a module that keeps its scaling
        (_ddp([1.0 / 3.0], [1.0 / 3.0]), _ddp([0.25], [0.25])),  # folding after one kept it
    ],
)
def test_a_conflicting_module_raises(first, second):
    patch_mod.fold_scaling_into_seed(first)
    before = _factors(second)
    with pytest.raises(RuntimeError, match="prescale_backward_by_dp: false"):
        patch_mod.fold_scaling_into_seed(second)
    assert _factors(second) == before


def test_a_module_kept_as_not_foldable_after_a_fold_raises():
    assert patch_mod.fold_scaling_into_seed(_ddp([0.25], [0.25]))
    with pytest.raises(RuntimeError, match="prescale_backward_by_dp: false"):
        patch_mod.fold_scaling_into_seed(_ddp([0.25], [0.25]), foldable=False)


def test_a_module_without_gradient_buffers_agrees_with_any():
    assert not patch_mod.fold_scaling_into_seed(_ddp([], []))
    assert patch_mod.fold_scaling_into_seed(_ddp([0.25], [0.25]))
    assert not patch_mod.fold_scaling_into_seed(_ddp([], []))
    assert patch_mod._PRESCALE == 0.25


@pytest.mark.parametrize(
    "config,ddp_config",
    [
        (SimpleNamespace(fp16=True), SimpleNamespace()),
        (SimpleNamespace(calculate_per_token_loss=True), SimpleNamespace()),
        (SimpleNamespace(), SimpleNamespace(average_in_collective=True)),
        (SimpleNamespace(moe_grad_scale_func=lambda: 1.0), SimpleNamespace()),
        (SimpleNamespace(mtp_grad_scale_func=lambda: 1.0), SimpleNamespace()),
    ],
)
def test_not_foldable(config, ddp_config):
    assert not patch_mod._foldable(config, ddp_config)


def test_foldable():
    assert patch_mod._foldable(SimpleNamespace(bf16=True), SimpleNamespace())


# applying the patch imports the Primus Megatron backend, which initializes aiter: a GPU in ROCm builds
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_scale_loss_carries_the_prescale_through_the_original(monkeypatch):
    pytest.importorskip("megatron.core.distributed")
    from megatron.core.distributed import DistributedDataParallel
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    from primus.backends.megatron.patches import _patch_guard

    def earlier_patch(self, loss):  # stays in effect underneath
        return self.get_loss_scale() * loss

    init, scale_loss = DistributedDataParallel.__init__, MegatronOptimizer.scale_loss
    try:
        MegatronOptimizer.scale_loss = earlier_patch
        patch_mod.patch_prescaled_backward(PatchContext(backend="megatron", phase="before_train"))
        patch_mod.patch_prescaled_backward(PatchContext(backend="megatron", phase="before_train"))
        assert MegatronOptimizer.scale_loss.__wrapped__ is earlier_patch  # wrapped once
        fake = SimpleNamespace(get_loss_scale=lambda: torch.tensor([2.0]))
        loss = torch.tensor([3.0])
        assert torch.equal(MegatronOptimizer.scale_loss(fake, loss), torch.tensor([6.0]))  # nothing folded
        monkeypatch.setattr(patch_mod, "_PRESCALE", 0.25)
        assert torch.equal(MegatronOptimizer.scale_loss(fake, loss), torch.tensor([1.5]))
    finally:
        DistributedDataParallel.__init__, MegatronOptimizer.scale_loss = init, scale_loss
        getattr(DistributedDataParallel, _patch_guard._SENTINEL_ATTR, set()).discard(patch_mod._PATCH_KEY)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_prescaled_seed_gives_the_scaled_gradients():
    """Two microbatches of a bf16 MLP accumulating into bf16 grads, as DDP's main_grad: seeding each
    backward with loss / 4 gives the bits of seeding with loss and scaling the sum by 1/4 (one fp32
    product rounded once, like the fused check pass)."""
    torch.manual_seed(0)
    hidden, inter, tokens = 512, 1024, 256
    w1 = (torch.randn(inter, hidden, device="cuda") * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(hidden, inter, device="cuda") * 0.05).to(torch.bfloat16)
    xs = [torch.randn(tokens, hidden, device="cuda").to(torch.bfloat16) for _ in range(2)]

    def grads(seed_scale):
        p1, p2 = w1.clone().requires_grad_(), w2.clone().requires_grad_()
        for x in xs:
            y = torch.nn.functional.linear(torch.nn.functional.gelu(torch.nn.functional.linear(x, p1)), p2)
            loss = y.float().pow(2).mean() / len(xs)
            (loss * seed_scale).backward()
        return p1.grad, p2.grad

    ref = [(g.float() * 0.25).to(torch.bfloat16) for g in grads(torch.tensor(1.0, device="cuda"))]
    got = grads(torch.tensor(0.25, device="cuda"))
    for r, g in zip(ref, got):
        assert torch.equal(r.view(torch.int16), g.view(torch.int16))
