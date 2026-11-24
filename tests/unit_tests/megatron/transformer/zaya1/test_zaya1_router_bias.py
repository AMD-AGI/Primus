###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The router's balancing-bias update: main's arithmetic bit for bit, and a host sync only on the first update."""

import pytest
import torch
from test_zaya1_math import _cfg  # noqa: E402  (pytest puts this dir on sys.path)

from primus.backends.megatron.core.models.zaya1.zaya1_modules import ZayaRouter

_BUFFERS = ("balancing_biases", "balance_m", "balance_v", "balance_step")
_DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU.")),
]


def _main_update(router, counts):
    """ZayaRouter._update_biases on main, from the data-parallel-summed counts on."""
    with torch.no_grad():
        total = counts.sum().clamp_min(1.0)
        grad = counts / total - (1.0 / router.num_scored)
        router.balance_step += 1
        step = int(router.balance_step.item())
        b1, b2 = router.balance_beta1, router.balance_beta2
        router.balance_m.mul_(b1).add_(grad, alpha=1.0 - b1)
        router.balance_v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
        mhat = router.balance_m / (1.0 - b1**step)
        vhat = router.balance_v / (1.0 - b2**step)
        if router.balance_wd != 0.0:
            router.balancing_biases.mul_(1.0 - router.balance_lr * router.balance_wd)
        router.balancing_biases.addcdiv_(mhat, vhat.sqrt().add_(1e-8), value=-router.balance_lr)


def _skewed_counts(steps, num_scored, device):
    """Summed token counts per update: most tokens on a few slots, often none on others."""
    gen = torch.Generator().manual_seed(0)
    weights = torch.rand(steps, num_scored, generator=gen) ** 4
    return torch.poisson(weights * 2000.0, generator=gen).to(device)


def _routers(device, **overrides):
    """Two routers holding the same weights and buffers."""
    torch.manual_seed(0)
    cfg = _cfg(num_moe_experts=16, **overrides)
    router = ZayaRouter(cfg).to(device)
    ref = ZayaRouter(cfg).to(device)
    ref.load_state_dict(router.state_dict())
    return router, ref


def _assert_same_buffers(router, ref, step):
    for name in _BUFFERS:
        got, expected = getattr(router, name), getattr(ref, name)
        assert got.dtype == expected.dtype and torch.equal(got, expected), f"{name} after update {step}"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_bias_update_matches_main_bit_for_bit(device, weight_decay):
    router, ref = _routers(device, zaya_balance_wd=weight_decay)
    for step, counts in enumerate(_skewed_counts(1000, router.num_scored, device), start=1):
        router._apply_bias_update(counts)
        _main_update(ref, counts)
        _assert_same_buffers(router, ref, step)


@pytest.mark.parametrize("device", _DEVICES)
def test_loading_a_checkpoint_rereads_the_step(device):
    router, ref = _routers(device)
    counts = _skewed_counts(30, router.num_scored, device)
    for c in counts[:10]:
        router._apply_bias_update(c)
        _main_update(ref, c)
    # A checkpoint ten updates further along than the router.
    for c in counts[10:20]:
        _main_update(ref, c)
    router.load_state_dict(ref.state_dict())
    for step, c in enumerate(counts[20:], start=21):
        router._apply_bias_update(c)
        _main_update(ref, c)
        _assert_same_buffers(router, ref, step)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU.")
def test_only_the_first_update_waits_on_the_host():
    router, _ = _routers("cuda")
    counts = _skewed_counts(4, router.num_scored, "cuda")
    router._apply_bias_update(counts[0])
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        try:
            counts[0].sum().item()
        except RuntimeError:
            pass
        else:
            pytest.skip("torch.cuda.set_sync_debug_mode doesn't catch host syncs on this platform.")
        for c in counts[1:]:
            router._apply_bias_update(c)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
