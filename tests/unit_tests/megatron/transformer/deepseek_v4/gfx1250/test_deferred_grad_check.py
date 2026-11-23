###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Deferred, sync-free DDP gradient NaN/Inf check (``patch_deferred_grad_nan_check``).

With the patch applied, ``check_grads`` of every bucket group only folds the bucket's sum of squares into a device
accumulator and ``DistributedDataParallel.finish_grad_sync`` validates it once per step:

* clean gradients pass;
* a NaN or an Inf in any bucket is rejected with a ``RuntimeError`` at ``finish_grad_sync``;
* with ``PRIMUS_GRAD_NORM_FROM_NAN_CHECK=1`` the accumulated sum of squares is exposed once through
  ``pop_last_grad_sumsq`` and matches the fp64 sum of squares of all buckets.

Switch: ``PRIMUS_DEFERRED_GRAD_NAN_CHECK`` (``0`` restores the upstream check). The Megatron classes are restored at
the end of each test.
"""

from types import SimpleNamespace as NS

import pytest
import torch

from .helpers import requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


class _FakeGroup:
    def __init__(self, grads):
        self.buckets = [NS(grad_data=g) for g in grads]

    def finish_grad_sync(self, force_all_reduce=False):
        pass


def _mk(n):
    return (torch.randn(n, device=DEV) * 1e-2).to(torch.bfloat16)


@pytest.fixture
def patched(monkeypatch):
    pytest.importorskip("megatron.core")
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as DDP,
    )
    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBucketGroup as G,
    )

    from primus.backends.megatron.patches import ddp_deferred_grad_check_patches as D

    monkeypatch.setenv("PRIMUS_DEFERRED_GRAD_NAN_CHECK", "1")
    monkeypatch.setenv("PRIMUS_GRAD_NORM_FROM_NAN_CHECK", "1")
    saved = (G.check_grads, DDP.finish_grad_sync)
    D.patch_deferred_grad_nan_check(None)
    assert getattr(G.check_grads, "_primus_deferred", False)
    yield D, DDP, G
    G.check_grads, DDP.finish_grad_sync = saved


def _step(DDP, G, groups):
    fake = NS(bucket_groups=groups, expert_parallel_bucket_groups=[])
    for g in groups:
        G.check_grads(g, True, False)
    DDP.finish_grad_sync(fake)


def test_clean_grads_pass_and_sumsq_is_exposed(patched):
    D, DDP, G = patched
    groups = [_FakeGroup([_mk(1000), _mk(4097)]), _FakeGroup([_mk(123)])]
    _step(DDP, G, groups)
    got = D.pop_last_grad_sumsq()
    assert got is not None
    exact = sum(b.grad_data.double().pow(2).sum().item() for g in groups for b in g.buckets)
    assert abs(got.item() - exact) / exact < 1e-5
    assert D.pop_last_grad_sumsq() is None  # consumed once


@pytest.mark.parametrize("bad", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nan_or_inf_is_rejected(patched, bad):
    _, DDP, G = patched
    groups = [_FakeGroup([_mk(1000)]), _FakeGroup([_mk(10), _mk(5000)])]
    groups[1].buckets[1].grad_data[1234] = bad
    with pytest.raises(RuntimeError):
        _step(DDP, G, groups)
