###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Embedding gradient written straight into main_grad (``embedding_grad_into``) and its gather / sum-of-squares helpers.

* The result equals ``aten::embedding_dense_backward`` byte for byte for more than 3072 tokens, where ATen takes its
  sorted path (skewed and uniform token ids, several repeats
  per row), with the output pre-filled with NaN so every row must be written or cleared.
* Two consecutive calls (second one with different ids) stay exact with the touched-rows clear
  (``PRIMUS_GLUE_EMB_TOUCHED``) on and off, and with the Triton sort (``PRIMUS_LAUNCH_EMB_SORT``) on and off.
* The folded partial sums (``sq=``) add up to sum(main_grad^2) and, fed to ``MultiSumsq`` as extras, give the same
  grad-norm^2 as reducing the whole gradient.
* The one-launch stable sort (``PRIMUS_LAUNCH_EMB_SORT=1``) returns the values and permutation of
  ``torch.sort(stable=True)``.
* Gathering rows from the FP32 master equals ``F.embedding`` of its BF16 rounding (ties, subnormals, large values).
"""

import pytest
import torch
import torch.nn.functional as F

from .helpers import bits_equal, requires_gfx1250

pytestmark = requires_gfx1250

DEV = "cuda"


def _ids(V, B, S, skew, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if skew:  # many repeats per row
        hot = torch.randint(0, V, (64,), device=DEV, generator=g)
        a = hot[torch.randint(0, 64, (B * S // 2,), device=DEV, generator=g)]
        b = torch.randint(0, V, (B * S - B * S // 2,), device=DEV, generator=g)
        return torch.cat([a, b])[torch.randperm(B * S, device=DEV, generator=g)].view(B, S)
    return torch.randint(0, V, (B, S), device=DEV, generator=g)


@pytest.mark.parametrize("touched", ["0", "1"])
@pytest.mark.parametrize("sort", ["0", "1"])
@pytest.mark.parametrize("V, D, B, S, skew", [(5000, 512, 1, 4096, True), (3000, 1536, 2, 2048, False)])
def test_embedding_grad_into_matches_aten(monkeypatch, V, D, B, S, skew, sort, touched):
    from primus.backends.megatron.patches.turbo import emb_main_grad_patches as EM

    monkeypatch.setenv("PRIMUS_GLUE_EMB_TOUCHED", touched)
    monkeypatch.setenv("PRIMUS_LAUNCH_EMB_SORT", sort)
    monkeypatch.setattr(EM, "_touched", {})  # the touched-rows state is keyed by the buffer address
    torch.manual_seed(0)
    out = torch.full((V, D), float("nan"), device=DEV, dtype=torch.bfloat16)
    for it in range(3):
        ids = _ids(V, B, S, skew, 100 + it)
        g = (torch.randn(B, S, D, device=DEV) * 3).to(torch.bfloat16)
        ref = torch.ops.aten.embedding_dense_backward(g, ids, V, -1, False)
        EM.embedding_grad_into(g, ids, out)
        torch.cuda.synchronize()
        assert bits_equal(ref, out), f"call {it}"


@pytest.mark.parametrize("touched", ["0", "1"])
def test_embedding_grad_sumsq_fold(monkeypatch, touched):
    from primus.backends.megatron.core.distributed.grad_sumsq import (
        MultiSumsq,
        grad_sumsq,
    )
    from primus.backends.megatron.patches.turbo import emb_main_grad_patches as EM

    monkeypatch.setenv("PRIMUS_GLUE_EMB_TOUCHED", touched)
    monkeypatch.setattr(EM, "_touched", {})
    V, D, N = 5000, 2048, 1024
    g = torch.Generator(device=DEV).manual_seed(1)
    grad = (torch.randn((1, N, D), device=DEV, generator=g) * 1e-3).to(torch.bfloat16)
    other = (torch.randn((4_000_000,), device=DEV, generator=g) * 1e-3).to(torch.bfloat16)
    mg_ref = torch.empty((V, D), dtype=torch.bfloat16, device=DEV).fill_(7.0)
    mg_new = torch.empty((V, D), dtype=torch.bfloat16, device=DEV).fill_(7.0)
    sq = torch.full((N * (D // 1024),), float("nan"), dtype=torch.float32, device=DEV)
    for it in range(2):
        ids = _ids(V, 1, N, True, 7 + it)
        EM.embedding_grad_into(grad, ids, mg_ref)
        EM.embedding_grad_into(grad, ids, mg_new, sq=sq)
        torch.cuda.synchronize()
        assert bits_equal(mg_ref, mg_new)
        assert bool(torch.isfinite(sq).all())
        emb = grad_sumsq(mg_ref.view(-1)).item()
        assert abs(sq.sum().item() - emb) / emb < 1e-6
        full = MultiSumsq([other, mg_ref.view(-1)], [])().item()
        fold = MultiSumsq([other], [sq])().item()
        assert abs(fold - full) / full <= 1e-6


def test_gather_from_master_equals_embedding_of_bf16():
    from primus.backends.megatron.patches.turbo import emb_main_grad_patches as EM

    V, D, N = 4000, 1024, 1024
    g = torch.Generator(device=DEV).manual_seed(2)
    master = torch.randn((V, D), device=DEV, generator=g) * 0.02
    mi = master.view(torch.int32)
    mi[:64] = (mi[:64] & ~0xFFFF) | 0x8000  # exact RNE ties
    master[96:128] *= 1e-38
    master[128:160] *= 1e30
    ids = torch.cat([_ids(V, 1, N - 200, True, 3).view(-1), torch.arange(200, device=DEV)]).view(1, N)
    a = F.embedding(ids, master.to(torch.bfloat16))
    b = EM._gather_from_master(ids, master)
    assert bits_equal(a, b)


@pytest.mark.parametrize(
    "n, v, seed", [(4096, 129280, 0), (4096, 50, 1), (3000, 1000, 3), (8192, 129280, 4), (1, 10, 5)]
)
def test_launch_sort_equals_torch_stable_sort(monkeypatch, n, v, seed):
    from primus.backends.megatron.patches.turbo import emb_main_grad_patches as EM

    monkeypatch.setenv("PRIMUS_LAUNCH_EMB_SORT", "1")
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.randint(0, v, (n,), device=DEV, generator=g, dtype=torch.int64)
    ref = torch.sort(idx, stable=True)
    got = EM._launch_sort(idx, 129280)
    assert got is not None
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
