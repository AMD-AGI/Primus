###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU tests of the replica planners and the logical -> physical remap.

Set MOONEP_REF_DIR to a MoonEP checkout to also compare plan_moonep with MoonEP's own torch planning reference
(tests/planning_reference.py)."""

import importlib.util
import os

import pytest
import torch

from primus.backends.megatron.core.transformer.moe.replica import (
    alloc_fractions,
    make_plan,
    owner_alloc,
    physical_counts,
    plan_moonep,
    remap_topk,
    resplit_alloc,
    resplit_if_balanced,
    slot_lookup,
)


def zipf_topk(R, S, K, E, alpha, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.arange(1, E + 1, dtype=torch.float64).pow(-alpha)
    p = p[torch.randperm(E, generator=g)]
    logp = torch.log(p / p.sum()).float()
    out = []
    for _ in range(R):
        u = torch.rand(S, E, generator=g).clamp_(1e-20, 1 - 1e-7)
        out.append(torch.topk(logp + -torch.log(-torch.log(u)), K, dim=-1).indices)
    return torch.stack(out)  # [R, S, K]


def counts_of(topk, E):
    return torch.stack([torch.bincount(t.reshape(-1), minlength=E) for t in topk])


CASES = [(8, 512, 8, 64, a) for a in (0.0, 0.6, 1.2)] + [(4, 256, 2, 8, 0.9), (8, 1024, 8, 256, 0.9)]


def check_alloc(alloc, counts):
    assert torch.equal(alloc.sum(dim=1), counts.sum(dim=0)), "per-expert rows not conserved"
    assert bool((alloc >= 0).all())


@pytest.mark.parametrize("R,S,K,E,alpha", CASES)
def test_moonep_exact(R, S, K, E, alpha):
    topk = zipf_topk(R, S, K, E, alpha, seed=1)
    counts = counts_of(topk, E)
    plan = make_plan("moonep", counts, num_slots=E // R)
    check_alloc(plan.alloc, counts)
    assert bool((plan.rank_loads() == S * K).all()), "moonep must give every rank exactly S*K rows"
    assert int((plan.slot_expert >= 0).sum(dim=1).max()) <= E // R


@pytest.mark.parametrize("R,S,K,E,alpha", CASES)
@pytest.mark.parametrize("slots,min_rows,block", [(2, 1, 1), (4, 64, 1), (4, 1, 32)])
def test_ultraep_budget(R, S, K, E, alpha, slots, min_rows, block):
    topk = zipf_topk(R, S, K, E, alpha, seed=2)
    counts = counts_of(topk, E)
    plan = make_plan("ultraep", counts, num_slots=slots, min_rows=min_rows, block=block)
    check_alloc(plan.alloc, counts)
    base = owner_alloc(counts.sum(dim=0), R).sum(dim=0)
    assert int(plan.rank_loads().max()) <= int(base.max())
    L = E // R
    for d in range(R):
        for c in range(slots):
            e = int(plan.slot_expert[d, c])
            if e >= 0:
                assert e // L != d
                assert int(plan.alloc[e, d]) >= max(1, min_rows)
        remote = [e for e in range(E) if plan.alloc[e, d] > 0 and e // L != d]
        assert sorted(remote) == sorted(int(x) for x in plan.slot_expert[d] if x >= 0)


def test_ultraep_balances_a_hot_rank():
    R, L = 4, 4
    counts = torch.zeros(R, R * L, dtype=torch.int64)
    counts[:, 0] = 300  # expert 0 on rank 0 is hot
    counts[:, 1:] = 20
    plan = make_plan("ultraep", counts, num_slots=2)
    assert int(plan.rank_loads().max()) < int(owner_alloc(counts.sum(0), R).sum(0).max())
    assert plan.num_replicas() >= 1


@pytest.mark.parametrize("seed", [9, 14])
@pytest.mark.parametrize("alpha", [0.6, 0.9, 1.2])
def test_ultraep_near_balance_with_min_rows(alpha, seed):
    # a remainder below min_rows must not make a tau infeasible (the binary search assumes monotone feasibility)
    R, S, K, E = 8, 8192, 8, 256
    counts = counts_of(zipf_topk(R, S, K, E, alpha, seed=seed), E)
    plan = make_plan("ultraep", counts, num_slots=4, min_rows=1024)
    check_alloc(plan.alloc, counts)
    assert int(plan.rank_loads().max()) <= 1.02 * S * K


def test_ultraep_min_gain_skips_small_imbalance():
    R, S, K, E = 8, 8192, 8, 256
    mild = counts_of(zipf_topk(R, S, K, E, 0.3, seed=10), E)
    hot = counts_of(zipf_topk(R, S, K, E, 0.9, seed=10), E)
    assert make_plan("ultraep", mild, num_slots=4, min_rows=1024).num_replicas() > 0
    skip = make_plan("ultraep", mild, num_slots=4, min_rows=1024, min_gain=0.2)
    assert skip.num_replicas() == 0
    assert torch.equal(skip.alloc, owner_alloc(mild.sum(dim=0), R))
    assert make_plan("ultraep", hot, num_slots=4, min_rows=1024, min_gain=0.2).num_replicas() > 0


def test_ultraep_replica_cost():
    R, S, K, E = 8, 8192, 8, 256
    hot = counts_of(zipf_topk(R, S, K, E, 0.9, seed=10), E)
    plans = [make_plan("ultraep", hot, num_slots=4, min_rows=1024, replica_cost=c) for c in (0, 2000, 10**9)]
    for p in plans:
        check_alloc(p.alloc, hot)
    n = [p.num_replicas() for p in plans]
    assert n[0] > 0 and n[1] <= n[0] and n[2] == 0
    # the busiest rank counting the cost of the replicas it hosts never ends above the plan without replicas
    raw = int(owner_alloc(hot.sum(dim=0), R).sum(dim=0).max())
    p = plans[1]
    assert int((p.rank_loads() + 2000 * (p.slot_expert >= 0).sum(dim=1)).max()) <= raw


def test_deterministic():
    topk = zipf_topk(8, 512, 8, 64, 0.9, seed=3)
    counts = counts_of(topk, 64)
    for planner in ("moonep", "ultraep"):
        a = make_plan(planner, counts, num_slots=8)
        b = make_plan(planner, counts.clone(), num_slots=8)
        assert torch.equal(a.alloc, b.alloc) and torch.equal(a.slot_expert, b.slot_expert)


@pytest.mark.parametrize("planner", ["moonep", "ultraep"])
def test_fixed_reuses_slots(planner):
    R, S, K, E = 8, 512, 8, 64
    first = counts_of(zipf_topk(R, S, K, E, 0.9, seed=4), E)
    plan = make_plan(planner, first, num_slots=E // R)
    later = counts_of(zipf_topk(R, S, K, E, 0.9, seed=5), E)
    again = make_plan("fixed", later, num_slots=E // R, slot_expert=plan.slot_expert)
    check_alloc(again.alloc, later)
    assert torch.equal(again.slot_expert, plan.slot_expert)
    L = E // R
    for e in range(E):
        for d in range(R):
            if again.alloc[e, d] > 0 and e // L != d:
                assert e in [int(x) for x in plan.slot_expert[d]]


@pytest.mark.parametrize("planner", ["moonep", "ultraep"])
def test_resplit_keeps_slots_and_shares(planner):
    R, S, K, E = 8, 512, 8, 64
    L = E // R
    first = counts_of(zipf_topk(R, S, K, E, 0.9, seed=4), E)
    plan = make_plan(planner, first, num_slots=E // R)
    cum = alloc_fractions(plan.alloc, L)
    assert torch.equal(resplit_alloc(cum, first.sum(dim=0)), plan.alloc)
    later = counts_of(zipf_topk(R, S, K, E, 0.9, seed=5), E)
    alloc = resplit_alloc(cum, later.sum(dim=0))
    check_alloc(alloc, later)
    held = (plan.alloc > 0) | (torch.arange(E).unsqueeze(1) // L == torch.arange(R))
    assert bool(((alloc > 0) <= held).all()), "rows sent to a rank that holds no copy of the expert"


@pytest.mark.parametrize("planner", ["moonep", "ultraep"])
def test_resplit_if_balanced_keeps_or_replans(planner):
    R, S, K, E = 8, 512, 8, 64
    first = counts_of(zipf_topk(R, S, K, E, 0.9, seed=4), E)
    plan = make_plan(planner, first, num_slots=E // R)
    later = counts_of(zipf_topk(R, S, K, E, 0.9, seed=5), E)
    fixed = make_plan("fixed", later, num_slots=E // R, slot_expert=plan.slot_expert)
    loads = fixed.rank_loads().to(torch.float64)
    imbalance = (loads.max() / loads.mean()).item()
    kept = resplit_if_balanced(later, plan.slot_expert, E // R, 1, imbalance + 1e-9)
    assert kept is not None
    assert torch.equal(kept.alloc, fixed.alloc) and torch.equal(kept.slot_expert, plan.slot_expert)
    assert resplit_if_balanced(later, plan.slot_expert, E // R, 1, imbalance - 1e-6) is None


@pytest.mark.parametrize("planner,slots", [("moonep", 8), ("ultraep", 4)])
def test_remap_matches_alloc(planner, slots):
    R, S, K, E = 8, 512, 8, 64
    L = E // R
    topk = zipf_topk(R, S, K, E, 1.2, seed=6)
    counts = counts_of(topk, E)
    plan = make_plan(planner, counts, num_slots=slots)
    lut = slot_lookup(plan)
    total = torch.zeros(R, L + slots, dtype=torch.int64)
    for r in range(R):
        phys = remap_topk(topk[r], counts, r, plan, lut, check=True)
        assert phys.shape == topk[r].shape
        assert int(phys.max()) < R * (L + slots)
        total += physical_counts(phys, R, L, slots)
    for e in range(E):
        for d in range(R):
            if e // L == d:
                got = int(total[d, e % L])
            elif plan.alloc[e, d] > 0:
                got = int(total[d, L + int(lut[d, e])])
            else:
                continue
            assert got == int(plan.alloc[e, d]), (e, d)
    assert bool((total.sum(dim=1) == plan.rank_loads()).all())


def test_remap_identity_without_replicas():
    R, S, K, E = 4, 128, 2, 16
    L, B = E // R, 2
    topk = zipf_topk(R, S, K, E, 0.0, seed=7)
    counts = counts_of(topk, E)
    plan = make_plan("fixed", counts, num_slots=B, slot_expert=torch.full((R, B), -1, dtype=torch.int64))
    for r in range(R):
        phys = remap_topk(topk[r], counts, r, plan)
        assert torch.equal(phys, (topk[r] // L) * (L + B) + topk[r] % L)


@pytest.mark.skipif(not os.environ.get("MOONEP_REF_DIR"), reason="MOONEP_REF_DIR not set")
@pytest.mark.parametrize("R,S,K,E,alpha", CASES)
def test_moonep_matches_reference(R, S, K, E, alpha):
    path = os.path.join(os.environ["MOONEP_REF_DIR"], "tests", "planning_reference.py")
    spec = importlib.util.spec_from_file_location("moonep_planning_reference", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    topk = zipf_topk(R, S, K, E, alpha, seed=8)
    counts = counts_of(topk, E)
    ctx = {"rank": 0, "R": R, "E": E, "S": S, "K": K, "NvS_capacity": S * K}
    want = ref.launch_planning_torch_reference(ctx, topk.int(), counts.int(), return_alloc=True)  # [R, E]
    assert torch.equal(plan_moonep(counts), want.t().to(torch.int64))
