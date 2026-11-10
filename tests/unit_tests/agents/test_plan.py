###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the tuning-agent deterministic seed planner (``plan.py``).

build_seed_plan generates a capped, baseline-anchored set of legal TrialConfigs
before the LLM runs. Pure CPU: derives legality + sweeps axes, no projection /
LLM calls.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("primus.agents.tuning_agent.plan")

from primus.agents.tuning_agent.config import (  # noqa: E402
    AgentConfig,
    BenchmarkHost,
    LLMConfig,
    OptimizationConfig,
    TargetCluster,
)
from primus.agents.tuning_agent.plan import SeedPlan, build_seed_plan  # noqa: E402
from primus.agents.tuning_agent.workload import ArchitectureRecord  # noqa: E402


def _arch(moe: bool = False, **kw) -> ArchitectureRecord:
    base = dict(
        model_name="moe" if moe else "dense",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        seq_length=4096,
        is_moe=moe,
        global_batch_size=128,
        micro_batch_size=2,
    )
    if moe:
        base.update(num_experts=8, moe_router_topk=2)
    base.update(kw)
    return ArchitectureRecord(**base)


def _cfg(num_nodes: int = 1, gpus_per_node: int = 8) -> AgentConfig:
    return AgentConfig(
        target_cluster=TargetCluster(num_nodes=num_nodes, gpus_per_node=gpus_per_node, gpu_arch="mi355x"),
        benchmark_host=BenchmarkHost(),
        optimization=OptimizationConfig(),
        llm=LLMConfig(),
    )


# (moe, max_candidates): dense default cluster vs MoE branch (2 nodes for EP room).
@pytest.mark.parametrize("moe, max_candidates", [(False, 6), (False, 3), (True, 12)])
def test_seed_plan_nonempty_and_capped(moe, max_candidates):
    nodes = 2 if moe else 1
    plan = build_seed_plan(_arch(moe=moe), _cfg(num_nodes=nodes), max_candidates=max_candidates)
    assert isinstance(plan, SeedPlan)
    assert 0 < len(plan.candidates) <= max_candidates
    assert plan.rationale  # non-empty explanation


def test_seed_plan_includes_baseline_parallelism():
    arch = _arch()
    plan = build_seed_plan(arch, _cfg(), max_candidates=12)
    baseline = (
        arch.tensor_model_parallel_size,
        arch.pipeline_model_parallel_size,
        arch.expert_model_parallel_size,
        arch.context_parallel_size,
    )
    # TrialConfig uses short axis names (tp/pp/ep/cp); ArchitectureRecord uses full names.
    combos = {(c.tp, c.pp, c.ep, c.cp) for c in plan.candidates}
    assert baseline in combos


def test_seed_plan_moe_nonempty_and_capped():
    # Exercises the MoE branch of build_seed_plan (EP axis derivation).
    plan = build_seed_plan(_arch(moe=True), _cfg(num_nodes=2, gpus_per_node=8), max_candidates=12)
    assert 0 < len(plan.candidates) <= 12


def test_seed_plan_candidates_are_unique():
    plan = build_seed_plan(_arch(), _cfg(), max_candidates=12)
    serialized = [json.dumps(c.as_dict(), sort_keys=True) for c in plan.candidates]
    assert len(serialized) == len(set(serialized))


def _gptoss_like(**kw) -> ArchitectureRecord:
    """A MoE model that ships no recompute and uses all 8 GPUs for EP, like gpt-oss-120B."""
    return _arch(
        moe=True,
        num_layers=36,
        hidden_size=2880,
        num_attention_heads=64,
        num_experts=128,
        moe_router_topk=4,
        expert_model_parallel_size=8,
        micro_batch_size=8,
        use_turbo_deepep=True,
        fp8="hybrid",
        **kw,
    )


def test_seed_plan_has_no_equivalent_duplicates():
    # No recompute vs "none", and VPP at PP=1, are the same run.
    from primus.agents.tuning_agent.legality import config_signature

    plan = build_seed_plan(_gptoss_like(), _cfg(), max_candidates=64)
    sigs = [config_signature(c.as_dict()) for c in plan.candidates]
    assert len(sigs) == len(set(sigs))
    assert all(c.vpp is None for c in plan.candidates if c.pp == 1)


def test_seed_plan_sweeps_recompute_when_the_workload_ships_none():
    plan = build_seed_plan(_gptoss_like(), _cfg(), max_candidates=24)
    full = [c for c in plan.candidates if c.recompute_granularity == "full"]
    assert {c.recompute_granularity for c in plan.candidates} >= {"selective", "full"}
    assert full and all(c.recompute_num_layers == 36 // c.pp for c in full)


def test_seed_plan_reaches_pipeline_layouts_with_a_fitted_ep():
    # Keeping EP=8 makes every PP>1 layout illegal on one node.
    plan = build_seed_plan(_gptoss_like(), _cfg(), max_candidates=64)
    layouts = {(c.tp, c.pp, c.ep) for c in plan.candidates}
    assert {(1, 2, 4), (1, 4, 2), (2, 1, 4)} <= layouts
    assert all(c.tp * c.pp * c.ep <= 8 for c in plan.candidates)
