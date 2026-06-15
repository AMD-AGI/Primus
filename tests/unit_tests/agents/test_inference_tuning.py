###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for the inference tuning search space (``inference_tuning.py``).

Focus: the DeepEP (MoE All-to-All overlap) axis — legality, validation, the
seed plan, and the evaluator CLI mapping that turns the knob into a
``--enable-deepep`` flag on ``projection inference``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("primus.agents.tuning_agent.inference_tuning")

from primus.agents.tuning_agent.config import (  # noqa: E402
    OptimizationConfig,
    TargetCluster,
)
from primus.agents.tuning_agent.inference_tuning import (  # noqa: E402
    InferenceTrialConfig,
    build_inference_seed_plan,
    derive_inference_legality,
    validate_inference,
)
from primus.agents.tuning_agent.workload import ArchitectureRecord  # noqa: E402
from primus.core.projection.inference_projection.collectives import (  # noqa: E402
    InferenceCollectiveModel,
    deepep_overlap_efficiency,
)
from primus.core.projection.inference_projection.memory import (  # noqa: E402
    project_inference_memory,
)
from primus.core.projection.training_config import (  # noqa: E402
    DisaggregationConfig,
    InferenceCollectiveConfig,
    InferenceConfig,
    InferenceRequestConfig,
    ModelConfig,
    ModelParallelConfig,
)


def _moe_model_config(**kw) -> ModelConfig:
    base = dict(
        num_layers=8,
        hidden_size=4096,
        num_attention_heads=32,
        num_experts=8,
        moe_router_topk=2,
        moe_pattern=[1] * 8,
    )
    base.update(kw)
    return ModelConfig(**base)


def _dense_arch(**kw) -> ArchitectureRecord:
    base = dict(
        model_name="dense",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        seq_length=4096,
        is_moe=False,
    )
    base.update(kw)
    return ArchitectureRecord(**base)


def _moe_arch(**kw) -> ArchitectureRecord:
    base = dict(
        model_name="moe",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        seq_length=4096,
        is_moe=True,
        num_experts=8,
        moe_router_topk=2,
    )
    base.update(kw)
    return ArchitectureRecord(**base)


def _cluster(num_nodes=1, gpus_per_node=8) -> TargetCluster:
    return TargetCluster(num_nodes=num_nodes, gpus_per_node=gpus_per_node, gpu_arch="mi355x")


# ─────────────────────────────────────────────────────────────────────────────
# legality
# ─────────────────────────────────────────────────────────────────────────────


def test_deepep_axis_only_offered_for_moe():
    assert derive_inference_legality(_moe_arch(), _cluster()).use_turbo_deepep == [False, True]
    assert derive_inference_legality(_dense_arch(), _cluster()).use_turbo_deepep == [False]


# ─────────────────────────────────────────────────────────────────────────────
# validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_deepep_rejected_for_dense():
    arch = _dense_arch()
    cluster = _cluster()
    leg = derive_inference_legality(arch, cluster)
    cfg = InferenceTrialConfig(tp=2, use_turbo_deepep=True)
    ok, reason = validate_inference(cfg, arch, cluster, leg)
    assert not ok
    assert "MoE" in reason


def test_validate_deepep_accepted_for_moe():
    arch = _moe_arch()
    cluster = _cluster()
    leg = derive_inference_legality(arch, cluster)
    cfg = InferenceTrialConfig(tp=2, ep=2, use_turbo_deepep=True)
    ok, reason = validate_inference(cfg, arch, cluster, leg)
    assert ok, reason


# ─────────────────────────────────────────────────────────────────────────────
# seed plan
# ─────────────────────────────────────────────────────────────────────────────


def test_seed_plan_includes_deepep_candidate_for_moe():
    arch = _moe_arch()
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    plan = build_inference_seed_plan(arch, cluster, OptimizationConfig(), max_candidates=64)
    assert any(c.use_turbo_deepep for c in plan.candidates)


def test_seed_plan_has_no_deepep_for_dense():
    arch = _dense_arch()
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    plan = build_inference_seed_plan(arch, cluster, OptimizationConfig(), max_candidates=64)
    assert all(not c.use_turbo_deepep for c in plan.candidates)


# ─────────────────────────────────────────────────────────────────────────────
# evaluator CLI mapping
# ─────────────────────────────────────────────────────────────────────────────


def test_build_inference_cmd_emits_enable_deepep():
    from pathlib import Path

    from primus.agents.tuning_agent.evaluator import _build_inference_cmd

    agent_cfg = SimpleNamespace(
        target_cluster=SimpleNamespace(gpu_arch="mi355x", gpu_clock_mhz=None),
        optimization=SimpleNamespace(hbm_capacity_gb=192.0),
    )
    primus_root = Path("/tmp/primus")

    on = _build_inference_cmd(
        Path("/tmp/w.yaml"),
        InferenceTrialConfig(tp=2, ep=2, use_turbo_deepep=True),
        agent_cfg,
        primus_root,
    )
    off = _build_inference_cmd(
        Path("/tmp/w.yaml"),
        InferenceTrialConfig(tp=2, ep=2, use_turbo_deepep=False),
        agent_cfg,
        primus_root,
    )
    assert "--enable-deepep" in on
    assert "--enable-deepep" not in off


# ─────────────────────────────────────────────────────────────────────────────
# oracle: DeepEP overlap in the inference projection (what the knob drives)
# ─────────────────────────────────────────────────────────────────────────────


def test_deepep_overlap_efficiency_ladder():
    assert deepep_overlap_efficiency(_moe_model_config()) == 0.0
    assert deepep_overlap_efficiency(_moe_model_config(use_turbo_deepep=True)) == 0.65
    assert deepep_overlap_efficiency(_moe_model_config(turbo_sync_free_moe_stage=1)) == 0.75
    assert deepep_overlap_efficiency(_moe_model_config(turbo_sync_free_moe_stage=2)) == 0.80
    assert deepep_overlap_efficiency(_moe_model_config(turbo_sync_free_moe_stage=3)) == 0.85


def test_ep_a2a_reduced_by_deepep_overlap():
    mp = ModelParallelConfig(expert_model_parallel_size=2)
    cc = InferenceCollectiveConfig()
    base = InferenceCollectiveModel(_moe_model_config(), mp, cc)
    deepep = InferenceCollectiveModel(_moe_model_config(use_turbo_deepep=True), mp, cc)

    a2a_base = base.ep_a2a_ms(batch=8, tokens=1024)
    a2a_deepep = deepep.ep_a2a_ms(batch=8, tokens=1024)
    assert a2a_base > 0.0
    # DeepEP hides 65% of the A2A → exposed cost is ~35% of the baseline.
    assert a2a_deepep == pytest.approx(a2a_base * (1.0 - 0.65), rel=1e-6)


def test_ep_a2a_zero_when_no_expert_parallelism():
    mp = ModelParallelConfig(expert_model_parallel_size=1)
    cc = InferenceCollectiveConfig()
    deepep = InferenceCollectiveModel(_moe_model_config(use_turbo_deepep=True), mp, cc)
    # EP<=1 → no A2A at all, DeepEP is a no-op.
    assert deepep.ep_a2a_ms(batch=8, tokens=1024) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# schema resolvers: cudagraph_mode + transfer_backend presets
# ─────────────────────────────────────────────────────────────────────────────


def test_cudagraph_mode_presets_resolve():
    # Unset → no overhead / penalty (legacy behaviour).
    req = InferenceRequestConfig()
    assert req.resolved_decode_step_overhead_us() == 0.0
    assert req.resolved_mixed_batch_penalty() == 0.0
    # "full" graph → tiny per-step overhead, no mixed penalty.
    full = InferenceRequestConfig(cudagraph_mode="full")
    assert full.resolved_decode_step_overhead_us() == 3.0
    assert full.resolved_mixed_batch_penalty() == 0.0
    # "piecewise" → moderate overhead + mixed-step penalty.
    pw = InferenceRequestConfig(cudagraph_mode="piecewise")
    assert pw.resolved_decode_step_overhead_us() == 8.0
    assert pw.resolved_mixed_batch_penalty() == 0.15
    # "none" (eager) → highest per-step overhead.
    none = InferenceRequestConfig(cudagraph_mode="none")
    assert none.resolved_decode_step_overhead_us() == 40.0


def test_cudagraph_explicit_overrides_preset():
    req = InferenceRequestConfig(cudagraph_mode="full", decode_step_overhead_us=12.0)
    # Explicit non-zero low-level knob wins over the preset.
    assert req.resolved_decode_step_overhead_us() == 12.0


def test_transfer_backend_presets_resolve():
    base = DisaggregationConfig()
    assert base.resolved_kv_transfer_bw_gbps() is None
    assert base.resolved_kv_transfer_latency_us() == 0.0

    nixl = DisaggregationConfig(transfer_backend="nixl")
    assert nixl.resolved_kv_transfer_bw_gbps() == 400.0
    assert nixl.resolved_kv_transfer_latency_us() == 5.0

    mooncake = DisaggregationConfig(transfer_backend="mooncake")
    assert mooncake.resolved_kv_transfer_bw_gbps() == 200.0

    # Explicit link knob wins over the preset.
    override = DisaggregationConfig(transfer_backend="nixl", kv_transfer_bw_gbps=123.0)
    assert override.resolved_kv_transfer_bw_gbps() == 123.0


# ─────────────────────────────────────────────────────────────────────────────
# preset knobs: agent legality / validation / seed plan / evaluator mapping
# ─────────────────────────────────────────────────────────────────────────────


def test_preset_knobs_legality_defaults():
    leg = derive_inference_legality(_moe_arch(), _cluster())
    assert leg.cudagraph_mode == ["none", "piecewise", "full"]
    assert leg.kv_cache_memory_fraction == [0.8, 0.85, 0.9]
    assert leg.transfer_backend == ["nixl", "mooncake", "mori"]


def test_validate_kv_fraction_range():
    arch, cluster = _moe_arch(), _cluster()
    leg = derive_inference_legality(arch, cluster)
    ok, _ = validate_inference(
        InferenceTrialConfig(tp=2, kv_cache_memory_fraction=0.9), arch, cluster, leg
    )
    assert ok
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, kv_cache_memory_fraction=1.5), arch, cluster, leg
    )
    assert not bad
    assert "kv_cache_memory_fraction" in reason


def test_validate_transfer_backend_requires_disaggregate():
    arch, cluster = _moe_arch(), _cluster()
    leg = derive_inference_legality(arch, cluster)
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, transfer_backend="nixl"), arch, cluster, leg
    )
    assert not bad
    assert "disaggregate" in reason


def test_validate_cudagraph_mode_membership():
    arch, cluster = _moe_arch(), _cluster()
    leg = derive_inference_legality(arch, cluster)
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, cudagraph_mode="bogus"), arch, cluster, leg
    )
    assert not bad
    assert "cudagraph_mode" in reason


def test_seed_plan_includes_preset_knob_candidates():
    arch = _moe_arch()
    # 2 nodes so the disaggregated prefill+decode pools (and thus the
    # transfer-backend candidate) fit within the GPU budget.
    cluster = _cluster(num_nodes=2, gpus_per_node=8)
    plan = build_inference_seed_plan(arch, cluster, OptimizationConfig(), max_candidates=128)
    assert any(c.cudagraph_mode == "full" for c in plan.candidates)
    assert any(c.kv_cache_memory_fraction for c in plan.candidates)
    assert any(c.transfer_backend == "nixl" for c in plan.candidates)


def test_kv_cache_memory_fraction_bounds_usable_hbm():
    mc = _moe_model_config()
    mp = ModelParallelConfig()
    full = project_inference_memory(
        InferenceConfig(
            model_config=mc,
            request_config=InferenceRequestConfig(input_seq_len=512, output_seq_len=128),
            model_parallel_config=mp,
        ),
        rank=0,
        hbm_capacity_gb=192.0,
        verbose=False,
    )
    # A tiny usable fraction starves the engine of headroom: the same workload
    # that fits in full HBM no longer fits once the fraction is applied.
    tiny = project_inference_memory(
        InferenceConfig(
            model_config=mc,
            request_config=InferenceRequestConfig(
                input_seq_len=512, output_seq_len=128, kv_cache_memory_fraction=0.0005
            ),
            model_parallel_config=mp,
        ),
        rank=0,
        hbm_capacity_gb=192.0,
        verbose=False,
    )
    assert full.fits is True
    assert tiny.fits is False


def test_build_inference_cmd_emits_preset_knob_flags():
    from pathlib import Path

    from primus.agents.tuning_agent.evaluator import _build_inference_cmd

    agent_cfg = SimpleNamespace(
        target_cluster=SimpleNamespace(gpu_arch="mi355x", gpu_clock_mhz=None),
        optimization=SimpleNamespace(hbm_capacity_gb=192.0),
    )
    primus_root = Path("/tmp/primus")

    cmd = _build_inference_cmd(
        Path("/tmp/w.yaml"),
        InferenceTrialConfig(
            tp=2,
            cudagraph_mode="full",
            kv_cache_memory_fraction=0.9,
            disaggregate=True,
            prefill_tp=2,
            decode_tp=2,
            transfer_backend="nixl",
        ),
        agent_cfg,
        primus_root,
    )
    assert "--cudagraph-mode" in cmd
    assert "full" in cmd
    assert "--kv-cache-memory-fraction" in cmd
    assert "--transfer-backend" in cmd
    # transfer-backend must not leak when disaggregation is off.
    no_disagg = _build_inference_cmd(
        Path("/tmp/w.yaml"),
        InferenceTrialConfig(tp=2, cudagraph_mode="none"),
        agent_cfg,
        primus_root,
    )
    assert "--transfer-backend" not in no_disagg
