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
    default_inference_trial,
    derive_inference_legality,
    inference_trial_from_dict,
    validate_inference,
)
from primus.agents.tuning_agent.workload import ArchitectureRecord  # noqa: E402
from primus.core.projection.inference_projection.collectives import (  # noqa: E402
    InferenceCollectiveModel,
    deepep_overlap_efficiency,
)
from primus.core.projection.inference_projection.kv_cache import (  # noqa: E402
    estimate_kv_cache,
    max_concurrent_sequences,
)
from primus.core.projection.inference_projection.memory import (  # noqa: E402
    project_inference_memory,
)
from primus.core.projection.inference_projection.performance import (  # noqa: E402
    InferencePerformanceProjector,
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


# ─────────────────────────────────────────────────────────────────────────────
# serving knobs: kv_block_size (paged-KV fragmentation)
# ─────────────────────────────────────────────────────────────────────────────


def _sized_moe_model_config(**kw) -> ModelConfig:
    # Realistic sizes so the analytical simulator (attention SDPA, grouped GEMM,
    # KV-cache bytes) has non-zero dimensions to work with.
    base = dict(
        kv_channels=128,
        num_query_groups=8,
        group_query_attention=True,
        ffn_hidden_size=14336,
        moe_ffn_hidden_size=14336,
        swiglu=True,
        padded_vocab_size=32000,
    )
    base.update(kw)
    return _moe_model_config(**base)


def _moe_inference_config(mp=None, **req_kw):
    return InferenceConfig(
        model_config=_sized_moe_model_config(),
        request_config=InferenceRequestConfig(**req_kw),
        model_parallel_config=mp or ModelParallelConfig(),
    )


def test_kv_block_size_rounds_up_per_sequence_bytes():
    # context_len = input + output = 100 + 5 = 105 → rounds up to 7 blocks of 16
    # (= 112 tokens), inflating per-sequence + total bytes and lowering capacity.
    layers = 8
    cont = _moe_inference_config(input_seq_len=100, output_seq_len=5)
    paged = _moe_inference_config(
        input_seq_len=100, output_seq_len=5, kv_block_size=16
    )
    kv_cont = estimate_kv_cache(cont, layers, concurrency=4)
    kv_paged = estimate_kv_cache(paged, layers, concurrency=4)

    assert kv_cont.max_context_len == 105
    # 105 rounds up to 112 → 112/105 inflation factor on bytes.
    assert kv_paged.bytes_per_sequence == pytest.approx(
        kv_cont.bytes_per_sequence * 112.0 / 105.0
    )
    assert kv_paged.bytes_total > kv_cont.bytes_total


def test_kv_block_size_lowers_max_concurrency():
    layers = 8
    cont = _moe_inference_config(input_seq_len=100, output_seq_len=5)
    paged = _moe_inference_config(
        input_seq_len=100, output_seq_len=5, kv_block_size=16
    )
    free = 4.0 * estimate_kv_cache(cont, layers, concurrency=1).bytes_per_sequence
    conc_cont = max_concurrent_sequences(cont, layers, free)
    conc_paged = max_concurrent_sequences(paged, layers, free)
    # Same free bytes hold strictly fewer paged sequences (rounding overhead).
    assert conc_paged < conc_cont


def test_kv_block_size_zero_is_noop():
    layers = 8
    cont = _moe_inference_config(input_seq_len=128, output_seq_len=128)
    paged = _moe_inference_config(
        input_seq_len=128, output_seq_len=128, kv_block_size=16
    )
    # 256 is already a multiple of 16 → no fragmentation, identical bytes.
    assert (
        estimate_kv_cache(paged, layers).bytes_per_sequence
        == estimate_kv_cache(cont, layers).bytes_per_sequence
    )


# ─────────────────────────────────────────────────────────────────────────────
# serving knobs: ep_load_balance + redundant_experts (MoE imbalance)
# ─────────────────────────────────────────────────────────────────────────────


def test_resolved_ep_imbalance_formula():
    # Balanced → no-op.
    assert InferenceRequestConfig(ep_load_balance=1.0).resolved_ep_imbalance(8) == 1.0
    # No redundancy → ratio passes through.
    assert InferenceRequestConfig(ep_load_balance=1.5).resolved_ep_imbalance(8) == 1.5
    # Redundant experts dilute the surplus: 1 + (1.5-1) * 8/(8+8) = 1.25.
    assert InferenceRequestConfig(
        ep_load_balance=1.5, redundant_experts=8
    ).resolved_ep_imbalance(8) == pytest.approx(1.25)
    # Anything <= 1.0 (or num_experts=0) is clamped / a no-op.
    assert InferenceRequestConfig(ep_load_balance=0.5).resolved_ep_imbalance(8) == 1.0
    assert InferenceRequestConfig(ep_load_balance=1.5).resolved_ep_imbalance(0) == 1.5


def _moe_sim_args():
    return SimpleNamespace(gpu_arch="mi300x", gpu_clock_mhz=None, gemm_backend=None)


def test_ep_load_balance_raises_moe_expert_compute():
    mp = ModelParallelConfig(expert_model_parallel_size=2)

    def moe_ms(bal, red=0):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                mp, input_seq_len=512, output_seq_len=128,
                ep_load_balance=bal, redundant_experts=red,
            ),
            args=_moe_sim_args(),
        )
        return proj._forward_times(8, 1, "decode", 600).moe_layer_ms

    balanced = moe_ms(1.0)
    imbalanced = moe_ms(1.5)
    mitigated = moe_ms(1.5, red=8)
    assert imbalanced > balanced
    # Redundant experts partially recover the loss → strictly between the two.
    assert balanced < mitigated < imbalanced


def test_ep_load_balance_noop_without_expert_parallelism():
    # EP=1 → imbalance is a no-op even with a skewed ep_load_balance.
    mp = ModelParallelConfig(expert_model_parallel_size=1)

    def moe_ms(bal):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                mp, input_seq_len=512, output_seq_len=128, ep_load_balance=bal
            ),
            args=_moe_sim_args(),
        )
        return proj._forward_times(8, 1, "decode", 600).moe_layer_ms

    assert moe_ms(1.5) == pytest.approx(moe_ms(1.0))


# ─────────────────────────────────────────────────────────────────────────────
# serving knobs: max_num_batched_tokens (per-step token cap)
# ─────────────────────────────────────────────────────────────────────────────


def test_max_num_batched_tokens_cap_raises_tpot():
    mp = ModelParallelConfig(expert_model_parallel_size=2)

    def metrics(cap):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                mp, input_seq_len=4096, output_seq_len=128, batch_size=16,
                chunked_prefill_size=2048, max_num_batched_tokens=cap,
            ),
            args=_moe_sim_args(),
        )
        return proj._continuous_decode_metrics(4096, 128, 16)

    uncapped = metrics(0)
    capped = metrics(512)  # tiny cap forces many small prefill chunks
    # A small cap splits prefill into more mixed steps → higher TPOT and a
    # larger mixed-step fraction than the uncapped path.
    assert capped["tpot_ms"] > uncapped["tpot_ms"]
    assert capped["mixed_step_fraction"] > uncapped["mixed_step_fraction"]


def test_max_num_batched_tokens_zero_is_noop():
    mp = ModelParallelConfig(expert_model_parallel_size=2)

    def metrics(cap):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                mp, input_seq_len=2048, output_seq_len=128, batch_size=8,
                chunked_prefill_size=1024, max_num_batched_tokens=cap,
            ),
            args=_moe_sim_args(),
        )
        return proj._continuous_decode_metrics(2048, 128, 8)

    # A cap large enough to never bind reproduces the uncapped TPOT exactly.
    assert metrics(0)["tpot_ms"] == pytest.approx(metrics(10_000_000)["tpot_ms"])


# ─────────────────────────────────────────────────────────────────────────────
# serving knobs: agent legality / validation / seed plan / evaluator mapping
# ─────────────────────────────────────────────────────────────────────────────


def test_serving_knob_legality_defaults():
    leg = derive_inference_legality(_moe_arch(), _cluster())
    assert leg.kv_block_size == [0, 16, 32]
    assert leg.max_num_batched_tokens == [0, 2048, 8192]
    assert leg.ep_load_balance == [1.0, 1.2, 1.5]
    assert leg.redundant_experts == [0, 8, 16]


def test_validate_kv_block_and_token_cap_ranges():
    arch, cluster = _moe_arch(), _cluster()
    leg = derive_inference_legality(arch, cluster)
    ok, _ = validate_inference(
        InferenceTrialConfig(tp=2, kv_block_size=16, max_num_batched_tokens=8192),
        arch, cluster, leg,
    )
    assert ok
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, kv_block_size=-1), arch, cluster, leg
    )
    assert not bad and "kv_block_size" in reason
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, max_num_batched_tokens=-5), arch, cluster, leg
    )
    assert not bad and "max_num_batched_tokens" in reason


def test_validate_ep_imbalance_moe_only():
    moe, dense, cluster = _moe_arch(), _dense_arch(), _cluster()
    leg_moe = derive_inference_legality(moe, cluster)
    leg_dense = derive_inference_legality(dense, cluster)
    # MoE accepts a skewed load + redundant experts.
    ok, reason = validate_inference(
        InferenceTrialConfig(tp=2, ep=2, ep_load_balance=1.5, redundant_experts=8),
        moe, cluster, leg_moe,
    )
    assert ok, reason
    # Non-MoE rejects either knob (mirrors the use_turbo_deepep MoE-only check).
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, ep_load_balance=1.5), dense, cluster, leg_dense
    )
    assert not bad and "MoE" in reason
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, redundant_experts=8), dense, cluster, leg_dense
    )
    assert not bad and "MoE" in reason
    # Out-of-range ratio rejected.
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, ep=2, ep_load_balance=0.5), moe, cluster, leg_moe
    )
    assert not bad and "ep_load_balance" in reason


def test_seed_plan_includes_serving_knob_candidates():
    arch = _moe_arch()
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    plan = build_inference_seed_plan(arch, cluster, OptimizationConfig(), max_candidates=128)
    assert any(c.kv_block_size == 16 for c in plan.candidates)
    assert any(c.max_num_batched_tokens == 8192 for c in plan.candidates)
    assert any(c.ep_load_balance == 1.3 for c in plan.candidates)
    assert any(c.redundant_experts == 8 for c in plan.candidates)


def test_seed_plan_no_ep_imbalance_for_dense():
    arch = _dense_arch()
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    plan = build_inference_seed_plan(arch, cluster, OptimizationConfig(), max_candidates=128)
    assert all(c.ep_load_balance == 1.0 for c in plan.candidates)
    assert all(c.redundant_experts == 0 for c in plan.candidates)


def test_build_inference_cmd_emits_serving_knob_flags():
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
            tp=2, ep=2, kv_block_size=16, max_num_batched_tokens=8192,
            ep_load_balance=1.3, redundant_experts=8,
        ),
        agent_cfg,
        primus_root,
    )
    assert "--kv-block-size" in cmd and "16" in cmd
    assert "--max-num-batched-tokens" in cmd and "8192" in cmd
    assert "--ep-load-balance" in cmd and "1.3" in cmd
    assert "--redundant-experts" in cmd

    # Defaults (balanced, no cap, no paging) must not leak any flags.
    plain = _build_inference_cmd(
        Path("/tmp/w.yaml"), InferenceTrialConfig(tp=2, ep=2), agent_cfg, primus_root
    )
    assert "--kv-block-size" not in plain
    assert "--max-num-batched-tokens" not in plain
    assert "--ep-load-balance" not in plain
    assert "--redundant-experts" not in plain


# ─────────────────────────────────────────────────────────────────────────────
# new serving knobs: schema resolvers (request_config / collective_config)
# ─────────────────────────────────────────────────────────────────────────────


def test_attention_backend_multiplier_presets():
    assert InferenceRequestConfig().resolved_attention_backend_multiplier() == 1.0
    assert InferenceRequestConfig(attention_backend="aiter").resolved_attention_backend_multiplier() == 0.85
    assert InferenceRequestConfig(attention_backend="ck").resolved_attention_backend_multiplier() == 0.90
    assert InferenceRequestConfig(attention_backend="triton").resolved_attention_backend_multiplier() == 1.0
    assert InferenceRequestConfig(attention_backend="hip").resolved_attention_backend_multiplier() == 1.10
    # Unknown backend → no-op.
    assert InferenceRequestConfig(attention_backend="bogus").resolved_attention_backend_multiplier() == 1.0


def test_sparse_attention_scale():
    # Dense (topk=0) → no scaling regardless of context.
    assert InferenceRequestConfig().resolved_sparse_attention_scale(8192) == 1.0
    # Short context (<= topk) → no scaling.
    assert InferenceRequestConfig(sparse_attention_topk=2048).resolved_sparse_attention_scale(1024) == 1.0
    # Long context → scales toward topk/context.
    s = InferenceRequestConfig(sparse_attention_topk=2048).resolved_sparse_attention_scale(8192)
    assert s == pytest.approx(2048.0 / 8192.0)
    # Very long context floors out (projections + indexer don't shrink).
    assert InferenceRequestConfig(sparse_attention_topk=64).resolved_sparse_attention_scale(100000) == 0.15


def test_moe_expert_dtype_speedup():
    assert InferenceRequestConfig().resolved_moe_expert_dtype_speedup() == 1.0
    assert InferenceRequestConfig(moe_expert_dtype="bf16").resolved_moe_expert_dtype_speedup() == 1.0
    assert InferenceRequestConfig(moe_expert_dtype="fp8").resolved_moe_expert_dtype_speedup() == 0.55
    assert InferenceRequestConfig(moe_expert_dtype="mxfp4").resolved_moe_expert_dtype_speedup() == 0.35


def test_fused_kernels_cut_step_overhead():
    # No fusion → preset overhead unchanged.
    pw = InferenceRequestConfig(cudagraph_mode="piecewise")
    assert pw.resolved_decode_step_overhead_us() == 8.0
    # Fused kernels cut the per-step launch overhead by 30%.
    fused = InferenceRequestConfig(cudagraph_mode="piecewise", fused_kernels=True)
    assert fused.resolved_decode_step_overhead_us() == pytest.approx(8.0 * 0.7)
    # Also applies on top of an explicit overhead.
    explicit = InferenceRequestConfig(decode_step_overhead_us=20.0, fused_kernels=True)
    assert explicit.resolved_decode_step_overhead_us() == pytest.approx(20.0 * 0.7)


def test_quick_reduce_and_fused_rmsnorm_speed_up_tp_allreduce():
    mp = ModelParallelConfig(tensor_model_parallel_size=2)
    base = InferenceCollectiveModel(_moe_model_config(), mp, InferenceCollectiveConfig())
    quick = InferenceCollectiveModel(
        _moe_model_config(), mp, InferenceCollectiveConfig(quick_reduce=True)
    )
    fused = InferenceCollectiveModel(
        _moe_model_config(), mp, InferenceCollectiveConfig(fuse_rmsnorm_allreduce=True)
    )
    both = InferenceCollectiveModel(
        _moe_model_config(),
        mp,
        InferenceCollectiveConfig(quick_reduce=True, fuse_rmsnorm_allreduce=True),
    )
    ar = base.tp_allreduce_ms(batch=8, tokens=1024)
    assert ar > 0.0
    assert quick.tp_allreduce_ms(8, 1024) == pytest.approx(ar * 0.6)
    assert fused.tp_allreduce_ms(8, 1024) == pytest.approx(ar * 0.8)
    assert both.tp_allreduce_ms(8, 1024) == pytest.approx(ar * 0.6 * 0.8)


# ─────────────────────────────────────────────────────────────────────────────
# new serving knobs: performance-model effects (simulation path)
# ─────────────────────────────────────────────────────────────────────────────


def test_attention_backend_scales_attention_compute():
    def total(backend):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                input_seq_len=512, output_seq_len=128, attention_backend=backend
            ),
            args=_moe_sim_args(),
        )
        return proj._forward_times(8, 1, "decode", 600).total_ms

    base = total(None)
    aiter = total("aiter")
    hip = total("hip")
    # aiter shaves attention time; hip inflates it (proves non-zero attention).
    assert aiter < base < hip


def test_sparse_attention_reduces_decode_attention():
    def total(topk):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                input_seq_len=512, output_seq_len=128, sparse_attention_topk=topk
            ),
            args=_moe_sim_args(),
        )
        return proj._forward_times(8, 1, "decode", 8192).total_ms

    dense = total(0)
    sparse = total(512)
    assert sparse < dense


def test_moe_expert_dtype_speeds_up_moe_layer():
    def moe_ms(dtype):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                input_seq_len=512, output_seq_len=128, moe_expert_dtype=dtype
            ),
            args=_moe_sim_args(),
        )
        return proj._forward_times(8, 1, "decode", 600).moe_layer_ms

    assert moe_ms("fp8") < moe_ms(None)
    assert moe_ms("mxfp4") < moe_ms("fp8")


def test_speculative_draft_cost_raises_decode_step():
    def step(dcf):
        proj = InferencePerformanceProjector(
            _moe_inference_config(
                input_seq_len=512, output_seq_len=128,
                speculative_num_tokens=4, speculative_acceptance_rate=0.7,
                speculative_draft_cost_factor=dcf,
            ),
            args=_moe_sim_args(),
        )
        return proj._decode_step_latency_ms(8, 600, q_len=4)

    assert step(0.0) < step(0.3)


def test_request_rate_queueing_adds_to_ttft():
    proj = InferencePerformanceProjector(
        _moe_inference_config(
            input_seq_len=512, output_seq_len=100,
            request_rate=5.0, arrival_model="poisson",
        ),
        args=_moe_sim_args(),
    )
    # mu = 1000/100 = 10 req/s, rho = 0.5 → Wq = 0.5/0.5 * (1000/10) = 100 ms.
    q = proj._request_rate_queueing(1000.0, 100, ttft_ms=50.0, request_latency_ms=200.0)
    assert q["utilization"] == pytest.approx(0.5)
    assert q["queue_wait_ms"] == pytest.approx(100.0)
    assert q["ttft_with_queue_ms"] == pytest.approx(150.0)
    assert q["saturated"] == 0.0
    # Deterministic arrivals → ~half the wait.
    proj_d = InferencePerformanceProjector(
        _moe_inference_config(
            input_seq_len=512, output_seq_len=100,
            request_rate=5.0, arrival_model="deterministic",
        ),
        args=_moe_sim_args(),
    )
    assert proj_d._request_rate_queueing(1000.0, 100, 50.0, 200.0)["queue_wait_ms"] == pytest.approx(50.0)


def test_request_rate_saturation_flags():
    proj = InferencePerformanceProjector(
        _moe_inference_config(
            input_seq_len=512, output_seq_len=100,
            request_rate=20.0, arrival_model="poisson",
        ),
        args=_moe_sim_args(),
    )
    # Offered 20 > sustainable 10 → saturated, large finite penalty.
    q = proj._request_rate_queueing(1000.0, 100, 50.0, 200.0)
    assert q["saturated"] == 1.0
    assert q["queue_wait_ms"] > 0.0


def test_request_rate_closed_loop_is_noop():
    proj = InferencePerformanceProjector(
        _moe_inference_config(input_seq_len=512, output_seq_len=100),
        args=_moe_sim_args(),
    )
    assert proj._request_rate_queueing(1000.0, 100, 50.0, 200.0) == {}


# ─────────────────────────────────────────────────────────────────────────────
# new serving knobs: agent legality / validation / seed / evaluator mapping
# ─────────────────────────────────────────────────────────────────────────────


def test_new_knob_legality_defaults():
    leg = derive_inference_legality(_moe_arch(), _cluster())
    assert leg.arrival_model == ["closed", "poisson", "deterministic"]
    assert leg.attention_backend == ["aiter", "triton", "ck", "hip"]
    assert leg.sparse_attention_topk == [0, 512, 2048]
    assert leg.moe_expert_dtype == ["bf16", "fp8", "mxfp4"]


def test_validate_request_rate_rules():
    arch, cluster = _moe_arch(), _cluster()
    leg = derive_inference_legality(arch, cluster)
    ok, _ = validate_inference(
        InferenceTrialConfig(tp=2, request_rate=8.0, arrival_model="poisson"),
        arch, cluster, leg,
    )
    assert ok
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, request_rate=8.0, arrival_model="closed"),
        arch, cluster, leg,
    )
    assert not bad and "request_rate" in reason


def test_validate_moe_expert_dtype_moe_only():
    moe, dense, cluster = _moe_arch(), _dense_arch(), _cluster()
    leg_moe = derive_inference_legality(moe, cluster)
    leg_dense = derive_inference_legality(dense, cluster)
    ok, reason = validate_inference(
        InferenceTrialConfig(tp=2, ep=2, moe_expert_dtype="mxfp4"), moe, cluster, leg_moe
    )
    assert ok, reason
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, moe_expert_dtype="mxfp4"), dense, cluster, leg_dense
    )
    assert not bad and "MoE" in reason


def test_validate_collective_ops_require_tp():
    arch, cluster = _moe_arch(), _cluster()
    leg = derive_inference_legality(arch, cluster)
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=1, quick_reduce=True), arch, cluster, leg
    )
    assert not bad and "quick_reduce" in reason
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=1, fuse_rmsnorm_allreduce=True), arch, cluster, leg
    )
    assert not bad and "fuse_rmsnorm_allreduce" in reason


def test_validate_draft_cost_requires_speculative():
    arch, cluster = _moe_arch(), _cluster()
    leg = derive_inference_legality(arch, cluster)
    bad, reason = validate_inference(
        InferenceTrialConfig(tp=2, speculative_draft_cost_factor=0.2), arch, cluster, leg
    )
    assert not bad and "speculative" in reason


def test_seed_plan_includes_new_knob_candidates():
    arch = _moe_arch()
    cluster = _cluster(num_nodes=1, gpus_per_node=8)
    plan = build_inference_seed_plan(arch, cluster, OptimizationConfig(), max_candidates=256)
    assert any(c.attention_backend == "aiter" for c in plan.candidates)
    assert any(c.moe_expert_dtype == "mxfp4" for c in plan.candidates)
    assert any(c.fused_kernels for c in plan.candidates)
    assert any(c.speculative_draft_cost_factor > 0 for c in plan.candidates)
    assert any(c.request_rate > 0 and c.arrival_model == "poisson" for c in plan.candidates)


def test_build_inference_cmd_emits_new_knob_flags():
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
            tp=2, ep=2,
            request_rate=8.0, arrival_model="poisson",
            attention_backend="aiter", sparse_attention_topk=2048,
            moe_expert_dtype="mxfp4", fused_kernels=True,
            speculative_num_tokens=4, speculative_acceptance_rate=0.7,
            speculative_draft_cost_factor=0.2,
            quick_reduce=True, fuse_rmsnorm_allreduce=True,
        ),
        agent_cfg,
        primus_root,
    )
    assert "--request-rate" in cmd and "--arrival-model" in cmd
    assert "--attention-backend" in cmd and "aiter" in cmd
    assert "--sparse-attention-topk" in cmd and "2048" in cmd
    assert "--moe-expert-dtype" in cmd and "mxfp4" in cmd
    assert "--fused-kernels" in cmd
    assert "--speculative-draft-cost-factor" in cmd
    assert "--quick-reduce" in cmd
    assert "--fuse-rmsnorm-allreduce" in cmd

    plain = _build_inference_cmd(
        Path("/tmp/w.yaml"), InferenceTrialConfig(tp=2, ep=2), agent_cfg, primus_root
    )
    assert "--request-rate" not in plain
    assert "--attention-backend" not in plain
    assert "--moe-expert-dtype" not in plain
    assert "--fused-kernels" not in plain
    assert "--quick-reduce" not in plain


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent plumbing: partial-proposal → full profile-anchored trial
# ─────────────────────────────────────────────────────────────────────────────


def test_default_inference_trial_profile_anchored():
    arch, cluster = _moe_arch(), _cluster(num_nodes=1, gpus_per_node=8)
    leg = derive_inference_legality(arch, cluster)
    base = default_inference_trial(arch, cluster, OptimizationConfig())
    # Largest legal TP that fits the 8-GPU budget; batch 1; bf16; profile lens.
    assert base.tp == max(t for t in leg.tp if t <= 8)
    assert base.batch_size == 1
    assert base.weight_dtype == "bf16" and base.kv_cache_dtype == "bf16"
    assert base.input_len == 1024 and base.output_len == 128
    # The baseline itself must be legal so the agent can always anchor on it.
    ok, reason = validate_inference(base, arch, cluster, leg)
    assert ok, reason


def test_inference_trial_from_dict_overlays_only_given_keys():
    arch, cluster = _moe_arch(), _cluster(num_nodes=1, gpus_per_node=8)
    base = default_inference_trial(arch, cluster, OptimizationConfig())
    cfg = inference_trial_from_dict(
        {
            "batch_size": 16,
            "request_rate": 8.0,
            "arrival_model": "poisson",
            "attention_backend": "aiter",
            "moe_expert_dtype": "mxfp4",
            "sparse_attention_topk": 2048,
            "fused_kernels": True,
        },
        arch,
        cluster,
        OptimizationConfig(),
    )
    # Given keys overlaid …
    assert cfg.batch_size == 16
    assert cfg.request_rate == 8.0 and cfg.arrival_model == "poisson"
    assert cfg.attention_backend == "aiter"
    assert cfg.moe_expert_dtype == "mxfp4"
    assert cfg.sparse_attention_topk == 2048
    assert cfg.fused_kernels is True
    # … everything else inherits the profile baseline.
    assert cfg.tp == base.tp and cfg.pp == base.pp and cfg.ep == base.ep
    assert cfg.input_len == base.input_len and cfg.output_len == base.output_len
    assert cfg.weight_dtype == "bf16"


def test_inference_trial_from_dict_partial_proposal_is_legal():
    # A partial proposal that only changes a couple of the *new* knobs should
    # still produce a fully-legal config (the rest filled from the baseline).
    arch, cluster = _moe_arch(), _cluster(num_nodes=1, gpus_per_node=8)
    leg = derive_inference_legality(arch, cluster)
    cfg = inference_trial_from_dict(
        {"tp": 2, "ep": 2, "batch_size": 16, "moe_expert_dtype": "fp8"},
        arch, cluster, OptimizationConfig(),
    )
    ok, reason = validate_inference(cfg, arch, cluster, leg)
    assert ok, reason


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent module (requires dspy): proposal keys + tool belt evaluate path
# ─────────────────────────────────────────────────────────────────────────────


def test_inference_agent_exposes_new_knobs_to_llm():
    pytest.importorskip("dspy")
    from primus.agents.tuning_agent.inference_agent import _PROPOSAL_KEYS

    # The new serving knobs must be in the proposal-key list the LLM is shown,
    # otherwise the agent would never search them.
    for key in (
        "request_rate",
        "arrival_model",
        "attention_backend",
        "sparse_attention_topk",
        "moe_expert_dtype",
        "fused_kernels",
        "quick_reduce",
        "fuse_rmsnorm_allreduce",
        "speculative_draft_cost_factor",
    ):
        assert key in _PROPOSAL_KEYS


def test_inference_agent_evaluate_tool_searches_new_knobs(tmp_path):
    pytest.importorskip("dspy")
    import json as _json

    from primus.agents.tuning_agent.evaluator import Evaluator
    from primus.agents.tuning_agent.history import History
    from primus.agents.tuning_agent.inference_agent import build_inference_tools
    from primus.agents.tuning_agent.scratchpad import Scratchpad

    arch, cluster = _moe_arch(), _cluster(num_nodes=1, gpus_per_node=8)
    leg = derive_inference_legality(arch, cluster)
    agent_cfg = SimpleNamespace(
        out_dir=tmp_path,
        target_cluster=cluster,
        extra_prompt="",
        benchmark_host=SimpleNamespace(has_gpu=False),
        optimization=SimpleNamespace(
            budget=SimpleNamespace(max_perf_calls=10, max_rounds=1, max_rlm_iterations=5),
            hbm_capacity_gb=192.0,
            memory_safety_margin=0.05,
            inference={},
            objective="decode_throughput_tps_per_gpu",
        ),
    )
    history = History.load(tmp_path / "inf.jsonl")
    evaluator = Evaluator(agent_cfg, arch, tmp_path, mode="dry")
    tools = build_inference_tools(
        agent_cfg, arch, leg, history, evaluator, Scratchpad(tmp_path / "sp.txt"),
        session_log=[], objective="decode_throughput_tps_per_gpu", quiet=True,
    )
    by_name = {t.__name__: t for t in tools}

    # The LLM can propose a partial config that flips the new knobs.
    out = _json.loads(
        by_name["evaluate_inference"](
            _json.dumps({"tp": 2, "ep": 2, "batch_size": 16, "moe_expert_dtype": "mxfp4",
                         "attention_backend": "aiter"})
        )
    )
    assert out["legal"] is True
    assert out["score"] is not None
    assert out["config"]["moe_expert_dtype"] == "mxfp4"
    assert out["config"]["attention_backend"] == "aiter"
    assert len(history.trials) == 1

    # Legal axes shown to the LLM include the new knobs, incl. the boolean
    # optimization toggles so LLM-driven search can actually explore them.
    axes = _json.loads(by_name["get_legal_axes"]())
    assert "attention_backend" in axes and "moe_expert_dtype" in axes
    assert "sparse_attention_topk" in axes
    for toggle in ("fused_kernels", "quick_reduce", "fuse_rmsnorm_allreduce"):
        assert toggle in axes, f"{toggle} missing from LLM search axes"
    # Budget tracking is wired.
    budget = _json.loads(by_name["get_budget_status"]())
    assert budget["eval_used"] == 1 and budget["eval_max"] == 10
