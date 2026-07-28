###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for the *performance* projection helpers
(``core/projection/performance_projection/projection.py``).

The performance projection module is large and most of it orchestrates GPU
layer benchmarks, but it carries a layer of pure-logic helpers that decide
parallelism placement, sub-node benchmark reduction, recompute accounting,
artifact schema upgrades, and per-chunk timing assembly.  Those are what we
exercise here (no GPU / torch.distributed required):

  * ``_calculate_min_gpus`` (MoE parallel-folding vs dense placement);
  * ``_has_dense_layers`` MoE-pattern sniffing;
  * ``_upgrade_artifact_to_v2`` schema migration;
  * ``load_hardware_config`` YAML loading;
  * recompute accounting (``_normalized_recompute_layer_ids``,
    ``_recompute_layer_count``, ``_layer_needs_recompute_fwd_in_bwd``);
  * ``_compute_micro_batches``;
  * per-layer-type timing extraction + IO-layer folding;
  * ``_limit_layers_for_projection`` / ``_calculate_single_node_config``
    benchmark config reduction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("primus.core.projection.performance_projection.projection")

from primus.core.projection.performance_projection.projection import (  # noqa: E402
    _BYTES_PER_GB,
    _add_io_layer_timings,
    _calculate_min_gpus,
    _calculate_single_node_config,
    _compute_ep_mlp_scale,
    _compute_micro_batches,
    _estimate_a2a_per_layer_ms,
    _estimate_ep_communication_overhead,
    _estimate_pp_communication_overhead,
    _extract_layer_type_timings,
    _get_deepep_overlap_efficiency,
    _get_parameter_memory,
    _has_dense_layers,
    _layer_needs_recompute_fwd_in_bwd,
    _limit_layers_for_projection,
    _load_artifact,
    _load_profiling_results,
    _normalized_recompute_layer_ids,
    _recompute_layer_count,
    _reduction_info_from_artifact_metadata,
    _rescale_expert_parallelism,
    _save_profiling_results,
    _summarize_bench_training_config,
    _upgrade_artifact_to_v2,
    calculate_collective_communication_time,
    extract_single_node_time_from_profiling,
    load_hardware_config,
)
from primus.core.projection.training_config import (  # noqa: E402
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# _calculate_min_gpus
# ─────────────────────────────────────────────────────────────────────────────


def test_min_gpus_dense_uses_cp():
    # Dense (ep<=1): TP × PP × CP.
    assert _calculate_min_gpus(tp=2, pp=2, ep=1, cp=2) == 8


def test_min_gpus_moe_folds_cp_into_ep():
    # MoE (ep>1): CP folded into EP → TP × PP × EP (CP excluded).
    assert _calculate_min_gpus(tp=2, pp=1, ep=8, cp=4) == 16


# ─────────────────────────────────────────────────────────────────────────────
# _has_dense_layers
# ─────────────────────────────────────────────────────────────────────────────


def test_has_dense_layers_none_is_true():
    assert _has_dense_layers(None) is True


def test_has_dense_layers_int_pattern():
    # 1 => every layer MoE → no dense.
    assert _has_dense_layers(1) is False
    assert _has_dense_layers(2) is True  # every 2nd layer MoE → dense present


def test_has_dense_layers_list_pattern():
    assert _has_dense_layers([1, 1, 1]) is False
    assert _has_dense_layers([0, 1, 1]) is True


def test_has_dense_layers_str_pattern():
    assert _has_dense_layers("[1, 1, 1]") is False
    assert _has_dense_layers("[0, 1, 1]") is True


# ─────────────────────────────────────────────────────────────────────────────
# _upgrade_artifact_to_v2
# ─────────────────────────────────────────────────────────────────────────────


def test_upgrade_artifact_idempotent_for_v2():
    payload = {"schema_version": 2, "metadata": {}, "profiling_results": {}, "memory_results": None}
    out = _upgrade_artifact_to_v2(dict(payload))
    assert out["schema_version"] == 2


def test_upgrade_artifact_hoists_memory_and_sets_version():
    v1 = {
        "metadata": {"x": 1},
        "profiling_results": {
            "0": {"type": "dense"},
            "_memory_benchmark": {"global_peak_allocated_bytes": 5},
        },
    }
    out = _upgrade_artifact_to_v2(v1)
    assert out["schema_version"] == 2
    # _memory_benchmark hoisted to top-level memory_results and removed from pr.
    assert out["memory_results"] == {"global_peak_allocated_bytes": 5}
    assert "_memory_benchmark" not in out["profiling_results"]


def test_upgrade_artifact_defaults_when_minimal():
    out = _upgrade_artifact_to_v2({})
    assert out["schema_version"] == 2
    assert out["metadata"] == {}
    assert out["profiling_results"] == {}
    assert out["memory_results"] is None


# ─────────────────────────────────────────────────────────────────────────────
# load_hardware_config
# ─────────────────────────────────────────────────────────────────────────────


def test_load_hardware_config(tmp_path):
    import yaml

    p = tmp_path / "hw.yaml"
    p.write_text(yaml.safe_dump({"hardware_config": {"gpu": "mi355x", "tflops": 1234}}))
    hw = load_hardware_config(str(p))
    assert hw == {"gpu": "mi355x", "tflops": 1234}


def test_load_hardware_config_missing_key_returns_empty(tmp_path):
    import yaml

    p = tmp_path / "hw.yaml"
    p.write_text(yaml.safe_dump({"something_else": {}}))
    assert load_hardware_config(str(p)) == {}


# ─────────────────────────────────────────────────────────────────────────────
# Recompute accounting
# ─────────────────────────────────────────────────────────────────────────────


def test_normalized_recompute_layer_ids_variants():
    assert _normalized_recompute_layer_ids(SimpleNamespace(recompute_layer_ids="0,3,7")) == frozenset(
        {0, 3, 7}
    )
    assert _normalized_recompute_layer_ids(SimpleNamespace(recompute_layer_ids=[1, 2])) == frozenset({1, 2})
    assert _normalized_recompute_layer_ids(SimpleNamespace(recompute_layer_ids=None)) is None
    # Empty string → None (nothing to recompute).
    assert _normalized_recompute_layer_ids(SimpleNamespace(recompute_layer_ids="")) is None


def test_recompute_layer_count_prefers_explicit_ids():
    mp = SimpleNamespace(recompute_layer_ids="0,3,7", recompute_num_layers=99)
    assert _recompute_layer_count(mp, total_layers=32) == 3


def test_recompute_layer_count_caps_num_layers_at_total():
    mp = SimpleNamespace(recompute_layer_ids=None, recompute_num_layers=40)
    assert _recompute_layer_count(mp, total_layers=32) == 32
    mp2 = SimpleNamespace(recompute_layer_ids=None, recompute_num_layers=4)
    assert _recompute_layer_count(mp2, total_layers=32) == 4


def test_layer_needs_recompute_with_explicit_ids():
    ids = frozenset({2, 5})
    assert (
        _layer_needs_recompute_fwd_in_bwd(
            global_layer_idx=2,
            local_layer_idx=0,
            recompute_num_layers=0,
            recompute_layer_ids=ids,
            total_layers=8,
            pp_size=1,
            vpp_size=1,
        )
        is True
    )
    assert (
        _layer_needs_recompute_fwd_in_bwd(
            global_layer_idx=3,
            local_layer_idx=3,
            recompute_num_layers=0,
            recompute_layer_ids=ids,
            total_layers=8,
            pp_size=1,
            vpp_size=1,
        )
        is False
    )


def test_layer_needs_recompute_local_vs_global_threshold():
    # Small num_layers (≤ typical chunk size) → per-chunk local index gate.
    assert _layer_needs_recompute_fwd_in_bwd(7, 1, 2, None, total_layers=8, pp_size=1, vpp_size=1) is True
    assert _layer_needs_recompute_fwd_in_bwd(7, 3, 2, None, total_layers=8, pp_size=1, vpp_size=1) is False

    # Large model-wide total (> typical chunk size) → global index gate.
    # total=61, pp=8 → typical_chunk≈8; recompute_num_layers=48 > 8 → global gate.
    assert _layer_needs_recompute_fwd_in_bwd(10, 99, 48, None, total_layers=61, pp_size=8, vpp_size=1) is True
    assert _layer_needs_recompute_fwd_in_bwd(50, 0, 48, None, total_layers=61, pp_size=8, vpp_size=1) is False


def test_layer_needs_recompute_disabled_cases():
    assert _layer_needs_recompute_fwd_in_bwd(0, 0, 0, None, total_layers=8, pp_size=1, vpp_size=1) is False
    assert _layer_needs_recompute_fwd_in_bwd(0, 0, 4, None, total_layers=0, pp_size=1, vpp_size=1) is False


# ─────────────────────────────────────────────────────────────────────────────
# _compute_micro_batches
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_micro_batches_basic():
    rt = SimpleNamespace(global_batch_size=128, micro_batch_size=2, data_parallel_size=8)
    # gbs / (mbs * dp) = 128 / 16 = 8.
    assert _compute_micro_batches(rt, None) == 8


def test_compute_micro_batches_rounds_up_and_floors_at_one():
    rt = SimpleNamespace(global_batch_size=10, micro_batch_size=4, data_parallel_size=1)
    assert _compute_micro_batches(rt, None) == 3  # ceil(10/4)
    rt2 = SimpleNamespace(global_batch_size=1, micro_batch_size=8, data_parallel_size=8)
    assert _compute_micro_batches(rt2, None) == 1


# ─────────────────────────────────────────────────────────────────────────────
# _extract_layer_type_timings + _add_io_layer_timings
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_layer_type_timings_one_per_type():
    layer_results = {
        0: {
            "type": "dense",
            "forward_time_ms": 1.0,
            "backward_time_ms": 2.0,
            "activation_memory_bytes": _BYTES_PER_GB,
        },
        1: {
            "type": "moe",
            "forward_time_ms": 3.0,
            "backward_time_ms": 4.0,
            "activation_memory_bytes": 2 * _BYTES_PER_GB,
        },
        2: {"type": "moe", "forward_time_ms": 9.0},  # duplicate type ignored
        3: {"type": "other", "forward_time_ms": 1.0},  # non dense/moe ignored
    }
    t = _extract_layer_type_timings(layer_results)
    assert set(t) == {"dense", "moe"}
    assert t["dense"]["forward"] == 1.0
    assert t["dense"]["backward"] == 2.0
    assert t["dense"]["wgrad"] == 0.0  # folded into backward
    assert t["dense"]["activation"] == pytest.approx(1.0)  # bytes → GB
    assert t["moe"]["activation"] == pytest.approx(2.0)


def test_extract_layer_type_timings_empty():
    assert _extract_layer_type_timings({}) == {}


def test_add_io_layer_timings_folds_embedding_and_output():
    chunk_timings = [
        [{"fwd": 1.0, "bwd": 1.0, "wgrad": 0.0, "activation": 0.0}],
        [{"fwd": 2.0, "bwd": 2.0, "wgrad": 0.0, "activation": 0.0}],
    ]
    profiling_results = {
        "embedding": {
            "forward_time_ms": 0.5,
            "backward_time_ms": 0.25,
            "activation_memory_bytes": _BYTES_PER_GB,
        },
        "output": {
            "forward_time_ms": 0.75,
            "backward_time_ms": 0.5,
            "activation_memory_bytes": 2 * _BYTES_PER_GB,
        },
    }
    _add_io_layer_timings(chunk_timings, profiling_results)
    # Embedding folds into the first chunk's first entry.
    first = chunk_timings[0][0]
    assert first["fwd"] == pytest.approx(1.5)
    assert first["bwd"] == pytest.approx(1.25)
    assert first["activation"] == pytest.approx(1.0)
    # Output folds into the last chunk's last entry.
    last = chunk_timings[-1][-1]
    assert last["fwd"] == pytest.approx(2.75)
    assert last["bwd"] == pytest.approx(2.5)
    assert last["activation"] == pytest.approx(2.0)


def test_add_io_layer_timings_noop_on_empty():
    # Must not raise on empty chunk list.
    _add_io_layer_timings([], {"embedding": {"forward_time_ms": 1.0}})


# ─────────────────────────────────────────────────────────────────────────────
# _limit_layers_for_projection
# ─────────────────────────────────────────────────────────────────────────────


def test_limit_layers_dense_single_layer():
    cfg = SimpleNamespace(
        num_experts=0,
        num_layers=32,
        moe_layer_freq=None,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
        pipeline_model_parallel_layout="t*1|t*1",
    )
    _limit_layers_for_projection(cfg)
    assert cfg.num_layers == 1
    assert cfg.moe_layer_freq == [0]
    # PP collapsed for single-node layer benchmarking.
    assert cfg.pipeline_model_parallel_size == 1
    assert cfg.virtual_pipeline_model_parallel_size == 1
    assert cfg.pipeline_model_parallel_layout is None


def test_limit_layers_moe_with_dense_keeps_two():
    cfg = SimpleNamespace(
        num_experts=8,
        num_layers=32,
        moe_layer_freq=[0, 1, 1, 1],  # dense present
        pipeline_model_parallel_size=2,
    )
    _limit_layers_for_projection(cfg)
    assert cfg.num_layers == 2
    # First dense, then moe so the extractor can classify both types.
    assert cfg.moe_layer_freq == [0, 1]


def test_limit_layers_moe_all_moe_single_layer():
    cfg = SimpleNamespace(num_experts=8, num_layers=24, moe_layer_freq=1, pipeline_model_parallel_size=1)
    _limit_layers_for_projection(cfg)
    assert cfg.num_layers == 1
    assert cfg.moe_layer_freq == [1]


# ─────────────────────────────────────────────────────────────────────────────
# _calculate_single_node_config
# ─────────────────────────────────────────────────────────────────────────────


def test_single_node_config_no_adjustment_when_fits():
    cfg = SimpleNamespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        num_experts=None,
    )
    info = _calculate_single_node_config(cfg, gpus_per_node=8, benchmark_gpus=8)
    assert info["adjusted"] is False
    assert info["benchmark_pp"] == info["original_pp"] == 4
    assert info["benchmark_tp"] == 2
    # Config object left unchanged.
    assert cfg.pipeline_model_parallel_size == 4


def test_single_node_config_reduces_pp_to_fit():
    cfg = SimpleNamespace(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=8,  # 2×8 = 16 > 8 → must reduce
        expert_model_parallel_size=1,
        context_parallel_size=1,
        num_experts=None,
    )
    info = _calculate_single_node_config(cfg, gpus_per_node=8, benchmark_gpus=8)
    assert info["adjusted"] is True
    assert info["original_pp"] == 8
    assert info["benchmark_pp"] == 1
    assert info["benchmark_tp"] == 2  # TP didn't need reducing once PP=1
    assert cfg.pipeline_model_parallel_size == 1


def test_single_node_config_reduces_ep_for_moe():
    cfg = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=16,  # 16 > 8 → reduce EP
        context_parallel_size=1,
        num_experts=16,
        moe_router_topk=2,
    )
    info = _calculate_single_node_config(cfg, gpus_per_node=8, benchmark_gpus=8)
    assert info["adjusted"] is True
    assert info["original_ep"] == 16
    assert info["benchmark_ep"] == 8
    # num_experts reduced proportionally (experts_per_rank preserved = 1).
    assert info["benchmark_num_experts"] == 8


# ─────────────────────────────────────────────────────────────────────────────
# _rescale_expert_parallelism (cap EP*TP at _MAX_EXPERT_PARALLEL_SIZE = 8)
# ─────────────────────────────────────────────────────────────────────────────


def test_rescale_ep_noop_within_limit():
    cfg = SimpleNamespace(
        expert_model_parallel_size=4, tensor_model_parallel_size=1, context_parallel_size=1, num_experts=8
    )
    assert _rescale_expert_parallelism(cfg) is None
    assert cfg.expert_model_parallel_size == 4


def test_rescale_ep_caps_and_scales_experts():
    cfg = SimpleNamespace(
        expert_model_parallel_size=16, tensor_model_parallel_size=1, context_parallel_size=1, num_experts=32
    )
    info = _rescale_expert_parallelism(cfg)
    assert (info["ep_before"], info["ep_after"]) == (16, 8)
    assert cfg.expert_model_parallel_size == 8
    assert cfg.num_experts == 16  # experts_per_rank (32/16=2) preserved -> 8*2


def test_rescale_ep_folds_tp_into_budget():
    # EP=8 is within the limit alone, but EP*TP=16 > 8, so EP is rescaled to 4.
    cfg = SimpleNamespace(
        expert_model_parallel_size=8, tensor_model_parallel_size=2, context_parallel_size=1, num_experts=8
    )
    info = _rescale_expert_parallelism(cfg)
    assert info["ep_after"] == 4
    assert cfg.expert_model_parallel_size == 4


def test_rescale_ep_handles_missing_num_experts():
    cfg = SimpleNamespace(
        expert_model_parallel_size=16, tensor_model_parallel_size=1, context_parallel_size=1, num_experts=None
    )
    info = _rescale_expert_parallelism(cfg)
    assert info["ep_after"] == 8
    assert info["num_experts_after"] is None


# ─────────────────────────────────────────────────────────────────────────────
# _get_deepep_overlap_efficiency / _compute_ep_mlp_scale
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stage,eff", [(0, 0.65), (1, 0.75), (2, 0.80), (3, 0.85), (5, 0.85)])
def test_deepep_overlap_efficiency(stage, eff):
    assert _get_deepep_overlap_efficiency(SimpleNamespace(turbo_sync_free_moe_stage=stage)) == eff


def test_deepep_overlap_efficiency_default_when_missing():
    assert _get_deepep_overlap_efficiency(SimpleNamespace()) == 0.65


def test_ep_mlp_scale_unity_when_ep_unchanged():
    assert _compute_ep_mlp_scale(SimpleNamespace(), 8, 8) == 1.0


def test_ep_mlp_scale_unity_when_experts_per_rank_preserved():
    assert (
        _compute_ep_mlp_scale(
            SimpleNamespace(),
            benchmark_ep=8,
            original_ep=16,
            original_num_experts=32,
            benchmark_num_experts=16,
        )
        == 1.0
    )


def test_ep_mlp_scale_unity_fallback_without_expert_counts():
    assert _compute_ep_mlp_scale(SimpleNamespace(), benchmark_ep=8, original_ep=16) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Communication-time estimators (use the analytical collective model)
# ─────────────────────────────────────────────────────────────────────────────


def _comm_tc(**mp):
    base_mp = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_model_parallel_size=1,
    )
    base_mp.update(mp)
    return SimpleNamespace(
        model_parallel_config=SimpleNamespace(**base_mp),
        model_config=SimpleNamespace(hidden_size=512, moe_router_topk=2),
        runtime_config=SimpleNamespace(micro_batch_size=1, sequence_length=128, global_batch_size=8),
    )


@pytest.fixture
def _single_node_env(monkeypatch):
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("NNODES", "1")


def test_pp_comm_overhead_zero_for_single_stage(_single_node_env):
    assert _estimate_pp_communication_overhead(_comm_tc(), pp_size=1) == 0.0


def test_pp_comm_overhead_positive_for_multistage(_single_node_env):
    assert _estimate_pp_communication_overhead(_comm_tc(), pp_size=2) > 0


def test_a2a_per_layer_zero_without_ep(_single_node_env):
    assert _estimate_a2a_per_layer_ms(_comm_tc(), ep=1) == 0.0


def test_a2a_per_layer_positive_with_ep(_single_node_env):
    assert _estimate_a2a_per_layer_ms(_comm_tc(expert_model_parallel_size=2), ep=2) > 0


def test_ep_comm_overhead_zero_when_not_scaling_up(_single_node_env):
    assert _estimate_ep_communication_overhead(_comm_tc(), original_ep=4, benchmark_ep=8) == (0.0, 0.0)


def test_ep_comm_overhead_symmetric_when_scaling_up(_single_node_env):
    fwd, bwd = _estimate_ep_communication_overhead(_comm_tc(), original_ep=4, benchmark_ep=2)
    assert fwd == bwd


# ─────────────────────────────────────────────────────────────────────────────
# _get_parameter_memory (param bytes/GB on a PP rank via the language-model spec)
# ─────────────────────────────────────────────────────────────────────────────


def _full_config(**mp):
    return TrainingConfig(
        model_config=ModelConfig(
            num_layers=4,
            hidden_size=512,
            ffn_hidden_size=1024,
            padded_vocab_size=32000,
            num_attention_heads=8,
            kv_channels=64,
            num_query_groups=8,
            swiglu=True,
            num_experts=None,
            moe_pattern=[0] * 4,
        ),
        runtime_config=RuntimeConfig(
            global_batch_size=8, micro_batch_size=1, sequence_length=128, data_parallel_size=1
        ),
        model_parallel_config=ModelParallelConfig(**mp),
    )


def test_get_parameter_memory_positive_gb(_single_node_env, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    assert _get_parameter_memory(_full_config(), pp_rank=0) > 0


def test_get_parameter_memory_rank_out_of_range_raises(_single_node_env, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="out of range"):
        _get_parameter_memory(_full_config(pipeline_model_parallel_size=2), pp_rank=5)


# ─────────────────────────────────────────────────────────────────────────────
# extract_single_node_time_from_profiling (extrapolate profiled layers -> model)
# ─────────────────────────────────────────────────────────────────────────────


def _profiling_tc(moe_pattern, **mp):
    base_mp = dict(recompute_granularity=None, recompute_num_layers=0, recompute_layer_ids=None)
    base_mp.update(mp)
    return SimpleNamespace(
        model_config=SimpleNamespace(moe_pattern=moe_pattern),
        model_parallel_config=SimpleNamespace(**base_mp),
    )


def test_extract_single_node_time_dense_with_io():
    tc = _profiling_tc([0, 0, 0, 0])
    pr = {
        0: {"forward_time_ms": 2, "backward_time_ms": 3, "type": "dense"},
        "embedding": {"forward_time_ms": 0.5, "backward_time_ms": 0.5},
        "output": {"forward_time_ms": 1, "backward_time_ms": 1},
    }
    # avg_dense=5 * 4 layers = 20; + embedding 1 + output 2 = 23
    assert extract_single_node_time_from_profiling(pr, tc) == pytest.approx(23.0)


def test_extract_single_node_time_dense_and_moe_buckets():
    tc = _profiling_tc([0, 1, 1, 1])  # 1 dense + 3 MoE
    pr = {
        0: {"forward_time_ms": 1, "backward_time_ms": 1, "type": "dense"},
        1: {"forward_time_ms": 2, "backward_time_ms": 2, "type": "moe"},
    }
    # dense 2 * 1 + moe 4 * 3 = 14
    assert extract_single_node_time_from_profiling(pr, tc) == pytest.approx(14.0)


def test_extract_single_node_time_adds_full_recompute_overhead():
    tc = _profiling_tc([0, 0, 0, 0], recompute_granularity="full", recompute_num_layers=4)
    pr = {0: {"forward_time_ms": 2, "backward_time_ms": 3, "type": "dense"}}
    # base 5 * 4 = 20; recompute all 4 dense layers adds avg_dense_fwd(2) * 4 = 8 -> 28
    assert extract_single_node_time_from_profiling(pr, tc) == pytest.approx(28.0)


# ─────────────────────────────────────────────────────────────────────────────
# artifact metadata helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_reduction_info_from_metadata_marks_adjusted():
    md = {
        "benchmark_pp": 1,
        "benchmark_tp": 2,
        "benchmark_ep": 8,
        "benchmark_gpus": 16,
        "original_pp": 2,
        "original_tp": 2,
        "original_ep": 16,
        "original_cp": 1,
        "original_num_experts": 64,
        "benchmark_num_experts": 32,
    }
    info = _reduction_info_from_artifact_metadata(md, gpus_per_node=8)
    assert info["adjusted"] is True
    assert (info["original_ep"], info["benchmark_ep"]) == (16, 8)
    # min GPUs = TP*PP*EP = 2*2*16 = 64 -> 8 nodes
    assert info["original_nodes_required"] == 8


def test_reduction_info_from_metadata_not_adjusted_when_identical():
    md = {
        "benchmark_pp": 1,
        "benchmark_tp": 1,
        "benchmark_ep": 1,
        "original_pp": 1,
        "original_tp": 1,
        "original_ep": 1,
    }
    assert _reduction_info_from_artifact_metadata(md, gpus_per_node=8)["adjusted"] is False


def test_summarize_bench_training_config_normalizes_recompute_ids():
    tc = SimpleNamespace(
        model_config=SimpleNamespace(num_layers=4, moe_pattern=[0, 1, 0, 1], num_experts=8),
        model_parallel_config=SimpleNamespace(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            recompute_granularity="full",
            recompute_num_layers=2,
            recompute_layer_ids="0,1",
        ),
        runtime_config=SimpleNamespace(micro_batch_size=1, sequence_length=128, global_batch_size=8),
    )
    s = _summarize_bench_training_config(tc)
    assert s["num_layers"] == 4 and s["moe_pattern"] == [0, 1, 0, 1]
    assert s["tensor_model_parallel_size"] == 2
    assert s["recompute_layer_ids"] == [0, 1]  # parsed + sorted


# ─────────────────────────────────────────────────────────────────────────────
# artifact save/load roundtrip
# ─────────────────────────────────────────────────────────────────────────────


def test_save_load_profiling_roundtrip(tmp_path):
    pr = {
        0: {"forward_time_ms": 1.0, "type": "dense"},
        1: {"forward_time_ms": 2.0, "type": "moe"},
        "_memory_benchmark": {"peak": 123},
    }
    path = str(tmp_path / "artifact.json")
    _save_profiling_results(pr, {"benchmark_ep": 8, "benchmark_tp": 2, "original_ep": 16}, path)
    loaded, metadata = _load_profiling_results(path)
    assert 0 in loaded and 1 in loaded  # int keys restored from JSON strings
    assert loaded["_memory_benchmark"] == {"peak": 123}  # hoisted then re-injected
    assert metadata["benchmark_ep"] == 8 and metadata["original_ep"] == 16


def test_load_artifact_upgrades_v1_payload(tmp_path):
    import json

    path = str(tmp_path / "v1.json")
    with open(path, "w") as f:
        json.dump({"profiling_results": {"0": {"forward_time_ms": 1}}}, f)
    payload = _load_artifact(path)
    assert payload["schema_version"] == 2
    assert payload["memory_results"] is None


# ─────────────────────────────────────────────────────────────────────────────
# calculate_collective_communication_time (analytical comm breakdown)
# ─────────────────────────────────────────────────────────────────────────────


def _full_moe_config(**mp):
    return TrainingConfig(
        model_config=ModelConfig(
            num_layers=4,
            hidden_size=512,
            ffn_hidden_size=1024,
            moe_ffn_hidden_size=1024,
            padded_vocab_size=32000,
            num_attention_heads=8,
            kv_channels=64,
            num_query_groups=8,
            swiglu=True,
            num_experts=8,
            moe_router_topk=2,
            moe_pattern=[1] * 4,
        ),
        runtime_config=RuntimeConfig(
            global_batch_size=8, micro_batch_size=1, sequence_length=128, data_parallel_size=1
        ),
        model_parallel_config=ModelParallelConfig(**mp),
    )


def test_collective_comm_zero_for_single_gpu(_single_node_env):
    tc = _full_config()
    total, breakdown, msg, per_layer = calculate_collective_communication_time(
        tc, num_nodes=1, gpus_per_node=8, tp=1, pp=1, ep=1, cp=1, dp=1
    )
    assert total == 0.0
    assert breakdown["gradient_allreduce"] == 0.0 and breakdown["moe_a2a_fwd"] == 0.0
    assert len(per_layer) == tc.model_config.num_layers


def test_collective_comm_gradient_allreduce_with_dp(_single_node_env):
    tc = _full_config()
    _, breakdown, msg, _ = calculate_collective_communication_time(
        tc, num_nodes=1, gpus_per_node=8, tp=1, pp=1, ep=1, cp=1, dp=4
    )
    assert breakdown["gradient_allreduce"] > 0
    assert msg["gradient_allreduce_size"] > 0


def test_collective_comm_moe_all_to_all(_single_node_env):
    tc = _full_moe_config()
    _, breakdown, msg, _ = calculate_collective_communication_time(
        tc, num_nodes=2, gpus_per_node=8, tp=1, pp=1, ep=2, cp=1, dp=8
    )
    assert breakdown["moe_a2a_fwd"] > 0
    assert msg["num_moe_layers"] == 4


def test_collective_comm_fsdp_breakdown(_single_node_env):
    tc = _full_config(use_torch_fsdp2=True)
    _, breakdown, msg, _ = calculate_collective_communication_time(
        tc, num_nodes=1, gpus_per_node=8, tp=1, pp=1, ep=1, cp=1, dp=4
    )
    assert breakdown["fsdp_allgather_fwd"] > 0
    assert msg["fsdp_enabled"] is True
