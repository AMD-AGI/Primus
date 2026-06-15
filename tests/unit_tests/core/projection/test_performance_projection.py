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
    _compute_micro_batches,
    _extract_layer_type_timings,
    _has_dense_layers,
    _layer_needs_recompute_fwd_in_bwd,
    _limit_layers_for_projection,
    _normalized_recompute_layer_ids,
    _recompute_layer_count,
    _upgrade_artifact_to_v2,
    load_hardware_config,
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
