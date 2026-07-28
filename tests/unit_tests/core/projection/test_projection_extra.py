###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU unit tests for projection pieces not covered by test_projection_profilers.py:
OptimizerProfiler, LossProfiler, transformer-layer aggregation/comm helpers,
the collective-comm hardware-config builder, and the memory-report printers.

All analytical / formula paths; no GPU benchmark or origami simulation backend.
"""

from types import SimpleNamespace

import pytest

from primus.core.projection.memory_projection.reports import (
    _gb,
    _pct,
    compare_simulate_vs_benchmark,
    print_per_rank_breakdown,
    report_dict,
)
from primus.core.projection.module_profilers.attention import AttentionProfiler
from primus.core.projection.module_profilers.collective_args import (
    CollectiveArgs,
    get_default_args,
)
from primus.core.projection.module_profilers.dense_mlp import DenseMLPProfiler
from primus.core.projection.module_profilers.layer_norm import LayerNormProfiler
from primus.core.projection.module_profilers.loss import (
    _BW_EFFICIENCY,
    _FALLBACK_HBM_BW_GBPS,
    LossProfiler,
)
from primus.core.projection.module_profilers.moe_mlp import MoEMLPProfiler
from primus.core.projection.module_profilers.optimizer import (
    _ADAM_BYTES_PER_PARAM,
    OptimizerProfiler,
)
from primus.core.projection.module_profilers.residual_add import ResidualAddProfiler
from primus.core.projection.module_profilers.router import RouterProfiler
from primus.core.projection.module_profilers.transformer_layer import (
    DenseTransformerLayerProfiler,
    MoETransformerLayerProfiler,
    _estimate_layernorm_residual_time_ms,
    _estimate_moe_a2a_time_ms,
    _estimate_tp_allreduce_time_ms,
    get_dense_transformer_layer_profiler_spec,
    get_moe_transformer_layer_profiler_spec,
)
from primus.core.projection.training_config import (
    ModelConfig,
    ModelParallelConfig,
    RuntimeConfig,
    TrainingConfig,
)

HIDDEN = 512
FFN = 1024
N_LAYERS = 4
VOCAB = 32000
GB = 1024**3


def _config(*, moe=False, fusion=False, **mp):
    base = dict(
        num_layers=N_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        padded_vocab_size=VOCAB,
        num_attention_heads=8,
        kv_channels=64,
        num_query_groups=8,
        swiglu=True,
        cross_entropy_loss_fusion=fusion,
    )
    if moe:
        base.update(num_experts=8, moe_ffn_hidden_size=FFN, moe_router_topk=2, moe_pattern=[1] * N_LAYERS)
    else:
        base.update(num_experts=None, moe_pattern=[0] * N_LAYERS)
    return TrainingConfig(
        model_config=ModelConfig(**base),
        runtime_config=RuntimeConfig(global_batch_size=8, micro_batch_size=1, sequence_length=128),
        model_parallel_config=ModelParallelConfig(**mp),
    )


class _FakeBackend:
    def __init__(self, bw=5300.0):
        self._bw = bw

    @property
    def hbm_bandwidth_gbps(self):
        return self._bw

    def name(self):
        return "fake"


@pytest.fixture
def _single_node_env(monkeypatch):
    """Single-node env so the collective-comm helpers see a known topology."""
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("NNODES", "1")


# =============================== OptimizerProfiler ===============================
# Dense, TP=PP=1: per layer = 4*h*h (attn) + 3*h*ffn (mlp); plus embedding+output
# = 2 * padded_vocab_size * hidden. The vocab term is a regression guard for a bug
# where OptimizerProfiler read model_config.vocab_size (a non-existent field, so it
# was always 0) instead of padded_vocab_size.
_EXPECTED_DENSE_PARAMS = N_LAYERS * (4 * HIDDEN * HIDDEN + 3 * HIDDEN * FFN) + 2 * VOCAB * HIDDEN


def _opt(bw=5300.0, **mp):
    return OptimizerProfiler(_config(**mp), gemm_backend=_FakeBackend(bw))


def test_optimizer_activation_memory_is_zero():
    assert _opt().estimated_activation_memory(batch_size=2, seq_len=128) == 0


def test_optimizer_num_params_includes_embedding_and_output():
    assert _opt().estimated_num_params() == _EXPECTED_DENSE_PARAMS


def test_optimizer_step_time_follows_bandwidth_formula():
    bw = 5300.0
    total_bytes = _EXPECTED_DENSE_PARAMS * _ADAM_BYTES_PER_PARAM
    assert _opt(bw=bw).estimated_step_time_ms() == pytest.approx(total_bytes / (bw * 1e9 / 1e3))


def test_optimizer_distributed_optimizer_shards_by_dp():
    full = _opt().estimated_step_time_ms(dp_size=4)
    sharded = _opt(use_distributed_optimizer=True).estimated_step_time_ms(dp_size=4)
    assert sharded == pytest.approx(full / 4)


def test_optimizer_fsdp_shards_by_dp():
    full = _opt().estimated_step_time_ms(dp_size=2)
    sharded = _opt(use_torch_fsdp2=True).estimated_step_time_ms(dp_size=2)
    assert sharded == pytest.approx(full / 2)


def test_optimizer_pipeline_parallel_halves_params():
    assert _opt(pipeline_model_parallel_size=2).estimated_num_params() == _EXPECTED_DENSE_PARAMS // 2


def test_optimizer_tensor_parallel_reduces_params():
    assert _opt(tensor_model_parallel_size=2).estimated_num_params() < _EXPECTED_DENSE_PARAMS


def test_optimizer_missing_backend_raises():
    p = OptimizerProfiler(_config(), gemm_backend=None)
    with pytest.raises(AssertionError, match="requires a GEMM simulation backend"):
        p.estimated_step_time_ms()


def test_optimizer_backend_without_bandwidth_raises():
    p = OptimizerProfiler(_config(), gemm_backend=_FakeBackend(None))
    with pytest.raises(AssertionError, match="does not report hbm_bandwidth_gbps"):
        p.estimated_step_time_ms()


# ================================= LossProfiler =================================
def test_loss_has_no_params():
    assert LossProfiler(_config()).estimated_num_params() == 0


def test_loss_activation_memory_formula():
    # tokens * vocab_per_rank * 4 bytes (fp32 softmax), tp=cp=1
    assert LossProfiler(_config()).estimated_activation_memory(2, 128) == 2 * 128 * VOCAB * 4


def test_loss_activation_memory_zero_with_fusion():
    assert LossProfiler(_config(fusion=True)).estimated_activation_memory(2, 128) == 0


def test_loss_tensor_parallel_shards_vocab():
    full = LossProfiler(_config()).estimated_activation_memory(1, 128)
    sharded = LossProfiler(_config(tensor_model_parallel_size=4)).estimated_activation_memory(1, 128)
    assert sharded == full // 4


def test_loss_forward_equals_backward_and_positive():
    p = LossProfiler(_config())
    fwd, bwd = p.estimated_forward_time(2, 128), p.estimated_backward_time(2, 128)
    assert fwd > 0 and fwd == pytest.approx(bwd)


def test_loss_time_zero_with_fusion():
    p = LossProfiler(_config(fusion=True))
    assert p.estimated_forward_time(2, 128) == 0.0
    assert p.estimated_backward_time(2, 128) == 0.0


def test_loss_time_uses_fallback_bandwidth():
    p = LossProfiler(_config())
    tensor_bytes = 2 * 128 * VOCAB * 2  # bf16 logits
    eff_bw = _FALLBACK_HBM_BW_GBPS * _BW_EFFICIENCY
    assert p.estimated_forward_time(2, 128) == pytest.approx(3.0 * tensor_bytes / (eff_bw * 1e6))


def test_loss_backend_bandwidth_speeds_up():
    p = LossProfiler(_config())
    fallback = p.estimated_forward_time(2, 128)
    p.set_gemm_backend(_FakeBackend(8000.0))
    assert p.estimated_forward_time(2, 128) < fallback


# ============================== transformer layer ===============================
def test_layernorm_residual_time_backward_heavier_than_forward():
    fwd, bwd = _estimate_layernorm_residual_time_ms(_config(), 2, 128)
    assert fwd > 0 and bwd > fwd  # 14 vs 12 memory passes


def test_layernorm_residual_time_faster_with_higher_bandwidth():
    base_fwd, _ = _estimate_layernorm_residual_time_ms(_config(), 2, 128)
    fast_fwd, _ = _estimate_layernorm_residual_time_ms(_config(), 2, 128, gemm_backend=_FakeBackend(8000.0))
    assert fast_fwd < base_fwd  # 8000 > fallback 5300


def test_tp_allreduce_zero_without_tp(_single_node_env):
    assert _estimate_tp_allreduce_time_ms(_config(tensor_model_parallel_size=1), 2, 128) == 0.0


def test_tp_allreduce_positive_with_tp(_single_node_env):
    assert _estimate_tp_allreduce_time_ms(_config(tensor_model_parallel_size=2), 2, 128) > 0


def test_moe_a2a_zero_without_ep(_single_node_env):
    assert _estimate_moe_a2a_time_ms(_config(moe=True, expert_model_parallel_size=1), 2, 128) == 0.0


def test_moe_a2a_positive_with_ep(_single_node_env):
    assert _estimate_moe_a2a_time_ms(_config(moe=True, expert_model_parallel_size=2), 2, 128) > 0


def test_dense_layer_params_and_activation_aggregate_subprofilers():
    cfg = _config()
    subs = {
        "layer_norm": LayerNormProfiler(cfg),
        "self_attention": AttentionProfiler(cfg),
        "residual_add": ResidualAddProfiler(cfg),
        "mlp": DenseMLPProfiler(cfg),
    }
    layer = DenseTransformerLayerProfiler(cfg, subs)
    assert layer.estimated_num_params() == (
        subs["layer_norm"].estimated_num_params() * 3
        + subs["self_attention"].estimated_num_params()
        + subs["mlp"].estimated_num_params()
        + subs["residual_add"].estimated_num_params() * 2
    )
    assert layer.estimated_activation_memory(1, 128) == (
        subs["layer_norm"].estimated_activation_memory(1, 128) * 3
        + subs["self_attention"].estimated_activation_memory(1, 128)
        + subs["mlp"].estimated_activation_memory(1, 128)
        + subs["residual_add"].estimated_activation_memory(1, 128) * 2
    )


def test_moe_layer_params_include_router():
    cfg = _config(moe=True)
    subs = {
        "layer_norm": LayerNormProfiler(cfg),
        "self_attention": AttentionProfiler(cfg),
        "residual_add": ResidualAddProfiler(cfg),
        "router": RouterProfiler(cfg),
        "mlp": MoEMLPProfiler(cfg),
    }
    layer = MoETransformerLayerProfiler(cfg, subs)
    assert layer.estimated_num_params() == (
        subs["layer_norm"].estimated_num_params() * 3
        + subs["self_attention"].estimated_num_params()
        + subs["mlp"].estimated_num_params()
        + subs["router"].estimated_num_params()
        + subs["residual_add"].estimated_num_params() * 2
    )


def test_layer_profiler_specs_expose_expected_sub_profilers():
    dense = get_dense_transformer_layer_profiler_spec(_config())
    assert dense.profiler is DenseTransformerLayerProfiler
    assert set(dense.sub_profiler_specs) == {"layer_norm", "self_attention", "residual_add", "mlp"}
    moe = get_moe_transformer_layer_profiler_spec(_config())
    assert moe.profiler is MoETransformerLayerProfiler
    assert "router" in moe.sub_profiler_specs


# ============================== collective_args =================================
def test_collective_topology_mapping_from_parallelism():
    args = get_default_args(num_nodes=4, gpus_per_node=8, tp=2, pp=2, ep=4, cp=1)
    assert (args.node_size, args.pod_size, args.num_nodes) == (8, 32, 4)
    assert (args.hp, args.pp, args.ep, args.cp) == (2, 2, 4, 1)


def test_collective_bw_eff_applied_once_raw_preserved():
    args = get_default_args()
    assert args.node_bw == CollectiveArgs.node_bw * CollectiveArgs.bw_eff
    assert args._raw_node_bw == CollectiveArgs.node_bw
    assert args._raw_pod_bw == CollectiveArgs.pod_bw


def test_collective_nics_default_to_gpus_when_none():
    args = get_default_args(gpus_per_node=8, hardware_config={"nics_per_node": None})
    assert args.nics_per_node == 8


def test_collective_hardware_config_string_floats():
    args = get_default_args(hardware_config={"node_bw": "2048", "bw_eff": "0.5"})
    assert args._raw_node_bw == 2048.0
    assert args.node_bw == 2048.0 * 0.5


def test_collective_hardware_config_int_scientific_notation():
    args = get_default_args(hardware_config={"ar_warmup_chunk_bytes": "1e3"})
    assert args.ar_warmup_chunk_bytes == 1000
    assert isinstance(args.ar_warmup_chunk_bytes, int)


def test_collective_hardware_config_string_bool():
    assert get_default_args(hardware_config={"switch_topology": "false"}).switch_topology is False


def test_collective_unknown_key_warns_and_is_ignored(capsys):
    args = get_default_args(hardware_config={"does_not_exist": 123})
    assert not hasattr(args, "does_not_exist")
    assert "Unknown hardware parameter" in capsys.readouterr().out


# ============================== memory reports ==================================
def _mock_projection(*, applied_correction=False, legacy_residual=False):
    bd = {
        "analytical_at_target_corrected": {
            "static": {"params_bytes": 10 * GB, "grads_bytes": 10 * GB, "optimizer_bytes": 40 * GB},
            "activations": {"transformer_layers_bytes": 5 * GB},
            "deepep_buffers_bytes": GB,
            "comm_buffers_bytes": GB,
        },
        "activation_correction": (
            {
                "applied_correction": True,
                "correction_factors": {"attn": 1.1, "mlp": 0.9},
                "uncorrected_activation_bytes": 4 * GB,
                "corrected_activation_bytes": 5 * GB,
            }
            if applied_correction
            else {"applied_correction": False}
        ),
        "safety_margin": 0.1,
    }
    if legacy_residual:
        bd["residual_allocated_bytes"] = GB
        bd["residual_reserved_bytes"] = 2 * GB
    else:
        bd["framework_overhead_bytes"] = GB
        bd["live_tensor_excess_bytes"] = GB
        bd["residual_total_bytes"] = 2 * GB
    diag = {
        "bench_global_peak_allocated_bytes": 50 * GB,
        "bench_global_peak_reserved_bytes": 55 * GB,
        "bench_world_size": 8,
        "target_world_size": 64,
        "analytical_share": 0.8,
        "residual_share": 0.2,
    }
    return SimpleNamespace(
        breakdown=bd, diagnostics=diag, point_estimate_bytes=60 * GB, upper_bound_bytes=70 * GB
    )


def test_reports_gb_and_pct_formatting():
    assert _gb(1024**3) == "1.000 GB"
    assert _pct(0.1234) == "12.3%"


def test_print_per_rank_breakdown_fits_and_runs(capsys):
    print_per_rank_breakdown(_mock_projection(), target_label="t", total_vram_bytes=192 * GB)
    out = capsys.readouterr().out
    assert "Per-rank peak memory" in out
    assert "FITS" in out  # 192 GB ceiling > 60 GB point estimate


def test_print_per_rank_breakdown_correction_oom_and_warning(capsys):
    proj = _mock_projection(applied_correction=True)
    proj.diagnostics["warning"] = "tight"
    print_per_rank_breakdown(proj, total_vram_bytes=50 * GB)  # 50 < 60 -> OOM
    out = capsys.readouterr().out
    assert "Activation correction" in out
    assert "OOM" in out
    assert "[WARNING] tight" in out


def test_print_per_rank_breakdown_legacy_residual(capsys):
    print_per_rank_breakdown(_mock_projection(legacy_residual=True))
    assert "Residual (bench peak" in capsys.readouterr().out


def test_compare_simulate_vs_benchmark_runs(capsys):
    proj = _mock_projection()
    compare_simulate_vs_benchmark(
        {"total_bytes": 58 * GB, "param_optimizer_bytes": 55 * GB, "activation_bytes": 3 * GB}, proj
    )
    assert "simulate vs benchmark" in capsys.readouterr().out


def test_compare_simulate_vs_benchmark_accepts_int_total(capsys):
    compare_simulate_vs_benchmark(58 * GB, _mock_projection())
    assert "TOTAL (point estimate)" in capsys.readouterr().out


def test_report_dict_snapshot():
    proj = _mock_projection()
    d = report_dict(proj)
    assert d["point_estimate_bytes"] == proj.point_estimate_bytes
    assert d["upper_bound_bytes"] == proj.upper_bound_bytes
    assert "breakdown" in d and "diagnostics" in d
