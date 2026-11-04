###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention (MSA) in the projection: selected-block core,
indexer, per-layer dispatch and the benchmark-layer reduction.

The dense FAv3 model the MSA core scales is stubbed with fixed times so the
tests run without Origami.
"""

from types import SimpleNamespace

import pytest

from primus.core.projection.module_profilers import attention as attn
from primus.core.projection.module_profilers.language_model import (
    _parse_minimax_sparse_attention_pattern,
    parse_layer_pattern,
)
from primus.core.projection.performance_projection.projection import (
    _limit_layers_for_projection,
)
from primus.core.projection.simulation_backends.base import SimulationResult
from primus.core.projection.simulation_backends.sdpa_simulator import (
    GPUHardwareSpec,
    SDPASimulator,
)
from primus.core.projection.training_config import ModelConfig, update_config_from_args

PEAK_TF = 2516.6
N_CU = 256
M3_PATTERN = "([0]*3+[1]*57)"

M3_MSA = dict(
    num_attention_heads=64,
    num_kv_heads=4,
    head_dim=128,
    hidden_size=6144,
    num_index_heads=4,
    index_head_dim=128,
    block_size=128,
    topk_blocks=16,
)


DENSE_FWD_MS = 2.0
DENSE_BWD_MS = 5.0


def _sim():
    sim = SDPASimulator.__new__(SDPASimulator)
    sim._hw = GPUHardwareSpec(peak_tflops_bf16=PEAK_TF, n_cu=N_CU)
    sim._mode = "stub"
    sim.dense_calls = []

    def dense_sdpa(**kw):
        sim.dense_calls.append(kw)
        return SimulationResult(forward_time_ms=DENSE_FWD_MS, backward_time_ms=DENSE_BWD_MS)

    sim.simulate_sdpa = dense_sdpa
    return sim


@pytest.mark.parametrize(
    "seq_len, expected_keys",
    [(4096, 1504.5), (8192, 1744.5), (32768, 1924.5), (131072, 1969.5)],
)
def test_keys_per_query_is_causal_selected_blocks(seq_len, expected_keys):
    keys = SDPASimulator.minimax_msa_keys_per_query(seq_len, block_size=128, topk_blocks=16)
    assert keys == pytest.approx(expected_keys)


@pytest.mark.parametrize("seq_len", [4096, 32768])
def test_keys_per_query_matches_flops_patch(seq_len):
    flops_patch = pytest.importorskip("primus.backends.megatron.patches.minimax_m3_flops_patches")
    assert SDPASimulator.minimax_msa_keys_per_query(seq_len, 128, 16) == pytest.approx(
        flops_patch.msa_keys_per_query(seq_len, 128, 16)
    )


@pytest.mark.parametrize("seq_len, keys", [(4096, 1504.5), (32768, 1924.5)])
def test_core_scales_dense_kernel_by_selected_keys(seq_len, keys):
    sim = _sim()
    r = sim.simulate_minimax_sparse_attention(batch_size=1, seq_len=seq_len, indexer_loss_coeff=1e-2, **M3_MSA)
    ((dense_call,),) = [sim.dense_calls]
    assert dense_call["seq_len"] == seq_len
    assert dense_call["num_heads"] == 64
    assert dense_call["num_heads_kv"] == 4
    assert dense_call["causal"] is True
    key_ratio = keys / (0.5 * seq_len)
    assert r.metadata["core_fwd_ms"] == pytest.approx(DENSE_FWD_MS * key_ratio)
    assert r.metadata["core_bwd_ms"] == pytest.approx(DENSE_BWD_MS * key_ratio)
    core_flops = 2.0 * 64 * seq_len * keys * (128 + 128)
    assert r.metadata["fwd_flops"] - r.metadata["indexer_fwd_flops"] == pytest.approx(core_flops)


def test_core_efficiency_ratio_divides_core_time(monkeypatch):
    base = _sim().simulate_minimax_sparse_attention(batch_size=1, seq_len=4096, indexer_loss_coeff=1e-2, **M3_MSA)
    monkeypatch.setenv("PRIMUS_MSA_CORE_EFFICIENCY_RATIO", "0.5")
    slow = _sim().simulate_minimax_sparse_attention(batch_size=1, seq_len=4096, indexer_loss_coeff=1e-2, **M3_MSA)
    assert slow.metadata["core_fwd_ms"] == pytest.approx(2.0 * base.metadata["core_fwd_ms"])
    assert slow.metadata["core_bwd_ms"] == pytest.approx(2.0 * base.metadata["core_bwd_ms"])
    assert slow.metadata["indexer_fwd_ms"] == pytest.approx(base.metadata["indexer_fwd_ms"])


def test_indexer_flops_match_flops_patch():
    sim = _sim()
    seq, hidden, heads, dim, topk = 4096, 6144, 4, 128, 16
    r = sim.simulate_minimax_sparse_attention(batch_size=1, seq_len=seq, indexer_loss_coeff=1e-2, **M3_MSA)
    projection = hidden * (heads * dim + dim)
    per_token = 2 * projection + heads * dim * (seq + 1) / 2 + 2 * heads * dim * topk
    assert r.metadata["indexer_fwd_flops"] + r.metadata["indexer_bwd_flops"] == pytest.approx(2.0 * seq * per_token)


def test_frozen_indexer_has_no_backward():
    r = _sim().simulate_minimax_sparse_attention(batch_size=1, seq_len=4096, indexer_loss_coeff=0.0, **M3_MSA)
    assert r.metadata["indexer_bwd_ms"] == 0.0
    assert r.metadata["indexer_bwd_flops"] == 0.0


def test_traffic_counts_kv_once_per_group():
    r = _sim().simulate_minimax_sparse_attention(batch_size=1, seq_len=4096, indexer_loss_coeff=1e-2, **M3_MSA)
    hbm_bytes_per_ms = GPUHardwareSpec().hbm_bandwidth_gbps * 1.0e6 * r.metadata["memory_efficiency"]
    kv = 4096 * 4 * 128 * 2 * 2.0
    scores = 4096 * 4 * 32 * 4.0 * 2.0
    ids = 4096 * 4 * 16 * 4.0 * 2.0
    assert r.metadata["traffic_fwd_ms"] == pytest.approx((kv + scores + ids) / hbm_bytes_per_ms)


def test_rejects_heads_not_divisible_by_kv_groups():
    with pytest.raises(ValueError):
        _sim().simulate_minimax_sparse_attention(
            batch_size=1, seq_len=4096, indexer_loss_coeff=0.0, **{**M3_MSA, "num_kv_heads": 3}
        )


@pytest.mark.parametrize(
    "value, expected",
    [
        (M3_PATTERN, [0] * 3 + [1] * 57),
        ("[0]*3+[1]*57", [0] * 3 + [1] * 57),
        (None, [1] * 60),
        (1, [1] * 60),
        ([0, 1] * 30, [0, 1] * 30),
    ],
)
def test_parse_layer_pattern(value, expected):
    assert parse_layer_pattern(value, 60, "sparse_attention_freq") == expected


def test_parse_layer_pattern_rejects_wrong_length():
    with pytest.raises(ValueError):
        parse_layer_pattern(M3_PATTERN, 2, "sparse_attention_freq")


def test_dense_models_have_no_msa_schedule():
    assert _parse_minimax_sparse_attention_pattern(SimpleNamespace(num_layers=4)) is None


def test_model_config_carries_msa_fields():
    args = SimpleNamespace(
        minimax_sparse_attention=True,
        sparse_attention_freq=M3_PATTERN,
        sparse_topk_blocks=8,
    )
    cfg = update_config_from_args(ModelConfig(), args)
    assert cfg.minimax_sparse_attention is True
    assert cfg.sparse_attention_freq == M3_PATTERN
    assert cfg.sparse_topk_blocks == 8
    assert cfg.sparse_block_size == 128


def test_benchmark_reduction_keeps_msa_on_moe_representative(monkeypatch):
    monkeypatch.delenv("PRIMUS_PROJ_MAX_LAYERS", raising=False)
    monkeypatch.delenv("PRIMUS_PROJ_COMPRESS_RATIOS", raising=False)
    module_config = SimpleNamespace(
        num_layers=60,
        num_experts=128,
        moe_layer_freq=M3_PATTERN,
        minimax_sparse_attention=True,
        sparse_attention_freq=M3_PATTERN,
        pipeline_model_parallel_size=4,
    )
    _limit_layers_for_projection(module_config)
    assert module_config.num_layers == 2
    assert module_config.moe_layer_freq == [0, 1]
    assert module_config.sparse_attention_freq == [0, 1]


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def simulate_minimax_sparse_attention(self, **kw):
        self.calls.append(("msa", kw))
        return SimulationResult(forward_time_ms=1.0, backward_time_ms=2.0)

    def simulate_sdpa(self, **kw):
        self.calls.append(("sdpa", kw))
        return SimulationResult(forward_time_ms=1.0, backward_time_ms=2.0)


def _m3_profiler(msa_layer):
    args = SimpleNamespace(
        hidden_size=6144,
        num_attention_heads=64,
        kv_channels=128,
        group_query_attention=True,
        num_query_groups=4,
        multi_latent_attention=False,
        qk_layernorm=True,
        use_flash_attn=False,
        fp8=None,
        minimax_sparse_attention=True,
        sparse_num_index_heads=4,
        sparse_index_dim=128,
        sparse_block_size=128,
        sparse_topk_blocks=16,
        sparse_indexer_loss_coeff=1e-2,
    )
    mp = SimpleNamespace(tensor_model_parallel_size=1, context_model_parallel_size=1)
    prof = attn.AttentionProfiler(SimpleNamespace(model_config=args, model_parallel_config=mp))
    backend = _RecordingBackend()
    prof.set_sdpa_backend(backend)
    prof.set_sim_minimax_sparse_attention(msa_layer)
    return prof, backend


def test_profiler_dispatches_msa_layers_to_sparse_core():
    prof, backend = _m3_profiler(msa_layer=True)
    prof.measured_forward_time(1, 4096)
    ((kind, call),) = backend.calls
    assert kind == "msa"
    assert call["num_attention_heads"] == 64
    assert call["num_kv_heads"] == 4
    assert call["topk_blocks"] == 16
    assert call["indexer_loss_coeff"] == pytest.approx(1e-2)


def test_profiler_keeps_dense_gqa_for_dense_attention_layers():
    prof, backend = _m3_profiler(msa_layer=False)
    prof.measured_forward_time(1, 4096)
    ((kind, call),) = backend.calls
    assert kind == "sdpa"
    assert call["num_heads_kv"] == 4


def test_msa_activation_has_no_score_matrix():
    sparse, _ = _m3_profiler(msa_layer=True)
    dense, _ = _m3_profiler(msa_layer=False)
    assert sparse.estimated_activation_memory(1, 4096) < dense.estimated_activation_memory(1, 4096)
