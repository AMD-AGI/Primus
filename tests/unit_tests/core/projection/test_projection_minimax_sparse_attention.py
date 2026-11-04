###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention (MSA) in the projection: selected-block core,
indexer, per-layer dispatch and the benchmark-layer reduction.

The dense FAv3 model the MSA core scales is stubbed with fixed times and the
1-CU tile GEMM with a fixed-rate matmul, so the tests run without Origami.
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
    num_index_heads=4,
    index_head_dim=128,
    block_size=128,
    topk_blocks=16,
)
HIDDEN = 6144

DENSE_FWD_MS = 2.0
DENSE_BWD_MS = 5.0
TILE_FLOPS_PER_MS = 1.0e9
VECTOR_FLOPS = 1.0e14


class _FixedRateGemm:
    """Prices a GEMM as 2mnk at a fixed rate and records its shape."""

    hbm_bandwidth_gbps = 5300.0

    def __init__(self):
        self.calls = []

    def simulate_gemm(self, m, n, k, dtype="bf16", **_kw):
        self.calls.append((m, n, k))
        return SimulationResult(forward_time_ms=2.0 * m * n * k / TILE_FLOPS_PER_MS)

    def vector_flops(self):
        return VECTOR_FLOPS


def _tile_ms(m, n, k):
    return 2.0 * m * n * k / TILE_FLOPS_PER_MS


def _sim():
    sim = SDPASimulator.__new__(SDPASimulator)
    sim._hw = GPUHardwareSpec(peak_tflops_bf16=PEAK_TF, n_cu=N_CU)
    sim._tile_gemm = _FixedRateGemm()
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


def _msa(seq_len=4096, trained=True, **overrides):
    return _sim().simulate_minimax_sparse_attention(
        batch_size=1, seq_len=seq_len, trained_indexer=trained, **{**M3_MSA, **overrides}
    )


@pytest.mark.parametrize("seq_len, expected", [(4096, 12.25), (131072, 16264 / 1024)])
def test_selected_blocks_per_query_fill_causally(seq_len, expected):
    assert SDPASimulator.minimax_msa_selected_blocks_per_query(seq_len, 128, 16) == pytest.approx(expected)


def test_plan_chunks_follow_flydsl_plan_chunks():
    # 4k M3 table: 4 heads x 4096 tokens x 16 slots over 4 x 32 (head, block) rows
    assert SDPASimulator.minimax_msa_plan_chunks(262144, 128) == (128 + 2048, 128)
    assert SDPASimulator.minimax_msa_plan_chunks(1000, 3) == (3 + 16, 64)


@pytest.mark.parametrize("seq_len, keys", [(4096, 1504.5), (32768, 1924.5)])
def test_core_scales_dense_kernel_by_selected_keys(seq_len, keys):
    sim = _sim()
    r = sim.simulate_minimax_sparse_attention(batch_size=1, seq_len=seq_len, trained_indexer=True, **M3_MSA)
    ((dense_call,),) = [sim.dense_calls]
    assert dense_call["seq_len"] == seq_len
    assert dense_call["num_heads"] == 64
    assert dense_call["num_heads_kv"] == 4
    assert dense_call["causal"] is True
    key_ratio = keys / (0.5 * seq_len)
    assert r.metadata["core_fwd_ms"] == pytest.approx(DENSE_FWD_MS * key_ratio)
    assert r.metadata["core_bwd_ms"] == pytest.approx(DENSE_BWD_MS * key_ratio)
    assert r.metadata["core_fwd_flops"] == pytest.approx(2.0 * 64 * seq_len * keys * (128 + 128))


def test_index_scores_run_as_causal_64_query_tiles():
    r = _msa(4096)
    # 64 queries x 4 index heads per work-group, 64 work-groups: one wave on 256 CUs
    assert r.metadata["index_score_waves"] == 1
    assert r.metadata["index_score_matmul_ms"] == pytest.approx(_tile_ms(256, 4096, 128) * 0.5)
    elems = 4 * 4096 * 4097 / 2
    assert r.metadata["index_score_vector_ms"] == pytest.approx(3.0 * elems / VECTOR_FLOPS * 1e3)


def test_index_score_flops_match_flops_patch():
    seq = 4096
    r = _msa(seq)
    assert r.metadata["index_score_flops"] == pytest.approx(2.0 * seq * 4 * 128 * (seq + 1) / 2)


def test_index_dk_walks_the_block_plan():
    r = _msa(4096)
    # 2176 chunks over 256 CUs -> 9 waves; 12.25 live slots per query
    live = 4 * 4096 * 12.25
    per_chunk = -(-int(live) // 2176)
    assert r.metadata["index_dk_ms"] == pytest.approx(_tile_ms(128, 128, per_chunk) * 9)
    assert r.metadata["index_dk_flops"] == pytest.approx(2.0 * live * 128)


def test_frozen_indexer_has_no_index_backward():
    r = _msa(4096, trained=False)
    assert r.metadata["index_dk_ms"] == 0.0
    assert r.backward_time_ms == pytest.approx(r.metadata["core_bwd_ms"])


def test_rejects_heads_not_divisible_by_kv_groups():
    with pytest.raises(ValueError):
        _msa(4096, num_kv_heads=3)


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
    assert call["trained_indexer"] is True


def _indexer_ms(trained, gemm=None, seq_len=4096):
    prof, _ = _m3_profiler(msa_layer=True)
    prof._gemm_backend = gemm
    shape = dict(M3_MSA, batch_size=1, seq_len=seq_len, trained_indexer=trained)
    meta = {"selected_blocks_per_query": 12.25}
    return prof._simulate_minimax_indexer(meta, **shape)


def test_indexer_projections_are_gemms_with_weight_gradients_only():
    gemm = _FixedRateGemm()
    _indexer_ms(trained=True, gemm=gemm)
    T = 4096
    # forward index-q / index-k, then their weight gradients: the input is detached
    assert gemm.calls == [(T, 512, HIDDEN), (T, 128, HIDDEN), (HIDDEN, 512, T), (HIDDEN, 128, T)]


def test_frozen_indexer_skips_its_gradients():
    gemm = _FixedRateGemm()
    trained_fwd, trained_bwd = _indexer_ms(trained=True, gemm=gemm)
    frozen_fwd, frozen_bwd = _indexer_ms(trained=False, gemm=_FixedRateGemm())
    assert frozen_bwd < trained_bwd
    assert frozen_fwd < trained_fwd  # no slot mass / KL loss in the forward either


def test_indexer_memory_terms_scale_with_hbm_bandwidth():
    slow = _FixedRateGemm()
    fast = _FixedRateGemm()
    fast.hbm_bandwidth_gbps = 2 * slow.hbm_bandwidth_gbps
    slow_fwd, _ = _indexer_ms(trained=True, gemm=slow)
    fast_fwd, _ = _indexer_ms(trained=True, gemm=fast)
    assert fast_fwd < slow_fwd


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
