###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 sparse-MLA attention core: valid-key counts and per-token tiles.

The 1-CU tile GEMM is stubbed as a fixed-rate matmul so the tests run without
Origami.
"""

from types import SimpleNamespace

import pytest

from primus.core.projection.module_profilers import attention as attn
from primus.core.projection.simulation_backends.base import SimulationResult
from primus.core.projection.simulation_backends.sdpa_simulator import (
    GPUHardwareSpec,
    SDPASimulator,
    _get_hardware_spec,
)

PEAK_TF = 2516.6
N_CU = 256


class _FixedRateTile:
    def __init__(self, per_cu_flops, vector_flops=None):
        self._rate = per_cu_flops
        self._vec = vector_flops

    def simulate_gemm(self, m, n, k, dtype="bf16", **_kw):
        return SimulationResult(forward_time_ms=2.0 * m * n * k / self._rate * 1e3)

    def vector_flops(self):
        return self._vec


def _sim(efficiency, vector_flops=None):
    sim = SDPASimulator.__new__(SDPASimulator)
    sim._hw = GPUHardwareSpec(peak_tflops_bf16=PEAK_TF, n_cu=N_CU)
    sim._tile_gemm = _FixedRateTile(efficiency * PEAK_TF * 1e12 / N_CU, vector_flops)
    sim._mode = "stub"
    return sim


@pytest.mark.parametrize(
    "arch, bf16_tf, fp8_tf",
    [("mi355x", 2516.6, 5033.2), ("gfx950", 2516.6, 5033.2), ("mi350x", 2309.6, 4614.0)],
)
def test_cdna4_peaks_match_published_peak(arch, bf16_tf, fp8_tf):
    spec = _get_hardware_spec(arch)
    assert spec.peak_tflops_bf16 == pytest.approx(bf16_tf)
    assert spec.peak_tflops_fp8 == pytest.approx(fp8_tf)


def test_clock_override_scales_from_spec_clock():
    assert _get_hardware_spec("mi355x", 2400).peak_tflops_bf16 == pytest.approx(2516.6)
    assert _get_hardware_spec("mi355x", 2200).peak_tflops_bf16 == pytest.approx(2309.6, rel=2e-3)


@pytest.mark.parametrize(
    "seq_len, window, expected",
    [(4096, 128, 126.02), (16384, 128, 127.50), (4096, 0, 2048.5), (100, 128, 50.5)],
)
def test_swa_band_is_full_width_not_halved(seq_len, window, expected):
    assert attn._swa_keys_per_query(seq_len, window) == pytest.approx(expected, abs=0.01)


def test_hca_pool_is_causal():
    assert attn._pool_keys_per_query(4096, 128) == pytest.approx(15.5, abs=0.01)


@pytest.mark.parametrize(
    "seq_len, topk, expected_keys",
    [
        (4096, 1024, 638.0),  # pro cr4 4K: pool still filling
        (16384, 1024, 1023.5),  # pro cr4 16K: top-K nearly full
        (4096, 512, 510.0),  # flash cr4 4K
        (16384, 512, 607.5),  # flash cr4 16K
    ],
)
def test_csa_keys_use_fill_ratio(seq_len, topk, expected_keys):
    pool = -(-seq_len // 4)
    k_cap = min(topk, pool)
    keys = attn._swa_keys_per_query(seq_len, 128) + k_cap * attn._csa_fill_ratio(seq_len, 4, k_cap)
    assert keys == pytest.approx(expected_keys, abs=0.1)


def test_sparse_mla_backward_scales_with_heads():
    sim = _sim(efficiency=0.5)
    pro = sim.simulate_sparse_mla(1, 128, 4096, 512, 126.0, heads_per_workgroup=128)
    flash = sim.simulate_sparse_mla(1, 64, 4096, 512, 126.0, heads_per_workgroup=128)
    assert pro.metadata["waves"] == flash.metadata["waves"]
    assert pro.backward_time_ms == pytest.approx(2.0 * flash.backward_time_ms)
    assert pro.forward_time_ms == pytest.approx(2.0 * flash.forward_time_ms)


def test_sparse_mla_never_beats_matrix_peak():
    sim = _sim(efficiency=2.0)
    r = sim.simulate_sparse_mla(1, 128, 16384, 512, 1023.5)
    assert r.forward_time_ms == pytest.approx(r.metadata["fwd_flops"] / (PEAK_TF * 1e12) * 1e3)
    assert r.backward_time_ms == pytest.approx(r.metadata["bwd_flops"] / (PEAK_TF * 1e12) * 1e3)
    assert r.tflops == pytest.approx(PEAK_TF)


def test_sparse_mla_adds_softmax_on_vector_units():
    no_vec = _sim(efficiency=0.5).simulate_sparse_mla(1, 128, 4096, 512, 638.0)
    vec = 150e12
    with_vec = _sim(efficiency=0.5, vector_flops=vec).simulate_sparse_mla(1, 128, 4096, 512, 638.0)
    softmax_ms = 5.0 * 128 * 4096 * 638.0 / vec * 1e3
    assert with_vec.forward_time_ms - no_vec.forward_time_ms == pytest.approx(softmax_ms)


class _RecordingSparseBackend:
    def __init__(self):
        self.calls = []

    def simulate_sparse_mla(self, **kw):
        self.calls.append(("sparse_mla", kw))
        return SimulationResult(forward_time_ms=1.0, backward_time_ms=2.0)

    def simulate_sdpa(self, **kw):
        self.calls.append(("sdpa", kw))
        return SimulationResult(forward_time_ms=1.0, backward_time_ms=2.0)


def _v4_profiler(cr, seq_len, heads, topk):
    args = SimpleNamespace(
        compress_ratios=[cr],
        hc_mult=1,
        hidden_size=4096,
        num_attention_heads=heads,
        kv_channels=512,
        q_lora_rank=0,
        o_lora_rank=0,
        o_groups=1,
        attn_sliding_window=128,
        index_topk=topk,
        index_head_dim=128,
        index_n_heads=64,
        multi_latent_attention=False,
        use_v4_triton_attention=True,
        fp8=None,
    )
    mp = SimpleNamespace(tensor_model_parallel_size=1, context_model_parallel_size=1)
    prof = attn.AttentionProfiler(SimpleNamespace(model_config=args, model_parallel_config=mp))
    backend = _RecordingSparseBackend()
    prof.set_sdpa_backend(backend)
    prof.set_sim_compress_ratio(cr)
    return prof, backend


@pytest.mark.parametrize(
    "cr, seq_len, heads, topk, expected_keys",
    [
        (0, 4096, 128, 1024, 126.02),
        (128, 4096, 128, 1024, 141.52),
        (4, 4096, 128, 1024, 638.02),
        (4, 16384, 128, 1024, 1023.50),
        (4, 16384, 64, 512, 607.50),
    ],
)
def test_v4_profiler_prices_valid_keys(cr, seq_len, heads, topk, expected_keys):
    prof, backend = _v4_profiler(cr, seq_len, heads, topk)
    prof.measured_forward_time(1, seq_len)
    ((kind, call),) = backend.calls
    assert kind == "sparse_mla"
    assert call["keys_per_query"] == pytest.approx(expected_keys, abs=0.01)
    assert call["num_heads"] == heads
    assert call["head_dim"] == 512


def test_v4_profiler_fav3_opt_out(monkeypatch):
    monkeypatch.setenv("PRIMUS_ATTN_V4_CORE", "fav3")
    prof, backend = _v4_profiler(0, 4096, 128, 1024)
    prof.measured_forward_time(1, 4096)
    ((kind, call),) = backend.calls
    assert kind == "sdpa"
    assert call["seq_len_kv"] == 128
