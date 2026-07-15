###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for the single-engine discrete-event simulator (``des.py``).

Phase 3 of the inference roadmap, converged with the ``tools/serving_sim``
vLLM-V1 packing model. These exercise the DES's pure-logic surface and its
end-to-end behaviour against a **fake cost kernel** — a tiny stand-in for
``InferencePerformanceProjector`` that implements just the three methods the DES
calls (``decode_step_latency_ms`` / ``mixed_step_latency_ms`` / ``project``).

Decoupling from the real profiler keeps these tests fast and independent of the
native Origami simulation backend, while still validating:

  * pure helpers (percentile, slope, arrivals, accept sampling, length sampling);
  * the vLLM-V1 unified-batch scheduler invariants (CONC cap, token budget,
    full-ISL KV reservation) — mirroring the serving_sim ``_selftest``;
  * workload heterogeneity, gamma burstiness, and workload-file replay;
  * latency behaviour — determinism, percentile ordering, TTFT floor, queueing
    growth, the saturation knee, speculative ITL spread; and
  * the packing summary + per-step records, and the throughput-latency sweep.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("primus.core.projection.inference_projection.des")

from primus.core.projection.inference_projection import des as des_mod  # noqa: E402
from primus.core.projection.inference_projection.des import (  # noqa: E402
    _build_workload,
    _generate_arrivals,
    _load_workload_file,
    _pct,
    _sample_accepted,
    _sample_len,
    _slope,
    run_des,
    simulate_once,
)
from primus.core.projection.training_config import (  # noqa: E402
    InferenceConfig,
    InferenceRequestConfig,
    ModelConfig,
    ModelParallelConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fake cost kernel — memory-bound-ish, monotonic, cheap. The DES only needs
# these three methods; using a stub avoids the native profiler entirely.
# ─────────────────────────────────────────────────────────────────────────────
class _FakeProjector:
    DECODE_FLOOR = 4.0        # ms, fixed per-step cost
    DECODE_PER_SEQ = 0.02     # ms per resident decode sequence
    DECODE_PER_KV = 0.0005    # ms per KV token attended
    PREFILL_PER_TOK = 0.02    # ms per prefill query token (compute-bound)
    SPEC_PER_TOK = 0.5        # ms per extra speculative verify token
    OVERHEAD = 0.2

    def decode_step_latency_ms(self, batch, kv_len, q_len=1):
        return (
            self.DECODE_FLOOR
            + self.DECODE_PER_SEQ * max(1, batch)
            + self.DECODE_PER_KV * max(1, kv_len)
            + self.SPEC_PER_TOK * max(0, q_len - 1)
            + self.OVERHEAD
        )

    def mixed_step_latency_ms(self, num_decode, chunk_tokens, decode_ctx, prefill_kv_len, q_len=1):
        prefill = self.PREFILL_PER_TOK * max(1, chunk_tokens)
        decode = 0.0
        if num_decode > 0:
            decode = (
                self.DECODE_PER_SEQ * num_decode
                + self.DECODE_PER_KV * max(1, decode_ctx)
                + self.SPEC_PER_TOK * max(0, q_len - 1)
            )
        return prefill + decode + self.OVERHEAD

    def project(self):
        # mu reference at full batch=64, mid context ~600.
        step = self.decode_step_latency_ms(64, 600)
        return SimpleNamespace(
            decode_throughput_tps=64 * 1000.0 / step,
            decode_step_latency_ms=step,
        )


def _cfg(**req) -> InferenceConfig:
    return InferenceConfig(
        model_config=ModelConfig(num_layers=4, hidden_size=1024, num_attention_heads=16),
        request_config=InferenceRequestConfig(**req),
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=1),
    )


@pytest.fixture
def proj():
    return _FakeProjector()


@pytest.fixture
def base_cfg():
    return _cfg(input_seq_len=512, output_seq_len=128, max_concurrency=64)


# ─────────────────────────────────────────────────────────────────────────────
# pure helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_pct_basic():
    xs = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _pct(xs, 0.0) == 10.0
    assert _pct(xs, 1.0) == 50.0
    assert _pct(xs, 0.5) == 30.0
    assert _pct([], 0.5) == 0.0
    assert _pct([7.0], 0.9) == 7.0


def test_slope_sign():
    assert _slope([0, 1, 2, 3], [0, 2, 4, 6]) == pytest.approx(2.0)
    assert _slope([0, 1, 2, 3], [5, 5, 5, 5]) == pytest.approx(0.0)
    assert _slope([1.0], [1.0]) == 0.0


def test_arrivals_deterministic_is_evenly_spaced():
    a = _generate_arrivals(5, rate_per_s=10.0, model="deterministic", rng=random.Random(0))
    assert a == pytest.approx([100.0, 200.0, 300.0, 400.0, 500.0])


def test_arrivals_infinite_rate_all_at_zero():
    a = _generate_arrivals(4, rate_per_s=float("inf"), model="poisson", rng=random.Random(0))
    assert a == [0.0, 0.0, 0.0, 0.0]


def test_arrivals_poisson_monotone_and_seeded():
    a = _generate_arrivals(50, 10.0, "poisson", random.Random(1))
    b = _generate_arrivals(50, 10.0, "poisson", random.Random(1))
    assert a == b
    assert all(a[i] <= a[i + 1] for i in range(len(a) - 1))
    assert 60.0 < (a[-1] / len(a)) < 160.0


def test_burstiness_increases_interarrival_variance():
    # Gamma shape < 1 is burstier (higher variance) than shape 1 (Poisson).
    def gaps(burst):
        a = _generate_arrivals(4000, 10.0, "poisson", random.Random(3), burstiness=burst)
        g = [a[i + 1] - a[i] for i in range(len(a) - 1)]
        m = sum(g) / len(g)
        return sum((x - m) ** 2 for x in g) / len(g)

    assert gaps(0.3) > gaps(1.0) > gaps(3.0)


def test_sample_accepted_bounds():
    rng = random.Random(0)
    assert _sample_accepted(rng, 0, 0.0, 5) == 1
    assert _sample_accepted(rng, 4, 1.0, 10) == 5
    assert _sample_accepted(rng, 4, 1.0, 2) == 2
    assert _sample_accepted(rng, 4, 0.0, 10) == 1


def test_sample_len_range():
    rng = random.Random(0)
    assert _sample_len(rng, 100, 1.0) == 100  # homogeneous
    for _ in range(200):
        v = _sample_len(rng, 100, 0.5)
        assert 50 <= v <= 100


def test_build_workload_heterogeneous_lengths():
    rng = random.Random(0)
    arr = [float(i) for i in range(200)]
    reqs = _build_workload(200, arr, input_len=1000, output_len=200, range_ratio=0.5, rng=rng)
    assert len(reqs) == 200
    assert all(500 <= r.prompt_len <= 1000 for r in reqs)
    assert all(100 <= r.output_len <= 200 for r in reqs)
    assert len({r.prompt_len for r in reqs}) > 1  # genuinely varied


# ─────────────────────────────────────────────────────────────────────────────
# scheduler invariants (mirror serving_sim _selftest)
# ─────────────────────────────────────────────────────────────────────────────


def test_batch_never_exceeds_concurrency(proj):
    cfg = _cfg(input_seq_len=64, output_seq_len=32, max_concurrency=4)
    r = simulate_once(cfg, proj, rate_per_s=float("inf"), num_requests=50, seed=0, record_steps=True)
    assert r.steps is not None
    assert all(s["batch_size"] <= 4 for s in r.steps)


def test_token_budget_respected(proj):
    cfg = _cfg(
        input_seq_len=512, output_seq_len=16, max_concurrency=64,
        max_num_batched_tokens=1024,
    )
    r = simulate_once(cfg, proj, rate_per_s=float("inf"), num_requests=40, seed=0, record_steps=True)
    assert all(s["total_query_tokens"] <= 1024 for s in r.steps)


def test_kv_reservation_gate_never_overflows(proj):
    cfg = _cfg(input_seq_len=100, output_seq_len=100, max_concurrency=100)
    # Pool holds ~3 full sequences (200 tokens each).
    r = simulate_once(
        cfg, proj, rate_per_s=float("inf"), num_requests=30, seed=0,
        kv_cache_tokens=600, record_steps=True,
    )
    assert all(s["kv_tokens_in_use"] <= 600 for s in r.steps)
    assert r.packing["kv_peak_tokens"] <= 600
    # At most 3 sequences resident at once given the pool.
    assert r.packing["max_batch_size"] <= 3


def test_chunked_prefill_produces_prefill_chunks(proj):
    # prompt 100, chunk 40 => 3 prefill chunks (40,40,20) for a lone request.
    cfg = _cfg(
        input_seq_len=100, output_seq_len=5, max_concurrency=4,
        chunked_prefill_size=40, max_num_batched_tokens=40,
    )
    r = simulate_once(cfg, proj, rate_per_s=float("inf"), num_requests=1, seed=0, record_steps=True)
    prefill_steps = [s for s in r.steps if s["num_prefill_reqs"] > 0]
    decode_steps = [s for s in r.steps if s["num_decode_reqs"] > 0]
    assert len(prefill_steps) == 3
    assert len(decode_steps) == 4  # OSL-1 (first token emitted at end of prefill)
    assert all(s["total_query_tokens"] <= 40 for s in r.steps)


# ─────────────────────────────────────────────────────────────────────────────
# latency behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_determinism(proj, base_cfg):
    a = simulate_once(base_cfg, proj, rate_per_s=40.0, num_requests=150, seed=11)
    b = simulate_once(base_cfg, proj, rate_per_s=40.0, num_requests=150, seed=11)
    assert a.ttft == b.ttft and a.e2e == b.e2e and a.tpot == b.tpot
    assert a.achieved_rate == b.achieved_rate


def test_percentiles_are_ordered(proj, base_cfg):
    r = simulate_once(base_cfg, proj, rate_per_s=40.0, num_requests=300, seed=0)
    for d in (r.ttft, r.tpot, r.itl, r.e2e):
        assert d["p50"] <= d["p90"] <= d["p99"]
        assert d["mean"] > 0.0


def test_ttft_floor_is_at_least_prefill(proj, base_cfg):
    r = simulate_once(base_cfg, proj, rate_per_s=5.0, num_requests=150, seed=0)
    prefill = proj.mixed_step_latency_ms(0, base_cfg.request_config.input_seq_len, 1, 512)
    assert r.ttft["p50"] >= 0.8 * prefill


def test_higher_load_raises_ttft_tail(proj, base_cfg):
    mu = proj.project().decode_throughput_tps / base_cfg.request_config.output_seq_len
    lo = simulate_once(base_cfg, proj, rate_per_s=0.3 * mu, num_requests=300, seed=0)
    hi = simulate_once(base_cfg, proj, rate_per_s=0.95 * mu, num_requests=300, seed=0)
    assert hi.ttft["p99"] > lo.ttft["p99"]


def test_saturation_flag_above_mu(proj, base_cfg):
    mu = proj.project().decode_throughput_tps / base_cfg.request_config.output_seq_len
    stable = simulate_once(base_cfg, proj, rate_per_s=0.25 * mu, num_requests=300, seed=0)
    overloaded = simulate_once(base_cfg, proj, rate_per_s=3.0 * mu, num_requests=400, seed=0)
    assert stable.saturated is False
    assert overloaded.saturated is True


def test_steady_state_agreement_at_high_load(proj, base_cfg):
    mu = proj.project().decode_throughput_tps / base_cfg.request_config.output_seq_len
    r = simulate_once(base_cfg, proj, rate_per_s=0.9 * mu, num_requests=400, seed=0)
    pure = proj.project().decode_step_latency_ms
    assert 0.5 * pure <= r.tpot["p50"] <= 2.0 * pure


def test_spec_decode_widens_itl_distribution(proj):
    cfg = _cfg(
        input_seq_len=512, output_seq_len=128, max_concurrency=32,
        speculative_num_tokens=4, speculative_acceptance_rate=0.7,
    )
    r = simulate_once(cfg, proj, rate_per_s=30.0, num_requests=200, seed=0)
    assert r.itl["p99"] > r.itl["mean"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# heterogeneity, KV pool, workload file, packing, sweep
# ─────────────────────────────────────────────────────────────────────────────


def test_range_ratio_widens_e2e_distribution(proj, base_cfg):
    homo = simulate_once(base_cfg, proj, rate_per_s=40.0, num_requests=300, seed=0, range_ratio=1.0)
    het = simulate_once(base_cfg, proj, rate_per_s=40.0, num_requests=300, seed=0, range_ratio=0.2)
    homo_spread = homo.e2e["p99"] - homo.e2e["p50"]
    het_spread = het.e2e["p99"] - het.e2e["p50"]
    assert het_spread > homo_spread


def test_kv_pool_limits_concurrency(proj, base_cfg):
    big = simulate_once(base_cfg, proj, rate_per_s=float("inf"), num_requests=120, seed=0)
    small = simulate_once(
        base_cfg, proj, rate_per_s=float("inf"), num_requests=120, seed=0, kv_cache_tokens=5 * 640
    )
    # A tiny pool holds far fewer resident sequences.
    assert small.packing["max_batch_size"] < big.packing["max_batch_size"]
    assert small.packing["kv_peak_tokens"] <= 5 * 640


def test_workload_file_json(proj, base_cfg, tmp_path):
    rows = [{"arrival": i * 20.0, "isl": 128, "osl": 16} for i in range(40)]
    p = tmp_path / "wl.json"
    p.write_text(json.dumps(rows))
    loaded = _load_workload_file(str(p))
    assert len(loaded) == 40 and loaded[0] == (0.0, 128, 16)
    r = simulate_once(base_cfg, proj, rate_per_s=0.0, num_requests=0, seed=0, workload_file=str(p))
    assert r.num_requests == 40
    assert r.e2e["p50"] > 0.0


def test_workload_file_csv(proj, base_cfg, tmp_path):
    p = tmp_path / "wl.csv"
    p.write_text("arrival,isl,osl\n0,64,8\n100,128,16\n200,256,32\n")
    loaded = _load_workload_file(str(p))
    assert [x[1] for x in loaded] == [64, 128, 256]
    r = simulate_once(base_cfg, proj, rate_per_s=0.0, num_requests=0, seed=0, workload_file=str(p))
    assert r.num_requests == 3


def test_packing_summary_populated(proj, base_cfg):
    r = simulate_once(base_cfg, proj, rate_per_s=40.0, num_requests=200, seed=0)
    pk = r.packing
    assert pk["num_steps"] > 0
    assert 0 < pk["avg_batch_size"] <= base_cfg.request_config.max_concurrency
    assert pk["max_batch_size"] <= base_cfg.request_config.max_concurrency
    assert pk["avg_query_tokens"] > 0


def test_record_steps_shapes(proj, base_cfg):
    r = simulate_once(base_cfg, proj, rate_per_s=40.0, num_requests=60, seed=0, record_steps=True)
    assert r.steps is not None and len(r.steps) > 0
    s0 = r.steps[0]
    for key in ("step", "batch_size", "num_prefill_reqs", "num_decode_reqs",
                "total_query_tokens", "kv_tokens_in_use", "requests"):
        assert key in s0
    assert all({"request_id", "phase", "query_len", "kv_len"} <= set(rq) for rq in s0["requests"])


def test_run_des_sweep_returns_curve(proj, base_cfg):
    out = run_des(
        base_cfg, proj, arrival_model="poisson", rate_per_s=40.0,
        num_requests=150, seed=0, sweep=True,
    )
    assert "point" in out and "curve" in out
    assert out["max_sustainable_rate"] > 0.0
    curve = out["curve"]
    assert len(curve) >= 4
    rates = [r.offered_rate for r in curve]
    assert rates == sorted(rates)
    tps = [r.system_throughput_tps for r in curve]
    assert tps[-1] >= tps[0]


def test_deterministic_arrival_model_runs(proj, base_cfg):
    r = simulate_once(
        base_cfg, proj, rate_per_s=40.0, arrival_model="deterministic",
        num_requests=150, seed=0,
    )
    assert r.num_requests > 0 and r.e2e["p50"] > 0.0


def test_explicit_arrivals_override_generation(proj, base_cfg):
    arr = [50.0 * (i + 1) for i in range(80)]
    r = simulate_once(
        base_cfg, proj, rate_per_s=20.0, arrival_model="poisson",
        num_requests=80, seed=0, arrivals=arr,
    )
    assert r.num_requests > 0 and r.ttft["p50"] > 0.0
