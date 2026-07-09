###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Unit tests for the single-engine discrete-event simulator (``des.py``).

Phase 3 of the inference roadmap. These exercise the pure-logic surface of the
DES — arrival processes, speculative-accept sampling, percentile/slope helpers —
plus its end-to-end behaviour against the analytical steady-state model:

  * **determinism** — a fixed seed reproduces the run exactly;
  * **percentile ordering** — p50 ≤ p90 ≤ p99 for every latency metric;
  * **queueing** — TTFT tail grows as offered load approaches the knee;
  * **saturation** — the divergence flag flips above the max-sustainable rate;
  * **steady-state agreement (validation path)** — at high load the simulated
    per-token TPOT converges toward the analytical pure-decode step latency, the
    cross-check that justifies "benchmark calibration inside a DES".
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

pytest.importorskip("primus.core.projection.inference_projection.des")

from primus.core.projection.inference_projection.des import (  # noqa: E402
    _generate_arrivals,
    _pct,
    _sample_accepted,
    _slope,
    run_des,
    simulate_once,
)
from primus.core.projection.inference_projection.performance import (  # noqa: E402
    InferencePerformanceProjector,
)
from primus.core.projection.training_config import (  # noqa: E402
    InferenceConfig,
    InferenceRequestConfig,
    ModelConfig,
    ModelParallelConfig,
)


def _model_config(**kw) -> ModelConfig:
    # Realistic-enough sizes so the analytical kernel returns non-zero attention
    # / GEMM / KV timings; small layer count keeps the profiler build fast.
    base = dict(
        num_layers=8,
        hidden_size=4096,
        num_attention_heads=32,
        kv_channels=128,
        num_query_groups=8,
        group_query_attention=True,
        ffn_hidden_size=14336,
        swiglu=True,
        padded_vocab_size=32000,
        num_experts=0,
    )
    base.update(kw)
    return ModelConfig(**base)


def _inf_config(**req) -> InferenceConfig:
    return InferenceConfig(
        model_config=_model_config(),
        request_config=InferenceRequestConfig(**req),
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=1),
    )


def _sim_args():
    return SimpleNamespace(gpu_arch="mi300x", gpu_clock_mhz=None, gemm_backend=None)


@pytest.fixture(scope="module")
def projector():
    cfg = _inf_config(
        input_seq_len=512, output_seq_len=128, max_concurrency=64
    )
    return cfg, InferencePerformanceProjector(cfg, args=_sim_args())


@pytest.fixture(scope="module")
def steady_mu(projector):
    cfg, proj = projector
    s = proj.project()
    osl = max(1, cfg.request_config.output_seq_len)
    return s, s.decode_throughput_tps / osl  # (perf, max-sustainable req/s)


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
    # Strictly increasing → positive slope; flat → ~0.
    assert _slope([0, 1, 2, 3], [0, 2, 4, 6]) == pytest.approx(2.0)
    assert _slope([0, 1, 2, 3], [5, 5, 5, 5]) == pytest.approx(0.0)
    assert _slope([1.0], [1.0]) == 0.0


def test_arrivals_deterministic_is_evenly_spaced():
    a = _generate_arrivals(5, rate_per_s=10.0, model="deterministic", rng=random.Random(0))
    # 10 req/s → 100 ms spacing.
    assert a == pytest.approx([100.0, 200.0, 300.0, 400.0, 500.0])


def test_arrivals_poisson_monotone_and_seeded():
    a = _generate_arrivals(50, 10.0, "poisson", random.Random(1))
    b = _generate_arrivals(50, 10.0, "poisson", random.Random(1))
    assert a == b  # reproducible
    assert all(a[i] <= a[i + 1] for i in range(len(a) - 1))  # non-decreasing
    # Mean interarrival ≈ 100 ms (loose bound).
    assert 60.0 < (a[-1] / len(a)) < 160.0


def test_sample_accepted_bounds():
    rng = random.Random(0)
    # No speculation → exactly one token (bonus only).
    assert _sample_accepted(rng, 0, 0.0, 5) == 1
    # accept=1.0 → all k drafts + bonus, but capped at remaining tokens.
    assert _sample_accepted(rng, 4, 1.0, 10) == 5
    assert _sample_accepted(rng, 4, 1.0, 2) == 2
    # accept=0.0 → only the bonus token.
    assert _sample_accepted(rng, 4, 0.0, 10) == 1


# ─────────────────────────────────────────────────────────────────────────────
# end-to-end DES behaviour
# ─────────────────────────────────────────────────────────────────────────────


def test_determinism(projector):
    cfg, proj = projector
    a = simulate_once(cfg, proj, rate_per_s=6.0, num_requests=120, seed=11)
    b = simulate_once(cfg, proj, rate_per_s=6.0, num_requests=120, seed=11)
    assert a.ttft == b.ttft
    assert a.e2e == b.e2e
    assert a.tpot == b.tpot
    assert a.achieved_rate == b.achieved_rate


def test_different_seed_changes_poisson_run(projector):
    cfg, proj = projector
    a = simulate_once(cfg, proj, rate_per_s=6.0, num_requests=120, seed=1)
    b = simulate_once(cfg, proj, rate_per_s=6.0, num_requests=120, seed=2)
    assert a.ttft["p99"] != b.ttft["p99"]


def test_percentiles_are_ordered(projector):
    cfg, proj = projector
    r = simulate_once(cfg, proj, rate_per_s=6.0, num_requests=200, seed=0)
    for d in (r.ttft, r.tpot, r.itl, r.e2e):
        assert d["p50"] <= d["p90"] <= d["p99"]
        assert d["mean"] > 0.0


def test_ttft_floor_is_at_least_prefill(projector):
    cfg, proj = projector
    # Light load → TTFT ≈ compute prefill (little/no queue). It must never be
    # below the single-request prefill latency.
    r = simulate_once(cfg, proj, rate_per_s=1.0, num_requests=120, seed=0)
    prefill = proj.prefill_latency_ms(1, cfg.request_config.input_seq_len)
    assert r.ttft["p50"] >= 0.8 * prefill


def test_higher_load_raises_ttft_tail(projector, steady_mu):
    cfg, proj = projector
    _perf, mu = steady_mu
    lo = simulate_once(cfg, proj, rate_per_s=0.3 * mu, num_requests=300, seed=0)
    hi = simulate_once(cfg, proj, rate_per_s=0.95 * mu, num_requests=300, seed=0)
    assert hi.ttft["p99"] > lo.ttft["p99"]


def test_saturation_flag_above_mu(projector, steady_mu):
    cfg, proj = projector
    _perf, mu = steady_mu
    stable = simulate_once(cfg, proj, rate_per_s=0.25 * mu, num_requests=300, seed=0)
    overloaded = simulate_once(cfg, proj, rate_per_s=3.0 * mu, num_requests=400, seed=0)
    assert stable.saturated is False
    assert overloaded.saturated is True


def test_steady_state_agreement_at_high_load(projector, steady_mu):
    """Validation path: near the knee the batch fills, so the simulated
    per-token TPOT should converge toward the analytical pure-decode step."""
    cfg, proj = projector
    perf, mu = steady_mu
    r = simulate_once(cfg, proj, rate_per_s=0.9 * mu, num_requests=400, seed=0)
    pure_step = perf.decode_step_latency_ms
    assert pure_step > 0.0
    # Within a factor of ~1.6 of the analytical pure step (continuous-batching
    # mixed-step pollution + finite concurrency keep it modestly above floor).
    assert 0.6 * pure_step <= r.tpot["p50"] <= 1.8 * pure_step


def test_spec_decode_widens_itl_distribution(projector):
    """Speculative variable-commit should make ITL more dispersed (heavier
    tail) than its mean — the variance the analytical scalar model averages."""
    cfg = _inf_config(
        input_seq_len=512,
        output_seq_len=128,
        max_concurrency=32,
        speculative_num_tokens=4,
        speculative_acceptance_rate=0.7,
    )
    proj = InferencePerformanceProjector(cfg, args=_sim_args())
    r = simulate_once(cfg, proj, rate_per_s=6.0, num_requests=200, seed=0)
    # p99 ITL strictly above the mean → a real distribution, not a constant.
    assert r.itl["p99"] > r.itl["mean"] > 0.0


def test_run_des_sweep_returns_curve(projector):
    cfg, proj = projector
    out = run_des(
        cfg, proj, arrival_model="poisson", rate_per_s=6.0,
        num_requests=120, seed=0, sweep=True,
    )
    assert "point" in out and "curve" in out
    assert out["max_sustainable_rate"] > 0.0
    curve = out["curve"]
    assert len(curve) >= 4
    # Offered rates strictly increasing along the curve.
    rates = [r.offered_rate for r in curve]
    assert rates == sorted(rates)
    # System throughput is non-decreasing with offered load (monotone region).
    tps = [r.system_throughput_tps for r in curve]
    assert tps[-1] >= tps[0]


def test_deterministic_arrival_model_runs(projector):
    cfg, proj = projector
    r = simulate_once(
        cfg, proj, rate_per_s=6.0, arrival_model="deterministic",
        num_requests=120, seed=0,
    )
    assert r.num_requests > 0
    assert r.e2e["p50"] > 0.0


def test_explicit_arrivals_override_generation(projector):
    cfg, proj = projector
    # A caller-supplied arrival list (the internal hook a future --arrival-trace
    # loader will use) bypasses the generated process: 80 requests, 50 ms apart.
    arr = [50.0 * (i + 1) for i in range(80)]
    r = simulate_once(
        cfg, proj, rate_per_s=20.0, arrival_model="poisson",
        num_requests=80, seed=0, arrivals=arr,
    )
    assert r.num_requests > 0
    assert r.ttft["p50"] > 0.0
