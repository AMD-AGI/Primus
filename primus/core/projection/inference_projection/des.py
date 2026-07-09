###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Lightweight single-engine discrete-event simulation (DES) for serving.

This is Phase 3 of the inference roadmap: an opt-in event loop that turns the
analytical steady-state projector into a *time-driven* simulation.  It answers
the questions the closed-form model structurally cannot:

  * **percentiles** — p50/p90/p99 of TTFT, TPOT/ITL, and end-to-end latency,
  * **request rate / arrivals** — open-loop Poisson / deterministic arrivals
    (or a caller-supplied explicit arrival list) with real queueing and
    admission, and
  * **throughput-vs-latency curve** — sweep offered load to find the knee.

The key design point ("benchmark calibration inside a DES"): every simulated
step's *duration* is drawn from the existing
:class:`InferencePerformanceProjector` cost kernel
(:meth:`decode_step_latency_ms` / :meth:`mixed_step_latency_ms`), which is
itself analytical *or* benchmark-calibrated.  The DES only adds the time axis
(arrivals, admission under ``max_num_seqs`` / ``max_num_batched_tokens``,
continuous-batching step composition, per-request speculative accept variance);
it does not re-estimate per-pass cost.

The steady-state model (``arrival_model == "closed"``) remains the validation
path: at low utilisation the DES means should agree with it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from primus.core.projection.training_config import InferenceConfig

from .performance import InferencePerformanceProjector

# Context-length bucket (tokens) for memoising step-cost kernel calls. Decode
# step latency varies slowly with context, so bucketing keeps the number of
# (expensive) profiler evaluations small while the DES iterates many steps.
_CTX_BUCKET = 256


def _slope(xs: List[float], ys: List[float]) -> float:
    """Least-squares slope dy/dx; 0.0 for degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / sxx


def _pct(xs: List[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 1]); 0.0 for empty input."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


@dataclass
class _Req:
    idx: int
    arrival_ms: float
    input_len: int
    output_len: int
    prefill_remaining: int = 0
    generated: int = 0
    prefill_done: bool = False
    first_token_ms: float = -1.0
    finish_ms: float = -1.0
    itls: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.prefill_remaining = self.input_len


@dataclass
class DESResult:
    arrival_model: str
    offered_rate: float            # req/s requested
    achieved_rate: float           # req/s completed over makespan
    utilization: float             # busy time / makespan
    num_requests: int
    makespan_ms: float
    system_throughput_tps: float   # output tokens/s
    saturated: bool
    # latency distributions (ms)
    ttft: Dict[str, float] = field(default_factory=dict)
    tpot: Dict[str, float] = field(default_factory=dict)
    itl: Dict[str, float] = field(default_factory=dict)
    e2e: Dict[str, float] = field(default_factory=dict)


class _CostKernel:
    """Memoised view over the projector's step-cost methods."""

    def __init__(self, projector: InferencePerformanceProjector, q_len: int):
        self._p = projector
        self._q = q_len
        self._decode: Dict[tuple, float] = {}
        self._mixed: Dict[tuple, float] = {}

    @staticmethod
    def _bucket(ctx: int) -> int:
        return max(_CTX_BUCKET, int(round(ctx / _CTX_BUCKET)) * _CTX_BUCKET)

    def decode_step_ms(self, batch: int, ctx: int) -> float:
        key = (batch, self._bucket(ctx))
        v = self._decode.get(key)
        if v is None:
            v = self._p.decode_step_latency_ms(batch, key[1], self._q)
            self._decode[key] = v
        return v

    def mixed_step_ms(self, num_decode: int, chunk_tokens: int, ctx: int, prefill_kv: int) -> float:
        key = (num_decode, int(chunk_tokens), self._bucket(ctx), self._bucket(prefill_kv))
        v = self._mixed.get(key)
        if v is None:
            v = self._p.mixed_step_latency_ms(num_decode, chunk_tokens, key[2], key[3], self._q)
            self._mixed[key] = v
        return v


def _generate_arrivals(
    n: int, rate_per_s: float, model: str, rng: random.Random
) -> List[float]:
    """Arrival timestamps (ms) for ``n`` requests at ``rate_per_s`` req/s."""
    if rate_per_s <= 0:
        return [0.0] * n
    mean_dt_ms = 1000.0 / rate_per_s
    t = 0.0
    out: List[float] = []
    for _ in range(n):
        if model == "deterministic":
            dt = mean_dt_ms
        else:  # poisson
            dt = rng.expovariate(1.0 / mean_dt_ms)
        t += dt
        out.append(t)
    return out


def _sample_accepted(rng: random.Random, k: int, accept: float, cap: int) -> int:
    """Tokens committed in one verify step under speculative decoding.

    Bonus token (always) + a run of accepted drafts until the first rejection
    (each accepted independently w.p. ``accept``), capped at ``k+1`` and at the
    remaining ``cap`` tokens. Reproduces the per-step *variable-commit* variance
    that the analytical scalar-expectation model averages away.
    """
    if k <= 0:
        return min(1, cap) if cap > 0 else 1
    accepted = 1
    if accept >= 1.0:
        accepted = k + 1
    else:
        for _ in range(k):
            if rng.random() < accept:
                accepted += 1
            else:
                break
    return max(1, min(accepted, cap if cap > 0 else accepted))


def simulate_once(
    inference_config: InferenceConfig,
    projector: InferencePerformanceProjector,
    *,
    rate_per_s: float,
    arrival_model: str = "poisson",
    num_requests: int = 400,
    warmup_frac: float = 0.1,
    seed: int = 0,
    arrivals: Optional[List[float]] = None,
) -> DESResult:
    """Run one single-engine DES at a fixed offered load.

    Continuous-batching scheduler (vLLM-ish): arrivals queue, are admitted up to
    ``max_concurrency`` resident sequences, prefill is chunked under the
    per-step token budget, and each step's duration comes from the cost kernel.
    """
    req = inference_config.request_config
    input_len = max(1, req.input_seq_len)
    output_len = max(1, req.output_seq_len)
    max_running = max(1, req.resolved_max_concurrency())
    token_budget = int(req.max_num_batched_tokens or 0)
    chunk_size = int(req.chunked_prefill_size or 0)
    spec_k = int(req.speculative_num_tokens or 0)
    accept = float(req.speculative_acceptance_rate or 0.0)
    q_len = (spec_k + 1) if spec_k > 0 else 1

    rng = random.Random(seed)
    if arrivals is None:
        arrivals = _generate_arrivals(num_requests, rate_per_s, arrival_model, rng)
    n = len(arrivals)
    kernel = _CostKernel(projector, q_len)

    pending = [
        _Req(idx=i, arrival_ms=arrivals[i], input_len=input_len, output_len=output_len)
        for i in range(n)
    ]
    next_arrival = 0
    waiting: List[_Req] = []
    active: List[_Req] = []  # admitted: prefilling or decoding
    done: List[_Req] = []

    now = 0.0
    busy_ms = 0.0
    backlog: List[tuple] = []  # (time_ms, waiting-queue depth) for saturation test
    # Exact worst-case iteration bound: batch=1 (every token its own step) gives
    # ``prefill_steps + output_len`` step-iterations per request, plus at most
    # one idle-jump ``continue`` per request. Never trips on a healthy run.
    prefill_steps = 1 if chunk_size <= 0 else max(1, math.ceil(input_len / chunk_size))
    max_steps = n * (output_len + prefill_steps + 1) + 16

    steps = 0
    while len(done) < n and steps < max_steps:
        steps += 1
        # 1) Move all requests that have arrived by ``now`` into the waiting q.
        while next_arrival < n and pending[next_arrival].arrival_ms <= now + 1e-9:
            waiting.append(pending[next_arrival])
            next_arrival += 1

        # 2) If nothing to do, jump the clock to the next arrival.
        if not active and not waiting:
            if next_arrival < n:
                now = pending[next_arrival].arrival_ms
                continue
            break

        # 3) Admit waiting requests up to the resident-sequence cap.
        while len(active) < max_running and waiting:
            active.append(waiting.pop(0))
        backlog.append((now, len(waiting)))

        # 4) Compose the step: decodes + (at most) one prefill chunk (FCFS).
        decodes = [r for r in active if r.prefill_done]
        prefiller = next((r for r in active if not r.prefill_done), None)

        decode_tokens = len(decodes) * q_len
        if decodes:
            mean_ctx = sum(r.input_len + r.generated for r in decodes) / len(decodes)
        else:
            mean_ctx = input_len

        if prefiller is not None:
            remaining = prefiller.prefill_remaining
            want = remaining if chunk_size <= 0 else min(chunk_size, remaining)
            if token_budget > 0:
                want = min(want, max(1, token_budget - decode_tokens))
            processed = prefiller.input_len - prefiller.prefill_remaining
            prefill_kv = processed + want
            step_dt = kernel.mixed_step_ms(len(decodes), want, mean_ctx, prefill_kv)
        else:
            step_dt = kernel.decode_step_ms(len(decodes), int(mean_ctx))
            want = 0

        # 5) Advance the clock.
        now += step_dt
        busy_ms += step_dt

        # 6) Apply prefill progress.
        if prefiller is not None:
            prefiller.prefill_remaining -= want
            if prefiller.prefill_remaining <= 0:
                prefiller.prefill_done = True
                prefiller.first_token_ms = now
                prefiller.generated = 1  # prefill emits the first token
                prefiller.itls.append(step_dt)
                if prefiller.generated >= prefiller.output_len:
                    prefiller.finish_ms = now
                    active.remove(prefiller)
                    done.append(prefiller)

        # 7) Apply decode progress (speculative variable commits).
        for r in list(decodes):
            cap = r.output_len - r.generated
            if cap <= 0:
                continue
            acc = _sample_accepted(rng, spec_k, accept, cap)
            r.generated += acc
            per_tok = step_dt / acc
            r.itls.extend([per_tok] * acc)
            if r.generated >= r.output_len:
                r.finish_ms = now
                active.remove(r)
                done.append(r)

    # ---- aggregate metrics (drop warmup by completion order) ----
    done.sort(key=lambda r: r.finish_ms)
    drop = int(len(done) * max(0.0, min(0.9, warmup_frac)))
    sample = done[drop:] if len(done) - drop >= 8 else done

    ttft = [r.first_token_ms - r.arrival_ms for r in sample if r.first_token_ms >= 0]
    e2e = [r.finish_ms - r.arrival_ms for r in sample if r.finish_ms >= 0]
    tpot = [
        (r.finish_ms - r.first_token_ms) / max(1, r.output_len - 1)
        for r in sample
        if r.finish_ms >= 0 and r.output_len > 1
    ]
    itl_all: List[float] = []
    for r in sample:
        itl_all.extend(r.itls)

    makespan = max((r.finish_ms for r in done), default=0.0)
    total_out = sum(r.generated for r in done)
    achieved_rate = (len(done) * 1000.0 / makespan) if makespan > 0 else 0.0
    sys_tps = (total_out * 1000.0 / makespan) if makespan > 0 else 0.0
    utilization = (busy_ms / makespan) if makespan > 0 else 0.0
    # Saturation: the *waiting* queue diverges (grows ~linearly with time) while
    # arrivals are still coming, rather than staying stationary. We restrict the
    # slope test to the arrival window (before the last arrival) because after
    # arrivals stop a finite run always drains to empty, masking the growth. This
    # is N-invariant — unlike the achieved-vs-offered rate, whose finite-run
    # drain tail makes a stable run look ~5-15% short — and unlike utilisation,
    # which sits near 1 whenever ≥1 request is resident (common even at low load
    # for memory-bound decode).
    saturated = False
    last_arrival = arrivals[-1] if arrivals else 0.0
    win = [(t, float(d)) for (t, d) in backlog if t <= last_arrival]
    if rate_per_s > 0 and len(win) >= 30:
        cut = int(0.2 * len(win))
        bs = win[cut:]
        ts = [t for t, _ in bs]
        ds = [d for _, d in bs]
        span = ts[-1] - ts[0]
        if span > 0:
            growth = _slope(ts, ds) * span  # queue tokens gained over the window
            saturated = growth > max_running and max(ds) > max_running
    # Hard fallback: the engine served far less than offered even after drain.
    if rate_per_s > 0 and achieved_rate < 0.5 * rate_per_s:
        saturated = True

    def dist(xs: List[float]) -> Dict[str, float]:
        return {
            "mean": (sum(xs) / len(xs)) if xs else 0.0,
            "p50": _pct(xs, 0.50),
            "p90": _pct(xs, 0.90),
            "p99": _pct(xs, 0.99),
        }

    return DESResult(
        arrival_model=arrival_model,
        offered_rate=rate_per_s,
        achieved_rate=achieved_rate,
        utilization=utilization,
        num_requests=len(done),
        makespan_ms=makespan,
        system_throughput_tps=sys_tps,
        saturated=saturated,
        ttft=dist(ttft),
        tpot=dist(tpot),
        itl=dist(itl_all),
        e2e=dist(e2e),
    )


def run_des(
    inference_config: InferenceConfig,
    projector: InferencePerformanceProjector,
    *,
    arrival_model: str,
    rate_per_s: float,
    num_requests: int = 400,
    seed: int = 0,
    warmup_frac: float = 0.1,
    sweep: bool = False,
) -> Dict[str, object]:
    """Run the DES at the configured load and (optionally) a load sweep.

    Returns a dict: ``{"point": DESResult, "curve": [DESResult, ...]}``.
    The sweep derives the engine's max-sustainable rate ``mu`` from the
    steady-state projection and samples fractions of it to trace the
    throughput-vs-latency knee.
    """
    out: Dict[str, object] = {}
    out["point"] = simulate_once(
        inference_config,
        projector,
        rate_per_s=rate_per_s,
        arrival_model=arrival_model,
        num_requests=num_requests,
        seed=seed,
        warmup_frac=warmup_frac,
    )

    if sweep:
        # mu = steady-state max sustainable request rate (decode-bound).
        steady = projector.project()
        osl = max(1, inference_config.request_config.output_seq_len)
        mu = (steady.decode_throughput_tps / osl) if steady.decode_throughput_tps > 0 else 0.0
        curve: List[DESResult] = []
        if mu > 0:
            sweep_n = min(num_requests, 300)
            fracs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
            rates = sorted({round(f * mu, 4) for f in fracs} | {round(rate_per_s, 4)})
            for lam in rates:
                if lam <= 0:
                    continue
                curve.append(
                    simulate_once(
                        inference_config,
                        projector,
                        rate_per_s=lam,
                        arrival_model=arrival_model,
                        num_requests=sweep_n,
                        seed=seed,
                        warmup_frac=warmup_frac,
                    )
                )
        out["curve"] = curve
        out["max_sustainable_rate"] = mu
    return out
