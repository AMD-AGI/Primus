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
  * **request rate / arrivals** — open-loop Poisson / gamma-bursty /
    deterministic arrivals (or a caller-supplied / file-based workload) with
    real queueing and admission, and
  * **throughput-vs-latency curve** — sweep offered load to find the knee.

**Scheduler fidelity (vLLM V1 unified batch).**  The step scheduler mirrors the
``tools/serving_sim`` token-step model: each forward pass first advances every
already-running request (decodes + in-progress prefill chunks) under a shared
``max_num_batched_tokens`` budget (**Phase 1**), then admits new waiting
requests subject to the resident-sequence cap and a **full-ISL KV reservation**
against a flat ``kv_cache_tokens`` pool (**Phase 2**).  Chunked prefill emerges
naturally from the budget; a step can therefore mix prefill chunks and decodes.
Per-request prompt/output lengths are heterogeneous (uniform sampling or a
workload file), and per-step batch composition (query/KV shapes) is recorded.

**The time axis ("benchmark calibration inside a DES").**  On top of that
packing model, every simulated step's *duration* is drawn from the existing
:class:`InferencePerformanceProjector` cost kernel
(:meth:`decode_step_latency_ms` / :meth:`mixed_step_latency_ms`), which is itself
analytical *or* benchmark-calibrated — so the DES turns accurate per-pass costs
into accurate latency **distributions**.  Speculative decoding is modelled as a
per-request draft→verify→commit cycle so accept variance is preserved.

The steady-state model (``arrival_model == "closed"``) remains the validation
path: at low utilisation the DES means should agree with it.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from primus.core.projection.training_config import InferenceConfig

from .performance import InferencePerformanceProjector

# Context-length bucket (tokens) for memoising step-cost kernel calls. Decode
# step latency varies slowly with context, so bucketing keeps the number of
# (expensive) profiler evaluations small while the DES iterates many steps.
_CTX_BUCKET = 256
# Query-token bucket for prefill chunk sizes (heterogeneous per step); keeps the
# mixed-step cost cache small without materially changing the composed cost.
_TOK_BUCKET = 64


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
    """One request's live scheduling state (vLLM-V1 style token accounting)."""

    idx: int
    arrival_ms: float
    prompt_len: int
    output_len: int
    num_computed: int = 0        # prompt tokens already processed (prefill progress)
    generated: int = 0           # output tokens emitted so far
    prefill_done: bool = False
    status: str = "WAITING"      # WAITING | RUNNING | FINISHED
    first_token_ms: float = -1.0
    finish_ms: float = -1.0
    itls: List[float] = field(default_factory=list)

    @property
    def reserved_kv(self) -> int:
        # Full-ISL reservation: the whole sequence is reserved up front.
        return self.prompt_len + self.output_len

    @property
    def in_prefill(self) -> bool:
        return not self.prefill_done

    @property
    def kv_len(self) -> int:
        """Context length attended to right now."""
        return self.prompt_len + self.generated if self.prefill_done else self.num_computed


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
    # batch-composition / packing summary (serving_sim-style)
    packing: Dict[str, float] = field(default_factory=dict)
    # optional per-step records (only when record_steps=True)
    steps: Optional[List[dict]] = None


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

    @staticmethod
    def _tok(n: int) -> int:
        return max(_TOK_BUCKET, int(round(n / _TOK_BUCKET)) * _TOK_BUCKET)

    def decode_step_ms(self, batch: int, ctx: int) -> float:
        key = (batch, self._bucket(ctx))
        v = self._decode.get(key)
        if v is None:
            v = self._p.decode_step_latency_ms(batch, key[1], self._q)
            self._decode[key] = v
        return v

    def mixed_step_ms(self, num_decode: int, prefill_tokens: int, ctx: int, prefill_kv: int) -> float:
        key = (num_decode, self._tok(prefill_tokens), self._bucket(ctx), self._bucket(prefill_kv))
        v = self._mixed.get(key)
        if v is None:
            v = self._p.mixed_step_latency_ms(num_decode, key[1], key[2], key[3], self._q)
            self._mixed[key] = v
        return v


def _generate_arrivals(
    n: int, rate_per_s: float, model: str, rng: random.Random, burstiness: float = 1.0
) -> List[float]:
    """Arrival timestamps (ms) for ``n`` requests at ``rate_per_s`` req/s.

    ``deterministic`` → fixed spacing; ``poisson`` → gamma inter-arrivals with
    shape ``burstiness`` (1.0 = exponential / Poisson, <1 = burstier, >1 =
    smoother). ``rate_per_s`` non-positive / infinite ⇒ all arrive at t=0.
    """
    if rate_per_s <= 0 or math.isinf(rate_per_s):
        return [0.0] * n
    mean_dt_ms = 1000.0 / rate_per_s
    t = 0.0
    out: List[float] = []
    for _ in range(n):
        if model == "deterministic":
            dt = mean_dt_ms
        else:  # poisson / gamma-bursty
            shape = max(1e-3, burstiness)
            dt = rng.gammavariate(shape, mean_dt_ms / shape)  # mean = mean_dt_ms
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


def _sample_len(rng: random.Random, max_len: int, range_ratio: float) -> int:
    """Uniform length in ``[range_ratio*max_len, max_len]`` (inclusive)."""
    max_len = max(1, int(max_len))
    lo = max(1, int(max_len * max(0.0, min(1.0, range_ratio))))
    return rng.randint(lo, max_len)


def _load_workload_file(path: str) -> List[Tuple[float, int, int]]:
    """Load ``(arrival_ms, isl, osl)`` rows from JSON (list of dicts) or CSV.

    Keys/columns are case-insensitive: ``arrival`` (ms; aliases arrival_ms/time/
    step), ``isl`` (aliases input_len/prompt_len), ``osl`` (aliases output_len).
    Missing arrival ⇒ 0; missing lengths ⇒ 1.
    """
    if path.endswith(".json"):
        with open(path) as f:
            rows = json.load(f)
    else:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))

    def get(row: dict, *names: str, default: float = 0.0) -> float:
        lower = {str(k).lower(): v for k, v in row.items()}
        for nm in names:
            if nm in lower and lower[nm] not in (None, ""):
                return float(lower[nm])
        return default

    out: List[Tuple[float, int, int]] = []
    for row in rows:
        arrival = get(row, "arrival", "arrival_ms", "time", "step", default=0.0)
        isl = int(get(row, "isl", "input_len", "prompt_len", default=1.0))
        osl = int(get(row, "osl", "output_len", default=1.0))
        out.append((float(arrival), max(1, isl), max(1, osl)))
    return out


def _build_workload(
    n: int,
    arrivals: List[float],
    input_len: int,
    output_len: int,
    range_ratio: float,
    rng: random.Random,
) -> List[_Req]:
    """Construct ``n`` requests with per-request sampled lengths (homogeneous
    when ``range_ratio >= 1``)."""
    reqs: List[_Req] = []
    for i in range(n):
        isl = _sample_len(rng, input_len, range_ratio)
        osl = _sample_len(rng, output_len, range_ratio)
        reqs.append(_Req(idx=i, arrival_ms=arrivals[i], prompt_len=isl, output_len=osl))
    return reqs


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
    burstiness: float = 1.0,
    range_ratio: float = 1.0,
    kv_cache_tokens: int = 0,
    workload_file: Optional[str] = None,
    record_steps: bool = False,
) -> DESResult:
    """Run one single-engine DES at a fixed offered load.

    vLLM-V1 unified-batch scheduler: each forward pass advances running requests
    (Phase 1) then admits waiting ones under the resident-sequence cap + full-ISL
    KV reservation (Phase 2), all sharing a per-step ``max_num_batched_tokens``
    budget; each step's duration comes from the (optionally benchmark-calibrated)
    cost kernel.
    """
    req = inference_config.request_config
    input_len = max(1, req.input_seq_len)
    output_len = max(1, req.output_seq_len)
    max_running = max(1, req.resolved_max_concurrency())
    token_budget = int(req.max_num_batched_tokens or 0)      # 0 = unlimited
    long_prefill = int(req.chunked_prefill_size or 0)        # per-request chunk cap
    max_model_len = max(2, int(req.resolved_max_context_len()))
    spec_k = int(req.speculative_num_tokens or 0)
    accept = float(req.speculative_acceptance_rate or 0.0)
    q_len = (spec_k + 1) if spec_k > 0 else 1
    kv_pool = int(kv_cache_tokens or 0)                       # 0 = unlimited

    rng = random.Random(seed)

    # ---- workload (arrivals + per-request lengths) ----
    if workload_file:
        rows = _load_workload_file(workload_file)
        rows.sort(key=lambda r: r[0])
        pending = [
            _Req(idx=i, arrival_ms=a, prompt_len=isl, output_len=osl)
            for i, (a, isl, osl) in enumerate(rows)
        ]
    else:
        if arrivals is None:
            arrivals = _generate_arrivals(num_requests, rate_per_s, arrival_model, rng, burstiness)
        pending = _build_workload(len(arrivals), arrivals, input_len, output_len, range_ratio, rng)
    pending.sort(key=lambda r: (r.arrival_ms, r.idx))
    n = len(pending)

    kernel = _CostKernel(projector, q_len)
    next_arrival = 0
    waiting: List[_Req] = []
    running: List[_Req] = []
    done: List[_Req] = []
    kv_used = 0

    now = 0.0
    busy_ms = 0.0
    backlog: List[tuple] = []       # (time_ms, waiting depth) for saturation test
    step_records: List[dict] = [] if record_steps else []
    # packing accumulators
    pk_steps = pk_batch = pk_maxbatch = pk_prefill_reqs = pk_decode_reqs = 0
    pk_qtokens = 0
    pk_prefill_steps = 0
    pk_kv_peak = 0

    # Exact worst-case iteration bound (batch=1: every token its own step).
    total_work = 0
    for r in pending:
        pf = 1 if long_prefill <= 0 else max(1, math.ceil(r.prompt_len / long_prefill))
        total_work += pf + r.output_len
    max_steps = total_work + n + 16

    steps = 0
    while len(done) < n and steps < max_steps:
        steps += 1
        # 1) Ingest arrivals due by ``now`` into the FCFS waiting queue.
        while next_arrival < n and pending[next_arrival].arrival_ms <= now + 1e-9:
            waiting.append(pending[next_arrival])
            next_arrival += 1

        # 2) Nothing resident and nothing waiting → jump to the next arrival.
        if not running and not waiting:
            if next_arrival < n:
                now = pending[next_arrival].arrival_ms
                continue
            break

        # 3) Build the step: Phase 1 (running) then Phase 2 (admit), sharing the
        #    per-step token budget. ``scheduled`` = (req, q, is_prefill, kv_start).
        budget = token_budget if token_budget > 0 else math.inf
        scheduled: List[Tuple[_Req, int, bool, int]] = []

        def _schedule_tokens(r: _Req, budget: float) -> int:
            if r.prefill_done:
                # decode: compute q_len speculative tokens (>=1) if not finished
                if r.generated >= r.output_len:
                    return 0
                return int(min(q_len, budget)) if budget >= 1 else 0
            need = r.prompt_len - r.num_computed
            if need <= 0:
                return 0
            if long_prefill > 0:
                need = min(need, long_prefill)
            need = min(need, budget)
            need = min(need, max_model_len - 1 - r.num_computed)
            return int(max(need, 0))

        # Phase 1 — already-running requests.
        for r in running:
            q = _schedule_tokens(r, budget)
            if q <= 0:
                continue
            scheduled.append((r, q, r.in_prefill, r.kv_len))
            budget -= q

        # Phase 2 — admit new waiting requests (full-ISL KV reservation gate).
        while waiting and len(running) < max_running and budget >= 1:
            cand = waiting[0]
            if kv_pool > 0 and kv_used + cand.reserved_kv > kv_pool:
                break  # head-of-line block until KV frees up
            q = _schedule_tokens(cand, budget)
            if q <= 0:
                break
            waiting.pop(0)
            cand.status = "RUNNING"
            running.append(cand)
            kv_used += cand.reserved_kv
            scheduled.append((cand, q, True, cand.kv_len))
            budget -= q

        if not scheduled:
            # Budget/KV starved this step with nothing runnable; advance to the
            # next arrival if possible, else we are stuck (bound will trip).
            if next_arrival < n:
                now = pending[next_arrival].arrival_ms
                continue
            break

        # 4) Step duration from the cost kernel (composition → time).
        pref = [(r, q, kv) for (r, q, p, kv) in scheduled if p]
        dec = [(r, q, kv) for (r, q, p, kv) in scheduled if not p]
        num_decode = len(dec)
        prefill_q = sum(q for _, q, _ in pref)
        if prefill_q > 0:
            prefill_kv = int(sum(kv + q for _, q, kv in pref) / len(pref))
            decode_ctx = int(sum(kv for _, _, kv in dec) / len(dec)) if dec else input_len
            step_dt = kernel.mixed_step_ms(num_decode, prefill_q, decode_ctx, prefill_kv)
        else:
            decode_ctx = int(sum(kv for _, _, kv in dec) / len(dec)) if dec else input_len
            step_dt = kernel.decode_step_ms(num_decode, decode_ctx)

        # 5) Advance the clock.
        now += step_dt
        busy_ms += step_dt

        # 6) Apply the forward pass (prefill progress; decode commits w/ spec).
        for (r, q, is_prefill, _kv) in scheduled:
            if not r.prefill_done:
                r.num_computed += q
                if r.num_computed >= r.prompt_len:
                    r.prefill_done = True
                    r.first_token_ms = now
                    r.generated = 1              # last prefill chunk emits token 1
                    r.itls.append(step_dt)
                    if r.generated >= r.output_len:
                        r.status = "FINISHED"
                        r.finish_ms = now
            else:
                cap = r.output_len - r.generated
                if cap <= 0:
                    continue
                acc = _sample_accepted(rng, spec_k, accept, cap)
                r.generated += acc
                per_tok = step_dt / acc
                r.itls.extend([per_tok] * acc)
                if r.generated >= r.output_len:
                    r.status = "FINISHED"
                    r.finish_ms = now

        # 7) Retire finished requests, free their KV reservation.
        still: List[_Req] = []
        for r in running:
            if r.status == "FINISHED":
                kv_used -= r.reserved_kv
                done.append(r)
            else:
                still.append(r)
        running = still

        # 8) Bookkeeping: waiting depth + packing stats (+ optional records).
        backlog.append((now, len(waiting)))
        bs = len(scheduled)
        pk_steps += 1
        pk_batch += bs
        pk_maxbatch = max(pk_maxbatch, bs)
        pk_prefill_reqs += len(pref)
        pk_decode_reqs += num_decode
        pk_qtokens += prefill_q + num_decode * q_len
        pk_prefill_steps += 1 if pref else 0
        pk_kv_peak = max(pk_kv_peak, kv_used)
        if record_steps:
            step_records.append(
                {
                    "step": pk_steps,
                    "time_ms": round(now, 4),
                    "step_ms": round(step_dt, 4),
                    "batch_size": bs,
                    "num_prefill_reqs": len(pref),
                    "num_decode_reqs": num_decode,
                    "total_query_tokens": prefill_q + num_decode * q_len,
                    "kv_tokens_in_use": kv_used,
                    "requests": [
                        {
                            "request_id": r.idx,
                            "phase": "prefill" if is_prefill else "decode",
                            "query_len": q,
                            "kv_len": kv,
                        }
                        for (r, q, is_prefill, kv) in scheduled
                    ],
                }
            )

    # ---- aggregate latency metrics (drop warmup by completion order) ----
    done.sort(key=lambda r: r.finish_ms)
    drop = int(len(done) * max(0.0, min(0.9, warmup_frac)))
    sample = done[drop:] if len(done) - drop >= 8 else done

    ttft = [r.first_token_ms - r.arrival_ms for r in sample if r.first_token_ms >= 0]
    e2e = [r.finish_ms - r.arrival_ms for r in sample if r.finish_ms >= 0]
    tpot = [
        (r.finish_ms - r.first_token_ms) / max(1, r.generated - 1)
        for r in sample
        if r.finish_ms >= 0 and r.generated > 1
    ]
    itl_all: List[float] = []
    for r in sample:
        itl_all.extend(r.itls)

    makespan = max((r.finish_ms for r in done), default=0.0)
    total_out = sum(r.generated for r in done)
    achieved_rate = (len(done) * 1000.0 / makespan) if makespan > 0 else 0.0
    sys_tps = (total_out * 1000.0 / makespan) if makespan > 0 else 0.0
    utilization = (busy_ms / makespan) if makespan > 0 else 0.0

    # Saturation: the waiting queue diverges (grows ~linearly) while arrivals are
    # still coming, rather than staying stationary. Restricted to the arrival
    # window because a finite run always drains to empty afterwards. N-invariant,
    # unlike achieved-vs-offered rate (drain-tail biased) or utilisation (~1
    # whenever ≥1 request is resident, common for memory-bound decode).
    saturated = False
    last_arrival = pending[-1].arrival_ms if pending else 0.0
    win = [(t, float(d)) for (t, d) in backlog if t <= last_arrival]
    if rate_per_s > 0 and len(win) >= 30:
        cut = int(0.2 * len(win))
        bs_win = win[cut:]
        ts = [t for t, _ in bs_win]
        ds = [d for _, d in bs_win]
        span = ts[-1] - ts[0]
        if span > 0:
            growth = _slope(ts, ds) * span
            saturated = growth > max_running and max(ds) > max_running
    if rate_per_s > 0 and achieved_rate < 0.5 * rate_per_s:
        saturated = True

    def dist(xs: List[float]) -> Dict[str, float]:
        return {
            "mean": (sum(xs) / len(xs)) if xs else 0.0,
            "p50": _pct(xs, 0.50),
            "p90": _pct(xs, 0.90),
            "p99": _pct(xs, 0.99),
        }

    packing = {
        "num_steps": float(pk_steps),
        "avg_batch_size": (pk_batch / pk_steps) if pk_steps else 0.0,
        "max_batch_size": float(pk_maxbatch),
        "avg_prefill_reqs": (pk_prefill_reqs / pk_steps) if pk_steps else 0.0,
        "avg_decode_reqs": (pk_decode_reqs / pk_steps) if pk_steps else 0.0,
        "avg_query_tokens": (pk_qtokens / pk_steps) if pk_steps else 0.0,
        "prefill_step_fraction": (pk_prefill_steps / pk_steps) if pk_steps else 0.0,
        "kv_peak_tokens": float(pk_kv_peak),
        "kv_utilization": (pk_kv_peak / kv_pool) if kv_pool > 0 else 0.0,
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
        packing=packing,
        steps=step_records if record_steps else None,
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
    burstiness: float = 1.0,
    range_ratio: float = 1.0,
    kv_cache_tokens: int = 0,
    workload_file: Optional[str] = None,
    record_steps: bool = False,
) -> Dict[str, object]:
    """Run the DES at the configured load and (optionally) a load sweep.

    Returns ``{"point": DESResult, "curve": [DESResult, ...],
    "max_sustainable_rate": mu}``. The sweep derives the engine's max-sustainable
    rate ``mu`` from the steady-state projection and samples fractions of it to
    trace the throughput-vs-latency knee.
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
        burstiness=burstiness,
        range_ratio=range_ratio,
        kv_cache_tokens=kv_cache_tokens,
        workload_file=workload_file,
        record_steps=record_steps,
    )

    if sweep and not workload_file:
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
                        burstiness=burstiness,
                        range_ratio=range_ratio,
                        kv_cache_tokens=kv_cache_tokens,
                    )
                )
        out["curve"] = curve
        out["max_sustainable_rate"] = mu
    return out
