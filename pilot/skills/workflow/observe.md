# Observe

**Status**: v1 (single-node, RunSnapshot-based)
**Stage**: `OPTIMIZE_LOOP.OBSERVE`
**Consumed by**: Observe Stage Worker, `pilot.tools.tune_single` after every `_run_one`
**Tool boundary**: `pilot.tools.observe.{snapshot, watch, compare_loss}`
**Anchors**: `diagnose.md` (downstream consumer)
**Schema**: `pilot/schemas/run_snapshot.schema.json`

A `RunSnapshot` is a structured point-in-time view of a training job's `train.log`. It is the **primary terminal artifact** that DIAGNOSE consumes; the engine also reads it during EXECUTE for early-stop decisions.

---

## 1. Top-level shape

```yaml
schema_version: "1.0"
run_id:         <str>
plan_id:        <str | null>
collected_at:   <iso8601>
status:         completed | early_stopped | oom | failed | killed | hung | unknown
progress:                                # how far into the requested iters we got
  current_iter:    <int>
  total_iters:     <int>
  pct:             <0..100>
  iters_per_min:   <float>
  last_iter_at:    <iso8601 | null>
  silent_for_s:    <float>                # since last iter line; vs hang threshold
metrics:
  loss_finite:     bool
  latest:                                  # most recent observed
    iter_time_ms:  <float>
    tflops:        <float>
    loss:          <float>
  history:                                 # last _HISTORY_CAP = 50 entries per series
    iter_time_ms:  [<float>...]
    tflops:        [<float>...]
    loss:          [<float>...]
symptoms:                                  # boolean classifiers, see §3
  hang_suspected:    bool
  oom_detected:      bool
  nccl_error:        bool
  cuda_error:        bool
  python_error:      bool
  loss_nan_or_inf:   bool
  hang_threshold_s:  <float>
  evidence:          [{kind, line_excerpt, ...}]   # up to _EVIDENCE_CAP = 16
warnings:          [<str>...]
```

---

## 2. Metric semantics

| Field | Unit | Source | Sampling window |
|---|---|---|---|
| `iter_time_ms` | milliseconds per iteration | Megatron `iteration <i>/<N>` log line | per-iter line |
| `tflops` | TFLOPs/s/GPU | same line, `TFLOPs/s/GPU` field | per-iter line |
| `loss` | scalar; positive for normal training | same line, `loss` field | per-iter line |
| `iters_per_min` | float | derived: `60 / median(iter_time_s)` over `history.iter_time_ms` | rolling, from `history` |

The default "steady-state" median used by `tune_single.summarize_snapshot` excludes nothing; warmup exclusion happens in `trace_analyze` (per `trace_analysis.md`), not in Observe.

---

## 3. Symptom classifiers

Each symptom is a boolean derived from regex matches on the last `_DEFAULT_TAIL_BYTES = 4 MiB` of `train.log`:

| Symptom | Trigger regex (illustrative; canonical in `observe.py`) | Downstream action |
|---|---|---|
| `oom_detected` | `out of memory`, `CUDA out of memory`, `HIP out of memory` | DIAGNOSE → `MEMORY_BOUND`; Settle marks plan `dead` |
| `hang_suspected` | `silent_for_s > hang_threshold_s` (default **120 s**) on a live process | DIAGNOSE → `HANG`; FailureReport routes to `PREFLIGHT` (env_probe) |
| `nccl_error` | `NCCL ERROR`, `ncclInternalError`, `NCCL operation timed out` | DIAGNOSE → `HANG` (treated as comm failure); not `COMM_BOUND` (those are perf, not error) |
| `cuda_error` | `CUDA error: ...`, `HIP error: ...`, `Misaligned address` | DIAGNOSE → `INVALID_CONFIG`; Settle marks plan `dead` |
| `python_error` | a `Traceback (most recent call last):` followed by an exception line | DIAGNOSE → `INVALID_CONFIG`; Settle marks plan `dead` |
| `loss_nan_or_inf` | per-line scan: `loss=nan` or `loss=inf` | DIAGNOSE → `NUMERICAL`; ABORT + escalate |

All symptoms write up to `_EVIDENCE_CAP = 16` evidence items into `symptoms.evidence` for later attribution.

---

## 4. Snapshot validity

A snapshot is **valid for promotion** in `settle.md` if **all** of:

1. `status == completed`
2. `metrics.loss_finite == True`
3. `metrics.history.iter_time_ms` has at least 8 finite entries (post-warmup steady state).
4. No "hard" symptom: `oom_detected`, `hang_suspected`, `cuda_error`, `python_error`, `loss_nan_or_inf` are all `false`.

A snapshot may still be **valid for diagnosis** even if it's not promotable (e.g. an OOM is still a valid input to DIAGNOSE; it just gets the `dead` verdict instead of a `tps` score).

---

## 5. observe.compare_loss (CORRECTNESS gate)

Single-node v1: short-window numeric health gate. Per `correctness.md`, this is not yet a token-aligned T0/T1/T2 reference curve. Implementation reads:

| Input | What it expects |
|---|---|
| `snapshot.metrics.history.loss` | tail of finite losses |
| `reference.expected_final_loss` OR `reference.expected_loss_at_iter[<iter>]` | scalar / dict for comparison |
| `tolerance_pct` | default **1.0** % per `correctness.md` |

Returns `{pass: bool, delta_pct: float, reason: str}`. A failure routes the `NUMERICAL` kind to ABORT.

---

## 6. Sampling windows for Watch mode

`observe watch --interval-s 5` polls every 5 s by default; tunable. Each poll produces a snapshot record on disk under `pilot/state/runs/<run_id>/snapshots/<ts>.yaml`. `tune_single` only reads the **terminal** snapshot for DIAGNOSE; Watch mode is for human inspection / early-stop dashboards.

---

## 7. Cross-references

- DIAGNOSE consumes this: `diagnose.md` §2 (Required input + Optional inputs).
- Trace-level metrics (overlap_ratio / bubble_ratio / kernel breakdowns) come from `profile.md` + `trace_analysis.md`, NOT this snapshot. Observe is log-driven; trace_analyze is profile-driven; both feed Diagnose.
- Schema: `pilot/schemas/run_snapshot.schema.json`.
- README design source: §8.3.
