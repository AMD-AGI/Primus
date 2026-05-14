# Correctness / Correctness-Lite (Numerical Health Gate)

**Status**: v1 (single-rank, log-derived)
**Stage**: `OPTIMIZE_LOOP.CORRECTNESS_LITE` (every round) and `CORRECTNESS`
          (full reference-curve gate, when called explicitly)
**Tool boundary**: `pilot.tools.observe.compare_loss`
**Authoritative schema**: the return shape below

This file is the **single source of truth** for the per-round numerical
gate that the OPTIMIZE_LOOP runs before SETTLE. Promoting a candidate
that diverges numerically is worse than not promoting at all — even a
+30% TFLOPS win is useless if the loss curve is broken.

---

## 1. Purpose

Given a trial run's `RunSnapshot` and a reference (a saved snapshot, a
scalar reference loss, or an explicit reference window), decide whether
the trial is **numerically sound enough to be a SETTLE candidate**.

The gate is intentionally conservative:

- It is **not** a convergence test. Convergence is whatever your
  upstream training criterion says; the gate only checks short-window
  health.
- It is **not** a token-aligned T0/T1/T2 reference-curve compare. That's
  the heavier `CORRECTNESS` gate which is invoked separately when a
  curve is available; the lite path is what runs every round.
- It is single-rank: rank 0's loss history is all we read.

---

## 2. Inputs

| Input | Source | Required |
|---|---|---|
| trial `RunSnapshot` | `state/runs/<run_id>/snapshots/<ts>.yaml` (latest) | yes (the gate fetches automatically if `--save` was used at observe time) |
| reference | one of: scalar `loss`, snapshot, explicit window list | yes |
| `window` | trailing finite-loss window length (default 5) | optional |
| `max_delta_pct` | gate threshold (default 25%) | optional |

The reference is resolved in this priority order:

1. `reference.window` — explicit `[float, ...]` (canonical curve).
2. `reference.snapshot.metrics.history.loss` — saved reference snapshot.
3. `reference.metrics.history.loss` — when `reference` itself IS a snapshot.
4. `reference.loss` / `reference.final_loss` / `reference.reference_loss`
   — singleton fallback for legacy callers.

---

## 3. Pass criteria (all must hold)

- `snapshot.metrics.loss_finite == true`.
- No hard symptom on the snapshot (`oom_detected`, `nccl_error`,
  `cuda_error`, `python_error`, `loss_nan_or_inf`).
- At least one finite loss observed in the trial window.
- If a reference is available: `(trial_median - ref_median) / |ref_median|
  * 100 ≤ max_delta_pct`.

The trial median is the median of the trailing-`window` finite losses
from the snapshot's `metrics.history.loss`. Window-mode is the default —
single-point compare was empirically too noisy (one warmup spike or
one profiler-perturbed iter would flip the gate). See
`IMPL_VS_DESIGN.md §1` for the regression that motivated this change.

---

## 4. Output

```yaml
stage:                 CORRECTNESS_LITE
status:                pass | fail
run_id:                <id>
snapshot_ref:          <path>
reference_ref:         <path>
loss:                  <float | null>           # trial median over window
reference_loss:        <float | null>           # reference median over window
loss_delta_pct:        <float | null>           # (loss - ref_loss) / |ref_loss| * 100
max_delta_pct:         <float>
loss_finite:           <bool>
hard_symptom:          <bool>
window:                <int>
trial_window_size:     <int>                    # actual sample count used
reference_window_size: <int>                    # 1 for scalar fallback
```

`status` is the only field the Orchestrator routes on; everything else
is for audit.

---

## 5. CLI

```bash
python -m pilot.tools.observe compare_loss \
    --run-id   <trial_run_id> \
    --reference state/<session>/baseline_snapshot.yaml \
    --log-dir  state/runs \
    --window   5 \
    --max-delta-pct 25
```

`compare_loss` is part of the `observe` module (it shares snapshot
loading with `observe snapshot`); it is **not** a separate
`pilot.tools.correctness` module.

---

## 6. When the gate fails

| Failure shape | Routing per `state_machine.md` §6 |
|---|---|
| `loss_nan_or_inf` / non-finite loss | `failure.kind = NUMERICAL` → ABORT + escalate |
| Hard symptom (`oom_detected`, etc.) | route per the symptom (OOM → REPLAN, NCCL → PREFLIGHT) |
| `loss_delta_pct > max_delta_pct` | candidate is rejected from SETTLE; record `reason=correctness_fail` in `run_history` and continue the round |
| Trial window too small (no finite losses) | escalate as `INVALID_PROFILE` if from a BASELINE-or-later run (see `profile.md §3`); otherwise treat as `correctness_fail`. |

---

## 7. Cross-references

- Snapshot shape: `observe.md`.
- Stop / promotion rules: `settle.md` (CORRECTNESS_LITE fails count as
  `dead` measurements when computing `dead_rate_in_subtree`).
- Profile-trace policy: `profile.md §3` (a trace-less BASELINE produces
  an `INVALID_PROFILE` failure that this gate will also surface).
- Schema for the `RunSnapshot` it consumes: `run_snapshot.schema.json`.
