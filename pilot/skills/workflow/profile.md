# Profile (Trace Capture)

**Status**: v0.1 (under iteration; expect skill changes after first real-trace pass)
**Stage**: pre-`OBSERVE` hook on every training run
**Tool boundary**: `pilot.tools.profile.inject(plan, run_dir)` — pure function, no side effects on disk
**Default**: ALWAYS ON.

### Per-stage policy for `submit run --no-profile`

| Stage | `--no-profile` allowed? | Notes |
|---|---|---|
| `SMOKE` | Yes, with a recorded reason | SMOKE is a liveness gate; a known profile-blocker (e.g. `PRIMUS_HIPBLASLT_TUNING=1`) may justify opting out. The reason must be echoed into `summary.headline`. |
| `BASELINE` | **No** | BASELINE produces the champion snapshot; a trace-less champion poisons every downstream DIAGNOSE round. See `@.cursor/rules/30-worker-baseline.mdc`. |
| `OPTIMIZE_LOOP.EXECUTE` | **No** | Trial snapshots must be comparable to the champion trace; `trace_analysis.md` rules require both sides to have traces. |
| `CORRECTNESS` (full) | **No** | Same reasoning — the numerical-gate run is also the trial that feeds back into the loop. |
| `CORRECTNESS_LITE` | N/A | Does not call `submit run`; reuses an existing run's snapshot. |
| `ENV_SWEEP` | N/A | Uses `env_probe sweep`, not `submit run`. |

If an internal `INVALID_PROFILE` fallback (see §5) self-disables profiling
on a BASELINE-or-later run, the run's snapshot is **not** a valid
champion / trial — the Orchestrator must re-run with the profile blocker
removed (typically via ENV_SWEEP) rather than accept the trace-less
snapshot.

---

## 1. Purpose

Every training run that Pilot launches (SMOKE, BASELINE, EXECUTE) must, by default,
emit **one** PyTorch Profiler chrome-trace covering **one** steady-state iteration.
That trace is the input substrate for `trace_analysis.md`; without it, all
downstream DIAGNOSE rules degrade to pure heuristics.

Why this design:
- We learned in the deepseek_v2_lite session that TFLOPs-vs-peak alone is too
  coarse to distinguish COMM_BOUND from COMPUTE_BOUND on MoE workloads.
- Capturing more than 1 iter blows out memory on heavy MoE/FP8 plans.
- Capturing earlier than iter 5 captures kernel-compile noise instead of
  steady-state behavior.

## 2. Capture configuration (defaults)

| Knob | Value | Why |
|---|---|---|
| `profile` | `true` | Megatron flag; gates everything |
| `use_pytorch_profiler` | `true` | torch.profiler chrome-trace JSON; the only format `trace_analysis` understands today |
| `profile_step_start` | **5** | warmup: skip kernel autotune + cuBLAS / hipBLASLt JIT |
| `profile_step_end` | **6** | one steady iter (`active = end - start = 1`) |
| `profile_ranks` | `[0]` | rank-0 trace is enough for COMM/PIPELINE/COMPUTE classification on a single node; we add rank-of-EP later when EP > 1 |
| `tensorboard_dir` | `<run_dir>/profile/tb/` | Megatron / Primus writes the trace to `<this>/<worker>.<pid>.<rand>.pt.trace.json[.gz]` |
| `torch_profiler_use_gzip` | `true` | a single trace can be 100-500 MB raw; gzip cuts ~5x |
| `torch_profiler_record_shapes` | `false` | shapes blow up the trace size; not needed for the rules in `trace_analysis.md` v0 |
| `torch_profiler_with_stack` | `false` | Python stack adds 3-10x overhead and we don't need it for kernel/comm bookkeeping |
| `disable_profiler_activity_cpu` | `false` | we need CPU range markers (e.g. `nvtx Optimizer`) to align kernels |

`train_iters` MUST be `>= profile_step_end + 2` (otherwise the run completes
before the trace handler flushes). `pilot.tools.profile.inject` enforces this
and bumps `train_iters` if too low.

## 3. Hard incompatibilities

| Knob | If set | Action |
|---|---|---|
| `PRIMUS_HIPBLASLT_TUNING=1` | profiler corrupts TE GEMM autotune | the engine logs and DROPS profile-on for this run |
| `PRIMUS_DETERMINISTIC=1` | OK; profiler tolerates determinism | no action |
| `train_iters < 8` | not enough room for warmup+capture+drain | bump `train_iters` to 8 minimum |
| `profile=true` already in plan | user already wants profiling | engine respects existing knobs; only fills missing fields |

## 4. Filesystem layout (the contract `trace_analysis` reads)

```
state/runs/<run_id>/
├── plan.effective.yaml             # the patched plan (profile knobs visible here for audit)
├── train.log
├── handle.yaml
├── snapshots/                      # observe output
└── profile/
    ├── trace_meta.json             # written by pilot.tools.profile after the run completes
    ├── tb/                         # tensorboard_dir; raw torch.profiler output
    │   └── primus-megatron-...-rank[0].<pid>.<rand>.pt.trace.json.gz
    └── (optional) torch_profile/   # only when upstream Megatron path is used
        └── rank-0.json.gz
```

`trace_meta.json` schema (v0.1):

```json
{
  "schema_version": "0.1",
  "run_id": "...",
  "captured_iter_start": 5,
  "captured_iter_end": 6,
  "captured_iter_count": 1,
  "ranks": [0],
  "trace_files": [
    {
      "rank": 0,
      "path": "profile/tb/primus-megatron-....pt.trace.json.gz",
      "bytes": 12345678,
      "format": "chrome_trace_v1"
    }
  ],
  "warnings": []
}
```

## 5. Run-time overhead budget

Empirical (FP8 MoE, EP=8, 1-iter capture):

| Cost | Typical |
|---|---|
| Wall-clock overhead | +5-10% on the captured iter; ~0% on warmup iters |
| HBM peak | +1-3 GB/rank during the captured iter |
| Disk per trace | 30-200 MB gzipped (model and seq dependent) |

If a captured-iter HBM spike trips OOM but the same plan ran clean without
profiling, classify it as `INVALID_PROFILE` (NOT `OOM`) and back off to
`profile=false` for that run only. The `pilot.tools.profile.inject` tool
records this in the run handle.

## 6. CLI contract

The user does not call this tool directly; `submit run` invokes it. But the
manual surface exists for debugging:

```
python -m pilot.tools.profile inject \
    --plan      state/runs/<id>/plan.effective.yaml \
    --run-dir   state/runs/<id> \
    [--profile-step-start 5] \
    [--profile-step-end   6] \
    [--ranks 0,4]                   # comma-separated; default "0"
    [--no-profile]                  # explicit kill switch (returns the plan unchanged).
                                    # Workers MUST NOT pass this from BASELINE
                                    # onward — see the per-stage policy table at
                                    # the top of this skill. Allowed only for
                                    # SMOKE (with a recorded reason) and for
                                    # debugging from a human shell.
    --out-plan  state/runs/<id>/plan.effective.profiled.yaml
```

Exit codes match the rest of `pilot.tools.*`:

| Code | Meaning |
|---|---|
| 0 | success (plan written) |
| 1 | hard incompatibility detected (e.g. HipBLASLT tuning on); profile not enabled; the original plan is left untouched |
| 2 | usage error |
| 3 | TOOL_ERROR |

## 7. Gotchas and known limitations

1. **Multi-rank traces**: by default we capture only rank-0. For EP > 1 in a
   single node, the alltoall pattern is symmetric across ranks; rank-0 is
   representative. For multi-node (when we get there), we will need to capture
   rank-0 + one rank-of-EP + one rank-of-PP. Tracked as a v0.2 follow-up.
2. **Trace flush latency**: torch.profiler flushes asynchronously after
   `prof.stop()`. The training process may exit before the flush completes,
   leaving a 0-byte file. `pilot.tools.profile.collect` waits up to 30s for
   the trace to settle (size stable for 3 consecutive 1s polls).
3. **gzip vs raw**: `trace_analysis` accepts both `.json` and `.json.gz`; gzip
   is preferred for space.
4. **Determinism**: with `profile_ranks=[0]`, only rank-0 takes the
   profiling-induced wallclock hit; this introduces a ~5% per-iter timing
   skew between rank-0 and others on the captured iter ONLY. The skew has not
   caused any deadlock or NCCL/RCCL drift in our testing, but if a future run
   reports `nccl_watchdog_timeout` exactly when `iter == profile_step_end`,
   the engine MUST set the warning `profile_caused_skew` in `trace_meta.json`.

## 8. Worker protocol

The PROFILE step is implicit: every Stage Worker that calls `submit run`
gets the patched plan automatically. Workers do NOT need to read this skill
unless they are debugging trace shape issues.

**Hard rules for Workers / the inline Orchestrator** (mirrors
`@.cursor/rules/00-pilot-core.mdc §I.4` and
`@.cursor/rules/30-worker-baseline.mdc`):

- From `BASELINE` onward, never pass `--no-profile` to `submit run`, and
  never override `profile=false` / `use_pytorch_profiler=false` via
  `--override`.
- If `profile.inject` self-disables on a BASELINE-or-later run
  (`decision.enabled=false` in `handle.yaml`), return
  `failure.kind = INVALID_PROFILE, escalate_to_orchestrator=true`
  instead of promoting the snapshot.

The DIAGNOSE Worker, however, MUST refuse to classify any run whose
`trace_meta.json` is missing or has empty `trace_files[]` — and instead
escalate `suggested_transition.to = OPTIMIZE_LOOP.EXECUTE` with `hint =
"re-run with profile enabled"`. This guarantees the trace-first invariant.
