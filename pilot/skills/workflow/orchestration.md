# Orchestration Protocol

**Status**: v1
**Read by**: Orchestrator (along with `state_machine.md`)

This document is the **contract between the Orchestrator role and the Stage
Worker role**. It governs:

- how a tuning session boots (new or resumed) — §0;
- which Skill subtree each Worker stage may read (SKILL_SCOPES) — §1;
- the shape and size of the `SubagentResult` payload a Worker returns — §2;
- the state-hygiene ritual the Orchestrator runs at every stage exit — §3;
- when the Orchestrator must trigger a session handoff (Strategy C) — §4;
- subagent spawn + failure routing + anti-patterns — §5–§7.

`state_machine.md` governs *routing* (what's the next stage); this document
governs *bootstrap, isolation, and bookkeeping* (how a session starts and
how Workers and the Orchestrator exchange information without bleeding
context).

The Orchestrator MUST follow every rule below. Workers MUST conform to the
SKILL_SCOPES table and the size constraints on `SubagentResult`.

---

## 0. Bootstrap — how a Cursor / Claude / Codex session starts driving Pilot

The user's launch prompt is intentionally **one line** ("start / resume the
Pilot tuning session for X"). The Orchestrator infers everything else from
this protocol.

### 0.1 Decide: new session vs resume

```text
session_dir = pilot/state/<session_id>/
if exists(session_dir/tuning_state.yaml) AND tuning_state.current_stage ∉ {DONE, REPORT, ABORT}:
    → resume protocol (§0.2)
else:
    → new-session protocol (§0.3)
```

### 0.2 Resume protocol

```bash
python -m pilot.tools.state resume <ABSOLUTE_PATH>/tuning_state.yaml
```

The tool prints the trimmed `TuningState` summary. The Orchestrator reads
**only** these fields into context:

- `session_id`, `current_stage`, `round_id`, `champion_id`, `budget_used`
- `stage_history[-3:]` (last 3 entries)
- `artifacts.*` (paths only — not contents)

Then jump straight to step 1 of `00-pilot-core.mdc` Part II (decide
`next_stage` from `state_machine.md`, given `current_stage`).

**Important**: `state.resume`'s relative-path argument is anchored at the
pilot package root, so it tries `pilot/pilot/state/...` for a relative
`pilot/state/...` input. Always pass the **absolute path** to
`tuning_state.yaml`.

### 0.3 New-session protocol

The user message must contain enough to assemble a `TargetVector`
(`schemas/target_vector.schema.json` §primary / constraints / budget). If
any of the four fields below are missing, ask for them once and stop —
do not guess defaults:

| Field | Default if user gave only "model X on cluster Y" | Source |
|---|---|---|
| `plan` (Primus exp YAML path) | required, ask if missing | user |
| `cluster_config` | `pilot/cluster.yaml` | user or `$PILOT_CLUSTER_CONFIG` |
| `primary` metric | `median_tflops` | conservative default |
| `budget` (`rounds`, `candidates_per_round`, `smoke_iters`, `train_iters`, `timeout_s`) | `4 / 3 / 10 / 20 / 900` | conservative default |

Then:

1. Bootstrap the session in **one** tool call:

   ```bash
   python -m pilot session init \
       --plan <PLAN_PATH> \
       --cluster-config <CLUSTER_YAML> \
       --primary <PRIMARY> \
       --rounds <N> --candidates-per-round <K> \
       --smoke-iters <N> --train-iters <N> --timeout-s <N>
   ```

   This single command (see `pilot/tools/session.py::init`):

   - generates `session_id = <model_id>__<cluster_id>__<utc_ts>` (or accepts
     a `--session-id` override; the rule lives in `_default_session_id`),
   - creates `pilot/state/<session_id>/`,
   - writes `tuning.yaml`, `target_vector.yaml`, and the initial
     `tuning_state.yaml` (`current_stage=PREFLIGHT`, `round_id=0`,
     `champion_id=null`),
   - seeds the `r0/` checkpoint.

   Optional flags: `--base-override key=value` (repeatable; e.g.
   `--base-override micro_batch_size=1`), `--constraint key=value`,
   `--trace-subdir`, `--node`, `--notes`, `--force`. Run
   `python -m pilot session init --help` for the full list.

2. Hand off to step 1 of `00-pilot-core.mdc` Part II — the first decision
   will be to spawn the PREFLIGHT Worker (or skip via cache; see
   `state_machine.md` PREFLIGHT row).

### 0.4 Stop conditions (terminate the loop cleanly)

After **every** `apply` step, check the three termination conditions:

- All `TargetVector.constraints` satisfied AND `primary` no longer
  improving → spawn REPORT Worker, then exit.
- `gain < ε_stop` for 2 consecutive rounds AND frontier has no
  high-priority candidates → REPORT, exit.
- `budget_used > budget.total_gpu_h` OR `round_id >= max_rounds` →
  REPORT, exit.

`tune_single.settle` already encodes (b) and (c); the Orchestrator only
needs to honour the Settle Worker's `suggested_transition: REPORT`.

---

## 1. SKILL_SCOPES — which Worker may read what

The Orchestrator passes `skill_scope` to `subagent.spawn` (or, in Cursor,
embeds it in the Task prompt). Workers MUST refuse to read paths outside
their listed scope, even if a transitive reference is tempting.

| Worker stage | Allowed Skill subtree (read) | Forbidden subtree | Rule file |
|---|---|---|---|
| `PREFLIGHT` | `workflow/preflight.md`, `profiling/preflight.md`, `profiling/network.md`, `profiling/env_probe.md`, `env/SKILL.md`, `env/presets.md` | `optimization/**`, `execution-model/**`, `workflow/{replan,diagnose,settle}.md` | `.cursor/rules/30-worker-preflight.mdc` |
| `PROJECTION` | `workflow/projection.md`, `workflow/profile.md`, `execution-model/*` | `optimization/**`, `workflow/{diagnose,replan,settle}.md` | `.cursor/rules/30-worker-projection.mdc` |
| `SMOKE` | `workflow/smoke.md`, `workflow/observe.md` | `workflow/{diagnose,replan,settle}.md`, `optimization/**` | `.cursor/rules/30-worker-smoke.mdc` |
| `BASELINE` | `workflow/smoke.md` §4 pass/fail, `workflow/observe.md` | `workflow/{diagnose,replan}.md`, `optimization/**`, `execution-model/**` | `.cursor/rules/30-worker-baseline.mdc` |
| `OBSERVE` | `workflow/observe.md`, `profiling/gpu.md`, `profiling/network.md` | `workflow/diagnose.md`, `optimization/**` | `.cursor/rules/30-worker-observe.mdc` |
| `DIAGNOSE` | `workflow/diagnose.md`, `workflow/observe.md`, `execution-model/*`, `env/SKILL.md`, `env/{rccl,alloc}.md` | `optimization/**` (Re-Plan's job) | `.cursor/rules/30-worker-diagnose.mdc` |
| `RE_PLAN` | `workflow/replan.md`, `workflow/plan_graph.md`, `optimization/**`, `constraints/**`, `axis_taxonomy.md` | `workflow/diagnose.md`, `workflow/observe.md` (those are inputs, already summarised by Diagnose) | `.cursor/rules/30-worker-replan.mdc` |
| `ENV_SWEEP` | `workflow/envsweep.md`, `env/**`, `profiling/env_probe.md` | `optimization/**`, `workflow/replan.md` | `.cursor/rules/30-worker-envsweep.mdc` |
| `CORRECTNESS` / `CORRECTNESS_LITE` | `workflow/correctness.md`, `workflow/observe.md` (just `compare_loss`) | everything else | `.cursor/rules/30-worker-correctness.mdc` |

The Orchestrator itself may only read **`workflow/state_machine.md`** and
**this file** (`workflow/orchestration.md`). Reading any other Skill from
the Orchestrator role is a violation of `.cursor/rules/00-pilot-core.mdc`
rule 2 (role separation).

---

## 2. SubagentResult — the only payload a Worker returns

Schema source of truth: `pilot/schemas/subagent_result.schema.json`. The
Worker writes its real artifact (Snapshot /
DiagnosisReport / CandidatePool / ...) under `pilot/state/...`, and returns
**only** the structure below as its tool return value:

```yaml
stage: <STAGE>                          # exactly the stage that was spawned
worker_id: <opaque, e.g. cursor_task_<task-id>>
status: success | tentative | failed
artifacts:                              # state-layer references; the Orchestrator does NOT inspect content
  - kind: ClusterProfile | RunSnapshot | DiagnosisReport | CandidatePool | EnvSweepResult | RunHandle | ProjectionReport
    ref: <relative path under pilot/state/...>
    size_bytes: <int, optional>
summary:                                # the ONLY natural-language portion; HARD-CAPPED < 200 tokens
  headline: <str, one line>             # see §2.1 templates
  key_metrics: { ... }                  # 5-8 scalars; numbers, not prose
suggested_transition:                   # advisory only — Orchestrator decides finally per state_machine.md
  to: <STAGE>
  reason: <str, < 80 chars>
cost:
  tokens_used: <int>
  tool_calls: <int>
  gpu_h: <float, optional>
  wallclock_s: <int, optional>
failure: null                           # required when status=failed; schema: pilot/schemas/failure_report.schema.json
```

### 2.1 `summary.headline` templates (per stage)

Worker MUST emit a headline matching the template below (the Orchestrator
displays this in `stage_history`):

| Stage | Headline template |
|---|---|
| PREFLIGHT | `"<n_healthy>/<n_total> nodes, peak BF16 <x> TFLOPs (<pct>%), env_baseline=<version> <status>"` |
| PROJECTION | `"<k> viable configs; top tps=<x>, top mem_pct=<y>"` |
| SMOKE | `"SMOKE <iters_done>/<iters_target> rc=<exit_code> <wallclock>s; iter=<ms> tps=<x> loss=<y> <finite\|nonfinite>"` |
| BASELINE | `"BASELINE <iters_done>/<iters_target> rc=<exit_code> <wallclock>s; iter=<ms> tps=<x> loss=<y> <finite\|nonfinite>"` |
| OBSERVE | `"tps=<x>, comm=<y>, bubble=<z>, status=<completed\|early_stopped\|oom>"` |
| DIAGNOSE | `"<BOTTLENECK>, confidence=<0.xx>, env_suspect=<count>"` |
| RE_PLAN | `"<bottleneck>, <selected_count> exploit + <explore_count> explore candidates, top priority=<p>"` |
| ENV_SWEEP | `"<n> env candidates tested, <m> safe, best Δtps=<x>"` |
| CORRECTNESS / CORRECTNESS_LITE | `"compare_loss vs <ref>: max_abs=<x>, max_rel=<y>%, verdict=<pass\|fail>"` |
| SETTLE | inline (no Worker spawned in single-node v1); recorded directly by Orchestrator |

### 2.2 Size constraints (the Orchestrator enforces all of these)

| Field | Cap | On breach |
|---|---|---|
| `summary.headline` | **200 tokens** | Reject the result, escalate as `failure.kind=CONTEXT_OVERFLOW` (Worker bug, not a task fail) |
| `summary` block (whole) | 500 tokens | same |
| `artifacts[*].ref` | path string, no inlined content | reject if the value looks like multi-line YAML |
| Worker peak context | 30K tokens (Diagnose / Re-Plan); 15K (Preflight, EnvSweep); 10K (Smoke, Baseline, Observe, Correctness) | force early-return + escalate |

The Orchestrator MUST run a textual length check on `summary.headline` /
`summary` before merging into `tuning_state.stage_history`.

---

## 3. State-hygiene ritual (Strategy A, §13.2)

At **every** stage exit (whether the Worker succeeded or failed), the
Orchestrator runs, in order:

1. **Validate** the returned `SubagentResult` against the schema and the
   size caps in §2.2. On violation: log, treat as `failure.kind=CONTEXT_OVERFLOW`,
   do not absorb the payload further.
2. **Apply** pointer-only fields:
   - `champion_id` (if Worker promoted a new champion)
   - `round_id` (only when stage = `OPTIMIZE_LOOP.SETTLE`; per
     `state_machine.md` Global rule 1)
   - `budget_used` (additive)
   - `last_decision_summary` ← `summary.headline`
   - `stage_history` ← append `{stage, status, headline}` (one line)
3. **Checkpoint**:
   ```bash
   python -m pilot.tools.state checkpoint \
       --session <session_id> --stage <stage>
   ```
4. **Trim** — drop everything that is not a pointer field:
   ```bash
   python -m pilot.tools.state trim \
       --keep session_id,current_stage,round_id,champion_id,budget_used,last_decision_summary
   ```
5. **Discard the SubagentResult object itself.** Do not paste it into the
   next prompt; do not summarise it further — the headline you just
   appended IS the summary.

If steps 3 or 4 fail (tool error), do **not** proceed to the next stage;
escalate as `failure.kind=TOOL_ERROR`.

---

## 4. Handoff thresholds (Strategy C, §13.4)

The Orchestrator must self-monitor its own context (token estimate from
the framework or `state.estimate_context_tokens()`). Two thresholds:

| Threshold | Action | Notes |
|---|---|---|
| `context_used > 0.5 × window` | **Recommended** handoff: at the next stage exit, after checkpoint + trim, if context is still above 0.5×, fire `state.handoff()` and exit cleanly. | The next session resumes from the last checkpoint. |
| `context_used > 0.75 × window` | **Mandatory** handoff: do **not** spawn the next Worker; fire `state.handoff()` immediately. | Prevents the next prompt from breaking the budget. |
| Periodic checkpoint: every `K=10` rounds | Force a handoff regardless of token usage. | Keeps long sessions reproducible. |

`state.handoff()` writes
`pilot/state/checkpoints/handoff/<session_id>.yaml` with `pending_decision`
(the next-stage hint). The next Cursor session must be started with the
same `session_id`; `state.resume()` will pick up the handoff automatically
(see `.cursor/rules/20-state-hygiene.mdc`).

---

## 5. Subagent spawn contract (Strategy B, §13.2)

In Cursor, "spawn a Worker" = call the Task tool with:

```text
description: "<stage> Worker"
subagent_type: generalPurpose
prompt: |
  You are the <STAGE> Stage Worker. Follow @.cursor/rules/30-worker-<stage>.mdc.
  Inputs:
    session_id: <id>
    round_id: <int>
    cluster_config_ref: <path>
    <stage-specific input_refs as a YAML block>
  Allowed Skill scope: <copy from §1 of this file>
  Return a SubagentResult conforming to pilot/skills/workflow/orchestration.md §2.
```

The Orchestrator MUST NOT:
- Read the Worker's reasoning trace (Task tool only returns the final
  message; the Orchestrator only consumes that final message).
- Re-spawn a Worker with the previous one's reasoning pasted in.
- Spawn two Workers in parallel for the same `(stage, round_id)` pair
  (parallel Workers are allowed only for *different* `run_id`s in
  OBSERVE — natural parallelism across independent runs).

If the framework lacks subagent support (Strategy B degradation per
§13.6), the Worker becomes a sub-conversation segment in the same
session; the SKILL_SCOPES + summary-cap rules still apply, but
isolation is weaker — the Orchestrator MUST run `state.trim()` extra
aggressively in that mode.

---

## 6. Failure routing summary

When the Worker returns `status=failed`, the Orchestrator first reads
`failure.kind` and routes per `state_machine.md` §12.2:

| failure.kind | Next stage | Counts against budget? |
|---|---|---|
| `OOM` | `RE_PLAN` (mark plan dead, force recompute / smaller MBS) | No |
| `HANG` | `PREFLIGHT` (env_probe only) | No |
| `INVALID_CONFIG` (constraint.check) | `RE_PLAN` | No |
| `NUMERICAL` (loss NaN/Inf, correctness fail) | `ABORT` + escalate | — |
| `CLUSTER` (cluster.yaml broken / SLURM gone / node down) | `ABORT` + escalate | — |
| `CONTEXT_OVERFLOW` (Worker breached size cap) | log + retry once with stricter prompt; 2nd breach → `ABORT` (Worker bug) | — |
| `TOOL_ERROR` (Pilot implementation gap) | `ABORT` + escalate | — |
| `TIMEOUT` | stage-specific; see `state_machine.md` row | — |
| other / `UNKNOWN` | `ABORT` + escalate | — |

The Orchestrator MUST NOT override `failure.kind=CLUSTER` or
`NUMERICAL` — these are hard escalations.

---

## 7. Anti-patterns (do NOT do these)

- **Read a stage Skill from the Orchestrator** to "decide better". The
  Skill belongs to the Worker; reading it in the Orchestrator is a
  regression to the single-agent anti-pattern (§13.5).
- **Inline the Worker's full reasoning** in `last_decision_summary` for
  audit. Audit lives in `artifacts[*].ref` + the State Layer, not in
  context.
- **Run two Workers sequentially without checkpoint + trim in between**.
  Token usage doubles, attention dilution kicks in.
- **Spawn one Worker per Skill file** (e.g. one Worker per
  `optimization/comm/*.md`). Skills cross-reference; the per-stage
  Worker is the right granularity.
- **Use `subagent.spawn` as "a bigger context"**. If a Worker bloats
  past its peak budget, the problem is not solved — see `state_machine.md`
  on-fail rules.

---

## 8. One-line creed

**Pointers in, summary out. Validate, apply, checkpoint, trim. Never
absorb Worker reasoning.**
