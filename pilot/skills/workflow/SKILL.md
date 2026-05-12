# Workflow — Tuning Loop Overview

**Status**: Single-node v1
**Read by**: Orchestrator + every Stage Worker
**Written by**: humans
**Domain**: tuning main flow

## Purpose

Entry point for the tuning workflow. Describes the outer/inner loop split and points to the per-stage Skills.

## Authoritative per-stage skills (v1)

| Stage / topic | File | Status |
|---|---|---|
| State machine + routing | `state_machine.md` | PREFLIGHT v1; remaining stages stub |
| Orchestrator ↔ Worker contract + bootstrap | `orchestration.md` | **v1** — §0 bootstrap (new vs resume), §1 SKILL_SCOPES, §2 SubagentResult caps, §3 hygiene ritual, §4 handoff thresholds, §5–§7 spawn + failure routing |
| PREFLIGHT protocol | `preflight.md` | v1 |
| PROJECTION / SMOKE | `projection.md`, `smoke.md` | smoke v1; projection stub |
| Profiling / trace analysis | `profile.md`, `trace_analysis.md` | v1 |
| DIAGNOSE rule engine | `diagnose.md` | v2 (trace-driven) |
| Axis catalog + radius rules | `axis_taxonomy.md` | v1 |
| **Re-Plan candidate generation** | `replan.md` | **v1** — priority formula, cost_proxy table, derivation source rules, Skill mapping per bottleneck |
| **Strategy Select** | `execution_strategy.md` | **v1** — decision tree, parameter defaults, calibration fallback |
| **PlanGraph search tree** | `plan_graph.md` | **v1** — node lifecycle, frontier, exhausted_neighborhoods, engine API |
| **Plan structure** | `plan.md` | **v1** — env.diff merge, predicted block, generated_by |
| **Observe / RunSnapshot** | `observe.md` | **v1** — metric definitions, symptom regexes, validity rules |
| **Settle / convergence** | `settle.md` | **v1** — promotion rules, default thresholds, escape mechanisms, stop conditions |
| Execute / EnvSweep / Learn / Correctness / Plan | rest of `workflow/*.md` | stub |

## Single-node v1

For `cluster.yaml mode=single`, the runnable path is:

```
PREFLIGHT/PROJECTION (optional) -> SMOKE -> BASELINE ->
OPTIMIZE_LOOP.{DIAGNOSE,REPLAN,EXECUTE,CORRECTNESS_LITE,SETTLE} -> REPORT -> LEARN draft
```

The non-interactive implementation is `python -m pilot.tools.tune_single run`.
Starting v1, it persists a `PlanGraph` (per `plan_graph.md`) at every Settle exit alongside the legacy `run_history`, picks champions per `settle.md`, and scores candidates per `replan.md`.

Round budget accounting (canonical defaults sourced from `settle.md` §4):

- SMOKE and BASELINE do not increment `budget_used.rounds`.
- Each completed SETTLE stage increments the effective tuning round.
- The loop stops when the configured round count is reached, no candidates are
  left, or SETTLE reports stagnation (gain `< ε_stop = 0.5%` for 2 consecutive rounds).

EnvSweep remains out of the single-node v1 loop. It can be reintroduced once
`preflight.env_probe` and `preflight.env_sweep` are promoted beyond stubs.
