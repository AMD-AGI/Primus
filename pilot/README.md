# Primus Pilot — Tuning Skill Collection

**Pilot is a set of Cursor skills for tuning Primus training jobs.** It is domain knowledge (formulas, decision tables, env catalog, optimization strategies) packaged as `SKILL.md` files that Cursor loads on demand.

Cursor itself is the runtime: the **main agent session drives the tuning loop**, **subagents run training**, and **chat history holds the state**. Nothing in `pilot/` is executable — the only files here are skills under `pilot/skills/`.

## Who reads what

- **Cursor main agent** reads `skills/tuning-loop/SKILL.md` and uses the other skills as decision references while it thinks.
- **Cursor subagents** are spawned by the main agent (one per training run) and read `skills/run-and-profile/SKILL.md` to actually launch Primus, parse profile, and return a markdown summary.
- The **main agent never runs training itself** and **never reads raw profile content** — only the subagent's markdown summary lands back in the main chat.

## The one hard convention

> **Every "submit training + collect profile" goes through a subagent following `skills/run-and-profile/SKILL.md`.**
> The subagent returns a single ~200-word markdown block (run id, plan, status, metrics, bottleneck hint, artifact paths). The main agent pastes that block into chat and discards everything else. Raw logs and profiler traces stay on disk under `output/pilot/runs/<run_id>/`.

This is the only architectural rule. There are no schemas to validate, no state files to manage, no orchestration protocol — just "training runs are subagents, summaries are markdown".

## Entry point

Start at **`skills/tuning-loop/SKILL.md`**. It walks the user-input → preflight → baseline → loop → report flow. Every other skill is loaded on demand from there.

## Skill index

| Skill | When the main agent loads it |
|---|---|
| `skills/tuning-loop/` | Always — the entry point and overall think loop |
| `skills/run-and-profile/` | Read by the spawned subagent each time training is launched |
| `skills/preflight/` | First time on a cluster, or when the cached `output/pilot/cluster-<id>.md` is stale (> 7 days, or driver/fabric changed) |
| `skills/primus-defaults/` | LOOP round 1 — Primus Turbo / DeepEP / fusion / bf16 precision-aware optimizer "should-be-on" flag batch, applied once per session before any bottleneck-specific tuning |
| `skills/execution-model/` | Sanity-checking a candidate's predicted T_step / Mem_peak before launching it |
| `skills/bottleneck-diagnose/` | After every run from round 2 onward, to classify the bottleneck (COMM / PIPELINE / MEMORY / COMPUTE / MOE_DISPATCH / MIXED) |
| `skills/optimize-comm/` | When the latest diagnosis is COMM_BOUND — bucket / overlap / topology / NCCL flags |
| `skills/optimize-pipeline/` | When PIPELINE_BOUND — vpp / mbs / gbs / stage balance |
| `skills/optimize-memory/` | When MEMORY_BOUND or after an OOM — recompute / shard / alloc env / offload |
| `skills/optimize-compute/` | When COMPUTE_BOUND — mbs / fusion / fp8 / threading env / hipblaslt |
| `skills/optimize-moe/` | When MOE_DISPATCH_BOUND — dispatch overlap / ep / capacity_factor / grouped GEMM |
| `skills/env-catalog/` | Whenever an `env_diff` field needs lookup (NCCL_*, RCCL_*, HSA_*, PYTORCH_HIP_ALLOC_CONF, OMP_NUM_THREADS, presets, dangerous combos) |

## Where things land

| Artifact | Path | Purpose |
|---|---|---|
| Cluster baseline | `output/pilot/cluster-<cluster_id>.md` | Reused across sessions; drives `execution-model` constants |
| Per-run artifacts | `output/pilot/runs/<run_id>/{plan.yaml, log.txt, profile/, snapshot.yaml, result.md}` | Subagent's working directory; main agent reads `result.md` only |
| Final session report | `output/pilot/<session_id>.md` | Best plan + decision trace; carry-forward for next session on same model+cluster |

Nothing else needs to persist. The `(champion / shelved / dead / tried axes)` ledger lives as plain markdown in chat — that *is* the state.

## Directory shape (enforced)

`pilot/` is intentionally minimal — the only things consumed at runtime are:

- `README.md` — this file (operational entry: one-page usage + skill index)
- `ARCHITECTURE.md` — design rationale, layered model, anti-patterns, extension guide
- `skills/<skill-name>/SKILL.md` — one directory per skill, each with a single `SKILL.md`

No Python, no JSON Schema, no YAML state, no `prompts/` / `tools/` / `schemas/` / `state/` / `integrations/` / `agent/` subdirectories. If a skill starts wanting one of those, collapse it back into a decision table inside the SKILL.md, or push the responsibility back to Cursor's native session / subagent / chat-history mechanism. See `ARCHITECTURE.md` §10 for the full anti-pattern list and the reasons.

## Self-check

```
ls pilot/skills/
# Expect: bottleneck-diagnose  env-catalog  execution-model
#         optimize-comm  optimize-compute  optimize-memory
#         optimize-moe  optimize-pipeline
#         preflight  primus-defaults  run-and-profile  tuning-loop
```

Each skill is a single `SKILL.md` with frontmatter (`name`, `description`) plus a `## Workflow` and `## Important Notes`.
