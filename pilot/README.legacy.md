# Primus Pilot — Training Tuning System

**Status**: Design spec

> Automatic tuning system for training jobs.
> An Agent (any tool-calling LLM such as Cursor / Claude / Codex) reads knowledge from Skills, calls Tools to execute actions, and closes the loop of modeling → search → convergence.

---

## Scope & Positioning

> **Pilot is a knowledge + toolkit for the tuning domain, not an agent runtime.**
> Anything about "how to reason, how to isolate context, how to spawn subagents" is delegated to the concrete agent framework (Claude Code / Cursor / Codex / your own harness). Pilot only provides "what to know, what to do, what state looks like" for tuning.

### Division of responsibility

| Concern | Pilot owns | Agent framework owns |
|---------|------------|----------------------|
| Tuning domain knowledge (bottleneck taxonomy, optimization strategies, env catalog, ...) | ✓ `skills/*.md` | — |
| **Role prompts** for Orchestrator / Stage Worker | ✓ `prompts/` | injects them |
| Business actions (preflight / submit / observe / constraint / ...) | ✓ `tools/` (CLI or MCP) | invokes them |
| Data contracts (TuningState / PlanGraph / SubagentResult / ...) | ✓ `schemas/` | parses & validates |
| State persistence (YAML/JSON on disk, checkpoint layout) | ✓ `state/` + `tools/state/*` | reads/writes |
| LLM calls, tool_use parsing, retry, rate limit | — | ✓ |
| **Subagent isolation** (independent context, restricted tool scope) | — | ✓ Claude Code Task / Cursor Task / ... |
| **Context management** (compression, window, handoff) | — | ✓ (Pilot only provides policy rule Skill, §13.2 strategy A) |
| Tool protocol (MCP / function calling / shell) | — | ✓ |

### Core rules

1. **Pilot's core directories (skills / prompts / tools / schemas / state) do not import any agent SDK.**
   Dependencies like anthropic / cursor-client / openai / ... are only allowed under `integrations/` (if any).
2. **Tools expose a process or MCP boundary, not a Python function boundary.**
   CLI (`python -m pilot.tools.submit --plan ...`) or MCP server methods; any framework can plug in via shell/MCP without binding to an SDK.
3. **Prompts are framework-agnostic Markdown.**
   No API formats (`input_schema` / `tool_choice` / ...). Tool registration is done by the integrations layer.
4. **Schemas are JSON-Schema-sourced**; for Python convenience a Pydantic mirror is generated.
5. **State is the file system.** Any agent reads/writes via shell or `tools/state/*` CLI.

### Where roles and responsibilities live

- **The Orchestrator / Stage Worker in §2.2 are "roles", not components Pilot itself implements.** The agent framework's main session takes the Orchestrator role; the framework's native subagent mechanism (Claude Code Task / Cursor subagent) takes the Stage Worker role. Pilot specifies the prompt, available tools, and output contract for each role.
- **Ownership of §13's three-tier context strategy**:
  - **Strategy A (State-first protocol)**: Pilot owns — written into `skills/workflow/orchestration.md` as rules the Agent must follow + `schemas/` constraining `SubagentResult.summary` size.
  - **Strategy B (Subagent isolation)**: **Framework owns** — Pilot only provides the boundary table for "which stages should spawn a subagent" (§13.2); the spawn mechanism itself is the framework's job.
  - **Strategy C (Session handoff)**: **Framework owns** — Pilot only specifies the file format of the handoff state (§8.7).

### About `pilot/agent/`

`pilot/agent/` is an **optional Python reference implementation / fallback** that demonstrates "what to write if you must build your own harness."
**Production path recommended**: have the agent framework consume `pilot/skills` + `pilot/prompts` + `pilot/tools` directly; thin framework adapters live under `pilot/integrations/<framework>/`.

---

## Table of contents

0. [Scope & Positioning](#scope--positioning)
0.1 [Prerequisites — what you prepare before invoking Pilot](#prerequisites)
1. [Problem & boundary](#1-problem--boundary)
2. [System architecture](#2-system-architecture)
3. [System flow](#3-system-flow)
4. [Directory layout](#4-directory-layout)
5. [Tool interfaces](#5-tool-interfaces)
6. [Execution Model (core knowledge)](#6-execution-model-core-knowledge)
7. [Search-space maintenance & solution guarantees](#7-search-space-maintenance--solution-guarantees)
8. [Data structures (Schema)](#8-data-structures-schema)
9. [Full iteration example](#9-full-iteration-example)
10. [Evaluation metrics](#10-evaluation-metrics)
11. [Integration with existing systems](#11-integration-with-existing-systems)
12. [Guardrails](#12-guardrails)
13. [Context Management & Multi-Agent Orchestration](#13-context-management--multi-agent-orchestration)
14. [One-line summary](#14-one-line-summary)

---

## Prerequisites

Pilot is a tuning **system**, not a cluster manager. Before any `python -m pilot.tools.*` command can do useful work, you (the user / SRE) must have prepared one of two runtime environments:

| Mode | What you prepare | What Pilot then does |
|------|------------------|----------------------|
| **`single`** | One container running locally with N GPUs visible | Runs everything inside that container; no cross-node fan-out |
| **`slurm`**  | A SLURM allocation (`salloc -N k`) plus container plumbing on each node | Attaches to the existing allocation via `srun --jobid=<id>`; does **not** create or destroy allocations |

You declare which mode you are in by writing a small `cluster.yaml` (schema: `schemas/cluster_config.schema.json`). Every Pilot tool reads this same file as its only environmental input. There is **no** ad-hoc fallback: if `cluster.yaml` is missing or stale, every tool exits with `failure.kind=CLUSTER` and a pointer to fix it.

→ See **`SETUP.md`** for the full step-by-step guide (docker run / salloc commands, container plumbing options, validation checks, common pitfalls).
→ See **`AGENTS.md` §4** for the agent-facing version of the same contract.

This boundary is deliberate: it keeps Pilot small, portable, and reproducible. Anything that would otherwise live as "ssh fan-out config", "docker compose", or "k8s job manifest" stays *outside* Pilot.

---

## 1. Problem & boundary

| Challenge | Concrete problem |
|-----------|------------------|
| **Parameter space explosion** | DP / TP / PP / EP / VPP / CP × MBS × recompute × communication params, 10⁴+ combinations |
| **Bottleneck localization is hard** | compute / comm / memory / bubble interleave; troubleshot from scratch each time |
| **High trial cost** | One multi-node experiment costs hundreds of GPU·h; bad runs are only revealed after completion |
| **Fragmented experience** | best-known configs scatter across Slack / wiki / heads; not reusable |

**In scope**: Dense / MoE bring-up, scaling-degradation diagnosis, joint parallelism + communication tuning.

**Out of scope**: automatic kernel / model architecture / communication library implementation changes.

---

## 2. System architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Agent Layer                                │
│         (any tool-calling LLM, e.g. Cursor / Claude / Codex)        │
│                                                                     │
│   - Reads Skills for knowledge (workflow / execution-model / opts)  │
│   - Reads/writes State as working memory (PlanGraph / TuningState)  │
│   - Calls Tools for execution (preflight / submit / observe / ...)  │
│   - Drives the Tuning Loop per state_machine.md transition rules    │
└────────────────────┬─────────────┬────────────────────┬─────────────┘
                     │             │                    │
                     ▼             ▼                    ▼
┌───────────────────────┐ ┌──────────────────┐ ┌───────────────────────┐
│     Skill Layer       │ │   State Layer    │ │      Tool Layer       │
│   (knowledge / .md)   │ │ (working mem/yml)│ │    (execution / py)   │
│                       │ │                  │ │                       │
│  skills/              │ │ state/           │ │  tools/               │
│  ├── workflow/        │ │ ├── cluster_     │ │  ├── preflight.py     │
│  │   (state_machine,  │ │ │   profile.yaml │ │  ├── env_probe.py     │
│  │    plan_graph,     │ │ ├── tuning_      │ │  ├── profiler.py      │
│  │    replan, settle, │ │ │   state.yaml   │ │  ├── submit.py        │
│  │    smoke,          │ │ ├── plan_        │ │  ├── observe.py       │
│  │    correctness …)  │ │ │   graph.yaml   │ │  ├── constraint.py    │
│  ├── execution-model/ │ │ ├── candidate_   │ │  ├── state.py         │
│  ├── optimization/    │ │ │   pool.yaml    │ │  └── knowledge.py     │
│  ├── env/             │ │ └── checkpoints/ │ │                       │
│  ├── profiling/       │ │   r0/, r1/, ...  │ │  Agent calls these    │
│  ├── constraints/     │ │                  │ │  via function calls;  │
│  └── knowledge/       │ │ Each stage exit  │ │  functions read/write │
│                       │ │ checkpoints; can │ │  the State Layer.     │
│  Agent reads .md for  │ │ be interrupted / │ │                       │
│  domain rules         │ │ resumed / replay │ │                       │
└───────────────────────┘ └──────────────────┘ └───────────────────────┘
         ▲                          ▲                       ▲
         │                          │                       │
         └────── LEARN stage writes best/failure cases ─────┘
                    back to skills/knowledge/
                    (the only Skill ← State reverse flow)
```

**Four-layer split of responsibilities**:

| Layer | Form | Who writes | Who reads | Example |
|-------|------|------------|-----------|---------|
| **Agent** | LLM inference | — | Skills + State | "comm_ratio=0.35, per skills/optimization/comm/SKILL.md try overlap" |
| **Skill** | Markdown | Humans (rare LEARN-stage automatic writes) | Agent | `T_bubble = (pp-1)/(pp-1+M) × T_comp` |
| **State** | YAML / JSON | Agent via `state.checkpoint()` | Agent + audit/replay | `PlanGraph.champion = r2_p4` |
| **Tool** | Python function | Humans | Agent (function call) | `preflight.run()` returns ClusterProfile, writes State |

**Key design principles**:
- **Skill ↔ State is one-way**: Agent reads Skill to decide what to do, reads State to decide next step; only LEARN writes back to Skill (knowledge precipitation).
- **State is single source of truth**: all cross-stage working memory lives in the State Layer; Tools are stateless functions (input → output + State update).
- **Skill is knowledge, not logic**: every "if X then Y" rule is also written in Markdown (e.g. transition table in `state_machine.md`). The Agent is the executor of rules, not the owner.
- **Observable = replayable**: State is fully persisted; any decision can be reconstructed from `state/checkpoints/rN/`.
- **The Agent is two-layered**: Orchestrator only holds pointers, Stage Worker absorbs details; see §2.2 and §13.

### 2.2 Agent Orchestration Model

> This section describes **Orchestrator / Stage Worker as two roles, not as components Pilot itself implements**. They are carried by the concrete agent framework (Claude Code's main session + Task-spawned subagent, Cursor's Agent + subagent, your own harness, ...). Pilot only specifies the responsibilities, visible tool set, context budget, and output contract for each role; "how to spawn, how to isolate, how to manage context" is delegated to the framework. See **Scope & Positioning** above.
>
> **Why split into two layers**: if a single session has to handle both "state-machine progression" (small but long-lived) and "per-stage reasoning" (large but short-lived), context grows linearly with rounds and lands in the attention-dilution zone. Splitting the roles, having the framework's native subagent mechanism deliver isolation, keeps the Orchestrator's steady-state context O(1).

```
┌───────────────────────────────────────────────────────────────────┐
│                Orchestrator Agent (long-lived)                    │
│                                                                   │
│  Holds (steady state < 2K tokens):                                │
│    - session_id, current_stage, round_id                          │
│    - champion_id (pointer into PlanGraph, not the node detail)    │
│    - budget_used, budget_remaining                                │
│    - last_decision (one-line summary)                             │
│                                                                   │
│  Responsibilities:                                                │
│    - Read skills/workflow/state_machine.md + orchestration.md     │
│    - Decide next stage by transition rules                        │
│    - Call subagent.spawn() to spawn a Stage Worker                │
│    - Receive SubagentResult (< 200 tokens), update pointers,      │
│      checkpoint                                                   │
│    - Never re-load Snapshot / CandidatePool / Skill details       │
└──────┬────────┬─────────────┬──────────────┬─────────────┬────────┘
       │        │             │              │             │ spawn
       ▼        ▼             ▼              ▼             ▼
   ┌───────┐┌─────────┐  ┌──────────┐  ┌─────────────┐┌──────────┐
   │Diagnose││Re-Plan │  │EnvSweep  │  │Correctness- ││Preflight │
   │ Worker ││ Worker │  │ Worker   │  │Lite Worker  ││ Worker   │
   │(one-   ││(one-   │  │(one-shot)│  │(one-shot)   ││(one-shot)│
   │ shot)  ││ shot)  │  │          │  │             ││          │
   └───┬───┘└───┬────┘  └────┬─────┘  └──────┬──────┘└────┬─────┘
       │        │             │              │             │
       │        │   - Reads only its slice of Skill subtree            │
       │        │   - Reads only its relevant slice of State           │
       │        │   - Calls business Tools (submit / observe / ...)    │
       │        │   - Writes back to State Layer                       │
       │        │   - Returns SubagentResult                           │
       ▼        ▼             ▼              ▼             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                State Layer (shared memory)                   │
  │   PlanGraph / CandidatePool / TuningState / checkpoints/     │
  │   Workers do *not* talk to each other directly; everything   │
  │   goes through here (isolation + auditability)               │
  └──────────────────────────────────────────────────────────────┘
```

**Two-layer contract**:

| Dimension | Orchestrator | Stage Worker |
|-----------|--------------|--------------|
| Lifetime | Whole session (may span days) | Single stage; destroyed on completion |
| Context budget | Steady < 2K tokens / round | Single peak < 30K tokens |
| Reads Skill | Only `workflow/state_machine.md` + `orchestration.md` | Only the relevant subtree (e.g. `optimization/comm/*`) |
| Reads State | Only `TuningState.summary` | On demand: PlanGraph / Snapshot / etc. slices |
| Writes State | Only pointer-class fields | Full stage product (DiagnosisReport / CandidatePool / ...) |
| Calls Tool | Only `state.*` / `subagent.spawn()` | All business Tools (submit / observe / constraint / ...) |
| Sees history | Cannot see Worker reasoning trace | Cannot see other Workers' context |

**Properties this split delivers**:
- Orchestrator only holds pointers; details live in the State Layer. Stage Workers are reborn each time, used and discarded, never accumulating context.
- Per-round Orchestrator delta < 500 tokens; over long loops (20+ rounds) context does not grow linearly.
- Side benefit: multi-plan Observe / Diagnose can be spawned in parallel — natural parallelism.

**Who actually spawns subagents**:

| Framework | Orchestrator carrier | Stage Worker carrier |
|-----------|----------------------|----------------------|
| Claude Code | main session | subagent spawned by Task tool |
| Cursor | Agent main session | subagent spawned by Task tool |
| OpenAI Codex | main session | its subagent mechanism |
| In-house harness (`pilot/agent/`) | `Orchestrator` Python class | `StageWorker` Python class + independent Claude API session |

Pilot's role is to **write one set of prompts + tool scope + output contracts** that all of these carriers consume, so Orchestrator behavior is consistent across frameworks.

---

## 3. System flow

> Three views: **§3.1 block-diagram main flow** gives stage I/O at a glance (most readable) → **§3.2 flow narrative** uses prose to thread each stage's responsibility and transition rules → **§3.3 internal swimlane** unfolds the role split (Agent / Skill / Tool) inside the Tuning Loop.

### 3.1 Block diagram (main trunk)

```
                         ┌──────────────────────────────┐
                         │        User Input            │
                         │  - Model Spec                │
                         │  - Cluster Size              │
                         │  - TargetVector              │
                         │    (primary / constraints /  │
                         │     budget)                  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        1. PREFLIGHT++                                │
│                                                                      │
│  Collect:                                                            │
│    - GEMM / MFMA peak                                                │
│    - IB / XGMI bandwidth                                             │
│    - AllReduce / All2All baseline                                    │
│    - env probe (NCCL/HSA/alloc connectivity + micro-bench)           │
│                                                                      │
│  Output:                                                             │
│    ClusterProfile = {                                                │
│        compute_peak, comm_bw, latency, overlap_capability,           │
│        env_baseline (cluster_shared default)                         │
│    }                                                                 │
│  (Reused across jobs by version + age; auto re-run when stale)      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     2. PROJECTION / Modeling                         │
│                                                                      │
│  Input: Model Spec + ClusterProfile                                  │
│                                                                      │
│  Step 1: Single-node profiling                                       │
│    (layers, mbs, recompute) → T_comp, Mem_peak                       │
│                                                                      │
│  Step 2: Build Execution Model                                       │
│    T_comp(l, mbs) / Mem(l, mbs) / T_comm / Bubble(P, M)              │
│                                                                      │
│  Step 3: Generate Initial Plans                                      │
│    Plan = { parallel, partition, mbs, recompute,                     │
│             env.diff (scale-aware), expected: {tps, bottleneck} }    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          3. SMOKE                                    │
│  Tiny scale × 100 step: verify it starts / no silent hang / no OOM   │
│  Failure → back to PROJECTION                                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         4. BASELINE                                  │
│  Full-scale baseline run; recorded as history[0]                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       5. CORRECTNESS                                 │
│  loss curve / grad norm vs reference alignment                       │
│  Failure → ABORT + escalate (numerical correctness broken — stop)    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │            Tuning Loop (core / two-layer)          │
        └────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   6. Outer · structural loop                         │
│                                                                      │
│   Execute → CORRECTNESS-LITE → Observe → Diagnose → Re-Plan          │
│                                                  │                   │
│                                                  ▼                   │
│   ┌───────────────────────────────────────────────────────────────┐  │
│   │  Re-Plan sub-flow                                              │  │
│   │                                                                │  │
│   │  ① Pick derivation source from PlanGraph                       │  │
│   │     (default: champion; on stagnation: shelved)                │  │
│   │  ② Skill mapping by bottleneck:                                │  │
│   │      COMM    → bucket / overlap                                │  │
│   │      PIPELINE→ vpp / microbatch                                │  │
│   │      MEMORY  → recompute / offload                             │  │
│   │      COMPUTE → mbs / parallel                                  │  │
│   │  ③ Generate CandidatePool (each candidate has                  │  │
│   │      predicted_gain × confidence / cost = priority)            │  │
│   │  ④ Constraint Check (OOM / invalid / env incompatible)        │  │
│   │  ⑤ exhausted_neighborhoods dedup                               │  │
│   │  ⑥ Strategy Select:                                            │  │
│   │      cluster_shared+weakly_local dominant → Champion-Challenger│  │
│   │      strongly_local + model trustworthy   → Per-Plan + Pruning │  │
│   │      ample budget / uncertain             → Successive Halving │  │
│   │  ⑦ Output top-K plans into Execute                             │  │
│   └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼ (conditional: env_suspect hit AND structurally stable)
┌──────────────────────────────────────────────────────────────────────┐
│                   7. Inner · EnvSweep (optional)                     │
│                                                                      │
│   - Lock the structure of outer best plan                            │
│   - Sweep weakly_local env axes (NCCL_BUFFSIZE / alloc_conf / ...)   │
│   - 30-50 step parallel short runs; merge best env diff into baseline│
│   - Per-call cap: ≤ 5 flags, ≤ 8 combinations                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    8. Settle / Convergence                           │
│                                                                      │
│   Maintain PlanGraph (tree + frontier + exhausted_neighborhoods):    │
│     - new_best.tps > champion × (1+ε_promote=2%) → promote champion  │
│         old champion → shelved (kept as backtrack candidate)         │
│     - marginal gain → champion unchanged; new best → shelved         │
│     - subtree dead-rate > 50% for 2 rounds → backtrack to frontier   │
│     - every K=3 rounds force one explore round (only from shelved)   │
│                                                                      │
│   Stop conditions:                                                   │
│       · TargetVector.constraints all met AND primary no longer       │
│         meaningfully improving                                       │
│       · gain < 2% (two consecutive rounds) AND no high-priority      │
│         candidate in frontier                                        │
│       · max_rounds / budget reached                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
              not converged ◄──┴──► converged
                  │                            │
       back to 6. outer next round              ▼
                              ┌───────────────────────────────────────┐
                              │             9. REPORT                 │
                              │   Final Best Config + decision trace  │
                              └────────────────┬──────────────────────┘
                                               │
                                               ▼
                              ┌───────────────────────────────────────┐
                              │             10. LEARN                 │
                              │   best / failure cases written back   │
                              │   to skills/knowledge/                │
                              └───────────────────────────────────────┘

【Reentry edges】(may fire from any stage; do not consume the round budget)
  Diagnose finds ClusterProfile stale       → PREFLIGHT
  Re-Plan finds structural invalidation      → PROJECTION
  HANG (NCCL/IB timeout)                     → PREFLIGHT (env_probe)
  CLUSTER (node down / driver error)         → PREFLIGHT
  OOM / INVALID_CONFIG                       → Re-Plan (mark dead)
  NUMERICAL drift (CORRECTNESS-LITE)         → ABORT + escalate
```

### 3.2 Flow narrative (state-machine driven, each stage exit persists `TuningState`, supports interrupt-and-resume)

1. **Requirement gathering**: user provides Model Spec / Cluster Size / **TargetVector** (primary + constraints + budget, see §8.6).
2. **Read workflow Skill**: Agent reads `skills/workflow/SKILL.md` + `state_machine.md` to learn the stage set and transition rules.
3. **Project info collection**: Agent calls Tools to gather cluster and model configuration.
4. **Enter the state-machine-driven workflow**:
   - **PREFLIGHT**: hardware baseline + cluster-level env baseline → `ClusterProfile` (reused across jobs, re-collected by `version`+`age`).
   - **PROJECTION**: single-node profiling + Execution Model + initial Plans (with scale-aware env diff).
   - **SMOKE**: full configuration + tiny scale (e.g. 1 node / 100 step); verify it starts and no silent hang; failure goes back to PROJECTION.
   - **BASELINE**: full-scale baseline run.
   - **CORRECTNESS**: loss curve / grad norm aligned with reference (numerical gate; failure ABORTs and escalates).
   - **OPTIMIZE_LOOP (two-layer + state machine)**:
     - Outer (structure): `Observe → Diagnose → Re-Plan → Execute → CORRECTNESS-LITE → Settle`.
     - Inner (EnvSweep, optional): triggered after Settle, lock structure and sweep env diff.
     - Any stage may jump back via `reentry_when` (e.g. Diagnose finds ClusterProfile stale → PREFLIGHT; Re-Plan finds structural invalidation → PROJECTION).
     - Guardrails fire explicit failure paths (see §12) without consuming the round budget.
   - **REPORT**: emit Final Config + decision trace.
   - **LEARN**: best config and failure cases write back to `skills/knowledge/` for next time.
5. **Acceptance**: run full tests, audit logs, confirm commit.

> **Why outer/inner are split**: env tuning does not change shape (no OOM risk), single-call cost is low (30-50 steps to differentiate), so it fits as a "safe sweep" after each outer baseline stabilizes. The optimal env value depends on structure (mbs / world_size), so it **cannot be done once and reused forever** — it must be embedded in each outer round.
>
> **Why a state machine instead of a linear pipeline**: adding a new stage / new transition / new run mode (e.g. RL post-training double loop) becomes "add a node + add an edge" rather than rewriting the main swimlane diagram; explicit declaration of `reentry_when` (jump-back conditions) and `on_fail` (failure transitions) gives the Agent rules to follow.

### 3.3 Tuning Loop internal swimlane (role split, expanded)

> Two views: **the outer Orchestrator only drives the state machine + spawns subagents**, **the inner Stage Worker (the "Agent" column in the swimlane below) handles Skill reading and Tool calling for that stage.** Between two Workers, the Orchestrator only sees a one-line `SubagentResult.summary`; it does not absorb Worker reasoning traces.

**Outer (Orchestrator view, ~< 500 tokens added per stage)**:

```
[loop over rounds]
  1. Read TuningState.summary (produced by state.resume, ~200 tokens)
  2. Decide next_stage per state_machine.md
  3. worker_result = subagent.spawn(
        stage = next_stage,
        input_refs = {snapshot_id, plan_graph_ref, ...},
        skill_scope = ["workflow/<stage>.md", "optimization/<bottleneck>/*"])
  4. Update pointers: champion_id / round_id / budget_used
  5. state.checkpoint() + state.trim()  ← discard Worker trace
  6. Decide Stop / Continue / Handoff
```

**Inner (Stage Worker view, single < 30K tokens, used and destroyed)**:

```
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│   Stage Worker      │        Skill            │        Tool             │
│  (one-shot agent)   │   (knowledge)           │   (execution)           │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│                     │                         │                         │
│ [Execute]           │                         │                         │
│     │               │                         │                         │
│     ├── read execute.md ► execution rules    │                         │
│     │               │   - early_stop thresh   │                         │
│     │               │   - serial/parallel pol │                         │
│     │               │                         │                         │
│     └── call Tool ─────────────────────────────► submit.run(plan)       │
│                     │                         │   └► returns run_id     │
│                     │                         │                         │
│ [CORRECTNESS-LITE]  │                         │                         │
│  (after Execute,    │                         │                         │
│   spot-check)       │                         │                         │
│     │               │                         │                         │
│     ├── read correctness.md ► numerical gate │                         │
│     │               │   - loss_delta_pct thr  │                         │
│     │               │   - grad_norm range     │                         │
│     │               │                         │                         │
│     └── call Tool ─────────────────────────────► observe.compare_loss() │
│                     │                         │   └► pass / drift       │
│                     │                         │                         │
│ [Observe]           │                         │                         │
│     │               │                         │                         │
│     ├── read observe.md ► snapshot schema    │                         │
│     │               │   - tps, bubble_ratio   │                         │
│     │               │   - comm_ratio, mem     │                         │
│     │               │                         │                         │
│     └── call Tool ─────────────────────────────► observe.snapshot(run_id)
│                     │                         │   └► returns Snapshot   │
│                     │                         │                         │
│ [Diagnose]          │                         │                         │
│     │               │                         │                         │
│     ├── read diagnose.md ► bottleneck rules  │                         │
│     │               │   if comm_ratio > X     │                         │
│     │               │     → COMM_BOUND        │                         │
│     │               │   if bubble > X         │                         │
│     │               │     → PIPELINE_BOUND    │                         │
│     │               │                         │                         │
│     └── Agent infers bottleneck               │                         │
│                     │                         │                         │
│ [Re-Plan]           │                         │                         │
│     │               │                         │                         │
│     ├── read optimization/{bottleneck}/ ─────┤                          │
│     │               │   returns strategies:   │                         │
│     │               │   - increase TP         │                         │
│     │               │   - tune bucket size    │                         │
│     │               │   - enable overlap      │                         │
│     │               │                         │                         │
│     └── call Tool ─────────────────────────────► constraint.check(plan) │
│                     │                         │   └► valid / invalid    │
│                     │                         │                         │
│ [EnvSweep]          │                         │                         │
│  (conditional:      │                         │                         │
│   bottleneck hit    │                         │                         │
│   AND struct stable)│                         │                         │
│     │               │                         │                         │
│     ├── read env/SKILL.md & env_probe.md ────┤                          │
│     │               │   - candidate flags     │                         │
│     │               │   - safe sweep proto    │                         │
│     │               │                         │                         │
│     ├── read optimization/{bottleneck}/env.md │                         │
│     │               │   - flags relevant      │                         │
│     │               │     under this bound    │                         │
│     │               │                         │                         │
│     ├── call Tool ─────────────────────────────► constraint.check_env() │
│     │               │                         │   └► reject dangerous   │
│     │               │                         │                         │
│     └── call Tool ─────────────────────────────► submit.run(plans, short)
│                     │                         │   └► 30-50 step parallel│
│                     │                         │      returns best diff  │
│                     │                         │                         │
│ [Settle]            │                         │                         │
│     │               │                         │                         │
│     ├── read settle.md ► convergence rules   │                         │
│     │               │   - greedy pick best    │                         │
│     │               │   - stop conditions     │                         │
│     │               │                         │                         │
│     └── Agent decides: continue / stop        │                         │
│           │                                   │                         │
│           ├── continue → back to [Execute]    │                         │
│           └── stop → emit Final Config        │                         │
│                     │                         │                         │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 4. Directory layout

### 4.1 Repository overall layout

Pilot organizes directories around the 5 product types declared in Scope & Positioning + 2 optional adapter types:

```
pilot/
│
├── skills/                         # Tuning domain knowledge (Markdown; see §4.2)
│   ├── workflow/                   #   Tuning main flow (state machine / orchestration / observe / ...)
│   ├── execution-model/            #   Training modeling formulas (T_comp / Mem / T_comm / Bubble)
│   ├── optimization/               #   Optimization strategies organized by bottleneck (comm / pipeline / memory / compute / moe)
│   ├── env/                        #   env tuning catalog (rccl / hsa / alloc / threading / presets)
│   ├── profiling/                  #   Data collection methods (preflight / env_probe / trace / ...)
│   ├── constraints/                #   Safety constraints (OOM est / config validity / env incompatibility matrix)
│   └── knowledge/                  #   LEARN write target (patterns / cases / anti-patterns)
│
├── prompts/                        # Role prompts (framework-agnostic Markdown)
│   ├── orchestrator.md             #   Orchestrator role: drive state machine + spawn subagent
│   └── worker/                     #   Per-Stage-Worker role prompts
│       ├── diagnose.md
│       ├── replan.md
│       ├── envsweep.md
│       ├── observe.md
│       ├── correctness_lite.md
│       └── preflight.md
│
├── tools/                          # Business actions (Python; exposed at CLI / MCP boundary)
│   ├── preflight.py                #   preflight.run / env_probe.run / env_probe.sweep
│   ├── profiler.py
│   ├── submit.py                   #   submit.run / submit.cancel
│   ├── observe.py                  #   observe.snapshot / observe.compare_loss
│   ├── constraint.py               #   constraint.check / check_env / estimate_mem / diagnose_failure
│   ├── state.py                    #   state.checkpoint / resume / trim / handoff
│   ├── subagent.py                 #   subagent.spawn protocol abstraction (concrete impl from integrations)
│   └── knowledge.py                #   knowledge.write
│
├── schemas/                        # Data contracts (JSON Schema source + Pydantic mirror)
│   ├── cluster_profile.schema.json
│   ├── plan.schema.json
│   ├── snapshot.schema.json
│   ├── diagnosis_report.schema.json
│   ├── env_sweep_result.schema.json
│   ├── target_vector.schema.json
│   ├── tuning_state.schema.json
│   ├── plan_graph.schema.json
│   ├── candidate_pool.schema.json
│   ├── subagent_result.schema.json
│   ├── failure_report.schema.json
│   └── pydantic/                   #   Generated from JSON Schema for Python convenience
│
├── state/                          # Runtime artifact directory (gitignored)
│   ├── cluster_profile.yaml        #   Reused across jobs (by version + age)
│   ├── tuning_state.yaml           #   Entry of stage-exit checkpoints
│   ├── plan_graph.yaml
│   ├── candidate_pool.yaml
│   └── checkpoints/
│       ├── r0/, r1/, r2/, ...      #   Per-round full snapshots (replayable)
│       └── handoff/                #   Orchestrator self-handoff landing point
│
├── integrations/                   # Thin adapters (optional, one per framework; not required)
│   ├── claude-code/
│   │   ├── README.md               #   How to run Pilot inside Claude Code
│   │   ├── CLAUDE.md               #   Entry prompt referencing skills / prompts / tools
│   │   └── mcp-server.py           #   Optional MCP wrapper around pilot/tools
│   ├── cursor/
│   │   ├── AGENTS.md               #   Cursor agent entry
│   │   └── rules/                  #   .cursor/rules/ template
│   └── codex/
│       └── README.md
│
└── agent/                          # Optional Python reference harness (fallback)
    │                               #   Production path uses integrations/<framework>/;
    │                               #   This directory is for: LLMs without native subagent /
    │                               #   headless long-running / reference impl.
    ├── orchestrator.py
    ├── subagent.py
    ├── schemas.py                  #   Local mirror of schemas/pydantic/
    ├── state.py
    ├── skills.py
    ├── orchestrator_tools.py
    ├── worker_tools.py
    └── prompts/
```

**Layer-responsibility cheat sheet**:

| Directory | Form | Tense | Who writes | Who reads |
|-----------|------|-------|------------|-----------|
| `skills/`       | Markdown (knowledge) | present (rules) | humans / LEARN | Agent / Worker |
| `prompts/`      | Markdown (roles) | present (identity) | humans | Agent framework injects |
| `tools/`        | Python (actions) | imperative | humans | Agent (via CLI / MCP) |
| `schemas/`      | JSON Schema (contracts) | static | humans | both sides validate |
| `state/`        | YAML (working memory) | past+present | Agent (via `state.*` tool) | full chain / audit |
| `integrations/` | Native entry per framework | static glue | humans | framework runtime |
| `agent/`        | Python (fallback harness) | imperative | humans | `python -m pilot.agent` |

**Versioning and gitignore**:
- `skills/` `prompts/` `tools/` `schemas/` `integrations/` `agent/` are all git-tracked.
- `state/` as a whole is in `.gitignore`; runtime artifacts archive externally by `session_id`.
- Regression / CI fixture `state/` lives at `tests/fixtures/state/`, separate from the runtime directory.

### 4.2 `skills/` detailed layout

```
skills/                                 # Single knowledge source (Agent reads)
│
├── workflow/                           # Main tuning flow (state-machine driven)
│   ├── SKILL.md                        # tuning loop overview (incl. outer/inner switch conditions)
│   ├── state_machine.md                # state set / transitions / reentry_when / on_fail
│   ├── orchestration.md                # Orchestrator ↔ Stage Worker protocol
│   │                                   #         + context hygiene rules + handoff protocol
│   ├── projection.md                   # modeling stage
│   ├── smoke.md                        # pre-flight self-check (tiny scale × 100 step)
│   ├── correctness.md                  # numerical gate (loss/grad vs reference)
│   ├── observe.md                      # observation data definition (snapshot schema)
│   ├── diagnose.md                     # bottleneck classification logic (incl. env_suspect / reentry triggers)
│   ├── plan.md                         # plan structure (incl. env.diff)
│   ├── plan_graph.md                   # solution-space maintenance (tree + frontier + exhausted_neighborhoods)
│   ├── replan.md                       # candidate generation + priority formula + derivation source
│   ├── axis_taxonomy.md                # axis categories: cluster_shared / weakly_local / strongly_local
│   ├── execution_strategy.md           # Champion-Challenger / Per-Plan / Successive Halving selection
│   ├── execute.md                      # execution and early stop
│   ├── envsweep.md                     # inner EnvSweep protocol (trigger / candidates / convergence)
│   ├── settle.md                       # convergence (greedy + soft rollback + explore round)
│   └── learn.md                        # protocol for writing best/failure cases back to knowledge/
│
├── execution-model/                    # Training modeling (core knowledge)
│   ├── SKILL.md                        # overview
│   ├── compute.md                      # T_comp(layers, mbs)
│   ├── memory.md                       # Mem(layers, mbs)
│   ├── communication.md                # T_comm / allreduce / alltoall
│   ├── pipeline.md                     # Bubble(pp, M)
│   ├── partition.md                    # layer partition / stage balance
│   └── examples.md                     # modeling examples (Dense / MoE)
│
├── optimization/                       # Optimization strategies organized by bottleneck
│   ├── SKILL.md                        # overall strategy
│   │
│   ├── comm/                           # Communication bottleneck
│   │   ├── SKILL.md                    # reduce_comm_pressure
│   │   ├── bucket.md                   # bucket tuning
│   │   ├── overlap.md                  # overlap optimization
│   │   ├── topology.md                 # cross-node vs intra-node
│   │   └── env.md                      # COMM_BOUND env candidates (→ env/rccl.md)
│   │
│   ├── pipeline/                       # Pipeline bottleneck
│   │   ├── SKILL.md                    # pipeline strategy
│   │   ├── vpp.md                      # VPP tuning
│   │   ├── microbatch.md               # MBS / GAS
│   │   └── balance.md                  # stage balance
│   │
│   ├── memory/                         # Memory bottleneck
│   │   ├── SKILL.md                    # memory strategy
│   │   ├── recompute.md                # activation recompute
│   │   ├── offload.md                  # CPU / NVMe offload
│   │   ├── fragmentation.md            # memory fragmentation
│   │   └── env.md                      # MEMORY_BOUND env candidates (→ env/alloc.md)
│   │
│   ├── compute/                        # Compute bottleneck
│   │   ├── SKILL.md                    # compute utilization
│   │   ├── mbs.md                      # mbs scaling
│   │   ├── parallel.md                 # dp/tp adjustments
│   │   ├── kernel.md                   # kernel-level hints
│   │   └── env.md                      # COMPUTE_BOUND env candidates (→ env/threading.md, hsa.md)
│   │
│   └── moe/                            # MoE-specific
│       ├── SKILL.md
│       ├── routing.md
│       ├── dispatch.md
│       └── load_balance.md
│
├── env/                                # env tuning catalog (source of truth, single point of maintenance)
│   ├── SKILL.md                        # env tuning overview / two-loop trigger conditions
│   ├── rccl.md                         # NCCL_*/RCCL_* full set (default / range / known pitfalls)
│   ├── hsa.md                          # HSA_*/HIP_*/GPU_MAX_HW_QUEUES
│   ├── alloc.md                        # PYTORCH_HIP_ALLOC_CONF / MALLOC_*
│   ├── threading.md                    # OMP_*/MKL_*/numactl
│   └── presets.md                      # per-cluster-type validated preset combinations
│
├── profiling/                          # Data collection methods
│   ├── SKILL.md
│   ├── preflight.md                    # cluster baseline
│   ├── gpu.md                          # GPU metrics
│   ├── network.md                      # IB / RCCL
│   ├── trace.md                        # timeline analysis
│   └── env_probe.md                    # env safe-probe protocol (connectivity → micro-bench → multi-node)
│
├── constraints/                        # Safety constraints
│   ├── SKILL.md
│   ├── oom.md                          # OOM estimation rules
│   ├── config.md                       # config validity
│   ├── validation.md                   # validation methods
│   └── env.md                          # env incompatibility matrix (mutually exclusive / dangerous combos)
│
└── knowledge/                          # Experience precipitation (LEARN stage write target)
    ├── SKILL.md                        # retrieval / writing protocol
    ├── patterns.md                     # general regularities ("MoE > 16 nodes always enable alltoall overlap")
    ├── cases.md                        # historical best-config case library (indexed by model × cluster)
    └── anti-patterns.md                # failure cases / known pitfalls
```

**Organization principle for env knowledge**:
- `skills/env/*.md` is the **only catalog** (each flag fully defined here once).
- `skills/optimization/{bottleneck}/env.md` only lists **"which flags to look at first under this bottleneck"**, referring to the catalog.
- This way adding a flag changes one catalog entry only — no scattered knowledge.

**What Skills do**: the Agent (Cursor / Claude / Codex / any tool-calling LLM) reads these Markdown files for domain knowledge instead of having knowledge hard-coded into code. This means:
- Knowledge can iterate independently of code.
- Different Agent runtimes share one knowledge base.
- New engineers can read Skills directly to understand the tuning logic.

---

## 5. Tool interfaces

Tools are Python functions called by the Agent via function calling:

| Tool | Function | Input | Output |
|------|----------|-------|--------|
| `preflight.run()` | Collect cluster hardware baseline | cluster_id | ClusterProfile |
| `env_probe.run()` | Probe / verify cluster-level env baseline (connectivity + RCCL micro-bench) | cluster_id, candidate_envs | EnvBaseline (written into ClusterProfile) |
| `env_probe.sweep()` | Inner EnvSweep: lock structure, scan env diff | base_plan, candidate_envs, max_steps | best_env_diff, results |
| `profiler.run()` | Single-node profiling | model_spec, configs | ProfilingResult |
| `submit.run()` | Submit a training job | plan, cluster | job_id |
| `submit.cancel()` | Cancel a job | job_id | status |
| `observe.snapshot()` | Collect runtime data | job_id | Snapshot |
| `observe.compare_loss()` | CORRECTNESS gate: align with reference loss | job_id, reference_curve | pass / drift, delta_pct |
| `constraint.check()` | Validate plan validity | plan, cluster | valid, violations |
| `constraint.check_env()` | Validate env combos (mutually exclusive / dangerous) | env_diff, baseline | valid, violations |
| `constraint.estimate_mem()` | Estimate memory | plan | mem_gb |
| `constraint.diagnose_failure()` | Failure attribution (OOM/hang/invalid → reason) | snapshot/error | failure_reason, suggested_transition |
| `state.checkpoint()` | Persist TuningState (called at each stage exit) | tuning_state | path |
| `state.resume()` | Resume from checkpoint | path | tuning_state |
| `state.trim()` | Orchestrator trim at stage exit, keep only pointer-class fields | tuning_state, keep_fields | trimmed_state |
| `state.handoff()` | Orchestrator self-handoff (when context near limit) | session_id | handoff_path |
| `subagent.spawn()` | Spawn a Stage Worker, isolate context, return structured result | stage, input_refs, skill_scope | SubagentResult |
| `knowledge.write()` | LEARN: write back best/failure cases | report, kind | written_paths |

The Agent decides "what to do" based on Skill knowledge, then calls Tools to execute.

---

## 6. Execution Model (core knowledge)

Defined in `skills/execution-model/*.md`, the Agent reads these formulas for estimation.

**Step time decomposition**:

```
T_step = T_comp + T_comm + T_bubble - T_overlap

T_comp = model_flops / (num_gpus × peak_tflops × efficiency)
  - efficiency comes from ClusterProfile + profiling data

T_comm = AllReduce(grad_size/dp) + AllToAll(moe_msg) + AllGather(zero_shard)
  - bandwidth looked up from ClusterProfile.rccl_baseline.<scope>.collectives.<coll>.roll_up.median_bw_gbs
  - <scope> ∈ {intra_node, inter_node, world} chosen by parallelism shape (TP/EP→intra_node, DP/FSDP→world)

T_bubble = (pp-1) / (pp-1 + M) × T_comp   # M = num_microbatch

T_overlap = min(T_comm_overlappable, T_comp_spare)
```

**Memory estimation**:

```
Mem = M_param + M_grad + M_optim + M_act + M_buffer

M_param = params / (tp × pp) × bytes_per_param
M_act   = f(seq, hidden, mbs, layers/pp, recompute)
```

---

## 7. Search-space maintenance & solution guarantees

> This section answers: "Why doesn't this greedy loop get stuck in local optima, repeat searches, or miss the truly best configuration?"
> This is the heart of Pilot's convergence design. Schema landings are §8.9 PlanGraph and §8.10 CandidatePool;
> here we discuss only **the mental model and mechanism**.

### 7.1 Two pitfalls of naive greed

The most intuitive approach: Settle maintains a baseline (current best); each round Re-Plan derives candidates from it, runs them, and picks the round's best as next baseline. Naive greed has two pitfalls:

| Pitfall | Symptom | Consequence |
|---------|---------|-------------|
| **Premature convergence** | Each round picks best and discards runners-up; but a runner-up may have opened a different bottleneck door (e.g. P_b is currently slightly worse but opens PIPELINE_BOUND with higher next-round potential). | Stuck forever in a local optimum |
| **Repeated search** | history dedupes by plan id only, but **derivation relations are lost**; cannot answer "which baseline did this config evolve from, and why didn't we keep going". | Repeatedly probes already-exhausted neighborhoods, wasting budget |

Pilot's response: **make the "solution space" explicit as a scored search tree (PlanGraph), make "candidates" explicit as a priority pool (CandidatePool), and stack 3 exploration mechanisms on top of greed.**

### 7.2 Mental model: search = maintaining a scored tree

What Re-Plan / Settle / Execute jointly maintain is not a chain but a tree:

```
                       root_plan (BASELINE)
                       tps=12000
                       /        |          \
                   P1            P2           P3
                tps=14200     tps=15800    tps=13100   ← Round 1 candidates
                              ★ champion       │
                              /     \          ↓
                            P4       P5      shelved
                          tps=17600  OOM    (kept in pool, may revive)
                          ★ champion │
                          /  \       ↓
                        ...  ...   dead       ← Round 2
```

Node states (four-class):

| State | Semantics | Can derive new candidates? |
|-------|-----------|---------------------------|
| **champion** | Current baseline; thick-arrow extension | Yes (default derivation source) |
| **shelved** | Did not win this round but alive; may revive later for backtrack | Yes (when exploring) |
| **dead** | OOM / invalid / numerical failure; permanent prune | No |
| **running** | Currently in Execute | — |

Derivation does not always start from champion — under certain conditions Re-Plan **derives from shelved** (see §7.5 Backtrack). This is the key to escaping local optima.

### 7.3 PlanGraph: solution-space maintenance

PlanGraph is a State-layer persisted structure, written at each stage exit. Listing only fields relevant to "solution guarantees" (full schema in §8.9):

| Field | Purpose |
|-------|---------|
| `champion` | Current baseline pointer |
| `champion_history` | Past champions (consecutive championships → high stability bonus) |
| `nodes[*].parent` | Derivation relation; lets audit replay "where it came from, why we stopped here" |
| `nodes[*].status` | completed / shelved / dead / running |
| `nodes[*].derived_axis` | Which axis moved relative to parent; novelty / neighborhood pruning depend on this |
| `frontier` | Currently derivable nodes (champion + shelved) |
| `exhausted_neighborhoods` | Already-explored radii; new candidates falling here are rejected to prevent repeated search |
| `metadata.rounds_since_promotion` | Stagnation counter |
| `metadata.rounds_since_explore` | Distance to last forced explore round |
| `metadata.dead_rate_in_subtree` | Per-node subtree failure rate; > 50% triggers backtrack |

### 7.4 Candidate generation: priority pool, not fixed K

What each Re-Plan outputs is not "K plans to run" but a **priority-weighted candidate pool**, with Strategy Select (see §8.10 selection) making the final cut:

```
priority(c) = expected_gain(c) × confidence(c) / est_cost(c)
            × novelty_bonus(c)              # exploring untouched axes +20%
            × parent_stability_bonus(c)     # derived from multi-time champion +10%
```

The pool mixes two sources:

- **exploit candidates**: derived from `champion`, fine-tuning along current best.
- **explore candidates**: derived from `shelved`, reviving previously-not-winning-but-viable branches.

Rejected candidates (hit `exhausted_neighborhoods` / failed `constraint.estimate_mem` / violated `constraint.check`) are also recorded in `candidate_pool.rejected[]` with traceable reasons.

### 7.5 Settle: greedy + soft rollback

Settle is not a simple "pick best as baseline." Beyond championship promotion/demotion, it decides whether to revive shelved, and whether to enter stagnation/explore mode:

```python
def settle(round_results, plan_graph):
    new_best = max(round_results, key=score)
    cur     = plan_graph.champion

    # Rule 1: significant gain → promote new champion
    if new_best.tps > cur.tps * (1 + ε_promote):     # ε_promote ≈ 2%
        plan_graph.champion = new_best.id
        plan_graph.set_status(cur, 'shelved')        # old champion not lost, goes to shelved
    # Rule 2: marginal gain → keep old champion, best goes to shelved for later revival
    elif new_best.tps > cur.tps:
        plan_graph.set_status(new_best.id, 'shelved')
    # Rule 3: all regressed → trigger backtrack (see §7.6)
    # Rule 4: gain < ε_stop for N consecutive rounds → enter stagnation mode
    if recent_gain_pct(plan_graph, n=2) < ε_stop:
        switch_to_explore_mode()                     # next Re-Plan derives from shelved
```

ε_promote / ε_stop default values are in `skills/pilot/workflow/settle.md`; specific numbers shift with TargetVector urgency (tighter budget → larger ε_promote → more conservative).

### 7.6 Three mechanisms to escape local optima

| Mechanism | Trigger | Action | Failure mode it prevents |
|-----------|---------|--------|--------------------------|
| **Backtrack (retreat to runner-up branch)** | Champion subtree dead-rate > 50% for 2 rounds; or stagnation persists 2 rounds | Pick the second-priority node from frontier as new champion, re-derive | Stuck in a dead end |
| **Diversification Bonus** | Every Re-Plan scoring | Bump priority weight for candidates that "cover untouched axes," avoid endless nudging on one axis | One axis fine-tuned to death, others ignored |
| **Periodic Exploration Round** | Every K rounds (default K=3) | Pool generated only from `shelved` + untouched axes, not from champion; this round may not gain or even regress | Long-term local optimum |

The 3 mechanisms are not mutually exclusive. Backtrack handles "sudden death," Diversification handles "boiled-frog stagnation," Periodic Exploration is the final safety net.

### 7.7 Convergence guarantees

On top of the above search structure, four static guarantees:

| Mechanism | Effect |
|-----------|--------|
| **Execution Model filtering** | Plans go through model estimation before Execute; high quality (low-confidence candidates pruned directly) |
| **PlanGraph + champion_history** | Champion is monotone or audit-traced rollback; no unjustified regression |
| **`exhausted_neighborhoods`** | Re-Plan excludes tried neighborhoods; search space shrinks per round |
| **Triple termination conditions** | TargetVector met / gain < ε_stop for 2 consecutive rounds / `max_rounds` or `budget.total_gpu_h` reached |

Cost upper bounds (typical Dense / MoE bring-up, single-node order; multi-node scales proportionally):

| Stage | Cost |
|-------|------|
| Preflight | ~30 min (first time; reused across jobs) |
| Projection | ~1 GPU·h (incl. single-node profiling) |
| Tuning Loop | ~3-5 GPU·h (incl. SMOKE / CORRECTNESS-LITE / optional EnvSweep) |
| **Total** | ~5-7 GPU·h |

> **Design cross-references**:
> - PlanGraph schema landing → §8.9
> - CandidatePool schema landing → §8.10
> - Promotion/demotion rules and stagnation thresholds → `skills/pilot/workflow/settle.md`
> - 7-step candidate generation → `skills/pilot/workflow/replan.md`
> - Explore/exploit switch policy → `skills/pilot/workflow/execution_strategy.md`
> - Axis exploration radius and pruning definition → `skills/pilot/workflow/axis_taxonomy.md`

---

## 8. Data structures (Schema)

Agent and Tool exchange structured data. Below are several core schemas.

### 8.1 ClusterProfile (Preflight output)

Schema 2.0: `rccl_baseline` is split into three scopes — `intra_node` (per-node columnar, parallel-measured), `inter_node` (single ring across 1 GPU/node), `world` (full N×gpus_per_node ring). Single-node mode populates `intra_node` only and leaves `inter_node` / `world` as `null`. Authoritative schema: `pilot/schemas/cluster_profile.schema.json`.

```yaml
schema_version: "2.0"
cluster_id: mi300x-16node
version: mi300x_8gpu-16node-v1
cluster_class: mi300x_8gpu
collected_at: 2026-04-15T10:00:00Z
nodes_total: 16
nodes_healthy: 16
gpus_per_node: 8

compute:
  peak_tflops_bf16: 1300         # measured GEMM peak (median across nodes)
  peak_tflops_fp8: 2600
  hbm_bandwidth_gbs: 5300
  hbm_capacity_gb: 192
  per_node_variance_pct: 1.4

interconnect:
  intra_node:
    type: xgmi
    bandwidth_gbs: 800            # echoed from rccl_baseline.intra_node AR @ max size
  inter_node:
    type: ib
    bandwidth_gbs: 400            # per-GPU effective bandwidth
    uniformity: uniform

rccl_baseline:
  intra_node:                     # per-node, parallel-measured (each node runs local 8-GPU PG)
    world_size: 8
    nnodes_measured: 16
    collectives:
      allreduce:
        sizes_mb: [1, 16, 64, 256]
        per_node_bw_gbs:
          smc01: [25.1, 182.2, 306.3, 377.2]
          smc02: [24.9, 181.0, 305.5, 374.0]
          # ... 14 more nodes
          smc16: [25.0, 181.5, 305.8, 376.1]
        per_node_latency_us:
          smc01: [73.1, 161.1, 383.4, 1245.3]
          # ...
        roll_up:
          median_bw_gbs: [25.0, 181.5, 305.8, 376.1]
          min_bw_gbs:    [22.3, 175.0, 290.0, 281.5]   # smc04 slow
          max_bw_gbs:    [25.2, 182.2, 306.3, 377.2]
          stddev_pct:    [3.2,  1.1,   1.5,   9.8]
          slow_nodes_at_max_size: [smc04]
      allgather:      { ... same shape ... }
      reduce_scatter: { ... }
      broadcast:      { ... }
      alltoall:       { ... }

  inter_node:                     # 1 GPU/node × 16 nodes single ring (isolates IB)
    world_size: 16
    nnodes: 16
    nproc_per_node: 1
    collectives:
      allreduce:
        sizes_mb:   [1, 16, 64, 256]
        bw_gbs:     [3.2, 18.5, 23.1, 25.4]
        latency_us: [580, 1580, 5060, 18400]
      # allgather / reduce_scatter / broadcast similar; alltoall typically omitted at this scope

  world:                          # full 128-GPU ring (the actual training topology)
    world_size: 128
    nnodes: 16
    nproc_per_node: 8
    collectives:
      allreduce:
        sizes_mb:   [1, 16, 64, 256]
        bw_gbs:     [1.8, 11.4, 19.7, 22.3]
        latency_us: [1020, 2580, 5940, 20800]
      # allgather / reduce_scatter / broadcast / alltoall similar

env_baseline:                      # cluster-level env golden default (probed once / reused across jobs)
  version: mi300x_8gpu-16node-v1
  status: validated                # validated / tentative
  rccl:
    NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
    NCCL_NET_GDR_LEVEL: 4
    NCCL_IB_GID_INDEX: 3
    NCCL_SOCKET_IFNAME: bond0
  hsa:
    HSA_FORCE_FINE_GRAIN_PCIE: 1
    GPU_MAX_HW_QUEUES: 2
  alloc:
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True"
  threading:
    OMP_NUM_THREADS: 8
```

**Reading the table** (consumer guidance for PROJECTION / SCHEDULER):

- For `intra_node`, always read from `roll_up.median_bw_gbs` for typical-case planning, `roll_up.min_bw_gbs` for worst-case bounds. Single-node mode: `min == median == max` and `stddev_pct == 0`.
- For `inter_node` and `world`, read directly from `bw_gbs` (single ring; no per-node fan-out).
- `slow_nodes_at_max_size` is the canonical place to source blacklist proposals; PREFLIGHT does not auto-blacklist, it only flags.

### 8.2 Plan (configuration to be executed)

```yaml
plan_id: r3_p2
parent_baseline: r2_p1            # last round's best
parallelism:
  tp: 4
  pp: 2
  dp: 16
  ep: 8
  vpp: 2
runtime:
  mbs: 2
  gbs: 1024
  recompute: selective
comm:
  bucket_size_mb: 64
  overlap: true
env:                               # only diff against env_baseline
  baseline_ref: mi300x-16node-v3
  diff:
    NCCL_MIN_NCHANNELS: 16
    NCCL_BUFFSIZE: 16777216
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:512"
    RCCL_MSCCL_ENABLE: 1
predicted:                         # Execution Model estimate
  tps: 18500
  mem_peak_gb: 165
  comm_ratio: 0.22
generated_by:
  bottleneck: PIPELINE_BOUND
  strategy: skills/optimization/pipeline/vpp.md
```

### 8.3 Snapshot (Observe output)

```yaml
run_id: job_8842
plan_id: r3_p2
collected_at: 2026-04-15T11:23:00Z
metrics:
  tps: 17800
  step_time_ms: 412
  comm_ratio: 0.18
  bubble_ratio: 0.09
  overlap_ratio: 0.61
  mem_peak_gb: 158
  gpu_util_avg: 0.74
status: completed                  # / early_stopped / oom / failed
warnings: []
```

### 8.4 DiagnosisReport (Diagnose output)

```yaml
snapshot_id: job_8842
bottleneck: COMPUTE_BOUND          # COMM / PIPELINE / MEMORY / COMPUTE
confidence: 0.85
evidence:
  - "comm_ratio=0.18 < threshold 0.25"
  - "bubble_ratio=0.09 < threshold 0.15"
  - "gpu_util=0.74 vs baseline peak 0.92 → headroom"
recommended_skills:
  - skills/optimization/compute/mbs.md
  - skills/optimization/compute/parallel.md
env_suspect:                       # if any → trigger EnvSweep (before structural changes)
  - flag: NCCL_BUFFSIZE
    reason: "comm_ratio=0.32 but msg_size_p95=4MB; buffer too small"
    hint: skills/env/rccl.md#buffsize
  - flag: PYTORCH_HIP_ALLOC_CONF
    reason: "mem_reserved/mem_alloc=1.45 → fragmentation high"
    hint: skills/env/alloc.md#expandable-segments
candidate_axes:                    # for Re-Plan: each axis type drives how to search
  - {axis: vpp,                  type: structural,     candidates: [1, 2, 4]}
  - {axis: mbs,                  type: structural,     candidates: [1, 2]}
  - {axis: NCCL_BUFFSIZE,        type: strongly_local, candidates: [8M, 16M, 32M]}
  - {axis: PYTORCH_HIP_ALLOC_CONF, type: weakly_local, candidates: [seg, no-seg]}
  - {axis: NCCL_IB_HCA,          type: cluster_shared, candidates: [baseline]}
suggested_strategy: Per-Plan       # Champion-Challenger / Per-Plan / Successive_Halving
  rationale: "strongly_local axis NCCL_BUFFSIZE present, model confidence>0.7"
```

### 8.5 EnvSweepResult (inner-loop output)

```yaml
sweep_id: r3_envsweep_1
parent_plan: r3_p2                 # locked structural baseline
trigger: COMM_BOUND
candidates:                        # env combos tried this round
  - env_diff: {NCCL_BUFFSIZE: 8388608}
    tps: 17900
    delta_pct: +0.6
  - env_diff: {NCCL_BUFFSIZE: 16777216, NCCL_MIN_NCHANNELS: 16}
    tps: 18650
    delta_pct: +4.8                # ← best
  - env_diff: {NCCL_BUFFSIZE: 33554432}
    tps: 17600
    delta_pct: -1.1
selected_diff:                     # merged into baseline.env.diff
  NCCL_BUFFSIZE: 16777216
  NCCL_MIN_NCHANNELS: 16
cost_gpu_h: 0.3
```

### 8.6 TargetVector (user input / Settle judgment basis)

Promote "meeting target" from single-TPS to multi-objective. Settle's stop condition = all `constraints` met AND `primary` no longer meaningfully improving / `budget` reached.

```yaml
target:
  primary: tps                     # also supports per_token_cost_usd / time_to_loss
  constraints:                     # hard constraints
    - mem_peak_gb <= 180
    - per_token_cost_usd <= 1.2e-7
    - correctness.loss_delta_pct <= 1.0
  budget:
    total_gpu_h: 10
    max_rounds: 5
    wallclock_h: 24
  preferences:                     # soft preferences (tiebreaker)
    prefer_lower_pp: true          # reduce bubble risk
    prefer_known_env_presets: true # prefer precipitated presets
```

### 8.7 TuningState (persisted at each stage exit)

Lets the Agent resume from any stage after interruption; also the unit of audit / replay.

```yaml
session_id: pilot_run_20260418_a3
current_stage: OPTIMIZE_LOOP.OBSERVE   # state-machine locator
stage_history:
  - {stage: PREFLIGHT,   exit: success, at: ...}
  - {stage: PROJECTION,  exit: success, at: ...}
  - {stage: SMOKE,       exit: success, at: ...}
  - {stage: BASELINE,    exit: success, at: ...}
  - {stage: CORRECTNESS, exit: success, at: ...}
  - {stage: OPTIMIZE_LOOP.SETTLE, exit: continue, round: 2, at: ...}

cluster_profile_ref: mi300x-16node-v3   # references ClusterProfile version
target: <TargetVector>
baseline: <Plan>                         # current best
history:                                 # for Re-Plan dedup
  - {plan_id: r0_p0, tps: 12000, status: completed}
  - {plan_id: r1_p1, tps: 14200, status: completed}
  - {plan_id: r2_p3, tps: 0,     status: oom}

budget_used:
  gpu_h: 4.2
  rounds: 2
  wallclock_h: 6.1

reentry_log:                             # jump-back records (audit)
  - {from: DIAGNOSE, to: PREFLIGHT, reason: cluster_profile.age>7d, at: ...}
```

### 8.8 FailureReport (Guardrail attribution output)

Generated by `constraint.diagnose_failure()`, decides which recovery path the Guardrail takes (see §12).

```yaml
run_id: job_8842
failure_kind: OOM                  # OOM / HANG / INVALID_CONFIG / NUMERICAL / CLUSTER
root_cause: "act_mem_underestimated"
evidence:
  - "predicted mem_peak=170GB, actual=192GB at step 47"
  - "no recompute on layers 12-23"
suggested_transition:              # hint for the state machine's failure path
  to: REPLAN
  hint: "mark plan dead, try recompute=full on rear half"
counts_against_budget: false       # OOM/invalid does not consume round budget
```

### 8.9 PlanGraph (solution-space maintenance)

Promote `history` from a flat list to a derivation tree. Re-Plan's source selection, Settle's championship promotion/demotion, and prevention of local optima all rely on this structure.

```yaml
plan_graph:
  champion: r2_p4                  # plan_id of current baseline
  champion_history:                # past champions (consecutive championships → high stability)
    - {round: 0, id: r0_p0}
    - {round: 1, id: r1_p2}
    - {round: 2, id: r2_p4}

  nodes:                           # every tried plan is a node
    r0_p0:
      parent: null
      status: completed            # completed / shelved / dead / running
      tps: 12000
      bottleneck: COMM_BOUND
      ★champion_at: [0]
      derived_axis: null           # root
    r1_p2:
      parent: r0_p0
      status: completed
      tps: 15800
      bottleneck: PIPELINE_BOUND
      ★champion_at: [1]
      derived_axis: {NCCL_overlap: enable}
    r1_p3:
      parent: r0_p0
      status: shelved              # didn't win but alive; revive next round
      tps: 13100
      reason: "ep reduction hurt capacity but still viable; kept as backtrack"
      derived_axis: {ep: 8→4}
    r2_p4:
      parent: r1_p2
      status: completed
      tps: 17600
      ★champion_at: [2]
      derived_axis: {vpp: 1→2}
    r2_p5:
      parent: r1_p2
      status: dead                 # permanent prune, no further consideration
      reason: OOM
      predicted_mem: 165
      actual_mem: 192
      derived_axis: {mbs: 1→2}

  frontier:                        # currently derivable active nodes (champion + shelved)
    - r2_p4    # champion
    - r1_p3    # shelved
    - r1_p1    # shelved

  exhausted_neighborhoods:         # explored radii; new candidates falling here are rejected
    - {around: r0_p0, axis: bucket_size_mb, tried: [16, 32, 64, 128]}
    - {around: r1_p2, axis: vpp,            tried: [1, 2]}
    - {around: r2_p4, axis: NCCL_BUFFSIZE,  tried: [8M, 16M]}

  metadata:
    rounds_since_promotion: 0      # for stagnation detection
    rounds_since_explore: 2        # rounds since last explore round (>=K=3 triggers)
    dead_rate_in_subtree:          # subtree failure rate (>50% triggers backtrack)
      r2_p4: 0.50                  # 1 of 2 children dead
```

### 8.10 CandidatePool (Re-Plan output)

Re-Plan no longer outputs "K plans to run directly"; instead it outputs a **priority-weighted pool** that Strategy Select finalizes.

```yaml
candidate_pool:
  generated_at: round_3
  derived_from:                    # derivation source (default champion; on stagnation: shelved)
    primary: r2_p4
    secondary: [r1_p3]             # if explore round enabled
  policy: explore_exploit          # exploit / explore / explore_exploit

  candidates:
    # exploit candidates (derived from champion)
    - id: r3_p1
      parent: r2_p4
      axis_change: {mbs: 1→2}
      predicted_tps: 18800
      predicted_gain_pct: 6.8
      confidence: 0.82             # Execution Model's confidence in this prediction
      est_cost_gpu_h: 0.4
      novelty_bonus: 1.0           # this axis untouched around r2_p4
      stability_bonus: 1.10        # derived from 2-round consecutive champion
      priority: 1.55               # = gain × confidence × bonuses / cost

    - id: r3_p2
      parent: r2_p4
      axis_change: {NCCL_BUFFSIZE: 8M→16M}
      predicted_tps: 18200
      confidence: 0.65
      est_cost_gpu_h: 0.3
      priority: 0.92

    # explore candidates (derived from shelved, escape local optima)
    - id: r3_p3
      parent: r1_p3
      axis_change: {ep: 4→8, with: alltoall_overlap}
      predicted_tps: 17900
      confidence: 0.55
      est_cost_gpu_h: 0.5
      novelty_bonus: 1.20          # reviving shelved earns explore bonus
      priority: 0.78
      tag: explore

  selection:                       # final Strategy Select output
    strategy: Champion-Challenger  # from axis_taxonomy decision
    pick_top_k: 3
    selected: [r3_p1, r3_p2, r3_p3]
    rejected:                      # rejected candidates also recorded for audit
      - {id: r3_p_x, axis_change: {bucket_size_mb: 32→64},
         reason: "exhausted_neighborhoods hit (around r0_p0)"}
      - {id: r3_p_y, axis_change: {pp: 2→4},
         reason: "constraint.estimate_mem over budget (210GB > 192GB cap)"}

  priority_formula: |
    priority(c) = expected_gain(c) × confidence(c) / est_cost(c)
                × novelty_bonus(c)              # untouched axis +20%
                × parent_stability_bonus(c)     # multi-time-champion parent +10%
```

### 8.11 SubagentResult (Stage Worker → Orchestrator)

The **only structured payload** the Stage Worker returns to the Orchestrator after completion. The Orchestrator parses only this object to update pointers; it absorbs no Worker intermediate trace. Raw details (Snapshot / DiagnosisReport / CandidatePool / EnvSweepResult) have already been written to the State Layer by the Worker; here we only reference them.

```yaml
subagent_result:
  stage: RE_PLAN                   # DIAGNOSE / RE_PLAN / ENV_SWEEP / CORRECTNESS_LITE / ...
  worker_id: sw_r3_replan_20260421_112345
  status: success                  # success / failed / escalate

  # State-layer references for structured artifacts (Orchestrator does not read content)
  artifacts:
    - kind: CandidatePool
      ref: state/round_3/candidate_pool.yaml
      size_bytes: 4280

  # One-line summary for Orchestrator (must be < 200 tokens)
  summary:
    headline: "COMPUTE_BOUND, 3 exploit + 1 explore candidates, top priority=1.55"
    key_metrics:
      selected_count: 3
      rejected_count: 2
      top_priority: 1.55
      est_cost_gpu_h: 1.2

  # Suggested state-machine transition (Orchestrator decides finally)
  suggested_transition:
    to: EXECUTE
    reason: "candidate_pool.selected non-empty, budget ok"

  # Resource usage (for evaluation and budget tracking)
  cost:
    wallclock_s: 12.4
    tokens_used: 18420           # LLM tokens consumed by this Worker invocation
    tool_calls: 7

  # Only present if status=failed
  failure:
    kind: SKILL_MISSING          # / TOOL_ERROR / CONSTRAINT_VIOLATION / ...
    message: "optimization/moe/dispatch.md not found"
    escalate_to_orchestrator: true
```

**Orchestrator-side processing protocol** (in `skills/workflow/orchestration.md`):

1. Read only `summary` + `suggested_transition` + `status`.
2. Decide based on `suggested_transition` referencing `state_machine.md`.
3. Append `summary.headline` to `TuningState.stage_history` (one-line log).
4. Discard the `SubagentResult` object itself (do not let the full YAML enter context).
5. Immediately `state.checkpoint()` + `state.trim()`.

---

## 9. Full iteration example

A real data flow through one Tuning Loop (MoE 16 nodes):

```
Round 0 (Baseline)
  Plan:     {tp:2, pp:4, ep:8, mbs:1, recompute:full}
  Snapshot: tps=12000, comm_ratio=0.38, bubble=0.12, mem=140GB
  Diagnose: COMM_BOUND (alltoall accounts for 28%)
  Re-Plan:  read skills/optimization/comm/ → generate 3 candidates
            P1: bucket_size 16→64 MB
            P2: enable alltoall overlap
            P3: ep 8→4 (reduce cross-node traffic)

Round 1 (outer)
  Execute:  3 plans run in parallel (each 50 step early-stop)
  Results:
    P1: tps=14200 (+18%)
    P2: tps=15800 (+32%)  ← best
    P3: tps=13100 (+9%, ep reduction lowered expert capacity)
  Settle:   pick P2 as new baseline

Round 1' (inner EnvSweep, since Diagnose output env_suspect=NCCL_BUFFSIZE)
  Sweep:    lock P2 structure, scan 3 candidates (30 step)
            E1: NCCL_BUFFSIZE=8M             → tps=15900 (+0.6%)
            E2: NCCL_BUFFSIZE=16M+MIN_NCH=16 → tps=16550 (+4.7%)  ← best
            E3: NCCL_BUFFSIZE=32M            → tps=15600 (-1.3%)
  Merge:    P2.env.diff += {NCCL_BUFFSIZE:16M, NCCL_MIN_NCHANNELS:16}
            new baseline tps=16550
  Cost:     0.3 GPU·h
  Diagnose: PIPELINE_BOUND (bubble rose to 0.18)
  Re-Plan:  read skills/optimization/pipeline/ → generate 2 candidates
            P4: vpp 1→2
            P5: mbs 1→2

Round 2
  Execute:  P4, P5
  Results:
    P4: tps=17600 (+11%)  ← best
    P5: OOM → mark dead (predicted_mem error > 15%)
  Settle:   promote P4 to new champion; P2 (old champion) → shelved
  PlanGraph:
    champion: P4
    shelved:  [P1, P3, P2]
    dead:     [P5]
    exhausted_neighborhoods: {around=P2, axis=NCCL_BUFFSIZE, tried=[8M,16M,32M]}
  Diagnose: COMPUTE_BOUND

Round 3 (CandidatePool demo)
  Re-Plan:  derive from P4 + 1 explore candidate (rounds_since_explore=2)
    Pool:
      - P6: P4 → mbs:2→3            priority=1.42
      - P7: P4 → tp:2→4             priority=0.95
      - P8: P3 revived → ep:4→8 + AT-overlap  priority=0.78 (tag=explore)
    Strategy Select: Champion-Challenger, top-3 → [P6, P7, P8]
    Rejected: P_x (vpp:2→4, exhausted around P4)
  Execute (50 step):
    P6: tps=18100 (+2.8%)  ← best
    P7: OOM (TP increase pushed activation memory) → dead
    P8: tps=15200 → not above champion, back to shelved
  Settle:   marginal gain (+2.8% < ε_promote=2%×1.5), champion still P4, P6 → shelved
  rounds_since_promotion: 1

Round 4
  Re-Plan:  rounds_since_promotion=1, still exploit; derive P9 from P4 (recompute=selective)
  Execute:  P9: tps=18400 (+1.7%)
  Settle:   gain<2%, 2 consecutive rounds without significant gain → enter stagnation
  Stop check: no priority>1.0 candidate in frontier → STOP

Final: champion=P9, tps=18400 (1.53× over baseline)
       PlanGraph fully replayable for all derivation paths
Time:  ~4.5 GPU·h (Tuning Loop part)
```

> **Value of PlanGraph**: search is an explicit tree, not a black box; every step is auditable, replayable, and migratable across jobs (next session on the same model + cluster directly loads `exhausted_neighborhoods` and avoids redundant exploration).

---

## 10. Evaluation metrics

How to judge whether the Pilot system itself works well:

| Dimension | Metric | Target |
|-----------|--------|--------|
| **Effectiveness** | Final TPS / baseline TPS | ≥ 1.3× |
| **Effectiveness** | Gap to human best-known config | ≥ 90% |
| **Efficiency** | Total GPU·h cost | ≤ 10 GPU·h |
| **Efficiency** | Convergence rounds | ≤ 5 rounds |
| **Reliability** | Success rate (no OOM / hang) | ≥ 95% |
| **Reliability** | Cost Model estimation error | ≤ 20% |
| **Explainability** | Each decision traceable to a Skill | 100% |
| **Scalability** | Orchestrator steady context / round | ≤ 2K tokens |
| **Scalability** | Stage Worker single peak context | ≤ 30K tokens |
| **Scalability** | Max rounds without handoff | ≥ 20 |
| **Scalability** | SubagentResult context injection into Orchestrator | ≤ 200 tokens / stage |

**Evaluation method**:
1. **Regression test set**: maintain 5-10 representative scenarios (Dense 7B/70B, MoE 8x7B, etc.); run automatically after each system update.
2. **Controlled experiment**: same model + cluster tuned by Pilot vs. by senior engineer; compare results.
3. **Ablation**: turn off Execution Model / turn off Cost Model filtering; measure how much Tuning Loop degrades.

---

## 11. Integration with existing systems

Pilot does not reinvent wheels; it composes existing capabilities:

| Existing system | Integration |
|-----------------|-------------|
| **Primus** | `submit.run()` calls Primus CLI to submit jobs; Preflight reuses Primus hardware detection |
| **Megatron / TorchTitan** | Tuning IR generates per-backend configs; Agent does not need to know specific framework |
| **WandB / Prometheus** | `observe.snapshot()` pulls metrics from WandB API |
| **rocprof / RCCL profiler** | `profiler.run()` wraps rocprof commands, outputs structured trace |
| **Slurm** | `submit.run()` internally generates sbatch scripts, reusing existing scheduling |

The benefit of this integration style: **if Pilot dies, all underlying tools remain usable**; existing engineers' muscle memory is preserved.

---

## 12. Guardrails

### 12.1 Preventive constraints

| Mechanism | Location | Effect |
|-----------|----------|--------|
| OOM estimation | `constraint.estimate_mem()` | Auto-filter high-risk configs |
| Numerical gate | `correctness.md` + `observe.compare_loss()` | Loss curve aligned with reference after BASELINE / every N rounds |
| Smoke pre-flight check | `smoke.md` + `submit.run(tiny)` | tiny scale × 100 step verifies it starts; failure → PROJECTION |
| Early stop | `observe.snapshot()` | OOM / throughput regression → terminate immediately |
| History dedup | Settle logic | Do not retry already-failed configs |
| max_rounds / budget | Settle + TargetVector.budget | Hard cost cap (GPU·h / wallclock) |
| env connectivity fail-fast | `env_probe.run()` | Wrong NCCL_IB_HCA etc. blow up within 30s; not in baseline |
| env incompatibility matrix | `constraint.check_env()` | Reject dangerous/exclusive combos (e.g. MSCCL × certain GDR modes) |
| EnvSweep per-call cap | inner loop | Each ≤ 5 flags, ≤ 8 combos, ≤ 50 step |
| env baseline versioning | `ClusterProfile.env_baseline.version` | Cluster upgrade / driver change triggers re-probe |
| env diff-only recording | `Plan.env.diff` | Audit, reproduce, rollback all based on diff |
| Audit log + persistence | `state.checkpoint()` at each stage exit | Replayable, interruptible-resumable |
| **Context hygiene** | `state.trim()` at each stage exit | Orchestrator context steady; details only in State Layer |
| **Subagent isolation** | `subagent.spawn()` + `orchestration.md` | Stage-level context does not pollute Orchestrator |
| **SubagentResult strict bound** | `orchestration.md` validation | Returned object's `summary` ≤ 200 tokens; over-limit rejected |
| **Context overflow path** | Orchestrator self-check `ctx_tokens > 0.5×window` | Auto `state.handoff()` without losing progress |
| **Worker budget cap** | `subagent.spawn(max_tokens=30K)` | Worker over-budget forces early-return + escalate |

### 12.2 Failure paths (state-machine `on_fail` / `reentry_when` realization)

After a Guardrail fires, `constraint.diagnose_failure()` outputs a `FailureReport` (§8.8) driving the state machine through one of these transitions. **`counts_against_budget=false` means no round-budget consumption**.

| Failure class | Attribution | Transition target | Note |
|---------------|-------------|-------------------|------|
| `OOM` | Memory over estimate | → `REPLAN` | Mark plan dead, force recompute / smaller mbs; not counted |
| `HANG` (NCCL/IB) | Comm stuck > timeout | → `PREFLIGHT` (env_probe) | env baseline suspected stale; re-probe; not counted |
| `INVALID_CONFIG` | constraint.check failed | → `REPLAN` | Drop only this plan, no Execute; not counted |
| `NUMERICAL` | loss drift / NaN | → `ABORT` + escalate | Numerical correctness broken, stop |
| `CLUSTER` | Node down / driver error | → `PREFLIGHT` | ClusterProfile marked stale; re-probe |
| `BUDGET_EXCEEDED` | gpu_h / wallclock exceeded | → `REPORT` | End early with current best |
| `STRUCTURAL_INVALIDATION` | Model/data spec changed | → `PROJECTION` | Re-model; history invalidated |
| **`CONTEXT_OVERFLOW`** | Orchestrator context > 0.5 × window | → `HANDOFF` | `state.handoff()` writes landing point; fresh Orchestrator `state.resume()` continues; not counted |
| **`SUBAGENT_FAILED`** | Worker status=failed / over budget | → `REPLAN` or `ABORT` | Inspect `SubagentResult.failure.kind`; recoverable → REPLAN, else escalate |
| `UNKNOWN` | Catch-all | → `ABORT` + escalate | Don't guess; defer to humans |

---

## 13. Context Management & Multi-Agent Orchestration

> Answer to a concrete question: why does letting a single Agent run the whole Tuning Loop start "forgetting" or "repeating mistakes" after about 5 rounds? The root cause is not LLM window size; it's that "state-machine progression" (small but long-lived) and "stage reasoning" (large but short-lived) share the same session, and context grows linearly with rounds into the attention-dilution zone. This chapter gives the protocol.
>
> **Ownership recap of this chapter** (see Scope & Positioning):
>
> | Strategy | Pilot owns | Agent framework owns |
> |----------|------------|----------------------|
> | A · State-first protocol | ✓ (`skills/workflow/orchestration.md` rules + schema size constraints) | executes |
> | B · Subagent isolation   | only the boundary table (§13.2) | ✓ (Task tool / subagent spawning) |
> | C · Session handoff      | only the handoff file format (§8.7) | ✓ (process management / session restart) |
>
> The "Orchestrator / Stage Worker behavior" described here is entirely **role specification**, landing in `prompts/` and `skills/workflow/orchestration.md`, realized by the framework's native session/subagent mechanism.

### 13.1 Problem: why a single Agent fails in long loops

Per-round context delta in a single Agent running the full Tuning Loop:

| Source | Delta (tokens) | Note |
|--------|---------------|------|
| Skill recall (optimization/comm/* + env/*) | ~3–5K | Per-round on-demand reference even if loaded |
| Snapshot YAML (§8.3) | ~0.5K | Per plan |
| DiagnosisReport (§8.4) | ~1K | Per round |
| CandidatePool (§8.10) | ~2K | Per round, includes rejected records |
| EnvSweepResult (§8.5, optional) | ~0.8K | When triggered |
| Agent free-form reasoning trace | ~3–5K | More with complexity |
| **Per-round subtotal** | **10–15K** | — |

5 rounds = 50–75K, 10 rounds = 100–150K. With a 200K context window there's apparent headroom, but:
- LLM attention quality drops noticeably after 60% utilization.
- Older Snapshot and CandidatePool are nearly useless to the present decision yet keep occupying space.
- A single Skill re-read pastes the entire .md back into context, amplifying again.

**Conclusion**: this is an architecture problem, not a window problem. Bigger windows only delay the failure.

### 13.2 Three-tier solution (light to heavy)

#### Strategy A: State-first protocol (must do, zero new components)

**Principle**: the State Layer (already a "single source of truth" in §2) is the Agent's working memory; context only stores pointers.

Mandatory rules (in `skills/workflow/orchestration.md`):

1. **Each stage exit must `state.checkpoint()` + `state.trim()`**.
2. **Before next stage, Orchestrator context retains only**:
   ```yaml
   {session_id, current_stage, round_id,
    champion_id, budget_used, last_decision_summary}
   ```
   Last round's Snapshot / CandidatePool / DiagnosisReport are discarded; if needed, slices are read back from the State Layer by reference.
3. **Skill is not re-loaded**: Skills loaded earlier in the same session are not re-read by the Orchestrator; the Stage Worker reads only Skills in its own scope.

**Effect**: Orchestrator steady context fixed < 2K tokens; does not grow with rounds.

#### Strategy B: Subagent isolation (recommended, core mechanism)

**Principle**: turn context-heavy stages into independent subagents — used and destroyed; the Orchestrator sees only `SubagentResult` summary.

**Subagent boundary table** (which stages must be subagents, which need not):

| Stage | Subagent? | Reason | Typical input | Typical output |
|-------|-----------|--------|---------------|----------------|
| **Preflight** | Yes | First time ~30 min; large micro-bench data; cross-job reuse only needs ClusterProfile | `cluster_id` | `ClusterProfile` |
| **Projection** | Yes | Reads entire execution-model/* subtree + profiling data | `model_spec, cluster_profile_ref` | `InitialPlans` ref |
| **Observe** | Yes (one per plan) | Naturally parallelizable when running multiple plans | `run_id` | `Snapshot` ref |
| **Diagnose** | Yes | Reads diagnose.md + execution-model/* + possibly profiling trace | `snapshot_id` | `DiagnosisReport` ref |
| **Re-Plan** | Yes (heaviest) | optimization/* subtree + env catalog + axis_taxonomy | `diagnosis_report_ref, plan_graph_ref` | `CandidatePool` ref |
| **EnvSweep (inner)** | Yes | Naturally an independent subloop; per-call ≤ 8 combos × 50 step | `base_plan_ref, candidate_envs` | `EnvSweepResult` ref |
| **Correctness-Lite** | Yes | Pulls reference curve for comparison | `run_id, reference_ref` | `{pass, delta_pct}` |
| **Execute** | No | Mostly `submit.run()` + polling, light logic | — | — |
| **Settle** | No | Pure numerical judgment + rule matching, very small | — | — |
| **State-machine progression** | No (Orchestrator's own job) | — | — | — |

**Boundary design principles**:

- **Cut where Skill span is largest**: Re-Plan / Diagnose / EnvSweep are typical "read multiple Skills + write one product" — best fit for subagents.
- **Granularity ceiling**: don't do "one subagent per Skill" — RPC disaster.
- **Granularity floor**: don't do "one subagent for the entire OPTIMIZE_LOOP" — that regresses to single-Agent absorbing all details.

**Calling protocol** (specified in `orchestration.md`):

```python
def orchestrator_step():
    state = state.resume(checkpoint_path)           # < 500 tokens
    next_stage = decide_next_stage(state)           # by state_machine.md

    result = subagent.spawn(
        stage = next_stage,
        input_refs = state.relevant_refs(next_stage),  # pointers only
        skill_scope = SKILL_SCOPES[next_stage],        # restrict Worker reading
        max_tokens = 30_000,                           # Worker budget cap
    )  # returns SubagentResult, §8.11

    if result.status == 'failed':
        return handle_failure(result.failure)

    state.apply(result.summary, result.suggested_transition)
    state.checkpoint()
    state.trim(keep=['session_id', 'current_stage', 'round_id',
                     'champion_id', 'budget_used'])
    return state
```

#### Strategy C: Session handoff (fallback, must-have for long-time tuning)

**Principle**: the Orchestrator itself is replaceable. Even with strategies A + B in effect, 20+ rounds or multi-day jobs still accumulate; handoff lets the Orchestrator "switch shifts" with itself.

**Triggers**:
- `context_tokens > 0.5 × window` (warning).
- `context_tokens > 0.75 × window` (forced).
- Proactive: every K rounds (K=10) periodic handoff for long-term stability.

**Protocol**:

```python
def handoff_if_needed():
    if self.ctx_tokens < 0.5 * window:
        return
    handoff_path = state.handoff(
        session_id = self.session_id,
        reason = 'context_pressure',
        next_action_hint = self.pending_decision,
    )
    # New Orchestrator launched, reads handoff_path to restore
    spawn_new_orchestrator(resume_from=handoff_path)
    sys.exit(0)
```

**Relation to §12.2**: the `CONTEXT_OVERFLOW` route uses this; not counted against round budget.

### 13.3 How the three strategies cooperate

```
┌─────────────────────────────────────────────────────────────────┐
│ Strategy A  State-first protocol                                 │
│   Each stage must checkpoint + trim → Orchestrator ctx steady    │
│   Skip it: per-round context cannot stabilize, B's many subagents│
│            can't save it                                         │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ prerequisite
                            │
┌─────────────────────────────────────────────────────────────────┐
│ Strategy B  Subagent isolation                                   │
│   Re-Plan / Diagnose / EnvSweep extracted → Worker ctx disposed  │
│   Skip it: Orchestrator forced to absorb Worker trace, A's       │
│            pointers polluted                                     │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ fallback supplement
                            │
┌─────────────────────────────────────────────────────────────────┐
│ Strategy C  Session handoff                                      │
│   Orchestrator self-handoff, last line of defense for very long  │
│   Skip it: 20+ rounds / multi-day jobs may still hit a wall      │
│            on incidentals (errors, diagnosis loops)              │
└─────────────────────────────────────────────────────────────────┘
```

### 13.4 Context budget ledger

| Component | Steady budget | Peak budget | Breach handling |
|-----------|---------------|-------------|-----------------|
| Orchestrator | < 2K tokens | < 10K tokens | Trigger `state.handoff()` |
| Stage Worker (Diagnose) | n/a (one-shot) | < 20K tokens | `subagent.spawn(max_tokens=30K)` early-return |
| Stage Worker (Re-Plan) | n/a | < 30K tokens | same |
| Stage Worker (EnvSweep) | n/a | < 15K tokens | same |
| SubagentResult.summary | — | < 200 tokens | `orchestration.md` rule rejection |
| State Layer | unlimited (disk) | — | only disk capacity |

### 13.5 Anti-patterns (explicitly NOT do)

| Anti-pattern | Why not |
|--------------|---------|
| "One subagent per Skill file" | Skills cross-reference (COMPUTE_BOUND also reads comm/overlap.md), RPC disaster; spawn overhead > benefit |
| "One Agent per round, sharded by round" | Cross-round PlanGraph / exhausted_neighborhoods evolution is tightly coupled; better to use Strategy A and stash in State Layer |
| "Use subagents as a 'larger context'" | If subagent itself bloats (reads all of optimization/**), the problem is just postponed; Workers must be short-lived and read narrowly |
| "Orchestrator absorbs Worker reasoning trace for audit" | Audit relies on State Layer + `SubagentResult.artifacts[]`, not on context replay |
| "Replace subagent with nested function calls" | No context isolation — same as one session swallowing all Worker trace |

### 13.6 Framework compatibility

Pilot's trunk (Skills + State + Tools + Schemas) is decoupled from any specific agent runtime, allowing different frameworks to deliver different tiers of context management:

| Framework capability | Pilot expectation | If framework lacks support |
|----------------------|-------------------|----------------------------|
| Native subagent spawning | Recommended: full Strategy B | Degrade: "Stage Worker" becomes a sub-conversation segment in the same session, manually clear context on exit — Strategy B disabled, only A in effect, protocol still correct |
| Native handoff / session restart | Recommended: full Strategy C | Degrade: rely on framework's context compression; manual Stop/Resume |
| MCP / function calling | Either is fine | Degrade: Pilot's `tools/` is CLI; `shell` invocation suffices |
| Long-running (>1 day) | Recommended: for >20 round jobs | Degrade: run in segments, each `state.resume()` continues |

**Degradation path is always available**: even with no advanced capability, as long as the framework can read .md + invoke shell + write .yaml, Pilot's trunk works — only context management capability is correspondingly weaker.

**Tool abstraction**: `subagent.*` / `state.handoff()` etc. are **protocol abstractions** — Pilot specifies input/output contracts and call timing; concrete impl comes from the framework (Claude Code / Cursor's Task tool, in-house harness's process spawning, ...).

### 13.7 Layered rollout

The three strategies need not all land at once; turn them on per integrated framework's capability and per task length:

| Tier | Action | Applicable | Prerequisites |
|------|--------|------------|---------------|
| Minimum | Strategy A only: drop in `state.trim()` + hygiene rules in `orchestration.md` | 5–8 round short tasks | None |
| Core | Add Strategy B: subagents for Re-Plan / Diagnose / EnvSweep | 10+ round mid tasks | Framework supports subagent spawning |
| Full | Add Strategy C: `state.handoff()` + `CONTEXT_OVERFLOW` path | Multi-day / 20+ round long tasks | Framework supports session restart / resume |

Each tier deploys independently and yields its own benefit; an upper tier failing does not affect the lower tier's correctness.

---

## 14. One-line summary

Primus Pilot = a tuning-domain **knowledge + toolkit** (`skills/` knowledge + `prompts/` role prompts + `tools/` business actions + `schemas/` data contracts + `state/` persistence); **runtime is delegated to the agent framework** (Claude Code / Cursor / in-house harness) — the framework uses its main session as the **Orchestrator** role, uses its native subagent mechanism to spawn **Stage Workers**, and progresses the state machine following Pilot's state-first protocol so that long loops keep context O(1) steady, closing the loop from Preflight → Projection → Tuning Loop.
