# Primus Pilot

**Status**: Design spec.
**Audience**: anyone changing `pilot/skills/*`, integrating Pilot into another agent harness, or proposing a new top-level file under `pilot/`.

> One-sentence framing — Primus Pilot is **a Cursor skill collection for tuning Primus training jobs**. Cursor is the runtime; Pilot is the knowledge that runs inside it. The single architectural artifact is one subagent contract (`run-and-profile`); everything else is decision tables in markdown.

**Quick start**: open `pilot/skills/tuning-loop/SKILL.md` — that is the entry skill the main agent reads to drive a tuning session. Everything below is the design rationale behind that skill and the rest of `pilot/skills/`.

---

## Table of contents

0. [Document scope and how to read it](#0-document-scope-and-how-to-read-it)
1. [Positioning — what Pilot is](#1-positioning--what-pilot-is)
2. [Design principles](#2-design-principles)
3. [Layered architecture](#3-layered-architecture)
4. [End-to-end flow (the tuning loop)](#4-end-to-end-flow-the-tuning-loop)
5. [Skill structure and on-demand loading](#5-skill-structure-and-on-demand-loading)
6. [The one interface — `run-and-profile`](#6-the-one-interface--run-and-profile)
7. [State, memory, and persistence](#7-state-memory-and-persistence)
8. [Long-session strategy](#8-long-session-strategy)
9. [Cursor runtime contract](#9-cursor-runtime-contract)
10. [Anti-patterns — things this design explicitly will not do](#10-anti-patterns--things-this-design-explicitly-will-not-do)
11. [Extension guide](#11-extension-guide)
12. [Acceptance and self-check](#12-acceptance-and-self-check)
13. [One-line summary](#13-one-line-summary)

---

## 0. Document scope and how to read it

This file answers three questions:

1. **What is Pilot?** — §1, §2
2. **How does it work?** — §3, §4, §5, §6, §7
3. **How does it survive long jobs without a custom runtime?** — §8, §9

If you only have five minutes:

- Read §1 (positioning) and §6 (the one interface).
- Skim §10 (anti-patterns) before proposing any new top-level directory under `pilot/`.
- Open `pilot/skills/tuning-loop/SKILL.md` for the entry skill the main agent actually executes.

If you are proposing a structural change (a new top-level directory, a new schema file, an "Orchestrator" component, a state store), read §2 and §10 first — those changes are explicitly out of scope.

---

## 1. Positioning — what Pilot is

### 1.1 The artifact

A **Cursor skill collection** packaged under `pilot/skills/<skill-name>/SKILL.md`. Each skill is a single markdown file with frontmatter (`name`, `description`) and a `## Workflow` body. Skills encode:

- Training-job execution-model formulas (`execution-model`)
- Bottleneck classification rules (`bottleneck-diagnose`)
- Optimization strategies per bottleneck (`optimize-comm`, `optimize-pipeline`, `optimize-memory`, `optimize-compute`, `optimize-moe`)
- The ordered rollout catalog of Primus "should-be-on" features (`primus-defaults`)
- The env-flag catalog (`env-catalog`)
- The cluster baseline / env probe procedure (`preflight`)
- The overall think loop (`tuning-loop`)
- The one subagent contract (`run-and-profile`)

That is the entire surface area.

### 1.2 Scope boundary

| Concern | In scope for Pilot | Out of scope |
|---|---|---|
| Tuning domain knowledge (formulas, decision tables, env catalog, optimization strategies) | **Yes** — `pilot/skills/*/SKILL.md` | — |
| The one interface contract for "submit a training run + return a short summary" | **Yes** — `pilot/skills/run-and-profile/SKILL.md` | — |
| LLM inference, tool dispatch, token budgeting | — | **Cursor** |
| Subagent spawning and context isolation | — | **Cursor** (native Task tool) |
| Chat history as working memory | — | **Cursor** (native chat) |
| File system reads/writes (`Read`, `Write`, `Glob`, `Grep`) | — | **Cursor** |
| Shell invocation (`bash`, `srun`, `salloc`, `docker exec`) | — | **Cursor** |
| Cluster preparation (SLURM allocation, container `docker run`) | — | **User / SRE** |
| Long-term cross-session memory (best plan per `<model, cluster>`) | **Pilot conventions** — two plain markdown files on disk (see §7) | — |
| Numerical correctness of training (NaN/inf, loss alignment) | — | **User / model owner** — Pilot escalates and stops |
| State-machine engine, schema validators, custom runtime | — | Not provided. The think loop is prose in `tuning-loop/SKILL.md`; the model walks it. |

Rule of thumb: if Cursor provides it natively, Pilot does not re-implement it.

### 1.3 Directory convention

```
pilot/
├── README.md                       # this file — entry point + design rationale
└── skills/
    └── <skill-name>/SKILL.md
```

That is the entire allowed shape. If any future change wants to add `pilot/prompts/`, `pilot/tools/`, `pilot/schemas/`, `pilot/state/`, `pilot/integrations/`, or `pilot/agent/` — read §10 first; the design has explicit reasons for not having any of them.

---

## 2. Design principles

### 2.1 Knowledge, not runtime

Pilot is knowledge delivered as Cursor skills. It does not own:

- the LLM,
- the chat history,
- the subagent spawn mechanism,
- a state machine,
- a persistence layer with schemas,
- a multi-framework abstraction layer.

Cursor provides all six natively. Re-implementing any of them inside Pilot would create two fighting copies (Cursor's chat history *and* Pilot's state store; Cursor's subagent *and* Pilot's "Stage Worker") and add a learning curve where new engineers have to internalize Pilot's runtime before they can change a tuning rule.

### 2.2 One contract, exactly one

The single nontrivial design call is: **every "submit training + collect profile" goes through a subagent following `run-and-profile`.**

That is the only contract. We keep it for one reason — the main chat must not absorb log files and profiler traces, because once a 50 MB profiler dump enters the context window it never leaves. Subagents are the cheapest way to enforce that separation, and Cursor provides them natively. Pilot only has to specify the input shape and the markdown summary the subagent returns.

Everything else — bottleneck classification, plan derivation, settle/stop decisions — happens in the main Cursor session reading whichever skill is in scope. No additional contract is needed because no other handoff crosses a process boundary.

### 2.3 Skills are decision tables, not prose essays

Knowledge inside SKILL.md files favours:

- **Decision tables** (`| Condition | Action |`) over paragraphs.
- **Formulas and thresholds** (`T_bubble = (pp-1)/(pp-1+M) × T_comp`) over descriptive text.
- **Enumerated steps** (`### Step 1: …`) over discursive workflow.

Prose paragraphs are reserved for *why* a decision exists, not *what* to do. The model is the consumer; it works better when the rule is a row it can look up than when it has to infer from prose.

### 2.4 Chat is the state

The in-session ledger (champion / shelved / dead / tried axes / rounds used / budget used) lives as plain markdown in the chat. The model updates it after each Settle step; it reads it before each Plan step. There is no separate state file for this — Cursor's chat history is the durable medium for in-session memory.

Across sessions, exactly two markdown files persist (see §7).

### 2.5 One variable per candidate

Every `run-and-profile` subagent invocation changes **exactly one variable** from its parent plan. The only allowance: a set of flags that *must* be set together to express a single logical feature is one variable (a "coupled bundle").

The rule exists because the user reads the final report's decision trace line by line. Splitting candidates costs more rounds but makes every gain auditable. Parallelism is the escape hatch — independent candidates from the same parent can run as N parallel subagents in one wallclock round.

### 2.6 Load on demand

Loading all 12 skills at session start would defeat the context-discipline argument in §8. Skills are loaded by Cursor's description-based matching only when their content is needed. The `description` frontmatter on each SKILL.md is engineered to trigger exactly when, and only when, that skill is relevant — see §5.2 for the per-skill policy.

### 2.7 Directory minimalism

`pilot/` contains exactly `README.md` and `skills/`. No Python, no JSON Schema, no YAML state files, no subdirectories beyond `skills/`. Every proposal to add one is checked against §10 (anti-patterns) first.

---

## 3. Layered architecture

The system is a four-layer stack. Two layers belong to Cursor (the runtime), two belong to Pilot (the knowledge and the one interface).

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Layer 4 — Cursor runtime                      │
│                                                                      │
│  • Main session: drives the tuning loop, holds the in-chat ledger    │
│  • Native tools: Read / Write / Shell / Glob / Grep / TodoWrite      │
│  • Native subagents: Task tool — context-isolated child processes    │
│  • Native chat history: durable in-session working memory            │
│  • No knowledge of training, profiling, or env tuning by default     │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ loads on demand
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                Layer 3 — Pilot skills (knowledge)                    │
│                                                                      │
│  pilot/skills/<name>/SKILL.md  (frontmatter + ## Workflow)           │
│  Cursor loads a skill when the user's request matches its            │
│  `description` keywords. Most skills are loaded *during* the loop,   │
│  not preloaded — see §5.2.                                            │
│                                                                      │
│  Always in scope at the start: tuning-loop, run-and-profile          │
│  Loaded once per cluster: preflight, env-catalog                     │
│  Loaded on bottleneck: optimize-comm / pipeline / memory / compute   │
│                          / moe                                       │
│  Loaded on demand: execution-model, bottleneck-diagnose,             │
│                    primus-defaults                                   │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ the only interface in Pilot
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│           Layer 2 — `run-and-profile` (the one contract)             │
│                                                                      │
│  Subagent input  : plan + cluster + purpose + max_steps  (YAML)      │
│  Subagent work   : translate to Primus CLI overrides → bash launch   │
│                    → poll → parse log + profiler → write artifacts   │
│  Subagent output : ~200-word markdown block back to main session     │
│                    + on-disk artifacts under output/pilot/runs/      │
│                                                                      │
│  This is the ONLY skill in Pilot that defines a data interface,      │
│  because it is the only place that absorbs heavy artifacts.          │
└────────────────────┬─────────────────────────────────────────────────┘
                     │ files only — no in-process state
                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Layer 1 — Disk artifacts (durable state)                │
│                                                                      │
│  output/pilot/cluster-<cluster_id>.md   ← from preflight, reused     │
│  output/pilot/<session_id>.md           ← final report per session   │
│  output/pilot/runs/<run_id>/            ← per-run subagent workdir   │
│    ├── plan.yaml                                                     │
│    ├── log.txt                                                       │
│    ├── profile/                                                      │
│    ├── snapshot.yaml                                                 │
│    └── result.md                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.1 What each layer is responsible for

- **Layer 4 (Cursor runtime)** drives the loop. It hosts the LLM, the chat history, and the spawn primitives. It has no built-in knowledge of training; everything it knows comes from skills it loads.
- **Layer 3 (skills)** supplies the knowledge. Skills are read-only references; they tell the model what to do next but never execute anything themselves.
- **Layer 2 (the one contract)** is the boundary across which training runs happen. It is the only place control crosses a process boundary in Pilot.
- **Layer 1 (disk artifacts)** is durable state. Per-run artifact directories are opaque to the main session; cross-session memory is two markdown files the model can read like prose.

### 3.2 Data flow direction

- Cursor → Skills: read-only; the main session reads SKILL.md files as decision references.
- Skills → Cursor: skills *tell the model what to do next*; they never execute anything themselves.
- Cursor → `run-and-profile` subagent: parent passes a plan + a purpose; that is the only place control crosses a process boundary in Pilot.
- `run-and-profile` subagent → disk: writes the full artifact directory.
- `run-and-profile` subagent → main session: returns one markdown block.
- Main session → disk: at the end of the session, writes `output/pilot/<session_id>.md`.

There is no automatic feedback edge from runs back into skills. Improvements to skills go through normal source-controlled edits.

---

## 4. End-to-end flow (the tuning loop)

The authoritative description of the flow lives in `pilot/skills/tuning-loop/SKILL.md`. This section sketches the same flow at architecture level so a reader unfamiliar with the skills can see how the pieces compose.

### 4.1 The five stages

```
                       ┌─────────────────────────────────────┐
                       │ User input: model spec / cluster /  │
                       │ goal (primary metric + constraints  │
                       │ + max rounds / GPU·h)               │
                       └────────────────┬────────────────────┘
                                        │
                                        ▼
        ┌─────────────────────────────────────────────────────────┐
        │  1. PREFLIGHT — first time on a cluster, or after       │
        │     cluster change. Subagent invocation; writes         │
        │     output/pilot/cluster-<cluster_id>.md.               │
        │     Skipped if cached and < 7 days old.                 │
        └─────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │  2. BASELINE — run the user's YAML as-given, no         │
        │     overrides. Subagent invocation; one markdown        │
        │     summary back. Becomes round_0 / champion.           │
        └─────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │  3. LOOP Phase A — primus-defaults rollout              │
        │     For each "should-be-on" feature in order:           │
        │       spawn 1 subagent with plan = champion + feature.  │
        │       Each candidate changes exactly one feature.       │
        │       Promote if gain ≥ 1% (Phase A threshold).         │
        │     Optional parallelism: independent features in one   │
        │     round → N parallel subagents.                       │
        └─────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │  4. LOOP Phase B — bottleneck-driven tuning             │
        │     Per round:                                          │
        │       (a) Diagnose latest run via bottleneck-diagnose   │
        │       (b) Pick optimize-<bottleneck> skill              │
        │       (c) Choose ONE candidate plan (one axis change)   │
        │       (d) Spawn run-and-profile subagent                │
        │       (e) Settle: champion / shelved / dead             │
        │       (f) Stop check: gain < 2% × 2 rounds / budget /   │
        │           exhausted axes                                │
        └─────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────┐
        │  5. REPORT — write final plan + decision trace to       │
        │     output/pilot/<session_id>.md. Next session on the   │
        │     same <model, cluster> reads it as prior best.       │
        └─────────────────────────────────────────────────────────┘
```

### 4.2 What runs where

| Action | Where | Why |
|---|---|---|
| Reading skills, deciding the next plan, maintaining the ledger | Main Cursor session | LLM reasoning + chat history |
| Submitting a training job (`bash …/run_pretrain_cli.sh`) | `run-and-profile` subagent | isolates the log/profiler from the main context |
| Parsing logs and profiler traces | Same subagent | same isolation reason |
| Running the cluster bench suite (GEMM peak, collective bandwidth) | `preflight` subagent | same isolation reason — bench output is verbose |
| Writing per-run artifacts to `output/pilot/runs/<run_id>/` | Same subagent | the subagent owns its run directory |
| Writing the final session report | Main session at the end of the loop | the model has the decision trace in chat |

The split is deterministic: heavy artifacts are absorbed by subagents and reduced to ~200-word summaries before crossing back into the main chat.

### 4.3 The one-variable rule in practice

`tuning-loop/SKILL.md` and `primus-defaults/SKILL.md` both restate this rule: every candidate changes exactly one variable (or one coupled bundle, with the bundle reason stated in `notes`).

Why a "coupled bundle" exists at all: some features (DeepEP, the bf16 precision-aware optimizer, Primus Turbo's grouped MLP path) only express their value as a tightly coupled flag set — enabling one flag without the others is a no-op or a misconfig. A bundle counts as one feature = one candidate = one round.

When in doubt: split into two candidates. The cost is one extra round; the benefit is a clean decision trace.

### 4.4 The in-chat ledger

After each round, the main session updates a few lines of markdown in chat:

```
Champion: r3_p1 (tps=18500, +5% vs r2)
Shelved:  r1_p2, r2_p3, r3_p2
Dead:     r2_p4 (OOM), r3_p3 (numerical)
Tried axes around champion: vpp{1,2}, mbs{1,2}, NCCL_BUFFSIZE{8M,16M}
Rounds used: 3 / 5  ·  GPU·h used: 2.1 / 4
```

That is the entire "state". Every decision the model needs (which axes to avoid, what the current champion is, whether budget is depleted) reads off these few lines.

### 4.5 What changes when the loop hits a wall

| Situation | Response (lives in `tuning-loop/SKILL.md`) |
|---|---|
| 2 consecutive rounds gain < 2% | Force one explore round derived from a `shelved` candidate, not the champion |
| Last 3 rounds all moved the same axis | Prefer a different axis category next |
| All `(parent, axis)` combinations exhausted | Stop |
| `budget.max_rounds` or `budget.total_gpu_h` reached | Stop |
| OOM on a candidate | Mark plan dead; do not retry that axis combo |
| HANG / NCCL/IB error on a candidate | Re-run `preflight` env_probe; mark the run as inconclusive |
| `numerical` failure | Stop, escalate to user — Pilot does not touch numerical correctness |

The rules above are prose checkpoints inside `tuning-loop/SKILL.md`; the model reads them when it gets to that decision point.

---

## 5. Skill structure and on-demand loading

### 5.1 Skill anatomy

Every skill follows the same shape (aligned with `.cursor/skills/` convention):

```markdown
---
name: <skill-name-kebab-case>
description: <single paragraph; describes what it is AND when Cursor should
  load it. The description doubles as the trigger condition — Cursor matches
  it against the user's request and current context. Include concrete keywords
  the user is likely to say (e.g. "NCCL_BUFFSIZE", "allreduce", "MoE
  dispatch") so the right skill surfaces at the right time.>
---

# <Title>

<2–3 sentence overview>

## Inputs (when relevant)
## Workflow
### Step 1: …
### Step 2: …
…
## Important Notes
```

The `description` field is the only thing standing between "this skill is in scope" and "Cursor never sees it this session". Trigger keywords matter — see `optimize-comm/SKILL.md` for an example whose description lists every plausible synonym (bucket, overlap, alltoall, NCCL_*, RCCL_*, …).

### 5.2 Load-on-demand policy

Pilot has 12 skills. Loading all of them at the start of every session would defeat the purpose of context discipline. The policy:

| Skill | Loaded when | Released when |
|---|---|---|
| `tuning-loop` | Always — the entry point | Session ends |
| `run-and-profile` | Always — every training run uses it | Session ends |
| `preflight` | First time on a cluster, or cached baseline > 7 days old | After cluster baseline is in chat as a 5-line summary |
| `primus-defaults` | At the start of LOOP Phase A | After Phase A completes |
| `bottleneck-diagnose` | After every run from round 2 onward | After the round's diagnosis is recorded |
| `execution-model` | When sanity-checking a candidate's predicted `T_step` / `Mem_peak` before launch | After the prediction is in chat |
| `optimize-comm` | When latest diagnosis is `COMM_BOUND` | When the bottleneck class changes |
| `optimize-pipeline` | When latest diagnosis is `PIPELINE_BOUND` | same |
| `optimize-memory` | When latest diagnosis is `MEMORY_BOUND`, or after an OOM | same |
| `optimize-compute` | When latest diagnosis is `COMPUTE_BOUND` | same |
| `optimize-moe` | When latest diagnosis is `MOE_DISPATCH_BOUND` | same |
| `env-catalog` | Whenever an `env_diff` field needs lookup | After the env decision is recorded |

The model decides when to load a skill (Cursor surfaces it based on description matching), but the policy column above is the design intent that every skill's `description` is written to support.

### 5.3 Skills carry both knowledge and intent

The skill the model is currently reading is the prompt for what it is doing right now. There is no separate "you are the Orchestrator" frame — skills *are* the prompts. A skill's `## Workflow` section is the imperative that drives the next few decisions; its decision tables are the references the model consults inside that workflow.

This conflation is deliberate. Splitting prompts from knowledge would require an extra coordination layer (when does the prompt change, who chooses it) without producing extra capability.

### 5.4 Why behaviours are skills, not code

The natural alternative would be a `pilot/tools/` directory of Python functions the model calls (`preflight.run()`, `submit.run()`, `observe.snapshot()`, `constraint.check()`, …). The design rejects that, because:

| Conceptual capability | Where it lives in Pilot |
|---|---|
| Run a cluster bench suite and write the baseline | The `preflight` skill — the subagent following it executes the bench tooling directly |
| Submit a training job | The `bash …/run_pretrain_cli.sh` invocation that `run-and-profile` performs |
| Cancel a hung job | `scancel` / `kill` in `run-and-profile`'s timeout path |
| Parse a run into metrics | The log+profiler parsing step inside `run-and-profile` |
| Decide whether a plan is valid | Decision tables inside each `optimize-*` skill |
| Estimate memory before launch | `execution-model/SKILL.md` formulas + the in-chat ledger's record of OOM cases |
| Attribute a failure | The status enum inside `run-and-profile` (oom / hang / failed / numerical) |
| Checkpoint state | The end-of-round ledger update in chat |
| Resume across sessions | The user reading `output/pilot/<session_id>.md` at the start of a new session |
| Trim context | Cursor's native context management |
| Spawn a subagent | Cursor's native Task tool |
| Capture a learning | A source-controlled edit to the relevant SKILL.md, or the `### Lessons` block of the session report |

Everything that would be a "tool" is either Cursor's job, the user's job, or a decision rule expressible as a markdown table inside a skill. The result is a single artifact format (`SKILL.md`) with no build step, no import path, and no Python dependency.

---

## 6. The one interface — `run-and-profile`

This is the only place Pilot defines a data interface. Everything in this section is also in `pilot/skills/run-and-profile/SKILL.md`; the description here exists to explain *why* it is the only interface, what it guards, and what is intentionally outside it.

### 6.1 What it guards

Two things never enter the main chat:

1. **Raw training log content** — multi-MB stdout/stderr with NCCL bootstrap chatter, warmup-iteration noise, sampler dumps. Once a 5 MB log enters the context it never leaves; six rounds in, the session is dead.
2. **Profiler trace content** — PyTorch profiler JSON / chrome trace JSON, easily 50 MB per run. Same problem, multiplied.

`run-and-profile` is the cordon sanitaire. The subagent absorbs both, reduces them to a ~200-word markdown block, and returns only that. Raw artifacts stay on disk; the main session reads them by path on demand (rarely needed — the summary is sized to be sufficient for the next decision).

### 6.2 The contract shape (high level — full schema in the skill)

**Input** (passed in the subagent prompt as YAML):

- `plan.parallelism` — tp / pp / dp / ep / vpp / cp
- `plan.runtime` — mbs / gbs / recompute / seq_len
- `plan.comm` — bucket_size_mb / overlap
- `plan.env_diff` — *diff vs cluster baseline only* (do not re-state the baseline)
- `cluster` — mode (single|slurm), nodes, gpus_per_node, partition, nodelist, image
- `config.exp_yaml` — path to the Primus YAML
- `purpose` — baseline / candidate-eval / env-sweep / preflight
- `max_steps` — typically 200 for baseline, 50–100 for candidate-eval, 30–50 for env-sweep
- `profile` — whether to enable PyTorch profiler (default true)
- `parent_run_id`, `notes` — optional, for the in-chat ledger

**Output** (single markdown block, ~200 words, returned to the parent):

```markdown
## Run Result (run_id: <id>)

- **plan**: <one-line YAML summary>
- **purpose**: <baseline|candidate-eval|env-sweep> (parent: <id>)
- **status**: completed | early_stopped | oom | hang | failed | numerical
- **metrics**: tps=<>, step_time_ms=<>, comm_ratio=<>, bubble_ratio=<>,
              overlap_ratio=<>, mem_peak_gb=<>, gpu_util_avg=<>
- **bottleneck hint**: COMM_BOUND | PIPELINE_BOUND | MEMORY_BOUND |
                       COMPUTE_BOUND | MOE_DISPATCH_BOUND | MIXED | UNKNOWN
- **one-line summary**: <≤200 chars; what changed, what happened, why>
- **artifacts**:
  - profile: output/pilot/runs/<run_id>/profile/
  - log:     output/pilot/runs/<run_id>/log.txt
  - snapshot.yaml: output/pilot/runs/<run_id>/snapshot.yaml
```

**On-disk artifacts** (the subagent's working directory):

```
output/pilot/runs/<run_id>/
├── plan.yaml          # the input plan, persisted for reproducibility
├── log.txt            # full stdout/stderr
├── profile/           # raw profiler trace
├── snapshot.yaml      # parsed metrics (machine-readable)
└── result.md          # the markdown block returned to parent
```

### 6.3 What the contract is and is not

It **is** a writing convention — a markdown block shape with named bullets. The main session can pattern-match the bullets when updating the ledger.

It **is not** a JSON Schema. There is no validator. Nothing rejects a malformed block at runtime. We rely on the subagent following the skill (and the skill being read carefully). The cost of a malformed block is one confused round, not a process crash; that is an acceptable trade for not maintaining a parser and a schema file.

### 6.4 Why one interface is sufficient

In this design the only place a structured handoff actually happens is the subagent → main-session return. Everything else is either:

- **Prose in chat** — main session writes to itself; no inter-process contract needed.
- **A file on disk** — `cluster-<id>.md` and `<session_id>.md` are markdown; the model writes them and the model reads them.
- **A per-run artifact directory** — `output/pilot/runs/<run_id>/` is opaque to anyone except the subagent that wrote it; the only piece the main session reads is `result.md`.

So one interface covers every handoff that crosses a process boundary.

### 6.5 The other subagent boundary: `preflight`

`preflight` also runs as a subagent (it executes a benchmark suite and parses outputs). Its "interface" is degenerate — the output is the `output/pilot/cluster-<cluster_id>.md` file itself, plus a 5-line summary the subagent returns to the parent. There is no separate input/output schema to specify because the inputs are just `cluster_id`, `mode`, `nodes`, `gpus_per_node`, and the output is a file the rest of Pilot reads with `Read`. We document it inside `preflight/SKILL.md` and stop there.

So strictly speaking there are two subagent boundaries: `run-and-profile` (every training run) and `preflight` (every new cluster). Only `run-and-profile` defines a formal markdown contract because only it is called per round.

---

## 7. State, memory, and persistence

### 7.1 The three tiers of memory

| Tier | Lifetime | Medium | What it holds |
|---|---|---|---|
| **In-session** | Current Cursor chat | Chat history (Cursor-managed) | The ledger, the user's goal, summaries returned from subagents, the model's reasoning |
| **Per-run** | The lifetime of a single training run | `output/pilot/runs/<run_id>/` on disk | plan.yaml, log.txt, profile/, snapshot.yaml, result.md |
| **Cross-session** | Until the cluster or model changes | Two markdown files on disk | `cluster-<cluster_id>.md` (per cluster), `<session_id>.md` (per session) |

There is no fourth tier.

### 7.2 In-session state — the in-chat ledger

The model maintains a few lines of markdown in chat:

```
Session: deepseek_v2_lite × 1node-mi300x × 20260515
Goal: max TPS under mem ≤ 180 GB, budget = 5 rounds / 4 GPU·h

Champion: r3_p1 (tps=18500)
Shelved:  r1_p2, r2_p3
Dead:     r2_p4 (OOM)
Tried axes around r3_p1: vpp{1,2}, mbs{1,2}, NCCL_BUFFSIZE{8M,16M}
Rounds used: 3 / 5  ·  GPU·h used: 2.1 / 4
```

This is updated after each `Settle` step. Cursor's native chat handling keeps it in context for the rest of the session.

**Why this is enough**: every decision the model needs (which axes to avoid, what the current champion is, whether budget is depleted) can be read directly off these few lines. Anything richer than this is over-engineered — the same information is one glance away.

### 7.3 Per-run state — the subagent's working directory

Each subagent invocation owns `output/pilot/runs/<run_id>/` exclusively. After the subagent returns:

- `plan.yaml` records exactly what was run (for reproduction).
- `log.txt` is the full stdout/stderr (for forensic debugging if the summary is insufficient).
- `profile/` is the raw profiler trace (rarely read by the main session; available on demand).
- `snapshot.yaml` is the machine-readable parsed metrics (rarely read by the main session).
- `result.md` is what the parent already saw.

The main session does not delete or garbage-collect this directory; users keep or prune at will.

### 7.4 Cross-session state — two files only

**`output/pilot/cluster-<cluster_id>.md`** — produced by `preflight`. Holds: cluster header, GEMM peak, interconnect bandwidth, RCCL/NCCL baseline (intra/inter/world for AllReduce/AllGather/ReduceScatter/AllToAll at sizes [1, 16, 64, 256] MB), slow-node list, validated env baseline. Reused across every session on the same cluster. Refreshed when stale (>7 days, or driver/fabric change) or when `force_refresh` is requested.

**`output/pilot/<session_id>.md`** — the final tuning report. Holds: session metadata, baseline metrics, final plan, full decision trace round by round, lessons. Read by the next tuning session on the same `<model, cluster>` as prior best.

That is it. No `plan_graph.yaml`, no `tuning_state.yaml`, no `candidate_pool.yaml`, no per-round checkpoint directories.

### 7.5 Why losing in-session state is acceptable

- The Cursor chat is itself persistent within the session.
- Across sessions, `output/pilot/<session_id>.md` carries the relevant decisions forward. The successor session starts from there, not from scratch.
- Per-run artifacts are durable on disk; even if a chat is lost, the user can grep for the best `result.md` and continue manually.

What this trades away: the ability to deterministically replay round 3 of session X with the exact same plan derivation. We have judged this not worth the schema burden. If a user really needs to re-run a candidate, the `plan.yaml` inside its run directory is sufficient.

---

## 8. Long-session strategy

A naive single-session tuning loop accumulates context per round. This section explains the per-round context budget Pilot expects, how the design holds the budget steady, and what to do when a session does eventually grow too large.

### 8.1 The per-round context budget

| Source | Per-round bytes |
|---|---|
| Skill recall (loaded on demand; typically one or two skills per round) | 2–4 K |
| `run-and-profile` summary (the markdown block) | ≤ 0.3 K |
| Diagnosis decision (one bottleneck class + 1–2 env_suspect flags) | 0.1 K |
| Ledger update | 0.1 K |
| Main-session reasoning | 1–2 K |
| **Per-round total** | **3.5–6.5 K** |

A 200 K window therefore comfortably hosts 20+ rounds without intervention.

### 8.2 The three properties that keep the budget steady

1. **Subagents absorb heavy artifacts.** Log content and profiler traces are absorbed by the `run-and-profile` subagent, reduced to a ~200-word summary, and persisted to disk. The main session never sees the raw bytes. This is enforced by the one-contract rule in §2.2.
2. **Skills load on demand.** Only `tuning-loop` and `run-and-profile` are always in scope. The other 10 skills enter and leave the context based on which bottleneck is active and which decision is being made (§5.2 lists the per-skill policy). This caps the skill-recall contribution at "the one or two skills relevant to this round".
3. **The ledger is a few lines, not a structured store.** Champion, shelved, dead, tried axes, rounds/budget used — all fit in 5–8 lines of markdown. The model reads them in a glance and updates them in a glance, instead of accumulating per-round narrative.

These three properties are properties of the design, not protocols the model has to follow. They fall out of "skills load on demand", "subagents absorb heavy artifacts", and "chat is the state".

### 8.3 When the user should start a new chat

For very long jobs (20+ rounds, multi-day), even a 200 K window eventually fills. The escape hatch is Cursor's own: start a new chat.

| Signal | What to do |
|---|---|
| Session has crossed 20–30 rounds and Cursor's context indicator shows >70% used | Wrap up: write the final report to `output/pilot/<session_id>.md`, then start a new Cursor chat for the next iteration |
| The user wants to continue tuning the same `<model, cluster>` next week | New chat; point it at the prior `output/pilot/<session_id>.md` and `output/pilot/cluster-<cluster_id>.md` |
| The model starts repeating an axis that the ledger lists in "Tried axes" | Context degradation — write the report and restart |

There is no `CONTEXT_OVERFLOW` state transition, no automatic handoff tool, no checkpointing protocol. The user notices it is time and starts fresh — which is what they would do in Cursor anyway, regardless of what Pilot prescribes.

### 8.4 Per-subagent context budget

| Subagent | Peak context | Why it stays bounded |
|---|---|---|
| `run-and-profile` | ≤ 30 K | Reads `run-and-profile/SKILL.md`, the plan, the YAML once, and parses log+profile programmatically (does not LLM-process them) |
| `preflight` | ≤ 30 K | Reads `preflight/SKILL.md`, the cluster info, runs bench commands, parses output programmatically |

Subagent contexts do not feed back into the parent; they die at return. No budget accounting needed beyond the per-skill workflow rules.

---

## 9. Cursor runtime contract

Pilot assumes Cursor is the runtime. This section makes the assumption explicit so anyone porting Pilot to another runtime knows what they would need.

### 9.1 What Pilot expects from Cursor

| Capability | Used for | If absent |
|---|---|---|
| Skill loading by `description` matching | Loading the right `optimize-*` / `env-catalog` / etc. when the bottleneck changes | Pilot still works — the user would have to instruct the model to "read pilot/skills/optimize-comm/SKILL.md" by hand |
| Native Task tool (subagent spawning) | The `run-and-profile` and `preflight` subagent boundaries | The main convention breaks — without subagents, log/profile content would land in the main chat |
| Native chat history | The in-chat ledger; session memory | The in-session state model breaks — would need an external store |
| Shell tool (`bash`, `srun`, `docker exec`) | The subagent's training launch and bench commands | Subagent cannot do its job |
| Read / Write / Glob / Grep | Reading SKILL.md files, reading cluster baseline, writing the session report | Foundational — Pilot cannot function |

### 9.2 What Pilot does **not** expect from Cursor

| Capability | Status |
|---|---|
| MCP servers | Not used. All Pilot tools are skills + shell. |
| Custom tool wrappers | Not used. No Python in `pilot/`. |
| Memory / RAG infrastructure beyond chat | Not used. `output/pilot/*.md` files are read by `Read`. |
| Long-running background agents | Not used. The user is in the loop; subagents are short-lived. |
| Multi-user / shared session | Not used. Single-user sessions only. |

### 9.3 Cursor session structure during a tuning run

```
Cursor main session
│
│   Loaded skills (on demand):
│     tuning-loop (always)
│     run-and-profile (always)
│     preflight (round 0 only)
│     primus-defaults (Phase A only)
│     bottleneck-diagnose (every round from round 2)
│     execution-model (when sanity-checking)
│     optimize-<class> (when bottleneck class matches)
│     env-catalog (when looking up an env flag)
│
│   In-chat artifacts:
│     - Session header (model, cluster, goal, budget)
│     - Round-by-round ledger
│     - Per-round summary blocks from subagents (~200 words each)
│
└─ spawns ──> Cursor subagent: run-and-profile
                  Reads: run-and-profile/SKILL.md, the plan YAML, the exp_yaml
                  Runs:  bash …/run_pretrain_cli.sh, polls, kills on timeout
                  Parses: log.txt + profile/*.json
                  Writes: output/pilot/runs/<run_id>/{plan.yaml,log.txt,
                          profile/,snapshot.yaml,result.md}
                  Returns: result.md content (the markdown block)
                  Dies.

└─ spawns ──> Cursor subagent: preflight  (only once per cluster)
                  Reads: preflight/SKILL.md, env-catalog/SKILL.md
                  Runs:  rocblas-bench, rccl-tests, env_probe sweep
                  Writes: output/pilot/cluster-<cluster_id>.md
                  Returns: 5-line summary
                  Dies.
```

This is the entire process topology. Nothing else spawns.

### 9.4 Porting to a non-Cursor runtime

If someone needs to run Pilot under another LLM agent framework:

1. The framework must support loading markdown files as "skills" that are pulled in based on a description-matching mechanism (or the user manually loads them).
2. The framework must support spawning a subagent / child task with its own context, returning a structured (markdown) result, and dying.
3. The framework must provide a Shell-equivalent tool for the subagent to invoke `bash`.
4. The framework must provide Read/Write tools for the main session and for the subagent.

If all four are present, Pilot works as-is — `pilot/skills/` is portable markdown. If any are absent, the user must compensate by manual prompting or by writing a thin harness *outside* `pilot/`. The `pilot/` directory itself stays minimal.

---

## 10. Anti-patterns — things this design explicitly will not do

If you find yourself wanting to add any of the following, this section is for you. Read it before opening a PR.

### 10.1 Do not add a new top-level directory under `pilot/`

`pilot/` contains only `README.md` and `skills/`. If you want to add `pilot/X/`:

- **`pilot/prompts/`** — collapse it into a SKILL.md. Roles are not separate from knowledge in this design.
- **`pilot/tools/`** — see §5.4. Each candidate tool is either (a) Cursor's job, (b) the user's job, or (c) a decision table inside a skill.
- **`pilot/schemas/`** — there are no schemas. The one interface (`run-and-profile`'s markdown block) is documented in prose. If you need a stricter contract, document it more carefully inside the existing SKILL.md.
- **`pilot/state/`** — chat is the state; two disk files cover cross-session memory.
- **`pilot/integrations/`** — Pilot targets Cursor. Other-framework harnesses live in those frameworks' own repos.
- **`pilot/agent/`** — there is no fallback runtime. If a user is not in Cursor, they manually drive the skills.
- **`pilot/lib/`**, **`pilot/utils/`**, **`pilot/templates/`** — same reasoning. Push the responsibility back into the skills that needed them.

If a skill genuinely needs a code artifact (e.g., a non-trivial parser script that the subagent invokes), put it under `tools/` at the repo root (not `pilot/tools/`) and reference it by path from the skill.

### 10.2 Do not introduce a schema validator

No JSON Schema, no Pydantic mirror, no markdown grammar checker. The model is the consumer. The cost of a malformed `run-and-profile` block is one confused round, not a process crash. We do not pay schema maintenance cost for that.

### 10.3 Do not run training in the main session

The single hard rule. Even for "just a quick sanity check" — spawn a subagent. The minute log/profiler content enters the main chat, the session degrades.

### 10.4 Do not preload all skills at session start

Cursor's description-based loading is doing real work. Forcing every `optimize-*` and `env-catalog` into the main context at session start defeats the entire context-discipline argument in §8. Trust the load-on-demand policy.

### 10.5 Do not paste profile / log content into chat

If the `run-and-profile` summary is insufficient for a decision, `Read` the specific file (`snapshot.yaml` or a profiler JSON) and quote only the relevant lines. Never paste the full file.

### 10.6 Do not change two structural axes in one candidate

`(vpp 1→2, mbs 1→2)` in the same candidate breaks the attribution chain. The user reading the final report cannot tell which change earned the gain. Split into two candidates and run them in parallel from the same parent if wallclock matters.

### 10.7 Do not spawn one subagent per skill

A subagent per `optimize-*` skill or per env flag is RPC disaster — the spawn overhead exceeds the work, and the parent ends up with multiple summaries to reconcile per round. Spawn one subagent per *training run*, not per *decision*.

### 10.8 Do not invent a "long-term memory" mechanism beyond the two disk files

If you want to remember "MoE > 16 nodes always enable AllToAll overlap", the place is either:

- The `## Important Notes` section of the relevant `optimize-*` SKILL.md (source-controlled).
- The `### Lessons` section of `output/pilot/<session_id>.md` for the session that discovered it.

Not a new schema, not a new index file, not a vector store.

### 10.9 Do not add "agent personas" or "role-play frames"

No "you are the Tuning Expert" preamble in skills. The skill is the prompt. The model reads the workflow and follows it.

### 10.10 Do not add a state machine

The flow in `tuning-loop/SKILL.md` is prose, not a state machine. If you want to add a new phase, write a new `## Workflow` step. Do not introduce `if state == X` tables or `on_fail` / `reentry_when` transition tables.

---

## 11. Extension guide

### 11.1 Adding a new skill

1. Decide the trigger. What words / situations will Cursor see that should pull this skill in? List 5–10 keywords. Put them in the `description` frontmatter field.
2. Create `pilot/skills/<skill-name>/SKILL.md`. Frontmatter `name` is kebab-case matching the directory.
3. Body shape: 2–3 sentence overview, `## Workflow` with numbered steps, `## Important Notes` at the end.
4. Use decision tables (`| Condition | Action |`) for "if symptom X then strategy Y" content. Prose paragraphs are for the *why*, not the *what*.
5. If the skill might be loaded mid-round, the `## Workflow` steps should be self-contained — the model may be reading you with limited context.
6. Cross-reference other skills by relative path (`pilot/skills/env-catalog/SKILL.md`). Do not duplicate content.
7. Add a row to the per-skill load-on-demand table in §5.2 of this file.

### 11.2 Adding a new bottleneck class

1. Add the new class to `pilot/skills/bottleneck-diagnose/SKILL.md`:
   - Decision rule: which metrics trip it.
   - `env_suspect` list: which env flags are commonly the proximate cause.
2. Add a new skill `pilot/skills/optimize-<class>/SKILL.md` with the standard shape.
3. Update `pilot/skills/run-and-profile/SKILL.md` Step 7 ("Pick a bottleneck hint") to include the new class.
4. Update `pilot/skills/tuning-loop/SKILL.md` §4.2 "Decide next plan" table to route the new class to the new skill.
5. Update the per-skill table in §5.2 of this file.

### 11.3 Adding a new must-on feature to `primus-defaults`

1. Add the feature to `pilot/skills/primus-defaults/SKILL.md`:
   - The bundle (single flag or coupled bundle).
   - Insertion position in `feature_order` (earlier features are evaluated first).
   - Model-class applicability (Dense / MoE / Turbo / DeepEP / etc.).
   - Expected gain band and the bisect strategy if the bundle fails.
2. Do not touch any other skill. Phase A in `tuning-loop` walks `feature_order` directly.

### 11.4 Adding support for a new training backend (e.g., torchtitan)

1. Confirm the launch wrapper exists at `examples/<backend>/...` and that it accepts the same `--key value` override convention.
2. Add a translation table to `pilot/skills/run-and-profile/SKILL.md` Step 2 mapping `plan.*` fields to the backend's override flag names.
3. If the backend has different default behaviour (e.g., different fusion flags), add it to `pilot/skills/primus-defaults/SKILL.md` with appropriate model-class gating.
4. Do not branch the entire `run-and-profile` workflow per backend; one workflow with a backend-aware translation step.

### 11.5 Changing the in-chat ledger shape

Discouraged. The ledger shape in `tuning-loop/SKILL.md` is what every other skill assumes the model has in scope. If you have a strong reason:

1. Update `tuning-loop/SKILL.md` first.
2. Audit every other SKILL.md for `Champion:` / `Shelved:` / `Dead:` / `Tried axes` references and update.
3. Note the breaking change in a session report.

### 11.6 Changing the `run-and-profile` markdown contract

This is the one contract; changing it has the widest blast radius.

1. Update `pilot/skills/run-and-profile/SKILL.md` Step 9 (the output block shape).
2. Update `pilot/skills/tuning-loop/SKILL.md` §4.3 ("Run via subagent") and §4.4 ("Settle") to consume the new shape.
3. Update this file's §6.2 and §3 diagram.
4. Add a migration note to `output/pilot/<session_id>.md` of any in-flight sessions.

### 11.7 What not to extend

- Adding cross-skill ordering ("optimize-comm must run before optimize-pipeline") — let the bottleneck class drive the order.
- Adding "personas" to skills — see §10.9.
- Adding logging / tracing infrastructure — Cursor's chat is the trace; `output/pilot/runs/<run_id>/` is the durable per-run record.

---

## 12. Acceptance and self-check

A correct implementation satisfies **all** of the following.

### 12.1 Directory shape

```
$ ls pilot/skills/
bottleneck-diagnose  env-catalog       execution-model
optimize-comm        optimize-compute  optimize-memory
optimize-moe         optimize-pipeline preflight
primus-defaults      run-and-profile   tuning-loop
```

At the top level `pilot/` consumes exactly `README.md` and `skills/`. No Python. No JSON. No YAML state files. No `prompts/` / `tools/` / `schemas/` / `state/` / `integrations/` / `agent/` subdirectories.

### 12.2 Skill shape

Every `pilot/skills/<name>/SKILL.md` has:

- Frontmatter with `name: <kebab-case matching directory>` and `description: <single paragraph with trigger keywords>`.
- A title (`# <Title>`).
- A `## Workflow` section with numbered steps.
- An `## Important Notes` section at the end.
- No code that needs a build step. (Shell commands inside markdown code fences are fine — they are documentation.)

### 12.3 Interface shape

`pilot/skills/run-and-profile/SKILL.md` defines:

- The YAML input shape (plan + cluster + config + purpose + max_steps + flags).
- The markdown output block shape (the 6-bullet structure).
- The on-disk artifact layout (`output/pilot/runs/<run_id>/{plan.yaml,log.txt,profile/,snapshot.yaml,result.md}`).
- A status enum: `completed | early_stopped | oom | hang | failed | numerical`.
- A bottleneck hint enum: `COMM_BOUND | PIPELINE_BOUND | MEMORY_BOUND | COMPUTE_BOUND | MOE_DISPATCH_BOUND | MIXED | UNKNOWN`.

### 12.4 Behavioural invariants

When the main session is running the tuning loop:

- Every `bash …/run_pretrain_cli.sh` invocation is inside a subagent. Zero exceptions.
- The main session has not `Read` any `log.txt` or any file under any `profile/` directory unless the subagent's summary was demonstrably insufficient.
- The in-chat ledger lists champion, shelved, dead, and tried axes around the current champion, after every Settle step.
- Each candidate plan differs from its parent by exactly one variable (or one coupled bundle, with the reason stated in `notes`).
- When the session ends, `output/pilot/<session_id>.md` exists.

### 12.5 What a PR reviewer should reject

- Any new top-level directory under `pilot/`.
- Any `.py`, `.json`, `.schema.json`, `.yaml` file under `pilot/` (other than YAML *examples* inside SKILL.md code fences).
- Any addition of an "Orchestrator", "Stage Worker", "state machine", or "context strategy" concept to a skill.
- Any cross-skill dependency that requires preloading.
- Any change to `run-and-profile`'s markdown output shape that has not also updated `tuning-loop` and this file's §6.

---

## 13. One-line summary

Primus Pilot = **a Cursor skill collection for tuning Primus training jobs**, with exactly one architectural artifact (the `run-and-profile` subagent contract) and one hard convention (every training run goes through a subagent, returns a ~200-word markdown block to the main session, and leaves its raw artifacts on disk) — Cursor's native runtime does the rest.
