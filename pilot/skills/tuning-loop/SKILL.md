---
name: tuning-loop
description: Entry skill for end-to-end Primus training tuning. Use whenever the user wants to tune throughput / step time / TPS / memory / scaling efficiency / bubble / MFU of a training job, asks for a "tuning plan" / "调优" / "auto-tuning" / "find the best parallelism / mbs / recompute / env", needs to bring up a new model on a cluster, debugs scaling degradation, or wants Pilot to drive a multi-round optimization loop. The main agent reads this skill to think; actual training runs are delegated to subagents per `run-and-profile`.
---

# Tuning Loop — Pilot Entry Skill

This is the entry point for any Primus tuning session. The **main Cursor agent** reads this skill to decide what to think about next; every "submit training + collect profile" is delegated to a **subagent** following the `run-and-profile` skill. The main agent never absorbs raw profile / log content — it only ever sees the markdown summary that subagents return.

Design constraints for this loop:

- **One variable per candidate.** Every `run-and-profile` subagent invocation changes **exactly one variable** from its parent plan. Without exception, every gain is attributable to a single move. The only allowance: a set of flags that **must** be set together to enable a single logical feature (e.g. `use_precision_aware_optimizer=true` + `main_grads_dtype=bf16` + `exp_avg_dtype=bf16` + `exp_avg_sq_dtype=bf16` is **one feature**, "bf16 precision-aware optimizer"). When in doubt: split into two candidates. This rule applies to **every** phase — `primus-defaults` rollout, env sweep, and bottleneck-driven tuning all obey it.
- **Chat is the state.** The ledger of `(champion / shelved / dead / tried axes / round / budget_used)` lives as plain markdown in this chat. Do not write YAML / JSON state files for it.
- **Subagent is the only place training runs.** The main agent never invokes `bash …/run_pretrain*.sh` directly and never reads raw profiler / log content; it only ever sees the markdown summary defined in `run-and-profile`.
- **Skills load on demand.** Only `tuning-loop` and `run-and-profile` are always in scope. `optimize-*`, `bottleneck-diagnose`, `execution-model`, `env-catalog`, `preflight`, `primus-defaults` are pulled in only when the current step needs them.
- **One artifact persists across sessions.** When the loop ends, write the final report to `output/pilot/<session_id>.md` (best plan + decision trace). The next tuning session on the same model + cluster reads it as prior best.
- **One persistent file per cluster.** `output/pilot/cluster-<cluster_id>.md` (produced by `preflight`) supplies hardware peaks / collective bandwidth / env baseline. Reused across sessions until the cluster changes.

## Workflow

### Step 1: Gather user input

Before doing anything, confirm three things from the user. If something is missing, ask.

| Slot | Example | If missing |
|---|---|---|
| Model spec | `examples/megatron/configs/MI300X/llama3.1_8B-BF16-pretrain.yaml`, or "DeepSeek V2 Lite, MoE" | Ask for a Primus YAML path or a model identifier |
| Cluster shape | `1 node × 8 MI300X` (single) or `16 nodes × 8 MI355X` (slurm) | Ask: how many nodes, GPU model, single-node or SLURM |
| Goal | "max TPS under mem ≤ 180 GB", "MFU ≥ 50%", "bring up without OOM" | Ask for the primary metric + any hard constraints + budget (max rounds, GPU·h) |

Record this in chat as a "session header" so the user can sanity-check before any GPU is burned.

### Step 2: PREFLIGHT (first time on a cluster, or after cluster change)

Before any tuning round, the cluster baseline must exist.

- If `output/pilot/cluster-<cluster_id>.md` already exists and is < 7 days old, reuse it.
- Otherwise run the `preflight` skill once. Preflight is itself a subagent invocation — it does GEMM peak / collective bandwidth / env probe and writes `output/pilot/cluster-<cluster_id>.md`.

What you keep in chat from preflight: a 5-line summary (peak TFLOPS, intra/inter bandwidth, env baseline tag, slow-node list if any, file path). Discard the rest.

### Step 3: BASELINE (run the user's YAML as-given, no overrides)

BASELINE is purely descriptive: submit `purpose: baseline` with the user's `exp_yaml` exactly as provided — **no parallelism / runtime / env overrides at all**. The point is to measure the as-shipped behavior so every subsequent optimization round can quote a real "+X% vs baseline".

Pilot does **not** apply Primus must-on flags, choose conservative defaults, or make any tuning decision here. Those all belong in the LOOP (Step 4) so the user can see exactly what each optimization batch contributes.

| Outcome | Action |
|---|---|
| `status: completed` | Treat as `champion = round_0`, proceed to LOOP |
| `status: oom` | The user's as-shipped YAML doesn't fit. Escalate: ask the user to confirm the cluster shape matches the YAML's expected scale, or to provide a smaller starting `mbs` / `seq_length` and re-baseline. Do **not** silently override. |
| `status: hang` or NCCL/IB error | Re-run `preflight` env_probe; check `output/pilot/cluster-*.md` for slow nodes |
| `status: numerical` (loss NaN/inf) | STOP. Numerical correctness is a model-side issue, not Pilot's job — escalate to user |
| `status: failed` (config error) | Fix the YAML / override; not a tuning round |

Record the baseline:

```
round_0 (baseline, YAML as-given):
  plan: <copy parallelism / mbs / gbs / recompute from the YAML, env=cluster-baseline>
  metrics: tps=<>, step_ms=<>, comm_ratio=<>, bubble=<>, mem=<>
  bottleneck: <hint from subagent>
```

### Step 4: LOOP — `primus-defaults` phase, then `Diagnose → Plan → Run → Settle`

The LOOP has two phases:

- **Phase A — primus-defaults rollout (rounds 1..K).** Enable each Primus must-on feature one at a time. Every round adds one feature on top of the current champion; per-feature gain becomes attributable.
- **Phase B — bottleneck-driven tuning (rounds K+1..N).** Standard Diagnose → Plan → Run → Settle.

#### 4.0 Phase A — `primus-defaults` rollout (rounds 1..K)

Load `primus-defaults`. It defines an ordered list of "features", where each feature is either a single flag or a small bundle of coupled flags that must be set together to express one logical capability. Run them in order:

```
for feature in primus-defaults.feature_order(model_class):
    if YAML already has it on: skip
    else:
        spawn 1 subagent: purpose=candidate-eval, plan = champion + {feature.flags}, notes="feature=<name>"
        wait for result
        if gain >= 1%:    promote to champion (this is ε_promote_phase_A — looser than Phase B's 2%)
        if gain in [-1%, 1%]: keep on, treat as neutral (small wins still compound)
        if gain < -1%:    drop the feature, mark as "not for this model" in ledger
        if status=failed: bisect inside the bundle (if multi-flag) or drop the feature
```

Optional parallelism: when several next-up features are independent (e.g. `apply_rope_fusion` and `cross_entropy_loss_fusion`), spawn them as parallel subagents in the same round — each candidate is still one feature on top of the same parent. Pick the winner, then derive the next round from it.

Phase A typically runs 5–10 rounds (depends on model class and what the YAML already had on). Each round inherits its parent's champion, so gains compound deterministically.

When `feature_order` is exhausted, transition to Phase B at 4.1.

#### 4.1 Diagnose

Read `bottleneck-diagnose` against the latest run's summary. Output one of: `COMM_BOUND`, `PIPELINE_BOUND`, `MEMORY_BOUND`, `COMPUTE_BOUND`, `MOE_DISPATCH_BOUND`, `MIXED`. Note any `env_suspect` flags (likely-misconfigured env that explains the symptom).

#### 4.2 Decide next plan

Based on bottleneck, load the matching optimization skill:

| Bottleneck | Optimization skill | Typical first move |
|---|---|---|
| COMM_BOUND | `optimize-comm` | overlap, bucket, bf16 grad reduce, drop TP (esp. on AMD) |
| PIPELINE_BOUND | `optimize-pipeline` | raise vpp, raise gbs/M, rebalance stage |
| MEMORY_BOUND | `optimize-memory` | recompute selective→full, raise tp, alloc env |
| COMPUTE_BOUND | `optimize-compute` | raise mbs, kernel hints, threading env |
| MOE_DISPATCH_BOUND | `optimize-moe` | dispatch overlap, capacity_factor, ep adjustment (DeepEP if multi-node EP) |
| `env_suspect` non-empty | `env-catalog` | sweep ≤ 5 flags in one round before structural moves |

Pick **one** (occasionally two) candidate plans per round. Record what you intend to test and *why*, in one line:

```
round_3 candidate: r3_p1 from r2_champion, axis=vpp 1→2, expect bubble↓, predict tps=18500
```

#### 4.3 Run via subagent

Call the `run-and-profile` subagent with `purpose: candidate-eval` and the plan diff. **Never** run training in the main session. The subagent returns the markdown block defined in `run-and-profile`. Paste only that block into chat.

#### 4.4 Settle (think, not a tool)

Compare the new run's primary metric to the current champion:

- New TPS > champion × 1.02 → new run becomes champion; record old champion as "shelved" in chat.
- 0 < gain < 2% → keep current champion; mark new run as "shelved" (worth reviving for explore).
- OOM / NaN / regression → mark plan dead; do not retry the same axis combo.

Update an in-chat short ledger like:

```
Champion: r3_p1 (tps=18500, +5% vs r2)
Shelved: r1_p2, r2_p3, r3_p2
Dead:    r2_p4 (OOM), r3_p3 (numerical)
Tried axes around champion: vpp{1,2}, mbs{1,2}, NCCL_BUFFSIZE{8M,16M}
```

#### 4.5 Stop check

Stop when **any** is true:

- All hard constraints met AND last 2 rounds had gain < 2%
- `budget.max_rounds` reached
- `budget.total_gpu_h` reached
- No candidate left that is not in "tried axes" or "dead"

Otherwise loop to Step 4.1.

### Step 5: Think tips (avoid common loop pathologies)

- **One variable per candidate, always** (restated for emphasis — see top-of-file constraint). If you find yourself wanting to change two unrelated knobs in one candidate "to save a round", split it into two candidates and run them in parallel instead. The only exception is a coupled-flag bundle that *must* be set together to enable one feature.
- **Coupled flags must be declared explicitly**. When a candidate ships >1 flag, write the bundle reason in the `notes` field, e.g. `notes: "feature=deepep, requires use_turbo_deepep + moe_router_dtype + turbo_deepep_num_cu"`. Anything not justified this way is a rule violation — split it.
- **No repeat trials**: before launching a candidate, scan the in-chat ledger; if `(parent_id, axis_change)` is already in "tried axes around <parent>", reject and pick another axis.
- **Stop greedy after 2 stagnation rounds**: if 2 consecutive rounds had gain < 2%, force one explore round — derive the next candidate from a `shelved` (not the champion). This breaks local optima.
- **Diversify**: if the last 3 rounds all moved the same axis, prefer a different axis category next (e.g. after 3 NCCL flag rounds, try a structural move).
- **Parallelize within a round, never within a candidate**. A single round can spawn N parallel subagents (each one variable on top of the same parent); that's the right way to compress wallclock without violating the rule.
- **EnvSweep stays small**: cap each round at ≤ 8 parallel candidates × ≤ 50 steps. Lock structure first; each candidate flips one env flag.
- **Don't paste profile content**: if a subagent's summary references `artifacts.profile: <path>`, only `Read` that file when the summary itself is insufficient to decide; never copy raw profile JSON into chat.
- **Trust the subagent's bottleneck hint as a starting point, not a verdict**: re-read `bottleneck-diagnose` if the metrics look inconsistent (e.g. high `comm_ratio` but low `mem_peak` and low `bubble` is COMM_BOUND, not MIXED).

### Step 6: Report and persist learnings

When the loop stops, emit a final report **in chat**:

```markdown
## Tuning Report — <model> on <cluster>

- Sessions ID: <session>
- Rounds: <N>, GPU·h spent: <X>
- Baseline tps: <>, Final tps: <>, gain: <Y%>
- Constraints met: yes / no (which violated)

### Final plan
- parallelism: tp=<> pp=<> dp=<> ep=<> vpp=<>
- runtime: mbs=<> gbs=<> recompute=<>
- comm: bucket=<> overlap=<>
- env diff vs baseline: { ... }

### Decision trace
- r0 (baseline): tps=<>, COMM_BOUND
- r1: r0 + alltoall overlap → tps=<> (+%), PIPELINE_BOUND
- ...

### Lessons (carry forward)
- "MoE > 16 nodes always enable alltoall overlap" / etc.
```

Then write the same report to `output/pilot/<session_id>.md` so the next session on the same model+cluster can grep it for prior best.

## Worked example (Round-by-round, MoE 16 nodes on MI355X)

Every line below is one `run-and-profile` subagent invocation; each candidate changes exactly one variable (or one coupled-flag bundle) from its parent.

```
Round 0 (BASELINE — YAML as-given, no overrides)
  plan:    tp=2 pp=4 ep=8 mbs=1 recompute=full   (whatever the YAML had)
  env:     cluster baseline, no diff
  result:  tps= 9100, comm_ratio=0.42, bubble=0.10, mem=128GB
  hint:    COMM_BOUND

— Phase A: primus-defaults rollout (one feature per round) —————————————

Round 1 (feature=cross_entropy_loss_fusion)
  diff vs r0: cross_entropy_fusion_impl=te + cross_entropy_loss_fusion=true   [coupled bundle: 1 feature]
  result:  tps= 9550 (+4.9%)  ← champion

Round 2 (feature=apply_rope_fusion)
  diff vs r1: apply_rope_fusion=true (+ enable_experimental=true)   [coupled bundle: 1 feature]
  result:  tps= 9820 (+2.8%)  ← champion

Round 3 (feature=bf16-precision-aware-optimizer)
  diff vs r2: use_precision_aware_optimizer=true
              + main_grads_dtype=bf16 + exp_avg_dtype=bf16 + exp_avg_sq_dtype=bf16   [coupled bundle: 1 feature]
  result:  tps=10500 (+6.9%)  ← champion

Round 4 (feature=turbo-attention)
  diff vs r3: enable_primus_turbo=true + use_turbo_attention=true   [coupled bundle: 1 feature]
  result:  tps=10920 (+4.0%)  ← champion

Round 5 (feature=turbo-grouped-mlp)
  diff vs r4: use_turbo_grouped_mlp=true + use_turbo_fused_act_with_probs=true   [bundle]
  result:  tps=11340 (+3.8%)  ← champion

Round 6 (feature=fused-router)
  diff vs r5: moe_use_fused_router_with_aux_score=true
  result:  tps=11550 (+1.9%)  ← champion (gain ≥ 1% threshold for Phase A)

Round 7 (feature=sync-free-moe-stage-2)
  diff vs r6: turbo_sync_free_moe_stage=2
  result:  tps=11770 (+1.9%)  ← champion

Round 8 (feature=deepep)
  diff vs r7: use_turbo_deepep=true + turbo_deepep_num_cu=80 + moe_router_dtype=fp32   [bundle]
  result:  tps=12000 (+2.0%)  ← champion
  Phase A done. Cumulative gain so far: 1.32× over baseline.

— Phase B: bottleneck-driven tuning ——————————————————————————————————————

→ diagnose r8: COMM_BOUND, env_suspect=[NCCL_BUFFSIZE]

Round 9 (optimize-comm, 2 parallel candidates each one variable)
  P9a: drop tp 2→1 (AMD intra-node TP cost, mem ok)
  P9b: bucket 16→64 MB
  Results:
    P9a tps=15800 (+32%)  ← new champion
    P9b tps=13100 (+9%)   → shelved

Round 10 (env sweep on env_suspect, parallel — each candidate one flag)
  E10a: NCCL_BUFFSIZE=8M                      tps=15900 (+0.6%)
  E10b: NCCL_BUFFSIZE=16M                     tps=16400 (+3.8%)
  E10c: NCCL_BUFFSIZE=32M                     tps=15600 (-1.3%)
  E10d: NCCL_MIN_NCHANNELS=16                 tps=16100 (+1.9%)
  Promote winner E10b only (one var rule); next round can stack E10d on top.

Round 11 (env sweep continued, on top of E10b)
  E11: NCCL_MIN_NCHANNELS=16                  tps=16550 (+4.7%)  ← champion
  → diagnose: PIPELINE_BOUND (bubble rose to 0.18 once comm dropped)

Round 12 (optimize-pipeline, 2 parallel)
  P12a: vpp 1→2          tps=17600 (+6%)  ← champion
  P12b: mbs 1→2          OOM → dead

→ diagnose: COMPUTE_BOUND

Round 13 (optimize-compute)
  P13: mbs 1→3           tps=18100 (+2.8%)  ← champion (just above ε_promote=2%)

Round 14
  P14: recompute=selective   tps=18400 (+1.7%)
  2 consecutive rounds gain < 2% → STOP

Final: champion=P14, tps=18400 (2.02× over baseline)
       Decomposition (visible because each round changed exactly 1 variable):
         - Phase A primus-defaults  : 9100 → 12000   (1.32×, 8 features)
         - tp 2→1                   : 12000 → 15800  (+32%)
         - NCCL_BUFFSIZE=16M        : 15800 → 16400  (+3.8%)
         - NCCL_MIN_NCHANNELS=16    : 16400 → 16550  (+0.9%)
         - vpp 1→2                  : 16550 → 17600  (+6.3%)
         - mbs 1→3                  : 17600 → 18100  (+2.8%)
         - recompute=selective      : 18100 → 18400  (+1.7%)
       Cost: ~6.5 GPU·h (more rounds than batched, but every gain is attributable)
```

> **Why this is worth more rounds**: when the user sees the final report, every line of the decomposition is a single, reproducible move. They can drop the `recompute=selective` change later if they discover a numerical issue, without unwinding seven other things at the same time.

## Important Notes

- **One hard rule**: every "submit training + collect profile" goes through a subagent following `run-and-profile`. The main session never runs training directly and never reads profile content.
- **Chat is the state**. The ledger of (champion / shelved / dead / tried axes) lives in chat as plain markdown. Do not introduce YAML / JSON state files for this — they were the legacy design that this rewrite removed.
- **Skills are loaded on demand**. Do not preload `optimize-*` skills at the start; load only the one matching the current bottleneck.
- **One axis per move when possible**. Mixing two structural axes (e.g. vpp + mbs) in one candidate makes it impossible to attribute the gain.
- **Long-term memory** = `output/pilot/<session_id>.md` (final report) + `output/pilot/cluster-<cluster_id>.md` (preflight result). Nothing else needs to persist.
- **Conflict with the user goal wins over the skill defaults**: if the user says "max throughput, ignore memory", do not waste rounds on memory optimization.
