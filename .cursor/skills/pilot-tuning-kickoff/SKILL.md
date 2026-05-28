---
name: pilot-tuning-kickoff
description: Generate a ready-to-paste kickoff prompt that starts a Primus tuning session driven by `pilot/skills/tuning-loop/SKILL.md`. Use when the user attaches or names a Primus config YAML and asks for a tuning prompt — for example "give me a prompt to tune X.yaml" / "kick off a tuning session for X" / "generate a Pilot kickoff for examples/megatron/configs/.../*.yaml" / "调一下 这个 yaml" / "用 pilot 调优 X". Inspects the YAML, classifies which `primus-defaults` features are already on / missing / explicitly off / model-class-conditional, asks the user for cluster shape and budget, and emits a single copy-paste prompt plus a Phase A preview table.
---

# Pilot Tuning Kickoff Prompt Generator

When the user wants to tune a Primus YAML with Pilot, this skill produces the kickoff prompt for them to paste into the Cursor main chat. The output bundles: cluster + goal + budget, the hard rules from `pilot/skills/tuning-loop/SKILL.md`, a per-YAML feature audit (so the agent skips already-on `primus-defaults` features), and call-outs for any model-class peculiarities found in the YAML (debug flags, MLA, etc.).

This skill does NOT run any tuning itself — it only generates the prompt. The user pastes the prompt into a fresh chat (or the same one) to actually start `tuning-loop`.

## Workflow

### Step 1: Resolve the YAML path

The user typically provides the YAML by `@`-attaching it or naming it. Resolve to an absolute or workspace-relative path under `examples/`. If multiple are mentioned, ask the user which one to tune (one kickoff = one YAML).

If the path is missing, ask:

> Which Primus config YAML do you want to tune? (paste the path, e.g. `examples/megatron/configs/MI355X/qwen3_8B-BF16-pretrain.yaml`)

### Step 2: Read the YAML and audit `primus-defaults`

Use `Read` on the YAML once. Walk the `primus-defaults` feature list (below) and classify each row:

| Status | Meaning | What the prompt should say |
|---|---|---|
| `already_on` | YAML sets the bundle's flags to the recommended values | "SKIP in Phase A: <feature names>" |
| `missing` | Flag(s) absent from the YAML | "Phase A round candidate: <feature>" |
| `explicit_off` | YAML sets a flag to `false` deliberately | "Currently OFF — confirm with user before flipping (likely intentional)" |
| `n/a` | Condition not met (e.g. MoE feature on dense YAML) | omit |

The `primus-defaults` rollout (must stay in sync with `pilot/skills/primus-defaults/SKILL.md` "Feature rollout order"):

| # | Feature | Bundle (flags to grep for) | Condition |
|---|---|---|---|
| 1 | `cross-entropy-fusion` | `cross_entropy_fusion_impl: te` + `cross_entropy_loss_fusion: true` | always |
| 2 | `rope-fusion` | `apply_rope_fusion: true` (+ `enable_experimental: true`) | always |
| 3 | `bf16-precision-aware-optimizer` | `use_precision_aware_optimizer: true` + `main_grads_dtype: bf16` + `exp_avg_dtype: bf16` + `exp_avg_sq_dtype: bf16` | DP/FSDP, esp. multi-node |
| 4 | `turbo-attention` | `enable_primus_turbo: true` + `use_turbo_attention: true` | always |
| 5 | `turbo-parallel-linear` | `use_turbo_parallel_linear: true` | TP > 1 only |
| 6 | `turbo-grouped-mlp` | `use_turbo_grouped_mlp: true` + `use_turbo_fused_act_with_probs: true` | MoE only |
| 7 | `fused-router` | `moe_use_fused_router_with_aux_score: true` | MoE only |
| 8 | `sync-free-moe-stage-2` | `turbo_sync_free_moe_stage: 2` | MoE only |
| 9 | `deepep` | `use_turbo_deepep: true` + `moe_router_dtype: fp32` + `turbo_deepep_num_cu: <80 ep≤8 / 32 ep 16-64>` | MoE, EP > 1 |
| 10 | `gradient-accumulation-fusion` | `gradient_accumulation_fusion: true` | large dense / MoE; small dense YAMLs ship `false` on purpose |
| 11 | `legacy-grouped-gemm` | `moe_use_legacy_grouped_gemm: true` | MoE on AMD |
| 12 | `turbo-attention-fp8` | `enable_turbo_attention_float8: true` | FP8 mode only |

### Step 3: Audit non-`primus-defaults` peculiarities

Scan the YAML for these patterns and surface them in the prompt as "things to check with the user":

| Pattern | Why to flag |
|---|---|
| `moe_router_force_load_balancing: true` | Debug flag — flipping off changes measurement semantics |
| `multi_latent_attention: false` on a DeepSeek-V2/V3 YAML | MLA is a DeepSeek architecture feature; not a Pilot tuning candidate, but worth flagging |
| `mock_data: true` | Performance numbers are valid; convergence numbers are not |
| `train_iters` ≤ 50 | Too short for stable post-warmup metrics; `run-and-profile` will override `max_steps` to its own defaults |
| `disable_last_saving: true` | Fine for tuning; flag if user expected real checkpoints |
| Backend `framework: torchtitan` (not megatron) | Some `primus-defaults` flag names differ; `run-and-profile` translates per-backend |
| `tensor_model_parallel_size: ${PRIMUS_TP:N}` with `N > 1` on AMD | TP > 1 default — `optimize-comm` AMD note will likely surface "drop TP" early |

Infer the **model class** (Dense / MoE / FP8) from the YAML — needed to know which Phase A features apply:

- MoE if `expert_model_parallel_size > 1` or any `moe_*` flag present
- FP8 if filename contains `FP8` or backend FP8 flag is set
- Otherwise Dense

Infer the **cluster class** from the path: `MI300X` / `MI355X` / `H100` segment in the YAML path → use it to suggest a cluster_id default.

### Step 4: Gather the runtime slots from the user

Use AskQuestion (or conversational ask if it's not available) to collect:

| Slot | Format | Default if user is vague |
|---|---|---|
| Cluster mode | `single` / `slurm` | `single` if `pp=1, dp_shard ≤ gpus_per_node` |
| Nodes | int | `1` for single; ask for slurm |
| GPUs/node | int | `8` |
| Cluster id (for cluster baseline file) | string | `<cluster_class>-<nodes>node` (e.g. `mi300x-1node`) |
| Image | string | `rocm/primus:v26.2` |
| Primary metric | `tps` / `mfu` / `step_time` | `tps` |
| Hard constraints | list | `mem_peak_gb <= 0.9 × hbm`, `no NaN` |
| Budget: max_rounds | int | `10–15` for Phase A + B |
| Budget: total_gpu_h | float | a sensible cap based on cluster size and round count |
| Budget: wallclock_h | float | typically 2× total_gpu_h for serial rounds, less if parallel |

If the user has already given you these (e.g. in their initial message), don't re-ask — just confirm in the emitted prompt's "Cluster" / "Goal" sections.

### Step 5: Emit the kickoff prompt

Use the template below. Fill every `<…>` placeholder with concrete values from Steps 1–4. Keep the `Hard rules` and `Pre-checks` sections verbatim — they are the contract with `tuning-loop`.

```
Use pilot/skills/tuning-loop/SKILL.md to tune this Primus training:

@<yaml_path>

Cluster:
  mode: <single|slurm>
  nodes: <N>
  gpus_per_node: <M>
  partition: <slurm partition>          # slurm only
  nodelist: <bracket-compressed list>   # slurm only, optional
  image: <docker image>
  cluster_id: <id-for-output/pilot/cluster-<id>.md>

Goal:
  primary: <tps|mfu|step_time>
  constraints:
    - <constraint 1>
    - <constraint 2>
  budget:
    max_rounds: <N>
    total_gpu_h: <X>
    wallclock_h: <Y>

Pre-checks before you start:
  - Confirm HF_TOKEN is exported (this YAML uses <mock_data | real data>).
  - Confirm a Primus container is up on the target node(s) with the image above.
    If not, surface the error and stop — Pilot does not run docker / salloc.
  - If output/pilot/cluster-<cluster_id>.md is missing or > 7 days old, run preflight first.

Hard rules to follow (from pilot/skills/tuning-loop/SKILL.md):
  - Every "submit + collect profile" goes through a subagent following
    pilot/skills/run-and-profile/SKILL.md. Never run training in this main chat
    and never paste profile / log content back.
  - One variable per candidate, always. The only exception is a coupled-flag
    bundle as defined in pilot/skills/primus-defaults/SKILL.md.
  - BASELINE = the YAML as-given, no overrides.

This YAML's primus-defaults audit (use this to plan Phase A):
  - Already on (SKIP in Phase A):
      <comma-separated feature names from already_on classification>
  - Missing (Phase A round candidates, in this order):
      1. <feature name>  — <bundle>
      2. <feature name>  — <bundle>
      ...
  - Explicitly OFF (confirm with user before flipping):
      <feature name>: <flag = false>   # likely intentional, ask first
  - Not applicable (model class doesn't need it):
      <feature names>

Things to ask the user before starting (do not silently change):
  <one-line per peculiarity from Step 3, e.g.:>
  - moe_router_force_load_balancing: true is set — ok to flip off so router behaves realistically?
  - multi_latent_attention: false on DeepSeek V2 — Pilot will not enable MLA as a tuning move (architecture decision); confirm this is intentional.

Persist results:
  - Per-run artifacts under output/pilot/runs/<run_id>/.
  - Final report at output/pilot/<exp_name>-<cluster_id>-<session>.md.
```

After emitting the prompt, also print:

### Step 6: Print the Phase A preview table

A short table the user can sanity-check before pasting:

```markdown
## Phase A preview for <yaml_basename>

| Round | Feature | Bundle | Expected gain |
|---|---|---|---|
| r1 | <missing feature 1> | <flags> | <range> |
| r2 | <missing feature 2> | <flags> | <range> |
| ... |

Phase A total: ~<N> rounds, ~<X> GPU·h.
Phase B (bottleneck-driven) starts at r<N+1>; expected first bottleneck: <COMM/MOE/COMPUTE> based on the YAML's <ep / mbs / parallelism> shape.
```

### Step 7 (optional): Suggest follow-ups

If you spotted things that aren't blockers but might matter, list them as suggestions the user can add to the prompt manually. Examples:

- "Want to compare BF16 vs FP8? After Phase A finishes, fork a subagent that flips to FP8 mode and re-baselines."
- "If multi-node, expect `deepep` and `bf16-precision-aware-optimizer` to dominate Phase A gains."
- "If the user wants to keep it short (< 2 GPU·h), reduce `max_rounds` to 6 and skip Phase A features with expected gain < 2%."

Keep this section to ≤ 3 bullets.

## Examples

### Example 1: Single-node, mostly-defaults-on YAML

User: "Generate a kickoff prompt for `@examples/megatron/configs/MI300X/deepseek_v2_lite-BF16-pretrain.yaml`"

Skill output:

> The YAML already has 8 of 12 `primus-defaults` features on (Turbo, DeepEP, sync-free MoE, fused router, CE fusion, overlap flags). Phase A reduces to 3–4 rounds. Cluster default: 1 node × 8 MI300X. Asking for budget...
>
> [emits the prompt with Phase A = `rope-fusion`, `bf16-precision-aware-optimizer`, `turbo-grouped-mlp` (already on, skip), `gradient-accumulation-fusion` (currently false — flag for user)]
>
> Plus call-out: `moe_router_force_load_balancing: true` and `multi_latent_attention: false` should be confirmed with the user.

### Example 2: Brand-new YAML with nothing on

User: "调一下 `@examples/megatron/configs/MI355X/some_new_model-BF16-pretrain.yaml`"

Skill output:

> YAML is bare — only baseline parallelism + data. All 12 `primus-defaults` features are candidates for Phase A (10 if non-MoE, 8 if dense + small). Estimated Phase A: 8–10 rounds, ~1.5 GPU·h on single-node MI355X. After Phase A, bottleneck-diagnose will likely surface MOE_DISPATCH_BOUND if MoE...
>
> [emits prompt with full Phase A list]

### Example 3: User attaches YAML + already states cluster

User: "@a-config.yaml — I have a slurm allocation for 4 nodes on amd-aig partition, just give me the prompt"

Skill: Skip the AskQuestion for cluster (user gave it); fill `mode: slurm, nodes: 4, partition: amd-aig`. Default budget guesses, audit YAML, emit prompt.

## Important Notes

- **One YAML per kickoff prompt.** Each kickoff starts one Pilot session against one model+cluster combo. Do not bundle multiple YAMLs.
- **Do not run any training, preflight, or subagent.** This skill emits a prompt only. The user (or a fresh chat) executes it.
- **Stay in sync with `pilot/skills/primus-defaults/SKILL.md`.** When that skill's feature list changes, update the table in Step 2 here. Keep the order identical.
- **`@<yaml_path>` is mandatory in the emitted prompt.** It forces the YAML into the next chat's context, so `tuning-loop` doesn't have to re-discover it.
- **Don't make tuning decisions** — flag peculiarities (debug flags, architecture flags like MLA) for the user to confirm; never pre-decide them in the prompt.
- **Use AskQuestion when available**, conversational asks otherwise. The slots in Step 4 should not be left blank; either ask or use the listed default and explicitly note the default in the emitted prompt.
- **Cluster mode prerequisites are not Pilot's job** — restate this in the Pre-checks section verbatim so the user is reminded that they need a working container / SLURM allocation already.
- **The output is markdown, not a code block.** Emit the kickoff prompt as a fenced ` ``` ` code block so the user can copy it cleanly; the preview table follows as plain markdown below.
