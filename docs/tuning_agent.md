# Tuning Agent

The **Tuning Agent** is an LLM-driven search for a near-optimal Primus
**training configuration** — the full parallelism strategy *plus* the coupled
batching, pipeline-schedule, memory, MoE-communication, and precision knobs —
on a target GPU cluster, **without running the workload at scale**. It drives
the [Primus Projection](./projection.md) tool (memory + simulate + optional
benchmark) as an evaluation oracle and returns the configuration that maximizes
`tokens/s/GPU` subject to a per-GPU memory safety margin. See
[Knobs Searched](#knobs-searched) for the full set of levers it tunes.

- **Package**: [`primus/agents/tuning_agent/`](../primus/agents/tuning_agent/)
- **Entry point**: `python -m primus.agents.tuning_agent`

This document is both the **user/operator guide** (installation, configuration,
execution modes, CLI reference, worked example, troubleshooting) and the
**design write-up** (paper-ready problem statement and deferred future
features) for the agent.

---

## Table of Contents

1. [Why a Tuning Agent](#why-a-tuning-agent)
2. [How It Works](#how-it-works)
3. [Knobs Searched](#knobs-searched)
4. [Installation](#installation)
5. [LLM Setup](#llm-setup)
6. [Quickstart](#quickstart)
7. [Execution Modes](#execution-modes)
8. [CLI Reference](#cli-reference)
9. [Target-Cluster YAML](#target-cluster-yaml)
10. [The Search Loop](#the-search-loop)
11. [Evaluator and Projection Modes](#evaluator-and-projection-modes)
12. [Output Artefacts](#output-artefacts)
13. [Worked Example](#worked-example)
14. [Troubleshooting](#troubleshooting)
15. [Limitations](#limitations)
16. [Design Notes & Future Features](#design-notes--future-features)

---

## Why a Tuning Agent

Choosing a training configuration for a large training (or inference) workload
is a combinatorial problem. The configuration is the joint choice of the
parallelism dimensions:

- **Data Parallel (DP)** — derived from world size and the other axes,
- **Tensor Parallel (TP)**,
- **Expert Parallel (EP)** — for MoE models,
- **Context Parallel (CP)**,
- **Pipeline Parallel (PP)**, with virtual pipeline (**VPP**) and the
  **pipeline schedule** (1F1B / interleaved / zero-bubble / ZBV-\* /
  Megatron-ILP),

together with the strongly coupled training knobs that decide how those
dimensions translate into in-flight work and memory pressure: global batch
size (**GBS**), micro batch size (**MBS**), activation recomputation
(`recompute_granularity`, `recompute_num_layers`), the overlap flags
(`overlap_grad_reduce`, `overlap_param_gather`), and a set of higher-impact
levers — FP8 precision, MoE DeepEP / sync-free communication, fused
cross-entropy, and optimizer-state sharding (distributed optimizer / FSDP2).
The full set is enumerated in [Knobs Searched](#knobs-searched).

The legal Cartesian product is large, the objective surface is non-convex and
architecture-specific, and exhaustive evaluation on real hardware is
infeasible. The agent replaces exhaustive sweeps with a small number of
informed analytical evaluations: an **LLM proposes** candidate configurations
conditioned on the resolved model architecture, the target cluster, and the
history of prior trials; the **projection evaluator scores** them for memory
feasibility and throughput. The loop keeps an incumbent and returns the best
legal configuration within a user-specified budget.

---

## How It Works

```
┌────────────────────────────────────────────────────────────────────────┐
│  workload YAML ──► resolve architecture record (layers, hidden, MoE …)   │
│  target-cluster YAML ──► cluster + budget + LLM config                   │
│                                                                          │
│   derive per-axis legal sets (divisibility + cluster size, in code)      │
│                              │                                           │
│            ┌─────────────────┴─────────────────┐                        │
│            ▼                                     ▼                        │
│   systematic seed grid                 DSPy planner + RLM rounds          │
│   (no LLM, coarse TP/PP/EP/CP)         (LLM proposes, tools score)        │
│            │                                     │                        │
│            └──────────────► Evaluator ◄──────────┘                       │
│                  projection memory / performance                         │
│                  (simulate · no GPU | benchmark · GPU)                    │
│                              │                                           │
│                              ▼                                           │
│        trials.jsonl · trials.png · scratchpad.txt · summary.json          │
│              best config (PRIMUS_* exports + re-runnable YAML)            │
└────────────────────────────────────────────────────────────────────────┘
```

The key idea is a separation of concerns: **legality is computed in code**
(the LLM can never spend budget proposing an obviously illegal config), while
**strategy is delegated to the LLM** (which axis to push next, when to trade
recompute for MBS, whether CP helps a given MoE shape).

---

## Knobs Searched

The agent sweeps far more than the five parallelism dimensions. Its trial
configuration (`TrialConfig` in `legality.py`) carries the full set of knobs
below; every knob is either **inherited from the workload YAML** (when the
agent leaves it unset) or **overridden for a trial** and translated into the
corresponding `projection` flags by the evaluator. Each is legality-checked in
code before it ever reaches the projection tool.

### Parallelism & batching

| Knob | Legal values | What it controls |
|------|--------------|------------------|
| `tp` | divisors of `num_attention_heads` **and** `hidden_size`, ≤ `gpus_per_node` | Tensor parallelism |
| `pp` | divisors of `num_layers` (plus layout-aware depths 2/4/8/16 for layout workloads, and the workload's own PP) | Pipeline parallelism |
| `ep` | divisors of `num_experts` (MoE only, else 1) | Expert parallelism |
| `cp` | dense: divisors of `seq_length` ≤ `gpus_per_node`; MoE: `CP ≤ EP` and `EP % CP == 0` (parallel folding) | Context parallelism |
| `vpp` | divisors of `num_layers / PP` (≤ 8), plus layout-aware VPP | Virtual pipeline (interleaving) |
| `mbs` | powers of two ≤ 16 | Micro batch size |
| `gbs` | multiple of `MBS × DP` | Global batch size |
| `overlap_grad_reduce` | `true` / `false` | Overlap the DP gradient all-reduce with backward |

### Pipeline schedule

| Knob | Legal values | What it controls |
|------|--------------|------------------|
| `pp_schedule` | VPP=1: `auto`, `zerobubble`, `zerobubble-heuristic`, `seaailab-ilp`; VPP=2: `auto`, `zbv-formatted`, `zbv-greedy-half`, `zbv-greedy-min`; other VPP: `auto` | Schedule algorithm (paired with VPP) |
| `enable_zero_bubble` | `true` / `false` / inherit | Split backward into B + W to fill pipeline bubbles |

### Memory levers

| Knob | Legal values | What it controls |
|------|--------------|------------------|
| `recompute_granularity` | `none` / `selective` / `full` | Activation recomputation strategy |
| `recompute_num_layers` | int (≤ layers per VPP stage) | Layers recomputed per stage under `full` |
| `cross_entropy_loss_fusion` | `true` / `false` / inherit | Fused cross-entropy — large-vocab memory + compute win |
| `use_distributed_optimizer` | `true` / `false` / inherit | ZeRO-1 optimizer-state sharding across DP |
| `use_torch_fsdp2` | `true` / `false` / inherit | FSDP2 sharding (mutually exclusive with `use_distributed_optimizer`) |

### MoE communication — *MoE only, high impact*

| Knob | Legal values | What it controls |
|------|--------------|------------------|
| `use_turbo_deepep` | `true` / `false` / inherit | DeepEP dispatch/combine kernels for MoE All-to-All (large MoE-comm win) |
| `sync_free_stage` | `0` / `1` / `2` / `3` | Sync-free MoE pipelining; stage ≥ 2 auto-enables DeepEP |
| `target_ep_size` | positive int / inherit | EP override used for All-to-All modeling |

### Precision — *high impact*

| Knob | Legal values | What it controls |
|------|--------------|------------------|
| `fp8` | `none` / `hybrid` (also `e4m3`, `delayed`) | FP8 on linear layers — roughly 2× compute on GEMMs |

### Coupling rules enforced in code

Some knobs interact; the validator rejects incoherent combinations before any
projection call, so the LLM never wastes budget on them:

- `use_torch_fsdp2` and `use_distributed_optimizer` are **mutually exclusive**
  (FSDP2 already shards the optimizer state).
- `sync_free_stage ≥ 2` **auto-enables** DeepEP, so `use_turbo_deepep=false`
  alongside it is contradictory.
- `use_turbo_deepep`, `sync_free_stage`, and `target_ep_size` are **MoE-only**
  and rejected on dense workloads.
- `fp8` combined with MLA attention is steered to a `tensorwise` recipe for
  container compatibility.

The `optimization.axes` toggles in the target-cluster YAML gate the core
parallelism / batching / schedule / recompute / overlap axes (set one to
`false` to pin it to the workload baseline). The remaining Tier-A knobs
(DeepEP, sync-free, FP8, loss fusion, distributed-optimizer / FSDP2) are
explored by the LLM proposer whenever they apply to the workload.

---

## Installation

```bash
python3 -m venv .venv-agent
source .venv-agent/bin/activate
pip install -r primus/agents/tuning_agent/requirements.txt

# Optional: only needed for the `simulate` projection backend (no GPU).
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

The agent itself is lightweight; the heavy dependencies (torch, Megatron,
Origami) are only needed by the evaluator paths you actually use:

| You want to run…                              | You need…                              |
|-----------------------------------------------|----------------------------------------|
| `--mode dry` (loop sanity check)              | nothing beyond the agent requirements  |
| `--mode memory-real`                          | Primus + torch (CPU is fine)           |
| `--profiling-mode simulate`                   | + Origami                              |
| `--profiling-mode benchmark`                  | + a ROCm GPU on the local host         |

---

## LLM Setup

The agent uses [DSPy](https://dspy.ai), which routes LLM calls through
[LiteLLM](https://docs.litellm.ai/docs/providers) internally — **no separate
proxy process is required**. Set credentials for whichever provider you use:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export LLM_MODEL=openai/gpt-4o            # default if unset

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=anthropic/claude-opus-4-5

# Any OpenAI-compatible endpoint (Ollama, vLLM, local LiteLLM proxy, …)
export OPENAI_API_KEY=dummy              # or the real key if required
export OPENAI_API_BASE=http://localhost:11434/v1
export LLM_MODEL=openai/llama3
```

The model string follows LiteLLM's provider-prefixed convention
(`<provider>/<model-name>`). Credentials can also live in a `.env` file; the
agent searches `$CWD/.env`, `<repo-root>/.env`, then `~/.env`. You may also
set them in the target-cluster YAML under `agent.llm` (see below); the
environment takes precedence over the YAML for `LLM_MODEL`.

> The LLM stage is **optional**. `--seed-only` runs the systematic seed grid
> against the real projection tool and reports the best seed config with no LLM
> credentials at all.

---

## Quickstart

```bash
# 1) Dry-run: no primus-cli, no LLM. Exercises the loop end-to-end with
#    synthesised metrics — verifies the install on a CPU-only host.
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_4nodes.yaml \
    --out-dir tuning_runs/dry-run \
    --dry-run --seed-only

# 2) Seed-only with the real projection tool (Origami needed for simulate):
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_4nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-seed \
    --seed-only

# 3) Full agent (planner + DSPy.RLM rounds):
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_4nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-full
```

---

## Execution Modes

The agent has two orthogonal mode switches: **what the evaluator does**
(`--mode`) and **whether the LLM stage runs** (`--seed-only` / `--agent-only`).

### Evaluator mode (`--mode`)

| Mode          | GPU | What it does |
|---------------|-----|--------------|
| `dry`         | No  | Synthesises metrics; never calls `primus-cli`. For testing the agent loop on a host with no Primus/torch/Origami install. (`--dry-run` is shorthand.) |
| `memory-real` | No  | Calls `projection memory` for real and synthesises `tokens/s` from a memory + axes heuristic. Use when `projection performance` cannot run (no Origami / no GPU). |
| `full`        | No\* | Calls `projection memory` **and** `projection performance`. Default. `simulate` profiling needs Origami (no GPU); `benchmark` profiling needs a GPU. |

\* `full` + `--profiling-mode simulate` runs entirely on CPU.

### LLM stage

| Flag           | Effect |
|----------------|--------|
| (default)      | Run the systematic seed grid, then the DSPy planner + RLM rounds. |
| `--seed-only`  | Evaluate the systematic seed plan only, then exit. No LLM. |
| `--no-agent`   | Alias for `--seed-only`. |
| `--agent-only` | Skip the seed grid; run the LLM agent on the existing `trials.jsonl`. |
| `--resume`     | Reuse an existing `trials.jsonl` in `--out-dir` (suppresses the overwrite warning). |

---

## CLI Reference

```bash
python -m primus.agents.tuning_agent \
    --workload <primus_pretrain.yaml> \
    --target-cluster <target_cluster.yaml> \
    [--out-dir <dir>] \
    [--mode {dry,memory-real,full}] \
    [--profiling-mode {simulate,benchmark}] \
    [--dry-run] [--seed-only] [--no-agent] [--agent-only] [--resume] \
    [--seed-budget N]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--workload` | *(required)* | Path to a Primus pretrain YAML. The agent follows the `extends:` chain to flatten the architecture. |
| `--target-cluster` | *(required)* | Path to a target-cluster YAML (see below). |
| `--out-dir` | `./tuning_runs/<cluster-name>` | Output directory for trials, plot, scratchpad, summary. |
| `--mode` | `full` | Evaluator mode (`dry` / `memory-real` / `full`). |
| `--profiling-mode` | `simulate` | Profiling backend for `projection performance` in `full` mode. `simulate` = Origami, no GPU; `benchmark` = real GPUs via `torch.distributed.run` on the local node, projected to `target_cluster.num_nodes` (requires `has_gpu: true`). |
| `--dry-run` | off | Shorthand for `--mode dry`. |
| `--seed-only` | off | Evaluate the systematic seed plan only; skip the LLM. |
| `--no-agent` | off | Alias for `--seed-only`. |
| `--agent-only` | off | Skip seed evaluation; run the LLM agent on existing history. |
| `--resume` | off | Reuse an existing `trials.jsonl` in `--out-dir`. |
| `--seed-budget` | `12` | Max seed candidates evaluated before the LLM takes over. |

---

## Target-Cluster YAML

A thin wrapper around the existing Primus `hardware_config` convention, so no
new networking format has to be invented; topology, bandwidths, and latencies
are consumed by the analytical communication model (see
[`projection.md` → Communication Modeling](./projection.md#communication-modeling)).

A complete example ships at
[`examples/agents/tuning_agent/target_cluster_mi355x_4nodes.yaml`](../examples/agents/tuning_agent/target_cluster_mi355x_4nodes.yaml):

```yaml
target_cluster:
  name: mi355x_4nodes
  num_nodes: 4
  gpus_per_node: 8
  gpu_arch: mi355x                          # selects the Origami / SDPA hardware profile
  hardware_config: examples/hardware_configs/mi355x.yaml
  gpu_clock_mhz: null                       # optional Origami clock override

available_for_benchmark:
  has_gpu: true                             # true → --profiling-mode benchmark enabled
  benchmark_gpus: 8                         # GPUs the agent may use locally
  benchmark_arch: mi355x                    # arch of the local host (may differ from target)

optimization:
  objective: tokens_per_s_per_gpu
  memory_safety_margin: 0.10                # hard cap = HBM × (1 − margin)
  hbm_capacity_gb: 288.0                    # MI300X/MI325X/MI355X = 192 / 256 / 288
  budget:
    max_proposals: 30                       # LLM-proposed candidates
    max_perf_calls: 30                      # simulate calls
    max_benchmark_calls: 12                 # GPU benchmark calls (ignored if has_gpu=false)
    max_rounds: 8                           # outer agent rounds
    max_rlm_iterations: 30                  # inner RLM iterations per round
  axes:                                     # toggle individual axes off to fix them
    tp: true
    pp: true
    ep: true
    cp: true
    mbs: true
    gbs: false
    vpp: true
    pp_schedule: true
    recompute: true
    overlap_grad_reduce: false

agent:
  llm:
    model: anthropic/claude-opus-4-5        # optional; env LLM_MODEL overrides
    api_key: sk-ant-...                     # optional if set in env
    base_url: ...                           # optional; override for custom endpoints
  prompt_extras:                            # optional free-form context for the LLM
    - "Prefer keeping EP within a single node when possible."
```

### Field reference

| Block / field | Default | Meaning |
|---------------|---------|---------|
| `target_cluster.num_nodes` / `gpus_per_node` | `1` / `8` | Target world size = `num_nodes × gpus_per_node`. |
| `target_cluster.gpu_arch` | `mi355x` | Selects the Origami / SDPA hardware profile for `simulate`. |
| `target_cluster.hardware_config` | `null` | Path to a Primus hardware YAML for communication modeling. |
| `available_for_benchmark.has_gpu` | `false` | When `true`, the `benchmark` profiling path is enabled. |
| `available_for_benchmark.benchmark_gpus` | `0` | GPUs the agent may use for sub-node benchmarking. |
| `optimization.objective` | `tokens_per_s_per_gpu` | The scalar maximized by the search. |
| `optimization.memory_safety_margin` | `0.10` | Feasibility cap: `projected_mem ≤ HBM × (1 − margin)`. |
| `optimization.hbm_capacity_gb` | `192.0` | Per-GPU HBM capacity used for the cap. |
| `optimization.budget.*` | see above | Stops the search when any budget is exhausted. |
| `optimization.axes.*` | most `true` | Set an axis `false` to pin it to the workload baseline. |
| `agent.llm.*` | env-resolved | Per-cluster LLM overrides (model / key / base_url). |
| `agent.prompt_extras` | `[]` | Free-form hints appended to the LLM prompt. |

---

## The Search Loop

1. **Resolve the workload.** Load the workload YAML, follow
   `modules.pre_trainer.model` into `primus/configs/models/megatron/<model>.yaml`,
   chase the `extends:` chain, and flatten an **architecture record**
   (`num_layers`, `hidden_size`, `ffn_hidden_size`, `num_attention_heads`,
   `num_query_groups`, `num_experts`, `moe_router_topk`, `moe_ffn_hidden_size`,
   attention type, `seq_length`, …) plus a **baseline overrides record** (the
   current TP/PP/EP/CP/MBS/GBS/recompute) used as the search starting point.

2. **Derive legal axes in code** (not delegated to the LLM):

   | Axis | Legality |
   |------|----------|
   | TP | divides `num_attention_heads` and `hidden_size`; ≤ `gpus_per_node` |
   | PP | divides `num_layers` (or any if explicit `pipeline_model_parallel_layout`) |
   | EP | divides `num_experts` (MoE only), else 1 |
   | CP | dense: free; MoE: `EP % CP == 0` and `CP ≤ EP` (parallel folding) |
   | DP | derived: `world_size / (TP × PP × EP)` (MoE) or `world_size / (TP × PP × CP)` (dense) |
   | MBS | positive integer with `GBS % (MBS × DP) == 0` |
   | GBS | multiple of `MBS × DP` |
   | VPP | divides `num_layers / PP` |
   | pipeline schedule | depends on VPP: 1F1B / Megatron-ILP / Zero-Bubble need VPP=1; Interleaved 1F1B needs VPP>1; ZBV-\* need VPP=2 |
   | recompute | `full` / `selective` / off; `recompute_num_layers ≤ layers_per_vpp_stage` |

3. **Seed** (no LLM): the workload baseline plus a coarse legal grid over
   (TP, PP, EP, CP), capped at `--seed-budget` candidates, each filtered by
   `projection memory` and scored by `projection performance`.

4. **LLM rounds** (DSPy `ChainOfThought` planner + `RLM` driver): each round
   sees the architecture, cluster, axis legality, the full compressed trial
   history, and a durable scratchpad, and proposes `k` new candidates. The
   agent validates legality in code, filters with memory, scores survivors,
   appends `(config, result)` to history, and updates the incumbent. It
   early-stops when no improvement occurs within budget.

5. **Emit** the best config (as `PRIMUS_*` env exports and a re-runnable
   workload overlay YAML), the trial log, and the plot.

---

## Evaluator and Projection Modes

The evaluator wraps the Primus Projection CLI behind a uniform interface, so
the agent does not need to know which mode produced a number:

```text
evaluate(config) -> {
  legal: bool,
  reason: str | None,                 # only when legal=False
  memory_per_gpu_gb: float,
  param_optimizer_gb: float,
  activation_gb: float,
  iteration_ms: float | None,
  tokens_per_s_per_gpu: float | None,
  tflops_per_s_per_gpu: float | None,
  bubble_ratio: float | None,
  comm_breakdown: dict[str, float],   # ms by collective
  source: "memory_only" | "simulate" | "benchmark" | "both",
}
```

The wrapper always picks the **cheapest sufficient** mode: a memory-only
pre-filter to reject infeasible configs before paying for a performance call,
then `simulate` (or `benchmark` for promising candidates when a GPU is
available). The tool belt exposed to the LLM mirrors this:

- `evaluate_memory_only(config_json)` — cheap pre-filter
- `evaluate_simulate(config_json)` — primary scoring path
- `evaluate_with_benchmark(config_json)` — only if `has_gpu: true`
- `get_history`, `get_best`, `get_legal_axes`, `get_architecture`,
  `get_cluster`, `get_budget_status`
- `note_to_scratchpad`, `read_scratchpad`
- `query_llm(prompt, system?)` — one-shot "LLM-inside-LLM" consultation

For the underlying projection math — memory components, the simulate vs.
benchmark trade-off, and the benchmark-based memory projection the agent uses
for OOM-accurate feasibility — see [`projection.md`](./projection.md).

---

## Output Artefacts

Everything lands in `--out-dir`:

```
trials.jsonl       all attempted configs and their results
trials.png         incumbent objective vs. trial number
scratchpad.txt     durable LLM notes carried across rounds
summary.json       agent-summarised winner
trials/*.yaml      one re-runnable workload-overlay YAML per trial
```

The run also prints the best configuration and ready-to-paste exports:

```text
=== BEST CONFIGURATION ===
  trial #17  source=simulate
  TP=1 PP=4 EP=8 CP=1
  MBS=2 GBS=128 VPP=2 schedule=interleaved_1f1b
  recompute=selective/None
  → tokens/s/GPU = 3,260
  → memory/GPU   = 182.8 GB

=== EXPORTS for primus-cli ===
  export PRIMUS_TP=1
  export PRIMUS_PP=4
  export PRIMUS_EP=8
  # then pass --micro-batch-size 2 --global-batch-size 128
```

---

## Worked Example

Search for the best Mixtral 8×22B configuration on a 4-node MI355X pod, with a
single idle 8-GPU node available for benchmarking:

```bash
# .env or exported: OPENAI_API_KEY / ANTHROPIC_API_KEY + LLM_MODEL

python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_4nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-mi355x \
    --mode full --profiling-mode benchmark
```

What happens:

1. The agent resolves Mixtral 8×22B (56 MoE layers, hidden 6144, 8 experts,
   topk 2) and derives legal axes given 32 GPUs.
2. The seed grid + LLM rounds propose configs; each is memory-filtered, then
   scored. With `--profiling-mode benchmark`, promising candidates are
   benchmarked on the local 8-GPU node and projected to 4 nodes.
3. The best legal config (highest `tokens/s/GPU` under the 10% memory margin)
   is printed with `PRIMUS_*` exports, and all trials are written to
   `tuning_runs/mixtral-22b-mi355x/`.

For a pure capacity-planning run with **no GPU**, swap in
`--profiling-mode simulate` and set `has_gpu: false` in the cluster YAML.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `WARNING: cannot import agent (...); skipping LLM stage` | `dspy` / `python-dotenv` not installed. Install `primus/agents/tuning_agent/requirements.txt`, or run with `--seed-only`. |
| `--profiling-mode benchmark but available_for_benchmark.has_gpu=false` | Benchmark trials are marked illegal. Set `has_gpu: true` (and `benchmark_gpus`) in the target-cluster YAML, or use `--profiling-mode simulate`. |
| `simulate` errors about Origami | Install Origami (see [Installation](#installation)) or fall back to `--mode memory-real`. |
| `<file> exists; re-using` warning | A prior `trials.jsonl` is in `--out-dir`. Pass `--resume` to reuse it intentionally, or point `--out-dir` elsewhere. |
| `no legal trial found` | Every candidate failed memory/legality. Loosen `memory_safety_margin`, raise `hbm_capacity_gb`, or widen the legal axes. |
| LLM stage produces no improvement | The seed incumbent may already be near-optimal; check `trials.png`. Increase `budget.max_rounds` / `max_proposals` to explore further. |

---

## Limitations

These are honest caveats, not future features:

1. **Simulator-vs-reality gap.** `--profiling-mode simulate` is analytical;
   [`projection.md`](./projection.md) explains why benchmarking is more
   accurate. In no-GPU mode the agent's numbers are analytical, not anchored.
   When a GPU is available, run a few `benchmark` trials as a sanity check.
2. **Memory-projection blind spots** in `simulate`: A2A buffers, allocator
   fragmentation, and communication scratch are not fully modeled. The
   `memory_safety_margin` compensates conservatively; the benchmark-based
   memory path (see `projection.md`) closes most of this gap.
3. **Search-space explosion** with all axes on. The agent mitigates by tiering
   — coarse over (TP, PP, EP, CP), polish over (MBS, schedule, recompute).
4. **Cluster-description lossiness**: averaged bandwidth/latency cannot capture
   contention or per-rail asymmetry.
5. **Schedule × VPP legality** is a coded table, not delegated to the LLM.

---

## Design Notes & Future Features

> This section captures the design rationale and the paper-ready problem
> statement, plus the list of features deliberately deferred from v1.

### Problem statement (paper-ready)

We address the problem of **automatically selecting a near-optimal
training configuration — the parallelism strategy plus the coupled batching,
schedule, memory, MoE-communication, and precision knobs — for a large-scale
distributed training workload on a target GPU cluster, without executing the
workload at scale**. The
configuration space is combinatorial: for each axis only a small set of values
is *legal* (constrained by divisibility against `num_attention_heads`,
`num_experts`, `num_layers`, `GBS / (MBS × DP)`; by the minimum-GPU requirement
`TP × PP × CP` for dense models or `TP × PP × EP` for MoE under parallel
folding; by per-GPU memory; and by axis interactions such as `EP % CP == 0` or
schedule-specific VPP), but their legal Cartesian product is large enough that
exhaustive evaluation on real hardware is infeasible.

The objective surface is non-convex and architecture-specific: increasing TP
reduces per-GPU parameters and activations but introduces AllReduce on the
critical path; increasing PP reduces per-GPU parameters but introduces bubbles
and requires `GA ≥ PP`; increasing EP shards expert weights but introduces
inter-node All-to-All when EP spans nodes; increasing MBS improves GEMM
efficiency but linearly inflates activation memory; recomputation trades ~33%
compute for one-to-two orders of magnitude less activation memory on MoE layers.

We formulate the search as **LLM-as-policy over a hybrid analytical /
benchmark-driven evaluator**. The evaluator is the Primus Projection tool,
which provides three calls of increasing fidelity and cost: (i) an analytical
**memory projection**; (ii) a fully analytical **performance projection** built
on the Origami GEMM model and an SDPA simulator; and (iii) a **hybrid
benchmark** that measures per-layer compute on as few as one GPU and
analytically scales PP/EP/DP to the target cluster. The LLM proposes candidate
configurations conditioned on the resolved architecture, the cluster
description, and the history of prior trials; the evaluator returns memory
feasibility, throughput, and a structured breakdown. The loop returns, within a
user-specified budget, the configuration that maximizes `tokens/s/GPU` subject
to a configurable per-GPU memory safety margin.

The contribution is a configuration-search methodology that exploits an LLM's
ability to reason over architectural priors (topk dominance of MoE activations,
the impact of MQA on attention activation, the inter-node/intra-node boundary
for All-to-All) to direct an analytical oracle — replacing exhaustive sweeps on
real hardware with a small number of informed analytical evaluations. The
system runs either entirely on a CPU-only host (simulate backend only) or in a
mixed mode where a small number of real-hardware benchmark runs calibrate the
analytical predictions.

### Scope

1. A standalone agent, distributed alongside Primus, that drives the Projection
   tool in a closed loop.
2. A **target-cluster YAML** wrapping the existing
   `examples/hardware_configs/*.yaml` convention, so no new networking format
   has to be invented.
3. A **single objective**: maximize `tokens/s/GPU` subject to
   `projected_memory ≤ HBM_capacity × (1 − safety_margin)`.
4. Agent search over **all** parallelism and coupled axes (TP, PP, EP, CP, MBS,
   GBS, VPP, pipeline schedule, recompute, overlap flags) plus the higher-impact
   levers (FP8, MoE DeepEP / sync-free, fused cross-entropy,
   distributed-optimizer / FSDP2) — restricted by per-architecture legality.
   See [Knobs Searched](#knobs-searched) for the complete list.
5. Two evaluator paths: a **no-GPU path** (`projection memory` + `projection
   performance --profiling-mode simulate`), always available; and a
   **with-GPU path** that adds benchmark runs for promising candidates and
   periodic calibration.
6. LLM driven via **DSPy/LiteLLM** (no separate proxy process); provider,
   model, and budget are user-configurable.
7. Output: the best legal configuration (YAML overlay + `PRIMUS_*` env
   overrides), the projected metrics, and a trial log.

### Deferred future features

These are recorded so they are not lost; they are deliberately left out to keep
the agent small.

- **F1. Multi-objective Pareto** — replace the scalar `tokens/s/GPU` objective
  with a Pareto frontier over (throughput, MFU, memory headroom, projected
  $/token, projected energy/token).
- **F2. Online cluster-spec retrieval** — pull the cluster description from an
  internal registry or known-archs catalog (MI300X / MI325X / MI355X reference
  pods) and fall back to user overrides; optionally infer topology from a small
  DCGM / ROCm-SMI dump.
- **F3. Persistent memory / configuration cache** — cache
  `(model_signature, cluster_signature) → best_known_configs` across runs;
  invalidate when ROCm / hipBLASLt / framework versions change.
- **F4. Agent-proposed scale-downs and microbenchmarks** — let the agent
  *propose* reduced-model proxies and targeted microbenchmarks to reduce the
  uncertainty of its current top-k (reusing the `moe_proxy_single_node.yaml`
  pattern).
- **F5. Telemetry plug-ins (rocprofiler / TraceLens / Magpie)** — after a
  benchmark run, optionally extract per-kernel time, GEMM efficiency, A2A bytes,
  NIC utilization to calibrate the analytical models and explain
  underperformance back to the agent. Exposed as a `SKILL.md`-described plug-in.
  References: [TraceLens](https://github.com/AMD-AGI/TraceLens-internal),
  [Magpie](https://github.com/AMD-AGI/Magpie).
- **F6. Calibration learning** — under `--profiling-mode both`, record
  per-(model, arch, dim) residuals between simulate and benchmark, fit a small
  correction model, and report a confidence band on subsequent simulate runs.
- **F7. Robustness / sensitivity report** — for the winning config, sweep ±1
  step on each axis and report whether the optimum is a sharp peak or a broad
  basin (cheap; only `simulate` calls).
- **F8. Cross-axis priors as DSPy modules** — a library of tunable "rules of
  thumb" that DSPy's optimizer can refine over time using the trial logs.
- **F9. Sub-agent "test-proposer"** — delegate targeted experiments (e.g. a
  2-layer scale-down forward+backward microbenchmark, or a stand-alone A2A probe
  at the proposed EP × hidden_size × topk), profiling the *test* with explicit
  synchronisation rather than the sandbox. A `run_proposed_experiment(plan,
  code)` tool can be added to `tools.py` without restructuring the loop.
- **F10. Sub-LLM expert router** — extend the existing `query_llm` tool into a
  *named expert* router (`query_llm(expert='moe', …)`).

### Known design holes

1. **Simulator-vs-reality gap** — see Limitations above; in no-GPU mode the
   agent reports a confidence caveat.
2. **Memory-projection blind spots** — A2A buffers, allocator fragmentation,
   and comm scratch are not modeled analytically; the benchmark-based memory
   projection closes most of this gap by anchoring on a measured peak.
3. **Search-space explosion** — mitigated by tiering.
4. **Cluster-description lossiness** — averaged bandwidth/latency cannot capture
   contention or per-rail asymmetry.
5. **Schedule × VPP legality** must be a coded table, not delegated to the LLM.

---

## Related Documentation

- [Projection](./projection.md) — memory + performance projection internals,
  including the benchmark-based memory projection the agent relies on.
- [Tuning Agent package README](../primus/agents/tuning_agent/README.md) —
  quickstart reference inside the source tree.
