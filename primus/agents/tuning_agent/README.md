# Tuning Agent

LLM-driven iterative search for an optimal Primus parallelization
configuration on a target cluster, using the **Primus Projection** tool
(memory + simulate + optional benchmark) as an oracle.

See:
- [`docs/agents/tuning_agent.md`](../../../docs/agents/tuning_agent.md) — full
  user/operator guide (modes, configuration, troubleshooting, worked
  example).
- [`docs/projection_optimizer.md`](../../../docs/projection_optimizer.md) —
  design write-up, paper-ready problem statement, and the list of deferred
  future features (cluster-spec retrieval, persistent memory cache,
  agent-proposed scale-down models, telemetry plug-ins, calibration learning,
  sub-agent test-proposer, …).

## Install

```bash
python3 -m venv ~/code/Primus/.venv-agent
source ~/code/Primus/.venv-agent/bin/activate
pip install -r primus/agents/tuning_agent/requirements.txt
# Optional, only if you want --profiling-mode simulate to run:
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

The agent picks up LLM credentials from a `.env` file. It searches:

```
$CWD/.env
<repo-root>/.env
~/code/.env       <-- shared with iterative_fix
~/.env
```

Required keys:

```bash
LITELLM_BASE_URL=https://llm-api.amd.com
OCP_APIM_SUBSCRIPTION_KEY=          # or use LITELLM_API_KEY for the same key
LITELLM_MODEL=GPT-oss-20B           # optional; default in code
```

## Quickstart

```bash
# Dry-run (no primus-cli, no LLM): just exercises the loop end-to-end with
# synthesised metrics. Useful to verify the install on a CPU-only host.
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_2nodes.yaml \
    --out-dir tuning_runs/dry-run \
    --dry-run --seed-only

# Seed-only with the real projection tool (requires Origami if simulate path):
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_2nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-mi355x-seed \
    --seed-only

# Full agent (planner + DSPy.RLM rounds):
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_2nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-mi355x-full
```

## Output artefacts

In `--out-dir`:

```
trials.jsonl       all attempted configs and their results
trials.png         incumbent vs. trial number plot
scratchpad.txt     durable LLM notes carried across rounds
summary.json       agent-summarised winner
trials/*.yaml      one workload-overlay YAML per trial (re-runnable)
```

## What the LLM gets

- The flattened model architecture record (a small JSON blob).
- The target cluster spec.
- The per-axis legal value sets (computed in code from divisibility +
  cluster size, so the LLM can't propose an obviously illegal config and
  spend budget being told no).
- A compact text summary of every prior trial (config + result + reason if
  rejected).
- A durable scratchpad it can write notes to, carried across rounds.
- A tool belt:
  - `evaluate_simulate(config_json)` — primary
  - `evaluate_memory_only(config_json)` — cheap pre-filter
  - `evaluate_with_benchmark(config_json)` — only if `has_gpu=true`
  - `get_history`, `get_best`, `get_legal_axes`, `get_architecture`,
    `get_cluster`, `get_budget_status`
  - `note_to_scratchpad`, `read_scratchpad`
  - `query_llm(prompt, system?)` — extra "LLM-inside-LLM" tool

## Two-stage flow

1. **Systematic seed** (no LLM): a coarse legal grid over (TP, PP, EP, CP)
   plus the workload baseline is evaluated and put into `trials.jsonl`.
2. **DSPy planner** (`ChainOfThought`) reads architecture + cluster + axis
   legality + seed history and emits a plan.
3. **DSPy.RLM** rounds run the search loop with the tool belt. Each round
   sees the *full* compressed history and the scratchpad. We early-stop
   when no improvement happens within budget.
