# Tuning Agent

LLM-driven iterative search for an optimal Primus training configuration on a
target cluster, using the **Primus Projection** tool as an oracle. Projection
provides two estimates — **memory** and **performance** — each of which runs
**benchmark-anchored by default** (measure what fits on a sub-node run, scale
the rest analytically) with a fully analytical **no-GPU `simulate`** fallback.

See [`docs/02-user-guide/tuning-agent.md`](../../../docs/02-user-guide/tuning-agent.md) — the full
user/operator guide (modes, configuration, troubleshooting, worked example)
plus the design write-up, paper-ready problem statement, and the list of
deferred future features (cluster-spec retrieval, persistent memory cache,
agent-proposed scale-down models, telemetry plug-ins, calibration learning,
sub-agent test-proposer, …).

## Install

```bash
python3 -m venv ~/code/Primus/.venv-agent
source ~/code/Primus/.venv-agent/bin/activate
pip install -r primus/agents/tuning_agent/requirements.txt
# Optional: only needed for --profiling-mode simulate
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

## LLM Setup

The agent uses [DSPy](https://dspy.ai), which routes LLM calls through
[LiteLLM](https://docs.litellm.ai/docs/providers) internally — **no separate
proxy process is required**.

Set credentials for whichever provider you use:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export LLM_MODEL=openai/gpt-4o           # default if unset

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=anthropic/claude-opus-4-5

# Any OpenAI-compatible endpoint (Ollama, vLLM, local LiteLLM proxy, …)
export OPENAI_API_KEY=dummy              # or the real key if required
export OPENAI_API_BASE=http://localhost:11434/v1
export LLM_MODEL=openai/llama3
```

The model string follows LiteLLM's provider-prefixed naming convention
(`<provider>/<model-name>`).  See the
[LiteLLM providers list](https://docs.litellm.ai/docs/providers) for all
supported backends.

Alternatively, put these in a `.env` file; the agent searches:

```
$CWD/.env
<repo-root>/.env
~/.env
```

You can also set them directly in your target-cluster YAML under `agent.llm`:

```yaml
agent:
  llm:
    model: anthropic/claude-opus-4-5
    api_key: sk-ant-...         # optional if set in env
    base_url: ...               # optional; override for custom endpoints
```

## Quickstart

```bash
# Dry-run (no primus-cli, no LLM): just exercises the loop end-to-end with
# synthesised metrics. Useful to verify the install on a CPU-only host.
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/megatron/guides/tuning_agent/target_cluster_mi355x_4nodes.yaml \
    --out-dir tuning_runs/dry-run \
    --dry-run --seed-only

# Seed-only with the real projection tool (requires Origami if simulate path):
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/megatron/guides/tuning_agent/target_cluster_mi355x_4nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-mi355x-seed \
    --seed-only

# Full agent (planner + DSPy.RLM rounds):
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/megatron/guides/tuning_agent/target_cluster_mi355x_4nodes.yaml \
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
    `get_op_coverage`, `get_cluster`, `get_budget_status`
  - `note_to_scratchpad`, `read_scratchpad`
  - `query_llm(prompt, system?)` — extra "LLM-inside-LLM" tool

## Op bring-up (new ops)

A new model often ships an op the projection does not model: a linear-attention
layer, a gate, a short convolution. The projection prices such an op as the
built-in block it resembles and prints a `PRIMUS_OP_GAP` line for it.

Adding those ops is the job of the **projection agent**
(`primus/agents/projection_agent/`), not of the tuning agent. The tuning agent
calls it before seeding, for the gaps the workload YAML reveals, and again for
any new gap a trial prints. The same agent serves a plain projection run, so a
workload gets its ops whether or not anyone is tuning it:

```bash
# add the missing ops, then project
primus projection performance --config <yaml> --profiling-mode simulate --op-bringup auto \
    --op-coder-cmd 'cursor-agent -p "$(cat {prompt_file})" --output-format text'

# add them only
python -m primus.agents.projection_agent --workload <yaml> --op-llm-model anthropic/claude-opus-4-5
```

For each gap the projection agent runs op bring-up (`projection_agent/bringup.py`):

1. **Context.** It writes `op_bringup/<op_id>/context.md`, which contains:
   - the gap's keys and the model YAML chain;
   - the HF `config.json`, fetched from `tokenizer_model` or `--hf-config`;
   - any `--op-paper` text;
   - the op-model API and the existing op models and profilers closest to the gap.
2. **Create.** A coder writes one op model:
   - `--op-coder-cmd` runs an external coding agent;
   - otherwise an LLM writes it: `--op-llm-model` / `$LLM_MODEL`, or, under the tuning agent, the search's own LLM.

   Attention-like ops are priced on the FAv3 or 1-CU tile simulator; everything else analytically.
3. **Validate.** The validator (`primus.core.projection.op_models.validate`) checks:
   - the roofline floor;
   - FLOPs against the paper's closed form;
   - the bwd/fwd ratio and monotonicity;
   - that every gap key is covered.

   Failures go back to the coder, up to 3 attempts.
4. **Register.** The model is copied into `primus/core/projection/op_models/generated/` with a regression test, and the trial is re-run. An op the coder judges negligible is acknowledged in `generated/acknowledged.yaml` instead.

The same `--op-*` flags work on the tuning agent, on `primus projection
{memory,performance,both}` and on `python -m primus.agents.projection_agent`:

| flag | meaning |
|---|---|
| `--op-bringup auto\|report\|off` | `auto` writes models. It falls back to `report` (context packs only) when no coder is available. Default `auto` for the tuning agent and the projection agent, `off` for `primus projection`. |
| `--op-coder auto\|dspy\|command\|none` | Who writes the model. |
| `--op-coder-cmd '...{prompt_file}...{target_file}...'` | Shell template for an external coding agent. |
| `--op-llm-model MODEL` | LiteLLM model string for the LLM coder, when it is not the tuning agent's. |
| `--op-paper FILE\|URL\|TEXT` | Paper text for the new op. Repeatable. |
| `--hf-config PATH\|URL\|REPO` | HF config, if the YAML has no `tokenizer_model`. |
| `--op-out-dir DIR` | Where context packs and reports go (the tuning agent uses its `--out-dir`). |
| `--op-pr off\|commit\|push\|pr` | Publish each registered or acknowledged op on its own branch off HEAD (`projection-agent/<op_id>-<time>`), built in a temporary worktree so the checkout is untouched. `commit` keeps it local, `push` pushes it, `pr` also opens a draft PR. Default `off`. |
| `--op-pr-remote`, `--op-pr-base`, `--op-pr-repo` | Remote to push to (`origin`); branch the PR targets (the remote's default); `owner/name` to open the PR on when pushing to a fork. |
| `--op-pr-author 'Name <email>'` | Commit identity when git has none configured (`$PRIMUS_OP_PR_AUTHOR`). |
| `--op-pr-signoff`, `--op-pr-ready` | Add `Signed-off-by`; open the PR ready for review instead of draft. |

With `--op-pr pr` the PR is opened with `gh` when it is installed, else through
the GitHub REST API with `$GITHUB_TOKEN` / `$GH_TOKEN`; with neither, the agent
prints the compare URL to open it from. The PR body is the bring-up report: the
gap, the validation checks and cases, and the module's references.

The `projection-op-bringup` skill covers doing the same work by hand.

## Two-stage flow

1. **Systematic seed** (no LLM): a coarse legal grid over (TP, PP, EP, CP)
   plus the workload baseline is evaluated and put into `trials.jsonl`.
2. **DSPy planner** (`ChainOfThought`) reads architecture + cluster + axis
   legality + seed history and emits a plan.
3. **DSPy.RLM** rounds run the search loop with the tool belt. Each round
   sees the *full* compressed history and the scratchpad. We early-stop
   when no improvement happens within budget.
