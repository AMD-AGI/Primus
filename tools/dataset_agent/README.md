# Dataset agent

Plug one real dataset into one Primus backend and model. One command runs the campaign on the host, outside any container and without a GPU. The gate reads the dataset id from the example YAML, streams a few real Hub rows, and passes them through the backend field parser. Documentation runs only after that check and the CPU pytest pass. The orchestrator is a Python phase machine. It does not use LangGraph.

## Run

From the Primus repo root:

```bash
pip install claude-agent-sdk pyyaml pytest datasets
export ANTHROPIC_API_KEY=your-api-key

# Optional. The CLI appends /v1/messages, so do not add a trailing /v1.
# export ANTHROPIC_BASE_URL=https://llm-api.amd.com/anthropic
# export ANTHROPIC_CUSTOM_HEADERS=$'Ocp-Apim-Subscription-Key: your-subscription-key'
# export DATASET_AGENT_MODEL=claude-sonnet-4-5
# export HF_TOKEN=hf_...          # only for private or gated datasets
# The campaign moves HF_HOME itself when ~/.cache/huggingface/token is a
# directory instead of a token file. Set HF_HOME only if you want a different cache.

python -m tools.dataset_agent \
  --backend megatron \
  --model llama3.1_8B \
  --dataset tau/scrolls \
  --stage posttrain
```

`--stage` is `pretrain` or `posttrain`. The integrator uses an existing format when that format can read the dataset. It adds a new format only when none of the current ones can, and the example YAML uses that name.

That is the whole campaign. Run it on the host with the dataset-agent virtualenv. It does not enter a container and it does not use a GPU. `--dry-run` only resolves the preset, the example path, and the Hub record.

When the gate passes, the example config is:

```text
examples/<backend>/configs/dataset_agent/<slug>.yaml
```

Logs, including the Hub JSON and the pytest output, are under `output/dataset_agent/<slug>/` when that directory is writable, and otherwise under `~/.cache/primus-dataset-agent/<slug>/`. The tool does not commit. Beside the example YAML it writes a `.patch` of the `primus/` and `docs/` edits plus that YAML, and a `.report.md` with the gate result, the config to try, and the command that reruns the CPU test.

## Flags

| Flag | Meaning |
|---|---|
| `--backend` | Primus backend name |
| `--model` | Model preset name, with or without `.yaml` |
| `--dataset` | Hugging Face dataset id |
| `--stage` | `pretrain` or `posttrain` |
| `--split` | Split streamed by the test. Default `train` |
| `--samples` | Records the test must read. Default `4` |
| `--max-retries` | Integrator resumes after a red gate. Default `3` |
| `--llm-model` | Claude model id. Overrides `DATASET_AGENT_MODEL` |
| `--no-branch` | Stay on the current git branch |
| `--dry-run` | Resolve files and the Hub record, then stop |

The probe uses the `claude` binary shipped in `claude_agent_sdk/_bundled/`. A `claude` on `PATH` is used instead when one is installed.
