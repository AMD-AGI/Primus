# integrations/codex/ — OpenAI Codex adapter

**Status**: Stub

Thin adapter for running Pilot inside OpenAI Codex. Mirror of `claude-code/` but for the Codex runtime.

## Files (TBD)

- `CODEX.md` — entry prompt; references `pilot/AGENTS.md`, `pilot/prompts/orchestrator.md`, `pilot/skills/workflow/orchestration.md`
- `subagent-backend.py` — concrete spawn impl registered into `pilot/tools/subagent.register_backend('codex', ...)`

## Setup (mandatory)

Before launching a session the user must prepare a `cluster.yaml` (see
`pilot/SETUP.md`); every cluster-touching Pilot tool reads it via
`--cluster-config <path>` (fallback: `$PRIMUS_PILOT_CLUSTER_CONFIG`, then
`./cluster.yaml`). When `subagent-backend.py` is implemented it must forward
the argument unchanged — the Codex Subagent must not redefine the contract.

## Constraints

- This adapter is the ONLY place openai / codex SDK imports may appear.
- See `pilot/AGENTS.md` §3.1 for the SDK isolation rule.
- Tool invocation must follow the same `cluster.yaml` contract as the Cursor
  adapter (see `pilot/integrations/cursor/rules/90-tool-invocation.mdc`).
