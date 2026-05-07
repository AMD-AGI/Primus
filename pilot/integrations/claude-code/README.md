# integrations/claude-code/ — Claude Code adapter

**Status**: Stub

Thin adapter for running Pilot inside Claude Code. The Orchestrator role is carried by Claude Code's main session; Stage Workers are carried by Task-tool subagents.

## Files (TBD)

- `CLAUDE.md` — entry prompt; references `pilot/AGENTS.md`, `pilot/prompts/orchestrator.md`, `pilot/skills/workflow/orchestration.md`
- `mcp-server.py` — optional MCP wrapper around `pilot/tools/*` (alternative to direct shell invocation)
- `task-spawner.py` — adapter for `pilot/tools/subagent.py`'s `register_backend()`; concrete spawn impl uses Claude Code's Task tool

## Setup

```bash
# Quick start: register CLAUDE.md and let Claude Code consume Pilot directly
ln -s $(pwd)/CLAUDE.md ~/.claude/projects/<your-project>/CLAUDE.md
```

Before launching a session, the user must prepare a `cluster.yaml` (see
`pilot/SETUP.md`); every cluster-touching Pilot tool consumes it via
`--cluster-config <path>` (fallback: `$PRIMUS_PILOT_CLUSTER_CONFIG`, then
`./cluster.yaml`). When the planned `mcp-server.py` wrapper is implemented, it
must forward this argument verbatim — Claude Code's Task subagents must NOT
reinvent the contract.

## Constraints

- This adapter is the ONLY place anthropic / claude-code SDK imports may appear.
- See `pilot/AGENTS.md` §3.1 for the SDK isolation rule.
- Tool invocation must follow the same `cluster.yaml` contract as the Cursor
  adapter (see `pilot/integrations/cursor/rules/90-tool-invocation.mdc`).
