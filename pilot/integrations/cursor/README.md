# Pilot × Cursor Integration

A thin adapter layer that lights up Pilot inside Cursor Desktop / CLI. It does
not modify Pilot's main tree; it only ships entry-point files Cursor can
consume natively.

## Directory contents

| File | Purpose | How Cursor consumes it |
|------|---------|------------------------|
| `AGENTS.md` | Project-level agent context: what Pilot is, how the Orchestrator runs, where to find knowledge | Cursor automatically loads `AGENTS.md` from the repo root or `pilot/` as the resident context for the main agent |
| `rules/*.mdc` | Sharded Cursor Rules: role prompts / tool-invocation conventions / state hygiene | Copy them into the user's `.cursor/rules/` and they take effect automatically |
| `mcp.json.example` | MCP-server config template (exposes `pilot/tools/*` as MCP tools) | Use as a reference, then place at `.cursor/mcp.json` or `~/.cursor/mcp.json` |

## Install steps

1. **Place AGENTS.md** (pick one):
   - **Repo-level**: copy `pilot/integrations/cursor/AGENTS.md` to `AGENTS.md`
     at the repo root (Cursor loads it globally).
   - **Directory-level**: keep `pilot/AGENTS.md` in place; Cursor loads it
     automatically when working under that directory.
2. **Install rules**:
   ```bash
   mkdir -p .cursor/rules
   cp pilot/integrations/cursor/rules/*.mdc .cursor/rules/
   ```
3. **(Optional) Configure MCP**:
   ```bash
   cp pilot/integrations/cursor/mcp.json.example .cursor/mcp.json
   ```
   Or just use shell to call `python -m pilot.tools.<name>` — MCP is not
   required.
4. **Prepare `cluster.yaml`** (mandatory before any tuning session): pick
   `single` or `slurm` mode and write it as documented in `pilot/SETUP.md`.
   All Pilot tools that touch the cluster require this file via
   `--cluster-config <path>` (or `$PRIMUS_PILOT_CLUSTER_CONFIG` /
   `./cluster.yaml` fallback). Without it tools exit with
   `failure.kind=CLUSTER`.
5. **Verify**: in the Cursor Agent panel, type
   `Start a tuning session for <model> on <cluster>`. The agent should pick up
   the Orchestrator role, ask for `cluster.yaml` if not found, and begin
   PREFLIGHT.

## How v2 roles map onto Cursor

| v2 role | Cursor side |
|---------|-------------|
| **Orchestrator** (main session) | Cursor's main Agent session; picks up its role from `AGENTS.md` + `rules/10-orchestrator-role.mdc` |
| **Stage Worker** (one-shot subagent) | Cursor's Task tool spawns a child agent; prompt comes from `rules/30-worker-*.mdc` (Agent Requested mode) |
| **State Layer** | Direct read/write of `pilot/state/*.yaml` |
| **Tool calls** | Shell to `python -m pilot.tools.*` or MCP (when enabled) |
| **Context hygiene** | `rules/00-pilot-core.mdc` + `rules/20-state-hygiene.mdc` mandate per-stage checkpoint + trim |
| **Handoff** | Cursor has no resident process; relies on `state/checkpoints/handoff/` + `state.resume()`, the next Cursor session continues from the saved point |

## Degradation notes

- Cursor's Task tool realizes Strategy B (subagent isolation). If a Cursor
  release lacks Task, the Worker degrades to a "sub-conversation segment"
  inside the same session; the scope rules from `rules/30-worker-*.mdc` still
  apply, but context isolation is weaker.
- Because Cursor has no resident process, Strategy C (session handoff)
  manifests as "the next time you open Cursor", not an automatic restart.

## Relationship to other integrations

`pilot/integrations/claude-code/` and `pilot/integrations/codex/` follow the
same pattern, each realizing the corresponding framework's native abilities.
The three core trees — Skills / Tools / Schemas — are fully shared, so adding
a new framework adapter never forks the main tree.
