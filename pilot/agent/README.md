# Primus Pilot — Agent Skeleton (Optional Python Runtime)

**Version**: v2.0 skeleton (aligned with `pilot/README.md`)

> ⚠️ **This is an optional reference implementation, not Pilot's primary integration path.**
>
> Per the **Scope & Positioning** section of `pilot/README.md`, Pilot itself is a
> **knowledge + tool package** for training-tuning (skills / prompts / tools /
> schemas / state); the runtime should be delegated to a concrete agent framework
> (Claude Code / Cursor / Codex / ...).
>
> **Recommended production path**: have your agent framework consume
> `pilot/skills` + `pilot/prompts` + `pilot/tools` directly (thin adapters will
> live under `pilot/integrations/<framework>/`).
>
> **When this directory is appropriate**:
> - You want an **unattended, headless long-running process** for long tuning
>   sessions (cron / CI).
> - You want **strict context-budget enforcement** and don't trust a given
>   framework's context management.
> - You're running on an LLM **without native subagent support** (e.g. raw API
>   only).
> - You want a reference implementation showing "how to roll your own harness".
>
> If you're using Claude Code / Cursor, **you do not need this directory** —
> their Task / subagent mechanisms already realize Strategy B and C from
> §13 of `pilot/README.md`.

---

## What this is

This is a **Python reference implementation** of Pilot's Agent layer, manually
realizing the two-tier agent architecture from v2.0:

- **Orchestrator Agent** (long-lived, thin context): `orchestrator.py`
- **Stage Worker** (one-shot, isolated context): `subagent.py`

Skeleton only: business tools (`submit.run` / `observe.snapshot` /
`constraint.check`, etc.) are stubbed with Anthropic tool schemas + placeholder
handlers. To wire up a real Primus / Slurm backend, implement the matching
handlers in `worker_tools.py`.

## Key design correspondence

| Skeleton file | Maps to README.md |
|---------------|-------------------|
| `orchestrator.py` | §2.2 Orchestrator, §13 Context Management |
| `subagent.py` | §2.2 Stage Worker, §13.2 Strategy B |
| `state.py::trim()` | §5 `state.trim()`, §12.1 Context hygiene, §13.2 Strategy A |
| `state.py::handoff()` | §5 `state.handoff()`, §13.2 Strategy C |
| `orchestrator_tools.py` | §5 Orchestrator-only tool set |
| `worker_tools.py` | §5 business tools (invisible to Orchestrator) |
| `schemas.py::SubagentResult` | §8.11 SubagentResult |
| `schemas.py::OrchestratorState` | §8.7 TuningState (trimmed view) |

## Run

```bash
# From the repo root
pip install -r pilot/agent/requirements.txt
export ANTHROPIC_API_KEY=sk-...

# Live mode
python -m pilot.agent --session demo_001 \
    --skills-dir pilot/skills \
    --state-dir pilot/state \
    --gpu-h 10 --max-rounds 5

# Dry-run (no API key needed; subagents return mock SubagentResult)
python -m pilot.agent --session demo_001 --dry-run
```

## Done vs TODO

**Done (protocol skeleton)**:
- Orchestrator main loop `resume → decide → spawn → apply → checkpoint → trim`
- `state.trim()` enforces pointer-only fields between iterations
- `subagent.spawn()` achieves context isolation via an independent Claude session
- `SubagentResult` schema validation (summary token cap)
- `state.handoff()` for handing off under context pressure
- Physical isolation between Orchestrator tool set and Worker tool set
  (Orchestrator cannot see business tools)

**TODO (business integration)**:
- `worker_tools.py`: real handlers per business tool (currently schema only)
- `subagent.py::_handle_tool_use`: business-tool dispatch
- `state.py::apply_result`: complete per-stage artifact field mapping
- `skills/workflow/orchestration.md` and `state_machine.md`: actual knowledge
  content (placeholders today)
- Token counting: currently estimated via `resp.usage`; production should wire
  in tiktoken / Anthropic's token counter.

## Suggested reading order

1. `orchestrator.py::Orchestrator.run()` — main-loop skeleton
2. `orchestrator.py::_decide()` — how each step keeps the Orchestrator thin
3. `orchestrator_tools.py` — Orchestrator has only 5 tools
4. `subagent.py::StageWorker.run()` — how a Worker runs in isolation
5. `state.py::StateStore.trim()` — context hygiene rules
