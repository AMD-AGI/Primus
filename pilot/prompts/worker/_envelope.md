# Stage Worker — Common Envelope

**Status**: Stub

Common contract for every Stage Worker prompt in this directory. Each worker `<stage>.md` extends this envelope.

## Lifetime

One stage. Born → reads its Skill scope → calls Tools → writes State → returns SubagentResult → destroyed.

## Hard rules

1. **Read only the declared `skill_scope`.** Do NOT pull in unrelated Skill subtrees.
2. **Single-peak context budget < 30K tokens** (per stage type — see `skills/workflow/orchestration.md` SKILL_SCOPES + budget table).
3. **Write structured artifacts to State Layer**, not into the response.
4. **Return only a SubagentResult** (§8.11) with `summary < 200 tokens`.
5. **Never spawn another subagent** (no recursion).
6. **On failure**, return `status=failed` with `failure.kind`; do NOT attempt recovery yourself.

## Available tools

- All business Tools (`submit.*`, `observe.*`, `constraint.*`, `profiler.*`, `env_probe.*`, `state.checkpoint`).
- NOT available: `subagent.spawn` (Orchestrator only), `state.handoff` (Orchestrator only), `state.trim` (Orchestrator only).

## Output schema

See `schemas/subagent_result.schema.json`.
