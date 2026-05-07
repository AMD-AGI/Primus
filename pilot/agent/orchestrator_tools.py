"""The 5 tools available to the Orchestrator Agent.

Design invariant (§2.2, §13):
  The Orchestrator CANNOT call business tools (submit / observe / constraint /
  profiler / ...). Those belong to Stage Workers only. Physically separating the
  tool sets is the primary enforcement mechanism.

  Orchestrator tools = state.* + subagent.spawn + orchestrator.stop
  Worker tools       = business tools + return_subagent_result
"""
from __future__ import annotations

from .schemas import Stage


SUBAGENT_SPAWN = {
    "name": "subagent_spawn",
    "description": (
        "Spawn a one-shot Stage Worker subagent to execute a single stage of the "
        "tuning workflow. The subagent runs in its own isolated Claude session, "
        "reads only its scoped Skills + State slice, and returns a SubagentResult "
        "whose `summary` is capped at ~200 tokens. Heavy artifacts "
        "(CandidatePool / Snapshot / DiagnosisReport / ...) are written to the "
        "State Layer by the worker; only their refs come back.\n\n"
        "Use this for: PREFLIGHT, PROJECTION, SMOKE, CORRECTNESS, OBSERVE, "
        "DIAGNOSE, REPLAN, ENV_SWEEP, CORRECTNESS_LITE, LEARN.\n\n"
        "Do NOT use this for: EXECUTE (handled inline by orchestrator), "
        "SETTLE (pure numeric rule, inline), BASELINE launch (inline)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "stage": {
                "type": "string",
                "enum": [s.value for s in Stage],
                "description": "The stage to execute. Must match state_machine.md.",
            },
            "input_refs": {
                "type": "object",
                "description": (
                    "Refs to State Layer artifacts the Worker needs, e.g. "
                    "{'snapshot_id': 'job_8842', 'plan_graph_ref': 'state/.../plan_graph.yaml'}. "
                    "Do NOT include raw content here."
                ),
            },
            "skill_scope": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Glob patterns (relative to skills/) of the Skill files the "
                    "Worker is permitted to read. Example for DIAGNOSE: "
                    "['workflow/diagnose.md', 'execution-model/*', 'profiling/*']"
                ),
            },
            "max_tokens": {
                "type": "integer",
                "default": 30000,
                "description": "Hard cap on the Worker's total LLM token usage.",
            },
            "reason": {
                "type": "string",
                "description": "One-line why you are dispatching this stage now (for audit).",
            },
        },
        "required": ["stage", "input_refs", "skill_scope", "reason"],
    },
}


STATE_CHECKPOINT = {
    "name": "state_checkpoint",
    "description": (
        "Persist the current OrchestratorState to disk. The outer loop already "
        "checkpoints after every tool call; use this tool only if you want an "
        "extra snapshot mid-turn (rare). Idempotent."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


STATE_TRIM = {
    "name": "state_trim",
    "description": (
        "Drop all heavy in-context fields, keep only pointer-class fields. "
        "Called automatically by the outer loop after every stage transition; "
        "invoke manually only if you just absorbed an unusually large summary."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


STATE_HANDOFF = {
    "name": "state_handoff",
    "description": (
        "Write a handoff checkpoint and terminate THIS Orchestrator session. "
        "The caller (shell / scheduler) is expected to spawn a fresh Orchestrator "
        "which will state.resume() from the handoff file. Use when you detect "
        "context pressure (e.g. budget_used.total_tokens > 0.5 * window) or "
        "after every K rounds for long runs. Does NOT consume round budget."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "e.g. 'context_pressure' / 'periodic_k10' / 'manual'",
            },
            "next_action_hint": {
                "type": "string",
                "description": "One-line hint for the next Orchestrator, e.g. 'about to spawn REPLAN with bottleneck=COMPUTE'",
            },
        },
        "required": ["reason"],
    },
}


ORCHESTRATOR_STOP = {
    "name": "orchestrator_stop",
    "description": (
        "Terminate the session. Use when TargetVector converged, budget "
        "exhausted, or unrecoverable error. Writes a final REPORT artifact."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": ["CONVERGED", "BUDGET_EXCEEDED", "ABORT", "STAGNATION"],
            },
            "final_summary": {
                "type": "string",
                "description": "One-paragraph final summary (goes into REPORT).",
            },
        },
        "required": ["reason", "final_summary"],
    },
}


ORCHESTRATOR_TOOLS = [
    SUBAGENT_SPAWN,
    STATE_CHECKPOINT,
    STATE_TRIM,
    STATE_HANDOFF,
    ORCHESTRATOR_STOP,
]


def tool_names() -> list[str]:
    return [t["name"] for t in ORCHESTRATOR_TOOLS]
