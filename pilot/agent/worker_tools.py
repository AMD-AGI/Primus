"""Business tools scoped per stage.

These are what Stage Workers may call. The Orchestrator NEVER sees these.
All handlers are stubs — wire them to your real Primus / Slurm / WandB
infrastructure as a follow-up.

Schemas mirror §5 of README.md.
"""
from __future__ import annotations

from typing import Any

from .schemas import Stage


# ---------------------------------------------------------------------------
# The return tool — every Worker must call this exactly once
# ---------------------------------------------------------------------------
RETURN_SUBAGENT_RESULT = {
    "name": "return_subagent_result",
    "description": (
        "Return the final SubagentResult to the Orchestrator. You MUST call "
        "this exactly once at the end. The `summary` field is hard-capped at "
        "~1200 chars (~200 tokens); put heavy detail in artifacts written "
        "via write_artifact and reference them here."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "failed", "escalate"]},
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "ref": {"type": "string"},
                        "size_bytes": {"type": "integer"},
                    },
                    "required": ["kind", "ref"],
                },
                "default": [],
            },
            "summary": {
                "type": "object",
                "description": "Keep under ~200 tokens. headline + key_metrics style.",
            },
            "suggested_transition": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
            "failure": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "message": {"type": "string"},
                    "escalate_to_orchestrator": {"type": "boolean"},
                },
            },
        },
        "required": ["status", "summary"],
    },
}


WRITE_ARTIFACT = {
    "name": "write_artifact",
    "description": (
        "Persist a heavy artifact (PlanGraph / CandidatePool / Snapshot / "
        "DiagnosisReport / EnvSweepResult / ...) to the State Layer. Returns "
        "a ref string you can put in SubagentResult.artifacts[]."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "name": {"type": "string"},
            "content": {
                "type": "string",
                "description": "YAML or JSON text. This is the only place heavy data should live.",
            },
        },
        "required": ["kind", "name", "content"],
    },
}


# ---------------------------------------------------------------------------
# Business tool stubs (map 1:1 to §5)
# ---------------------------------------------------------------------------
PREFLIGHT_RUN = {
    "name": "preflight_run",
    "description": "Collect cluster hardware baseline. Returns ClusterProfile YAML.",
    "input_schema": {
        "type": "object",
        "properties": {"cluster_id": {"type": "string"}},
        "required": ["cluster_id"],
    },
}

ENV_PROBE_RUN = {
    "name": "env_probe_run",
    "description": "Probe / validate cluster env baseline (connectivity + RCCL micro-bench).",
    "input_schema": {
        "type": "object",
        "properties": {
            "cluster_id": {"type": "string"},
            "candidate_envs": {"type": "object"},
        },
        "required": ["cluster_id"],
    },
}

ENV_PROBE_SWEEP = {
    "name": "env_probe_sweep",
    "description": "Inner-loop env sweep: fix structure, scan env diff.",
    "input_schema": {
        "type": "object",
        "properties": {
            "base_plan_ref": {"type": "string"},
            "candidate_envs": {"type": "object"},
            "max_steps": {"type": "integer", "default": 50},
        },
        "required": ["base_plan_ref", "candidate_envs"],
    },
}

PROFILER_RUN = {
    "name": "profiler_run",
    "description": "Single-node profiling. Returns ProfilingResult ref.",
    "input_schema": {
        "type": "object",
        "properties": {
            "model_spec": {"type": "object"},
            "configs": {"type": "object"},
        },
        "required": ["model_spec", "configs"],
    },
}

SUBMIT_RUN = {
    "name": "submit_run",
    "description": "Submit a training job. Returns job_id.",
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_ref": {"type": "string"},
            "cluster_id": {"type": "string"},
            "max_steps": {"type": "integer"},
        },
        "required": ["plan_ref", "cluster_id"],
    },
}

OBSERVE_SNAPSHOT = {
    "name": "observe_snapshot",
    "description": "Collect runtime metrics. Returns Snapshot ref.",
    "input_schema": {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    },
}

OBSERVE_COMPARE_LOSS = {
    "name": "observe_compare_loss",
    "description": "CORRECTNESS gate: compare loss curve with reference.",
    "input_schema": {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "reference_curve_ref": {"type": "string"},
        },
        "required": ["job_id", "reference_curve_ref"],
    },
}

CONSTRAINT_CHECK = {
    "name": "constraint_check",
    "description": "Validate plan config legality (OOM / invalid combinations).",
    "input_schema": {
        "type": "object",
        "properties": {"plan_ref": {"type": "string"}},
        "required": ["plan_ref"],
    },
}

CONSTRAINT_CHECK_ENV = {
    "name": "constraint_check_env",
    "description": "Validate env combination (mutex / dangerous combos).",
    "input_schema": {
        "type": "object",
        "properties": {
            "env_diff": {"type": "object"},
            "baseline": {"type": "string"},
        },
        "required": ["env_diff"],
    },
}

CONSTRAINT_ESTIMATE_MEM = {
    "name": "constraint_estimate_mem",
    "description": "Estimate memory for a plan.",
    "input_schema": {
        "type": "object",
        "properties": {"plan_ref": {"type": "string"}},
        "required": ["plan_ref"],
    },
}

KNOWLEDGE_WRITE = {
    "name": "knowledge_write",
    "description": "LEARN stage: write best/failure case back to skills/knowledge/.",
    "input_schema": {
        "type": "object",
        "properties": {
            "report_ref": {"type": "string"},
            "kind": {"type": "string", "enum": ["best", "failure", "pattern"]},
        },
        "required": ["report_ref", "kind"],
    },
}


# ---------------------------------------------------------------------------
# Per-stage tool set (§13.2 scope enforcement)
# A Worker only sees the tools relevant to its stage.
# ---------------------------------------------------------------------------
STAGE_TOOL_MAP: dict[Stage, list[dict[str, Any]]] = {
    Stage.PREFLIGHT: [PREFLIGHT_RUN, ENV_PROBE_RUN, WRITE_ARTIFACT, RETURN_SUBAGENT_RESULT],
    Stage.PROJECTION: [PROFILER_RUN, CONSTRAINT_ESTIMATE_MEM, WRITE_ARTIFACT, RETURN_SUBAGENT_RESULT],
    Stage.SMOKE: [SUBMIT_RUN, OBSERVE_SNAPSHOT, WRITE_ARTIFACT, RETURN_SUBAGENT_RESULT],
    Stage.BASELINE: [SUBMIT_RUN, OBSERVE_SNAPSHOT, WRITE_ARTIFACT, RETURN_SUBAGENT_RESULT],
    Stage.CORRECTNESS: [OBSERVE_COMPARE_LOSS, RETURN_SUBAGENT_RESULT],
    Stage.OBSERVE: [OBSERVE_SNAPSHOT, WRITE_ARTIFACT, RETURN_SUBAGENT_RESULT],
    Stage.DIAGNOSE: [WRITE_ARTIFACT, RETURN_SUBAGENT_RESULT],
    Stage.REPLAN: [
        CONSTRAINT_CHECK,
        CONSTRAINT_CHECK_ENV,
        CONSTRAINT_ESTIMATE_MEM,
        WRITE_ARTIFACT,
        RETURN_SUBAGENT_RESULT,
    ],
    Stage.ENV_SWEEP: [
        ENV_PROBE_SWEEP,
        CONSTRAINT_CHECK_ENV,
        WRITE_ARTIFACT,
        RETURN_SUBAGENT_RESULT,
    ],
    Stage.CORRECTNESS_LITE: [OBSERVE_COMPARE_LOSS, RETURN_SUBAGENT_RESULT],
    Stage.SETTLE: [WRITE_ARTIFACT, RETURN_SUBAGENT_RESULT],
    Stage.LEARN: [KNOWLEDGE_WRITE, RETURN_SUBAGENT_RESULT],
}


def tools_for_stage(stage: Stage) -> list[dict[str, Any]]:
    return STAGE_TOOL_MAP.get(stage, [RETURN_SUBAGENT_RESULT])


# ---------------------------------------------------------------------------
# Handler registry — Worker routes tool_use blocks through here.
# Each handler takes tool args + a StateStore and returns a JSON-serializable result.
# Real implementations call out to Primus / Slurm / WandB; here we stub them.
# ---------------------------------------------------------------------------
def handle_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    state_store,  # StateStore, typed loosely to avoid circular import
) -> Any:
    if tool_name == "write_artifact":
        art = state_store.write_artifact(
            kind=tool_input["kind"],
            name=tool_input["name"],
            content=tool_input["content"],
        )
        return {"ref": art.ref, "size_bytes": art.size_bytes}

    # --- business tool stubs ---
    if tool_name == "preflight_run":
        return {"cluster_profile_ref": "STUB:cluster_profile_v1",
                "note": "wire to preflight.py"}
    if tool_name == "env_probe_run":
        return {"env_baseline_ref": "STUB:env_baseline_v1"}
    if tool_name == "env_probe_sweep":
        return {"best_env_diff": {}, "results": [], "note": "STUB"}
    if tool_name == "profiler_run":
        return {"profiling_result_ref": "STUB:profiling_v1"}
    if tool_name == "submit_run":
        return {"job_id": "STUB_JOB_0001"}
    if tool_name == "observe_snapshot":
        return {"snapshot_ref": "STUB:snapshot_v1",
                "tps": 0, "comm_ratio": 0.0, "bubble_ratio": 0.0}
    if tool_name == "observe_compare_loss":
        return {"pass": True, "delta_pct": 0.1}
    if tool_name == "constraint_check":
        return {"valid": True, "violations": []}
    if tool_name == "constraint_check_env":
        return {"valid": True, "violations": []}
    if tool_name == "constraint_estimate_mem":
        return {"mem_gb": 150}
    if tool_name == "knowledge_write":
        return {"written_paths": ["STUB:skills/knowledge/cases.md"]}

    raise NotImplementedError(f"no handler for tool: {tool_name}")
