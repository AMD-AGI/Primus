"""Pydantic schemas — the wire format between Orchestrator, Stage Worker, and State Layer.

Maps to:
  - §8.7 TuningState        → OrchestratorState (trimmed version)
  - §8.11 SubagentResult    → SubagentResult
  - §12 FailureReport       → SubagentFailure (inline on SubagentResult)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Stages — one-to-one with state_machine.md
# ---------------------------------------------------------------------------
class Stage(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    PROJECTION = "PROJECTION"
    SMOKE = "SMOKE"
    BASELINE = "BASELINE"
    CORRECTNESS = "CORRECTNESS"
    # OPTIMIZE_LOOP substages
    OBSERVE = "OPTIMIZE_LOOP.OBSERVE"
    DIAGNOSE = "OPTIMIZE_LOOP.DIAGNOSE"
    REPLAN = "OPTIMIZE_LOOP.REPLAN"
    EXECUTE = "OPTIMIZE_LOOP.EXECUTE"
    CORRECTNESS_LITE = "OPTIMIZE_LOOP.CORRECTNESS_LITE"
    ENV_SWEEP = "OPTIMIZE_LOOP.ENV_SWEEP"
    SETTLE = "OPTIMIZE_LOOP.SETTLE"
    # Terminal
    REPORT = "REPORT"
    LEARN = "LEARN"
    ABORT = "ABORT"


SUBAGENT_STAGES: set[Stage] = {
    Stage.PREFLIGHT,
    Stage.PROJECTION,
    Stage.OBSERVE,
    Stage.DIAGNOSE,
    Stage.REPLAN,
    Stage.ENV_SWEEP,
    Stage.CORRECTNESS_LITE,
    Stage.SMOKE,
    Stage.CORRECTNESS,
    Stage.LEARN,
}
"""Stages the Orchestrator delegates to subagents (§13.2 boundary table)."""


ORCHESTRATOR_INLINE_STAGES: set[Stage] = {
    Stage.EXECUTE,
    Stage.SETTLE,
    Stage.BASELINE,
}
"""Stages that stay in the Orchestrator (lightweight, no heavy Skills)."""


# ---------------------------------------------------------------------------
# Budget (§8.6 TargetVector.budget)
# ---------------------------------------------------------------------------
class Budget(BaseModel):
    gpu_h: float = 0.0
    rounds: int = 0
    wallclock_h: float = 0.0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# OrchestratorState — the THIN state held in Orchestrator's context.
# §13.2 Strategy A: trim() guarantees only these fields survive between iterations.
# ---------------------------------------------------------------------------
class OrchestratorState(BaseModel):
    session_id: str
    current_stage: Stage
    round_id: int = 0

    # Pointers only — never the payload itself
    champion_id: str | None = None
    cluster_profile_ref: str | None = None
    plan_graph_ref: str | None = None
    target_vector_ref: str | None = None

    # Budget tracking
    budget_total: Budget
    budget_used: Budget = Field(default_factory=Budget)

    # One-line summaries
    last_decision_summary: str = ""
    stage_history: list[dict[str, Any]] = Field(default_factory=list)
    reentry_log: list[dict[str, Any]] = Field(default_factory=list)

    def budget_remaining(self) -> Budget:
        return Budget(
            gpu_h=self.budget_total.gpu_h - self.budget_used.gpu_h,
            rounds=self.budget_total.rounds - self.budget_used.rounds,
            wallclock_h=self.budget_total.wallclock_h - self.budget_used.wallclock_h,
            total_tokens=self.budget_total.total_tokens - self.budget_used.total_tokens,
        )

    def is_terminal(self) -> bool:
        return self.current_stage in {Stage.REPORT, Stage.ABORT}


# ---------------------------------------------------------------------------
# SubagentResult — §8.11
# The ONLY structured payload a Stage Worker returns to Orchestrator.
# ---------------------------------------------------------------------------
class Artifact(BaseModel):
    kind: str  # "ClusterProfile" / "CandidatePool" / "DiagnosisReport" / ...
    ref: str   # path or URI in State Layer
    size_bytes: int = 0


class Transition(BaseModel):
    to: Stage
    reason: str = ""


class SubagentFailure(BaseModel):
    kind: Literal[
        "SKILL_MISSING",
        "TOOL_ERROR",
        "CONSTRAINT_VIOLATION",
        "TIMEOUT",
        "TOKEN_BUDGET",
        "OOM",
        "HANG",
        "NUMERICAL",
        "UNKNOWN",
    ]
    message: str = ""
    escalate_to_orchestrator: bool = False


class SubagentResult(BaseModel):
    stage: Stage
    worker_id: str
    status: Literal["success", "failed", "escalate"]

    artifacts: list[Artifact] = Field(default_factory=list)

    # Hard-capped summary (Orchestrator's only source of detail)
    summary: dict[str, Any] = Field(default_factory=dict)

    suggested_transition: Transition | None = None
    cost: dict[str, Any] = Field(default_factory=dict)  # {tokens_used, wallclock_s, tool_calls}
    failure: SubagentFailure | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("summary")
    @classmethod
    def _summary_size_cap(cls, v: dict[str, Any]) -> dict[str, Any]:
        """§12.1 Guardrail: summary.headline + key_metrics should stay < ~200 tokens.

        Rough heuristic: < 1200 chars (≈ 200 tokens English, fewer for code/yaml).
        """
        import json

        raw = json.dumps(v, ensure_ascii=False)
        if len(raw) > 1200:
            raise ValueError(
                f"SubagentResult.summary exceeds 1200-char budget ({len(raw)} chars). "
                "Move details into artifacts[] and keep summary as headline + key_metrics."
            )
        return v


# ---------------------------------------------------------------------------
# Spawn request — Orchestrator → StageWorker
# ---------------------------------------------------------------------------
class SpawnRequest(BaseModel):
    stage: Stage
    input_refs: dict[str, Any] = Field(default_factory=dict)
    skill_scope: list[str] = Field(default_factory=list)
    max_tokens: int = 30_000
    parent_session: str = ""
