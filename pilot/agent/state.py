"""State Layer I/O.

The State Layer is the single source of truth for cross-stage memory (§2).
This module implements the four Orchestrator-facing tools (§5):

  - state.resume()      load the latest checkpoint
  - state.checkpoint()  persist current OrchestratorState
  - state.trim()        enforce context hygiene (§13.2 Strategy A)
  - state.handoff()     write a handoff point for a fresh Orchestrator

Heavy artifacts (PlanGraph / CandidatePool / Snapshot / DiagnosisReport)
are written by Stage Workers; the Orchestrator only stores refs to them.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .schemas import (
    Artifact,
    Budget,
    OrchestratorState,
    Stage,
    SubagentResult,
)

log = logging.getLogger(__name__)


class StateStore:
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.ckpt_dir = self.session_dir / "checkpoints"
        self.handoff_dir = self.session_dir / "handoffs"
        self.artifact_dir = self.session_dir / "artifacts"
        for d in (self.ckpt_dir, self.handoff_dir, self.artifact_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---------------- initial / resume ----------------------------------
    def initial(self, session_id: str, budget_total: Budget) -> OrchestratorState:
        return OrchestratorState(
            session_id=session_id,
            current_stage=Stage.PREFLIGHT,
            budget_total=budget_total,
        )

    def resume(self) -> OrchestratorState | None:
        latest = self._latest(self.ckpt_dir)
        if latest is None:
            return None
        return OrchestratorState.model_validate_json(latest.read_text())

    # ---------------- checkpoint ----------------------------------------
    def checkpoint(self, state: OrchestratorState) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        stage_tag = state.current_stage.name
        path = self.ckpt_dir / f"r{state.round_id:03d}_{stage_tag}_{ts}.json"
        path.write_text(state.model_dump_json(indent=2))
        log.debug("checkpoint: %s", path)
        return path

    # ---------------- trim (context hygiene, §13.2 Strategy A) ----------
    def trim(self, state: OrchestratorState) -> OrchestratorState:
        """Return a state object with only pointer-class fields kept.

        The outer loop of the Orchestrator calls this right after checkpoint(),
        guaranteeing that the next iteration's Claude call sees only the
        minimal summary, never the prior round's heavy payloads.

        Heuristic: keep last 3 stage_history entries and last 1 reentry entry.
        Everything else stays on disk via the checkpoint.
        """
        return OrchestratorState(
            session_id=state.session_id,
            current_stage=state.current_stage,
            round_id=state.round_id,
            champion_id=state.champion_id,
            cluster_profile_ref=state.cluster_profile_ref,
            plan_graph_ref=state.plan_graph_ref,
            target_vector_ref=state.target_vector_ref,
            budget_total=state.budget_total,
            budget_used=state.budget_used,
            last_decision_summary=state.last_decision_summary,
            stage_history=state.stage_history[-3:],
            reentry_log=state.reentry_log[-1:] if state.reentry_log else [],
        )

    # ---------------- handoff (context overflow, §13.2 Strategy C) ------
    def handoff(self, state: OrchestratorState, reason: str) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.handoff_dir / f"{ts}_{reason}.json"
        payload = {
            "handoff_reason": reason,
            "handoff_at": datetime.utcnow().isoformat(),
            "state": state.model_dump(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        log.info("handoff written: %s (reason=%s)", path, reason)
        return path

    def resume_from_handoff(self, path: Path) -> OrchestratorState:
        payload = json.loads(path.read_text())
        return OrchestratorState.model_validate(payload["state"])

    # ---------------- apply SubagentResult ------------------------------
    def apply_result(
        self, state: OrchestratorState, result: SubagentResult
    ) -> OrchestratorState:
        """Fold a SubagentResult into state. Only pointer/summary fields are touched."""
        state.last_decision_summary = str(result.summary.get("headline", ""))
        state.stage_history.append(
            {
                "round": state.round_id,
                "stage": result.stage.value,
                "status": result.status,
                "headline": state.last_decision_summary,
                "at": datetime.utcnow().isoformat(),
            }
        )

        # Stage-specific pointer updates. Add more as stages come online.
        for art in result.artifacts:
            if art.kind == "ClusterProfile":
                state.cluster_profile_ref = art.ref
            elif art.kind == "PlanGraph":
                state.plan_graph_ref = art.ref
            elif art.kind == "TargetVector":
                state.target_vector_ref = art.ref
            elif art.kind == "Champion":
                # Settle Worker promotes a new champion by returning its id
                state.champion_id = art.ref

        # Budget tracking
        state.budget_used.total_tokens += int(result.cost.get("tokens_used", 0))
        state.budget_used.wallclock_h += float(result.cost.get("wallclock_s", 0.0)) / 3600.0
        state.budget_used.gpu_h += float(result.cost.get("gpu_h", 0.0))
        if result.stage == Stage.SETTLE:
            state.round_id += 1
            state.budget_used.rounds += 1

        # Stage transition: Worker suggests, state_machine.md (loaded in Orchestrator system prompt) adjudicates
        if result.suggested_transition is not None:
            state.current_stage = result.suggested_transition.to

        return state

    # ---------------- artifact helpers ----------------------------------
    def write_artifact(self, kind: str, name: str, content: str) -> Artifact:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.artifact_dir / f"{ts}_{kind}_{name}"
        path.write_text(content)
        return Artifact(kind=kind, ref=str(path), size_bytes=path.stat().st_size)

    # ---------------- internals -----------------------------------------
    @staticmethod
    def _latest(folder: Path) -> Path | None:
        files = sorted(folder.glob("*.json"))
        return files[-1] if files else None
