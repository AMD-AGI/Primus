"""Stage Worker — one-shot Claude agent with isolated context.

Architectural anchors (pilot/README.md):
  §2.2  two-tier agent model
  §13.2 Strategy B subagent isolation
  §8.11 SubagentResult

Each Stage Worker:
  - Lives only for the duration of one subagent.spawn() call
  - Has its own Claude message thread (independent of Orchestrator)
  - Reads only the Skills in its declared `skill_scope`
  - Can call only the business tools scoped to its stage (see worker_tools.py)
  - MUST end by calling return_subagent_result exactly once

The Worker's internal context can grow to ~30K tokens during its run; that
is fine because it is discarded at the end. The Orchestrator never sees
any of it — only the SubagentResult summary + artifact refs.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .config import OrchestratorConfig
from .schemas import (
    Artifact,
    SpawnRequest,
    Stage,
    SubagentFailure,
    SubagentResult,
    Transition,
)
from .skills import SkillLoader
from .state import StateStore
from .worker_tools import handle_tool_call, tools_for_stage

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDTRIPS = 12  # guardrail: prevent runaway tool loops


class StageWorker:
    def __init__(
        self,
        cfg: OrchestratorConfig,
        client: Any | None,
        skills: SkillLoader,
        state_store: StateStore,
    ):
        self.cfg = cfg
        self.client = client
        self.skills = skills
        self.state_store = state_store

    # -----------------------------------------------------------------
    def run(self, req: SpawnRequest) -> SubagentResult:
        worker_id = f"sw_{req.stage.name}_{uuid.uuid4().hex[:8]}"
        log.info(
            "worker start: id=%s stage=%s scope=%s",
            worker_id,
            req.stage.value,
            req.skill_scope,
        )

        if self.cfg.dry_run or self.client is None:
            return self._dry_run_result(worker_id, req)

        system = self._build_system_prompt(req)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._seed_user_message(req)}
        ]
        tools = tools_for_stage(req.stage)

        total_tokens = 0
        for step in range(MAX_TOOL_ROUNDTRIPS):
            resp = self.client.messages.create(
                model=self.cfg.worker_model,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=tools,
            )
            try:
                total_tokens += resp.usage.input_tokens + resp.usage.output_tokens
            except AttributeError:
                pass

            if total_tokens > req.max_tokens:
                return self._token_budget_failure(worker_id, req, total_tokens)

            stop = getattr(resp, "stop_reason", "")
            if stop == "end_turn":
                return self._extract_return(worker_id, req, resp, total_tokens)

            if stop == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = self._execute_tools(resp.content, req, worker_id)
                messages.append({"role": "user", "content": tool_results})
                # Check if return_subagent_result was called — if so, we're done
                if any(
                    getattr(b, "type", "") == "tool_use"
                    and b.name == "return_subagent_result"
                    for b in resp.content
                ):
                    return self._extract_return(worker_id, req, resp, total_tokens)
                continue

            log.warning("worker unexpected stop_reason=%s", stop)
            break

        return SubagentResult(
            stage=req.stage,
            worker_id=worker_id,
            status="failed",
            summary={"headline": "worker did not return within roundtrip budget"},
            failure=SubagentFailure(
                kind="UNKNOWN",
                message=f"hit MAX_TOOL_ROUNDTRIPS={MAX_TOOL_ROUNDTRIPS}",
                escalate_to_orchestrator=True,
            ),
            cost={"tokens_used": total_tokens},
        )

    # -----------------------------------------------------------------
    # System prompt & seed user message
    # -----------------------------------------------------------------
    def _build_system_prompt(self, req: SpawnRequest) -> str:
        skills_body = self.skills.read_scope(req.skill_scope)
        stage_tools = [t["name"] for t in tools_for_stage(req.stage)]

        return f"""You are a Stage Worker for stage `{req.stage.value}` in Primus Pilot.

# HARD RULES
  1. You are one-shot. Do your job and return.
  2. Produce exactly ONE `return_subagent_result` call at the end.
  3. `summary` field must fit in ~200 tokens; use `write_artifact` for the rest.
  4. You may only call these tools: {stage_tools}.
  5. You do not see the Orchestrator's history and cannot spawn further workers.
  6. If you cannot complete the task (missing skill / invalid input / etc.), return
     `status: failed` with a populated `failure` field.

# STAGE SKILLS

{skills_body}
"""

    def _seed_user_message(self, req: SpawnRequest) -> str:
        body = {
            "stage": req.stage.value,
            "input_refs": req.input_refs,
            "instruction": (
                "Execute this stage according to the Skills above. Use the "
                "listed tools as needed. Finish with return_subagent_result."
            ),
        }
        return "```json\n" + json.dumps(body, indent=2, default=str) + "\n```"

    # -----------------------------------------------------------------
    # Tool execution loop
    # -----------------------------------------------------------------
    def _execute_tools(
        self, content_blocks, req: SpawnRequest, worker_id: str
    ) -> list[dict[str, Any]]:
        """Return a list of tool_result blocks for the Anthropic messages API."""
        results: list[dict[str, Any]] = []
        for b in content_blocks:
            if getattr(b, "type", "") != "tool_use":
                continue
            if b.name == "return_subagent_result":
                # The extract step handles this; we still need a tool_result
                # block or Anthropic will reject the next turn.
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": "ok",
                    }
                )
                continue
            try:
                out = handle_tool_call(b.name, dict(b.input), self.state_store)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": json.dumps(out, default=str),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("tool %s failed", b.name)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "is_error": True,
                        "content": f"{type(exc).__name__}: {exc}",
                    }
                )
        return results

    # -----------------------------------------------------------------
    # Extract the final return_subagent_result tool call
    # -----------------------------------------------------------------
    def _extract_return(
        self,
        worker_id: str,
        req: SpawnRequest,
        resp: Any,
        total_tokens: int,
    ) -> SubagentResult:
        for b in resp.content:
            if getattr(b, "type", "") == "tool_use" and b.name == "return_subagent_result":
                args = dict(b.input)
                artifacts = [Artifact(**a) for a in args.get("artifacts", [])]
                transition = None
                st = args.get("suggested_transition")
                if st:
                    transition = Transition(to=Stage(st["to"]), reason=st.get("reason", ""))
                failure = None
                if args.get("failure"):
                    failure = SubagentFailure(**args["failure"])
                return SubagentResult(
                    stage=req.stage,
                    worker_id=worker_id,
                    status=args["status"],
                    artifacts=artifacts,
                    summary=args.get("summary", {}),
                    suggested_transition=transition,
                    failure=failure,
                    cost={"tokens_used": total_tokens},
                )
        # Worker ended without returning — that's a protocol failure.
        return SubagentResult(
            stage=req.stage,
            worker_id=worker_id,
            status="failed",
            summary={"headline": "worker ended without return_subagent_result"},
            failure=SubagentFailure(
                kind="UNKNOWN",
                message="protocol violation: missing final return tool call",
                escalate_to_orchestrator=True,
            ),
            cost={"tokens_used": total_tokens},
        )

    # -----------------------------------------------------------------
    # Dry-run path — deterministic SubagentResult per stage
    # -----------------------------------------------------------------
    def _dry_run_result(self, worker_id: str, req: SpawnRequest) -> SubagentResult:
        """Emits a canned success result + mocks the next transition so the
        Orchestrator loop can walk through stages without an API key."""
        next_stage = _DRY_RUN_NEXT.get(req.stage, Stage.REPORT)
        artifacts: list[Artifact] = []
        if req.stage == Stage.PREFLIGHT:
            artifacts.append(
                Artifact(kind="ClusterProfile", ref="STUB:cluster_profile_v1", size_bytes=0)
            )
        if req.stage == Stage.PROJECTION:
            artifacts.append(
                Artifact(kind="PlanGraph", ref="STUB:plan_graph_r0", size_bytes=0)
            )
        return SubagentResult(
            stage=req.stage,
            worker_id=worker_id,
            status="success",
            artifacts=artifacts,
            summary={
                "headline": f"[dry-run] {req.stage.value} ok",
                "key_metrics": {},
            },
            suggested_transition=Transition(
                to=next_stage, reason="dry-run deterministic schedule"
            ),
            cost={"tokens_used": 0, "wallclock_s": 0.0},
        )

    def _token_budget_failure(
        self, worker_id: str, req: SpawnRequest, tokens: int
    ) -> SubagentResult:
        return SubagentResult(
            stage=req.stage,
            worker_id=worker_id,
            status="failed",
            summary={
                "headline": f"worker exceeded token budget ({tokens}/{req.max_tokens})"
            },
            failure=SubagentFailure(
                kind="TOKEN_BUDGET",
                message="early return at budget cap",
                escalate_to_orchestrator=True,
            ),
            cost={"tokens_used": tokens},
        )


# Simple stage progression for dry-run mode
_DRY_RUN_NEXT: dict[Stage, Stage] = {
    Stage.PREFLIGHT: Stage.PROJECTION,
    Stage.PROJECTION: Stage.SMOKE,
    Stage.SMOKE: Stage.BASELINE,
    Stage.BASELINE: Stage.CORRECTNESS,
    Stage.CORRECTNESS: Stage.OBSERVE,
    Stage.OBSERVE: Stage.DIAGNOSE,
    Stage.DIAGNOSE: Stage.REPLAN,
    Stage.REPLAN: Stage.EXECUTE,
    Stage.EXECUTE: Stage.OBSERVE,
    Stage.CORRECTNESS_LITE: Stage.SETTLE,
    Stage.ENV_SWEEP: Stage.SETTLE,
    Stage.SETTLE: Stage.OBSERVE,  # next round
    Stage.LEARN: Stage.REPORT,
}
