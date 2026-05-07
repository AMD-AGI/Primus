"""Orchestrator / Master Agent — v2.0 implementation.

Architectural anchors (see pilot/README.md):
  §2.2  two-tier agent model (Orchestrator + Stage Worker)
  §13   context management & multi-agent orchestration
  §8.7  TuningState / §8.11 SubagentResult
  §12.1 context hygiene guardrails

The Orchestrator is a thin Claude agent with three invariants:

  (I1)  Fresh messages list per iteration.
        We do NOT accumulate conversation history across stages. Each iteration
        builds a new `messages=[{"role":"user", ...}]` list from the trimmed
        state alone. This is what keeps Orchestrator context O(1).

  (I2)  No business tools in scope.
        Physically separate tool sets: orchestrator_tools only has
        state.* / subagent.spawn / orchestrator.stop. Business tools
        (submit/observe/...) live exclusively inside StageWorker.

  (I3)  State Layer is the bus.
        Cross-iteration / cross-worker communication goes through files on
        disk (StateStore), never through in-memory context passing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .orchestrator_tools import ORCHESTRATOR_TOOLS
from .schemas import Budget, OrchestratorState, SpawnRequest, Stage, SubagentResult
from .skills import SkillLoader
from .state import StateStore
from .subagent import StageWorker

log = logging.getLogger(__name__)


class OrchestratorError(RuntimeError):
    pass


class Orchestrator:
    """Long-lived, thin agent driving the state machine."""

    def __init__(self, cfg: OrchestratorConfig, client: Any | None = None):
        """
        Args:
            cfg:    OrchestratorConfig
            client: anthropic.Anthropic instance. Required unless cfg.dry_run=True.
        """
        cfg.ensure_dirs()
        self.cfg = cfg
        self.client = client
        self.state_store = StateStore(cfg.session_dir)
        self.skills = SkillLoader(cfg.skills_dir)
        self._system_prompt = self._build_system_prompt()

        if not cfg.dry_run and client is None:
            raise OrchestratorError(
                "Anthropic client is required unless OrchestratorConfig.dry_run=True"
            )

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------
    def run(self) -> OrchestratorState:
        """Drive the state machine until a terminal stage or handoff."""
        state = self._bootstrap()
        log.info(
            "orchestrator start: session=%s stage=%s round=%d",
            state.session_id,
            state.current_stage.value,
            state.round_id,
        )

        while not state.is_terminal():
            state = self.step(state)

            if self._should_handoff(state):
                self._handoff(state, reason="context_pressure_auto")
                return state

            if self._budget_exhausted(state):
                log.info("budget exhausted; transitioning to REPORT")
                state.current_stage = Stage.REPORT

        log.info("orchestrator terminal: stage=%s", state.current_stage.value)
        return state

    # -----------------------------------------------------------------
    # One orchestrator turn (Invariant I1 lives here)
    # -----------------------------------------------------------------
    def step(self, state: OrchestratorState) -> OrchestratorState:
        """
        One iteration:

            1. Ask Claude for the single next action, given only the trimmed state.
            2. Execute that action (typically subagent_spawn).
            3. Fold SubagentResult into state (pointers only).
            4. Checkpoint + trim.

        The `messages` list sent to Claude is rebuilt from scratch every call
        — that is the core trick that keeps Orchestrator context bounded.
        """
        action = self._decide(state)
        state = self._apply_action(state, action)
        self.state_store.checkpoint(state)
        return self.state_store.trim(state)

    # -----------------------------------------------------------------
    # Claude call — THIN context
    # -----------------------------------------------------------------
    def _decide(self, state: OrchestratorState) -> dict[str, Any]:
        """Call Claude and return a single {name, input} tool call description."""
        user_content = self._build_user_message(state)

        if self.cfg.dry_run:
            # In dry-run mode, synthesize a deterministic decision so the
            # skeleton runs end-to-end without an API key.
            return self._dry_run_decide(state)

        # Real call — the only place we touch the Anthropic SDK.
        resp = self.client.messages.create(  # type: ignore[union-attr]
            model=self.cfg.orchestrator_model,
            max_tokens=self.cfg.orchestrator_response_tokens,
            system=self._system_prompt,
            tools=ORCHESTRATOR_TOOLS,
            tool_choice={"type": "any"},  # force exactly one tool call
            messages=[{"role": "user", "content": user_content}],
        )

        # Bookkeeping: track orchestrator's own token consumption toward budget
        try:
            used = resp.usage.input_tokens + resp.usage.output_tokens
            state.budget_used.total_tokens += used
            log.debug("orchestrator turn used %d tokens", used)
        except AttributeError:
            pass

        tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if len(tool_uses) != 1:
            raise OrchestratorError(
                f"Orchestrator must emit exactly one tool_use, got {len(tool_uses)}. "
                "Check the base prompt's 'one tool call per turn' rule."
            )
        tu = tool_uses[0]
        return {"name": tu.name, "input": dict(tu.input)}

    def _build_user_message(self, state: OrchestratorState) -> str:
        """The full user-side payload — just the trimmed state as JSON.

        We keep it structured so Claude doesn't waste tokens on prose parsing.
        """
        remaining = state.budget_remaining()
        payload = {
            "trimmed_state": state.model_dump(
                exclude={"stage_history", "reentry_log"}, mode="json"
            ),
            "recent_stage_history": state.stage_history[-3:],
            "recent_reentry": state.reentry_log[-1:] if state.reentry_log else [],
            "budget_remaining": {
                "gpu_h": remaining.gpu_h,
                "rounds": remaining.rounds,
                "wallclock_h": remaining.wallclock_h,
            },
            "instruction": (
                "Emit exactly one tool call to advance the state machine. "
                "See base rules in system prompt. If current_stage is a "
                "subagent stage, use subagent_spawn."
            ),
        }
        return "```json\n" + json.dumps(payload, indent=2, default=str) + "\n```"

    # -----------------------------------------------------------------
    # Action dispatch
    # -----------------------------------------------------------------
    def _apply_action(
        self, state: OrchestratorState, action: dict[str, Any]
    ) -> OrchestratorState:
        name = action["name"]
        args = action["input"]

        if name == "subagent_spawn":
            return self._handle_spawn(state, args)
        if name == "state_checkpoint":
            return state  # outer loop checkpoints anyway
        if name == "state_trim":
            return state  # outer loop trims anyway
        if name == "state_handoff":
            self._handoff(state, reason=args.get("reason", "manual"))
            state.current_stage = Stage.REPORT
            return state
        if name == "orchestrator_stop":
            log.info("orchestrator_stop: %s", args)
            state.current_stage = (
                Stage.ABORT if args.get("reason") == "ABORT" else Stage.REPORT
            )
            state.last_decision_summary = args.get("final_summary", "")
            return state

        raise OrchestratorError(f"unknown tool emitted by orchestrator: {name}")

    def _handle_spawn(
        self, state: OrchestratorState, args: dict[str, Any]
    ) -> OrchestratorState:
        req = SpawnRequest(
            stage=Stage(args["stage"]),
            input_refs=args.get("input_refs", {}),
            skill_scope=args.get("skill_scope", []),
            max_tokens=args.get("max_tokens", self.cfg.worker_max_tokens),
            parent_session=state.session_id,
        )
        log.info(
            "spawn: stage=%s scope=%s reason=%s",
            req.stage.value,
            req.skill_scope,
            args.get("reason", ""),
        )

        worker = StageWorker(
            cfg=self.cfg,
            client=self.client,
            skills=self.skills,
            state_store=self.state_store,
        )
        result: SubagentResult = worker.run(req)

        if result.status == "failed" and result.failure:
            log.warning(
                "subagent failure: kind=%s escalate=%s msg=%s",
                result.failure.kind,
                result.failure.escalate_to_orchestrator,
                result.failure.message,
            )
            # state_machine.md's on_fail table routes from here; for the
            # skeleton we record and let the next decide() turn choose
            # the recovery action.
            state.reentry_log.append(
                {
                    "from": result.stage.value,
                    "failure_kind": result.failure.kind,
                    "message": result.failure.message,
                }
            )

        return self.state_store.apply_result(state, result)

    # -----------------------------------------------------------------
    # Bootstrap / Handoff / Hygiene
    # -----------------------------------------------------------------
    def _bootstrap(self) -> OrchestratorState:
        resumed = self.state_store.resume()
        if resumed is not None:
            log.info("resumed from checkpoint; stage=%s round=%d",
                     resumed.current_stage.value, resumed.round_id)
            return resumed
        log.info("no prior checkpoint — fresh session")
        return self.state_store.initial(self.cfg.session_id, self.cfg.budget_total)

    def _should_handoff(self, state: OrchestratorState) -> bool:
        """§13.4 context budget check — two signals:

          (a) token budget consumed > ratio of total budget, OR
          (b) rounds_since_handoff >= max_rounds_before_handoff
        """
        if state.budget_total.total_tokens > 0:
            ratio = (
                state.budget_used.total_tokens / state.budget_total.total_tokens
            )
            if ratio > self.cfg.handoff_token_ratio:
                return True
        if state.round_id > 0 and (state.round_id % self.cfg.max_rounds_before_handoff == 0):
            return True
        return False

    def _budget_exhausted(self, state: OrchestratorState) -> bool:
        r = state.budget_remaining()
        return r.rounds < 0 or r.gpu_h < 0 or r.wallclock_h < 0

    def _handoff(self, state: OrchestratorState, reason: str) -> None:
        path = self.state_store.handoff(state, reason)
        log.info("handoff written: %s (reason=%s)", path, reason)

    # -----------------------------------------------------------------
    # System prompt — loaded ONCE at construction time
    # -----------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        """The total Skill knowledge the Orchestrator is allowed to carry.

        Only two Skill files + the base instructions:
          - prompts/orchestrator_base.md
          - skills/workflow/state_machine.md
          - skills/workflow/orchestration.md

        Everything else (optimization/*, env/*, execution-model/*, ...) is
        delegated to Stage Workers — they load what they need when spawned.
        """
        base_path = Path(__file__).parent / "prompts" / "orchestrator_base.md"
        base = base_path.read_text(encoding="utf-8")
        state_machine = self.skills.read("workflow/state_machine.md")
        orchestration = self.skills.read("workflow/orchestration.md")
        return "\n\n---\n\n".join(
            [
                base,
                "# skills/workflow/state_machine.md\n\n" + state_machine,
                "# skills/workflow/orchestration.md\n\n" + orchestration,
            ]
        )

    # -----------------------------------------------------------------
    # Dry-run helper (no Anthropic API needed)
    # -----------------------------------------------------------------
    def _dry_run_decide(self, state: OrchestratorState) -> dict[str, Any]:
        """Deterministic stage scheduler used for skeleton tests.

        Walks the state machine in a fixed order. Not meant to replace
        real LLM reasoning; just lets the skeleton run end-to-end.
        """
        stage = state.current_stage
        if stage == Stage.PREFLIGHT:
            return _spawn_call(stage, ["workflow/preflight.md"])
        if stage == Stage.PROJECTION:
            return _spawn_call(stage, ["workflow/projection.md", "execution-model/*"])
        if stage == Stage.SMOKE:
            return _spawn_call(stage, ["workflow/smoke.md"])
        if stage == Stage.BASELINE:
            return _spawn_call(Stage.OBSERVE, ["workflow/observe.md"])
        if stage == Stage.CORRECTNESS:
            return _spawn_call(stage, ["workflow/correctness.md"])
        if stage == Stage.OBSERVE:
            return _spawn_call(stage, ["workflow/observe.md", "profiling/*"])
        if stage == Stage.DIAGNOSE:
            return _spawn_call(
                stage, ["workflow/diagnose.md", "execution-model/*"]
            )
        if stage == Stage.REPLAN:
            return _spawn_call(
                stage, ["workflow/replan.md", "optimization/*", "constraints/*"]
            )
        if stage == Stage.ENV_SWEEP:
            return _spawn_call(stage, ["workflow/envsweep.md", "env/*"])
        if stage == Stage.EXECUTE:
            # EXECUTE is inline in real operation; in dry-run we skip straight
            # to CORRECTNESS_LITE to keep the loop flowing.
            return _spawn_call(Stage.CORRECTNESS_LITE, ["workflow/correctness.md"])
        if stage == Stage.CORRECTNESS_LITE:
            return _spawn_call(stage, ["workflow/correctness.md"])
        if stage == Stage.SETTLE:
            # Check budget BEFORE spawning so we don't overshoot
            if state.round_id >= state.budget_total.rounds:
                return {
                    "name": "orchestrator_stop",
                    "input": {
                        "reason": "BUDGET_EXCEEDED",
                        "final_summary": f"dry-run done after {state.round_id} rounds",
                    },
                }
            # Spawn SETTLE so apply_result sees stage==SETTLE and increments round_id.
            # (In real impl SETTLE can be inline per §13.2; keeping it as a spawn
            # here keeps the dry-run scheduler uniform.)
            return _spawn_call(stage, ["workflow/settle.md"])
        if stage in (Stage.REPORT, Stage.LEARN):
            return {
                "name": "orchestrator_stop",
                "input": {"reason": "CONVERGED", "final_summary": "dry-run terminal"},
            }
        return {
            "name": "orchestrator_stop",
            "input": {"reason": "ABORT", "final_summary": f"unhandled stage {stage}"},
        }


def _spawn_call(stage: Stage, scope: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "name": "subagent_spawn",
        "input": {
            "stage": stage.value,
            "input_refs": extra,
            "skill_scope": scope,
            "reason": f"dry-run schedule: {stage.value}",
        },
    }
