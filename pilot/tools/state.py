"""state tools: persist / resume / trim / handoff.

Pilot's State Layer is the single source of truth across stages. Tools here
read and write YAML / JSON under `state/`.

Status: skeleton.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


def checkpoint(tuning_state: dict, *, root: str = "state") -> str:
    """Persist TuningState (§8.7) at stage exit. Returns the written path."""
    raise NotImplementedError("pilot.tools.state.checkpoint")


def resume(path: str | Path) -> dict:
    """Load TuningState from a checkpoint path."""
    raise NotImplementedError("pilot.tools.state.resume")


def trim(tuning_state: dict, *, keep: list[str]) -> dict:
    """Orchestrator-only: discard non-pointer fields at stage exit (§13.2 strategy A).

    `keep` is the allowlist of pointer-class fields:
        {session_id, current_stage, round_id, champion_id, budget_used,
         last_decision_summary}
    """
    raise NotImplementedError("pilot.tools.state.trim")


def handoff(session_id: str, *, reason: str, next_action_hint: str) -> str:
    """Orchestrator-only: write a session-handoff landing point (§13.2 strategy C).

    Returns the handoff file path; new Orchestrator process resumes from it.
    """
    raise NotImplementedError("pilot.tools.state.handoff")


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.state")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("checkpoint", "resume", "trim", "handoff"):
        sub.add_parser(name)
    args = p.parse_args()
    raise NotImplementedError(f"CLI dispatch for {args.cmd!r} not implemented")


if __name__ == "__main__":
    sys.exit(_cli())
