"""knowledge tool: write LEARN drafts.

Per §S4 governance, this tool writes ONLY to `state/knowledge_drafts/`.
It cannot write to `skills/knowledge/` — that path is reserved for
curator-merged git PRs.

Status: skeleton.
"""

from __future__ import annotations
import argparse
import sys
from typing import Literal


DraftKind = Literal[
    "final_best_case",
    "failure_pattern",
    "env_recipe",
    "model_calibration_drift",
]


def write(report: dict, kind: DraftKind, *,
          drafts_root: str = "state/knowledge_drafts") -> dict:
    """Emit a knowledge draft (§S4.2).

    Validates against schemas/knowledge_draft.schema.json. Auto-rejects
    drafts that violate Stage-A anti-patterns (§S4.2):
        - empty or all-LLM-inference evidence
        - headline > 200 chars
        - empty / wildcard binding
        - direct contradiction with skills/knowledge/anti-patterns.md

    Returns:
        {written_path: str, draft_id: str, accepted: bool, reasons: list[str]}
    """
    raise NotImplementedError("pilot.tools.knowledge.write")


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.knowledge")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--kind", required=True, choices=[
        "final_best_case", "failure_pattern", "env_recipe", "model_calibration_drift"])
    w.add_argument("--report", required=True)
    args = p.parse_args()
    raise NotImplementedError(f"CLI dispatch for {args.cmd!r} not implemented")


if __name__ == "__main__":
    sys.exit(_cli())
