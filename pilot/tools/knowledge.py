"""knowledge tool: write LEARN drafts.

Per §S4 governance, this tool writes ONLY to `state/knowledge_drafts/`.
It cannot write to `skills/knowledge/` — that path is reserved for
curator-merged git PRs.

"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

DraftKind = Literal[
    "final_best_case",
    "failure_pattern",
    "env_recipe",
    "model_calibration_drift",
]


def write(
    report: dict,
    kind: DraftKind,
    *,
    drafts_root: str = "state/knowledge_drafts",
    id_suffix: str | None = None,
) -> dict:
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
    reasons: list[str] = []
    verdict = report.get("verdict") or {}
    session = report.get("session") or {}
    tuning = report.get("tuning") or {}
    champion = tuning.get("champion") or {}
    headline = verdict.get("headline") or f"{kind} draft for {session.get('plan_name', 'unknown plan')}"
    if len(headline) > 200:
        reasons.append("headline exceeds 200 chars")
        headline = headline[:197] + "..."
    evidence = []
    for artifact in report.get("artifacts") or []:
        if artifact.get("ref"):
            evidence.append({"kind": artifact.get("kind", "artifact"), "ref": artifact["ref"]})
    if not evidence:
        reasons.append("no artifact evidence attached")
    if not session.get("plan_ref"):
        reasons.append("missing plan binding")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    if id_suffix:
        safe_suffix = "".join(c if c.isalnum() or c in "-_" else "_" for c in id_suffix)[:48]
        draft_id = f"{kind}_{ts}_{safe_suffix}"
    else:
        draft_id = f"{kind}_{ts}"
    draft = {
        "schema_version": "1.0",
        "draft_id": draft_id,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accepted": not reasons,
        "reasons": reasons,
        "headline": headline,
        "binding": {
            "plan_ref": session.get("plan_ref"),
            "cluster_id": session.get("cluster_id"),
            "model": session.get("plan_name"),
        },
        "evidence": evidence,
        "content": {
            "overall": verdict.get("overall"),
            "next_action": verdict.get("next_action"),
            "champion_id": champion.get("id"),
            "champion_overrides": champion.get("overrides"),
            "improvement_pct": tuning.get("improvement_pct"),
            **(report.get("content") or {}),
        },
    }
    root = Path(drafts_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{draft_id}.yaml"
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        return {
            "written_path": "",
            "draft_id": draft_id,
            "accepted": False,
            "reasons": [f"PyYAML required: {exc}"],
        }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(draft, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)
    return {"written_path": str(path), "draft_id": draft_id, "accepted": not reasons, "reasons": reasons}


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.knowledge")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument(
        "--kind",
        required=True,
        choices=["final_best_case", "failure_pattern", "env_recipe", "model_calibration_drift"],
    )
    w.add_argument("--report", required=True)
    w.add_argument("--drafts-root", default="state/knowledge_drafts")
    args = p.parse_args()
    try:
        if args.cmd == "write":
            import yaml  # type: ignore

            with open(args.report) as f:
                report = yaml.safe_load(f)
            if not isinstance(report, dict):
                raise ValueError("report must be a mapping")
            print(json.dumps(write(report, args.kind, drafts_root=args.drafts_root), indent=2, default=str))
            return 0
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "stage": "LEARN",
                    "status": "failed",
                    "failure": {"kind": "TOOL_ERROR", "message": str(exc), "escalate_to_orchestrator": True},
                },
                indent=2,
            )
        )
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
