"""state tools: persist / resume / trim / handoff.

Pilot's State Layer is the single source of truth across stages. Tools here
read and write YAML / JSON under ``pilot/state`` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent
_DEFAULT_KEEP = [
    "session_id",
    "current_stage",
    "round_id",
    "champion_id",
    "cluster_profile_ref",
    "current_plan_ref",
    "plan_graph_ref",
    "candidate_pool_ref",
    "target_vector_ref",
    "budget_used",
    "budget_remaining",
    "last_decision_summary",
]


class _StateError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _resolve_pilot_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else _PILOT_ROOT / p


def _yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise _StateError("DEP_MISSING", f"PyYAML required for state tools: {exc}") from exc
    return yaml


def _load_data(path: str | Path | None) -> dict[str, Any]:
    yaml = _yaml()
    if path is None or str(path) == "-":
        raw_text = sys.stdin.read()
        source = "<stdin>"
    else:
        p = _resolve_pilot_path(path)
        source = str(p)
        if p.is_dir():
            p = p / "tuning_state.yaml"
        if not p.exists():
            raise _StateError("USAGE", f"state file not found: {p}")
        raw_text = p.read_text()

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise _StateError("USAGE", f"{source} is not valid YAML/JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _StateError("USAGE", f"{source} must contain a mapping")
    return data


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    yaml = _yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


def _round_id(tuning_state: dict[str, Any]) -> int:
    raw = tuning_state.get("round_id")
    if raw is None:
        raw = (tuning_state.get("budget_used") or {}).get("rounds", 0)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise _StateError("USAGE", f"round_id must be an integer, got {raw!r}")
    return max(0, value)


def _validate_minimal_state(tuning_state: dict[str, Any]) -> None:
    missing = [k for k in ("session_id", "current_stage", "stage_history") if k not in tuning_state]
    if missing:
        raise _StateError("USAGE", f"tuning_state missing required fields: {missing}")
    if not isinstance(tuning_state["stage_history"], list):
        raise _StateError("USAGE", "tuning_state.stage_history must be a list")


def checkpoint(tuning_state: dict, *, root: str = "state") -> str:
    """Persist TuningState at stage exit.

    Writes both ``<root>/tuning_state.yaml`` and
    ``<root>/checkpoints/r<N>/tuning_state.yaml``. Returns the checkpoint path.
    """
    state = deepcopy(tuning_state)
    _validate_minimal_state(state)
    state.setdefault("schema_version", "1.0")
    state["checkpointed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    root_path = _resolve_pilot_path(root)
    current_path = root_path / "tuning_state.yaml"
    checkpoint_path = root_path / "checkpoints" / f"r{_round_id(state)}" / "tuning_state.yaml"
    _atomic_write_yaml(current_path, state)
    _atomic_write_yaml(checkpoint_path, state)
    return str(checkpoint_path)


def resume(path: str | Path) -> dict:
    """Load TuningState from a file or checkpoint directory."""
    return _load_data(path)


def trim(tuning_state: dict, *, keep: list[str]) -> dict:
    """Discard non-pointer fields at stage exit using an explicit allowlist."""
    if not keep:
        raise _StateError("USAGE", "trim keep list cannot be empty")
    return {k: deepcopy(tuning_state[k]) for k in keep if k in tuning_state}


def update_pointers(
    session: str,
    *,
    fields: dict[str, Any] | None = None,
    append_history: dict[str, Any] | None = None,
    root: str = "state",
) -> str:
    """Granular write tool for the Orchestrator's per-round bookkeeping.

    Replaces the read-yaml→mutate-dict-in-python→checkpoint pattern that
    every round of session 20260513T024603Z hand-rolled in inline Python
    (see ``IMPL_VS_DESIGN.md §4``). Two operations:

    - ``fields``: shallow merge into the top-level ``tuning_state`` mapping.
      Use for pointer-class fields the Orchestrator owns: ``current_stage``,
      ``round_id``, ``champion_id``, ``budget_used``, ``last_decision_summary``,
      ``current_plan_ref``, ``candidate_pool_ref``, ``plan_graph_ref``.
    - ``append_history``: append one entry to ``stage_history`` (a YAML list).
      The entry is added verbatim; the caller is responsible for shape
      (``{stage, status, headline}`` per orchestration.md §3 step 2).

    Both arguments are optional; the call still rewrites the file (and
    a per-round checkpoint) so the ``checkpointed_at`` timestamp moves
    forward — matching the state-hygiene ritual.

    Returns the per-round checkpoint path (same shape as ``checkpoint``).

    The session_id argument is used to locate the file. The ``root`` is
    the same default as ``checkpoint``.
    """
    root_path = _resolve_pilot_path(root) / session
    current_path = root_path / "tuning_state.yaml"
    if not current_path.exists():
        raise _StateError(
            "USAGE",
            f"tuning_state.yaml not found for session {session!r} at {current_path}",
        )
    state_doc = _load_data(current_path)
    if fields:
        for k, v in fields.items():
            state_doc[k] = deepcopy(v)
    if append_history is not None:
        history = state_doc.setdefault("stage_history", [])
        if not isinstance(history, list):
            raise _StateError("USAGE", "stage_history must be a list")
        history.append(deepcopy(append_history))
    return checkpoint(state_doc, root=str(root_path))


def handoff(session_id: str, *, reason: str, next_action_hint: str) -> str:
    """Write a session-handoff landing point and return its path."""
    if not session_id:
        raise _StateError("USAGE", "session_id is required")
    payload = {
        "schema_version": "1.0",
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "next_action_hint": next_action_hint,
    }
    path = _PILOT_ROOT / "state" / "checkpoints" / "handoff" / f"{session_id}.yaml"
    _atomic_write_yaml(path, payload)
    return str(path)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "STATE",
        "status": "failed",
        "failure": {"kind": kind, "message": message, "escalate_to_orchestrator": True},
    }


def _split_keep(values: list[str]) -> list[str]:
    keep: list[str] = []
    for value in values:
        keep.extend([v.strip() for v in value.split(",") if v.strip()])
    return keep


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.state")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ckpt = sub.add_parser("checkpoint")
    p_ckpt.add_argument("--input", "-i", default="-", help="YAML/JSON state file, or '-' for stdin.")
    p_ckpt.add_argument("--root", default="state")

    p_resume = sub.add_parser("resume")
    p_resume.add_argument("path", nargs="?", default="state/tuning_state.yaml")

    p_trim = sub.add_parser("trim")
    p_trim.add_argument("--input", "-i", default="-", help="YAML/JSON state file, or '-' for stdin.")
    p_trim.add_argument(
        "--keep", action="append", default=[], help="Field to keep. Can be repeated or comma-separated."
    )

    p_handoff = sub.add_parser("handoff")
    p_handoff.add_argument("--session-id", required=True)
    p_handoff.add_argument("--reason", required=True)
    p_handoff.add_argument("--next-action-hint", required=True)

    p_update = sub.add_parser(
        "update_pointers",
        help="Merge pointer-class fields and/or append a stage_history entry.",
    )
    p_update.add_argument("--session", required=True, help="Session id (directory name under <root>/).")
    p_update.add_argument("--root", default="state")
    p_update.add_argument(
        "--fields", default=None, help="JSON object of pointer fields to merge (top-level shallow)."
    )
    p_update.add_argument("--append-history", default=None, help="JSON object to append to stage_history.")

    args = p.parse_args()
    try:
        if args.cmd == "checkpoint":
            path = checkpoint(_load_data(args.input), root=args.root)
            _emit({"stage": "STATE", "status": "success", "path": path})
            return 0
        if args.cmd == "resume":
            _emit(resume(args.path))
            return 0
        if args.cmd == "trim":
            keep = _split_keep(args.keep) if args.keep else list(_DEFAULT_KEEP)
            _emit(trim(_load_data(args.input), keep=keep))
            return 0
        if args.cmd == "handoff":
            path = handoff(
                args.session_id,
                reason=args.reason,
                next_action_hint=args.next_action_hint,
            )
            _emit({"stage": "STATE", "status": "success", "path": path})
            return 0
        if args.cmd == "update_pointers":
            fields = json.loads(args.fields) if args.fields else None
            history_entry = json.loads(args.append_history) if args.append_history else None
            if fields is not None and not isinstance(fields, dict):
                raise _StateError("USAGE", "--fields must be a JSON object")
            if history_entry is not None and not isinstance(history_entry, dict):
                raise _StateError("USAGE", "--append-history must be a JSON object")
            path = update_pointers(
                args.session,
                fields=fields,
                append_history=history_entry,
                root=args.root,
            )
            _emit({"stage": "STATE", "status": "success", "path": path})
            return 0
    except _StateError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return 2
    except json.JSONDecodeError as exc:
        _emit(_failure("USAGE", f"invalid JSON argument: {exc}"))
        return 2

    return 2


if __name__ == "__main__":
    sys.exit(_cli())
