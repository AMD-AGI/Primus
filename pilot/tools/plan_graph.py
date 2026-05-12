"""PlanGraph engine: tree-shaped solution-space maintenance.

Implements the contract in ``pilot/skills/workflow/plan_graph.md``. This module
is **pure**: every public function returns a fresh dict (deepcopy semantics)
and never mutates its inputs in place. The single I/O surface is
``persist(graph, path)`` / ``load(path)`` for atomic disk writes.

Three sets of public operations:

1. **Construction / mutation** (``new``, ``add_node``, ``record_result``,
   ``shelve``, ``promote``, ``mark_exhausted``, ``mark_dead``).
2. **Read-only queries** (``frontier_excluding_dead``, ``subtree_dead_rate``,
   ``is_exhausted``, ``champion_node``).
3. **Counter maintenance** (``bump_promotion_counter`` /
   ``reset_promotion_counter`` / ``bump_explore_counter`` /
   ``reset_explore_counter``).

The escape mechanisms (Backtrack / Diversification bonus / Periodic
Exploration) are realized as helpers consumed by Re-Plan and Settle:
``should_backtrack``, ``pick_backtrack_target``, ``should_explore_round``,
``novelty_bonus_for``. They depend on the counters maintained above.

All numeric thresholds are sourced from ``skills/workflow/settle.md`` and
``skills/workflow/plan_graph.md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Constants (mirror of the Skill defaults — kept in lockstep with
# skills/workflow/settle.md \u00a74 and skills/workflow/plan_graph.md)
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = "1.0"

# Source: skills/workflow/settle.md \u00a74.
DEFAULT_DEAD_RATE_BACKTRACK: float = 0.50
DEFAULT_DEAD_RATE_WINDOW_ROUNDS: int = 2
DEFAULT_EXPLORE_PERIOD_K: int = 3

# Source: skills/workflow/replan.md \u00a73.
DEFAULT_NOVELTY_BONUS: float = 1.20
DEFAULT_STABILITY_BONUS: float = 1.10

# Source: skills/workflow/plan_graph.md \u00a75.
_WEAKLY_LOCAL_NUMERIC_RADIUS_PCT: float = 0.25  # \u00b125%


class PlanGraphError(Exception):
    """Raised on invalid PlanGraph mutations (unknown node, root re-shelve, etc.)."""


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def new(
    *,
    session_id: str,
    root_id: str,
    root_tps: float | None = None,
    root_bottleneck: str | None = None,
    measurement_ref: str | None = None,
    explore_period_K: int = DEFAULT_EXPLORE_PERIOD_K,
) -> dict[str, Any]:
    """Create a fresh PlanGraph with a single root node (the BASELINE).

    The root is marked ``completed`` and elected champion immediately;
    ``champion_history[0] = {round: 0, id: root_id}``.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": _SCHEMA_VERSION,
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "champion": root_id,
        "champion_history": [{"round": 0, "id": root_id, "at": now}],
        "nodes": {
            root_id: {
                "parent": None,
                "status": "completed",
                "tps": root_tps,
                "bottleneck": root_bottleneck,
                "champion_at": [0],
                "derived_axis": None,
                "reason": "BASELINE root",
                "measurement_ref": measurement_ref,
                "round_id": 0,
            }
        },
        "frontier": [root_id],
        "exhausted_neighborhoods": [],
        "metadata": {
            "rounds_since_promotion": 0,
            "rounds_since_explore": 0,
            "dead_rate_in_subtree": {},
            "explore_period_K": int(explore_period_K),
            "explore_rounds_completed": 0,
        },
    }


def add_node(
    graph: dict[str, Any],
    *,
    plan_id: str,
    parent: str,
    derived_axis: dict[str, Any] | None,
    round_id: int,
    reason: str = "",
    measurement_ref: str | None = None,
) -> dict[str, Any]:
    """Append a new ``running`` node; raises if id already exists or parent unknown."""
    g = copy.deepcopy(graph)
    if plan_id in g["nodes"]:
        raise PlanGraphError(f"plan_id already in graph: {plan_id}")
    if parent not in g["nodes"]:
        raise PlanGraphError(f"unknown parent plan_id: {parent}")
    g["nodes"][plan_id] = {
        "parent": parent,
        "status": "running",
        "tps": None,
        "bottleneck": None,
        "champion_at": [],
        "derived_axis": derived_axis,
        "reason": reason,
        "measurement_ref": measurement_ref,
        "round_id": int(round_id),
    }
    g["updated_at"] = _now()
    return g


def record_result(
    graph: dict[str, Any],
    *,
    plan_id: str,
    status: str,
    tps: float | None = None,
    bottleneck: str | None = None,
    reason: str = "",
    measurement_ref: str | None = None,
) -> dict[str, Any]:
    """Transition a node from ``running`` to ``completed`` or ``dead``.

    Adding to ``frontier`` is handled inside ``_recompute_frontier``.
    """
    if status not in ("completed", "dead"):
        raise PlanGraphError(f"record_result status must be completed|dead, got {status!r}")
    g = copy.deepcopy(graph)
    node = _require_node(g, plan_id)
    node["status"] = status
    if tps is not None:
        node["tps"] = float(tps)
    if bottleneck is not None:
        node["bottleneck"] = bottleneck
    if reason:
        node["reason"] = reason
    if measurement_ref is not None:
        node["measurement_ref"] = measurement_ref
    _recompute_frontier(g)
    _recompute_dead_rates(g)
    g["updated_at"] = _now()
    return g


def shelve(graph: dict[str, Any], *, plan_id: str, reason: str = "") -> dict[str, Any]:
    """Transition a ``completed`` node to ``shelved`` (e.g. a marginal-gain runner-up)."""
    g = copy.deepcopy(graph)
    node = _require_node(g, plan_id)
    if node["status"] != "completed":
        raise PlanGraphError(
            f"shelve requires status=completed, got {node['status']!r} for {plan_id}"
        )
    node["status"] = "shelved"
    if reason:
        node["reason"] = reason
    _recompute_frontier(g)
    g["updated_at"] = _now()
    return g


def promote(graph: dict[str, Any], *, plan_id: str, round_id: int) -> dict[str, Any]:
    """Make ``plan_id`` the new champion; old champion is demoted to ``shelved``.

    Also resets ``rounds_since_promotion`` to 0 and appends to ``champion_history``.
    """
    g = copy.deepcopy(graph)
    if plan_id not in g["nodes"]:
        raise PlanGraphError(f"unknown plan_id for promote: {plan_id}")
    new_champ_node = g["nodes"][plan_id]
    if new_champ_node["status"] not in ("completed", "shelved"):
        raise PlanGraphError(
            f"promote requires status in (completed, shelved), got {new_champ_node['status']!r}"
        )
    old_champ = g["champion"]
    if old_champ and old_champ != plan_id:
        old_node = g["nodes"].get(old_champ)
        if old_node is not None and old_node["status"] == "completed":
            old_node["status"] = "shelved"
            old_node["reason"] = old_node.get("reason") or f"demoted at r{round_id}"
    # Promotion implies the node is "completed" going forward, even if it was shelved.
    new_champ_node["status"] = "completed"
    cap = list(new_champ_node.get("champion_at") or [])
    if round_id not in cap:
        cap.append(int(round_id))
    new_champ_node["champion_at"] = cap
    g["champion"] = plan_id
    g["champion_history"].append({"round": int(round_id), "id": plan_id, "at": _now()})
    g["metadata"]["rounds_since_promotion"] = 0
    _recompute_frontier(g)
    _recompute_dead_rates(g)
    g["updated_at"] = _now()
    return g


def mark_exhausted(
    graph: dict[str, Any],
    *,
    around: str,
    axis: str,
    value: Any,
    axis_type: str | None = None,
) -> dict[str, Any]:
    """Append ``value`` to the matching ``exhausted_neighborhoods`` row.

    Creates the row when no existing match is found.
    """
    g = copy.deepcopy(graph)
    if around not in g["nodes"]:
        raise PlanGraphError(f"unknown around plan_id: {around}")
    for row in g["exhausted_neighborhoods"]:
        if row["around"] == around and row["axis"] == axis:
            if value not in row["tried"]:
                row["tried"].append(value)
            if axis_type and not row.get("axis_type"):
                row["axis_type"] = axis_type
            g["updated_at"] = _now()
            return g
    g["exhausted_neighborhoods"].append({
        "around": around,
        "axis": axis,
        "axis_type": axis_type,
        "tried": [value],
    })
    g["updated_at"] = _now()
    return g


# ---------------------------------------------------------------------------
# Counters (per skills/workflow/settle.md \u00a75)
# ---------------------------------------------------------------------------


def bump_promotion_counter(graph: dict[str, Any]) -> dict[str, Any]:
    g = copy.deepcopy(graph)
    g["metadata"]["rounds_since_promotion"] = int(g["metadata"].get("rounds_since_promotion", 0)) + 1
    g["updated_at"] = _now()
    return g


def reset_promotion_counter(graph: dict[str, Any]) -> dict[str, Any]:
    g = copy.deepcopy(graph)
    g["metadata"]["rounds_since_promotion"] = 0
    g["updated_at"] = _now()
    return g


def bump_explore_counter(graph: dict[str, Any]) -> dict[str, Any]:
    g = copy.deepcopy(graph)
    g["metadata"]["rounds_since_explore"] = int(g["metadata"].get("rounds_since_explore", 0)) + 1
    g["updated_at"] = _now()
    return g


def reset_explore_counter(graph: dict[str, Any]) -> dict[str, Any]:
    """Reset `rounds_since_explore` to 0 and increment the lifetime
    explore-rounds counter (used by Settle's stop-deferral logic)."""
    g = copy.deepcopy(graph)
    g["metadata"]["rounds_since_explore"] = 0
    g["metadata"]["explore_rounds_completed"] = int(
        g["metadata"].get("explore_rounds_completed", 0)
    ) + 1
    g["updated_at"] = _now()
    return g


# ---------------------------------------------------------------------------
# Read-only queries
# ---------------------------------------------------------------------------


def champion_node(graph: dict[str, Any]) -> dict[str, Any]:
    return graph["nodes"][graph["champion"]]


def frontier_excluding_dead(graph: dict[str, Any]) -> list[str]:
    """Return the current frontier (champion + shelved), sorted by tps desc.

    Always re-derived from ``nodes`` to stay consistent with statuses.
    """
    items: list[tuple[float, str]] = []
    for nid, n in graph["nodes"].items():
        if n["status"] in ("shelved",) or nid == graph["champion"]:
            tps = n.get("tps")
            score = float(tps) if isinstance(tps, (int, float)) else float("-inf")
            items.append((score, nid))
    items.sort(key=lambda x: (-x[0], x[1]))
    return [nid for _, nid in items]


def subtree_dead_rate(graph: dict[str, Any], plan_id: str) -> float:
    """Fraction of `plan_id`'s direct + transitive descendants that are ``dead``.

    Returns 0.0 when there are no descendants. The plan itself is not counted.
    """
    if plan_id not in graph["nodes"]:
        return 0.0
    descendants = _descendants_of(graph, plan_id)
    if not descendants:
        return 0.0
    dead = sum(1 for d in descendants if graph["nodes"][d]["status"] == "dead")
    return dead / len(descendants)


def is_exhausted(
    graph: dict[str, Any],
    *,
    around: str,
    axis: str,
    value: Any,
    axis_type: str | None = None,
) -> bool:
    """Apply the radius rules from ``skills/workflow/plan_graph.md`` \u00a75."""
    for row in graph["exhausted_neighborhoods"]:
        if row["around"] != around or row["axis"] != axis:
            continue
        tried = row.get("tried") or []
        atype = axis_type or row.get("axis_type")
        if atype == "weakly_local" and isinstance(value, (int, float)) and not isinstance(value, bool):
            for prior in tried:
                if isinstance(prior, (int, float)) and not isinstance(prior, bool) and prior != 0:
                    if abs(float(value) - float(prior)) / abs(float(prior)) <= _WEAKLY_LOCAL_NUMERIC_RADIUS_PCT:
                        return True
            # boolean/enum weakly_local still falls through to exact match below
        if value in tried:
            return True
    return False


# ---------------------------------------------------------------------------
# Escape mechanism helpers (skills/workflow/settle.md \u00a75)
# ---------------------------------------------------------------------------


def should_backtrack(
    graph: dict[str, Any],
    *,
    dead_rate_threshold: float = DEFAULT_DEAD_RATE_BACKTRACK,
    window_rounds: int = DEFAULT_DEAD_RATE_WINDOW_ROUNDS,
) -> bool:
    """True iff Backtrack should fire (per settle.md \u00a75).

    Backtrack triggers when the current champion's subtree has had a
    dead-rate above ``dead_rate_threshold`` for ``window_rounds`` consecutive
    rounds. The caller is responsible for honoring the window: this helper
    only checks the current dead-rate. ``window_rounds`` is documented so the
    caller can refuse to fire Backtrack until it has observed the high
    dead-rate for the required number of rounds (the engine tracks this via
    ``metadata.rounds_since_promotion`` + ``dead_rate_in_subtree`` history,
    which the caller maintains).
    """
    champ = graph.get("champion")
    if champ is None:
        return False
    rate = float(graph["metadata"].get("dead_rate_in_subtree", {}).get(champ, 0.0))
    # window_rounds is intentionally honored by the caller (Settle); we
    # surface the current-round rate so the caller can roll its own window.
    _ = window_rounds
    return rate > dead_rate_threshold


def pick_backtrack_target(graph: dict[str, Any]) -> str | None:
    """Pick the highest-tps shelved node that is NOT the current champion."""
    frontier = frontier_excluding_dead(graph)
    for nid in frontier:
        if nid == graph["champion"]:
            continue
        if graph["nodes"][nid]["status"] != "shelved":
            continue
        return nid
    return None


def should_explore_round(
    graph: dict[str, Any], *, explore_period_K: int | None = None
) -> bool:
    """True iff the next Re-Plan should be a Periodic Exploration round."""
    K = int(explore_period_K or graph["metadata"].get("explore_period_K", DEFAULT_EXPLORE_PERIOD_K))
    return int(graph["metadata"].get("rounds_since_explore", 0)) >= K


def novelty_bonus_for(
    graph: dict[str, Any], *, parent: str, axis: str
) -> float:
    """Return DEFAULT_NOVELTY_BONUS if `axis` has not yet been used as
    ``derived_axis.axis`` under any direct child of ``parent``; else 1.0.
    """
    seen: set[str] = set()
    for n in graph["nodes"].values():
        if n.get("parent") == parent:
            da = n.get("derived_axis") or {}
            a = da.get("axis")
            if isinstance(a, str):
                seen.add(a)
            for entry in da.get("composite") or []:
                ca = entry.get("axis")
                if isinstance(ca, str):
                    seen.add(ca)
    return DEFAULT_NOVELTY_BONUS if axis not in seen else 1.0


def stability_bonus_for(graph: dict[str, Any], *, parent: str) -> float:
    """Return DEFAULT_STABILITY_BONUS if `parent` has ≥2 entries in
    ``champion_at`` (i.e. has been champion for at least two rounds)."""
    node = graph["nodes"].get(parent)
    if not node:
        return 1.0
    return DEFAULT_STABILITY_BONUS if len(node.get("champion_at") or []) >= 2 else 1.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist(graph: dict[str, Any], path: str | Path) -> str:
    """Atomically write the graph as YAML to ``path``."""
    import yaml  # local import: pilot core declares yaml as optional dep elsewhere
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    g = copy.deepcopy(graph)
    g["updated_at"] = _now()
    with tmp.open("w") as f:
        yaml.safe_dump(g, f, sort_keys=False, default_flow_style=False)
    tmp.replace(p)
    return str(p)


def load(path: str | Path) -> dict[str, Any]:
    """Read a graph back from disk."""
    import yaml
    return yaml.safe_load(Path(path).read_text())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_node(graph: dict[str, Any], plan_id: str) -> dict[str, Any]:
    n = graph["nodes"].get(plan_id)
    if n is None:
        raise PlanGraphError(f"unknown plan_id: {plan_id}")
    return n


def _recompute_frontier(graph: dict[str, Any]) -> None:
    """Re-derive ``frontier`` from node statuses (called inside mutations)."""
    graph["frontier"] = frontier_excluding_dead(graph)


def _children_of(graph: dict[str, Any], parent: str) -> list[str]:
    return [nid for nid, n in graph["nodes"].items() if n.get("parent") == parent]


def _descendants_of(graph: dict[str, Any], plan_id: str) -> list[str]:
    out: list[str] = []
    stack: list[str] = list(_children_of(graph, plan_id))
    while stack:
        nid = stack.pop()
        out.append(nid)
        stack.extend(_children_of(graph, nid))
    return out


def _recompute_dead_rates(graph: dict[str, Any]) -> None:
    rates: dict[str, float] = {}
    for nid in graph["nodes"]:
        rates[nid] = subtree_dead_rate(graph, nid)
    graph["metadata"]["dead_rate_in_subtree"] = rates


# ---------------------------------------------------------------------------
# CLI (lightweight inspection helpers; mutations live behind tune_single)
# ---------------------------------------------------------------------------


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.plan_graph")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show")
    p_show.add_argument("path")

    p_frontier = sub.add_parser("frontier")
    p_frontier.add_argument("path")

    p_dead = sub.add_parser("dead-rate")
    p_dead.add_argument("path")
    p_dead.add_argument("--plan-id", required=True)

    p_exh = sub.add_parser("is-exhausted")
    p_exh.add_argument("path")
    p_exh.add_argument("--around", required=True)
    p_exh.add_argument("--axis", required=True)
    p_exh.add_argument("--value", required=True)
    p_exh.add_argument("--axis-type", default=None)

    args = p.parse_args()
    if args.cmd == "show":
        _emit(load(args.path))
        return 0
    if args.cmd == "frontier":
        _emit({"frontier": frontier_excluding_dead(load(args.path))})
        return 0
    if args.cmd == "dead-rate":
        _emit({"plan_id": args.plan_id, "rate": subtree_dead_rate(load(args.path), args.plan_id)})
        return 0
    if args.cmd == "is-exhausted":
        raw = args.value
        try:
            parsed_value: Any = json.loads(raw)
        except json.JSONDecodeError:
            parsed_value = raw
        _emit({
            "around": args.around,
            "axis": args.axis,
            "value": parsed_value,
            "axis_type": args.axis_type,
            "is_exhausted": is_exhausted(
                load(args.path),
                around=args.around,
                axis=args.axis,
                value=parsed_value,
                axis_type=args.axis_type,
            ),
        })
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
