"""Single-node end-to-end tuning runner.

This module is the first complete Pilot loop for ``cluster.yaml`` mode
``single``. It intentionally stays narrow: Primus exp YAML in, override
candidates out, short foreground training runs, RunSnapshot-based diagnosis,
champion selection, and a final TuningState checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.tools import constraint, observe, plan_graph as pg, report, state, submit
from pilot.tools import _axis_translator as _axt

# `pilot.tools.diagnose` v2 is trace-driven; tune_single passes the
# already-loaded trace_analysis dict when available (caller is responsible
# for calling pilot.tools.trace_analyze first). Lazy import so a missing
# trace_analyze install does not block the rest of tune_single.
try:
    from pilot.tools import diagnose as _diagnose_engine  # type: ignore
except Exception:  # pragma: no cover
    _diagnose_engine = None
from pilot.tools._cluster_config import ClusterConfigError, load_cluster_config, preflight_check


# ---------------------------------------------------------------------------
# Canonical Skill-sourced constants
# ---------------------------------------------------------------------------
# All numeric thresholds below are sourced from the matching Skill files and
# MUST be kept in sync with them (the Skill is the single source of truth;
# this module is a faithful mirror). Each constant's docstring names the
# authoritative Skill section.
# ---------------------------------------------------------------------------

# Cost proxy per axis type. Source: skills/workflow/replan.md §3.1
# (cross-referenced by skills/workflow/axis_taxonomy.md §1).
# `weakly_local`: cheapest, no structural impact, no env-baseline change.
# `strongly_local`: still per-run but bigger blast radius (e.g. recompute).
# `structural`: invalidates memory predictions and may change `world_size`
#   composition; constraint.check must rerun.
# `cluster_shared`: changes the env baseline; only Champion-Challenger.
_AXIS_COST_PROXY: dict[str, float] = {
    "weakly_local": 1.0,
    "strongly_local": 1.2,
    "structural": 2.0,
    "cluster_shared": 3.0,
}

# Re-Plan priority-formula extra factors. Source: skills/workflow/replan.md §3.
_CONFIDENCE_FLOOR: float = 0.05         # priority(c) uses max(_CONFIDENCE_FLOOR, confidence)
_DEFAULT_NOVELTY_BONUS: float = 1.20     # axis not yet seen as a sibling under same parent
_DEFAULT_STABILITY_BONUS: float = 1.10   # parent has ≥2 entries in champion_history

# Special-purpose candidate priorities. Source: skills/workflow/replan.md §3.2.
_DEFAULT_ENV_SUSPECT_PRIORITY: float = 0.6   # env_suspect not covered by an axis: 0.6 × confidence
_DEFAULT_CONTROL_PRIORITY: float = 0.05      # engine-path control rerun (noise check)
_DEFAULT_LEGACY_CONTROL_PRIORITY: float = 0.50  # legacy-path control rerun; see replan.md §6.

# Settle thresholds. Source: skills/workflow/settle.md §4.
_DEFAULT_EPSILON_PROMOTE: float = 0.02   # 2%: min relative gain to promote
_DEFAULT_EPSILON_STOP: float = 0.005     # 0.5%: per-round gain that counts as "stagnant"
_DEFAULT_STAGNATION_ROUNDS: int = 2      # consecutive stagnant rounds → STOP
_DEFAULT_DEAD_RATE_BACKTRACK: float = 0.50      # subtree dead-rate that triggers Backtrack
_DEFAULT_DEAD_RATE_WINDOW_ROUNDS: int = 2       # for how many rounds the dead-rate must hold
_DEFAULT_EXPLORE_PERIOD_K: int = 3        # force one Explore round every K rounds

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent
_REPO_ROOT: Path = _PILOT_ROOT.parent


class _TuneError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise _TuneError("DEP_MISSING", f"PyYAML required for tune_single: {exc}") from exc
    return yaml


def _resolve_pilot_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else _PILOT_ROOT / p


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.exists():
        # State artifact refs are often pilot-relative.
        p2 = _resolve_pilot_path(path)
        if p2.exists():
            p = p2
    if not p.exists():
        raise _TuneError("USAGE", f"file not found: {path}")
    data = _yaml().safe_load(p.read_text())
    if not isinstance(data, dict):
        raise _TuneError("USAGE", f"{p} must contain a YAML/JSON mapping")
    return data


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    yaml = _yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


def _parse_override(spec: str) -> tuple[str, Any]:
    return submit._parse_override(spec)  # noqa: SLF001 - shared CLI semantics.


def _base_overrides(plan: dict[str, Any]) -> dict[str, Any]:
    return (
        plan.get("modules", {})
        .get("pre_trainer", {})
        .get("overrides", {})
        or {}
    )


def _latest_cluster_profile(root: str) -> dict[str, Any] | None:
    """Return the freshest ClusterProfile YAML under ``root/cluster_profiles``.

    Returns None if none exists. Best-effort: only used to enrich diagnose calls.
    """
    profiles_dir = _resolve_pilot_path(root) / "cluster_profiles"
    if not profiles_dir.exists():
        return None
    candidates = sorted(profiles_dir.rglob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            data = _load_yaml(p)
            if isinstance(data, dict) and "compute" in data:
                return data
        except Exception:
            continue
    return None


def _merged_plan(base_plan: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    plan = deepcopy(base_plan)
    target = plan.setdefault("modules", {}).setdefault("pre_trainer", {}).setdefault("overrides", {})
    target.update(overrides)
    return plan


def _finite_values(values: list[Any]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out.append(float(v))
    return out


def summarize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Extract stable metrics from a RunSnapshot."""
    metrics = snapshot.get("metrics") or {}
    history = metrics.get("history") or {}
    latest = metrics.get("latest") or {}
    iter_times = _finite_values(history.get("iter_time_ms") or [])
    tflops = _finite_values(history.get("tflops") or [])
    losses = _finite_values(history.get("loss") or [])
    median_iter = statistics.median(iter_times) if iter_times else latest.get("iter_time_ms")
    median_tflops = statistics.median(tflops) if tflops else latest.get("tflops")
    final_loss = losses[-1] if losses else latest.get("loss")
    throughput = (1000.0 / float(median_iter)) if median_iter else None
    return {
        "status": snapshot.get("status"),
        "median_iter_time_ms": round(float(median_iter), 4) if median_iter else None,
        "median_tflops": round(float(median_tflops), 4) if median_tflops else None,
        "throughput_steps_per_s": round(throughput, 6) if throughput else None,
        "final_loss": round(float(final_loss), 8) if isinstance(final_loss, (int, float)) and math.isfinite(float(final_loss)) else None,
        "loss_finite": bool(metrics.get("loss_finite", True)),
        "progress": snapshot.get("progress") or {},
        "symptoms": snapshot.get("symptoms") or {},
    }


def score_measurement(measurement: dict[str, Any]) -> float | None:
    if measurement.get("status") not in ("completed", "success", "pass"):
        return None
    if measurement.get("loss_finite") is False:
        return None
    if measurement.get("median_iter_time_ms"):
        return 1000.0 / float(measurement["median_iter_time_ms"])
    if measurement.get("throughput_steps_per_s"):
        return float(measurement["throughput_steps_per_s"])
    return None


def diagnose(
    snapshot: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    cluster_profile: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    champion_snapshot: dict[str, Any] | None = None,
    profile_path: str | None = None,
    trace_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a single-node RunSnapshot for candidate generation.

    Two paths:
    1. **Engine path (preferred, v2)**: when ``trace_analysis`` is supplied
       (a loaded `trace_analysis.json` per `skills/workflow/trace_analysis.md`),
       delegates to `pilot.tools.diagnose.run`, the trace-driven rule
       engine in `skills/workflow/diagnose.md`. Returns the legacy
       back-compat dict (``bottleneck``/``candidate_axes``/``measurement``)
       enriched with the full DiagnosisReport under ``meta.engine_report``.
    2. **Legacy fallback**: pre-engine heuristic for callers that don't
       have a trace yet. Defaults to COMPUTE when in doubt.
    """
    summary = summarize_snapshot(snapshot)

    if trace_analysis is not None and _diagnose_engine is not None:
        try:
            engine_report = _diagnose_engine.run(
                trace_analysis=trace_analysis,
                snapshot=snapshot,
                plan=plan,
                cluster_profile=cluster_profile,
                champion_snapshot=champion_snapshot,
            )
        except Exception:
            engine_report = None
        if engine_report is not None:
            return _bridge_engine_to_legacy(engine_report, summary, snapshot)

    return _legacy_diagnose(snapshot, summary, baseline)


def _legacy_diagnose(
    snapshot: dict[str, Any],
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    failure = constraint.diagnose_failure(snapshot)
    symptoms = summary["symptoms"]
    bottleneck = "UNKNOWN"
    axes: list[str] = []

    if failure["kind"] == "OOM":
        bottleneck = "MEMORY"
        axes = ["micro_batch_size", "recompute"]
    elif failure["kind"] in ("HANG", "TOOL_ERROR", "NUMERICAL"):
        bottleneck = failure["kind"]
        axes = ["stability"]
    elif summary.get("status") in ("failed", "killed", "hung", "unknown"):
        bottleneck = "RUNTIME"
        axes = ["stability"]
        failure = {
            **failure,
            "kind": failure.get("kind") or "UNKNOWN",
            "message": failure.get("message") or f"run status={summary.get('status')}",
        }
    else:
        current_iter = summary.get("median_iter_time_ms")
        baseline_iter = (baseline or {}).get("median_iter_time_ms")
        current_tflops = summary.get("median_tflops")
        baseline_tflops = (baseline or {}).get("median_tflops")
        if current_iter and baseline_iter and current_iter > baseline_iter * 1.05:
            bottleneck = "REGRESSION"
            axes = ["micro_batch_size", "recompute"]
        elif current_tflops and baseline_tflops and current_tflops < baseline_tflops * 0.9:
            bottleneck = "COMPUTE"
            axes = ["micro_batch_size", "tensor_model_parallel_size"]
        elif symptoms.get("nccl_error"):
            bottleneck = "COMM"
            axes = ["tensor_model_parallel_size"]
        else:
            bottleneck = "COMPUTE"
            axes = ["micro_batch_size", "tensor_model_parallel_size"]

    return {
        "status": "success",
        "bottleneck": bottleneck,
        "candidate_axes": axes,
        "measurement": summary,
        "failure": failure if failure["kind"] != "UNKNOWN" or summary.get("status") in ("failed", "killed", "hung", "unknown") else None,
        "source": "legacy",
    }


_ENGINE_TO_LEGACY_BOTTLENECK = {
    "COMM_BOUND": "COMM",
    "PIPELINE_BOUND": "PIPELINE",
    "MEMORY_BOUND": "MEMORY",
    "COMPUTE_BOUND": "COMPUTE",
}


def _bridge_engine_to_legacy(
    report: dict[str, Any],
    summary: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    meta = report.get("meta") or {}
    extended = meta.get("bottleneck_extended", report["bottleneck"])
    legacy_bottleneck = _ENGINE_TO_LEGACY_BOTTLENECK.get(report["bottleneck"], "COMPUTE")
    if extended in ("HANG", "NUMERICAL", "INVALID_CONFIG", "CANCELLED", "REGRESSION", "UNKNOWN"):
        legacy_bottleneck = extended

    legacy_axes: list[str] = []
    seen: set[str] = set()
    for entry in report.get("candidate_axes") or []:
        axis = entry.get("axis")
        if isinstance(axis, str) and axis not in seen:
            legacy_axes.append(axis)
            seen.add(axis)
    if not legacy_axes:
        legacy_axes = ["stability"] if extended in ("HANG", "NUMERICAL", "INVALID_CONFIG", "CANCELLED") else ["micro_batch_size", "tensor_model_parallel_size"]

    failure = constraint.diagnose_failure(snapshot)
    if failure.get("kind") in (None, "UNKNOWN") and extended in ("HANG", "NUMERICAL", "INVALID_CONFIG", "CANCELLED"):
        failure = {
            "kind": extended,
            "message": (report.get("evidence") or [extended])[0],
            "suggested_transition": (report.get("suggested_transition") or {}).get("to") or "OPTIMIZE_LOOP.REPLAN",
        }

    return {
        "status": "success",
        "bottleneck": legacy_bottleneck,
        "candidate_axes": legacy_axes,
        "measurement": summary,
        "failure": failure if failure.get("kind") not in (None, "UNKNOWN") or summary.get("status") in ("failed", "killed", "hung", "unknown") else None,
        "source": "engine",
        "meta": {"engine_report": report},
    }


def _candidate_id(round_id: int, idx: int, overrides: dict[str, Any]) -> str:
    parts = [f"{k}={overrides[k]}" for k in sorted(overrides)]
    import hashlib

    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]
    return f"r{round_id}_c{idx}_{digest}"


def _dedupe_candidates(candidates: list[dict[str, Any]], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    for item in history:
        ovs = item.get("overrides") or {}
        seen.add(json.dumps(ovs, sort_keys=True, default=str))
    out: list[dict[str, Any]] = []
    local: set[str] = set()
    for cand in candidates:
        key = json.dumps(cand["overrides"], sort_keys=True, default=str)
        if key in seen or key in local:
            continue
        local.add(key)
        out.append(cand)
    return out


def _engine_report(diagnosis: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the trace-driven engine report out of a diagnosis dict, if any."""
    meta = diagnosis.get("meta") or {}
    er = meta.get("engine_report")
    if isinstance(er, dict) and er.get("schema_version") and er.get("candidate_axes") is not None:
        return er
    return None


def _gain_mid(axis_entry: dict[str, Any]) -> float:
    band = axis_entry.get("expected_gain_band_pct") or [0, 0]
    try:
        lo, hi = float(band[0]), float(band[1])
    except (TypeError, ValueError, IndexError):
        return 0.0
    return (lo + hi) / 2.0 / 100.0


def _axis_priority(axis_entry: dict[str, Any], confidence: float) -> float:
    """priority = gain_mid * max(_CONFIDENCE_FLOOR, confidence) / cost_proxy.

    Source: skills/workflow/replan.md §3 (the canonical formula). novelty_bonus
    and stability_bonus are NOT applied here because they depend on PlanGraph
    state; callers compose them on top of this scalar.
    """
    gain = _gain_mid(axis_entry)
    cost = _AXIS_COST_PROXY.get(axis_entry.get("type") or "weakly_local", 1.0)
    return gain * max(_CONFIDENCE_FLOOR, confidence) / cost


def _engine_replan(
    *,
    base_plan: dict[str, Any],
    cluster: dict[str, Any],
    diagnosis: dict[str, Any],
    engine_report: dict[str, Any],
    round_id: int,
    history: list[dict[str, Any]],
    max_candidates: int,
    train_iters: int,
    plan_graph: dict[str, Any] | None = None,
    parent_plan_id: str | None = None,
) -> dict[str, Any]:
    """Generate candidates from a trace-driven DiagnosisReport.

    For each engine-emitted axis x value pair, asks ``_axis_translator`` what
    the change actually IS (trainer YAML override / structural / env var),
    composes a candidate, validates structural changes through
    ``constraint.check``, and ranks by gain × confidence ÷ cost_proxy.

    Env-only ``env_suspect`` flags that aren't already covered by an axis
    are emitted as their own (env-only) candidate so Champion-Challenger can
    isolate the env effect.

    When ``plan_graph`` and ``parent_plan_id`` are provided, the engine also:
      - rejects candidates that hit ``exhausted_neighborhoods`` (per
        skills/workflow/plan_graph.md §5 radius rules);
      - applies the novelty bonus (1.20) for axes never used as a sibling
        under ``parent_plan_id`` (per replan.md §3);
      - applies the stability bonus (1.10) when ``parent_plan_id`` has been
        champion for ≥2 rounds (per replan.md §3).
    """
    base_overrides = deepcopy(_base_overrides(base_plan))
    confidence = float(engine_report.get("confidence") or 0.5)
    axis_entries = list(engine_report.get("candidate_axes") or [])
    env_suspects = list(engine_report.get("env_suspect") or [])
    axis_names_emitted = {e.get("axis") for e in axis_entries}
    skipped_unknown_axes: list[str] = []
    rejected_audit: list[dict[str, Any]] = []  # for CandidatePool.selection.rejected

    use_pg = plan_graph is not None and parent_plan_id is not None
    stability = pg.stability_bonus_for(plan_graph, parent=parent_plan_id) if use_pg else 1.0  # type: ignore[arg-type]

    raw: list[tuple[float, str, dict[str, Any], dict[str, str], dict[str, Any]]] = []

    def add(
        *,
        priority: float,
        reason: str,
        plan_overrides: dict[str, Any],
        env_overrides: dict[str, str],
        meta: dict[str, Any],
    ) -> None:
        # PlanGraph-aware: drop candidates whose (axis, value) already lives
        # in exhausted_neighborhoods around the same parent.
        if use_pg and meta.get("axis") not in (None, "_control"):
            if pg.is_exhausted(
                plan_graph,  # type: ignore[arg-type]
                around=parent_plan_id,  # type: ignore[arg-type]
                axis=str(meta["axis"]),
                value=meta.get("value"),
                axis_type=meta.get("type"),
            ):
                rejected_audit.append({
                    "axis": meta.get("axis"),
                    "value": meta.get("value"),
                    "reason": "exhausted_neighborhoods hit",
                })
                return

        # Novelty bonus: axis not yet seen as a sibling under the same parent.
        novelty = (
            pg.novelty_bonus_for(plan_graph, parent=parent_plan_id, axis=str(meta.get("axis")))  # type: ignore[arg-type]
            if use_pg and meta.get("axis") not in (None, "_control")
            else 1.0
        )
        bonused_priority = priority * novelty * stability

        merged_plan_overrides = {**base_overrides, **plan_overrides, "train_iters": train_iters}
        plan = _merged_plan(base_plan, merged_plan_overrides)
        verdict = constraint.check(plan, cluster)
        if not verdict["valid"]:
            skipped_unknown_axes.append(
                f"{meta.get('axis', 'composite')}: constraint violations {verdict['violations']}"
            )
            rejected_audit.append({
                "axis": meta.get("axis"),
                "value": meta.get("value"),
                "reason": f"constraint: {verdict['violations']}",
            })
            return
        # Stash bonuses in meta for audit / downstream inspection.
        meta = {
            **meta,
            "novelty_bonus": novelty,
            "stability_bonus": stability,
            "priority_pre_bonus": round(priority, 6),
        }
        raw.append((bonused_priority, reason, merged_plan_overrides, env_overrides, meta))

    for entry in axis_entries:
        axis = entry.get("axis")
        if not axis:
            continue
        priority = _axis_priority(entry, confidence)
        rationale = entry.get("rationale") or ""
        for value in entry.get("candidates") or []:
            action = _axt.translate(axis, value)
            if action is None:
                skipped_unknown_axes.append(
                    f"{axis}={value!r}: not in pilot.tools._axis_translator catalog"
                )
                continue
            plan_ov: dict[str, Any] = {}
            env_ov: dict[str, str] = {}
            if action.channel == "env":
                env_ov[action.key] = action.rendered_value
            else:  # trainer_override or structural
                plan_ov[action.key] = action.rendered_value
            add(
                priority=priority,
                reason=f"{axis}={value} :: {rationale}",
                plan_overrides=plan_ov,
                env_overrides=env_ov,
                meta={
                    "axis": axis,
                    "value": value,
                    "type": entry.get("type"),
                    "channel": action.channel,
                    "expected_gain_band_pct": entry.get("expected_gain_band_pct"),
                    "rationale": rationale,
                    "source": "engine.candidate_axes",
                },
            )

    for sus in env_suspects:
        flag = sus.get("flag")
        if not flag or flag in axis_names_emitted:
            continue
        action = _axt.translate(flag, True)
        if action is None or action.channel != "env":
            skipped_unknown_axes.append(
                f"env_suspect {flag}: not an env-channel axis in translator catalog"
            )
            continue
        add(
            priority=_DEFAULT_ENV_SUSPECT_PRIORITY * confidence,
            reason=f"env_suspect {flag}=true :: {sus.get('reason', '')}",
            plan_overrides={},
            env_overrides={action.key: action.rendered_value},
            meta={
                "axis": flag,
                "value": True,
                "type": "weakly_local",
                "channel": "env",
                "expected_gain_band_pct": [1, 5],
                "rationale": sus.get("reason"),
                "source": "engine.env_suspect",
            },
        )

    add(
        priority=_DEFAULT_CONTROL_PRIORITY,
        reason="control rerun with current structure (engine path baseline)",
        plan_overrides={},
        env_overrides={},
        meta={"axis": "_control", "value": None, "type": "control", "channel": "control",
              "source": "engine.control"},
    )

    candidates: list[dict[str, Any]] = []
    for idx, (priority, reason, plan_ov, env_ov, meta) in enumerate(
        sorted(raw, key=lambda x: x[0], reverse=True), 1
    ):
        cand: dict[str, Any] = {
            "id": _candidate_id(round_id, idx, {**plan_ov, **{f"env::{k}": v for k, v in env_ov.items()}}),
            "round_id": round_id,
            "priority": round(priority, 6),
            "reason": reason,
            "overrides": plan_ov,
            "env_overrides": env_ov,
            "axis_meta": meta,
        }
        candidates.append(cand)
    candidates = _dedupe_candidates(candidates, history)[:max_candidates]

    selection = {
        "strategy": "Per-Plan",        # single-node v1 default; see execution_strategy.md §6
        "pick_top_k": len(candidates),
        "selected": [c["id"] for c in candidates],
        "rejected": rejected_audit,
    }

    return {
        "schema_version": "1.0",
        "round_id": round_id,
        "status": "ready" if candidates else "empty",
        "diagnosis": diagnosis,
        "candidates": candidates,
        "selection": selection,
        "derived_from": {"primary": parent_plan_id, "secondary": []} if parent_plan_id else None,
        "source": "engine",
        "engine_meta": {
            "rule_id": (engine_report.get("meta") or {}).get("rule_id"),
            "bottleneck": engine_report.get("bottleneck"),
            "confidence": confidence,
            "skipped": skipped_unknown_axes,
        },
        "priority_formula": (
            "priority(c) = gain_mid × max(0.05, confidence) / cost_proxy × novelty_bonus × stability_bonus"
        ),
    }


def _legacy_replan(
    *,
    base_plan: dict[str, Any],
    cluster: dict[str, Any],
    diagnosis: dict[str, Any],
    round_id: int,
    history: list[dict[str, Any]],
    max_candidates: int,
    train_iters: int,
) -> dict[str, Any]:
    """Legacy axis-name replan path. Retained for callers that don't have a
    trace yet (e.g. SMOKE / BASELINE failure paths in tune_single).

    Priorities here come from `skills/workflow/replan.md` §6 (the legacy-path
    table). They are intentionally flat (not derived from the §3 formula)
    because the legacy path has no engine-emitted `expected_gain_band_pct`
    nor `confidence` to plug into the formula.
    """
    overrides = deepcopy(_base_overrides(base_plan))
    tp = constraint._to_int(overrides.get("tensor_model_parallel_size"), 1) or 1  # noqa: SLF001
    pp = constraint._to_int(overrides.get("pipeline_model_parallel_size"), 1) or 1  # noqa: SLF001
    ep = constraint._to_int(overrides.get("expert_model_parallel_size"), 1) or 1  # noqa: SLF001
    mbs = constraint._to_int(overrides.get("micro_batch_size"), 1) or 1  # noqa: SLF001
    gbs = constraint._to_int(overrides.get("global_batch_size"), max(1, mbs * 8)) or max(1, mbs * 8)  # noqa: SLF001
    _, _, world = constraint._cluster_world(cluster)  # noqa: SLF001
    axes = diagnosis.get("candidate_axes") or ["micro_batch_size"]

    raw: list[tuple[float, str, dict[str, Any]]] = []

    def add(priority: float, reason: str, delta: dict[str, Any]) -> None:
        merged = {**overrides, **delta, "train_iters": train_iters}
        merged.setdefault("pipeline_model_parallel_size", pp)
        merged.setdefault("expert_model_parallel_size", ep)
        plan = _merged_plan(base_plan, merged)
        verdict = constraint.check(plan, cluster)
        if verdict["valid"]:
            raw.append((priority, reason, merged))

    # Per skills/workflow/replan.md §6: hand-written priority rules.
    _LEGACY_PRIORITY_MBS_UP = 1.00
    _LEGACY_PRIORITY_MBS_DOWN = 0.95
    _LEGACY_PRIORITY_RECOMPUTE_FULL = 0.90
    _LEGACY_PRIORITY_TP_FLIP = 0.85

    if "micro_batch_size" in axes:
        add(_LEGACY_PRIORITY_MBS_UP, "probe higher micro_batch_size for better compute occupancy", {
            "micro_batch_size": mbs + 1,
            "global_batch_size": max(gbs, (mbs + 1) * max(1, world // max(1, tp * pp * ep))),
        })
        if mbs > 1:
            add(_LEGACY_PRIORITY_MBS_DOWN, "reduce micro_batch_size after memory or stability signal", {
                "micro_batch_size": max(1, mbs - 1),
                "global_batch_size": max(1, gbs),
            })
    if "tensor_model_parallel_size" in axes:
        for next_tp in (tp * 2, max(1, tp // 2)):
            if next_tp != tp and next_tp >= 1 and world % max(1, next_tp * pp * ep) == 0:
                add(_LEGACY_PRIORITY_TP_FLIP, f"probe tensor_model_parallel_size={next_tp}", {
                    "tensor_model_parallel_size": next_tp,
                })
    if "recompute" in axes or diagnosis.get("bottleneck") in ("MEMORY", "OOM"):
        add(_LEGACY_PRIORITY_RECOMPUTE_FULL, "enable full block recompute to relieve activation memory", {
            "recompute_granularity": "full",
            "recompute_method": "block",
            "recompute_num_layers": int(overrides.get("recompute_num_layers") or 1),
        })

    add(_DEFAULT_LEGACY_CONTROL_PRIORITY,
        "control rerun with current structure and fixed short train_iters",
        {"train_iters": train_iters})

    candidates: list[dict[str, Any]] = []
    for idx, (priority, reason, cand_overrides) in enumerate(sorted(raw, key=lambda x: x[0], reverse=True), 1):
        candidates.append({
            "id": _candidate_id(round_id, idx, cand_overrides),
            "round_id": round_id,
            "priority": priority,
            "reason": reason,
            "overrides": cand_overrides,
            "env_overrides": {},
            "axis_meta": {"axis": None, "type": "legacy", "channel": "trainer_override",
                          "source": "legacy.axis_name"},
        })
    candidates = _dedupe_candidates(candidates, history)[:max_candidates]
    return {
        "schema_version": "1.0",
        "round_id": round_id,
        "status": "ready" if candidates else "empty",
        "diagnosis": diagnosis,
        "candidates": candidates,
        "source": "legacy",
    }


def replan(
    *,
    base_plan: dict[str, Any],
    cluster: dict[str, Any],
    diagnosis: dict[str, Any],
    round_id: int,
    history: list[dict[str, Any]] | None = None,
    max_candidates: int = 3,
    train_iters: int = 20,
    plan_graph: dict[str, Any] | None = None,
    parent_plan_id: str | None = None,
) -> dict[str, Any]:
    """Generate a small priority pool of single-node candidates.

    Two paths:

    1. **Engine path** when ``diagnosis['meta']['engine_report']`` is present
       (i.e. ``diagnose()`` was called with ``trace_analysis=...``). Consumes
       the trace-driven candidate_axes / env_suspect verbatim, translates each
       (axis, value) pair via ``_axis_translator``, ranks by
       ``gain_mid × confidence ÷ cost_proxy`` × novelty × stability, and
       validates structural candidates through ``constraint.check``. Emits
       ``env_overrides`` per candidate when an axis is env-channel.

    2. **Legacy path** when only the legacy ``candidate_axes`` (a list of axis
       names) is available. Mirrors round-0 behavior so SMOKE / BASELINE
       failure routing keeps working.

    Optional ``plan_graph`` enables:
      - dedup against ``exhausted_neighborhoods`` (see plan_graph.md §5)
      - the novelty / stability bonuses from replan.md §3
      - the Periodic Exploration Round derivation source (settle.md §5)
    When ``plan_graph`` is None all PlanGraph-aware enhancements are skipped
    and behavior is identical to the pre-engine path.
    """
    history = history or []
    er = _engine_report(diagnosis)
    if er is not None:
        return _engine_replan(
            base_plan=base_plan,
            cluster=cluster,
            diagnosis=diagnosis,
            engine_report=er,
            round_id=round_id,
            history=history,
            max_candidates=max_candidates,
            train_iters=train_iters,
            plan_graph=plan_graph,
            parent_plan_id=parent_plan_id,
        )
    return _legacy_replan(
        base_plan=base_plan,
        cluster=cluster,
        diagnosis=diagnosis,
        round_id=round_id,
        history=history,
        max_candidates=max_candidates,
        train_iters=train_iters,
    )


def settle(
    history: list[dict[str, Any]],
    *,
    champion_id: str | None = None,
    epsilon_promote: float = _DEFAULT_EPSILON_PROMOTE,
    epsilon_stop: float = _DEFAULT_EPSILON_STOP,
    stagnation_rounds: int = _DEFAULT_STAGNATION_ROUNDS,
    plan_graph: dict[str, Any] | None = None,
    round_id: int | None = None,
    is_explore_round: bool = False,
) -> dict[str, Any]:
    """Pick a champion and decide whether the loop should stop.

    Source of truth: ``skills/workflow/settle.md``. Promotion rules (R1/R2/R3)
    live in §3; the default thresholds (`epsilon_promote`, `epsilon_stop`,
    `stagnation_rounds`) come from §4.

    Stop semantics:
      - With a ``plan_graph``: stagnation is "metadata.rounds_since_promotion
        ≥ stagnation_rounds after this round" (skills/workflow/settle.md §6.2).
        This is the canonical per-round signal; the flat ``history`` lookup is
        used only as a fallback for legacy callers without a graph.
      - With ``is_explore_round=True``: ``stop`` is forced to False per
        skills/workflow/settle.md §6 last paragraph ("the Periodic Exploration
        Round does not count toward stagnation").
      - With a ``plan_graph``: also fires the **Backtrack** signal when the
        current champion's subtree dead-rate exceeds the threshold
        (skills/workflow/settle.md §5). Backtrack returns a recommendation in
        ``backtrack`` and never auto-applies; the caller decides whether to
        rebase the champion onto ``backtrack.new_champion``.

    The single-node v1 implementation covers R1/R2/R3, stagnation,
    Backtrack signaling, and explore-round exemption. Diversification is
    realized by ``replan`` (the novelty/stability bonuses in
    skills/workflow/replan.md §3) and does not need a Settle hook.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    by_id = {item.get("id"): item for item in history if item.get("id")}
    current = by_id.get(champion_id) if champion_id else None
    current_score = score_measurement((current or {}).get("measurement") or {}) if current else None

    for item in history:
        score = score_measurement(item.get("measurement") or {})
        if score is not None:
            scored.append((score, item))

    if not scored:
        return {
            "status": "failed",
            "champion": current,
            "champion_id": current.get("id") if current else champion_id,
            "promoted": False,
            "stop": True,
            "reason": "no successful candidates with timing metrics",
        }

    best_score, best = max(scored, key=lambda x: x[0])
    promoted = False
    gain = 0.0
    if current_score is None or best_score > current_score * (1.0 + epsilon_promote):
        promoted = best is not current
        champion = best
        gain = 0.0 if current_score is None else (best_score / current_score - 1.0)
    else:
        champion = current or best
        gain = 0.0 if current_score is None else (best_score / current_score - 1.0)

    if plan_graph is not None:
        # Canonical stop: rounds_since_promotion is bumped by the caller AFTER
        # Settle returns when no promotion happened, so the value we read here
        # is the count BEFORE this round's increment. Predict the post-round
        # value: +1 if not promoted, 0 if promoted.
        cur_rsp = int(plan_graph["metadata"].get("rounds_since_promotion", 0))
        post_round_rsp = 0 if promoted else cur_rsp + 1
        stagnant = (
            stagnation_rounds > 0 and post_round_rsp >= stagnation_rounds
        )
    else:
        # Legacy fallback for callers without a PlanGraph.
        recent = history[-stagnation_rounds:] if stagnation_rounds > 0 else []
        recent_gains = [
            float(item.get("gain_vs_champion", 0.0))
            for item in recent
            if isinstance(item.get("gain_vs_champion", 0.0), (int, float))
        ]
        stagnant = (
            len(recent_gains) >= stagnation_rounds
            and all(g < epsilon_stop for g in recent_gains)
        )

    # Periodic Exploration Round is exempt from stagnation (settle.md §6 last ¶).
    if is_explore_round:
        stagnant = False

    # Backtrack signal: emit a recommendation but do not auto-apply
    # (settle.md §5 / plan_graph.md §6).
    backtrack: dict[str, Any] = {"fired": False, "reason": None, "new_champion": None}
    if plan_graph is not None and not promoted:
        if pg.should_backtrack(plan_graph):
            target = pg.pick_backtrack_target(plan_graph)
            if target is not None:
                backtrack = {
                    "fired": True,
                    "reason": (
                        f"subtree dead-rate "
                        f"{plan_graph['metadata']['dead_rate_in_subtree'].get(plan_graph['champion'], 0.0):.2f}"
                        f" > {_DEFAULT_DEAD_RATE_BACKTRACK}"
                    ),
                    "new_champion": target,
                }

    base_reason = (
        "stagnation" if stagnant
        else ("promoted new champion" if promoted else "kept champion")
    )
    reason = f"{base_reason}; backtrack→{backtrack['new_champion']}" if backtrack["fired"] else base_reason

    return {
        "status": "success",
        "champion": champion,
        "champion_id": champion.get("id") if champion else None,
        "promoted": promoted,
        "gain": round(gain, 6),
        "stop": stagnant and not backtrack["fired"],
        "reason": reason,
        "backtrack": backtrack,
        "is_explore_round": is_explore_round,
    }


def _maybe_analyze_trace(run_id: str, log_dir: str | Path) -> dict[str, Any] | None:
    """Best-effort: if `<run_dir>/profile/trace_meta.json` exists (default
    profile=True path in submit.run), run trace_analyze and write
    `<run_dir>/trace_analysis.{json,md}`. Returns the loaded dict or None.

    Failures here are intentionally non-fatal: a bad/missing trace must not
    break the tuning loop. Errors are recorded into the returned dict's
    `analysis_error` so downstream logging can surface them.
    """
    run_dir = _resolve_pilot_path(log_dir) / run_id
    meta_path = run_dir / "profile" / "trace_meta.json"
    if not meta_path.exists():
        return None
    try:
        from pilot.tools import trace_analyze as _ta
    except Exception as exc:  # pragma: no cover
        return {"analysis_error": f"trace_analyze unavailable: {exc}"}
    try:
        trace_meta = json.loads(meta_path.read_text())
        report = _ta.run(trace_meta, run_dir=run_dir)
    except Exception as exc:  # noqa: BLE001
        return {"analysis_error": f"{type(exc).__name__}: {exc}"}

    out_json = run_dir / "trace_analysis.json"
    out_md = run_dir / "trace_analysis.md"
    try:
        out_json.write_text(json.dumps(report, indent=2, default=str))
        out_md.write_text(_ta.render_md(report))
    except OSError as exc:  # pragma: no cover
        report["analysis_error"] = f"write failed: {exc}"
    return report


def _run_one(
    *,
    cfg: Any,
    launch_plan: Any,
    plan_path: Path,
    run_id: str,
    overrides: dict[str, Any],
    train_iters: int,
    timeout_s: int,
    log_dir: str | Path,
    env_overrides: dict[str, str] | None = None,
    analyze_trace: bool = True,
) -> dict[str, Any]:
    effective_overrides = {**overrides, "train_iters": train_iters}
    submit_result = submit.run(
        cfg,
        launch_plan,
        plan_path=plan_path,
        overrides=effective_overrides,
        env_overrides=env_overrides,
        run_id=run_id,
        log_dir=log_dir,
        foreground=True,
        foreground_timeout_s=timeout_s,
    )
    snap = observe.snapshot(run_id, log_dir=log_dir, save=True)
    measurement = summarize_snapshot(snap)
    measurement["status"] = "completed" if submit_result["status"] == "completed" and snap["status"] == "completed" else snap["status"]
    trace_analysis: dict[str, Any] | None = None
    if analyze_trace and measurement["status"] == "completed":
        trace_analysis = _maybe_analyze_trace(run_id, log_dir)
    return {
        "id": run_id,
        "run_id": run_id,
        "overrides": effective_overrides,
        "env_overrides": env_overrides or {},
        "submit": submit_result,
        "snapshot": snap,
        "measurement": measurement,
        "trace_analysis": trace_analysis,
    }


def run_session(
    *,
    cluster_config: str | Path,
    plan_path: str | Path,
    session_id: str | None = None,
    base_overrides: dict[str, Any] | None = None,
    rounds: int = 2,
    candidates_per_round: int = 3,
    smoke_iters: int = 20,
    train_iters: int = 30,
    timeout_s: int = 1800,
    root: str = "state",
    log_dir: str = "state/runs",
) -> dict[str, Any]:
    """Run the single-node tuning loop using real foreground training jobs."""
    cfg, launch_plan = preflight_check(str(cluster_config))
    if cfg.mode != "single":
        raise _TuneError("USAGE", f"tune_single only supports mode=single, got {cfg.mode!r}")

    cfg_dict = load_cluster_config(str(cluster_config)).__dict__
    plan_path_abs = Path(plan_path).resolve()
    base_plan = _load_yaml(plan_path_abs)
    session = session_id or f"{plan_path_abs.stem}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    plan_path_for_runs = plan_path_abs
    if base_overrides:
        base_plan = _merged_plan(base_plan, base_overrides)
        plan_path_for_runs = _resolve_pilot_path(root) / "sessions" / session / "base.effective.yaml"
        _atomic_write_yaml(plan_path_for_runs, base_plan)
    started = datetime.now(timezone.utc)
    run_history: list[dict[str, Any]] = []
    cluster_profile_for_diag = _latest_cluster_profile(root)

    tuning_state: dict[str, Any] = {
        "session_id": session,
        "current_stage": "SMOKE",
        "round_id": 0,
        "stage_history": [],
        "target": {
            "primary": {"metric": "median_iter_time_ms", "direction": "minimize"},
            "budget": {"rounds": rounds, "max_candidates_per_round": candidates_per_round},
        },
        "budget_used": {"gpu_h": 0.0, "rounds": 0, "wallclock_h": 0.0},
        "current_plan_ref": str(plan_path_for_runs),
        "run_history": run_history,
    }

    latest_checkpoint: str | None = None

    def checkpoint_stage(stage_name: str, status: str, headline: str) -> None:
        nonlocal latest_checkpoint
        tuning_state["stage_history"].append({
            "stage": stage_name,
            "exit": status,
            "round": tuning_state.get("round_id", 0),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "headline": headline,
        })
        latest_checkpoint = state.checkpoint(tuning_state, root=root)

    def finalize_report(note: str) -> dict[str, Any]:
        tuning_state["current_stage"] = "REPORT"
        tuning_state["budget_used"]["rounds"] = tuning_state.get("round_id", 0)
        tuning_state["budget_used"]["wallclock_h"] = round(
            (datetime.now(timezone.utc) - started).total_seconds() / 3600.0, 5
        )
        checkpoint_stage("REPORT", "ready", note)
        current_state_path = _resolve_pilot_path(root) / "tuning_state.yaml"
        try:
            report_result = report.build(
                plan_path=plan_path_for_runs,
                cluster_config_ref=str(cluster_config),
                tuning_state=str(current_state_path),
                report_id=f"{session}_report",
            )
            tuning_state["report_ref"] = report_result["artifacts"][0]["ref"]
            state.checkpoint(tuning_state, root=root)
        except Exception as exc:  # noqa: BLE001
            tuning_state["report_error"] = f"{type(exc).__name__}: {exc}"
            state.checkpoint(tuning_state, root=root)
        return tuning_state

    smoke = _run_one(
        cfg=cfg,
        launch_plan=launch_plan,
        plan_path=plan_path_for_runs,
        run_id=f"{session}_smoke",
        overrides={},
        train_iters=smoke_iters,
        timeout_s=timeout_s,
        log_dir=log_dir,
        # SMOKE only confirms the launcher boots; analyzing the trace adds
        # latency without buying us anything (the trace is too short to be
        # representative of the steady iter and DIAGNOSE never reads it).
        analyze_trace=False,
    )
    smoke["stage"] = "SMOKE"
    run_history.append(smoke)
    diagnosis = diagnose(
        smoke["snapshot"],
        cluster_profile=cluster_profile_for_diag,
        plan=base_plan,
    )
    checkpoint_stage("SMOKE", "success" if diagnosis.get("failure") is None else "failed", diagnosis["bottleneck"])
    if diagnosis.get("failure"):
        tuning_state["failure"] = diagnosis["failure"]
        return finalize_report("single-node tuning stopped after SMOKE failure")

    tuning_state["current_stage"] = "BASELINE"
    baseline = _run_one(
        cfg=cfg,
        launch_plan=launch_plan,
        plan_path=plan_path_for_runs,
        run_id=f"{session}_baseline",
        overrides={},
        train_iters=train_iters,
        timeout_s=timeout_s,
        log_dir=log_dir,
    )
    baseline["stage"] = "BASELINE"
    baseline["measurement"]["status"] = "completed" if baseline["submit"]["status"] == "completed" else baseline["measurement"]["status"]
    run_history.append(baseline)
    baseline_diag = diagnose(
        baseline["snapshot"],
        baseline=smoke["measurement"],
        cluster_profile=cluster_profile_for_diag,
        plan=base_plan,
        champion_snapshot=smoke.get("snapshot"),
        trace_analysis=baseline.get("trace_analysis"),
    )
    if baseline_diag.get("failure"):
        tuning_state["failure"] = baseline_diag["failure"]
        tuning_state["champion"] = baseline
        tuning_state["champion_id"] = baseline["id"]
        checkpoint_stage("BASELINE", "failed", baseline_diag["bottleneck"])
        return finalize_report("single-node tuning stopped after BASELINE failure")
    champion_id = baseline["id"]
    tuning_state["champion"] = baseline
    tuning_state["champion_id"] = champion_id

    # ---- PlanGraph initialization ----
    # Build the tree of derivations with the baseline as the root. See
    # skills/workflow/plan_graph.md §6 for the rollout: this is the single
    # source of truth for "what we've tried, what's been promoted, what's
    # exhausted" from this point forward; `run_history` is still kept as
    # the flat audit log.
    baseline_score = score_measurement(baseline["measurement"])
    plan_graph = pg.new(
        session_id=session,
        root_id=baseline["id"],
        root_tps=baseline_score,
        root_bottleneck=baseline_diag.get("bottleneck"),
        measurement_ref=baseline.get("run_id"),
    )
    plan_graph_path = _resolve_pilot_path(root) / "plan_graphs" / f"{session}.yaml"
    plan_graph_path = Path(pg.persist(plan_graph, plan_graph_path))
    tuning_state["plan_graph_ref"] = str(plan_graph_path)
    checkpoint_stage("BASELINE", "success", f"baseline iter={baseline['measurement'].get('median_iter_time_ms')}")

    for round_id in range(1, rounds + 1):
        tuning_state["round_id"] = round_id
        tuning_state["current_stage"] = "OPTIMIZE_LOOP.DIAGNOSE"
        # DIAGNOSE always speaks to the CURRENT CHAMPION ("what's bottlenecking
        # the best plan we have today?"). When the loop hasn't promoted yet,
        # champion is the baseline, so its trace + snapshot are the inputs.
        cur_champ = tuning_state.get("champion") or baseline
        cur_champ_id = champion_id

        # Periodic Exploration: if rounds_since_explore ≥ K, the next Re-Plan
        # derives from the highest-tps shelved node instead of the champion
        # (see skills/workflow/settle.md §5 and replan.md §5).
        derive_parent = cur_champ_id
        derive_policy = "exploit"
        if pg.should_explore_round(plan_graph):
            shelved = [
                nid for nid in pg.frontier_excluding_dead(plan_graph)
                if nid != plan_graph["champion"]
                and plan_graph["nodes"][nid]["status"] == "shelved"
            ]
            if shelved:
                derive_parent = shelved[0]
                derive_policy = "explore"

        diag = diagnose(
            cur_champ["snapshot"],
            baseline=baseline["measurement"],
            cluster_profile=cluster_profile_for_diag,
            plan=base_plan,
            champion_snapshot=cur_champ.get("snapshot"),
            trace_analysis=cur_champ.get("trace_analysis"),
        )
        pool = replan(
            base_plan=base_plan,
            cluster=cfg_dict,
            diagnosis=diag,
            round_id=round_id,
            history=run_history,
            max_candidates=candidates_per_round,
            train_iters=train_iters,
            plan_graph=plan_graph,
            parent_plan_id=derive_parent,
        )
        # Annotate the persisted pool with the derivation policy for audit.
        pool["policy"] = derive_policy
        pool_path = _resolve_pilot_path(root) / "candidate_pools" / f"{session}_r{round_id}.yaml"
        _atomic_write_yaml(pool_path, pool)
        tuning_state["candidate_pool_ref"] = str(pool_path)
        checkpoint_stage("OPTIMIZE_LOOP.REPLAN", pool["status"], f"{len(pool['candidates'])} candidates ({derive_policy})")
        if not pool["candidates"]:
            break

        current_score = score_measurement((tuning_state.get("champion") or {}).get("measurement") or {})
        round_executed: list[str] = []  # plan_ids attempted this round (for explore-round bookkeeping)
        for cand in pool["candidates"]:
            tuning_state["current_stage"] = "OPTIMIZE_LOOP.EXECUTE"
            # Register the candidate in the PlanGraph BEFORE running it
            # (the graph captures `running` nodes so a kill / crash leaves
            # an auditable trail).
            axis_meta = cand.get("axis_meta") or {}
            derived_axis = {
                "axis": axis_meta.get("axis"),
                "value": axis_meta.get("value"),
                "type": axis_meta.get("type"),
            } if axis_meta.get("axis") not in (None, "_control") else None
            # PlanGraph nodes are keyed by the same id used in run_history
            # (the full run_id), so Settle's `champion_id` lookup matches.
            cand_node_id = f"{session}_{cand['id']}"
            plan_graph = pg.add_node(
                plan_graph,
                plan_id=cand_node_id,
                parent=derive_parent,
                derived_axis=derived_axis,
                round_id=round_id,
                reason=cand.get("reason", ""),
            )

            run = _run_one(
                cfg=cfg,
                launch_plan=launch_plan,
                plan_path=plan_path_for_runs,
                run_id=f"{session}_{cand['id']}",
                overrides=cand["overrides"],
                env_overrides=cand.get("env_overrides") or {},
                train_iters=train_iters,
                timeout_s=timeout_s,
                log_dir=log_dir,
            )
            run["stage"] = "OPTIMIZE_LOOP.EXECUTE"
            run["candidate"] = cand
            score = score_measurement(run["measurement"])
            run["gain_vs_champion"] = (
                (score / current_score - 1.0)
                if score is not None and current_score else 0.0
            )
            run_history.append(run)
            round_executed.append(cand_node_id)

            # Record the outcome in the PlanGraph + register the (axis, value)
            # as exhausted so a future round doesn't blindly re-emit it.
            node_status = "completed" if score is not None else "dead"
            plan_graph = pg.record_result(
                plan_graph,
                plan_id=cand_node_id,
                status=node_status,
                tps=score,
                reason=run["measurement"].get("status", ""),
                measurement_ref=run["run_id"],
            )
            if derived_axis is not None:
                plan_graph = pg.mark_exhausted(
                    plan_graph,
                    around=derive_parent,
                    axis=str(derived_axis["axis"]),
                    value=derived_axis["value"],
                    axis_type=derived_axis.get("type"),
                )

        tuning_state["current_stage"] = "OPTIMIZE_LOOP.SETTLE"
        settle_result = settle(
            run_history,
            champion_id=champion_id,
            plan_graph=plan_graph,
            round_id=round_id,
            is_explore_round=(derive_policy == "explore"),
        )

        # Apply Settle's decision to the PlanGraph.
        new_champ_id = settle_result.get("champion_id")
        if settle_result.get("promoted") and new_champ_id and new_champ_id in plan_graph["nodes"]:
            plan_graph = pg.promote(plan_graph, plan_id=new_champ_id, round_id=round_id)
            tuning_state["champion"] = settle_result["champion"]
            tuning_state["champion_id"] = new_champ_id
            champion_id = new_champ_id
            # Shelve the round's runners-up (every other completed child).
            for cid in round_executed:
                if cid == new_champ_id:
                    continue
                node = plan_graph["nodes"].get(cid, {})
                if node.get("status") == "completed":
                    plan_graph = pg.shelve(
                        plan_graph,
                        plan_id=cid,
                        reason=f"runner-up r{round_id}; champion promoted to {new_champ_id}",
                    )
        else:
            # No promotion: shelve every still-completed child of this round
            # so it stays in the frontier as a backtrack candidate.
            for cid in round_executed:
                if plan_graph["nodes"].get(cid, {}).get("status") == "completed":
                    plan_graph = pg.shelve(
                        plan_graph,
                        plan_id=cid,
                        reason="marginal or no gain; kept for backtrack",
                    )
            plan_graph = pg.bump_promotion_counter(plan_graph)

        # Backtrack rescue (settle.md §5): rebase the champion onto the
        # recommended shelved node. ``promote`` enforces the new champion
        # swap and reshelves the prior champion regardless of whether this
        # round also promoted; we look the run record up by id in
        # run_history so the live tuning_state.champion mirrors the new
        # champion's measurement / snapshot (needed by the next DIAGNOSE).
        backtrack = settle_result.get("backtrack") or {}
        if backtrack.get("fired") and backtrack.get("new_champion") in plan_graph["nodes"]:
            target = backtrack["new_champion"]
            plan_graph = pg.promote(plan_graph, plan_id=target, round_id=round_id)
            rescued = next(
                (r for r in run_history if r.get("id") == target),
                None,
            )
            if rescued is not None:
                tuning_state["champion"] = rescued
                tuning_state["champion_id"] = target
                champion_id = target

        # Maintain the periodic-exploration counter.
        if derive_policy == "explore":
            plan_graph = pg.reset_explore_counter(plan_graph)
        else:
            plan_graph = pg.bump_explore_counter(plan_graph)

        # Persist after each Settle exit.
        plan_graph_path = Path(pg.persist(plan_graph, plan_graph_path))
        tuning_state["plan_graph_ref"] = str(plan_graph_path)

        # Stop gating with shelved-reprieve (settle.md §6 last ¶):
        # if Settle wants to stop but (a) there are still shelved candidates
        # in the frontier AND (b) the loop hasn't yet completed an explore
        # round, defer the stop so we get at least one Periodic Exploration
        # Round before declaring stagnation. Once an explore round has
        # fired and the loop is still stagnant, stop sticks.
        should_stop = settle_result["stop"]
        if should_stop:
            shelved_exists = any(
                n["status"] == "shelved" for n in plan_graph["nodes"].values()
            )
            already_explored = bool(
                plan_graph["metadata"].get("explore_rounds_completed", 0) > 0
            )
            if shelved_exists and not already_explored:
                should_stop = False

        checkpoint_stage(
            "OPTIMIZE_LOOP.SETTLE",
            settle_result["status"],
            settle_result["reason"] + (
                " (stop deferred for explore round)"
                if settle_result["stop"] and not should_stop else ""
            ),
        )
        if should_stop:
            break

    return finalize_report("single-node tuning loop complete")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "TUNE_SINGLE",
        "status": "failed",
        "failure": {"kind": kind, "message": message, "escalate_to_orchestrator": True},
    }


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.tune_single")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_diag = sub.add_parser("diagnose")
    p_diag.add_argument("--snapshot", required=True)
    p_diag.add_argument("--baseline", default=None)

    p_replan = sub.add_parser("replan")
    p_replan.add_argument("--plan", required=True)
    p_replan.add_argument("--cluster-config", required=True)
    p_replan.add_argument("--snapshot", required=True)
    p_replan.add_argument("--round-id", type=int, default=1)
    p_replan.add_argument("--max-candidates", type=int, default=3)
    p_replan.add_argument("--train-iters", type=int, default=20)

    p_settle = sub.add_parser("settle")
    p_settle.add_argument("--history", required=True)
    p_settle.add_argument("--champion-id", default=None)

    p_run = sub.add_parser("run")
    p_run.add_argument("--cluster-config", required=True)
    p_run.add_argument("--plan", required=True)
    p_run.add_argument("--session-id", default=None)
    p_run.add_argument("--override", action="append", default=[],
                       help="Base override key=value applied before all tuning stages.")
    p_run.add_argument("--rounds", type=int, default=2)
    p_run.add_argument("--candidates-per-round", type=int, default=3)
    p_run.add_argument("--smoke-iters", type=int, default=20)
    p_run.add_argument("--train-iters", type=int, default=30)
    p_run.add_argument("--timeout-s", type=int, default=1800)
    p_run.add_argument("--root", default="state")
    p_run.add_argument("--log-dir", default="state/runs")

    args = p.parse_args()
    try:
        if args.cmd == "diagnose":
            baseline = _load_yaml(args.baseline) if args.baseline else None
            _emit(diagnose(_load_yaml(args.snapshot), baseline=baseline))
            return 0
        if args.cmd == "replan":
            cfg = load_cluster_config(args.cluster_config)
            snap = _load_yaml(args.snapshot)
            diag = diagnose(snap)
            _emit(replan(
                base_plan=_load_yaml(args.plan),
                cluster=cfg.__dict__,
                diagnosis=diag,
                round_id=args.round_id,
                max_candidates=args.max_candidates,
                train_iters=args.train_iters,
            ))
            return 0
        if args.cmd == "settle":
            data = _load_yaml(args.history)
            hist = data.get("run_history") or data.get("history") or []
            _emit(settle(hist, champion_id=args.champion_id))
            return 0
        if args.cmd == "run":
            overrides: dict[str, Any] = {}
            for spec in args.override:
                k, v = _parse_override(spec)
                overrides[k] = v
            result = run_session(
                cluster_config=args.cluster_config,
                plan_path=args.plan,
                session_id=args.session_id,
                base_overrides=overrides,
                rounds=args.rounds,
                candidates_per_round=args.candidates_per_round,
                smoke_iters=args.smoke_iters,
                train_iters=args.train_iters,
                timeout_s=args.timeout_s,
                root=args.root,
                log_dir=args.log_dir,
            )
            _emit(result)
            return 0
    except ClusterConfigError as exc:
        _emit(_failure("CLUSTER", exc.to_message()))
        return 4
    except (submit._SubmitError, observe._ObserveError, _TuneError) as exc:  # noqa: SLF001
        _emit(_failure(getattr(exc, "kind", "UNKNOWN"), str(exc)))
        return 2

    return 2


if __name__ == "__main__":
    sys.exit(_cli())
