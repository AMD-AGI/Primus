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

from pilot.tools import constraint, observe, report, state, submit
from pilot.tools._cluster_config import ClusterConfigError, load_cluster_config, preflight_check


_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent
_REPO_ROOT: Path = _PILOT_ROOT.parent
_DEFAULT_EPSILON_PROMOTE = 0.02
_DEFAULT_EPSILON_STOP = 0.005


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


def diagnose(snapshot: dict[str, Any], *, baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify a single-node RunSnapshot for candidate generation."""
    summary = summarize_snapshot(snapshot)
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


def replan(
    *,
    base_plan: dict[str, Any],
    cluster: dict[str, Any],
    diagnosis: dict[str, Any],
    round_id: int,
    history: list[dict[str, Any]] | None = None,
    max_candidates: int = 3,
    train_iters: int = 20,
) -> dict[str, Any]:
    """Generate a small priority pool of single-node override candidates."""
    history = history or []
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

    if "micro_batch_size" in axes:
        add(1.0, "probe higher micro_batch_size for better compute occupancy", {
            "micro_batch_size": mbs + 1,
            "global_batch_size": max(gbs, (mbs + 1) * max(1, world // max(1, tp * pp * ep))),
        })
        if mbs > 1:
            add(0.95, "reduce micro_batch_size after memory or stability signal", {
                "micro_batch_size": max(1, mbs - 1),
                "global_batch_size": max(1, gbs),
            })
    if "tensor_model_parallel_size" in axes:
        for next_tp in (tp * 2, max(1, tp // 2)):
            if next_tp != tp and next_tp >= 1 and world % max(1, next_tp * pp * ep) == 0:
                add(0.85, f"probe tensor_model_parallel_size={next_tp}", {
                    "tensor_model_parallel_size": next_tp,
                })
    if "recompute" in axes or diagnosis.get("bottleneck") in ("MEMORY", "OOM"):
        add(0.9, "enable full block recompute to relieve activation memory", {
            "recompute_granularity": "full",
            "recompute_method": "block",
            "recompute_num_layers": int(overrides.get("recompute_num_layers") or 1),
        })

    add(0.5, "control rerun with current structure and fixed short train_iters", {
        "train_iters": train_iters,
    })

    candidates: list[dict[str, Any]] = []
    for idx, (priority, reason, cand_overrides) in enumerate(sorted(raw, key=lambda x: x[0], reverse=True), 1):
        candidates.append({
            "id": _candidate_id(round_id, idx, cand_overrides),
            "round_id": round_id,
            "priority": priority,
            "reason": reason,
            "overrides": cand_overrides,
        })
    candidates = _dedupe_candidates(candidates, history)[:max_candidates]
    return {
        "schema_version": "1.0",
        "round_id": round_id,
        "status": "ready" if candidates else "empty",
        "diagnosis": diagnosis,
        "candidates": candidates,
    }


def settle(
    history: list[dict[str, Any]],
    *,
    champion_id: str | None = None,
    epsilon_promote: float = _DEFAULT_EPSILON_PROMOTE,
    epsilon_stop: float = _DEFAULT_EPSILON_STOP,
    stagnation_rounds: int = 2,
) -> dict[str, Any]:
    """Pick a champion and decide whether the loop should stop."""
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

    recent = history[-stagnation_rounds:] if stagnation_rounds > 0 else []
    recent_gains = [
        float(item.get("gain_vs_champion", 0.0))
        for item in recent
        if isinstance(item.get("gain_vs_champion", 0.0), (int, float))
    ]
    stagnant = len(recent_gains) >= stagnation_rounds and all(g < epsilon_stop for g in recent_gains)
    return {
        "status": "success",
        "champion": champion,
        "champion_id": champion.get("id") if champion else None,
        "promoted": promoted,
        "gain": round(gain, 6),
        "stop": stagnant,
        "reason": "stagnation" if stagnant else ("promoted new champion" if promoted else "kept champion"),
    }


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
) -> dict[str, Any]:
    effective_overrides = {**overrides, "train_iters": train_iters}
    submit_result = submit.run(
        cfg,
        launch_plan,
        plan_path=plan_path,
        overrides=effective_overrides,
        run_id=run_id,
        log_dir=log_dir,
        foreground=True,
        foreground_timeout_s=timeout_s,
    )
    snap = observe.snapshot(run_id, log_dir=log_dir, save=True)
    measurement = summarize_snapshot(snap)
    measurement["status"] = "completed" if submit_result["status"] == "completed" and snap["status"] == "completed" else snap["status"]
    return {
        "id": run_id,
        "run_id": run_id,
        "overrides": effective_overrides,
        "submit": submit_result,
        "snapshot": snap,
        "measurement": measurement,
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
    )
    smoke["stage"] = "SMOKE"
    run_history.append(smoke)
    diagnosis = diagnose(smoke["snapshot"])
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
    baseline_diag = diagnose(baseline["snapshot"], baseline=smoke["measurement"])
    if baseline_diag.get("failure"):
        tuning_state["failure"] = baseline_diag["failure"]
        tuning_state["champion"] = baseline
        tuning_state["champion_id"] = baseline["id"]
        checkpoint_stage("BASELINE", "failed", baseline_diag["bottleneck"])
        return finalize_report("single-node tuning stopped after BASELINE failure")
    champion_id = baseline["id"]
    tuning_state["champion"] = baseline
    tuning_state["champion_id"] = champion_id
    checkpoint_stage("BASELINE", "success", f"baseline iter={baseline['measurement'].get('median_iter_time_ms')}")

    for round_id in range(1, rounds + 1):
        tuning_state["round_id"] = round_id
        tuning_state["current_stage"] = "OPTIMIZE_LOOP.DIAGNOSE"
        diag = diagnose(baseline["snapshot"], baseline=baseline["measurement"])
        pool = replan(
            base_plan=base_plan,
            cluster=cfg_dict,
            diagnosis=diag,
            round_id=round_id,
            history=run_history,
            max_candidates=candidates_per_round,
            train_iters=train_iters,
        )
        pool_path = _resolve_pilot_path(root) / "candidate_pools" / f"{session}_r{round_id}.yaml"
        _atomic_write_yaml(pool_path, pool)
        tuning_state["candidate_pool_ref"] = str(pool_path)
        checkpoint_stage("OPTIMIZE_LOOP.REPLAN", pool["status"], f"{len(pool['candidates'])} candidates")
        if not pool["candidates"]:
            break

        current_score = score_measurement((tuning_state.get("champion") or {}).get("measurement") or {})
        for cand in pool["candidates"]:
            tuning_state["current_stage"] = "OPTIMIZE_LOOP.EXECUTE"
            run = _run_one(
                cfg=cfg,
                launch_plan=launch_plan,
                plan_path=plan_path_for_runs,
                run_id=f"{session}_{cand['id']}",
                overrides=cand["overrides"],
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

        tuning_state["current_stage"] = "OPTIMIZE_LOOP.SETTLE"
        settle_result = settle(run_history, champion_id=champion_id)
        if settle_result.get("champion"):
            tuning_state["champion"] = settle_result["champion"]
            tuning_state["champion_id"] = settle_result.get("champion_id")
            champion_id = settle_result.get("champion_id")
        checkpoint_stage("OPTIMIZE_LOOP.SETTLE", settle_result["status"], settle_result["reason"])
        if settle_result["stop"]:
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
