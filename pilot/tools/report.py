"""pilot.tools.report — aggregate stage artifacts into a TuningReport.

REPORT is the *terminal* stage: it consumes artifacts produced by upstream
stages (PREFLIGHT ClusterProfile, SMOKE RunHandle + RunSnapshot, optionally
SCAN_PARALLEL/BASELINE outputs) and emits a single deliverable that closes
the tuning session:

  ``state/reports/<report_id>.yaml``  validated against ``schemas/tuning_report.schema.json``
  ``state/reports/<report_id>.md``    human-readable rendering of the same data

The CLI does **not** launch any subprocess and does not need a GPU; aggregation
is pure I/O + dict munging.

CLI surface
-----------
::

    report build  --plan <plan.yaml> [--cluster-config cluster.yaml]
                  [--cluster-profile <profile.yaml>]   # default: latest in state/cluster_profiles/
                  [--smoke-run-id <id>] [--smoke-snapshot <path>]
                  [--report-id <id>]
                  [--out-dir state/reports]
                  [--render-markdown]                   # default true

    report show   --report-id <id>                        # print the markdown to stdout

Anything *not* passed in is treated as `skipped`: the schema captures the
distinction between "stage failed" and "stage was never run".
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.tools._schema import validate as _validate_schema


# ---------------------------------------------------------------------------
# Path anchoring (mirror the rest of the toolset)
# ---------------------------------------------------------------------------

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent
_REPO_ROOT: Path = _PILOT_ROOT.parent


def _resolve_pilot_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else _PILOT_ROOT / pp


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _ReportError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _ReportError("DEP_MISSING", f"PyYAML required: {exc}")
    if not path.exists():
        raise _ReportError("USAGE", f"file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise _ReportError("USAGE", f"{path} is not a YAML mapping")
    return data


def _atomic_write_yaml(path: Path, data: dict[str, Any], *, schema_name: str | None = None) -> None:
    import yaml
    if schema_name is not None:
        _validate_schema(data, schema_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _latest_cluster_profile(profiles_dir: Path) -> Path | None:
    if not profiles_dir.exists():
        return None
    yamls = sorted(profiles_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return yamls[0] if yamls else None


def _latest_snapshot(run_dir: Path) -> Path | None:
    snap_dir = run_dir / "snapshots"
    if not snap_dir.exists():
        return None
    yamls = sorted(snap_dir.glob("*.yaml"))
    return yamls[-1] if yamls else None


def _git_rev(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return None


def _operator_hint() -> str | None:
    return os.environ.get("USER") or os.environ.get("USERNAME") or None


# ---------------------------------------------------------------------------
# Per-stage extractors
# ---------------------------------------------------------------------------


def _summarize_preflight(profile: dict[str, Any]) -> dict[str, Any]:
    """Distill a ClusterProfile into the terse `stages.preflight` block.

    Status ladder:
      - fail  : status == 'failed' (sentinel exists in profile)
      - warn  : peak_pct_of_spec_bf16 < 0.40 OR slow_nodes non-empty
      - pass  : otherwise
    """
    rccl = profile.get("rccl_baseline", {}) or {}
    intra = rccl.get("intra_node") or {}
    ar_block = (intra.get("collectives") or {}).get("allreduce") or {}
    sizes = ar_block.get("sizes_mb", [])
    medians = (ar_block.get("roll_up") or {}).get("median_bw_gbs", [])
    ar_256 = None
    for s, bw in zip(sizes, medians):
        if s == 256:
            ar_256 = bw
            break

    slow_nodes = (ar_block.get("roll_up") or {}).get("slow_nodes_at_max_size") or []
    peak_pct = (profile.get("compute") or {}).get("peak_pct_of_spec_bf16")

    warnings: list[str] = []
    status = "pass"
    if peak_pct is not None and peak_pct < 0.40:
        warnings.append(f"BF16 peak only {peak_pct*100:.1f}% of vendor spec (<40%)")
        status = "warn"
    if slow_nodes:
        warnings.append(f"slow nodes flagged at max collective size: {slow_nodes}")
        status = "warn"
    if profile.get("status") == "failed":
        status = "fail"

    return {
        "status": status,
        "summary": {
            "cluster_class":         profile.get("cluster_class"),
            "gpus_per_node":         profile.get("gpus_per_node"),
            "nnodes":                profile.get("nodes_total"),
            "rccl_ar_256mb_gbs":     ar_256,
            "peak_pct_of_spec_bf16": peak_pct,
            "slow_nodes":            list(slow_nodes),
        },
        "warnings": warnings,
    }


def _summarize_smoke(handle: dict[str, Any], snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Distill RunHandle + last RunSnapshot into the `stages.smoke` block.

    Status ladder:
      - pass : handle.status == 'completed' AND no symptoms.* AND loss_finite
      - warn : handle.status == 'completed' BUT iters_per_min low / loss low
      - fail : handle.status in {failed, killed, hung, unknown}
    """
    rid = handle.get("run_id", "")
    handle_status = handle.get("status", "unknown")
    exit_code = handle.get("exit_code")

    iters_done: int | None = None
    iters_target: int | None = None
    median_iter_time_ms: float | None = None
    median_tflops: float | None = None
    loss_finite: bool | None = None
    symptoms: dict[str, bool] = {}

    snapshot_ref: str | None = None
    if snapshot:
        snapshot_ref = snapshot.get("_ref")  # opportunistic
        prog = snapshot.get("progress", {}) or {}
        metrics = snapshot.get("metrics", {}) or {}
        hist = metrics.get("history", {}) or {}
        iters_done = prog.get("current_iter")
        iters_target = prog.get("total_iters")
        loss_finite = metrics.get("loss_finite")
        for key, dest in (("iter_time_ms", "median_iter_time_ms"),
                          ("tflops",       "median_tflops")):
            vals = [v for v in (hist.get(key) or []) if v is not None]
            if vals:
                if dest == "median_iter_time_ms":
                    median_iter_time_ms = round(statistics.median(vals), 2)
                else:
                    median_tflops = round(statistics.median(vals), 2)
        symptoms = {k: bool(v) for k, v in (snapshot.get("symptoms") or {}).items()
                    if isinstance(v, bool)}

    warnings: list[str] = []
    if handle_status in ("launching", "running"):
        # Non-terminal — REPORT cannot decide yet; bubble up as a
        # session-level `incomplete` via _decide_verdict.
        status = "in_flight"
        warnings.append(f"SMOKE is still {handle_status}; cannot build a final verdict.")
    elif handle_status in ("failed", "killed", "hung", "unknown"):
        status = "fail"
        warnings.append(f"handle.status={handle_status}, exit_code={exit_code}")
    elif handle_status == "completed":
        if loss_finite is False:
            status = "fail"
            warnings.append("loss is NaN or Inf")
        elif any(symptoms.get(k, False) for k in
                 ("oom_detected", "nccl_error", "cuda_error", "python_error")):
            status = "fail"
            warnings.append(f"hard symptom in evidence: {[k for k,v in symptoms.items() if v]}")
        elif iters_done is None or iters_target is None:
            status = "warn"
            warnings.append("training completed but iter progress could not be parsed")
        else:
            status = "pass"
    else:
        status = "warn"
        warnings.append(f"unexpected handle.status={handle_status}")

    return {
        "status":       status,
        "run_id":       rid,
        "handle_ref":   "",  # filled by caller (knows the file path)
        "snapshot_ref": snapshot_ref,
        "summary": {
            "iters_done":          iters_done,
            "iters_target":        iters_target,
            "median_iter_time_ms": median_iter_time_ms,
            "median_tflops":       median_tflops,
            "loss_finite":         loss_finite,
            "wallclock_s":         handle.get("wallclock_s"),
            "exit_code":           exit_code,
        },
        "symptoms": symptoms,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Verdict rules
# ---------------------------------------------------------------------------


def _decide_verdict(stages: dict[str, Any]) -> dict[str, Any]:
    """Distill stage-level pass/warn/fail into the top-level verdict."""
    pf = stages.get("preflight")
    sm = stages.get("smoke")

    blockers: list[str] = []
    next_action = "wait"
    overall = "incomplete"

    if pf and pf.get("status") == "fail":
        blockers.append("PREFLIGHT failed; cluster baseline is unsafe.")
        return {
            "overall": "fail",
            "headline": "Aborted: PREFLIGHT failure.",
            "next_action": "rerun_preflight",
            "blockers": blockers,
        }

    if sm is None:
        blockers.append("SMOKE was not executed; cannot promote without a tiny-scale liveness check.")
        return {
            "overall": "incomplete",
            "headline": "PREFLIGHT only — SMOKE pending.",
            "next_action": "rerun_smoke",
            "blockers": blockers,
        }

    if sm.get("status") == "in_flight":
        blockers.append("SMOKE has not terminated; final verdict requires a completed run.")
        return {
            "overall": "incomplete",
            "headline": "SMOKE still running — re-build the report once it finishes.",
            "next_action": "wait",
            "blockers": blockers,
        }

    if sm.get("status") == "fail":
        for w in sm.get("warnings") or []:
            blockers.append(f"SMOKE: {w}")
        return {
            "overall": "fail",
            "headline": "SMOKE failed; do not promote.",
            "next_action": "diagnose",
            "blockers": blockers,
        }

    if sm.get("status") == "warn" or (pf and pf.get("status") == "warn"):
        return {
            "overall": "warn",
            "headline": "SMOKE passed with warnings; review before promotion.",
            "next_action": "promote",
            "blockers": [],
        }

    return {
        "overall": "pass",
        "headline": "PREFLIGHT + SMOKE both clean; safe to proceed.",
        "next_action": "promote",
        "blockers": [],
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(
    plan_path: str | Path,
    *,
    cluster_config_ref: str | None = None,
    cluster_profile: str | Path | None = None,
    smoke_run_dir: str | Path | None = None,
    smoke_snapshot: str | Path | None = None,
    report_id: str | None = None,
    out_dir: str | Path = "state/reports",
    render_markdown: bool = True,
) -> dict[str, Any]:
    """Aggregate inputs → TuningReport (validated, persisted)."""
    started = datetime.now(timezone.utc)
    plan_path = Path(plan_path).resolve()
    if not plan_path.exists():
        raise _ReportError("USAGE", f"--plan not found: {plan_path}")

    rid = report_id or f"{plan_path.stem}__{started.strftime('%Y%m%dT%H%M%S')}"
    if not rid.replace("-", "").replace("_", "").isalnum():
        raise _ReportError("USAGE", f"invalid --report-id: {rid!r}")

    artifacts: list[dict[str, str]] = [
        {"kind": "PrimusPlan", "ref": str(plan_path)},
    ]

    # PREFLIGHT
    stages: dict[str, Any] = {}
    if cluster_profile is None:
        cluster_profile = _latest_cluster_profile(_PILOT_ROOT / "state" / "cluster_profiles")
    if cluster_profile:
        cp_path = Path(cluster_profile).resolve()
        profile = _load_yaml(cp_path)
        _validate_schema(profile, "cluster_profile")
        pf_block = _summarize_preflight(profile)
        pf_block["ref"] = str(cp_path)
        stages["preflight"] = pf_block
        artifacts.append({"kind": "ClusterProfile", "ref": str(cp_path)})
    else:
        stages["preflight"] = None

    # SMOKE
    if smoke_run_dir:
        run_dir = Path(smoke_run_dir).resolve()
        handle_path = run_dir / "handle.yaml"
        if not handle_path.exists():
            raise _ReportError("USAGE", f"smoke run handle not found: {handle_path}")
        handle = _load_yaml(handle_path)
        _validate_schema(handle, "run_handle")

        snap_path = (
            Path(smoke_snapshot).resolve() if smoke_snapshot
            else _latest_snapshot(run_dir)
        )
        snap = None
        if snap_path:
            snap = _load_yaml(snap_path)
            _validate_schema(snap, "run_snapshot")
            snap["_ref"] = str(snap_path)  # local note; stripped before persist
            artifacts.append({"kind": "RunSnapshot", "ref": str(snap_path)})

        sm_block = _summarize_smoke(handle, snap)
        sm_block["handle_ref"] = str(handle_path)
        if snap:
            snap.pop("_ref", None)
        stages["smoke"] = sm_block
        artifacts.append({"kind": "RunHandle", "ref": str(handle_path)})
    else:
        stages["smoke"] = None

    # SCAN_PARALLEL + BASELINE not yet implemented
    stages["scan_parallel"] = None
    stages["baseline"] = None

    verdict = _decide_verdict(stages)

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "report_id": rid,
        "generated_at": started.isoformat(timespec="seconds"),
        "session": {
            "plan_ref": str(plan_path),
            "plan_name": plan_path.name,
            "cluster_config_ref": cluster_config_ref,
            "cluster_id": (stages.get("preflight") or {}).get("summary", {}).get("cluster_class"),
            "operator": _operator_hint(),
            "primus_git_rev": _git_rev(_REPO_ROOT),
        },
        "stages": stages,
        "verdict": verdict,
        "artifacts": artifacts,
        "meta": {
            "pilot_version": None,
            "host": socket.gethostname(),
            "wallclock_s": round((datetime.now(timezone.utc) - started).total_seconds(), 2),
        },
    }

    out_yaml = _resolve_pilot_path(out_dir) / f"{rid}.yaml"
    _atomic_write_yaml(out_yaml, report, schema_name="tuning_report")

    if render_markdown:
        md = render_markdown_text(report)
        out_md = out_yaml.with_suffix(".md")
        out_md.write_text(md)
        artifacts.append({"kind": "TuningReportMarkdown", "ref": str(out_md)})

    return _build_subagent_result(report, out_yaml, started)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_markdown_text(report: dict[str, Any]) -> str:
    """Render the report into a human-readable Markdown string."""
    parts: list[str] = []
    p = parts.append

    verdict = report.get("verdict", {})
    overall = verdict.get("overall", "?")
    badge = {
        "pass":       "PASS",
        "warn":       "PASS (with warnings)",
        "fail":       "FAIL",
        "incomplete": "INCOMPLETE",
    }.get(overall, overall.upper())

    s = report.get("session", {})
    p(f"# Tuning Report — {report['report_id']}")
    p("")
    p(f"**Verdict**: {badge} — {verdict.get('headline','')}")
    p(f"**Next action**: `{verdict.get('next_action','wait')}`")
    p("")
    p("## Session")
    p("")
    p(f"- **Plan**: `{s.get('plan_name')}` ({s.get('plan_ref')})")
    p(f"- **Cluster config**: `{s.get('cluster_config_ref') or '(none)'}`")
    p(f"- **Operator**: `{s.get('operator') or '?'}`")
    p(f"- **Primus rev**: `{s.get('primus_git_rev') or '?'}`")
    p(f"- **Generated**: `{report.get('generated_at')}`")

    blockers = verdict.get("blockers") or []
    if blockers:
        p("")
        p("## Blockers")
        p("")
        for b in blockers:
            p(f"- {b}")

    # PREFLIGHT
    pf = (report.get("stages") or {}).get("preflight")
    p("")
    p("## PREFLIGHT")
    p("")
    if pf is None:
        p("_skipped_")
    else:
        sm = pf.get("summary") or {}
        p(f"- **Status**: `{pf.get('status')}`")
        p(f"- **Cluster class**: `{sm.get('cluster_class')}`  "
          f"(nodes={sm.get('nnodes')}, gpus_per_node={sm.get('gpus_per_node')})")
        p(f"- **AllReduce@256MB intra-node median**: "
          f"`{sm.get('rccl_ar_256mb_gbs')} GB/s`")
        peak_pct = sm.get("peak_pct_of_spec_bf16")
        if peak_pct is not None:
            p(f"- **BF16 peak vs vendor spec**: `{peak_pct*100:.1f}%`")
        slow = sm.get("slow_nodes") or []
        if slow:
            p(f"- **Slow nodes**: `{', '.join(slow)}`")
        for w in pf.get("warnings") or []:
            p(f"  - warning: {w}")

    # SMOKE
    sm = (report.get("stages") or {}).get("smoke")
    p("")
    p("## SMOKE")
    p("")
    if sm is None:
        p("_skipped_")
    else:
        s2 = sm.get("summary") or {}
        p(f"- **Status**: `{sm.get('status')}`  (run_id `{sm.get('run_id')}`)")
        p(f"- **Iterations**: `{s2.get('iters_done')}/{s2.get('iters_target')}`")
        p(f"- **Median iter time**: `{s2.get('median_iter_time_ms')} ms`")
        p(f"- **Median TFLOPS/GPU**: `{s2.get('median_tflops')}`")
        p(f"- **Wallclock**: `{s2.get('wallclock_s')} s`  "
          f"(exit_code=`{s2.get('exit_code')}`, loss_finite=`{s2.get('loss_finite')}`)")
        sym = sm.get("symptoms") or {}
        flagged = [k for k, v in sym.items() if v]
        if flagged:
            p(f"- **Symptoms**: `{', '.join(flagged)}`")
        for w in sm.get("warnings") or []:
            p(f"  - warning: {w}")
        p(f"- handle: `{sm.get('handle_ref')}`")
        if sm.get("snapshot_ref"):
            p(f"- snapshot: `{sm.get('snapshot_ref')}`")

    # SCAN_PARALLEL / BASELINE
    for label, key in (("SCAN_PARALLEL", "scan_parallel"), ("BASELINE", "baseline")):
        p("")
        p(f"## {label}")
        p("")
        p("_skipped (stage not implemented yet)_"
          if (report.get("stages") or {}).get(key) is None
          else f"`{json.dumps((report['stages'])[key], indent=2)}`")

    # Artifacts
    p("")
    p("## Artifacts")
    p("")
    for a in report.get("artifacts") or []:
        p(f"- **{a.get('kind')}**: `{a.get('ref')}`")

    p("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------


def show(report_id: str, *, out_dir: str | Path = "state/reports") -> str:
    yaml_path = _resolve_pilot_path(out_dir) / f"{report_id}.yaml"
    if not yaml_path.exists():
        raise _ReportError("USAGE", f"no report at {yaml_path}")
    md_path = yaml_path.with_suffix(".md")
    if md_path.exists():
        return md_path.read_text()
    # Fall back: render on the fly.
    return render_markdown_text(_load_yaml(yaml_path))


# ---------------------------------------------------------------------------
# SubagentResult
# ---------------------------------------------------------------------------


def _build_subagent_result(report: dict[str, Any], yaml_path: Path, started_dt: datetime) -> dict[str, Any]:
    elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
    verdict = report["verdict"]
    return {
        "stage": "REPORT",
        "status": "completed",
        "artifacts": [
            {"kind": "TuningReport", "ref": str(yaml_path)},
            {"kind": "TuningReportMarkdown", "ref": str(yaml_path.with_suffix('.md'))},
        ],
        "summary": {
            "headline": f"{verdict['overall']}: {verdict['headline']}",
            "key_metrics": {
                "report_id":   report["report_id"],
                "overall":     verdict["overall"],
                "next_action": verdict["next_action"],
                "blockers":    len(verdict.get("blockers") or []),
            },
            "warnings": [],
        },
        "suggested_transition": {
            "to": {
                "promote":         "DONE",
                "retune":          "PROJECTION",
                "rerun_smoke":     "SMOKE",
                "rerun_preflight": "PREFLIGHT",
                "diagnose":        "DIAGNOSE",
                "wait":            "WAIT",
            }.get(verdict.get("next_action", "wait"), "WAIT"),
            "reason": verdict.get("headline", ""),
        },
        "cost": {
            "gpu_h": 0.0,
            "wallclock_s": round(elapsed, 2),
            "tool_calls": 1,
        },
        "failure": None if verdict["overall"] != "fail" else {
            "kind": "REPORT_FAIL",
            "message": verdict["headline"],
            "escalate_to_orchestrator": True,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK = 0
_EXIT_USAGE = 2


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "REPORT",
        "status": "failed",
        "failure": {"kind": kind, "message": message,
                    "escalate_to_orchestrator": True},
    }


def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="pilot.tools.report",
        description="Aggregate stage artifacts into a TuningReport (yaml + markdown).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("build", help="Build a new report.")
    p_b.add_argument("--plan", required=True, help="Path to the Primus exp.yaml that was tuned.")
    p_b.add_argument("--cluster-config", default=None,
                     help="Optional: cluster.yaml path (recorded only; not validated here).")
    p_b.add_argument("--cluster-profile", default=None,
                     help="Path to a ClusterProfile yaml. Default: latest under state/cluster_profiles/.")
    p_b.add_argument("--smoke-run-dir", default=None,
                     help="Path to state/runs/<run_id>/ produced by `submit run`.")
    p_b.add_argument("--smoke-run-id", default=None,
                     help="Convenience: shorthand that resolves to state/runs/<id>/.")
    p_b.add_argument("--smoke-snapshot", default=None,
                     help="Optional: pin a specific snapshot YAML. Default: latest in <run_dir>/snapshots/.")
    p_b.add_argument("--report-id", default=None)
    p_b.add_argument("--out-dir", default="state/reports")
    p_b.add_argument("--no-markdown", action="store_true", help="Skip markdown rendering.")

    p_s = sub.add_parser("show", help="Print a previously-built report's markdown to stdout.")
    p_s.add_argument("--report-id", required=True)
    p_s.add_argument("--out-dir", default="state/reports")

    args = p.parse_args()

    try:
        if args.cmd == "build":
            smoke_run_dir = args.smoke_run_dir
            if not smoke_run_dir and args.smoke_run_id:
                smoke_run_dir = _PILOT_ROOT / "state" / "runs" / args.smoke_run_id
            result = build(
                plan_path=args.plan,
                cluster_config_ref=args.cluster_config,
                cluster_profile=args.cluster_profile,
                smoke_run_dir=smoke_run_dir,
                smoke_snapshot=args.smoke_snapshot,
                report_id=args.report_id,
                out_dir=args.out_dir,
                render_markdown=not args.no_markdown,
            )
            _emit(result)
            return _EXIT_OK

        if args.cmd == "show":
            print(show(args.report_id, out_dir=args.out_dir))
            return _EXIT_OK

    except _ReportError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return _EXIT_USAGE
    except Exception as exc:  # imported lazily so module always loads
        from pilot.tools._schema import SchemaValidationError
        if isinstance(exc, SchemaValidationError):
            _emit(_failure("TOOL_ERROR", f"schema validation failed: {exc}"))
            return _EXIT_USAGE
        raise

    return _EXIT_USAGE


if __name__ == "__main__":
    sys.exit(_cli())
