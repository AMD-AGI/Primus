"""pilot.tools.diagnose — bottleneck classifier (DIAGNOSE stage).

Authoritative protocol: ``skills/workflow/diagnose.md`` (rules R0..R5,
threshold table §10, axis catalog `axis_taxonomy.md`).

This is a deterministic rule engine — given the same RunSnapshot,
ClusterProfile, and Plan, the output is bit-exactly reproducible. There is no
LLM call inside this module; the agent only consults the recommended skills
referenced in the report.

CLI surface (stable contract per skills/workflow/diagnose.md §12)::

    python -m pilot.tools.diagnose run \\
        --snapshot         <run_snapshot.yaml> \\
        --cluster-profile  <cluster_profile.yaml> \\
        --plan             <plan.effective.yaml> \\
        [--champion-snapshot <champion_run_snapshot.yaml>] \\
        [--plan-graph        <plan_graph.yaml>] \\
        [--profile           <profiler.json>] \\
        [--thresholds        <thresholds.yaml>] \\
        [--out               <output_path.yaml>]

All outputs:
- emit a single JSON document on stdout matching ``schemas/diagnosis_report.schema.json``
- log progress to stderr
- exit 0 on success, 1 on stage-failed (failure routing returned non-bottleneck),
  2 on usage error, 3 on TOOL_ERROR (NotImplementedError / dep missing /
  schema validation failed).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.tools import constraint as _constraint


# ---------------------------------------------------------------------------
# Path anchoring
# ---------------------------------------------------------------------------

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/
_DEFAULT_THRESHOLDS_PATH: Path = _PILOT_ROOT / "state" / "thresholds" / "diagnose.yaml"


def _resolve_pilot_path(p: str | Path) -> Path:
    """Resolve `p` against CWD first; if missing, fall back to the pilot/ root.

    This mirrors how the rest of pilot.tools.* (submit, observe) treat paths:
    a leading "state/..." string works both when invoked from the repo root
    AND when invoked from anywhere with the pilot/state-relative form.
    """
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    cwd_path = Path.cwd() / pp
    if cwd_path.exists():
        return cwd_path
    pilot_path = _PILOT_ROOT / pp
    if pilot_path.exists():
        return pilot_path
    return cwd_path  # fall back to CWD-relative for the "file not found" message


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _DiagnoseError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# YAML/JSON loading
# ---------------------------------------------------------------------------


def _yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise _DiagnoseError("DEP_MISSING", f"PyYAML required: {exc}") from exc
    return yaml


def _load_mapping(path: str | Path) -> dict[str, Any]:
    p = _resolve_pilot_path(path)
    if not p.exists():
        raise _DiagnoseError("USAGE", f"file not found: {p}")
    text = p.read_text()
    if p.suffix.lower() == ".json":
        data: Any = json.loads(text)
    else:
        data = _yaml().safe_load(text)
    if not isinstance(data, dict):
        raise _DiagnoseError("USAGE", f"{p} must contain a YAML/JSON mapping")
    return data


# ---------------------------------------------------------------------------
# Default thresholds (skills/workflow/diagnose.md §10)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    "MEM_TIGHT_PCT": 0.92,
    "MEM_FRAGMENTATION_RATIO": 1.4,
    "BUBBLE_HIGH": 0.15,
    "COMM_HIGH": 0.25,
    "COMPUTE_PEAK_LOW": 0.30,
    "COMPUTE_PEAK_VERY_LOW": 0.15,
    "COMPUTE_PEAK_HIGH": 0.55,
    "REGRESSION_PCT": 1.05,
    "STALE_PROFILE_DAYS": 7.0,
    "MEASUREMENT_NOISE_CV": 0.05,
    "DROP_FIRST_ITERS": 2.0,
}


def _load_thresholds(path: str | Path | None) -> dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    if path is None:
        if _DEFAULT_THRESHOLDS_PATH.exists():
            try:
                override = _load_mapping(_DEFAULT_THRESHOLDS_PATH)
                for k, v in override.items():
                    if k in out and isinstance(v, (int, float)):
                        out[k] = float(v)
            except _DiagnoseError:
                pass
        return out
    override = _load_mapping(path)
    for k, v in override.items():
        if k in out and isinstance(v, (int, float)):
            out[k] = float(v)
        else:
            print(
                f"diagnose: ignoring unknown threshold key {k!r}", file=sys.stderr,
            )
    return out


# ---------------------------------------------------------------------------
# Axis catalog (kept in skills/workflow/axis_taxonomy.md as the source-of-truth
# for humans; we mirror only the structural facts the engine needs to validate
# emitted axes).
# ---------------------------------------------------------------------------

AXIS_CATALOG: dict[str, str] = {
    # parallelism (structural)
    "tensor_model_parallel_size": "structural",
    "pipeline_model_parallel_size": "structural",
    "expert_model_parallel_size": "structural",
    "context_parallel_size": "structural",
    "virtual_pipeline_model_parallel_size": "structural",
    "micro_batch_size": "structural",
    "global_batch_size": "structural",
    "seq_length": "structural",
    # recompute / memory
    "recompute_granularity": "strongly_local",
    "recompute_method": "strongly_local",
    "recompute_num_layers": "strongly_local",
    "optimizer_offload": "strongly_local",
    "MOE_BUFFER_PCT": "weakly_local",
    # comm
    "overlap_grad_reduce": "weakly_local",
    "overlap_param_gather": "weakly_local",
    "gradient_accumulation_fusion": "weakly_local",
    "turbo_deepep_use_comm_stream": "weakly_local",
    "turbo_deepep_num_cu": "weakly_local",
    "moe_shared_expert_overlap": "weakly_local",
    "moe_router_force_load_balancing": "weakly_local",
    "NCCL_BUFFSIZE": "strongly_local",
    "NCCL_MIN_NCHANNELS": "strongly_local",
    "NCCL_NET_GDR_LEVEL": "weakly_local",
    "NCCL_IB_DISABLE": "strongly_local",
    "RCCL_MSCCL_ENABLE": "weakly_local",
    "NCCL_IB_HCA": "cluster_shared",
    # compute kernels
    "attention_kernel": "weakly_local",
    "MOE_PERMUTE_FUSION": "strongly_local",
    "fp8_e4m3_fnuz": "strongly_local",
    # allocator
    "PYTORCH_HIP_ALLOC_CONF": "weakly_local",
    "PYTORCH_CUDA_ALLOC_CONF": "weakly_local",
}


# Hard-constraint deny-list (mutex pairs). The engine still emits these axes
# as candidates only when the *base plan* state would not violate the mutex.
HARD_CONSTRAINTS: list[tuple[str, Any, str, Any, str]] = [
    # (axis_a, value_a, axis_b, value_b, learn_seed_id)
    ("use_turbo_deepep", True, "moe_shared_expert_overlap", True,
     "deepep_x_sharedovrlp_mutex"),
    ("use_turbo_deepep", True, "moe_router_force_load_balancing", False,
     "deepep_x_real_router_hang"),
]


# ---------------------------------------------------------------------------
# Plan extraction
# ---------------------------------------------------------------------------


def _extract_overrides(plan: dict[str, Any]) -> dict[str, Any]:
    if "modules" in plan:
        return (
            plan.get("modules", {})
            .get("pre_trainer", {})
            .get("overrides", {})
            or {}
        )
    if isinstance(plan.get("overrides"), dict):
        return plan["overrides"]
    return plan


_ENV_DEFAULT_RE = re.compile(r"^\$\{[^:}]+:(?P<default>[^}]+)\}$")


def _to_int(value: Any, default: int = 1) -> int:
    if isinstance(value, str):
        m = _ENV_DEFAULT_RE.match(value.strip())
        if m:
            value = m.group("default")
    if value in (None, "null", "None", ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _plan_axes(overrides: dict[str, Any]) -> dict[str, int]:
    return {
        "tp": _to_int(overrides.get("tensor_model_parallel_size"), 1),
        "pp": _to_int(overrides.get("pipeline_model_parallel_size"), 1),
        "ep": _to_int(overrides.get("expert_model_parallel_size"), 1),
        "cp": _to_int(overrides.get("context_parallel_size"), 1),
        "mbs": _to_int(overrides.get("micro_batch_size"), 1),
        "gbs": _to_int(overrides.get("global_batch_size"), 1),
        "seq": _to_int(overrides.get("seq_length"), 1),
    }


def _plan_uses_alltoall(overrides: dict[str, Any]) -> bool:
    if _to_int(overrides.get("expert_model_parallel_size"), 1) > 1:
        return True
    if _to_int(overrides.get("num_experts"), 0) > 0:
        return True
    if overrides.get("use_turbo_deepep") is True:
        return True
    if overrides.get("multi_latent_attention") is True and _to_int(overrides.get("expert_model_parallel_size"), 1) > 1:
        return True
    return False


def _plan_uses_allreduce_heavy(overrides: dict[str, Any]) -> bool:
    return _to_int(overrides.get("tensor_model_parallel_size"), 1) > 1


def _plan_dtype_for_peak(overrides: dict[str, Any], cluster_profile: dict[str, Any]) -> tuple[str, float | None]:
    """Pick the right hardware peak for the plan's compute dtype.

    Falls back to BF16 peak if FP8 peak isn't measured, even when the plan is
    FP8 (this is the common case today). The fallback is recorded in `meta`.
    """
    compute = (cluster_profile or {}).get("compute") or {}
    bf16 = compute.get("peak_tflops_bf16")
    fp8 = compute.get("peak_tflops_fp8")
    is_fp8 = (
        overrides.get("fp8") is True
        or overrides.get("fp8_format") not in (None, "no", "false", "")
        or "fp8" in str(overrides.get("autocast_dtype") or "").lower()
    )
    if is_fp8 and isinstance(fp8, (int, float)) and fp8 > 0:
        return "fp8", float(fp8)
    if isinstance(bf16, (int, float)) and bf16 > 0:
        return "bf16", float(bf16)
    return "unknown", None


# ---------------------------------------------------------------------------
# Snapshot signal extraction
# ---------------------------------------------------------------------------


_RE_MEM_RATIO = re.compile(r"usage_ratio:\s*([0-9.]+)\s*%")


def _signals_from_snapshot(snapshot: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    metrics = snapshot.get("metrics") or {}
    history = metrics.get("history") or {}
    iter_ms = list(history.get("iter_time_ms") or [])
    tflops = list(history.get("tflops") or [])
    drop = int(thresholds["DROP_FIRST_ITERS"])

    iter_ms_steady = [v for v in iter_ms[drop:] if isinstance(v, (int, float))]
    tflops_steady = [v for v in tflops[drop:] if isinstance(v, (int, float))]

    iter_ms_median: float | None = None
    tflops_median: float | None = None
    iter_ms_cv: float | None = None
    if iter_ms_steady:
        iter_ms_median = float(statistics.median(iter_ms_steady))
        if len(iter_ms_steady) >= 2:
            mean = statistics.fmean(iter_ms_steady)
            sd = statistics.pstdev(iter_ms_steady)
            if mean > 0:
                iter_ms_cv = sd / mean
    if tflops_steady:
        tflops_median = float(statistics.median(tflops_steady))

    iters_observed = len(iter_ms)

    # Memory pct: try to recover from the train.log (snapshot doesn't carry it
    # natively as of run_snapshot.schema v1.0). Best-effort.
    mem_pct = _try_extract_mem_pct(snapshot)

    return {
        "iter_ms_steady_median": iter_ms_median,
        "tflops_steady_median": tflops_median,
        "iter_ms_cv": iter_ms_cv,
        "iters_observed": iters_observed,
        "mem_pct": mem_pct,
    }


def _try_extract_mem_pct(snapshot: dict[str, Any]) -> float | None:
    log = (snapshot.get("log") or {}).get("ref")
    if not log:
        return None
    p = Path(log)
    if not p.exists() or not p.is_file():
        return None
    try:
        # Tail at most 256 KiB to keep this cheap.
        size = p.stat().st_size
        cap = 256 * 1024
        with p.open("rb") as f:
            f.seek(max(0, size - cap))
            blob = f.read()
    except OSError:
        return None
    text = blob.decode("utf-8", errors="replace")
    matches = _RE_MEM_RATIO.findall(text)
    if not matches:
        return None
    try:
        last = float(matches[-1]) / 100.0
    except ValueError:
        return None
    if 0.0 <= last <= 1.0:
        return last
    return None


# ---------------------------------------------------------------------------
# Bubble estimate (fallback when no profiler)
# ---------------------------------------------------------------------------


def _bubble_estimate(plan_axes: dict[str, int], cluster_profile: dict[str, Any]) -> tuple[float, str]:
    """Closed-form estimate of pipeline bubble ratio.

    With pp stages and N microbatches per global step,
    bubble = (pp - 1) / (N + pp - 1).  Returns 0.0 and source 'n/a' when pp <= 1.
    """
    pp = plan_axes["pp"]
    if pp <= 1:
        return 0.0, "n/a"
    nodes = (cluster_profile or {}).get("nodes_total") or 1
    gpus = (cluster_profile or {}).get("gpus_per_node") or 1
    world = max(1, int(nodes) * int(gpus))
    model_parallel = max(1, plan_axes["tp"] * plan_axes["pp"] * plan_axes["ep"] * plan_axes["cp"])
    dp = max(1, world // model_parallel)
    mbs = plan_axes["mbs"]
    gbs = plan_axes["gbs"]
    n_micro = max(1, gbs // max(1, mbs * dp))
    return (pp - 1) / max(1, (n_micro + pp - 1)), "fallback"


# ---------------------------------------------------------------------------
# Profiler trace ingest (best-effort; opens the door for future work)
# ---------------------------------------------------------------------------


def _profile_signals(profile_path: str | None) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "comm_ratio": None,
        "bubble_ratio": None,
        "overlap_ratio": None,
        "mem_reserved_over_alloc": None,
    }
    if not profile_path:
        return out
    p = _resolve_pilot_path(profile_path)
    if not p.exists():
        return out
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out
    for k in out:
        v = data.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Re-entry triggers (skills/workflow/diagnose.md §8)
# ---------------------------------------------------------------------------


def _profile_age_days(cluster_profile: dict[str, Any]) -> float | None:
    ts = cluster_profile.get("collected_at")
    if not isinstance(ts, str):
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)


def _check_reentry(
    cluster_profile: dict[str, Any],
    plan_axes: dict[str, int],
    thresholds: dict[str, float],
) -> dict[str, Any] | None:
    age = _profile_age_days(cluster_profile or {})
    if age is not None and age > thresholds["STALE_PROFILE_DAYS"]:
        return {
            "to": "PREFLIGHT",
            "reason": f"reentry_stale: ClusterProfile age {age:.1f}d > {thresholds['STALE_PROFILE_DAYS']:.0f}d",
            "counts_against_budget": False,
            "hint": "re-collect with --reason reentry_stale",
        }
    nodes = int((cluster_profile or {}).get("nodes_total") or 1)
    gpus = int((cluster_profile or {}).get("gpus_per_node") or 1)
    world = max(1, nodes * gpus)
    mp = plan_axes["tp"] * plan_axes["pp"] * plan_axes["ep"] * plan_axes["cp"]
    if mp > world or world % max(1, mp) != 0:
        return {
            "to": "PROJECTION",
            "reason": (
                f"plan parallelism (tp*pp*ep*cp={mp}) not representable in "
                f"world={world} (nodes={nodes}, gpus_per_node={gpus})"
            ),
            "counts_against_budget": False,
            "hint": "re-derive Plan from Projection with current cluster",
        }
    return None


# ---------------------------------------------------------------------------
# Failure routing (R0; skills/workflow/diagnose.md §7)
# ---------------------------------------------------------------------------


def _failure_routing(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    failure = _constraint.diagnose_failure(snapshot)
    kind = (failure.get("kind") or "UNKNOWN").upper()
    status = snapshot.get("status")
    symptoms = snapshot.get("symptoms") or {}

    # Explicit hang detection — observe sets status=hung when silent_for_s
    # exceeds hang_threshold_s on a live process.
    if status == "hung" or symptoms.get("hang_suspected"):
        return {
            "bottleneck": "COMM_BOUND",
            "bottleneck_extended": "HANG",
            "confidence": 0.90,
            "evidence": [
                f"status={status} hang_suspected={bool(symptoms.get('hang_suspected'))}; silent_for_s={(snapshot.get('progress') or {}).get('silent_for_s')}",
            ],
            "transition": {
                "to": "OPTIMIZE_LOOP.REPLAN",
                "reason": "HANG: mark candidate dead, derive a different axis change",
                "counts_against_budget": False,
            },
            "rule_id": "R0_HANG",
        }
    if status == "killed":
        return {
            "bottleneck": "COMPUTE_BOUND",
            "bottleneck_extended": "CANCELLED",
            "confidence": 1.00,
            "evidence": ["status=killed (operator/orchestrator cancelled)"],
            "transition": {
                "to": "OPTIMIZE_LOOP.REPLAN",
                "reason": "CANCELLED: skip this candidate",
                "counts_against_budget": False,
            },
            "rule_id": "R0_CANCELLED",
        }
    if status == "unknown":
        return {
            "bottleneck": "COMPUTE_BOUND",
            "bottleneck_extended": "UNKNOWN",
            "confidence": 0.30,
            "evidence": ["status=unknown; observe could not determine terminal state"],
            "transition": {
                "to": "WAIT",
                "reason": "re-snapshot before classifying",
                "counts_against_budget": False,
            },
            "rule_id": "R0_UNKNOWN",
        }
    if status not in (None, "completed"):
        if kind == "OOM":
            return {
                "bottleneck": "MEMORY_BOUND",
                "bottleneck_extended": "MEMORY_BOUND",
                "confidence": 0.95,
                "evidence": ["OOM detected via observe.symptoms.oom_detected"],
                "recommended_skills": [
                    "skills/optimization/memory/recompute.md",
                    "skills/optimization/memory/offload.md",
                ],
                "transition": {
                    "to": "OPTIMIZE_LOOP.REPLAN",
                    "reason": "OOM: mark dead and lower memory pressure",
                    "counts_against_budget": False,
                },
                "rule_id": "R0_OOM",
            }
        if kind == "NUMERICAL":
            return {
                "bottleneck": "COMPUTE_BOUND",
                "bottleneck_extended": "NUMERICAL",
                "confidence": 1.00,
                "evidence": ["loss became NaN/Inf (loss_finite=false)"],
                "transition": {
                    "to": "ABORT",
                    "reason": "NUMERICAL: escalate to human; do not auto-tune",
                    "counts_against_budget": False,
                },
                "rule_id": "R0_NUMERICAL",
            }
        if kind == "HANG":
            return {
                "bottleneck": "COMM_BOUND",
                "bottleneck_extended": "HANG",
                "confidence": 0.85,
                "evidence": ["NCCL/RCCL error or hang detected"],
                "transition": {
                    "to": "PREFLIGHT",
                    "reason": "HANG: re-probe env baseline before retrying",
                    "counts_against_budget": False,
                },
                "rule_id": "R0_HANG_NCCL",
            }
        if kind == "TOOL_ERROR":
            # Distinguish CUDA vs Python: CUDA → CLUSTER, Python → INVALID_CONFIG.
            if symptoms.get("cuda_error"):
                return {
                    "bottleneck": "COMPUTE_BOUND",
                    "bottleneck_extended": "INVALID_CONFIG",
                    "confidence": 1.00,
                    "evidence": ["CUDA/HIP runtime error"],
                    "transition": {
                        "to": "PREFLIGHT",
                        "reason": "CUDA/HIP runtime error; re-collect cluster baseline",
                        "counts_against_budget": False,
                    },
                    "rule_id": "R0_CUDA_ERROR",
                }
            # Heuristic: python tracebacks containing "support" / "supported" /
            # "config" / "argument" / "incompatible" are config errors. Otherwise
            # we fall through to the generic stage-failed routing.
            evidence_lines = [e.get("line", "") for e in symptoms.get("evidence", []) if isinstance(e, dict)]
            joined = " ".join(evidence_lines).lower()
            if any(token in joined for token in (
                "not support", "not supported", "incompatible",
                "invalid config", "missing required", "must be",
            )):
                return {
                    "bottleneck": "COMPUTE_BOUND",
                    "bottleneck_extended": "INVALID_CONFIG",
                    "confidence": 0.95,
                    "evidence": [
                        "Python error suggests INVALID_CONFIG (matched keyword in traceback)",
                    ],
                    "transition": {
                        "to": "OPTIMIZE_LOOP.REPLAN",
                        "reason": "INVALID_CONFIG: mark candidate dead",
                        "counts_against_budget": False,
                    },
                    "rule_id": "R0_INVALID_CONFIG",
                }
        # Fall-through: generic stage failure
        return {
            "bottleneck": "COMPUTE_BOUND",
            "bottleneck_extended": "UNKNOWN",
            "confidence": 0.40,
            "evidence": [f"status={status} kind={kind}"],
            "transition": {
                "to": "OPTIMIZE_LOOP.REPLAN",
                "reason": f"stage failed (status={status}, kind={kind})",
                "counts_against_budget": False,
            },
            "rule_id": "R0_GENERIC_FAILURE",
        }
    return None


# ---------------------------------------------------------------------------
# Bottleneck rules R1..R5 (skills/workflow/diagnose.md §5)
# ---------------------------------------------------------------------------


def _classify(
    snapshot: dict[str, Any],
    cluster_profile: dict[str, Any],
    plan: dict[str, Any],
    overrides: dict[str, Any],
    plan_axes: dict[str, int],
    signals: dict[str, Any],
    profile_signals: dict[str, float | None],
    champion_signals: dict[str, Any] | None,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    evidence: list[str] = []
    env_suspect: list[dict[str, str]] = []
    recommended_skills: list[str] = []
    bottleneck_extended: str | None = None
    rule_id = "R4_DEFAULT_COMPUTE"
    bottleneck = "COMPUTE_BOUND"
    base_conf = 0.55

    mem_pct = signals["mem_pct"]
    iter_ms = signals["iter_ms_steady_median"]
    tflops = signals["tflops_steady_median"]

    dtype, peak_tflops = _plan_dtype_for_peak(overrides, cluster_profile)
    util: float | None = None
    if isinstance(tflops, (int, float)) and isinstance(peak_tflops, (int, float)) and peak_tflops > 0:
        util = tflops / peak_tflops

    # R1 - Memory tightness
    symptoms = snapshot.get("symptoms") or {}
    if symptoms.get("oom_detected"):
        bottleneck = "MEMORY_BOUND"
        rule_id = "R1_OOM"
        base_conf = 0.95
        evidence.append("symptoms.oom_detected=true")
        recommended_skills += [
            "skills/optimization/memory/recompute.md",
            "skills/optimization/memory/offload.md",
        ]
    elif (frag := profile_signals.get("mem_reserved_over_alloc")) and frag > thresholds["MEM_FRAGMENTATION_RATIO"]:
        bottleneck = "MEMORY_BOUND"
        rule_id = "R1_FRAGMENTATION"
        base_conf = 0.70
        evidence.append(
            f"mem_reserved/mem_alloc={frag:.2f} > {thresholds['MEM_FRAGMENTATION_RATIO']}",
        )
        env_suspect.append({
            "flag": "PYTORCH_HIP_ALLOC_CONF",
            "reason": f"fragmentation ratio {frag:.2f} > {thresholds['MEM_FRAGMENTATION_RATIO']}",
            "hint": "skills/optimization/memory/fragmentation.md",
        })
    elif isinstance(mem_pct, (int, float)) and mem_pct > thresholds["MEM_TIGHT_PCT"]:
        bottleneck = "MEMORY_BOUND"
        rule_id = "R1_MEM_TIGHT"
        base_conf = 0.80
        evidence.append(
            f"peak mem_pct={mem_pct:.2f} > {thresholds['MEM_TIGHT_PCT']:.2f}",
        )
        recommended_skills += [
            "skills/optimization/memory/recompute.md",
        ]

    # R2 - Pipeline bubble (pp > 1)
    bubble_source = "n/a"
    bubble = None
    if bottleneck != "MEMORY_BOUND":
        if isinstance(profile_signals.get("bubble_ratio"), (int, float)):
            bubble = float(profile_signals["bubble_ratio"])
            bubble_source = "profiler"
        elif plan_axes["pp"] > 1:
            bubble, bubble_source = _bubble_estimate(plan_axes, cluster_profile)
        if bubble is not None and plan_axes["pp"] > 1 and bubble > thresholds["BUBBLE_HIGH"]:
            bottleneck = "PIPELINE_BOUND"
            rule_id = f"R2_BUBBLE_HIGH({bubble_source})"
            base_conf = 0.85 if bubble_source == "profiler" else 0.70
            evidence.append(
                f"bubble_ratio={bubble:.2f} > {thresholds['BUBBLE_HIGH']:.2f} ({bubble_source})",
            )
            recommended_skills += [
                "skills/optimization/pipeline/vpp.md",
                "skills/optimization/pipeline/microbatch.md",
            ]

    # R3 - Communication
    if bottleneck == "COMPUTE_BOUND":   # not yet claimed by R1/R2
        comm_ratio = profile_signals.get("comm_ratio")
        if isinstance(comm_ratio, (int, float)) and comm_ratio > thresholds["COMM_HIGH"]:
            bottleneck = "COMM_BOUND"
            rule_id = "R3a_PROFILER_COMM"
            base_conf = 0.90
            evidence.append(
                f"comm_ratio={comm_ratio:.2f} > {thresholds['COMM_HIGH']:.2f} (profiler)",
            )
            recommended_skills += [
                "skills/optimization/comm/SKILL.md",
                "skills/optimization/comm/overlap.md",
            ]
        elif (
            isinstance(util, float)
            and util < thresholds["COMPUTE_PEAK_LOW"]
            and (_plan_uses_alltoall(overrides) or _plan_uses_allreduce_heavy(overrides) or plan_axes["cp"] > 1)
        ):
            bottleneck = "COMM_BOUND"
            rule_id = "R3b_PROFILER_LESS_COMM"
            base_conf = 0.80 if util < thresholds["COMPUTE_PEAK_VERY_LOW"] else 0.75
            evidence.append(
                f"tflops/peak_{dtype}={util:.3f} < {thresholds['COMPUTE_PEAK_LOW']:.2f}; plan has heavy collectives",
            )
            recommended_skills += ["skills/optimization/comm/SKILL.md"]
            if _plan_uses_alltoall(overrides):
                recommended_skills += ["skills/optimization/moe/dispatch.md"]
            if _plan_uses_allreduce_heavy(overrides):
                recommended_skills += ["skills/optimization/comm/bucket.md"]

    # R4 - Compute headroom (only if nothing claimed yet)
    if bottleneck == "COMPUTE_BOUND":
        if isinstance(util, float) and util >= thresholds["COMPUTE_PEAK_HIGH"]:
            rule_id = "R4_COMPUTE_BOUND_HIGH"
            base_conf = 0.85
            evidence.append(
                f"tflops/peak_{dtype}={util:.3f} >= {thresholds['COMPUTE_PEAK_HIGH']:.2f}",
            )
            recommended_skills += [
                "skills/optimization/compute/mbs.md",
                "skills/optimization/compute/parallel.md",
            ]
        elif isinstance(util, float):
            rule_id = "R4_COMPUTE_BOUND_MID"
            base_conf = 0.55
            evidence.append(
                f"tflops/peak_{dtype}={util:.3f} in [{thresholds['COMPUTE_PEAK_LOW']:.2f}, {thresholds['COMPUTE_PEAK_HIGH']:.2f})",
            )
            recommended_skills += ["skills/optimization/compute/SKILL.md"]
        else:
            rule_id = "R4_COMPUTE_BOUND_NO_PEAK"
            base_conf = 0.40
            evidence.append("no measured peak available; defaulted to COMPUTE_BOUND with low confidence")

    # R5 - Regression vs champion
    if (
        champion_signals
        and isinstance(iter_ms, float)
        and isinstance(champion_signals.get("iter_ms_steady_median"), float)
    ):
        ch = champion_signals["iter_ms_steady_median"]
        if iter_ms > ch * thresholds["REGRESSION_PCT"]:
            bottleneck_extended = "REGRESSION"
            evidence.append(
                f"iter_ms={iter_ms:.0f} > champion {ch:.0f} × {thresholds['REGRESSION_PCT']:.2f} (regression)",
            )

    # Confidence adjustments
    confidence = base_conf
    cv = signals.get("iter_ms_cv")
    if isinstance(cv, float) and cv > thresholds["MEASUREMENT_NOISE_CV"]:
        confidence -= 0.10
        evidence.append(f"measurement noise cv={cv:.3f} > {thresholds['MEASUREMENT_NOISE_CV']:.2f}")
    if signals.get("iters_observed", 0) < 5:
        confidence -= 0.10
        evidence.append(f"iters_observed={signals.get('iters_observed')} < 5")
    if all(not symptoms.get(k) for k in ("nccl_error", "cuda_error", "python_error", "hang_suspected")):
        confidence += 0.05
    confidence = max(0.0, min(1.0, round(confidence, 2)))

    return {
        "bottleneck": bottleneck,
        "bottleneck_extended": bottleneck_extended or bottleneck,
        "confidence": confidence,
        "evidence": evidence,
        "env_suspect": env_suspect,
        "recommended_skills": list(dict.fromkeys(recommended_skills)),
        "rule_id": rule_id,
        "util": util,
        "dtype": dtype,
        "peak_tflops": peak_tflops,
        "bubble_ratio": bubble,
        "bubble_source": bubble_source,
    }


# ---------------------------------------------------------------------------
# Candidate axis production (skills/workflow/diagnose.md §9)
# ---------------------------------------------------------------------------


def _emit_axes(
    bottleneck: str,
    overrides: dict[str, Any],
    plan_axes: dict[str, int],
    cluster_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    nodes = int(cluster_profile.get("nodes_total") or 1)
    gpus = int(cluster_profile.get("gpus_per_node") or 1)
    world = max(1, nodes * gpus)
    mp = max(1, plan_axes["tp"] * plan_axes["pp"] * plan_axes["ep"] * plan_axes["cp"])
    dp = max(1, world // mp)

    out: list[dict[str, Any]] = []

    def add(axis: str, candidates: list[Any], gain: tuple[float, float], cost: float, why: str) -> None:
        atype = AXIS_CATALOG.get(axis)
        if atype is None:
            return
        if not _passes_hard_constraints(axis, candidates, overrides):
            return
        out.append({
            "axis": axis,
            "type": atype,
            "candidates": candidates,
            "expected_gain_band_pct": [gain[0], gain[1]],
            "est_cost_gpu_h": cost,
            "rationale": why,
        })

    if bottleneck == "COMM_BOUND":
        if overrides.get("turbo_deepep_use_comm_stream") is False:
            add("turbo_deepep_use_comm_stream", [True], (5, 15), 0.21,
                "default off; flipping enables overlap on a separate comm stream")
        cu = _to_int(overrides.get("turbo_deepep_num_cu"), 64)
        if cu == 64:
            add("turbo_deepep_num_cu", [80], (0, 5), 0.21,
                "deepEP CU pool widening; safe single-axis change for EP=8")
        if overrides.get("overlap_grad_reduce") is False:
            add("overlap_grad_reduce", [True], (0, 4), 0.21, "overlap grad reduce")
        if overrides.get("overlap_param_gather") is False:
            add("overlap_param_gather", [True], (0, 4), 0.21, "overlap param gather")
        # Holdout structurals
        if plan_axes["ep"] > 1 and plan_axes["mbs"] >= 2:
            add("micro_batch_size", [max(1, plan_axes["mbs"] - 2), plan_axes["mbs"] + 2],
                (-5, 12), 0.42,
                "amortize alltoall over more tokens (mbs+) or reduce per-step traffic (mbs-)")

    elif bottleneck == "PIPELINE_BOUND":
        add("virtual_pipeline_model_parallel_size", [2, 4], (5, 25), 0.25,
            "interleaving reduces bubble at small comm cost")
        gbs = plan_axes["gbs"]
        n_micro = max(1, gbs // max(1, plan_axes["mbs"] * dp))
        add("micro_batch_size", [max(1, plan_axes["mbs"] // 2)],
            (3, 12), 0.21, f"more microbatches (currently {n_micro}) reduces bubble")
        add("recompute_method", ["block"], (-5, 5), 0.21, "trade memory for finer pipeline scheduling")

    elif bottleneck == "MEMORY_BOUND":
        if overrides.get("recompute_granularity") in (None, "null"):
            add("recompute_granularity", ["selective"], (-5, 0), 0.21,
                "selective recompute saves activations cheaply")
        if overrides.get("recompute_granularity") == "selective":
            add("recompute_granularity", ["full"], (-15, -5), 0.21,
                "full recompute as last-resort memory hedge")
        add("PYTORCH_HIP_ALLOC_CONF",
            ["expandable_segments:True,max_split_size_mb:512"],
            (0, 5), 0.21,
            "reduces fragmentation; safe weakly_local change")
        if plan_axes["mbs"] > 1:
            add("micro_batch_size", [max(1, plan_axes["mbs"] // 2)],
                (-15, 0), 0.21, "halve mbs to relieve activation memory")

    elif bottleneck == "COMPUTE_BOUND":
        # Only suggest mbs+ when we have memory headroom (mem_pct < 0.80). The
        # caller passes mem via signals; we conservatively check via overrides
        # that recompute is off (otherwise increasing mbs probably won't fit).
        if overrides.get("recompute_granularity") in (None, "null"):
            add("micro_batch_size", [plan_axes["mbs"] + 2],
                (3, 10), 0.21, "increase per-step compute density")
        if overrides.get("gradient_accumulation_fusion") is False:
            add("gradient_accumulation_fusion", [True], (0, 3), 0.21,
                "fuse grad accumulation kernels")
        if overrides.get("attention_kernel") not in (None, "flash"):
            add("attention_kernel", ["flash"], (5, 20), 0.21,
                "FlashAttention typically dominates at sequence length > 1k")

    return out


def _passes_hard_constraints(axis: str, candidates: list[Any], overrides: dict[str, Any]) -> bool:
    """Refuse to emit a candidate that would trip a known hard constraint pair."""
    for axis_a, val_a, axis_b, val_b, _ in HARD_CONSTRAINTS:
        if axis == axis_b and val_b in candidates:
            if overrides.get(axis_a) == val_a:
                return False
        if axis == axis_a and val_a in candidates:
            if overrides.get(axis_b) == val_b:
                return False
    return True


def _suggest_strategy(candidate_axes: list[dict[str, Any]]) -> tuple[str, str]:
    if not candidate_axes:
        return "Per-Plan", "no axes; nothing to suggest"
    types = {a["type"] for a in candidate_axes}
    if "cluster_shared" in types:
        return "Champion-Challenger", "cluster_shared axis present; re-validate baseline before structural moves"
    if types == {"weakly_local"} and len(candidate_axes) <= 4:
        return "Per-Plan", "all weakly_local and <= 4 axes; explore each independently"
    if len(candidate_axes) > 6:
        return "Successive_Halving", f"{len(candidate_axes)} axes total; halve worst half each round"
    if types == {"structural"}:
        return "Per-Plan", "only structural axes; explore per-plan with explicit warning that bigger structural moves are coming"
    return "Per-Plan", "mixed types but small pool; per-plan is the cheapest exploration"


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run(
    snapshot: dict[str, Any],
    cluster_profile: dict[str, Any],
    plan: dict[str, Any],
    *,
    champion_snapshot: dict[str, Any] | None = None,
    plan_graph: dict[str, Any] | None = None,
    profile_path: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Public API: classify one snapshot. Output matches diagnosis_report.schema.json."""
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({k: v for k, v in thresholds.items() if k in DEFAULT_THRESHOLDS})

    overrides = _extract_overrides(plan)
    plan_axes = _plan_axes(overrides)

    # R0 — failure routing short-circuits everything else.
    routed = _failure_routing(snapshot)
    if routed is not None:
        report = _assemble_report(
            snapshot=snapshot,
            cluster_profile=cluster_profile,
            plan=plan,
            overrides=overrides,
            plan_axes=plan_axes,
            classification=routed,
            transition_override=routed.get("transition"),
            recommended_skills=routed.get("recommended_skills"),
            thresholds=th,
            signals=None,
            profile_signals={"comm_ratio": None, "bubble_ratio": None, "overlap_ratio": None, "mem_reserved_over_alloc": None},
            champion_signals=None,
        )
        return report

    # Re-entry triggers bypass classification.
    reentry = _check_reentry(cluster_profile, plan_axes, th)
    if reentry is not None:
        skeleton: dict[str, Any] = {
            "bottleneck": "COMPUTE_BOUND",
            "bottleneck_extended": "UNKNOWN",
            "confidence": 0.30,
            "evidence": [reentry["reason"]],
            "env_suspect": [],
            "recommended_skills": [],
            "rule_id": "REENTRY",
            "util": None, "dtype": "unknown", "peak_tflops": None,
            "bubble_ratio": None, "bubble_source": "n/a",
        }
        return _assemble_report(
            snapshot=snapshot, cluster_profile=cluster_profile, plan=plan,
            overrides=overrides, plan_axes=plan_axes, classification=skeleton,
            transition_override=reentry, recommended_skills=None, thresholds=th,
            signals=None,
            profile_signals={"comm_ratio": None, "bubble_ratio": None, "overlap_ratio": None, "mem_reserved_over_alloc": None},
            champion_signals=None,
        )

    signals = _signals_from_snapshot(snapshot, th)
    profile_signals = _profile_signals(profile_path)
    champion_signals = (
        _signals_from_snapshot(champion_snapshot, th) if champion_snapshot else None
    )

    classification = _classify(
        snapshot=snapshot, cluster_profile=cluster_profile, plan=plan,
        overrides=overrides, plan_axes=plan_axes, signals=signals,
        profile_signals=profile_signals, champion_signals=champion_signals,
        thresholds=th,
    )

    return _assemble_report(
        snapshot=snapshot, cluster_profile=cluster_profile, plan=plan,
        overrides=overrides, plan_axes=plan_axes, classification=classification,
        transition_override=None, recommended_skills=None,
        thresholds=th, signals=signals, profile_signals=profile_signals,
        champion_signals=champion_signals,
    )


def _assemble_report(
    *,
    snapshot: dict[str, Any],
    cluster_profile: dict[str, Any],
    plan: dict[str, Any],
    overrides: dict[str, Any],
    plan_axes: dict[str, int],
    classification: dict[str, Any],
    transition_override: dict[str, Any] | None,
    recommended_skills: list[str] | None,
    thresholds: dict[str, float],
    signals: dict[str, Any] | None,
    profile_signals: dict[str, float | None],
    champion_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    bottleneck = classification["bottleneck"]
    candidate_axes: list[dict[str, Any]] = []
    if transition_override is None:
        candidate_axes = _emit_axes(bottleneck, overrides, plan_axes, cluster_profile)
    strategy, strategy_why = _suggest_strategy(candidate_axes)

    if transition_override is None:
        # Default transition for normal classification: into REPLAN with
        # the env_suspect routing decided by Re-Plan.
        transition = {
            "to": "OPTIMIZE_LOOP.REPLAN",
            "reason": (
                "env_suspect detected; consider EnvSweep before structural moves"
                if classification.get("env_suspect")
                else f"classified as {bottleneck}; derive candidates from {strategy}"
            ),
            "counts_against_budget": True,
        }
    else:
        transition = dict(transition_override)
        transition.setdefault("counts_against_budget", False)

    skills = list(classification.get("recommended_skills") or [])
    if recommended_skills:
        skills = list(dict.fromkeys(list(recommended_skills) + skills))

    meta: dict[str, Any] = {
        "bottleneck_extended": classification.get("bottleneck_extended", bottleneck),
        "rule_id": classification.get("rule_id"),
        "iter_ms_steady_median": (signals or {}).get("iter_ms_steady_median"),
        "tflops_steady_median": (signals or {}).get("tflops_steady_median"),
        "tflops_pct_of_peak": classification.get("util"),
        "iter_ms_cv": (signals or {}).get("iter_ms_cv"),
        "iters_observed": (signals or {}).get("iters_observed"),
        "mem_pct": (signals or {}).get("mem_pct"),
        "bubble_ratio_estimate": classification.get("bubble_ratio"),
        "bubble_ratio_source": classification.get("bubble_source", "n/a"),
        "comm_ratio": profile_signals.get("comm_ratio") if profile_signals else None,
        "strategy_rationale": strategy_why,
        "thresholds": thresholds,
    }

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "snapshot_id": snapshot.get("run_id") or snapshot.get("snapshot_id") or "<unknown>",
        "bottleneck": bottleneck,
        "confidence": float(classification["confidence"]),
        "evidence": list(classification.get("evidence") or []),
        "recommended_skills": skills,
        "env_suspect": list(classification.get("env_suspect") or []),
        "candidate_axes": candidate_axes,
        "suggested_strategy": strategy,
        "suggested_transition": transition,
        "meta": meta,
    }

    # Validate against the schema before returning.
    try:
        from pilot.tools._schema import validate
        validate(report, "diagnosis_report")
    except ImportError:  # pragma: no cover
        pass
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK = 0
_EXIT_STAGE_FAILED = 1
_EXIT_USAGE = 2
_EXIT_TOOL_ERROR = 3


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="pilot.tools.diagnose",
        description="Classify a RunSnapshot into a DiagnosisReport (skills/workflow/diagnose.md).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="Run the diagnose engine on one snapshot.")
    p_run.add_argument("--snapshot", required=True, help="Path to RunSnapshot YAML/JSON.")
    p_run.add_argument("--cluster-profile", required=True, help="Path to ClusterProfile YAML.")
    p_run.add_argument("--plan", required=True, help="Path to plan.effective.yaml.")
    p_run.add_argument("--champion-snapshot", default=None)
    p_run.add_argument("--plan-graph", default=None)
    p_run.add_argument("--profile", default=None, help="Optional profiler trace JSON.")
    p_run.add_argument("--thresholds", default=None)
    p_run.add_argument("--out", default=None, help="If set, also writes the report to this YAML path.")

    args = p.parse_args()
    if args.cmd != "run":  # pragma: no cover
        return _EXIT_USAGE

    try:
        snapshot = _load_mapping(args.snapshot)
        cluster_profile = _load_mapping(args.cluster_profile)
        plan = _load_mapping(args.plan)
        champion = _load_mapping(args.champion_snapshot) if args.champion_snapshot else None
        plan_graph = _load_mapping(args.plan_graph) if args.plan_graph else None
        thresholds = _load_thresholds(args.thresholds)

        report = run(
            snapshot=snapshot,
            cluster_profile=cluster_profile,
            plan=plan,
            champion_snapshot=champion,
            plan_graph=plan_graph,
            profile_path=args.profile,
            thresholds=thresholds,
        )
    except _DiagnoseError as exc:
        _emit({"stage": "DIAGNOSE", "status": "failed", "failure": {"kind": exc.kind, "message": str(exc)}})
        return _EXIT_USAGE if exc.kind == "USAGE" else _EXIT_TOOL_ERROR
    except Exception as exc:
        from pilot.tools._schema import SchemaValidationError
        if isinstance(exc, SchemaValidationError):
            _emit({"stage": "DIAGNOSE", "status": "failed", "failure": {"kind": "TOOL_ERROR", "message": f"schema validation failed: {exc}"}})
            return _EXIT_TOOL_ERROR
        raise

    _emit(report)

    if args.out:
        out = _resolve_pilot_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_yaml().safe_dump(report, sort_keys=False))

    transition = (report.get("suggested_transition") or {}).get("to", "OPTIMIZE_LOOP.REPLAN")
    if transition in ("ABORT",):
        return _EXIT_STAGE_FAILED
    return _EXIT_OK


if __name__ == "__main__":
    sys.exit(_cli())
