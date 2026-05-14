"""pilot.tools.diagnose — trace-driven bottleneck classifier (v2).

Authoritative protocol: ``skills/workflow/diagnose.md`` (rules R0..R5,
threshold table §10, axis catalog `axis_taxonomy.md`).

This is a deterministic rule engine — given the same trace_analysis.json
(plus optional snapshot/plan/champion), the output is bit-exactly
reproducible. There is no LLM call inside this module; the agent only
consults the recommended skills referenced in the report.

Primary input: ``trace_analysis.json`` produced by
``pilot.tools.trace_analyze`` (per ``skills/workflow/trace_analysis.md``).

CLI surface (stable contract per skills/workflow/diagnose.md §12)::

    python -m pilot.tools.diagnose run \\
        --trace-analysis    <trace_analysis.json>     [REQUIRED] \\
        [--snapshot          <run_snapshot.yaml>] \\
        [--plan              <plan.effective.yaml>] \\
        [--cluster-profile   <cluster_profile.yaml>] \\
        [--champion-snapshot <champion_run_snapshot.yaml>] \\
        [--plan-graph        <plan_graph.yaml>] \\
        [--thresholds        <thresholds.yaml>] \\
        [--out               <output_path.json>]

Outputs:
- emit a single JSON document on stdout matching
  ``schemas/diagnosis_report.schema.json``
- log progress to stderr
- exit 0 on success, 1 on stage-failed (failure routing returned a
  non-bottleneck), 2 on usage error, 3 on TOOL_ERROR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path anchoring (mirrors pilot.tools.* convention)
# ---------------------------------------------------------------------------

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/


def _resolve(p: str | Path) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    cwd_path = Path.cwd() / pp
    if cwd_path.exists():
        return cwd_path
    pilot_path = _PILOT_ROOT / pp
    if pilot_path.exists():
        return pilot_path
    return cwd_path


# ---------------------------------------------------------------------------
# Errors and exit codes
# ---------------------------------------------------------------------------


class _DiagError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


_EXIT_OK = 0
_EXIT_STAGE_FAILED = 1
_EXIT_USAGE = 2
_EXIT_TOOL_ERROR = 3


# ---------------------------------------------------------------------------
# Threshold defaults (skills/workflow/diagnose.md §10)
# ---------------------------------------------------------------------------


_DEFAULT_THRESHOLDS: dict[str, float] = {
    "MEM_TIGHT_PCT": 0.92,
    "BUBBLE_HIGH": 0.10,
    "BUBBLE_MED": 0.05,
    "BUBBLE_LOW": 0.02,
    "COMM_HIGH": 0.20,
    "COMM_VERY_HIGH": 0.30,
    "COMM_LOW": 0.10,
    "PURE_COMM_HIGH": 0.10,
    "SERIAL_COMM_HIGH_MS": 5.0,
    "COMPUTE_HIGH": 0.80,
    "COMPUTE_GEMM_HIGH": 0.70,
    "OVERLAP_LOW": 0.02,
    "REGRESSION_PCT": 1.05,
    "STALE_PROFILE_DAYS": 7.0,
}


def _load_thresholds(path: str | Path | None) -> dict[str, float]:
    """Merge user override (YAML) on top of defaults. Returns a fresh dict."""
    out = dict(_DEFAULT_THRESHOLDS)
    if path is None:
        return out
    p = _resolve(path)
    if not p.exists():
        return out
    try:
        import yaml  # type: ignore
    except ImportError:
        return out
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except Exception:  # pragma: no cover
        return out
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        if k in out and isinstance(v, (int, float)):
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        raise _DiagError("USAGE", f"file not found: {p}")
    return json.loads(p.read_text())


def _load_yaml_or_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = _resolve(path)
    if not p.exists():
        raise _DiagError("USAGE", f"file not found: {p}")
    text = p.read_text()
    if p.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _DiagError("DEP_MISSING", f"PyYAML required for {p}: {exc}") from exc
    return yaml.safe_load(text) or {}


# ---------------------------------------------------------------------------
# Trace evidence extraction
# ---------------------------------------------------------------------------


def _build_evidence(trace: dict[str, Any]) -> dict[str, Any]:
    """Pull the fields the rule engine needs into one flat dict."""
    # trace_analysis.json schema may put per-rank reports in `per_rank` or
    # at the top level (single-rank case).
    if "per_rank" in trace and isinstance(trace["per_rank"], list) and trace["per_rank"]:
        analysis = trace["per_rank"][0]
    else:
        analysis = trace

    ratios = analysis.get("ratios") or {}
    cb = analysis.get("cost_breakdown") or {}
    pl = analysis.get("pipeline_timeline") or {}
    phases = analysis.get("phases") or {}
    kernels_full = analysis.get("kernels_full") or []
    overlap_block = analysis.get("overlap") or {}

    iter_ms = analysis.get("iter_wallclock_ms") or 0.0

    # Stream observation: how many GPU streams did the trace see, and do
    # comm_moe_dispatch and compute_gemm SHARE a stream?
    stream_count = pl.get("stream_count_total") or len(pl.get("streams") or [])
    moe_streams: set[int] = set()
    gemm_streams: set[int] = set()
    for k in kernels_full:
        b = k.get("bucket")
        for s in k.get("streams") or []:
            if b == "comm_moe_dispatch":
                moe_streams.add(s)
            elif b in ("compute_gemm", "compute_gemm_fp8_grouped"):
                gemm_streams.add(s)
    moe_shares_compute_stream = bool(moe_streams) and bool(moe_streams & gemm_streams)

    # FP8 prep: fraction of fp8_prep self-time spent on amax/cast/scale
    # kernels (used as a heuristic for "permute fusion would help").
    fp8_amax_share = 0.0
    fp8_kernels = [k for k in kernels_full if k.get("bucket") == "compute_fp8_prep"]
    if fp8_kernels:
        amax_self = sum(
            (k.get("self_ms") or 0.0)
            for k in fp8_kernels
            if any(t in (k.get("name_short") or "").lower() for t in ("amax", "cast", "scale"))
        )
        all_self = sum((k.get("self_ms") or 0.0) for k in fp8_kernels) or 1.0
        fp8_amax_share = amax_self / all_self

    # Attention p99 / p50 ratio (long-tail signal).
    attention_long_tail = False
    for k in kernels_full:
        if k.get("bucket") == "compute_attention":
            p50 = k.get("p50_ms") or 0
            p99 = k.get("p99_ms") or 0
            if p50 > 0 and p99 / p50 > 1.5:
                attention_long_tail = True
                break

    # Small-kernel storm: many tiny compute_other kernels = launch overhead.
    other_kernels = [k for k in kernels_full if k.get("bucket") == "compute_other"]
    other_kernel_count_total = sum((k.get("calls") or 0) for k in other_kernels)
    other_avg_ms = 0.0
    if other_kernel_count_total > 0:
        other_self = sum((k.get("self_ms") or 0.0) for k in other_kernels)
        other_avg_ms = other_self / other_kernel_count_total

    return {
        "iter_wallclock_ms": iter_ms,
        "ratios": dict(ratios),
        "cost_breakdown": {
            "pure_compute_pct": cb.get("pure_compute_pct", 0.0),
            "pure_comm_pct": cb.get("pure_comm_pct", 0.0),
            "overlap_pct": cb.get("overlap_pct", 0.0),
            "bubble_pct": cb.get("bubble_pct", 0.0),
            "longest_serialized_comm_ms": cb.get(
                "longest_serialized_comm_ms", overlap_block.get("longest_serialized_comm_ms", 0.0)
            ),
        },
        "phases": {
            "fwd_pct": (phases.get("fwd_ms") / iter_ms) if (iter_ms and phases.get("fwd_ms")) else None,
            "bwd_pct": (phases.get("bwd_ms") / iter_ms) if (iter_ms and phases.get("bwd_ms")) else None,
            "optim_pct": (phases.get("optim_ms") / iter_ms) if (iter_ms and phases.get("optim_ms")) else None,
        },
        "streams_observed": stream_count,
        "moe_shares_compute_stream": moe_shares_compute_stream,
        "fp8_amax_share": round(fp8_amax_share, 4),
        "attention_long_tail": attention_long_tail,
        "compute_other_count": other_kernel_count_total,
        "compute_other_avg_ms": round(other_avg_ms, 4),
        "warnings": list(analysis.get("warnings") or []),
    }


# ---------------------------------------------------------------------------
# Plan / snapshot helpers
# ---------------------------------------------------------------------------


def _plan_overrides(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Pull the relevant trainer overrides out of an effective plan."""
    if not plan:
        return {}
    modules = plan.get("modules") or {}
    pre_trainer = modules.get("pre_trainer") or {}
    return pre_trainer.get("overrides") or {}


def _plan_dim(plan: dict[str, Any] | None, key: str, default: int = 1) -> int:
    """Read a parallel dim from plan overrides; default to 1."""
    ov = _plan_overrides(plan)
    val = ov.get(key, default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _snapshot_failure_kind(snapshot: dict[str, Any] | None) -> str | None:
    if not snapshot:
        return None
    status = (snapshot.get("status") or "").lower()
    if status not in ("failed", "killed", "hung", "unknown"):
        return None
    symptoms = snapshot.get("symptoms") or {}
    if status == "failed":
        if symptoms.get("oom_detected"):
            return "OOM"
        if symptoms.get("nccl_error"):
            return "NCCL"
        if symptoms.get("cuda_error"):
            return "CUDA"
        if symptoms.get("loss_nan_or_inf"):
            return "NUMERICAL"
        if symptoms.get("python_error"):
            return "PYTHON"
        return "FAILED"
    return status.upper()  # KILLED / HUNG / UNKNOWN


def _snapshot_mem_pct(snapshot: dict[str, Any] | None) -> float | None:
    if not snapshot:
        return None
    metrics = snapshot.get("metrics") or {}
    history = metrics.get("history") or {}
    mems = history.get("mem_pct") or history.get("usage_ratio") or []
    if not mems:
        return None
    try:
        return float(max(mems))
    except (TypeError, ValueError):
        return None


def _snapshot_iter_ms(snapshot: dict[str, Any] | None) -> float | None:
    if not snapshot:
        return None
    metrics = snapshot.get("metrics") or {}
    history = metrics.get("history") or {}
    times = history.get("iter_time_ms") or history.get("iter_ms") or []
    times = [t for t in times if isinstance(t, (int, float))]
    if len(times) < 3:
        return None
    sample = sorted(times[2:])
    return sample[len(sample) // 2]


# ---------------------------------------------------------------------------
# Failure routing (R0)
# ---------------------------------------------------------------------------


_FAILURE_ROUTING = {
    "OOM": (
        "MEMORY_BOUND",
        "MEMORY_BOUND",
        0.95,
        "OPTIMIZE_LOOP.REPLAN",
        "oom; mark dead and try mem-relief axes",
    ),
    "NCCL": ("COMM_BOUND", "COMM_BOUND", 0.85, "PREFLIGHT", "nccl error; re-probe env"),
    "CUDA": (
        "COMPUTE_BOUND",
        "UNKNOWN",
        1.00,
        "PREFLIGHT",
        "cuda error suggests hardware/driver issue; full re-collect",
    ),
    "PYTHON": ("COMPUTE_BOUND", "INVALID_CONFIG", 0.95, "OPTIMIZE_LOOP.REPLAN", "python_error; mark dead"),
    "NUMERICAL": ("COMPUTE_BOUND", "NUMERICAL", 1.00, "ABORT", "loss nan/inf; escalate to human"),
    "HUNG": ("COMM_BOUND", "HANG", 0.90, "OPTIMIZE_LOOP.REPLAN", "hung run; mark dead"),
    "KILLED": ("COMPUTE_BOUND", "CANCELLED", 1.00, "OPTIMIZE_LOOP.REPLAN", "cancelled by orchestrator"),
    "UNKNOWN": ("COMPUTE_BOUND", "UNKNOWN", 0.30, "WAIT", "snapshot status unknown; re-snapshot"),
    "FAILED": ("COMPUTE_BOUND", "UNKNOWN", 0.40, "OPTIMIZE_LOOP.REPLAN", "unspecified failure"),
}


def _route_failure(kind: str) -> dict[str, Any]:
    bottleneck, extended, conf, target, reason = _FAILURE_ROUTING[kind]
    return {
        "bottleneck": bottleneck,
        "bottleneck_extended": extended,
        "confidence": conf,
        "rule_id": f"R0-{kind}",
        "transition_to": target,
        "transition_reason": reason,
    }


# ---------------------------------------------------------------------------
# Bottleneck classification (R1..R4)
# ---------------------------------------------------------------------------


def _classify_bottleneck(
    ev: dict[str, Any],
    th: dict[str, float],
    plan: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    """Returns {bottleneck, confidence, rule_id, evidence_lines, ...}."""
    evidence_lines: list[str] = []

    # ---- R1 Memory ------------------------------------------------------
    mem_pct = _snapshot_mem_pct(snapshot)
    if snapshot and (snapshot.get("symptoms") or {}).get("oom_detected"):
        evidence_lines.append("R1-OOM: snapshot.symptoms.oom_detected=true")
        return {
            "bottleneck": "MEMORY_BOUND",
            "confidence": 0.95,
            "rule_id": "R1-OOM",
            "evidence_lines": evidence_lines,
        }
    if mem_pct is not None and mem_pct > th["MEM_TIGHT_PCT"]:
        evidence_lines.append(f"R1-MEM: mem_pct={mem_pct:.3f} > MEM_TIGHT_PCT={th['MEM_TIGHT_PCT']}")
        return {
            "bottleneck": "MEMORY_BOUND",
            "confidence": 0.80,
            "rule_id": "R1-MEM",
            "evidence_lines": evidence_lines,
        }

    pp = _plan_dim(plan, "pipeline_model_parallel_size", 1)
    bubble_pct = ev["cost_breakdown"]["bubble_pct"] or 0.0
    comm_ratio = ev["ratios"].get("comm_ratio") or 0.0
    pure_comm_pct = ev["cost_breakdown"]["pure_comm_pct"] or 0.0
    serial_ms = ev["cost_breakdown"]["longest_serialized_comm_ms"] or 0.0
    iter_ms = ev["iter_wallclock_ms"] or 1.0

    # ---- R2 Pipeline bubble (only if pp>1) ------------------------------
    if pp > 1 and bubble_pct >= th["BUBBLE_HIGH"]:
        evidence_lines.append(
            f"R2-A: bubble_pct={bubble_pct:.3f} ≥ BUBBLE_HIGH={th['BUBBLE_HIGH']} (pp={pp})"
        )
        return {
            "bottleneck": "PIPELINE_BOUND",
            "confidence": 0.85,
            "rule_id": "R2-A",
            "evidence_lines": evidence_lines,
        }
    if pp > 1 and bubble_pct >= th["BUBBLE_MED"]:
        evidence_lines.append(f"R2-B: bubble_pct={bubble_pct:.3f} ≥ BUBBLE_MED={th['BUBBLE_MED']} (pp={pp})")
        return {
            "bottleneck": "PIPELINE_BOUND",
            "confidence": 0.70,
            "rule_id": "R2-B",
            "evidence_lines": evidence_lines,
        }

    # ---- R3 Communication ---------------------------------------------
    A = comm_ratio >= th["COMM_HIGH"]
    B = pure_comm_pct >= th["PURE_COMM_HIGH"]
    C = serial_ms >= th["SERIAL_COMM_HIGH_MS"] and (serial_ms / iter_ms) >= 0.10
    if A and comm_ratio >= th["COMM_VERY_HIGH"]:
        evidence_lines.append(
            f"R3-A-strong: comm_ratio={comm_ratio:.3f} ≥ COMM_VERY_HIGH={th['COMM_VERY_HIGH']}"
        )
        return {
            "bottleneck": "COMM_BOUND",
            "confidence": 0.90,
            "rule_id": "R3-A-strong",
            "evidence_lines": evidence_lines,
        }
    if sum([A, B, C]) >= 2:
        flags = "".join(["A" if A else "", "B" if B else "", "C" if C else ""])
        evidence_lines.append(
            f"R3-{flags}: comm_ratio={comm_ratio:.3f} pure_comm={pure_comm_pct:.3f} "
            f"longest_serial_ms={serial_ms:.2f} ({serial_ms / iter_ms * 100:.1f}% of iter)"
        )
        bonus = 0.05 if (A and B and C) else 0.0
        return {
            "bottleneck": "COMM_BOUND",
            "confidence": min(1.0, 0.80 + bonus),
            "rule_id": f"R3-{flags}",
            "evidence_lines": evidence_lines,
        }
    if A:
        evidence_lines.append(
            f"R3-A-weak: comm_ratio={comm_ratio:.3f} ≥ COMM_HIGH={th['COMM_HIGH']} "
            "but only 1/3 sub-signals fired"
        )
        return {
            "bottleneck": "COMM_BOUND",
            "confidence": 0.70,
            "rule_id": "R3-A-weak",
            "evidence_lines": evidence_lines,
        }

    # ---- R4 Compute (default) -----------------------------------------
    compute_gemm_ratio = ev["ratios"].get("compute_gemm_ratio") or 0.0
    compute_ratio = ev["ratios"].get("compute_ratio") or 0.0
    if compute_gemm_ratio >= th["COMPUTE_GEMM_HIGH"] and bubble_pct < th["BUBBLE_LOW"]:
        evidence_lines.append(
            f"R4-A: compute_gemm_ratio={compute_gemm_ratio:.3f} ≥ COMPUTE_GEMM_HIGH={th['COMPUTE_GEMM_HIGH']} "
            f"AND bubble_pct={bubble_pct:.3f} < BUBBLE_LOW={th['BUBBLE_LOW']} (gemm-dominated steady iter)"
        )
        return {
            "bottleneck": "COMPUTE_BOUND",
            "confidence": 0.85,
            "rule_id": "R4-A",
            "evidence_lines": evidence_lines,
        }
    if compute_ratio >= th["COMPUTE_HIGH"] and comm_ratio < th["COMM_LOW"]:
        evidence_lines.append(
            f"R4-B: compute_ratio={compute_ratio:.3f} ≥ COMPUTE_HIGH={th['COMPUTE_HIGH']} "
            f"AND comm_ratio={comm_ratio:.3f} < COMM_LOW={th['COMM_LOW']}"
        )
        return {
            "bottleneck": "COMPUTE_BOUND",
            "confidence": 0.75,
            "rule_id": "R4-B",
            "evidence_lines": evidence_lines,
        }
    evidence_lines.append(
        f"R4-DEFAULT: no other rule fired "
        f"(comm={comm_ratio:.3f} bubble={bubble_pct:.3f} compute_gemm={compute_gemm_ratio:.3f})"
    )
    return {
        "bottleneck": "COMPUTE_BOUND",
        "confidence": 0.55,
        "rule_id": "R4-DEFAULT",
        "evidence_lines": evidence_lines,
    }


# ---------------------------------------------------------------------------
# candidate_axes production (§9)
# ---------------------------------------------------------------------------


def _build_axes(
    ev: dict[str, Any],
    th: dict[str, float],
    plan: dict[str, Any] | None,
    snapshot: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Returns (candidate_axes, env_suspect, skipped_axes_log)."""
    axes: list[dict[str, Any]] = []
    env_suspect: list[dict[str, Any]] = []
    skipped: list[str] = []

    ratios = ev["ratios"]
    cb = ev["cost_breakdown"]
    moe_ratio = ratios.get("comm_moe_dispatch_ratio") or 0.0
    coll_ratio = ratios.get("comm_collective_ratio") or 0.0
    fp8_prep = ratios.get("compute_fp8_prep_ratio") or 0.0
    attn = ratios.get("compute_attention_ratio") or 0.0
    other = ratios.get("compute_other_ratio") or 0.0
    gemm = ratios.get("compute_gemm_ratio") or 0.0
    comm = ratios.get("comm_ratio") or 0.0
    overlap_pct = cb.get("overlap_pct") or 0.0
    bubble_pct = cb.get("bubble_pct") or 0.0
    moe_shares = ev.get("moe_shares_compute_stream", False)
    streams_observed = ev.get("streams_observed", 0)

    pp = _plan_dim(plan, "pipeline_model_parallel_size", 1)
    ep = _plan_dim(plan, "expert_model_parallel_size", 1)
    _plan_dim(plan, "tensor_model_parallel_size", 1)
    mbs = _plan_dim(plan, "micro_batch_size", 1)
    overrides = _plan_overrides(plan)

    # ---- MoE EP comm sharing the compute stream (env_suspect + axis) ----
    if moe_ratio >= 0.05 and overlap_pct < th["OVERLAP_LOW"] and streams_observed <= 2 and moe_shares:
        axes.append(
            {
                "axis": "turbo_deepep_use_comm_stream",
                "type": "weakly_local",
                "candidates": [True],
                "expected_gain_band_pct": [3, 8],
                "rationale": (
                    f"MoE EP comm ({moe_ratio*100:.1f}% of iter) shares stream with compute_gemm; "
                    f"overlap_pct={overlap_pct:.3f} < OVERLAP_LOW={th['OVERLAP_LOW']} "
                    f"and only {streams_observed} GPU stream(s) observed"
                ),
            }
        )
        env_suspect.append(
            {
                "flag": "turbo_deepep_use_comm_stream",
                "reason": "MoE EP comm rides the compute stream",
                "hint": "set true and re-profile",
            }
        )

    # ---- MoE EP CU allocation ----
    if moe_ratio >= 0.05:
        axes.append(
            {
                "axis": "turbo_deepep_num_cu",
                "type": "weakly_local",
                "candidates": [64, 80, 96],
                "expected_gain_band_pct": [1, 5],
                "rationale": f"MoE EP dispatch is {moe_ratio*100:.1f}% of iter; tune CU allocation",
            }
        )

    # ---- Grad-reduction overlap ----
    if coll_ratio >= 0.05 and overlap_pct < th["OVERLAP_LOW"]:
        axes.append(
            {
                "axis": "overlap_grad_reduce",
                "type": "weakly_local",
                "candidates": [True],
                "expected_gain_band_pct": [1, 4],
                "rationale": (
                    f"comm_collective ratio {coll_ratio*100:.1f}% with overlap_pct={overlap_pct:.3f} "
                    "(grad-reduction not overlapped)"
                ),
            }
        )
        axes.append(
            {
                "axis": "overlap_param_gather",
                "type": "weakly_local",
                "candidates": [True],
                "expected_gain_band_pct": [1, 3],
                "rationale": "DP zero-1/zero-2 param gather can overlap with bwd",
            }
        )

    # ---- FP8 amax/cast hot path ----
    if fp8_prep >= 0.03 and ev.get("fp8_amax_share", 0) >= 0.5:
        axes.append(
            {
                "axis": "MOE_PERMUTE_FUSION",
                "type": "strongly_local",
                "candidates": [True],
                "expected_gain_band_pct": [2, 6],
                "rationale": (
                    f"FP8 prep is {fp8_prep*100:.1f}% of iter and "
                    f"{ev['fp8_amax_share']*100:.0f}% of fp8_prep is amax/cast/scale "
                    "(permute fusion can absorb these)"
                ),
            }
        )

    # ---- FP8 recipe (axis_taxonomy.md §2.6 — empirically dominant) ----
    # Trigger when FP8 prep is non-trivial AND the trainer is on the default
    # `tensorwise` recipe. The `delayed` recipe removes per-iter amax in the
    # FP8 hot path; observed +24.85% on DeepSeek-V2-Lite (session R4). This is
    # *the* highest-prior axis on FP8 stacks and must be emitted by the engine
    # so REPLAN doesn't depend on orchestrator hand-injection.
    fp8_recipe_now = overrides.get("fp8_recipe")
    if fp8_prep >= 0.02 and fp8_recipe_now in (None, "", "tensorwise"):
        axes.append(
            {
                "axis": "fp8_recipe",
                "type": "strongly_local",
                "candidates": ["delayed"],
                "expected_gain_band_pct": [10, 25],
                "rationale": (
                    f"compute_fp8_prep_ratio={fp8_prep:.3f} on default `tensorwise` recipe; "
                    "`delayed` removes per-iter amax (axis_taxonomy.md §2.6)"
                ),
            }
        )

    # ---- Attention long-tail ----
    if attn >= 0.05 and ev.get("attention_long_tail"):
        axes.append(
            {
                "axis": "attention_kernel",
                "type": "weakly_local",
                "candidates": ["flash", "sdpa"],
                "expected_gain_band_pct": [1, 5],
                "rationale": f"attention is {attn*100:.1f}% of iter and shows long-tail kernels (p99 > 1.5 × p50)",
            }
        )

    # ---- RoPE fusion (axis_taxonomy.md §2.7) ----
    # Fuses the RoPE rotation into attention; observed +2.61% (session R9)
    # whenever attention is non-trivial and rope_fusion is left off.
    if attn >= 0.03 and overrides.get("apply_rope_fusion") in (None, False, "false", 0):
        axes.append(
            {
                "axis": "apply_rope_fusion",
                "type": "weakly_local",
                "candidates": [True],
                "expected_gain_band_pct": [1, 4],
                "rationale": (
                    f"attention is {attn*100:.1f}% of iter with apply_rope_fusion off; "
                    "RoPE-into-attention fusion removes a kernel per layer"
                ),
            }
        )

    # ---- Small-kernel storm (launch overhead) ----
    if (
        other >= 0.05
        and ev.get("compute_other_count", 0) > 10000
        and ev.get("compute_other_avg_ms", 0) < 0.05
    ):
        axes.append(
            {
                "axis": "gradient_accumulation_fusion",
                "type": "weakly_local",
                "candidates": [True],
                "expected_gain_band_pct": [1, 3],
                "rationale": (
                    f"compute_other has {ev['compute_other_count']} kernels averaging "
                    f"{ev['compute_other_avg_ms']*1000:.1f} µs each (launch-overhead dominated)"
                ),
            }
        )
        # Companion fusion knobs (axis_taxonomy.md §2.7); cheap to flip and
        # they each shave a few launch-bound kernels.
        for fuse_axis in ("masked_softmax_fusion", "bias_dropout_fusion"):
            if overrides.get(fuse_axis) in (None, False, "false", 0):
                axes.append(
                    {
                        "axis": fuse_axis,
                        "type": "weakly_local",
                        "candidates": [True],
                        "expected_gain_band_pct": [0, 2],
                        "rationale": ("small-kernel storm above; fusion removes a launch per block"),
                    }
                )

    # ---- Compute-bound: bigger MBS to amortize launch ----
    if gemm >= th["COMPUTE_GEMM_HIGH"] and bubble_pct < th["BUBBLE_LOW"] and comm < th["COMM_LOW"]:
        if plan is None:
            skipped.append("micro_batch_size (structural; no plan provided)")
        else:
            axes.append(
                {
                    "axis": "micro_batch_size",
                    "type": "structural",
                    "candidates": [mbs + 2, mbs + 4],
                    "expected_gain_band_pct": [2, 6],
                    "rationale": (
                        f"compute_gemm_ratio={gemm:.3f} dominant and bubble_pct={bubble_pct:.3f} "
                        "tiny; bigger batch amortizes launch & fills tile shapes"
                    ),
                }
            )

    # ---- EP-comm dominant (suggest reducing EP) ----
    if comm >= th["COMM_HIGH"] and gemm < 0.40:
        if plan is None:
            skipped.append("expert_model_parallel_size (structural; no plan provided)")
        elif ep > 1:
            axes.append(
                {
                    "axis": "expert_model_parallel_size",
                    "type": "structural",
                    "candidates": [max(1, ep // 2)],
                    "expected_gain_band_pct": [3, 10],
                    "rationale": (
                        f"comm_ratio={comm:.3f} ≥ {th['COMM_HIGH']} and compute_gemm={gemm:.3f} < 0.40 "
                        f"(EP={ep} comm-bound)"
                    ),
                }
            )

    # ---- VPP for PP bubble ----
    if pp > 1 and bubble_pct >= th["BUBBLE_HIGH"]:
        if plan is None:
            skipped.append("virtual_pipeline_model_parallel_size (structural; no plan provided)")
        else:
            axes.append(
                {
                    "axis": "virtual_pipeline_model_parallel_size",
                    "type": "structural",
                    "candidates": [2, 4],
                    "expected_gain_band_pct": [3, 8],
                    "rationale": f"bubble_pct={bubble_pct:.3f} with pp={pp}; VPP can split the bubble",
                }
            )

    # ---- Memory pressure ----
    mem_pct = _snapshot_mem_pct(snapshot)
    if mem_pct is not None and mem_pct > th["MEM_TIGHT_PCT"]:
        axes.append(
            {
                "axis": "recompute_granularity",
                "type": "strongly_local",
                "candidates": ["selective"],
                "expected_gain_band_pct": [1, 5],
                "rationale": f"mem_pct={mem_pct:.3f} > MEM_TIGHT_PCT={th['MEM_TIGHT_PCT']}",
            }
        )
        if mem_pct > 0.95:
            if plan is None:
                skipped.append("micro_batch_size (structural; no plan provided)")
            else:
                axes.append(
                    {
                        "axis": "micro_batch_size",
                        "type": "structural",
                        "candidates": [max(1, mbs - 1)],
                        "expected_gain_band_pct": [0, 0],
                        "rationale": f"mem_pct={mem_pct:.3f} very tight; reduce MBS to fit",
                    }
                )

    # ---- env_suspect (extra triggers beyond MoE shared-stream) ----
    env_diff = (plan or {}).get("env_diff", {}) or {}
    if coll_ratio >= 0.05 and not (overrides.get("RCCL_MSCCL_ENABLE") or env_diff.get("RCCL_MSCCL_ENABLE")):
        env_suspect.append(
            {
                "flag": "RCCL_MSCCL_ENABLE",
                "reason": f"comm_collective is {coll_ratio*100:.1f}% of iter; algorithm pick can change collective shape",
                "hint": "try true; algorithm pick can change collective shape",
            }
        )
    # RCCL protocol/algorithm picks become candidates when collectives are a
    # measurable share of iter (axis_taxonomy.md §2.11).
    if coll_ratio >= 0.05 and not env_diff.get("RCCL_PROTO"):
        env_suspect.append(
            {
                "flag": "RCCL_PROTO",
                "reason": f"comm_collective is {coll_ratio*100:.1f}% of iter; protocol pick can shift latency floor",
                "hint": "try LL128 first (Simple is the default fallback)",
            }
        )
    if mem_pct is not None and mem_pct >= 0.92:
        env_suspect.append(
            {
                "flag": "PYTORCH_HIP_ALLOC_CONF",
                "reason": "mem_pct ≥ 0.92",
                "hint": "set `expandable_segments:True`",
            }
        )

    # ---- Host-launch bubble env_suspects (axis_taxonomy.md §2.10) ----
    # When the bubble is high relative to GPU work, the cause is usually
    # CPU-side: OMP thread contention, GC pauses, or untuned HW queue depth.
    # OMP_NUM_THREADS=4 was empirically +3.34% (session R7); manual_gc=true
    # eliminates jitter from CPython's amortised collection.
    if bubble_pct >= 0.20:
        if not env_diff.get("OMP_NUM_THREADS"):
            env_suspect.append(
                {
                    "flag": "OMP_NUM_THREADS",
                    "reason": f"bubble_pct={bubble_pct:.3f} ≥ 0.20 (host-launch dominant)",
                    "hint": "set 4 (PyTorch default oversubscribes EPYC threads)",
                }
            )
        if overrides.get("manual_gc") in (None, False, "false", 0):
            axes.append(
                {
                    "axis": "manual_gc",
                    "type": "weakly_local",
                    "candidates": [True],
                    "expected_gain_band_pct": [0, 2],
                    "rationale": (
                        f"bubble_pct={bubble_pct:.3f} likely includes CPython GC pauses; "
                        "manual_gc removes amortised-collection jitter"
                    ),
                }
            )

    # ---- CUDA graph family (axis_taxonomy.md §2.8 — stack-blocked) ----
    # Engine still names the axes so REPLAN/ENV_SWEEP can register the
    # known-blocker tag in axis_meta and downrank priority. This way the
    # frontier surfaces "host-launch bubble exists, fix is upstream" as a
    # signal instead of silently exhausting the search.
    if bubble_pct >= 0.25 and overrides.get("cuda_graph_impl") is None:
        axes.append(
            {
                "axis": "cuda_graph_impl",
                "type": "strongly_local",
                "candidates": ["local"],
                "expected_gain_band_pct": [5, 20],
                "rationale": (
                    f"bubble_pct={bubble_pct:.3f} ≥ 0.25 (host-launch dominates); "
                    "cuda graph capture would absorb the launch storm"
                ),
                "known_blocker": "cuda_graph_family",
                "blocker_note": (
                    "Megatron arguments.py:958 enum-vs-str bug + DeepEP intranode "
                    "dispatch is not capture-friendly (axis_taxonomy.md §2.8). "
                    "REPLAN should downrank by 0.1 until upstream lands."
                ),
            }
        )

    return axes, env_suspect, skipped


# ---------------------------------------------------------------------------
# Strategy mapping (§11)
# ---------------------------------------------------------------------------


def _suggest_strategy(axes: list[dict[str, Any]]) -> tuple[str, str]:
    n = len(axes)
    types = [a["type"] for a in axes]
    if any(t == "cluster_shared" for t in types):
        return "Champion-Challenger", "cluster_shared axis present; re-validate baseline"
    if all(t == "weakly_local" for t in types) and n <= 4:
        return "Per-Plan", "all axes weakly_local and ≤ 4 axes; explore independently"
    total_candidates = sum(len(a.get("candidates", [])) for a in axes)
    if len(set(types)) > 1 and total_candidates > 6:
        return "Successive_Halving", "mixed axis types and > 6 candidate values; budget-aware pruning"
    if all(t == "structural" for t in types) and n > 0:
        return "Per-Plan", "only structural axes; explore each (with constraint check)"
    return "Per-Plan", "default"


# ---------------------------------------------------------------------------
# Recommended skills (§9.x)
# ---------------------------------------------------------------------------


_BOTTLENECK_SKILL = {
    "COMM_BOUND": "skills/optimization/comm/SKILL.md",
    "PIPELINE_BOUND": "skills/optimization/pipeline/SKILL.md",
    "MEMORY_BOUND": "skills/optimization/memory/SKILL.md",
    "COMPUTE_BOUND": "skills/optimization/compute/SKILL.md",
}


def _recommend_skills(bottleneck: str, ev: dict[str, Any]) -> list[str]:
    out = [_BOTTLENECK_SKILL.get(bottleneck, "skills/optimization/compute/SKILL.md")]
    moe_ratio = ev["ratios"].get("comm_moe_dispatch_ratio") or 0.0
    coll_ratio = ev["ratios"].get("comm_collective_ratio") or 0.0
    if moe_ratio > coll_ratio and moe_ratio >= 0.05:
        out.append("skills/optimization/comm/moe_dispatch.md")
    elif coll_ratio >= 0.05:
        out.append("skills/optimization/comm/tp_allreduce.md")
    return out


# ---------------------------------------------------------------------------
# Main entry: run()
# ---------------------------------------------------------------------------


def run(
    *,
    trace_analysis: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
    cluster_profile: dict[str, Any] | None = None,
    champion_snapshot: dict[str, Any] | None = None,
    plan_graph: dict[str, Any] | None = None,
    thresholds_path: str | Path | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Build a DiagnosisReport from a trace_analysis.json (+ optional context)."""
    th = _load_thresholds(thresholds_path)
    ev = _build_evidence(trace_analysis)

    snapshot_id = snapshot_id or (snapshot or {}).get("run_id") or trace_analysis.get("run_id") or "unknown"

    # ---- R0 short-circuit ----
    failure_kind = _snapshot_failure_kind(snapshot)
    if failure_kind:
        f = _route_failure(failure_kind)
        report: dict[str, Any] = {
            "schema_version": "1.0",
            "snapshot_id": snapshot_id,
            "bottleneck": f["bottleneck"],
            "confidence": round(f["confidence"], 2),
            "evidence": [f"R0-{failure_kind}: snapshot status={(snapshot or {}).get('status')}"],
            "recommended_skills": [_BOTTLENECK_SKILL[f["bottleneck"]]],
            "env_suspect": [],
            "candidate_axes": [],
            "suggested_strategy": "Per-Plan",
            "suggested_transition": {
                "to": f["transition_to"],
                "reason": f["transition_reason"],
                "counts_against_budget": False,
            },
            "meta": {
                "rule_id": f["rule_id"],
                "bottleneck_extended": f["bottleneck_extended"],
            },
        }
        return report

    # ---- R1..R4 ----
    cls = _classify_bottleneck(ev, th, plan, snapshot)
    bottleneck = cls["bottleneck"]
    confidence = cls["confidence"]
    rule_id = cls["rule_id"]
    evidence_lines = list(cls["evidence_lines"])

    # Confidence adjustments per §6
    if ev["iter_wallclock_ms"] < 100:
        confidence -= 0.10
    if ev["warnings"]:
        confidence -= 0.10
    confidence = max(0.0, min(1.0, round(confidence, 2)))

    # ---- R5 regression overlay ----
    bottleneck_extended = bottleneck
    if champion_snapshot is not None:
        champ_iter = _snapshot_iter_ms(champion_snapshot)
        cur_iter = ev["iter_wallclock_ms"]
        if champ_iter and cur_iter and cur_iter > champ_iter * th["REGRESSION_PCT"]:
            evidence_lines.append(
                f"R5-REGRESSION: iter_ms={cur_iter:.2f} > champion_iter_ms={champ_iter:.2f} × "
                f"REGRESSION_PCT={th['REGRESSION_PCT']}"
            )
            bottleneck_extended = "REGRESSION"
            confidence = 0.95

    # ---- candidate_axes ----
    axes, env_suspect, skipped = _build_axes(ev, th, plan, snapshot)
    strategy, strategy_rationale = _suggest_strategy(axes)

    # ---- Suggested transition ----
    if bottleneck_extended == "REGRESSION":
        transition = {
            "to": "OPTIMIZE_LOOP.REPLAN",
            "reason": "regression vs champion; this candidate is dead",
            "counts_against_budget": False,
            "hint": "do not derive from this plan",
        }
    elif "iter_boundary_fallback" in ev["warnings"]:
        transition = {
            "to": "WAIT",
            "reason": "trace iter boundary used wallclock fallback; verdict deferred",
            "counts_against_budget": False,
        }
    else:
        transition = {
            "to": "OPTIMIZE_LOOP.REPLAN",
            "reason": f"{bottleneck.lower()} verdict via {rule_id}",
            "counts_against_budget": True,
        }

    # ---- Build report ----
    report = {
        "schema_version": "1.0",
        "snapshot_id": snapshot_id,
        "bottleneck": bottleneck,
        "confidence": confidence,
        "evidence": evidence_lines,
        "recommended_skills": _recommend_skills(bottleneck, ev),
        "env_suspect": env_suspect,
        "candidate_axes": axes,
        "suggested_strategy": strategy,
        "suggested_transition": transition,
        "meta": {
            "rule_id": rule_id,
            "bottleneck_extended": bottleneck_extended,
            "iter_ms_steady_median": ev["iter_wallclock_ms"],
            "comm_ratio": ev["ratios"].get("comm_ratio"),
            "bubble_ratio_estimate": ev["cost_breakdown"]["bubble_pct"],
            "bubble_ratio_source": "profiler" if ev["iter_wallclock_ms"] > 0 else "n/a",
            "strategy_rationale": strategy_rationale,
            "thresholds": th,
            "trace_evidence": ev,
        },
    }
    if skipped:
        report["meta"]["skipped_axes"] = skipped
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.diagnose")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="Classify bottleneck from a trace_analysis.json.")
    p_run.add_argument(
        "--trace-analysis",
        required=True,
        help="Path to trace_analysis.json (per skills/workflow/trace_analysis.md)",
    )
    p_run.add_argument("--snapshot", default=None, help="Optional RunSnapshot YAML")
    p_run.add_argument("--plan", default=None, help="Optional effective Plan YAML")
    p_run.add_argument("--cluster-profile", default=None, help="Optional ClusterProfile YAML")
    p_run.add_argument("--champion-snapshot", default=None, help="Optional champion RunSnapshot YAML")
    p_run.add_argument("--plan-graph", default=None, help="Optional PlanGraph YAML")
    p_run.add_argument("--thresholds", default=None, help="Optional override file")
    p_run.add_argument("--out", default=None, help="Write the report JSON to this path")
    p_run.add_argument(
        "--snapshot-id", default=None, help="Override snapshot_id in the report (else inferred)"
    )
    args = p.parse_args()

    try:
        trace = _load_json(args.trace_analysis)
        snapshot = _load_yaml_or_json(args.snapshot)
        plan = _load_yaml_or_json(args.plan)
        cluster_profile = _load_yaml_or_json(args.cluster_profile)
        champion_snapshot = _load_yaml_or_json(args.champion_snapshot)
        plan_graph = _load_yaml_or_json(args.plan_graph)
        report = run(
            trace_analysis=trace,
            snapshot=snapshot,
            plan=plan,
            cluster_profile=cluster_profile,
            champion_snapshot=champion_snapshot,
            plan_graph=plan_graph,
            thresholds_path=args.thresholds,
            snapshot_id=args.snapshot_id,
        )
    except _DiagError as exc:
        _emit({"stage": "DIAGNOSE", "status": "failed", "failure": {"kind": exc.kind, "message": str(exc)}})
        return _EXIT_USAGE if exc.kind == "USAGE" else _EXIT_TOOL_ERROR

    if args.out:
        out_path = _resolve(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
    _emit(report)
    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
