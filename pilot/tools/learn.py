"""learn tool: derive between-session findings from a finished tuning session.

Closes the last link in the autonomy chain: every session improves the next.

Inputs (read-only):
  * ``tuning_state.yaml``           — required; the structured spine.
  * ``r*_exec_results.json``        — optional; per-trial subprocess output.
  * ``r*_snapshots.json``           — optional; per-trial steady-state metrics.
  * ``run_history`` (inline or sidecar) — optional; if present, supersedes the
    json-glob fallback.

Outputs (write-only into ``state/learn/<session_id>/`` and
``state/knowledge_drafts/`` via :mod:`pilot.tools.knowledge`):

  * ``learn_analysis.yaml``         — :class:`LearnAnalysis` doc.
  * one or more ``KnowledgeDraft`` YAMLs (when ``--emit-drafts``).

Findings classes:

  * **catalog_gaps**          — axes that moved a champion (or were tried)
    but :mod:`pilot.tools._axis_translator` returns ``None`` for. These are
    candidates for the next ``axis_taxonomy.md`` patch.
  * **constraint_gaps**       — failure messages that match a known mutex
    shape from ``axis_taxonomy.md §2.14`` but weren't pre-empted by
    ``constraint.check`` in this session. Indicates either a false-negative
    in the rule loader or a brand-new mutex worth registering.
  * **calibration_drifts**    — engine-predicted gain band vs measured gain
    differs by more than ``2x`` on either side. Surfaces axes whose Re-Plan
    prior is mis-calibrated (e.g. ``fp8_recipe`` had band ``[10, 25]`` but
    measured ``+24.85%`` — top of band, calibration is OK; ``cuda_graph``
    family had band ``[5, 20]`` but measured ``-100%`` due to known_blocker).
  * **anti_pattern_signals**  — env / trainer values that *consistently*
    regressed by ≥5% across rounds. Candidates for ``axis_taxonomy.md``
    DANGER notes (e.g. ``HSA_ENABLE_INTERRUPT=0`` measured -13.28%).

Each finding maps to one of the four existing :class:`KnowledgeDraft`
``kind`` values, so this tool does NOT change the schema:

  ===========================  ======================================
  Finding                      Draft kind
  ===========================  ======================================
  catalog_gaps + champion      ``final_best_case``
  catalog_gaps + env-only      ``env_recipe``
  constraint_gaps              ``failure_pattern``
  calibration_drifts           ``model_calibration_drift``
  anti_pattern_signals         ``failure_pattern``
  ===========================  ======================================

Drafts are gated by :func:`pilot.tools.knowledge.write`, which already
enforces the §S4.2 anti-patterns (no empty evidence / over-broad binding /
direct contradiction). ``learn`` does not write to ``skills/`` directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.tools import _axis_translator as _axt
from pilot.tools import knowledge as _knowledge

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Known-mutex fingerprint table (mirrors axis_taxonomy.md §2.14)
# ---------------------------------------------------------------------------
# Each entry: rule_id, regex on the trial failure message, the mutex axis
# combination, and a short rationale. Order matters: first match wins.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MutexFingerprint:
    rule_id: str
    pattern: re.Pattern
    axes: tuple[str, ...]
    rationale: str


_MUTEX_FINGERPRINTS: tuple[_MutexFingerprint, ...] = (
    _MutexFingerprint(
        "REQ-PP-DEFER-EMB",
        re.compile(r"defer.*embedding.*wgrad.*pipeline.*not used", re.I),
        ("defer_embedding_wgrad_compute", "pipeline_model_parallel_size"),
        "defer_embedding_wgrad_compute requires pp >= 2",
    ),
    _MutexFingerprint(
        "REQ-PP-OVRLP-P2P",
        re.compile(r"interleaved pipeline parallelism", re.I),
        ("overlap_p2p_communication", "virtual_pipeline_model_parallel_size"),
        "overlap_p2p_communication requires pp >= 2 and vpp > 1",
    ),
    _MutexFingerprint(
        "MUTEX-CG-IMPL",
        re.compile(r"--enable-cuda-graph.*--cuda-graph-impl|--cuda-graph-impl.*--enable-cuda-graph", re.I),
        ("enable_cuda_graph", "cuda_graph_impl"),
        "enable_cuda_graph and cuda_graph_impl are mutually exclusive",
    ),
    _MutexFingerprint(
        "MUTEX-DEEPEP-ROUTER",
        re.compile(r"DeepEP.*float32.*probs|moe_router_dtype.*fp32", re.I),
        ("use_turbo_deepep", "moe_router_dtype"),
        "use_turbo_deepep requires moe_router_dtype in {unset, fp32}",
    ),
    _MutexFingerprint(
        "KNOWN-BLOCKER-CG-ENUM",
        re.compile(r"requires string as left operand, not CudaGraphScope", re.I),
        ("cuda_graph_impl",),
        "Megatron arguments.py:958 enum-vs-str bug (cuda_graph_family)",
    ),
    _MutexFingerprint(
        "KNOWN-BLOCKER-CG-HIP",
        re.compile(r"HIP error: invalid argument", re.I),
        ("external_cuda_graph",),
        "DeepEP intranode dispatch is not capture-friendly (cuda_graph_family)",
    ),
    _MutexFingerprint(
        "MUTEX-PROFILE-HIPBLASLT",
        re.compile(r"PRIMUS_HIPBLASLT_TUNING.*profile|profile.*HIPBLASLT", re.I),
        ("PRIMUS_HIPBLASLT_TUNING",),
        "PRIMUS_HIPBLASLT_TUNING=1 conflicts with default torch.profiler injection",
    ),
)


# ---------------------------------------------------------------------------
# Findings dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CatalogGap:
    axis: str
    values_tried: list[Any]
    best_observed_gain_pct: float | None
    proposed_section: str | None
    proposed_type: str | None
    channel: str | None
    evidence_run_ids: list[str] = field(default_factory=list)


@dataclass
class ConstraintGap:
    rule_id: str
    failure_match: str
    axes: list[str]
    rationale: str
    observed_in_runs: list[str]


@dataclass
class CalibrationDrift:
    axis: str
    predicted_band_pct: tuple[float, float]
    measured_gain_pct: float
    verdict: str  # "in_band" | "above_band" | "below_band" | "regressed"


@dataclass
class AntiPatternSignal:
    axis: str
    value: Any
    measured_regression_pct: float
    proposed_note: str
    evidence_run_ids: list[str]


@dataclass
class LearnAnalysis:
    schema_version: str
    session_id: str
    generated_at: str
    inputs: dict[str, Any]
    findings: dict[str, list[Any]]
    suggested_drafts: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Internal: shared utilities
# ---------------------------------------------------------------------------


def _yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"PyYAML required for learn: {exc}") from exc
    return yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = _yaml().safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: expected YAML mapping")
    return data


def _load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    yaml = _yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Internal: per-trial extractors
# ---------------------------------------------------------------------------


_LOG_TAIL_BYTES = 16 * 1024  # peek the last 16 KiB of train.log


def _tail_log(log_path: str | Path | None) -> str:
    if not log_path:
        return ""
    p = Path(log_path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            if size > _LOG_TAIL_BYTES:
                f.seek(-_LOG_TAIL_BYTES, 2)
            chunk = f.read()
        return chunk.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _extract_log_ref(inner: dict[str, Any]) -> str | None:
    for art in inner.get("artifacts") or []:
        if (art or {}).get("kind") == "TrainLog" and art.get("ref"):
            return art["ref"]
    return None


def _iter_exec_trials(session_dir: Path) -> list[dict[str, Any]]:
    """Each `r*_exec_results.json` is a list of `{run_id, exit, stdout, stderr}`.

    The `stdout` is itself a JSON-encoded `SubmitResult`. We unpack it so
    callers see a single flat dict per trial, with `failure` lifted out
    AND ``log_tail`` populated from the trailing 16 KiB of ``train.log``
    when the SubmitResult points at one. The mutex / known-blocker
    fingerprints in :data:`_MUTEX_FINGERPRINTS` need the actual stack
    trace, which is in the log — `failure.message` only carries the
    generic ``job exited with status=failed`` line.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(session_dir.glob("r*_exec_results.json")):
        try:
            arr = _load_json(path)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(arr, list):
            continue
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            stdout = entry.get("stdout") or "{}"
            try:
                inner = json.loads(stdout) if isinstance(stdout, str) else stdout
            except json.JSONDecodeError:
                inner = {}
            log_ref = _extract_log_ref(inner)
            out.append(
                {
                    "source_file": str(path.name),
                    "run_id": entry.get("run_id")
                    or (inner.get("summary", {}) or {}).get("key_metrics", {}).get("run_id"),
                    "exit": entry.get("exit"),
                    "elapsed_s": entry.get("elapsed_s"),
                    "status": inner.get("status"),
                    "failure": inner.get("failure"),
                    "summary": inner.get("summary") or {},
                    "log_ref": log_ref,
                    "log_tail": _tail_log(log_ref) if (entry.get("exit") or 0) != 0 else "",
                }
            )
    return out


def _iter_snapshot_entries(session_dir: Path) -> list[dict[str, Any]]:
    """Each `r*_snapshots.json` is either a list or a dict of trial summaries.

    We normalize to a list of dicts; callers expect each to carry at least
    ``run_id`` and ``median_tflops`` / ``median_iter_time_ms`` / ``gain_pct``
    fields, but tolerate missing ones.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(session_dir.glob("r*_snapshots.json")):
        try:
            data = _load_json(path)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    v = dict(v)
                    v.setdefault("run_id", k)
                    out.append(v)
        elif isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Internal: stage_history headline parser (last-resort signal extractor)
# ---------------------------------------------------------------------------


# Headlines look like:
#   "R1 promote: champion=r1_c0_deepep80 (turbo_deepep_num_cu=80) +2.13% TFLOPS"
#   "R7 c2 OMP_NUM_THREADS=4 +3.34% TFLOPS PROMOTED"
#   "R4 c2 fp8_recipe=delayed +24.85% TFLOPS"
#
# Headline parsing is a *fallback* signal source — the canonical input is
# `tuning_state.run_history` (structured per-trial). When that's absent (as
# in session 20260513T024603Z), we extract axes from headlines but with
# strict guards to avoid false positives on prose like "rc=0", "MBS=1",
# "c0=134.2".

_HEADLINE_AXIS_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*=\s*([A-Za-z0-9_./\-]+)\b[^%]{0,120}?([+\-]\d+(?:\.\d+)?)\s*%",
)

# Known noise tokens that look like `axis=value` but aren't axes.
_AXIS_BLOCKLIST: frozenset[str] = frozenset(
    {
        "champion",  # champion=<plan_id>
        "champion_id",
        "session",
        "session_id",
        "round",
        "round_id",
        "iter",
        "iter_time",
        "iter_ms",
        "loss",
        "tps",
        "tflops",
        "stage",
        "exit",
        "exit_code",
        "rc",  # process exit
        "elapsed",
        "elapsed_s",
        "wallclock_s",
        "MBS",  # parameter description prefix (real axis is micro_batch_size)
        "GBS",
        "PP",
        "TP",
        "EP",
        "CP",
        "VPP",
        "DP",
        "rank",
        "world",
        "nnodes",
        "gpus",
        "headline",
        "status",
        "kind",
        "ref",
        "path",
    }
)

# Tokens that should be expanded to a known full axis name. Most LLM-prose
# headlines abbreviate; this is the canonical resolver.
_AXIS_PREFIX_ALIAS: dict[str, str] = {
    "OMP": "OMP_NUM_THREADS",
    "HSA_INT": "HSA_ENABLE_INTERRUPT",
    "HSA_NSR": "HSA_NO_SCRATCH_RECLAIM",
    "RCCL_PROTO": "RCCL_PROTO",
    "RCCL_ALGO": "RCCL_ALGO",
    "deepep_cu": "turbo_deepep_num_cu",
    "fp8": "fp8_recipe",
    "rope": "apply_rope_fusion",
    "rope_fusion": "apply_rope_fusion",
}


def _canonicalize_axis(token: str) -> str:
    """Resolve a headline-prose axis token to its catalog-canonical form.

    Falls back to the original token if no alias matches.
    """
    if token in _AXIS_PREFIX_ALIAS:
        return _AXIS_PREFIX_ALIAS[token]
    # Prefix-match against known axes (e.g. `OMP_NUM` → `OMP_NUM_THREADS`).
    if not _axt.is_known(token):
        for known in (
            _axt._TRAINER_OVERRIDE_AXES.keys() | _axt._STRUCTURAL_AXES.keys() | _axt._ENV_AXES.keys()
        ):
            if known.startswith(token + "_") or known.lower() == token.lower():
                return known
    return token


def _extract_axis_value_gain(
    text: str,
    *,
    require_promote_marker: bool = False,
) -> list[tuple[str, str, float]]:
    """Best-effort axis=value+gain extraction from a headline.

    Guards (lower the false-positive rate observed on session
    20260513T024603Z):
      - Axis token must be ≥4 chars and not in the blocklist.
      - Gain magnitude must be ≥0.5% (filters "exit rc=0 ... -100%" noise).
      - When ``require_promote_marker``: text must contain "promote" or
        "PROMOTED" (case-insensitive). Used by the catalog-gap finder
        to consume only ground-truth winning rows.
    """
    if require_promote_marker and not re.search(r"promot", text or "", re.I):
        return []
    out: list[tuple[str, str, float]] = []
    for m in _HEADLINE_AXIS_RE.finditer(text or ""):
        token = m.group(1)
        if token in _AXIS_BLOCKLIST:
            continue
        canonical = _canonicalize_axis(token)
        try:
            gain = float(m.group(3))
        except ValueError:
            continue
        if abs(gain) < 0.5:
            continue
        out.append((canonical, m.group(2), gain))
    return out


# ---------------------------------------------------------------------------
# Finding builders
# ---------------------------------------------------------------------------


def _find_constraint_gaps(trials: list[dict[str, Any]]) -> list[ConstraintGap]:
    by_rule: dict[str, ConstraintGap] = {}
    for t in trials:
        # Fingerprint matching: the `failure.message` is usually generic
        # ("job exited with status=failed"), so we union it with the
        # tail of `train.log` (already populated by `_iter_exec_trials`
        # for non-zero-exit trials).
        failure = t.get("failure") or {}
        haystack = " ".join(x for x in [failure.get("message") or "", t.get("log_tail") or ""] if x).strip()
        if not haystack:
            continue
        for fp in _MUTEX_FINGERPRINTS:
            m = fp.pattern.search(haystack)
            if not m:
                continue
            gap = by_rule.get(fp.rule_id)
            run_id = t.get("run_id") or "unknown"
            # Surface the matched text (the line that actually fired) rather
            # than the generic failure.message — that's what the curator
            # needs to write the next constraint rule.
            evidence_line = m.group(0)[:200]
            if gap is None:
                by_rule[fp.rule_id] = ConstraintGap(
                    rule_id=fp.rule_id,
                    failure_match=evidence_line,
                    axes=list(fp.axes),
                    rationale=fp.rationale,
                    observed_in_runs=[run_id],
                )
            else:
                if run_id not in gap.observed_in_runs:
                    gap.observed_in_runs.append(run_id)
            break
    return list(by_rule.values())


def _aggregate_axis_signals(
    stage_history: list[dict[str, Any]],
    run_history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Best-observed gain + run ids per (axis, value)."""
    agg: dict[str, dict[str, Any]] = {}

    def _record(axis: str, value: Any, gain: float, run_id: str | None) -> None:
        key = f"{axis}={value}"
        slot = agg.setdefault(
            key,
            {"axis": axis, "value": value, "gains": [], "runs": []},
        )
        slot["gains"].append(gain)
        if run_id and run_id not in slot["runs"]:
            slot["runs"].append(run_id)

    for entry in stage_history or []:
        text = entry.get("headline") or ""
        stage = (entry.get("stage") or "").upper()
        # Only consume promote / settle headlines as ground truth — every
        # other stage (DIAGNOSE, REPLAN, EXECUTE) describes intent /
        # progress, not the post-correctness verdict.
        require_marker = stage not in ("SETTLE",)
        for axis, value, gain in _extract_axis_value_gain(text, require_promote_marker=require_marker):
            _record(axis, value, gain, None)

    for item in run_history or []:
        gain = item.get("gain_vs_champion_pct")
        if gain is None and "gain_vs_champion" in item:
            try:
                gain = float(item["gain_vs_champion"]) * 100.0
            except (TypeError, ValueError):
                gain = None
        if gain is None:
            continue
        run_id = item.get("id") or item.get("run_id")
        for axis, value in (item.get("overrides") or {}).items():
            _record(axis, value, float(gain), run_id)
        for axis, value in (item.get("env_overrides") or {}).items():
            _record(axis, value, float(gain), run_id)

    return agg


def _classify_catalog_gap(axis: str) -> tuple[str, str | None, str | None]:
    """Return (status, channel_hint, section_hint).

    status ∈ {"known", "unknown"}.
    """
    if _axt.is_known(axis):
        return "known", _axt.channel_of(axis), None
    # Heuristic taxonomy hints for the proposal payload.
    upper_axis = axis.isupper() or "_" in axis and axis.split("_")[0].isupper()
    if upper_axis:
        return "unknown", "env", "§2.10/§2.11/§2.12 (env families)"
    return "unknown", "trainer_override", "§2.6/§2.7 (FP8 / fusions)"


def _find_catalog_gaps(agg: dict[str, dict[str, Any]]) -> list[CatalogGap]:
    by_axis: dict[str, CatalogGap] = {}
    for slot in agg.values():
        axis = slot["axis"]
        status, channel, section = _classify_catalog_gap(axis)
        if status == "known":
            continue
        positive_gains = [g for g in slot["gains"] if g > 0]
        best = max(positive_gains) if positive_gains else None
        gap = by_axis.setdefault(
            axis,
            CatalogGap(
                axis=axis,
                values_tried=[],
                best_observed_gain_pct=best,
                proposed_section=section,
                proposed_type=("strongly_local" if best and best >= 10 else "weakly_local"),
                channel=channel,
            ),
        )
        if slot["value"] not in gap.values_tried:
            gap.values_tried.append(slot["value"])
        for r in slot["runs"]:
            if r not in gap.evidence_run_ids:
                gap.evidence_run_ids.append(r)
        if best is not None and (gap.best_observed_gain_pct is None or best > gap.best_observed_gain_pct):
            gap.best_observed_gain_pct = best
            gap.proposed_type = "strongly_local" if best >= 10 else "weakly_local"
    return list(by_axis.values())


def _find_anti_patterns(agg: dict[str, dict[str, Any]]) -> list[AntiPatternSignal]:
    out: list[AntiPatternSignal] = []
    for slot in agg.values():
        # Anti-pattern when ALL observations are regressions and the worst is
        # ≤ -5%. A single random regression doesn't qualify.
        gains = slot["gains"]
        if not gains:
            continue
        if any(g >= 0 for g in gains):
            continue
        worst = min(gains)
        if worst > -5.0:
            continue
        out.append(
            AntiPatternSignal(
                axis=slot["axis"],
                value=slot["value"],
                measured_regression_pct=worst,
                proposed_note=(
                    f"axis_taxonomy DANGER row: setting {slot['axis']}={slot['value']} "
                    f"measured {worst:.2f}% (worst across {len(gains)} obs)"
                ),
                evidence_run_ids=list(slot["runs"]),
            )
        )
    return out


def _find_calibration_drifts(
    diagnosis_reports: list[dict[str, Any]],
    agg: dict[str, dict[str, Any]],
) -> list[CalibrationDrift]:
    out: list[CalibrationDrift] = []
    band_by_axis: dict[str, tuple[float, float]] = {}
    for report in diagnosis_reports or []:
        for ax in report.get("candidate_axes") or []:
            band = ax.get("expected_gain_band_pct")
            if isinstance(band, list) and len(band) == 2:
                try:
                    band_by_axis[ax["axis"]] = (float(band[0]), float(band[1]))
                except (TypeError, ValueError):
                    continue
    for axis, band in band_by_axis.items():
        gains = [slot["gains"][0] for slot in agg.values() if slot["axis"] == axis and slot["gains"]]
        if not gains:
            continue
        measured = max(gains, key=abs)
        lo, hi = band
        if measured < min(0, lo):
            verdict = "regressed"
        elif lo <= measured <= hi:
            verdict = "in_band"
        elif measured > hi:
            verdict = "above_band"
        else:
            verdict = "below_band"
        out.append(
            CalibrationDrift(
                axis=axis,
                predicted_band_pct=(lo, hi),
                measured_gain_pct=measured,
                verdict=verdict,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Suggested drafts
# ---------------------------------------------------------------------------


def _draft_for_constraint_gap(gap: ConstraintGap, session_id: str) -> dict[str, Any]:
    headline = f"{gap.rule_id}: {gap.rationale}"[:200]
    return {
        "kind": "failure_pattern",
        "id_suffix": gap.rule_id,
        "headline": headline,
        "evidence": [{"kind": "round_result", "ref": rid} for rid in gap.observed_in_runs],
        "binding": {"session_id": session_id},
        "content": {
            "rule_id": gap.rule_id,
            "axes": gap.axes,
            "rationale": gap.rationale,
            "failure_match": gap.failure_match,
        },
    }


def _draft_for_catalog_gap(gap: CatalogGap, session_id: str) -> dict[str, Any]:
    win = f" (best gain {gap.best_observed_gain_pct:+.2f}%)" if gap.best_observed_gain_pct is not None else ""
    headline = (f"axis catalog gap: {gap.axis} (channel={gap.channel}, " f"type={gap.proposed_type}){win}")[
        :200
    ]
    is_env_only = gap.channel == "env"
    is_promoted_winner = gap.best_observed_gain_pct is not None and gap.best_observed_gain_pct >= 2.0
    if is_env_only and is_promoted_winner:
        kind = "env_recipe"
    elif is_promoted_winner:
        kind = "final_best_case"
    else:
        kind = "failure_pattern"
    return {
        "kind": kind,
        "id_suffix": f"axis_{gap.axis}",
        "headline": headline,
        "evidence": [{"kind": "round_result", "ref": rid} for rid in gap.evidence_run_ids],
        "binding": {"session_id": session_id},
        "content": {
            "axis": gap.axis,
            "values_tried": gap.values_tried,
            "best_observed_gain_pct": gap.best_observed_gain_pct,
            "proposed_section": gap.proposed_section,
            "proposed_type": gap.proposed_type,
            "channel": gap.channel,
        },
    }


def _draft_for_anti_pattern(sig: AntiPatternSignal, session_id: str) -> dict[str, Any]:
    return {
        "kind": "failure_pattern",
        "id_suffix": f"danger_{sig.axis}",
        "headline": (
            f"DANGER: {sig.axis}={sig.value} measured " f"{sig.measured_regression_pct:+.2f}% (anti-pattern)"
        )[:200],
        "evidence": [{"kind": "round_result", "ref": rid} for rid in sig.evidence_run_ids],
        "binding": {"session_id": session_id},
        "content": {
            "axis": sig.axis,
            "value": sig.value,
            "measured_regression_pct": sig.measured_regression_pct,
            "proposed_note": sig.proposed_note,
        },
    }


def _draft_for_calibration_drift(d: CalibrationDrift, session_id: str) -> dict[str, Any]:
    return {
        "kind": "model_calibration_drift",
        "id_suffix": f"drift_{d.axis}",
        "headline": (
            f"calibration drift: {d.axis} predicted {list(d.predicted_band_pct)} "
            f"measured {d.measured_gain_pct:+.2f}% ({d.verdict})"
        )[:200],
        "evidence": [{"kind": "round_result", "ref": d.axis}],
        "binding": {"session_id": session_id},
        "content": asdict(d),
    }


# ---------------------------------------------------------------------------
# Public entry: analyze + emit
# ---------------------------------------------------------------------------


def analyze(
    session_dir: str | Path,
    *,
    diagnosis_reports: list[dict[str, Any]] | None = None,
) -> LearnAnalysis:
    """Run the full set of finders and produce a :class:`LearnAnalysis`."""
    sd = Path(session_dir).expanduser()
    if not sd.is_absolute():
        sd = _PILOT_ROOT / sd
    ts_path = sd / "tuning_state.yaml"
    if not ts_path.exists():
        raise FileNotFoundError(f"tuning_state.yaml not found under {sd}")

    tuning_state = _load_yaml(ts_path)
    session_id = tuning_state.get("session_id") or sd.name
    stage_history = tuning_state.get("stage_history") or []
    run_history = tuning_state.get("run_history") or []

    trials = _iter_exec_trials(sd)
    snapshots = _iter_snapshot_entries(sd)

    constraint_gaps = _find_constraint_gaps(trials)
    agg = _aggregate_axis_signals(stage_history, run_history)
    catalog_gaps = _find_catalog_gaps(agg)
    anti_patterns = _find_anti_patterns(agg)
    calibration = _find_calibration_drifts(diagnosis_reports or [], agg)

    suggested: list[dict[str, Any]] = []
    suggested.extend(_draft_for_constraint_gap(g, session_id) for g in constraint_gaps)
    suggested.extend(_draft_for_catalog_gap(g, session_id) for g in catalog_gaps)
    suggested.extend(_draft_for_anti_pattern(s, session_id) for s in anti_patterns)
    suggested.extend(_draft_for_calibration_drift(d, session_id) for d in calibration)

    return LearnAnalysis(
        schema_version="1.0",
        session_id=session_id,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        inputs={
            "tuning_state_ref": str(ts_path),
            "trial_count": len(trials),
            "snapshot_count": len(snapshots),
            "stage_history_len": len(stage_history),
            "run_history_len": len(run_history),
            "diagnosis_reports_provided": bool(diagnosis_reports),
        },
        findings={
            "catalog_gaps": [asdict(g) for g in catalog_gaps],
            "constraint_gaps": [asdict(g) for g in constraint_gaps],
            "calibration_drifts": [asdict(d) for d in calibration],
            "anti_pattern_signals": [asdict(s) for s in anti_patterns],
        },
        suggested_drafts=suggested,
    )


def emit_drafts(
    analysis: LearnAnalysis,
    *,
    drafts_root: str = "state/knowledge_drafts",
) -> list[dict[str, Any]]:
    """For each suggested draft, call :func:`pilot.tools.knowledge.write`.

    Returns the list of draft-write results so the caller can inspect which
    were accepted vs auto-rejected by the §S4.2 anti-patterns.
    """
    results: list[dict[str, Any]] = []
    for d in analysis.suggested_drafts:
        report = {
            "session": {"plan_name": analysis.session_id, "plan_ref": d["binding"].get("session_id")},
            "verdict": {"headline": d["headline"]},
            "tuning": {"champion": {}},
            "artifacts": d["evidence"],
            "content": d.get("content") or {},
        }
        try:
            res = _knowledge.write(
                report,
                d["kind"],
                drafts_root=drafts_root,
                id_suffix=d.get("id_suffix"),
            )
        except Exception as exc:  # noqa: BLE001
            res = {
                "written_path": "",
                "draft_id": "",
                "accepted": False,
                "reasons": [f"knowledge.write raised: {exc}"],
            }
        res["proposed_kind"] = d["kind"]
        res["proposed_headline"] = d["headline"]
        results.append(res)
    return results


def write_analysis(analysis: LearnAnalysis, *, root: str = "state/learn") -> str:
    """Persist the analysis YAML under ``state/learn/<session_id>/``."""
    out_dir = _PILOT_ROOT / root / analysis.session_id
    out_path = out_dir / "learn_analysis.yaml"
    payload = asdict(analysis)
    _atomic_write_yaml(out_path, payload)
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.learn")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser(
        "analyze",
        help="Derive between-session findings from a finished tuning session.",
    )
    p_an.add_argument(
        "--session", required=True, help="Path to the session directory (containing tuning_state.yaml)."
    )
    p_an.add_argument(
        "--diagnosis-glob",
        default=None,
        help="Optional glob for DiagnosisReport YAMLs to feed calibration drift.",
    )
    p_an.add_argument(
        "--write-analysis", action="store_true", help="Persist the analysis under state/learn/<session>/."
    )
    p_an.add_argument(
        "--emit-drafts", action="store_true", help="Also emit one KnowledgeDraft per suggested finding."
    )
    p_an.add_argument("--drafts-root", default="state/knowledge_drafts")

    args = p.parse_args()
    try:
        if args.cmd == "analyze":
            diagnosis_reports: list[dict[str, Any]] = []
            if args.diagnosis_glob:
                for path in glob.glob(args.diagnosis_glob):
                    try:
                        diagnosis_reports.append(_load_yaml(path))
                    except Exception:  # noqa: BLE001
                        continue
            analysis = analyze(args.session, diagnosis_reports=diagnosis_reports)
            payload = asdict(analysis)
            payload["written_path"] = None
            payload["drafts"] = []
            if args.write_analysis:
                payload["written_path"] = write_analysis(analysis)
            if args.emit_drafts:
                payload["drafts"] = emit_drafts(analysis, drafts_root=args.drafts_root)
            _emit(payload)
            return 0
    except Exception as exc:  # noqa: BLE001
        _emit(
            {
                "stage": "LEARN",
                "status": "failed",
                "failure": {
                    "kind": "TOOL_ERROR",
                    "message": str(exc),
                    "escalate_to_orchestrator": True,
                },
            }
        )
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
