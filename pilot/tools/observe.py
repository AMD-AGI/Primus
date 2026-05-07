"""pilot.tools.observe — point-in-time view of a running training job.

Consumes a `RunHandle` (produced by `pilot.tools.submit run`) and tails the
captured `train.log` to produce a structured `RunSnapshot` (validated against
``schemas/run_snapshot.schema.json``). The snapshot is the primary artifact
SMOKE / SCAN_PARALLEL / DIAGNOSE consume.

Two CLIs:

    observe snapshot --run-id <id> [--log-dir state/runs] [--tail-bytes N]
                                   [--hang-threshold-s 120] [--save]
    observe watch    --run-id <id> [--interval-s 5] [--max-snapshots N]
                                   [--until-terminal]

`compare_loss` (CORRECTNESS gate) remains a stub for now; it requires a
reference curve which only exists post-T0.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.tools import submit as _submit


# ---------------------------------------------------------------------------
# Path helpers (mirror submit.py)
# ---------------------------------------------------------------------------

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent


def _resolve_pilot_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else _PILOT_ROOT / pp


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _ObserveError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Cap parsing cost. 4 MiB tail covers ~30k Megatron log lines with TFLOPS.
_DEFAULT_TAIL_BYTES = 4 * 1024 * 1024
# Max history retained per metric in the snapshot.
_HISTORY_CAP = 50
# Symptom evidence cap (most-recent N matches across all kinds).
_EVIDENCE_CAP = 16
# Default hang threshold (seconds without a new iter line on a live process).
_DEFAULT_HANG_THRESHOLD_S = 120.0


# ---------------------------------------------------------------------------
# Regex (compiled once)
# ---------------------------------------------------------------------------

# ANSI color stripper (Primus + tee both color the log).
_RE_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Megatron's per-iter log line. Format (from Megatron-LM training.py):
#   [YYYY-MM-DD HH:MM:SS.ffffff] iteration <i>/<N> | consumed samples: ... |
#       elapsed time per iteration (ms): X.X | [throughput per GPU (TFLOP/s/GPU): T |]
#       learning rate: ... | global batch size: ... | <key>: <val> | ...
_RE_ITER = re.compile(
    r"\[(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]"
    r".*?iteration\s+(?P<cur>\d+)\s*/\s*(?P<tot>\d+)\s*\|"
)
_RE_KV_INT = lambda key: re.compile(rf"{key}:\s*(?P<v>-?\d+)")  # noqa: E731
_RE_KV_FLOAT = lambda key: re.compile(  # noqa: E731
    rf"{key}:\s*(?P<v>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
)
_RE_CONSUMED = _RE_KV_INT("consumed samples")
_RE_ELAPSED_MS = _RE_KV_FLOAT(r"elapsed time per iteration \(ms\)")
_RE_TFLOPS = _RE_KV_FLOAT(r"throughput per GPU \(TFLOP/s/GPU\)")
_RE_LR = _RE_KV_FLOAT("learning rate")
_RE_GRAD_NORM = _RE_KV_FLOAT("grad norm")
# Loss key is something like `lm loss` / `loss`. Megatron prints the dict keys
# verbatim; pick whichever matches first.
_RE_LOSS = re.compile(
    r"(?:^|\|)\s*(?:lm\s+)?loss:\s*(?P<v>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|nan|inf|-inf)"
)

# Symptom regexes — case-insensitive.
_SYMPTOM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # OOM (CUDA + ROCm flavors).
    ("oom", re.compile(
        r"(?i)\b(out of memory|cuda out of memory|hip out of memory|"
        r"hipErrorOutOfMemory|cudaErrorMemoryAllocation)\b"
    )),
    # NCCL / RCCL / collective watchdog timeouts.
    ("nccl", re.compile(
        r"(?i)\b(NCCL|RCCL)[A-Z_ ]*(error|failure|timeout|abort)|"
        r"ProcessGroupNCCL.*Timeout|"
        r"Watchdog caught collective operation timeout|"
        r"NCCL_PROXY_TIMEOUT|"
        r"NCCL WARN"
    )),
    # CUDA / HIP runtime errors (excluding the OOM ones we already captured).
    ("cuda", re.compile(
        r"(?i)\b(CUDA error|HIP error|hipError(?!OutOfMemory)|"
        r"cudaError(?!MemoryAllocation)|illegal memory access|"
        r"device-side assert)\b"
    )),
    # Python tracebacks / unhandled exceptions.
    ("python_error", re.compile(
        r"(?m)^(Traceback \(most recent call last\):|"
        r"\s*RuntimeError:|\s*AssertionError:|\s*ValueError:)"
    )),
]

# Hang hints — these don't change `hang_suspected` directly (that's
# wall-time-based) but get recorded as evidence. Word boundaries are required
# because bare "hang" matches inside benign words like "exchange_algo".
_RE_HANG_HINT = re.compile(
    r"(?i)\b("
    r"stuck|deadlock(?:ed)?|hung|hanging\s+process|"
    r"barrier\s+timeout|gloo\s+socket\s+timeout|watchdog\s+timeout|"
    r"no\s+progress\s+for"
    r")\b"
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclass
class _ParsedIter:
    iter_num: int
    total_iters: int
    ts: str
    consumed_samples: int | None = None
    iter_time_ms: float | None = None
    tflops: float | None = None
    learning_rate: float | None = None
    loss: float | None = None
    grad_norm: float | None = None


@dataclass
class _ParseResult:
    iters: list[_ParsedIter] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    saw_loss_nan_or_inf: bool = False


def _to_float(s: str) -> float:
    s = s.strip().lower()
    if s == "nan":
        return float("nan")
    if s == "inf":
        return float("inf")
    if s == "-inf":
        return float("-inf")
    return float(s)


def _parse_log(
    log_path: Path,
    *,
    tail_bytes: int = _DEFAULT_TAIL_BYTES,
) -> tuple[_ParseResult, dict[str, int | None]]:
    """Tail the log file (last ``tail_bytes``) and extract iter rows + symptoms.

    Returns (parsed, log_meta). log_meta has total file size + line count of
    the tail we actually read.
    """
    result = _ParseResult()
    meta: dict[str, int | None] = {"bytes": None, "lines": None, "tailed_bytes": None}

    if not log_path.exists():
        return result, meta

    try:
        size = log_path.stat().st_size
    except OSError:
        return result, meta
    meta["bytes"] = size

    seek = max(0, size - max(0, int(tail_bytes)))
    try:
        with log_path.open("rb") as f:
            f.seek(seek)
            blob = f.read()
    except OSError:
        return result, meta

    text = blob.decode("utf-8", errors="replace")
    # If we seeked past the start, the first line is likely partial. Drop it.
    if seek > 0:
        nl = text.find("\n")
        text = text[nl + 1 :] if nl >= 0 else ""

    meta["tailed_bytes"] = len(text.encode("utf-8"))

    lines = text.splitlines()
    meta["lines"] = len(lines)

    # Megatron sometimes emits the same iter line at two log levels (utils.py
    # 425 + 429). Dedupe by (iter_num, ts) so history isn't doubled.
    seen_iters: set[tuple[int, str]] = set()

    # Track most-recent evidence per kind, but keep insertion order overall.
    evidence: list[dict[str, Any]] = []

    for offset, raw in enumerate(lines, start=1):
        line = _RE_ANSI.sub("", raw)

        m = _RE_ITER.search(line)
        if m:
            key = (int(m.group("cur")), m.group("ts"))
            if key in seen_iters:
                continue
            seen_iters.add(key)
            it = _ParsedIter(
                iter_num=int(m.group("cur")),
                total_iters=int(m.group("tot")),
                ts=m.group("ts"),
            )
            mm = _RE_CONSUMED.search(line)
            if mm:
                it.consumed_samples = int(mm.group("v"))
            mm = _RE_ELAPSED_MS.search(line)
            if mm:
                try:
                    it.iter_time_ms = _to_float(mm.group("v"))
                except ValueError:
                    pass
            mm = _RE_TFLOPS.search(line)
            if mm:
                try:
                    it.tflops = _to_float(mm.group("v"))
                except ValueError:
                    pass
            mm = _RE_LR.search(line)
            if mm:
                try:
                    it.learning_rate = _to_float(mm.group("v"))
                except ValueError:
                    pass
            mm = _RE_GRAD_NORM.search(line)
            if mm:
                try:
                    it.grad_norm = _to_float(mm.group("v"))
                except ValueError:
                    pass
            mm = _RE_LOSS.search(line)
            if mm:
                try:
                    val = _to_float(mm.group("v"))
                    it.loss = val
                    if not math.isfinite(val):
                        result.saw_loss_nan_or_inf = True
                except ValueError:
                    pass
            result.iters.append(it)
            continue

        # Symptom matching (keep at most _EVIDENCE_CAP, most recent wins).
        for kind, pat in _SYMPTOM_PATTERNS:
            if pat.search(line):
                evidence.append({
                    "kind": kind,
                    "line_no": offset,
                    "line": line.strip()[:512],
                })
                break
        else:
            if _RE_HANG_HINT.search(line):
                evidence.append({
                    "kind": "hang_hint",
                    "line_no": offset,
                    "line": line.strip()[:512],
                })

    result.evidence = evidence[-_EVIDENCE_CAP:]
    return result, meta


# ---------------------------------------------------------------------------
# Snapshot assembly
# ---------------------------------------------------------------------------


def _isoformat_to_aware(ts: str) -> datetime | None:
    """Megatron prints `2026-05-04 03:16:01.123456` — local time, no tz.

    We assume UTC for snapshot math; this is consistent with how we render
    `created_at` and `snapshot_at`.
    """
    fmts = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
    for fmt in fmts:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _iters_per_min(iters: list[_ParsedIter], window: int = 5) -> float | None:
    if len(iters) < 2:
        return None
    tail = iters[-window:] if len(iters) >= window else iters
    t0 = _isoformat_to_aware(tail[0].ts)
    t1 = _isoformat_to_aware(tail[-1].ts)
    if not t0 or not t1:
        return None
    elapsed = (t1 - t0).total_seconds()
    if elapsed <= 0:
        return None
    n = tail[-1].iter_num - tail[0].iter_num
    if n <= 0:
        return None
    return round(n / elapsed * 60.0, 2)


def _recommend(
    *,
    process_alive: bool,
    handle_status: str,
    progress_pct: float | None,
    silent_for_s: float | None,
    hang_threshold_s: float,
    symptoms_summary: dict[str, bool],
) -> dict[str, str]:
    """Map status + symptoms to a next-step suggestion. Advisory only."""
    if any(symptoms_summary[k] for k in ("oom_detected", "nccl_error", "cuda_error")):
        return {"next": "cancel_and_diagnose",
                "reason": "hard symptom detected (OOM / collective / CUDA error)"}
    if symptoms_summary["loss_nan_or_inf"]:
        return {"next": "cancel_and_diagnose",
                "reason": "loss is NaN or Inf"}
    if symptoms_summary["hang_suspected"]:
        return {"next": "cancel_and_diagnose",
                "reason": f"no new iter line for >{hang_threshold_s:.0f}s while process alive"}
    if handle_status == "completed":
        return {"next": "promote", "reason": "training reached terminal state with rc=0"}
    if handle_status == "failed":
        return {"next": "escalate_diagnose",
                "reason": "process exited with non-zero rc; check train.log + symptoms"}
    if handle_status == "killed":
        return {"next": "wait", "reason": "run was cancelled; nothing to observe"}
    if handle_status == "running" and process_alive:
        if progress_pct is not None and progress_pct >= 100.0:
            return {"next": "wait", "reason": "all iters logged; awaiting clean exit"}
        return {"next": "continue",
                "reason": f"training advancing ({progress_pct:.1f}%)"
                if progress_pct is not None
                else "training initializing; first iter not yet logged"}
    return {"next": "wait", "reason": f"status={handle_status}; no action yet"}


def snapshot(
    run_id: str,
    *,
    log_dir: str | Path = "state/runs",
    tail_bytes: int = _DEFAULT_TAIL_BYTES,
    hang_threshold_s: float = _DEFAULT_HANG_THRESHOLD_S,
    save: bool = False,
) -> dict[str, Any]:
    """Produce a RunSnapshot for ``run_id``."""
    snapshot_at = datetime.now(timezone.utc)

    # 1) Refresh handle (this also reconciles status: running/killed/etc.)
    refreshed = _submit.status(run_id, log_dir=log_dir)
    handle_path, handle = _submit._load_handle(run_id, log_dir)  # noqa: SLF001
    handle_status: str = handle["status"]
    pid = handle["launch"].get("pid")
    log_ref = handle["log"]["stdout"]
    log_path = Path(log_ref)

    process_alive = _submit._is_alive_not_zombie(pid)  # noqa: SLF001

    # 2) Parse log tail.
    parsed, meta = _parse_log(log_path, tail_bytes=tail_bytes)

    # 3) Progress.
    iters = parsed.iters
    last_iter_at: datetime | None = None
    silent_for_s: float | None = None
    current_iter: int | None = None
    total_iters: int | None = None
    pct: float | None = None
    if iters:
        last = iters[-1]
        current_iter = last.iter_num
        total_iters = last.total_iters
        if total_iters > 0:
            pct = round(min(100.0, current_iter / total_iters * 100.0), 1)
        last_iter_at = _isoformat_to_aware(last.ts)
        if last_iter_at:
            silent_for_s = round((snapshot_at - last_iter_at).total_seconds(), 2)

    # 4) Symptoms.
    kinds_seen: set[str] = {ev["kind"] for ev in parsed.evidence}
    hang_suspected = bool(
        process_alive
        and silent_for_s is not None
        and silent_for_s > hang_threshold_s
    )
    symptoms_summary = {
        "hang_suspected":  hang_suspected,
        "oom_detected":    "oom" in kinds_seen,
        "nccl_error":      "nccl" in kinds_seen,
        "cuda_error":      "cuda" in kinds_seen,
        "python_error":    "python_error" in kinds_seen,
        "loss_nan_or_inf": parsed.saw_loss_nan_or_inf,
    }

    # 5) Effective status.
    if hang_suspected and handle_status == "running":
        eff_status = "hung"
    else:
        eff_status = handle_status

    # 6) History (cap last N).
    hist_iters = [i.iter_num for i in iters[-_HISTORY_CAP:]]
    hist_loss = [i.loss for i in iters[-_HISTORY_CAP:]]
    hist_iter_time = [i.iter_time_ms for i in iters[-_HISTORY_CAP:]]
    hist_tflops = [i.tflops for i in iters[-_HISTORY_CAP:]]

    # 7) Latest metrics.
    latest = iters[-1] if iters else None
    latest_metrics: dict[str, Any] = {
        "loss":             latest.loss if latest else None,
        "iter_time_ms":     latest.iter_time_ms if latest else None,
        "tflops":           latest.tflops if latest else None,
        "consumed_samples": latest.consumed_samples if latest else None,
        "learning_rate":    latest.learning_rate if latest else None,
        "grad_norm":        latest.grad_norm if latest else None,
    }
    # Sanitize NaN/Inf for JSON.
    for k, v in list(latest_metrics.items()):
        if isinstance(v, float) and not math.isfinite(v):
            latest_metrics[k] = None

    snap: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "snapshot_at": snapshot_at.isoformat(timespec="seconds"),
        "status": eff_status,
        "process": {
            "pid": pid,
            "alive": bool(process_alive),
            "exit_code": refreshed.get("exit_code"),
            "wallclock_s": refreshed.get("wallclock_s"),
        },
        "progress": {
            "current_iter": current_iter,
            "total_iters":  total_iters,
            "pct":          pct,
            "iters_per_min": _iters_per_min(iters),
            "last_iter_at": last_iter_at.isoformat(timespec="seconds") if last_iter_at else None,
            "silent_for_s": silent_for_s,
        },
        "metrics": {
            "latest": latest_metrics,
            "history": {
                "iters":        hist_iters,
                "loss":         [None if (v is None or not math.isfinite(v)) else v
                                 for v in hist_loss],
                "iter_time_ms": [None if (v is None or not math.isfinite(v)) else v
                                 for v in hist_iter_time],
                "tflops":       [None if (v is None or not math.isfinite(v)) else v
                                 for v in hist_tflops],
            },
            "loss_finite": not parsed.saw_loss_nan_or_inf,
        },
        "symptoms": {
            **symptoms_summary,
            "hang_threshold_s": hang_threshold_s,
            "evidence": parsed.evidence,
        },
        "log": {
            "ref":          str(log_path),
            "bytes":        meta["bytes"],
            "lines":        meta["lines"],
            "tailed_bytes": meta["tailed_bytes"],
        },
        "recommendation": _recommend(
            process_alive=process_alive,
            handle_status=handle_status,
            progress_pct=pct,
            silent_for_s=silent_for_s,
            hang_threshold_s=hang_threshold_s,
            symptoms_summary=symptoms_summary,
        ),
    }

    # Schema-validate before any caller touches the result. Catches both
    # parser drift and recommendation enum bugs at the source.
    from pilot.tools._schema import validate
    validate(snap, "run_snapshot")

    if save:
        _persist_snapshot(run_id, log_dir, snap)

    return snap


def _persist_snapshot(run_id: str, log_dir: str | Path, snap: dict[str, Any]) -> Path:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _ObserveError("DEP_MISSING", f"PyYAML required: {exc}")
    snap_dir = _resolve_pilot_path(log_dir) / run_id / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = snap["snapshot_at"].replace(":", "").replace("-", "")
    out = snap_dir / f"{ts}.yaml"
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(snap, f, sort_keys=False, default_flow_style=False)
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------


_TERMINAL = {"completed", "failed", "killed", "unknown", "hung"}


def watch(
    run_id: str,
    *,
    log_dir: str | Path = "state/runs",
    interval_s: float = 5.0,
    max_snapshots: int | None = None,
    until_terminal: bool = True,
    tail_bytes: int = _DEFAULT_TAIL_BYTES,
    hang_threshold_s: float = _DEFAULT_HANG_THRESHOLD_S,
    save: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Poll observe.snapshot until ``max_snapshots`` taken or terminal status."""
    n = 0
    last: dict[str, Any] | None = None
    while True:
        last = snapshot(
            run_id,
            log_dir=log_dir,
            tail_bytes=tail_bytes,
            hang_threshold_s=hang_threshold_s,
            save=save,
        )
        n += 1
        if verbose:
            _print_one_line(last)
        if until_terminal and last["status"] in _TERMINAL:
            break
        if max_snapshots is not None and n >= max_snapshots:
            break
        time.sleep(max(0.5, interval_s))
    return last or {}


def _print_one_line(snap: dict[str, Any]) -> None:
    p = snap["progress"]
    s = snap["status"]
    cur = p["current_iter"]
    tot = p["total_iters"]
    pct = p["pct"]
    silent = p["silent_for_s"]
    loss = snap["metrics"]["latest"]["loss"]
    tflops = snap["metrics"]["latest"]["tflops"]
    print(
        f"[{snap['snapshot_at']}] status={s} "
        f"iter={cur}/{tot} ({pct}%) "
        f"loss={loss} tflops={tflops} silent_for={silent}s "
        f"→ {snap['recommendation']['next']}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# compare_loss — kept as stub
# ---------------------------------------------------------------------------


def compare_loss(run_id: str, reference_curve: str) -> dict[str, Any]:
    """CORRECTNESS gate: tokens-aligned comparison vs T0/T1 reference (§S2).

    Stub — requires a reference loss curve produced by an earlier T0 baseline,
    which is out of scope for the SMOKE slice.
    """
    raise NotImplementedError("pilot.tools.observe.compare_loss")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK = 0
_EXIT_USAGE = 2


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "OBSERVE",
        "status": "failed",
        "failure": {"kind": kind, "message": message,
                    "escalate_to_orchestrator": True},
    }


def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="pilot.tools.observe",
        description="Snapshot or watch a training run launched by submit.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot", help="One-shot snapshot.")
    p_snap.add_argument("--run-id", required=True)
    p_snap.add_argument("--log-dir", default="state/runs")
    p_snap.add_argument("--tail-bytes", type=int, default=_DEFAULT_TAIL_BYTES)
    p_snap.add_argument("--hang-threshold-s", type=float, default=_DEFAULT_HANG_THRESHOLD_S)
    p_snap.add_argument("--save", action="store_true",
                        help="Persist snapshot YAML under state/runs/<id>/snapshots/.")

    p_watch = sub.add_parser("watch", help="Poll snapshot in a loop.")
    p_watch.add_argument("--run-id", required=True)
    p_watch.add_argument("--log-dir", default="state/runs")
    p_watch.add_argument("--interval-s", type=float, default=5.0)
    p_watch.add_argument("--max-snapshots", type=int, default=None)
    p_watch.add_argument("--no-until-terminal", action="store_true",
                         help="Don't auto-stop when status becomes terminal.")
    p_watch.add_argument("--tail-bytes", type=int, default=_DEFAULT_TAIL_BYTES)
    p_watch.add_argument("--hang-threshold-s", type=float, default=_DEFAULT_HANG_THRESHOLD_S)
    p_watch.add_argument("--no-save", action="store_true",
                         help="Don't persist per-tick YAML snapshots.")
    p_watch.add_argument("--quiet", action="store_true",
                         help="Suppress per-tick one-line summary.")

    p_cmp = sub.add_parser("compare_loss", help="(stub) CORRECTNESS gate.")
    p_cmp.add_argument("--run-id", required=True)
    p_cmp.add_argument("--reference", required=True)

    args = p.parse_args()
    try:
        if args.cmd == "snapshot":
            snap = snapshot(
                args.run_id,
                log_dir=args.log_dir,
                tail_bytes=args.tail_bytes,
                hang_threshold_s=args.hang_threshold_s,
                save=args.save,
            )
            _emit(snap)
            return _EXIT_OK

        if args.cmd == "watch":
            last = watch(
                args.run_id,
                log_dir=args.log_dir,
                interval_s=args.interval_s,
                max_snapshots=args.max_snapshots,
                until_terminal=not args.no_until_terminal,
                tail_bytes=args.tail_bytes,
                hang_threshold_s=args.hang_threshold_s,
                save=not args.no_save,
                verbose=not args.quiet,
            )
            # Final snapshot also emitted as JSON for downstream tools.
            _emit(last)
            return _EXIT_OK

        if args.cmd == "compare_loss":
            _emit(_failure("TOOL_ERROR", "compare_loss not implemented"))
            return _EXIT_USAGE

    except _ObserveError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return _EXIT_USAGE
    except _submit._SubmitError as exc:  # noqa: SLF001 — handle propagation
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
