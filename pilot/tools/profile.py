"""pilot.tools.profile — inject torch.profiler knobs into a plan.

Authoritative protocol: ``skills/workflow/profile.md`` (defaults, layout,
incompatibilities, gotchas).

Two responsibilities:

1. ``inject(plan, run_dir, ...)``:
   Returns the SAME plan dict with the profile-related Megatron flags filled in
   (``profile``, ``use_pytorch_profiler``, ``profile_step_start``,
   ``profile_step_end``, ``profile_ranks``, ``tensorboard_dir``, gzip toggle).
   Refuses to inject if a hard incompatibility is detected (HipBLASLT tuning).
2. ``collect(run_dir)``:
   After the training process exits, scans ``<run_dir>/profile/tb`` for the
   chrome-trace files torch.profiler emitted, waits up to 30s for the flush
   to settle, and writes ``<run_dir>/profile/trace_meta.json`` (the contract
   the trace_analyze tool reads).

Both functions are pure / idempotent: re-running them does not double-write or
crash.

CLI surface (manual debugging; the production path is ``submit run`` which
calls ``inject`` and ``collect`` automatically):

    python -m pilot.tools.profile inject \\
        --plan      state/runs/<id>/plan.effective.yaml \\
        --run-dir   state/runs/<id> \\
        --out-plan  state/runs/<id>/plan.effective.profiled.yaml

    python -m pilot.tools.profile collect \\
        --run-dir   state/runs/<id> \\
        [--wait-s 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/

DEFAULTS: dict[str, Any] = {
    "profile_step_start": 5,
    "profile_step_end": 6,
    "profile_ranks": [0],
    "min_train_iters": 8,
    "use_pytorch_profiler": True,
    "torch_profiler_use_gzip": True,
    "torch_profiler_record_shapes": False,
    "torch_profiler_with_stack": False,
    "disable_profiler_activity_cpu": False,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _ProfileError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


_EXIT_OK = 0
_EXIT_STAGE_FAILED = 1
_EXIT_USAGE = 2
_EXIT_TOOL_ERROR = 3


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise _ProfileError("DEP_MISSING", f"PyYAML required: {exc}") from exc
    return yaml


def _resolve_path(p: str | Path) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return pp
    cwd_path = Path.cwd() / pp
    if cwd_path.exists():
        return cwd_path
    pilot_path = _PILOT_ROOT / pp
    return pilot_path if pilot_path.exists() else cwd_path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise _ProfileError("USAGE", f"file not found: {p}")
    with p.open() as f:
        data = _yaml().safe_load(f)
    if not isinstance(data, dict):
        raise _ProfileError("USAGE", f"{p} must be a YAML mapping")
    return data


def _atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        _yaml().safe_dump(data, f, sort_keys=False)
    tmp.replace(path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


def _get_overrides(plan: dict[str, Any]) -> dict[str, Any]:
    if "modules" in plan:
        modules = plan.setdefault("modules", {})
        pre_trainer = modules.setdefault("pre_trainer", {})
        return pre_trainer.setdefault("overrides", {})
    if isinstance(plan.get("overrides"), dict):
        return plan["overrides"]
    return plan


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# inject() — the production entry point called from submit.run
# ---------------------------------------------------------------------------


def inject(
    plan: dict[str, Any],
    run_dir: str | Path,
    *,
    profile_step_start: int | None = None,
    profile_step_end: int | None = None,
    ranks: list[int] | None = None,
    enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (patched_plan, decision) where ``decision`` records what we did.

    ``decision`` is a small dict the caller embeds in the run handle so a
    reviewer can later see why profiling was on/off without re-deriving the
    rules. Shape::

        {
          "enabled": bool,
          "skipped_reason": str | None,
          "knobs": {...},
          "trace_dir": "<run_dir>/profile/tb",
          "warnings": [...],
        }
    """
    decision: dict[str, Any] = {
        "enabled": False,
        "skipped_reason": None,
        "knobs": {},
        "trace_dir": str(Path(run_dir) / "profile" / "tb"),
        "warnings": [],
    }
    out = deepcopy(plan)
    overrides = _get_overrides(out)

    if not enabled:
        decision["skipped_reason"] = "user passed --no-profile"
        return out, decision

    if str(overrides.get("PRIMUS_HIPBLASLT_TUNING") or "0") == "1":
        decision["skipped_reason"] = "PRIMUS_HIPBLASLT_TUNING=1 (incompatible with torch.profiler)"
        return out, decision

    start = profile_step_start if profile_step_start is not None else DEFAULTS["profile_step_start"]
    end = profile_step_end if profile_step_end is not None else DEFAULTS["profile_step_end"]
    if end <= start:
        end = start + 1

    train_iters = _to_int(overrides.get("train_iters"), 0)
    min_iters = max(DEFAULTS["min_train_iters"], end + 2)
    bumped = False
    if train_iters < min_iters:
        overrides["train_iters"] = min_iters
        bumped = True
        decision["warnings"].append(
            f"bumped train_iters from {train_iters} to {min_iters} so profiler can flush"
        )

    user_ranks = ranks if ranks is not None else DEFAULTS["profile_ranks"]
    knobs = {
        "profile": True,
        "use_pytorch_profiler": True,
        "profile_step_start": start,
        "profile_step_end": end,
        "profile_ranks": list(user_ranks),
        "tensorboard_dir": str(Path(run_dir) / "profile" / "tb"),
        "torch_profiler_use_gzip": DEFAULTS["torch_profiler_use_gzip"],
        "torch_profiler_record_shapes": DEFAULTS["torch_profiler_record_shapes"],
        "torch_profiler_with_stack": DEFAULTS["torch_profiler_with_stack"],
        "disable_profiler_activity_cpu": DEFAULTS["disable_profiler_activity_cpu"],
    }
    for k, v in knobs.items():
        overrides.setdefault(k, v)

    decision["enabled"] = True
    decision["knobs"] = {k: overrides[k] for k in knobs}
    decision["train_iters_bumped"] = bumped
    return out, decision


# ---------------------------------------------------------------------------
# collect() — called after training exits to write trace_meta.json
# ---------------------------------------------------------------------------


def collect(
    run_dir: str | Path,
    *,
    wait_s: int = 30,
    poll_s: float = 1.0,
    extra_search_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    rd = Path(run_dir)
    profile_dir = rd / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Primus's tensorboard_path_patches.py overrides any user-supplied
    # tensorboard_dir with `<exp_root>/tensorboard`. Until we patch that
    # behavior upstream, also scan well-known Primus exp_root locations so
    # the chrome trace doesn't escape our analysis pipeline.
    primus_root = Path(__file__).resolve().parent.parent.parent
    primus_output = primus_root / "output"
    primus_tb_globs: list[Path] = []
    if primus_output.exists():
        primus_tb_globs = list(primus_output.glob("*/*/*/tensorboard"))

    candidate_dirs = [
        profile_dir / "tb",
        profile_dir / "torch_profile",
        *(extra_search_dirs or []),
        *primus_tb_globs,
    ]
    waited = 0.0
    last_sizes: dict[str, int] = {}
    stable_polls = 0
    deadline = time.time() + wait_s
    trace_files: list[Path] = []
    while time.time() < deadline:
        trace_files = []
        for d in candidate_dirs:
            if not d.exists():
                continue
            for p in d.rglob("*.json*"):
                if p.is_file() and p.stat().st_size > 0:
                    trace_files.append(p)
        sizes = {str(p): p.stat().st_size for p in trace_files}
        if trace_files and sizes == last_sizes:
            stable_polls += 1
            if stable_polls >= 3:
                break
        else:
            stable_polls = 0
            last_sizes = sizes
        time.sleep(poll_s)
        waited += poll_s

    meta: dict[str, Any] = {
        "schema_version": "0.1",
        "run_id": rd.name,
        "captured_iter_start": DEFAULTS["profile_step_start"],
        "captured_iter_end": DEFAULTS["profile_step_end"],
        "captured_iter_count": DEFAULTS["profile_step_end"] - DEFAULTS["profile_step_start"],
        "ranks": DEFAULTS["profile_ranks"],
        "trace_files": [],
        "warnings": [],
    }

    if not trace_files:
        meta["warnings"].append(
            f"no trace files found under {[str(d) for d in candidate_dirs]} after {waited:.1f}s; "
            "profiler may have failed to flush, or training exited before profile_step_end"
        )
    else:
        # Mirror traces that landed outside the run_dir (Primus exp_root override)
        # into <run_dir>/profile/tb/ via symlink, so the analyzer + audit trail
        # only need to look in one place.
        tb_dir = profile_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(trace_files):
            try:
                rel = p.relative_to(rd)
                stored_path = str(rel)
                origin = "in_run_dir"
            except ValueError:
                link = tb_dir / p.name
                if not link.exists():
                    try:
                        link.symlink_to(p)
                    except OSError:
                        try:
                            import shutil
                            shutil.copy2(p, link)
                        except OSError as exc:
                            meta["warnings"].append(
                                f"could not mirror {p} into {link}: {exc}"
                            )
                            continue
                stored_path = str(link.relative_to(rd))
                origin = "primus_exp_root"
                meta["warnings"].append(
                    f"trace originated outside run_dir at {p}; "
                    "this is the Primus tensorboard_path_patches.py override"
                )
            rank = _rank_from_filename(p.name)
            meta["trace_files"].append({
                "rank": rank,
                "path": stored_path,
                "origin_path": str(p),
                "origin": origin,
                "bytes": p.stat().st_size,
                "format": "chrome_trace_v1" if p.suffix in (".json", ".gz") else "unknown",
            })

    out_path = profile_dir / "trace_meta.json"
    _atomic_write_json(out_path, meta)
    return meta


def _rank_from_filename(name: str) -> int | None:
    import re
    m = re.search(r"rank-?(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"rank\[(\d+)\]", name)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.profile")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_inj = sub.add_parser("inject", help="Patch a plan with profiler knobs.")
    p_inj.add_argument("--plan", required=True)
    p_inj.add_argument("--run-dir", required=True)
    p_inj.add_argument("--profile-step-start", type=int, default=None)
    p_inj.add_argument("--profile-step-end", type=int, default=None)
    p_inj.add_argument("--ranks", default=None, help="Comma-separated rank list (default: 0).")
    p_inj.add_argument("--no-profile", action="store_true")
    p_inj.add_argument("--out-plan", required=True)

    p_col = sub.add_parser("collect", help="Write trace_meta.json after training exits.")
    p_col.add_argument("--run-dir", required=True)
    p_col.add_argument("--wait-s", type=int, default=30)

    args = p.parse_args()

    try:
        if args.cmd == "inject":
            plan = _load_yaml(args.plan)
            ranks = (
                [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
                if args.ranks else None
            )
            patched, decision = inject(
                plan, args.run_dir,
                profile_step_start=args.profile_step_start,
                profile_step_end=args.profile_step_end,
                ranks=ranks,
                enabled=not args.no_profile,
            )
            out_path = _resolve_path(args.out_plan)
            _atomic_write_yaml(out_path, patched)
            _emit({"stage": "PROFILE.INJECT", "status": "ok", "decision": decision, "out_plan": str(out_path)})
            return _EXIT_OK if decision["enabled"] or args.no_profile else _EXIT_STAGE_FAILED

        if args.cmd == "collect":
            meta = collect(args.run_dir, wait_s=args.wait_s)
            _emit({"stage": "PROFILE.COLLECT", "status": "ok", "meta": meta})
            return _EXIT_OK if meta["trace_files"] else _EXIT_STAGE_FAILED
    except _ProfileError as exc:
        _emit({"stage": "PROFILE", "status": "failed", "failure": {"kind": exc.kind, "message": str(exc)}})
        return _EXIT_USAGE if exc.kind == "USAGE" else _EXIT_TOOL_ERROR

    return _EXIT_USAGE


if __name__ == "__main__":
    sys.exit(_cli())
