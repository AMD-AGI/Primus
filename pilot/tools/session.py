"""session tools: bootstrap a new tuning session.

A session is a single Pilot tuning run, namespaced by ``session_id`` and rooted
at ``pilot/state/<session_id>/``. ``pilot session init`` is the first step
every new session goes through: it materialises the directory and the three
canonical artifacts that subsequent stages read from / write to:

* ``tuning.yaml``        — session-wide bootstrap config (schema:
  ``tuning_config.schema.json``). Declares ``log_dir_prefix`` and per-stage
  ``trace/<stage>`` paths so every downstream tool agrees on where each stage
  writes its trace artifacts.
* ``target_vector.yaml`` — optimization target (legacy mirror; kept for tools
  that read it directly).
* ``tuning_state.yaml``  — initial TuningState (``current_stage=PREFLIGHT``);
  the ``state.checkpoint`` tool then seeds ``checkpoints/r0/`` from this.

Subsequent stages MUST log under ``<log_dir_prefix>/<trace_subdir>/<stage>/``:

  trace/preflight/, trace/projection/, trace/smoke/, trace/baseline/,
  trace/t1/, trace/t2/, ...,  trace/report/

The ``optimize_loop.dir_pattern`` field uses ``{trial_id}`` substitution at
trial launch (``trace/t{trial_id}`` → ``trace/t1``, ``trace/t2``, ...).

CLI surface
-----------
::

    session init   --plan <yaml> [--cluster-config <yaml>] [--session-id <id>]
                   [--primary <metric>] [--constraint <expr> ...]
                   [--rounds N]  [--candidates-per-round N]
                   [--smoke-iters N] [--train-iters N] [--timeout-s N]
                   [--base-override key=value ...]
                   [--node <hostname>] [--notes <str>] [--force]

Outputs (under ``pilot/state/<session_id>/``)
---------------------------------------------
* ``tuning.yaml``        — TuningConfig
* ``target_vector.yaml`` — TargetVector mirror
* ``tuning_state.yaml``  — initial TuningState
* ``checkpoints/r0/tuning_state.yaml`` — seeded via ``state.checkpoint``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/
_PRIMUS_ROOT: Path = _PILOT_ROOT.parent

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _SessionError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _resolve_pilot_path(p: str | Path) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else _PILOT_ROOT / pp


def _resolve_repo_path(p: str | Path) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else _PRIMUS_ROOT / pp


def _yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise _SessionError("DEP_MISSING", f"PyYAML required for session tools: {exc}") from exc
    return yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    yaml = _yaml()
    if not path.exists():
        raise _SessionError("USAGE", f"file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise _SessionError("USAGE", f"{path} must be a YAML mapping")
    return data


def _atomic_write_yaml(path: Path, data: dict[str, Any], *, schema_name: str | None = None) -> None:
    yaml = _yaml()
    if schema_name is not None:
        from pilot.tools._schema import validate
        validate(data, schema_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Identifier derivation
# ---------------------------------------------------------------------------


def _derive_model_id(plan_path: Path) -> str:
    """Lowercase, hyphen-stripped stem of the plan yaml (e.g. `deepseek_v2_lite_fp8`)."""
    stem = plan_path.stem
    stem = stem.lower().replace("-", "_").replace(".", "_")
    return re.sub(r"_+", "_", stem).strip("_") or "session"


def _derive_cluster_id(cluster_cfg_path: Path) -> str:
    cfg = _load_yaml(cluster_cfg_path)
    cid = cfg.get("cluster_id")
    if not isinstance(cid, str) or not cid:
        raise _SessionError(
            "CLUSTER",
            f"cluster_config {cluster_cfg_path} missing required `cluster_id`",
        )
    return cid


def _default_session_id(plan_path: Path, cluster_cfg_path: Path) -> str:
    model = _derive_model_id(plan_path)
    cluster = _derive_cluster_id(cluster_cfg_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model}__{cluster}__{ts}"


# ---------------------------------------------------------------------------
# Override parsing (shared shape with submit._parse_override)
# ---------------------------------------------------------------------------


def _parse_override(spec: str) -> tuple[str, Any]:
    if "=" not in spec:
        raise _SessionError("USAGE", f"--base-override expects key=value, got {spec!r}")
    k, _, raw = spec.partition("=")
    k = k.strip()
    raw = raw.strip()
    if not k:
        raise _SessionError("USAGE", f"--base-override has empty key: {spec!r}")
    low = raw.lower()
    if low in ("true", "false"):
        return k, low == "true"
    if low in ("null", "none", "~"):
        return k, None
    for cast in (int, float):
        try:
            return k, cast(raw)
        except ValueError:
            continue
    return k, raw


def _parse_constraint(spec: str) -> tuple[str, Any]:
    """Same coercion rules as base-override but used for target.constraints."""
    return _parse_override(spec)


# ---------------------------------------------------------------------------
# Build the three canonical artifacts
# ---------------------------------------------------------------------------


def _build_tuning_config(
    *,
    session_id: str,
    plan_ref: str,
    cluster_config_ref: str,
    primary: str,
    constraints: dict[str, Any],
    budget: dict[str, int],
    base_overrides: dict[str, Any],
    node: str | None,
    notes: str | None,
    trace_subdir: str = "trace",
) -> dict[str, Any]:
    log_dir_prefix = f"state/{session_id}"
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan": plan_ref,
        "cluster_config": cluster_config_ref,
        "log_dir_prefix": log_dir_prefix,
        "trace_subdir": trace_subdir,
        "target": {
            "primary": primary,
            "constraints": dict(constraints),
            "budget": dict(budget),
            "base_overrides": dict(base_overrides),
        },
        "stages": {
            "preflight":  {"dir": f"{trace_subdir}/preflight"},
            "projection": {"dir": f"{trace_subdir}/projection"},
            "smoke": {
                "dir": f"{trace_subdir}/smoke",
                "iters": budget["smoke_iters"],
                "timeout_s": budget["timeout_s"],
            },
            "baseline": {
                "dir": f"{trace_subdir}/baseline",
                "iters": budget["train_iters"],
                "timeout_s": budget["timeout_s"],
            },
            "optimize_loop": {
                "dir_pattern": f"{trace_subdir}/t{{trial_id}}",
                "iters": budget["train_iters"],
                "timeout_s": budget["timeout_s"],
            },
            "report":      {"dir": f"{trace_subdir}/report"},
            "env_sweep":   {"dir": f"{trace_subdir}/env_sweep"},
            "correctness": {"dir": f"{trace_subdir}/correctness"},
        },
        "related": {
            "target_vector_ref": "target_vector.yaml",
            "tuning_state_ref": "tuning_state.yaml",
        },
    }
    if node:
        payload["node"] = node
    if notes:
        payload["notes"] = notes
    return payload


def _build_target_vector(
    *,
    session_id: str,
    plan_ref: str,
    cluster_config_ref: str,
    primary: str,
    constraints: dict[str, Any],
    budget: dict[str, int],
    base_overrides: dict[str, Any],
    node: str | None,
    notes: str | None,
) -> dict[str, Any]:
    """Legacy mirror — kept so tools that read target_vector.yaml directly keep working."""
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "session_id": session_id,
        "plan": plan_ref,
        "cluster_config": cluster_config_ref,
        "primary": primary,
        "constraints": dict(constraints),
        "budget": dict(budget),
        "base_overrides": dict(base_overrides),
    }
    if node:
        payload["node"] = node
    if notes:
        payload["notes"] = notes
    return payload


def _build_initial_tuning_state(
    *,
    session_id: str,
    base_overrides: dict[str, Any],
    log_dir_prefix: str,
) -> dict[str, Any]:
    pairs = ", ".join(f"{k}={v}" for k, v in base_overrides.items())
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "current_stage": "PREFLIGHT",
        "round_id": 0,
        "champion_id": None,
        "budget_used": {"wallclock_s": 0, "gpu_h": 0.0},
        "stage_history": [],
        "artifacts": {},
        "flags": {
            "base_overrides_applied": pairs or "(none)",
            "log_dir_prefix": log_dir_prefix,
        },
        "last_decision_summary": "session bootstrapped via pilot session init",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init(
    *,
    plan: str | Path,
    cluster_config: str | Path | None = None,
    session_id: str | None = None,
    primary: str = "median_tflops",
    constraints: dict[str, Any] | None = None,
    rounds: int = 4,
    candidates_per_round: int = 3,
    smoke_iters: int = 10,
    train_iters: int = 20,
    timeout_s: int = 900,
    base_overrides: dict[str, Any] | None = None,
    trace_subdir: str = "trace",
    node: str | None = None,
    notes: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Bootstrap ``state/<session_id>/`` with tuning.yaml + target_vector.yaml + tuning_state.yaml.

    Idempotency: by default refuses to write into a non-empty session directory
    (``force=False``). Pass ``force=True`` to overwrite.

    Returns a JSON-serialisable summary that the CLI prints to stdout.
    """
    plan_path = _resolve_repo_path(plan)
    if not plan_path.exists():
        raise _SessionError("USAGE", f"--plan file not found: {plan_path}")

    cluster_cfg_in = cluster_config or "pilot/cluster.yaml"
    cluster_cfg_path = _resolve_repo_path(cluster_cfg_in)
    if not cluster_cfg_path.exists():
        raise _SessionError("USAGE", f"--cluster-config file not found: {cluster_cfg_path}")

    sid = session_id or _default_session_id(plan_path, cluster_cfg_path)
    if not _SESSION_ID_RE.match(sid):
        raise _SessionError("USAGE", f"invalid --session-id: {sid!r}")

    session_dir = _PILOT_ROOT / "state" / sid
    if session_dir.exists() and any(session_dir.iterdir()) and not force:
        raise _SessionError(
            "USAGE",
            f"session directory {session_dir} already exists and is non-empty. "
            "Pass --force to overwrite or pick a different --session-id.",
        )
    session_dir.mkdir(parents=True, exist_ok=True)

    constraints = constraints or {"finite_loss": True, "max_iter_time_regression_pct": 5.0}
    base_overrides = base_overrides or {}
    budget = {
        "rounds": int(rounds),
        "candidates_per_round": int(candidates_per_round),
        "smoke_iters": int(smoke_iters),
        "train_iters": int(train_iters),
        "timeout_s": int(timeout_s),
    }

    # Path refs are stored relative to the repo root so they round-trip whether
    # the caller is inside the container, on the controller, or anywhere else
    # that mounts the same shared filesystem at /shared/.../Primus.
    plan_ref = str(_relative_to_repo(plan_path))
    cluster_config_ref = str(_relative_to_repo(cluster_cfg_path))

    tuning_config = _build_tuning_config(
        session_id=sid,
        plan_ref=plan_ref,
        cluster_config_ref=cluster_config_ref,
        primary=primary,
        constraints=constraints,
        budget=budget,
        base_overrides=base_overrides,
        node=node,
        notes=notes,
        trace_subdir=trace_subdir,
    )
    target_vector = _build_target_vector(
        session_id=sid,
        plan_ref=plan_ref,
        cluster_config_ref=cluster_config_ref,
        primary=primary,
        constraints=constraints,
        budget=budget,
        base_overrides=base_overrides,
        node=node,
        notes=notes,
    )
    tuning_state = _build_initial_tuning_state(
        session_id=sid,
        base_overrides=base_overrides,
        log_dir_prefix=tuning_config["log_dir_prefix"],
    )

    tuning_yaml = session_dir / "tuning.yaml"
    target_vector_yaml = session_dir / "target_vector.yaml"
    tuning_state_yaml = session_dir / "tuning_state.yaml"

    # tuning.yaml is validated against the new schema; the other two have
    # permissive stub schemas already (target_vector / tuning_state) so we
    # skip schema validation for them here to avoid drag during bring-up.
    _atomic_write_yaml(tuning_yaml, tuning_config, schema_name="tuning_config")
    _atomic_write_yaml(target_vector_yaml, target_vector)
    _atomic_write_yaml(tuning_state_yaml, tuning_state)

    # Pre-create the trace tree so downstream tools can land their per-stage
    # logs without a separate mkdir dance.
    trace_root = session_dir / trace_subdir
    for stage_dir_rel in _trace_subpaths(tuning_config):
        (session_dir / stage_dir_rel).mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    # Seed r0 checkpoint via the canonical state.checkpoint path.
    from pilot.tools import state as _state
    checkpoint_path = _state.checkpoint(tuning_state, root=str(session_dir))

    return {
        "stage": "SESSION",
        "status": "success",
        "session_id": sid,
        "session_dir": str(session_dir),
        "artifacts": {
            "tuning_config":  str(tuning_yaml),
            "target_vector":  str(target_vector_yaml),
            "tuning_state":   str(tuning_state_yaml),
            "r0_checkpoint":  checkpoint_path,
        },
        "log_dir_prefix": tuning_config["log_dir_prefix"],
        "trace_subdir":   trace_subdir,
        "stage_trace_dirs": {
            name: str(session_dir / block["dir"])
            for name, block in tuning_config["stages"].items()
            if isinstance(block, dict) and "dir" in block
        },
        "optimize_loop_dir_pattern": tuning_config["stages"]["optimize_loop"]["dir_pattern"],
    }


def _trace_subpaths(tuning_config: dict[str, Any]) -> list[str]:
    """Trace dirs that have a fixed name (excludes optimize_loop's per-trial pattern)."""
    out: list[str] = []
    for block in tuning_config.get("stages", {}).values():
        if isinstance(block, dict) and "dir" in block:
            out.append(block["dir"])
    return out


def _relative_to_repo(p: Path) -> Path:
    try:
        return p.resolve().relative_to(_PRIMUS_ROOT)
    except ValueError:
        return p.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK = 0
_EXIT_STAGE_FAILED = 1
_EXIT_USAGE = 2


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "SESSION",
        "status": "failed",
        "failure": {"kind": kind, "message": message, "escalate_to_orchestrator": True},
    }


def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="pilot.tools.session",
        description="Bootstrap a Pilot tuning session (creates state/<session_id>/{tuning,target_vector,tuning_state}.yaml + r0 checkpoint).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a new session directory and seed bootstrap artifacts.")
    p_init.add_argument("--plan", required=True, help="Path to the base Primus exp YAML.")
    p_init.add_argument("--cluster-config", default=None,
                        help="Path to cluster.yaml. Defaults to pilot/cluster.yaml.")
    p_init.add_argument("--session-id", default=None,
                        help="Override the auto-generated session_id (default: <model>__<cluster>__<utc-timestamp>).")
    p_init.add_argument("--primary", default="median_tflops",
                        help="Primary objective metric (default: median_tflops).")
    p_init.add_argument("--constraint", action="append", default=[],
                        help="Hard constraint key=value (repeatable). Defaults: finite_loss=true, max_iter_time_regression_pct=5.0.")
    p_init.add_argument("--rounds", type=int, default=4)
    p_init.add_argument("--candidates-per-round", type=int, default=3)
    p_init.add_argument("--smoke-iters", type=int, default=10)
    p_init.add_argument("--train-iters", type=int, default=20)
    p_init.add_argument("--timeout-s", type=int, default=900)
    p_init.add_argument("--base-override", action="append", default=[],
                        help="Plan-level override key=value applied to every stage (repeatable). Example: --base-override micro_batch_size=1 --base-override global_batch_size=8.")
    p_init.add_argument("--trace-subdir", default="trace",
                        help="Sub-directory under log_dir_prefix for per-stage trace folders (default: trace).")
    p_init.add_argument("--node", default=None, help="Informational hint (e.g. mi355-gpu-26).")
    p_init.add_argument("--notes", default=None)
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite a non-empty existing session directory.")

    args = p.parse_args()
    try:
        if args.cmd == "init":
            constraints: dict[str, Any] = {}
            if args.constraint:
                for spec in args.constraint:
                    k, v = _parse_constraint(spec)
                    constraints[k] = v
            else:
                constraints = {"finite_loss": True, "max_iter_time_regression_pct": 5.0}

            base_overrides: dict[str, Any] = {}
            for spec in args.base_override:
                k, v = _parse_override(spec)
                base_overrides[k] = v

            result = init(
                plan=args.plan,
                cluster_config=args.cluster_config,
                session_id=args.session_id,
                primary=args.primary,
                constraints=constraints,
                rounds=args.rounds,
                candidates_per_round=args.candidates_per_round,
                smoke_iters=args.smoke_iters,
                train_iters=args.train_iters,
                timeout_s=args.timeout_s,
                base_overrides=base_overrides,
                trace_subdir=args.trace_subdir,
                node=args.node,
                notes=args.notes,
                force=args.force,
            )
            _emit(result)
            return _EXIT_OK
    except _SessionError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return _EXIT_USAGE if exc.kind == "USAGE" else _EXIT_STAGE_FAILED
    except Exception as exc:
        from pilot.tools._schema import SchemaValidationError
        if isinstance(exc, SchemaValidationError):
            _emit(_failure("TOOL_ERROR", f"schema validation failed: {exc}"))
            return _EXIT_STAGE_FAILED
        raise

    return _EXIT_USAGE


if __name__ == "__main__":
    sys.exit(_cli())
