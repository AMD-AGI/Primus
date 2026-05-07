"""pilot.tools.submit — launch / cancel / status of training jobs.

Wraps Primus's `examples/run_pretrain.sh` launcher. Does **not** introduce a
new scheduler; it follows the universal Pilot input contract by reading
`cluster.yaml` (single | slurm) and translating it into the `NNODES /
NODE_RANK / GPUS_PER_NODE / MASTER_ADDR / MASTER_PORT / EXP / TRAIN_LOG`
environment variables that `run_pretrain.sh` already understands.

CLI surface
-----------
::

    submit run     --cluster-config cluster.yaml --plan <yaml>
                   [--override key=value ...]   [--run-id <id>]
                   [--log-dir state/runs]       [--liveness-s 5]
                   [--foreground]
    submit cancel  --run-id <id> [--log-dir state/runs]
    submit status  --run-id <id> [--log-dir state/runs]

Outputs (per submission, under ``state/runs/<run_id>/``)
--------------------------------------------------------
* ``handle.yaml``         — RunHandle (schemas/run_handle.schema.json)
* ``plan.effective.yaml`` — base plan with Pilot's ``--override`` patches applied
* ``train.log``           — combined stdout+stderr captured from the launcher
* ``exit_code.txt``       — written by the wrapper once the subprocess exits
* ``submit.json``         — initial SubagentResult (stdout of ``submit run``)

Pre-existing skeleton (`run`, `cancel`, `_cli`) is replaced. The CLI now
performs the same fast-fail cluster.yaml check as `pilot.tools.preflight` so
mis-configured environments are surfaced as ``failure.kind=CLUSTER`` (exit 4)
without ever touching the launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pilot.tools._cluster_config import (
    ClusterConfig,
    ClusterConfigError,
    LaunchPlan,
    cluster_config_failure,
    preflight_check,
)


# ---------------------------------------------------------------------------
# Path anchoring (mirrors preflight.py)
# ---------------------------------------------------------------------------

_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent  # tools/ -> pilot/
_PRIMUS_ROOT: Path = _PILOT_ROOT.parent                      # pilot/ -> repo root
_RUN_PRETRAIN_SH: Path = _PRIMUS_ROOT / "examples" / "run_pretrain.sh"


def _resolve_pilot_path(p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else _PILOT_ROOT / pp


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class _SubmitError(Exception):
    """Measurement-layer / launch-layer failures (not cluster-contract issues)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Run-id + plan composition
# ---------------------------------------------------------------------------

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _generate_run_id(plan_path: Path) -> str:
    stem = plan_path.stem.replace(".", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stem}__{ts}"


def _parse_override(spec: str) -> tuple[str, Any]:
    """Parse ``key=value`` override; coerce value to bool/int/float when obvious."""
    if "=" not in spec:
        raise _SubmitError("USAGE", f"--override expects key=value, got {spec!r}")
    k, _, raw = spec.partition("=")
    k = k.strip()
    raw = raw.strip()
    if not k:
        raise _SubmitError("USAGE", f"--override has empty key: {spec!r}")
    low = raw.lower()
    if low in ("true", "false"):
        return k, (low == "true")
    if low in ("null", "none", "~"):
        return k, None
    # int
    try:
        return k, int(raw)
    except ValueError:
        pass
    # float
    try:
        return k, float(raw)
    except ValueError:
        pass
    return k, raw


def _apply_overrides(plan: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Patch ``modules.pre_trainer.overrides.<k>`` in-place for every (k, v).

    The Primus exp.yaml convention is to put per-run hyperparams under
    ``modules.pre_trainer.overrides``. We leave the rest of the file alone.
    """
    if not overrides:
        return plan
    pre_trainer = (
        plan.setdefault("modules", {}).setdefault("pre_trainer", {})
    )
    target = pre_trainer.setdefault("overrides", {})
    for k, v in overrides.items():
        target[k] = v
    return plan


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _SubmitError("DEP_MISSING", f"PyYAML required for submit: {exc}")
    if not path.exists():
        raise _SubmitError("USAGE", f"plan file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise _SubmitError("USAGE", f"plan {path} is not a YAML mapping")
    return data


def _atomic_write_yaml(
    path: Path,
    data: dict[str, Any],
    *,
    schema_name: str | None = None,
) -> None:
    """Validate (when ``schema_name`` is given) → write tmp → rename.

    The validation step ensures broken artifacts never escape the writer.
    Schema errors raise :class:`pilot.tools._schema.SchemaValidationError`,
    surfaced by the CLI as ``failure.kind=TOOL_ERROR``.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise _SubmitError("DEP_MISSING", f"PyYAML required for submit: {exc}")
    if schema_name is not None:
        from pilot.tools._schema import validate
        validate(data, schema_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    cfg: ClusterConfig,
    plan: LaunchPlan,
    *,
    plan_path: str | Path,
    overrides: dict[str, Any] | None = None,
    run_id: str | None = None,
    log_dir: str | Path = "state/runs",
    liveness_s: int = 5,
    foreground: bool = False,
    foreground_timeout_s: int | None = None,
) -> dict[str, Any]:
    """Launch a training job. Returns a SubagentResult dict.

    In detached mode (default), this returns once the subprocess has been
    spawned and survived ``liveness_s`` seconds without crashing. In
    ``foreground=True`` mode this blocks until the subprocess exits (or
    ``foreground_timeout_s`` elapses, in which case the subprocess is killed).
    """
    started = time.time()
    overrides = overrides or {}
    base_path = Path(plan_path).resolve()
    if not base_path.exists():
        raise _SubmitError("USAGE", f"--plan file not found: {base_path}")

    if not _RUN_PRETRAIN_SH.exists():
        raise _SubmitError(
            "ENV_MISSING",
            f"Primus launcher not found at {_RUN_PRETRAIN_SH}. "
            "Pilot expects to live alongside Primus (examples/run_pretrain.sh).",
        )

    rid = run_id or _generate_run_id(base_path)
    if not _RUN_ID_RE.match(rid):
        raise _SubmitError("USAGE", f"invalid --run-id: {rid!r}")

    run_dir = _resolve_pilot_path(log_dir) / rid
    if run_dir.exists() and any(run_dir.iterdir()):
        raise _SubmitError(
            "USAGE",
            f"run_id collision: {run_dir} already exists and is non-empty. "
            "Pass an explicit --run-id or remove the directory first.",
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Effective plan = base + overrides
    base_plan = _load_yaml(base_path)
    effective_plan = _apply_overrides(base_plan, overrides)
    effective_path = run_dir / "plan.effective.yaml"
    _atomic_write_yaml(effective_path, effective_plan)

    # Compose env
    env_diff = _compose_env(cfg, plan, effective_path, run_dir)

    # Build command
    if cfg.mode == "single":
        cmd_argv = ["bash", str(_RUN_PRETRAIN_SH)]
    elif cfg.mode == "slurm":
        cmd_argv = _build_slurm_argv(plan)
    else:
        raise _SubmitError("USAGE", f"unsupported mode: {cfg.mode!r}")

    cmd_str = " ".join(shlex.quote(a) for a in cmd_argv)
    log_path = run_dir / "train.log"
    sentinel = run_dir / "exit_code.txt"

    # Wrap the inner command so we always get exit_code.txt regardless of how
    # the launcher exits. The wrapper preserves stdout/stderr semantics by
    # `exec`-ing the inner command in a subshell.
    wrapped = (
        f"({cmd_str}); rc=$?; echo $rc > {shlex.quote(str(sentinel))}; exit $rc"
    )

    # Compose final env: inherit current, then overlay env_diff
    spawn_env = os.environ.copy()
    for k, v in env_diff.items():
        spawn_env[k] = v

    # Persist initial handle (status=launching)
    handle: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": rid,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cluster_config_ref": cfg.source_path,
        "plan": {
            "base_ref": str(base_path),
            "effective_ref": str(effective_path),
            "overrides": overrides,
        },
        "launch": {
            "mode": cfg.mode,
            "nnodes": plan.nnodes,
            "gpus_per_node": _detect_gpus_per_node(),
            "master_addr": env_diff["MASTER_ADDR"],
            "master_port": int(env_diff["MASTER_PORT"]),
            "cmd": cmd_str,
            "env_diff": {k: str(v) for k, v in env_diff.items()},
            "pid": None,
            "slurm_job_id": plan.slurm_job_id,
            "slurm_step_id": None,
        },
        "log": {
            "stdout": str(log_path),
            "stderr": None,
            "exit_code_sentinel": str(sentinel),
        },
        "status": "launching",
        "exit_code": None,
        "wallclock_s": None,
    }
    handle_path = run_dir / "handle.yaml"
    _atomic_write_yaml(handle_path, handle, schema_name="run_handle")

    # Spawn
    log_fp = log_path.open("ab", buffering=0)  # binary, line-unbuffered enough for tee-style
    try:
        # cwd = PRIMUS_ROOT so any relative paths inside run_pretrain.sh resolve.
        proc = subprocess.Popen(
            ["bash", "-c", wrapped],
            cwd=str(_PRIMUS_ROOT),
            env=spawn_env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from parent's process group
        )
    except OSError as exc:
        log_fp.close()
        raise _SubmitError("LAUNCH", f"failed to spawn launcher: {exc}")

    handle["launch"]["pid"] = proc.pid
    _atomic_write_yaml(handle_path, handle, schema_name="run_handle")

    # Foreground mode: block until completion
    if foreground:
        try:
            rc = proc.wait(timeout=foreground_timeout_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
            handle["status"] = "killed"
        else:
            handle["status"] = "completed" if rc == 0 else "failed"
        handle["exit_code"] = rc
        handle["wallclock_s"] = round(time.time() - started, 2)
        _atomic_write_yaml(handle_path, handle, schema_name="run_handle")
        log_fp.close()
        return _build_subagent_result(handle, started)

    # Detached mode: liveness check + return
    deadline = time.time() + max(0, liveness_s)
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    rc = proc.poll()
    if rc is None:
        handle["status"] = "running"
    else:
        handle["status"] = "completed" if rc == 0 else "failed"
        handle["exit_code"] = rc
        handle["wallclock_s"] = round(time.time() - started, 2)
    _atomic_write_yaml(handle_path, handle, schema_name="run_handle")

    return _build_subagent_result(handle, started)


def cancel(
    run_id: str,
    *,
    log_dir: str | Path = "state/runs",
    timeout_s: int = 10,
) -> dict[str, Any]:
    """Send SIGTERM (then SIGKILL after timeout) to the launched process group."""
    handle_path, handle = _load_handle(run_id, log_dir)
    pid = handle["launch"].get("pid")
    mode = handle["launch"]["mode"]
    rid = handle["run_id"]

    if handle["status"] in ("completed", "failed", "killed"):
        return {
            "run_id": rid,
            "status": handle["status"],
            "exit_code": handle.get("exit_code"),
            "message": f"already terminal ({handle['status']})",
        }

    if mode == "slurm":
        step_id = handle["launch"].get("slurm_step_id")
        if step_id:
            subprocess.run(
                ["scancel", "--signal=TERM", str(step_id)],
                check=False, capture_output=True, timeout=10,
            )
        elif pid:
            _terminate_pgid(pid, timeout_s)
    else:
        if not pid:
            raise _SubmitError("STATE", f"handle for {rid} has no pid")
        _terminate_pgid(pid, timeout_s)

    handle["status"] = "killed"
    handle["exit_code"] = handle.get("exit_code")
    if handle.get("wallclock_s") is None:
        try:
            t0 = datetime.fromisoformat(handle["created_at"])
            handle["wallclock_s"] = round(
                (datetime.now(timezone.utc) - t0).total_seconds(), 2,
            )
        except Exception:
            pass
    _atomic_write_yaml(handle_path, handle, schema_name="run_handle")
    return {
        "run_id": rid,
        "status": "killed",
        "exit_code": handle.get("exit_code"),
        "message": "SIGTERM/SIGKILL sent",
    }


def status(
    run_id: str,
    *,
    log_dir: str | Path = "state/runs",
) -> dict[str, Any]:
    """Refresh and return the run's status by checking pid liveness + sentinel."""
    handle_path, handle = _load_handle(run_id, log_dir)
    rid = handle["run_id"]

    # Already terminal: just return.
    if handle["status"] in ("completed", "failed", "killed"):
        return _status_view(handle)

    pid = handle["launch"].get("pid")
    sentinel = handle["log"].get("exit_code_sentinel")
    sentinel_path = Path(sentinel) if sentinel else None

    # Sentinel takes priority: if the wrapper wrote it, the subprocess
    # finished cleanly (success or natural failure) and we should record
    # the exit code regardless of whether the orphaned bash is still in
    # the proc table.
    rc: int | None = None
    if sentinel_path and sentinel_path.exists():
        try:
            rc = int(sentinel_path.read_text().strip())
        except Exception:
            rc = None

    if rc is not None:
        handle["status"] = "completed" if rc == 0 else "failed"
        handle["exit_code"] = rc
    else:
        # No sentinel → check pid liveness, but rule out zombie processes
        # which `os.kill(pid, 0)` reports as alive.
        alive = _is_alive_not_zombie(pid)
        handle["status"] = "running" if alive else "unknown"

    if handle["status"] != "running" and handle.get("wallclock_s") is None:
        try:
            t0 = datetime.fromisoformat(handle["created_at"])
            handle["wallclock_s"] = round(
                (datetime.now(timezone.utc) - t0).total_seconds(), 2,
            )
        except Exception:
            pass

    _atomic_write_yaml(handle_path, handle, schema_name="run_handle")
    return _status_view(handle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_gpus_per_node() -> int:
    """Best-effort GPU count for env composition. Honors HIP_VISIBLE_DEVICES /
    CUDA_VISIBLE_DEVICES; falls back to rocm-smi / nvidia-smi when set, then 8.
    """
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        v = os.environ.get(var)
        if v:
            return len([t for t in v.split(",") if t.strip()])
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 8


def _free_master_port() -> int:
    """Pick a free TCP port in the ephemeral range. Avoids the rdzv default 29400."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _compose_env(
    cfg: ClusterConfig,
    plan: LaunchPlan,
    effective_plan: Path,
    run_dir: Path,
) -> dict[str, str]:
    """Compose the env diff that run_pretrain.sh expects."""
    gpus = _detect_gpus_per_node()
    if cfg.mode == "single":
        master_addr = "127.0.0.1"
        master_port = _free_master_port()
    else:  # slurm
        master_addr = plan.head_host
        # plan.rdzv_endpoint is "host:port"; reuse the port if present.
        if ":" in plan.rdzv_endpoint:
            master_port = int(plan.rdzv_endpoint.rsplit(":", 1)[1])
        else:
            master_port = plan.rdzv_port

    env_diff: dict[str, str] = {
        "EXP": str(effective_plan),
        "NNODES": str(plan.nnodes),
        "GPUS_PER_NODE": str(gpus),
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "TRAIN_LOG": str(run_dir / "train.log"),
        # NODE_RANK: single-mode is always 0; slurm-mode each task picks up
        # $SLURM_NODEID inside the wrapper (see _build_slurm_argv).
        "NODE_RANK": "0",
    }
    return env_diff


def _build_slurm_argv(plan: LaunchPlan) -> list[str]:
    """Build the srun command for slurm mode.

    Each task on each allocated node runs `run_pretrain.sh` with NODE_RANK
    derived from `$SLURM_NODEID`. We cannot put a literal `$SLURM_NODEID`
    into env_diff because Python composes that string at submit time, before
    srun has assigned node IDs. So we wrap the actual launcher in a tiny
    `bash -c` that re-exports NODE_RANK at runtime on each node.
    """
    if plan.slurm_job_id is None:
        raise _SubmitError("STATE", "slurm mode but plan.slurm_job_id is None")

    # Inner script: NODE_RANK comes from SLURM_NODEID at runtime.
    # Single-quote the whole thing so $SLURM_NODEID isn't expanded by Python.
    inner = (
        f"export NODE_RANK=${{SLURM_NODEID:-0}}; "
        f"exec bash {shlex.quote(str(_RUN_PRETRAIN_SH))}"
    )
    argv = [
        "srun",
        f"--jobid={plan.slurm_job_id}",
        f"--nodes={plan.nnodes}",
        "--ntasks-per-node=1",
        "--export=ALL",  # propagate every env var Pilot set
        "bash", "-c", inner,
    ]
    return argv


def _is_alive_not_zombie(pid: int | None) -> bool:
    """Return True iff the pid is in the proc table AND not a zombie.

    `os.kill(pid, 0)` reports zombies as alive; we explicitly read
    `/proc/<pid>/stat` to filter them out. Falls back to the kill(0) probe
    when /proc isn't available (non-Linux).
    """
    if not pid:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            data = proc_stat.read_text()
            # Format: pid (comm) state ...
            # comm can contain spaces/parens, so split after the trailing ')'.
            tail = data.rsplit(")", 1)[1].strip()
            state = tail.split(" ", 1)[0]
            return state not in ("Z", "X")
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_pgid(pid: int, timeout_s: int) -> None:
    """SIGTERM the process group, then SIGKILL after timeout."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + max(1, timeout_s)
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    # Still alive — escalate.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _load_handle(run_id: str, log_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    run_dir = _resolve_pilot_path(log_dir) / run_id
    handle_path = run_dir / "handle.yaml"
    if not handle_path.exists():
        raise _SubmitError("USAGE", f"no handle for run_id={run_id!r} at {handle_path}")
    return handle_path, _load_yaml(handle_path)


def _status_view(handle: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": handle["run_id"],
        "status": handle["status"],
        "exit_code": handle.get("exit_code"),
        "wallclock_s": handle.get("wallclock_s"),
        "log": handle["log"]["stdout"],
        "pid": handle["launch"].get("pid"),
        "slurm_step_id": handle["launch"].get("slurm_step_id"),
    }


def _build_subagent_result(handle: dict[str, Any], started: float) -> dict[str, Any]:
    elapsed = time.time() - started
    rid = handle["run_id"]
    st = handle["status"]
    nnodes = handle["launch"]["nnodes"]
    gpn = handle["launch"]["gpus_per_node"]
    plan_name = Path(handle["plan"]["base_ref"]).name

    failure = None
    if st in ("failed", "killed", "unknown"):
        failure = {
            "kind": "TRAIN_LAUNCH" if st == "failed" else "CANCELLED" if st == "killed" else "UNKNOWN",
            "message": (
                f"job exited with status={st}, "
                f"exit_code={handle.get('exit_code')}; "
                f"see {handle['log']['stdout']}"
            ),
            "escalate_to_orchestrator": st != "killed",
        }

    return {
        "stage": "SUBMIT",
        "status": st,
        "artifacts": [
            {"kind": "RunHandle", "ref": str(_handle_path_from_handle(handle))},
            {"kind": "TrainLog",  "ref": handle["log"]["stdout"]},
        ],
        "summary": {
            "headline": (
                f"submit {plan_name} → run_id={rid} "
                f"({handle['launch']['mode']}, {nnodes}x{gpn}gpu) status={st}"
            ),
            "key_metrics": {
                "run_id": rid,
                "nnodes": nnodes,
                "gpus_per_node": gpn,
                "pid": handle["launch"].get("pid"),
                "slurm_step_id": handle["launch"].get("slurm_step_id"),
                "exit_code": handle.get("exit_code"),
                "wallclock_s": handle.get("wallclock_s"),
            },
            "warnings": [],
        },
        "suggested_transition": {
            "to": "OBSERVE" if st == "running" else (
                "REPORT" if st == "completed" else "DIAGNOSE"
            ),
            "reason": (
                "job running; poll observe.snapshot" if st == "running"
                else f"submit terminal: {st}"
            ),
        },
        "cost": {
            "gpu_h": 0.0,  # submit itself doesn't burn gpu-h; track in observe
            "wallclock_s": round(elapsed, 2),
            "tool_calls": 1,
        },
        "failure": failure,
    }


def _handle_path_from_handle(handle: dict[str, Any]) -> Path:
    log = Path(handle["log"]["stdout"])
    return log.parent / "handle.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_EXIT_OK = 0
_EXIT_STAGE_FAILED = 1
_EXIT_USAGE = 2
_EXIT_TOOL_ERROR = 3
_EXIT_CLUSTER = 4


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "SUBMIT",
        "status": "failed",
        "failure": {
            "kind": kind,
            "message": message,
            "escalate_to_orchestrator": True,
        },
    }


def _add_cluster_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cluster-config",
        default=None,
        help="Path to cluster.yaml (fallback: $PRIMUS_PILOT_CLUSTER_CONFIG, then ./cluster.yaml).",
    )


def _cli() -> int:
    p = argparse.ArgumentParser(
        prog="pilot.tools.submit",
        description="Launch / cancel / status of training jobs (cluster.yaml driven).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run", help="Submit a training job (detached by default).")
    _add_cluster_config_arg(p_run)
    p_run.add_argument("--plan", required=True, help="Path to a Primus exp.yaml.")
    p_run.add_argument(
        "--override", action="append", default=[],
        help="key=value override for modules.pre_trainer.overrides (repeatable).",
    )
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument(
        "--log-dir", default="state/runs",
        help="Where to write state/runs/<run_id>/. Anchored at pilot package root if relative.",
    )
    p_run.add_argument("--liveness-s", type=int, default=5)
    p_run.add_argument(
        "--foreground", action="store_true",
        help="Block until the job exits (useful for SMOKE / debug). Default: detached.",
    )
    p_run.add_argument(
        "--foreground-timeout-s", type=int, default=None,
        help="Wallclock cap for --foreground; subprocess is killed on timeout.",
    )

    # cancel
    p_cancel = sub.add_parser("cancel", help="Terminate a running job by run_id.")
    p_cancel.add_argument("--run-id", required=True)
    p_cancel.add_argument("--log-dir", default="state/runs")
    p_cancel.add_argument("--timeout-s", type=int, default=10)

    # status
    p_status = sub.add_parser("status", help="Report a run's status (refreshes handle.yaml).")
    p_status.add_argument("--run-id", required=True)
    p_status.add_argument("--log-dir", default="state/runs")

    args = p.parse_args()

    try:
        if args.cmd == "run":
            cfg, plan = preflight_check(args.cluster_config)
            overrides: dict[str, Any] = {}
            for spec in args.override:
                k, v = _parse_override(spec)
                overrides[k] = v
            result = run(
                cfg, plan,
                plan_path=args.plan,
                overrides=overrides,
                run_id=args.run_id,
                log_dir=args.log_dir,
                liveness_s=args.liveness_s,
                foreground=args.foreground,
                foreground_timeout_s=args.foreground_timeout_s,
            )
            _emit(result)
            return _EXIT_OK if result["status"] in ("running", "completed") else _EXIT_STAGE_FAILED

        if args.cmd == "cancel":
            result = cancel(args.run_id, log_dir=args.log_dir, timeout_s=args.timeout_s)
            _emit(result)
            return _EXIT_OK

        if args.cmd == "status":
            result = status(args.run_id, log_dir=args.log_dir)
            _emit(result)
            return _EXIT_OK

    except ClusterConfigError as exc:
        _emit(cluster_config_failure(exc, stage="SUBMIT"))
        return _EXIT_CLUSTER
    except _SubmitError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return _EXIT_USAGE if exc.kind == "USAGE" else _EXIT_STAGE_FAILED
    except NotImplementedError as exc:
        _emit(_failure("TOOL_ERROR", f"not implemented: {exc}"))
        return _EXIT_TOOL_ERROR
    except Exception as exc:  # imported lazily so module always loads
        from pilot.tools._schema import SchemaValidationError
        if isinstance(exc, SchemaValidationError):
            _emit(_failure("TOOL_ERROR", f"schema validation failed: {exc}"))
            return _EXIT_TOOL_ERROR
        raise

    return _EXIT_USAGE


if __name__ == "__main__":
    sys.exit(_cli())
