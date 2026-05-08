"""constraint tools: safety checks consumed by Re-Plan / EnvSweep / Execute."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


_PILOT_ROOT: Path = Path(__file__).resolve().parent.parent
_ENV_DEFAULT_RE = re.compile(r"^\$\{[^:}]+:(?P<default>[^}]+)\}$")


class _ConstraintError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise _ConstraintError("DEP_MISSING", f"PyYAML required: {exc}") from exc
    return yaml


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else _PILOT_ROOT / p


def _load_mapping(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        raise _ConstraintError("USAGE", f"file not found: {p}")
    data = _yaml().safe_load(p.read_text())
    if not isinstance(data, dict):
        raise _ConstraintError("USAGE", f"{p} must contain a YAML/JSON mapping")
    return data


def _extract_overrides(plan: dict[str, Any]) -> dict[str, Any]:
    if "modules" in plan:
        return (
            plan.get("modules", {})
            .get("pre_trainer", {})
            .get("overrides", {})
            or {}
        )
    if "overrides" in plan and isinstance(plan["overrides"], dict):
        return plan["overrides"]
    return plan


def _env_default(value: Any) -> Any:
    if isinstance(value, str):
        m = _ENV_DEFAULT_RE.match(value.strip())
        if m:
            return m.group("default")
    return value


def _to_int(value: Any, default: int | None = None) -> int | None:
    value = _env_default(value)
    if value in (None, "null", "None", ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    value = _env_default(value)
    if value in (None, "null", "None", ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cluster_world(cluster: dict[str, Any]) -> tuple[int, int, int]:
    single = cluster.get("single") or {}
    launch = cluster.get("launch") or {}
    nnodes = _to_int(cluster.get("nnodes") or launch.get("nnodes"), 1) or 1
    gpus = (
        _to_int(cluster.get("gpus_per_node"))
        or _to_int(cluster.get("visible_gpus"))
        or _to_int(single.get("max_local_gpus"))
        or _to_int(launch.get("gpus_per_node"))
        or 1
    )
    return max(1, nnodes), max(1, gpus), max(1, nnodes * gpus)


def _parallelism(overrides: dict[str, Any]) -> dict[str, int]:
    return {
        "tp": _to_int(overrides.get("tensor_model_parallel_size"), 1) or 1,
        "pp": _to_int(overrides.get("pipeline_model_parallel_size"), 1) or 1,
        "ep": _to_int(overrides.get("expert_model_parallel_size"), 1) or 1,
        "cp": _to_int(overrides.get("context_parallel_size"), 1) or 1,
    }


def check(plan: dict, cluster: dict) -> dict:
    """Static config validity for a single-node Primus/Megatron plan."""
    overrides = _extract_overrides(plan)
    nnodes, gpus_per_node, world = _cluster_world(cluster)
    par = _parallelism(overrides)
    violations: list[str] = []
    warnings: list[str] = []

    for name, value in par.items():
        if value < 1:
            violations.append(f"{name} must be >= 1, got {value}")
    model_parallel = par["tp"] * par["pp"] * par["ep"] * par["cp"]
    if model_parallel > world:
        violations.append(
            f"tp*pp*ep*cp={model_parallel} exceeds world_size={world} "
            f"(nnodes={nnodes}, gpus_per_node={gpus_per_node})"
        )
    elif world % model_parallel != 0:
        violations.append(
            f"world_size={world} must be divisible by tp*pp*ep*cp={model_parallel}"
        )

    mbs = _to_int(overrides.get("micro_batch_size"))
    gbs = _to_int(overrides.get("global_batch_size"))
    train_iters = _to_int(overrides.get("train_iters"))
    seq_length = _to_int(overrides.get("seq_length"))
    for key, value in (
        ("micro_batch_size", mbs),
        ("global_batch_size", gbs),
        ("train_iters", train_iters),
        ("seq_length", seq_length),
    ):
        if value is not None and value <= 0:
            violations.append(f"{key} must be > 0, got {value}")

    if mbs and gbs and model_parallel <= world:
        dp = max(1, world // max(1, model_parallel))
        unit = mbs * dp
        if unit > 0 and gbs % unit != 0:
            violations.append(
                f"global_batch_size={gbs} must be divisible by "
                f"micro_batch_size*dp={unit} (dp={dp})"
            )

    recompute_granularity = overrides.get("recompute_granularity")
    if recompute_granularity not in (None, "full", "selective"):
        violations.append(
            "recompute_granularity must be one of null/full/selective, "
            f"got {recompute_granularity!r}"
        )
    recompute_method = overrides.get("recompute_method")
    if recompute_method not in (None, "uniform", "block"):
        violations.append(
            f"recompute_method must be one of null/uniform/block, got {recompute_method!r}"
        )

    if par["pp"] > 1 and train_iters is not None and train_iters < par["pp"]:
        warnings.append("train_iters is smaller than pipeline stages; timing may be noisy")

    return {
        "valid": not violations,
        "violations": violations,
        "warnings": warnings,
        "derived": {
            "nnodes": nnodes,
            "gpus_per_node": gpus_per_node,
            "world_size": world,
            "model_parallel": model_parallel,
            "data_parallel": world // model_parallel if model_parallel and world % model_parallel == 0 else None,
        },
    }


def check_env(env_diff: dict, baseline: dict) -> dict:
    """Validate env combination against a small single-node incompatibility matrix."""
    violations: list[str] = []
    warnings: list[str] = []
    if not isinstance(env_diff, dict):
        raise _ConstraintError("USAGE", "env_diff must be a mapping")

    for key, value in env_diff.items():
        if not isinstance(key, str):
            violations.append(f"env key must be a string, got {key!r}")
        if value is None:
            warnings.append(f"{key} is set to null; unset it instead if inheritance is desired")

    if "HIP_VISIBLE_DEVICES" in env_diff and "CUDA_VISIBLE_DEVICES" in env_diff:
        violations.append("set only one of HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES")
    if env_diff.get("NCCL_IB_DISABLE") in (1, "1", True) and env_diff.get("NCCL_NET_GDR_LEVEL"):
        warnings.append("NCCL_NET_GDR_LEVEL is ignored when NCCL_IB_DISABLE=1")
    if env_diff.get("RCCL_ENABLE_COLLTRACE") in (1, "1", True):
        warnings.append("RCCL_ENABLE_COLLTRACE can perturb timing; use only for diagnostics")

    inherited = set((baseline or {}).keys()) & set(env_diff.keys())
    if inherited:
        warnings.append(f"overrides baseline env keys: {sorted(inherited)}")

    return {"valid": not violations, "violations": violations, "warnings": warnings}


def estimate_mem(plan: dict) -> dict:
    """Conservative single-node memory estimate.

    The estimate is intentionally coarse until model metadata is standardized in
    Plan. It is useful for comparing candidates, not for admission control.
    """
    overrides = _extract_overrides(plan)
    par = _parallelism(overrides)
    tp = max(1, par["tp"])
    pp = max(1, par["pp"])

    hidden = _to_int(overrides.get("hidden_size"))
    layers = _to_int(overrides.get("num_layers"))
    ffn = _to_int(overrides.get("ffn_hidden_size"), None)
    vocab = _to_int(overrides.get("vocab_size"), 128_256) or 128_256
    seq = _to_int(overrides.get("seq_length"), 4096) or 4096
    mbs = _to_int(overrides.get("micro_batch_size"), 1) or 1
    bytes_per_param = 2.0

    confidence = 0.35
    if hidden and layers:
        ffn = ffn or hidden * 4
        # Transformer rough count: attention/proj ~4h^2 + MLP ~2h*ffn per layer.
        params = layers * (4 * hidden * hidden + 2 * hidden * ffn) + vocab * hidden
        param_gb = params * bytes_per_param / 1e9 / tp / pp
        confidence = 0.6
    else:
        # Fallback heuristic tied to activation size; deliberately conservative.
        hidden = hidden or 4096
        layers = layers or 32
        param_gb = 8.0 / tp / pp

    grad_gb = param_gb
    optim_gb = param_gb * 4.0
    act_factor = 2.0 if overrides.get("recompute_granularity") else 6.0
    act_gb = mbs * seq * hidden * max(1, layers // pp) * bytes_per_param * act_factor / 1e9
    buffer_gb = max(2.0, 0.12 * (param_gb + grad_gb + optim_gb + act_gb))
    total = param_gb + grad_gb + optim_gb + act_gb + buffer_gb
    return {
        "mem_gb": round(total, 2),
        "components": {
            "param": round(param_gb, 2),
            "grad": round(grad_gb, 2),
            "optim": round(optim_gb, 2),
            "act": round(act_gb, 2),
            "buffer": round(buffer_gb, 2),
        },
        "confidence": confidence,
        "notes": ["heuristic estimate; validate with a smoke run before promotion"],
    }


def diagnose_failure(snapshot_or_error: dict) -> dict:
    """Attribute failures from RunSnapshot symptoms or a generic error dict."""
    symptoms = snapshot_or_error.get("symptoms") or {}
    status = snapshot_or_error.get("status")
    failure = snapshot_or_error.get("failure") or {}
    kind = "UNKNOWN"
    transition = "OPTIMIZE_LOOP.DIAGNOSE"
    message = ""

    if symptoms.get("oom_detected"):
        kind = "OOM"
        transition = "OPTIMIZE_LOOP.REPLAN"
        message = "OOM detected; reduce micro_batch_size or enable recompute"
    elif symptoms.get("hang_suspected") or symptoms.get("nccl_error"):
        kind = "HANG"
        transition = "PREFLIGHT"
        message = "hang or collective error detected"
    elif symptoms.get("cuda_error"):
        kind = "TOOL_ERROR"
        transition = "ABORT"
        message = "CUDA/HIP runtime error detected"
    elif symptoms.get("python_error"):
        kind = "TOOL_ERROR"
        transition = "ABORT"
        message = "Python exception detected in training log"
    elif symptoms.get("loss_nan_or_inf") or snapshot_or_error.get("metrics", {}).get("loss_finite") is False:
        kind = "NUMERICAL"
        transition = "OPTIMIZE_LOOP.REPLAN"
        message = "loss became NaN or Inf"
    elif status in ("failed", "killed", "hung", "unknown"):
        kind = "UNKNOWN"
        transition = "OPTIMIZE_LOOP.DIAGNOSE"
        message = f"run status={status}"
    elif failure:
        kind = str(failure.get("kind") or "UNKNOWN")
        message = str(failure.get("message") or "")

    return {
        "kind": kind,
        "message": message or "no hard failure symptoms detected",
        "suggested_transition": transition,
        "escalate_to_orchestrator": transition in ("ABORT", "PREFLIGHT"),
        "evidence": symptoms.get("evidence", []),
    }


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "CONSTRAINT",
        "status": "failed",
        "failure": {"kind": kind, "message": message, "escalate_to_orchestrator": True},
    }


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.constraint")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check")
    p_check.add_argument("--plan", required=True)
    p_check.add_argument("--cluster", required=True)

    p_env = sub.add_parser("check_env")
    p_env.add_argument("--env", required=True)
    p_env.add_argument("--baseline", default=None)

    p_mem = sub.add_parser("estimate_mem")
    p_mem.add_argument("--plan", required=True)

    p_diag = sub.add_parser("diagnose_failure")
    p_diag.add_argument("--snapshot", required=True)

    args = p.parse_args()
    try:
        if args.cmd == "check":
            _emit(check(_load_mapping(args.plan), _load_mapping(args.cluster)))
            return 0
        if args.cmd == "check_env":
            baseline = _load_mapping(args.baseline) if args.baseline else {}
            _emit(check_env(_load_mapping(args.env), baseline))
            return 0
        if args.cmd == "estimate_mem":
            _emit(estimate_mem(_load_mapping(args.plan)))
            return 0
        if args.cmd == "diagnose_failure":
            _emit(diagnose_failure(_load_mapping(args.snapshot)))
            return 0
    except _ConstraintError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return 2

    return 2


if __name__ == "__main__":
    sys.exit(_cli())
