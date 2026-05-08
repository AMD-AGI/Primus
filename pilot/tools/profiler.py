"""profiler tool: single-node profiling for Projection.

The first shipped profiler is a lightweight GPU-aware estimator. It does not
run a training job; it converts model/config shapes into comparable estimates
and records the visible device inventory for the Orchestrator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class _ProfilerError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise _ProfilerError("DEP_MISSING", f"PyYAML required for profiler: {exc}") from exc
    return yaml


def _load_mapping(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        raise _ProfilerError("USAGE", f"file not found: {p}")
    data = _yaml().safe_load(p.read_text())
    if not isinstance(data, dict):
        raise _ProfilerError("USAGE", f"{p} must contain a YAML/JSON mapping")
    return data


def _load_configs(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return [{}]
    data = _load_mapping(path)
    raw = data.get("configs", data.get("candidates", data))
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    if isinstance(raw, dict):
        return [raw]
    raise _ProfilerError("USAGE", f"{path} must contain configs as a mapping or list")


def _gpu_inventory() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise _ProfilerError("DEP_MISSING", f"torch is required on the training node: {exc}") from exc
    if not torch.cuda.is_available():
        raise _ProfilerError("CLUSTER", "no GPU visible to torch.cuda")
    devices = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        devices.append({
            "index": idx,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "major": props.major,
            "minor": props.minor,
        })
    return {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def _num(raw: Any, default: float) -> float:
    if isinstance(raw, str) and raw.startswith("${") and ":" in raw:
        raw = raw.rsplit(":", 1)[1].rstrip("}")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return value


def _estimate(model_spec: dict[str, Any], cfg: dict[str, Any], gpus: int) -> dict[str, Any]:
    hidden = _num(cfg.get("hidden_size", model_spec.get("hidden_size")), 4096)
    layers = _num(cfg.get("num_layers", model_spec.get("num_layers")), 32)
    seq = _num(cfg.get("seq_length", model_spec.get("seq_length")), 4096)
    mbs = _num(cfg.get("micro_batch_size", model_spec.get("micro_batch_size")), 1)
    tp = max(1.0, _num(cfg.get("tensor_model_parallel_size", model_spec.get("tensor_model_parallel_size")), 1))
    pp = max(1.0, _num(cfg.get("pipeline_model_parallel_size", model_spec.get("pipeline_model_parallel_size")), 1))
    recompute = bool(cfg.get("recompute_granularity") or model_spec.get("recompute_granularity"))

    # Rough dense transformer math. Good enough for candidate ordering.
    tokens = mbs * seq
    flops = 6.0 * layers * tokens * hidden * hidden
    effective_gpus = max(1.0, min(float(gpus), tp * pp))
    assumed_tflops_per_gpu = _num(model_spec.get("assumed_tflops_per_gpu"), 1000.0)
    t_comp_ms = flops / (effective_gpus * assumed_tflops_per_gpu * 1e12) * 1000.0
    act_gb = tokens * hidden * max(1.0, layers / pp) * 2.0 * (2.0 if recompute else 6.0) / 1e9
    return {
        "config": cfg,
        "estimated_t_comp_ms": round(t_comp_ms, 3),
        "estimated_activation_gb": round(act_gb, 3),
        "recompute": recompute,
        "confidence": 0.45 if math.isfinite(t_comp_ms) else 0.0,
    }


def run(model_spec: dict, configs: list[dict]) -> dict:
    """Run single-node profiling estimates across candidate variants."""
    inventory = _gpu_inventory()
    results = [_estimate(model_spec, cfg, inventory["device_count"]) for cfg in configs]
    return {
        "stage": "PROJECTION",
        "status": "success",
        "gpu_inventory": inventory,
        "results": results,
        "summary": {
            "configs": len(results),
            "best_by_t_comp": min(results, key=lambda r: r["estimated_t_comp_ms"]) if results else None,
        },
    }


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _failure(kind: str, message: str) -> dict[str, Any]:
    return {
        "stage": "PROFILER",
        "status": "failed",
        "failure": {"kind": kind, "message": message, "escalate_to_orchestrator": True},
    }


def _cli() -> int:
    p = argparse.ArgumentParser(prog="pilot.tools.profiler")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--model-spec", required=True)
    p_run.add_argument("--configs", default=None)
    args = p.parse_args()
    try:
        if args.cmd == "run":
            _emit(run(_load_mapping(args.model_spec), _load_configs(args.configs)))
            return 0
    except _ProfilerError as exc:
        _emit(_failure(exc.kind, str(exc)))
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
