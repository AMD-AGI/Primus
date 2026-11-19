###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Validate an op model before it is registered.

An op model is accepted only when its cost is physically possible and agrees
with the math it came from.  On the workload's own shape plus two scaled
variants (2x batch, 1/2 sequence) it must:

* be structurally complete: name, valid slot, references, the gap's keys
  consumed or explicitly ignored;
* actually match the workload and run on a sane share of layers;
* return finite, positive time, FLOPs and bytes, with a breakdown that sums
  to the totals;
* never beat the roofline -- analytical time below ``max(FLOPs / peak,
  bytes / HBM BW)`` is impossible -- and not sit absurdly far above it;
* have a plausible backward / forward ratio for a training op;
* agree with :meth:`OpModel.reference_flops`, the closed form written from
  the paper independently of the kernel decomposition;
* grow with the token count.

The CLI prints a report (``--json`` for the tuning agent) and exits non-zero
when any check fails::

    python -m primus.core.projection.op_models.validate \\
        --config examples/megatron/configs/MI355X/qwen3_5_35B_A3B-BF16-pretrain.yaml \\
        --op-file primus/core/projection/op_models/generated/gated_delta_net.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from .base import SLOTS, OpContext, OpCost, OpModel

# Tolerances.  Analytical work may not undercut the roofline (a few percent of
# slack for clock / efficiency rounding in the backends) and should stay within
# 50x of it: further out the decomposition is pricing something it should not.
ROOFLINE_SLACK = 0.97
ROOFLINE_MAX_RATIO = 50.0
BWD_FWD_RATIO = (0.8, 4.0)
REFERENCE_FLOPS_TOL = 0.25
BREAKDOWN_TOL = 0.02

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class ValidationReport:
    op_name: str
    checks: List[Check] = field(default_factory=list)
    cases: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op_name,
            "ok": self.ok,
            "checks": [c.__dict__ for c in self.checks],
            "cases": self.cases,
        }

    def feedback(self) -> str:
        """What to fix, phrased for the coder that wrote the op."""
        lines = [f"Validation of op model '{self.op_name}': {'PASSED' if self.ok else 'FAILED'}"]
        for c in self.checks:
            if c.status != PASS:
                lines.append(f"- [{c.status.upper()}] {c.name}: {c.detail}")
        for case in self.cases:
            lines.append(
                f"  case {case['label']}: fwd={case['fwd_ms']:.4f} ms bwd={case['bwd_ms']:.4f} ms "
                f"roofline_fwd={case['roofline_fwd_ms']:.4f} ms flops_fwd={case['fwd_flops']:.3e}"
            )
        return "\n".join(lines)

    def text(self) -> str:
        out = [f"Op model '{self.op_name}': {'OK' if self.ok else 'FAILED'}"]
        for c in self.checks:
            out.append(f"  {c.status.upper():4s} {c.name}" + (f" -- {c.detail}" if c.detail else ""))
        for case in self.cases:
            out.append(
                f"  case {case['label']:>10s}: fwd {case['fwd_ms']:.4f} ms / bwd {case['bwd_ms']:.4f} ms"
                f"  (roofline fwd {case['roofline_fwd_ms']:.4f} ms, {case['fwd_flops']:.3e} FLOPs,"
                f" {case['fwd_bytes']:.3e} B)"
            )
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Workload loading
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    training_config: Any
    gemm_backend: Any
    sdpa_backend: Any
    batch_size: int
    seq_len: int


def load_workload(
    config_path: str,
    *,
    gpu_arch: Optional[str] = None,
    overrides: Sequence[str] = (),
    simulate_backends: bool = True,
) -> Workload:
    """Build the TrainingConfig and simulation backends the simulate path would use."""
    from primus.core.launcher.parser import load_primus_config
    from primus.core.projection.frameworks import normalize_primus_config
    from primus.core.projection.performance_projection.projection import (
        _limit_layers_for_projection,
        _rescale_expert_parallelism,
    )
    from primus.core.projection.training_config import (
        convert_primus_config_to_projection_config,
    )

    if gpu_arch:
        os.environ.setdefault("PRIMUS_GPU_ARCH", gpu_arch)
    primus_config, _ = load_primus_config(SimpleNamespace(config=str(config_path)), list(overrides))
    normalize_primus_config(primus_config)
    module_config = primus_config.get_module_config("pre_trainer")
    _limit_layers_for_projection(module_config)
    _rescale_expert_parallelism(module_config)
    tc = convert_primus_config_to_projection_config(primus_config)

    gemm = sdpa = None
    if simulate_backends:
        from primus.core.projection.simulation_backends.factory import (
            get_gemm_simulation_backend,
            get_sdpa_simulation_backend,
        )

        try:
            gemm = get_gemm_simulation_backend(backend_name=None, gpu_arch=gpu_arch, gpu_clock_mhz=None)
            sdpa = get_sdpa_simulation_backend(gpu_arch=gpu_arch, gpu_clock_mhz=None)
        except Exception as exc:  # backends are optional; the roofline fallback still validates
            print(f"[Primus:OpValidate] simulation backends unavailable ({exc}); using roofline primitives")
    rt = tc.runtime_config
    return Workload(tc, gemm, sdpa, int(rt.micro_batch_size), int(rt.sequence_length))


def load_op_class(path: str) -> Type[OpModel]:
    """Import *path* and return the OpModel subclass it registers."""
    from .registry import load_op_model_file, registered_op_models

    registered_op_models()  # discover first so the file's registration wins
    found = load_op_model_file(path)
    if not found:
        raise ValueError(f"{path} registers no OpModel (decorate the class with @register_op_model)")
    if len(found) > 1:
        raise ValueError(f"{path} registers {len(found)} op models; put one op per file")
    return found[0]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _finite_pos(x: Any) -> bool:
    try:
        return math.isfinite(float(x)) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def _case_configs(wl: Workload) -> List[Tuple[str, int, int]]:
    b, s = wl.batch_size, wl.seq_len
    cases = [("workload", b, s), ("2x_batch", 2 * b, s)]
    if s >= 256:
        cases.append(("half_seq", b, s // 2))
    return cases


def validate_op_model(
    op: Union[OpModel, Type[OpModel]],
    wl: Workload,
    *,
    gap_keys: Sequence[str] = (),
) -> ValidationReport:
    """Run every check on *op* against workload *wl*."""
    inst = op() if isinstance(op, type) else op
    cls = type(inst)
    rep = ValidationReport(getattr(cls, "name", "") or cls.__name__)
    mc = wl.training_config.model_config

    # ---- structure ----
    problems = []
    if not _NAME_RE.match(cls.name or ""):
        problems.append(f"name {cls.name!r} is not snake_case")
    if cls.slot not in SLOTS:
        problems.append(f"slot {cls.slot!r} not in {SLOTS}")
    if not cls.claims:
        problems.append("claims is empty: the coverage check cannot tell this op is modeled")
    if not cls.references:
        problems.append("references is empty: cite the paper / model card / kernel the equations come from")
    rep.add("structure", FAIL if problems else PASS, "; ".join(problems))

    covered = set(cls.claims) | set(cls.consumes) | set(getattr(cls, "ignores", {}) or {})
    missing = sorted(k for k in gap_keys if k not in covered)
    rep.add(
        "gap_keys_covered",
        FAIL if missing else PASS,
        (f"keys {missing} are neither consumed nor listed in `ignores` with a reason" if missing else ""),
    )

    # ---- presence ----
    try:
        matched = bool(cls.matches(mc))
    except Exception as exc:
        matched = False
        rep.add("matches", FAIL, f"matches() raised {exc!r}")
    else:
        rep.add(
            "matches", PASS if matched else FAIL, "" if matched else "matches() is False for this workload"
        )
    try:
        frac = float(inst.layer_fraction(mc))
        ok = 0.0 < frac <= 1.0
        rep.add(
            "layer_fraction", PASS if ok else FAIL, f"{frac:.4f}" + ("" if ok else " (must be in (0, 1])")
        )
    except Exception as exc:
        rep.add("layer_fraction", FAIL, f"layer_fraction() raised {exc!r}")

    # ---- per-shape cost checks ----
    costs: List[Tuple[str, int, int, OpCost, OpContext]] = []
    for label, b, s in _case_configs(wl):
        ctx = OpContext(wl.training_config, b, s, wl.gemm_backend, wl.sdpa_backend)
        try:
            c = inst.cost(ctx)
        except Exception as exc:
            rep.add(f"cost[{label}]", FAIL, f"cost() raised {type(exc).__name__}: {exc}")
            continue
        if not isinstance(c, OpCost):
            rep.add(f"cost[{label}]", FAIL, f"cost() returned {type(c).__name__}, expected OpCost")
            continue
        costs.append((label, b, s, c, ctx))

    for label, b, s, c, ctx in costs:
        bad = [
            n
            for n in ("fwd_ms", "bwd_ms", "fwd_flops", "bwd_flops", "fwd_bytes", "bwd_bytes")
            if not _finite_pos(getattr(c, n))
        ]
        rep.add(
            f"finite_positive[{label}]",
            FAIL if bad else PASS,
            f"{bad} must be finite and > 0" if bad else "",
        )
        roof_f = ctx.roofline_ms(c.fwd_flops, c.fwd_bytes)
        roof_b = ctx.roofline_ms(c.bwd_flops, c.bwd_bytes)
        rep.cases.append(
            {
                "label": label,
                "batch_size": b,
                "seq_len": s,
                "fwd_ms": float(c.fwd_ms),
                "bwd_ms": float(c.bwd_ms),
                "fwd_flops": float(c.fwd_flops),
                "bwd_flops": float(c.bwd_flops),
                "fwd_bytes": float(c.fwd_bytes),
                "bwd_bytes": float(c.bwd_bytes),
                "roofline_fwd_ms": roof_f,
                "roofline_bwd_ms": roof_b,
                "breakdown": {k: list(v) for k, v in c.breakdown.items()},
            }
        )
        if bad:
            continue
        msgs, status = [], PASS
        for tag, t, roof in (("fwd", c.fwd_ms, roof_f), ("bwd", c.bwd_ms, roof_b)):
            if t < ROOFLINE_SLACK * roof:
                status = FAIL
                msgs.append(
                    f"{tag} {t:.4f} ms beats the roofline {roof:.4f} ms (FLOPs or bytes overstated, or time too low)"
                )
            elif roof > 0 and t > ROOFLINE_MAX_RATIO * roof:
                status = FAIL if status == FAIL else WARN
                msgs.append(f"{tag} {t:.4f} ms is {t / roof:.0f}x the roofline {roof:.4f} ms")
        rep.add(f"roofline[{label}]", status, "; ".join(msgs))

        ratio = c.bwd_ms / c.fwd_ms
        lo, hi = BWD_FWD_RATIO
        rep.add(
            f"bwd_fwd_ratio[{label}]",
            PASS if lo <= ratio <= hi else FAIL,
            f"{ratio:.2f}" + ("" if lo <= ratio <= hi else f" (expected {lo}..{hi} for a trained op)"),
        )

        if c.breakdown:
            sf = sum(v[0] for v in c.breakdown.values())
            sb = sum(v[1] for v in c.breakdown.values())
            off = max(abs(sf - c.fwd_ms) / c.fwd_ms, abs(sb - c.bwd_ms) / c.bwd_ms)
            rep.add(
                f"breakdown_sums[{label}]",
                PASS if off <= BREAKDOWN_TOL else FAIL,
                (
                    f"breakdown sums to {sf:.4f}/{sb:.4f} ms vs totals {c.fwd_ms:.4f}/{c.bwd_ms:.4f} ms"
                    if off > BREAKDOWN_TOL
                    else ""
                ),
            )

        try:
            ref = inst.reference_flops(ctx)
        except Exception as exc:
            rep.add(f"reference_flops[{label}]", FAIL, f"reference_flops() raised {exc!r}")
            ref = None
        else:
            if ref is None:
                rep.add(f"reference_flops[{label}]", WARN, "no closed form given; FLOPs not cross-checked")
        if ref is not None:
            rf, rb = (float(x) for x in ref)
            errs = []
            for tag, got, want in (("fwd", c.fwd_flops, rf), ("bwd", c.bwd_flops, rb)):
                if want > 0 and abs(got - want) / want > REFERENCE_FLOPS_TOL:
                    errs.append(f"{tag} {got:.3e} vs closed form {want:.3e} ({(got - want) / want:+.0%})")
            rep.add(
                f"reference_flops[{label}]",
                FAIL if errs else PASS,
                "; ".join(errs) or f"within {REFERENCE_FLOPS_TOL:.0%}",
            )

    by_label = {label: c for label, _, _, c, _ in costs}
    base = by_label.get("workload")
    for label, more in (("2x_batch", True), ("half_seq", False)):
        other = by_label.get(label)
        if base is None or other is None or not (_finite_pos(base.fwd_ms) and _finite_pos(other.fwd_ms)):
            continue
        grows = other.fwd_ms >= base.fwd_ms * 0.999 if more else other.fwd_ms <= base.fwd_ms * 1.001
        rep.add(
            f"monotonic[{label}]",
            PASS if grows else FAIL,
            (
                ""
                if grows
                else f"fwd {other.fwd_ms:.4f} ms vs workload {base.fwd_ms:.4f} ms: cost must grow with tokens"
            ),
        )
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: Optional[Sequence[str]]) -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="Primus workload YAML")
    p.add_argument("--op-file", required=True, help="Python file defining the OpModel")
    p.add_argument("--gpu-arch", default=os.getenv("PRIMUS_GPU_ARCH"))
    p.add_argument("--gap-keys", default="", help="Comma-separated config keys the op must cover")
    p.add_argument("--no-backends", action="store_true", help="Validate on roofline primitives only")
    p.add_argument("--json", action="store_true")
    return p.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, overrides = _parse(argv)
    os.environ.setdefault("PRIMUS_OP_COVERAGE", "0")
    try:
        op_cls = load_op_class(args.op_file)
        wl = load_workload(
            args.config, gpu_arch=args.gpu_arch, overrides=overrides, simulate_backends=not args.no_backends
        )
        gap_keys = [k for k in args.gap_keys.split(",") if k]
        rep = validate_op_model(op_cls, wl, gap_keys=gap_keys)
    except Exception as exc:
        rep = ValidationReport(os.path.basename(args.op_file))
        rep.add("load", FAIL, f"{type(exc).__name__}: {exc}")
    if args.json:
        print(json.dumps(rep.as_dict(), indent=2, default=str))
    else:
        print(rep.text())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
