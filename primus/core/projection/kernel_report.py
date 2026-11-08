###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""The kernels one training iteration runs, as simulate-mode projection prices them.

Simulate mode prices one representative layer of each type and reuses it for
every layer, so every GEMM, attention call and analytical term it prices is
recorded once here, labelled with its layer type (*scope*), op and phase
(``fwd``, ``dgrad``, ``wgrad``, ``bwd`` or ``step``). :func:`build_report` then
counts each row's calls per iteration on each pipeline stage.

Two documents adjust the terms as they are priced, so the pipeline simulation
and the iteration time see the adjusted times:

- a calibration (``primus.kernel_calibration``): a factor per op, built from a
  profile of the same stack, and the engine's own kernels the model lacks
  (``extras``), each a share of a layer's time in one phase;
- measured speedups (``primus.kernel_speedups``): what a tuned kernel bought,
  per op and optionally per shape.

Recording is off unless :func:`activate` was called; then every helper here is
a pass-through.
"""

from __future__ import annotations

import contextlib
import json
import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPORT_SCHEMA = "primus.kernel_report"
CALIBRATION_SCHEMA = "primus.kernel_calibration"
SPEEDUPS_SCHEMA = "primus.kernel_speedups"

#: Megatron's operand layouts of a linear layer's three GEMMs (weight stored [N, K]).
_LAYOUT = {"fwd": "NT", "dgrad": "NN", "wgrad": "TN"}

_SCOPE: ContextVar[str] = ContextVar("primus_kernel_scope", default="model")
_MODULE: ContextVar[Optional[str]] = ContextVar("primus_kernel_module", default=None)
_LABEL: ContextVar[Optional[Tuple[str, str]]] = ContextVar("primus_kernel_label", default=None)


def direction(phase: str) -> str:
    """``fwd``, ``bwd`` or ``step``: the pass a phase runs in (``step`` is once per iteration)."""
    return phase if phase in ("fwd", "step") else "bwd"


def _match(entry: dict, term: dict) -> Optional[int]:
    """How specific ``entry`` is for ``term`` (more fields matched is higher), or None if it doesn't apply.

    ``op`` ``*`` matches any op, so a measured shape applies wherever that shape is priced.
    """
    if entry.get("op") not in (term["op"], "*"):
        return None
    score = 1 if entry["op"] == "*" else 2
    for key in ("phase", "scope", "kind", "dtype"):
        if entry.get(key) is not None:
            if entry[key] != term.get(key):
                return None
            score += 1
    for key, value in (entry.get("dims") or {}).items():
        if term["dims"].get(key) != value:
            return None
        score += 1
    return score


def _best(entries: List[dict], term: dict) -> Optional[dict]:
    scored = [(s, e) for e in entries if (s := _match(e, term)) is not None]
    return max(scored, key=lambda se: se[0])[1] if scored else None


def _speedup(entry: dict) -> float:
    """An entry's speedup, given as such or as the baseline and tuned times it was measured at."""
    if entry.get("speedup"):
        return float(entry["speedup"])
    base, tuned = entry.get("baseline_us"), entry.get("tuned_us")
    return float(base) / float(tuned) if base and tuned else 0.0


def _load(path: str, schema: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get("schema") != schema:
        raise ValueError(f"{path} is not a {schema} document")
    return doc


@dataclass
class Row:
    scope: str
    op: str
    phase: str
    kind: str
    ms: float
    base_ms: float
    src: str
    dims: Dict[str, Any] = field(default_factory=dict)
    per_layer: int = 1


class Ledger:
    """The terms one projection priced, and the documents that adjust them."""

    def __init__(self, calibration: Optional[dict] = None, speedups: Optional[dict] = None) -> None:
        self.rows: List[Row] = []
        self.calibration = calibration or {}
        self.speedups = speedups or {}
        self.result: Dict[str, Any] = {}

    @classmethod
    def from_paths(cls, calibration: str = "", speedups: str = "") -> "Ledger":
        return cls(
            _load(calibration, CALIBRATION_SCHEMA) if calibration else None,
            _load(speedups, SPEEDUPS_SCHEMA) if speedups else None,
        )

    def adjust(
        self, scope: str, op: str, phase: str, ms: float, dims: dict, kind: str = ""
    ) -> Tuple[float, str]:
        """``ms`` as calibrated and sped up, and a note of what was applied."""
        term = {
            "scope": scope,
            "op": op,
            "phase": phase,
            "kind": kind,
            "dtype": dims.get("dtype"),
            "dims": dims,
        }
        notes = []
        cal = _best(self.calibration.get("factors") or [], term)
        if cal is not None:
            ms *= float(cal["factor"])
            notes.append("calibrated")
        fast = _best(self.speedups.get("speedups") or [], term)
        speedup = _speedup(fast) if fast is not None else 0.0
        if speedup > 0:
            ms /= speedup
            notes.append(f"x{speedup:g}")
        return ms, "+".join(notes)

    def record(
        self, op: str, phase: str, kind: str, ms: float, src: str, dims: dict, scope: Optional[str] = None
    ) -> float:
        scope = scope or _SCOPE.get()
        adjusted, note = self.adjust(scope, op, phase, ms, dims, kind)
        self.rows.append(
            Row(scope, op, phase, kind, adjusted, ms, src + (f"+{note}" if note else ""), dict(dims))
        )
        return adjusted

    def modelled_ms(self, scope: str, phase: str) -> float:
        """One call of ``scope``'s modelled time in ``phase`` (``fwd``, ``bwd`` or ``step``), before calibration and speedups."""
        seen = {}
        for row in self.rows:
            if row.scope == scope and row.src != "calibration" and direction(row.phase) == phase:
                seen.setdefault(
                    (row.op, row.phase, json.dumps(row.dims, sort_keys=True)), row.base_ms * row.per_layer
                )
        return sum(seen.values())

    def extras(self, scope: str, phase: str) -> float:
        """The engine's own time in ``scope``'s ``phase`` (a share of its modelled time), recorded."""
        added = 0.0
        for extra in self.calibration.get("extras") or []:
            if extra.get("scope") == scope and extra.get("phase") == phase:
                ms = float(extra["share"]) * self.modelled_ms(scope, phase)
                self.rows.append(Row(scope, str(extra["op"]), phase, "engine", ms, ms, "calibration", {}))
                added += ms
        return added


_ledger: Optional[Ledger] = None


def activate(ledger: Optional[Ledger]) -> Optional[Ledger]:
    """Record into ``ledger`` (``None`` stops recording); the previous ledger."""
    global _ledger
    previous, _ledger = _ledger, ledger
    return previous


def active() -> Optional[Ledger]:
    return _ledger


@contextlib.contextmanager
def _set(var: ContextVar, value: Any) -> Iterator[None]:
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


def scope(name: str) -> contextlib.AbstractContextManager:
    """Terms priced inside belong to ``name`` (a layer type, ``embedding``, ``output`` or ``iteration``)."""
    return _set(_SCOPE, name)


def module(name: str) -> contextlib.AbstractContextManager:
    """GEMMs priced inside by the shared MLP and attention helpers are named ``<name>_<projection>``."""
    return _set(_MODULE, name)


def module_name(default: str) -> str:
    return _MODULE.get() or default


def layer_scope(layer_type: str, compress_ratio: Optional[int] = None) -> str:
    """The scope of a decoder layer: its type, and its compress ratio when the model varies it per layer."""
    return layer_type if compress_ratio is None else f"{layer_type}_cr{compress_ratio}"


def label(op: str, phase: str) -> contextlib.AbstractContextManager:
    """The GEMM priced inside is ``op`` in ``phase``."""
    return _set(_LABEL, (op, phase))


def gemm(backend: Any, op: str, phase: str, *args: Any, **kwargs: Any) -> Any:
    """``backend.simulate_gemm(*args, **kwargs)``, priced as ``op`` in ``phase``."""
    with label(op, phase):
        return backend.simulate_gemm(*args, **kwargs)


def term(
    op: str, phase: str, ms: float, kind: str = "elementwise", scope_name: Optional[str] = None, **dims: Any
) -> float:
    """Record an analytically priced term; its time as calibrated and sped up (``ms`` when not recording)."""
    if _ledger is None or not ms:
        return ms
    return _ledger.record(op, phase, kind, float(ms), "analytic", dims, scope=scope_name)


def mark() -> int:
    """A position in the ledger, for :func:`repeat_since`."""
    return len(_ledger.rows) if _ledger is not None else 0


def repeat_since(position: int, times: int) -> None:
    """The rows recorded since ``position`` run ``times`` times per layer (a sequential per-expert loop)."""
    if _ledger is not None:
        for row in _ledger.rows[position:]:
            row.per_layer *= max(1, int(times))


def extras(scope_name: str, phase: str) -> float:
    """The calibration's engine time to add to ``scope_name``'s ``phase`` (0 when not recording)."""
    return _ledger.extras(scope_name, phase) if _ledger is not None else 0.0


# ── recording backends ───────────────────────────────────────────────────────


def wrap_backends(gemm_backend: Any, sdpa_backend: Any) -> Tuple[Any, Any]:
    """The backends to simulate with: recording ones when a ledger is active."""
    if _ledger is None:
        return gemm_backend, sdpa_backend
    from .simulation_backends.base import (
        GEMMSimulationBackend,
        SDPASimulationBackend,
        SimulationResult,
    )

    class RecordingGEMMBackend(GEMMSimulationBackend):
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def name(self) -> str:
            return self.inner.name()

        def is_available(self) -> bool:
            return self.inner.is_available()

        @property
        def hbm_bandwidth_gbps(self):
            return self.inner.hbm_bandwidth_gbps

        def __getattr__(self, item: str) -> Any:
            return getattr(self.inner, item)

        def simulate_gemm(self, m, n, k, dtype="bf16", trans_a=False, trans_b=False, batch=1):
            res = self.inner.simulate_gemm(m, n, k, dtype, trans_a=trans_a, trans_b=trans_b, batch=batch)
            op, phase = _LABEL.get() or (f"{module_name('model')}_gemm", "any")
            dims = {"m": int(m), "n": int(n), "k": int(k), "batch": int(batch), "dtype": dtype}
            if phase in _LAYOUT:
                dims["layout"] = _LAYOUT[phase]
            kind = "grouped_gemm" if batch > 1 else "gemm"
            if _ledger is None:
                return res
            ms = _ledger.record(op, phase, kind, res.forward_time_ms, self.inner.name(), dims)
            return SimulationResult(
                forward_time_ms=ms,
                backward_time_ms=res.backward_time_ms,
                tflops=res.tflops,
                bandwidth_gbps=res.bandwidth_gbps,
                metadata=res.metadata,
            )

    class RecordingSDPABackend(SDPASimulationBackend):
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def name(self) -> str:
            return self.inner.name()

        def is_available(self) -> bool:
            return self.inner.is_available()

        def __getattr__(self, item: str) -> Any:
            return getattr(self.inner, item)

        def simulate_sdpa(
            self,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            causal=True,
            dtype="bf16",
            seq_len_kv=None,
            num_heads_kv=None,
            head_dim_v=None,
        ):
            res = self.inner.simulate_sdpa(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                causal=causal,
                dtype=dtype,
                seq_len_kv=seq_len_kv,
                num_heads_kv=num_heads_kv,
                head_dim_v=head_dim_v,
            )
            if _ledger is None:
                return res
            dims = {
                "batch": int(batch_size),
                "heads": int(num_heads),
                "kv_heads": int(num_heads_kv or num_heads),
                "seq": int(seq_len),
                "seq_kv": int(seq_len_kv or seq_len),
                "head_dim": int(head_dim),
                "head_dim_v": int(head_dim_v or head_dim),
                "causal": bool(causal),
                "dtype": dtype,
            }
            op = f"{module_name('attn')}_sdpa"
            fwd = _ledger.record(op, "fwd", "attention", res.forward_time_ms, self.inner.name(), dims)
            bwd = _ledger.record(op, "bwd", "attention", res.backward_time_ms, self.inner.name(), dims)
            return SimulationResult(
                forward_time_ms=fwd,
                backward_time_ms=bwd,
                tflops=res.tflops,
                bandwidth_gbps=res.bandwidth_gbps,
                metadata=res.metadata,
            )

    gemm = RecordingGEMMBackend(gemm_backend) if gemm_backend is not None else None
    sdpa = RecordingSDPABackend(sdpa_backend) if sdpa_backend is not None else None
    return gemm, sdpa


# ── the report ───────────────────────────────────────────────────────────────


def _stage_layers(training_config) -> List[List[int]]:
    """Each pipeline stage's decoder layers (all virtual chunks), as the pipeline simulation splits them."""
    from .module_profilers.language_model import LanguageModelProfiler

    mc, mp = training_config.model_config, training_config.model_parallel_config
    total = int(getattr(mc, "num_layers", 0) or 0)
    pp = getattr(mp, "pipeline_model_parallel_size", 1) or 1
    if pp <= 1:
        return [list(range(total))]
    tp = getattr(mp, "tensor_model_parallel_size", 1) or 1
    cp = getattr(mp, "context_model_parallel_size", 1) or 1
    ep = getattr(mp, "expert_model_parallel_size", 1) or 1
    stages = []
    for rank in range(pp):
        chunks = LanguageModelProfiler.get_virtual_stage_layers_for_rank(
            None,
            global_rank=rank * tp * cp * ep,
            n_layers=total,
            pp_size=pp,
            tp_size=tp,
            cp_size=cp,
            ep_size=ep,
            num_virtual_pipeline_stages=getattr(mp, "virtual_pipeline_model_parallel_size", 1) or 1,
            decoder_first_pipeline_num_layers=getattr(mp, "decoder_first_pipeline_num_layers", None),
            decoder_last_pipeline_num_layers=getattr(mp, "decoder_last_pipeline_num_layers", None),
            pipeline_model_parallel_layout=getattr(mp, "pipeline_model_parallel_layout", None),
        )
        stages.append([layer for chunk in chunks or [] for layer in chunk])
    return stages


def _recomputed(training_config, layers: List[int]) -> List[int]:
    """The stage ``layers`` that re-run their forward in backward (full recompute)."""
    mp = training_config.model_parallel_config
    if getattr(mp, "recompute_granularity", None) != "full":
        return []
    n = getattr(mp, "recompute_num_layers", None)
    if getattr(mp, "recompute_method", None) == "block" and n is not None:
        return layers[: int(n)]
    return list(layers)


def _layer_scopes(training_config) -> List[str]:
    """Each decoder layer's scope, as the layer loop names its representative."""
    from .module_profilers.language_model import _parse_compress_ratios

    mc = training_config.model_config
    total = int(getattr(mc, "num_layers", 0) or 0)
    pattern = list(getattr(mc, "moe_pattern", None) or [0] * total)
    ratios = _parse_compress_ratios(mc) if getattr(mc, "compress_ratios", None) is not None else None
    scopes = []
    for i in range(total):
        cr = int(ratios[i]) if ratios and i < len(ratios) else None
        scopes.append(layer_scope("moe" if i < len(pattern) and pattern[i] else "dense", cr))
    return scopes


def build_report(ledger: Ledger, training_config, num_gpus: int, provenance: Optional[dict] = None) -> dict:
    """Every recorded term with its calls per iteration on each pipeline stage, one rank of the stage."""
    mc, mp, rt = (
        training_config.model_config,
        training_config.model_parallel_config,
        training_config.runtime_config,
    )
    tp = getattr(mp, "tensor_model_parallel_size", 1) or 1
    pp = getattr(mp, "pipeline_model_parallel_size", 1) or 1
    cp = getattr(mp, "context_model_parallel_size", 1) or 1
    dp = max(1, num_gpus // (tp * pp * cp))
    mbs = getattr(rt, "micro_batch_size", 1) or 1
    gbs = getattr(rt, "global_batch_size", mbs * dp) or mbs * dp
    microbatches = max(1, math.ceil(gbs / (mbs * dp)))
    scope_of = _layer_scopes(training_config)
    stages = _stage_layers(training_config)
    selective = getattr(mp, "recompute_granularity", None) == "selective"

    by_key: Dict[tuple, Row] = {}
    for row in ledger.rows:
        key = (row.scope, row.op, row.phase, json.dumps(row.dims, sort_keys=True))
        by_key.setdefault(key, row)

    rows = []
    for stage, layers in enumerate(stages):
        per_scope: Dict[str, int] = {}
        recomputed: Dict[str, int] = {}
        for i in layers:
            per_scope[scope_of[i]] = per_scope.get(scope_of[i], 0) + 1
        for i in _recomputed(training_config, layers):
            recomputed[scope_of[i]] = recomputed.get(scope_of[i], 0) + 1
        for row in by_key.values():
            if row.scope.startswith(("dense", "moe")):
                calls = per_scope.get(row.scope, 0) * microbatches
                if row.phase == "fwd":
                    calls += recomputed.get(row.scope, 0) * microbatches
                    if selective and row.kind == "attention":
                        calls += per_scope.get(row.scope, 0) * microbatches
            elif row.scope == "embedding":
                calls = microbatches if stage == 0 else 0
            elif row.scope == "output":
                calls = microbatches if stage == len(stages) - 1 else 0
            elif row.scope == "iteration":
                calls = 1
            else:
                calls = microbatches
            calls *= row.per_layer
            if calls <= 0:
                continue
            rows.append(
                {
                    "stage": stage,
                    "scope": row.scope,
                    "op": row.op,
                    "phase": row.phase,
                    "kind": row.kind,
                    "dims": row.dims,
                    "est_ms": round(row.ms, 6),
                    "base_ms": round(row.base_ms, 6),
                    "calls": calls,
                    "est_total_ms": round(row.ms * calls, 6),
                    "src": row.src,
                }
            )
    rows.sort(key=lambda r: (r["stage"], -r["est_total_ms"]))
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "deployment": {
            "tp": tp,
            "pp": pp,
            "ep": getattr(mp, "expert_model_parallel_size", 1) or 1,
            "cp": cp,
            "dp": dp,
            "vpp": getattr(mp, "virtual_pipeline_model_parallel_size", 1) or 1,
            "gpus": num_gpus,
            "micro_batch_size": mbs,
            "global_batch_size": gbs,
            "microbatches": microbatches,
            "seq_length": getattr(rt, "sequence_length", None),
            "stage_layers": stages,
        },
        "iteration": dict(ledger.result),
        "provenance": dict(provenance or {}),
        "kernels": rows,
    }


def write_report(
    ledger: Ledger, training_config, num_gpus: int, path: str, provenance: Optional[dict] = None
) -> dict:
    doc = build_report(ledger, training_config, num_gpus, provenance)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return doc
