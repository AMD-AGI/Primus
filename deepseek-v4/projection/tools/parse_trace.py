#!/usr/bin/env python3
"""Turn per-cr chrome traces into a projection breakdown JSON.

Input: one rank-0 PyTorch/Kineto chrome trace per compression-ratio (cr), each
captured by ``script/deepseek_v4_layer_trace-projection.sh`` (1 layer, seq 4096,
GA=2, recompute off, overlap off, profiler window iter 6->7).

Output: a single ``<model>.json`` matching ``design/03-json-schema.md``.

Attribution (validated against the real ROCm/Kineto trace):
  * GPU kernels (cat=="kernel") link to their launching CPU op via the shared
    ``External id`` arg. Optimizer (``multi_tensor_apply``) and DP-comm
    (``nccl``) kernels carry no External id and are classified by name.
  * The module comes from the enclosing ``nn.Module: <Class>_n`` python_function
    events (with_stack) on the CPU op's thread; fwd/bwd from "Backward"/
    "autograd" in the CPU op name or an enclosing frame.
  * Clean per-call time = ``min`` over launches grouped by
    ``(phase, module, kernel, input-dims)`` (overlap is off, so this just
    removes warm-up/jitter); calls-per-microbatch is treated as 1 for a single
    captured layer.
  * Compute-bound FLOP class from the kernel name; GEMM FLOPs from input dims.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kernel_module_map import flop_class_from_kernel

PRO_COMPRESS = [128, 128] + [4 if i % 2 == 0 else 128 for i in range(2, 60)] + [0]
FLASH_COMPRESS = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(2, 42)] + [0]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "pro": {
        "num_layers": 61,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "kv_channels": 512,
        "num_experts": 384,
        "moe_router_topk": 6,
        "moe_ffn_hidden_size": 3072,
        "moe_shared_expert_intermediate_size": 3072,
        "index_topk": 1024,
        "vocab_size": 129280,
        "compress_ratios": PRO_COMPRESS,
    },
    "flash": {
        "num_layers": 43,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "kv_channels": 512,
        "num_experts": 256,
        "moe_router_topk": 6,
        "moe_ffn_hidden_size": 2048,
        "moe_shared_expert_intermediate_size": 2048,
        "index_topk": 512,
        "vocab_size": 129280,
        "compress_ratios": FLASH_COMPRESS,
    },
}

# MI355X: BF16 matrix 2.5 PFLOPS, HBM3E 8 TB/s (AMD product page). MI455X
# (MI400): HBM4 19.6 TB/s; BF16 dense not officially published — estimated
# ~10 PFLOPS (half of the 20 PFLOPS FP8 spec). The site can override these.
DEFAULT_HARDWARE = {
    "MI355X": {"peak_tflops_bf16": 2500.0, "hbm_bandwidth_gbps": 8000.0},
    "MI455X": {"peak_tflops_bf16": 10000.0, "hbm_bandwidth_gbps": 19600.0},
}

ALL_MODULES = (
    "attn.proj",
    "attn.core",
    "attn.indexer",
    "attn.norm",
    "moe.router",
    "moe.dispatch",
    "moe.grouped_gemm",
    "moe.shared_expert",
    "moe.combine",
    "embedding",
    "output",
    "loss",
    "other",
)


def _arg(ev: dict, *keys: str) -> Any:
    args = ev.get("args") or {}
    for k in keys:
        if k in args:
            return args[k]
    return None


def _dims_key(dims: Any) -> str:
    if dims is None:
        return ""
    try:
        return json.dumps(dims, separators=(",", ":"))
    except TypeError:
        return str(dims)


def _gemm_flops(dims_key: str) -> float | None:
    if not dims_key:
        return None
    try:
        dims = json.loads(dims_key)
    except json.JSONDecodeError:
        return None
    shapes = [d for d in dims if isinstance(d, list) and len(d) >= 2 and all(isinstance(x, int) for x in d)]
    if len(shapes) >= 2:
        a, b = shapes[0], shapes[1]
        m, k = a[-2], a[-1]
        k2, n = b[-2], b[-1]
        if k == k2:
            batch = 1
            for x in a[:-2]:
                batch *= x
            return 2.0 * batch * m * n * k
    return None


def _resolve_module(kname: str, cpu_name: str, anc_classes: set[str], flop_class: str | None) -> str:
    """Logical module for a kernel given its name, CPU op name, and enclosing
    nn.Module classes. Returns '__optimizer__' / '__dpcomm__' for non-layer
    buckets handled separately. Priority: kernel name > cpu-op name > enclosing
    nn.Module (forward only; backward ops aren't inside module forward ranges)."""
    n = kname
    low = n.lower()
    cn = cpu_name or ""

    def has(*xs: str) -> bool:
        return any(any(x in a for a in anc_classes) for x in xs)

    # 1) kernel-name rules (most reliable; present for fwd and bwd)
    if "multi_tensor_apply" in n or "fusedadam" in low or "adamw" in low:
        return "__optimizer__"
    if "nccl" in low:
        return "__dpcomm__"
    if "deep_ep" in n or "deepep" in low:
        return "moe.combine" if "combine" in low else "moe.dispatch"
    if "_v4_csa" in n or "_v4_attention" in n or "_hc_" in n:
        return "attn.core"
    if "_sinkhorn" in n or "_v4_router" in n:
        return "moe.router"
    if "GroupedGemm" in n or "_grouped" in n or "group_gemm" in low or "grouped_variable" in n:
        return "moe.grouped_gemm"

    # 2) cpu-op (autograd Function / aten) name rules — needed for backward
    if "Attention" in cn or "CSAPool" in cn or "MLA" in cn:
        return "attn.core"
    if "Indexer" in cn or "Compressor" in cn:
        return "attn.indexer"
    if "Sinkhorn" in cn or "Router" in cn:
        return "moe.router"
    if "RMSNorm" in cn or "LayerNorm" in cn or "layer_norm" in cn.lower():
        return "attn.norm"
    if "cross_entropy" in cn.lower() or "nll_loss" in cn.lower():
        return "loss"
    if "embedding" in cn.lower():
        return "output" if flop_class == "gemm" else "embedding"
    if "LinearWithGradAccumulation" in cn or cn in ("aten::mm", "aten::addmm", "aten::matmul", "aten::bmm"):
        if has("Embedding") or (
            has("DeepseekV4Model")
            and not has(
                "DeepseekV4Attention", "DeepseekV4HybridLayer", "Compressor", "Indexer", "MLP", "Expert"
            )
        ):
            return "output"
        if has("Compressor", "Indexer"):
            return "attn.indexer"
        if has("MLP", "Expert"):
            return "moe.grouped_gemm"
        return "attn.proj"

    # 3) enclosing nn.Module (forward only)
    if has("Compressor", "Indexer"):
        return "attn.indexer"
    if has("SharedExpert"):
        return "moe.shared_expert"
    if has("GroupedMLP", "SequentialMLP", "GroupedExperts", "Experts"):
        return "moe.grouped_gemm"
    if has("Router"):
        return "moe.router"
    if has("DeepseekV4Attention", "MLASelfAttention", "SelfAttention"):
        return "attn.proj" if flop_class == "gemm" else "attn.norm"
    if has("Embedding"):
        return "output" if flop_class == "gemm" else "embedding"
    return "other"


def parse_trace(path: Path):
    payload = json.loads(path.read_text())
    events = payload.get("traceEvents", [])

    # index cpu ops by External id
    cpu_by_extid: dict[Any, dict] = {}
    # interesting python_function intervals per tid: (ts, end, name)
    pf_by_tid: dict[Any, list[tuple[float, float, str]]] = defaultdict(list)
    kernels: list[dict] = []

    for ev in events:
        cat = (ev.get("cat") or "").lower()
        ph = ev.get("ph")
        if cat == "cpu_op" and ph == "X":
            ext = _arg(ev, "External id")
            if ext is not None and ext not in cpu_by_extid:
                cpu_by_extid[ext] = ev
        elif cat == "python_function" and ph == "X":
            name = ev.get("name", "")
            if name.startswith("nn.Module:") or "ackward" in name or "autograd" in name:
                ts = ev.get("ts")
                dur = ev.get("dur") or 0
                if ts is not None:
                    pf_by_tid[ev.get("tid")].append((ts, ts + dur, name))
        elif cat == "kernel" and ph == "X" and ev.get("dur") is not None:
            kernels.append(ev)

    for tid in pf_by_tid:
        pf_by_tid[tid].sort(key=lambda t: t[0])

    def enclosing(cpu: dict) -> set[str]:
        """Return enclosing nn.Module class names for a cpu op (forward only;
        backward ops live under the autograd engine, not module forward ranges)."""
        tid = cpu.get("tid")
        ts = cpu.get("ts")
        end = ts + (cpu.get("dur") or 0)
        classes: set[str] = set()
        for pts, pend, name in pf_by_tid.get(tid, ()):
            if pts > ts:
                break
            if pend >= end and name.startswith("nn.Module:"):
                classes.add(name.split(":", 1)[1].strip().rsplit("_", 1)[0])
        return classes

    # group key -> list of durations (us)
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    group_flops: dict[tuple[str, str, str, str], float | None] = {}
    optimizer_us = 0.0
    dpcomm_us = 0.0

    for ev in kernels:
        name = ev.get("name", "")
        dur = float(ev["dur"])
        ext = _arg(ev, "External id")
        cpu = cpu_by_extid.get(ext) if ext is not None else None
        cpu_name = cpu.get("name", "") if cpu else ""
        anc = enclosing(cpu) if cpu else set()
        flop_class = flop_class_from_kernel(name)
        module = _resolve_module(name, cpu_name, anc, flop_class)
        if module == "__optimizer__":
            optimizer_us += dur
            continue
        if module == "__dpcomm__":
            dpcomm_us += dur
            continue
        # Phase: a kernel-name _fwd_/_bwd_ tag is authoritative (V4 triton
        # kernels encode it, and dense attention re-runs its _fwd_ kernel inside
        # backward — whose CPU op carries a "Fwd thread id" — so the name must
        # win). Otherwise use the backward-only "Fwd thread id" arg / cpu name.
        ln = name.lower()
        if "_fwd" in ln and "_bwd" not in ln:
            phase = "forward"
        elif "_bwd" in ln:
            phase = "backward"
        else:
            fwd_tid = _arg(cpu, "Fwd thread id") if cpu else None
            phase = "backward" if (fwd_tid is not None or "ackward" in cpu_name) else "forward"
        dims_key = _dims_key(_arg(cpu, "Input Dims")) if cpu else ""
        key = (phase, module, name, dims_key)
        groups[key].append(dur)
        if key not in group_flops and flop_class == "gemm":
            group_flops[key] = _gemm_flops(dims_key)

    # collapse: min per group; aggregate to module rows
    agg: dict[tuple[str, str], dict] = {}
    for (phase, module, kname, _dims), durs in groups.items():
        clean = min(durs)
        flop_class = flop_class_from_kernel(kname)
        flops = group_flops.get((phase, module, kname, _dims))
        row = agg.setdefault(
            (phase, module),
            {
                "module": module,
                "time_us": 0.0,
                "flops": 0.0,
                "has_flops": False,
                "flop_class": None,
                "kernels": [],
            },
        )
        row["time_us"] += clean
        if flop_class:
            row["flop_class"] = flop_class
        if flops:
            row["flops"] += flops
            row["has_flops"] = True
        row["kernels"].append({"name": kname[:80], "time_us": round(clean, 3), "launches": 1})

    out = {b: {"forward": {}, "backward": {}} for b in ("attention", "moe", "embedding", "output", "loss")}
    for (phase, module), row in agg.items():
        compute = row["flop_class"] is not None
        flops = row["flops"] if row["has_flops"] else None
        time_s = row["time_us"] * 1e-6
        tflops = (flops / time_s / 1e12) if (flops and time_s > 0) else None
        entry = {
            "module": module,
            "time_us": round(row["time_us"], 3),
            "class": "compute_bound" if compute else "memory_bound",
            "flop_class": row["flop_class"],
            "flops": flops,
            "tflops": round(tflops, 1) if tflops else None,
            "kernels": sorted(row["kernels"], key=lambda k: -k["time_us"])[:6],
        }
        bucket = (
            "attention"
            if module.startswith("attn.")
            else (
                "moe"
                if module.startswith("moe.") or module == "other"
                else module if module in ("embedding", "output", "loss") else "moe"
            )
        )
        out[bucket][phase][module] = entry
    return out, optimizer_us, dpcomm_us


def _lists(bd: dict) -> dict:
    return {p: sorted(bd[p].values(), key=lambda r: -r["time_us"]) for p in ("forward", "backward")}


def cr_layer_counts(compress: list[int]) -> dict[str, int]:
    counts = {"0": 0, "4": 0, "128": 0}
    for c in compress:
        counts[str(c)] = counts.get(str(c), 0) + 1
    return counts


def _fwd_total(buckets: dict) -> float:
    return sum(r["time_us"] for r in buckets["attention"]["forward"].values()) + sum(
        r["time_us"] for r in buckets["moe"]["forward"].values()
    )


def build(model: str, traces: dict[str, Path]) -> dict[str, Any]:
    cfg = MODEL_CONFIGS[model]
    per_cr, opt_us = {}, []
    for cr, path in traces.items():
        buckets, o, _dp = parse_trace(path)
        per_cr[cr] = buckets
        opt_us.append(o)

    # Some cr layers (pure dense cr=0 / HCA cr=128) get CUDA-graph / stream-
    # captured, so their compute kernels are not individually visible in the
    # trace (only optimizer/comm/elementwise appear). Fall back to the eager
    # cr=4 sample for those so the full-model projection isn't zeroed; flag it.
    ref = (
        "4"
        if "4" in per_cr and _fwd_total(per_cr["4"]) >= 1000
        else max(per_cr, key=lambda c: _fwd_total(per_cr[c]))
    )
    graphed = []
    for cr in list(per_cr):
        if cr != ref and _fwd_total(per_cr[cr]) < 1000:
            graphed.append(cr)
            per_cr[cr] = per_cr[ref]

    src = per_cr[ref]
    layers = {cr: {"attention": _lists(b["attention"]), "moe": _lists(b["moe"])} for cr, b in per_cr.items()}
    optimizer_us = round(sum(opt_us) / len(opt_us), 1) if opt_us else None

    return {
        "schema_version": 1,
        "model": model,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "traces": {cr: str(p) for cr, p in traces.items()},
            "graphed_crs_estimated_from_cr4": graphed,
            "note": (
                "cr in graphed_crs_estimated_from_cr4 were CUDA-graph/stream-captured "
                "(compute not visible in trace); their breakdown is copied from cr=4 as an "
                "estimate. Re-run those cr with graph capture disabled for exact numbers."
            ),
        },
        "capture": {
            "gpu": "MI355X",
            "seq_length": 4096,
            "micro_batch_size": 1,
            "tokens_per_microbatch": 4096,
            "ep": 8,
            "ga_for_capture": 2,
            "optimizer": "adam",
            "distributed_optimizer": True,
            "recompute": "off",
            "measured_iter_time_ms": None,
        },
        "model_config": {**cfg, "cr_layer_counts": cr_layer_counts(cfg["compress_ratios"])},
        "hardware": DEFAULT_HARDWARE,
        "layers": layers,
        "non_layer": {k: _lists(src[k]) for k in ("embedding", "output", "loss")},
        "optimizer": {
            "type": "adam",
            "measured_params": None,
            "time_us": optimizer_us,
            "bytes_per_param": 18,
            "class": "memory_bound",
            "note": "measured one-layer optimizer-step kernel time on this rank; the site scales by per-rank params",
        },
        "comm": {
            "ep_dispatch_us": None,
            "ep_combine_us": None,
            "note": "EP dispatch/combine are memory_bound rows inside moe; informational",
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build projection breakdown JSON from per-cr traces.")
    p.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    p.add_argument("--trace", action="append", default=[], metavar="cr=PATH")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    traces: dict[str, Path] = {}
    for spec in args.trace:
        if "=" not in spec:
            raise SystemExit(f"--trace must be cr=PATH, got: {spec}")
        cr, path = spec.split("=", 1)
        traces[cr.replace("cr", "")] = Path(path)
    if not traces:
        raise SystemExit("at least one --trace cr=PATH is required")
    for cr, path in traces.items():
        if not path.exists():
            raise SystemExit(f"trace not found for cr={cr}: {path}")

    doc = build(args.model, traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"[parse_trace] wrote {args.out} (model={args.model}, crs={sorted(traces)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
