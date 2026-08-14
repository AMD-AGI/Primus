"""Where the KDA backends' time goes, kernel by kernel.

Profiles the forward and the backward **separately** — the backward by
replaying a graph built outside the profiled region with ``retain_graph=True``,
so its kernel list is not contaminated by the forward's — and prints a
per-kernel table with launch counts. That is what turns "flydsl is 1.8x fla"
into "this stage is the 1.8x".

    python bench/profile_kda.py --shapes prod_T4096 --dtypes bf16

Also reports the host-side enqueue cost, because a stage that is only dispatch
needs a different fix from one that is device work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Tuple

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.bench_kda_backends import DTYPES, SHAPES, make_inputs, timed  # noqa: E402

RESULTS = os.path.join(REPO, "bench", "results")


def _device_events(prof) -> List[Tuple[str, float, float]]:
    """Every on-device kernel event as ``(name, start_us, end_us)``."""
    out = []
    try:
        from torch.autograd import DeviceType

        for ev in prof.events():
            if getattr(ev, "device_type", None) == DeviceType.CUDA:
                tr = ev.time_range
                out.append((ev.key, tr.start, tr.end))
    except Exception:
        pass
    if out:
        return out
    for ev in prof.events():
        for k in getattr(ev, "kernels", []) or []:
            out.append((k.name, 0.0, float(k.duration)))
    return out


def _summarise(evs, iters, top_n) -> Dict[str, object]:
    if not evs:
        return {"n_kernels": 0, "busy_us": 0.0, "span_us": 0.0, "top": []}
    spans = sorted((s, e) for _, s, e in evs)
    busy, cur_s, cur_e = 0.0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    per: Dict[str, List[float]] = {}
    for name, s, e in evs:
        d = per.setdefault(name, [0.0, 0.0])
        d[0] += 1
        d[1] += e - s
    top = sorted(per.items(), key=lambda kv: -kv[1][1])[:top_n]
    return {
        "n_kernels": len(evs) / iters,
        "busy_us": busy / iters,
        "span_us": (spans[-1][1] - spans[0][0]) / iters,
        "top": [{"name": n[:78], "count": c / iters, "us": t / iters} for n, (c, t) in top],
    }


def _by_aten_op(prof, iters, top_n) -> List[Dict[str, object]]:
    """Device time attributed to the **ATen op** that launched it.

    The kernel-name view says `elementwise_kernel_manual_unroll` nine times; this
    one says which nine calls those are, which is the difference between knowing
    there is glue left and knowing where to delete it.
    """
    rows = []
    for ev in prof.key_averages():
        dev = getattr(ev, "device_time_total", 0.0) or 0.0
        if dev <= 0 or getattr(ev, "key", "").startswith(("cudaLaunch", "hipLaunch")):
            continue
        rows.append({"op": ev.key[:60], "count": ev.count / iters, "us": dev / iters})
    rows.sort(key=lambda r: -r["us"])
    # key_averages nests: a top-level op's device time includes its children, so
    # keep only leaf-ish ATen ops (those with no '::' children is not knowable
    # here, so report both and let the reader see the nesting).
    return rows[:top_n]


def profile_fn(fn, iters=8, warmup=3, top_n=25) -> Dict[str, object]:
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    out = _summarise(_device_events(prof), iters, top_n)
    try:
        out["by_op"] = _by_aten_op(prof, iters, top_n)
    except Exception:
        out["by_op"] = []
    return out


def host_us(fn, iters=20, warmup=5) -> float:
    """CPU time to enqueue one call, with no synchronisation at all."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return (t1 - t0) / iters * 1e6


def analyse(backend, inputs, iters, warmup, top_n) -> Dict[str, object]:
    q, k, v, g, beta = inputs
    out: Dict[str, object] = {}

    def fwd():
        with torch.no_grad():
            backend(q, k, v, g, beta)

    out["fwd_evt_us"] = timed(fwd, iters, warmup)
    out["fwd_host_us"] = host_us(fwd, iters, warmup)
    out["fwd"] = profile_fn(fwd, max(iters // 2, 3), 2, top_n)

    # The backward on its own: one graph, replayed. `retain_graph` keeps the
    # saved tensors alive so the same graph can be walked `iters` times, which
    # is what isolates the backward's kernels from the forward's.
    o, _ = backend(q, k, v, g, beta)
    grad = torch.randn_like(o)

    def bwd():
        for t in inputs:
            t.grad = None
        o.backward(grad, retain_graph=True)

    out["bwd_evt_us"] = timed(bwd, iters, warmup)
    out["bwd_host_us"] = host_us(bwd, iters, warmup)
    out["bwd"] = profile_fn(bwd, max(iters // 2, 3), 2, top_n)
    for t in inputs:
        t.grad = None
    del o, grad
    torch.cuda.empty_cache()
    return out


def render(name: str, sec: Dict[str, object]) -> str:
    lines = [
        f"\n#### {name}: {sec['n_kernels']:.0f} launches, "
        f"{sec['busy_us']:.0f} µs busy, {sec['span_us']:.0f} µs span",
        "",
        "| kernel | n | µs | share |",
        "|---|---|---|---|",
    ]
    total = max(sum(t["us"] for t in sec["top"]), 1e-9)
    for t in sec["top"]:
        lines.append(f"| `{t['name']}` | {t['count']:.1f} | {t['us']:.1f} | {t['us'] / total * 100:.0f}% |")
    if sec.get("by_op"):
        lines += ["", "| launching op | n | µs |", "|---|---|---|"]
        for t in sec["by_op"]:
            lines.append(f"| `{t['op']}` | {t['count']:.1f} | {t['us']:.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="profile")
    ap.add_argument("--shapes", default="prod_T4096")
    ap.add_argument("--dtypes", default="bf16")
    ap.add_argument("--backends", default="flydsl,fla")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"profile_{args.tag}.json")

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        load_fla_kda_backend,
        load_flydsl_kda_backend,
    )

    loaders = {"fla": load_fla_kda_backend, "flydsl": load_flydsl_kda_backend}
    backends = {n: loaders[n]() for n in args.backends.split(",") if n}

    report: Dict[str, object] = {"tag": args.tag, "torch": torch.__version__}
    text = []
    for name in args.shapes.split(","):
        b, t, h, k, v = SHAPES[name]
        for dt in args.dtypes.split(","):
            inputs = make_inputs(b, t, h, k, v, "cuda", DTYPES[dt])
            for bname, backend in backends.items():
                key = f"{name}_{dt}_{bname}"
                try:
                    report[key] = analyse(backend, inputs, args.iters, args.warmup, args.top)
                except Exception:
                    report[key] = {"error": traceback.format_exc()}
                    print(report[key]["error"], flush=True)
                    continue
                e = report[key]
                text.append(
                    f"\n### {key}  fwd {e['fwd_evt_us']:.0f} µs "
                    f"(host {e['fwd_host_us']:.0f}) | bwd {e['bwd_evt_us']:.0f} µs "
                    f"(host {e['bwd_host_us']:.0f})"
                )
                text.append(render("forward", e["fwd"]))
                text.append(render("backward", e["bwd"]))
                with open(out_path, "w") as fh:
                    json.dump(report, fh, indent=2)
            del inputs
            torch.cuda.empty_cache()
    print("\n".join(text))
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
