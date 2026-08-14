"""KDA backend microbenchmark: `eager` vs `fla` vs `flydsl`.

Measures forward, forward+backward and (by subtraction) backward for every KDA
backend at the geometries Kimi K3 actually trains at, checks each fused backend
against the eager reference evaluated in fp32, and reports peak allocator
residency.

    python bench/bench_kda_backends.py --shapes prod_T4096 --dtypes bf16
    python bench/bench_kda_backends.py --parity-only
    python bench/bench_kda_backends.py --tag baseline

Timing follows the convention every earlier pass used (`torch.cuda.Event`,
median of `--iters` after `--warmup`, forward measured under `no_grad`, backward
by subtracting the forward from a fwd+bwd), so the numbers are directly
comparable with `wp9_p*/results/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Dict, Optional, Tuple

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RESULTS = os.path.join(REPO, "bench", "results")

# B, T, H, K, V.  `prod_*` are the historical wp9 shapes, kept so this harness
# reproduces the earlier passes' numbers; `off8L_*` and `curve_*` are the two
# recipes in examples/megatron/configs/MI355X/ at their real per-micro-batch
# shape (KDA is full-width and has no GQA, so H = num_attention_heads and
# K = V = linear_key_head_dim = 128).
SHAPES: Dict[str, Tuple[int, int, int, int, int]] = {
    "unit_small": (2, 128, 4, 64, 64),
    "unit_mid": (2, 512, 8, 128, 128),
    "prod_T2048": (1, 2048, 96, 128, 128),
    "prod_T4096": (1, 4096, 96, 128, 128),
    "off8L_mbs2": (2, 7168, 96, 128, 128),
    "curve_mbs8": (8, 2048, 16, 128, 128),
    "curve_mbs16": (16, 2048, 16, 128, 128),
}

# Shapes the eager reference can be run at without the [B,H,NC,C,K] per-column
# intermediates blowing up; everything else is validated against `fla` instead.
PARITY_SHAPES = ("unit_small", "unit_mid", "parity_prod")
PARITY_EXTRA = {"parity_prod": (1, 1024, 96, 128, 128)}

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def gemm_flops(b: int, t: int, h: int, k: int, v: int, chunk: int = 64) -> float:
    """GEMM FLOPs of the chunkwise-parallel form, identical for every backend.

    Counted per ``(batch, head, chunk)`` and summed, so it is a fair relative
    measure across backends even though each one schedules the work differently:

        Aqk, Akk, W   3 x 2·C²·K      intra-chunk scores and the UT-transformed W
        U, Aqk@T      2 x 2·C²·V      the value-side intra-chunk products
        QΓ@S, W@S,
        KGᵀ@T         3 x 2·C·K·V     the three state-sized products

    The ``(I−L)^{-1}`` transform is left out: it is O(C³/3) on a 64x64 matrix,
    under 2 % of the above, and the backends compute it by different algorithms.
    """
    per_chunk = 3 * 2 * chunk * chunk * k + 2 * 2 * chunk * chunk * v + 3 * 2 * chunk * k * v
    return float(per_chunk * (t // chunk) * b * h)


def make_inputs(b, t, h, k, v, dev, dtype, seed=0, lb=-5.0):
    """The five KDA inputs, as the attention module hands them to a backend.

    ``q``/``k`` arrive L2-normalised (the module calls ``kda_l2norm`` itself),
    ``g`` is a bounded log-decay in ``[lb, 0]`` (``kda_gate``'s range) and
    ``beta`` is already sigmoid-activated.
    """
    gen = torch.Generator(device=dev).manual_seed(seed)

    def l2(x):
        return x / x.norm(dim=-1, keepdim=True)

    q = l2(torch.randn(b, t, h, k, generator=gen, device=dev, dtype=torch.float32))
    kk = l2(torch.randn(b, t, h, k, generator=gen, device=dev, dtype=torch.float32))
    vv = torch.randn(b, t, h, v, generator=gen, device=dev, dtype=torch.float32)
    z = torch.randn(b, t, h, k, generator=gen, device=dev, dtype=torch.float32) * 3.0
    g = lb * torch.sigmoid(z)
    beta = torch.sigmoid(torch.randn(b, t, h, generator=gen, device=dev, dtype=torch.float32))
    return [x.to(dtype).detach().requires_grad_(True) for x in (q, kk, vv, g, beta)]


def timed(fn, iters=20, warmup=5) -> float:
    """Median wall time of `fn` in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def count_launches(fn, iters=4, warmup=2) -> float:
    """Device kernel launches per call of `fn`.

    A first-class metric here, not a diagnostic: the whole remaining gap to
    `fla` is launch count and the intermediates that come with it, so every
    round has to report it next to the latency it is trying to move.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    n = sum(1 for ev in prof.events() if getattr(ev, "device_type", None) == DeviceType.CUDA)
    return n / iters


def bench_backend(backend, inputs, iters, warmup, do_backward=True) -> Dict[str, float]:
    q, k, v, g, beta = inputs
    out: Dict[str, float] = {}

    def fwd():
        with torch.no_grad():
            backend(q, k, v, g, beta)

    out["fwd_us"] = timed(fwd, iters, warmup)
    out["fwd_launches"] = count_launches(fwd)
    if not do_backward:
        return out

    o, _ = backend(q, k, v, g, beta)
    grad = torch.randn_like(o)

    # The backward's launches on their own, from a graph replayed with
    # `retain_graph`, so the forward's kernels are not counted twice.
    def bwd_only():
        for t in inputs:
            t.grad = None
        o.backward(grad, retain_graph=True)

    out["bwd_launches"] = count_launches(bwd_only)
    for t in inputs:
        t.grad = None
    del o

    def fwd_bwd():
        for t in inputs:
            t.grad = None
        o_, _ = backend(q, k, v, g, beta)
        o_.backward(grad)

    out["fwd_bwd_us"] = timed(fwd_bwd, iters, warmup)
    out["bwd_us"] = out["fwd_bwd_us"] - out["fwd_us"]

    # peak residency of one fwd+bwd, measured on its own so the timing loop's
    # cached blocks do not count
    for t in inputs:
        t.grad = None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fwd_bwd()
    torch.cuda.synchronize()
    out["peak_mem_gb"] = (torch.cuda.max_memory_allocated() - base) / 2**30
    for t in inputs:
        t.grad = None
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# numerical parity
# ---------------------------------------------------------------------------


def _err(got: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """Max-abs, scale-relative max and relative RMS against an fp32 oracle.

    ``max_rel`` is normalised by the *tensor's* scale rather than elementwise:
    a bf16 output has entries many orders of magnitude apart and an elementwise
    ratio on the smallest of them says nothing about whether the kernel is
    right. ``rel_rms`` is the norm-wise figure the earlier passes quoted.
    """
    got = got.detach().float()
    ref = ref.detach().float()
    d = (got - ref).abs()
    scale = ref.abs().amax().clamp_min(1e-30)
    return {
        "max_abs": d.amax().item(),
        "max_rel": (d.amax() / scale).item(),
        "rel_rms": (d.norm() / ref.norm().clamp_min(1e-30)).item(),
    }


def parity(backend, inputs, oracle_fn) -> Dict[str, Dict[str, float]]:
    """Compare `backend` against the eager reference run in fp32 on the same data.

    Both sides get the identical (bf16-rounded) inputs, so the difference is the
    kernel's own arithmetic and not a different starting point. The oracle runs
    in fp32 because the reference is what defines the correct answer, not the
    dtype the kernel happens to use.
    """
    ref_inputs = [t.detach().float().requires_grad_(True) for t in inputs]
    o_ref, _ = oracle_fn(*ref_inputs)
    grad = torch.randn_like(o_ref)
    o_ref.backward(grad)

    got_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
    o_got, _ = backend(*got_inputs)
    o_got.backward(grad.to(o_got.dtype))

    report = {"o": _err(o_got, o_ref)}
    for name, a, b in zip(
        ("dq", "dk", "dv", "dg", "dbeta"),
        (t.grad for t in got_inputs),
        (t.grad for t in ref_inputs),
    ):
        report[name] = _err(a, b) if a is not None else {"missing": True}
    return report


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--shapes", default="", help="comma list; default = every shape")
    ap.add_argument("--dtypes", default="bf16,fp32")
    ap.add_argument("--backends", default="fla,flydsl")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--eager-shapes", default="unit_small,unit_mid", help="where to also time eager")
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--skip-parity", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"bench_{args.tag}.json")

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        eager_chunk_kda,
        load_fla_kda_backend,
        load_flydsl_kda_backend,
    )

    loaders = {
        "eager": lambda: eager_chunk_kda,
        "fla": load_fla_kda_backend,
        "flydsl": load_flydsl_kda_backend,
    }
    wanted = [b for b in args.backends.split(",") if b]
    backends = {}
    for name in wanted:
        try:
            backends[name] = loaders[name]()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            print(f"[warn] backend {name} unavailable: {exc}", flush=True)

    import fla

    report: Dict[str, object] = {
        "device": torch.cuda.get_device_properties(0).name,
        "arch": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "fla": fla.__version__,
        "iters": args.iters,
        "tag": args.tag,
        "backends": list(backends),
    }

    def flush():
        with open(out_path, "w") as fh:
            json.dump(report, fh, indent=2)

    dtypes = [d for d in args.dtypes.split(",") if d]
    eager_shapes = set(s for s in args.eager_shapes.split(",") if s)

    # ---- parity ------------------------------------------------------------
    if not args.skip_parity:
        shapes = dict(SHAPES)
        shapes.update(PARITY_EXTRA)
        for name in PARITY_SHAPES:
            if name not in shapes:
                continue
            b, t, h, k, v = shapes[name]
            for dt in dtypes:
                inputs = make_inputs(b, t, h, k, v, "cuda", DTYPES[dt])
                for bname, backend in backends.items():
                    if bname == "eager":
                        continue
                    key = f"parity_{name}_{dt}_{bname}"
                    try:
                        report[key] = parity(backend, inputs, eager_chunk_kda)
                    except Exception:
                        report[key] = {"error": traceback.format_exc().splitlines()[-1]}
                    print(f"[{key}] {json.dumps(report[key])}", flush=True)
                    flush()
                del inputs
                torch.cuda.empty_cache()
    if args.parity_only:
        print("\nwrote", out_path)
        return

    # ---- speed -------------------------------------------------------------
    want = set(s for s in args.shapes.split(",") if s) or set(SHAPES)
    for name, (b, t, h, k, v) in SHAPES.items():
        if name not in want:
            continue
        for dt in dtypes:
            tag = f"{name}_{dt}"
            entry: Dict[str, object] = {
                "shape": [b, t, h, k, v],
                "gflop_fwd": gemm_flops(b, t, h, k, v) / 1e9,
                "tokens": b * t,
            }
            report[tag] = entry
            inputs = make_inputs(b, t, h, k, v, "cuda", DTYPES[dt])
            for bname, backend in backends.items():
                try:
                    entry[bname] = bench_backend(backend, inputs, args.iters, args.warmup)
                except Exception:
                    entry[bname] = {"error": traceback.format_exc().splitlines()[-1]}
                    torch.cuda.empty_cache()
            if name in eager_shapes and "eager" not in backends:
                try:
                    entry["eager"] = bench_backend(
                        eager_chunk_kda, inputs, max(args.iters // 4, 3), 1
                    )
                except Exception:
                    entry["eager"] = {"error": traceback.format_exc().splitlines()[-1]}
                    torch.cuda.empty_cache()
            del inputs
            torch.cuda.empty_cache()
            print(f"[{tag}] {json.dumps(entry)}", flush=True)
            flush()

    print(table(report, want, dtypes))
    flush()
    print("wrote", out_path)


def table(report, want, dtypes) -> str:
    lines = [
        "",
        "### forward / backward, microseconds (median), ratio = fla / flydsl",
        "",
        "| shape | dtype | fly fwd | fla fwd | fwd | fly bwd | fla bwd | bwd | fly f+b | fla f+b | f+b "
        "| launches fly f/b | launches fla f/b | fly GB | fla GB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in SHAPES:
        if name not in want:
            continue
        for dt in dtypes:
            e = report.get(f"{name}_{dt}")
            if not e:
                continue
            f_, a_ = e.get("flydsl", {}), e.get("fla", {})
            if "fwd_us" not in f_ or "fwd_us" not in a_:
                lines.append(f"| {name} | {dt} | " + " | ".join(["(err)"] * 14) + " |")
                continue

            def r(x, y):
                return f"{y / x:.2f}x" if x and y else "-"

            def launches(d):
                return f"{d.get('fwd_launches', 0):.0f}/{d.get('bwd_launches', 0):.0f}"

            lines.append(
                f"| {name} | {dt} | {f_['fwd_us']:.1f} | {a_['fwd_us']:.1f} | "
                f"{r(f_['fwd_us'], a_['fwd_us'])} | {f_.get('bwd_us', 0):.1f} | "
                f"{a_.get('bwd_us', 0):.1f} | {r(f_.get('bwd_us'), a_.get('bwd_us'))} | "
                f"{f_.get('fwd_bwd_us', 0):.1f} | {a_.get('fwd_bwd_us', 0):.1f} | "
                f"{r(f_.get('fwd_bwd_us'), a_.get('fwd_bwd_us'))} | "
                f"{launches(f_)} | {launches(a_)} | "
                f"{f_.get('peak_mem_gb', 0):.2f} | {a_.get('peak_mem_gb', 0):.2f} |"
            )
    lines += ["", "(ratio > 1.00x means FlyDSL is faster than fla)"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
