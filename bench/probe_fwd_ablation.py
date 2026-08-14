"""An exact ledger of the FlyDSL KDA forward, stage by stage, at one geometry.

Round 2 attributed 440 µs of the forward's 672 µs gap to "the `[NB,C,D]` fp32
layout". That was read off a kernel-name profile, where several unrelated copies
share one Triton kernel name, so it is an attribution and not a measurement. This
script measures each stage on exactly the tensors the real assembly hands it, so
the prize for each candidate change is a number rather than an estimate, and the
work can be ordered by measured value.

    python bench/probe_fwd_ablation.py --shape prod_T4096

Prints the ledger, its sum against the measured whole, and for each line the
forward ratio against `fla` that removing it entirely would give.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.bench_kda_backends import SHAPES, make_inputs, timed  # noqa: E402

RESULTS = os.path.join(REPO, "bench", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="prod_T4096")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="ablation")
    args = ap.parse_args()

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        load_fla_kda_backend,
        load_flydsl_kda_backend,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import chunk as ch
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores,
        ut_inverse,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.prep import (
        chunk_prep,
    )
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.sweep import (
        _transposed_operand,
        fused_chunk_sweep,
    )

    b, t, h, k_dim, v_dim = SHAPES[args.shape]
    c = args.chunk
    q, kk, vv, g, beta = make_inputs(b, t, h, k_dim, v_dim, "cuda", torch.bfloat16)
    scale = k_dim**-0.5
    flydsl, fla = load_flydsl_kda_backend(), load_fla_kda_backend()

    led: Dict[str, float] = {}
    with torch.no_grad():
        led["WHOLE flydsl forward"] = timed(
            lambda: flydsl(q, kk, vv, g, beta), args.iters, args.warmup
        )
        led["WHOLE fla forward"] = timed(lambda: fla(q, kk, vv, g, beta), args.iters, args.warmup)

        # ---- the layout region, and the part of it q/k account for ----
        led["prepare_operands (q,k,v,cg,beta)"] = timed(
            lambda: ch._prepare_operands(q, kk, vv, g, beta, c), args.iters, args.warmup
        )
        qf, kf, vf, cg, betaf = ch._prepare_operands(q, kk, vv, g, beta, c)
        nb = qf.shape[0]

        def lay_one(x):
            return ch._lay_out(x, c, 0)

        led["  one [B,T,H,D] -> [NB,C,D] fp32 cast+transpose"] = timed(
            lambda: lay_one(q), args.iters, args.warmup
        )
        led["  cg = lay_out(g).cumsum"] = timed(
            lambda: ch._lay_out(g, c, 0).cumsum(dim=-2), args.iters, args.warmup
        )

        # ---- the kernels ----
        led["decay_scores kernel"] = timed(lambda: decay_scores(qf, kf, cg), args.iters, args.warmup)
        aqk, akk = decay_scores(qf, kf, cg)
        led["beta products (low, scale_ut)"] = timed(
            lambda: ch._scale_ut(ch._low_from_scores(akk, betaf), betaf), args.iters, args.warmup
        )
        low = ch._low_from_scores(akk, betaf)
        led["ut_inverse kernel"] = timed(lambda: ut_inverse(low), args.iters, args.warmup)
        ut = ch._scale_ut(ut_inverse(low), betaf)
        led["chunk_prep kernel"] = timed(
            lambda: chunk_prep(qf, kf, cg, torch.bfloat16, use_kernel=True), args.iters, args.warmup
        )
        qw, kgam, kg, dec = chunk_prep(qf, kf, cg, torch.bfloat16, use_kernel=True)
        led["W GEMM (ut @ kgam)"] = timed(lambda: ut @ kgam, args.iters, args.warmup)
        w = ut @ kgam
        led["  store_w (fp32 -> bf16 into qw[:, C:])"] = timed(
            lambda: ch._store_w(qw, w, c), args.iters, args.warmup
        )
        led["U GEMM (ut @ vf)"] = timed(lambda: ut @ vf, args.iters, args.warmup)
        u = ut @ vf
        led["KG transpose (_transposed_operand)"] = timed(
            lambda: _transposed_operand(kg, torch.bfloat16), args.iters, args.warmup
        )
        led["fused_chunk_sweep (kernel + output baddbmm)"] = timed(
            lambda: fused_chunk_sweep(
                qw, u, aqk, kg, dec, None, num_chunks=t // c, op_dtype=torch.bfloat16, scale=scale
            ),
            args.iters,
            args.warmup,
        )
        o, _ = fused_chunk_sweep(
            qw, u, aqk, kg, dec, None, num_chunks=t // c, op_dtype=torch.bfloat16, scale=scale
        )
        led["lay_back (output [NB,C,V] -> [B,T,H,V])"] = timed(
            lambda: ch._lay_back(o, b, h, t, torch.bfloat16), args.iters, args.warmup
        )

    whole = led["WHOLE flydsl forward"]
    ref = led["WHOLE fla forward"]
    parts = {n: v for n, v in led.items() if not n.startswith("WHOLE")}
    accounted = sum(v for n, v in parts.items() if not n.startswith("  "))

    width = max(len(n) for n in led)
    print(f"\n### {args.shape} bf16 — forward stage ledger (µs, median of {args.iters})\n")
    print(f"{'stage':{width}s} {'µs':>8s} {'% of whole':>11s} {'ratio if free':>14s}")
    print("-" * (width + 36))
    for n, val in led.items():
        if n.startswith("WHOLE"):
            print(f"{n:{width}s} {val:8.1f}")
            continue
        freed = whole - val
        print(f"{n:{width}s} {val:8.1f} {val / whole * 100:10.1f}% {ref / freed:13.2f}x")
    print("-" * (width + 36))
    print(f"{'sum of top-level stages':{width}s} {accounted:8.1f} {accounted / whole * 100:10.1f}%")
    print(f"{'flydsl / fla ratio today':{width}s} {'':8s} {'':11s} {ref / whole:13.2f}x")
    print(
        "\n(indented lines are a component of the line above them, so they are not "
        "summed; 'ratio if free' is the forward ratio vs fla with that stage's whole "
        "cost removed, which is an upper bound on what optimising it can give.)"
    )

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"fwd_{args.tag}_{args.shape}.json")
    with open(out, "w") as fh:
        json.dump(led, fh, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
