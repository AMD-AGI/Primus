"""What is the sweep's chunk loop actually bound by, and is 300 µs reachable?

The sweep region is 43 % of the FlyDSL KDA forward and the only stage whose
removal would put the forward past `fla` ("ratio if free" 1.12×,
`bench/probe_fwd_ablation.py`). Rounds 2–4 closed every candidate that had been
proposed for it: `block_v`/`waves_per_eu` are at their optimum, prefetching the
exposed HBM latency took 65 µs, and splitting it the way `fla` does **costs**
120 µs (`bench/probe_sweep_split.py`). Round 4 left one hypothesis: that the loop
is bound by the **LDS traffic of the MFMA B fragment**, on the evidence that
removing half the MFMAs (`emit_rq=False`) bought only 14 %.

This script tests that hypothesis by deletion rather than argument, using the
measurement-only build variants in `kda_state_sweep_kernel.PROBES`. **Every
variant computes the wrong answer** — that is the point; each removes exactly one
cost so the delta attributes it.

    full     the real kernel
    lds1     the B fragment from one LDS read instead of LANE_K strided ones
             -> the delta is the b-fragment LDS traffic, and an upper bound on
                what any LDS-layout change (transpose, bf16 shadow) can give
    areuse   every ks step reads the same A fragment
             -> the delta is the A operand's global traffic

and it prices the two arithmetic-only candidates that remain:

    C=128    halve the serial chain; intra-chunk scores go as T*C and the
             triangular solve as T*C^2, so this is decidable by measurement of
             the *whole* forward, not of the sweep alone
    scan     a log-depth parallel scan over the affine transition, costed in
             FLOPs and HBM bytes below

    python bench/probe_sweep_bound.py
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


def bound_the_loop(shape, chunk, iters, warmup) -> Dict[str, float]:
    """Time the real kernel and each deletion variant on identical operands."""
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
        kda_state_sweep_kernel as mod,
    )

    b, t, h, kd, vd = SHAPES[shape]
    c = chunk
    nc, nb, nbh = t // c, b * h * (t // c), b * h
    dev = "cuda"
    amat = torch.randn(nb, 2 * c, kd, dtype=torch.bfloat16, device=dev) * 0.05
    yc = torch.randn(nb, c, vd, dtype=torch.float32, device=dev)
    xt = torch.randn(nb, kd, c, dtype=torch.bfloat16, device=dev) * 0.05
    dec = torch.rand(nb, kd, dtype=torch.float32, device=dev)
    s0 = torch.zeros(nbh, kd, vd, dtype=torch.float32, device=dev)
    dummy = torch.empty(1, dtype=torch.float32, device=dev)
    rq = torch.empty(nb, c, vd, dtype=torch.float32, device=dev)
    t_all = torch.empty_like(rq)
    s_final = torch.empty(nbh, kd, vd, dtype=torch.float32, device=dev)

    out: Dict[str, float] = {}
    for probe in mod.PROBES:
        launch = mod.build_kda_state_sweep(
            chunk_size=c,
            k_dim=kd,
            v_dim=vd,
            block_v=16,
            mode="mfma",
            emit_rq=True,
            emit_states=False,
            has_e=False,
            sgn_t=-1.0,
            sgn_x=1.0,
            reverse=False,
            probe=probe,
        )
        fn = lambda: launch(  # noqa: E731
            amat.reshape(-1),
            yc.reshape(-1),
            xt.reshape(-1),
            dec.reshape(-1),
            dummy,
            s0.reshape(-1),
            rq.reshape(-1),
            t_all.reshape(-1),
            dummy,
            s_final.reshape(-1),
            nbh,
            nc,
        )
        fn()
        torch.cuda.synchronize()
        out[probe] = timed(fn, iters, warmup)
        print(f"  sweep probe {probe:8s} {out[probe]:7.1f} µs", flush=True)
    return out


def price_chunk_size(shape, iters, warmup) -> Dict[str, float]:
    """The whole forward at C = 64 and C = 128, both backends.

    A longer chunk halves the serial chain but the intra-chunk work grows: the
    score matrices are ``[C, C]`` per chunk over ``T/C`` chunks, so they go as
    ``T·C``, and the triangular solve as ``T·C²``. Only the whole forward decides
    it, so that is what is timed.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        load_flydsl_kda_backend,
    )

    b, t, h, kd, vd = SHAPES[shape]
    flydsl = load_flydsl_kda_backend()
    q, k, v, g, beta = make_inputs(b, t, h, kd, vd, "cuda", torch.bfloat16)
    out: Dict[str, float] = {}
    with torch.no_grad():
        for c in (64, 128):
            try:
                flydsl(q, k, v, g, beta, chunk_size=c)
                torch.cuda.synchronize()
                out[f"forward C={c}"] = timed(
                    lambda: flydsl(q, k, v, g, beta, chunk_size=c), iters, warmup
                )
            except Exception as exc:  # noqa: BLE001
                out[f"forward C={c}"] = float("nan")
                print(f"  C={c}: {exc!r:.140}", flush=True)
            print(f"  forward C={c:3d} {out[f'forward C={c}']:7.1f} µs", flush=True)
    return out


def price_parallel_scan(shape, chunk) -> Dict[str, float]:
    """FLOPs and HBM bytes for a log-depth scan over the affine transition.

    Substituting ``T_n = U_n − W_n S_n`` into the recurrence gives

        S_{n+1} = (Diag(dec_n) − KG_nᵀ W_n) S_n + KG_nᵀ U_n  =  A_n S_n + B_n

    so the transition is an **affine map with a full ``[K, K]`` matrix**, not a
    diagonal one. Affine maps compose associatively, which is what would allow a
    scan — but each composition is ``A' = A₂A₁`` (``2K³``) and
    ``B' = A₂B₁ + B₂`` (``2K²V``), against a direct step's three ``C``-sized
    products. A Blelloch scan over ``NC`` elements does ~``2·NC`` compositions.
    """
    b, t, h, kd, vd = SHAPES[shape]
    c, nc, nbh = chunk, t // chunk, b * h
    direct_per_chunk = 2 * (2 * c * kd * vd) + 2 * kd * c * vd  # Rq, W@S, KGᵀ@T
    direct = direct_per_chunk * nc * nbh
    compose = 2 * kd**3 + 2 * kd**2 * vd
    scan = 2 * nc * compose * nbh
    # A_n has to be materialised to be composed at all.
    a_bytes = nc * nbh * kd * kd * 4
    return {
        "direct GFLOP": direct / 1e9,
        "scan GFLOP": scan / 1e9,
        "scan / direct FLOP ratio": scan / direct,
        "new [K,K] A-matrix residency GB": a_bytes / 2**30,
        "scan µs at an optimistic 300 TFLOP/s": scan / 300e12 * 1e6,
        "scan µs floor from A-matrix traffic alone at 5.3 TB/s": a_bytes * 3 / 5.3e12 * 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="prod_T4096")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    print("\n--- 1. what the chunk loop is bound by (deletion variants) ---")
    loop = bound_the_loop(args.shape, args.chunk, args.iters, args.warmup)
    print("\n--- 2. candidate: longer chunk (whole forward) ---")
    cs = price_chunk_size(args.shape, args.iters, args.warmup)
    print("\n--- 3. candidate: log-depth parallel scan (arithmetic only) ---")
    scan = price_parallel_scan(args.shape, args.chunk)
    for k, val in scan.items():
        print(f"  {k:52s} {val:12.2f}")

    full = loop.get("full", float("nan"))
    print("\n### verdict inputs")
    for name, label in (
        ("lds1", "b-fragment LDS traffic"),
        ("areuse", "A operand global traffic"),
    ):
        d = full - loop.get(name, float("nan"))
        print(
            f"  {label:26s} = {d:6.1f} µs of {full:.0f}  ({d / full * 100:4.1f} %) "
            f"-> deleting it entirely leaves {loop.get(name, float('nan')):.0f} µs"
        )
    print(
        "\n  target: the sweep region must lose ~300 µs for the forward to pass fla."
    )

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"sweep_bound_{args.shape}.json")
    with open(out, "w") as fh:
        json.dump({"loop": loop, "chunk_size": cs, "scan": scan}, fh, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
