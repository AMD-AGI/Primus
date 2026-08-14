"""Is the sweep's chunk loop concurrency-bound? Vary the workgroup count only.

`bench/probe_sweep_bound.py` falsified the LDS hypothesis (deleting 7 of the 8
b-fragment reads bought 0.4 %) and priced the A operand's global traffic at 111 µs
of 501. MFMA issue accounts for ~8 µs and the stores and `y` loads for ~115 µs at
peak bandwidth, so **~265 µs of the 501 is unaccounted for** — which points at
latency on the 64-step serial chain with only three workgroups per CU.

That is testable without touching the kernel: hold the total chunk-step count
roughly fixed and vary how many workgroups it is spread over. The kernel's
workgroup count is `nbh * (V/block_v)` and its chain length is `nc`, both runtime
arguments, so the same compiled kernel can be run at 768 workgroups x 64 steps and
at 6144 x 8. If time per chunk-step falls as the workgroup count rises, the loop
is concurrency-bound and the production geometry (`B*H = 96`) simply does not have
enough of it — which is a property of the shape, not of the kernel.

    python bench/probe_sweep_conc.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.bench_kda_backends import timed  # noqa: E402

RESULTS = os.path.join(REPO, "bench", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-dim", type=int, default=128)
    ap.add_argument("--v-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--block-v", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
        kda_state_sweep_kernel as mod,
    )

    c, kd, vd, bv = args.chunk, args.k_dim, args.v_dim, args.block_v
    nvb = vd // bv
    dev = "cuda"
    launch = mod.build_kda_state_sweep(
        chunk_size=c,
        k_dim=kd,
        v_dim=vd,
        block_v=bv,
        mode="mfma",
        emit_rq=True,
        emit_states=False,
        has_e=False,
        sgn_t=-1.0,
        sgn_x=1.0,
        reverse=False,
    )

    # (nbh, nc) pairs whose product is fixed, so the arithmetic is constant and
    # only the shape of the parallelism changes. 96 x 64 is the production one.
    cases = [(96, 64), (192, 32), (384, 16), (768, 8), (1536, 4), (6144, 1)]
    rows = []
    for nbh, nc in cases:
        nb = nbh * nc
        amat = torch.randn(nb, 2 * c, kd, dtype=torch.bfloat16, device=dev) * 0.05
        yc = torch.randn(nb, c, vd, dtype=torch.float32, device=dev)
        xt = torch.randn(nb, kd, c, dtype=torch.bfloat16, device=dev) * 0.05
        dec = torch.rand(nb, kd, dtype=torch.float32, device=dev)
        s0 = torch.zeros(nbh, kd, vd, dtype=torch.float32, device=dev)
        dummy = torch.empty(1, dtype=torch.float32, device=dev)
        rq = torch.empty(nb, c, vd, dtype=torch.float32, device=dev)
        t_all = torch.empty_like(rq)
        s_final = torch.empty(nbh, kd, vd, dtype=torch.float32, device=dev)
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
        us = timed(fn, args.iters, args.warmup)
        wg = nbh * nvb
        steps = nb * nvb
        rows.append(
            {
                "nbh": nbh,
                "nc": nc,
                "workgroups": wg,
                "wg_per_cu": wg / 256.0,
                "us": us,
                "ns_per_chunk_step": us * 1e3 / steps,
            }
        )
        print(
            f"  nbh {nbh:5d} x nc {nc:3d} -> {wg:6d} workgroups "
            f"({wg / 256.0:5.1f}/CU): {us:7.1f} µs, {us * 1e3 / steps:6.2f} ns/chunk-step",
            flush=True,
        )
        del amat, yc, xt, dec, s0, rq, t_all, s_final
        torch.cuda.empty_cache()

    base = rows[0]
    best = min(rows, key=lambda r: r["ns_per_chunk_step"])
    print(
        f"\nproduction geometry (96 x 64) is {base['ns_per_chunk_step']:.2f} ns/chunk-step; "
        f"best is {best['ns_per_chunk_step']:.2f} at {best['workgroups']} workgroups "
        f"-> {base['ns_per_chunk_step'] / best['ns_per_chunk_step']:.2f}x of headroom that "
        "exists only if the parallelism does"
    )

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "sweep_concurrency.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
