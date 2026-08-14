"""Minimal driver: run one backend's forward N times, for rocprofv3 to profile.

    rocprofv3 --pmc <counters> -- python bench/probe_counters_driver.py --backend fla

Kept separate from the benchmark harness so the profiled region contains nothing
but the kernel under test.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.bench_kda_backends import SHAPES, make_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fla", choices=("fla", "flydsl"))
    ap.add_argument("--shape", default="prod_T4096")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels import (
        load_fla_kda_backend,
        load_flydsl_kda_backend,
    )

    backend = {"fla": load_fla_kda_backend, "flydsl": load_flydsl_kda_backend}[args.backend]()
    b, t, h, k, v = SHAPES[args.shape]
    q, kk, vv, g, beta = make_inputs(b, t, h, k, v, "cuda", torch.bfloat16)

    with torch.no_grad():
        for _ in range(2):  # warm up the JIT / autotune outside the counted region
            backend(q, kk, vv, g, beta)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            backend(q, kk, vv, g, beta)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
