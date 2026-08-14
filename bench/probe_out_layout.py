"""Is the fused output's write pattern what costs? Test it independently of `block_v`.

The last unexplained fact in `bench/FINDINGS.md` §5e: at identical workload we are
1.85x slower than `fla`, and the only surviving suspicion is the store pattern — at
`block_v = 16` a workgroup owns 16 consecutive floats (64 B) of a 512 B row stride,
so every store instruction writes half a cache line, and two *different*
workgroups have to fill the other half. `block_v = 64` would write 256 B runs but
measures worse overall, which is why the suspicion failed its own test.

Those two things are separable. Writing the output **V-block-major**
(`[NB, NVB, C, BV]` instead of `[NB, C, V]`) makes a workgroup's whole `C x BV`
tile contiguous while leaving `block_v`, the workgroup count and everything else
alone. The values are identical, only permuted — and `_lay_back` already makes a
full pass over this tensor, so it could absorb the permutation for nothing.

This times the two layouts on the real fused kernel and checks the permuted output
against the row-major one, so a win here is directly bankable.

    python bench/probe_out_layout.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.bench_kda_backends import SHAPES, timed  # noqa: E402

RESULTS = os.path.join(REPO, "bench", "results")


def main():
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
        kda_state_sweep_kernel as mod,
    )

    b, t, h, kd, vd = SHAPES["prod_T4096"]
    c, bv = 64, 16
    nc, nb, nbh, nvb = t // c, b * h * (t // c), b * h, vd // bv
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)

    def rnd(*shape, dtype=torch.float32, scale=1.0):
        return torch.randn(*shape, generator=g, device=dev, dtype=dtype) * scale

    amat = rnd(nb, 2 * c, kd, dtype=torch.bfloat16, scale=0.05)
    yc = rnd(nb, c, vd)
    xt = rnd(nb, kd, c, dtype=torch.bfloat16, scale=0.05)
    dec = torch.rand(nb, kd, generator=g, device=dev)
    aqk = rnd(nb, c, c, scale=0.1)
    s0 = torch.zeros(nbh, kd, vd, device=dev)
    dummy = torch.empty(1, device=dev)
    t_all = torch.empty(nb, c, vd, device=dev)
    s_final = torch.empty(nbh, kd, vd, device=dev)

    out = {}
    res = {}
    for probe in ("full", "ocontig"):
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
            probe=probe,
            fuse_out=True,
            emit_t=False,
            scale=kd**-0.5,
        )
        o = torch.zeros(nb, c, vd, device=dev)
        fn = lambda: launch(  # noqa: E731
            amat.reshape(-1),
            yc.reshape(-1),
            xt.reshape(-1),
            dec.reshape(-1),
            dummy,
            s0.reshape(-1),
            dummy,
            dummy,
            dummy,
            s_final.reshape(-1),
            aqk.reshape(-1),
            o.reshape(-1),
            nbh,
            nc,
        )
        fn()
        torch.cuda.synchronize()
        res[probe] = o.clone()
        out[probe] = timed(fn, 20, 5)
        print(f"  output layout {probe:8s} {out[probe]:7.1f} µs", flush=True)

    # `ocontig` wrote [NB, NVB, C, BV]; permute it back and compare bit-for-bit.
    perm = res["ocontig"].view(nb, nvb, c, bv).permute(0, 2, 1, 3).reshape(nb, c, vd)
    err = (perm - res["full"]).abs().amax().item()
    out["max_abs_after_unpermuting"] = err
    d = out["full"] - out["ocontig"]
    print(
        f"\n  contiguous output saves {d:.1f} µs of {out['full']:.1f} "
        f"({d / out['full'] * 100:.1f} %); permuted result differs by {err:.3e}"
    )
    print(
        "  (a difference of exactly 0 means the two layouts are the same computation, "
        "so any saving is bankable by teaching _lay_back the permutation)"
    )

    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "out_layout.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", p)


if __name__ == "__main__":
    main()
