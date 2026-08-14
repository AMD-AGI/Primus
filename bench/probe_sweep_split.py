"""Would splitting the sweep the way `fla` does actually pay? Measure it first.

Round 3 costed "split the sweep into a sequential state-history kernel plus a
fully parallel output kernel" at ~250 µs. That estimate came from comparing our
845 µs stage against `fla`'s 234 µs `..._fwd_kernel_h` **plus** its 193 µs
`chunk_gla_fwd_kernel_o`, which is a different decomposition, and the estimate
assumed the sequential half would land near `fla`'s.

The kernel already has the two build flags the split needs — `emit_rq` and
`emit_states` — so the sequential half can be measured *before* any output kernel
is written:

  A  emit_rq=True,  emit_states=False   what the forward runs today
  B  emit_rq=False, emit_states=True    the split's sequential half: drop Rq from
                                        the loop, write the [NB,K,V] history
  C  emit_rq=False, emit_states=False   B without the history write, i.e. the
                                        floor the split could ever reach

and the output half it would then owe is `Rq = QG @ S` over all chunks at once,
which is the price the split has to pay back. If `A < B + (QG @ S)` the split
loses, and it loses before a line of it is written.

    python bench/probe_sweep_split.py
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

from bench.bench_kda_backends import SHAPES, timed  # noqa: E402

RESULTS = os.path.join(REPO, "bench", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="prod_T4096")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--block-v", type=int, default=0, help="0 = sweep {16,32,64}")
    args = ap.parse_args()

    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
        kda_state_sweep_kernel as mod,
    )

    b, t, h, kd, vd = SHAPES[args.shape]
    c = args.chunk
    nc = t // c
    nb = b * h * nc
    nbh = b * h
    dev = "cuda"

    # Operands as `_FusedSweep` hands them over: only the two MFMA A operands are
    # in the operand dtype, everything else fp32.
    amat = torch.randn(nb, 2 * c, kd, dtype=torch.bfloat16, device=dev) * 0.05
    yc = torch.randn(nb, c, vd, dtype=torch.float32, device=dev)
    xt = torch.randn(nb, kd, c, dtype=torch.bfloat16, device=dev) * 0.05
    # `dec = exp(chunk_total)` is in (0, 1]; random values in [0,1) keep the
    # recurrence bounded, which a raw randn would not.
    dec = torch.rand(nb, kd, dtype=torch.float32, device=dev)
    s0 = torch.zeros(nbh, kd, vd, dtype=torch.float32, device=dev)
    dummy = torch.empty(1, dtype=torch.float32, device=dev)
    rq = torch.empty(nb, c, vd, dtype=torch.float32, device=dev)
    t_all = torch.empty_like(rq)
    states = torch.empty(nb, kd, vd, dtype=torch.float32, device=dev)
    s_final = torch.empty(nbh, kd, vd, dtype=torch.float32, device=dev)

    def run_cfg(block_v, emit_rq, emit_states):
        launch = mod.build_kda_state_sweep(
            chunk_size=c,
            k_dim=kd,
            v_dim=vd,
            block_v=block_v,
            mode="mfma",
            emit_rq=emit_rq,
            emit_states=emit_states,
            has_e=False,
            sgn_t=-1.0,
            sgn_x=1.0,
            reverse=False,
        )
        out_rq = rq if emit_rq else dummy
        out_st = states if emit_states else dummy
        return lambda: launch(
            amat.reshape(-1),
            yc.reshape(-1),
            xt.reshape(-1),
            dec.reshape(-1),
            dummy,
            s0.reshape(-1),
            out_rq.reshape(-1),
            t_all.reshape(-1),
            out_st.reshape(-1),
            s_final.reshape(-1),
            nbh,
            nc,
        )

    led: Dict[str, float] = {}
    bvs = (16, 32, 64) if args.block_v == 0 else (args.block_v,)
    with torch.no_grad():
        for bv in bvs:
            for name, (erq, est) in (
                ("A today (rq=1, states=0)", (True, False)),
                ("B split seq (rq=0, states=1)", (False, True)),
                ("C floor  (rq=0, states=0)", (False, False)),
            ):
                key = f"bv{bv:<3d} {name}"
                try:
                    fn = run_cfg(bv, erq, est)
                    fn()
                    torch.cuda.synchronize()
                    led[key] = timed(fn, args.iters, args.warmup)
                except Exception as exc:  # noqa: BLE001
                    led[key] = float("nan")
                    print(f"  {key}: {exc!r:.120}", flush=True)
                print(f"  {key}: {led[key]:.1f} µs", flush=True)

        # the bill the split would owe: Rq = QG @ S over every chunk at once
        qg = amat[:, :c].float().contiguous()
        led["OWED Rq = QG @ S, fp32 batched GEMM"] = timed(
            lambda: torch.bmm(qg, states), args.iters, args.warmup
        )
        qg_bf = amat[:, :c]
        st_bf = states.to(torch.bfloat16)
        led["OWED Rq = QG @ S, bf16 batched GEMM"] = timed(
            lambda: torch.bmm(qg_bf, st_bf), args.iters, args.warmup
        )
        led["  (cast S to bf16, part of the bf16 line)"] = timed(
            lambda: states.to(torch.bfloat16), args.iters, args.warmup
        )
        aqk = torch.randn(nb, c, c, dtype=torch.float32, device=dev)
        led["REF output baddbmm (Aqk @ T + Rq), as today"] = timed(
            lambda: torch.baddbmm(rq, aqk, t_all, beta=0.5, alpha=0.5), args.iters, args.warmup
        )

    width = max(len(k) for k in led)
    print(f"\n### {args.shape} — sweep split feasibility (µs, median of {args.iters})\n")
    for k, v in led.items():
        print(f"{k:{width}s} {v:8.1f}")

    for bv in bvs:
        a = led.get(f"bv{bv:<3d} A today (rq=1, states=0)", float("nan"))
        bb = led.get(f"bv{bv:<3d} B split seq (rq=0, states=1)", float("nan"))
        for gemm_kind in ("fp32", "bf16"):
            owed = led[f"OWED Rq = QG @ S, {gemm_kind} batched GEMM"]
            if gemm_kind == "bf16":
                owed += led["  (cast S to bf16, part of the bf16 line)"]
            print(
                f"\nbv={bv}: split total = B {bb:.0f} + owed({gemm_kind}) {owed:.0f} "
                f"= {bb + owed:.0f} µs against today's A {a:.0f} µs "
                f"-> {'SAVES' if bb + owed < a else 'COSTS'} {abs(a - bb - owed):.0f} µs"
            )

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"sweep_split_{args.shape}.json")
    with open(out, "w") as fh:
        json.dump(led, fh, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
