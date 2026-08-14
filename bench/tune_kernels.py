"""Sweep the three FlyDSL kernels' build-time parameters at production geometry.

Round 1 removed the glue; what is left is the kernels themselves, and before
rewriting one it is worth asking whether the *existing* one is built at the wrong
point. All three take a ``waves_per_eu``, the score adjoint's thread mapping is
set by a module constant (``THREADS_D``, which fixes its ``MR``/``NR`` register
tile and therefore its register pressure), and the sweep's ``block_v`` was tuned
once at an earlier pass.

Each configuration is timed on exactly the operands the real assembly hands the
kernel, and **checked bit-equal against the shipped configuration** — these are
scheduling parameters, so anything but bit-equality means a bug, not a tradeoff.

    python bench/tune_kernels.py --which scores_bwd,scores_fwd,sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from bench.bench_kda_backends import make_inputs, timed  # noqa: E402

RESULTS = os.path.join(REPO, "bench", "results")

# The production KDA geometry, as [B, T, H, K, V].
SHAPE = (1, 4096, 96, 128, 128)
CHUNK = 64


def build_operands(shape=SHAPE, chunk=CHUNK, dtype=torch.bfloat16):
    """``(qf, kf, cg, betaf, aqk, akk)`` exactly as ``_assemble`` produces them."""
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1.ops import (
        decay_scores,
    )

    b, t, h, k, v = shape
    q, kk, vv, g, beta = make_inputs(b, t, h, k, v, "cuda", dtype)
    with torch.no_grad():
        nb = b * h * (t // chunk)

        def lay(x):
            out = torch.empty((b, h, t, x.shape[-1]), dtype=torch.float32, device=x.device)
            out.copy_(x.transpose(1, 2))
            return out.view(nb, chunk, x.shape[-1])

        qf, kf, vf = lay(q), lay(kk), lay(vv)
        cg = lay(g).cumsum(dim=-2)
        aqk, akk = decay_scores(qf, kf, cg)
    return qf, kf, cg, vf, aqk, akk


def _reset_caches():
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import ops, sweep

    ops._KERNEL_CACHE.clear()
    ops._BWD_KERNEL_CACHE.clear()
    sweep._KERNEL_CACHE.clear()


# ---------------------------------------------------------------------------


def tune_scores_fwd(qf, kf, cg, iters, warmup) -> List[Dict[str, object]]:
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
        kda_decay_scores_kernel as mod,
    )

    nb, c, kd = qf.shape
    aqk = torch.empty(nb, c, c, dtype=torch.float32, device=qf.device)
    akk = torch.empty_like(aqk)
    ref = None
    rows = []
    for wpe in (1, 2, 4, 8):
        try:
            launch = mod.build_kda_decay_scores(chunk_size=c, k_dim=kd, waves_per_eu=wpe)
            run = lambda: launch(  # noqa: E731
                qf.reshape(-1), kf.reshape(-1), cg.reshape(-1), aqk.reshape(-1), akk.reshape(-1), nb
            )
            run()
            torch.cuda.synchronize()
            got = (aqk.clone(), akk.clone())
            if ref is None:
                ref = got
            err = max((g - r).abs().amax().item() for g, r in zip(got, ref))
            rows.append({"waves_per_eu": wpe, "us": timed(run, iters, warmup), "max_abs_vs_ref": err})
        except Exception:
            rows.append({"waves_per_eu": wpe, "error": traceback.format_exc().splitlines()[-1]})
        print(f"  scores_fwd {json.dumps(rows[-1])}", flush=True)
    return rows


def tune_scores_bwd(qf, kf, cg, aqk, akk, iters, warmup) -> List[Dict[str, object]]:
    """``waves_per_eu`` x ``THREADS_D``.

    ``THREADS_D`` sets the accumulator tile: ``TR = 256 / TD`` threads down the
    rows, ``MR = SB / TR`` rows and ``NR = KD / TD`` channels per thread, so it
    trades register pressure against LDS reads per FMA. The shipped 32 gives
    ``MR=2, NR=4`` — 32 accumulator registers before any temporaries, which is
    the occupancy hypothesis this sweep tests.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
        kda_decay_scores_bwd_kernel as mod,
    )

    nb, c, kd = qf.shape
    d_aqk = torch.randn_like(aqk)
    d_akk = torch.randn_like(akk)
    dq = torch.empty_like(qf)
    dk = torch.empty_like(qf)
    dcg = torch.empty_like(qf)
    shipped = mod.THREADS_D
    ref = None
    rows = []
    for td in (16, 32, 64, 128):
        for wpe in (1, 2, 4):
            cfg = {"threads_d": td, "waves_per_eu": wpe}
            try:
                mod.THREADS_D = td
                if mod.supports_bwd_geometry(c, kd) is not None:
                    cfg["skipped"] = mod.supports_bwd_geometry(c, kd)
                    rows.append(cfg)
                    continue
                launch = mod.build_kda_decay_scores_bwd(chunk_size=c, k_dim=kd, waves_per_eu=wpe)
                run = lambda: launch(  # noqa: E731
                    qf.reshape(-1),
                    kf.reshape(-1),
                    cg.reshape(-1),
                    d_aqk.reshape(-1),
                    d_akk.reshape(-1),
                    dq.reshape(-1),
                    dk.reshape(-1),
                    dcg.reshape(-1),
                    nb,
                )
                run()
                torch.cuda.synchronize()
                got = (dq.clone(), dk.clone(), dcg.clone())
                if ref is None:
                    ref = got
                scale = max(r.abs().amax().item() for r in ref)
                cfg["max_rel_vs_ref"] = (
                    max((g - r).abs().amax().item() for g, r in zip(got, ref)) / scale
                )
                cfg["us"] = timed(run, iters, warmup)
            except Exception:
                cfg["error"] = traceback.format_exc().splitlines()[-1]
            finally:
                mod.THREADS_D = shipped
            rows.append(cfg)
            print(f"  scores_bwd {json.dumps(cfg)}", flush=True)
    return rows


def tune_sweep(qf, kf, cg, vf, aqk, akk, iters, warmup) -> List[Dict[str, object]]:
    from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._flydsl_v1 import (
        kda_state_sweep_kernel as mod,
    )

    nb, c, kd = qf.shape
    v_dim = vf.shape[-1]
    num_chunks = 64
    nbh = nb // num_chunks
    dev = qf.device
    amat = torch.randn(nb, 2 * c, kd, dtype=torch.bfloat16, device=dev)
    yc = torch.randn(nb, c, v_dim, dtype=torch.float32, device=dev)
    xt = torch.randn(nb, kd, c, dtype=torch.bfloat16, device=dev)
    dec = torch.rand(nb, kd, dtype=torch.float32, device=dev)
    s0 = torch.zeros(nbh, kd, v_dim, dtype=torch.float32, device=dev)
    dummy = torch.empty(1, dtype=torch.float32, device=dev)
    rq = torch.empty(nb, c, v_dim, dtype=torch.float32, device=dev)
    t_all = torch.empty_like(rq)
    states = torch.empty(nb, kd, v_dim, dtype=torch.float32, device=dev)
    s_final = torch.empty(nbh, kd, v_dim, dtype=torch.float32, device=dev)

    ref = None
    rows = []
    for bv in (8, 16, 32, 64):
        for wpe in (1, 2, 4):
            cfg = {"block_v": bv, "waves_per_eu": wpe}
            try:
                reason = mod.supports_sweep_geometry(c, kd, v_dim, bv)
                if reason is not None:
                    cfg["skipped"] = reason
                    rows.append(cfg)
                    continue
                launch = mod.build_kda_state_sweep(
                    chunk_size=c,
                    k_dim=kd,
                    v_dim=v_dim,
                    block_v=bv,
                    mode="mfma",
                    emit_rq=True,
                    emit_states=True,
                    has_e=False,
                    sgn_t=-1.0,
                    sgn_x=1.0,
                    reverse=False,
                    waves_per_eu=wpe,
                )
                run = lambda: launch(  # noqa: E731
                    amat.reshape(-1),
                    yc.reshape(-1),
                    xt.reshape(-1),
                    dec.reshape(-1),
                    dummy,
                    s0.reshape(-1),
                    rq.reshape(-1),
                    t_all.reshape(-1),
                    states.reshape(-1),
                    s_final.reshape(-1),
                    nbh,
                    num_chunks,
                )
                run()
                torch.cuda.synchronize()
                got = (rq.clone(), t_all.clone(), s_final.clone())
                if ref is None:
                    ref = got
                scale = max(r.abs().amax().item() for r in ref)
                cfg["max_rel_vs_ref"] = (
                    max((g - r).abs().amax().item() for g, r in zip(got, ref)) / scale
                )
                cfg["us"] = timed(run, iters, warmup)
            except Exception:
                cfg["error"] = traceback.format_exc().splitlines()[-1]
            rows.append(cfg)
            print(f"  sweep {json.dumps(cfg)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="scores_fwd,scores_bwd,sweep")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--tag", default="tune")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"tune_{args.tag}.json")
    want = set(args.which.split(","))

    qf, kf, cg, vf, aqk, akk = build_operands()
    report: Dict[str, object] = {"shape": list(SHAPE), "chunk": CHUNK}
    if "scores_fwd" in want:
        _reset_caches()
        report["scores_fwd"] = tune_scores_fwd(qf, kf, cg, args.iters, args.warmup)
    if "scores_bwd" in want:
        _reset_caches()
        report["scores_bwd"] = tune_scores_bwd(qf, kf, cg, aqk, akk, args.iters, args.warmup)
    if "sweep" in want:
        _reset_caches()
        report["sweep"] = tune_sweep(qf, kf, cg, vf, aqk, akk, args.iters, args.warmup)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)

    for name, rows in report.items():
        if not isinstance(rows, list):
            continue
        ok = [r for r in rows if isinstance(r, dict) and "us" in r]
        if not ok:
            continue
        best = min(ok, key=lambda r: r["us"])
        shipped = ok[0]
        print(
            f"\n{name}: best {best['us']:.1f} µs at "
            f"{ {k: v for k, v in best.items() if k not in ('us', 'max_rel_vs_ref', 'max_abs_vs_ref')} }"
            f" vs first-listed {shipped['us']:.1f} µs -> {shipped['us'] / best['us']:.2f}x"
        )
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
