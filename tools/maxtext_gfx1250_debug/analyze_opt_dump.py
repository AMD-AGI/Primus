###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Offline analysis of PRIMUS_MAXTEXT_OPT_DEBUG_DUMP arrays from `run.sh maxtext` (lr=0, wd=0).

usage: analyze_opt_dump.py <dump_dir> [--replay]

Checks the AdamW invariants that hold at lr=0 with a fixed batch, maps where
non-finite values sit in each tracked kernel, and with --replay feeds the
reconstructed step-0 gradient through standalone optax.adamw (needs JAX).
"""

import glob
import os
import sys

import numpy as np

D = sys.argv[1]
REPLAY = "--replay" in sys.argv


def load(pattern):
    hits = sorted(glob.glob(os.path.join(D, pattern)))
    if len(hits) != 1:
        raise SystemExit(f"pattern {pattern}: {hits}")
    return np.load(hits[0])


def desc(name, x):
    fin = np.isfinite(x)
    print(
        f"  {name}: nan={np.isnan(x).sum()} inf={np.isinf(x).sum()} zero={(x == 0).sum()} "
        f"absmax={np.abs(x[fin]).max() if fin.any() else float('nan'):.4e}"
    )


def nan_layout(name, bad):
    if not bad.any():
        print(f"  {name}: no non-finite")
        return
    rows = np.where(bad.any(1))[0]
    cols = np.where(bad.any(0))[0]
    flat = np.flatnonzero(bad)
    print(
        f"  {name}: count={bad.sum()} frac={bad.mean():.4f} rows_hit={rows.size}/{bad.shape[0]} "
        f"cols_hit={cols.size}/{bad.shape[1]} flat_first={flat[0]} flat_last={flat[-1]} "
        f"first_byte_off_bf16={flat[0] * 2}"
    )
    per_row = bad.sum(1)
    print(
        f"    per-row count: min={per_row.min()} max={per_row.max()} "
        f"rows fully bad={(per_row == bad.shape[1]).sum()}"
    )
    # Contiguity of bad elements in flat order.
    breaks = np.count_nonzero(np.diff(flat) != 1) + 1
    print(f"    contiguous runs in flat order={breaks}")
    # 1 MiB granularity map of bad fraction.
    chunk = (1 << 20) // 2
    nchunks = (bad.size + chunk - 1) // chunk
    frac = np.array([bad.ravel()[i * chunk : (i + 1) * chunk].mean() for i in range(nchunks)])
    print(f"    1MiB chunks: total={nchunks} any_bad={(frac > 0).sum()} fully_bad={(frac == 1).sum()}")
    hist = np.histogram(frac, bins=[0, 1e-9, 0.01, 0.1, 0.5, 0.99, 1.0 + 1e-9])[0]
    print(f"    chunk bad-frac hist [0,~0,1%,10%,50%,99%,100%]: {hist.tolist()}")


def main():
    for k in ("wi_0", "wi_1"):
        print(f"\n===== {k} =====")
        p_pre = load(f"pre_step0_param_*{k}*kernel.npy")
        p0 = load(f"post_step0_param_*{k}*kernel.npy")
        p1 = load(f"post_step1_param_*{k}*kernel.npy")
        mu0 = load(f"post_step0_opt_*mu*{k}*.npy")
        nu0 = load(f"post_step0_opt_*nu*{k}*.npy")
        mu1 = load(f"post_step1_opt_*mu*{k}*.npy")
        nu1 = load(f"post_step1_opt_*nu*{k}*.npy")
        for n, x in (
            ("p_pre", p_pre),
            ("p0", p0),
            ("p1", p1),
            ("mu0", mu0),
            ("nu0", nu0),
            ("mu1", mu1),
            ("nu1", nu1),
        ):
            desc(n, x)

        print(
            " lr=0 invariant p0 == p_pre:",
            "exact" if np.array_equal(p0, p_pre) else f"DIFF n={(p0 != p_pre).sum()}",
        )
        print(
            " lr=0 invariant p1 == p_pre (finite part):",
            f"diff_finite={((p1 != p_pre) & np.isfinite(p1)).sum()}",
        )

        g0 = mu0 / 0.1
        fin = np.isfinite(nu0) & np.isfinite(mu0)
        pred_nu0 = 0.05 * g0 * g0
        rel = np.abs(nu0 - pred_nu0) / np.maximum(np.abs(pred_nu0), 1e-30)
        big = fin & (pred_nu0 > 1e-30) & (rel > 0.05)
        print(
            f" step0 nu0 vs 0.05*(mu0/0.1)^2: mismatch>5% = {big.sum()} "
            f"(nu0==0 & mu0!=0: {((nu0 == 0) & (mu0 != 0)).sum()}, nu0!=0 & mu0==0: {((nu0 != 0) & (mu0 == 0)).sum()})"
        )

        r_mu = np.abs(mu1 - 1.9 * mu0) / np.maximum(np.abs(1.9 * mu0), 1e-30)
        r_nu = np.abs(nu1 - 1.95 * nu0) / np.maximum(np.abs(1.95 * nu0), 1e-30)
        print(
            f" step1 mu1 vs 1.9*mu0: mismatch>5% = {(np.isfinite(mu1) & (np.abs(mu0) > 1e-30) & (r_mu > 0.05)).sum()}"
        )
        print(
            f" step1 nu1 vs 1.95*nu0: mismatch>5% = {(np.isfinite(nu1) & (np.abs(nu0) > 1e-30) & (r_nu > 0.05)).sum()}"
        )

        bad1 = ~np.isfinite(p1)
        nan_layout("p1 non-finite", bad1)
        nan_layout("mu1 non-finite", ~np.isfinite(mu1))
        nan_layout("nu1 non-finite", ~np.isfinite(nu1))
        if bad1.any():
            print("  at bad p1 positions:")
            for n, x in (("mu0", mu0), ("nu0", nu0), ("mu1", mu1), ("nu1", nu1), ("p_pre", p_pre)):
                v = x[bad1]
                vf = v[np.isfinite(v)]
                print(
                    f"    {n}: finite={vf.size}/{v.size} zero={(v == 0).sum()} "
                    f"absmed={np.median(np.abs(vf)) if vf.size else float('nan'):.3e}"
                )
            good = ~bad1
            for n, x in (("mu0", mu0), ("nu0", nu0)):
                v = x[good]
                print(
                    f"    (good positions) {n}: zero={(v == 0).sum()}/{v.size} absmed={np.median(np.abs(v)):.3e}"
                )

        if REPLAY:
            replay(k, p_pre, g0)


def replay(k, p_pre, g0):
    import jax
    import jax.numpy as jnp
    import optax

    print(f" replay {k}: standalone optax.adamw lr=0 wd=0 bf16 mu, g=mu0/0.1, 2 steps")
    w = jnp.asarray(p_pre, dtype=jnp.bfloat16)
    g = jnp.asarray(g0, dtype=jnp.bfloat16)
    tx = optax.adamw(0.0, b1=0.9, b2=0.95, eps=1e-8, eps_root=0.0, weight_decay=0.0, mu_dtype=jnp.bfloat16)
    st = tx.init(w)

    @jax.jit
    def step(w, st):
        u, st = tx.update(g, st, w)
        return optax.apply_updates(w, u), u, st

    for s in range(2):
        w, u, st = step(w, st)
        print(
            f"   step{s}: w nan={int(jnp.isnan(w).sum())} u nan={int(jnp.isnan(u).sum())} "
            f"u inf={int(jnp.isinf(u).sum())} mu nan={int(jnp.isnan(st[0].mu).sum())} "
            f"nu nan={int(jnp.isnan(st[0].nu).sum())}"
        )


if __name__ == "__main__":
    main()
