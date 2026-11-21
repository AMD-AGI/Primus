###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Standalone Optax AdamW vs SGD on Gemma-4 31B MLP wi shape. No MaxText."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import optax


def stats(name, x):
    x = x.block_until_ready()
    n = x.size
    nnan = int(jnp.isnan(x).sum())
    ninf = int(jnp.isinf(x).sum())
    finite = jnp.where(jnp.isfinite(x), x.astype(jnp.float32), 0)
    print(
        f"{name}: dtype={x.dtype} shape={tuple(x.shape)} nbytes={x.nbytes} "
        f"nan={nnan} inf={ninf} frac_nan={nnan/n:.6f} "
        f"finite_absmax={float(jnp.max(jnp.abs(finite)))}",
        flush=True,
    )
    return nnan, ninf


def run(label, shape, opt_kind, lr, wd, mu_dtype, nsteps=2, grad_scale=1.0):
    print(f"\n=== {label} shape={shape} {opt_kind} lr={lr} wd={wd} mu={mu_dtype} ===", flush=True)
    k = jax.random.key(0)
    k1, k2 = jax.random.split(k)
    w = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
    g = (jax.random.normal(k2, shape, dtype=jnp.bfloat16) * grad_scale).astype(jnp.bfloat16)
    stats("init_w", w)
    stats("init_g", g)

    if opt_kind == "adamw":
        tx = optax.adamw(
            lr,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            eps_root=0.0,
            weight_decay=wd,
            mu_dtype=mu_dtype,
        )
    elif opt_kind == "sgd":
        tx = optax.sgd(lr)
    else:
        raise ValueError(opt_kind)

    opt_state = tx.init(w)
    apply = jax.jit(tx.update)

    for step in range(nsteps):
        updates, opt_state = apply(g, opt_state, w)
        stats(f"step{step}_updates", updates)
        w = jax.jit(lambda p, u: optax.apply_updates(p, u))(w, updates)
        nnan, ninf = stats(f"step{step}_w", w)
        if nnan or ninf:
            print(f"NONFINITE at step {step}", flush=True)
            return False
    print("FINITE", flush=True)
    return True


def main():
    print("jax", jax.__version__, "optax", optax.__version__, flush=True)
    print("devices", jax.devices(), flush=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

    cases = [
        ("tiny_adamw_bf16mu", (2048, 4096), "adamw", 3e-5, 0.1, jnp.bfloat16),
        ("large_adamw_bf16mu", (5376, 21504), "adamw", 3e-5, 0.1, jnp.bfloat16),
        ("large_adamw_lr0_wd0", (5376, 21504), "adamw", 0.0, 0.0, jnp.bfloat16),
        ("large_sgd", (5376, 21504), "sgd", 3e-5, 0.0, None),
        ("large_adamw_fp32mu", (5376, 21504), "adamw", 3e-5, 0.1, jnp.float32),
        ("large_adamw_bf16mu_tinygrad", (5376, 21504), "adamw", 3e-5, 0.1, jnp.bfloat16),
    ]
    results = []
    for i, (label, shape, kind, lr, wd, mu) in enumerate(cases):
        extra = {}
        if label.endswith("tinygrad"):
            extra["grad_scale"] = 1e-3
        ok = run(label, shape, kind, lr, wd, mu, **extra)
        results.append((label, ok))
    print("\n=== SUMMARY ===", flush=True)
    for label, ok in results:
        print(f"{label}: {'FINITE' if ok else 'NONFINITE'}", flush=True)


if __name__ == "__main__":
    main()
