###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""AdamW on the 31B MLP wi shape with realistic gradient magnitudes (tiny, denormal, zero)."""

import jax
import jax.numpy as jnp
import optax

SHAPE = (5376, 21504)


def stats(name, x):
    x = x.block_until_ready()
    nnan = int(jnp.isnan(x).sum())
    ninf = int(jnp.isinf(x).sum())
    f32 = x.astype(jnp.float32)
    tiny = int(((f32 != 0) & (jnp.abs(f32) < 1.1754944e-38)).sum())
    zero = int((f32 == 0).sum())
    print(f"{name}: nan={nnan} inf={ninf} denormal={tiny} zero={zero}", flush=True)
    return nnan + ninf


def make_grads(kind):
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    sign = jnp.where(jax.random.bernoulli(k1, 0.5, SHAPE), 1.0, -1.0)
    if kind == "loguniform_1e-45_1e-2":
        e = jax.random.uniform(k2, SHAPE, minval=-45.0, maxval=-2.0)
        g = sign * jnp.power(10.0, e)
    elif kind == "normal_1e-6":
        g = jax.random.normal(k2, SHAPE) * 1e-6
    elif kind == "half_zero_normal":
        g = jnp.where(jax.random.bernoulli(k3, 0.5, SHAPE), 0.0, jax.random.normal(k2, SHAPE) * 1e-4)
    else:
        raise ValueError(kind)
    return g.astype(jnp.bfloat16)


def run(kind, lr, wd):
    print(f"\n=== grads={kind} lr={lr} wd={wd} ===", flush=True)
    w = jax.random.normal(jax.random.key(0), SHAPE, dtype=jnp.bfloat16)
    g = make_grads(kind)
    stats("grad", g)
    clip = optax.clip_by_global_norm(1.0)
    tx = optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, eps_root=0.0, weight_decay=wd, mu_dtype=jnp.bfloat16)
    opt_state = tx.init(w)
    clip_state = clip.init(w)

    @jax.jit
    def step(w, g, opt_state, clip_state):
        g, clip_state = clip.update(g, clip_state)
        updates, opt_state = tx.update(g, opt_state, w)
        return optax.apply_updates(w, updates), updates, opt_state, clip_state

    w, updates, opt_state, clip_state = step(w, g, opt_state, clip_state)
    bad = stats("updates", updates) + stats("w_after", w)
    adam = opt_state[0]
    stats("mu", adam.mu)
    stats("nu", adam.nu)
    print("NONFINITE" if bad else "FINITE", flush=True)
    return not bad


def main():
    print("jax", jax.__version__, "optax", optax.__version__, jax.devices(), flush=True)
    res = []
    for kind in ("loguniform_1e-45_1e-2", "normal_1e-6", "half_zero_normal"):
        for lr, wd in ((0.0, 0.0), (3e-5, 0.1)):
            res.append((f"{kind} lr={lr}", run(kind, lr, wd)))
    print("\n=== SUMMARY ===")
    for name, ok in res:
        print(f"{name}: {'FINITE' if ok else 'NONFINITE'}")


if __name__ == "__main__":
    main()
