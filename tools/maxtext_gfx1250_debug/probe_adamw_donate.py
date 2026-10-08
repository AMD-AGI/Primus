###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Multi-step AdamW on the 31B MLP wi shapes with buffer donation, like MaxText's p_train_step."""

import jax
import jax.numpy as jnp
import optax

SHAPE = (5376, 21504)


def bad(x):
    return int(jnp.isnan(x).sum()), int(jnp.isinf(x).sum())


def run(name, lr, wd, donate, steps=4):
    print(f"\n=== {name} lr={lr} wd={wd} donate={donate} ===", flush=True)
    params = {
        "wi_0": jax.random.normal(jax.random.key(0), SHAPE, dtype=jnp.bfloat16) * 0.02,
        "wi_1": jax.random.normal(jax.random.key(1), SHAPE, dtype=jnp.bfloat16) * 0.02,
    }
    grads = {
        "wi_0": jax.random.normal(jax.random.key(2), SHAPE, dtype=jnp.bfloat16) * 1e-3,
        "wi_1": jax.random.normal(jax.random.key(3), SHAPE, dtype=jnp.bfloat16) * 1e-3,
    }
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, eps_root=0.0, weight_decay=wd, mu_dtype=jnp.bfloat16),
    )
    opt_state = tx.init(params)

    def step(params, opt_state, grads):
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    fn = jax.jit(step, donate_argnums=(0, 1) if donate else ())
    ok = True
    for s in range(steps):
        params, opt_state = fn(params, opt_state, grads)
        adam = opt_state[1][0]
        line = [f"step{s} count={int(adam.count)}"]
        for k in ("wi_0", "wi_1"):
            p, m, v = bad(params[k]), bad(adam.mu[k]), bad(adam.nu[k])
            line.append(f"{k} p(nan,inf)={p} mu={m} nu={v}")
            ok &= not any(p + m + v)
        print("  " + " | ".join(line), flush=True)
    print("FINITE" if ok else "NONFINITE", flush=True)
    return ok


def main():
    print("jax", jax.__version__, "optax", optax.__version__, jax.devices(), flush=True)
    res = []
    for donate in (True, False):
        res.append((f"lr0 donate={donate}", run("lr0_wd0", 0.0, 0.0, donate)))
        res.append((f"lr3e-5 donate={donate}", run("lr3e-5_wd0.1", 3e-5, 0.1, donate)))
    print("\n=== SUMMARY ===")
    for name, ok in res:
        print(f"{name}: {'FINITE' if ok else 'NONFINITE'}")


if __name__ == "__main__":
    main()
