###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused MLP fwd+bwd+AdamW on 31B-width shapes. Closer to MaxText train_step than isolated tx.update."""

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
        f"{name}: dtype={x.dtype} shape={tuple(x.shape)} nan={nnan} inf={ninf} "
        f"frac_nan={nnan/n:.6f} absmax={float(jnp.max(jnp.abs(finite)))}",
        flush=True,
    )
    return nnan, ninf


def tree_stats(prefix, tree):
    bad = False
    for k, v in tree.items():
        nnan, ninf = stats(f"{prefix}.{k}", v)
        bad = bad or nnan or ninf
    return bad


def silu(x):
    return x * jax.nn.sigmoid(x)


def mlp_loss(params, x, y):
    # Gemma-4 gated MLP: (silu(x@wi0) * (x@wi1)) @ wo
    h = silu(x @ params["wi_0"]) * (x @ params["wi_1"])
    out = h @ params["wo"]
    return jnp.mean((out.astype(jnp.float32) - y.astype(jnp.float32)) ** 2)


def make_step(tx, clip_tx):
    def step(params, opt_state, clip_state, x, y):
        loss, grads = jax.value_and_grad(mlp_loss)(params, x, y)
        grads, clip_state = clip_tx.update(grads, clip_state)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, clip_state, loss, grads

    return jax.jit(step)


def run(label, fused, lr=3e-5, wd=0.1, mu=jnp.bfloat16, nsteps=2):
    print(f"\n=== {label} fused={fused} ===", flush=True)
    k = jax.random.key(1)
    ks = jax.random.split(k, 6)
    params = {
        "wi_0": jax.random.normal(ks[0], (5376, 21504), dtype=jnp.bfloat16),
        "wi_1": jax.random.normal(ks[1], (5376, 21504), dtype=jnp.bfloat16),
        "wo": jax.random.normal(ks[2], (21504, 5376), dtype=jnp.bfloat16),
    }
    x = jax.random.normal(ks[3], (1, 1024, 5376), dtype=jnp.bfloat16)
    y = jax.random.normal(ks[4], (1, 1024, 5376), dtype=jnp.bfloat16)
    tree_stats("init", params)
    tx = optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8, eps_root=0.0, weight_decay=wd, mu_dtype=mu)
    clip_tx = optax.clip_by_global_norm(1.0)
    opt_state = tx.init(params)
    clip_state = clip_tx.init(params)
    step_fn = make_step(tx, clip_tx)
    split_loss = jax.jit(jax.value_and_grad(mlp_loss))
    split_clip = jax.jit(clip_tx.update)
    split_upd = jax.jit(tx.update)
    split_apply = jax.jit(optax.apply_updates)
    for i in range(nsteps):
        if fused:
            params, opt_state, clip_state, loss, grads = step_fn(params, opt_state, clip_state, x, y)
        else:
            loss, grads = split_loss(params, x, y)
            grads, clip_state = split_clip(grads, clip_state)
            updates, opt_state = split_upd(grads, opt_state, params)
            params = split_apply(params, updates)
        print(f"step {i} loss={float(loss)}", flush=True)
        if tree_stats(f"step{i}_g", grads) or tree_stats(f"step{i}_w", params):
            print("NONFINITE", flush=True)
            return False
    print("FINITE", flush=True)
    return True


def main():
    print("jax", jax.__version__, "optax", optax.__version__, "devices", jax.devices(), flush=True)
    r1 = run("split_fwd_bwd_then_adamw", fused=False)
    r2 = run("fused_train_step_jit", fused=True)
    print("\n=== SUMMARY ===", flush=True)
    print("split:", "FINITE" if r1 else "NONFINITE")
    print("fused:", "FINITE" if r2 else "NONFINITE")


if __name__ == "__main__":
    main()
