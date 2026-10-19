###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Isolate 31B-shaped allocations that HSA-fault MaxText before first step."""

import traceback

import jax
import jax.numpy as jnp

print("devices", jax.devices(), flush=True)


def run(name, fn):
    print(f"=== {name} ===", flush=True)
    try:
        fn()
        print(f"OK {name}", flush=True)
        return True
    except Exception:
        traceback.print_exc()
        print(f"FAIL {name}", flush=True)
        return False


def small():
    x = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
    y = (x @ x).block_until_ready()
    print("small gemm", float(y[0, 0]), flush=True)


def emb_table():
    # gemma4-31b: vocab 262144 x emb 5376
    k = jax.random.key(0)
    x = jax.random.normal(k, (262144, 5376), dtype=jnp.bfloat16)
    x.block_until_ready()
    print("emb mean", float(x.reshape(-1)[:8].mean()), "nbytes", x.nbytes, flush=True)


def mlp_kernel():
    k = jax.random.key(1)
    x = jax.random.normal(k, (5376, 21504), dtype=jnp.bfloat16)
    x.block_until_ready()
    print("mlp nbytes", x.nbytes, flush=True)


def mlp_gemm():
    k = jax.random.key(2)
    a = jax.random.normal(k, (1024, 5376), dtype=jnp.bfloat16)
    b = jax.random.normal(jax.random.fold_in(k, 1), (5376, 21504), dtype=jnp.bfloat16)
    y = (a @ b).block_until_ready()
    print("mlp gemm", y.shape, float(y[0, 0]), flush=True)


def logits_gemm():
    k = jax.random.key(3)
    a = jax.random.normal(k, (1024, 5376), dtype=jnp.bfloat16)
    b = jax.random.normal(jax.random.fold_in(k, 1), (5376, 262144), dtype=jnp.bfloat16)
    y = (a @ b).block_until_ready()
    print("logits gemm", y.shape, float(y[0, 0]), flush=True)


if __name__ == "__main__":
    for name, fn in [
        ("small_gemm", small),
        ("emb_table_init", emb_table),
        ("mlp_kernel_init", mlp_kernel),
        ("mlp_gemm", mlp_gemm),
        ("logits_gemm", logits_gemm),
    ]:
        if not run(name, fn):
            raise SystemExit(1)
    print("ALL OK", flush=True)
