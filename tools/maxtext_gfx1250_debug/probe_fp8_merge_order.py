###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""CPU probe: does MaxText's NNX train step update FP8 delayed-scaling stats and give kernel grads?

Mirrors the NNX branch of maxtext/trainers/pre_train/train.py (split -> grad -> State.merge update).
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import variablelib
from maxtext.layers import linears, quantizations

OWG = variablelib.variable_type_from_name("_overwrite_with_gradient", allow_register=True)
X = jax.random.normal(jax.random.PRNGKey(1), (256, 1024), jnp.bfloat16)
LOSS_SCALE = float(os.environ.get("LOSS_SCALE", "1e-4"))


def stats(a):
    a = jnp.asarray(a, jnp.float32)
    return f"absmax={float(jnp.abs(a).max()):.3e} zero_frac={float((a == 0).mean()):.4f}"


def owg_summary(layer):
    s = nnx.state(layer, OWG)
    out = {}
    for path, v in jax.tree_util.tree_leaves_with_path(s):
        name = str(path[-2] if str(path[-1]).endswith("value')") or "value" in str(path[-1]) else path[-1])
        out[name] = float(jnp.max(jnp.abs(jnp.asarray(v))))
    return out


def run(quant, label, fixed_merge=False):
    layer = linears.DenseGeneral(
        1024, 1024, quant=quant, dtype=jnp.bfloat16, weight_dtype=jnp.float32, rngs=nnx.Rngs(0)
    )
    print(f"== {label}")
    for step in range(4):
        graphdef, params, custom, rest = nnx.split(layer, nnx.Param, nnx.Any(OWG), ...)

        def diff(params, custom, rest):
            m = nnx.merge(graphdef, params, custom, rest, copy=True)
            loss = (m(X).astype(jnp.float32) ** 2).mean() * LOSS_SCALE
            npr = nnx.state(m, nnx.Not(nnx.Any(nnx.Param, nnx.Intermediate)))
            return loss, npr

        (loss, npr), (g, cg) = jax.value_and_grad(diff, argnums=(0, 1), has_aux=True)(params, custom, rest)
        print(f" step {step}: loss={float(loss):.4e} kernel_grad {stats(g['kernel'].get_value())}")
        if quant is not None:
            print(f"   custom_grads (new fp8 stats): {owg_summary_state(cg)}")
            nnx.update(layer, nnx.State.merge(npr, cg) if fixed_merge else nnx.State.merge(cg, npr))
            print(
                f"   layer fp8 stats after train.py-style update: {owg_summary_state(nnx.state(layer, OWG))}"
            )


def owg_summary_state(s):
    out = {}
    for path, v in jax.tree_util.tree_leaves_with_path(s):
        keys = [getattr(p, "key", getattr(p, "name", str(p))) for p in path]
        name = next(
            (k for k in reversed(keys) if isinstance(k, str) and ("scale" in k or "amax" in k)), str(keys)
        )
        out[name] = f"{float(jnp.max(jnp.abs(jnp.asarray(v)))):.3e}"
    return out


run(None, "bf16 (no quant)")
run(quantizations.Fp8Quantization(), "fp8 (quantization: fp8)")
run(quantizations.Fp8Quantization(), "fp8, merge order swapped (custom_grads win)", fixed_merge=True)
