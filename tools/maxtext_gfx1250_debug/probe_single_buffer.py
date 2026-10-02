###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""One-shot GPU alloc. SHAPE='rows,cols' MODE=zeros|ones|normal DTYPE=bf16|fp32"""

import os

import jax
import jax.numpy as jnp

shape = tuple(int(x) for x in os.environ.get("SHAPE", "1024,1024").split(","))
mode = os.environ.get("MODE", "normal")
dt = {"bf16": jnp.bfloat16, "fp32": jnp.float32}[os.environ.get("DTYPE", "bf16")]
print("devices", jax.devices(), "shape", shape, "mode", mode, "dtype", dt, flush=True)
if mode == "zeros":
    x = jnp.zeros(shape, dtype=dt)
elif mode == "ones":
    x = jnp.ones(shape, dtype=dt)
else:
    x = jax.random.normal(jax.random.key(0), shape, dtype=dt)
x.block_until_ready()
print("OK nbytes", x.nbytes, "sample", float(x.reshape(-1)[0]), flush=True)
