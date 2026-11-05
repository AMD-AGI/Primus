###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Silent corruption from a >2 GiB device buffer, no MaxText.

Fills sentinel buffers with a known value, then produces one f32 output larger
than 2 GiB (the size of an f32 copy of a 131072x5376 embedding). If stores past
byte 2^31 of that output land 4 GiB below it, they overwrite sentinels instead
of faulting, and the output's tail keeps whatever was in memory before.

Env: BIG_ROWS (default 131072), BIG_COLS (default 5376),
     SENTINEL_MB (per buffer, default 256), SENTINEL_COUNT (default 24).
"""

import os

import jax
import jax.numpy as jnp

ROWS = int(os.environ.get("BIG_ROWS", "131072"))
COLS = int(os.environ.get("BIG_COLS", "5376"))
SENT_ELEMS = int(os.environ.get("SENTINEL_MB", "256")) * (1 << 20) // 2
SENT_COUNT = int(os.environ.get("SENTINEL_COUNT", "24"))
SENT_VALUE = 3.0


def main():
    print("devices", jax.devices(), flush=True)
    sentinels = [jnp.full((SENT_ELEMS,), SENT_VALUE, dtype=jnp.bfloat16) for _ in range(SENT_COUNT)]
    jax.block_until_ready(sentinels)
    src = jnp.ones((ROWS, COLS), dtype=jnp.bfloat16)
    src.block_until_ready()
    out_bytes = ROWS * COLS * 4
    print(
        f"sentinels: {SENT_COUNT} x {SENT_ELEMS * 2} B; f32 output: {out_bytes} B "
        f"({out_bytes / 2**30:.3f} GiB), bytes past 2^31: {max(0, out_bytes - 2**31)}",
        flush=True,
    )

    big = jax.jit(lambda x: x.astype(jnp.float32) * 2.0)(src)
    big.block_until_ready()

    flat = big.reshape(-1)
    split = min(flat.size, (2**31) // 4)
    head_bad = int((flat[:split] != 2.0).sum())
    tail_bad = int((flat[split:] != 2.0).sum()) if flat.size > split else 0
    print(f"output: wrong below 2^31 B = {head_bad}, wrong past 2^31 B = {tail_bad} of {flat.size - split}")

    corrupted_bytes = 0
    for i, s in enumerate(sentinels):
        bad = int((s != SENT_VALUE).sum())
        if bad:
            corrupted_bytes += bad * 2
            print(f"sentinel {i}: {bad} elements overwritten")
    print(f"sentinel bytes overwritten: {corrupted_bytes}")
    ok = head_bad == 0 and tail_bad == 0 and corrupted_bytes == 0
    print("CLEAN" if ok else "CORRUPTED", flush=True)


if __name__ == "__main__":
    main()
