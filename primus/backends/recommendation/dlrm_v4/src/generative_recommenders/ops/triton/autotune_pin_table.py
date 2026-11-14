"""Robust per-key Triton autotune pins (``DLRM_HSTU_FP8_AUTOTUNE_PINS``).

Default ON through the gin binding ``apply_env_bootstrap.DLRM_HSTU_FP8_AUTOTUNE_PINS =
True`` (yambda_5b.gin, ship-4 item (10)); absent from the environment = off. Opt out per
run with ``DLRM_HSTU_FP8_AUTOTUNE_PINS=0``. Ported from the Arbor testbed (commit
14006f6, iteration 20) onto b97d9d9; the kernel modules and their autotune config
lists are identical between the two trees.

Why: every cold start benchmarks ~21 multi-config autotune keys inside the MLPerf timed
region (~31 s/run, iter-18 ``startup-attribution``), ranks pick different winners on
near-ties (coin flips, ~0.10 ms/rank-step), and the winner chosen on the early-window
shape is kept for the whole run although the keys do not contain the token count M
(early winners at late shapes cost ~0.51 ms/rank-step, iter-19 ``autotune-pin-table``).

What: seed each Autotuner's in-memory ``cache[key]`` with the config that minimises the
WORST normalised autotune time across four shape regimes (START_TS 0 twice, 150, 280;
8 ranks x 2 hosts). Source: eval/eval_autotune_regret.py over probes/iter19/attime_{a,b};
table proposals/autotune_pin_table_iter19.md. A seeded key skips benchmarking; any key not
in the table (new shape/dtype) is autotuned exactly as before. Only configs that are
already in the kernel's own autotune list are used, so every pinned launch is one the
autotuner could have chosen itself. ``TRITON_FULL_AUTOTUNE=1`` disables the pins.
The regimes above (and the posenc rows below) came from real-data length pools; under
the 2026-10-12 user rule the pins must be re-justified on SYNTHETIC lengths (pins re-derivation
kit).
STATUS: all 23 rows re-evaluated on SYNTHETIC lengths (pins kit run 70576,
same worst-regret rule over the config-limit regimes of gen_lengths.py): 9 SAME, 7 CHANGE
(switched to the synthetic pick) and 7 NEAR, kept at their earlier real-length value
because it lies within tolerance of the synthetic pick. Per-row comments give the regret.

Numerics: BLOCK_N / num_warps change rows per program. Per-row LayerNorm statistics are
unchanged, but the dw/db partial-sum reduction order of ``*_bwd*`` kernels can differ
from whatever winner a rank would have picked (which already varies run to run).
Declared as a reduction-order change.

Ship-12 item (33) (yambda_5b.gin): the last two rows pin the item-(20) posenc concat
kernel (``triton_position._add_timestamp_position_embeddings_concat_kernel``; int32 and
int16 index keys). They are seeded when ``DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT`` is
on (gin binding; code default off) and the table is on. Kill switch
``DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT=0`` (forwarded by launch_slurm.sh's
``DLRM_HSTU_FP8_`` prefix rule): the table then seeds the other 21 rows exactly as
before and the concat kernel autotunes as before.
"""

import ast
import logging
import os
from typing import Dict, List, Tuple

logger: logging.Logger = logging.getLogger(__name__)

# Grep-able arming literal: a run log with "armed OK: [autotune-pins] seeded 23/23 keys"
# on every rank proves the table is active (21/21 with the item (33) row switched off,
# which the line then says). Not printed when 0 keys were seeded.
ARMED_PREFIX = "armed OK: [autotune-pins]"

# Ship-12 item (33): the posenc concat rows and their own switch (code default
# off; the gin binding apply_env_bootstrap.DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT
# turns them on; "0" is the kill switch).
_POSENC_CONCAT_KERNEL = "_add_timestamp_position_embeddings_concat_kernel"
_POSENC_CONCAT_SWITCH = "DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT"


def _posenc_concat_row_on() -> bool:
    """Item (33) row switch: on when DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT is set
    and not 0/false/off/no (the gin binding exports "1"; absent = off). Read when
    the pins are applied (after the env bootstrap)."""
    value = os.environ.get(_POSENC_CONCAT_SWITCH, "0").strip().lower()
    return value not in ("", "0", "false", "off", "no")

_PKG = "generative_recommenders.ops.triton."

# (module, kernel, str(autotune key), robust config)
PIN_TABLE: List[Tuple[str, str, str, Dict[str, int]]] = [
    ('triton_hstu_linear', '_ln_mul_dropout_bwd_dwdb', "(512, 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.bfloat16')", {'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),  # synthetic worst regret 0.00% (LENGTH-INDEPENDENT; 1 regimes, pins kit)
    ('triton_hstu_linear', '_ln_mul_dropout_fwd_rng', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.int8')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.00% (7 regimes, pins kit)
    ('triton_hstu_linear', '_ln_mul_dropout_fwd_rng', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.int8')", {'BLOCK_N': 2, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 3.14% (7 regimes, pins kit)
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(1, 'torch.int64', 'torch.int64', 'torch.int64', 'torch.int64')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.00% (7 regimes, pins kit)
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(1, 'torch.int64', 'torch.int64', 'torch.int64', 'torch.int64', 'torch.int64')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.00% (7 regimes, pins kit)
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(32, 'torch.float32', 'torch.float32', 'torch.int64', 'torch.int64', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.00% (7 regimes, pins kit)
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.int64', 'torch.bfloat16')", {'BLOCK_N': 2, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 3.72% (7 regimes, pins kit)
    ('triton_jagged_tensors', 'split_2D_jagged_multirow', "(32, 'torch.float32', 'torch.int64', 'torch.int64', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.00% (7 regimes, pins kit)
    ('triton_jagged_tensors', 'split_2D_jagged_multirow', "(512, 'torch.bfloat16', 'torch.int64', 'torch.bfloat16', 'torch.bfloat16')", {'BLOCK_N': 4, 'num_warps': 2, 'num_stages': 3}),  # synthetic worst regret 7.06% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(256, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.00% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 6.39% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 8.80% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(512, 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.96% (LENGTH-INDEPENDENT; 1 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(256, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 4, 'num_stages': 3}),  # synthetic worst regret 0.00% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32')", {'BLOCK_N': 4, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 8.58% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 4, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 7.05% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16')", {'BLOCK_N': 2, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 6.79% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 4, 'num_stages': 3}),  # synthetic worst regret 6.91% (7 regimes, pins kit)
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # synthetic worst regret 0.57% (LENGTH-INDEPENDENT; 1 regimes, pins kit)
    ('triton_position', '_add_embeddings_bwd_kernel', "(4096, 1024, 512, 'torch.bfloat16', 'torch.int32', 'torch.int64', 'torch.float32')", {'BLOCK': 64, 'num_warps': 2, 'num_stages': 3}),  # synthetic worst regret 1.84% (7 regimes, pins kit)
    ('triton_position', '_add_timestamp_position_embeddings_kernel', "(4096, 'torch.bfloat16', 'torch.int64', 'torch.int64', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.int64', 'torch.int32', 'torch.int32', 'torch.int64')", {'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}),  # synthetic worst regret 3.08% (7 regimes, pins kit)
    # Ship-12 item (33): the ship8 item (20) POSENC_CONCAT_FOLD kernel (no row before: the
    # first training step compiled + benchmarked all 18 configs inside the timed region,
    # +4.6-4.9 s cold at 1/2/4n). Elementwise per row: every config is bitwise identical
    # (checked). Originally picked over 4 length regimes built from real sample-length pools
    # (prod/dense pools x1.0/x0.5, B=1024, gfx950; worst regret 3.6%, worst config
    # +97%; Arbor dlstart iter 18 probes/iter18/v10/out/posenc.json). Those pools were
    # training-data derived and are removed (user rule 2026-10-12); the pin must be
    # re-justified on SYNTHETIC lengths (pins re-derivation kit). STATUS:
    # done (pins kit run 70576): synthetic worst regret of the row 1.8%, worst config +80%. Kill switch: DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT=0.
    ('triton_position', '_add_timestamp_position_embeddings_concat_kernel', "(4096, 'torch.bfloat16', 'torch.bfloat16', 'torch.int64', 'torch.int64', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.int64', 'torch.int32', 'torch.int32', 'torch.int32', 'torch.int64')", {'BLOCK_N': 32, 'num_warps': 2, 'num_stages': 2}),  # synthetic worst regret 1.80% (7 regimes, pins kit)
    # Ship-12 merge: int16 twin of the row above. The rccl (41)(b) POSENC_SORT16
    # item makes PosInds / TsInds int16, so the int32 key would never match; same
    # config (the kernel is elementwise per row: bitwise for any config). Covered by
    # the same row switch (DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT filters by kernel).
    ('triton_position', '_add_timestamp_position_embeddings_concat_kernel', "(4096, 'torch.bfloat16', 'torch.bfloat16', 'torch.int64', 'torch.int64', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.int64', 'torch.int16', 'torch.int16', 'torch.int32', 'torch.int64')", {'BLOCK_N': 32, 'num_warps': 2, 'num_stages': 2}),  # synthetic worst regret 2.16% (7 regimes, pins kit)
]


def _cfg_matches(cfg, want: Dict[str, int]) -> bool:
    for name, val in want.items():
        if name == "num_warps":
            if cfg.num_warps != val:
                return False
        elif name == "num_stages":
            if cfg.num_stages != val:
                return False
        elif cfg.kwargs.get(name) != val:
            return False
    # every tunable kwarg of the config must be named by the pin
    return set(cfg.kwargs) <= set(want)


def apply_autotune_pins(force: bool = False) -> Tuple[int, int]:
    """Seed the autotune caches. Returns (seeded, total). Safe to call more than once."""
    if not force and os.environ.get("DLRM_HSTU_FP8_AUTOTUNE_PINS", "0") != "1":
        print(
            "[autotune-pins] OFF (DLRM_HSTU_FP8_AUTOTUNE_PINS!=1): autotuning every key",
            flush=True,
        )
        return 0, 0
    # Item (33) row switch: with it off the table is exactly the pre-item-(33) table.
    posenc_concat_on = _posenc_concat_row_on()
    table = [
        row for row in PIN_TABLE
        if posenc_concat_on or row[1] != _POSENC_CONCAT_KERNEL
    ]
    if os.environ.get("TRITON_FULL_AUTOTUNE", "0") == "1":
        print("[autotune-pins] OFF: TRITON_FULL_AUTOTUNE=1 disables the pins", flush=True)
        return 0, len(table)
    import importlib

    seeded = 0
    problems = []
    for mod_name, kern, key_str, want in table:
        try:
            mod = importlib.import_module(_PKG + mod_name)
            at = getattr(mod, kern)
            # unwrap @triton.heuristics / other decorators layered over the Autotuner
            for _ in range(4):
                if hasattr(at, "cache") and hasattr(at, "configs"):
                    break
                at = at.fn
            cache = at.cache
            configs = at.configs
        except Exception as e:  # pragma: no cover - environment specific
            problems.append(f"{kern}: {type(e).__name__}: {e}")
            continue
        match = [c for c in configs if _cfg_matches(c, want)]
        if len(match) != 1:
            problems.append(f"{kern} {key_str[:40]}: {len(match)} configs match {want}")
            continue
        key = ast.literal_eval(key_str)
        if key in cache and cache[key] is not match[0]:
            problems.append(f"{kern} {key_str[:40]}: already tuned, left as is")
            continue
        cache[key] = match[0]
        seeded += 1
    if seeded > 0:
        msg = f"{ARMED_PREFIX} seeded {seeded}/{len(table)} keys"
    else:
        msg = f"[autotune-pins] NOT armed: seeded 0/{len(table)} keys"
    if not posenc_concat_on:
        msg += f"; item (33) posenc concat row OFF ({_POSENC_CONCAT_SWITCH}=0)"
    if problems:
        msg += "; skipped: " + " | ".join(problems)
    print(msg, flush=True)
    return seeded, len(table)
