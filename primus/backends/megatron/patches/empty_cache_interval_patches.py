###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Megatron empty_cache() interval patch.

PROBLEM
-------
Megatron's ``train_step`` calls ``torch.cuda.empty_cache()`` before every
``optimizer.step()`` when ``args.empty_unused_memory_level >= 1``
(megatron/training/training.py:1759).

On the Pure-GDN 1B / 100B-tokens MI300X run profiling showed this single
call was responsible for ~4.6 s of ``hipMalloc`` and ~2.3 s of ``hipFree``
EVERY iter — 91% of the 7.65 s/iter wall time.  Removing it entirely
crashes NCCL on iter 1 with ``Failed to CUDA calloc 4 MiB`` because
NCCL's lazy workspace allocation needs a contiguous block that
fragmentation prevents at 81% VRAM usage.

SOLUTION
--------
Call ``empty_cache()`` only every N iters instead of every iter.  Once
NCCL's workspace is allocated on the iter where ``empty_cache()`` ran,
it stays cached on subsequent iters that DO NOT call ``empty_cache()``
— so we avoid the per-iter ``hipMalloc`` cost while still periodically
returning fragmented cached blocks to the driver as a safety net.

CONFIGURATION
-------------
The interval is set, in order of precedence:

    1. ``empty_cache_interval: N`` in the EXP YAML's ``overrides:`` block
       (preferred — co-located with the rest of the run config)
    2. ``PRIMUS_EMPTY_CACHE_INTERVAL=N`` env var (for ad-hoc overrides
       without editing the YAML; also lets a single launcher script set
       a default for backwards compatibility)
    3. Default ``1`` (passthrough — no behavioural change vs vanilla
       Megatron)

Values:
    - 1     : ORIGINAL behaviour — call empty_cache() every iter
              (only when args.empty_unused_memory_level >= 1, which is
              the gating condition Megatron itself uses; this patch
              NEVER enables empty_cache when Megatron's flag is 0).
    - N >= 2: call empty_cache() only on iters where
              ((iteration_count_since_train_start) % N == 0), i.e. once
              every N iters.  Iter 0 ALWAYS runs empty_cache (so the
              first NCCL workspace allocation succeeds against a clean
              cache).
    - 0     : NEVER call empty_cache() in train_step (CAUTION: risks the
              NCCL OOM if the model is memory-tight; only safe if NCCL
              has already been warmed up via some other mechanism).

The patch leaves ``args.empty_unused_memory_level``'s OTHER call sites
(checkpoint save in training.py:3214, and the level-2 call after
optimizer.step in training.py:1803) UNAFFECTED — those run rarely
enough that they are not a per-iter cost concern.

LOSS IMPACT
-----------
None.  ``torch.cuda.empty_cache()`` only returns unused cached blocks
to the driver — it does not modify any tensor data.  The only thing
this patch changes is HOW OFTEN we return those cached blocks, which
is purely a memory-bookkeeping decision.

TURNING OFF
-----------
Set ``empty_cache_interval: 1`` in the EXP YAML (or unset the env var)
to get the original per-iter behaviour back.  Alternatively, set
``empty_unused_memory_level: 0`` in the EXP YAML — the patch is
short-circuited when Megatron's own flag is 0 (which is also the
Megatron default for non-OOM-prone runs).
"""

from __future__ import annotations

import os
from typing import Optional

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


_DEFAULT_INTERVAL = 1


def _coerce(value, source: str) -> Optional[int]:
    """Best-effort parse of a user-supplied value to int; warn on bad input."""
    if value is None:
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        # Best-effort warn; if logger is uninitialised (e.g. unit tests) we
        # fall back to print to avoid a secondary AttributeError.
        msg = (
            f"[Patch:megatron.empty_cache_interval] WARN: invalid "
            f"{source}={value!r}, ignoring."
        )
        try:
            log_rank_0(msg)
        except Exception:
            print(msg)
        return None


def _resolve_interval(args) -> int:
    """Resolve the effective empty_cache_interval.

    Precedence: args.empty_cache_interval > PRIMUS_EMPTY_CACHE_INTERVAL env > 1.
    """
    yaml_val = _coerce(getattr(args, "empty_cache_interval", None), "args.empty_cache_interval (YAML)")
    if yaml_val is not None:
        return yaml_val

    env_val = _coerce(os.environ.get("PRIMUS_EMPTY_CACHE_INTERVAL"), "PRIMUS_EMPTY_CACHE_INTERVAL env")
    if env_val is not None:
        return env_val

    return _DEFAULT_INTERVAL


@register_patch(
    "megatron.training.empty_cache_interval.wrap_train_step",
    backend="megatron",
    phase="before_train",
    description=(
        "Skip torch.cuda.empty_cache() in train_step on non-trigger iters when "
        "empty_cache_interval > 1.  Eliminates the ~5 s/iter hipMalloc/hipFree "
        "thrash on the Pure-GDN 1B run while still flushing periodically."
    ),
)
def patch_train_step_with_empty_cache_interval(ctx: PatchContext) -> None:
    import megatron.training.training as training  # type: ignore
    from megatron.training.global_vars import get_args as get_megatron_args

    original_train_step = training.train_step
    if getattr(original_train_step, "_primus_empty_cache_interval_wrapped", False):
        return

    counter = {"n": 0, "interval": None, "logged": False}

    def _train_step_with_interval(*args, **kwargs):
        mg_args = get_megatron_args()

        # Resolve once on the first call.  We can't do this in patch
        # registration time because args isn't fully built yet.
        if counter["interval"] is None:
            counter["interval"] = _resolve_interval(mg_args)
            interval = counter["interval"]
            src = (
                "YAML(empty_cache_interval)"
                if getattr(mg_args, "empty_cache_interval", None) is not None
                else (
                    "env(PRIMUS_EMPTY_CACHE_INTERVAL)"
                    if os.environ.get("PRIMUS_EMPTY_CACHE_INTERVAL") is not None
                    else "default"
                )
            )
            mode = (
                "every iter (no-op)"
                if interval == 1
                else ("NEVER (risky)" if interval == 0 else f"every {interval} iters")
            )
            log_rank_0(
                f"[Patch:megatron.empty_cache_interval] empty_cache_interval={interval} "
                f"({mode}); source={src}; "
                f"empty_unused_memory_level={getattr(mg_args, 'empty_unused_memory_level', 0)}"
            )
            counter["logged"] = True

        interval = counter["interval"]
        # Original gating in training.py:1759 is `if args.empty_unused_memory_level >= 1`.
        # We only intervene when the user actually wanted empty_cache (so we never
        # accidentally enable it).  Behaviour:
        #   - interval == 1 : passthrough (every iter behaviour unchanged)
        #   - interval >= 2 : on iters NOT divisible by interval, temporarily
        #                     downgrade empty_unused_memory_level to 0 so
        #                     train_step skips the empty_cache call.  On the
        #                     trigger iter, restore the original value so the
        #                     empty_cache fires as designed.
        #   - interval == 0 : always downgrade (never call empty_cache in
        #                     train_step).  Risky; documented above.
        orig_level = getattr(mg_args, "empty_unused_memory_level", 0)
        downgrade = False

        if orig_level >= 1 and interval != 1:
            if interval == 0:
                downgrade = True
            else:
                # iter 0 (first call) ALWAYS triggers so NCCL's first allocation
                # happens against a clean cache.  After that, fire on every Nth iter.
                if counter["n"] != 0 and (counter["n"] % interval != 0):
                    downgrade = True

        try:
            if downgrade:
                # Megatron reads args.empty_unused_memory_level inline in
                # train_step.  Temporarily clear it for this call only.
                mg_args.empty_unused_memory_level = 0
            return original_train_step(*args, **kwargs)
        finally:
            if downgrade:
                mg_args.empty_unused_memory_level = orig_level
            counter["n"] += 1

    setattr(_train_step_with_interval, "_primus_empty_cache_interval_wrapped", True)
    training.train_step = _train_step_with_interval

    log_rank_0(
        "[Patch:megatron.empty_cache_interval] Wrapped train_step(); "
        "actual interval will be resolved + logged on first train_step call."
    )
