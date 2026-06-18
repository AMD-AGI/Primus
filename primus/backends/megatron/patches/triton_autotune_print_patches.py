###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Silence Triton's autotuner stdout spam.

Triton's ``Autotuner._bench`` (``triton/runtime/autotuner.py``) emits a bare
``print()`` for *every* (kernel, config) pair it benchmarks when the
``knobs.autotuning.print`` flag (env: ``TRITON_PRINT_AUTOTUNING``) is on::

    Autotuning kernel _permute_kernel with config BLOCK_SIZE: 64, num_warps: 4, ...

On a multi-node run, every autotuned Triton kernel (e.g. the MoE
permute / unpermute / sort kernels in
``primus/backends/transformer_engine/pytorch/triton/permutation.py``) produces a
whole batch of these lines during warmup, flooding the logs.

Because it is a ``print()`` (not a ``logging`` record), a logging-level change
cannot suppress it -- the only lever is the Triton ``knobs.autotuning.print``
flag. We force it off here in the ``before_train`` phase, which runs inside the
training worker after model setup but before the first training step (and thus
before any kernel is autotuned). Programmatic assignment to the knob shadows the
``TRITON_PRINT_AUTOTUNING`` env value, so this works regardless of how the flag
got turned on.

Set ``PRIMUS_VERBOSE_TRITON=1`` to keep the full autotune trace.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.triton.silence_autotune_print",
    backend="megatron",
    phase="before_train",
    description=(
        "Force Triton knobs.autotuning.print off so the autotuner does not "
        "flood stderr with one print line per kernel config it benchmarks."
    ),
    condition=lambda ctx: os.environ.get("PRIMUS_VERBOSE_TRITON", "0") != "1",
)
def patch_silence_triton_autotune_print(ctx: PatchContext):
    """Turn off Triton autotuner printing. Idempotent / best-effort."""
    del ctx
    try:
        from triton import knobs
    except Exception as exc:  # triton missing -- nothing to do.
        log_rank_0(f"[Patch:megatron.triton.silence_autotune_print] triton unavailable: {exc!r}")
        return

    try:
        knobs.autotuning.print = False
    except Exception as exc:  # knobs API changed -- never block training.
        log_rank_0(
            f"[Patch:megatron.triton.silence_autotune_print] could not set "
            f"knobs.autotuning.print: {exc!r}"
        )
        return

    log_rank_0("[Patch:megatron.triton.silence_autotune_print] Triton autotuner printing disabled.")
