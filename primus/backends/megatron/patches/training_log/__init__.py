###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron training_log patches package.

This package currently exposes:

    - print_rank_last_patches: scoped print_rank_last hook for training_log
      that injects ROCm memory and throughput statistics.
    - wall_clock_timer_patch: wraps train_step with a wall-clock timer for
      NeMo-comparable throughput measurement.
    - first_iteration_reset_patches: a first iteration on the log schedule
      is logged and reset as a regular logged iteration (one switch for every
      model, ``log_reset_first_iteration``, default off; the OLMo-3 example
      configs set it).

The actual patch registration is handled via ``@register_patch``; this
``__init__`` exists mainly to make ``training_log`` a proper package so
that the auto-import logic in ``primus.backends.megatron.patches.__init__``
can discover and import the patch modules automatically.
"""

from primus.backends.megatron.patches.training_log import (  # noqa: F401
    first_iteration_reset_patches,
    print_rank_last_patches,
    wall_clock_timer_patch,
)

__all__ = ["first_iteration_reset_patches", "print_rank_last_patches", "wall_clock_timer_patch"]
