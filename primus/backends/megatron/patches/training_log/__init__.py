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

The actual patch registration is handled via ``@register_patch``; this
``__init__`` exists mainly to make ``training_log`` a proper package so
that the auto-import logic in ``primus.backends.megatron.patches.__init__``
can discover and import the patch modules automatically.
"""

from primus.backends.megatron.patches.training_log import (  # noqa: F401
    print_rank_last_patches,
    wall_clock_timer_patch,
)

__all__ = ["print_rank_last_patches", "wall_clock_timer_patch"]
