###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron training_log patches package.

This package currently exposes:

    - print_rank_last_patches: scoped print_rank_last hook for training_log
      that injects ROCm memory and throughput statistics.

The actual patch registration is handled by ``print_rank_last_patches`` via
``@register_patch``; this ``__init__`` exists mainly to make
``training_log`` a proper package so that the auto-import logic in
``primus.backends.megatron.patches.__init__`` can discover and import
``print_rank_last_patches`` automatically.
"""

from primus.backends.megatron.patches.training_log import (  # noqa: F401
    print_rank_last_patches,
)

__all__ = ["print_rank_last_patches"]
