###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo GPT Output Layer Patches

Patches for replacing GPT output layer with PrimusTurbo implementation.
"""

import importlib.util

from primus.core.patches import PatchContext, get_args


def _is_turbo_parallel_linear_enabled(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo parallel linear is enabled.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
      - use_turbo_parallel_linear == True
    """
    # Check if primus_turbo package is available
    if importlib.util.find_spec("primus_turbo") is None:
        return False

    args = get_args(ctx)
    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))
    use_turbo_parallel_linear = bool(getattr(args, "use_turbo_parallel_linear", False))

    return tp_size == 1 and enable_primus_turbo and use_turbo_parallel_linear
