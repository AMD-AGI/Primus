###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Utility helpers for TorchTitan Primus-Turbo patches.

These helpers are shared across TorchTitan Primus-Turbo patch modules
to keep condition logic consistent (e.g., checking whether Primus-Turbo is
enabled in the job config).
"""

from primus.core.patches import PatchContext, get_args


def get_primus_turbo_config(ctx: PatchContext):
    """
    Return the Primus-Turbo config object from module_config.params,
    or None if it is not available.
    """
    return getattr(get_args(ctx), "primus_turbo", None)
