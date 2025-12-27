###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan backend integration for Primus.

This package wires TorchTitan components (models, protocols, Primus-Turbo
extensions) into the Primus stack.

On import, it also triggers registration of TorchTitan-specific patches so
that they can be managed via the unified Primus patch system.
"""

# Trigger registration of all TorchTitan patches (classic attention, etc.)
import primus.backends.torchtitan.patches  # noqa: F401
