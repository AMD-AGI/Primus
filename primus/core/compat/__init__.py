###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Compatibility workarounds for upstream framework bugs.

These are applied early in the training lifecycle, before backend imports,
to prevent known import-time failures.
"""

from primus.core.compat.torch_inductor import apply_inductor_compat

__all__ = ["apply_inductor_compat"]
