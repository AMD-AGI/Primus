###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Runtime initialization and management for Primus.

This module provides global runtime context and initialization functions
for distributed training, logging, and environment setup.
"""

from .context import RuntimeContext
from .distributed import init_distributed_env
from .logging import init_global_logger, update_module_name

__all__ = [
    "RuntimeContext",
    "init_distributed_env",
    "init_global_logger",
    "update_module_name",
]
