###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Logger Patch

This patch ensures TorchTitan uses a named logger instead of the root logger,
which allows proper source file and line number tracking in Primus logs.

IMPORTANT: This patch must be applied BEFORE importing any TorchTitan modules.
"""

import logging

import torchtitan.tools.logging as titan_logging


def patch_torchtitan_logger() -> None:
    """
    Patch TorchTitan's logger to use a named logger instead of root logger.

    This allows Primus's InterceptHandler to correctly extract module names
    and line numbers from TorchTitan logs.
    """
    # Replace TorchTitan's root logger with a properly-named logger
    titan_logging.logger = logging.getLogger("torchtitan")

    # Disable TorchTitan's logger initialization (Primus manages logging)
    titan_logging.init_logger = lambda: None
