###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
TorchTitan Logger Patch
=======================

This patch redirects TorchTitan's logging to Primus's unified logger.

Purpose:
--------
TorchTitan uses its own logger instance (torchtitan.tools.logging.logger),
which by default logs to stdout/stderr without integration with Primus's
logging infrastructure. This patch replaces TorchTitan's logger with Primus's
logger to ensure consistent log formatting, filtering, and destination.

Behavior:
---------
1. Replaces `torchtitan.tools.logging.logger` with Primus's logger
2. Disables `torchtitan.tools.logging.init_logger()` to prevent re-initialization

Configuration:
--------------
This patch is always enabled for the TorchTitan backend and runs during
the "setup" phase before any TorchTitan modules are initialized.

Usage:
------
This patch is automatically applied - no configuration needed.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    patch_id="torchtitan.logger",
    backend="torchtitan",
    phase="setup",
    description="Redirect TorchTitan logger to Primus unified logger",
    condition=lambda ctx: True,  # Always enabled
)
def patch_torchtitan_logger(ctx: PatchContext) -> None:
    """
    Replace TorchTitan's logger with Primus's unified logger.
    """
    from primus.core.utils.logger import _logger as primus_logger

    log_rank_0(
        "[Patch:torchtitan.logger] " "Monkey patching TorchTitan logger to use Primus unified logger...",
    )
    import torchtitan.tools.logging as titan_logging

    # Replace TorchTitan's logger with Primus's logger
    titan_logging.logger = primus_logger

    # Disable TorchTitan's logger initialization
    titan_logging.init_logger = lambda: None

    log_rank_0(
        "[Patch:torchtitan.logger] " "TorchTitan logger successfully redirected to Primus logger.",
    )
