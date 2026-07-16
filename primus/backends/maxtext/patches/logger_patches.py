###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Logger Patch

This patch redirects MaxText's logging to Primus's unified logger.
"""

import logging

from primus.core.patches import PatchContext, register_patch
from primus.core.utils import checker
from primus.core.utils.logger import _logger as primus_logger
from primus.core.utils.module_utils import error_rank_0, log_rank_0, warning_rank_0


def _resolve_max_logging():
    """Resolve MaxText's ``max_logging`` module across MaxText versions.

    v26.4+ ships it at ``maxtext.utils.max_logging``; v26.3 and earlier expose
    it as ``MaxText.max_logging``. Returns ``None`` if neither is importable.
    """
    try:
        from maxtext.utils import max_logging as maxtext_logging

        return maxtext_logging
    except ImportError:
        try:
            import MaxText.max_logging as maxtext_logging

            return maxtext_logging
        except ImportError:
            return None


@register_patch(
    patch_id="maxtext.logger",
    backend="maxtext",
    phase="setup",
    description="Redirect MaxText logger to Primus unified logger",
    condition=lambda ctx: True,  # Always enabled
)
def patch_maxtext_logger(ctx: PatchContext) -> None:
    """
    Replace MaxText's logger with Primus's unified logger.
    """
    log_rank_0("[Patch:maxtext.logger] Patching MaxText logger...")

    maxtext_logging = _resolve_max_logging()
    if maxtext_logging is None:
        error_rank_0("[Patch:maxtext.logger] Failed to import MaxText's logging module.")
    elif hasattr(maxtext_logging, "log"):
        maxtext_logging.log = primus_logger.info
        warning_rank_0("[Patch:maxtext.logger] MaxText logger patched successfully.")
    else:
        error_rank_0("[Patch:maxtext.logger] MaxText logging module does not have a 'log' function.")

    # Configure JAX logger level based on module config
    level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    try:
        module_config = ctx.extra.get("module_config")
        stderr_sink_level = getattr(module_config.params, "stderr_sink_level", "INFO")
        checker.check_true(stderr_sink_level in level_map)
        logging_level = level_map[stderr_sink_level]

        jax_loggers = [logging.getLogger("jax"), logging.getLogger("jaxlib")]
        for jax_logger in jax_loggers:
            jax_logger.setLevel(logging_level)

        log_rank_0(f"[Patch:maxtext.logger] Set JAX logging level to {logging_level}")
    except Exception as e:
        warning_rank_0(f"[Patch:maxtext.logger] Failed to set JAX logging level: {e}")
