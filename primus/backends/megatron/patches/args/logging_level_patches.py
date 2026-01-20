###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.args.logging_level",
    backend="megatron",
    phase="build_args",
    description="Set logging level based on stderr_sink_level",
)
def patch_logging_level(ctx: PatchContext):
    """
    Configure logging level.

    Maps stderr_sink_level (DEBUG/INFO/WARNING/ERROR) to numeric
    logging level and sets args.logging_level accordingly.
    """
    args = ctx.extra.get("args")
    if not args:
        return

    level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    stderr_level = getattr(args, "stderr_sink_level", "INFO")
    if stderr_level not in level_map:
        log_rank_0(
            f"[Patch:megatron.args.logging_level][WARN] "
            f"Invalid stderr_sink_level '{stderr_level}', using INFO"
        )
        stderr_level = "INFO"

    logging_level = level_map[stderr_level]

    if hasattr(args, "logging_level") and args.logging_level is not None:
        if args.logging_level != logging_level:
            log_rank_0(
                "[Patch:megatron.args.logging_level][WARN] args.logging_level is deprecated; "
                f"setting to {logging_level} (from stderr_sink_level={stderr_level})"
            )

    args.logging_level = logging_level
    log_rank_0(f"[Patch:megatron.args.logging_level] logging_level={logging_level} ({stderr_level})")
