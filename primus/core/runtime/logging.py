###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Global logging initialization for Primus.

This module handles one-time initialization of the logging system for
distributed training.
"""

import builtins

from primus.core.config.primus_config import PrimusConfig
from primus.core.utils import logger
from primus.core.utils.distributed_logging import debug_rank_all, set_logging_rank

from .context import RuntimeContext


def init_global_logger(
    primus_config: PrimusConfig,
    module_name: str = "primus",
    file_sink_level: str = "INFO",
    stderr_sink_level: str = "INFO",
) -> None:
    """
    Initialize global logging system for distributed training.

    This function should be called once at the start of training to set up
    the logging system. It:
        1. Configures file and stderr logging
        2. Sets up rank-aware logging (log_rank_0, etc.)
        3. Monkey patches print function for distributed logging

    Args:
        primus_config: Primus configuration object
        module_name: Name for the logger (default: "primus")
        file_sink_level: Log level for file output
        stderr_sink_level: Log level for stderr output
    """
    context = RuntimeContext.get_instance()

    # Skip if already initialized
    if context.logger_initialized:
        print("[Primus:Runtime] Logger already initialized")
        return

    # Ensure distributed environment is initialized
    if not context.distributed_initialized:
        raise RuntimeError(
            "[Primus:Runtime] Distributed environment must be initialized before logger. "
            "Call init_distributed_env() first."
        )

    # Store primus_config in context
    context.set_primus_config(primus_config)

    # Create logger configuration
    logger_cfg = logger.LoggerConfig(
        exp_root_path=primus_config.exp_root_path,
        work_group=primus_config.exp_meta_info["work_group"],
        user_name=primus_config.exp_meta_info["user_name"],
        exp_name=primus_config.exp_meta_info["exp_name"],
        module_name=module_name,
        file_sink_level=file_sink_level,
        stderr_sink_level=stderr_sink_level,
        node_ip=context.platform.get_addr(),
        rank=context.rank,
        world_size=context.world_size,
    )

    # Setup logger
    logger.setup_logger(logger_cfg, is_head=False)

    # Set logging rank for rank-aware logging (log_rank_0, etc.)
    set_logging_rank(context.rank, context.world_size)

    # Monkey patch print function for distributed logging
    builtins.print = debug_rank_all

    context.logger_initialized = True

    print(
        f"[Primus:Runtime] Global logger initialized (rank={context.rank}, world_size={context.world_size})"
    )
