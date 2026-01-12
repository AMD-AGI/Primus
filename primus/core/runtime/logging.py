###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Global logging initialization for Primus.

This module handles initialization and management of the logging system for
distributed training.

Key features:
    - One-time initialization with init_global_logger()
    - Dynamic module_name updates for workflow scenarios (pretrain -> sft -> posttrain)
    - Separate log directories per module_name
    - Configurable logging levels per module

Workflow example:
    >>> from primus.core.runtime import init_global_logger, update_module_name
    >>>
    >>> # Initialize for pretrain
    >>> init_global_logger(config, module_name="pretrain")
    >>> # Logs go to: logs/pretrain/rank-0/
    >>>
    >>> # Switch to sft phase
    >>> update_module_name("sft")
    >>> # Logs now go to: logs/sft/rank-0/
"""

import builtins
from typing import Any

from primus.core.utils import logger
from primus.core.utils.env import get_torchrun_env
from primus.modules.module_utils import debug_rank_all, set_logging_rank

# from primus.core.utils.distributed_logging import debug_rank_all, set_logging_rank


def init_worker_logger(primus_config: Any, module_name: str = "primus", module_config=None) -> None:
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
        module_config: Optional module config to read logging levels from.
                       Supports: sink_level, file_sink_level, stderr_sink_level
    """

    # Determine logging levels from module_config if provided
    file_sink_level = "DEBUG"  # Default from module_base.yaml
    stderr_sink_level = "INFO"  # Default from module_base.yaml

    if module_config is not None:
        # Access logging config from params dict
        sink_level = getattr(module_config.params, "sink_level", None)

        # If sink_level is set, it overrides both file and stderr levels
        if sink_level is not None:
            file_sink_level = sink_level
            stderr_sink_level = sink_level
        else:
            # Otherwise, use individual levels if specified
            file_sink_level = getattr(module_config.params, "file_sink_level", "DEBUG")
            stderr_sink_level = getattr(module_config.params, "stderr_sink_level", "INFO")

    dist_env = get_torchrun_env()

    # Create logger configuration
    logger_cfg = logger.LoggerConfig(
        exp_root_path=primus_config.exp_root_path,
        work_group=primus_config.exp_meta_info["work_group"],
        user_name=primus_config.exp_meta_info["user_name"],
        exp_name=primus_config.exp_meta_info["exp_name"],
        module_name=module_name,
        file_sink_level=file_sink_level,
        stderr_sink_level=stderr_sink_level,
        node_ip=dist_env["master_addr"],
        rank=dist_env["rank"],
        world_size=dist_env["world_size"],
    )
    # Setup logger and track file sink handlers
    logger.setup_logger(logger_cfg, is_head=False)

    # Set logging rank for rank-aware logging (log_rank_0, etc.)
    set_logging_rank(dist_env["rank"], dist_env["world_size"])

    # Monkey patch print function for distributed logging
    builtins.print = debug_rank_all

    print(
        f"[Primus:Runtime] Global logger initialized (rank={dist_env['rank']}, world_size={dist_env['world_size']})"
    )


def update_module_name(new_module_name: str, new_module_config=None) -> None:
    """
    Update module_name for the logger (for workflow scenarios).

    This allows changing the logging context when transitioning between
    workflow stages (e.g., pretrain -> sft -> posttrain). It will:
        1. Remove old file sinks
        2. Add new file sinks with updated module_name path
        3. Re-bind module_name to logger
        4. Optionally update logging levels from new_module_config

    Args:
        new_module_name: New module name (e.g., "sft", "posttrain")
        new_module_config: Optional new module config to update logging levels from.
                          Supports: sink_level, file_sink_level, stderr_sink_level

    Raises:
        RuntimeError: If logger has not been initialized yet

    Example:
        >>> init_global_logger(config, module_name="pretrain")
        >>> # ... pretrain phase ...
        >>> update_module_name("sft")  # Switch to sft phase
        >>> # ... sft phase logs to new directory ...
    """
    context = RuntimeContext.get_instance()

    if not context.logger_initialized:
        raise RuntimeError(
            "[Primus:Runtime] Logger must be initialized before updating module_name. "
            "Call init_global_logger() first."
        )

    from loguru import logger as loguru_logger

    from primus.core.utils import logger as logger_utils

    # 1. Remove old file sinks
    for handler_id in context.file_sink_handlers:
        try:
            loguru_logger.remove(handler_id)
        except ValueError:
            # Handler already removed, skip
            pass
    context.file_sink_handlers.clear()

    # 2. Update module_name in stored config
    old_logger_cfg = context.logger_config

    # Determine new logging levels
    file_sink_level = old_logger_cfg.file_sink_level
    stderr_sink_level = old_logger_cfg.stderr_sink_level

    if new_module_config is not None:
        sink_level = new_module_config.params.get("sink_level")
        if sink_level is not None:
            file_sink_level = sink_level
            stderr_sink_level = sink_level
        else:
            if new_module_config.params.get("file_sink_level") is not None:
                file_sink_level = new_module_config.params.get("file_sink_level")
            if new_module_config.params.get("stderr_sink_level") is not None:
                stderr_sink_level = new_module_config.params.get("stderr_sink_level")

    # Create new logger config with updated module_name
    new_logger_cfg = logger.LoggerConfig(
        exp_root_path=old_logger_cfg.exp_root_path,
        work_group=old_logger_cfg.work_group,
        user_name=old_logger_cfg.user_name,
        exp_name=old_logger_cfg.exp_name,
        module_name=new_module_name,
        file_sink_level=file_sink_level,
        stderr_sink_level=stderr_sink_level,
        node_ip=old_logger_cfg.node_ip,
        rank=old_logger_cfg.rank,
        world_size=old_logger_cfg.world_size,
    )

    # Store updated config
    context.logger_config = new_logger_cfg

    # 3. Re-bind module_name to logger (this updates the extra field)
    # Note: We need to access the global _logger from logger_utils
    import primus.core.utils.logger as logger_module

    logger_module._logger = logger_module._logger.bind(module_name=new_module_name)

    # 4. Add new file sinks with updated path
    log_path = f"{new_logger_cfg.exp_root_path}/logs/{new_module_name}/rank-{new_logger_cfg.rank}"
    logger_utils.create_path_if_not_exists(log_path)

    sinked_levels = ["debug", "info", "warning", "error"]
    for sinked_level in sinked_levels:
        handler_id = logger_utils.add_file_sink(
            logger_module._logger,
            log_path,
            new_logger_cfg.file_sink_level,
            "",  # prefix
            False,  # is_head
            sinked_level,
        )
        if handler_id is not None:
            context.file_sink_handlers.append(handler_id)

    print(
        f"[Primus:Runtime] Logger module_name updated: {old_logger_cfg.module_name} -> {new_module_name} "
        f"(rank={context.rank})"
    )
