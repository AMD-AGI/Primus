###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Distributed logging utilities for Primus.

This module provides rank-aware logging functions that can be used before
PyTorch distributed is initialized. The logging functions use global rank
variables that are set once at the start of training.

Functions:
    - log_rank_0: Log only from rank 0
    - log_rank_last: Log only from last rank
    - log_rank_all: Log from all ranks
    - log_kv_rank_0: Log key-value pairs from rank 0
    - debug_rank_0: Debug log from rank 0
    - debug_rank_all: Debug log from all ranks
    - warning_rank_0: Warning log from rank 0
    - error_rank_0: Error log from rank 0
    - log_dict_aligned: Log dictionary/namespace in aligned column format
"""

import inspect
from types import SimpleNamespace
from typing import Any, Dict, Union

from primus.core.utils import logger

# Global rank variables (set once at initialization)
_rank = 0
_world_size = 1


def set_logging_rank(rank: int, world_size: int):
    """
    Set global rank variables for distributed logging.

    This should be called once during initialization (in init_global_logger).

    Args:
        rank: Global rank of this process
        world_size: Total number of processes
    """
    global _rank
    global _world_size
    _rank = rank
    _world_size = world_size


def log_rank_0(msg, *args, **kwargs):
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if _rank == 0:
        log_func(msg, module_name, function_name, line)


def log_rank_last(msg, *args, **kwargs):
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if _rank == _world_size - 1:
        log_func(msg, module_name, function_name, line)


def log_rank_all(msg, *args, **kwargs):
    log_func = logger.info_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    log_func(msg, module_name, function_name, line)


def log_kv_rank_0(key, value):
    log_func = logger.log_kv_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if _rank == 0:
        log_func(key, value, module_name, function_name, line)


def debug_rank_0(msg, *args, **kwargs):
    log_func = logger.debug_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if _rank == 0:
        log_func(msg, module_name, function_name, line)


def debug_rank_all(msg, *args, **kwargs):
    log_func = logger.debug_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    log_func(msg, module_name, function_name, line)


def warning_rank_0(msg, *args, **kwargs):
    log_func = logger.warning_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if _rank == 0:
        log_func(msg, module_name, function_name, line)


def error_rank_0(msg, *args, **kwargs):
    log_func = logger.error_with_caller

    caller = inspect.stack()[1]
    caller_frame = caller.frame
    function_name = caller_frame.f_code.co_name
    module_name = caller_frame.f_globals["__name__"].split(".")[-1]
    line = caller.lineno

    if _rank == 0:
        log_func(msg, module_name, function_name, line)


def log_dict_aligned(
    title: str,
    data: Union[Dict[str, Any], SimpleNamespace],
    indent: str = "  ",
    rank_filter: str = "rank_0",
):
    """
    Log a dictionary or namespace in an aligned column format.

    This function logs key-value pairs with aligned columns for better readability.
    Values are aligned vertically at the same column position.

    Args:
        title: Title to display before the data
        data: Dictionary or SimpleNamespace to log
        indent: Indentation prefix for each line (default: "  ")
        rank_filter: Which ranks should log ("rank_0", "rank_all", "rank_last")

    Example output:
        Backend args:
          account_for_embedding_in_pipeline_split : False
          adam_beta1                              : 0.9
          adam_eps                                : 1e-08
          add_bias_linear                         : False

    Usage:
        # Log from rank 0 only (default)
        log_dict_aligned("Config", my_config)

        # Log from all ranks
        log_dict_aligned("Local state", state_dict, rank_filter="rank_all")

        # Log distributed environment info
        log_dict_aligned("Distributed info", dist_env)
    """
    # Select appropriate log function based on rank filter
    if rank_filter == "rank_0":
        log_func = log_rank_0
    elif rank_filter == "rank_all":
        log_func = log_rank_all
    elif rank_filter == "rank_last":
        log_func = log_rank_last
    else:
        log_func = log_rank_0  # Default to rank 0

    log_func(f"{title}:")

    # Convert SimpleNamespace to dict if needed
    if isinstance(data, SimpleNamespace):
        data_dict = vars(data)
    else:
        data_dict = data

    if not data_dict:
        log_func(f"{indent}(empty)")
        return

    # Find the longest key for alignment
    max_key_length = max(len(str(key)) for key in data_dict.keys())

    # Log each key-value pair with alignment
    for key, value in sorted(data_dict.items()):
        log_func(f"{indent}{str(key):<{max_key_length}} : {value}")
