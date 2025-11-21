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
"""

import inspect

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
