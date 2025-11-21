###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import inspect

from primus.core.utils import logger

_rank = 0
_world_size = 1


######################################################log before torch distributed initialized
def set_logging_rank(rank, world_size):
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
