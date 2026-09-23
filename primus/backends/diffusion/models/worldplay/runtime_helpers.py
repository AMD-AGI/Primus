###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from Tencent HunyuanVideo-1.5 / HY-WorldPlay official implementation.

"""Small runtime helpers required by the WorldPlay transformer."""

from __future__ import annotations

import collections.abc
import warnings
from functools import wraps
from itertools import repeat

import torch


def _ntuple(n):
    def parse(value):
        if isinstance(value, collections.abc.Iterable) and not isinstance(value, str):
            value = tuple(value)
            return tuple(repeat(value[0], n)) if len(value) == 1 else value
        return tuple(repeat(value, n))

    return parse


to_2tuple = _ntuple(2)


def get_infer_state():
    return None


def torch_compile_wrapper():
    """Retain the source decorator API without enabling inference-only compilation."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _flash2_available() -> bool:
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func  # noqa: F401

        return True
    except Exception:
        return False


def _flash3_available() -> bool:
    try:
        from flash_attn_interface import flash_attn_varlen_func  # noqa: F401

        return True
    except Exception:
        return False


def maybe_fallback_attn_mode(attn_mode, infer_state=None, block_idx=None):
    del infer_state, block_idx
    if attn_mode == "flash":
        if _flash3_available():
            return "flash3"
        if _flash2_available():
            return "flash2"
        warnings.warn("Flash Attention is unavailable; falling back to torch attention.")
        return "torch"
    if attn_mode == "flash3" and not _flash3_available():
        warnings.warn("Flash Attention 3 is unavailable; falling back to torch attention.")
        return "torch"
    if attn_mode == "flash2" and not _flash2_available():
        warnings.warn("Flash Attention 2 is unavailable; falling back to torch attention.")
        return "torch"
    if attn_mode == "flex-block-attn":
        try:
            from flex_block_attn import flex_block_attn_func  # noqa: F401
        except Exception as error:
            raise ValueError("flex-block-attn is not installed") from error
    return attn_mode
