###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared requirements and small helpers for the gfx1250 DeepSeek-V4 kernel tests."""

import functools
import importlib

import pytest
import torch


@functools.lru_cache(maxsize=1)
def is_gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # pragma: no cover - defensive
        return False


@functools.lru_cache(maxsize=1)
def gluon_available() -> bool:
    """Gluon kernels need Triton >= 3.9 (``triton.experimental.gluon`` with the gfx1250 dialect)."""
    try:
        import triton

        major, minor = (int(x) for x in triton.__version__.split("+")[0].split(".")[:2])
        if (major, minor) < (3, 9):
            return False
        importlib.import_module("triton.experimental.gluon")
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def turbo_flydsl_available() -> bool:
    try:
        importlib.import_module("flydsl")
        importlib.import_module("primus_turbo.flydsl")
        return True
    except Exception:
        return False


requires_gfx1250 = pytest.mark.skipif(not is_gfx1250(), reason="requires a gfx1250 GPU")
requires_gluon = pytest.mark.skipif(
    not (is_gfx1250() and gluon_available()), reason="requires gfx1250 and Triton >= 3.9 with Gluon"
)
requires_turbo_flydsl = pytest.mark.skipif(
    not (is_gfx1250() and turbo_flydsl_available()), reason="requires gfx1250 and Primus-Turbo with FlyDSL"
)

_INT_OF = {
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.float8_e4m3fn: torch.uint8,
    torch.float8_e5m2: torch.uint8,
}


def as_bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret a floating tensor as integers of the same width (so NaN payloads compare too)."""
    return t.view(_INT_OF[t.dtype]) if t.dtype in _INT_OF else t


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(as_bits(a), as_bits(b))


def bit_diff_count(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((as_bits(a) != as_bits(b)).sum().item())
