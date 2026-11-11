###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TE hipBLASLt GEMM workspace size on ROCm (``PRIMUS_TE_GEMM_WORKSPACE_MB``).

Seen on gfx950: hipBLASLt picks split-K algorithms for large-K / small-N GEMMs whose
workspace exceeds TE's 64 MiB default and fails with "HIPBLASLT Error: 6". A model
patch calls :func:`enlarge_te_gemm_workspace` to raise it (GLM-5.3, Muse Spark); it acts
on every ROCm GPU (no architecture check), only for the models whose patch calls it
(other models keep TE's default). The size is ``PRIMUS_TE_GEMM_WORKSPACE_MB``, default
512. It only enlarges: a TE size already >= the request is kept, so a value below TE's
default changes nothing (there is no opt-out below TE's size). Off ROCm nothing is done
and the variable is not read.

``strict=True`` (Muse Spark): a set value must be ASCII digits (optional surrounding
whitespace) from 1 to ``MAX_WORKSPACE_MB``; anything else raises ``ValueError`` naming
the variable. ``strict=False`` (GLM-5.3) keeps GLM-5.3's original reading, a plain
``int()``, so its existing launch settings behave as before.
"""

import os
import re

from primus.core.utils.module_utils import log_rank_0

WORKSPACE_ENV = "PRIMUS_TE_GEMM_WORKSPACE_MB"
DEFAULT_WORKSPACE_MB = 512
# 64 GiB: far above the few hundred MiB a split-K GEMM workspace needs; a larger value is
# taken as a typo
MAX_WORKSPACE_MB = 64 * 1024
_ASCII_DIGITS = re.compile(r"[0-9]+")


def parse_workspace_mb(value) -> int:
    """``PRIMUS_TE_GEMM_WORKSPACE_MB`` as MiB: None (unset) is ``DEFAULT_WORKSPACE_MB``;
    a set value must be ASCII digits with optional surrounding whitespace, from 1 to
    ``MAX_WORKSPACE_MB``. Anything else (blank, ``1.5``, ``1e3``, ``+5``, ``0``, non-ASCII
    digits, ``99999999999``) raises ``ValueError`` naming the variable."""
    if value is None:
        return DEFAULT_WORKSPACE_MB
    text = str(value).strip()
    # bounded digit count before int(): no int() of a long string (Python's 4300-digit
    # limit would raise a bare ValueError)
    digits = text.lstrip("0") or "0"
    ok = _ASCII_DIGITS.fullmatch(text) and len(digits) <= len(str(MAX_WORKSPACE_MB))
    mb = int(digits) if ok else 0
    if not 1 <= mb <= MAX_WORKSPACE_MB:
        raise ValueError(
            f"{WORKSPACE_ENV} must be an integer number of MiB from 1 to {MAX_WORKSPACE_MB}, got {value!r}"
        )
    return mb


def enlarge_te_gemm_workspace(tag: str, strict: bool = True) -> None:
    """Raise TE's GEMM workspace to ``PRIMUS_TE_GEMM_WORKSPACE_MB`` (default 512 MiB) on
    ROCm; ``tag`` names the calling patch in the log. A larger TE size is kept. ``strict``
    selects :func:`parse_workspace_mb` (True) or a plain ``int()`` (False, GLM-5.3's
    original rule: e.g. ``0`` or a negative value is accepted and changes nothing)."""
    import torch

    if torch.version.hip is None:
        return
    value = os.environ.get(WORKSPACE_ENV)
    if strict:
        mb = parse_workspace_mb(value)
    else:
        mb = int(value if value is not None else str(DEFAULT_WORKSPACE_MB))
    size = mb * 1024 * 1024
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm

    if te_gemm.get_cublas_workspace_size_bytes() >= size:
        return
    te_gemm.get_cublas_workspace_size_bytes = lambda: size
    te_gemm.get_cublas_workspace.cache_clear()
    log_rank_0(f"[Patch:{tag}] TE GEMM workspace -> {size >> 20} MiB")
