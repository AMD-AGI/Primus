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
(other models keep TE's default; Muse Spark's ``after_train`` patch calls
:func:`restore_te_gemm_workspace`, so a later model in the same process gets TE's default
back). The size is ``PRIMUS_TE_GEMM_WORKSPACE_MB``, default
512, per workspace: TE allocates it once for the plain GEMM workspace, once per stream for
the grouped-GEMM (MoE expert) workspaces and 3x for the UB workspace, so the GPU memory it
costs is the size times the number of workspaces (~2.5 GiB at 512 MiB with TE 2.17's 4
streams). It only enlarges: a TE size already >= the request is kept, so a value below TE's
default changes nothing (there is no opt-out below TE's size). Off ROCm nothing is done
and the variable is not read.

``strict=True`` (Muse Spark): a set value must be ASCII digits (optional surrounding
whitespace) from 1 to ``MAX_WORKSPACE_MB``; anything else raises ``ValueError`` naming
the variable. ``strict=False`` (GLM-5.3): a plain ``int()`` of the value with no range
check (0 or a negative value changes nothing, a value above ``MAX_WORKSPACE_MB`` is used
as given); a value ``int()`` rejects raises ``ValueError`` naming the variable.

Call it before the first TE GEMM (the patches run in ``before_train``, before the model
runs): the cached workspaces are dropped and reallocated at the new size, but a buffer
already captured in a CUDA graph or registered with userbuffers keeps its old size.
"""

import os
import re

from primus.core.utils.module_utils import log_rank_0

WORKSPACE_ENV = "PRIMUS_TE_GEMM_WORKSPACE_MB"
DEFAULT_WORKSPACE_MB = 512
# The size applies to EACH cached TE workspace: the plain GEMM one, one per cuBLAS / hipBLASLt
# stream for grouped GEMM (tex.get_num_cublas_streams(): 4 in TE 2.17+rocm), and the UB one
# (3x the size) when userbuffers are on. The default 512 MiB thus costs ~2.5 GiB per GPU
# (1 + 4 workspaces). 4 GiB per workspace (~20 GiB per GPU at 5 workspaces) is far above
# what a split-K GEMM needs; a larger value is taken as a typo.
MAX_WORKSPACE_MB = 4 * 1024
MiB = 1024 * 1024
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
    selects :func:`parse_workspace_mb` (True) or a plain ``int()`` (False: no range check,
    e.g. ``0`` or a negative value is accepted and changes nothing). Call it before the
    first TE GEMM (see the module docstring)."""
    import torch

    if torch.version.hip is None:
        return
    value = os.environ.get(WORKSPACE_ENV)
    if strict:
        mb = parse_workspace_mb(value)
    elif value is None:
        mb = DEFAULT_WORKSPACE_MB
    else:
        try:
            mb = int(value)
        except ValueError as exc:
            raise ValueError(f"{WORKSPACE_ENV} must be an integer number of MiB, got {value!r}") from exc
    size = mb * MiB
    te_mod, reset = _te_workspace_owner()
    if te_mod.get_cublas_workspace_size_bytes() >= size:
        return
    current = te_mod.get_cublas_workspace_size_bytes
    original = getattr(current, "__wrapped__", current)  # TE's own, also after an earlier enlarge

    def enlarged_size():
        return size

    enlarged_size.__wrapped__ = original  # TE's own getter, for restore_te_gemm_workspace
    enlarged_size.enlarged_by = tag  # the patch that enlarged it (see enlarged_by)
    te_mod.get_cublas_workspace_size_bytes = enlarged_size
    reset()
    log_rank_0(f"[Patch:{tag}] TE GEMM workspace -> {mb} MiB ({te_mod.__name__})")


def enlarged_by():
    """The tag of the patch whose :func:`enlarge_te_gemm_workspace` call is in effect, or None
    (not enlarged, off ROCm, TE not imported). Does not import TE."""
    import sys

    for name in ("transformer_engine.pytorch.cpp_extensions.gemm", "transformer_engine.pytorch.module.base"):
        te_mod = sys.modules.get(name)
        tag = getattr(getattr(te_mod, "get_cublas_workspace_size_bytes", None), "enlarged_by", None)
        if tag is not None:
            return tag
    return None


def restore_te_gemm_workspace(tag: str) -> bool:
    """Undo :func:`enlarge_te_gemm_workspace`: put TE's own size getter back and drop the
    cached workspaces, so a later model in the same process allocates TE's default again.
    Called by the model's ``after_train`` patch. True when something was restored; a no-op
    off ROCm or when the getter was not enlarged."""
    import torch

    if torch.version.hip is None:
        return False
    te_mod, reset = _te_workspace_owner()
    original = getattr(te_mod.get_cublas_workspace_size_bytes, "__wrapped__", None)
    if original is None:
        return False
    te_mod.get_cublas_workspace_size_bytes = original
    reset()
    log_rank_0(f"[Patch:{tag}] TE GEMM workspace restored to TE's default ({te_mod.__name__})")
    return True


def _te_workspace_owner():
    """The TE module that owns ``get_cublas_workspace_size_bytes`` and a function that drops
    its cached workspaces, so the next GEMM allocates them at the new size.

    * TE >= 2.1x: ``pytorch.cpp_extensions.gemm`` (``get_cublas_workspace`` is an
      ``lru_cache`` serving the plain, grouped-GEMM multi-stream and UB workspaces);
    * older TE (e.g. 2.8): ``pytorch.module.base`` (globals ``_cublas_workspace`` and
      ``_multi_stream_cublas_workspace``).

    Neither layout: ``RuntimeError`` naming the TE version (not a silent no-op)."""
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm

    if hasattr(te_gemm, "get_cublas_workspace_size_bytes") and hasattr(
        getattr(te_gemm, "get_cublas_workspace", None), "cache_clear"
    ):
        return te_gemm, te_gemm.get_cublas_workspace.cache_clear
    import transformer_engine.pytorch.module.base as te_base

    if hasattr(te_base, "get_cublas_workspace_size_bytes"):

        def reset():
            if hasattr(te_base, "_cublas_workspace"):
                te_base._cublas_workspace = None
            if hasattr(te_base, "_multi_stream_cublas_workspace"):
                te_base._multi_stream_cublas_workspace = []

        return te_base, reset
    import transformer_engine

    raise RuntimeError(
        f"{WORKSPACE_ENV}: Transformer Engine {getattr(transformer_engine, '__version__', '?')} has "
        "get_cublas_workspace_size_bytes neither in pytorch.cpp_extensions.gemm nor in "
        "pytorch.module.base; the GEMM workspace cannot be enlarged"
    )
