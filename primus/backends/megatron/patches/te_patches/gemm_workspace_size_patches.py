###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine GEMM workspace size (ROCm).

On ROCm, hipBLASLt can pick split-K algorithms for large-K / small-N GEMMs whose workspace
exceeds Transformer Engine's 64 MiB default, and the GEMM fails with "HIPBLASLT Error: 6" (seen
on gfx950). The helper applies on every ROCm arch: it costs up to the set size per device and
stream, so only models that hit the error call it.
``enlarge_te_gemm_workspace`` raises TE's workspace to ``PRIMUS_TE_GEMM_WORKSPACE_MB`` (default
512). A helper module (no patch is registered here): a model whose GEMM shapes need it calls it from its
own patch, so other models keep TE's default. A no-op on CUDA, without Transformer Engine, with
``PRIMUS_TE_GEMM_WORKSPACE_MB`` <= 0, or when the workspace is already at least that large. It
raises ``ValueError`` for a non-integer ``PRIMUS_TE_GEMM_WORKSPACE_MB`` and ``RuntimeError`` on
ROCm when the installed TE has neither known workspace layout (set the variable to 0 to skip it).
"""

import importlib
import importlib.util
import os
import sys

import torch

from primus.core.utils.module_utils import log_rank_0

DEFAULT_WORKSPACE_MB = 512


def workspace_mb_from_env() -> int:
    """``PRIMUS_TE_GEMM_WORKSPACE_MB`` (default ``DEFAULT_WORKSPACE_MB``) as an integer. A value of
    0 or below means "keep TE's default"; a value that is not an integer raises ``ValueError``
    naming the variable."""
    raw = os.environ.get("PRIMUS_TE_GEMM_WORKSPACE_MB", str(DEFAULT_WORKSPACE_MB)).strip()
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"PRIMUS_TE_GEMM_WORKSPACE_MB={raw!r}: expected an integer number of MiB (e.g. {DEFAULT_WORKSPACE_MB})"
        ) from None


def te_cpp_extensions_gemm_layout_supported(te_gemm) -> bool:
    """Whether ``transformer_engine.pytorch.cpp_extensions.gemm`` has the layout this helper
    changes first (TE 2.17 ROCm build, the project image): the size function
    ``get_cublas_workspace_size_bytes()`` and the ``lru_cache``'d ``get_cublas_workspace(device,
    ub, grouped_gemm)`` that allocates the buffer ``general_gemm`` passes to hipBLASLt."""
    return callable(getattr(te_gemm, "get_cublas_workspace_size_bytes", None)) and hasattr(
        getattr(te_gemm, "get_cublas_workspace", None), "cache_clear"
    )


def te_module_base_layout_supported(te_base) -> bool:
    """Whether ``transformer_engine.pytorch.module.base`` has the older layout (TE 2.8 ROCm): the
    size function ``get_cublas_workspace_size_bytes()`` read at allocation time by ``get_workspace()``
    and ``get_multi_stream_cublas_workspace()``, which cache the buffers in the module globals
    ``_cublas_workspace`` and ``_multi_stream_cublas_workspace``."""
    return (
        callable(getattr(te_base, "get_cublas_workspace_size_bytes", None))
        and callable(getattr(te_base, "get_workspace", None))
        and hasattr(te_base, "_cublas_workspace")
    )


def _te_installed() -> bool:
    """Whether Transformer Engine is installed (found by the import system; not imported)."""
    try:
        return importlib.util.find_spec("transformer_engine") is not None
    except ValueError:  # loaded with no __spec__ (e.g. a module set in sys.modules): it exists
        return sys.modules.get("transformer_engine") is not None


def _import_or_none(name):
    """The module, or None when it does not exist: a ``ModuleNotFoundError`` for ``name`` itself or
    one of its parent packages (a TE without that submodule). Any other failure (a broken install:
    an undefined symbol or missing ``.so``, a missing dependency of the module) raises, with its
    original traceback."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing and (name == missing or name.startswith(missing + ".")):
            return None
        raise


def enlarge_te_gemm_workspace(log_tag: str = "te_gemm_workspace") -> int:
    """Set TE's cuBLAS / hipBLASLt workspace to ``PRIMUS_TE_GEMM_WORKSPACE_MB`` MiB on ROCm.

    Returns the effective workspace size in bytes: the new size if enlarged, the current size if
    it is already at least that large, or 0 when not applicable (CUDA,
    ``PRIMUS_TE_GEMM_WORKSPACE_MB`` <= 0, or Transformer Engine not installed; each logged except
    CUDA). Two TE layouts are handled: ``cpp_extensions.gemm`` (TE 2.17) and ``module.base``
    (TE 2.8); each one this TE has is enlarged (the smaller resulting size is returned). A TE with
    neither raises ``RuntimeError`` (the caller needs the larger workspace;
    ``PRIMUS_TE_GEMM_WORKSPACE_MB=0`` opts out).
    ``PRIMUS_TE_GEMM_WORKSPACE_MB`` must be an integer (MiB); 0 or below leaves TE's default."""
    if torch.version.hip is None:
        return 0
    size = workspace_mb_from_env() * 1024 * 1024
    if size <= 0:
        log_rank_0(
            f"[Patch:{log_tag}] PRIMUS_TE_GEMM_WORKSPACE_MB <= 0: TE GEMM workspace left at its default"
        )
        return 0
    # TE is heavy and optional: imported only here (by name, from sys.modules when loaded)
    if not _te_installed():
        log_rank_0(
            f"[Patch:{log_tag}] WARNING: Transformer Engine not installed; TE GEMM workspace not changed"
        )
        return 0
    # every layout this TE has is enlarged: a build with both would otherwise keep the old size for
    # the buffers allocated through the other one
    te_gemm = _import_or_none("transformer_engine.pytorch.cpp_extensions.gemm")
    te_base = _import_or_none("transformer_engine.pytorch.module.base")
    sizes = []
    if te_gemm is not None and te_cpp_extensions_gemm_layout_supported(te_gemm):
        sizes.append(_enlarge_cpp_extensions_gemm_workspace(te_gemm, size, log_tag))
    if te_base is not None and te_module_base_layout_supported(te_base):
        sizes.append(_enlarge_module_base_workspace(te_base, size, log_tag))
    if sizes:
        return min(sizes)
    # the model asked for the larger workspace because its GEMMs fail without it: going on with
    # TE's default would fail later, far from the cause ("HIPBLASLT Error: 6")
    raise RuntimeError(
        f"[Patch:{log_tag}] this Transformer Engine has neither cpp_extensions.gemm's "
        "get_cublas_workspace_size_bytes / cached get_cublas_workspace nor module.base's "
        "get_cublas_workspace_size_bytes / get_workspace, so the TE GEMM workspace cannot be "
        f"raised to {size >> 20} MiB. Use a TE with one of those layouts (TE 2.8 or 2.17 ROCm), "
        "or set PRIMUS_TE_GEMM_WORKSPACE_MB=0 to keep TE's default workspace."
    )


def _warn_allocated(log_tag: str) -> None:
    # buffers already handed out (e.g. kept by comm-overlap code) keep their old size: the helper
    # must run before the first GEMM (the before_train patches do)
    log_rank_0(
        f"[Patch:{log_tag}] WARNING: TE GEMM workspace already allocated; "
        "references to it taken before this call keep the old size"
    )


def _enlarge_cpp_extensions_gemm_workspace(te_gemm, size: int, log_tag: str) -> int:
    """The TE 2.17 layout: replace ``cpp_extensions.gemm.get_cublas_workspace_size_bytes`` and
    clear the ``get_cublas_workspace`` cache so the next GEMM allocates at the new size."""
    current = te_gemm.get_cublas_workspace_size_bytes()
    if current >= size:
        return current
    info = getattr(te_gemm.get_cublas_workspace, "cache_info", None)
    if callable(info) and info().currsize > 0:
        _warn_allocated(log_tag)
    te_gemm.get_cublas_workspace_size_bytes = lambda: size
    te_gemm.get_cublas_workspace.cache_clear()
    log_rank_0(f"[Patch:{log_tag}] TE GEMM workspace -> {size >> 20} MiB")
    return size


def _enlarge_module_base_workspace(te_base, size: int, log_tag: str) -> int:
    """The TE 2.8 layout: replace ``module.base.get_cublas_workspace_size_bytes`` (looked up in the
    module globals when ``get_workspace`` / ``get_multi_stream_cublas_workspace`` allocate) and drop
    the cached buffers so the next GEMM allocates them at the new size."""
    current = te_base.get_cublas_workspace_size_bytes()
    if current >= size:
        return current
    if getattr(te_base, "_cublas_workspace", None) is not None or bool(
        getattr(te_base, "_multi_stream_cublas_workspace", None)
    ):
        _warn_allocated(log_tag)
    te_base.get_cublas_workspace_size_bytes = lambda: size
    te_base._cublas_workspace = None
    if hasattr(te_base, "_multi_stream_cublas_workspace"):
        te_base._multi_stream_cublas_workspace = []
    log_rank_0(f"[Patch:{log_tag}] TE GEMM workspace -> {size >> 20} MiB (module.base layout)")
    return size
