###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine GEMM workspace size (ROCm).

On ROCm, hipBLASLt can pick split-K algorithms for large-K / small-N GEMMs whose workspace
exceeds Transformer Engine's 64 MiB default, and the GEMM fails with "HIPBLASLT Error: 6" (seen
on gfx950). ``enlarge_te_gemm_workspace`` raises TE's workspace to
``PRIMUS_TE_GEMM_WORKSPACE_MB`` (default 512). It costs up to that size per device and stream, so
this is a helper module (no patch is registered here): a model whose GEMM shapes need it calls it
from its own patch, and other models keep TE's default.

The helper is a no-op on CUDA, without Transformer Engine, with ``PRIMUS_TE_GEMM_WORKSPACE_MB``
<= 0, or when the workspace is already at least that large. On ROCm it raises ``ValueError`` for a
non-integer ``PRIMUS_TE_GEMM_WORKSPACE_MB`` and ``RuntimeError`` when the installed TE has neither
known workspace layout (set the variable to 0 or below to skip it); on CUDA the variable is not
read.
"""

import importlib
import importlib.util
import os
import sys

import torch

from primus.backends.megatron.patches.te_patches.hipblaslt_workspace_patches import (
    import_or_none,
    resize_te_gemm_workspace,
    te_cpp_extensions_gemm_layout_supported,
    te_module_base_layout_supported,
    te_workspace_modules,
)
from primus.core.utils.module_utils import log_rank_0

# PRIMUS_TE_GEMM_WORKSPACE_MB is GLM-5.3's variable (main's glm5_next patch read it, in MiB despite
# its name) and keeps its name so GLM-5.3 runs see no rename. The resize itself is the one in
# hipblaslt_workspace_patches.py, also used by PRIMUS_TE_GEMM_WORKSPACE_MIB (opt-in, any model,
# unset by default). Both only ever enlarge the workspace: with both set the larger value applies.
DEFAULT_WORKSPACE_MB = 512

__all__ = [
    "DEFAULT_WORKSPACE_MB",
    "enlarge_te_gemm_workspace",
    "te_cpp_extensions_gemm_layout_supported",
    "te_module_base_layout_supported",
    "workspace_mb_from_env",
]


def workspace_mb_from_env() -> int:
    """``PRIMUS_TE_GEMM_WORKSPACE_MB`` in MiB (unset or empty: ``DEFAULT_WORKSPACE_MB``) as an
    integer. A value of 0 or below means "keep TE's default"; a value that is not an integer raises
    ``ValueError`` naming the variable."""
    # unset or empty (an unset template such as ${PRIMUS_TE_GEMM_WORKSPACE_MB:}): the default
    raw = os.environ.get("PRIMUS_TE_GEMM_WORKSPACE_MB", "").strip() or str(DEFAULT_WORKSPACE_MB)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"PRIMUS_TE_GEMM_WORKSPACE_MB={raw!r}: expected an integer number of MiB (e.g. {DEFAULT_WORKSPACE_MB})"
        ) from None


def _te_installed() -> bool:
    """Whether Transformer Engine is installed (found by the import system; not imported)."""
    try:
        return importlib.util.find_spec("transformer_engine") is not None
    except ValueError:  # loaded with no __spec__ (e.g. a module set in sys.modules): it exists
        return sys.modules.get("transformer_engine") is not None


def enlarge_te_gemm_workspace(log_tag: str = "te_gemm_workspace") -> int:
    """Set TE's cuBLAS / hipBLASLt workspace to ``PRIMUS_TE_GEMM_WORKSPACE_MB`` MiB on ROCm.

    Returns the effective workspace size in bytes: the new size if enlarged, the current size if
    it is already at least that large, or 0 when not applicable (CUDA,
    ``PRIMUS_TE_GEMM_WORKSPACE_MB`` <= 0, or Transformer Engine not installed; each logged except
    CUDA). Two TE layouts are handled: ``cpp_extensions.gemm`` (TE 2.17) and ``module.base``
    (TE 2.8); each one this TE has is enlarged (the smaller resulting size is returned). A TE with
    neither raises ``RuntimeError`` (the caller needs the larger workspace;
    ``PRIMUS_TE_GEMM_WORKSPACE_MB`` 0 or below opts out). When the workspace was already allocated,
    the new size is returned and a warning is printed on every rank: references taken before the
    call keep the old buffer.
    ``PRIMUS_TE_GEMM_WORKSPACE_MB`` must be an integer (MiB); 0 or below leaves TE's default."""
    if torch.version.hip is None:
        return 0
    size = workspace_mb_from_env() * 1024 * 1024
    if size <= 0:
        log_rank_0(
            f"[Patch:{log_tag}] PRIMUS_TE_GEMM_WORKSPACE_MB <= 0: TE GEMM workspace left at its default"
        )
        return 0
    # TE is optional: check it is installed without importing it; its submodules are imported
    # below only when present
    if not _te_installed():
        log_rank_0(
            f"[Patch:{log_tag}] WARNING: Transformer Engine not installed; TE GEMM workspace not changed"
        )
        return 0
    # every layout this TE has is enlarged: a build with both would otherwise keep the old size for
    # the buffers allocated through the other one
    te_gemm, te_base = te_workspace_modules()
    if te_gemm is None and te_base is None and import_or_none("transformer_engine.pytorch") is None:
        raise RuntimeError(
            f"[Patch:{log_tag}] Transformer Engine is installed without its PyTorch bindings "
            "(transformer_engine.pytorch), so the TE GEMM workspace cannot be raised to "
            f"{size >> 20} MiB. Install TE's PyTorch build, or set PRIMUS_TE_GEMM_WORKSPACE_MB=0."
        )
    # the one resize both workspace variables use (hipblaslt_workspace_patches.py)
    sizes = resize_te_gemm_workspace(size, log_tag, te_gemm=te_gemm, te_base=te_base)
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
