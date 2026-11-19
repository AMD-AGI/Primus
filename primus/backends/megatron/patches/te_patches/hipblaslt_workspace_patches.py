###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine hipBLASLt GEMM workspace patches.

TE fixes its GEMM workspace at 64 MiB on gfx950 (32 MiB elsewhere) with no way
to change it. That is too small for a weight-gradient GEMM whose reduction axis
is long while its output is small and square: hipBLASLt answers such a shape
with a split-K solution, the number of splits grows with the reduction length,
and each split needs its own ``M x N x 4`` accumulator. Once the splits no
longer fit, the call fails outright instead of falling back::

    RuntimeError: rocm_gemm.hip:1707 in function hipblaslt_gemm: HIPBLASLT Error: 6

WAN 2.1 T2V-1.3B hits this in its 1536 -> 1536 linears from roughly 70,000
tokens up (micro_batch_size 7 at 10,920 tokens per sample); 1536 -> 8960 and
8960 -> 1536 are unaffected because their larger outputs rule split-K out
anyway. Seen on rocm/primus:v26.5 (TE 2.15.0.dev0).

Set ``PRIMUS_TE_GEMM_WORKSPACE_MIB`` to enlarge the workspace; 128 clears the
WAN shapes. Unset, nothing is touched, so a bigger workspace never silently
changes hipBLASLt's solution choice for runs that do not need it. It only ever
enlarges: a workspace that is already at least that size, such as the one
GLM-5's patch sets, is left alone.
"""

import importlib
import os
import sys

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

ENV_VAR = "PRIMUS_TE_GEMM_WORKSPACE_MIB"

# ---- one TE workspace resize, shared with gemm_workspace_size_patches.py (GLM-5.3, Mistral Large 4)


def te_cpp_extensions_gemm_layout_supported(te_gemm) -> bool:
    """Whether ``transformer_engine.pytorch.cpp_extensions.gemm`` has the TE 2.17 layout: the size
    function ``get_cublas_workspace_size_bytes()`` and the ``lru_cache``'d ``get_cublas_workspace``
    that allocates the buffer ``general_gemm`` passes to hipBLASLt."""
    return callable(getattr(te_gemm, "get_cublas_workspace_size_bytes", None)) and hasattr(
        getattr(te_gemm, "get_cublas_workspace", None), "cache_clear"
    )


def te_module_base_layout_supported(te_base) -> bool:
    """Whether ``transformer_engine.pytorch.module.base`` has the TE 2.8 layout: the size function
    ``get_cublas_workspace_size_bytes()`` read at allocation time by ``get_workspace()`` and
    ``get_multi_stream_cublas_workspace()``, which cache the buffers in the module globals
    ``_cublas_workspace`` and ``_multi_stream_cublas_workspace``."""
    return (
        callable(getattr(te_base, "get_cublas_workspace_size_bytes", None))
        and callable(getattr(te_base, "get_workspace", None))
        and hasattr(te_base, "_cublas_workspace")
    )


def import_or_none(name):
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


def te_workspace_modules():
    """``(cpp_extensions.gemm, module.base)`` of the installed TE, each None when TE lacks it."""
    return (
        import_or_none("transformer_engine.pytorch.cpp_extensions.gemm"),
        import_or_none("transformer_engine.pytorch.module.base"),
    )


def warn_allocated(log_tag: str) -> None:
    # buffers already handed out (e.g. kept by comm-overlap code) keep their old size: the resize
    # must run before the first GEMM (the before_train patches do). Logged on every rank (the rank
    # that holds such a reference may not be rank 0), with the rank in the message.
    rank = os.environ.get("RANK", "?")
    print(
        f"[Patch:{log_tag}] WARNING (rank {rank}): TE GEMM workspace already allocated; references to it "
        "taken before this call keep the old size, and the old buffer stays alive while they do",
        file=sys.stderr,
        flush=True,
    )


def _enlarge_cpp_extensions_gemm_workspace(te_gemm, size: int, log_tag: str) -> int:
    """TE 2.17: replace ``cpp_extensions.gemm.get_cublas_workspace_size_bytes`` and clear the
    ``get_cublas_workspace`` cache so the next GEMM allocates at the new size."""
    current = te_gemm.get_cublas_workspace_size_bytes()
    if current >= size:
        return current
    info = getattr(te_gemm.get_cublas_workspace, "cache_info", None)
    if callable(info) and info().currsize > 0:
        warn_allocated(log_tag)
    te_gemm.get_cublas_workspace_size_bytes = lambda: size
    te_gemm.get_cublas_workspace.cache_clear()
    log_rank_0(f"[Patch:{log_tag}] TE GEMM workspace {current >> 20} MiB -> {size >> 20} MiB")
    return size


def _enlarge_module_base_workspace(te_base, size: int, log_tag: str) -> int:
    """TE 2.8: replace ``module.base.get_cublas_workspace_size_bytes`` (looked up in the module
    globals when ``get_workspace`` / ``get_multi_stream_cublas_workspace`` allocate) and drop the
    cached buffers so the next GEMM allocates them at the new size."""
    current = te_base.get_cublas_workspace_size_bytes()
    if current >= size:
        return current
    if getattr(te_base, "_cublas_workspace", None) is not None or bool(
        getattr(te_base, "_multi_stream_cublas_workspace", None)
    ):
        warn_allocated(log_tag)
    te_base.get_cublas_workspace_size_bytes = lambda: size
    te_base._cublas_workspace = None
    if hasattr(te_base, "_multi_stream_cublas_workspace"):
        te_base._multi_stream_cublas_workspace = []
    log_rank_0(
        f"[Patch:{log_tag}] TE GEMM workspace {current >> 20} MiB -> {size >> 20} MiB (module.base layout)"
    )
    return size


def resize_te_gemm_workspace(size_bytes: int, log_tag: str, te_gemm=None, te_base=None) -> list:
    """Enlarge TE's GEMM workspace to ``size_bytes`` in every layout given (TE 2.17
    ``cpp_extensions.gemm``, TE 2.8 ``module.base``; None or a module without the layout is
    skipped). Never lowers it. Returns the effective size of each layout handled, in bytes (empty
    when neither layout is there; the caller decides whether that is an error). Each layout is
    enlarged in turn (a size-function swap and a cache drop); a failure in the second leaves the
    first enlarged (larger, never wrong)."""
    sizes = []
    if te_cpp_extensions_gemm_layout_supported(te_gemm):  # False for None too
        sizes.append(_enlarge_cpp_extensions_gemm_workspace(te_gemm, size_bytes, log_tag))
    if te_module_base_layout_supported(te_base):  # False for None too
        sizes.append(_enlarge_module_base_workspace(te_base, size_bytes, log_tag))
    return sizes


def _requested_workspace_mib():
    """Return the requested workspace in MiB, or None when unset/invalid."""
    raw = os.getenv(ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        requested = int(raw)
    except ValueError:
        requested = 0
    if requested > 0:
        return requested
    warning_rank_0(
        f"{ENV_VAR}={raw!r} is not a positive whole number of MiB (e.g. 128); "
        "leaving TE's GEMM workspace at its default"
    )
    return None


def _needs_te_gemm_workspace_patch(_ctx: PatchContext) -> bool:
    # not gated on model_type: any model opts in by setting PRIMUS_TE_GEMM_WORKSPACE_MIB
    if _requested_workspace_mib() is None:
        return False
    try:
        te_gemm, te_base = te_workspace_modules()
    except ImportError:
        return False
    return te_cpp_extensions_gemm_layout_supported(te_gemm) or te_module_base_layout_supported(te_base)


@register_patch(
    "megatron.te.hipblaslt_gemm_workspace",
    backend="megatron",
    phase="before_train",
    description="Resize TE's hipBLASLt GEMM workspace (PRIMUS_TE_GEMM_WORKSPACE_MIB)",
    condition=_needs_te_gemm_workspace_patch,
)
def patch_te_gemm_workspace_size(ctx: PatchContext):
    """Enlarge the workspace TE hands to hipBLASLt for every GEMM (TE 2.17 and TE 2.8 layouts)."""
    del ctx

    requested_bytes = _requested_workspace_mib() * 1024 * 1024
    te_gemm, te_base = te_workspace_modules()
    # another patch may already have enlarged it (GLM-5.3's / Mistral Large 4's, through
    # PRIMUS_TE_GEMM_WORKSPACE_MB): never lowered
    sizes = resize_te_gemm_workspace(
        requested_bytes, "megatron.te.hipblaslt_gemm_workspace", te_gemm=te_gemm, te_base=te_base
    )
    if sizes and min(sizes) > requested_bytes:
        log_rank_0(
            "[Patch:megatron.te.hipblaslt_gemm_workspace] TE GEMM workspace is already "
            f"{min(sizes) // (1024 * 1024)} MiB, at least the "
            f"{requested_bytes // (1024 * 1024)} MiB {ENV_VAR} asks for; left as is"
        )
