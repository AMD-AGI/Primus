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
changes hipBLASLt's solution choice for runs that do not need it.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0

ENV_VAR = "PRIMUS_TE_GEMM_WORKSPACE_MIB"


def _requested_workspace_mib():
    """Return the requested workspace in MiB, or None when unset/invalid."""
    raw = os.getenv(ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        requested = int(raw)
    except ValueError:
        return None
    return requested if requested > 0 else None


def _needs_te_gemm_workspace_patch(_ctx: PatchContext) -> bool:
    if _requested_workspace_mib() is None:
        return False
    try:
        from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm
    except (ImportError, ModuleNotFoundError):
        return False
    return hasattr(te_gemm, "get_cublas_workspace_size_bytes")


@register_patch(
    "megatron.te.hipblaslt_gemm_workspace",
    backend="megatron",
    phase="before_train",
    description="Resize TE's hipBLASLt GEMM workspace (PRIMUS_TE_GEMM_WORKSPACE_MIB)",
    condition=_needs_te_gemm_workspace_patch,
)
def patch_te_gemm_workspace_size(ctx: PatchContext):
    """Enlarge the workspace TE hands to hipBLASLt for every GEMM."""
    del ctx

    from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm

    requested_bytes = _requested_workspace_mib() * 1024 * 1024
    try:
        previous_bytes = te_gemm.get_cublas_workspace_size_bytes()
    except Exception:  # noqa: BLE001 - only used for the log line
        previous_bytes = None

    te_gemm.get_cublas_workspace_size_bytes = lambda: requested_bytes
    # Already-allocated workspaces are memoized per device.
    if hasattr(te_gemm.get_cublas_workspace, "cache_clear"):
        te_gemm.get_cublas_workspace.cache_clear()

    was = "unknown" if previous_bytes is None else f"{previous_bytes // (1024 * 1024)} MiB"
    log_rank_0(
        "[Patch:megatron.te.hipblaslt_gemm_workspace] TE GEMM workspace "
        f"{was} -> {requested_bytes // (1024 * 1024)} MiB ({ENV_VAR})"
    )
