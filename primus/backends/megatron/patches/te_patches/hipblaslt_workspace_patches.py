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

A model patch that needs a larger workspace by default calls
``requested_workspace_mib`` / ``te_gemm_module`` / ``enlarge_te_gemm_workspace``
directly instead of writing the variable.
"""

import os
import traceback

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

ENV_VAR = "PRIMUS_TE_GEMM_WORKSPACE_MIB"
_BYTES_PER_MIB = 1024 * 1024
# the largest default TE sets (64 MiB on gfx950, 32 MiB elsewhere): a request at least this large
# cannot shrink TE's default even when the current size cannot be read
_MAX_TE_DEFAULT_MIB = 64


def requested_workspace_mib(warn: bool = True):
    """The workspace ``PRIMUS_TE_GEMM_WORKSPACE_MIB`` asks for, in MiB; None when it is unset or
    not a positive whole number (then, with ``warn``, a warning names the value)."""
    raw = os.getenv(ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        requested = int(raw)
    except ValueError:
        requested = 0
    if requested > 0:
        return requested
    if warn:
        warning_rank_0(f"{ENV_VAR}={raw!r} is not a positive whole number of MiB (e.g. 128); ignored")
    return None


# appended to every warning that skips a requested resize
_SKIP_HINT = "GEMMs that need a larger workspace can fail with 'HIPBLASLT Error: 6'"


# a missing module with one of these names means this environment has no TE PyTorch GEMM module
# (TE not installed, a JAX- or core-only build, or a TE without cpp_extensions.gemm)
_TE_ABSENT_MODULES = (
    "transformer_engine",
    "transformer_engine.pytorch",
    "transformer_engine.pytorch.cpp_extensions",
    "transformer_engine.pytorch.cpp_extensions.gemm",
)


# what enlarge_te_gemm_workspace needs from TE's GEMM module: te_gemm_module accepts exactly these
_WORKSPACE_API = ("get_cublas_workspace_size_bytes", "get_cublas_workspace")


def te_gemm_module(source: str):
    """TE's PyTorch GEMM module when it imports and has ``get_cublas_workspace_size_bytes`` and
    ``get_cublas_workspace`` (``_WORKSPACE_API``, what ``enlarge_te_gemm_workspace`` needs);
    otherwise None, with a warning naming ``source`` (who asks for the resize, e.g.
    ``"PRIMUS_TE_GEMM_WORKSPACE_MIB=128"``) and the reason: no TE GEMM module
    (``_TE_ABSENT_MODULES``), a module without one of those functions, or a TE that fails to import (any
    ``Exception``: an undefined symbol, its compiled extension, a dependency, a version check),
    logged with its traceback. Never raises an ``Exception``: a broken or absent TE only skips the
    resize (with a warning) instead of failing the before_train patch that asks for it."""
    asks = f"{source} asks for a larger TE GEMM workspace"
    try:
        from transformer_engine.pytorch.cpp_extensions import gemm as te_gemm
    except Exception as exc:  # noqa: BLE001 - a broken TE install is reported here, not raised
        # a TE package without the gemm submodule raises a plain ImportError naming its parent
        absent = getattr(exc, "name", None) in _TE_ABSENT_MODULES and (
            isinstance(exc, ModuleNotFoundError) or "cannot import name 'gemm'" in str(exc)
        )
        if absent:
            warning_rank_0(
                f"{asks} but this environment has no Transformer Engine PyTorch GEMM module "
                f"({exc}); TE's GEMM workspace is not resized ({_SKIP_HINT})"
            )
        else:
            warning_rank_0(
                f"{asks} but Transformer Engine fails to import; TE's GEMM workspace is not "
                f"resized ({_SKIP_HINT}):\n"
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
        return None
    missing = [n for n in _WORKSPACE_API if not hasattr(te_gemm, n)]
    if missing:
        warning_rank_0(
            f"{asks} but this Transformer Engine's cpp_extensions.gemm has no {', '.join(missing)}; "
            f"TE's GEMM workspace is not resized ({_SKIP_HINT})"
        )
        return None
    return te_gemm


def _needs_te_gemm_workspace_patch(_ctx: PatchContext) -> bool:
    """True when the variable is set to a positive whole number and TE's GEMM module has the
    workspace API (``te_gemm_module``). Otherwise False, with one warning when the variable is set:
    an invalid value, or no usable TE GEMM module (absent, without the API, or failing to import),
    so an explicit request is never dropped silently and the patch is not reported as applied.

    Not gated on model_type: any model opts in by setting the variable."""
    requested = requested_workspace_mib()
    if requested is None:
        return False
    return te_gemm_module(f"{ENV_VAR}={requested}") is not None


BASE_PATCH_ID = "megatron.te.hipblaslt_gemm_workspace"


def enlarge_te_gemm_workspace(te_gemm, mib: int, source: str, patch_id: str = BASE_PATCH_ID) -> None:
    """Make TE hand hipBLASLt a ``mib`` MiB workspace for every GEMM, unless it is already at
    least that large (it only ever enlarges). ``te_gemm`` is the module ``te_gemm_module``
    returned; ``source`` names the request in the log, as for ``te_gemm_module``; ``patch_id`` is
    the patch that asks (log lines are tagged ``[Patch:<patch_id>]``). Warns and skips, never
    raises: a ``mib`` that is not a positive int, or a module without the functions
    ``te_gemm_module`` checks (``_WORKSPACE_API``), leaves everything unchanged with a warning.
    When the current size cannot be read (TE's own getter failed, so no patch has replaced it),
    a request of at least ``_MAX_TE_DEFAULT_MIB`` is installed (logged as ``unknown -> <mib> MiB``)
    and a smaller one is skipped with a warning, since it could be below TE's default."""
    tag = f"[Patch:{patch_id}]"
    if isinstance(mib, bool) or not isinstance(mib, int) or mib <= 0:
        warning_rank_0(
            f"{tag} {source} asks for {mib!r} MiB, not a positive whole number; TE's GEMM workspace "
            f"is not resized ({_SKIP_HINT})"
        )
        return
    missing = [n for n in _WORKSPACE_API if not hasattr(te_gemm, n)]
    if missing:
        warning_rank_0(
            f"{tag} {te_gemm!r} has no {', '.join(missing)} (not TE's GEMM module, see te_gemm_module); "
            f"TE's GEMM workspace is not resized ({_SKIP_HINT})"
        )
        return
    get_workspace = te_gemm.get_cublas_workspace
    requested_bytes = mib * _BYTES_PER_MIB
    try:
        previous_bytes = te_gemm.get_cublas_workspace_size_bytes()
    except Exception:  # noqa: BLE001 - unknown size: see the docstring
        previous_bytes = None
    if previous_bytes is None and mib < _MAX_TE_DEFAULT_MIB:
        warning_rank_0(
            f"{tag} TE GEMM workspace size cannot be read; the {mib} MiB {source} asks for could be "
            f"below TE's default (up to {_MAX_TE_DEFAULT_MIB} MiB), so it is left as is ({_SKIP_HINT})"
        )
        return

    # Another patch may already have enlarged it (GLM-5's, through its own variable).
    if previous_bytes is not None and previous_bytes >= requested_bytes:
        log_rank_0(
            f"{tag} TE GEMM workspace is already {previous_bytes // _BYTES_PER_MIB} MiB, at least the "
            f"{mib} MiB {source} asks for; left as is"
        )
        return

    te_gemm.get_cublas_workspace_size_bytes = lambda: requested_bytes
    # Already-allocated workspaces are memoized per device.
    if hasattr(get_workspace, "cache_clear"):
        get_workspace.cache_clear()

    was = "unknown" if previous_bytes is None else f"{previous_bytes // _BYTES_PER_MIB} MiB"
    log_rank_0(f"{tag} TE GEMM workspace {was} -> {mib} MiB ({source})")


@register_patch(
    BASE_PATCH_ID,
    backend="megatron",
    phase="before_train",
    description="Resize TE's hipBLASLt GEMM workspace (PRIMUS_TE_GEMM_WORKSPACE_MIB)",
    condition=_needs_te_gemm_workspace_patch,
)
def patch_te_gemm_workspace_size(ctx: PatchContext):
    """Enlarge the workspace TE hands to hipBLASLt for every GEMM."""
    del ctx
    requested = requested_workspace_mib(warn=False)  # the condition has warned about a bad value
    if requested is None:
        return
    source = f"{ENV_VAR}={requested}"
    # the condition has checked the module; None here only if it changed since: it has said why
    te_gemm = te_gemm_module(source)
    if te_gemm is not None:
        enlarge_te_gemm_workspace(te_gemm, requested, source)
